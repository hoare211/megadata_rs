"""外部基线对比基准：megadata_rs vs HuggingFace datasets vs PyTorch DataLoader。

回答"和 PyTorch DataLoader / HF datasets 比到底快多少"：同一份 jsonl、同一个分词器
（bbpe Python 包装 → 三条链路 token 流一致），双场景（PT 流式 packed 读 / SFT 变长监督批）
× 三链路：

  1. megadata_rs   BatchStream 顺序读 phase9 scale 产物（预处理引用 phase9 实测）；
  2. HF datasets   生态最佳实践：load_dataset(json) → map(tokenize, batched, num_proc=N)
                   → arrow 缓存 → DataLoader；PT 额外 group_texts 拼接切块（等价 packing）；
  3. PyTorch DataLoader 无预处理生态：IterableDataset 流式读 jsonl + 现算 tokenize
                   （num_workers=N 字节偏移分片），PT 动态拼窗口、SFT collate pad
                   ——给基线最有利的形态（预处理=0，全部成本融合进读取）。

公平性口径（诚实声明，随报告输出）：
  - 三链路用同一个 bbpe 分词器的 Python 包装；megadata 原生路径在 Rust 内直连分词，
    无逐文档 Python 调用开销——这是架构差异的一部分，不是本 bench 的误差；
  - megadata 构建期数字引用 phase9 §5.6（同机、RAYON_NUM_THREADS=26 实测），不重跑；
  - SFT 基线按生态惯例做 pad batching（megadata 是构建期 packing，零 padding），
    padding 浪费率单独报告；PT 三链路 token 流逐位同构（逐 doc encode_ordinary+eos
    后全局拼接切块），吞吐可直接比较；
  - 读取顺序 megadata → HF → DataLoader：7.8GB 输入在 251GB 内存的机器上跑完第一遍后
    全部进 page cache，后续链路磁盘 IO 基本为零（对 megadata 略不利，标注）；
  - RSS 测的是**进程树峰值**（含 HF num_proc 子进程 / DataLoader worker）。

正确性守恒（基线实现错了对比就无效，随跑随验）：
  - PT：有效 token 总数 == manifest total_tokens，且首窗口逐位 == bbpe 独立复算的真值前缀；
  - SFT：有效 token 总数 == manifest total_tokens + 截断丢弃数（基线不截断），
    抽样前 8 条 (ids, labels) == 独立监督掩码状态机逐位复算。

指标：预处理耗时 / 读取耗时 / 有效 token 吞吐 / 进程树峰值 RSS / 首批延迟 / p50/p99。

用法：
    .venv/bin/python bench/compare_baselines.py --smoke              # mini 数据快速验证管线
    .venv/bin/python bench/compare_baselines.py                      # 全量（scale 数据，需先跑过 phase9）
    .venv/bin/python bench/compare_baselines.py --scenario pt        # 只跑 PT 场景
环境变量：BENCH_CPU=26（CPU 预算，三链路统一）。
输出：bench/out/compare_baselines.{md,json}。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA = os.path.join(ROOT, "data")
VERIFY_OUT = os.path.join(ROOT, "tools", "verify", "out")
BENCH_OUT = os.path.join(ROOT, "bench", "out")
HF_CACHE = os.path.join(VERIFY_OUT, "bench_hf_cache")
SCALE_PT_INDEX = os.path.join(VERIFY_OUT, "scale_pt", "index.json")
SCALE_SFT_INDEX = os.path.join(VERIFY_OUT, "scale_sft", "index.json")
SMOKE_PT_INDEX = os.path.join(VERIFY_OUT, "bench_smoke_pt", "index.json")
SMOKE_SFT_INDEX = os.path.join(VERIFY_OUT, "bench_smoke_sft", "index.json")
TOKENIZER_DIR = os.path.abspath(os.path.join(ROOT, "..", "bbpe_rs", "model_learn_tokenizer"))

PT_IN = os.path.join(DATA, "pretrain_t2t.jsonl")
SFT_IN = os.path.join(DATA, "sft_t2t_mini_x.jsonl")
SMOKE_PT_IN = os.path.join(DATA, "pretrain_t2t_mini.jsonl")
SMOKE_SFT_IN = os.path.join(DATA, "sft_t2t_mini.jsonl")

PT_BS, PT_MSL = 16, 2048
SFT_BS, SFT_MSL = 8, 2048
# phase9 §5.6 实测构建期（同机、RAYON_NUM_THREADS=26）：PT 328s / SFT 88s。bench 引用不重跑。
PHASE9_BUILD_S = {"pt": 328.0, "sft": 88.0}

RESULT_PREFIX = "RESULT_JSON: "


# ======================================================================
# 公共：分词器与真值复算（与 tools/verify/phase9_scale.py 同一规则）
# ======================================================================
_TOK = None
EOS_ID = None
TURN_START_ID = None
TURN_END_ID = None


def _bbpe():
    global _TOK
    if _TOK is None:
        from bbpe_tokenizer import BBTokenizer
        _TOK = BBTokenizer.from_pretrained(TOKENIZER_DIR)
    return _TOK


def _init_ids():
    global EOS_ID, TURN_START_ID, TURN_END_ID
    t = _bbpe()
    EOS_ID = t.encode("<|endoftext|>", allowed_special=["all"]).ids[0]
    TURN_START_ID = t.encode("<|im_start|>", allowed_special=["all"]).ids[0]
    TURN_END_ID = t.encode("<|im_end|>", allowed_special=["all"]).ids[0]


def render_chat(conv) -> str:
    # datasets 的 struct 列给 map 的元素是消息列表本身；torch 路径 json.loads 给的是行 dict
    msgs = conv if isinstance(conv, list) else (conv.get("conversations") or conv.get("messages") or [])
    return "".join(
        "<|im_start|>" + str(m["role"]) + "\n" + str(m["content"]) + "<|im_end|>\n" for m in msgs
    )


def _sft_gt_doc(conv):
    """真值复算：渲染文本 encode + 监督掩码状态机（assistant 段监督，标签右移一位）。"""
    e = _bbpe().encode(render_chat(conv), allowed_special=["all"])
    ids, toks = list(e.ids), list(e.tokens)
    n = len(ids)
    sup, in_asst = [False] * n, False
    for k in range(n):
        tk = toks[k] if k < len(toks) else ""
        if ids[k] == TURN_START_ID or (tk and "<|im_start|>" in tk):
            role = []
            for j in range(k + 1, min(k + 8, n)):
                nx = toks[j] if j < len(toks) else ""
                if "\n" in nx or "<|im_end|>" in nx or "<|im_start|>" in nx:
                    break
                role.append(nx)
            in_asst = "".join(role).strip().startswith("assistant")
        if in_asst:
            sup[k] = True
        if ids[k] == TURN_END_ID or (tk and "<|im_end|>" in tk):
            if in_asst:
                sup[k] = True
                in_asst = False
    labels = [-100] * n
    for k in range(n - 1):
        if sup[k]:
            labels[k] = ids[k + 1]
    return ids, labels


def _pt_gt_prefix(path, n_tok):
    """真值流前缀：逐 doc encode_ordinary + eos 全局拼接的前 n_tok 个 token。"""
    st = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            st.extend(_bbpe().encode_ordinary(json.loads(line)["text"]))
            st.append(EOS_ID)
            if len(st) >= n_tok:
                return st[:n_tok]
    raise RuntimeError(f"input too small for {n_tok}-token prefix: {path}")


def _first_convs(path, k):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if len(out) >= k:
                break
    return out


def _percentile(sorted_xs, p):
    return sorted_xs[min(int(len(sorted_xs) * p), len(sorted_xs) - 1)] if sorted_xs else 0.0


def _emit(result):
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)


def _check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}", flush=True)
    return {"check": name, "ok": bool(ok), "detail": str(detail)}


# ======================================================================
# 链路 1：megadata_rs BatchStream 顺序读
# ======================================================================
def run_megadata(spec):
    from megadata import BatchStream

    bs, msl = spec["bs"], spec["msl"]
    _init_ids()
    loader = BatchStream(index_json_path=spec["index_json"], batch_size=bs, max_seq_len=msl,
                         world_size=1, rank=0, shuffle=False, shard_order="sequential")
    t_start = time.perf_counter()
    it = iter(loader)
    first = None
    useful = pad = nb = 0
    lat = []
    first_window = None
    docs = []
    last_t = time.perf_counter()

    def consume(b):
        nonlocal useful, pad, nb, first, first_window, docs
        nb += 1
        if first is None:
            first = time.perf_counter() - t_start
        if spec["scenario"] == "pt":
            ids, cu = b["input_ids"], b["cu_seqlens"]
            valid = int(cu[-1])
            useful += valid
            pad += ids.numel() - valid
            if first_window is None:
                first_window = ids[0][:msl].tolist()
        else:
            cu = b["cu_seqlens"]
            useful += int(cu[-1])
            ids_rows, lab_rows = b["input_ids"], b["labels"]
            pos_rows, row_lens = b["position_ids"], b["row_lens"]
            di = 0
            for r in range(len(row_lens)):
                rl, pos = int(row_lens[r]), 0
                while di < len(cu) - 1 and pos < rl:
                    ln = int(cu[di + 1] - cu[di])
                    if ln <= 0:
                        di += 1
                        continue
                    if len(docs) < 8:
                        docs.append((ids_rows[r][pos:pos + ln].tolist(),
                                     lab_rows[r][pos:pos + ln].tolist(),
                                     pos_rows[r][pos:pos + ln].tolist()))
                    pos += ln
                    di += 1

    consume(next(it))
    while True:
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        lat.append((time.perf_counter() - t0) * 1000)
        consume(b)
        if time.perf_counter() - last_t > 30:
            last_t = time.perf_counter()
            print(f"[progress] megadata/{spec['scenario']} batches={nb} tokens={useful:,}", flush=True)
    read_s = time.perf_counter() - t_start
    lat.sort()

    checks = []
    if spec["scenario"] == "pt":
        checks.append(_check("[megadata/PT] token 守恒 == manifest", useful == spec["expected_tokens"],
                             f"{useful:,} vs {spec['expected_tokens']:,}"))
        exp = _pt_gt_prefix(spec["input"], msl)
        checks.append(_check("[megadata/PT] 首窗口逐位 == 真值前缀", first_window == exp,
                             f"len={len(first_window or [])}"))
    else:
        checks.append(_check("[megadata/SFT] token 守恒 == manifest（含截断口径）",
                             useful == spec["expected_tokens"],
                             f"{useful:,} vs {spec['expected_tokens']:,}"))
        bad = []
        for i, (ids, lab, pos) in enumerate(docs):
            gt_ids, gt_lab = _sft_gt_doc(_first_convs(spec["input"], 8)[i])
            if ids != gt_ids[:len(ids)] or lab != gt_lab[:len(lab)] or pos != list(range(len(ids))):
                bad.append(i)
        checks.append(_check("[megadata/SFT] 前 8 条 ids+labels+pos 逐位 == 状态机复算", not bad, f"{bad}"))

    _emit({"chain": "megadata", "scenario": spec["scenario"],
           "preprocess_s": None,  # 引用 phase9，不测
           "read_s": read_s, "batches": nb, "useful_tokens": useful,
           "pad_tokens": pad, "first_batch_s": first,
           "p50_ms": _percentile(lat, 0.50), "p99_ms": _percentile(lat, 0.99),
           "checks": checks})


# ======================================================================
# 链路 2：HF datasets（arrow 缓存 + DataLoader）
# ======================================================================
def _hf_pt_tok_map(batch):
    import numpy as np
    t = _bbpe()
    return {"ids": [np.asarray(t.encode_ordinary(x) + [EOS_ID], dtype=np.int32) for x in batch["text"]]}


def _hf_pt_group_map(batch, msl):
    """HF group_texts 惯用法：拼接切块成 msl 窗口（每 map 批内拼接，尾块保留不丢 token）。"""
    import numpy as np
    ids = np.concatenate(batch["ids"]) if batch["ids"] else np.array([], dtype=np.int32)
    n = (len(ids) + msl - 1) // msl
    return {"ids": [ids[i * msl:(i + 1) * msl] for i in range(n)]}


def _hf_sft_map(batch):
    import numpy as np
    key = "conversations" if "conversations" in batch else "messages"
    ids_l, lab_l, len_l = [], [], []
    for conv in batch[key]:
        ids, labels = _sft_gt_doc(conv)
        ids_l.append(np.asarray(ids, dtype=np.int32))
        lab_l.append(np.asarray(labels, dtype=np.int32))
        len_l.append(len(ids))
    return {"ids": ids_l, "labels": lab_l, "doc_len": len_l}


def _hf_pt_collate(feats):
    import torch
    rows = [f["ids"] for f in feats]
    lens = torch.tensor([len(r) for r in rows])
    m = int(lens.max())
    ids = torch.zeros(len(rows), m, dtype=torch.int32)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
    return ids, lens


def _hf_sft_collate(feats):
    import torch
    lens = torch.tensor([f["doc_len"] for f in feats])
    m = int(max(len(f["ids"]) for f in feats))
    ids = torch.zeros(len(feats), m, dtype=torch.int32)
    labels = torch.full((len(feats), m), -100, dtype=torch.int32)
    for i, f in enumerate(feats):
        ln = len(f["ids"])
        ids[i, :ln] = torch.tensor(f["ids"], dtype=torch.int32)
        labels[i, :ln] = torch.tensor(f["labels"], dtype=torch.int32)
    return ids, labels, lens


def run_hf(spec):
    import datasets
    import torch
    from torch.utils.data import DataLoader

    datasets.disable_progress_bars()
    torch.set_num_threads(1)
    _init_ids()
    bs, msl = spec["bs"], spec["msl"]

    t0 = time.perf_counter()
    raw = datasets.load_dataset("json", data_files=spec["input"], split="train", cache_dir=HF_CACHE)
    if spec["scenario"] == "pt":
        ds = raw.map(_hf_pt_tok_map, batched=True, batch_size=1000, num_proc=spec["n"],
                     remove_columns=raw.column_names, desc="hf-tokenize")
        ds = ds.map(_hf_pt_group_map, batched=True, batch_size=100000, num_proc=1,
                    fn_kwargs={"msl": msl}, desc="hf-group")
        first_window = list(ds[0]["ids"])
    else:
        ds = raw.map(_hf_sft_map, batched=True, batch_size=500, num_proc=spec["n"],
                     remove_columns=raw.column_names, desc="hf-tokenize+mask")
        first_window = None
    preprocess_s = time.perf_counter() - t0

    collate = _hf_pt_collate if spec["scenario"] == "pt" else _hf_sft_collate
    t0 = time.perf_counter()
    dl = DataLoader(ds, batch_size=bs, num_workers=spec["n"], collate_fn=collate)
    it = iter(dl)
    first = None
    useful = pad = nb = 0
    lat = []
    last_t = time.perf_counter()
    while True:
        t1 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        lat.append((time.perf_counter() - t1) * 1000)
        nb += 1
        if first is None:
            first = time.perf_counter() - t0
        if spec["scenario"] == "pt":
            ids, lens = b
            got = int(lens.sum())
            useful += got
            pad += ids.numel() - got
        else:
            ids, labels, lens = b
            got = int(lens.sum())
            useful += got
            pad += ids.numel() - got
        if time.perf_counter() - last_t > 30:
            last_t = time.perf_counter()
            print(f"[progress] hf/{spec['scenario']} batches={nb} tokens={useful:,}", flush=True)
    read_s = time.perf_counter() - t0
    lat.sort()

    checks = []
    if spec["scenario"] == "pt":
        checks.append(_check("[hf/PT] token 守恒 == manifest", useful == spec["expected_tokens"],
                             f"{useful:,} vs {spec['expected_tokens']:,}"))
        exp = _pt_gt_prefix(spec["input"], msl)
        checks.append(_check("[hf/PT] 首窗口逐位 == 真值前缀", first_window == exp,
                             f"len={len(first_window)}"))
    else:
        checks.append(_check("[hf/SFT] 有效 token == manifest+截断数（基线不截断）",
                             useful == spec["expected_baseline_tokens"],
                             f"{useful:,} vs {spec['expected_baseline_tokens']:,}"))
        bad = []
        convs = _first_convs(spec["input"], 8)
        for i, conv in enumerate(convs):
            gt_ids, gt_lab = _sft_gt_doc(conv)
            row = ds[i]
            if list(row["ids"]) != gt_ids or list(row["labels"]) != gt_lab:
                bad.append(i)
        checks.append(_check("[hf/SFT] 前 8 条 ids+labels 逐位 == 状态机复算", not bad, f"{bad}"))

    _emit({"chain": "hf", "scenario": spec["scenario"],
           "preprocess_s": preprocess_s, "read_s": read_s, "batches": nb,
           "useful_tokens": useful, "pad_tokens": pad, "first_batch_s": first,
           "p50_ms": _percentile(lat, 0.50), "p99_ms": _percentile(lat, 0.99),
           "checks": checks})


# ======================================================================
# 链路 3：PyTorch DataLoader（jsonl 现算，无预处理生态）
# ======================================================================
def _seek_line_start(f, start):
    """把读指针对齐到本 worker 范围内第一条完整行的行首。

    边界恰好落在行首时（前一字节是 \\n）不能丢弃，否则该行无人处理（丢 token）；
    落在行中间时丢弃半行（归属上一个 worker），否则会与上一个 worker 重复。
    """
    if start == 0:
        return
    f.seek(start - 1)
    if f.read(1) != b"\n":
        f.readline()


def _shard_range(path, wid, nw):
    size = os.path.getsize(path)
    per = size // nw
    start = wid * per
    end = size if wid == nw - 1 else start + per
    return start, end


def _iter_lines(path, start, end):
    """按字节范围 [start, end) 迭代行：行起点落在范围内的归本 worker。

    关键：先 tell()（= 下一行起点）判界再 readline——跨边界行起点 < end，
    由本 worker 处理完后下轮 pos >= end 自然退出；若先读后判，该行会被读出
    却不处理，而下个 worker 又按"起点在前一范围"丢弃它 → 每边界丢一行。
    """
    with open(path, "rb") as f:
        _seek_line_start(f, start)
        while True:
            pos = f.tell()
            if pos >= end:
                return
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            yield line


def _iter_pt_windows(path, msl, wid, nw):
    """字节偏移分片流式读 jsonl，逐 doc encode_ordinary+eos，动态拼 msl 窗口。

    每个 worker 末尾至多产生 1 个 <msl 的尾窗（保留不丢 token，等价 megadata
    的非满批语义）。
    """
    t = _bbpe()
    start, end = _shard_range(path, wid, nw)
    buf = []
    for line in _iter_lines(path, start, end):
        buf.extend(t.encode_ordinary(json.loads(line)["text"]))
        buf.append(EOS_ID)
        while len(buf) >= msl:
            yield buf[:msl]
            buf = buf[msl:]
    if buf:
        yield buf


def _iter_sft_docs(path, wid, nw):
    """字节偏移分片流式读 jsonl，逐条渲染+encode+掩码（与真值同一状态机）。"""
    start, end = _shard_range(path, wid, nw)
    for line in _iter_lines(path, start, end):
        yield _sft_gt_doc(json.loads(line))


def _make_pt_dataset(path, msl):
    from torch.utils.data import IterableDataset

    class DS(IterableDataset):
        def __iter__(self):
            import torch
            info = torch.utils.data.get_worker_info()
            wid, nw = (info.id, info.num_workers) if info else (0, 1)
            yield from _iter_pt_windows(path, msl, wid, nw)
    return DS()


def _make_sft_dataset(path):
    from torch.utils.data import IterableDataset

    class DS(IterableDataset):
        def __iter__(self):
            import torch
            info = torch.utils.data.get_worker_info()
            wid, nw = (info.id, info.num_workers) if info else (0, 1)
            yield from _iter_sft_docs(path, wid, nw)
    return DS()


def _torch_pt_collate(rows):
    import torch
    lens = torch.tensor([len(r) for r in rows])
    m = int(lens.max())
    ids = torch.zeros(len(rows), m, dtype=torch.int32)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r, dtype=torch.int32)
    return ids, lens


def _torch_sft_collate(rows):
    import torch
    lens = torch.tensor([len(ids) for ids, _ in rows])
    m = int(lens.max())
    ids = torch.zeros(len(rows), m, dtype=torch.int32)
    labels = torch.full((len(rows), m), -100, dtype=torch.int32)
    for i, (a, b) in enumerate(rows):
        ids[i, :len(a)] = torch.tensor(a, dtype=torch.int32)
        labels[i, :len(b)] = torch.tensor(b, dtype=torch.int32)
    return ids, labels, lens


def run_torch(spec):
    import torch
    from torch.utils.data import DataLoader

    torch.set_num_threads(1)
    _init_ids()
    bs, msl = spec["bs"], spec["msl"]

    # ---- 不限时正确性快查（独立单 worker 小迭代，不污染计时）----
    checks = []
    if spec["scenario"] == "pt":
        first_window = next(_iter_pt_windows(spec["input"], msl, 0, 1))
        exp = _pt_gt_prefix(spec["input"], msl)
        checks.append(_check("[torch/PT] 首窗口逐位 == 真值前缀", list(first_window) == exp,
                             f"len={len(first_window)}"))
    else:
        got = []
        for ids, lab in _iter_sft_docs(spec["input"], 0, 1):
            got.append((ids, lab))
            if len(got) >= 8:
                break
        bad = []
        for i, conv in enumerate(_first_convs(spec["input"], 8)):
            gt_ids, gt_lab = _sft_gt_doc(conv)
            if got[i][0] != gt_ids or got[i][1] != gt_lab:
                bad.append(i)
        checks.append(_check("[torch/SFT] 前 8 条 ids+labels 逐位 == 状态机复算", not bad, f"{bad}"))

    ds = _make_pt_dataset(spec["input"], msl) if spec["scenario"] == "pt" else _make_sft_dataset(spec["input"])
    collate = _torch_pt_collate if spec["scenario"] == "pt" else _torch_sft_collate
    t0 = time.perf_counter()
    dl = DataLoader(ds, batch_size=bs, num_workers=spec["n"], collate_fn=collate)
    it = iter(dl)
    first = None
    useful = pad = nb = 0
    lat = []
    last_t = time.perf_counter()
    while True:
        t1 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        lat.append((time.perf_counter() - t1) * 1000)
        nb += 1
        if first is None:
            first = time.perf_counter() - t0
        if spec["scenario"] == "pt":
            ids, lens = b
            got = int(lens.sum())
        else:
            ids, labels, lens = b
            got = int(lens.sum())
        useful += got
        pad += ids.numel() - got
        if time.perf_counter() - last_t > 30:
            last_t = time.perf_counter()
            print(f"[progress] torch/{spec['scenario']} batches={nb} tokens={useful:,}", flush=True)
    read_s = time.perf_counter() - t0
    lat.sort()

    if spec["scenario"] == "pt":
        checks.append(_check("[torch/PT] token 守恒 == manifest", useful == spec["expected_tokens"],
                             f"{useful:,} vs {spec['expected_tokens']:,}"))
    else:
        checks.append(_check("[torch/SFT] 有效 token == manifest+截断数（基线不截断）",
                             useful == spec["expected_baseline_tokens"],
                             f"{useful:,} vs {spec['expected_baseline_tokens']:,}"))

    _emit({"chain": "torch", "scenario": spec["scenario"],
           "preprocess_s": 0.0, "read_s": read_s, "batches": nb,
           "useful_tokens": useful, "pad_tokens": pad, "first_batch_s": first,
           "p50_ms": _percentile(lat, 0.50), "p99_ms": _percentile(lat, 0.99),
           "checks": checks})


# ======================================================================
# worker 入口 / 进程树 RSS / 父进程编排
# ======================================================================
def _worker_main(spec_path):
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    try:
        {"megadata": run_megadata, "hf": run_hf, "torch": run_torch}[spec["chain"]](spec)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)


class RssTreeMeter:
    """采样目标进程及其全部后代的 VmRSS 之和的峰值（KB）。"""

    def __init__(self, root_pid):
        self.root = root_pid
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _tree_rss_kb(self):
        kids = {}
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
                    data = fh.read()
                ppid = int(data[data.rindex(")") + 2:].split()[1])
                kids.setdefault(ppid, []).append(int(pid))
            except (OSError, ValueError, IndexError):
                continue
        total, stack = 0, [self.root]
        while stack:
            p = stack.pop()
            stack.extend(kids.get(p, []))
            try:
                with open(f"/proc/{p}/status", encoding="utf-8") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
                            break
            except OSError:
                continue
        return total

    def _run(self):
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._tree_rss_kb())
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)


def _spawn_worker(spec, n):
    spec_path = os.path.join(BENCH_OUT, f"_spec_{spec['chain']}_{spec['scenario']}.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False)
    env = {**os.environ,
           "RAYON_NUM_THREADS": str(n),
           "TOKENIZERS_PARALLELISM": "false",
           "HF_DATASETS_CACHE": HF_CACHE,
           "HF_HOME": HF_CACHE}
    print(f"\n>>> 运行 {spec['chain']}/{spec['scenario']}（CPU 预算 {n}）...", flush=True)
    rss = None
    with subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--worker", spec_path],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    ) as proc:
        with RssTreeMeter(proc.pid) as meter:
            result, tail = None, []
            for line in proc.stdout:
                if line.startswith(RESULT_PREFIX):
                    result = json.loads(line[len(RESULT_PREFIX):])
                else:
                    print("    " + line, end="", flush=True)
                    tail.append(line)
                    tail = tail[-15:]
            proc.wait()
        rss = meter.peak / 1e6
    if result is None:
        result = {"error": "".join(tail)[-1500:]}
    result["peak_rss_gb"] = rss
    return result


def _manifest_stats(index_json):
    with open(index_json, encoding="utf-8") as f:
        gs = json.load(f)["global_stats"]
    return int(gs["total_tokens"]), int(gs.get("truncated_tokens", 0) or 0)


def _ensure_smoke_products(n):
    """smoke 模式：mini 数据现场构建 megadata 产物（幂等）。"""
    from megadata import DataBuilder
    from megadata.builder import BuildConfig, Marker

    _init_ids()
    jobs = [(SMOKE_PT_IN, SMOKE_PT_INDEX.replace("index.json", ""), "pt"),
            (SMOKE_SFT_IN, SMOKE_SFT_INDEX.replace("index.json", ""), "sft")]
    for inp, out_dir, mode in jobs:
        if os.path.exists(os.path.join(out_dir, "index.json")):
            continue
        os.makedirs(out_dir, exist_ok=True)
        if mode == "pt":
            cfg = BuildConfig(input_paths=[inp], output_dir=out_dir, mode="pt",
                              batch_size=PT_BS, batches_per_shard=0, compression="zstd",
                              template_str="", eos=Marker(text="<|endoftext|>", id=EOS_ID))
        else:
            cfg = BuildConfig(input_paths=[inp], output_dir=out_dir, mode="sft",
                              batch_size=SFT_BS, batches_per_shard=0, compression="zstd",
                              template_str=("{% for m in conversations %}<|im_start|>{{ m.role }}\n"
                                            "{{ m.content }}<|im_end|>\n{% endfor %}"),
                              turn_start=Marker(id=TURN_START_ID), turn_end=Marker(id=TURN_END_ID))
        print(f"[smoke] 构建 {mode}: {inp} -> {out_dir}", flush=True)
        DataBuilder(tokenizer_path=TOKENIZER_DIR, max_seq_len=2048).build_dataset(cfg)


# ======================================================================
# 报告
# ======================================================================
def _fmt_cell(run, key, fmt, na="—"):
    if run is None or "error" in run:
        return "ERR"
    v = run.get(key)
    return na if v is None else fmt.format(v)


def _report(all_runs, meta):
    lines = ["# 外部基线对比：megadata_rs vs HF datasets vs PyTorch DataLoader", ""]
    for sc in ("pt", "sft"):
        runs = all_runs.get(sc, {})
        if not runs:
            continue
        sc_name = ("PT 流式 packed 读" if sc == "pt" else "SFT 变长监督批")
        bs, msl = (PT_BS, PT_MSL) if sc == "pt" else (SFT_BS, SFT_MSL)
        exp = meta[sc]["expected_tokens"]
        lines += [f"## {sc.upper()}：{sc_name}（bs={bs}，msl={msl}，"
                  f"有效 token {exp:,}）", "",
                  "| 链路 | 预处理 | 读取 | 端到端 | 有效吞吐 | 峰值RSS(树) | 首批 | p50/p99 | 校验 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for chain, label in (("megadata", "megadata_rs (BatchStream)"),
                             ("hf", "HF datasets (arrow缓存+DataLoader)"),
                             ("torch", "PyTorch DataLoader (jsonl现算)")):
            r = runs.get(chain)
            if r is None:
                continue
            if "error" in r:
                lines.append(f"| {label} | ERR | ERR | ERR | ERR | {r['peak_rss_gb']:.2f}GB | — | — | FAIL |")
                continue
            pre = r.get("preprocess_s")
            if chain == "megadata" and not meta["smoke"]:
                pre_cell, pre_val = f"{PHASE9_BUILD_S[sc]:.0f}s†", PHASE9_BUILD_S[sc]
            elif pre is None:
                pre_cell, pre_val = "—", 0.0
            else:
                pre_cell, pre_val = f"{pre:.0f}s", pre
            e2e = pre_val + r["read_s"]
            ok = all(c["ok"] for c in r["checks"]) if r["checks"] else False
            lines.append(
                f"| {label} | {pre_cell} | {r['read_s']:.0f}s | {e2e:.0f}s "
                f"| {r['useful_tokens'] / r['read_s'] / 1e6:.1f} Mtok/s "
                f"| {r['peak_rss_gb']:.2f}GB | {r['first_batch_s']:.2f}s "
                f"| {r['p50_ms']:.2f}/{r['p99_ms']:.2f}ms | {'PASS' if ok else 'FAIL'} |")
        if sc == "sft":
            for chain in ("hf", "torch"):
                r = runs.get(chain)
                if r and "error" not in r and r["pad_tokens"]:
                    total_alloc = r["useful_tokens"] + r["pad_tokens"]
                    lines.append(f"- {chain} SFT padding 浪费率：{r['pad_tokens'] / total_alloc * 100:.1f}%"
                                 "（megadata 构建期 packing，0 padding）")
        lines.append("")
    lines += [
        "## 口径与公平性说明", "",
        f"- CPU 预算统一 {meta['n']} 核：megadata `RAYON_NUM_THREADS={meta['n']}`，"
        f"HF `map(num_proc={meta['n']})`，DataLoader `num_workers={meta['n']}`。",
        "- † megadata 构建期引用 phase9 §5.6 实测（同机、RAYON_NUM_THREADS=26）；HF 预处理为现场实测；"
        "DataLoader 无预处理生态（成本融合进读取）。",
        "- 三链路使用同一个 bbpe 分词器的 Python 包装（token 流一致，可比）；megadata 原生路径在 "
        "Rust 内直连分词、无逐文档 Python 调用开销，属于架构差异的一部分。",
        "- PT 三链路 token 流逐位同构（逐 doc encode_ordinary+eos 后全局拼接切块），吞吐直接可比；"
        "SFT 基线按生态惯例 pad batching，megadata 为构建期 packing（0 padding），浪费率已单列。",
        "- 读取顺序 megadata → HF → DataLoader：首遍之后输入全部进 page cache（251GB 内存），"
        "后续链路磁盘 IO 近零，对 megadata 略不利。",
        "- RSS 为父进程采样的**进程树峰值**（含 HF num_proc 子进程与 DataLoader worker）。",
        "- p50/p99 为主进程 next() 延迟；DataLoader 多 worker 预取下该值近似队列弹出耗时，"
        "吞吐才是有效指标。",
    ]
    if meta.get("smoke"):
        lines.insert(1, "> **SMOKE 模式（mini 数据）**：仅验证管线与守恒校验，数字不代表规模性能。")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description="外部基线对比基准")
    ap.add_argument("--scenario", choices=["pt", "sft", "all"], default="all")
    ap.add_argument("--smoke", action="store_true", help="mini 数据快速验证管线")
    ap.add_argument("--n", type=int, default=int(os.environ.get("BENCH_CPU", "26")), help="CPU 预算")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        _worker_main(args.worker)
        return

    os.makedirs(BENCH_OUT, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)
    smoke = args.smoke
    if smoke:
        _ensure_smoke_products(args.n)

    scenarios = [s for s in ("pt", "sft") if args.scenario in (s, "all")]
    meta = {"n": args.n, "smoke": smoke}
    all_runs = {}
    for sc in scenarios:
        if sc == "pt":
            inp = SMOKE_PT_IN if smoke else PT_IN
            index_json = SMOKE_PT_INDEX if smoke else SCALE_PT_INDEX
            bs, msl = PT_BS, PT_MSL
        else:
            inp = SMOKE_SFT_IN if smoke else SFT_IN
            index_json = SMOKE_SFT_INDEX if smoke else SCALE_SFT_INDEX
            bs, msl = SFT_BS, SFT_MSL
        if not os.path.exists(index_json):
            print(f"[error] megadata 产物不存在: {index_json}（全量模式需先运行 tools/verify/phase9_scale.py）",
                  flush=True)
            sys.exit(2)
        total, trunc = _manifest_stats(index_json)
        meta[sc] = {"input": inp, "index_json": index_json, "bs": bs, "msl": msl,
                    "expected_tokens": total,
                    "expected_baseline_tokens": total + (trunc if sc == "sft" else 0)}
        runs = {}
        for chain in ("megadata", "hf", "torch"):
            spec = {"chain": chain, "scenario": sc, "input": inp, "index_json": index_json,
                    "bs": bs, "msl": msl, "n": args.n,
                    "expected_tokens": total,
                    "expected_baseline_tokens": meta[sc]["expected_baseline_tokens"]}
            r = _spawn_worker(spec, args.n)
            runs[chain] = r
        all_runs[sc] = runs

    md = _report(all_runs, meta)
    with open(os.path.join(BENCH_OUT, "compare_baselines.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(BENCH_OUT, "compare_baselines.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "runs": all_runs}, f, ensure_ascii=False, indent=2)
    print("\n" + md, flush=True)
    print(f"[done] 结果写入 {BENCH_OUT}/compare_baselines.md", flush=True)


if __name__ == "__main__":
    main()
