"""阶段9：百万行级验证（按需运行；需要 data/ 下的大数据集）。

- PT：data/pretrain_t2t.jsonl（~847 万行 / 7.8GB）
- SFT：data/sft_t2t_mini_x.jsonl（~91 万条 / 1.7GB）

全量构建 + 全量读回，验证规模下的正确性不变式与资源行为：
- 构建端：docs_read == docs_tokenized == 非空行数（无静默丢弃）；吞吐与峰值 RSS；
- PT 读回（shard_order="sequential" 顺序扫描）：
  * **eos 计数守恒**：流中 special eos 只能来自逐 doc 追加（ordinary 编码不可能
    产生 special id）→ eos 总数 == 输入行数（无漏/无重的强不变式）；
  * **F3 规模不变式**：非满批（cu[-1] < batch 容量）全数据集至多 1 个且必在最后；
  * **token 守恒**：读回有效 token 总数 == manifest global_stats.total_tokens；
  * 抽样 64 篇逐位真值（bbpe_tokenizer 独立复算 encode_ordinary + eos）；
- SFT 读回：文档数 == 输入行数（一行一 doc，无漏/无重）、row_lens/cu/padding 契约、
  抽样 64 条逐位真值（渲染文本 encode + 监督掩码状态机复算）；
- 读回吞吐（batches/s、tokens/s）。

运行：.venv/bin/python tools/verify/phase9_scale.py（不进 phase0~8 常规流水线）
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

from megadata import BatchStream, DataBuilder
from megadata.builder import BuildConfig, Marker

R = V.Report("PHASE9-SCALE")
V.init_ids()

PT_INPUT = os.path.join(V.DATA, "pretrain_t2t.jsonl")
SFT_INPUT = os.path.join(V.DATA, "sft_t2t_mini_x.jsonl")
PT_BS, PT_MSL = 16, 2048
SFT_BS, SFT_MSL = 8, 2048
N_SAMPLE = 64
# PHASE9_REUSE=1：产物已存在时跳过构建（读回+真值断言照跑），便于迭代调试真值逻辑
REUSE = os.environ.get("PHASE9_REUSE") == "1"


class RssMeter:
    """后台线程采样进程峰值 RSS（KB），验证规模下内存有界（不随数据量增长）。"""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                with open("/proc/self/status", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            self.peak = max(self.peak, int(line.split()[1]))
                            break
            except OSError:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=2)


def scan_lines(path, sample_idx):
    """单遍流式扫描：统计非空行数，仅保留抽样行文本（避免把 7.8GB 读进内存）。"""
    want = set(sample_idx)
    n, picked = 0, {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if n in want:
                picked[n] = json.loads(line)
            n += 1
    return n, picked


def sample_indices(n):
    return sorted({i * n // N_SAMPLE for i in range(N_SAMPLE)})


def _build(name, *, mode, inputs, batch_size, max_seq_len, eos=None, turn_start=None, turn_end=None):
    out_dir = os.path.join(V.OUT, name)
    if REUSE and os.path.exists(os.path.join(out_dir, "index.json")):
        gs = json.load(open(os.path.join(out_dir, "index.json"), encoding="utf-8"))["global_stats"]
        return out_dir, {"total_batches": gs["total_batches"], "total_shards": gs["total_shards"],
                         "total_tokens": gs["total_tokens"], "progress": None}, 0.0
    V._clear_dir(out_dir)
    cfg = BuildConfig(
        input_paths=list(inputs), output_dir=out_dir, mode=mode,
        batch_size=batch_size, batches_per_shard=0,  # 自动聚合（~128MB/shard），验证 F11/聚合规模行为
        compression="zstd", template_str="" if mode == "pt" else V.TEMPLATE,
        eos=eos or Marker(), turn_start=turn_start or Marker(), turn_end=turn_end or Marker(),
    )
    b = DataBuilder(tokenizer_path=V.TOKENIZER_DIR, max_seq_len=max_seq_len)
    t0 = time.time()
    summary = b.build_dataset(cfg)
    dt = time.time() - t0
    summary["progress"] = b.get_progress()
    return out_dir, summary, dt


# ================================================================
# 1. PT：847 万行全量构建 + 全量读回
# ================================================================
print(f"[phase9] PT 输入: {PT_INPUT}", flush=True)
pt_n, _ = scan_lines(PT_INPUT, set())
_, pt_picked = scan_lines(PT_INPUT, set(sample_indices(pt_n)))
pt_sampled = set(sample_indices(pt_n))

with RssMeter() as pt_rss:
    out_pt, s_pt, pt_dt = _build(
        "scale_pt", mode="pt", inputs=[PT_INPUT], batch_size=PT_BS, max_seq_len=PT_MSL,
        eos=Marker(text="<|endoftext|>", id=V.EOS_ID),
    )
in_mb = os.path.getsize(PT_INPUT) / 1e6
if s_pt["progress"] is not None:
    R.check("[PT] docs_read == docs_tokenized == 非空行数（无静默丢弃）",
            s_pt["progress"]["docs_read"] == s_pt["progress"]["docs_tokenized"] == pt_n,
            f"docs={s_pt['progress']['docs_tokenized']}/{pt_n} shards={s_pt['total_shards']} "
            f"batches={s_pt['total_batches']} tokens={s_pt['total_tokens']:,} "
            f"build={pt_dt:.0f}s {in_mb / pt_dt:.0f}MB/s {s_pt['total_tokens'] / pt_dt / 1e6:.2f}Mtok/s "
            f"peakRSS={pt_rss.peak / 1e6:.2f}GB")
else:
    print(f"[phase9] PT 复用已有产物：shards={s_pt['total_shards']} "
          f"batches={s_pt['total_batches']} tokens={s_pt['total_tokens']:,}", flush=True)

eos_total = nonfull = 0
tok_total = 0
batches = 0
collected = {}
cur_doc = -1
last_nonfull = False
t0 = time.time()
loader = BatchStream(index_json_path=os.path.join(out_pt, "index.json"),
                     batch_size=PT_BS, max_seq_len=PT_MSL, world_size=1, rank=0,
                     shuffle=False, shard_order="sequential")
for b in loader:
    batches += 1
    ids = b["input_ids"]          # torch 张量 (B, S)
    cu = b["cu_seqlens"]
    valid = int(cu[-1])
    tok_total += valid
    flat = ids.reshape(-1)[:valid]
    last_nonfull = valid < ids.numel()
    nonfull += int(last_nonfull)
    eos_mask = flat == V.EOS_ID
    eos_total += int(eos_mask.sum())
    eos_pos = eos_mask.nonzero().flatten().tolist()
    start = 0
    for ep in eos_pos:
        cur_doc += 1
        if cur_doc in pt_sampled:
            collected.setdefault(cur_doc, []).extend(flat[start:ep].tolist())
        start = ep + 1
    if start < len(flat) and cur_doc + 1 in pt_sampled:
        # 无 eos 结尾的尾段 = 长文档被跨批切分的延续段（逻辑 doc 尚未闭合）
        collected.setdefault(cur_doc + 1, []).extend(flat[start:].tolist())
pt_read_dt = time.time() - t0

R.check("[PT] eos 计数 == 输入行数（special eos 只能来自逐 doc 追加，无漏/无重）",
        eos_total == pt_n, f"eos={eos_total} lines={pt_n}")
R.check("[PT] token 守恒：读回有效 token 总数 == manifest total_tokens",
        tok_total == s_pt["total_tokens"], f"walk={tok_total:,} manifest={s_pt['total_tokens']:,}")
R.check("[PT] F3 规模不变式：非满批至多 1 个且必在最后",
        nonfull <= 1 and (nonfull == 0 or last_nonfull), f"nonfull={nonfull} batches={batches}")
R.check("[PT] 读回吞吐统计",
        batches == s_pt["total_batches"],
        f"{batches / pt_read_dt:.0f}batches/s {tok_total / pt_read_dt / 1e6:.1f}Mtok/s "
        f"({pt_read_dt:.0f}s)")

bad = []
for i, rec in pt_picked.items():
    txt = rec["text"] if isinstance(rec, dict) else rec
    exp = list(V.tok().encode_ordinary(txt))  # eos 是分隔符本身，collected 片不含它（eos 计数另行守恒）
    got = collected.get(i)
    if got != exp:
        bad.append((i, len(got or []), len(exp)))
R.check(f"[PT] 抽样 {N_SAMPLE} 篇逐位真值 == 独立复算(encode_ordinary, eos 分隔另测)", not bad, f"{bad[:3]}")

# ================================================================
# 2. SFT：91 万条对话全量构建 + 全量读回
# ================================================================
print(f"[phase9] SFT 输入: {SFT_INPUT}", flush=True)
sft_n, _ = scan_lines(SFT_INPUT, set())
_, sft_picked = scan_lines(SFT_INPUT, set(sample_indices(sft_n)))
sft_sampled = set(sample_indices(sft_n))

with RssMeter() as sft_rss:
    out_sft, s_sft, sft_dt = _build(
        "scale_sft", mode="sft", inputs=[SFT_INPUT], batch_size=SFT_BS, max_seq_len=SFT_MSL,
        turn_start=Marker(id=V.TURN_START_ID), turn_end=Marker(id=V.TURN_END_ID),
    )
in_mb = os.path.getsize(SFT_INPUT) / 1e6
if s_sft["progress"] is not None:
    R.check("[SFT] docs_read == docs_tokenized == 非空行数（无静默丢弃）",
            s_sft["progress"]["docs_read"] == s_sft["progress"]["docs_tokenized"] == sft_n,
            f"docs={s_sft['progress']['docs_tokenized']}/{sft_n} shards={s_sft['total_shards']} "
            f"batches={s_sft['total_batches']} tokens={s_sft['total_tokens']:,} "
            f"build={sft_dt:.0f}s {in_mb / sft_dt:.0f}MB/s {s_sft['total_tokens'] / sft_dt / 1e6:.2f}Mtok/s "
            f"peakRSS={sft_rss.peak / 1e6:.2f}GB")
else:
    print(f"[phase9] SFT 复用已有产物：shards={s_sft['total_shards']} "
          f"batches={s_sft['total_batches']} tokens={s_sft['total_tokens']:,}", flush=True)
gs = V.manifest(out_sft)["global_stats"]
R.check("[SFT] 截断审计字段内部一致（有截断必有丢 token，反之亦然）",
        ("truncated_docs" in gs) == ("truncated_tokens" in gs),
        f"truncated_docs={gs.get('truncated_docs')} truncated_tokens={gs.get('truncated_tokens')}")

docs_seen = 0
tok_total = 0
bad_contract = []
collected_sft = {}
t0 = time.time()
loader = BatchStream(index_json_path=os.path.join(out_sft, "index.json"),
                     batch_size=SFT_BS, max_seq_len=SFT_MSL, world_size=1, rank=0,
                     shuffle=False, shard_order="sequential")
for b in loader:
    cu = b["cu_seqlens"]
    tok_total += int(cu[-1])
    ids_rows = b["input_ids"]
    lab_rows = b["labels"]
    pos_rows = b["position_ids"]
    row_lens = b["row_lens"]
    di = 0
    for r in range(len(row_lens)):
        rl = int(row_lens[r])
        pos = 0
        while di < len(cu) - 1 and pos < rl:
            L = int(cu[di + 1] - cu[di])
            if L <= 0:
                di += 1
                continue
            if L > SFT_MSL or int(pos_rows[r][pos]) != 0:
                bad_contract.append((docs_seen, "len/pos-head"))
            if docs_seen in sft_sampled:
                collected_sft[docs_seen] = (
                    ids_rows[r][pos:pos + L].tolist(),
                    lab_rows[r][pos:pos + L].tolist(),
                    pos_rows[r][pos:pos + L].tolist())
            pos += L
            di += 1
            docs_seen += 1
        if rl and any(t != 0 for t in ids_rows[r][rl:]):
            bad_contract.append((r, "pad-ids"))
sft_read_dt = time.time() - t0

R.check("[SFT] 还原文档数 == 输入行数（无漏/无重）", docs_seen == sft_n,
        f"docs={docs_seen} lines={sft_n}")
R.check("[SFT] token 守恒：读回 == manifest", tok_total == s_sft["total_tokens"],
        f"walk={tok_total:,} manifest={s_sft['total_tokens']:,}")
R.check("[SFT] 行级契约（row_lens<=msl, position_ids 从 0 起, padding=pad_id）",
        not bad_contract, f"{bad_contract[:3]}")
R.check("[SFT] 读回吞吐统计",
        True, f"{(sft_n / SFT_BS) / sft_read_dt:.0f}batches/s "
              f"{tok_total / sft_read_dt / 1e6:.1f}Mtok/s ({sft_read_dt:.0f}s)")

bad = []
for i in sorted(sft_sampled):
    conv = sft_picked[i]
    text = V.render_chat(conv)
    e = V.tok().encode(text, allowed_special=["all"])
    ids_exp, toks_exp = list(e.ids), list(e.tokens)
    got = collected_sft.get(i)
    if got is None or got[0] != ids_exp:
        bad.append((i, "ids"))
        continue
    _ids, lab, pos = got
    # 独立状态机复算监督掩码（assistant 段监督；与 phase2 同一规则）
    sup, in_asst = [False] * len(ids_exp), False
    n = len(ids_exp)
    for k in range(n):
        tk = toks_exp[k] if k < len(toks_exp) else ""
        if ids_exp[k] == V.TURN_START_ID or (tk and "<|im_start|>" in tk):
            role = []
            for j in range(k + 1, min(k + 8, n)):
                nx = toks_exp[j] if j < len(toks_exp) else ""
                if "\n" in nx or "<|im_end|>" in nx or "<|im_start|>" in nx:
                    break
                role.append(nx)
            in_asst = "".join(role).strip().startswith("assistant")
        if in_asst:
            sup[k] = True
        if ids_exp[k] == V.TURN_END_ID or (tk and "<|im_end|>" in tk):
            if in_asst:
                sup[k] = True
                in_asst = False
    exp_lab = [-100] * n
    for k in range(n - 1):
        if sup[k]:
            exp_lab[k] = ids_exp[k + 1]
    if lab != exp_lab or pos != list(range(n)):
        bad.append((i, "labels/pos"))
R.check(f"[SFT] 抽样 {N_SAMPLE} 条逐位真值（ids+监督掩码+position_ids == 独立复算）",
        not bad, f"{bad[:3]}")

ok = R.done()
sys.exit(0 if ok else 1)
