"""阶段8：HF/第三方分词器端到端验证（Python 回调路径，PT + SFT 双链）。

与 phase0~7（bbpe 默认路径）互补：本阶段用 `tokenizers` 库在本地训练一个标准
HuggingFace BPE 分词器（ByteLevel，special token 训练后添加），经
`HfTokenizerAdapter`（Python 分词回调路径）走完整的 build → 落盘 → 读取链路，
并用**官方 `tokenizers.Tokenizer` 独立实例**复算全部真值（与被测路径不共享任何
分词代码）。

覆盖：
- PT：有效 token 流逐位 == 独立复算（ordinary 语义：标记串按普通文本切碎 + eos）、
  文档边界与解码往返、读头流 == 落盘流、manifest 分词器指纹；
- SFT：文档序列无漏无重、labels 监督掩码逐位 == 独立状态机复算、next-token 不变式、
  position_ids、row_lens/cu/padding 契约、语义级 assistant 字符区间核对、
  截断审计（F2：truncated_docs / truncated_tokens）、注入检测不误报（F7）。

本阶段自含构建（不依赖 phase0 的 builds.json），可单独运行：
    .venv/bin/python tools/verify/phase8_hf.py
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

from megadata import DataBuilder, HfTokenizerAdapter
from megadata.builder import BuildConfig, Marker

try:
    from tokenizers import Tokenizer, decoders, models, trainers
    from tokenizers.pre_tokenizers import ByteLevel
except ImportError as e:  # pragma: no cover - 环境缺软依赖时显式失败（验证框架不静默跳过）
    print(f"FAIL: phase8 requires the `tokenizers` package: {e}", flush=True)
    sys.exit(1)

R = V.Report("PHASE8-HF")

# ================================================================
# 1. 训练一个标准 HF BPE 分词器（真·HuggingFace tokenizer.json）
#    语料取自 data/*.jsonl 前 N 行（确定性子采样），保证被编码的真实数据
#    与训练语料同分布、BPE merge 行为非平凡。
# ================================================================
N_PT_TRAIN, N_SFT_TRAIN = 3000, 2000
TRAIN_CORPUS = V.read_pt_texts()[:N_PT_TRAIN] + [
    V.render_chat(c) for c in V.read_sft_convs()[:N_SFT_TRAIN]
]
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

hf = Tokenizer(models.BPE())
hf.pre_tokenizer = ByteLevel(add_prefix_space=False)  # 标准 GPT-2 正则切分；不注入前导空格 → 解码无损
hf.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=600,
    min_frequency=1,
    initial_alphabet=ByteLevel.alphabet(),
)
hf.train_from_iterator(TRAIN_CORPUS, trainer=trainer)
hf.add_special_tokens(SPECIALS)  # 训练后添加：id 落在 merges 之后

TOK_DIR = os.path.join(V.OUT, "hf_tok")
os.makedirs(TOK_DIR, exist_ok=True)
HF_TOK_PATH = os.path.join(TOK_DIR, "tokenizer.json")
with open(HF_TOK_PATH, "w", encoding="utf-8") as f:
    f.write(hf.to_str())  # 老版 tokenizers 无 to_file，to_str 序列化等价

# 两个**相互独立**的实例：official 做真值，adapter 走被测路径
official = Tokenizer.from_file(HF_TOK_PATH)
ADAPTER = HfTokenizerAdapter.from_file(HF_TOK_PATH)
IM_ID = official.token_to_id("<|im_start|>")
EM_ID = official.token_to_id("<|im_end|>")
EOS_ID_HF = official.token_to_id("<|endoftext|>")

R.check("HF 分词器可从 tokenizer.json 加载且 special id 互异",
        None not in (IM_ID, EM_ID, EOS_ID_HF) and len({IM_ID, EM_ID, EOS_ID_HF}) == 3,
        f"vocab={official.get_vocab_size()} im={IM_ID} em={EM_ID} eos={EOS_ID_HF}")


def hf_build(name, *, mode, inputs, batch_size, max_seq_len, batches_per_shard=16,
             compression="none", template_str=None, eos=None, turn_start=None, turn_end=None):
    """HF adapter 路径构建（vlib.build 走 bbpe 固定路径，此处参数化）。"""
    out_dir = os.path.join(V.OUT, name)
    V._clear_dir(out_dir)
    cfg = BuildConfig(
        input_paths=list(inputs), output_dir=out_dir, mode=mode,
        batch_size=batch_size, batches_per_shard=batches_per_shard,
        compression=compression, template_str=template_str or "",
        eos=eos or Marker(), turn_start=turn_start or Marker(), turn_end=turn_end or Marker(),
    )
    b = DataBuilder(tokenizer=ADAPTER, max_seq_len=max_seq_len)
    summary = b.build_dataset(cfg)
    return out_dir, summary, b


# ================================================================
# 2. PT 链：ordinary 语义 + eos，全量 20000 篇
# ================================================================
pt_texts = V.read_pt_texts()
out_pt, summary_pt, _ = hf_build(
    "hf_pt", mode="pt", inputs=[V.PT_JSONL], batch_size=8, max_seq_len=256,
    eos=Marker(text="<|endoftext|>", id=EOS_ID_HF),
)
R.check("[PT] 构建产出与输入篇数一致", summary_pt["total_shards"] >= 1,
        f"batches={summary_pt['total_batches']} shards={summary_pt['total_shards']}")

# 独立复算：ordinary 语义（标记串按普通文本切碎）——与适配器 PT 方向同一开关语义
official.encode_special_tokens = True
exp_pt = []
for t in pt_texts:
    exp_pt.extend(official.encode(t, add_special_tokens=False).ids)
    exp_pt.append(EOS_ID_HF)
official.encode_special_tokens = False

got_pt = V.pt_stream(out_pt)
R.check("[PT] 落盘有效流逐位 == 独立复算(ordinary 语义 + eos, 顺序)",
        got_pt == exp_pt, f"got={len(got_pt)} exp={len(exp_pt)} "
        f"first_diff={next((i for i, (a, b) in enumerate(zip(got_pt, exp_pt)) if a != b), None)}")

# 文档边界还原 + 解码往返（ByteLevel 解码器无损）
docs, cur = [], []
for t in got_pt:
    if t == EOS_ID_HF:
        docs.append(cur)
        cur = []
    else:
        cur.append(t)
R.check("[PT] 文档数 == 输入行数(无漏/无重, 含空文本篇)",
        len(docs) == len(pt_texts), f"docs={len(docs)} input={len(pt_texts)}")
bad_rt = [i for i, (d, t) in enumerate(zip(docs, pt_texts)) if official.decode(d) != t]
R.check("[PT] 每篇文档解码往返 == 原始文本(ByteLevel 无损)", not bad_rt, f"{bad_rt[:3]}")

# 读头 == 落盘 == 真值（三方可对齐）。
# 注意：shuffle="none" 只关闭 batch 级置换，shard 访问顺序仍由 (seed, epoch) 决定
# （F6 文档化行为）——要与落盘顺序逐位对齐，显式指定 shard_order="sequential"。
read_ids = []
for b in V.read_pt_batches(os.path.join(out_pt, "index.json"), 8, 256,
                           shard_order="sequential"):
    read_ids.extend(b["valid"])
R.check("[PT] 读头流 == 独立复算(shard_order=sequential, 读取路径同错位自由)",
        read_ids == exp_pt, f"read={len(read_ids)}")
read_any = []
for b in V.read_pt_batches(os.path.join(out_pt, "index.json"), 8, 256):
    read_any.extend(b["valid"])
R.check("[PT] 默认 shard 序(random)读头流 == 真值的多重集(只换序不换数)",
        sorted(read_any) == sorted(exp_pt), f"read={len(read_any)}")

# manifest 分词器指纹（T8）
fp = V.manifest(out_pt).get("tokenizer") or {}
R.check("[PT] manifest 分词器指纹(backend/identity/vocab_size)",
        fp.get("backend") == "python" and fp.get("identity") == HF_TOK_PATH
        and fp.get("vocab_size") == official.get_vocab_size(),
        f"{ {k: fp.get(k) for k in ('backend', 'identity', 'vocab_size')} }")

# ================================================================
# 3. SFT 链：监督掩码语义，全量 15000 条对话（msl=256 无截断）
# ================================================================
convs = V.read_sft_convs()
out_sft, summary_sft, builder_sft = hf_build(
    "hf_sft", mode="sft", inputs=[V.SFT_JSONL], batch_size=8, max_seq_len=256,
    template_str=V.TEMPLATE, turn_start=Marker(id=IM_ID), turn_end=Marker(id=EM_ID),
)

exp_ids, exp_tokens = [], []
for c in convs:
    e = official.encode(V.render_chat(c), add_special_tokens=False)  # special 识别为单 token
    exp_ids.append(list(e.ids))
    exp_tokens.append(list(e.tokens))

docs_sft = V.sft_docs(out_sft)
R.check("[SFT] 还原文档数 == 输入对话数(无漏/无重)", len(docs_sft) == len(convs),
        f"docs={len(docs_sft)}")
R.check("[SFT] 文档 token 序列逐条 == 独立复算(渲染/顺序/内容)",
        docs_sft == exp_ids,
        f"first_diff={next((i for i, (a, b) in enumerate(zip(docs_sft, exp_ids)) if a != b), None)}")

# 独立复现监督状态机（与 phase2 同规则，token 串来自 official 实例）
TU, TE = IM_ID, EM_ID


def expected_supervision(ids, tokstrs):
    n = len(ids)
    sup = [False] * n
    in_asst = False
    for i in range(n):
        tok = tokstrs[i] if i < len(tokstrs) else ""
        if ids[i] == TU or (tok and "<|im_start|>" in tok):
            role = []
            for j in range(i + 1, min(i + 8, n)):
                nx = tokstrs[j] if j < len(tokstrs) else ""
                if "\n" in nx or "<|im_end|>" in nx or "<|im_start|>" in nx:
                    break
                role.append(nx)
            in_asst = "".join(role).strip().startswith("assistant")
        if in_asst:
            sup[i] = True
        if ids[i] == TE or (tok and "<|im_end|>" in tok):
            if in_asst:
                sup[i] = True
                in_asst = False
    return sup


def expected_labels(ids, sup):
    lab = [-100] * len(ids)
    for i in range(len(ids) - 1):
        if sup[i]:
            lab[i] = ids[i + 1]
    return lab


# 从落盘 shard 逐文档抽取 (ids, labels, pos)
doc_triples = []
for b in V.shard_batches(out_sft):
    cu = b["cu"]
    di = 0
    for r, rl in enumerate(b["row_lens"]):
        pos = 0
        while di < len(cu) - 1 and pos < rl:
            L = cu[di + 1] - cu[di]
            if L <= 0:
                di += 1
                continue
            doc_triples.append((
                b["rows"][r][pos:pos + L],
                b["label_rows"][r][pos:pos + L],
                b["pos_rows"][r][pos:pos + L],
            ))
            pos += L
            di += 1

bad_sup, bad_next, bad_pos = [], [], []
for gdi, (ids, lab, pss) in enumerate(doc_triples):
    sup = expected_supervision(ids, exp_tokens[gdi])
    if lab != expected_labels(ids, sup):
        bad_sup.append(gdi)
    for t in range(len(ids) - 1):
        if lab[t] != -100 and lab[t] != ids[t + 1]:
            bad_next.append((gdi, t))
    if pss != list(range(len(ids))):
        bad_pos.append(gdi)
R.check("[SFT] 监督掩码逐位 == 独立状态机复算(assistant 段监督)", not bad_sup, f"{bad_sup[:3]}")
R.check("[SFT] next-token 不变式(labels[t]==ids[t+1])", not bad_next, f"{bad_next[:3]}")
R.check("[SFT] position_ids == 文档内相对位置", not bad_pos, f"{bad_pos[:3]}")

# 行级契约：row_lens / cu / padding
bad_rl, bad_pad, bad_cu = [], [], []
for bi, b in enumerate(V.shard_batches(out_sft)):
    if sum(b["row_lens"]) != b["cu"][-1]:
        bad_rl.append(("sum", bi))
    if any(l < 0 or l > 256 for l in b["row_lens"]):
        bad_rl.append(("range", bi))
    if b["cu"][0] != 0 or any(y < x for x, y in zip(b["cu"], b["cu"][1:])):
        bad_cu.append(bi)
    for r, rl in enumerate(b["row_lens"]):
        if any(t != 0 for t in b["rows"][r][rl:]):
            bad_pad.append(("ids", bi, r))
        if any(t != -100 for t in b["label_rows"][r][rl:]):
            bad_pad.append(("labels", bi, r))
R.check("[SFT] row_lens 契约(sum==cu[-1], ≤max_seq_len)", not bad_rl, f"{bad_rl[:3]}")
R.check("[SFT] cu_seqlens 单调且起点为 0", not bad_cu, f"{bad_cu[:3]}")
R.check("[SFT] 行内 padding: ids=pad_id(0) 且 labels=-100", not bad_pad, f"{bad_pad[:3]}")

# 语义级：监督位集合恰为 assistant 轮的**字节区间**（独立于状态机实现）。
# 注意：ByteLevel 的 token 串是"字节 → 可见 unicode 字符"的映射结果（Ġ/Ċ 等），
# 不是原文本身——字符级 span 需先把每个 token 还原成原始字节，在字节空间匹配。
def _bytes_to_unicode_map() -> dict:
    """GPT-2 / HF ByteLevel 的字节 ↔ 字符映射表的逆表（字符 → 字节）。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_INV_BYTE = _bytes_to_unicode_map()
_SPECIAL_STRS = set(SPECIALS)


def _token_bytes(tok: str) -> bytes:
    """单个 token 串 → 原始字节（special token 用其字面内容的 UTF-8 字节）。"""
    if tok in _SPECIAL_STRS:
        return tok.encode("utf-8")
    return bytes(_INV_BYTE[ch] for ch in tok)


bad_sem = []
for gdi, (ids, lab, _pss) in enumerate(doc_triples):
    tk = exp_tokens[gdi]
    raw_bytes = [_token_bytes(t) for t in tk]
    spans, off = [], 0
    for bs in raw_bytes:
        spans.append((off, off + len(bs)))
        off += len(bs)
    full = b"".join(raw_bytes)
    want = set()
    for m in re.finditer(rb"<\|im_start\|>assistant\n", full):
        s0 = m.start()
        e0 = full.index(b"<|im_end|>", m.end()) + len(b"<|im_end|>")
        for ti, (a, bb) in enumerate(spans):
            if a >= s0 and bb <= e0:
                want.add(ti)
    got = {i for i in range(len(ids)) if lab[i] != -100}
    want.discard(len(ids) - 1)   # 末位无 next token，监督位必被剔除
    got -= {len(ids) - 1}
    if got != want:
        bad_sem.append((gdi, sorted(want ^ got)[:5]))
R.check("[SFT] 监督位集合 == 文本层 assistant 轮字节区间(语义正确)", not bad_sem, f"{bad_sem[:3]}")

# F7：干净数据不误报注入；msl=256 下该语料无截断 → manifest 省略 truncated_* 字段
snap = builder_sft.get_progress()
R.check("[SFT] 干净数据注入检测不计数(F7 不误报)", snap["docs_flagged_injection"] == 0,
        f"flagged={snap['docs_flagged_injection']}")
gs_a = V.manifest(out_sft)["global_stats"]
R.check("[SFT] 无截断时 manifest 不含 truncated_* 字段(向后兼容)",
        "truncated_docs" not in gs_a and "truncated_tokens" not in gs_a, "")

# ================================================================
# 4. SFT 截断链（msl=64 + 合成超长对话）：F2 审计 + 前缀一致
#    （HF BPE 对本语料压缩率高，msl=64 时原语料无超长文档——F2 计数的前提
#    不成立，故用确定性合成的超长 user 轮保证截断必然发生。）
# ================================================================
trunc_records = []
for i in range(400):
    # 纯 ASCII 短对话：ByteLevel 下每字符 1 字节，渲染串 ≈ 15-20 token，稳定 < 64
    trunc_records.append({"conversations": [
        {"role": "user", "content": f"q{i}"},
        {"role": "assistant", "content": f"a{i} ok"},
    ]})
for i in range(100):
    trunc_records.append({"conversations": [
        {"role": "user", "content": "很长很长的用户提问" * 300},
        {"role": "assistant", "content": "短答"},
    ]})
os.makedirs(V.EDGE, exist_ok=True)
trunc_jsonl = os.path.join(V.EDGE, "hf_trunc.jsonl")
with open(trunc_jsonl, "w", encoding="utf-8") as f:
    for o in trunc_records:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

out_t, _s, builder_t = hf_build(
    "hf_sft_trunc64", mode="sft", inputs=[trunc_jsonl], batch_size=8, max_seq_len=64,
    template_str=V.TEMPLATE, turn_start=Marker(id=IM_ID), turn_end=Marker(id=EM_ID),
)
t_ids, t_tokens = [], []
for c in trunc_records:
    e = official.encode(V.render_chat(c), add_special_tokens=False)
    t_ids.append(list(e.ids))
    t_tokens.append(list(e.tokens))
n_trunc = sum(1 for d in t_ids if len(d) > 64)
assert n_trunc == 100, f"合成数据应恰有 100 篇超长，实际 {n_trunc}"

docs_t = V.sft_docs(out_t)
R.check("[trunc64] 超长文档 tail 截断到 max_seq_len(前缀一致)",
        docs_t == [d[:64] for d in t_ids], f"docs={len(docs_t)}")
lost = sum(len(d) - 64 for d in t_ids if len(d) > 64)
kept = sum(min(len(d), 64) for d in t_ids)
gs = V.manifest(out_t)["global_stats"]
R.check("[trunc64] manifest truncated_docs/truncated_tokens == 独立复算(F2)",
        gs.get("truncated_docs") == n_trunc and gs.get("truncated_tokens") == lost
        and gs.get("total_tokens") == kept,
        f"docs={gs.get('truncated_docs')}/{n_trunc} tokens={gs.get('truncated_tokens')}/{lost}")
snap_t = builder_t.get_progress()
R.check("[trunc64] 进度快照截断/废样本计数(F2)",
        snap_t["docs_truncated"] == n_trunc and snap_t["all_masked_docs"] == 100
        and snap_t["tokens_dropped_to_truncation"] == lost,
        f"trunc={snap_t['docs_truncated']} masked={snap_t['all_masked_docs']}")

ok = R.done()
sys.exit(0 if ok else 1)
