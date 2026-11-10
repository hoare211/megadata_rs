"""阶段2：SFT（对话微调）数据正确性验证。

维度：完整性/顺序、模板渲染一致性、row_lens 与物理布局、labels 掩码语义（next-token、
仅 assistant 段监督、padding=-100、user 段绝不监督）、position_ids、超长文档截断。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

V.init_ids()
R = V.Report("PHASE2-SFT")
convs = V.read_sft_convs()
BUILDS = json.load(open(V.BUILDS_JSON))
TU = V.TURN_START_ID
TE = V.TURN_END_ID
PAD = 0


def enc_full(text):
    r = V.tok().encode(text, allowed_special=["all"])
    return list(r.ids), list(r.tokens)


exp_docs, exp_tokens = [], []
for c in convs:
    i, t = enc_full(V.render_chat(c))
    exp_docs.append(i)
    exp_tokens.append(t)

R.check("SFT 语料读取", len(convs) == 15000, f"convs={len(convs)}")
R.check("SFT 渲染+编码(独立复算)", all(len(d) > 0 for d in exp_docs),
        f"tokens={sum(len(d) for d in exp_docs)}")


def ref_batches(name):
    """从 Arrow shard 直接读顺序 epoch 的 (ids,labels,row_lens,cu)。"""
    import pyarrow as pa
    info = BUILDS[name]
    d = info["dir"]
    m = V.manifest(d)
    out = []
    for s in m["shards"]:
        with pa.memory_map(os.path.join(d, s["filename"]), "rb") as src:
            rd = pa.ipc.open_file(src)
            assert rd.num_record_batches == s["num_batches"], (rd.num_record_batches, s["num_batches"])
            for i in range(rd.num_record_batches):
                b = rd.get_batch(i)
                out.append({
                    "ids": b.column("input_ids").to_pylist(),
                    "labels": b.column("labels").to_pylist(),
                    "pos": b.column("position_ids").to_pylist(),
                    "cu": b.column("cu_seqlens").to_pylist()[0],
                    "row_lens": b.column("row_lens").to_pylist()[0],
                })
    return out


def docs_of_batch(b, msl):
    """从单个 batch 按 row_lens/cu 还原文档序列。"""
    docs, cu = [], b["cu"]
    di = 0
    for r, rl in enumerate(b["row_lens"]):
        pos = 0
        while di < len(cu) - 1 and pos < rl:
            L = cu[di + 1] - cu[di]
            if L <= 0:
                di += 1
                continue
            docs.append(b["ids"][r][pos:pos + L])
            pos += L
            di += 1
    return docs


def read_epoch(name, **kw):
    info = BUILDS[name]
    from megadata import BatchStream
    L = BatchStream(os.path.join(info["dir"], "index.json"),
                    batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                    world_size=1, rank=0, **kw)
    out = []
    for b in L:
        out.append({"ids": b["input_ids"].tolist(), "labels": b["labels"].tolist(),
                    "pos": b["position_ids"].tolist(), "cu": b["cu_seqlens"].tolist(),
                    "row_lens": b["row_lens"].tolist()})
    return out


# ---------------------------------------------------------------- 独立复现掩码状态机
def expected_supervision(ids, tokstrs):
    """按文档规则独立复现：assistant 段（含 turn_start..turn_end）为监督位。"""
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


# ---------------------------------------------------------------- 主校验
for name in ("sft_a", "sft_ref"):
    info = BUILDS[name]
    msl = info["max_seq_len"]
    ref = ref_batches(name)
    R.check(f"[{name}] Arrow 直读 batch 数 == manifest total_batches",
            len(ref) == info["total_batches"], f"{len(ref)} vs {info['total_batches']}")
    # 还原文档序列
    docs = [d for b in ref for d in docs_of_batch(b, msl)]
    R.check(f"[{name}] 还原文档数 == 输入对话数(无漏/无重)",
            len(docs) == len(convs), f"docs={len(docs)}")
    R.check(f"[{name}] 文档 token 序列逐条 == 独立复算(渲染/顺序/内容)",
            docs == exp_docs,
            f"docs={len(docs)} exp={len(exp_docs)} "
            f"first_diff={next((i for i,(a,b) in enumerate(zip(docs,exp_docs)) if a!=b), None)}")
    # row_lens / cu / padding 契约
    bad_rl, bad_pad, bad_cu = [], [], []
    for bi, b in enumerate(ref):
        cap = info["batch_size"] * msl
        if len(b["row_lens"]) != info["batch_size"]:
            bad_rl.append(("len", bi))
        if sum(b["row_lens"]) != b["cu"][-1]:
            bad_rl.append(("sum", bi))
        if any(l < 0 or l > msl for l in b["row_lens"]):
            bad_rl.append(("range", bi))
        if b["cu"][0] != 0 or any(y < x for x, y in zip(b["cu"], b["cu"][1:])):
            bad_cu.append(bi)
        for r, rl in enumerate(b["row_lens"]):
            if any(t != PAD for t in b["ids"][r][rl:]):
                bad_pad.append(("ids", bi, r))
            if any(t != -100 for t in b["labels"][r][rl:]):
                bad_pad.append(("labels", bi, r))
    R.check(f"[{name}] row_lens 契约(长度=B, sum=cu[-1], ≤max_seq_len)", not bad_rl, f"{bad_rl[:3]}")
    R.check(f"[{name}] cu_seqlens 单调且起点为 0", not bad_cu, f"{bad_cu[:3]}")
    R.check(f"[{name}] 行内 padding: input_ids=pad_id(0) 且 labels=-100", not bad_pad, f"{bad_pad[:3]}")
    # labels 语义
    bad_next, bad_sup, bad_pos = [], [], []
    gdi = 0
    for bi, b in enumerate(ref):
        di = 0
        for r, rl in enumerate(b["row_lens"]):
            pos = 0
            while di < len(b["cu"]) - 1 and pos < rl:
                L = b["cu"][di + 1] - b["cu"][di]
                if L <= 0:
                    di += 1
                    continue
                ids = b["ids"][r][pos:pos + L]
                lab = b["labels"][r][pos:pos + L]
                pss = b["pos"][r][pos:pos + L]
                tk = exp_tokens[gdi]
                sup = expected_supervision(ids, tk)
                exp_lab = expected_labels(ids, sup)
                if lab != exp_lab:
                    bad_sup.append((bi, r, pos, L, gdi))
                # next-token 不变式
                for t in range(len(ids) - 1):
                    if lab[t] != -100 and lab[t] != ids[t + 1]:
                        bad_next.append((bi, r, t))
                # position_ids 为文档内相对位置
                if pss != list(range(L)):
                    bad_pos.append((bi, r, pos, pss[:5]))
                pos += L
                di += 1
                gdi += 1
    R.check(f"[{name}] labels[t]∈有效位 ⇒ labels[t]==input_ids[t+1](next-token)", not bad_next, f"{bad_next[:3]}")
    R.check(f"[{name}] 监督位掩码 == 独立复现的 assistant 段规则(逐位)", not bad_sup, f"{bad_sup[:3]}")
    R.check(f"[{name}] position_ids == 文档内相对位置(0..L-1, 逐文档重置)", not bad_pos, f"{bad_pos[:3]}")

# ---- 语义级（文本对齐，独立于实现细节）：监督位必须恰为 assistant 轮的字符区间 ----
import re
bad_sem, bad_concat = [], []
gdi = 0
for bi, b in enumerate(ref_batches("sft_a")):
    di = 0
    for r, rl in enumerate(b["row_lens"]):
        pos = 0
        while di < len(b["cu"]) - 1 and pos < rl:
            L = b["cu"][di + 1] - b["cu"][di]
            if L <= 0:
                di += 1
                continue
            ids = b["ids"][r][pos:pos + L]
            lab = b["labels"][r][pos:pos + L]
            tk = exp_tokens[gdi]
            if "".join(tk) != V.render_chat(convs[gdi]):
                bad_concat.append(gdi)
            # token → 字符区间
            spans, off = [], 0
            for t in tk:
                spans.append((off, off + len(t)))
                off += len(t)
            text = "".join(tk)
            want = set()
            for m in re.finditer(r"<\|im_start\|>assistant\n", text):
                s0 = m.start()
                e0 = text.index("<|im_end|>", m.end()) + len("<|im_end|>")
                for ti, (a, bb) in enumerate(spans):
                    if a >= s0 and bb <= e0:
                        want.add(ti)
            got = {i for i in range(len(ids)) if lab[i] != -100}
            # 末位无法监督(无 next token)，从期望集中剔除
            want.discard(len(ids) - 1)
            got -= {len(ids) - 1}
            if got != want:
                bad_sem.append((bi, r, gdi, sorted(want ^ got)[:5]))
            pos += L
            di += 1
            gdi += 1
R.check("token 串拼接 == 原始渲染文本(分词无损)", not bad_concat, f"{bad_concat[:3]}")
R.check("监督位集合 == 文本层 assistant 轮字符区间(语义正确)",
        not bad_sem, f"{bad_sem[:3]}")

# 交叉：sft_a(msl=256) 与 sft_ref(msl=512) 文档序列一致（无截断场景）
docs_a = [d for b in ref_batches("sft_a") for d in docs_of_batch(b, 256)]
docs_r = [d for b in ref_batches("sft_ref") for d in docs_of_batch(b, 512)]
R.check("SFT 文档序列与 max_seq_len(256/512) 无关(无截断时)", docs_a == docs_r,
        f"{len(docs_a)} vs {len(docs_r)}")

# 交叉：Arrow zstd 与 none 一致
docs_z = [d for b in ref_batches("sft_zstd") for d in docs_of_batch(b, 256)]
R.check("sft_zstd(Arrow zstd) 解压结果 == sft_a", docs_z == docs_a, "")

# 截断场景：msl=64
info = BUILDS["sft_trunc64"]
ref = ref_batches("sft_trunc64")
docs_t = [d for b in ref for d in docs_of_batch(b, 64)]
exp_trunc = [d[:64] for d in exp_docs]
R.check("[sft_trunc64] 超长文档被 tail 截断到 max_seq_len(前缀一致)",
        docs_t == exp_trunc, f"docs={len(docs_t)}")
lost = sum(len(d) - len(d[:64]) for d in exp_docs)
kept = sum(len(d[:64]) for d in exp_docs)
R.check("[sft_trunc64] 数据量损失可量化(非无声)",
        info["total_tokens"] == kept and lost > 0,
        f"保留={kept} 丢弃={lost} ({(lost/(kept+lost))*100:.1f}%) manifest_total_tokens={info['total_tokens']}")

# ---------------------------------------------------------------- F2：截断统计审计
# 修复后：截断不再是静默行为——manifest global_stats 记录 truncated_docs / truncated_tokens
gs_t = V.manifest(info["dir"])["global_stats"]
n_trunc = sum(1 for d in exp_docs if len(d) > 64)
R.check("[sft_trunc64] manifest global_stats.truncated_docs == 超长文档数(F2)",
        gs_t.get("truncated_docs") == n_trunc,
        f"manifest={gs_t.get('truncated_docs')} exp={n_trunc}")
R.check("[sft_trunc64] manifest global_stats.truncated_tokens == 丢弃 token 数(F2)",
        gs_t.get("truncated_tokens") == lost,
        f"manifest={gs_t.get('truncated_tokens')} exp={lost}")
gs_a = V.manifest(BUILDS["sft_a"]["dir"])["global_stats"]
R.check("[sft_a] 无截断时 manifest 不含 truncated_* 字段(序列化省略, 向后兼容)",
        "truncated_docs" not in gs_a and "truncated_tokens" not in gs_a, "")

ok = R.done()
sys.exit(0 if ok else 1)
