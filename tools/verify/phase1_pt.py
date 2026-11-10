"""阶段1：PT（预训练）数据正确性验证。

校验口径：先绕过读头、直接从落盘 shard（.bin/.idx/.cu）解析出"顺序 epoch 批次序列"作为
规范真值，再验证读头产出的 epoch 是它的**双射**（不丢/不重），并按顺序拼接后逐位等于
独立复算的 token 流。
"""
import hashlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V
import zstandard as zstd

V.init_ids()
R = V.Report("PHASE1-PT")
texts = V.read_pt_texts()
expected = V.pt_expected_stream(texts)
exp_doc_lens = [len(V.tok().encode_ordinary(t)) + 1 for t in texts]
BUILDS = json.load(open(V.BUILDS_JSON))

R.check("PT 语料读取", len(texts) == 20000, f"docs={len(texts)}")
R.check("PT 期望流长度(独立复算)", len(expected) == sum(exp_doc_lens), f"tokens={len(expected)}")
R.check("文档内不含 eos id", all(V.EOS_ID not in V.tok().encode_ordinary(t) for t in texts), f"eos_id={V.EOS_ID}")


def h(ids):
    return hashlib.md5(struct.pack(f"<{len(ids)}i", *ids) if ids else b"").hexdigest()


def ref_batches(name):
    """从落盘 shard 直接解析顺序 epoch 的批次列表（真值，不经过读头）。"""
    info = BUILDS[name]
    d = info["dir"]
    m = V.manifest(d)
    out = []
    for s in m["shards"]:
        fn, stem = s["filename"], os.path.splitext(s["filename"])[0]
        raw = open(os.path.join(d, stem + ".idx"), "rb").read()
        offsets = list(struct.unpack(f"<{len(raw)//8}Q", raw))
        cu_recs = V.parse_cu_file(os.path.join(d, stem + ".cu"))
        blob = open(os.path.join(d, fn), "rb").read()
        assert len(cu_recs) == s["num_batches"]
        for i in range(s["num_batches"]):
            if m["format"] == "Zstd-MDS":
                data = zstd.ZstdDecompressor().decompressobj().decompress(blob[offsets[i]:offsets[i + 1]])
                ids = list(struct.unpack(f"<{len(data)//4}i", data))
            else:
                ids = list(struct.unpack(f"<{offsets[i+1]-offsets[i]}i", blob[offsets[i] * 4: offsets[i + 1] * 4]))
            assert len(ids) == info["batch_size"] * info["max_seq_len"], (len(ids), name)
            out.append((ids[:cu_recs[i][-1]], cu_recs[i]))
    return out


def read_epoch(name, **kw):
    info = BUILDS[name]
    from megadata import BatchStream
    L = BatchStream(os.path.join(info["dir"], "index.json"),
                    batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                    world_size=1, rank=0, **kw)
    out = []
    for b in L:
        ids = b["input_ids"].reshape(-1).tolist()
        cu = b["cu_seqlens"].tolist()
        out.append({"valid": ids[:cu[-1]], "full": ids, "cu": cu, "raw": b})
    return out


def verify_epoch(name, shuffle, verbose=False):
    """校验一次 epoch：双射 + 顺序拼接 == 真值 + cu/padding 契约。"""
    info = BUILDS[name]
    ref = ref_batches(name)
    ref_stream = [t for ids, _ in ref for t in ids]
    ref_keys = [h(ids) for ids, _ in ref]
    assert ref_stream == expected, f"{name}: shard 直读流 != 独立真值"
    got = read_epoch(name, shuffle=shuffle)
    tag = f"{name}/shuffle={shuffle}"
    R.check(f"[{tag}] epoch 批次数 == manifest total_batches",
            len(got) == info["total_batches"] == len(ref),
            f"read={len(got)} manifest={info['total_batches']} ref={len(ref)}")
    pos = {}
    for i, k in enumerate(ref_keys):
        pos.setdefault(k, []).append(i)
    idxs, unknown = [], 0
    for g in got:
        cand = pos.get(h(g["valid"]))
        if not cand:
            unknown += 1
            idxs.append(-1)
        else:
            idxs.append(cand.pop(0))
    R.check(f"[{tag}] 每个读出的 batch 都能在真值序列中一一对应(内容零损坏)",
            unknown == 0, f"unknown_batches={unknown}")
    R.check(f"[{tag}] epoch 是顺序批次序列的双射(不丢/不重)",
            sorted(idxs) == list(range(len(ref))),
            f"covered={len(set(idxs))}/{len(ref)}")
    concat = [t for i in sorted(range(len(idxs)), key=lambda j: idxs[j]) for t in got[i]["valid"]] \
        if -1 not in idxs else None
    R.check(f"[{tag}] 按真值顺序拼接后逐位等于独立复算 token 流",
            concat == expected, f"len={len(concat) if concat else None}")
    # cu 契约：段边界只能是"eos 之后"或"batch 末尾"
    bad_cu, bad_pad, cap = [], [], info["batch_size"] * info["max_seq_len"]
    for bi, g in enumerate(got):
        cu = g["cu"]
        if cu[0] != 0 or any(b < a for a, b in zip(cu, cu[1:])) or cu[-1] != len(g["valid"]):
            bad_cu.append(("inv", bi))
        for a, b in zip(cu, cu[1:]):
            if b < cap and g["full"][b - 1] != V.EOS_ID:
                bad_cu.append(("boundary_not_after_eos", bi, b))
        if b > cap:
            bad_cu.append(("overflow", bi, b))
        if any(t != V.EOS_ID for t in g["full"][cu[-1]:]):
            bad_pad.append(bi)
    R.check(f"[{tag}] cu_seqlens 契约(起点0/单调/末值=有效长/边界=文档末)", not bad_cu, f"{bad_cu[:3]}")
    R.check(f"[{tag}] padding 全为 eos", not bad_pad, f"{bad_pad[:3]}")
    # 复原文档文本
    docs, cur = [], []
    for t in concat:
        if t == V.EOS_ID:
            docs.append(cur)
            cur = []
        else:
            cur.append(t)
    ok_texts = all(V.tok().decode(d) == t for d, t in zip(docs, texts))
    R.check(f"[{tag}] 复原 {len(texts)} 篇文档文本与原文逐条一致(顺序/无重)",
            len(docs) == len(texts) and ok_texts, f"docs={len(docs)}")
    # shuffle=none 的承诺：置换成 shard 粒度——batch 只可能在 shard 边界处"跳号"
    if shuffle is False:
        sha = V.manifest(info["dir"])["shards"]
        bounds = []
        acc = 0
        for s in sha:
            acc += s["num_batches"]
            bounds.append(acc)
        bs_set = set(bounds)  # 允许断开的边界（含整轮总批次边界）
        breaks = [j for j in range(1, len(idxs)) if idxs[j] != idxs[j - 1] + 1]
        R.check(f"[{tag}] shuffle=none 的跳号只发生在 shard 边界(shard 内部严格顺序)",
                all(idxs[j - 1] + 1 in bs_set for j in breaks),
                f"breaks={len(breaks)} runs<=shards={len(sha)}")
    return idxs, got


# ---------------------------------------------------------------- 三形状一致性
idxs_a, _ = verify_epoch("pt_a", False)
verify_epoch("pt_a", True)
verify_epoch("pt_ref", False)
verify_epoch("pt_bs1_msl64", False)   # 文档(最长136) > msl(64) → 跨 batch 切分
verify_epoch("pt_zstd", False)
verify_epoch("pt_auto", False)        # 1 个 shard
verify_epoch("pt_cols", False)

# 各形状 manifest total_tokens 一致
R.check("所有 PT 形状的 total_tokens 相同 == 独立复算",
        all(BUILDS[n]["total_tokens"] == len(expected) for n in
            ("pt_ref", "pt_a", "pt_zstd", "pt_cols", "pt_auto", "pt_bs1_msl64")),
        str({n: BUILDS[n]["total_tokens"] for n in BUILDS if n.startswith("pt")}))

# 压缩正确性：zstd 与 none 的 epoch 批次内容集合相同
eq = sorted(h(ids) for ids, _ in ref_batches("pt_a")) == sorted(h(ids) for ids, _ in ref_batches("pt_zstd"))
R.check("pt_zstd 解压后批次内容集合 == pt_a(压缩无损)", eq, "")

# ---------------------------------------------------------------- F3：跨读取块保留未满 batch
# 修复前：每个 5000 行读取块末尾强制 flush 一个未满 batch（非满批数 == ceil(行数/5000)）；
# 修复后：GreedyPacker 有状态化，全数据集至多 1 个非满批（在有效流末尾，eos 填充）。
for name in ("pt_ref", "pt_a", "pt_bs1_msl64", "pt_zstd"):
    cap_n = BUILDS[name]["batch_size"] * BUILDS[name]["max_seq_len"]
    partial = [i for i, (ids, _) in enumerate(ref_batches(name)) if len(ids) < cap_n]
    R.check(f"[{name}] 非满批数 == 1(F3 修复后；修复前 == 读取块数)",
            len(partial) <= 1, f"partials={partial}")

# ---------------------------------------------------------------- hash 完整性
for name in ("pt_a", "pt_zstd", "pt_cols"):
    info = BUILDS[name]
    m = V.manifest(info["dir"])
    bad = []
    for s in m["shards"]:
        if hashlib.sha256(open(os.path.join(info["dir"], s["filename"]), "rb").read()).hexdigest() != s["hashes"].get("sha256"):
            bad.append(s["filename"])
        for ext, key in ((".cu", "sha256:cu"), (".idx", "sha256:idx"),
                         (".col_doc_ids", "sha256:col_doc_ids"),
                         (".col_loss_weights", "sha256:col_loss_weights")):
            sp = os.path.join(info["dir"], os.path.splitext(s["filename"])[0] + ext)
            if key in s["hashes"] and hashlib.sha256(open(sp, "rb").read()).hexdigest() != s["hashes"][key]:
                bad.append(os.path.basename(sp))
    R.check(f"[{name}] manifest sha256(含 .cu/.idx/文档级旁路列) 与落盘一致", not bad, f"bad={bad[:3]}")

# ---------------------------------------------------------------- 随机访问
info = BUILDS["pt_a"]
from megadata import RandomAccessBatchReader
ra = RandomAccessBatchReader(os.path.join(info["dir"], "index.json"),
                             batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                             world_size=1, rank=0)
epoch_none = read_epoch("pt_a", shuffle=False)
bad = []
for i in [0, 1, 5, 100, 3, 500, len(epoch_none) - 2, len(epoch_none) - 1]:
    g = ra.get_batch(i)
    if g["input_ids"].reshape(-1).tolist() != epoch_none[i]["full"]:
        bad.append(i)
    if g["cu_seqlens"].tolist() != epoch_none[i]["cu"]:
        bad.append(("cu", i))
R.check("RandomAccessBatchReader.get_batch(i) == 访问序(BatchStream/none)第 i 个 batch",
        not bad, f"bad={bad}")
R.check("RandomAccessBatchReader.__len__ == total_batches", len(ra) == info["total_batches"], f"{len(ra)}")

# ---------------------------------------------------------------- doc_ids / loss_weights
info = BUILDS["pt_cols"]
B, S, cap = info["batch_size"], info["max_seq_len"], info["batch_size"] * info["max_seq_len"]
d = info["dir"]
m = V.manifest(d)
# 按文档所述"定宽算术寻址"直接读列文件（顺序 epoch）
col_offsets = {}
acc = 0
per_shard = []
for s in m["shards"]:
    per_shard.append((s, acc))
    acc += s["num_batches"]
bad_doc, bad_w, bad_seg, uids, lens = [], [], [], [], {}
batch_i = 0
for s, base in per_shard:
    stem = os.path.splitext(s["filename"])[0]
    did_blob = open(os.path.join(d, stem + ".col_doc_ids"), "rb").read()
    lw_blob = open(os.path.join(d, stem + ".col_loss_weights"), "rb").read()
    assert len(did_blob) == s["num_batches"] * cap * 4, (len(did_blob), s["num_batches"] * cap * 4)
    assert len(lw_blob) == s["num_batches"] * cap * 2, (len(lw_blob), s["num_batches"] * cap * 2)
    cu_recs = V.parse_cu_file(os.path.join(d, stem + ".cu"))
    for i in range(s["num_batches"]):
        did = list(struct.unpack_from(f"<{cap}i", did_blob, i * cap * 4))
        lw = list(struct.unpack_from(f"<{cap}e", lw_blob, i * cap * 2))
        cu = cu_recs[i]
        valid = cu[-1]
        if len(did) != cap:
            bad_doc.append(("shape", batch_i))
        if any(x != -1 for x in did[valid:]):
            bad_doc.append(("pad", batch_i))
        if any(abs(w) > 1e-6 for w in lw[valid:]) or any(abs(w - 1) > 1e-6 for w in lw[:valid]):
            bad_w.append(batch_i)
        for a, b in zip(cu, cu[1:]):
            u = set(did[a:b])
            if len(u) != 1:
                bad_seg.append((batch_i, a, b))
            else:
                uids.append(u.pop())
        for x in did[:valid]:
            lens[x] = lens.get(x, 0) + 1
        batch_i += 1
R.check("pt_cols doc_ids 列文件长度 == num_batches×cap×4(定宽)", not [x for x in bad_doc if x[0] == "shape"], f"{bad_doc[:3]}")
R.check("doc_ids[padding] == -1", not [x for x in bad_doc if x[0] == "pad"], f"{bad_doc[:3]}")
R.check("loss_weights: 有效=1.0 / padding=0.0", not bad_w, f"{bad_w[:3]}")
R.check("doc_ids 在 cu 段内恒定(段=同文档)", not bad_seg, f"{bad_seg[:3]}")
first_seen = [u for i, u in enumerate(uids) if i == 0 or u != uids[i - 1]]
R.check("doc_ids uid 严格递增且为 0..19999(同文档跨 batch 切分时 uid 保持一致)",
        first_seen == list(range(len(texts))) and max(uids) == len(texts) - 1,
        f"segments={len(uids)} uniq={len(set(uids))}")
R.check("doc_ids 每 uid 覆盖 token 数 == 文档长度(含 eos)",
        [lens[i] for i in range(len(lens))] == exp_doc_lens, f"uids={len(lens)}")
# 与读头侧张量对照
gotc = read_epoch("pt_cols", shuffle=False)
refd = {}
bad_sync = []
for g in gotc:
    key = h(g["valid"])
    if key in refd:
        continue
    refd[key] = (g["raw"]["doc_ids"].reshape(-1).tolist(), g["raw"]["loss_weights"].reshape(-1).tolist())
R.check("读头侧 doc_ids/loss_weights 与落盘列同构(张量形状)", 
        all(len(v[0]) == cap for v in refd.values()), f"n={len(refd)}")

ok = R.done()
sys.exit(0 if ok else 1)
