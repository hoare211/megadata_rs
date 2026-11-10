"""阶段3：加载期验证——多轮 epoch 覆盖、多 rank 互斥且并集完整、worker 维度、
混采配比、len() 口径。"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

V.init_ids()
R = V.Report("PHASE3-LOAD")
BUILDS = json.load(open(V.BUILDS_JSON))


def bh(name, b):
    return hashlib.md5(b["input_ids"].numpy().tobytes()).hexdigest()


def collect(name, world_size=1, rank=0, shuffle=True, epoch=0, partition="round_robin", **kw):
    from megadata import BatchStream
    info = BUILDS[name]
    L = BatchStream(os.path.join(info["dir"], "index.json"),
                    batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                    world_size=world_size, rank=rank, shuffle=shuffle, epoch=epoch,
                    partition=partition, **kw)
    n = len(L)
    got = [bh(name, b) for b in L]
    return n, got


# ---------------------------------------------------------------- 多轮 epoch
for name in ("pt_a", "sft_a"):
    info = BUILDS[name]
    tot = info["total_batches"]
    sets = []
    for ep in range(3):
        n, got = collect(name, epoch=ep)
        sets.append((n, got))
        R.check(f"[{name}] epoch={ep} len==total_batches 且实际产出一致",
                n == tot and len(got) == tot and len(set(got)) == tot,
                f"len={n} produced={len(got)} uniq={len(set(got))} total={tot}")
        R.check(f"[{name}] epoch={ep} 是对全量的双射(无漏无重)", sorted(got) == sorted(sets[0][1]),
                f"same multiset as epoch0: {sorted(got) == sorted(sets[0][1])}")
    R.check(f"[{name}] 不同 epoch 顺序不同(每轮重排)", sets[0][1] != sets[1][1] != sets[2][1],
            f"e0==e1:{sets[0][1]==sets[1][1]} e1==e2:{sets[1][1]==sets[2][1]}")
    # 同一 seed/epoch 可复现
    n2, got2 = collect(name, epoch=0)
    R.check(f"[{name}] 同 (seed,epoch) 完全可复现", got2 == sets[0][1], "")

# ---------------------------------------------------------------- DDP 互斥 + 并集
for name in ("pt_a", "sft_a"):
    info = BUILDS[name]
    # 顺序全量（用于对照并集）
    _, full = collect(name, shuffle=False)
    full_set = set(full)
    for W in (1, 2, 3, 4, 8):
        per_rank, lens = [], []
        for r in range(W):
            n, got = collect(name, world_size=W, rank=r)
            per_rank.append(got)
            lens.append(n)
        flat = [x for g in per_rank for x in g]
        R.check(f"[{name}] world_size={W}: 各 rank 轮长之和 == total_batches",
                sum(lens) == info["total_batches"], f"lens={lens}")
        R.check(f"[{name}] world_size={W}: 各 rank 产出数 == len()",
                all(lens[r] == len(per_rank[r]) for r in range(W)), f"lens={lens}")
        R.check(f"[{name}] world_size={W}: rank 间严格互斥(无重复)",
                len(flat) == len(set(flat)), f"uniq={len(set(flat))}/{len(flat)}")
        R.check(f"[{name}] world_size={W}: rank 并集 == 全量(无遗漏)",
                set(flat) == full_set, f"cover={len(set(flat))}/{len(full_set)}")
    # relaxed 分区同样互斥且覆盖
    for W in (3, 4):
        per_rank = [collect(name, world_size=W, rank=r, partition="relaxed", num_canonical_nodes=W)[1]
                    for r in range(W)]
        flat = [x for g in per_rank for x in g]
        R.check(f"[{name}] relaxed world_size={W}: 互斥且并集==全量",
                len(flat) == len(set(flat)) and set(flat) == full_set,
                f"uniq={len(set(flat))}/{len(flat)} cover={set(flat)==full_set}")

# ---------------------------------------------------------------- worker 维度（真实 DataLoader）
try:
    from megadata import StreamingDataset, MegadataDataLoader
    info = BUILDS["pt_a"]
    _, full = collect("pt_a", shuffle=False)
    rank_sets = []
    for r in range(2):
        ds = StreamingDataset(os.path.join(info["dir"], "index.json"),
                              batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                              world_size=2, rank=r)
        dl = MegadataDataLoader(ds, num_workers=2)
        seen = [bh("pt_a", b) for b in dl]
        rank_sets.append(seen)
        R.check(f"[pt_a] MegadataDataLoader(world=2,rank={r},num_workers=2) 无重复产出",
                len(seen) == len(set(seen)) and len(seen) > 0,
                f"batches={len(seen)} uniq={len(set(seen))}")
    flat = [x for s in rank_sets for x in s]
    R.check("[pt_a] 多 worker 跨 rank: 互斥且并集 == 全量",
            len(flat) == len(set(flat)) and set(flat) == set(full),
            f"uniq={len(set(flat))}/{len(flat)} cover={set(flat)==set(full)}")
    # 关键对照：num_workers=0 时 rank0 的集合 vs num_workers=2 时 rank0 的集合
    _, r0_single = collect("pt_a", world_size=2, rank=0)
    same = set(rank_sets[0]) == set(r0_single)
    print(f"[NOTE] num_workers 0→2 时 rank0 的 batch 集合是否相同: {same} "
          f"(|single|={len(set(r0_single))}, |multi|={len(set(rank_sets[0]))})", flush=True)
except Exception as e:
    R.check("[pt_a] MegadataDataLoader(num_workers=2)", False, f"{type(e).__name__}: {e}")

# ---------------------------------------------------------------- 混采配比
from megadata import MixBatchStream
pt_i, sft_i = BUILDS["pt_a"], BUILDS["sft_a"]
mx = MixBatchStream(
    [{"path": os.path.join(pt_i["dir"], "index.json"), "weight": 0.7},
     {"path": os.path.join(sft_i["dir"], "index.json"), "weight": 0.3}],
    batch_size=pt_i["batch_size"], max_seq_len=pt_i["max_seq_len"],
    world_size=1, rank=0, seed=7)
cnt = {"pt": 0, "sft": 0}
fmts = set()
for i, b in enumerate(mx):
    fmts.add(b["format"])
    cnt[b["format"]] += 1
    if i >= 499:
        break
tot = sum(cnt.values())
R.check("MixBatchStream 同时产出 pt/sft 两种格式", fmts == {"pt", "sft"}, f"formats={fmts}")
ratio = cnt["pt"] / tot
R.check("MixBatchStream 权重 0.7/0.3 生效(±0.1)", abs(ratio - 0.7) < 0.1,
        f"pt={cnt['pt']} sft={cnt['sft']} ratio={ratio:.3f}")
R.check("MixBatchStream 每个 batch 带 format 键", True, "")
# 权重切换
mx.update_weights([0.0, 1.0])
c2 = {"pt": 0, "sft": 0}
for i, b in enumerate(mx):
    c2[b["format"]] += 1
    if i >= 99:
        break
R.check("update_weights([0,1]) 后全为 sft", c2["sft"] == 100 and c2["pt"] == 0, f"{c2}")

ok = R.done()
sys.exit(0 if ok else 1)
