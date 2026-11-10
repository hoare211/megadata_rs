"""阶段4：断点续训一致性 + mask_adapter 变换一致性。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

V.init_ids()
R = V.Report("PHASE4-RESUME")
BUILDS = json.load(open(V.BUILDS_JSON))


def mk(name, **kw):
    from megadata import BatchStream
    info = BUILDS[name]
    return BatchStream(os.path.join(info["dir"], "index.json"),
                       batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                       world_size=1, rank=0, **kw)


def ids_of(name, b):
    import hashlib
    return hashlib.md5(b["input_ids"].numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------- BatchStream 续训
for name in ("pt_a", "sft_a"):
    info = BUILDS[name]
    tot = info["total_batches"]
    base = [ids_of(name, b) for b in mk(name, seed=4242)]
    R.check(f"[{name}] 基线 epoch 长度", len(base) == tot, f"{len(base)}")
    for k in (0, 1, 137, tot - 1):
        L = mk(name, seed=4242)
        it = iter(L)
        for _ in range(k):
            next(it)
        st = L.state_dict()
        nxt = ids_of(name, next(it))
        del it
        L2 = mk(name, seed=4242)
        L2.load_state_dict(st)
        R.check(f"[{name}] k={k}: state_dict 后 len() == 剩余步数",
                len(L2) == tot - k, f"len={len(L2)} expect={tot-k}")
        seq = [ids_of(name, b) for b in L2]
        R.check(f"[{name}] k={k}: 恢复后剩余序列逐位对齐基线",
                seq == base[k:], f"got={len(seq)} exp={len(base)-k}")

# 轮末快照归一化
info = BUILDS["pt_a"]
L = mk("pt_a", seed=4242)
for _ in L:
    pass
st = L.state_dict()
R.check("pt_a 整轮结束后的快照归一化为 (epoch+1, 0)",
        st["in_epoch_pos"] == 0 and st["epoch"] == 1, f"{st}")
L2 = mk("pt_a", seed=4242)
L2.load_state_dict(st)
R.check("pt_a 轮末快照恢复后 len() == 整轮长", len(L2) == info["total_batches"], f"{len(L2)}")
# epoch 推进：第二轮顺序与第一轮不同
e1 = [ids_of("pt_a", b) for b in mk("pt_a", seed=4242, epoch=1)]
base0 = [ids_of("pt_a", b) for b in mk("pt_a", seed=4242)]
R.check("pt_a epoch 1 与 epoch 0 顺序不同且同为全量",
        e1 != base0 and sorted(e1) == sorted(base0), "")

# ---------------------------------------------------------------- Tier-1 精确续训(num_workers=0)
from megadata import StreamingDataset, MegadataDataLoader
info = BUILDS["sft_a"]
ds = StreamingDataset(os.path.join(info["dir"], "index.json"),
                      batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                      world_size=1, rank=0, seed=99)
base_dl = MegadataDataLoader(ds, num_workers=0)
base = [ids_of("sft_a", b) for b in base_dl]
R.check("Tier-1 基线可产出整轮", len(base) == info["total_batches"], f"{len(base)}")
for k in (1, 250):
    ds1 = StreamingDataset(os.path.join(info["dir"], "index.json"),
                           batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                           world_size=1, rank=0, seed=99)
    dl1 = MegadataDataLoader(ds1, num_workers=0)
    for i, b in enumerate(dl1):
        if i + 1 >= k:
            break
    st = dl1.state_dict()
    R.check(f"Tier-1 k={k}: resume_granularity==batch", st.get("resume_granularity") == "batch", f"{st.get('resume_granularity')}")
    ds2 = StreamingDataset(os.path.join(info["dir"], "index.json"),
                           batch_size=info["batch_size"], max_seq_len=info["max_seq_len"],
                           world_size=1, rank=0, seed=99)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    dl2.load_state_dict(st)
    seq = [ids_of("sft_a", b) for b in dl2]
    R.check(f"Tier-1 k={k}: 恢复后序列 == 基线剩余", seq == base[k:],
            f"got={len(seq)} exp={len(base)-k}")

# ---------------------------------------------------------------- MixBatchStream 确定性续训
from megadata import MixBatchStream
pt_i, sft_i = BUILDS["pt_a"], BUILDS["sft_a"]
CFG = [{"path": os.path.join(pt_i["dir"], "index.json"), "weight": 0.7},
       {"path": os.path.join(sft_i["dir"], "index.json"), "weight": 0.3}]
N = 200
base = []
mx = MixBatchStream(CFG, batch_size=pt_i["batch_size"], max_seq_len=pt_i["max_seq_len"],
                    world_size=1, rank=0, seed=13, steps_per_epoch=N)
for b in mx:
    base.append(ids_of("pt_a", b))
R.check("MixBatchStream 确定性模式轮长 == steps_per_epoch", len(base) == N, f"{len(base)}")
for k in (1, 77):
    mx1 = MixBatchStream(CFG, batch_size=pt_i["batch_size"], max_seq_len=pt_i["max_seq_len"],
                         world_size=1, rank=0, seed=13, steps_per_epoch=N)
    it = iter(mx1)
    for _ in range(k):
        next(it)
    st = mx1.state_dict()
    del it
    mx2 = MixBatchStream(CFG, batch_size=pt_i["batch_size"], max_seq_len=pt_i["max_seq_len"],
                         world_size=1, rank=0, seed=13, steps_per_epoch=N)
    mx2.load_state_dict(st)
    R.check(f"MixBatchStream k={k}: 恢复后 len() == 剩余步数",
            len(mx2) == N - k, f"{len(mx2)} vs {N-k}")
    seq = [ids_of("pt_a", b) for b in mx2]
    R.check(f"MixBatchStream k={k}: 恢复后剩余序列逐位对齐基线", seq == base[k:],
            f"got={len(seq)} exp={len(base)-k}")
    R.check(f"MixBatchStream k={k}: 恢复位置被消费后 len() 回到整轮长",
            len(mx2) == N, f"{len(mx2)}")

# ---------------------------------------------------------------- mask_adapter 变换
import torch
from megadata import (flatten_and_unpad, repad_to_batch, to_flash_varlen,
                      cu_seqlens_to_doc_ids, valid_len)

# PT
info = BUILDS["pt_a"]
b = next(iter(mk("pt_a")))
cu = b["cu_seqlens"]
x = b["input_ids"]
f1 = flatten_and_unpad(x, cu, packer="pt")
R.check("PT flatten_and_unpad 长度 == cu[-1]", f1.numel() == int(cu[-1]), f"{f1.numel()} vs {int(cu[-1])}")
rp = repad_to_batch(f1, cu, x.shape[0], x.shape[1], packer="pt")
R.check("PT repad_to_batch(flatten(x)) == x(布局对称)", torch.equal(rp, x), "")
cu32, ms = to_flash_varlen(cu)
R.check("PT to_flash_varlen 返回 int32 cu 且首 0/末值正确",
        str(cu32.dtype) == "torch.int32" and int(cu32[0]) == 0 and int(cu32[-1]) == int(cu[-1]),
        f"dtype={cu32.dtype} max={ms}")
did = cu_seqlens_to_doc_ids(cu, total_len=x.numel())
R.check("PT cu_seqlens_to_doc_ids 长度 == 全 batch 容量",
        did.numel() == x.numel(), f"{did.numel()} vs {x.numel()}")

# SFT（行内 padding 分散，layout 不同）
b = next(iter(mk("sft_a")))
cu, x, lab, rl = b["cu_seqlens"], b["input_ids"], b["labels"], b["row_lens"]
f1 = flatten_and_unpad(x, cu, packer="sft", row_lens=rl)
fl = flatten_and_unpad(lab, cu, packer="sft", row_lens=rl)
R.check("SFT flatten 长度 == valid_len(cu)", f1.numel() == valid_len(cu),
        f"{f1.numel()} vs {valid_len(cu)}")
rp = repad_to_batch(f1, cu, x.shape[0], x.shape[1], packer="sft", row_lens=rl)
R.check("SFT repad_to_batch(flatten(x)) == x", torch.equal(rp, x), "")
R.check("SFT labels flatten 与 ids flatten 同长", fl.numel() == f1.numel(),
        f"{fl.numel()} vs {f1.numel()}")
# flatten 后 next-token 对齐：flatten(labels)[i] == flatten(ids)[i+1]（同一行内，非行尾）
ok_next = True
for r in range(x.shape[0]):
    st = sum(rl[:r])
    L = int(rl[r])
    seg = fl[st:st + L].tolist()
    seg_ids = f1[st:st + L].tolist()
    for t in range(L - 1):
        if seg[t] != -100 and seg[t] != seg_ids[t + 1]:
            ok_next = False
R.check("SFT flatten 流内 labels[t]==ids[t+1](行内对齐)", ok_next, "")

ok = R.done()
sys.exit(0 if ok else 1)
