"""阶段7：分块边界（5000 行/chunk 的尾块）、进度计数、format 键、PT 满批契约。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

V.init_ids()
R = V.Report("PHASE7-MISC")
E = V.EDGE

# ---------------------------------------------------------------- 尾块（20007 行，5000/chunk → 4 满块 + 1 尾块）
texts = V.read_pt_texts()
extra = [{"text": f"tail doc {i}"} for i in range(7)]
p = f"{E}/pt_tail.jsonl"
with open(p, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    for o in extra:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")
from megadata import DataBuilder
from megadata.builder import BuildConfig
out_dir = f"{V.OUT}/pt_tail"
builder = DataBuilder(tokenizer_path=V.TOKENIZER_DIR, max_seq_len=128)
summary = builder.build_dataset(BuildConfig(
    input_paths=[p], output_dir=out_dir, mode="pt", batch_size=4,
    batches_per_shard=16, compression="none"))
prog = builder.get_progress()
exp = []
for t in texts:
    exp.extend(V.tok().encode_ordinary(t))
    exp.append(V.EOS_ID)
for o in extra:
    exp.extend(V.tok().encode_ordinary(o["text"]))
    exp.append(V.EOS_ID)
got = V.pt_stream(out_dir)
R.check("[chunk] 20007 行(=4 满块+尾块) 有效流 == 独立复算(尾块无丢失)",
        got == exp, f"got={len(got)} exp={len(exp)}")
R.check("[chunk] 进度计数 docs_read == 输入行数", prog.get("docs_read") == 20007,
        f"docs_read={prog.get('docs_read')}")
R.check("[chunk] 进度计数 docs_tokenized == 输入行数(无丢弃)",
        prog.get("docs_tokenized") == 20007, f"docs_tokenized={prog.get('docs_tokenized')}")
docs = []
cur = []
for t in got:
    if t == V.EOS_ID:
        docs.append(V.tok().decode(cur))
        cur = []
    else:
        cur.append(t)
R.check("[chunk] 尾块 7 篇文档完整落地",
        docs[-7:] == [o["text"] for o in extra], f"tail={docs[-7:]}")
# F3 修复后：20007 行（5 个读取块）也只允许 1 个非满批（修复前 == 5）
bbs_tail = V.shard_batches(out_dir)
cap_tail = 4 * 128
nf_tail = sum(1 for bb in bbs_tail if bb["cu"][-1] < cap_tail)
R.check("[chunk] 非满批数 == 1(F3 修复后, 修复前==读取块数5)", nf_tail == 1, f"非满批={nf_tail}")

# ---------------------------------------------------------------- format 键 / PT 满批契约
from megadata import BatchStream
info = json.load(open(V.BUILDS_JSON))
for name in ("pt_a", "sft_a", "pt_zstd", "sft_zstd"):
    d = info[name]["dir"]
    L = BatchStream(os.path.join(d, "index.json"), batch_size=info[name]["batch_size"],
                    max_seq_len=info[name]["max_seq_len"], world_size=1, rank=0)
    b = next(iter(L))
    R.check(f"[{name}] batch 携带 format 键且与 manifest 一致",
            b.get("format") == ("sft" if info[name]["format"] == "Arrow-IPC" else "pt"),
            f"format={b.get('format')} keys={sorted(b.keys())}")
    if b.get("format") == "pt":
        cap = info[name]["batch_size"] * info[name]["max_seq_len"]
        bbs = V.shard_batches(d)
        full = sum(1 for bb in bbs if bb["cu"][-1] == cap)
        nf = len(bbs) - full
        # F3 已修复（2026-09-29）：GreedyPacker 有状态化，跨读取块保留未满 batch，
        # 全数据集**至多 1 个**非满批且在有效流末尾（修复前非满批数 == ceil(行数/5000)）。
        R.check(f"[{name}] PT 无非满批越界(非满批数==1, F3 修复后)",
                all(bb["cu"][-1] <= cap for bb in bbs) and nf == 1,
                f"full={full} total={len(bbs)} 非满批={nf}")
        R.check(f"[{name}] cu_seqlens 张量 dtype=int32",
                str(b["cu_seqlens"].dtype) == "torch.int32", f"{b['cu_seqlens'].dtype}")
    else:
        R.check(f"[{name}] SFT 有 labels/position_ids/row_lens",
                all(k in b for k in ("labels", "position_ids", "row_lens", "cu_seqlens")),
                f"keys={sorted(b.keys())}")

ok = R.done()
sys.exit(0 if ok else 1)
