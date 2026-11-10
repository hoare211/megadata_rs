"""阶段0：构建全部验证用数据集。"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

os.makedirs(V.OUT, exist_ok=True)
V.init_ids()
print("EOS", V.EOS_ID, "TURN_START", V.TURN_START_ID, "TURN_END", V.TURN_END_ID, flush=True)

BUILDS = [
    dict(name="pt_ref", mode="pt", inputs=[V.PT_JSONL], batch_size=1, max_seq_len=8192,
         batches_per_shard=8, compression="none"),
    dict(name="pt_a", mode="pt", inputs=[V.PT_JSONL], batch_size=8, max_seq_len=256,
         batches_per_shard=16, compression="none"),
    dict(name="pt_zstd", mode="pt", inputs=[V.PT_JSONL], batch_size=8, max_seq_len=256,
         batches_per_shard=16, compression="zstd"),
    dict(name="pt_cols", mode="pt", inputs=[V.PT_JSONL], batch_size=8, max_seq_len=256,
         batches_per_shard=16, compression="none", extra_columns=True),
    dict(name="pt_auto", mode="pt", inputs=[V.PT_JSONL], batch_size=8, max_seq_len=256,
         batches_per_shard=0, compression="auto"),
    dict(name="pt_bs1_msl64", mode="pt", inputs=[V.PT_JSONL], batch_size=1, max_seq_len=64,
         batches_per_shard=64, compression="none"),
    dict(name="sft_a", mode="sft", inputs=[V.SFT_JSONL], batch_size=8, max_seq_len=256,
         batches_per_shard=16, compression="none", template_str=V.TEMPLATE),
    dict(name="sft_zstd", mode="sft", inputs=[V.SFT_JSONL], batch_size=8, max_seq_len=256,
         batches_per_shard=16, compression="zstd", template_str=V.TEMPLATE),
    dict(name="sft_ref", mode="sft", inputs=[V.SFT_JSONL], batch_size=1, max_seq_len=512,
         batches_per_shard=8, compression="none", template_str=V.TEMPLATE),
    dict(name="sft_trunc64", mode="sft", inputs=[V.SFT_JSONL], batch_size=8, max_seq_len=64,
         batches_per_shard=16, compression="none", template_str=V.TEMPLATE),
]

results = {}
for cfg in BUILDS:
    name = cfg.pop("name")
    out_dir, summary = V.build(name, **cfg)
    m = V.manifest(out_dir)
    results[name] = {
        "dir": out_dir,
        "mode": summary["mode"],
        "batch_size": summary["batch_size"],
        "max_seq_len": summary["max_seq_len"],
        "total_batches": summary["total_batches"],
        "total_shards": summary["total_shards"],
        "total_tokens": summary["total_tokens"],
        "format": m.get("format"),
        "schema": m.get("schema"),
        "files": sorted(os.listdir(out_dir)),
    }
    print(f"BUILT {name}: batches={summary['total_batches']} shards={summary['total_shards']} "
          f"tokens={summary['total_tokens']} fmt={m.get('format')}", flush=True)

with open(V.BUILDS_JSON, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("OK")
