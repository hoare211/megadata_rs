"""阶段6：项目自带自检器交叉验证 + 损坏检测 + 契约校验 + 模板空白行为。"""
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib as V

V.init_ids()
R = V.Report("PHASE6-SELFCHECK")
BUILDS = json.load(open(V.BUILDS_JSON))
PY = os.path.join(V.ROOT, ".venv", "bin", "python")

# ---------------------------------------------------------------- validate.py
for name, ws in [("pt_a", 8), ("pt_zstd", 4), ("pt_cols", 3), ("sft_a", 8), ("sft_zstd", 2), ("pt_auto", 1)]:
    idx = os.path.join(BUILDS[name]["dir"], "index.json")
    p = subprocess.run([PY, "-m", "megadata.validate", idx, "--world-size", str(ws), "--deep"],
                       capture_output=True, text=True, cwd=V.ROOT)
    fails = [l for l in p.stdout.splitlines() if l.startswith("[FAIL]")]
    verdict = [l for l in p.stdout.splitlines() if l.startswith("Verdict")]
    R.check(f"[validate] {name} --deep(自带自检器) 通过", p.returncode == 0 and not fails,
            f"rc={p.returncode} {verdict} fails={fails[:2]}")

# pt_auto 用 batches_per_shard=0（自动）→ 只有 1 个 shard：validate 必须拦下 world_size>1
_idx = os.path.join(BUILDS["pt_auto"]["dir"], "index.json")
_p = subprocess.run([PY, "-m", "megadata.validate", _idx, "--world-size", "4"],
                    capture_output=True, text=True, cwd=V.ROOT)
R.check("[validate] pt_auto(shards=1) + --world-size 4 被拦下(默认聚合粒度陷阱, 报告 F11)",
        _p.returncode != 0 and any("shard count >= world_size" in l for l in _p.stdout.splitlines()),
        [l for l in _p.stdout.splitlines() if l.startswith("[FAIL]")][:1])

# 篡改检测：改坏 shard 一个字节 / 截断 shard
src = BUILDS["pt_a"]["dir"]
bad = f"{V.OUT}/pt_corrupt"
V._clear_dir(bad)
shutil.copytree(src, bad)
fn = sorted(f for f in os.listdir(bad) if f.endswith(".bin"))[0]
p = os.path.join(bad, fn)
with open(p, "r+b") as f:
    f.seek(100)
    b = f.read(1)
    f.seek(100)
    f.write(bytes([b[0] ^ 0xFF]))
err = None
from megadata import BatchStream
try:
    L = BatchStream(os.path.join(bad, "index.json"), batch_size=8, max_seq_len=256,
                    world_size=1, rank=0, shuffle=False)
    for _ in L:          # 读完整轮，确保损坏的 shard 一定被读到
        pass
except Exception as e:
    err = e
vp = subprocess.run([PY, "-m", "megadata.validate", os.path.join(bad, "index.json"), "--deep"],
                    capture_output=True, text=True, cwd=V.ROOT)
# 报告 F1：本地读取路径不做 sha256 校验 ⇒ 等长位翻转被静默读出（读取端不报错）
R.check("[corrupt] 位翻转(等长)在本地读取时**不被发现**(报告 F1)",
        err is None, f"reader_error={type(err).__name__ if err else None}")
R.check("[corrupt] validate --deep 能发现 sha256 不一致", vp.returncode != 0,
        [l for l in vp.stdout.splitlines() if l.startswith("[FAIL]")][:2])

# 截断 shard → 读取端报错（长度不足可被检出）
bad2 = f"{V.OUT}/pt_trunc"
V._clear_dir(bad2)
shutil.copytree(src, bad2)
p2 = os.path.join(bad2, fn)
with open(p2, "r+b") as f:
    f.truncate(os.path.getsize(p2) // 2)
err2 = None
try:
    L = BatchStream(os.path.join(bad2, "index.json"), batch_size=8, max_seq_len=256,
                    world_size=1, rank=0, shuffle=False)
    for _ in L:
        pass
except Exception as e:
    err2 = e
R.check("[corrupt] shard 被截断 → 读取端抛 OSError(不静默当 EOF)", err2 is not None,
        f"{type(err2).__name__}: {str(err2)[:100]}")

# ---------------------------------------------------------------- 契约校验
info = BUILDS["pt_a"]
err3 = None
try:
    BatchStream(os.path.join(info["dir"], "index.json"), batch_size=16, max_seq_len=256)
except Exception as e:
    err3 = e
R.check("[contract] batch_size 与构建期不一致 → 构造期报错", err3 is not None,
        f"{type(err3).__name__}: {str(err3)[:80]}")
err4 = None
try:
    BatchStream(os.path.join(info["dir"], "index.json"), batch_size=8, max_seq_len=128)
except Exception as e:
    err4 = e
R.check("[contract] max_seq_len 与构建期不一致 → 构造期报错", err4 is not None,
        f"{type(err4).__name__}: {str(err4)[:80]}")
err5 = None
try:
    BatchStream(os.path.join(info["dir"], "index.json"), batch_size=8, max_seq_len=256,
                world_size=64, rank=0)
except Exception as e:
    err5 = e
R.check("[contract] shard 数(47) < world_size(64) → 构造期报错", err5 is not None,
        f"{type(err5).__name__}: {str(err5)[:80]}")

# ---------------------------------------------------------------- 模板空白行为
E = V.EDGE
p = f"{E}/sft_ws.jsonl"
conv = {"conversations": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
with open(p, "w", encoding="utf-8") as f:
    f.write(json.dumps(conv, ensure_ascii=False) + "\n")
multi = ("{% for m in conversations %}\n"
         "  <|im_start|>{{ m.role }}\n"
         "  {{ m.content }}<|im_end|>\n"
         "{% endfor %}\n")
out, _ = V.build("edge_sft_ws", mode="sft", inputs=[p], template_str=multi,
                 batch_size=4, max_seq_len=128, batches_per_shard=8, compression="none")
docs = V.sft_docs(out)
got = V.tok().decode(docs[0]) if docs else ""
naive = ("\n  <|im_start|>user\n  q<|im_end|>\n" "\n  <|im_start|>assistant\n  a<|im_end|>\n")
print(f"[NOTE] 多行模板实际渲染: {got!r}", flush=True)
print(f"[NOTE] 朴素(无 trim/lstrip)预期: {naive!r}", flush=True)
R.check("[tmpl] 多行模板受 minijinja trim_blocks/lstrip_blocks 影响(渲染 != 朴素拼接)",
        got != naive, f"equal={got == naive}")

ok = R.done()
sys.exit(0 if ok else 1)
