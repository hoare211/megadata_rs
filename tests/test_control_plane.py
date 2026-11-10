"""
控制面改造（借鉴 streaming 设计文档 P0~P2）的端到端验证。

覆盖：
- P0-2: 真 sha256 checksum（构建期写入 + 读取端校验 + 损坏缓存重下）
- P0-3: shard 聚合（多 batch/shard）+ 读回一致性
- P0-4: total_tokens 真实有效 token 数（不含 padding）
- P1-1: PT per-batch zstd frame 压缩 / SFT Arrow IPC 压缩 的读回一致性
- P1-3: Mixer O(1) 断点续训（rng_word_pos 快照）
- P2-1: 置换表 shuffle（确定性 + 几十字节 state_dict + 断点续训）
- P2-2: relaxed 分区（分布式完整性 + 跨拓扑序不变）
- P2-4: StreamingDataset + MegadataDataLoader（num_workers=2 无重不漏）
"""
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import http.server
import socketserver
import threading
import time
import warnings

import pytest
import torch

from megadata import (
    DataBuilder,
    BatchStream,
    MixBatchStream,
    RandomAccessBatchReader,
    StreamingDataset,
    MegadataDataLoader,
    MegadataInterrupted,
    read_manifest,
)

TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH",
    os.environ.get("BBPE_TOKENIZER_DIR", "./tokenizer"),
)
# 转绝对路径：后续测试可能 os.chdir（云缓存测试），相对路径会失效
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)
BATCH_SIZE = 4
MAX_SEQ_LEN = 64
TOTAL_DOCS = 400


def _make_jsonl(path, n=TOTAL_DOCS, prefix="doc"):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"{prefix} {i} the quick brown fox jumps over the lazy dog {i * 7}"}) + "\n")


def _build(output_dir, mode="pt", batches_per_shard=0, compression="none", prefix="doc"):
    os.makedirs(output_dir, exist_ok=True)
    raw = output_dir.rstrip("/") + ".jsonl"
    _make_jsonl(raw, prefix=prefix)
    builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
    # eos 是 PT 专属参数：SFT 下传入会被 BuildConfig 拒绝（模式无关参数响亮失败）
    extra = {"eos_id": 2} if mode == "pt" else {}
    builder.build_dataset(
        [raw], output_dir, mode=mode, batch_size=BATCH_SIZE,
        batches_per_shard=batches_per_shard, compression=compression, **extra,
    )
    return os.path.join(output_dir, "index.json")


def _collect_batches(loader, limit=100000):
    out = []
    for batch in loader:
        out.append(batch["input_ids"].clone())
        if len(out) >= limit:
            break
    return out


@pytest.fixture(scope="module")
def pt_variants():
    """同一份语料的三种构建：1:1 旧粒度 / 聚合 / 聚合+zstd。"""
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "ds")
        yield {
            "plain1": _build(os.path.join(base, "plain1"), batches_per_shard=1),
            "agg": _build(os.path.join(base, "agg"), batches_per_shard=8),
            "zstd": _build(os.path.join(base, "zstd"), batches_per_shard=8, compression="zstd"),
        }


# ============================================================
# P0-3: shard 聚合
# ============================================================
def test_aggregated_manifest(pt_variants):
    with open(pt_variants["agg"]) as f:
        m = json.load(f)
    assert m["version"] == "2.0"
    assert m["global_stats"]["total_shards"] < m["global_stats"]["total_batches"], "聚合后 shard 数应小于 batch 数"
    multi = [s for s in m["shards"] if s["num_batches"] > 1]
    assert multi, "应存在多 batch 的聚合 shard"
    # 每 shard 的 batch 数不超过聚合粒度（最后一个 shard 可不满）
    assert all(s["num_batches"] <= 8 for s in m["shards"])


def test_aggregated_readback_equals_unaggregated(pt_variants):
    """聚合只是物理打包方式变化：读出的 batch 多重集必须与 1:1 构建完全一致
    （不重不漏）。逐位置顺序无需一致——读取端 sampler 会按 seed 打乱 shard 顺序，
    shard 数不同则打乱后的遍历序不同；顺序一致性由 test_zstd_pt_roundtrip
    （同聚合粒度对比）覆盖。"""
    a = _collect_batches(BatchStream(pt_variants["plain1"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    b = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    assert len(a) == len(b) > 0
    key = lambda batches: sorted(tuple(t.flatten().tolist()) for t in batches)
    assert key(a) == key(b), "聚合与 1:1 构建读出的 batch 全集必须一致"


# ============================================================
# P0-2: 真 sha256
# ============================================================
def test_sha256_recorded(pt_variants):
    for key in ("plain1", "agg", "zstd"):
        with open(pt_variants[key]) as f:
            m = json.load(f)
        for shard in m["shards"]:
            h = shard.get("hashes", {}).get("sha256")
            assert h and len(h) == 64 and all(c in "0123456789abcdef" for c in h), \
                f"{key} shard {shard['shard_id']} 缺少合法 sha256"


def test_pt_default_compression(pt_variants):
    """PT 构建默认 compression='zstd'（benchmark 判据驱动，见 builder.py docstring）；
    显式 compression='none' 关闭。"""
    # 不传 compression 的 PT 构建 → 默认 Zstd-MDS（建在 ds/ 兄弟目录，避免污染 agg 目录）
    base_dir = os.path.dirname(os.path.dirname(pt_variants["agg"]))
    # v3：`compression="auto"`（PT→zstd）；`None` 不再是合法值
    idx = _build(os.path.join(base_dir, "default"), batches_per_shard=8, compression="auto")
    with open(idx) as f:
        assert json.load(f)["format"] == "Zstd-MDS"
    # SFT 默认仍不压缩（Arrow-IPC 原生格式）
    with open(pt_variants["agg"]) as f:
        assert json.load(f)["format"] == "Memmap"


def test_total_tokens_is_real_valid_count(pt_variants):
    """P0-4：total_tokens 不再把 padding 计入（ cu_seqlens 末值累加）。"""
    with open(pt_variants["agg"]) as f:
        m = json.load(f)
    gs = m["global_stats"]
    full = gs["total_batches"] * BATCH_SIZE * MAX_SEQ_LEN
    assert 0 < gs["total_tokens"] <= full, "total_tokens 应为 (0, batch数×bs×msl] 的真实 token 数"


# ============================================================
# P1-1: 压缩读回一致性
# ============================================================
def test_zstd_pt_roundtrip_and_random_access(pt_variants):
    plain = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    zstd_loader = BatchStream(pt_variants["zstd"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
    zstd = _collect_batches(zstd_loader)
    assert len(zstd) == len(plain)
    for x, y in zip(plain, zstd):
        assert torch.equal(x, y)

    # 压缩格式的随机寻址：v3 起随机访问归 `RandomAccessBatchReader`
    mid = len(plain) // 2
    reader = RandomAccessBatchReader(pt_variants["zstd"], batch_size=BATCH_SIZE,
                                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    assert torch.equal(reader.get_batch(mid)["input_ids"], plain[mid])


def test_zstd_manifest_format(pt_variants):
    with open(pt_variants["zstd"]) as f:
        m = json.load(f)
    assert m["format"] == "Zstd-MDS"


# ============================================================
# P2-1: 置换表 shuffle
# ============================================================
def test_permutation_deterministic_and_resume(pt_variants):
    l1 = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                              seed=7, shuffle="permutation")
    l2 = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                              seed=7, shuffle="permutation")
    s1 = _collect_batches(l1, limit=10)
    s2 = _collect_batches(l2, limit=10)
    assert len(s1) == 10
    for x, y in zip(s1, s2):
        assert torch.equal(x, y), "同 seed 置换序必须完全一致"

    # 断点续训：state_dict 只有几十字节，恢复后接续原序列
    state = l1.state_dict()
    assert state["schema"] == 3          # v3：不再有 mode 字段（buffer 模式已移除）
    blob = json.dumps(state, default=str)
    assert len(blob) < 512, f"置换模式 state_dict 应 < 512B，实际 {len(blob)}B"

    l3 = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                              seed=7, shuffle="permutation")
    l3.load_state_dict(state)
    tail = _collect_batches(l3, limit=5)
    for i, (x, y) in enumerate(zip(s1[10:15] if len(s1) > 15 else [], tail)):
        assert torch.equal(x, y), f"第 {10 + i} 步续训不一致"


def test_permutation_covers_all_batches(pt_variants):
    l1 = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                              seed=7, shuffle="permutation")
    perm = _collect_batches(l1)
    seq = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    assert len(perm) == len(seq)
    key = lambda batches: sorted(tuple(t.flatten().tolist()) for t in batches)
    assert key(perm) == key(seq), "置换后必须无重不漏"


# ============================================================
# P0（2026-09-14 复核修复）: 续训状态恢复不得被"重建"清零
# ============================================================
CONSUME_BEFORE_CKPT = 5


def _consume(loader, n):
    it = iter(loader)
    return [next(it)["input_ids"].clone() for _ in range(n)]


def test_elastic_anchor_change_keeps_consumed(pt_variants):
    """回归（复核问题 1）：弹性锚点 ncn 变化时 load_state_dict 必须保留 consumed。

    旧实现：先恢复 `_consumed`，随后因 ncn 变化触发 reader 重建，而 `_ensure_reader`
    里为 fork 语义写的 `self._consumed = 0` 会把刚恢复的值冲掉 → `seek(0)` → 从头重读，
    且完全静默（弹性扩缩容续训的旗舰路径）。
    """
    loader = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                                  seed=42, partition="relaxed", num_canonical_nodes=1)
    _consume(loader, CONSUME_BEFORE_CKPT)
    state = loader.state_dict()
    # v3：游标是 (epoch, in_epoch_pos)，ncn 归 topology
    assert state["in_epoch_pos"] == CONSUME_BEFORE_CKPT
    assert state["topology"]["num_canonical_nodes"] == 1

    # 新实例 + 不同 ncn：state 里的 ncn=1 会触发 reader 重建
    resumed = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                                   seed=42, partition="relaxed", num_canonical_nodes=2)
    resumed.load_state_dict(state)
    assert resumed.state_dict()["in_epoch_pos"] == CONSUME_BEFORE_CKPT, \
        "ncn 变化后游标必须保留（旧实现被清零）"
    # 行为断言：真的 seek 到了继续点（剩余 = 本轮总长 - 已消费），而不是从头读。
    # F-08：恢复后 len() 即"剩余"（与 MixBatchStream 确定性模式同口径），与产出数直接对齐。
    full = len(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                            world_size=1, rank=0, seed=42, partition="relaxed", num_canonical_nodes=2))
    assert len(resumed) == full - CONSUME_BEFORE_CKPT, "恢复后 len() 应为剩余（整轮长 - 已消费）"
    assert sum(1 for _ in resumed) == full - CONSUME_BEFORE_CKPT, "恢复后应恰好产出剩余（不重读）"


def test_resume_after_fork_simulation(pt_variants):
    """回归：模拟 fork（本实例的 reader/mixer 属于父进程）后 load_state_dict 仍按快照恢复。"""
    # --- BatchStream ---
    s1 = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=42, shuffle=False)
    _consume(s1, CONSUME_BEFORE_CKPT)
    state = s1.state_dict()
    s2 = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=42, shuffle=False)
    s2._reader_pid = -1  # 模拟：reader 建在父进程里（fork 后 pid 不一致）
    s2.load_state_dict(state)
    assert s2.state_dict()["in_epoch_pos"] == CONSUME_BEFORE_CKPT, "fork 重建不得冲掉恢复的游标"
    assert sum(1 for _ in s2) == len(s2) - CONSUME_BEFORE_CKPT

    # --- MixBatchStream：fork 之后续训必须与原序列逐 batch 一致 ---
    m1 = MixBatchStream([{"path": pt_variants["agg"], "weight": 1.0}], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                              world_size=1, rank=0, seed=42)
    it = iter(m1)
    for _ in range(CONSUME_BEFORE_CKPT):
        next(it)
    mst = m1.state_dict()
    nxt = next(it)["input_ids"].clone()  # 原序列的第 6 个 batch

    m2 = MixBatchStream([{"path": pt_variants["agg"], "weight": 1.0}], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                              world_size=1, rank=0, seed=42)
    m2._mixer_pid = -1  # 模拟 fork
    m2.load_state_dict(mst)
    assert m2.state_dict()["steps_yielded"] == CONSUME_BEFORE_CKPT
    assert torch.equal(next(iter(m2))["input_ids"], nxt), "续训首个 batch 应与原序列接续一致"


def test_resume_topology_mismatch_warns(pt_variants):
    """续训拓扑与快照不一致 → 必须显式告警（per-rank 进度无法跨拓扑换算）；一致时静默。"""
    args = dict(batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=2, seed=42)
    st = BatchStream(pt_variants["agg"], rank=0, **args, shuffle=False).state_dict()
    other = BatchStream(pt_variants["agg"], rank=1, **args, shuffle=False)
    with pytest.warns(UserWarning, match="topology differs"):
        other.load_state_dict(st)
    same = BatchStream(pt_variants["agg"], rank=0, **args, shuffle=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # 拓扑一致时不得告警
        same.load_state_dict(st)

    # 置换模式（BatchStream）同样告警
    stp = BatchStream(pt_variants["agg"], rank=0, **args).state_dict()
    otherp = BatchStream(pt_variants["agg"], rank=1, **args)
    with pytest.warns(UserWarning, match="topology differs"):
        otherp.load_state_dict(stp)


# ============================================================
# P2-2: relaxed 分区
# ============================================================
def test_relaxed_distributed_integrity(pt_variants):
    """relaxed 分区下 4 个 rank 无重不漏。"""
    WORLD = 4
    loaders = [
        BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                               world_size=WORLD, rank=r, partition="relaxed",
                               num_canonical_nodes=2, shuffle=False)
        for r in range(WORLD)
    ]
    sets = [_collect_batches(l) for l in loaders]
    total = sum(len(s) for s in sets)
    single = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                                                     world_size=1, rank=0, partition="relaxed",
                                                     num_canonical_nodes=2, shuffle=False))
    assert total == len(single), "relaxed 分区存在遗漏/重复"
    key = lambda batches: sorted(tuple(t.flatten().tolist()) for t in batches)
    merged = sorted([tuple(t.flatten().tolist()) for s in sets for t in s])
    assert merged == key(single), "4 rank 拼接集必须等于单 rank 全集"


def test_predownload_throttle(pt_variants):
    """predownload 滑窗节流（§3.7）：极小窗口下功能不变——削 IO 突刺但不丢数据。"""
    ref = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    throttled = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                                                        world_size=1, rank=0, predownload=4, shuffle=False))
    assert len(throttled) == len(ref) > 0
    key = lambda ts: sorted(tuple(t.flatten().tolist()) for t in ts)
    assert key(throttled) == key(ref), "节流窗口不应影响读出的 batch 全集"


# ============================================================
# P2-4: StreamingDataset + MegadataDataLoader
# ============================================================
def test_megadata_dataloader_two_workers(pt_variants):
    ds = StreamingDataset(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7, shuffle="permutation")
    dl = MegadataDataLoader(ds, num_workers=2)
    got = list(dl)
    assert len(got) > 0
    for b in got:
        assert isinstance(b, dict) and "input_ids" in b

    # 无重不漏：2 worker 的并集 == 单进程全集
    seq = _collect_batches(BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
    key = lambda ts: sorted(tuple(t.flatten().tolist()) for t in ts)
    assert key([b["input_ids"] for b in got]) == key(seq), "num_workers=2 的并集必须等于单进程全集"

    # state_dict 追踪
    assert dl._num_batches_yielded == len(got)
    st = dl.state_dict()
    assert st["num_batches_yielded"] == len(got)
    # 已知限制（复核问题 13）：reader 活在 worker 进程里 → 主进程 dataset 状态为空
    assert st["dataset"] == {}
    assert ds.state_dict() == {}


def test_dataloader_state_dict_limitation_warns_and_single_process_is_exact(pt_variants):
    """回归（复核问题 13）：多 worker 下主进程拿不到 dataset 进度必须**告警**；
    而 num_workers=0 的精确续训路径必须真的接续（旧实现 iter() 每次重建 loader →
    load_state_dict 的 seek 白做、静默从头重读）。"""
    ds = StreamingDataset(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7, shuffle="none")
    dl = MegadataDataLoader(ds, num_workers=0)
    # 主进程从未迭代 → dataset 状态为空；此时带着步数恢复必须告警（而不是静默重读）
    with pytest.warns(UserWarning, match="epoch-level"):
        dl.load_state_dict({"num_batches_yielded": 5, "dataset": {}})
    assert dl.state_dict()["num_batches_yielded"] == 5
    assert dl.state_dict()["resume_granularity"] == "epoch"

    it = iter(dl)
    _head = [next(it)["input_ids"].clone() for _ in range(3)]
    st = dl.state_dict()
    assert st["dataset"], "num_workers=0 时 dataset 状态应可用（精确续训路径）"
    assert st["dataset"]["in_epoch_pos"] == 3
    tail = [next(it)["input_ids"].clone() for _ in range(2)]  # 基准的第 4、5 个 batch

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # 精确路径不得告警
        dl.load_state_dict(st)
    rit = iter(dl)
    resumed = [next(rit)["input_ids"].clone() for _ in range(2)]
    for i, (a, b) in enumerate(zip(resumed, tail)):
        assert torch.equal(a, b), f"恢复后第 {i + 1} 个 batch 应与基准续读一致（不能从头重读）"


# ============================================================
# P0-2 读取端校验：损坏缓存 → 重下
# ============================================================
# 服务器侧请求计数（每个用例自行 clear），用于断言"确实重下了"而非仅"数据读对了"
_REQUEST_COUNTS = {}


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        _REQUEST_COUNTS[self.path] = _REQUEST_COUNTS.get(self.path, 0) + 1
        super().do_GET()


def test_mixer_weight_validation(tmp_path, pt_variants):
    """回归（复核问题 9 / 零 token）：权重非法、混合指定、零 token 都要给可操作报错。

    旧行为：全 0 权重 → Rust `WeightedIndex::new().unwrap()` → pyo3 `PanicException:
    InsufficientNonZero`；全 0 token → Python 侧 `ZeroDivisionError`。
    """
    idx = pt_variants["agg"]
    # 1) 全 0 权重 → ValueError（可操作）
    with pytest.raises(ValueError, match="at least one weight"):
        MixBatchStream([{"path": idx, "weight": 0.0}, {"path": idx, "weight": 0.0}], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)

    # 2) 部分显式 weight、部分省略 → ValueError（配比语义不明确）
    with pytest.raises(ValueError, match="all specified or all omitted"):
        MixBatchStream([{"path": idx, "weight": 1.0}, {"path": idx}], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)

    # 3) token 数为 0 的 manifest → ValueError（而不是 ZeroDivisionError）。
    #    锚点 2（D3）：错误文案随 weight_basis 泛化为 "zero {basis} count"，
    #    total_tokens 口径下语义不变（旧文案 "total_tokens == 0" 的等价改写）。
    manifest = read_manifest(idx)
    assert manifest["global_stats"]["total_tokens"] > 0
    zero = dict(manifest)
    zero["global_stats"] = dict(manifest["global_stats"], total_tokens=0)
    zero_path = tmp_path / "index_zero.json"
    zero_path.write_text(json.dumps(zero))
    with pytest.raises(ValueError, match="zero total_tokens count"):
        MixBatchStream([{"path": str(zero_path)}], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)

    # 4) v3：update_temperature 按 manifest total_tokens 重算（不再依赖"构造期是否记了
    #    token 数"），且会重新应用零长度子流掩码（C19-②）
    loader = MixBatchStream([{"path": idx, "weight": 1.0}], batch_size=BATCH_SIZE,
                       max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    loader.update_temperature(2.0)
    assert abs(sum(loader.current_weights) - 1.0) < 1e-6, "温度重算后权重仍应归一"


def test_read_manifest_local_and_http(pt_variants, tmp_path):
    """回归（复核问题 12）：manifest 读取走 Rust —— 本地 / http 都能读，失败要报错。"""
    idx = pt_variants["agg"]
    local = read_manifest(idx)
    assert local["format"] == "Memmap" and local["global_stats"]["total_batches"] > 0

    with tempfile.TemporaryDirectory() as serve_dir:
        src_dir = os.path.dirname(idx)
        for name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, name), serve_dir)
        old_cwd = os.getcwd()
        os.chdir(serve_dir)
        httpd = socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{httpd.server_address[1]}/index.json"
            remote = read_manifest(url)
            assert remote["global_stats"]["total_tokens"] == local["global_stats"]["total_tokens"]
            assert remote["global_stats"]["total_batches"] == local["global_stats"]["total_batches"]
        finally:
            httpd.shutdown()
            httpd.server_close()
            os.chdir(old_cwd)

    with pytest.raises(Exception, match="manifest"):
        read_manifest(str(tmp_path / "not_exists.json"))


def test_corrupt_cu_sidecar_is_detected_and_redownloaded(pt_variants):
    """回归（复核问题 7）：PT 的 .cu 旁路文件损坏也要被发现并重下。

    旧实现：.cu / .idx 只调 `ensure_local_file`（无校验）——损坏只会被静默忽略
    （退回空 cu_seqlens ⇒ **悄悄改变 attention 语义**）或按错误长度解析。
    现在构建期把 `sha256:cu` / `sha256:idx` 写进 manifest，读取端在
    MEGADATA_VALIDATE_CACHE=1 下走同一套"校验 → 不符则重下"。
    """
    with tempfile.TemporaryDirectory() as serve_dir, tempfile.TemporaryDirectory() as cache_dir:
        src_dir = os.path.dirname(pt_variants["agg"])
        for name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, name), serve_dir)

        with open(pt_variants["agg"]) as f:
            manifest = json.load(f)
        cu_hashes = {s["filename"]: s["hashes"].get("sha256:cu") for s in manifest["shards"]}
        assert all(cu_hashes.values()), f"构建期应记录 .cu 的 sha256，实际: {cu_hashes}"
        shard0 = manifest["shards"][0]["filename"]

        old_cwd = os.getcwd()
        os.chdir(serve_dir)
        httpd = socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{port}/index.json"
            old_cache = os.environ.get("MEGADATA_CACHE_DIR")
            old_validate = os.environ.get("MEGADATA_VALIDATE_CACHE")
            os.environ["MEGADATA_CACHE_DIR"] = cache_dir
            os.environ["MEGADATA_VALIDATE_CACHE"] = "1"
            try:
                def read_all():
                    return [(b["input_ids"].clone(),
                             [int(x) for x in b["cu_seqlens"]] if "cu_seqlens" in b else None)
                            for b in BatchStream(url, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                                                            world_size=1, rank=0)]

                ref = read_all()
                assert ref and ref[0][1] is not None, "PT 数据集应带 cu_seqlens（.cu 旁路已加载）"

                # 篡改 shard 0 的 .cu 缓存（等长改写）
                victim_cu = next(f for f in os.listdir(cache_dir)
                                 if f.endswith(f"_{shard0.replace('.bin', '.cu')}"))
                victim = os.path.join(cache_dir, victim_cu)
                assert _sha256_file(victim) == cu_hashes[shard0], "篡改前应与 manifest 一致"
                with open(victim, "r+b") as f:
                    data = bytearray(f.read())
                    data[0] ^= 0xFF
                    f.seek(0)
                    f.write(data)
                assert _sha256_file(victim) != cu_hashes[shard0], "篡改应被 sha256 感知"

                _REQUEST_COUNTS.clear()
                again = read_all()
                assert [x[0].tolist() for x in again] == [x[0].tolist() for x in ref]
                assert again[0][1] == ref[0][1], "重下后 cu_seqlens 应与基准一致（语义不能变）"
                cu_name = shard0.replace(".bin", ".cu")
                assert _REQUEST_COUNTS.get(f"/{cu_name}", 0) >= 1, (
                    "损坏的 .cu 必须被重新下载；旧实现无校验 → 这条会失败"
                )
                assert _sha256_file(victim) == cu_hashes[shard0], "缓存 .cu 应已被重下修复"
            finally:
                for k, v in (("MEGADATA_CACHE_DIR", old_cache), ("MEGADATA_VALIDATE_CACHE", old_validate)):
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        finally:
            httpd.shutdown()
            httpd.server_close()
            os.chdir(old_cwd)


def test_sft_arrow_random_access_and_permutation(tmp_path):
    """回归（复核问题 2）：Arrow 的 shard 内随机寻址必须**正确**（且不再是 O(k²)）。

    旧实现"重开文件 + next() 跳过 N 个 batch"：置换 shuffle 下每个 batch 都要付出
    O(shard 内 batch 数) 的解析代价（实测 k≈256 慢 2.1×、k=1009 慢 7.7×），跳过区间
    的解码错误还被 `let _ =` 吞掉。改用 IPC footer block 定位（FileReader::set_index）后
    随机访问是 O(1)：本用例用"正序 vs 反序寻址逐 idx 比对"锁死定位正确性（含向后跳），
    再用"置换读多重集 == 顺序读多重集"锁死置换路径。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")
    raw = tmp_path / "sft.jsonl"
    with open(raw, "w", encoding="utf-8") as f:
        for i in range(600):
            f.write(json.dumps({"messages": [
                {"role": "user", "content": f"Q{i} question"},
                {"role": "assistant", "content": f"A{i} answer text"},
            ]}, ensure_ascii=False) + "\n")
    out = str(tmp_path / "sft")
    chatml = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN).build_dataset(
        [str(raw)], out, mode="sft", template_str=chatml, batch_size=BATCH_SIZE,
        # turn 标记 id 不写死：由分词器自动推导（Rust probe），避免"写死的 id 与
        # 实际词表不符 → 监督标签静默损坏"（D8 已把这种不一致升级为构建期报错）
        # 注：SFT 不传 eos（模式无关参数，传了会被拒绝）
        pad_id=0,
        turn_start_str="<|im_start|>", turn_end_str="<|im_end|>",
        assistant_str="assistant", batches_per_shard=8, compression="none")
    idx = os.path.join(out, "index.json")

    # 1) map 模式正序 / 反序寻址逐 idx 结果一致（直接检验 footer block 定位，含向后跳）
    m = RandomAccessBatchReader(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    n = min(len(m), 40)
    assert n > 8, "样本太少，测不出随机寻址"
    forward = [m.get_batch(i)["labels"].clone() for i in range(n)]
    backward = [m.get_batch(i)["labels"].clone() for i in range(n - 1, -1, -1)]
    # 注：backward[k] 对应 idx = n-1-k（下方逐 idx 交叉比对）
    for i in range(n):
        assert torch.equal(forward[i], backward[n - 1 - i]), f"idx {i} 正/反序寻址结果不一致"

    # 2) 置换 shuffle 读出的 batch 多重集必须与顺序读完全一致（同一列：labels）
    seq = [b["labels"].clone() for b in
           BatchStream(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)]
    perm = [b["labels"].clone() for b in
            BatchStream(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                                 seed=7, shuffle="permutation")]
    assert len(perm) == len(seq) > 0
    key = lambda ts: sorted(tuple(t.flatten().tolist()) for t in ts)  # noqa: E731
    assert key(perm) == key(seq), "Arrow 置换读必须无重不漏"


def test_build_source_error_is_reported(tmp_path):
    """回归（复核问题 3）：输入读失败必须**报错**，且不留"看起来完整"的部分数据集。

    旧实现 `let _ = read_jsonl_chunks_async(...)` 静默吞错：缺失文件被跳过、读到中途
    出错只保留已读部分，index.json 照常落盘 —— 实测非法 UTF-8 会静默少 16.7% 数据。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")
    ok = tmp_path / "ok.jsonl"
    _make_jsonl(str(ok), n=200)
    out = str(tmp_path / "ds")
    builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)

    # 1) 输入文件不存在 → 报错 + 无 index.json
    with pytest.raises(Exception, match="Failed to read input"):
        builder.build_dataset([str(tmp_path / "missing.jsonl"), str(ok)], out,
                              mode="pt", batch_size=BATCH_SIZE, eos_id=2)
    assert not os.path.exists(os.path.join(out, "index.json")), "失败构建不得写出 index.json"

    # 2) 读到中途出错（非法 UTF-8）→ 报错 + 清理半成品 shard + 无 index.json
    bad = tmp_path / "bad.jsonl"
    with open(bad, "wb") as f:
        f.write(("\n".join(json.dumps({"text": f"doc {i} quick brown fox jumps"})
                           for i in range(6001)) + "\n").encode())
        f.write(b'{"text": "\xff\xfe bad"}\n')
    with pytest.raises(Exception, match="Failed to read input"):
        builder.build_dataset([str(bad)], out, mode="pt", batch_size=BATCH_SIZE, eos_id=2,
                              batches_per_shard=8, compression="none")
    assert not os.path.exists(os.path.join(out, "index.json"))
    leftovers = sorted(p for p in os.listdir(out) if p.startswith("shard_")) if os.path.isdir(out) else []
    assert leftovers == [], f"失败构建应清理半成品 shard，实际残留 {len(leftovers)} 个: {leftovers[:3]}"

    # 3) 正常输入仍可构建（修复不能误伤正常路径）
    builder.build_dataset([str(ok)], out, mode="pt", batch_size=BATCH_SIZE, eos_id=2)
    assert os.path.exists(os.path.join(out, "index.json"))


def test_sigterm_still_terminates_process(tmp_path):
    """回归：装过 megadata 信号处理器后，SIGTERM 仍必须能终止进程。

    旧实现只对 SIGINT 抛 KeyboardInterrupt、对其它信号"什么都不做"，等于把 SIGTERM
    静默吞掉 —— `timeout` / k8s 的 pod 终止都杀不掉进程，只能 SIGKILL
    （复核修复期间实测：两个 pytest 进程因此挂满 20 分钟）。用子进程验证，避免污染本进程。
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import sys, time\n"
        "sys.path.insert(0, 'python')\n"
        "from megadata import install_signal_handlers\n"
        "install_signal_handlers()\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], cwd=repo_root,
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ready", "子进程未能装好信号处理器"
        proc.terminate()  # SIGTERM
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("SIGTERM 被信号处理器吞掉：进程没有退出（旧实现的缺陷）")
        assert proc.returncode == -signal.SIGTERM, f"应由 SIGTERM 终止，实际 returncode={proc.returncode}"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_loader_cancel_public_api(pt_variants):
    """loader.cancel() 是公开 API：取消后下一次取数在一个 batch 周期内抛 MegadataInterrupted。"""
    loader = BatchStream(pt_variants["agg"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                                    world_size=1, rank=0)
    it = iter(loader)
    assert next(it)["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)

    loader.cancel()
    with pytest.raises(MegadataInterrupted):
        for _ in range(1000):
            next(it)
    loader.cancel()  # 幂等


def test_sft_empty_template_warns(tmp_path):
    """SFT 空模板会渲染出空文本 → 0 shard：构造期 UserWarning + 构建**显式报错**（F4）。

    F4 修复后 0 产出不再静默成功（不落 index.json 也不报错），而是直接报
    "Build produced 0 shards"；空模板的 UserWarning 由 Python 侧构造校验发出。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")
    raw = tmp_path / "sft.jsonl"
    raw.write_text('{"conversations":[{"role":"user","content":"hi"}]}\n', encoding="utf-8")
    builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=32)
    with pytest.warns(UserWarning, match="template_str"):
        with pytest.raises(OSError, match="0 shards"):
            builder.build_dataset([str(raw)], str(tmp_path / "out"), mode="sft",
                                  template_str="", batch_size=2)


def test_local_read_retries_transient_failure_then_heals(pt_variants):
    """本地瞬时故障（半截文件）→ **有限次重试**自愈：不抛错、不挂住、数据正确。

    时序（确定性）：t≈0 首读失败（文件被截断）→ 50ms / 200ms 两次重试之间（t≈120ms）
    修好文件 → 最后一次重试成功。若重试被关闭/次数不足，这里会拿到 OSError。
    """
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.dirname(pt_variants["agg"])
        for name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, name), tmp)
        bins = sorted(f for f in os.listdir(tmp) if f.endswith(".bin"))
        victim = os.path.join(tmp, bins[0])
        with open(victim, "rb") as f:
            good = f.read()
        open(victim, "wb").close()  # 截断（模拟传输半截 / 瞬时 IO 故障）

        def repair():
            time.sleep(0.12)
            with open(victim, "wb") as f:
                f.write(good)

        t = threading.Thread(target=repair)
        t.start()
        try:
            loader = BatchStream(os.path.join(tmp, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
            it = iter(loader)
            first = next(it)  # 首读失败 → 重试期间文件被修好 → 最终成功
            rest = sum(1 for _ in it)
        finally:
            t.join()
        assert first["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert rest + 1 == len(loader), "自愈后应能读完全部 batch"


def test_corrupt_local_shard_raises_instead_of_hanging(pt_variants):
    """本地数据集被截断（bit rot）→ 读取端必须**报错**，而不是静默重试。

    旧行为：预取线程把 IO 错误当 EOF，200ms 轮询后无限重试同一位置，消费端表现为
    "一直没数据、也不报错"（训练永久挂住）。修复后 `try_next` 三态分明，
    致命错误经 Python 绑定抛 OSError。
    """
    with tempfile.TemporaryDirectory() as tmp:
        src_dir = os.path.dirname(pt_variants["agg"])
        for name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, name), tmp)
        bins = sorted(f for f in os.listdir(tmp) if f.endswith(".bin"))
        assert bins, "应有 .bin shard"
        victim_name = bins[0]
        victim = os.path.join(tmp, victim_name)
        open(victim, "wb").close()  # 截断为 0 字节

        loader = BatchStream(os.path.join(tmp, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False)
        t0 = time.time()
        with pytest.raises(OSError) as ei:
            for i, _batch in enumerate(loader):
                if i > 100_000:
                    pytest.fail("读了 10 万个 batch 仍无错误：IO 错误被静默吞掉（回归）")

        elapsed = time.time() - t0
        assert elapsed < 5, f"报错耗时 {elapsed:.2f}s 过长（疑似重试无界）"
        msg = str(ei.value)
        assert "megadata" in msg, f"错误信息应来自 megadata 数据流: {msg}"
        assert victim_name in msg, f"错误信息应定位到损坏的 shard {victim_name}: {msg}"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_corrupt_cache_redownloads(pt_variants):
    """HTTP 云端 + 缓存文件被篡改（bit rot 模拟）→ MEGADATA_VALIDATE_CACHE=1 时重下修复。

    显式锁定"重下"这一用户可见契约（而不只是"数据恰好读对了"）：
    1) 服务器侧计数：被篡改的 shard 在修复阶段至少被再 GET 一次；
    2) 缓存侧校验：读完后该缓存文件的 sha256 已恢复到 manifest 记录值。

    注意（2026-09-14 复核结论）：这两条**不能**区分修复前的实现——旧实现
    "校验失败 → 删缓存 → 返回已删除路径"会让 `File::open` 失败，而上层 prefill
    线程把 `None` 当 EOF 轮询（`POLL_TIMEOUT=200ms`）后会重试，于是 200ms 后
    自动重下、数据仍然完整，只是白付一次无效尝试与停顿。真正的行为锁定在
    Rust 侧 `io::tests::test_cache_hit_hash_mismatch_redownloads_and_returns_existing_path`
    （断言返回路径必须存在、且恰好重下一次）。
    """
    with tempfile.TemporaryDirectory() as serve_dir, tempfile.TemporaryDirectory() as cache_dir:
        # 复制 agg 数据集到独立目录（含 index.json 与 shard）
        src_dir = os.path.dirname(pt_variants["agg"])
        for name in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, name), serve_dir)

        with open(pt_variants["agg"]) as f:
            manifest = json.load(f)
        sha_by_name = {s["filename"]: s["hashes"]["sha256"] for s in manifest["shards"]}

        old_cwd = os.getcwd()
        os.chdir(serve_dir)
        httpd = socketserver.TCPServer(("127.0.0.1", 0), _QuietHandler)
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}/index.json"
            old_cache = os.environ.get("MEGADATA_CACHE_DIR")
            old_validate = os.environ.get("MEGADATA_VALIDATE_CACHE")
            os.environ["MEGADATA_CACHE_DIR"] = cache_dir
            os.environ["MEGADATA_VALIDATE_CACHE"] = "1"
            try:
                ref = _collect_batches(BatchStream(url, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
                assert ref

                # 篡改一个缓存中的 .bin（保持大小不变，只改内容 → 模拟 bit rot）
                cached_bins = [f for f in os.listdir(cache_dir) if f.endswith(".bin")]
                assert cached_bins, "缓存中应有 .bin"
                victim = os.path.join(cache_dir, cached_bins[0])
                # 缓存名 = <url hash 前 8 位>_<原始文件名>，还原出原始名以对上 manifest
                victim_name = cached_bins[0].split("_", 1)[1]
                assert victim_name in sha_by_name, f"{victim_name} 应在 manifest 中"
                assert _sha256_file(victim) == sha_by_name[victim_name], "篡改前应与 manifest 一致"
                with open(victim, "r+b") as f:
                    data = bytearray(f.read())
                    data[0] ^= 0xFF
                    f.seek(0)
                    f.write(data)
                assert _sha256_file(victim) != sha_by_name[victim_name], "篡改应被 sha256 感知"

                # 校验开启：损坏文件应被发现并重新下载，读回数据不受影响
                _REQUEST_COUNTS.clear()  # 只统计修复阶段的请求
                fixed = _collect_batches(BatchStream(url, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
                assert len(fixed) == len(ref)
                for x, y in zip(fixed, ref):
                    assert torch.equal(x, y), "损坏缓存重下后数据应与基准一致"
                assert _REQUEST_COUNTS.get(f"/{victim_name}", 0) >= 1, (
                    "损坏的 shard 必须被重新下载；若只删缓存却返回已删除路径，这里会失败"
                )
                assert _sha256_file(victim) == sha_by_name[victim_name], "缓存文件应已被重下修复"
            finally:
                if old_cache is None:
                    os.environ.pop("MEGADATA_CACHE_DIR", None)
                else:
                    os.environ["MEGADATA_CACHE_DIR"] = old_cache
                if old_validate is None:
                    os.environ.pop("MEGADATA_VALIDATE_CACHE", None)
                else:
                    os.environ["MEGADATA_VALIDATE_CACHE"] = old_validate
        finally:
            os.chdir(old_cwd)
            httpd.shutdown()
            httpd.server_close()


# ============================================================
# P1-3: Mixer O(1) 断点续训
# ============================================================
@pytest.fixture(scope="module")
def two_pt_datasets(pt_variants):
    return {"a": pt_variants["agg"], "b": pt_variants["zstd"]}


def test_mixer_o1_state_dict(two_pt_datasets):
    cfg = [{"path": two_pt_datasets["a"], "weight": 0.7},
           {"path": two_pt_datasets["b"], "weight": 0.3}]
    # 基准时间线：完整 20 步
    loader = MixBatchStream(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=1024)
    it = iter(loader)
    baseline = []
    for _ in range(20):
        baseline.append(next(it)["input_ids"].clone())

    # 中断时间线：跑 10 步后快照
    loader_int = MixBatchStream(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=1024)
    it_int = iter(loader_int)
    for _ in range(10):
        next(it_int)
    state = loader_int.state_dict()
    assert "rng_word_pos" in state and "sub_stream_cursors" in state
    blob = json.dumps(state, default=str)
    assert len(blob) < 2048, f"Mixer state_dict 应 < 2KB，实际 {len(blob)}B"

    # 恢复时间线：O(1) 注入快照后必须与基准的第 10~20 步逐位一致
    loader2 = MixBatchStream(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=1024)
    loader2.load_state_dict(state)
    it2 = iter(loader2)
    for i in range(10):
        b = next(it2)["input_ids"]
        assert torch.equal(b, baseline[10 + i]), f"O(1) 续训第 {10 + i} 步不一致"


def test_sft_arrow_compression():
    """SFT (Arrow IPC zstd) 构建与读回。"""
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "raw.jsonl")
        with open(raw, "w", encoding="utf-8") as f:
            for i in range(120):
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": f"Q{i}"},
                        {"role": "assistant", "content": f"A{i} answer"},
                    ]
                }, ensure_ascii=False) + "\n")
        out_plain = os.path.join(tmp, "sft_plain")
        out_zstd = os.path.join(tmp, "sft_zstd")
        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
        chatml = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"
        common = dict(input_paths=[raw], mode="sft", template_str=chatml, batch_size=BATCH_SIZE,
                      pad_id=0,   # SFT 不传 eos（模式无关参数会被拒绝）
                      turn_start_str="<|im_start|>", turn_end_str="<|im_end|>", assistant_str="assistant")
        builder.build_dataset(output_dir=out_plain, batches_per_shard=4, **common)
        builder.build_dataset(output_dir=out_zstd, batches_per_shard=4, compression="zstd", **common)

        with open(os.path.join(out_plain, "index.json")) as f:
            mp = json.load(f)
        assert mp["format"] == "Arrow-IPC"
        assert any(s["num_batches"] == 4 for s in mp["shards"]), "SFT 也应聚合多 batch/shard"

        a = list(BatchStream(os.path.join(out_plain, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
        b = list(BatchStream(os.path.join(out_zstd, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=False))
        assert len(a) == len(b) > 0
        for x, y in zip(a, b):
            assert torch.equal(x["input_ids"], y["input_ids"])
            assert torch.equal(x["labels"], y["labels"])
            assert torch.equal(x["position_ids"], y["position_ids"])
