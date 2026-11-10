"""20260918 评审（`docs/reviews/code/20260918/评审专家1.md`）修复回归。

覆盖：
- P1-4：轮末 checkpoint（`in_epoch_pos == 轮长`）归一化为下一轮开头——`len()` 不说谎、
  恢复不再 `OSError: seek 越界`（旧版本快照的 (epoch, 轮长) 也走防御性归一化）；
- P0-3：`_OPTION_DEFAULTS` 由 `LoaderOptions` 推导（单一事实源）+ 冲突比较前归一化
  （`True` / `"permutation"` 同义写法不再误报）；
- P0-1：`MixBatchStream` 构造期**不再做集合通信**（本地静态判定；危险直接报错，
  不可判告警）+ `verify_ddp_consistency()` 显式精确校验；`MixStreamingDataset.__len__`
  自然模式走纯函数、不构造引擎；
- P0-2：spawn 平台序列化（`__getstate__` 丢弃 Rust 读头）+ 真实 spawn worker 迭代；
- P2-4 / P2-5 / P2-7 / P2-8：导出面一致 / 版本白名单单一事实源 / unhashable 输入报
  `ValueError` / `Marker.strict` 对 eos 不再是死旋钮。

数据集用**合成 PT**（自写 index.json + shard + `.cu`），不依赖 tokenizer。
"""
from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import sys
import warnings

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "python"))

import megadata._manifest as manifest_mod  # noqa: E402  # P2-11：缓存全局量移到定义模块
import megadata._topology as topology_mod  # noqa: E402  # P2-11：DDP 告警标志移到定义模块
import megadata.dataloader as dataloader_mod  # noqa: E402
from megadata import (  # noqa: E402
    BatchStream,
    LoaderOptions,
    MegadataDataLoader,
    MixBatchStream,
    MixStreamingDataset,
    RandomAccessBatchReader,
    StreamingDataset,
)
from megadata.dataloader import _natural_mode_length_verdict  # noqa: E402

BATCH_SIZE = 2
MAX_SEQ_LEN = 8
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 合成 PT 数据集（memmap + .cu 旁路；不依赖 tokenizer）
# ============================================================
def make_pt_dataset(root, num_batches, *, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                    shards=None):
    """每个 batch 的首元素写入可识别的全局 id；`shards` 可指定每 shard 的 batch 数。"""
    os.makedirs(root, exist_ok=True)
    per_shard = shards or [num_batches]
    meta = []
    total = 0
    for si, nb in enumerate(per_shard):
        fn = f"shard_{si:05d}.bin"
        buf = bytearray()
        for j in range(nb):
            bid = total + j
            for v in [bid] + [0] * (batch_size * max_seq_len - 1):
                buf += int(v).to_bytes(4, "little", signed=True)
        with open(os.path.join(root, fn), "wb") as f:
            f.write(bytes(buf))
        meta.append({"shard_id": si, "filename": fn, "num_batches": nb,
                     "size_bytes": len(buf), "hashes": {}})
        total += nb
        cu = bytearray()
        for _ in range(nb):
            cu += int(2).to_bytes(4, "little")
            cu += int(0).to_bytes(4, "little", signed=True)
            cu += int(batch_size * max_seq_len).to_bytes(4, "little", signed=True)
        with open(os.path.join(root, fn.replace(".bin", ".cu")), "wb") as f:
            f.write(bytes(cu))

    manifest = {
        "version": "2.0",
        "format": "Memmap",
        "max_seq_len": max_seq_len,
        "global_stats": {"total_tokens": total * batch_size * max_seq_len,
                         "total_batches": total, "total_shards": len(per_shard),
                         "batch_size": batch_size},
        "schema": {"input_ids": "int32"},
        "shards": meta,
    }
    path = os.path.join(root, "index.json")
    with open(path, "w") as f:
        json.dump(manifest, f)
    return path


@pytest.fixture(scope="module")
def pt_index(tmp_path_factory):
    """6 个等长 shard × 4 batch = 24 batch（等长 shard ⇒ 静态判定可证真/证伪）。"""
    return make_pt_dataset(str(tmp_path_factory.mktemp("rv0918")), 24, shards=[4] * 6)


@pytest.fixture(scope="module")
def tokenizer_path():
    path = os.path.abspath(os.environ.get(
        "TOKENIZER_PATH", os.path.join(REPO_ROOT, "..", "bbpe_rs", "model_learn_tokenizer")))
    if not os.path.exists(path):
        pytest.skip(f"未找到分词器：{path}")
    return path


def _stream(index, **kw):
    return BatchStream(index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                       world_size=1, rank=0, seed=7, **kw)


# ============================================================
# P1-4：轮末 checkpoint 的归一化与可恢复性
# ============================================================
def test_p1_4_epoch_end_snapshot_normalized_and_resumable(pt_index):
    l1 = _stream(pt_index)
    total = len(l1)
    last = None
    for _ in l1:
        last = l1.state_dict()          # 最后一次 = "轮内最后一个 batch 之后"
    assert (last["epoch"], last["in_epoch_pos"]) == (1, 0), \
        "轮末快照必须按'下一个待产出位置'归一化为下一轮开头"

    l2 = _stream(pt_index)
    l2.load_state_dict(last)
    assert len(l2) == total, "恢复后 len() 应是完整一轮（旧实现返回说谎的 1）"
    assert sum(1 for _ in l2) == total, "旧实现这里直接 OSError（seek 越界）"


def test_p1_4_legacy_epoch_end_snapshot_is_normalized(pt_index):
    """旧版本写出的 (epoch, 轮长) 快照也必须能恢复（防御性归一化）。"""
    l1 = _stream(pt_index)
    total = len(l1)
    st = dict(l1.state_dict())
    st["epoch"], st["in_epoch_pos"] = 0, total      # 旧实现的轮末位置
    l2 = _stream(pt_index)
    l2.load_state_dict(st)
    assert len(l2) == total
    assert sum(1 for _ in l2) == total


def test_p1_4_oversized_position_warns_and_clamps(pt_index):
    l1 = _stream(pt_index)
    total = len(l1)
    st = dict(l1.state_dict())
    st["epoch"], st["in_epoch_pos"] = 0, total + 3  # 快照不自洽
    l2 = _stream(pt_index)
    with pytest.warns(UserWarning, match="inconsistent snapshot"):
        l2.load_state_dict(st)
    assert len(l2) == total
    assert sum(1 for _ in l2) == total


def test_p1_4_tier1_epoch_end_snapshot_resume(pt_index):
    """Tier-1（HF Trainer `save_steps` 整除轮长时命中轮末）的端到端恢复。"""
    ds = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=0)
    total = 0
    last = None
    for _ in dl:
        total += 1
        last = dl.state_dict()               # 最后一次 = 轮内最后一个 batch 之后
    assert last["dataset"]["in_epoch_pos"] == 0, "轮末快照已归一化为下一轮开头"

    ds2 = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    try:
        dl2.load_state_dict(last)            # 旧实现：load 后首次迭代 OSError（seek 越界）
        assert len(dl2) == total
        assert sum(1 for _ in dl2) == total
        assert ds2.epoch == last["tier1"]["epoch"] + 1
    finally:
        dl2.close()
        dl.close()


# ============================================================
# P0-3：默认值单一事实源 + 冲突比较归一化
# ============================================================
def test_p0_3_conflict_check_normalizes_synonyms(pt_index):
    opts = LoaderOptions(batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    BatchStream(pt_index, options=opts, shuffle=True).close()
    BatchStream(pt_index, options=opts, shuffle="permutation").close()   # 同义写法不得误报
    with pytest.raises(ValueError, match="both options= and flat arguments"):
        BatchStream(pt_index, options=opts, shuffle="none")
    with pytest.raises(ValueError, match="both options= and flat arguments"):
        BatchStream(pt_index, options=opts, partition="relaxed")
    # 非法值：报"带可选值"的 ValueError，而不是被误判成冲突
    with pytest.raises(ValueError, match="buffer mode was removed in v3"):
        BatchStream(pt_index, options=opts, shuffle="buffer")


def test_p0_3_defaults_derived_from_loader_options():
    import dataclasses
    from megadata.dataloader import _derive_option_defaults, _OPTION_DEFAULTS
    assert _OPTION_DEFAULTS == _derive_option_defaults()
    assert set(_OPTION_DEFAULTS) == {f.name for f in dataclasses.fields(LoaderOptions)}
    assert _OPTION_DEFAULTS["seed"] == LoaderOptions().seed


# ============================================================
# P0-1：构造期无集合通信 + 静态判定 + 显式精确校验
# ============================================================
@pytest.mark.parametrize("shards,world_size,num_workers,expected", [
    ([4] * 8, 2, 0, "safe"),        # 8 % 2 == 0：每 rank 4 shard
    ([4] * 7, 2, 0, "danger"),      # 7 % 2 == 1：rank0 4 shard / rank1 3 shard
    ([4] * 8, 2, 2, "safe"),        # T=4：8 % 4 == 0
    ([4] * 6, 2, 2, "danger"),      # T=4：rank0 4 shard / rank1 2 shard
    ([4, 3, 4], 1, 0, "unknown"),   # shard 尺寸不齐：静态不可判
])
def test_p0_1_static_verdict(shards, world_size, num_workers, expected):
    manifests = [{"shards": [{"num_batches": n} for n in shards]}]
    verdict, why = _natural_mode_length_verdict(
        manifests, world_size=world_size, num_workers=num_workers)
    assert verdict == expected, (verdict, why)


def _fake_ddp(monkeypatch, *, world_size=2, record=None):
    """把 torch.distributed 伪造成"已初始化的 DDP"，并记录 all_gather_object 调用。"""
    import torch.distributed as dist
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)

    def all_gather_object(out, obj):
        if record is not None:
            record.append(obj)
        out[:] = [obj for _ in out]

    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(topology_mod, "_DDP_NATURAL_WARNED", False)


def test_p0_1_ctor_makes_no_collective_call(pt_index, monkeypatch):
    calls = []
    _fake_ddp(monkeypatch, record=calls)
    cfg = [{"path": pt_index, "weight": 1.0}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MixBatchStream(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=2, rank=0, seed=7)
    try:
        assert calls == [], "构造期不允许出现集合通信（rank 不对称会 hang）"
    finally:
        m.close()


def test_p0_1_danger_verdict_raises_without_collective(tmp_path, monkeypatch):
    calls = []
    _fake_ddp(monkeypatch, record=calls)
    idx = make_pt_dataset(str(tmp_path / "odd"), 28, shards=[4] * 7)  # 7 shard / 2 rank
    with pytest.raises(RuntimeError, match="provably has unequal"):
        MixBatchStream([{"path": idx, "weight": 1.0}], batch_size=BATCH_SIZE,
                       max_seq_len=MAX_SEQ_LEN, world_size=2, rank=0, seed=7)
    assert calls == [], "报错也必须由本地静态判定给出，不依赖集合通信"


def test_p0_1_explicit_verify_calls_collective_once(pt_index, monkeypatch):
    calls = []
    _fake_ddp(monkeypatch, record=calls)
    cfg = [{"path": pt_index, "weight": 1.0}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = MixBatchStream(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=2, rank=0, seed=7)
    try:
        assert calls == []
        m.verify_ddp_consistency()          # 显式入口：这里才是集合通信
        assert len(calls) == 1, "显式校验应恰好一次 all_gather_object"
    finally:
        m.close()


def test_p0_1_mix_tier1_natural_len_is_pure(pt_index):
    cfg = [{"path": pt_index, "weight": 1.0}]
    mx = MixStreamingDataset(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                             world_size=1, rank=0, seed=7)
    try:
        n = len(mx)                          # 自然模式：纯函数求和，不构造 mixer
        assert mx._current_loader is None, "len() 不得构造引擎"
        m = MixBatchStream(cfg, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=7)
        try:
            assert n == len(m), "纯函数口径必须与真实 mixer 的名义轮长一致"
        finally:
            m.close()
    finally:
        mx.close()


# ============================================================
# P0-2：spawn 平台序列化
# ============================================================
def test_p0_2_dataset_with_resumed_loader_is_picklable(pt_index):
    engine = _stream(pt_index)
    it = iter(engine)
    for _ in range(2):
        next(it)
    st = engine.state_dict()
    assert (st["epoch"], st["in_epoch_pos"]) == (0, 2)

    ds = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    ds.load_state_dict(st)                   # 构造并持有 loader（含 Rust 读头）
    assert ds._current_loader is not None
    with pytest.warns(UserWarning, match="spawn serialization dropped"):
        blob = pickle.dumps(ds)              # 旧实现：TypeError: cannot pickle RustStreamReader
    clone = pickle.loads(blob)
    assert clone._current_loader is None and clone._resume_pending is False
    clone.close()
    ds.close()
    engine.close()


def test_p0_2_spawn_workers_iterate_after_resume(pt_index):
    if "spawn" not in multiprocessing.get_all_start_methods():
        pytest.skip("平台不支持 spawn")
    engine = _stream(pt_index)
    it = iter(engine)
    for _ in range(2):
        next(it)
    st = engine.state_dict()

    ds = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=2, multiprocessing_context="spawn")
    try:
        dl.load_state_dict({"num_batches_yielded": 2, "dataset": st,
                            "tier1": {"epoch": 0}, "resume_granularity": "batch"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = sum(1 for _ in dl)         # worker 启动即 pickle dataset
        assert got > 0
    finally:
        dl.close()
        engine.close()


# ============================================================
# P2：导出面 / 版本白名单 / unhashable / eos strict
# ============================================================
def test_p2_4_top_level_surface_matches_mask_adapter_all():
    import megadata
    from megadata.mask_adapter import __all__ as mask_all
    for name in mask_all:
        assert hasattr(megadata, name), f"mask_adapter.__all__ 的 {name} 未在顶层导出"


def test_p2_5_manifest_version_whitelist_single_source(pt_index):
    import megadata._rust as rust
    from megadata.validate import _SUPPORTED_MANIFEST_VERSIONS, validate
    assert tuple(rust.SUPPORTED_MANIFEST_VERSIONS) == tuple(_SUPPORTED_MANIFEST_VERSIONS)
    report = validate(pt_index)
    assert report["ok"], report["checks"]


def test_p2_7_unhashable_choice_raises_value_error():
    with pytest.raises(ValueError, match="shuffle"):
        LoaderOptions(shuffle=["permutation"])
    with pytest.raises(ValueError, match="partition"):
        LoaderOptions(partition=["round_robin"])
    from megadata._options import normalize_compression
    with pytest.raises(ValueError, match="compression"):
        normalize_compression({"a": 1})


def test_p2_8_eos_strict_knob_is_honored(tokenizer_path):
    from megadata import DataBuilder
    from megadata.builder import BuildConfig, Marker
    # 默认值分级：eos 宽松（strict=False），Marker 本身默认严格
    assert BuildConfig(input_paths=["a.jsonl"], output_dir="out", mode="pt").eos.strict is False
    assert Marker().strict is True

    b = DataBuilder(tokenizer_path=tokenizer_path, max_seq_len=32)
    text = "多 token 的 eos 文本 hello world"
    got = list(b._builder.probe_special_ids(text) or [])
    if len(got) <= 1:
        pytest.skip("该分词器把测试文本编成单 token，无法验证 strict 分支")
    with pytest.warns(UserWarning, match="taking the first"):
        assert b._resolve_eos_id(text, None) == int(got[0])          # 默认宽松
    with pytest.raises(ValueError, match="strict=True"):
        b._resolve_eos_id(text, None, strict=True)                   # 显式严格
    assert b._resolve_eos_id(text, 123, strict=True) == 123          # 显式 id 始终优先


# ============================================================
# 20260926 补强：spawn 告警去重 / 粒度自述 / manifest 缓存上限 / 构建摘要
# ============================================================
def test_p0_2_spawn_drop_warning_dedup_and_granularity(pt_index):
    """核对汇总 §五.1/§五.2：同一 dataset 反复 spawn 序列化只告警一次（旧实现
    2 epoch × 2 worker 会重复 4 次）；降级后 Tier-1 快照的 resume_granularity
    自述为 "epoch"，不再谎报 "batch"。"""
    engine = _stream(pt_index)
    it = iter(engine)
    for _ in range(2):
        next(it)
    st = engine.state_dict()

    ds = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=0)
    try:
        dl.load_state_dict({"num_batches_yielded": 2, "dataset": st,
                            "tier1": {"epoch": 0}, "resume_granularity": "batch"})
        assert dl.state_dict()["resume_granularity"] == "batch"   # 未降级：精确到批
        with pytest.warns(UserWarning, match="spawn serialization dropped") as rec:
            blob = pickle.dumps(ds)
            pickle.dumps(ds)                 # 同一实例反复序列化：只告警一次
        assert len(rec) == 1
        clone = pickle.loads(blob)
        assert clone._current_loader is None and clone._resume_pending is False
        assert clone._spawn_resume_degraded is True          # 副本携带降级标记
        assert dl.state_dict()["resume_granularity"] == "epoch"   # 降级自述
        clone.close()
    finally:
        dl.close()
        engine.close()


def test_manifest_cache_is_bounded(tmp_path, monkeypatch):
    """P2-6：`_MANIFEST_CACHE` FIFO 有界——超限淘汰最旧，淘汰后重读内容仍正确。"""
    paths = [make_pt_dataset(str(tmp_path / f"c{i}"), 4) for i in range(3)]
    monkeypatch.setattr(manifest_mod, "_MANIFEST_CACHE_MAX", 2)
    dataloader_mod._MANIFEST_CACHE.clear()
    try:
        for p in paths:
            m = dataloader_mod._load_manifest_cached(p)
            assert m["global_stats"]["total_batches"] == 4
            assert len(dataloader_mod._MANIFEST_CACHE) <= 2
        assert list(dataloader_mod._MANIFEST_CACHE) == [paths[1], paths[2]], \
            "超过上限时应 FIFO 淘汰最旧条目"
        m0 = dataloader_mod._load_manifest_cached(paths[0])   # 淘汰后重读仍正确
        assert m0["global_stats"]["total_batches"] == 4
        assert len(dataloader_mod._MANIFEST_CACHE) == 2
    finally:
        dataloader_mod._MANIFEST_CACHE.clear()


def test_p2_10_build_dataset_returns_summary(tokenizer_path, tmp_path):
    """P2-10：`build_dataset` 返回构建摘要（此前无返回值）。"""
    from megadata import DataBuilder
    from megadata.builder import BuildConfig
    src = tmp_path / "tiny.jsonl"
    src.write_text(
        "\n".join(json.dumps({"text": f"第 {i} 条训练文本：hello megadata world"},
                             ensure_ascii=False) for i in range(20)) + "\n",
        encoding="utf-8",
    )
    b = DataBuilder(tokenizer_path=tokenizer_path, max_seq_len=64)
    cfg = BuildConfig(input_paths=[str(src)], output_dir=str(tmp_path / "out"),
                      mode="pt", batch_size=2)
    summary = b.build_dataset(cfg)
    assert summary["mode"] == "pt" and summary["batch_size"] == 2
    assert summary["compression"] == "zstd"              # PT auto → zstd
    assert summary["total_batches"] > 0
    assert summary["total_shards"] >= 1
    assert summary["manifest"] and os.path.exists(summary["manifest"])
    assert summary["manifest"].endswith("index.json")


# ============================================================
# P1-1：五入口构造器签名收敛（扁平 vs options= 等价 / 差异声明 / epoch 校验统一）
# ============================================================
def test_p1_1_batchstream_flat_and_options_are_equivalent(pt_index):
    """同一参数集走扁平签名与 options= 必须收敛到相同实例状态与相同产出序列。"""
    common = dict(batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                  world_size=1, rank=0, seed=43, shuffle="none")
    a = BatchStream(pt_index, **common)
    b = BatchStream(pt_index, options=LoaderOptions(**common))
    try:
        for name in ("_seed", "_shuffle", "_partition", "_num_canonical_nodes", "_predownload"):
            assert getattr(a, name) == getattr(b, name), f"公共拷贝字段 {name} 不一致"
        va = next(iter(a))["input_ids"][0, 0].item()
        vb = next(iter(b))["input_ids"][0, 0].item()
        assert va == vb, "扁平与 options= 产出的首 batch 必须一致（shard 级打乱同参同序）"
    finally:
        a.close()
        b.close()


def test_p1_1_tier1_kwargs_converge(pt_index):
    """Tier-1 两入口：扁平签名与 options= 归一化收敛后的 kwargs 必须逐键一致。"""
    common = dict(batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=43)
    flat = StreamingDataset(pt_index, **common)
    via = StreamingDataset(pt_index, options=LoaderOptions(**common))
    try:
        assert flat._kwargs == via._kwargs
    finally:
        flat.close()
        via.close()
    cfg = [{"path": pt_index, "weight": 1.0}]
    mflat = MixStreamingDataset(cfg, **common)
    mvia = MixStreamingDataset(cfg, options=LoaderOptions(**common))
    try:
        assert mflat._mix_common == mvia._mix_common
    finally:
        mflat.close()
        mvia.close()


def test_p1_1_reader_ignores_shuffle_in_options(pt_index):
    """差异声明：Reader 无置换表——`options.shuffle` 照单接收但有意忽略，不报错且
    产出与不传 shuffle 时一致（扁平签名本就无 shuffle 形参，两路都不得拷贝该字段）。"""
    assert RandomAccessBatchReader._consumes_shuffle is False
    common = dict(batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=43)
    r1 = RandomAccessBatchReader(pt_index, **common)
    r2 = RandomAccessBatchReader(pt_index, options=LoaderOptions(**common, shuffle=False))
    try:
        assert not hasattr(r1, "_shuffle") and not hasattr(r2, "_shuffle"), \
            "_consumes_shuffle=False 时公共拷贝必须跳过 shuffle 字段"
        v1 = r1.get_batch(0)["input_ids"][0, 0].item()
        v2 = r2.get_batch(0)["input_ids"][0, 0].item()
        assert v1 == v2, "options 携带 shuffle 与不传必须产出相同数据（有意忽略）"
    finally:
        r1.close()
        r2.close()


def test_p1_1_negative_epoch_rejected_uniformly(pt_index):
    """五入口的 epoch 构造期校验统一收敛到 `_resolve_options`（负值此前会让 pyo3
    抛难读的 OverflowError）。校验发生在任何 manifest IO 之前。"""
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        BatchStream(pt_index, epoch=-1)
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        BatchStream(pt_index, options=LoaderOptions(), epoch=-1)   # options= 路径同样校验
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        RandomAccessBatchReader(pt_index, epoch=-1)
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        StreamingDataset(pt_index, epoch=-1)
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        MixStreamingDataset([{"path": pt_index, "weight": 1.0}], epoch=-1)
    with pytest.raises(ValueError, match="epoch must be >= 0"):
        MixBatchStream([{"path": pt_index, "weight": 1.0}], epoch=-1)


# ============================================================
# P2-9：Tier-1 快照点号键改嵌套（tier1.epoch）+ 旧键静默兜底
# ============================================================
def test_p2_9_snapshot_uses_nested_tier1_epoch(pt_index):
    """新格式：Tier-1 轮号嵌套在 `state["tier1"]["epoch"]`，点号键不复存在；
    往返恢复后轮号与剩余步数都不说谎。"""
    ds = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=0)
    it = iter(dl)
    for _ in range(5):
        next(it)
    st = dl.state_dict()
    assert st["tier1"]["epoch"] == ds.epoch == 0
    assert "dataset.epoch" not in st, "点号键必须从新快照中移除（P2-9）"

    ds2 = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    try:
        dl2.load_state_dict(st)
        assert ds2.epoch == 0
        assert sum(1 for _ in dl2) == 24 - 5, "嵌套键恢复后应从轮内位置精确续读"
    finally:
        dl2.close()
        dl.close()


def test_p2_9_legacy_dotted_epoch_key_fallback(pt_index):
    """兜底：旧格式快照（点号键 `dataset.epoch`、无 `tier1` 节）仍可恢复——
    静默兼容（不告警），Tier-1 轮号照常回填。"""
    ds = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=0)
    it = iter(dl)
    for _ in range(5):
        next(it)
    st = dl.state_dict()
    legacy = {k: v for k, v in st.items() if k != "tier1"}
    legacy["dataset.epoch"] = st["tier1"]["epoch"]      # 旧写法：点号键摊平在顶层

    ds2 = StreamingDataset(pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")               # 兜底路径必须静默
            dl2.load_state_dict(legacy)
        assert ds2.epoch == 0, "旧点号键必须照常回填 Tier-1 轮号"
        assert sum(1 for _ in dl2) == 24 - 5
    finally:
        dl2.close()
        dl.close()
