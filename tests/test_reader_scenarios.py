"""读取侧（引擎层 + Tier 1）全场景测试。

覆盖真实使用中会遇到的所有读取场景：拓扑网格（world_size × num_workers × 分区）、
shuffle/epoch 语义、弹性分区、predownload、取消/中断、manifest 与参数的错误路径、
MixBatchStream 的双模式与权重、RandomAccessBatchReader 的随机寻址。
"""

from __future__ import annotations

import json
import os
import shutil

import pytest

from _scenario_kit import (
    BATCH_SIZE,
    MAX_SEQ_LEN,
    batch_key,
    build_dataset,
    pt_records,
    read_batches,
    sft_records,
    total_batches,
)

WORLD_SIZES = [1, 3]
PARTITIONS = ["round_robin", "relaxed"]


# ==============================================================================
# 夹具
# ==============================================================================
@pytest.fixture(scope="module")
def pt_index(tmp_path_factory):
    """PT 数据集：足够多的 shard，支持 world_size 到 3。"""
    d = tmp_path_factory.mktemp("rd_pt")
    return build_dataset(str(d / "pt"), pt_records(500), "pt", batches_per_shard=4,
                         compression="zstd")


@pytest.fixture(scope="module")
def sft_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("rd_sft")
    return build_dataset(str(d / "sft"), sft_records(500), "sft", batches_per_shard=4)


@pytest.fixture(scope="module")
def all_pt_keys(pt_index):
    """全集指纹（world_size=1 / shuffle=False 的一轮）。"""
    return set(batch_key(b) for b in read_batches(pt_index))


def collect_tier1(index, *, world_size=1, num_workers=0, partition="round_robin",
                  shuffle=True, seed=42, nodes=1):
    """用 Tier 1（StreamingDataset + MegadataDataLoader）收集某拓扑下的全部 batch。"""
    from megadata import MegadataDataLoader, StreamingDataset

    keys = []
    for rank in range(world_size):
        ds = StreamingDataset(index_json_path=index, batch_size=BATCH_SIZE,
                              max_seq_len=MAX_SEQ_LEN, world_size=world_size, rank=rank,
                              seed=seed, shuffle=shuffle, partition=partition,
                              num_canonical_nodes=nodes or None)
        dl = MegadataDataLoader(ds, num_workers=num_workers)
        keys += [batch_key(b) for b in dl]
    return keys


# ==============================================================================
# 1. 拓扑网格：无重叠、无遗漏
# ==============================================================================
@pytest.mark.parametrize("partition", PARTITIONS)
@pytest.mark.parametrize("world_size", WORLD_SIZES)
@pytest.mark.parametrize("num_workers", [0, 2])
def test_topology_grid_disjoint_and_covering(pt_index, all_pt_keys, partition, world_size,
                                             num_workers):
    """任意 (world_size, num_workers, partition) 组合下：数据两两不交且并集为全集。"""
    keys = collect_tier1(pt_index, world_size=world_size, num_workers=num_workers,
                         partition=partition)
    assert keys, f"拓扑 {world_size=} {num_workers=} {partition=} 没有任何产出"
    assert len(set(keys)) == len(keys), "同一拓扑下出现重复 batch（worker 未互斥）"
    assert set(keys) == all_pt_keys, (
        f"拓扑 {world_size=} {num_workers=} {partition=} 的并集与全集不符"
        f"（缺 {len(all_pt_keys - set(keys))} 个 / 多 {len(set(keys) - all_pt_keys)} 个）"
    )


def test_every_batch_reachable_for_each_rank(pt_index):
    """world_size=3 时每个 rank 都必须分到数据（不能有空转 rank）。"""
    for rank in range(3):
        keys = collect_tier1(pt_index, world_size=3, num_workers=0, partition="round_robin")
        del keys
        break
    for rank in range(3):
        from megadata import BatchStream

        loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                              max_seq_len=MAX_SEQ_LEN, world_size=3, rank=rank, seed=42)
        n = len(loader)
        assert n > 0, f"rank {rank} 的轮长为 0（高位 rank 空转）"


# ==============================================================================
# 2. shuffle / epoch 语义
# ==============================================================================
def test_shuffle_semantics(pt_index):
    """shuffle=True：每轮同集合不同顺序、轮内不重复；shuffle=False：每轮同集合。"""
    from megadata import BatchStream

    def run(shuffle):
        loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                              max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                              seed=42, shuffle=shuffle)
        return [[batch_key(b) for b in loader] for _ in range(3)]

    shuffled = run(True)
    assert len({tuple(r) for r in shuffled}) == 3, "shuffle=True 时三轮顺序应各不相同"
    assert all(set(r) == set(shuffled[0]) for r in shuffled), "每轮应覆盖同一集合"
    assert all(len(set(r)) == len(r) for r in shuffled), "轮内不得重复"

    fixed = run(False)
    assert all(set(r) == set(fixed[0]) for r in fixed), "shuffle=False 时每轮应覆盖同一集合"
    assert all(len(set(r)) == len(r) for r in fixed), "轮内不得重复"


def test_sequential_mode_order_contract(pt_index):
    """`shuffle=False` 的确定性承诺：同 seed/epoch 的**两次独立会话**顺序完全一致。

    注意：轮**之间**顺序会变——shard 级归属逐轮轮换（保证尾 shard 均衡），
    `shuffle=False` 关闭的是 **batch 级置换表**（`batch_permutation` 退化为恒等）。
    需要"跨轮完全一致"的场景请用 `RandomAccessBatchReader`（访问轮由 `epoch=` 固定）。
    """
    from megadata import BatchStream

    def first_epoch():
        loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                              max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                              seed=42, shuffle=False)
        return [batch_key(b) for b in loader]

    assert first_epoch() == first_epoch(), "同 seed 的两次独立会话顺序必须一致（可复现）"


def test_shard_assignment_rotates_each_epoch(pt_index):
    """记录并锁定"shard 归属逐轮轮换"这一行为（尾 shard 均衡的代价）。"""
    from megadata.simulation import SimulationConfig, plan_from_rust

    n = total_batches(pt_index)
    cfg = SimulationConfig(
        index_path=pt_index, world_size=1, num_nodes=1, num_workers=0, seed=42,
        epochs=2, shuffle="none", partition="round_robin", num_canonical_nodes=None,
        bandwidth_mbps=1000.0, cache_limit_bytes=0, compute_ms_per_batch=250.0,
        predownload=0, max_steps=0,
    )
    seq = plan_from_rust(cfg)[0]
    epoch0, epoch1 = seq[:n], seq[n:2 * n]
    assert set(epoch0) == set(epoch1), "每轮都必须覆盖全部 shard"
    # shuffle=none 仍会在轮间换 shard 归属（顺序不同）；这不是"数据错"，是设计取舍
    assert epoch0 != epoch1


@pytest.mark.parametrize("alias,expected", [(True, True), ("permutation", True),
                                            (False, False), ("none", False)])
def test_shuffle_aliases(pt_index, alias, expected):
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle=alias)
    assert bool(loader._shuffle) is expected


def test_invalid_shuffle_rejected(pt_index):
    from megadata import BatchStream

    with pytest.raises(ValueError, match="shuffle"):
        BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, shuffle="buffer")


def test_epoch_constructor_and_advance(pt_index):
    """`epoch=k` 起始轮：构造后首轮从第 k 轮开始，len() 对应该轮。"""
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, epoch=5)
    assert loader.epoch == 5
    state = loader.state_dict()          # 未迭代 ⇒ (5, 0)
    assert state["epoch"] == 5 and state["in_epoch_pos"] == 0
    n = sum(1 for _ in loader)
    assert loader.epoch == 6, "读完一轮应推进轮号"
    assert n == total_batches(pt_index)


# ==============================================================================
# 3. 弹性分区（relaxed）
# ==============================================================================
def test_relaxed_global_order_is_topology_independent(pt_index):
    """relaxed 的"全局样本序"只依赖 (seed, epoch, num_canonical_nodes)：

    固定 `num_canonical_nodes` 后，换 world_size / worker 数，展开各 worker 的连续
    切片得到的全局序列应完全一致 —— 这正是"换拓扑可续训"的前提。
    """
    from megadata.simulation import SimulationConfig, plan_from_rust

    NCN = 4  # 必须固定：它是全局序的输入之一（不固定就等于换了序列）

    def plan(world_size, num_workers, shuffle="none"):
        cfg = SimulationConfig(
            index_path=pt_index, world_size=world_size, num_nodes=1,
            num_workers=num_workers, seed=42, epochs=1, shuffle=shuffle,
            partition="relaxed", num_canonical_nodes=NCN, bandwidth_mbps=1000.0,
            cache_limit_bytes=0, compute_ms_per_batch=250.0, predownload=0, max_steps=0,
        )
        return [s for worker in plan_from_rust(cfg) for s in worker]

    # shard 级（用 shuffle="none" 排除 per-worker 置换表的干扰）：
    # 各 worker 取全局序的**连续切片**，按 worker 顺序拼接 == 同一个全局序
    base = plan(1, 1)
    assert plan(2, 1) == base, "relaxed 换 world_size 后全局 shard 序应不变"
    assert plan(1, 3) == base, "relaxed 换 worker 数后全局 shard 序应不变"
    assert plan(4, 2) == base, "relaxed 换拓扑后全局 shard 序应不变"
    assert plan(8, 4) == base, "relaxed 大幅扩缩容后全局 shard 序应不变"

    # 已知边界（记录在案）：置换表是 **per-worker** 的（作用在该 worker 自己的
    # batch 空间上），因此 **batch 级** 全局序仍依赖拓扑。跨拓扑续训依赖的是
    # 逐 rank 游标 + 上面的 shard 级不变性，不是 batch 级全局序。
    assert plan(2, 1, shuffle="permutation") != plan(4, 2, shuffle="permutation")


def test_round_robin_is_not_topology_independent(pt_index):
    """对照组：round_robin 明确不保证弹性（文档承诺如此，避免误用）。"""
    from megadata.simulation import SimulationConfig, plan_from_rust

    def plan(world_size):
        cfg = SimulationConfig(
            index_path=pt_index, world_size=world_size, num_nodes=1, num_workers=0,
            seed=42, epochs=1, shuffle="permutation", partition="round_robin",
            num_canonical_nodes=None, bandwidth_mbps=1000.0, cache_limit_bytes=0,
            compute_ms_per_batch=250.0, predownload=0, max_steps=0,
        )
        return [s for worker in plan_from_rust(cfg) for s in worker]

    assert plan(1) != plan(2), "round_robin 换拓扑应改变分区（如相等说明该承诺失真）"


# ==============================================================================
# 4. predownload / 读取参数不影响内容
# ==============================================================================
@pytest.mark.parametrize("predownload", [0, 1, 8])
def test_predownload_does_not_change_data(pt_index, predownload):
    from megadata import BatchStream

    keys = [batch_key(b) for b in BatchStream(
        index_json_path=pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
        world_size=1, rank=0, seed=42, shuffle=False, predownload=predownload)]
    ref = [batch_key(b) for b in read_batches(pt_index)]
    assert keys == ref, f"predownload={predownload} 改变了读取结果"


# ==============================================================================
# 5. 取消 / 中断语义（P6）
# ==============================================================================
def test_cancel_stream_loader(pt_index):
    from megadata import MegadataInterrupted, BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    it = iter(loader)
    next(it)
    loader.cancel()
    with pytest.raises(MegadataInterrupted):
        next(it)
    del it
    # 取消是"可重复调用"的
    loader.cancel()


def test_cancel_before_first_batch(pt_index):
    from megadata import MegadataInterrupted, BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    loader.cancel()
    with pytest.raises(MegadataInterrupted):
        next(iter(loader))


def test_cancel_mix_loader(pt_index, sft_index):
    from megadata import MegadataInterrupted, MixBatchStream

    loader = MixBatchStream(configs=[{"path": pt_index, "weight": 0.5},
                                {"path": sft_index, "weight": 0.5}],
                       batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                       steps_per_epoch=1000)
    it = iter(loader)
    next(it)
    loader.cancel()
    with pytest.raises(MegadataInterrupted):
        next(it)


def test_cancel_random_access(pt_index):
    from megadata import MegadataInterrupted, RandomAccessBatchReader

    reader = RandomAccessBatchReader(index_json_path=pt_index, batch_size=BATCH_SIZE,
                                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    reader.cancel()
    with pytest.raises(MegadataInterrupted):
        reader.get_batch(0)


def test_cancel_cascades_to_tier1(pt_index):
    """`MegadataDataLoader.cancel()` 透传到 dataset/引擎层。"""
    from megadata import MegadataDataLoader, MegadataInterrupted, StreamingDataset

    ds = StreamingDataset(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    dl = MegadataDataLoader(ds, num_workers=0)
    it = iter(dl)
    next(it)
    dl.cancel()
    with pytest.raises(MegadataInterrupted):
        next(it)


# ==============================================================================
# 6. 错误路径（每一类都必须响亮失败）
# ==============================================================================
def _clone_dataset(index_path, tmp_path, name, patch=None):
    """复制数据集目录并可选地改写 index.json，返回新的 index 路径。"""
    src = os.path.dirname(index_path)
    dst = tmp_path / name
    shutil.copytree(src, dst)
    new_index = dst / "index.json"
    if patch is not None:
        data = json.loads(new_index.read_text(encoding="utf-8"))
        patch(data)
        new_index.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(new_index)


def test_missing_index_file(tmp_path):
    from megadata import BatchStream

    with pytest.raises(Exception):
        BatchStream(index_json_path=str(tmp_path / "nope" / "index.json"),
                     batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)


def test_index_path_is_directory(pt_index, tmp_path):
    from megadata import BatchStream

    with pytest.raises(Exception):
        BatchStream(index_json_path=os.path.dirname(pt_index),
                     batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)


def test_unsupported_manifest_version(pt_index, tmp_path):
    from megadata import BatchStream

    idx = _clone_dataset(pt_index, tmp_path, "badver",
                         lambda d: d.update({"version": "9.9"}))
    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                     world_size=1, rank=0)
    assert "Unsupported index.json version" in str(exc.value)


def test_manifest_with_no_shards(pt_index, tmp_path):
    from megadata import BatchStream

    def patch(d):
        d["shards"] = []
        d["global_stats"]["total_shards"] = 0
        d["global_stats"]["total_batches"] = 0
    idx = _clone_dataset(pt_index, tmp_path, "noshard", patch)
    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                     world_size=1, rank=0)
    assert "shard" in str(exc.value).lower()


def test_manifest_truncated_stats_mismatch(pt_index, tmp_path):
    """被截断/手改的 manifest（统计量与 shard 列表不符）必须拒绝。"""
    from megadata import BatchStream

    idx = _clone_dataset(pt_index, tmp_path, "mismatch",
                         lambda d: d["global_stats"].__setitem__("total_batches", 999999))
    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                     world_size=1, rank=0)
    assert "矛盾" in str(exc.value) or "total_batches" in str(exc.value)


def test_severely_fewer_shards_than_world_size(pt_index, tmp_path):
    """shard 数 < world_size：高位 rank 会空转，必须拒绝并给出可操作建议。"""
    from megadata import BatchStream

    idx = _clone_dataset(pt_index, tmp_path, "fewshard", lambda d: d["shards"].__setitem__(
        slice(1, None), []))
    idx = _clone_dataset(pt_index, tmp_path, "fewshard2", lambda d: (
        d["shards"].__setitem__(slice(1, None), []),
        d["global_stats"].__setitem__("total_shards", 1),
        d["global_stats"].__setitem__("total_batches", d["shards"][0]["num_batches"]),
    )[0])
    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                     world_size=8, rank=0)
    assert "world_size" in str(exc.value)


def test_batch_size_mismatch_rejected(pt_index):
    from megadata import BatchStream

    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE + 1,
                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    assert "batch_size" in str(exc.value)


def test_max_seq_len_mismatch_rejected(pt_index):
    from megadata import BatchStream

    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                     max_seq_len=MAX_SEQ_LEN * 2, world_size=1, rank=0)
    assert "max_seq_len" in str(exc.value)


def test_missing_load_params_with_legacy_manifest(pt_index, tmp_path):
    """旧数据集 manifest 没记 batch_size：必须提示显式传参，而不是静默算出 0。"""
    from megadata import BatchStream

    idx = _clone_dataset(pt_index, tmp_path, "notsizes",
                         lambda d: d["global_stats"].__setitem__("batch_size", 0))
    with pytest.raises(ValueError, match="batch_size"):
        BatchStream(index_json_path=idx, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)


def test_invalid_partition_rejected(pt_index):
    from megadata import BatchStream

    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, partition="magic")
    assert "分区" in str(exc.value) or "partition" in str(exc.value).lower()


def test_negative_epoch_rejected(pt_index):
    from megadata import BatchStream

    with pytest.raises(ValueError, match="epoch"):
        BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, epoch=-1)


def test_concurrent_iteration_rejected(pt_index):
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    it = iter(loader)
    next(it)
    with pytest.raises(RuntimeError, match="active iteration session"):
        iter(loader)
    del it


def test_load_state_dict_during_active_session_rejected(pt_index):
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    state = loader.state_dict()
    it = iter(loader)
    next(it)
    with pytest.raises(RuntimeError):
        loader.load_state_dict(state)
    del it


def test_foreign_schema_state_rejected(pt_index):
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    with pytest.raises(ValueError, match="schema"):
        loader.load_state_dict({"schema": 2, "epoch": 0, "in_epoch_pos": 0})


def test_mismatched_dataset_state_rejected(pt_index, sft_index):
    """拿另一个数据集的 checkpoint 恢复到本数据集 → 必须拒绝（不能静默错位读）。"""
    from megadata import BatchStream

    other = BatchStream(index_json_path=sft_index, batch_size=BATCH_SIZE,
                         max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    state = other.state_dict()
    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    with pytest.raises(ValueError, match="fingerprint"):
        loader.load_state_dict(state)


def test_python_objects_reject_data_loader_wrapping(pt_index):
    """RandomAccessBatchReader 明确不是 Dataset：不能套 torch DataLoader。"""
    from megadata import RandomAccessBatchReader
    from torch.utils.data import DataLoader

    reader = RandomAccessBatchReader(index_json_path=pt_index, batch_size=BATCH_SIZE,
                                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    assert not hasattr(reader, "__getitem__"), "一旦支持 __getitem__ 就会被误当 map-style Dataset"
    with pytest.raises(Exception):
        list(DataLoader(reader, batch_size=None))   # 缺 __getitem__ → 读取期报错


# ==============================================================================
# 7. MixBatchStream 细节
# ==============================================================================
def test_mixloader_validation(pt_index):
    from megadata import MixBatchStream

    with pytest.raises(ValueError, match="must not be empty"):
        MixBatchStream(configs=[], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="steps_per_epoch"):
        MixBatchStream(configs=[{"path": pt_index}], batch_size=BATCH_SIZE,
                  max_seq_len=MAX_SEQ_LEN, steps_per_epoch=0)
    with pytest.raises(ValueError, match="epoch"):
        MixBatchStream(configs=[{"path": pt_index}], batch_size=BATCH_SIZE,
                  max_seq_len=MAX_SEQ_LEN, epoch=-1)
    with pytest.raises(ValueError, match="all specified or all omitted"):
        MixBatchStream(configs=[{"path": pt_index, "weight": 1.0}, {"path": pt_index}],
                  batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="at least one weight"):
        MixBatchStream(configs=[{"path": pt_index, "weight": 0.0}, {"path": pt_index, "weight": 0.0}],
                  batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)


def test_mixloader_size_consistency(pt_index, sft_index, tmp_path):
    """子流尺寸不一致必须拒绝（异构指 PT/SFT 之别，不是尺寸之别）。"""
    from _scenario_kit import build_dataset
    from megadata import MixBatchStream

    other_msl = build_dataset(str(tmp_path / "other"), sft_records(60), "sft",
                              max_seq_len=MAX_SEQ_LEN * 2, batches_per_shard=4)
    with pytest.raises(ValueError) as exc:
        MixBatchStream(configs=[{"path": sft_index}, {"path": other_msl}],
                  batch_size=BATCH_SIZE)
    assert "max_seq_len" in str(exc.value)


def test_mixloader_natural_mode_stops_at_exhaustion(pt_index, sft_index):
    from megadata import MixBatchStream

    loader = MixBatchStream(configs=[{"path": pt_index, "weight": 0.5},
                                {"path": sft_index, "weight": 0.5}],
                       batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=3)
    n = sum(1 for _ in loader)
    assert 0 < n <= len(loader), "自然模式应在于流耗尽时结束（且不超过 len 上界）"


def test_mixloader_update_weights_changes_distribution(pt_index, sft_index):
    """课程学习：改权重后采样分布应变化（且权重被零长度掩码重新应用）。"""
    from megadata import MixBatchStream

    configs = [{"path": pt_index}, {"path": sft_index}]
    loader = MixBatchStream(configs=configs, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                       steps_per_epoch=60, seed=11)
    before = list(loader.current_weights)
    assert abs(sum(before) - 1.0) < 1e-6
    loader.update_weights([0.9, 0.1])
    after = list(loader.current_weights)
    assert after[0] > after[1], "权重更新未生效"

    fmts = [b["format"] for b in loader]
    assert fmts.count("pt") > fmts.count("sft"), "采样分布未跟随权重"


def test_mixloader_update_temperature(pt_index, sft_index):
    from megadata import MixBatchStream

    loader = MixBatchStream(configs=[{"path": pt_index}, {"path": sft_index}],
                       batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                       steps_per_epoch=20, seed=5)
    base = list(loader.current_weights)
    loader.update_temperature(3.0)
    warm = list(loader.current_weights)
    loader.update_temperature(0.5)
    cold = list(loader.current_weights)
    assert abs(sum(warm) - 1.0) < 1e-6 and abs(sum(cold) - 1.0) < 1e-6
    # 低温更"贪心"：方差变大
    assert max(cold) - min(cold) > max(warm) - min(warm)
    del base


def test_mixloader_all_ranks_empty_is_rejected(pt_index):
    """world_size 太大导致本 rank 所有子流轮长为 0 → 报错（而不是空转）。"""
    from megadata import MixBatchStream

    with pytest.raises(Exception):
        MixBatchStream(configs=[{"path": pt_index}], batch_size=BATCH_SIZE,
                  max_seq_len=MAX_SEQ_LEN, world_size=1024, rank=1023)


# ==============================================================================
# 8. RandomAccessBatchReader 细节
# ==============================================================================
def test_random_access_bounds_and_order(pt_index):
    from megadata import RandomAccessBatchReader

    reader = RandomAccessBatchReader(index_json_path=pt_index, batch_size=BATCH_SIZE,
                                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    n = len(reader)
    assert n == total_batches(pt_index)
    for bad in (n, n + 10, -n - 1):
        with pytest.raises(IndexError):
            reader.get_batch(bad)
    # 顺序访问 fast-path 与随机 seek 应给出相同结果
    seq = [batch_key(reader.get_batch(i)) for i in range(min(6, n))]
    jumped = [batch_key(reader.get_batch(i)) for i in reversed(range(min(6, n)))]
    assert jumped == list(reversed(seq))


def test_random_access_respects_world_size(pt_index):
    """按 rank 分片后互不重叠（评估场景：每个 rank 看自己的那份）。"""
    from megadata import RandomAccessBatchReader

    seen = []
    for rank in range(2):
        reader = RandomAccessBatchReader(index_json_path=pt_index, batch_size=BATCH_SIZE,
                                         max_seq_len=MAX_SEQ_LEN, world_size=2, rank=rank)
        seen.append({batch_key(reader.get_batch(i)) for i in range(len(reader))})
    assert not (seen[0] & seen[1]), "两个 rank 的随机访问集合重叠"
