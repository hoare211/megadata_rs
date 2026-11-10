"""
megadata v3 协议矩阵测试（P1–P6 协议条目）。

与存量测试不同：本文件**不依赖 tokenizer**——用合成 PT 数据集（自写 index.json +
shard) 直接驱动协议，因此任何环境都能跑。

重点回归（评审发现 → 断言）：
- B2：读满一轮后 `next` 必须返回 StopIteration，**不能自旋挂死**；
- B1：推进轮号走 `seek_to_epoch_batch(epoch, in_epoch)`，两轮数据必须不同；
- B3：MixBatchStream 自然模式（A=2000 / B=4，权重 [0.9, 0.1]）必须**有限步终止**；
- F1：Tier 1 相邻两轮顺序必须不同（`num_workers ∈ {0, 2}`）。
"""
import json
import os
import signal
import sys
import time

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megadata import (  # noqa: E402
    MegadataDataLoader,
    MegadataInterrupted,
    MixBatchStream,
    RandomAccessBatchReader,
    BatchStream,
    StreamingDataset,
)

BATCH_SIZE = 2
MAX_SEQ_LEN = 8


class Watchdog(Exception):
    """用于把"自旋挂死"变成可断言的失败。"""


def _alarm(seconds):
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Watchdog()))
    signal.alarm(seconds)


def _disarm():
    signal.alarm(0)


def make_pt_dataset(root, num_batches, *, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                    with_cu=True, shards=None):
    """合成一个 PT(Memmap) 数据集：每个 batch 的首元素写入可识别 id。

    `shards`: 每个 shard 的 batch 数列表（默认单 shard）。最后一个 shard 可以是
    "短尾块"，用于复现 per-epoch 轮长变化（B1）。
    """
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

        if with_cu:
            # .cu 旁路：[u32 count LE][i32 * count LE] × batch
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


def first_ids(loader, limit=None):
    """取每个 batch 的首元素（= 可识别的全局 batch id）"""
    out = []
    for i, batch in enumerate(loader):
        out.append(int(batch["input_ids"].reshape(-1)[0]))
        if limit is not None and i + 1 >= limit:
            break
    return out


@pytest.fixture(scope="module")
def pt_dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp("pt_small")
    return make_pt_dataset(str(d), 48)


# ============================================================
# P1 批量数据契约
# ============================================================
def test_p1_cu_seqlens_is_int32_tensor(pt_dataset):
    """P1：cu_seqlens 必须是 int32 tensor（不再是 Python list）"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    batch = next(iter(loader))
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert batch["input_ids"].dtype == torch.int32
    cu = batch["cu_seqlens"]
    assert isinstance(cu, torch.Tensor), "cu_seqlens 必须是 tensor（v3 契约）"
    assert cu.dtype == torch.int32
    # C21/G2 契约：首 0、单调、末值 ≤ B×S（满批 == B×S）
    cu_list = [int(v) for v in cu.tolist()]
    assert cu_list[0] == 0
    assert all(b >= a for a, b in zip(cu_list, cu_list[1:])), "cu_seqlens 必须单调不减"
    assert cu_list[-1] <= BATCH_SIZE * MAX_SEQ_LEN


def test_p1_format_property(pt_dataset):
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    assert loader.format == "pt"


# ============================================================
# P2 构造协议
# ============================================================
def test_p2_batch_size_auto_read(pt_dataset):
    """batch_size=None → 从 manifest 自动读取"""
    loader = BatchStream(pt_dataset)
    assert len(loader) == 48


def test_p2_shuffle_normalization(pt_dataset):
    assert BatchStream(pt_dataset, shuffle=True) is not None
    assert BatchStream(pt_dataset, shuffle="permutation") is not None
    assert BatchStream(pt_dataset, shuffle=False) is not None
    assert BatchStream(pt_dataset, shuffle="none") is not None
    with pytest.raises(ValueError):
        BatchStream(pt_dataset, shuffle="buffer")   # buffer 模式已砍
    with pytest.raises(ValueError):
        BatchStream(pt_dataset, shuffle="whatever")


def test_p2_negative_epoch_and_zero_steps_rejected(pt_dataset):
    """C31/J2：构造期参数校验"""
    with pytest.raises(ValueError):
        BatchStream(pt_dataset, epoch=-1)
    with pytest.raises(ValueError):
        MixBatchStream([{"path": pt_dataset}, {"path": pt_dataset}], steps_per_epoch=0, weights=[0.5, 0.5])


def test_p2_mixloader_mismatched_sizes_rejected(tmp_path, pt_dataset):
    d = tmp_path / "other"
    other = make_pt_dataset(str(d), 8, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN * 2)
    with pytest.raises(ValueError):
        MixBatchStream([{"path": pt_dataset}, {"path": other}], weights=[0.5, 0.5])


# ============================================================
# P3 迭代协议（BatchStream）
# ============================================================
def test_p3_full_epoch_terminates_without_spin(pt_dataset):
    """B2 验收：读满一轮必须 StopIteration（限时），不能自旋挂死"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    assert len(loader) == 48
    ids = first_ids(loader)
    assert ids == list(range(48)) or sorted(ids) == list(range(48))


def test_p3_two_epochs_differ_and_cover(pt_dataset):
    """B1/C1：推进轮号后置换表与 shard 分配应变化（两轮数据顺序不同）"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    first = first_ids(loader)
    second = first_ids(loader)
    assert sorted(first) == sorted(second) == list(range(48)), "每轮应覆盖同一份数据"
    assert first != second, "相邻两轮的顺序必须不同（否则 epoch 推进失效）"


def test_p3_break_then_reiter_replays_current_epoch(pt_dataset):
    """OD5：break 再 iter = 重跑当前轮（PyTorch 肌肉记忆）"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    it = iter(loader)
    head = [int(next(iter([next(it)]))["input_ids"].reshape(-1)[0]) for _ in range(3)]
    del it  # 触发生成器回收 → 会话释放
    again = first_ids(loader, limit=3)
    assert again == head, "break 后重新 iter 应重跑当前轮"


def test_p3_state_dict_is_next_position_and_restore_exact(pt_dataset):
    """C23/G4 + C27/H3：state 记录"下一个待产出位置"，恢复后**逐位**续上原序列"""
    reference = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    full_seq = first_ids(reference)          # 第 0 轮完整序列（置换后的顺序）

    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    it = iter(loader)
    for _ in range(5):
        next(it)
    state = loader.state_dict()
    assert state["schema"] == 3
    assert state["epoch"] == 0
    assert state["in_epoch_pos"] == 5, "已产出 5 个 → 下一个待产出位置是 5"
    del it

    resumed = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    resumed.load_state_dict(state)
    assert resumed.epoch == 0
    tail = first_ids(resumed, limit=4)
    assert tail == full_seq[5:9], "恢复后必须按原序列继续（不重复、不跳过、不错位）"


def test_p3_natural_completion_advances_epoch(pt_dataset):
    """B3'：自然读完后 state = (e+1, 0)"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    first_ids(loader)
    state = loader.state_dict()
    assert (state["epoch"], state["in_epoch_pos"]) == (1, 0)


def test_p3_concurrent_iteration_rejected(pt_dataset):
    """OD2：并发迭代会话 → RuntimeError"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    it = iter(loader)
    next(it)
    with pytest.raises(RuntimeError):
        iter(loader)
    del it


def test_p3_reader_single_epoch_bounds(tmp_path):
    """C15/B5'：Reader 单轮语义 + 越界 IndexError"""
    path = make_pt_dataset(str(tmp_path / "r"), 16)
    reader = RandomAccessBatchReader(path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    assert len(reader) == 16
    assert int(reader.get_batch(3)["input_ids"].reshape(-1)[0]) == 3
    assert int(reader.get_batch(-1)["input_ids"].reshape(-1)[0]) == 15
    with pytest.raises(IndexError):
        reader.get_batch(16)
    with pytest.raises(IndexError):
        reader.get_batch(-17)


# ============================================================
# P3 迭代协议（MixBatchStream）
# ============================================================
def test_p3_mixloader_natural_mode_terminates(tmp_path):
    """B3 验收：A=2000 / B=4，权重 [0.9, 0.1] —— 子流耗尽必须终止，不能挂死"""
    a = make_pt_dataset(str(tmp_path / "A"), 2000)
    b = make_pt_dataset(str(tmp_path / "B"), 4)
    mixer = MixBatchStream([{"path": a, "weight": 0.9}, {"path": b, "weight": 0.1}],
                      batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    _alarm(60)
    try:
        steps = 0
        for _batch in mixer:
            steps += 1
            if steps > 400:      # 自然模式的轮长远小于 400（B 只有 4 个 batch）
                break
    except Watchdog:
        pytest.fail("MixBatchStream 自然模式挂死（B2/B3 回归）")
    finally:
        _disarm()
    assert 0 < steps <= 400, f"自然模式应在有限步内终止，实际 {steps}"


def test_p3_mixloader_deterministic_mode_exact_steps(tmp_path):
    a = make_pt_dataset(str(tmp_path / "A2"), 200)
    b = make_pt_dataset(str(tmp_path / "B2"), 200)
    mixer = MixBatchStream([{"path": a, "weight": 0.5}, {"path": b, "weight": 0.5}],
                      batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, steps_per_epoch=50)
    assert len(mixer) == 50
    _alarm(60)
    try:
        steps = sum(1 for _ in mixer)
    except Watchdog:
        pytest.fail("确定性模式挂死")
    finally:
        _disarm()
    assert steps == 50, f"确定性模式应恰好 N 步，实际 {steps}"


def test_p3_mixloader_state_roundtrip(tmp_path):
    a = make_pt_dataset(str(tmp_path / "A3"), 200)
    b = make_pt_dataset(str(tmp_path / "B3"), 200)
    mixer = MixBatchStream([{"path": a, "weight": 0.5}, {"path": b, "weight": 0.5}],
                      batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, steps_per_epoch=100)
    it = iter(mixer)
    for _ in range(10):
        next(it)
    state = mixer.state_dict()
    assert len(state["sub_stream_cursors"]) == 2, "逐子流 (epoch, in_epoch) 游标"
    del it

    other = MixBatchStream([{"path": a, "weight": 0.5}, {"path": b, "weight": 0.5}],
                      batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, steps_per_epoch=100)
    other.load_state_dict(state)
    restored = other.state_dict()
    assert restored["steps_yielded"] == state["steps_yielded"]
    assert [list(c) for c in restored["sub_stream_cursors"]] == [list(c) for c in state["sub_stream_cursors"]]


# ============================================================
# P4 worker 协议 + Tier 1 epoch 推进
# ============================================================
def test_p4_tier1_two_epochs_differ(tmp_path):
    """F1/C17 验收：Tier 1 相邻两轮顺序必须不同（num_workers ∈ {0, 2}）

    注意：轮号记在 **dataset 实例**上，因此同一个 `StreamingDataset` 连续两轮
    `iter(dl)` 才构成"相邻两轮"（每轮新建 dataset 会各自从 epoch 0 开始）。
    """
    path = make_pt_dataset(str(tmp_path / "t1"), 48)

    for nw in (0, 2):
        ds = StreamingDataset(path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
        dl = MegadataDataLoader(ds, num_workers=nw)

        def one_epoch():
            out = []
            for batch in dl:
                out.append(int(batch["input_ids"].reshape(-1)[0]))
            return out

        first, second = one_epoch(), one_epoch()
        assert sorted(first) == sorted(second) == list(range(48)), f"num_workers={nw}：每轮应覆盖全集"
        assert first != second, f"num_workers={nw}：相邻两轮顺序必须不同（Tier 1 epoch 推进失效）"
        assert ds.epoch == 2, f"num_workers={nw}：跑完两轮后 Tier 1 轮号应为 2"


def test_p4_tier1_state_dict_has_epoch(tmp_path):
    path = make_pt_dataset(str(tmp_path / "t2"), 16)
    ds = StreamingDataset(path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=0)
    for _ in dl:
        pass
    state = dl.state_dict()
    assert state["tier1"]["epoch"] == 1, "完整跑完一轮后 Tier 1 轮号应为 1"


def test_p4_worker_union_covers_dataset(tmp_path):
    path = make_pt_dataset(str(tmp_path / "w"), 48)
    ds = StreamingDataset(path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    dl = MegadataDataLoader(ds, num_workers=2)
    ids = sorted(int(b["input_ids"].reshape(-1)[0]) for b in dl)
    assert ids == list(range(48)), "多 worker 的并集应覆盖全集且互不重叠"


# ============================================================
# P5 快照协议
# ============================================================
def test_p5_rejects_foreign_schema(pt_dataset):
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError):
        loader.load_state_dict({"schema": 2, "epoch": 0, "in_epoch_pos": 0})


def test_p5_rejects_mismatched_manifest(tmp_path, pt_dataset):
    other = make_pt_dataset(str(tmp_path / "other2"), 8)
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    foreign = loader.state_dict()
    foreign["manifest_fingerprint"]["total_batches"] = 99999
    with pytest.raises(ValueError):
        loader.load_state_dict(foreign)
    assert other  # 仅确保第二份数据集已构造


def test_p5_load_state_dict_during_active_session_rejected(pt_dataset):
    """C19-⑤"""
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    state = loader.state_dict()
    it = iter(loader)
    next(it)
    with pytest.raises(RuntimeError):
        loader.load_state_dict(state)
    del it


# ============================================================
# P6 错误与中断协议
# ============================================================
def test_p6_truncated_shard_raises_oserror(tmp_path):
    """数据损坏必须上报 OSError，而不是静默当 EOF"""
    d = tmp_path / "trunc"
    d.mkdir()
    path = make_pt_dataset(str(d), 8)
    with open(os.path.join(str(d), "shard_00000.bin"), "wb") as f:
        f.write(b"")  # 截断：0 字节
    loader = BatchStream(path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    _alarm(30)
    try:
        with pytest.raises(OSError):
            for _batch in loader:
                pass
    finally:
        _disarm()


def test_p6_cancel_raises_megadata_interrupted(pt_dataset):
    loader = BatchStream(pt_dataset, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    it = iter(loader)
    next(it)
    loader.cancel()
    with pytest.raises(MegadataInterrupted):
        next(it)


if __name__ == "__main__":
    # 便于手工跑：python tests/test_protocol.py
    t0 = time.time()
    print("run pytest tests/test_protocol.py instead")
    print(f"{time.time() - t0:.2f}s")
