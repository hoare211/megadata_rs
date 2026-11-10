"""评审修复回归（依据 `docs/reviews/code/20260916/接口设计评审汇总.md` 第四部分验收标准）。

覆盖：
- R1 / F-01：Tier-1 精确续训轮号回填（StreamingDataset 与 MixStreamingDataset）
- R2 / F-02：update_weights 错误输入 + 失败原子性；`weights=` 参数生效且与 configs 互斥
- R3 / F-07：跨轮"零长度子流"身份恒定性（确认性用例，锁定"无需跨轮重算掩码"的结论）
- R4 / F-08：恢复后 `len()` == 实际产出数（两个引擎同口径）
- F-03 / F-05 / F-10 / F-18 / F-24 / F-32 的接口契约
"""
import json
import os

import pytest
import torch

import megadata._rust as _rust
from megadata import (
    DataBuilder,
    MixBatchStream,
    MixStreamingDataset,
    MegadataDataLoader,
    BatchStream,
    StreamingDataset,
    cu_seqlens_to_doc_ids,
    normalize_cu_seqlens,
    valid_len,
)
from megadata.validate import validate

TOKENIZER_PATH = os.environ.get(
    "TOKENIZER_PATH",
    os.environ.get("BBPE_TOKENIZER_DIR", "./tokenizer"),
)
TOKENIZER_PATH = os.path.abspath(TOKENIZER_PATH)
BATCH_SIZE = 2
MAX_SEQ_LEN = 64
TEMPLATE = "{% for m in conversations %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"


@pytest.fixture(scope="module")
def dsdirs(tmp_path_factory):
    base = tmp_path_factory.mktemp("review_regr")

    pt_raw = base / "pt.jsonl"
    with open(pt_raw, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({"text": f"pt doc {i} " + "alpha beta gamma delta " * 6}) + "\n")
    pt_dir = base / "pt"
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN).build_dataset(
        [str(pt_raw)], str(pt_dir), mode="pt",
        batch_size=BATCH_SIZE, batches_per_shard=4, compression="zstd")

    sft_raw = base / "sft.jsonl"
    with open(sft_raw, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i} text"},
            ]}) + "\n")
    sft_dir = base / "sft"
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN).build_dataset(
        [str(sft_raw)], str(sft_dir), mode="sft", template_str=TEMPLATE,
        batch_size=BATCH_SIZE, batches_per_shard=4)

    return {"pt": str(pt_dir / "index.json"), "sft": str(sft_dir / "index.json")}


def _stream(idx):
    return BatchStream(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                        world_size=1, rank=0, seed=7)


# ============================================================
# R1 / F-01：Tier-1 续训轮号回填
# ============================================================
def test_r1_f01_tier1_resume_epoch_backfill(dsdirs):
    idx = dsdirs["pt"]
    total = len(_stream(idx))

    ds = StreamingDataset(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    dl = MegadataDataLoader(ds, num_workers=0)
    for _ in dl:                                   # epoch 0 完整
        pass
    assert ds.epoch == 1
    it = iter(dl)
    next(it)                                       # epoch 1 内 1 步（保持会话存活，不再使用）
    st = dl.state_dict()
    assert st["tier1"]["epoch"] == 1

    ds2 = StreamingDataset(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=7)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    dl2.load_state_dict(st)
    assert ds2.epoch == 1, "load_state_dict 后轮号必须回填（不能是构造期 0）"

    n_resumed = sum(1 for _ in dl2)                # 恢复轮：产出 total-1
    assert n_resumed == total - 1
    assert ds2.epoch == 2, "恢复轮跑完后轮号必须前进到 k+1（不得回退/重放）"

    n_next = sum(1 for _ in iter(dl2))             # 下一轮：完整一轮（未被重放污染）
    assert n_next == total


def test_r1_f01_mix_tier1_resume_epoch_backfill(dsdirs):
    ds = MixStreamingDataset(
        [{"path": dsdirs["pt"], "weight": 1.0}, {"path": dsdirs["sft"], "weight": 0.3}],
        batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7,
        steps_per_epoch=10)
    dl = MegadataDataLoader(ds, num_workers=0)
    assert sum(1 for _ in dl) == 10
    assert ds.epoch == 1

    it = iter(dl)
    for _ in range(3):
        next(it)
    st = dl.state_dict()
    assert st["tier1"]["epoch"] == 1

    ds2 = MixStreamingDataset(
        [{"path": dsdirs["pt"], "weight": 1.0}, {"path": dsdirs["sft"], "weight": 0.3}],
        batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7,
        steps_per_epoch=10)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    dl2.load_state_dict(st)
    assert ds2.epoch == 1
    assert sum(1 for _ in dl2) == 7                # N - k = 10 - 3（确定性模式剩余步数）
    assert ds2.epoch == 2


# ============================================================
# R2 / F-02：update_weights 校验与失败原子性
# ============================================================
def test_r2_f02_update_weights_validation_and_atomicity(dsdirs):
    mix = MixBatchStream([{"path": dsdirs["pt"]}, {"path": dsdirs["sft"]}],
                    batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                    world_size=1, rank=0, seed=7)
    with pytest.raises(ValueError):
        mix.update_weights([0.5])                  # 2 子流却只给 1 个：长度不符必须报错
    before = list(mix.current_weights)
    for bad in ([1.0, float("nan")], [-1.0, 1.0], [0.0, 0.0]):
        with pytest.raises(ValueError):
            mix.update_weights(bad)
        assert mix.current_weights == before, f"失败路径不得改状态（{bad}）"
    assert list(mix.state_dict()["weights"]) == before, "state_dict 不得导出未生效的权重"
    # 合法更新仍工作
    mix.update_weights([1.0, 0.0])
    assert mix.current_weights == [1.0, 0.0]


def test_r2_f02_weights_kwarg_effective_and_exclusive(dsdirs):
    mix = MixBatchStream([{"path": dsdirs["pt"]}, {"path": dsdirs["sft"]}], weights=[0.9, 0.1],
                    batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                    world_size=1, rank=0, seed=7)
    assert abs(mix.current_weights[0] - 0.9) < 1e-9, "weights= 参数必须生效"
    with pytest.raises(ValueError):
        MixBatchStream([{"path": dsdirs["pt"], "weight": 1.0}, {"path": dsdirs["sft"]}],
                  weights=[0.5, 0.5], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                  world_size=1, rank=0)
    with pytest.raises(ValueError):
        MixBatchStream([{"path": dsdirs["pt"]}, {"path": dsdirs["sft"]}], weights=[0.5],
                  batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)


# ============================================================
# R3 / F-07：跨轮"零长度子流"身份恒定性（确认性）
# ============================================================
@pytest.mark.parametrize("partition", ["round_robin", "relaxed"])
@pytest.mark.parametrize("world_size,num_workers", [(1, 4), (2, 2)])
def test_r3_f07_zero_length_identity_stable(dsdirs, partition, world_size, num_workers):
    """F-07 施工期确认：固定拓扑下"轮长为 0"的 (worker, 子流) 身份跨轮恒定。"""
    idx = dsdirs["pt"]
    total_workers = world_size * num_workers
    prev = [[] for _ in range(total_workers)]
    first = None
    for e in range(1, 5):
        cur = _rust.plan_shard_sequence(idx, world_size, num_workers, 11, e,
                                        True, partition, "random", 0)
        lens = [len(c) - len(p) for c, p in zip(cur, prev)]
        prev = cur
        zero = [l == 0 for l in lens]
        if first is None:
            first = zero
        else:
            assert zero == first, f"{partition} W={total_workers}: 轮长 0 身份跨轮变化"


# ============================================================
# R4 / F-08：恢复后 len() == 实际产出数
# ============================================================
def test_r4_f08_stream_len_exact(dsdirs):
    l1 = _stream(dsdirs["pt"])
    for _ in l1:
        pass
    it = iter(l1)
    for _ in range(3):
        next(it)
    st = l1.state_dict()
    l2 = _stream(dsdirs["pt"])
    l2.load_state_dict(st)
    length = len(l2)
    produced = sum(1 for _ in l2)
    assert length == produced, f"len()={length} != 实际产出={produced}"


def test_r4_f08_mix_len_matches_produced_after_resume(dsdirs):
    cfg = [{"path": dsdirs["pt"], "weight": 1.0}, {"path": dsdirs["sft"], "weight": 0.3}]
    m1 = MixBatchStream(cfg, steps_per_epoch=10, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                   world_size=1, rank=0, seed=7)
    it = iter(m1)
    for _ in range(3):
        next(it)
    st = m1.state_dict()
    m2 = MixBatchStream(cfg, steps_per_epoch=10, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                   world_size=1, rank=0, seed=7)
    m2.load_state_dict(st)
    assert len(m2) == 7
    assert sum(1 for _ in m2) == 7


# ============================================================
# 接口契约回归：F-03 / F-05 / F-10 / F-18 / F-24 / F-32
# ============================================================
def test_f03_megadata_dataloader_rejects_packaging_kwargs(dsdirs):
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    with pytest.raises(ValueError, match="pre-packed"):
        MegadataDataLoader(ds, num_workers=0, batch_size=8)
    for kw in ("shuffle", "sampler"):
        with pytest.raises(ValueError):
            MegadataDataLoader(ds, num_workers=0, **{kw: True})


def test_f05_close_is_terminal(dsdirs):
    with _stream(dsdirs["pt"]) as l:
        assert sum(1 for _ in l) > 0
    l.close()                                      # 幂等
    with pytest.raises(RuntimeError):
        iter(l)


def test_f10_all_batches_carry_format(dsdirs):
    b_pt = next(iter(_stream(dsdirs["pt"])))
    b_sft = next(iter(_stream(dsdirs["sft"])))
    assert b_pt["format"] == "pt" and b_sft["format"] == "sft"
    mix_same_source = MixBatchStream([{"path": dsdirs["pt"], "weight": 1.0},
                                 {"path": dsdirs["pt"], "weight": 0.5}],
                                batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                                world_size=1, rank=0, seed=7)
    assert next(iter(mix_same_source))["format"] == "pt"


def test_f18_set_epoch_and_direct_iteration_warns(dsdirs):
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    with pytest.warns(UserWarning, match="advance the epoch counter"):
        for _, _b in zip(range(2), ds):            # 直接迭代（不经 MegadataDataLoader）
            pass
    ds.set_epoch(3)
    assert ds.epoch == 3
    with pytest.raises(ValueError):
        ds.set_epoch(-1)


def test_f24_mix_epoch_advances_and_resumes(dsdirs):
    cfg = [{"path": dsdirs["pt"], "weight": 1.0}, {"path": dsdirs["sft"], "weight": 0.3}]
    m = MixBatchStream(cfg, steps_per_epoch=1, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                  world_size=1, rank=0, seed=7)
    assert m.epoch == 0
    for _ in range(5):
        next(iter(m))                              # N=1：每轮 1 步，轮结束生成器自然 return
    assert m.epoch == 5, "epoch 属性必须随迭代前进（不再冻结）"
    st = m.state_dict()
    m2 = MixBatchStream(cfg, steps_per_epoch=1, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                   world_size=1, rank=0, seed=7)
    m2.load_state_dict(st)
    assert m2.epoch == 5, "恢复后 epoch 必须反映恢复到的轮号"


@pytest.mark.real_tokenizer
def test_f32_validate_reports(dsdirs):
    rep = validate(dsdirs["pt"], world_size=1, expect_batch_size=BATCH_SIZE)
    assert rep["ok"], [c for c in rep["checks"] if not c["ok"]]
    rep_bad = validate(dsdirs["pt"], world_size=10000)
    assert not rep_bad["ok"], "shard 数不足必须被 validate 检出"


# ============================================================
# mask_adapter 纯函数契约（参数化；property-based 的轻量替代）
# ============================================================
@pytest.mark.parametrize("cu,expect_valid", [
    ([0, 3, 8], 8),
    ([0, 1, 1, 5, 10], 10),        # 连续重复边界点被去重
    ([0, 16], 16),
    ([3, 5, 9], 9),                # 首元素非 0 → 自动补 0
])
def test_mask_doc_ids_contract(cu, expect_valid):
    n = valid_len(cu)
    assert n == expect_valid
    ids = cu_seqlens_to_doc_ids(cu)
    assert ids.numel() == n, "doc_ids 长度必须等于 valid_len"
    assert ids.dtype == torch.long
    if n:
        assert torch.all(ids[1:] >= ids[:-1]), "doc_ids 必须单调不减"
        assert int(ids[0]) == 0 and int(ids[-1]) == int(ids[-1])
    norm = normalize_cu_seqlens(cu)
    assert int(norm[0]) == 0, "归一化后首元素必须为 0"
    assert int(norm[-1]) == n
