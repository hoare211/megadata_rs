"""解决方案（接口评审修复）的验收回归。

锁定四条设计原则的落地行为：
- A1 单一状态权威：close 终态 / release 非终态 / 推进单一实现点 / 类型守卫接线
- A2 意图与生效分离：base_weights 不被温度覆写；温度幂等；update_weights 接管 base
- A3 查询零副作用：`len()` 不构造读头；resume 精度自述
- A4 数据自描述优先：SFT row_lens 列 + flatten/repad 对称往返
- A5 失败响亮且论据诚实：模式无关参数报错、精度告警只在真丢位置时发出
"""
import json
import os
import warnings

import pytest
import torch

from megadata import (
    DataBuilder,
    LoaderOptions,
    MegadataDataLoader,
    MixBatchStream,
    MixStreamingDataset,
    RandomAccessBatchReader,
    BatchStream,
    StreamingDataset,
    is_pt_batch,
    is_sft_batch,
)
from megadata._options import (
    normalize_compression,
    normalize_mode,
    normalize_partition,
    normalize_shuffle,
)
from megadata.builder import BuildConfig, Marker
from megadata.mask_adapter import flatten_and_unpad, repad_to_batch, sft_row_lens

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
    base = tmp_path_factory.mktemp("solution2")

    pt_raw = base / "pt.jsonl"
    with open(pt_raw, "w", encoding="utf-8") as f:
        for i in range(120):
            f.write(json.dumps({"text": f"pt doc {i} " + "alpha beta gamma delta " * 5}) + "\n")
    pt_dir = base / "pt"
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN).build_dataset(
        BuildConfig(input_paths=[str(pt_raw)], output_dir=str(pt_dir), mode="pt",
                    batch_size=BATCH_SIZE, batches_per_shard=4, compression="zstd"))

    sft_raw = base / "sft.jsonl"
    with open(sft_raw, "w", encoding="utf-8") as f:
        for i in range(120):
            f.write(json.dumps({"conversations": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": f"answer {i} text"},
            ]}) + "\n")
    sft_dir = base / "sft"
    DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN).build_dataset(
        BuildConfig(input_paths=[str(sft_raw)], output_dir=str(sft_dir), mode="sft",
                    template_str=TEMPLATE, batch_size=BATCH_SIZE, batches_per_shard=4))

    return {"pt": str(pt_dir / "index.json"), "sft": str(sft_dir / "index.json")}


def _stream(idx):
    return BatchStream(idx, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                        world_size=1, rank=0, seed=7)


def _mix_cfg(dsdirs):
    # 不带 configs 内 weight：权重来源由各用例显式给出（weights= 或 token 配比），
    # 二者与 configs.weight 互斥（权重来源必须唯一）
    return [{"path": dsdirs["pt"]}, {"path": dsdirs["sft"]}]


# ============================================================
# A2 / #7b：权重意图与生效分离
# ============================================================
def test_a2_temperature_keeps_explicit_weights(dsdirs):
    mix = MixBatchStream(_mix_cfg(dsdirs), weights=[0.8, 0.2],
                    batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                    world_size=1, rank=0, seed=7)
    mix.update_temperature(1.0)
    assert mix.current_weights == pytest.approx([0.8, 0.2], abs=1e-9), \
        "T=1 必须精确保持显式权重（不得回退 token 配比）"
    # 显式权重 + T>1：作用于权重而非 token 配比
    mix2 = MixBatchStream(_mix_cfg(dsdirs), weights=[0.8, 0.2],
                     batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                     world_size=1, rank=0, seed=7)
    mix2.update_temperature(2.0)
    exp = [0.8 ** 0.5, 0.2 ** 0.5]
    s = sum(exp)
    assert mix2.current_weights == pytest.approx([e / s for e in exp], abs=1e-9)


def test_a2_temperature_is_idempotent(dsdirs):
    a = MixBatchStream(_mix_cfg(dsdirs), weights=[0.8, 0.2], batch_size=BATCH_SIZE,
                  max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    a.update_temperature(2.0)
    a.update_temperature(4.0)
    b = MixBatchStream(_mix_cfg(dsdirs), weights=[0.8, 0.2], batch_size=BATCH_SIZE,
                  max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    b.update_temperature(4.0)
    assert a.current_weights == pytest.approx(b.current_weights, abs=1e-12), \
        "连续退火必须幂等（不得复合为 1/(T1*T2)）"


def test_a2_update_weights_takes_over_base(dsdirs):
    mix = MixBatchStream(_mix_cfg(dsdirs), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                    world_size=1, rank=0, seed=7)   # 构造期省略权重 = token 配比
    mix.update_weights([0.9, 0.1])
    mix.update_temperature(2.0)
    exp = [0.9 ** 0.5, 0.1 ** 0.5]
    s = sum(exp)
    assert mix.current_weights == pytest.approx([e / s for e in exp], abs=1e-9), \
        "update_weights 后的退火必须作用于用户最新意图（而不是 token 配比）"
    # 再 update_weights：温度清零（用户显式接管）
    mix.update_weights([0.5, 0.5])
    assert mix.current_weights == pytest.approx([0.5, 0.5], abs=1e-9)


def test_a2_state_dict_carries_base_and_temperature(dsdirs):
    mix = MixBatchStream(_mix_cfg(dsdirs), weights=[0.7, 0.3], batch_size=BATCH_SIZE,
                    max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    mix.update_temperature(3.0)
    st = mix.state_dict()
    assert st["base_weights"] == [0.7, 0.3] and st["temperature"] == 3.0
    mix2 = MixBatchStream(_mix_cfg(dsdirs), weights=[0.7, 0.3], batch_size=BATCH_SIZE,
                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    mix2.load_state_dict(st)
    assert mix2.current_weights == pytest.approx(mix.current_weights, abs=1e-12)
    # 旧快照（无 base_weights/temperature）回退：weights 当 base
    legacy = {k: v for k, v in st.items()
              if k not in ("base_weights", "temperature")}
    mix3 = MixBatchStream(_mix_cfg(dsdirs), weights=[0.7, 0.3], batch_size=BATCH_SIZE,
                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    mix3.load_state_dict(legacy)
    assert mix3.current_weights == pytest.approx(mix.current_weights, abs=1e-9)


# ============================================================
# A1 / #2：Tier-1 生命周期（close 终态 / release 非终态 / with）
# ============================================================
def test_a1_tier1_with_and_close_terminal(dsdirs):
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    with MegadataDataLoader(ds, num_workers=0) as dl:
        assert sum(1 for _ in dl) > 0
    # with 退出 = close 终态：数据集不可再用
    with pytest.raises(RuntimeError, match="terminal state"):
        next(iter(ds))
    with pytest.raises(RuntimeError, match="terminal state"):
        len(ds)
    ds.close()   # 幂等


def test_a1_tier1_release_is_not_terminal(dsdirs):
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    dl = MegadataDataLoader(ds, num_workers=0)
    assert sum(1 for _ in dl) > 0       # 第一轮
    ds.release()                        # 非终态：释放读头
    assert sum(1 for _ in dl) > 0       # 下一轮自动重建、仍可迭代
    ds.close()
    with pytest.raises(RuntimeError):
        next(iter(dl))


def test_a1_engine_len_after_close_raises(dsdirs):
    """close 后 `len()` / `state_dict()` 必须报错（旧实现 len 会偷偷重建读头）。"""
    l = _stream(dsdirs["pt"])
    assert len(l) > 0
    l.close()
    with pytest.raises(RuntimeError):
        len(l)
    with pytest.raises(RuntimeError):
        l.state_dict()


def test_a1_engine_set_epoch(dsdirs):
    l = _stream(dsdirs["pt"])
    l.set_epoch(5)
    assert l.epoch == 5
    l2 = _stream(dsdirs["pt"])
    with pytest.raises(ValueError):
        l2.set_epoch(-1)
    it = iter(l2)
    next(it)                            # 活跃会话
    with pytest.raises(RuntimeError):
        l2.set_epoch(3)


def test_a1_mix_set_epoch_rebuilds_and_rejects_active(dsdirs):
    m = MixBatchStream(_mix_cfg(dsdirs), steps_per_epoch=4, batch_size=BATCH_SIZE,
                  max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    m.set_epoch(3)
    assert m.epoch == 3
    assert sum(1 for _ in m) == 4       # 从轮 3 起完整跑一轮
    assert m.epoch == 4


def test_a1_mix_tier1_lifecycle_and_len(dsdirs):
    ds = MixStreamingDataset(_mix_cfg(dsdirs), steps_per_epoch=6, batch_size=BATCH_SIZE,
                             max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=7)
    dl = MegadataDataLoader(ds, num_workers=0)
    assert len(dl) == 6                     # 确定性模式：len 直接由 steps_per_epoch 给出
    assert sum(1 for _ in dl) == 6
    ds.release()                            # 非终态：释放读头后仍可迭代
    assert sum(1 for _ in iter(dl)) == 6
    with MegadataDataLoader(ds, num_workers=0) as dl2:
        assert sum(1 for _ in dl2) == 6
    with pytest.raises(RuntimeError, match="terminal state"):   # with 退出 = close（终态）
        next(iter(ds))


def test_a1_random_access_set_epoch(dsdirs):
    r = RandomAccessBatchReader(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                                world_size=1, rank=0, seed=7)
    n0 = len(r)
    assert n0 > 0
    b = r.get_batch(0)
    assert "input_ids" in b
    r.set_epoch(1)
    assert len(r) > 0
    r.get_batch(0)
    r.close()
    with pytest.raises(RuntimeError):
        len(r)


# ============================================================
# A3 / #6b / #6a：len 纯函数 + resume 精度自述
# ============================================================
def test_a3_tier1_len_is_pure_query(dsdirs):
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    n1 = len(ds)                        # 未迭代过：纯函数查询
    assert ds._current_loader is None, "len() 不得构造读头（A3 零副作用）"
    assert n1 == len(_stream(dsdirs["pt"]))
    dl = MegadataDataLoader(ds, num_workers=0)
    assert len(dl) == n1, "Tier-1 len 与引擎层同口径"
    ds.close()


def test_a3_len_after_resume_is_remaining(dsdirs):
    total = len(_stream(dsdirs["pt"]))
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    dl = MegadataDataLoader(ds, num_workers=0)
    it = iter(dl)
    for _ in range(3):
        next(it)
    st = dl.state_dict()
    ds2 = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=7)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    dl2.load_state_dict(st)
    assert len(dl2) == total - 3
    assert sum(1 for _ in dl2) == total - 3
    ds.close(); ds2.close()


def test_a5_resume_granularity_self_report(dsdirs):
    ds = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=7)
    dl = MegadataDataLoader(ds, num_workers=0)
    for _ in dl:
        pass
    st = dl.state_dict()
    assert st["resume_granularity"] == "batch", "num_workers=0 应自述批级精度"
    # 轮级快照 + 有产出 → 必须告警
    ds2 = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=7)
    dl2 = MegadataDataLoader(ds2, num_workers=0)
    with pytest.warns(UserWarning, match="epoch-level"):
        dl2.load_state_dict({"num_batches_yielded": 3, "dataset": {},
                             "resume_granularity": "epoch"})
    # 未迭代就 save（yielded=0）→ 不得误报
    ds3 = StreamingDataset(dsdirs["pt"], batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=7)
    dl3 = MegadataDataLoader(ds3, num_workers=0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dl3.load_state_dict({"num_batches_yielded": 0, "dataset": {},
                             "resume_granularity": "epoch"})
        assert not [r for r in rec if "epoch-level" in str(r.message)], \
            "未产出过批次时轮级快照不告警（论据诚实：不存在'重复'）"
    ds.close(); ds2.close(); ds3.close()


# ============================================================
# A4 / #6c：SFT row_lens 列 + flatten/repad 对称
# ============================================================
def test_a4_sft_row_lens_column_and_roundtrip(dsdirs):
    b = next(iter(_stream(dsdirs["sft"])))
    assert "row_lens" in b, "新构建的 SFT 数据集必须携带 row_lens 列（A4 数据自描述）"
    assert tuple(b["row_lens"].shape) == (BATCH_SIZE,)
    rl = [int(v) for v in b["row_lens"].tolist()]
    assert sum(rl) == int(b["cu_seqlens"][-1])
    # 列直读 + labels 行尾不变式校验
    assert sft_row_lens(b["cu_seqlens"], BATCH_SIZE, MAX_SEQ_LEN,
                        row_lens=b["row_lens"], labels=b["labels"]) == rl
    # flatten / repad 对称往返
    flat = flatten_and_unpad(b["input_ids"], b["cu_seqlens"], packer="sft",
                             row_lens=b["row_lens"])
    assert int(flat.shape[0]) == int(b["cu_seqlens"][-1])
    back = repad_to_batch(flat, b["cu_seqlens"], BATCH_SIZE, MAX_SEQ_LEN,
                          pad_value=0, packer="sft", row_lens=b["row_lens"])
    assert torch.equal(back, b["input_ids"]), "SFT repad 必须与 flatten 严格互逆"
    # 复刻路径（模拟旧数据集：不给 row_lens）结果一致
    assert sft_row_lens(b["cu_seqlens"], BATCH_SIZE, MAX_SEQ_LEN) == rl
    # PT 数据集不得有 row_lens 键
    bp = next(iter(_stream(dsdirs["pt"])))
    assert "row_lens" not in bp


# ============================================================
# A5 / #7a：BuildConfig 模式相关性校验 + 兼容 shim
# ============================================================
def test_a5_buildconfig_mode_guards():
    with pytest.raises(ValueError, match="does not use eos"):
        BuildConfig(input_paths=["a"], output_dir="o", mode="sft", eos=Marker(id=5))
    with pytest.raises(ValueError, match="does not use eos"):
        BuildConfig(input_paths=["a"], output_dir="o", mode="sft", eos=Marker(text="<x>"))
    with pytest.raises(ValueError, match="does not use turn_start"):
        BuildConfig(input_paths=["a"], output_dir="o", mode="pt", turn_start=Marker(id=5))
    with pytest.raises(ValueError, match="does not use turn_end"):
        BuildConfig(input_paths=["a"], output_dir="o", mode="pt", turn_end=Marker(id=5))
    with pytest.raises(ValueError):
        BuildConfig(input_paths=["a"], output_dir="o", mode="bogus")
    with pytest.raises(ValueError):
        BuildConfig(input_paths=["a"], output_dir="o", mode="pt", compression=None)
    # 合法组合
    BuildConfig(input_paths=["a"], output_dir="o", mode="pt", eos=Marker(id=5))
    BuildConfig(input_paths=["a"], output_dir="o", mode="sft", turn_start=Marker("<|im_start|>"))
    # 新旧写法混用必须报错（不静默取舍）
    with pytest.raises(TypeError, match="signatures mixed"):
        BuildConfig.from_kwargs({"input_paths": ["a"], "output_dir": "o", "mode": "pt",
                                 "eos": Marker(id=1), "eos_id": 2})
    with pytest.raises(TypeError, match="unknown arguments"):
        BuildConfig.from_kwargs({"input_paths": ["a"], "output_dir": "o", "bogus_kw": 1})


def test_a5_legacy_kwargs_path_warns_and_is_guarded(dsdirs, tmp_path):
    raw = tmp_path / "raw.jsonl"
    with open(raw, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"text": f"legacy doc {i} quick fox"}) + "\n")
    out = tmp_path / "legacy_pt"
    b = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
    # 旧签名（带弃用告警）仍可用
    with pytest.warns(DeprecationWarning, match="flat signature"):
        b.build_dataset([str(raw)], str(out), mode="pt", batch_size=BATCH_SIZE, eos_id=2)
    assert os.path.exists(str(out / "index.json"))
    # 旧签名在 SFT 下传 eos 同样响亮拒绝（连位置参数形式一起走 shim）
    with pytest.raises(ValueError, match="does not use eos"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b.build_dataset([str(raw)], str(tmp_path / "legacy_sft"), mode="sft",
                            eos_id=2, template_str=TEMPLATE)


# ============================================================
# S5 / #5：LoaderOptions 配置收敛
# ============================================================
def test_loader_options_shared_and_normalized(dsdirs):
    opts = LoaderOptions(batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                         world_size=1, rank=0, seed=7, shuffle="none")
    assert opts.shuffle is False and opts.partition == "round_robin"
    # 同一份参数集可复用于引擎层与 Tier-1
    l = BatchStream(dsdirs["pt"], options=opts)
    ds = StreamingDataset(dsdirs["pt"], options=opts)
    assert len(l) == len(ds)
    l.close(); ds.close()
    # options= 与扁平参数互斥（值偏离默认即报错）
    with pytest.raises(ValueError, match="both options= and flat arguments"):
        BatchStream(dsdirs["pt"], options=opts, batch_size=4)
    # 非法取值在构造期报错
    with pytest.raises(ValueError, match="shuffle"):
        LoaderOptions(shuffle="buffer")
    with pytest.raises(ValueError, match="partition"):
        LoaderOptions(partition="bogus")
    with pytest.raises(ValueError, match="must not be negative"):
        LoaderOptions(batch_size=-1)
    # 0 与 None 同义（取默认 / manifest 值），不得报错
    assert LoaderOptions(predownload=0, batch_size=0, max_seq_len=0) is not None


# ============================================================
# S5 / #8：开关归一化统一
# ============================================================
def test_normalize_choice_unified():
    assert normalize_shuffle("Permutation") is True
    assert normalize_shuffle(False) is False
    with pytest.raises(ValueError, match="buffer"):
        normalize_shuffle("buffer")
    assert normalize_partition("ROUND_ROBIN") == "round_robin"
    with pytest.raises(ValueError, match="partition"):
        normalize_partition("bogus")
    assert normalize_mode("SFT") == "sft"
    with pytest.raises(ValueError, match="compression=None"):
        normalize_compression(None)
    with pytest.raises(ValueError, match="compression"):
        normalize_compression("gzip")


# ============================================================
# A5 / #9：类型守卫接线
# ============================================================
def test_type_guards_wired(dsdirs):
    b_pt = next(iter(_stream(dsdirs["pt"])))
    b_sft = next(iter(_stream(dsdirs["sft"])))
    assert is_pt_batch(b_pt) and not is_sft_batch(b_pt)
    assert is_sft_batch(b_sft) and not is_pt_batch(b_sft)
    assert not is_pt_batch({"format": "unknown"})
