"""**全项目**端到端用户旅程测试（不是只看单个模块）。

每条用例都是一次真实的"用户做事流程"：构建 → 训练读 → 断点续训 → 混采 →
评估/容量规划，并断言每一步的**用户可见契约**（数据不重不漏、形状/类型、
轮次语义、异常语义）。
"""

from __future__ import annotations


import pytest
import torch

from _scenario_kit import (
    BATCH_SIZE,
    MAX_SEQ_LEN,
    batch_key,
    build_dataset,
    pt_records,
    read_batches,
    sft_records,
    tokenizer_path,
    total_batches,
)


# ==============================================================================
# 旅程 1：PT 数据集 —— 构建 → 多轮训练 → 断点续训
# ==============================================================================
@pytest.fixture(scope="module")
def pt_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("journey_pt")
    return build_dataset(str(d / "pt"), pt_records(400), "pt", batches_per_shard=4,
                         compression="zstd")


def test_journey_1_pt_three_epochs_are_disjoint_and_covering(pt_index):
    """3 轮训练：每轮 batch 数 == len()，每轮内部无重复，每轮并集都是全集。"""
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                          seed=42, shuffle=True)
    expected_total = total_batches(pt_index)
    epoch_keys = []
    for _ in range(3):
        keys = [batch_key(b) for b in loader]
        assert len(keys) == expected_total, f"len() 与实际产出不一致：{len(keys)} vs {expected_total}"
        assert len(set(keys)) == len(keys), "同一轮内出现重复 batch（置换表非双射）"
        epoch_keys.append(set(keys))

    assert len(epoch_keys[0]) == expected_total, "一轮没有覆盖全部 batch"
    assert epoch_keys[0] == epoch_keys[1] == epoch_keys[2], "每轮应覆盖同一集合（顺序不同）"
    assert loader.epoch == 3, "自然读完应推进轮号"


def test_journey_1b_pt_resume_is_exact(pt_index):
    """断点续训：保存 → 新 loader 恢复 → 剩余批次与原会话完全一致。"""
    from megadata import BatchStream

    def make():
        return BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                            max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                            seed=42, shuffle=True)

    # 参照：完整跑一轮
    ref = [batch_key(b) for b in make()]

    # 真实：跑 3 步 → 存 → 新 loader 恢复 → 跑完
    loader = make()
    it = iter(loader)
    head = [batch_key(next(it)) for _ in range(3)]
    state = loader.state_dict()
    del it, loader

    restored = make()
    restored.load_state_dict(state)
    tail = [batch_key(b) for b in restored]
    assert head + tail == ref, "续训后序列与原始序列不一致（重读/跳读/错位）"


def test_journey_1c_pt_checkpoint_schema_contract(pt_index):
    """state_dict 的对外契约：schema=3 + 指纹 + 拓扑。"""
    from megadata import BatchStream

    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    state = loader.state_dict()
    assert state["schema"] == 3
    assert state["manifest_fingerprint"]["batch_size"] == BATCH_SIZE
    assert state["manifest_fingerprint"]["max_seq_len"] == MAX_SEQ_LEN
    assert state["manifest_fingerprint"]["total_batches"] == total_batches(pt_index)
    assert state["topology"]["world_size"] == 1 and state["topology"]["rank"] == 0


# ==============================================================================
# 旅程 2：SFT 数据集 —— 构建 → 标签掩码 → 注意力后端联动
# ==============================================================================
@pytest.fixture(scope="module")
def sft_index(tmp_path_factory):
    d = tmp_path_factory.mktemp("journey_sft")
    return build_dataset(str(d / "sft"), sft_records(200), "sft", batches_per_shard=4)


@pytest.mark.real_tokenizer
def test_journey_2_sft_batch_contract(sft_index):
    """SFT batch 的用户可见契约：形状 / dtype / 掩码 / cu_seqlens 单调。"""
    batches = read_batches(sft_index)
    assert batches, "SFT 数据集应有 batch"
    for b in batches:
        assert set(b) >= {"input_ids", "labels", "position_ids", "cu_seqlens"}
        assert b["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert b["labels"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert b["position_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert b["input_ids"].dtype == torch.int32
        # P1：cu_seqlens 是 int32 tensor（不再是 Python list）
        assert isinstance(b["cu_seqlens"], torch.Tensor)
        assert b["cu_seqlens"].dtype == torch.int32
        cu = b["cu_seqlens"].tolist()
        assert cu[0] == 0 and all(cu[i] <= cu[i + 1] for i in range(len(cu) - 1))
        valid = cu[-1]
        assert all(lab == -100 for lab in b["labels"][0][valid:].tolist()), "padding 必须 -100"
        assert (b["labels"] != -100).any(), "至少要有被监督的 token"


def test_journey_2b_sft_to_flex_attention(sft_index):
    """SFT 走 FlexAttention：真实 batch 的 cu_seqlens 能构造 BlockMask 并前向。"""
    from megadata import to_flex_block_mask

    b = read_batches(sft_index)[0]
    cu = b["cu_seqlens"]
    valid = max(cu.tolist())
    mask = to_flex_block_mask(cu, causal=True, block_size=16, compile_mask=False)
    # BlockMask.shape == (B, H, Q_LEN, KV_LEN)；Q/KV 等长且 pad 到 block_size 倍数
    assert mask.shape[2] == mask.shape[3]
    assert mask.shape[2] % 16 == 0
    assert mask.shape[2] >= valid, "pad 后的长度不能小于有效长度"


# ==============================================================================
# 旅程 3：混采异构数据集（PT + SFT）
# ==============================================================================
@pytest.fixture(scope="module")
def mixed_pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("journey_mix")
    pt = build_dataset(str(d / "pt"), pt_records(200, tag="mixpt"), "pt", batches_per_shard=4)
    sft = build_dataset(str(d / "sft"), sft_records(200, tag="mixsft"), "sft",
                        batches_per_shard=4)
    return pt, sft


def test_journey_3_mixed_pt_sft(mixed_pair):
    """异构混采：batch 带 format 键、恰好 N 步、两种格式都出现。"""
    from megadata import MixBatchStream

    pt, sft = mixed_pair
    steps = 40
    loader = MixBatchStream(
        configs=[{"path": pt, "weight": 0.5}, {"path": sft, "weight": 0.5}],
        batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, steps_per_epoch=steps,
    )
    assert len(loader) == steps, "确定性模式 len() 应等于 steps_per_epoch"
    formats = []
    for b in loader:
        assert "format" in b, "异构混采必须携带 format 键"
        formats.append(b["format"])
        assert b["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
    assert len(formats) == steps
    assert set(formats) == {"pt", "sft"}, f"两种格式都应被采样到：{set(formats)}"


def test_journey_3b_mixed_deterministic_resume(mixed_pair):
    """混采的 O(1) 断点续训：恢复后剩余步序与原会话一致。"""
    from megadata import MixBatchStream

    pt, sft = mixed_pair
    configs = [{"path": pt, "weight": 0.7}, {"path": sft, "weight": 0.3}]

    def make():
        return MixBatchStream(configs=configs, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
                         steps_per_epoch=30, seed=7)

    ref = [batch_key(b) for b in make()]
    loader = make()
    it = iter(loader)
    head = [batch_key(next(it)) for _ in range(5)]
    state = loader.state_dict()
    del it, loader

    restored = make()
    restored.load_state_dict(state)
    assert len(restored) == 25, "恢复后 len() 应是本轮剩余步数（N - k）"
    tail = [batch_key(b) for b in restored]
    assert head + tail == ref, "混采续训后步序偏移（RNG / 游标未精确恢复）"
    assert len(tail) == 25, "恢复后应只跑完本轮剩余步数，而不是又跑满 N 步"


# ==============================================================================
# 旅程 4：HF Trainer / Lightning 风格循环（Tier 1）
# ==============================================================================
@pytest.mark.parametrize("num_workers", [0, 2])
def test_journey_4_tier1_trainer_loop(pt_index, num_workers):
    """Tier 1：`for batch in dl` 两轮，每轮都推进轮号且数据不同。"""
    from megadata import MegadataDataLoader, StreamingDataset

    ds = StreamingDataset(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0, seed=1)
    dl = MegadataDataLoader(ds, num_workers=num_workers)
    rounds = []
    for _ in range(2):
        keys = [batch_key(b) for b in dl]
        assert keys, "每轮都应有数据"
        rounds.append(set(keys))
    assert rounds[0] == rounds[1], "每轮应覆盖同一批数据"
    assert ds.epoch == 2, f"轮号应由 dataset 计数（实际 {ds.epoch}）"
    st = dl.state_dict()
    assert st["num_batches_yielded"] > 0
    assert st["tier1"]["epoch"] == ds.epoch


# ==============================================================================
# 旅程 5：评估/抽查（随机访问）与流式读取一致
# ==============================================================================
def test_journey_5_random_access_covers_same_batches(pt_index):
    """同一 seed/shuffle=False 下，随机访问全量遍历 == 流式一轮（同集合）。"""
    from megadata import RandomAccessBatchReader

    reader = RandomAccessBatchReader(index_json_path=pt_index, batch_size=BATCH_SIZE,
                                     max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0)
    n = len(reader)
    assert n == total_batches(pt_index)
    seq_keys = [batch_key(reader.get_batch(i)) for i in range(n)]
    assert len(set(seq_keys)) == n, "随机访问出现重复 batch"
    stream_keys = set(batch_key(b) for b in read_batches(pt_index))
    assert set(seq_keys) == stream_keys, "随机访问与流式读取应覆盖同一批 batch"

    # 反序访问（触发 seek）结果一致
    reverse = [batch_key(reader.get_batch(i)) for i in reversed(range(n))]
    assert reverse == list(reversed(seq_keys)), "seek 后结果与顺序访问不一致"

    with pytest.raises(IndexError):
        reader.get_batch(n)
    assert batch_key(reader.get_batch(-1)) == seq_keys[-1], "负索引应为倒数"


# ==============================================================================
# 旅程 6：容量规划（仿真器）用真实 manifest
# ==============================================================================
def test_journey_6_capacity_simulation(pt_index):
    """仿真器对真实数据集给出可用的容量规划结论。"""
    from megadata.simulation import (
        SimulationConfig,
        format_report_markdown,
        load_shard_sizes,
        simulate,
        sweep_cache,
    )

    sizes = load_shard_sizes(pt_index)
    assert sizes and all(s > 0 for s in sizes)

    cfg = SimulationConfig(
        index_path=pt_index, world_size=2, num_nodes=1, num_workers=0,
        seed=42, epochs=1, shuffle="permutation", partition="round_robin",
        num_canonical_nodes=None, bandwidth_mbps=1000.0, cache_limit_bytes=0,
        compute_ms_per_batch=250.0, predownload=8, max_steps=0,
    )
    report = simulate(cfg)
    assert report.steps > 0
    assert report.total_download_bytes > 0
    assert 0.0 <= report.hit_rate <= 1.0

    sweep = sweep_cache(cfg, ks=[0, 1, None])
    assert len(sweep) == 3
    assert isinstance(format_report_markdown(report), str)


# ==============================================================================
# 旅程 7：完整"训练 → 中断 → 恢复 → 继续训练"
# ==============================================================================
def test_journey_7_interrupt_and_resume(pt_index, tmp_path):
    """模拟真实训练循环：跑 4 步 → ctrl-c 式中断 → 从 checkpoint 继续到跑完一轮。"""
    from megadata import BatchStream

    ckpt = tmp_path / "ckpt.pt"
    loader = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                          seed=123, shuffle=True)
    ref = [batch_key(b) for b in BatchStream(
        index_json_path=pt_index, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
        world_size=1, rank=0, seed=123, shuffle=True)]

    seen = []
    for i, batch in enumerate(loader):
        seen.append(batch_key(batch))
        if i + 1 == 4:
            torch.save({"dataloader": loader.state_dict(), "step": i + 1}, ckpt)
            break                       # 真实场景：max_steps / 提前结束
    del loader

    state = torch.load(ckpt, weights_only=False)
    resumed = BatchStream(index_json_path=pt_index, batch_size=BATCH_SIZE,
                           max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                           seed=123, shuffle=True)
    resumed.load_state_dict(state["dataloader"])
    seen += [batch_key(b) for b in resumed]
    assert seen == ref, "中断恢复后整轮序列应与从未中断时一致"
    assert state["step"] == 4


# ==============================================================================
# 旅程 8：协议面符号（对外 API 面）
# ==============================================================================
def test_journey_8_public_api_surface():
    """`megadata` 的协议面符号必须齐备且可用（文档承诺的对外契约）。"""
    import megadata

    expected = {
        "DataBuilder", "BatchStream", "MixBatchStream", "RandomAccessBatchReader",
        "StreamingDataset", "MegadataDataLoader", "DatasetConfig",
        "MegadataInterrupted", "install_signal_handlers", "uninstall_signal_handlers",
        "read_manifest", "PTBatch", "SFTBatch",
        "BatchTokenizer", "HfTokenizerAdapter",
    }
    missing = expected - set(dir(megadata))
    assert not missing, f"协议面缺少符号：{missing}"
    assert set(megadata.__all__) >= expected, "符号未在 __all__ 中导出"
    assert issubclass(megadata.MegadataInterrupted, KeyboardInterrupt)
    assert tokenizer_path()  # 环境自检（缺失会 skip）
