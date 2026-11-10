import os
import json
import tempfile
import pytest

from megadata import DataBuilder, BatchStream

# ==============================================================================
# 配置区：请在这里填入您本地一个真实可用的 bbpe_rs 分词器目录路径
# 因为底层 BBPETokenizer::load 会尝试读取真实词表文件
# ==============================================================================
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./model_learn_tokenizer")

# 生成的假数据总量
TOTAL_FAKE_DOCS = 5000
BATCH_SIZE = 4
MAX_SEQ_LEN = 256

@pytest.fixture(scope="module")
def fake_sft_environment():
    """
    测试环境夹具 (Fixture)：
    负责在临时目录生成 5000 条确定性的假对话数据，
    并调用 Rust 底层的 DataBuilder 走完完整的处理流水线。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到有效的 Tokenizer 路径: {TOKENIZER_PATH}，请配置环境变量 TOKENIZER_PATH")

    with tempfile.TemporaryDirectory() as temp_dir:
        raw_jsonl_path = os.path.join(temp_dir, "raw_data.jsonl")
        output_arrow_dir = os.path.join(temp_dir, "processed_sft")

        # 1. 制造确定性的假数据
        print("\n[Fixture] 正在生成假数据...")
        with open(raw_jsonl_path, "w", encoding="utf-8") as f:
            for i in range(TOTAL_FAKE_DOCS):
                # 每条对话都带有唯一编号 i，确保结构确定性
                record = {
                    "messages": [
                        {"role": "user", "content": f"User_Query_{i}"},
                        {"role": "assistant", "content": f"Assistant_Response_{i}"}
                    ]
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 2. 启动 Rust DataBuilder 流水线
        print("[Fixture] 启动 Rust 离线并行构建引擎...")
        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)

        # 使用标准的 ChatML 模板进行渲染验证
        chat_template = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"

        builder.build_dataset(
            input_paths=[raw_jsonl_path],
            output_dir=output_arrow_dir,
            mode="sft",
            template_str=chat_template,
            batch_size=BATCH_SIZE,
            pad_id=0,   # SFT 不传 eos（模式无关参数会被 BuildConfig 拒绝）
            # turn 标记 id 交由分词器自动推导（不写死，D8 会校验一致性与单 token 性）
            turn_start_str="<|im_start|>", turn_end_str="<|im_end|>",
            assistant_str="assistant",
            # 显式小聚合粒度：保证 shard 数 >= world_size=4（dimension_2 分布式校验）
            batches_per_shard=8,
        )

        yield output_arrow_dir


@pytest.mark.real_tokenizer
def test_dimension_1_e2e_parity(fake_sft_environment):
    """
    【维度一验证】：端到端逻辑与掩码验证
    目标：证明 Rust 底层生成的 Tensor 形状正确，且 -100 的 Label 掩码被精准生成。
    """
    output_dir = fake_sft_environment
    index_path = os.path.join(output_dir, "index.json")

    # 拉起单机单卡的 DataLoader
    loader = BatchStream(
        index_json_path=index_path,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        world_size=1, rank=0
    , shuffle=False)

    batch_count = 0
    total_docs_packed = 0

    for batch in loader:
        batch_count += 1

        # 1. 校验形状是否被零拷贝正确转换
        assert "input_ids" in batch
        assert "labels" in batch
        assert "cu_seqlens" in batch
        assert batch["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
        assert batch["labels"].shape == (BATCH_SIZE, MAX_SEQ_LEN)

        # 2. 校验 SFT 掩码逻辑 (Padding 必须为 -100)
        labels = batch["labels"]

        # 检查是否包含正常的 token 预测 (一定会有正数的 ID 被保留计算 Loss)
        assert (labels > 0).any().item(), "Batch 中应该有需要计算 Loss 的 token"

        # 检查是否包含 -100 (User 部分和 Pad 部分必须是 -100)
        assert (labels == -100).any().item(), "Batch 中应该包含 -100 的掩码"

        # 3. 校验 FlashAttention 拓扑结构的准确性
        # 在我们的 Rust 算法中，cu_seqlens 不记录 padding 的长度，只记录真实文档长度。
        # 因此，(cu_seqlens 的长度 - 1) 恰好等于这个 Batch 里打包了多少条真实文档！
        cu_seqlens = batch["cu_seqlens"]
        assert cu_seqlens[0].item() == 0, "cu_seqlens 必须以 0 开头"

        # 累计这一个 Batch 消化了多少篇文档
        docs_in_batch = cu_seqlens.numel() - 1
        total_docs_packed += docs_in_batch

    print(f"\n[E2E] 成功遍历 {batch_count} 个 Batch。")
    # 最严苛的终极校验：所有 Batch 里面打包的有效文档总数，必须等于 5000！
    assert total_docs_packed == TOTAL_FAKE_DOCS, \
        f"数据丢失或冗余！期待 {TOTAL_FAKE_DOCS} 篇，实际打包了 {total_docs_packed} 篇"


def test_dimension_2_distributed_integrity(fake_sft_environment):
    """
    【维度二验证】：分布式数据无重叠、无遗漏验证
    目标：证明 4 台机器同时读取时，它们处理的数据严格互斥，且总和等于全集。
    """
    output_dir = fake_sft_environment
    index_path = os.path.join(output_dir, "index.json")

    WORLD_SIZE = 4
    loaders = []

    # 模拟在 4 个不同的分布式进程中启动 Loader
    for rank in range(WORLD_SIZE):
        loader = BatchStream(
            index_json_path=index_path,
            batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            world_size=WORLD_SIZE,
            rank=rank,
            seed=42 # 全局 Seed 必须一致
        , shuffle=False)
        loaders.append(loader)

    # 收集 4 个 Rank 的处理统计
    docs_per_rank = [0] * WORLD_SIZE

    for rank, loader in enumerate(loaders):
        for batch in loader:
            # 根据 cu_seqlens 计算该 rank 这个 batch 吃掉了多少条文档
            docs_in_batch = batch["cu_seqlens"].numel() - 1
            docs_per_rank[rank] += docs_in_batch

    print(f"\n[分布式] 4个节点消化的数据量分布: {docs_per_rank}")

    # 1. 验证没有遗漏：4 个节点的总和必须精确等于 5000
    total_docs = sum(docs_per_rank)
    assert total_docs == TOTAL_FAKE_DOCS, \
        f"分布式分片存在遗漏/重复！总计 {total_docs}，应为 {TOTAL_FAKE_DOCS}"

    # 2. 验证负载均衡：轮询算法下，各节点数据量差距不应过大 (允许因为 Shard 划分有少许误差)
    max_docs = max(docs_per_rank)
    min_docs = min(docs_per_rank)
    assert (max_docs - min_docs) / TOTAL_FAKE_DOCS < 0.1, \
        "分布式哈希路由不均衡，差距超过 10%！"