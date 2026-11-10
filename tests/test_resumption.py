import os
import json
import tempfile
import pytest
import torch

from megadata import DataBuilder, MixBatchStream

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./model_learn_tokenizer")
TOTAL_FAKE_DOCS = 5000
BATCH_SIZE = 4
MAX_SEQ_LEN = 128

@pytest.fixture(scope="module")
def fake_pt_mixer_environment():
    """
    环境夹具：生成两个预训练 (PT) 数据集（模拟 Code 和 Math），
    并使用 Memmap 格式落盘，供 Mixer 混合。
    """
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到有效的 Tokenizer 路径: {TOKENIZER_PATH}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # 准备两个数据集的目录
        code_dir = os.path.join(temp_dir, "code_pt")
        math_dir = os.path.join(temp_dir, "math_pt")

        raw_code_path = os.path.join(temp_dir, "raw_code.jsonl")
        raw_math_path = os.path.join(temp_dir, "raw_math.jsonl")

        # 1. 制造确定性的 PT 假数据
        with open(raw_code_path, "w", encoding="utf-8") as fc, \
             open(raw_math_path, "w", encoding="utf-8") as fm:
            for i in range(TOTAL_FAKE_DOCS):
                # PT 模式直接抽取 "text" 字段
                fc.write(json.dumps({"text": f"def code_function_{i}(): pass"}, ensure_ascii=False) + "\n")
                fm.write(json.dumps({"text": f"The math answer for {i} is {i*2}."}, ensure_ascii=False) + "\n")

        # 2. 启动 Rust DataBuilder，使用 mode="pt" (底层会生成 .bin 和 .idx)
        print("\n[Fixture] 正在构建 Code 预训练数据集 (Memmap格式)...")
        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)

        builder.build_dataset(
            input_paths=[raw_code_path],
            output_dir=code_dir,
            mode="pt", # 激活 Memmap + GreedyPacker
            batch_size=BATCH_SIZE,
            eos_id=2
        )

        print("[Fixture] 正在构建 Math 预训练数据集 (Memmap格式)...")
        builder.build_dataset(
            input_paths=[raw_math_path],
            output_dir=math_dir,
            mode="pt",
            batch_size=BATCH_SIZE,
            eos_id=2
        )

        yield {"code": os.path.join(code_dir, "index.json"), "math": os.path.join(math_dir, "index.json")}


def test_time_travel_resumption(fake_pt_mixer_environment):
    """
    【神级校验】：时间旅行与断点续训验证
    目标：证明 Mixer 引擎不仅能保存字节偏移量，还能精准恢复随机数抽样状态！
    """
    index_code = fake_pt_mixer_environment["code"]
    index_math = fake_pt_mixer_environment["math"]

    datasets_config = [
        {"path": index_code, "weight": 0.7},
        {"path": index_math, "weight": 0.3}
    ]

    # ==========================================
    # 阶段 1：基准时间线 (运行 20 步)
    # ==========================================
    loader_baseline = MixBatchStream(
        configs=datasets_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        seed=1024 # 固定随机种子
    )

    iter_baseline = iter(loader_baseline)
    baseline_tensors = []

    print("\n[Time-Travel] 运行基准时间线...")
    for _ in range(20):
        batch = next(iter_baseline)
        # clone() 很重要，防止底层内存复用被覆盖
        baseline_tensors.append(batch["input_ids"].clone())

    # ==========================================
    # 阶段 2：中断时间线 (运行 10 步并截取快照)
    # ==========================================
    loader_interrupt = MixBatchStream(
        configs=datasets_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        seed=1024 # 必须与基准 Seed 一致
    )

    iter_interrupt = iter(loader_interrupt)

    print("[Time-Travel] 运行中断时间线...")
    for _ in range(10):
        batch = next(iter_interrupt)

    # 💣 此时机器突然宕机，我们抢救出了 state_dict
    checkpoint_state = loader_interrupt.state_dict()
    print("[Time-Travel] 机器宕机，已截取第 10 步快照！")

    # ==========================================
    # 阶段 3：恢复时间线 (注入快照并运行 10 步)
    # ==========================================
    loader_resume = MixBatchStream(
        configs=datasets_config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        seed=1024
    )

    # 🌟 魔法发生：瞬间恢复游标与随机数！
    loader_resume.load_state_dict(checkpoint_state)
    iter_resume = iter(loader_resume)

    resume_tensors = []
    print("[Time-Travel] 重启加载快照，运行恢复时间线...")
    for _ in range(10):
        batch = next(iter_resume)
        resume_tensors.append(batch["input_ids"].clone())

    # ==========================================
    # 阶段 4：时空对齐断言
    # ==========================================
    print("[Time-Travel] 开始时空对齐校验...")
    for i in range(10):
        tensor_baseline = baseline_tensors[10 + i]
        tensor_resume = resume_tensors[i]

        # 绝对相等的断言！如果这里过了，证明我们的 DLPack 和 断点续训天下无敌！
        assert torch.equal(tensor_baseline, tensor_resume), \
            f"❌ 断点续训失败！第 {10 + i} 步的 Tensor 与基准时间线不一致！"

    print("✅ 时间旅行校验完美通过！")