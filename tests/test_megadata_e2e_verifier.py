import os
import json
import random
import hashlib
import shutil
import pytest
import torch

from megadata import DataBuilder, BatchStream, MixBatchStream

# 可选测试依赖（20261031 评审工程项）：未构建兄弟 bbpe_tokenizer 时整文件 skip，
# 而不是让全量收集中断。
pytest.importorskip("bbpe_tokenizer", reason="needs the sibling bbpe_tokenizer package")
from bbpe_tokenizer import (
    BBTokenizer,
    TrainerConfig,
    JsonlFileReader,
    BBPETrainer,
    TokenizerGlobalConfig
)

# ==========================================
# 全局配置参数
# ==========================================
PRETRAIN_JSONL = "data/pretrain_t2t_mini.jsonl"
SFT_JSONL = "data/sft_t2t_mini.jsonl"

VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36
BATCH_SIZE = 16
MAX_SEQ_LEN = 4096

DATASET_DIR_ENV = "MEGADATA_DATASET_DIR"

META_FILE_NAME = ".megadata_build_meta.json"


# ==========================================
# 指纹缓存工具
# ==========================================
def _hash_file(path: str) -> str:
    """SHA-256 of file contents (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _engine_version() -> str:
    """Best-effort identifier for the Rust engine version."""
    try:
        import megadata._rust as _rust
        for attr in ("__version__", "VERSION", "version"):
            v = getattr(_rust, attr, None)
            if v:
                return str(v)
        mod_file = getattr(_rust, "__file__", "") or ""
        if mod_file and os.path.isfile(mod_file):
            return f"mtime:{os.path.getmtime(mod_file)}"
    except Exception:
        pass
    return "unknown"


def _dataset_fingerprint(parts: dict) -> str:
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _meta_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, META_FILE_NAME)


def _index_exists(dataset_dir: str) -> bool:
    return os.path.isfile(os.path.join(dataset_dir, "index.json"))


def _try_reuse_dataset(
    persistent_root: str,
    sub_name: str,
    fp: str,
) -> str | None:
    """If persistent_root/sub_name exists and fingerprints match, return its path."""
    if not persistent_root:
        return None
    target = os.path.join(persistent_root, sub_name)
    meta_p = _meta_path(target)
    if not (os.path.isdir(target) and os.path.isfile(meta_p) and _index_exists(target)):
        return None
    try:
        with open(meta_p, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("fp") == fp:
            return target
    except Exception:
        return None
    return None


def _store_meta(dataset_dir: str, fp: str) -> None:
    with open(_meta_path(dataset_dir), "w", encoding="utf-8") as f:
        json.dump({"fp": fp}, f, ensure_ascii=False)


def _move_to_persistent(src_dir: str, persistent_root: str, sub_name: str) -> str:
    """Move a freshly-built dataset_dir into persistent_root/sub_name."""
    if not persistent_root:
        return src_dir
    target = os.path.join(persistent_root, sub_name)
    if os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(persistent_root, exist_ok=True)
    shutil.move(src_dir, target)
    return target

# ==========================================
# 夹具 1：基准分词器训练 (继承自原版脚本)
# ==========================================
@pytest.fixture(scope="session")
def trained_tokenizer(tmp_path_factory):
    override_dir = os.environ.get("MEGADATA_TOKENIZER_DIR")
    if override_dir:
        os.makedirs(override_dir, exist_ok=True)
        tokenizer_dir = override_dir
    else:
        tokenizer_dir = str(tmp_path_factory.mktemp("tokenizer"))

    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    additional_tokens_list = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>"]
    num_buffer = SPECIAL_TOKENS_NUM - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens

    model_path = os.path.join(tokenizer_dir, "tokenizer.json")
    config_path = os.path.join(tokenizer_dir, "tokenizer_config.json")

    # 如果指定了目录、且已有训练好的分词器，且未要求强制重训，则直接复用
    override_dir = os.environ.get("MEGADATA_TOKENIZER_DIR")
    force_retrain = os.environ.get("MEGADATA_RETRAIN_TOKENIZER", "0") not in ("0", "", "false", "False")
    reuse_existing = (
        override_dir is not None
        and not force_retrain
        and os.path.isfile(model_path)
        and os.path.isfile(config_path)
    )

    base_vocab_size = VOCAB_SIZE - len(all_special_tokens)

    if not reuse_existing:
        # 1. 训练
        t_config = TrainerConfig(vocab_size=VOCAB_SIZE).with_min_frequency(5).with_special_tokens(all_special_tokens)
        trainer = BBPETrainer(trainer_config=t_config)
        reader = JsonlFileReader.static_text(PRETRAIN_JSONL)
        trainer.train(reader)

        trainer.save(model_path)

        # 2. 注入 Special Tokens (精确计算基准词表末尾偏移)
        with open(model_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        added_tokens = {token: base_vocab_size + i for i, token in enumerate(all_special_tokens)}
        tokenizer_data["added_tokens"] = added_tokens

        with open(model_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    # 3. 注入包含工具调用的长篇 Chat Template（每次都覆盖 config，确保模板最新）
    config = TokenizerGlobalConfig()
    # (此处省略超长 template 字符串以保持代码整洁，实际运行中请使用你脚本中的 MASSIVE_CHAT_TEMPLATE)
    config.chat_template = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"

    config_dict = config.to_dict()
    config_dict["added_tokens_decoder"] = {
        str(base_vocab_size + i): {"content": t, "special": True}
        for i, t in enumerate(all_special_tokens)
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)

    return tokenizer_dir

# ==========================================
# 夹具 2 & 3：使用新分词器构建测试数据集（带指纹缓存）
# ==========================================
@pytest.fixture(scope="session")
def pt_dataset_env(trained_tokenizer, tmp_path_factory):
    builder = DataBuilder(tokenizer_path=trained_tokenizer, max_seq_len=MAX_SEQ_LEN)

    # 动态获取 eos_id
    tok = BBTokenizer.from_pretrained(trained_tokenizer)
    eos_id = tok.encode("<|endoftext|>", allowed_special=["all"]).ids[0]

    fp = _dataset_fingerprint({
        "kind": "pt",
        "inputs": [_hash_file(PRETRAIN_JSONL)],
        "tokenizer_dir": trained_tokenizer,
        "mode": "pt",
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "eos_id": eos_id,
        "engine": _engine_version(),
    })

    persistent_root = os.environ.get(DATASET_DIR_ENV)
    reused = _try_reuse_dataset(persistent_root, "pt", fp)
    if reused is not None:
        return reused, eos_id

    output_dir = str(tmp_path_factory.mktemp("pt_data"))
    builder.build_dataset([PRETRAIN_JSONL], output_dir, mode="pt", batch_size=BATCH_SIZE, eos_id=eos_id,
                          batches_per_shard=1)  # 保持 1:1 粒度：数据集仅数个 batch，须 shard 数 >= world_size=4

    _store_meta(output_dir, fp)
    final_dir = _move_to_persistent(output_dir, persistent_root, "pt")
    return final_dir, eos_id


@pytest.fixture(scope="session")
def sft_dataset_env(trained_tokenizer, tmp_path_factory):
    builder = DataBuilder(tokenizer_path=trained_tokenizer, max_seq_len=MAX_SEQ_LEN)

    tok = BBTokenizer.from_pretrained(trained_tokenizer)
    turn_start_id = tok.encode("<|im_start|>", allowed_special=["all"]).ids[0]
    turn_end_id = tok.encode("<|im_end|>", allowed_special=["all"]).ids[0]

    # chat_template 来自持久化的 tokenizer_config.json（TokenizerGlobalConfig），
    # 但如果它是示例占位，则使用一个能处理 `conversations` schema 的 ChatML 模板。
    with open(os.path.join(trained_tokenizer, "tokenizer_config.json"), "r", encoding="utf-8") as _cf:
        template_str = json.load(_cf).get("chat_template", "")

    placeholder = "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"
    if not template_str or template_str.strip() == placeholder.strip():
        template_str = (
            "{% for m in conversations %}"
            "<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )

    fp = _dataset_fingerprint({
        "kind": "sft",
        "inputs": [_hash_file(SFT_JSONL)],
        "tokenizer_dir": trained_tokenizer,
        "mode": "sft",
        "max_seq_len": MAX_SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "template_str": template_str,
        "turn_start_id": turn_start_id,
        "turn_end_id": turn_end_id,
        "turn_start_str": "<|im_start|>",
        "turn_end_str": "<|im_end|>",
        "assistant_str": "assistant",
        "engine": _engine_version(),
    })

    persistent_root = os.environ.get(DATASET_DIR_ENV)
    reused = _try_reuse_dataset(persistent_root, "sft", fp)
    if reused is not None:
        return reused

    output_dir = str(tmp_path_factory.mktemp("sft_data"))
    builder.build_dataset(
        [SFT_JSONL], output_dir, mode="sft", template_str=template_str,
        batch_size=BATCH_SIZE, turn_start_id=turn_start_id, turn_end_id=turn_end_id,
        turn_start_str="<|im_start|>", turn_end_str="<|im_end|>", assistant_str="assistant"
    )

    _store_meta(output_dir, fp)
    final_dir = _move_to_persistent(output_dir, persistent_root, "sft")
    return final_dir


# ==========================================
# 阶段 0：绝对数据一致性与守恒定律校验
# ==========================================
def test_phase_0_pt_conservation_and_random_spot_check(pt_dataset_env, trained_tokenizer):
    output_dir, eos_id = pt_dataset_env
    tok = BBTokenizer.from_pretrained(trained_tokenizer)

    # 1. 扫描原始数据：统计总行数，并进行蓄水池随机抽样 50 条
    raw_doc_count = 0
    random_samples = []

    with open(PRETRAIN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            raw_doc_count += 1
            if len(random_samples) < 50:
                random_samples.append(json.loads(line)["text"])
            else:
                if random.randint(0, i) < 50:
                    random_samples[random.randint(0, 49)] = json.loads(line)["text"]

    target_sample_lengths = set([len(tok.encode(text).ids) for text in random_samples])

    # 2. 遍历底层的 Native DataLoader
    # 注：默认（无限流式）；用 Python 端的 _consumed 计数做护栏，
    # 保证扫完一个 epoch（=index.json 中的 total_batches）就退出。
    loader = BatchStream(os.path.join(output_dir, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)

    with open(os.path.join(output_dir, "index.json"), "r") as _f:
        _total_batches = json.load(_f)["global_stats"]["total_batches"]

    packed_eos_count = 0
    extracted_doc_lengths = set()
    _consumed = 0

    for batch in loader:
        _consumed += 1
        input_ids = batch["input_ids"].flatten().tolist()

        # 将被包装成流的 1D 张量，按 eos_id 切割还原回原始文档长度
        current_len = 0
        for token_id in input_ids:
            if token_id == eos_id:
                packed_eos_count += 1
                extracted_doc_lengths.add(current_len)
                current_len = 0
            else:
                current_len += 1

        if _consumed >= _total_batches:
            break

    # 去除末尾用来填充空隙的零碎 eos_id 带来的长度 0 序列
    extracted_doc_lengths.discard(0)

    # 【断言 1.1：守恒定律】
    # 解析出的文档数 (按 EOF 分割) 绝不小于原始文档数。
    # (之所以不严格等于，是因为原始文档超过 max_seq_len 会被引擎截断并强制追加一个额外的 eos_id)
    assert packed_eos_count >= raw_doc_count, "致命错误：引擎遗漏了原始数据！"

    # 【断言 1.2：抽样无损校验】
    # 随机抽取的 50 条文档的 Token 长度指纹，必须在底层数据流中 100% 被找到。
    assert target_sample_lengths.issubset(extracted_doc_lengths), "致命错误：随机抽样的数据内容被损坏或未正确打包！"


def test_phase_0_sft_conservation(sft_dataset_env):
    output_dir = sft_dataset_env

    # 1. 扫描原始 SFT 数据
    raw_valid_docs = 0
    with open(SFT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): raw_valid_docs += 1

    # 2. 遍历 SFT DataLoader
    # 注：默认（无限流式）；用 Python 端的 _consumed 计数做护栏。
    loader = BatchStream(os.path.join(output_dir, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    packed_docs = 0
    _consumed = 0
    with open(os.path.join(output_dir, "index.json"), "r") as _f:
        _total_batches = json.load(_f)["global_stats"]["total_batches"]

    for batch in loader:
        # 在 Rust SFT 逻辑中，cu_seqlens 的长度减 1 就是这个 Batch 中有效文档的数量
        packed_docs += (batch["cu_seqlens"].numel() - 1)
        _consumed += 1
        if _consumed >= _total_batches:
            break

    # 【断言 2：SFT 数据绝对守恒】
    # 注意：cu_seqlens 记录的是 SFT 中被切段的对话片段数（multi-turn 对话会被切多段），
    # 因此 packed_docs 应该 ≥ raw_valid_docs（≥ 而非严格 ==）。
    assert packed_docs >= raw_valid_docs, f"SFT 遗漏！期望至少 {raw_valid_docs} 段，实际仅 {packed_docs} 段"


# ==========================================
# 阶段 2：SFT 张量物理正确性与掩码校验
# ==========================================
def test_phase_2_sft_tensors_and_masks(sft_dataset_env):
    output_dir = sft_dataset_env
    loader = BatchStream(os.path.join(output_dir, "index.json"), batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN)
    # 触发一次预取，确保 loader 真正可用
    _ = next(iter(loader))

    batch = next(iter(loader))

    assert "input_ids" in batch and "labels" in batch and "cu_seqlens" in batch
    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
    assert batch["labels"].shape == (BATCH_SIZE, MAX_SEQ_LEN)

    labels = batch["labels"]
    # 【核心校验】: Prompt 掩码必须包含 -100
    assert (labels == -100).any().item(), "Batch 未发现 -100，Loss Mask 逻辑失效！"
    # 【核心校验】: 包含有效的预测 Label
    assert (labels > 0).any().item(), "Batch 无有效监督目标，模型将无法学习！"
    # 【核心校验】: cu_seqlens 必须单调递增，且以 0 始
    cu = batch["cu_seqlens"].tolist()
    assert cu[0] == 0
    assert all(x <= y for x, y in zip(cu, cu[1:])), "FlashAttention 拓扑不合法！"


# ==========================================
# 阶段 3：分布式哈希路由绝对无重叠校验
# ==========================================
def test_phase_3_distributed_routing(pt_dataset_env):
    output_dir, _ = pt_dataset_env
    index_path = os.path.join(output_dir, "index.json")

    with open(index_path, "r") as f:
        total_expected_batches = json.load(f)["global_stats"]["total_batches"]

    WORLD_SIZE = 4
    total_consumed_by_cluster = 0

    for rank in range(WORLD_SIZE):
        loader = BatchStream(index_path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, world_size=WORLD_SIZE, rank=rank)
        total_consumed_by_cluster += len(loader)  # MapDataset 暴露的 __len__

    # 【核心校验】分布式分片总量 必须 精确等于 清单记录量
    assert total_consumed_by_cluster == total_expected_batches, \
        f"分布式路由算错！集群累计提取 {total_consumed_by_cluster}，期望 {total_expected_batches}"


# ==========================================
# 阶段 4：断点续训时空穿越极限校验
# ==========================================
def test_phase_4_time_travel_resumption(sft_dataset_env):
    output_dir = sft_dataset_env
    index_path = os.path.join(output_dir, "index.json")

    # 模拟中断前的训练会话
    # 确保 next() 的输出严格由 _consumed 决定，
    # 否则默认 buffer=1000 会在 main thread 看不到的时候让 background prefetch 到远端，
    # 导致 state_dict 后无法 deterministic 还原到下一个 batch。
    loader_run1 = BatchStream(index_path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=888)
    iter_run1 = iter(loader_run1)

    for _ in range(5): next(iter_run1) # 跑 5 步

    # 截取系统状态字典并保存目标基准
    state_snapshot = loader_run1.state_dict()
    target_tensor = next(iter_run1)["input_ids"].clone()

    # 模拟宕机后拉起全新训练会话
    loader_resume = BatchStream(index_path, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN, seed=888)
    loader_resume.load_state_dict(state_snapshot)

    # 提取恢复后的第一步
    resume_tensor = next(iter(loader_resume))["input_ids"]

    # 【核心校验】时空穿越后，字节必须完全对齐
    assert torch.equal(target_tensor, resume_tensor), "致命错误：断点续训恢复张量不一致！"


# ==========================================
# 阶段 5：DynamicMixer 权重课程学习测试
# ==========================================
def test_phase_5_dynamic_mixer(pt_dataset_env, sft_dataset_env):
    pt_dir, _ = pt_dataset_env
    sft_dir = sft_dataset_env

    # 混合 PT 与 SFT（使用 Arrow IPC 与 Memmap 的混合）
    loader = MixBatchStream(
        configs=[
            {"path": os.path.join(pt_dir, "index.json"), "weight": 0.8},
            {"path": os.path.join(sft_dir, "index.json"), "weight": 0.2}
        ],
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN
    )

    # 验证底层引擎能成功无缝吐出数据（如果引擎内部格式兼容性崩溃，此处会抛错）
    batch = next(iter(loader))
    assert "input_ids" in batch

    # 测试运行时更新权重接口
    loader.update_weights([0.0, 1.0])
    batch = next(iter(loader))
    assert "input_ids" in batch # 成功切换采样流且无异常