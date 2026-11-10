"""
MixBatchStream 单元测试。

覆盖 test_resumption.py 未涉及的 API 契约:
  - update_weights / update_temperature 的权重更新
  - __len__ 在/1 下的契约（不再 raise TypeError）
  - 权重采样的区分度（[1,0] 与 [0,1] 产出的 batch 内容必须不同）
  - state_dict / load_state_dict 往返
"""
import os
import json
import tempfile
import pytest
import torch

from megadata import DataBuilder, MixBatchStream

TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "./model_learn_tokenizer")
TOTAL_FAKE_DOCS = 2000
BATCH_SIZE = 4
MAX_SEQ_LEN = 128


@pytest.fixture(scope="module")
def two_pt_datasets():
    """构建两个内容可区分的 PT 数据集（code / math），供混采测试使用。"""
    if not os.path.exists(TOKENIZER_PATH):
        pytest.skip(f"未找到 Tokenizer: {TOKENIZER_PATH}")

    with tempfile.TemporaryDirectory() as temp_dir:
        code_dir = os.path.join(temp_dir, "code_pt")
        math_dir = os.path.join(temp_dir, "math_pt")
        raw_code = os.path.join(temp_dir, "raw_code.jsonl")
        raw_math = os.path.join(temp_dir, "raw_math.jsonl")

        # code 数据集每行 "def code_function_{i}(): pass"
        # math 数据集每行 "The math answer for {i} is {i*2}."
        # 两者 tokenize 后 token 分布明显不同，可用于区分来源。
        with open(raw_code, "w", encoding="utf-8") as fc, \
             open(raw_math, "w", encoding="utf-8") as fm:
            for i in range(TOTAL_FAKE_DOCS):
                fc.write(json.dumps({"text": f"def code_function_{i}(): pass"}, ensure_ascii=False) + "\n")
                fm.write(json.dumps({"text": f"The math answer for {i} is {i*2}."}, ensure_ascii=False) + "\n")

        builder = DataBuilder(tokenizer_path=TOKENIZER_PATH, max_seq_len=MAX_SEQ_LEN)
        builder.build_dataset(input_paths=[raw_code], output_dir=code_dir,
                              mode="pt", batch_size=BATCH_SIZE, eos_id=2)
        builder.build_dataset(input_paths=[raw_math], output_dir=math_dir,
                              mode="pt", batch_size=BATCH_SIZE, eos_id=2)

        yield {
            "code": os.path.join(code_dir, "index.json"),
            "math": os.path.join(math_dir, "index.json"),
        }


def _build_loader(datasets, weights=None):
    """构造一个两源混采 loader。weights=[w_code, w_math]。

    v3：没有 `num_epochs`（轮数归外层），温度平滑走 `update_temperature()`。
    """
    key_to_idx = {"code": 0, "math": 1}
    configs = []
    for key in ("code", "math"):
        cfg = {"path": datasets[key]}
        if weights is not None:
            cfg["weight"] = weights[key_to_idx[key]]
        configs.append(cfg)
    return MixBatchStream(
        configs=configs,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )


def test_len_contract_finite(two_pt_datasets):
    """ 时 __len__ 返回有限正整数。"""
    loader = _build_loader(two_pt_datasets)
    n = len(loader)
    assert isinstance(n, int)
    assert n > 0


def test_infinite_stream_is_while_true_over_for(two_pt_datasets):
    """v3：无限流 = `while True:` 包住 for（无哨兵、无 sys.maxsize）。

    每个 for 恰好一轮；外层 while 负责跨轮。本例用步数上限退出，验证
    "一轮读完 → StopIteration → 下一轮继续"确实成立（B2 回归：读满一轮必须停）。
    """
    loader = _build_loader(two_pt_datasets)
    assert isinstance(len(loader), int) and len(loader) > 0

    rounds = 0
    steps = 0
    for _round in range(2):
        for _batch in loader:      # 每个 for = 一轮
            steps += 1
        rounds += 1
        if steps > 400:            # 护栏：自然模式步数远小于此
            break
    assert rounds >= 1 and steps > 0, "至少应完成一轮"


def test_basic_iteration(two_pt_datasets):
    """混采能正常产出 batch，且 input_ids 形状与 dtype 正确。"""
    loader = _build_loader(two_pt_datasets)
    it = iter(loader)
    batch = next(it)
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)
    assert batch["input_ids"].dtype == torch.int32


def test_update_weights_distinguishes_streams(two_pt_datasets):
    """权重 [1,0] 与 [0,1] 产出的 batch 内容必须不同，证明 update_weights 生效。

    分别用只开 code 流、只开 math 流的权重各跑 5 步，比较两组 tensor。
    若完全相同，说明权重未真正影响采样分布。
    """
    # 1) 只开 code 流
    loader_code = _build_loader(two_pt_datasets, weights=[1.0, 0.0])
    it_code = iter(loader_code)
    code_tensors = []
    for _ in range(5):
        batch = next(it_code)
        code_tensors.append(batch["input_ids"].clone())

    # 2) 只开 math 流
    loader_math = _build_loader(two_pt_datasets, weights=[0.0, 1.0])
    it_math = iter(loader_math)
    math_tensors = []
    for _ in range(5):
        batch = next(it_math)
        math_tensors.append(batch["input_ids"].clone())

    # 3) 两组内容必须不同
    differ = any(not torch.equal(ct, mt) for ct, mt in zip(code_tensors, math_tensors))
    assert differ, "权重 [1,0] 与 [0,1] 产出的 batch 完全相同，update_weights 未生效"


def test_update_weights_runtime(two_pt_datasets):
    """运行期 update_weights 能即时切换权重，且 current_weights 被更新。"""
    loader = _build_loader(two_pt_datasets)
    # 初始权重各 0.5（由温度 1.0 平滑得到）
    assert abs(sum(loader.current_weights) - 1.0) < 1e-6

    loader.update_weights([0.0, 1.0])
    assert loader.current_weights == [0.0, 1.0]

    # 切换后仍能正常迭代
    it = iter(loader)
    batch = next(it)
    assert batch["input_ids"].shape == (BATCH_SIZE, MAX_SEQ_LEN)


def test_update_temperature(two_pt_datasets):
    """update_temperature 基于温度平滑重算权重，且 current_weights 被刷新。"""
    loader = _build_loader(two_pt_datasets)

    loader.update_temperature(3.0)
    new_weights = list(loader.current_weights)

    # 温度变化后权重和仍为 1
    assert abs(sum(new_weights) - 1.0) < 1e-6
    assert len(new_weights) == 2


def test_state_dict_roundtrip(two_pt_datasets):
    """state_dict / load_state_dict 能往返保存与恢复。

    v3 口径：`steps_yielded`（成功产出步数）+ 逐子流 `sub_stream_cursors`
    （`(epoch, in_epoch)`，"下一个待产出位置"）。
    """
    loader1 = _build_loader(two_pt_datasets)
    it1 = iter(loader1)
    for _ in range(5):
        next(it1)
    state = loader1.state_dict()
    assert state["schema"] == 3
    assert state["steps_yielded"] == 5
    assert len(state["sub_stream_cursors"]) == 2

    loader2 = _build_loader(two_pt_datasets)
    loader2.load_state_dict(state)
    restored = loader2.state_dict()
    assert restored["steps_yielded"] == 5
    assert [list(c) for c in restored["sub_stream_cursors"]] == \
        [list(c) for c in state["sub_stream_cursors"]], "逐子流游标必须原样恢复"


def test_pt_cu_seqlens_contract(two_pt_datasets):
    """PT 模式产出的 batch 应包含 cu_seqlens（文档边界），格式满足 FlashAttention varlen 契约。

    契约：
    - cu_seqlens[0] == 0
    - cu_seqlens 单调递增
    - cu_seqlens[-1] <= batch_size * max_seq_len（有效 token 数，不含 EOS padding）
    """
    loader = _build_loader(two_pt_datasets)
    it = iter(loader)
    batch = next(it)

    assert "cu_seqlens" in batch, "PT batch 应包含 cu_seqlens（文档边界）"
    cu = batch["cu_seqlens"]
    # v3（P1）契约：cu_seqlens 是 int32 **tensor**（不再是 Python list）
    assert isinstance(cu, torch.Tensor), "v3 契约：cu_seqlens 必须是 tensor"
    assert cu.dtype == torch.int32, f"v3 契约：cu_seqlens dtype 应为 int32，实际 {cu.dtype}"
    cu_list = [int(x) for x in cu.tolist()]
    assert len(cu_list) >= 2, "cu_seqlens 至少应有 [0, valid_len] 两个元素"
    assert cu_list[0] == 0, f"cu_seqlens 应以 0 开头，实际 {cu_list[0]}"
    # C21/G2：满批 == B×S；末尾残余批 ≤ B×S（不能对任意 batch 无条件断言 ==）
    assert 0 < cu_list[-1] <= BATCH_SIZE * MAX_SEQ_LEN, \
        f"cu_seqlens 末值应为有效 token 数 (0, {BATCH_SIZE * MAX_SEQ_LEN}]，实际 {cu_list[-1]}"
    for i in range(1, len(cu_list)):
        assert cu_list[i] >= cu_list[i - 1], \
            f"cu_seqlens 应单调递增，{cu_list[i]} < {cu_list[i-1]} at i={i}"
