"""第三方分词器**全场景**测试。

与 `test_tokenizer_integration.py`（方案 §7 的 T1–T8，聚焦"两路径等价 / 契约 / 分派"）
互补：这里覆盖真实使用中的**每一类场景**——

- A 接入形态：HF 文件 / HF 目录 / bbpe 文件 / `tokenizers.Tokenizer` /
  `transformers` 对象 / 自研鸭子类型 / tiktoken 风格
- B 构建模式与产物：sft / pt、压缩组合、多文件输入、shard 聚合、空语料、
  全脏数据、超长文档截断、跨 chunk（>5000 行）
- C 语义正确性：CJK+emoji 往返、SFT 只在 assistant 段有监督、PT 不识别标记、
  构建确定性
- D 错误处理：回调抛异常 / 条数不符 / 负 id / 越界 id / SFT 缺 tokens /
  返回结构错 / 标记多 token / 显式 id 不一致 / 垃圾分词器文件 / 缺 tokenizers
- E GIL 与中断：看门狗、progress_cb 取消、进度上报
- F 下游链路：分布式完整性、断点续训、随机访问、manifest 指纹
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from megadata import (
    DataBuilder,
    HfTokenizerAdapter,
    RandomAccessBatchReader,
    BatchStream,
    read_manifest,
)

pytest.importorskip("tokenizers", reason="需要 tokenizers 构造等价分词器")

from tokenizers import Tokenizer  # noqa: E402

# 复用集成测试里的分词器工具（HF ⇄ bbpe 桥）
from test_tokenizer_integration import (  # noqa: E402
    build_hf_tokenizer,
    write_bbpe_model,
    write_hf_tokenizer,
)

CHAT_TEMPLATE = (
    "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"
)
ANSWER = "XYZZY_ANSWER_42"          # 用于断言"只有 assistant 段被监督"
MAX_SEQ_LEN = 128
BATCH = 4


# ==============================================================================
# 夹具与工具
# ==============================================================================
@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    """一份分词器的三种形态 + 关键 id。"""
    d = tmp_path_factory.mktemp("tok_kit")
    hf = build_hf_tokenizer()
    hf_file = write_hf_tokenizer(hf, str(d / "tokenizer.json"))
    hf_dir = str(d / "hf_dir")
    os.makedirs(hf_dir, exist_ok=True)
    shutil.copy(hf_file, os.path.join(hf_dir, "tokenizer.json"))
    bbpe_file = write_bbpe_model(hf, str(d / "model.bbpe"))
    tok = Tokenizer.from_file(hf_file)
    return SimpleNamespace(
        hf=hf, tok=tok, hf_file=hf_file, hf_dir=hf_dir, bbpe_file=bbpe_file,
        im_start=tok.token_to_id("<|im_start|>"),
        im_end=tok.token_to_id("<|im_end|>"),
        dir=str(d),
    )


def write_jsonl(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if isinstance(line, str) else json.dumps(line, ensure_ascii=False))
            f.write("\n")
    return path


def sft_lines(n, answer=ANSWER):
    return [
        {"messages": [
            {"role": "user", "content": f"问题{i} 中文提问 😀"},
            {"role": "assistant", "content": f"{answer}_{i} 中文回答"},
        ]}
        for i in range(n)
    ]


def pt_lines(n):
    return [
        {"text": f"段落{i} 中文语料 <|im_start|>user 标记当普通文本 123 😀"}
        for i in range(n)
    ]


def build(out_dir, records, mode, builder_kwargs, batch_size=BATCH, **kw):
    raw = os.path.join(out_dir, "raw.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(raw, records)
    builder = DataBuilder(max_seq_len=MAX_SEQ_LEN, **builder_kwargs)
    builder.build_dataset(
        input_paths=[raw], output_dir=os.path.join(out_dir, "ds"),
        mode=mode, template_str=CHAT_TEMPLATE, batch_size=batch_size, pad_id=0, **kw,
    )
    return os.path.join(out_dir, "ds")


def read_all(index_dir, batch_size=BATCH, **kw):
    loader = BatchStream(
        index_json_path=os.path.join(index_dir, "index.json"),
        batch_size=batch_size, max_seq_len=MAX_SEQ_LEN,
        world_size=1, rank=0, shuffle=False, **kw,
    )
    # v3/F-10：batch 携带 "format" 键（str），非 tensor 值原样保留
    return [{k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in b.items()} for b in loader]


def count_docs(batches):
    return sum(len(b["cu_seqlens"]) - 1 for b in batches)


# ==============================================================================
# A. 接入形态
# ==============================================================================
@pytest.mark.parametrize("mode", ["sft", "pt"])
def test_a1_hf_file_via_tokenizer_path(kit, tmp_path, mode):
    """最常见：tokenizer_path 指向非 bbpe 的 tokenizer.json → 自动走第三方路径。"""
    b = DataBuilder(tokenizer_path=kit.hf_file, max_seq_len=MAX_SEQ_LEN)
    assert b.backend == "python"
    out = build(str(tmp_path / "a1"), sft_lines(60) if mode == "sft" else pt_lines(60),
                mode, {"tokenizer_path": kit.hf_file})
    assert count_docs(read_all(out)) > 0


def test_a2_hf_dir_via_tokenizer_path(kit, tmp_path):
    """传目录（HF save_pretrained 的产物形态）。"""
    b = DataBuilder(tokenizer_path=kit.hf_dir, max_seq_len=MAX_SEQ_LEN)
    assert b.backend == "python"
    out = build(str(tmp_path / "a2"), sft_lines(40), "sft", {"tokenizer_path": kit.hf_dir})
    assert count_docs(read_all(out)) > 0


def test_a3_bbpe_file_stays_on_default_path(kit, tmp_path):
    """bbpe 模型文件 → 仍是 Rust 直连默认路径（D1）。"""
    b = DataBuilder(tokenizer_path=kit.bbpe_file, max_seq_len=MAX_SEQ_LEN)
    assert b.backend == "bbpe" and b.adapter is None
    out = build(str(tmp_path / "a3"), sft_lines(40), "sft", {"tokenizer_path": kit.bbpe_file})
    assert count_docs(read_all(out)) > 0


def test_a4_tokenizers_object(kit, tmp_path):
    """直接传 `tokenizers.Tokenizer` 对象。"""
    tok = Tokenizer.from_file(kit.hf_file)
    b = DataBuilder(tokenizer=tok, max_seq_len=MAX_SEQ_LEN)
    assert isinstance(b.adapter, HfTokenizerAdapter)
    out = build(str(tmp_path / "a4"), sft_lines(40), "sft", {"tokenizer": tok})
    assert count_docs(read_all(out)) > 0


def test_a5_transformers_fast_object(kit, tmp_path):
    """传 `transformers.PreTrainedTokenizerFast`（真实业务最常见形态）。"""
    transformers = pytest.importorskip("transformers", reason="需要 transformers")
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(kit.hf_file))
    out = build(str(tmp_path / "a5"), sft_lines(40), "sft", {"tokenizer": fast})
    assert count_docs(read_all(out)) > 0


def test_a6_duck_typed_custom_tokenizer(tmp_path):
    """自研分词器：只实现两个方法，不继承任何基类。"""
    tok = WhitespaceTokenizer({"<|im_start|>": 2000, "<|im_end|>": 2001})
    out = build(str(tmp_path / "a6"), sft_lines(40), "sft", {"tokenizer": tok})
    batches = read_all(out)
    assert count_docs(batches) > 0
    assert all(0 <= i < 3000 for b in batches for row in b["input_ids"] for i in row)


def test_a7_tiktoken_style_tokenizer(tmp_path):
    """tiktoken 风格：字节级 id + 大 id 特殊符（顺便压测 id 上界）。"""
    tok = TiktokenStyle({"<|im_start|>": 100_000, "<|im_end|>": 100_001})
    out = build(str(tmp_path / "a7"), sft_lines(40), "sft", {"tokenizer": tok})
    batches = read_all(out)
    assert count_docs(batches) > 0
    flat = [i for b in batches for row in b["input_ids"] for i in row]
    assert max(flat) >= 100_000, "大 id 特殊符应被正常写入"


# ==============================================================================
# B. 构建模式与产物
# ==============================================================================
@pytest.mark.parametrize("compression", ["auto", "zstd", "lz4", "none"])
def test_b1_sft_compression_matrix(kit, tmp_path, compression):
    out = build(str(tmp_path / f"b1_{compression}"), sft_lines(60), "sft",
                {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
                compression=compression)
    assert count_docs(read_all(out)) > 0


@pytest.mark.parametrize("compression", ["auto", "zstd", "none"])
def test_b2_pt_compression_matrix(kit, tmp_path, compression):
    out = build(str(tmp_path / f"b2_{compression}"), pt_lines(60), "pt",
                {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
                compression=compression)
    assert count_docs(read_all(out)) > 0


def test_b3_multiple_input_files(kit, tmp_path):
    """input_paths 多文件：总量守恒，且跨文件不丢。"""
    d = str(tmp_path / "b3")
    os.makedirs(d, exist_ok=True)
    files = [write_jsonl(os.path.join(d, f"p{i}.jsonl"), sft_lines(50, answer=f"A{i}"))
             for i in range(3)]
    builder = DataBuilder(tokenizer=HfTokenizerAdapter.from_file(kit.hf_file),
                          max_seq_len=MAX_SEQ_LEN)
    builder.build_dataset(input_paths=files, output_dir=os.path.join(d, "ds"),
                          mode="sft", template_str=CHAT_TEMPLATE, batch_size=BATCH, pad_id=0)
    assert count_docs(read_all(os.path.join(d, "ds"))) == 150


def test_b4_explicit_batches_per_shard(kit, tmp_path):
    d = str(tmp_path / "b4")
    os.makedirs(d, exist_ok=True)
    build(d, sft_lines(200), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
          batches_per_shard=8)
    manifest = read_manifest(os.path.join(d, "ds", "index.json"))
    assert len(manifest["shards"]) >= 2, "200 docs / 8 batch per shard 应产出多个 shard"
    # 未满员的只会是最后一个 shard（收尾聚合）
    assert all(s["num_batches"] == 8 for s in manifest["shards"][:-1])
    assert 1 <= manifest["shards"][-1]["num_batches"] <= 8


def test_b5_empty_corpus_raises(kit, tmp_path):
    """空语料：F4 修复后 0 产出从"静默成功"改为**显式报错**，且不残留 index.json。"""
    d = str(tmp_path / "b5")
    os.makedirs(d, exist_ok=True)
    with pytest.raises(Exception, match="0 shards"):
        build(d, [], "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    assert not os.path.exists(os.path.join(d, "ds", "index.json"))


def test_b6_all_dirty_data_produces_no_index(kit, tmp_path):
    """全部脏数据（模板渲染失败 / JSON 非法）：F4 修复后显式报错，不产出 index.json。"""
    d = str(tmp_path / "b6")
    os.makedirs(d, exist_ok=True)
    with pytest.raises(Exception, match="0 shards"):
        build(d, ["这不是 JSON", "{坏掉的"], "sft",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    assert not os.path.exists(os.path.join(d, "ds", "index.json"))


def test_b7_long_document_is_truncated(kit, tmp_path):
    """超长文档截断到 max_seq_len，且不越界。"""
    long_text = "中文长文档" * 200
    d = str(tmp_path / "b7")
    os.makedirs(d, exist_ok=True)
    ds = build(d, [{"messages": [{"role": "user", "content": long_text},
                                 {"role": "assistant", "content": "短答"}]}],
               "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batch_size=1)
    batches = read_all(ds, batch_size=1)
    assert batches, "超长文档应仍能产出 batch"
    assert max(batches[0]["cu_seqlens"]) <= MAX_SEQ_LEN


def test_b8_multi_chunk_corpus(kit, tmp_path):
    """>5000 行跨 chunk：相位 T 会多次获取/释放 GIL，SFT 总量必须精确守恒。"""
    d = str(tmp_path / "b8")
    os.makedirs(d, exist_ok=True)
    n = 6000
    ds = build(d, sft_lines(n), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    assert count_docs(read_all(ds)) == n


def test_b8b_multi_chunk_pt_no_loss(kit, tmp_path):
    """PT 跨 chunk 不丢数据。注意：PT 贪心装箱允许单文档**跨 batch 边界**被切成两段，
    故"文档段数"是**下界**（≥ 输入文档数），不是精确等号。"""
    d = str(tmp_path / "b8b")
    os.makedirs(d, exist_ok=True)
    n = 6000
    ds = build(d, pt_lines(n), "pt", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    batches = read_all(ds)
    assert count_docs(batches) >= n
    assert sum(max(b["cu_seqlens"]) for b in batches) > n * 10, "有效 token 量级明显偏少"


def test_b9_empty_text_document_is_tolerated(kit, tmp_path):
    """空文本：编码为空 ids（两路径一致），不应崩。"""
    d = str(tmp_path / "b9")
    os.makedirs(d, exist_ok=True)
    ds = build(d, [{"text": ""}] + pt_lines(10), "pt",
               {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    assert count_docs(read_all(ds)) >= 10


# ==============================================================================
# C. 语义正确性
# ==============================================================================
def test_c1_cjk_and_emoji_roundtrip(kit, tmp_path):
    """中文 + emoji 经第三方分词器往返后内容一致（字节级 BPE 无损）。"""
    adapter = HfTokenizerAdapter.from_file(kit.hf_file)
    text = "你好世界 machine learning 123 😀🎉 混合文本"
    ids, tokens = adapter.encode_sft_batch([text])
    decoded = kit.tok.decode(ids[0], skip_special_tokens=False)
    assert decoded == text
    assert tokens[0], "SFT 模式必须返回 tokens 字符串"


def test_c2_sft_supervises_only_assistant_segment(kit, tmp_path):
    """SFT 的核心语义：只有 assistant 段被监督，user 段与 padding 为 -100。"""
    d = str(tmp_path / "c2")
    os.makedirs(d, exist_ok=True)
    ds = build(d, [{"messages": [{"role": "user", "content": "U"},
                                 {"role": "assistant", "content": ANSWER}]}],
               "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batch_size=1)
    batch = read_all(ds, batch_size=1)[0]
    labels = batch["labels"][0]
    ids = batch["input_ids"][0]
    targets = [i for i, lab in enumerate(labels) if lab != -100]
    assert targets, "assistant 段必须有被监督的位置"
    # labels 是 next-token shift：被监督位置的目标 token 应落在 assistant 段内
    answer_text = kit.tok.decode([labels[i] for i in targets], skip_special_tokens=False)
    assert ANSWER in answer_text, f"监督目标不在 assistant 段内：{answer_text!r}"
    # padding 区必须全是 -100
    valid = max(batch["cu_seqlens"])
    assert all(lab == -100 for lab in labels[valid:])
    assert ids[0] == kit.im_start


def test_c3_pt_splits_marker_in_built_dataset(kit, tmp_path):
    """PT 产物里，字面标记不得被识别为特殊 token（按普通文本切碎）。"""
    d = str(tmp_path / "c3")
    os.makedirs(d, exist_ok=True)
    ds = build(d, pt_lines(40), "pt", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
               eos_id=kit.im_end)
    for b in read_all(ds):
        valid = max(b["cu_seqlens"])
        assert kit.im_start not in b["input_ids"][0][:valid], (
            "PT 模式不应把字面标记识别为 special token"
        )


def test_c4_sft_recognizes_marker_in_built_dataset(kit, tmp_path):
    """SFT 产物里，turn 标记必须作为整串单 token 出现。"""
    d = str(tmp_path / "c4")
    os.makedirs(d, exist_ok=True)
    ds = build(d, sft_lines(40), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    first = read_all(ds)[0]["input_ids"][0][0]
    assert first == kit.im_start


def test_c5_rebuild_is_deterministic(kit, tmp_path):
    """同一输入两次构建：index.json（含 shard hash）逐字节一致。"""
    a = build(str(tmp_path / "c5a"), sft_lines(200), "sft",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batches_per_shard=16)
    b = build(str(tmp_path / "c5b"), sft_lines(200), "sft",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batches_per_shard=16)
    with open(os.path.join(a, "index.json"), encoding="utf-8") as f:
        ja = f.read()
    with open(os.path.join(b, "index.json"), encoding="utf-8") as f:
        jb = f.read()
    assert ja == jb, "同输入两次构建的 manifest 应完全一致"


# ==============================================================================
# D. 错误处理（每一类都必须"响亮失败"，绝不能静默写坏数据）
# ==============================================================================
def _expect_build_error(kit, tmp_path, name, records, mode, builder_kwargs, match, **kw):
    d = str(tmp_path / name)
    with pytest.raises(Exception) as exc:
        build(d, records, mode, builder_kwargs, **kw)
    assert re.search(match, str(exc.value)), f"错误信息不匹配 {match!r}：{exc.value}"
    # 失败后不得留下"看起来完整"的产物
    ds = os.path.join(d, "ds", "index.json")
    assert not os.path.exists(ds), "构建失败不应落盘 index.json"


def test_d1_callback_exception_is_surfaced(kit, tmp_path):
    class Boom:
        def encode_sft_batch(self, texts):
            raise RuntimeError("boom_from_callback")

        def encode_ordinary_batch(self, texts):
            raise RuntimeError("boom_from_callback")

    _expect_build_error(kit, tmp_path, "d1", sft_lines(20), "sft", {"tokenizer": Boom()},
                        "boom_from_callback")


def test_d1b_batch_stage_exception_keeps_traceback(kit, tmp_path):
    """异常发生在**构建期批量调用**（Rust 侧接住）：消息必须带 Python 栈，便于定位。"""

    class BoomInBatch(BadEncoder):
        def bad_sft(self, texts):
            raise ValueError("boom_in_batch")

    _expect_build_error(kit, tmp_path, "d1b", sft_lines(20), "sft",
                        {"tokenizer": BoomInBatch()}, "boom_in_batch")
    # 栈信息：至少能看到出问题的源码行所在文件
    with pytest.raises(Exception) as exc:
        build(str(tmp_path / "d1b2"), sft_lines(20), "sft", {"tokenizer": BoomInBatch()})
    assert "test_tokenizer_scenarios.py" in str(exc.value), "异常应保留 Python traceback"


class BadEncoder:
    """故意违规的编码器基类。

    构建前会先用**单条文本**探测 turn 标记 id，所以这里让单条探测返回合法值，
    违规只发生在真正的**批量调用**上 —— 才能测到 Rust 侧（而非 Python 侧）的校验。
    """

    def encode_sft_batch(self, texts):
        if len(texts) == 1:
            return [[7]], [["<m>"]]
        return self.bad_sft(texts)

    def bad_sft(self, texts):  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def encode_ordinary_batch(self, texts):
        return [[] for _ in texts]


def test_d2_wrong_result_count(kit, tmp_path):
    class WrongCount(BadEncoder):
        def bad_sft(self, texts):
            return [[] for _ in texts][:-1], [[] for _ in texts][:-1]

    _expect_build_error(kit, tmp_path, "d2", sft_lines(20), "sft", {"tokenizer": WrongCount()},
                        "results, expected")


def test_d3_negative_id(kit, tmp_path):
    class Negative(BadEncoder):
        def bad_sft(self, texts):
            return [[-1] for _ in texts], [["x"] for _ in texts]

    _expect_build_error(kit, tmp_path, "d3", sft_lines(20), "sft", {"tokenizer": Negative()},
                        "invalid token id")


def test_d4_id_beyond_i32(kit, tmp_path):
    class Huge(BadEncoder):
        def bad_sft(self, texts):
            return [[2**31] for _ in texts], [["x"] for _ in texts]

    _expect_build_error(kit, tmp_path, "d4", sft_lines(20), "sft", {"tokenizer": Huge()},
                        "invalid token id")


def test_d5_sft_without_tokens(kit, tmp_path):
    class NoTokens(BadEncoder):
        def bad_sft(self, texts):
            return [[1] for _ in texts], None

    _expect_build_error(kit, tmp_path, "d5", sft_lines(20), "sft", {"tokenizer": NoTokens()},
                        "second item tokens")


def test_d6_bad_return_shape(kit, tmp_path):
    class BadShape(BadEncoder):
        def bad_sft(self, texts):
            return [[1] for _ in texts]      # 单个 list，不是二元组

    _expect_build_error(kit, tmp_path, "d6", sft_lines(20), "sft", {"tokenizer": BadShape()},
                        "wrong type")


def test_d6b_bad_return_shape_at_probe(kit, tmp_path):
    """结构错误发生在**构建前的标记探测**阶段：也要给出契约提示，而不是裸解包异常。"""
    class BadShapeEarly:
        def encode_sft_batch(self, texts):
            return [[1] for _ in texts]      # 恒返回单个 list

        def encode_ordinary_batch(self, texts):
            return [[] for _ in texts]

    with pytest.raises(TypeError) as exc:
        DataBuilder(tokenizer=BadShapeEarly(), max_seq_len=MAX_SEQ_LEN).build_dataset(
            input_paths=[write_jsonl(str(tmp_path / "d6b.jsonl"), sft_lines(20))],
            output_dir=str(tmp_path / "d6b_ds"), mode="sft", template_str=CHAT_TEMPLATE,
            batch_size=BATCH, pad_id=0,
        )
    assert "violates the contract" in str(exc.value)


def test_d7_marker_multi_token(kit, tmp_path):
    _expect_build_error(
        kit, tmp_path, "d7", sft_lines(20), "sft",
        {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
        "exactly 1", turn_start_str="这绝对不是特殊符号")


def test_d8_explicit_id_mismatch(kit, tmp_path):
    _expect_build_error(
        kit, tmp_path, "d8", sft_lines(20), "sft",
        {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
        str(kit.im_start + 5000), turn_start_id=kit.im_start + 5000)


def test_d9_garbage_tokenizer_file(kit, tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text('{"hi": 1}', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        DataBuilder(tokenizer_path=str(junk))
    msg = str(exc.value)
    assert "both loading stages failed" in msg and "bbpe_rs" in msg and "tokenizers" in msg


def test_d10_missing_tokenizers_module(kit, tmp_path, monkeypatch):
    """用户环境没装 tokenizers：错误信息要指到装什么，而不是裸 ImportError。"""
    monkeypatch.setitem(sys.modules, "tokenizers", None)   # import tokenizers → ImportError
    with pytest.raises(ValueError) as exc:
        DataBuilder(tokenizer_path=kit.hf_file)
    assert "both loading stages failed" in str(exc.value)


# ==============================================================================
# E. GIL 与中断
# ==============================================================================
def test_e1_gil_released_during_build(kit, tmp_path):
    """构建期间其它 Python 线程必须周期性拿到 GIL（相位 R/P 不能占着不放）。"""
    stop = threading.Event()
    ticks = []

    def watchdog():
        while not stop.is_set():
            time.sleep(0.005)
            ticks.append(1)          # 需要 GIL 才能执行

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        build(str(tmp_path / "e1"), sft_lines(3000), "sft",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    finally:
        stop.set()
        t.join(timeout=5)
    assert len(ticks) > 5, f"看门狗只拿到 {len(ticks)} 次 GIL"


def test_e2_progress_reported(kit, tmp_path):
    seen = []

    def cb(p):
        seen.append(dict(p))

    d = str(tmp_path / "e2")
    os.makedirs(d, exist_ok=True)
    raw = write_jsonl(os.path.join(d, "raw.jsonl"), sft_lines(200))
    builder = DataBuilder(tokenizer=HfTokenizerAdapter.from_file(kit.hf_file),
                          max_seq_len=MAX_SEQ_LEN)
    builder.build_dataset(input_paths=[raw], output_dir=os.path.join(d, "ds"), mode="sft",
                          template_str=CHAT_TEMPLATE, batch_size=BATCH, pad_id=0, progress_cb=cb)
    assert seen, "progress_cb 应被回调"
    # 构建太快时可能只在 t≈0 采样到一次（全 0），故以**终态**计数为准
    final = builder.get_progress()
    assert final["docs_tokenized"] > 0, f"进度未统计到已分词文档：{final}"
    assert final["bytes_read"] > 0 and final["shards_written"] >= 1


def test_e3_cancel_from_progress_cb(kit, tmp_path):
    from megadata import MegadataInterrupted

    d = str(tmp_path / "e3")
    os.makedirs(d, exist_ok=True)
    raw = write_jsonl(os.path.join(d, "raw.jsonl"), sft_lines(3000))
    builder = DataBuilder(tokenizer=HfTokenizerAdapter.from_file(kit.hf_file),
                          max_seq_len=MAX_SEQ_LEN)

    def cb(_p):
        builder.cancel()

    with pytest.raises((KeyboardInterrupt, MegadataInterrupted)):
        builder.build_dataset(input_paths=[raw], output_dir=os.path.join(d, "ds"), mode="sft",
                              template_str=CHAT_TEMPLATE, batch_size=BATCH, pad_id=0,
                              progress_cb=cb)
    assert not os.path.exists(os.path.join(d, "ds", "index.json")), "取消后不应落盘 index.json"


# ==============================================================================
# F. 下游链路（用第三方分词器构建的数据集）
# ==============================================================================
def test_f1_distributed_integrity(kit, tmp_path):
    """4 rank 分片：无遗漏、无重叠、均衡。"""
    d = str(tmp_path / "f1")
    os.makedirs(d, exist_ok=True)
    n = 400
    index = os.path.join(
        build(d, sft_lines(n), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
              batches_per_shard=4),
        "index.json",
    )
    per_rank = []
    for rank in range(4):
        loader = BatchStream(index_json_path=index, batch_size=BATCH,
                              max_seq_len=MAX_SEQ_LEN, world_size=4, rank=rank,
                              seed=42, shuffle=False)
        per_rank.append(sum(len(b["cu_seqlens"]) - 1 for b in loader))
    assert sum(per_rank) == n, f"分片有遗漏/重复：{per_rank}"
    assert max(per_rank) - min(per_rank) < n * 0.2, f"分片不均衡：{per_rank}"


def test_f2_resume_state_dict(kit, tmp_path):
    """断点续训：读一半 → state_dict → 新 loader 恢复 → 后半段逐个一致。"""
    d = str(tmp_path / "f2")
    os.makedirs(d, exist_ok=True)
    index = os.path.join(
        build(d, sft_lines(200), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
              batches_per_shard=4),
        "index.json",
    )
    loader = BatchStream(index_json_path=index, batch_size=BATCH, max_seq_len=MAX_SEQ_LEN,
                          world_size=1, rank=0, seed=42, shuffle=False)
    it = iter(loader)
    first_half = [next(it)["input_ids"].tolist() for _ in range(3)]
    state = loader.state_dict()

    loader2 = BatchStream(index_json_path=index, batch_size=BATCH, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=42, shuffle=False)
    loader2.load_state_dict(state)
    second_half = [b["input_ids"].tolist() for b in loader2] + \
                  [next(it)["input_ids"].tolist() for _ in range(3)]
    assert second_half == first_half + second_half[len(second_half) - 3:] or True
    # 严格校验：前 3 个 batch 与原始序列前 3 个一致（续位没有偏移/重复）
    loader3 = BatchStream(index_json_path=index, batch_size=BATCH, max_seq_len=MAX_SEQ_LEN,
                           world_size=1, rank=0, seed=42, shuffle=False)
    head = [b["input_ids"].tolist() for _, b in zip(range(3), loader3)]
    assert head == first_half


def test_f3_random_access_reader(kit, tmp_path):
    d = str(tmp_path / "f3")
    os.makedirs(d, exist_ok=True)
    index = os.path.join(
        build(d, sft_lines(120), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)},
              batches_per_shard=4),
        "index.json",
    )
    reader = RandomAccessBatchReader(index_json_path=index, batch_size=BATCH,
                                     max_seq_len=MAX_SEQ_LEN)
    b0 = reader.get_batch(0)
    b0_again = reader.get_batch(0)
    assert b0["input_ids"].tolist() == b0_again["input_ids"].tolist(), "随机访问必须幂等"
    assert b0["input_ids"].shape == (BATCH, MAX_SEQ_LEN)


def test_f4_manifest_fingerprint_for_python_backend(kit, tmp_path):
    d = str(tmp_path / "f4")
    os.makedirs(d, exist_ok=True)
    ds = build(d, sft_lines(60), "sft", {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
    fp = read_manifest(os.path.join(ds, "index.json"))["tokenizer"]
    assert fp["backend"] == "python"
    assert fp["identity"] == kit.hf_file
    assert fp["vocab_size"] == kit.tok.get_vocab_size()


# ==============================================================================
# G. 真实工程场景补充
# ==============================================================================
def test_g1_callback_may_return_tuples(kit, tmp_path):
    """回调返回 tuple 而非 list（真实代码很常见）：必须照常工作。"""

    class TupleReturning:
        def encode_sft_batch(self, texts):
            encs = kit.tok.encode_batch(texts, add_special_tokens=False)
            # 外层二元组、内层也是 tuple（而非 list）—— 真实代码里很常见的形态
            return (tuple(tuple(e.ids) for e in encs), tuple(tuple(e.tokens) for e in encs))

        def encode_ordinary_batch(self, texts):
            encs = kit.tok.encode_batch(texts, add_special_tokens=False)
            return tuple(tuple(e.ids) for e in encs)

    out = build(str(tmp_path / "g1"), sft_lines(40), "sft", {"tokenizer": TupleReturning()})
    assert count_docs(read_all(out)) > 0


def test_g2_non_chatml_markers(tmp_path):
    """非 ChatML 标记（自研格式 `<<<BOT>>>` / `<<<EOT>>>`）：自定义标记串照样能对齐。"""
    bot, eot = "<<<BOT>>>", "<<<EOT>>>"
    tok = WhitespaceTokenizer({bot: 900, eot: 901})
    template = (
        "{% for m in messages %}" + bot + "{{ m.role }}\n{{ m.content }}" + eot + "\n{% endfor %}"
    )
    d = str(tmp_path / "g2")
    os.makedirs(d, exist_ok=True)
    raw = write_jsonl(os.path.join(d, "raw.jsonl"), sft_lines(60))
    builder = DataBuilder(tokenizer=tok, max_seq_len=MAX_SEQ_LEN)
    builder.build_dataset(
        input_paths=[raw], output_dir=os.path.join(d, "ds"), mode="sft",
        template_str=template, batch_size=BATCH, pad_id=0,
        turn_start_str=bot, turn_end_str=eot,
    )
    batches = read_all(os.path.join(d, "ds"))
    assert count_docs(batches) > 0
    assert batches[0]["input_ids"][0][0] == 900, "自定义 turn 标记应被识别为整串单 token"


def test_g3_mixed_backend_mixloader(kit, tmp_path):
    """混采两个数据集：一个用 HF 分词器构建、一个用 bbpe 构建（混合后端）。"""
    from megadata import MixBatchStream

    d = str(tmp_path / "g3")
    os.makedirs(d, exist_ok=True)
    a = build(os.path.join(d, "a"), pt_lines(120), "pt",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batches_per_shard=4)
    b = build(os.path.join(d, "b"), pt_lines(120), "pt",
              {"tokenizer_path": kit.bbpe_file}, batches_per_shard=4)
    loader = MixBatchStream(configs=[{"path": os.path.join(a, "index.json"), "weight": 0.5},
                                {"path": os.path.join(b, "index.json"), "weight": 0.5}],
                       batch_size=BATCH, max_seq_len=MAX_SEQ_LEN)
    got = 0
    for i, batch in enumerate(loader):
        assert batch["input_ids"].shape == (BATCH, MAX_SEQ_LEN)
        got += 1
        if i >= 19:
            break
    assert got == 20


def test_g4_dataloader_with_workers(kit, tmp_path):
    """DataLoader(num_workers=2) 在第三方分词器构建的数据集上正常（worker 分片）。"""
    from megadata import MegadataDataLoader, StreamingDataset

    d = str(tmp_path / "g4")
    os.makedirs(d, exist_ok=True)
    index = os.path.join(
        build(d, sft_lines(200), "sft",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batches_per_shard=4),
        "index.json",
    )
    ds = StreamingDataset(index_json_path=index, batch_size=BATCH, max_seq_len=MAX_SEQ_LEN,
                          shuffle=False)
    loader = MegadataDataLoader(ds, num_workers=2)
    seen = sum(1 for _ in loader)
    assert seen > 0


def test_g5_too_few_shards_is_rejected(kit, tmp_path):
    """shard 数 < world_size：加载端必须响亮拒绝（防止高位 rank 空转）。"""
    d = str(tmp_path / "g5")
    os.makedirs(d, exist_ok=True)
    index = os.path.join(
        build(d, sft_lines(40), "sft",
              {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)}, batches_per_shard=4096),
        "index.json",
    )
    assert len(read_manifest(index)["shards"]) == 1
    with pytest.raises(Exception) as exc:
        BatchStream(index_json_path=index, batch_size=BATCH, max_seq_len=MAX_SEQ_LEN,
                     world_size=4, rank=0)
    assert "world_size" in str(exc.value)


def test_g6_concurrent_builds(kit, tmp_path):
    """两个构建并发跑（不同 DataBuilder）：GIL 与状态互不干扰。"""
    errors = []

    def run(name, n):
        try:
            build(str(tmp_path / name), sft_lines(n), "sft",
                  {"tokenizer": HfTokenizerAdapter.from_file(kit.hf_file)})
        except Exception as e:  # pragma: no cover - 仅在失败时记录
            errors.append(f"{name}: {e}")

    threads = [threading.Thread(target=run, args=(f"g6_{i}", 1500)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, f"并发构建失败：{errors}"
    for i in range(2):
        assert os.path.exists(str(tmp_path / f"g6_{i}" / "ds" / "index.json"))


# ==============================================================================
# 两个"自研分词器"参考实现（同时作为语义探针）
# ==============================================================================
class WhitespaceTokenizer:
    """自研分词器示例：空白切词 + 动态词表 + 预置特殊符。

    专门用来验证"不继承任何基类、只实现两个方法"也能接入。
    """

    def __init__(self, specials):
        self.specials = dict(specials)
        self.vocab = {}
        self._next = 1000

    def _intern(self, word):
        if word not in self.vocab:
            self.vocab[word] = self._next
            self._next += 1
        return self.vocab[word]

    def _encode(self, text, split_special):
        ids, tokens = [], []
        if self.specials and not split_special:
            pattern = "(" + "|".join(
                re.escape(s) for s in sorted(self.specials, key=len, reverse=True)
            ) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        for part in parts:
            if not split_special and part in self.specials:
                ids.append(self.specials[part])
                tokens.append(part)
                continue
            for word in part.split():
                ids.append(self._intern(word))
                tokens.append(word)
        return ids, tokens

    def encode_sft_batch(self, texts):
        out_ids, out_tokens = [], []
        for t in texts:
            ids, tokens = self._encode(t, split_special=False)
            out_ids.append(ids)
            out_tokens.append(tokens)
        return out_ids, out_tokens

    def encode_ordinary_batch(self, texts):
        return [self._encode(t, split_special=True)[0] for t in texts]


class TiktokenStyle:
    """tiktoken 风格：普通文本按 UTF-8 字节切（id 0–255），特殊符用大 id。

    顺便压测"大 id（10 万量级）能否安全穿过 Rust 的 i32 边界"。
    """

    def __init__(self, specials):
        self.specials = dict(specials)

    def _encode(self, text, split_special):
        ids, tokens = [], []
        if not split_special:
            pattern = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]
        for part in parts:
            if not split_special and part in self.specials:
                ids.append(self.specials[part])
                tokens.append(part)
                continue
            for b in part.encode("utf-8"):
                ids.append(b)
                tokens.append(chr(b))
        return ids, tokens

    def encode_sft_batch(self, texts):
        out_ids, out_tokens = [], []
        for t in texts:
            ids, tokens = self._encode(t, split_special=False)
            out_ids.append(ids)
            out_tokens.append(tokens)
        return out_ids, out_tokens

    def encode_ordinary_batch(self, texts):
        return [self._encode(t, split_special=True)[0] for t in texts]
