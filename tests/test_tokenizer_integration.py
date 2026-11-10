"""第三方分词器接入测试（方案 §7：T1–T6 / T8）。

**CI 无网络**：用 `tokenizers` 在本地训练一个小 BPE（中文 + emoji + special token），
再把它导出成等价的 **bbpe_rs 模型文件**，于是"同一个分词器"可以同时走两条路径：

- 默认路径：Rust 直连 bbpe_rs；
- Python 路径：`HfTokenizerAdapter` 包 `tokenizers.Tokenizer`。

T1（双路径逐 batch 等价）是招牌用例：任何一侧语义映射漂移、或"bbpe ↔ HF 双向
兼容"这个前提本身被破坏，它都会失败。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from megadata import DataBuilder, HfTokenizerAdapter, BatchStream, read_manifest

# `tokenizers` 是软依赖：缺失时整组用例跳过（bbpe 默认路径的 CI 不受影响）
tokenizers = pytest.importorskip("tokenizers", reason="需要 tokenizers 才能构造等价分词器")

from tokenizers import Tokenizer, Regex, decoders, models, trainers  # noqa: E402
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split  # noqa: E402

# 两引擎都能编译的预分词正则（bbpe 用 fancy-regex，HF 用 onig）：
# 完全覆盖 —— 字母 / 数字 / 单个非字母数字非空白 / 空白，不留未匹配间隙。
PRE_TOKENIZER_PATTERN = r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]|\s+"

CORPUS = [
    "你好世界，这是一段中文语料 hello world",
    "机器学习 machine learning 需要大量数据 😀😀",
    "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好呀<|im_end|>",
    "abc abcab abcabc 123 4567 😀 你好你好你好",
] * 40

SPECIALS = ["<|im_start|>", "<|im_end|>"]


# ==============================================================================
# 工具：造一个"两条路径都认"的分词器
# ==============================================================================
def _bytes_to_unicode() -> dict:
    """GPT-2 / HF ByteLevel 的字节 ↔ 字符映射表。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def build_hf_tokenizer(vocab_size: int = 400) -> "Tokenizer":
    """训练一个小 BPE：ByteLevel(use_regex=False) + 自定义 Split（与 bbpe 同正则）。

    基础字节的 id 由 HF 按 `ByteLevel.alphabet()` 顺序分配（与字节值无关），
    `write_bbpe_model` 里会做一次 id 置换来对齐 bbpe 的口径。
    """
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = Sequence([
        Split(Regex(PRE_TOKENIZER_PATTERN), behavior="isolated"),
        ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer=trainer)
    # special token 必须在训练后加：id 落在 merges 之后（bbpe 校验 V5 的"尾部性质"）
    tok.add_special_tokens(SPECIALS)
    return tok


def _align_ids(hf: "Tokenizer"):
    """算出 HF → 规范 id 的置换 σ，并拆出模型各部件。

    为什么要置换：bbpe 的 BPE 内核**直接把字节值当初始 id** 用
    （`bpe_core_full`: `bytes.map(|b| b as i32)`），即要求 `vocab[id] == [id]`；
    而 HF 按 `ByteLevel.alphabet()` 的顺序（≈ 按码点排序）分配基础字节 id，
    `id 0` 是 `'!'`（字节 33）。`initial_alphabet` 的顺序会被 HF 忽略，所以只能
    在**文件层面**把两边的 id 统一到"基础字节 id == 字节值"这一规范布局上：

    - 基础字节 id  ⟶ 字节值
    - 第 i 条 merge 的产物 ⟶ `256 + i`（rank 不变，满足 bbpe 校验 V3 的连续性）
    - special token ⟶ 保持原 id（必须 > `255 + m`，满足 V5 的尾部性质）
    """
    raw = json.loads(hf.to_str())
    model = raw["model"]
    vocab_str = model["vocab"]
    merges_raw = model["merges"]
    added = {t["content"]: t["id"] for t in raw.get("added_tokens", [])}
    inv = {ch: b for b, ch in _bytes_to_unicode().items()}

    base_byte = {i: inv[hf.id_to_token(i)] for i in range(256)}
    assert sorted(base_byte.values()) == list(range(256)), "基础字节 id 不是 0..255 的排列"
    assert all(hid > 255 + len(merges_raw) for hid in added.values()), (
        "special token 的 id 必须落在 merges 之后（bbpe 校验 V5 的尾部性质）"
    )

    sigma = {i: base_byte[i] for i in range(256)}
    for j in range(len(merges_raw)):
        sigma[256 + j] = 256 + j
    for hid in added.values():
        sigma[hid] = hid
    return raw, vocab_str, merges_raw, added, inv, sigma


def write_hf_tokenizer(hf: "Tokenizer", out_path: str) -> str:
    """导出**规范 id 布局**的 HF tokenizer.json（与 bbpe 侧逐位对齐）。

    `Tokenizer.from_file` 会原样采用 json 里的 id，所以改写 `model.vocab` 即可。
    """
    raw, vocab_str, _merges, _added, _inv, sigma = _align_ids(hf)
    raw["model"]["vocab"] = {tok: sigma[hid] for tok, hid in vocab_str.items()}
    for t in raw.get("added_tokens", []):
        t["id"] = sigma[t["id"]]
    Path(out_path).write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return out_path


def write_bbpe_model(hf: "Tokenizer", out_path: str) -> str:
    """把同一个分词器导出成 bbpe_rs v1 模型文件（HF → bbpe 桥，纯机械翻译）。

    - `vocab`：byte-level 字符串 → id  ⟶  规范 id → 原始字节
    - `merges`：`[[a, b], ...]` 按序  ⟶  `"σ(id(a)),σ(id(b))": 256 + i`
    - `added_tokens` ⟶ `special_tokens`
    """
    _raw, vocab_str, merges_raw, added, inv, sigma = _align_ids(hf)

    vocab = {}
    for tok_str, hid in vocab_str.items():
        vocab[str(sigma[hid])] = [inv[ch] for ch in tok_str]
    for content, hid in added.items():
        vocab[str(sigma[hid])] = list(content.encode("utf-8"))

    merges = {}
    for i, pair in enumerate(merges_raw):
        a, b = (pair.split(" ", 1) if isinstance(pair, str) else pair[:2])
        new_id = 256 + i
        assert vocab_str[a + b] == new_id, (
            "HF merges 顺序与 id 布局不符（bbpe 校验 V3 要求 new_id 连续）"
        )
        merges[f"{sigma[vocab_str[a]]},{sigma[vocab_str[b]]}"] = new_id

    payload = {
        "format": "bbpe-rs-model",
        "format_version": 1,
        "vocab": vocab,
        "merges": merges,
        "added_tokens": added,
        "pre_tokenizer_pattern": PRE_TOKENIZER_PATTERN,
    }
    Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
    return out_path


@pytest.fixture(scope="module")
def tokenizer_pair(tmp_path_factory):
    """`(hf_tokenizer_json, bbpe_model_json, marker_ids)`：同一分词器的两种文件形态。

    `marker_ids` 给 bbpe 路径用：bbpe 路径的 ID 自动推导依赖 `bbpe_tokenizer`
    软依赖（CI 未必装），显式传入同样能被 Rust 入口的等价性校验覆盖（T4）。
    """
    d = tmp_path_factory.mktemp("tok_pair")
    hf = build_hf_tokenizer()
    hf_path = write_hf_tokenizer(hf, str(d / "tokenizer.json"))
    bbpe_path = write_bbpe_model(hf, str(d / "model.bbpe"))
    aligned = Tokenizer.from_file(hf_path)
    marker_ids = {
        "turn_start_id": aligned.token_to_id("<|im_start|>"),
        "turn_end_id": aligned.token_to_id("<|im_end|>"),
    }
    return hf_path, bbpe_path, marker_ids


@pytest.fixture
def sft_jsonl(tmp_path):
    """小型 ChatML 语料（确定性）。"""
    p = tmp_path / "sft.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": f"问题{i}：你好吗 😀"},
                    {"role": "assistant", "content": f"回答{i}：我很好，机器学习真有趣"},
                ]
            }, ensure_ascii=False) + "\n")
    return str(p)


@pytest.fixture
def pt_jsonl(tmp_path):
    """PT 语料：故意混入字面标记串（T3 的素材）。"""
    p = tmp_path / "pt.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i in range(200):
            f.write(json.dumps({
                "text": f"段落{i} 中文语料 <|im_start|>user 标记当普通文本 123 😀\n"
            }, ensure_ascii=False) + "\n")
    return str(p)


CHAT_TEMPLATE = (
    "{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}"
)

MAX_SEQ_LEN = 256
BATCH_SIZE = 4


def _build(input_path, output_dir, mode, builder_kwargs, template=CHAT_TEMPLATE, **kw):
    builder = DataBuilder(max_seq_len=MAX_SEQ_LEN, **builder_kwargs)
    builder.build_dataset(
        input_paths=[input_path],
        output_dir=output_dir,
        mode=mode,
        template_str=template,
        batch_size=BATCH_SIZE,
        pad_id=0,
        **kw,
    )
    return builder


def _dump(output_dir, mode):
    """按 batch 顺序读出全部张量（确定性：world_size=1 / shuffle=False）。"""
    loader = BatchStream(
        index_json_path=os.path.join(output_dir, "index.json"),
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        world_size=1,
        rank=0,
        shuffle=False,
    )
    out = []
    for batch in loader:
        # v3/F-10：batch 携带 "format" 键（str），非 tensor 值原样保留
        out.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in batch.items()})
    return out


# ==============================================================================
# T1：双路径等价（招牌）
# ==============================================================================
def test_t1_dual_path_equivalence_sft(tokenizer_pair, sft_jsonl, tmp_path):
    """同一分词器：bbpe 默认路径 vs HfTokenizerAdapter → shard 逐 batch 一致。"""
    hf_path, bbpe_path, marker_ids = tokenizer_pair

    out_bbpe = str(tmp_path / "bbpe_sft")
    out_py = str(tmp_path / "py_sft")

    _build(sft_jsonl, out_bbpe, "sft", {"tokenizer_path": bbpe_path}, **marker_ids)
    _build(sft_jsonl, out_py, "sft", {"tokenizer": HfTokenizerAdapter.from_file(hf_path)},
           **marker_ids)

    a, b = _dump(out_bbpe, "sft"), _dump(out_py, "sft")
    assert a, "bbpe 路径产出为空"
    assert len(a) == len(b), f"batch 数不一致：{len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys()
        for k in x:
            assert x[k] == y[k], f"batch {i} 的 {k} 不一致（双路径语义漂移）"


def test_t1_dual_path_equivalence_pt(tokenizer_pair, pt_jsonl, tmp_path):
    """PT 模式的双路径等价（含字面标记串的语料）。"""
    hf_path, bbpe_path, marker_ids = tokenizer_pair
    eos_str = "<|im_end|>"

    out_bbpe = str(tmp_path / "bbpe_pt")
    out_py = str(tmp_path / "py_pt")
    _build(pt_jsonl, out_bbpe, "pt", {"tokenizer_path": bbpe_path},
           eos_str=eos_str, eos_id=marker_ids["turn_end_id"])
    _build(pt_jsonl, out_py, "pt", {"tokenizer": HfTokenizerAdapter.from_file(hf_path)},
           eos_str=eos_str, eos_id=marker_ids["turn_end_id"])

    a, b = _dump(out_bbpe, "pt"), _dump(out_py, "pt")
    assert a and len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x["input_ids"] == y["input_ids"], f"PT batch {i} 的 input_ids 不一致"


# ==============================================================================
# T2：适配器契约
# ==============================================================================
def test_t2_adapter_matches_official_library(tokenizer_pair):
    """adapter 的输出与官方 `tokenizers` 逐位一致；空文本；无 BOS/EOS 注入。"""
    hf_path, _, _ = tokenizer_pair
    adapter = HfTokenizerAdapter.from_file(hf_path)
    official = Tokenizer.from_file(hf_path)

    texts = ["", "你好世界", "hello <|im_start|>assistant<|im_end|>", "😀😀 123", "   "]
    ids, tokens = adapter.encode_sft_batch(texts)

    assert len(ids) == len(texts)
    assert ids[0] == [] and tokens[0] == [], "空文本应产出空 ids / 空 tokens"
    for text, got_ids, got_tokens in zip(texts, ids, tokens):
        ref = official.encode(text, add_special_tokens=False)
        assert got_ids == list(ref.ids), f"{text!r} 的 ids 与官方库不一致"
        assert got_tokens == list(ref.tokens), f"{text!r} 的 tokens 与官方库不一致"

    # special token 必须是整串单 token（SFT 语义）
    sft_ids, sft_tokens = adapter.encode_sft_batch(["<|im_start|>"])
    assert sft_ids[0] == [official.token_to_id("<|im_start|>")]
    assert sft_tokens[0] == ["<|im_start|>"]

    # add_special_tokens 恒 False：不注入任何 post_processor 产物
    plain_ids, _ = adapter.encode_sft_batch(["hello"])
    assert plain_ids[0] == list(official.encode("hello", add_special_tokens=False).ids)


def test_t2_vocab_size_and_identity(tokenizer_pair):
    hf_path, _, _ = tokenizer_pair
    adapter = HfTokenizerAdapter.from_file(hf_path)
    assert adapter.vocab_size == Tokenizer.from_file(hf_path).get_vocab_size()
    assert adapter.identity == hf_path


# ==============================================================================
# T3：PT 的 special 语义（标记按普通文本切碎）
# ==============================================================================
def test_t3_pt_splits_marker_as_ordinary_text(tokenizer_pair):
    """PT 方向：`encode_ordinary_batch` 必须把字面标记串切碎，与 bbpe encode_ordinary 同向。"""
    hf_path, _, _ = tokenizer_pair
    adapter = HfTokenizerAdapter.from_file(hf_path)
    official = Tokenizer.from_file(hf_path)

    text = "<|im_start|>user"
    marker_id = official.token_to_id("<|im_start|>")

    ordinary = adapter.encode_ordinary_batch([text])[0]
    assert marker_id not in ordinary, "PT 模式不应把字面标记识别为 special token"

    # 与 transformers 的 split_special_tokens=True 口径一致（这是唯一的语义坑）
    try:
        from transformers import PreTrainedTokenizerFast

        ref = PreTrainedTokenizerFast(tokenizer_object=official)(
            text, add_special_tokens=False, split_special_tokens=True
        )["input_ids"]
        assert ordinary == list(ref)
    except ImportError:
        pytest.skip("transformers 不可用，跳过交叉验证")

    # SFT 方向相反：必须识别为单 token
    sft_ids, _ = adapter.encode_sft_batch([text])
    assert sft_ids[0][0] == marker_id


# ==============================================================================
# T4：构建期校验（D8）
# ==============================================================================
def test_t4_marker_multi_token_raises(tokenizer_pair, sft_jsonl, tmp_path):
    """标记编码出多个 token → 报错（SFT 标签正确性优先）。"""
    hf_path, _, _ = tokenizer_pair
    builder = DataBuilder(tokenizer=HfTokenizerAdapter.from_file(hf_path), max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError, match="exactly 1"):
        builder.build_dataset(
            input_paths=[sft_jsonl],
            output_dir=str(tmp_path / "bad"),
            mode="sft",
            template_str=CHAT_TEMPLATE,
            batch_size=BATCH_SIZE,
            turn_start_str="这不是任何特殊符号",
        )


def test_t4_explicit_id_mismatch_raises(tokenizer_pair, sft_jsonl, tmp_path):
    """显式传入的 id 与实际编码不一致 → 报错，信息含实际 token。"""
    hf_path, _, _ = tokenizer_pair
    real_id = Tokenizer.from_file(hf_path).token_to_id("<|im_start|>")
    wrong = real_id + 1000
    builder = DataBuilder(tokenizer=HfTokenizerAdapter.from_file(hf_path), max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(ValueError, match=str(wrong)):
        builder.build_dataset(
            input_paths=[sft_jsonl],
            output_dir=str(tmp_path / "bad"),
            mode="sft",
            template_str=CHAT_TEMPLATE,
            batch_size=BATCH_SIZE,
            turn_start_id=wrong,
        )


def test_t4_bbpe_path_marker_validated_in_rust(tokenizer_pair, sft_jsonl, tmp_path):
    """bbpe 路径的对称校验在 Rust 入口：多 token 标记 → 构建失败。"""
    _, bbpe_path, _ids = tokenizer_pair
    builder = DataBuilder(tokenizer_path=bbpe_path, max_seq_len=MAX_SEQ_LEN)
    with pytest.raises(Exception, match="exactly 1"):
        builder.build_dataset(
            input_paths=[sft_jsonl],
            output_dir=str(tmp_path / "bad"),
            mode="sft",
            template_str=CHAT_TEMPLATE,
            batch_size=BATCH_SIZE,
            turn_start_str="这不是任何特殊符号",
            turn_start_id=7,  # 显式给出，绕开 Python 侧软依赖推导
            turn_end_id=8,
        )


# ==============================================================================
# T5：GIL 回归（§4.2 红线 1/2）
# ==============================================================================
def test_t5_gil_is_released_during_rust_phases(tokenizer_pair, sft_jsonl, tmp_path):
    """构建期间看门狗线程必须周期性拿到 GIL —— 防相位 R/P 漏掉 GIL 释放。"""
    hf_path, _, _ = tokenizer_pair
    stop = threading.Event()
    ticks = []

    def watchdog():
        while not stop.is_set():
            time.sleep(0.005)      # 等 GIL；若 Rust 全程持 GIL 这里会饿死
            ticks.append(1)        # 需要 GIL 才能执行

    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    try:
        _build(sft_jsonl, str(tmp_path / "gil"), "sft",
               {"tokenizer": HfTokenizerAdapter.from_file(hf_path)})
    finally:
        stop.set()
        t.join(timeout=5)

    assert len(ticks) > 5, f"看门狗只拿到 {len(ticks)} 次 GIL：Rust 相位可能持着 GIL 不放"


def test_t5_cancel_from_progress_cb_yields_interrupt(tokenizer_pair, sft_jsonl, tmp_path):
    """`tokenizer=` 构建可被打断 → KeyboardInterrupt / MegadataInterrupted。"""
    from megadata import MegadataInterrupted

    hf_path, _, _ = tokenizer_pair
    builder = DataBuilder(tokenizer=HfTokenizerAdapter.from_file(hf_path), max_seq_len=MAX_SEQ_LEN)

    def cb(_progress):
        builder.cancel()

    with pytest.raises((KeyboardInterrupt, MegadataInterrupted)):
        builder.build_dataset(
            input_paths=[sft_jsonl],
            output_dir=str(tmp_path / "cancelled"),
            mode="sft",
            template_str=CHAT_TEMPLATE,
            batch_size=BATCH_SIZE,
            progress_cb=cb,
        )


# ==============================================================================
# T6：加载分派（§3.3）
# ==============================================================================
def test_t6_bbpe_file_uses_default_backend(tokenizer_pair):
    _, bbpe_path, _ = tokenizer_pair
    b = DataBuilder(tokenizer_path=bbpe_path)
    assert b.backend == "bbpe" and b.adapter is None


def test_t6_hf_file_falls_back_to_python(tokenizer_pair):
    hf_path, _, _ = tokenizer_pair
    b = DataBuilder(tokenizer_path=hf_path)
    assert b.backend == "python"
    assert isinstance(b.adapter, HfTokenizerAdapter)


def test_t6_garbage_file_lists_both_failures(tmp_path):
    junk = tmp_path / "junk.json"
    junk.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        DataBuilder(tokenizer_path=str(junk))
    msg = str(exc.value)
    assert "both loading stages failed" in msg
    assert "bbpe_rs" in msg and "tokenizers" in msg, "必须并列两级失败原因"


def test_t6_tokenizer_and_path_are_mutually_exclusive(tokenizer_pair):
    hf_path, bbpe_path, _ = tokenizer_pair
    with pytest.raises(ValueError, match="exactly one of"):
        DataBuilder(tokenizer_path=bbpe_path, tokenizer=HfTokenizerAdapter.from_file(hf_path))
    # P2 (mm)：两者都不给 → backend="none"（不再报错）；仅供 mode="vlm" 使用，
    # pt/sft 在 build_dataset 期据此显式报错（见 test_mm_p2_vlm_packer 的回归用例）。
    b = DataBuilder()
    assert b.backend == "none"


# ==============================================================================
# T8：manifest 分词器指纹（§6）
# ==============================================================================
def test_t8_manifest_tokenizer_fingerprint(tokenizer_pair, sft_jsonl, tmp_path):
    hf_path, bbpe_path, marker_ids = tokenizer_pair

    out_bbpe = str(tmp_path / "fp_bbpe")
    out_py = str(tmp_path / "fp_py")
    _build(sft_jsonl, out_bbpe, "sft", {"tokenizer_path": bbpe_path}, **marker_ids)
    _build(sft_jsonl, out_py, "sft", {"tokenizer": HfTokenizerAdapter.from_file(hf_path)},
           **marker_ids)

    fp_bbpe = read_manifest(os.path.join(out_bbpe, "index.json"))["tokenizer"]
    fp_py = read_manifest(os.path.join(out_py, "index.json"))["tokenizer"]

    assert fp_bbpe["backend"] == "bbpe"
    assert fp_bbpe["identity"].endswith("@v1")      # sha256[:16] + format_version
    assert fp_bbpe["vocab_size"] > 256

    assert fp_py["backend"] == "python"
    assert fp_py["identity"] == hf_path
    assert fp_py["vocab_size"] == Tokenizer.from_file(hf_path).get_vocab_size()


def test_t8_manifest_without_tokenizer_field_is_backward_compatible():
    """向后兼容：旧 index.json 没有 tokenizer 字段时，读取端解析不受影响。"""
    with tempfile.TemporaryDirectory() as d:
        index = Path(d) / "index.json"
        index.write_text(json.dumps({
            "version": "2.0", "format": "Memmap", "max_seq_len": 8,
            "global_stats": {"total_tokens": 8, "total_batches": 1, "total_shards": 1, "batch_size": 1},
            "schema": {}, "shards": [],
        }), encoding="utf-8")
        manifest = read_manifest(str(index))
        assert "tokenizer" not in manifest, "旧 manifest 不应凭空多出 tokenizer 字段"
