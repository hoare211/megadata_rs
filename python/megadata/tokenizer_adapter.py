"""第三方分词器接入适配层（方案 §3）。

设计要点：
- **零硬依赖**：本模块只 import 标准库。`tokenizers` / `transformers` 都是软依赖，
  只在真正需要加载 HF 分词器时才 import —— 用 bbpe 默认路径的用户完全不受影响。
- **两条路径共享全部后段**：本模块只负责"把文本变成 ids/tokens"，渲染（Rust 模板
  引擎）、pack、压缩、写盘、索引全部由 Rust 承担（方案 §2）。
- **语义映射收口在此**：`add_special_tokens` 恒 False、PT 方向标记按普通文本切碎
  等容易踩的坑，全部由适配器内部处理，不让用户自己踩（§3.2）。
"""

from __future__ import annotations

import contextlib
import os
import warnings
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

__all__ = ["BatchTokenizer", "HfTokenizerAdapter", "adapt_tokenizer", "load_tokenizer_file"]


@runtime_checkable
class BatchTokenizer(Protocol):
    """megadata 构建管线可用的分词器最小契约（鸭子类型，无需继承）。

    两个方法分别对应两种构建模式：
    - `encode_sft_batch`：SFT 模式。文本中出现的标记字符串（如 `<|im_start|>`）
      **识别为单个 token**，等价于 bbpe 的 `encode_with_types(_, {}, true)`。
      必须同时返回 tokens 字符串 —— packer 靠它做 turn 标记的 `contains` 匹配。
    - `encode_ordinary_batch`：PT 模式。标记字符串**按普通文本切碎**，等价于
      bbpe 的 `encode_ordinary`。

    约定：
    - ids 为 Python int，必须非负且 <= 2^31-1（Rust 侧会做范围校验，越界报错）；
    - 空文本 → 空 ids 合法（packer 现有 0 长文档行为，两路径一致）；
    - 返回值外层长度必须等于入参长度。
    """

    def encode_sft_batch(self, texts: List[str]) -> Tuple[List[List[int]], List[List[str]]]:
        """返回 `(ids_batch, tokens_batch)`。"""
        ...

    def encode_ordinary_batch(self, texts: List[str]) -> List[List[int]]:
        """返回 `ids_batch`。"""
        ...


def _fast_backend(tokenizer: Any) -> Optional[Any]:
    """取出底层 `tokenizers.Tokenizer`（HF fast 引擎本身是 Rust，encode_batch 内部自并行）。

    返回 None 表示这是慢分词器（纯 Python 实现），走 `__call__` 回退路径。
    """
    if hasattr(tokenizer, "encode_batch"):
        return tokenizer
    fast = getattr(tokenizer, "_tokenizer", None)
    if fast is not None and hasattr(fast, "encode_batch"):
        return fast
    return None


def _infer_vocab_size(tokenizer: Any, fast: Optional[Any]) -> Optional[int]:
    for obj in (fast, tokenizer):
        if obj is None:
            continue
        getter = getattr(obj, "get_vocab_size", None)
        if getter is not None:
            try:
                return int(getter())
            except Exception:
                pass
    try:
        return len(tokenizer)
    except Exception:
        return None


@contextlib.contextmanager
def _special_tokens_split(fast: Any, split: bool):
    """临时切换 `encode_special_tokens` 开关（退出时还原）。

    `encode_special_tokens=True` 即 bbpe `encode_ordinary` 的语义：文本里出现的
    标记字符串按普通文本切碎。这是 PT 方向唯一的语义坑（§3.2 / T3），收口在此。

    切换是**进程内全局**的（作用在 `tokenizers.Tokenizer` 对象上），因此必须在
    finally 里还原。相位 T 是单点串行调用（§4.2），不存在并发竞争。
    """
    prev = getattr(fast, "encode_special_tokens", None)
    if prev is None:
        # 老版本 tokenizers 没有该开关 —— 无法表达 ordinary 语义，交由调用方回退
        raise RuntimeError(
            "the current `tokenizers` version does not support the `encode_special_tokens` switch, so the "
            "'special-token strings split as ordinary text' (PT) semantics cannot be expressed. "
            "Upgrade tokenizers (>=0.20), or pass a transformers tokenizer object "
            "(the split_special_tokens fallback path)."
        )
    try:
        fast.encode_special_tokens = bool(split)
        yield
    finally:
        fast.encode_special_tokens = prev


class HfTokenizerAdapter:
    """把 HF 分词器适配成 `BatchTokenizer`。

    接受 `tokenizers.Tokenizer` 或 `transformers.PreTrainedTokenizerFast`（含
    slow → fast 自动转换的 sentencepiece 系）。

    语义映射与一致性守卫（§3.2）：

    | 语义 | bbpe_rs（默认路径） | 本适配器 |
    |---|---|---|
    | SFT 标记识别 | `encode_with_types(_, {}, true)` | `encode_batch(add_special_tokens=False)` |
    | PT 标记按普通文本 | `encode_ordinary` | `encode_special_tokens=True` |
    | add_special_tokens | 无此概念（不注入） | **恒 False**（防 post_processor 注入 BOS/EOS 错位） |
    | offsets / types | 产出但不消费 | 不进契约 |
    """

    def __init__(self, tokenizer: Any, name_or_path: Optional[str] = None):
        self._tok = tokenizer
        self._fast = _fast_backend(tokenizer)
        self.name_or_path = name_or_path
        self.vocab_size = _infer_vocab_size(tokenizer, self._fast)

    # ---------------- 构造入口 ----------------

    @classmethod
    def from_file(cls, path: str) -> "HfTokenizerAdapter":
        """从 `tokenizer.json`（或含它的目录）加载。`tokenizers` 为软依赖。"""
        from tokenizers import Tokenizer

        candidate = path
        if os.path.isdir(path):
            candidate = os.path.join(path, "tokenizer.json")
            if not os.path.exists(candidate):
                raise FileNotFoundError(f"no tokenizer.json under directory {path!r}")
        return cls(Tokenizer.from_file(candidate), name_or_path=path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "HfTokenizerAdapter":
        """用 `transformers.AutoTokenizer` 加载（含 trust_remote_code 等全部特性）。"""
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(path, **kwargs), name_or_path=path)

    # ---------------- 契约实现 ----------------

    def encode_sft_batch(self, texts: List[str]) -> Tuple[List[List[int]], List[List[str]]]:
        """SFT 模式：标记字符串识别为单 token，返回 `(ids, tokens)`。"""
        return self._encode(list(texts), split_special_tokens=False)

    def encode_ordinary_batch(self, texts: List[str]) -> List[List[int]]:
        """PT 模式：标记字符串按普通文本切碎。"""
        ids, _tokens = self._encode(list(texts), split_special_tokens=True)
        return ids

    # ---------------- manifest 指纹信息 ----------------

    @property
    def identity(self) -> str:
        """adapter 报告的标识，进 manifest 的 `tokenizer.identity`。"""
        for src in (self.name_or_path, getattr(self._tok, "name_or_path", None)):
            if isinstance(src, str) and src:
                return src
        return type(self._tok).__name__ or "opaque"

    # ---------------- 内部实现 ----------------

    def _encode(
        self, texts: List[str], split_special_tokens: bool
    ) -> Tuple[List[List[int]], List[List[str]]]:
        if not texts:
            return [], []
        fast = self._fast
        # 需要 ordinary 语义但底层引擎不支持该开关 → 走 transformers 的
        # `split_special_tokens` 回退路径（慢路径，但语义正确优先）。
        if fast is None or (split_special_tokens and not hasattr(fast, "encode_special_tokens")):
            return self._encode_slow(texts, split_special_tokens)

        with _special_tokens_split(fast, split_special_tokens):
            # add_special_tokens 恒 False：post_processor 自动注入 BOS/EOS 会让
            # 所有文档长度错位（§3.2 / T2）。
            encodings = fast.encode_batch(texts, add_special_tokens=False)
        return (
            [list(e.ids) for e in encodings],
            [list(e.tokens) for e in encodings],
        )

    def _encode_slow(
        self, texts: List[str], split_special_tokens: bool
    ) -> Tuple[List[List[int]], List[List[str]]]:
        """纯 Python / transformers 慢分词器回退路径。"""
        kwargs: dict = {"add_special_tokens": False}
        if split_special_tokens:
            kwargs["split_special_tokens"] = True
        try:
            encoded = self._tok(texts, **kwargs)
        except TypeError:
            # 旧版 transformers 不支持 split_special_tokens：静默回退会让
            # "ordinary 语义" 悄悄失效（特殊 token 未被切分），记一次以便用户感知。
            kwargs.pop("split_special_tokens", None)
            warnings.warn(
                "the current tokenizer does not support split_special_tokens=True; "
                "falling back to encoding without it (special tokens will not be split). "
                "Upgrade `transformers` for ordinary-tokenization semantics.",
                stacklevel=2,
            )
            encoded = self._tok(texts, **kwargs)
        ids_batch = [list(row) for row in encoded["input_ids"]]
        convert = getattr(self._tok, "convert_ids_to_tokens", None)
        if convert is None:
            return ids_batch, [[] for _ in ids_batch]
        return ids_batch, [[convert(i) for i in row] for row in ids_batch]


def adapt_tokenizer(obj: Any) -> Any:
    """把任意分词器对象适配为 `BatchTokenizer` 契约。

    - 已是 `HfTokenizerAdapter` → 原样返回；
    - 已实现协议的两个方法（自研 / tiktoken 包装）→ 原样返回；
    - HF 分词器对象 → 包成 `HfTokenizerAdapter`；
    - 其余 → TypeError（列出契约要求）。
    """
    if isinstance(obj, HfTokenizerAdapter):
        return obj
    if hasattr(obj, "encode_sft_batch") and hasattr(obj, "encode_ordinary_batch"):
        return obj
    if (
        hasattr(obj, "encode_batch")
        or hasattr(obj, "_tokenizer")
        or hasattr(obj, "batch_encode_plus")
    ):
        return HfTokenizerAdapter(obj)
    raise TypeError(
        "cannot adapt the object to BatchTokenizer: it must implement the encode_sft_batch / "
        "encode_ordinary_batch methods, or be an HF tokenizer object "
        "(tokenizers.Tokenizer / transformers tokenizer). "
        f"Got type: {type(obj).__name__}"
    )


def load_tokenizer_file(path: str) -> HfTokenizerAdapter:
    """两级加载分派的**第二级**：把 `tokenizer.json`（或目录）包成适配器。

    失败时抛出的异常信息会被 builder.py 与"第一级（bbpe）失败原因"并列展示（T6）。
    """
    return HfTokenizerAdapter.from_file(path)
