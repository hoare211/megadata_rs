# python/megadata/builder.py
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from megadata._rust import DataBuilder as RustDataBuilder
from megadata._rust import read_manifest_json
# F-36：从 _signals 取 `_register`（不再 import dataloader → 构建路径不拖入 torch/pyarrow）
from megadata._signals import _register
from megadata._options import normalize_compression, normalize_mode
from megadata.tokenizer_adapter import adapt_tokenizer, load_tokenizer_file

# bbpe_rs 模型文件（`ExtendedTokenizerModel`）的 format 常量。加载器据此给出
# 干净的失败信号（F4）；这里用"文件头部是否含该字面量"做**零成本**的一级判别：
# serde 按结构体字段序序列化，`format` 是第一个字段，64KB 头部足够覆盖。
_BBPE_FORMAT_MAGIC = b"bbpe-rs-model"
_BBPE_SNIFF_BYTES = 64 * 1024

#: 哨兵：区分"未传入"与"显式传入 None / 默认值"（兼容 shim 用）
_UNSET = object()

#: 模式默认标记文本（`Marker.text=None` 表示未干预时使用）
_DEFAULT_EOS = "<|endoftext|>"
_DEFAULT_TURN_START = "<|im_start|>"
_DEFAULT_TURN_END = "<|im_end|>"


@dataclass(frozen=True)
class Marker:
    """特殊标记（eos / turn_start / turn_end）的用户意图。

    ``text=None`` 表示**未干预**（用模式默认文本）——因此"只传 str"与"只传 id"
    都会被识别为显式设置；配合 `BuildConfig` 的模式相关性校验，
    "传了不生效"的参数在构造期即报错。

    ``strict``：非单 token 时的口径——True 报错、False 告警取首个。

    分级是有意的（**默认值也按标记区分**）：
    - `turn_start` / `turn_end`：默认 `strict=True`（多 token 会损坏监督标签）；
    - `eos`：`BuildConfig.eos` 的默认值是 `Marker(strict=False)`——eos 只影响文档边界，
      非掩码关键路径。显式传 `Marker(strict=True)` 时 eos 同样强校验"恰好 1 个 token"
      （此前该字段对 eos 是死旋钮：传了不起作用也不报错）。
    """
    text: Optional[str] = None
    id: Optional[int] = None
    strict: bool = True

    @property
    def intervened(self) -> bool:
        """用户是否对本标记做过显式设置（text 或 id 任一非 None）。"""
        return self.text is not None or self.id is not None


@dataclass(frozen=True)
class BuildConfig:
    """数据集构建配置值对象（对齐 `SimulationConfig` 范式）。

    划分标准：**配置 = 决定产物 / 进 manifest 指纹的输入；回调 = 运行期副作用**——
    因此 `progress_cb` 留在 `build_dataset` 签名里，不在本类。

    模式相关性（A5：传了不生效的参数直接报错）：
    - ``mode="sft"`` 不使用 `eos`（监督边界由 turn 标记决定，Rust DocumentAwarePacker
      不消费 eos_id）；
    - ``mode="pt"`` 不使用 `turn_start` / `turn_end`（GreedyPacker 无对话轮次概念）；
    - ``mode="vlm"`` 不使用 `eos` / `turn_start` / `turn_end` / `template_str`
      （图文编码全在 `modality_encoder` 回调内自理，样本段表来自 P0 读取侧）。

    用法::

        cfg = BuildConfig(input_paths=["a.jsonl"], output_dir="out/sft", mode="sft",
                          template_str=TEMPLATE, batch_size=16)
        DataBuilder(tokenizer_path=...).build_dataset(cfg)

    vlm 专用字段（``mode="vlm"`` 时必填/生效，P2）：
    - ``modality_id_ranges``：各模态专属 token id 区间 ``[(start, end, kind), ...]``，
      ``kind ∈ {"image","audio","video"}``（D6：不声明则 token 校验无从谈起 → 必填）；
    - ``max_visual_tokens_per_sample``：D7 样本级总视觉预算；None = 行宽（msl）；
    - ``image_label``：图像段 label（T3，默认 -100）；
    - ``visual_budget_hard_fail``：超预算策略，False=TruncateAndCount（默认）。

    vlm 专用字段（P3a，audio 流；``mode="vlm"`` 时生效）：
    - ``w_ms``：对齐窗长（整数毫秒；默认 2000。含 audio 段的构建必须 ≥1）；
    - ``audio_window_token_rate``：音频窗内 token 率（tokens/s；None=未声明——
      含 audio 段的构建缺它即 fail-fast，禁静默降级）；
    - ``audio_max_seq_len`` / ``audio_batch_size`` / ``audio_batches_per_shard``：
      audio 流独立尺寸（None = 同 Vlm 流）；
    - ``audio_strict_monotonic``：start_ms 单调性倒退时 True=fail-fast（缺省 WARN）。
    """
    input_paths: List[str]
    output_dir: str
    mode: str = "sft"
    batch_size: int = 16
    pad_id: int = 0
    template_str: str = ""
    #: PT 的文档边界标记；默认**宽松**（strict=False：非单 token 告警取首个，见 Marker）。
    eos: Marker = field(default_factory=lambda: Marker(strict=False))
    turn_start: Marker = field(default_factory=Marker)
    turn_end: Marker = field(default_factory=Marker)
    assistant_str: str = "assistant"
    batches_per_shard: int = 0
    compression: str = "auto"
    #: 是否落盘**文档级旁路列** ``doc_ids`` / ``loss_weights``（仅 PT/Memmap 数据集）。
    #: 开启后：每个 shard 多出 `.col_doc_ids` / `.col_loss_weights` 两个定宽文件，
    #: 读取端按 manifest schema 自动透出为 DLPack capsule（``doc_ids`` 为 int32，
    #: ``loss_weights`` 为 float16），可用于文档级 loss 加权 / 样本追溯。
    #: 默认 False——零额外磁盘开销（约 +1.5× token 字节量）。
    extra_columns: bool = False
    # ---- vlm 专用（mode="vlm"；pt/sft 传了报错——模式无关参数 loud-fail 原则）----
    #: 各模态专属 token id 区间 ``[(start, end, kind), ...]``（D6；vlm 必填）
    modality_id_ranges: Optional[List[Tuple[int, int, str]]] = None
    #: D7 样本级总视觉预算（None = 行宽 msl，即不额外收紧）
    max_visual_tokens_per_sample: Optional[int] = None
    #: 图像段 label（T3，默认 -100）
    image_label: int = -100
    #: 超预算策略：False=TruncateAndCount（默认），True=HardFail
    visual_budget_hard_fail: bool = False
    # ---- vlm 专用（P3a：audio 流对齐/装箱；pt/sft 传了报错）----
    #: 对齐窗长 W_ms（整数毫秒；默认 2000，对齐终稿 DP1）
    w_ms: int = 2000
    #: 音频窗内 token 率（tokens/s）；None = 未声明（含 audio 段的构建 fail-fast）
    audio_window_token_rate: Optional[int] = None
    #: audio 流 max_seq_len / batch_size / batches_per_shard（None = 同 Vlm 流）
    audio_max_seq_len: Optional[int] = None
    audio_batch_size: Optional[int] = None
    audio_batches_per_shard: Optional[int] = None
    #: INV-2 软校验升级：start_ms 单调性倒退时 True=fail-fast（缺省 WARN）
    audio_strict_monotonic: bool = False
    # ---- vlm 专用（P3b：video 定帧采样确定性；pt/sft 传了报错）----
    #: 采样帧数 N（uniform 定帧；D9 定宽预算，默认 16）
    video_n_frames: int = 16
    #: resize 目标分辨率 (w, h)（以指纹为准、rec 透传；默认 (336, 336)）
    video_resize: Tuple[int, int] = (336, 336)
    #: 采样器版本占位（uniform 策略不消费；seeded 随机采样预留，默认 0）
    video_seed: int = 0
    #: fps 兜底值（MediaMeta.fps 缺失 → WARN + 此值 + 计数入 manifest；默认 30.0）
    video_fps_default: float = 30.0
    #: Vlm 流 shard 字节均衡上限（0 = 不限；默认 512 MiB）
    video_shard_max_bytes: int = 512 * 1024 * 1024
    # ---- vlm 专用（P4：源字段映射；tar/parquet reader 消费，JSONL 不读）----
    #: dict：image_fields/text_fields/audio_fields/video_fields/meta_fields/projection/
    #: timestamp_field；None = 默认映射（贴合 webdataset 惯例）
    field_map: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_mode(self.mode))
        object.__setattr__(self, "compression", normalize_compression(self.compression))
        if int(self.batch_size) < 1:
            raise ValueError(f"BuildConfig.batch_size must be >= 1, got {self.batch_size!r}")
        if int(self.batches_per_shard) < 0:
            raise ValueError(
                f"BuildConfig.batches_per_shard must be >= 0 (0 = auto), got {self.batches_per_shard!r}"
            )
        if int(self.pad_id) < 0:
            raise ValueError(f"BuildConfig.pad_id must be >= 0, got {self.pad_id!r}")
        # 模式无关参数：干预过即报错（text 也算干预——str 只用于推导 id，传了同样不生效）
        _vlm_why = ("VLM encodes text+images inside the modality_encoder callback "
                    "(samples come from the P0 reader, no template rendering)")
        _why = {
            "sft": {"eos": "SFT's supervision boundaries are determined by turn_start/turn_end; "
                           "the Rust DocumentAwarePacker does not consume eos"},
            "pt": {"turn_start": "PT is a plain token stream (GreedyPacker has no notion of dialogue turns)",
                   "turn_end": "PT is a plain token stream (GreedyPacker has no notion of dialogue turns)"},
            "vlm": {"eos": _vlm_why, "turn_start": _vlm_why, "turn_end": _vlm_why},
        }[self.mode]
        for name, why in _why.items():
            if getattr(self, name).intervened:
                raise ValueError(
                    f"mode={self.mode!r} does not use {name} ({why}): remove the argument — "
                    "it would have no effect (fail loudly rather than silently ignore)."
                )
        if self.mode == "vlm":
            if str(self.template_str):
                raise ValueError(
                    f"mode='vlm' does not use template_str ({_vlm_why}): remove the argument "
                    "(fail loudly rather than silently ignore)."
                )
            # D6：vlm 必须声明模态专属 id 区间，否则 token id 校验无从谈起。
            ranges = self.modality_id_ranges
            if not ranges:
                raise ValueError(
                    "mode='vlm' requires modality_id_ranges: declare each modality's exclusive "
                    "token id range as [(start, end, kind), ...] with kind in "
                    "('image', 'audio', 'video') — without it the D6 token-range "
                    "validation cannot be enforced."
                )
            seen: set = set()
            for item in ranges:
                try:
                    start, end, kind = item
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"modality_id_ranges items must be (start, end, kind) tuples, got {item!r}"
                    ) from e
                if kind not in ("image", "audio", "video"):
                    raise ValueError(
                        f"modality_id_ranges kind must be image/audio/video, got {kind!r} "
                        "(text tokens live in the ordinary vocab, no declared range needed)"
                    )
                if not (0 <= start < end):
                    raise ValueError(
                        f"modality_id_ranges range must satisfy 0 <= start < end, got {item!r}"
                    )
                for other_start, other_end, _ in seen:
                    if start < other_end and other_start < end:
                        raise ValueError(
                            f"modality_id_ranges overlap: [{start},{end}) vs "
                            f"[{other_start},{other_end}) (D6: ranges must be pairwise disjoint)"
                        )
                seen.add((start, end, kind))
            if self.max_visual_tokens_per_sample is not None and int(self.max_visual_tokens_per_sample) < 1:
                raise ValueError(
                    f"max_visual_tokens_per_sample must be >= 1 (or None = max_seq_len), "
                    f"got {self.max_visual_tokens_per_sample!r}"
                )
            # P3a：audio 流对齐/装箱参数校验（vlm 内）
            if int(self.w_ms) < 1:
                raise ValueError(f"w_ms must be >= 1 (integer milliseconds), got {self.w_ms!r}")
            if self.audio_window_token_rate is not None and int(self.audio_window_token_rate) < 1:
                raise ValueError(
                    f"audio_window_token_rate must be >= 1 (tokens/s) or None (undeclared), "
                    f"got {self.audio_window_token_rate!r}"
                )
            for f in ("audio_max_seq_len", "audio_batch_size", "audio_batches_per_shard"):
                v = getattr(self, f)
                if v is not None and int(v) < 1:
                    raise ValueError(f"{f} must be >= 1 (or None = same as the Vlm stream), got {v!r}")
            # P3b：video 采样参数校验
            if int(self.video_n_frames) < 1:
                raise ValueError(f"video_n_frames must be >= 1, got {self.video_n_frames!r}")
            w, h = self.video_resize
            if int(w) < 1 or int(h) < 1:
                raise ValueError(f"video_resize must be (w>=1, h>=1), got {self.video_resize!r}")
            if float(self.video_fps_default) <= 0:
                raise ValueError(f"video_fps_default must be > 0, got {self.video_fps_default!r}")
            if int(self.video_shard_max_bytes) < 0:
                raise ValueError(
                    f"video_shard_max_bytes must be >= 0 (0 = unlimited), got {self.video_shard_max_bytes!r}"
                )
            # P4：field_map 结构校验（dict；角色字段为 list[str]，timestamp_field 为 str）
            if self.field_map is not None:
                if not isinstance(self.field_map, dict):
                    raise ValueError(f"field_map must be a dict, got {type(self.field_map).__name__}")
                for k in ("image_fields", "text_fields", "audio_fields", "video_fields",
                          "meta_fields", "projection"):
                    v = self.field_map.get(k)
                    if v is not None and not (isinstance(v, (list, tuple))
                                              and all(isinstance(x, str) for x in v)):
                        raise ValueError(f"field_map[{k!r}] must be list[str], got {v!r}")
                tsf = self.field_map.get("timestamp_field")
                if tsf is not None and not isinstance(tsf, str):
                    raise ValueError(f"field_map['timestamp_field'] must be str, got {tsf!r}")
        elif any(
            getattr(self, f) is not None
            for f in ("modality_id_ranges", "max_visual_tokens_per_sample")
        ) or self.image_label != -100 or self.visual_budget_hard_fail:
            raise ValueError(
                "modality_id_ranges / max_visual_tokens_per_sample / image_label / "
                "visual_budget_hard_fail are vlm-only arguments: mode="
                f"{self.mode!r} does not use them (fail loudly rather than silently ignore)."
            )
        elif (
            self.w_ms != 2000
            or self.audio_window_token_rate is not None
            or self.audio_max_seq_len is not None
            or self.audio_batch_size is not None
            or self.audio_batches_per_shard is not None
            or self.audio_strict_monotonic
        ):
            # P3a：audio 流参数同为 vlm-only（pt/sft 无 audio 流，loud-fail 防静默忽略）
            raise ValueError(
                "w_ms / audio_window_token_rate / audio_max_seq_len / audio_batch_size / "
                "audio_batches_per_shard / audio_strict_monotonic are vlm-only arguments: mode="
                f"{self.mode!r} does not use them (fail loudly rather than silently ignore)."
            )
        elif self.field_map is not None:
            # P4：field_map 同为 vlm-only（pt/sft 无 tar/parquet 源，loud-fail）
            raise ValueError(
                "field_map is a vlm-only argument (consumed by the tar/parquet source "
                f"readers): mode={self.mode!r} does not use it (fail loudly rather than "
                "silently ignore)."
            )
        elif (
            self.video_n_frames != 16
            or tuple(self.video_resize) != (336, 336)
            or self.video_seed != 0
            or self.video_fps_default != 30.0
            or self.video_shard_max_bytes != 512 * 1024 * 1024
        ):
            # P3b：video 参数同为 vlm-only（pt/sft 无 video 流，loud-fail）
            raise ValueError(
                "video_n_frames / video_resize / video_seed / video_fps_default / "
                "video_shard_max_bytes are vlm-only arguments: mode="
                f"{self.mode!r} does not use them (fail loudly rather than silently ignore)."
            )
        if self.mode == "sft" and not str(self.template_str).strip():
            # 空模板会渲染出空文本 → 构建出 0 shard 的空数据集（加载时才报错），
            # 属于"静默产出废数据"，构造期即告警。
            warnings.warn(
                "mode='sft' but template_str is empty: the rendered text would be empty and the build would "
                "produce an empty dataset with 0 shards. Pass a chat template (e.g. "
                "\"{% for m in conversations %}<|im_start|>{{ m.role }}\\n"
                "{{ m.content }}<|im_end|>\\n{% endfor %}\").",
                stacklevel=3,
            )

    @classmethod
    def from_kwargs(cls, kwargs: Dict[str, Any]) -> "BuildConfig":
        """旧扁平签名 → BuildConfig（兼容期 shim；调用方负责发 DeprecationWarning）。

        旧键 `{marker}_str` / `{marker}_id` 合并为 `Marker`；新旧写法混用
        （如同时给 ``eos`` 与 ``eos_str``）直接报错，不静默取舍。
        兼容语义：出现在旧签名里的键一律视为"显式传入"（含显式传默认值）。
        """
        kw = dict(kwargs)
        kw.pop("progress_cb", None)   # 回调不进 config（防御：签名捕获后不应出现）

        def _marker(prefix: str, strict: bool) -> Marker:
            text = kw.pop(f"{prefix}_str", _UNSET)
            ident = kw.pop(f"{prefix}_id", _UNSET)
            return Marker(
                text=None if text is _UNSET else text,
                id=None if ident is _UNSET else ident,
                strict=strict,
            )

        markers: Dict[str, Marker] = {}
        for name, strict in (("eos", False), ("turn_start", True), ("turn_end", True)):
            legacy = _marker(name, strict)
            direct = kw.pop(name, _UNSET)
            if direct is not _UNSET and legacy.intervened:
                raise TypeError(
                    f"build_dataset() got both {name}= and {name}_str/{name}_id= (old and new signatures mixed): "
                    "keep only one."
                )
            markers[name] = legacy if direct is _UNSET else direct

        known = set(cls.__dataclass_fields__) - set(markers)
        unknown = set(kw) - known
        if unknown:
            raise TypeError(
                f"build_dataset() got unknown arguments {sorted(unknown)}: use config=BuildConfig(...) "
                "(recommended) or the legacy flat signature; do not mix the two."
            )
        missing = [k for k in ("input_paths", "output_dir") if k not in kw]
        if missing:
            raise TypeError(f"build_dataset() missing required arguments {missing}")
        return cls(**markers, **kw)


class DataBuilder:
    """
    离线数据集构建器。

    分词器接入（两级加载分派，§3.3）：
    - ``DataBuilder(tokenizer_path=...)``：先按 bbpe_rs 判别（默认路径，Rust 直连，
      零改动）；不是 bbpe 文件时回退到 HF ``tokenizers``（Python 路径）。
    - ``DataBuilder(tokenizer=...)``：任意第三方分词器对象（HF / remote_code /
      tiktoken / 自研），走 Python 路径。
    - 两者**互斥**：同时给出报错。都**不给**也允许（``backend="none"``）：仅供
      ``mode="vlm"`` 使用（文本编码在 modality_encoder 回调内自理，构建不需要
      分词器）；pt/sft 构建在 build_dataset 期显式报错。

    token ID 自动推导（**按模式区分**；模式相关性见 `BuildConfig`）：
    - **PT**：`eos` 未显式传入时自动从默认文本（`<|endoftext|>`）编码推导
      （GreedyPacker 的文档结束符与 padding 依赖它，必需）；
      `turn_start` / `turn_end` 在 PT 下**不接受**（传了报错）。
    - **SFT**：`turn_start` / `turn_end` 自动推导并强校验"恰好 1 个 token"；
      **SFT 不使用 `eos`**（监督边界由 turn 标记决定，Rust DocumentAwarePacker 不消费）
      ——传了报错，不再静默忽略。
    - 显式传参始终优先（`Marker.id`）；bbpe 路径的自动推导依赖 bbpe_tokenizer 可导入
      （不可用时须显式传 ID）；Python 路径直接用当前 adapter 编码。
    """

    def __init__(
        self,
        tokenizer_path: Optional[str] = None,
        max_seq_len: int = 8192,
        tokenizer: Optional[Any] = None,
    ):
        if tokenizer_path is not None and tokenizer is not None:
            raise ValueError(
                "exactly one of tokenizer_path and tokenizer must be given (mutually exclusive): "
                "tokenizer_path uses the two-stage 'bbpe first + HF fallback' loading; tokenizer accepts "
                "any third-party tokenizer object."
            )
        if tokenizer_path is not None and not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer path does not exist: {tokenizer_path!r}")

        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        # 都不给：backend="none"，仅 vlm 构建可用（文本编码在 modality encoder 内自理）；
        # pt/sft 在 build_dataset 期报错（构造期不知道 mode，无法更早拒绝）。
        self._builder = RustDataBuilder(tokenizer_path or "", max_seq_len)
        self._tok = None  # 懒加载的 BBTokenizer（可能为 None 表示不可用）
        self._adapter: Optional[Any] = None
        self.backend: str = "bbpe"  # "bbpe"（Rust 直连）| "python"（第三方回调）| "none"（仅 vlm）
        if tokenizer_path is None and tokenizer is None:
            # 都不给：backend="none"，仅 vlm 构建可用（文本编码在 modality encoder
            # 内自理）；pt/sft 在 build_dataset 期据此显式报错。
            self.backend = "none"
            return
        self._resolve_backend(tokenizer)

    # ------------------------------------------------------------------
    # 分词器接入
    # ------------------------------------------------------------------
    def _resolve_backend(self, tokenizer: Optional[Any]) -> None:
        """两级加载分派（§3.3）。失败时把**两级失败原因并列**抛出（T6）。"""
        if tokenizer is not None:
            self.backend = "python"
            self._adapter = adapt_tokenizer(tokenizer)
            return
        # 一级快判（零成本）不中的，交给 Rust **真实加载器**裁决：v0 旧格式没有
        # `format` 字段、目录形态等边角都能被判对（F4 已确认非 bbpe 文件给出干净失败）。
        if self._looks_like_bbpe_model(self.tokenizer_path) or (
            self.tokenizer_path and self._builder.probe_bbpe()
        ):
            self.backend = "bbpe"  # 默认路径：Rust 直连，字面零改动（D1）
            return
        # 第二级：HF tokenizers（含目录内的 tokenizer.json）
        try:
            self._adapter = load_tokenizer_file(self.tokenizer_path)
        except Exception as e:
            raise ValueError(
                f"cannot load tokenizer {self.tokenizer_path!r}: both loading stages failed.\n"
                f"  ① bbpe_rs (default path): the Rust loader rejected the file (not a bbpe model format)\n"
                f"  ② tokenizers/transformers (third-party path): {type(e).__name__}: {e}\n"
                "Remedy: point bbpe-trained artifacts at the model file (or its directory); point standard HF "
                "tokenizers at tokenizer.json (or its directory); for trust_remote_code / "
                "tiktoken / custom tokenizers use DataBuilder(tokenizer=...) instead."
            ) from e
        self.backend = "python"

    @classmethod
    def _looks_like_bbpe_model(cls, path: Optional[str]) -> bool:
        """一级快判：文件头部是否含 bbpe v1 的 `format` 常量（serde 按字段序写出，
        `format` 是第一个字段，64KB 足够覆盖）。零成本，命中就不必动用真实加载器。"""
        path = cls._resolve_tokenizer_file(path)
        if not path or not os.path.isfile(path):
            return False
        try:
            with open(path, "rb") as f:
                return _BBPE_FORMAT_MAGIC in f.read(_BBPE_SNIFF_BYTES)
        except OSError:
            return False

    @staticmethod
    def _resolve_tokenizer_file(path: Optional[str]) -> Optional[str]:
        """目录 → 目录下的 `tokenizer.json`（与 Rust 侧同样的解析规则）。"""
        if path and os.path.isdir(path):
            return os.path.join(path, "tokenizer.json")
        return path

    @property
    def adapter(self) -> Optional[Any]:
        """Python 路径的 `BatchTokenizer` 适配器；bbpe 路径为 None。"""
        return self._adapter

    def _callback(self, mode: str) -> Callable[[List[str]], Any]:
        """把 `BatchTokenizer` 适配为 Rust 侧要的**单一 callable**（模式烘焙进去，§4.3）。

        返回 `(ids_batch, tokens_batch|None)`：SFT 给 tokens，PT 给 None。
        """
        adapter = self._adapter
        if mode == "sft":
            def encode(texts: List[str]):
                return adapter.encode_sft_batch(texts)
        else:
            def encode(texts: List[str]):
                return adapter.encode_ordinary_batch(texts), None
        return encode

    def _cancel(self):
        self._builder.cancel()

    def cancel(self):
        """请求取消正在进行的构建。"""
        self._builder.cancel()

    def get_progress(self) -> dict:
        """取回当前构建进度快照。dict 含 bytes_read/docs_read/docs_tokenized/
        shards_written/bytes_written/elapsed_secs。构建未开始时返回空 dict。"""
        return dict(self._builder.get_progress())

    def _get_tok(self):
        if self._tok is None:
            try:
                from bbpe_tokenizer import BBTokenizer  # 软依赖
                self._tok = BBTokenizer.from_pretrained(self.tokenizer_path)
            except Exception:
                self._tok = False
        return self._tok if self._tok is not False else None

    def _derive_id(self, text: str, explicit: Optional[int]) -> int:
        if explicit is not None:
            return explicit
        # 优先走 Rust：bbpe 路径手上有真实分词器，推导必然与构建期一致，
        # 不再依赖 bbpe_tokenizer 软依赖（方案 §3.5 的可选增强）。
        got = self._builder.probe_special_ids(text)
        if got:
            if len(got) != 1:
                # 非单 token：这里只告警取首个（eos 的宽松口径），
                # turn 标记的严格校验在 Rust 构建入口（D8）。
                warnings.warn(
                    f"encoding {text!r} produced {len(got)} tokens {got}; taking the first. "
                    f"Verify it is a single-token special symbol.",
                    stacklevel=2,
                )
            return got[0]
        tok = self._get_tok()
        if tok is None:
            raise ValueError(
                f"cannot auto-derive the token ID (special symbol not found in the tokenizer and "
                f"bbpe_tokenizer unavailable). Pass the ID explicitly. Text: {text!r}"
            )
        ids = tok.encode(text, allowed_special=["all"]).ids
        if len(ids) != 1:
            warnings.warn(
                f"encoding {text!r} produced {len(ids)} tokens {ids}; taking the first. Verify it is a single-token special symbol.",
                stacklevel=2,
            )
        return ids[0]

    def _resolve_marker_id(self, text: str, explicit: Optional[int], label: str,
                           *, strict: bool = True) -> int:
        """推导标记 id：`strict=True` **强校验**（turn 标记，§4.4 / D8），
        `strict=False` 走 eos 式宽松口径（非单 token 告警取首个；分级是有意的——
        turn 损坏监督标签，eos 只影响文档边界）。

        Python 路径在此校验（手上有 adapter）；bbpe 路径交给 Rust 入口用真实
        tokenizer 校验（等效且更早失败）。任一侧发现"标记不是恰好 1 个 token"或
        "与显式 id 不一致"都直接报错 —— SFT 标签正确性优先于构建能跑完。
        """
        if not strict:
            return self._derive_id(text, explicit)
        if self.backend != "python":
            return self._derive_id(text, explicit)

        ids, tokens = self._probe_encode_sft(text)
        got = ids[0] if ids else []
        shown = tokens[0] if tokens else None
        if len(got) != 1:
            raise ValueError(
                f"{label} marker {text!r} encodes to {len(got)} tokens "
                f"({shown if shown is not None else got}); exactly 1 is required: "
                "multi-token markers corrupt assistant-span detection and the supervision labels. "
                "Check that the marker string exactly matches a tokenizer special symbol."
            )
        if explicit is not None and got[0] != explicit:
            raise ValueError(
                f"{label} marker {text!r} encodes to id {got[0]} (token {shown!r}), which does not match the "
                f"explicit {label}_id={explicit}. Check the value, or omit the argument to auto-derive."
            )
        return got[0]

    def _probe_encode_sft(self, text: str):
        """调一次 `encode_sft_batch` 并校验**返回结构**。

        结构错误若放任下去，只会冒出一句 `ValueError: not enough values to unpack`
        之类的裸异常，用户根本不知道是自己分词器的返回值写错了。这里翻译成
        "契约 + 实际类型"的可操作提示（构建期防呆，方案 §4.3）。
        """
        name = type(self.adapter).__name__
        try:
            result = self.adapter.encode_sft_batch([text])
            ids_batch, tokens_batch = result
        except Exception as e:
            raise TypeError(
                f"the return value of tokenizer {name}.encode_sft_batch violates the contract: it must be "
                f"a (ids_batch, tokens_batch) 2-tuple, but it cannot be unpacked as one "
                f"({type(e).__name__}: {e})."
            ) from e
        if not isinstance(ids_batch, (list, tuple)):
            raise TypeError(
                f"ids_batch returned by tokenizer {name}.encode_sft_batch should be list[list[int]], "
                f"got {type(ids_batch).__name__}."
            )
        return ids_batch, tokens_batch

    def _resolve_eos_id(self, eos_str: str, explicit: Optional[int], *, strict: bool = False) -> int:
        """eos_id 推导。

        默认**宽松口径**（非掩码关键路径，§4.4：多 token 告警取首个）；
        `Marker(strict=True)` 时按 turn 标记同一严格口径校验（必须恰好 1 个 token），
        使 `Marker.strict` 对 eos 不再是死旋钮。
        """
        if explicit is not None:
            return explicit
        if strict:
            if self.backend == "python":
                ids, tokens = self._probe_encode_sft(eos_str)
                got = ids[0] if ids else []
                if len(got) != 1:
                    shown = tokens[0] if tokens else None
                    raise ValueError(
                        f"eos marker {eos_str!r} encodes to {len(got)} tokens "
                        f"({shown if shown is not None else got}); strict=True requires exactly 1: "
                        "check that the string exactly matches a tokenizer special symbol, "
                        "or use Marker(strict=False) (lenient: warn and take the first)."
                    )
                return int(got[0])
            got = list(self._builder.probe_special_ids(eos_str) or [])
            if len(got) != 1:
                raise ValueError(
                    f"eos marker {eos_str!r} encodes to {len(got)} tokens {got} "
                    "(strict=True requires exactly 1): if the symbol is not in the vocab, pass eos_id explicitly, "
                    "or use Marker(strict=False) (lenient)."
                )
            return int(got[0])
        if self.backend != "python":
            return self._derive_id(eos_str, None)
        ids, _ = self._probe_encode_sft(eos_str)
        if not ids or not ids[0]:
            raise ValueError(f"eos_str {eos_str!r} encodes to nothing; cannot derive eos_id — pass eos_id explicitly.")
        return ids[0][0]

    def _fingerprint_kwargs(self) -> dict:
        """manifest 分词器指纹（§6）。Python 路径由 adapter 报告标识与词表大小。"""
        if self.backend != "python":
            return {}
        adapter = self.adapter
        identity = getattr(adapter, "identity", None) or type(adapter).__name__ or "opaque"
        vocab = getattr(adapter, "vocab_size", None)
        return {
            "tokenizer_identity": identity,
            "tokenizer_vocab_size": int(vocab) if vocab else None,
        }

    def build_dataset(
        self,
        *args: Any,
        config: Optional[BuildConfig] = None,
        progress_cb: Optional[Callable[[dict], None]] = None,
        modality_encoder: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        **legacy: Any,
    ):
        """Build a dataset from JSONL sources and write it into ``output_dir``.

        Recommended form::

            cfg = BuildConfig(input_paths=["a.jsonl"], output_dir="out/sft", mode="sft",
                              template_str=TEMPLATE, batch_size=16)
            summary = builder.build_dataset(cfg, progress_cb=print)

        Args:
            config: A :class:`BuildConfig` value object (preferred). Omit it to use
                the legacy flat keyword form, which still works but emits a
                ``DeprecationWarning``.
            progress_cb: Optional callable invoked roughly every 100 ms with a
                progress ``dict``.
            modality_encoder: Optional callback mapping raw multimodal records to
                encoded ones (VLM builds).

        Returns:
            dict: build summary — ``output_dir`` / ``mode`` / ``batch_size`` /
            ``compression`` etc. plus manifest-derived ``manifest``,
            ``total_batches``, ``total_shards``, ``total_tokens``, ``max_seq_len``
            (``manifest: None`` plus ``manifest_error`` if reading it failed).

        Raises:
            ValueError: invalid ``mode``, a mode-irrelevant argument was supplied
                (e.g. ``eos=`` with ``mode="sft"``), or the build produced no shards.
            OSError: I/O failure while reading inputs or writing shards.

        Note:
            ``batch_size`` and ``max_seq_len`` are frozen into the shards; loaders
            must be constructed with the same values or they raise immediately.

        全能构建引擎（**推荐**传 `config=BuildConfig(...)`；旧扁平签名仍可用，带弃用告警）。

        ```python
        cfg = BuildConfig(input_paths=["a.jsonl"], output_dir="out/sft", mode="sft",
                          template_str=TEMPLATE, batch_size=16)
        builder.build_dataset(cfg, progress_cb=print)
        ```

        vlm 多模态构建（P2）::

            from megadata.modality_adapter import make_vlm_encode_callback
            cfg = BuildConfig(input_paths=["imgs.jsonl"], output_dir="out/vlm", mode="vlm",
                              batch_size=2, modality_id_ranges=[(32000, 32100, "image")],
                              max_visual_tokens_per_sample=256)
            cb = make_vlm_encode_callback(tokenizer=my_tok, image_range_start=32000)
            DataBuilder(max_seq_len=1024).build_dataset(cfg, modality_encoder=cb)

        `modality_encoder`：``(samples: list[dict]) -> list[dict]``（长度不变），
        契约与并行化责任见 `megadata.modality_adapter` 模块 docstring；
        运行期回调，不属于 `BuildConfig`（同 `progress_cb`）。vlm 模式必填，
        pt/sft 传入报错（Rust 入口防护）。

        - ChatML (Qwen等): ``turn_start=Marker("<|im_start|>")``
        - Llama-3: ``turn_start=Marker("<|start_header_id|>")``,
          ``turn_end=Marker("<|eot_id|>")``
        - Gemma: ``turn_start=Marker("<start_of_turn>")``, ``assistant_str="model"``

        token ID 未显式传入时自动从 tokenizer 推导（推荐）。
        `progress_cb`: 可选进度回调 ``cb(progress_dict)``（含
        bytes_read/docs_read/docs_tokenized/shards_written/bytes_written/elapsed_secs）；
        它是运行期副作用，**不属于** `BuildConfig`。

        shard 聚合与压缩（借鉴 streaming/MDS 的控制面设计）：
        - ``batches_per_shard``: 单个 shard 文件聚合的 batch 数。0 = 自动
          （约 128MB/shard）。注意 shard 数仍须 >= world_size（加载端强校验，
          否则高位 rank 分不到 shard）。
        - ``compression``: ``"auto" | "zstd" | "lz4" | "none"``：
            * ``"auto"``（默认）：PT → "zstd"（per-batch zstd frame，Zstd-MDS 格式，
              保留 O(1) 随机寻址）；SFT → 不压缩（Arrow IPC 原生压缩需显式指定）。
            * ``"zstd"`` / ``"lz4"``：显式指定算法（SFT 走 Arrow IPC 原生压缩，
              读取端透明）。
            * ``"none"``：显式关闭压缩。
          ⚠️ ``None`` 不再是合法值（v1/v2 里 ``None`` 与 ``"none"`` 的视觉陷阱已消灭）。

          默认值的依据（bench/compression_bench.py 实测，真实中文网页语料
          + 151k 词表 BPE 预分词，14.6M tokens）：zstd:3 压缩到 42.2%
          （节省 57.8% 磁盘/对象存储流量），解压吞吐 629MB/s，随机访问仅
          +103µs/batch——三项判据（≤75% / ≥500MB/s / ≤1ms）全部满足。
          高 level 收益递减（zstd:7 → 39.9% 但压缩吞吐 -70%），不必调高。

        返回**构建摘要** dict（P2-10）：`output_dir/mode/batch_size/compression/...`
        + 从落盘 manifest 汇总的 `manifest/total_batches/total_shards/total_tokens/
        max_seq_len`（manifest 读取失败时为 `manifest: None` + `manifest_error`）。
        """
        # `config` 允许两种写法：`build_dataset(cfg)` 与 `build_dataset(config=cfg)`
        if args and isinstance(args[0], BuildConfig):
            if config is not None:
                raise TypeError("build_dataset() got multiple config=BuildConfig (duplicated via positional and keyword)")
            config, args = args[0], args[1:]

        if config is None:
            # ---- 旧扁平签名兼容路径（弃用）----
            for name, val in zip(("input_paths", "output_dir"), args):
                if name in legacy:
                    raise TypeError(f"build_dataset() got duplicate argument {name!r}")
                legacy[name] = val
            if len(args) > 2:
                raise TypeError(
                    f"build_dataset() takes at most 2 positional arguments (input_paths, output_dir), got {len(args)}"
                )
            if not legacy:
                raise TypeError(
                    "build_dataset() requires config=BuildConfig(...) (recommended) "
                    "or legacy arguments (input_paths=..., output_dir=...)"
                )
            warnings.warn(
                "build_dataset's old flat signature is deprecated: use config=BuildConfig(...) instead "
                "(value object with construction-time validation; mode-irrelevant arguments are rejected "
                "explicitly instead of silently ignored).",
                DeprecationWarning, stacklevel=2,
            )
            config = BuildConfig.from_kwargs(legacy)
        elif args or legacy:
            raise TypeError(
                "build_dataset() cannot take both config= and legacy arguments: the configuration source "
                "must be unique (to avoid silent ambiguity about which one takes effect)."
            )

        mode = config.mode

        # compression：值对象已归一化（"auto" → 具体算法；"none" → None 关闭）
        comp = config.compression
        if comp == "none":
            compression = None
        elif comp == "auto":
            compression = "zstd" if mode == "pt" else None
        else:
            compression = comp

        # 按模式推导所需 ID（未显式传入时自动从**当前生效**分词器编码）。
        # 模式无关参数已由 BuildConfig 在构造期拒绝（传了报错，不再静默忽略/填 0）：
        # - SFT 不使用 eos（监督边界由 turn 标记决定，DocumentAwarePacker 不消费）；
        # - PT 不使用 turn_start/turn_end（GreedyPacker 无对话轮次概念）。
        # 下面写进 Rust 签名的 0 只是**占位值**，不是"默认值"——它没有任何语义，
        # 因为对应的参数在另一模式下必然为占位。
        if mode == "vlm":
            # vlm：文本+视觉编码全在 modality_encoder 回调内；eos/turn id 是
            # pt/sft 概念（BuildConfig 已拒绝干预），传 0 占位；不需要分词器。
            if modality_encoder is None:
                raise ValueError(
                    "mode='vlm' requires modality_encoder: a callback mapping "
                    "list[sample dict] -> list[encoded dict] (length preserving). "
                    "See megadata.modality_adapter.make_vlm_encode_callback for the "
                    "reference implementation and the parallelism contract."
                )
            eos_id = turn_start_id = turn_end_id = 0   # 占位：vlm 下无语义
            turn_start_str = turn_end_str = ""
        elif mode == "pt":
            if self.backend == "none":
                raise ValueError(
                    "DataBuilder was constructed without a tokenizer (backend='none'): "
                    "pt/sft builds need one. Use mode='vlm' with a modality_encoder, "
                    "or construct DataBuilder(tokenizer_path=... / tokenizer=...)."
                )
            eos_str = config.eos.text if config.eos.text is not None else _DEFAULT_EOS
            eos_id = self._resolve_eos_id(eos_str, config.eos.id, strict=config.eos.strict)
            turn_start_id = 0     # 占位：PT 下无语义（BuildConfig 已拒绝用户干预）
            turn_end_id = 0       # 占位
            turn_start_str = _DEFAULT_TURN_START
            turn_end_str = _DEFAULT_TURN_END
        else:  # sft
            if self.backend == "none":
                raise ValueError(
                    "DataBuilder was constructed without a tokenizer (backend='none'): "
                    "pt/sft builds need one. Use mode='vlm' with a modality_encoder, "
                    "or construct DataBuilder(tokenizer_path=... / tokenizer=...)."
                )
            eos_id = 0            # 占位：SFT 下无语义（BuildConfig 已拒绝用户干预）
            turn_start_str = (
                config.turn_start.text if config.turn_start.text is not None else _DEFAULT_TURN_START
            )
            turn_end_str = (
                config.turn_end.text if config.turn_end.text is not None else _DEFAULT_TURN_END
            )
            turn_start_id = self._resolve_marker_id(
                turn_start_str, config.turn_start.id, "turn_start",
                strict=config.turn_start.strict,
            )
            turn_end_id = self._resolve_marker_id(
                turn_end_str, config.turn_end.id, "turn_end",
                strict=config.turn_end.strict,
            )

        _register(self)
        kwargs: Dict[str, Any] = {}
        if mode == "vlm":
            # P2 (mm)：vlm 参数透传 Rust（modality_encoder + D6 区间 + D7 预算/T3 标签）。
            # backend="none" 时不传指纹（D6 vocab 检查跳过，只查区间互斥）；
            # backend="python" 时把 adapter 的 identity/vocab_size 带给 Rust（D6 全量检查）。
            kwargs["modality_encoder"] = modality_encoder
            kwargs["modality_id_ranges"] = [tuple(r) for r in config.modality_id_ranges]
            kwargs["max_visual_tokens_per_sample"] = (
                None if config.max_visual_tokens_per_sample is None
                else int(config.max_visual_tokens_per_sample)
            )
            kwargs["image_label"] = int(config.image_label)
            kwargs["visual_budget_hard_fail"] = bool(config.visual_budget_hard_fail)
            # P3a (mm)：audio 流对齐/装箱参数透传（None → Rust 0 = 同 Vlm 流 /
            # 未声明；含 audio 段的构建缺 window_token_rate 由 Rust fail-fast）
            kwargs["w_ms"] = int(config.w_ms)
            kwargs["audio_window_token_rate"] = (
                0 if config.audio_window_token_rate is None else int(config.audio_window_token_rate)
            )
            kwargs["audio_max_seq_len"] = (
                0 if config.audio_max_seq_len is None else int(config.audio_max_seq_len)
            )
            kwargs["audio_batch_size"] = (
                0 if config.audio_batch_size is None else int(config.audio_batch_size)
            )
            kwargs["audio_batches_per_shard"] = (
                0 if config.audio_batches_per_shard is None else int(config.audio_batches_per_shard)
            )
            kwargs["audio_strict_monotonic"] = bool(config.audio_strict_monotonic)
            # P3b (mm)：video 定帧采样/确定性参数透传
            kwargs["video_n_frames"] = int(config.video_n_frames)
            kwargs["video_resize"] = (int(config.video_resize[0]), int(config.video_resize[1]))
            kwargs["video_seed"] = int(config.video_seed)
            kwargs["video_fps_default"] = float(config.video_fps_default)
            kwargs["video_shard_max_bytes"] = int(config.video_shard_max_bytes)
            # P4 (mm)：源字段映射（tar/parquet 源消费；None = Rust 默认映射）
            kwargs["field_map"] = config.field_map
            if self.backend == "python":
                kwargs.update(self._fingerprint_kwargs())
        else:
            # Python 路径：把适配器烘焙成单一 callback 交给 Rust（§4.3）；bbpe 路径不传，
            # 走现有 Rust 直连管线（D1 零改动）。
            if self.backend == "python":
                kwargs["batch_encoder"] = self._callback(mode)
                kwargs.update(self._fingerprint_kwargs())

        self._builder.build_dataset(
            list(config.input_paths), config.output_dir, mode, config.template_str,
            int(config.batch_size), int(config.pad_id),
            eos_id, turn_start_id, turn_end_id,
            turn_start_str, turn_end_str, config.assistant_str,
            int(config.batches_per_shard), compression,
            progress_cb,
            extra_columns=bool(config.extra_columns),
            **kwargs,
        )

        # P2-10：返回**构建摘要**（此前无返回值，调用方只能自己再去读 index.json）。
        # 摘要从落盘 manifest 汇总（构建成功则 index.json 必在；读取失败不 raise，
        # 以 manifest_error 字段披露——构建本身已成功，摘要缺字段不该让调用方拿到异常）。
        summary: Dict[str, Any] = {
            "output_dir": config.output_dir,
            "mode": mode,
            "batch_size": int(config.batch_size),
            "batches_per_shard": int(config.batches_per_shard),
            "compression": compression,
            "input_paths": list(config.input_paths),
        }
        manifest_path = os.path.join(config.output_dir, "index.json")
        try:
            manifest = json.loads(read_manifest_json(manifest_path))
            stats = manifest.get("global_stats", {})
            summary.update({
                "manifest": manifest_path,
                "total_batches": int(stats.get("total_batches") or 0),
                "total_shards": int(stats.get("total_shards") or 0),
                "total_tokens": int(stats.get("total_tokens") or 0),
                "max_seq_len": int(manifest.get("max_seq_len") or 0),
            })
        except Exception as e:
            summary.update({"manifest": None, "manifest_error": f"{type(e).__name__}: {e}"})
        return summary
