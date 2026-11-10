# python/megadata/modality_adapter.py
"""多模态 P2：vlm 模态编码回调的参考实现（Rust `PyModalityEncoder` 契约的 Python 侧）。

回调数据契约（与 Rust `sample_to_dict` / `encoded_sample_from_dict` 一一对应）：

- 输入 ``samples: list[dict]``，每项 = 一个 Sample::

      {"sample_id": int, "source_ref": str|None, "payload": bytes|None,
       "codec": str|None,
       "segments": [{"kind": "text"|"image"|..., "order": int, "len_hint": int|None,
                     "supervise": "on"|"off"|"weight",
                     "text": str|None,          # Text 段内容（P2：vlm 编码数据源）
                     "bytes": bytes|None,       # 段级压缩字节（统一取用：单模态样本
                     "codec": str|None}],       # 的媒体段也透出样本级 payload 的一份）
       "media_meta": {"width","height","sample_rate","channels","duration_sec","fps",
                      "n_frames_total"} | None,   # 各键 int/float|None
       "error_class": str|None}

  **图像字节统一读 ``segments[].bytes``**（无需区分单模态/交错两种形态）；
  样本级 ``payload`` 是单模态样本的快捷方式（两者内容相同）。

- 输出 ``list[dict]``（**长度必须等于输入**），每项 = 一个 EncodedSample::

      {"sample_id": int, "token_seq": list[int],
       "segment_table": [{"kind","order","token_start","token_end",
                          "media_record_id",   # 媒体段必填（指向 media_refs 下标）；文本段 None
                          "loss_weight"}],     # 0..=255，255=1.0
       "labels": list[int],                    # len == len(token_seq)
       "loss_weights": bytes|None,             # f16 LE，2*len(token_seq)；None=packer 默认
       "stats": {"text_tokens","image_tokens","audio_tokens","video_tokens",
                 "images","audio_clips","video_clips"},
       "media_refs": [{"kind","seg_order","bytes","grid_h","grid_w","flags"}]}

GIL / 并行化契约（**必读**，Rust 侧红线）：

- Rust 侧**单点编排、逐 chunk 调用**本回调（唯一持有 GIL 的编排线程），**不做**
  工作线程 fan-out——工作窃取 + GIL 等待 = 死锁（pyo3 官方警告）。
- 因此**并行化的责任在回调实现者**：
  - 线程池：有效。PIL/numpy 的 decode/resize 在 C 层释放 GIL，多线程真正并行；
  - 多进程：有效。每进程独立 GIL；注意 chunk 数据要可 pickle（本参考实现返回
    纯 bytes/int，天然可 pickle）。
- **禁止**在回调里再起操作 Rust 对象的线程（`DataBuilder` / 构建进度等都不是
  线程安全的），也禁止在回调里调用 `builder.build_dataset` / `builder.cancel()`。

grid（视觉 token 网格）决定方式（D7 一期固定 patch 网格）：

- ``tokens = grid_h × grid_w``；placeholder id 取自声明的 image 专属 id 区间
  （``make_vlm_encode_callback(image_range_start=...)``，区间与 id_ranges 声明一致）；
- 默认 ``grid_fn``：``ceil(width/patch) × ceil(height/patch)``，尺寸来自样本的
  ``media_meta.width/height``（JSONL ``image_meta`` 字段带出）。**缺失即报错**
  （不静默猜尺寸——错误 grid 会让 placeholder 数与真实视觉内容脱钩）；
  需要真实解码（如 PIL resize 后按 patch 重算）时传入自定义 ``grid_fn``。
- ⚠️ resize/patch 参数属于 manifest 指纹范围（P3b 确定性）：一期请把
  ``patch_size`` 与 ``grid_fn`` 的行为视为数据集配置的一部分，改动它们 =
  重新构建数据集。

不依赖分词器：``tokenizer=None`` 时文本段用**恒等映射**（逐字符 ``ord``），
仅供测试/最小示例；真实构建请传入实现了 ``encode(text) -> list[int]`` 的
分词器对象。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

#: EncodedSample.stats 的全部键（缺失键 Rust 侧按 0 处理；这里显式补全）
_STATS_KEYS = ("text_tokens", "image_tokens", "audio_tokens", "video_tokens",
               "images", "audio_clips", "video_clips")

#: 图像段 loss 权重定点数（0..=255，255=1.0）。图像段不参与 loss（packer 还会
#: 强制 label=image_label），这里显式给 0 与 VLM 标签语义一致。
_WEIGHT_ON = 255
_WEIGHT_OFF = 0

# P3a/P3b：caption 前置 Text 段的新 order 基址——媒体段保持源侧 segment_order
# （demux 的 (sample_id, segment_order) 源侧查找契约），caption Text 取
# 900 起的不冲突值（SegmentEntry.order 是不透明标识符，无需 0 起连续）。
_CAPTION_ORDER_BASE = 900

# P3a：audio 段缺 duration_sec 时的兜底 token 数（pipeline 回退 token 隐含
# 时长切窗，自洽；与测试缺省值一致）。
_AUDIO_FALLBACK_TOKENS = 100


def _default_grid(payload: Optional[bytes], media_meta: Optional[Dict[str, Any]],
                  patch_size: int) -> Tuple[int, int]:
    """默认网格：``ceil(w/patch) × ceil(h/patch)``（media_meta 驱动，零图像解码）。"""
    w = (media_meta or {}).get("width")
    h = (media_meta or {}).get("height")
    if not w or not h:
        raise ValueError(
            "image sample has no width/height in media_meta (JSONL image_meta missing?): "
            "the default grid_fn refuses to guess the visual token count (a wrong grid "
            "silently decouples placeholders from pixels). Provide image_meta "
            '{"width": W, "height": H} or pass a custom grid_fn.'
        )
    return math.ceil(int(w) / patch_size), math.ceil(int(h) / patch_size)


def _identity_text_tokens(text: str) -> List[int]:
    """恒等映射：每字符一个 token（ord）。仅用于测试/示例（不依赖分词器）。"""
    return [ord(c) for c in text]


def validate_encoded_sample(sample: Dict[str, Any]) -> None:
    """EncodedSample 四条不变式的 Python 侧自查（返回前快速失败，报错更友好）。

    Rust 侧（``EncodedSample::validate``）会再次校验并硬失败——这里提前拦截，
    错误信息带 Python 侧上下文，省一次跨语言往返：

    1. segment_table 按 token_start 排序后**连续覆盖**整个 token_seq（无重叠/空洞）；
    2. ``Σ segment.token_len == len(token_seq)``；
    3. ``len(labels) == len(token_seq)``；
    4. ``loss_weights`` 为 None 或长度 = ``2 * len(token_seq)``（f16 LE 交错对）。
    """
    tok = sample.get("token_seq")
    table = sample.get("segment_table")
    labels = sample.get("labels")
    if not isinstance(tok, list) or not isinstance(table, list):
        raise ValueError("token_seq / segment_table must be lists")
    if labels is None or not isinstance(labels, list):
        raise ValueError("labels must be a list")
    sid = sample.get("sample_id")

    expect = 0
    prev_end = 0
    for i, seg in enumerate(table):
        start, end = seg.get("token_start"), seg.get("token_end")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"sample {sid}: segment {i} has bad token range [{start}, {end})")
        if start != prev_end:
            raise ValueError(
                f"sample {sid}: segment {i} starts at {start}, expected {prev_end} "
                "(segment_table must contiguously cover token_seq in order)"
            )
        prev_end = end
        expect += end - start
    if prev_end != len(tok):
        raise ValueError(
            f"sample {sid}: segments cover {prev_end} tokens but token_seq has {len(tok)}"
        )
    if len(labels) != len(tok):
        raise ValueError(f"sample {sid}: labels len {len(labels)} != token_seq len {len(tok)}")
    lw = sample.get("loss_weights")
    if lw is not None and len(lw) != 2 * len(tok):
        raise ValueError(
            f"sample {sid}: loss_weights len {len(lw)} != 2*token_seq len {2 * len(tok)} (f16 LE pairs)"
        )


def sample_frame_indices(total: int, n_frames: int, sid: int) -> List[int]:
    """P3b：uniform 定帧采样——linspace 均匀间隔（确定性：同 total + 同
    n_frames → 同索引；D9 "uniform 定帧采样 → 固定帧数"的主流做法）。

    边界钉死：``n_frames > total`` → fail-fast（不静默产生重复帧索引——
    linspace 在 n>total 时会重复）；``total <= 0`` → fail-fast。
    seed 不参与帧索引（uniform 策略下索引由 (total, n_frames) 唯一确定；
    指纹保留 seed 键作采样器版本占位，P3b §6.1）。
    """
    if n_frames > total:
        raise ValueError(
            f"sample {sid}: video_n_frames={n_frames} exceeds total frames={total} "
            "(fail-fast; do not silently duplicate frame indices)"
        )
    if total <= 0:
        raise ValueError(f"sample {sid}: video has non-positive total frames ({total})")
    import numpy as np
    return np.linspace(0, total - 1, n_frames, dtype=int).tolist()


def make_vlm_encode_callback(
    tokenizer: Optional[Any] = None,
    *,
    image_range_start: int = 32000,
    patch_size: int = 16,
    grid_fn: Optional[Callable[[Optional[bytes], Optional[Dict[str, Any]]], Tuple[int, int]]] = None,
    audio_range_start: Optional[int] = None,
    audio_token_rate: float = 50.0,
    video_range_start: Optional[int] = None,
    video_n_frames: int = 16,
    video_resize: Tuple[int, int] = (336, 336),
    video_seed: int = 0,
    video_fps_default: float = 30.0,
    frame_fn: Optional[Callable[[bytes, List[int]], List[bytes]]] = None,
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """构造符合 Rust ``PyModalityEncoder`` 契约的 vlm 编码回调（参考实现）。

    参数：
    - ``tokenizer``：实现 ``encode(text) -> list[int]`` 的分词器对象（HF fast /
      bbpe wrapper 等）。``None`` → 文本段恒等映射（``ord``），仅测试/示例用。
    - ``image_range_start``：image 专属 id 区间的**起始**（与 ``modality_id_ranges``
      里声明的 image 区间 start 一致）；placeholder id 依 grid 顺序递增。
    - ``patch_size``：D7 一期固定 patch 网格的边长（默认 grid_fn 使用）。
    - ``grid_fn``：``(payload, media_meta) -> (grid_h, grid_w)``；默认从
      media_meta 尺寸按 patch 上取整（零图像解码）。需要真实解码（PIL resize /
      动态分辨率）时自定义——**并行化责任在实现者**（见模块 docstring）。
    - ``audio_range_start``：P3a audio 专属 id 区间起始；``None`` → audio 段
      出现即报错。``audio_token_rate``：audio token 率（tokens/s，**须与构建的
      ``audio_window_token_rate`` 一致**——pipeline 按它切窗）。
    - ``video_range_start``：P3b video 专属 id 区间起始（与 modality_id_ranges 的
      video 区间 start 一致）；``None`` → video 段出现即报错（须显式配置区间）。
    - ``video_n_frames`` / ``video_resize`` / ``video_seed``：P3b 定帧采样参数
      （uniform linspace；seed 为采样器版本占位，不参与帧索引）。
    - ``video_fps_default``：``MediaMeta.fps`` 缺失时的兜底值（限频 WARN +
      fps_milli 写此值；计数见 manifest ``video_fps_defaulted_clips``）。
    - ``frame_fn``：``(payload, indices) -> list[frame bytes]`` 帧解码后端注入点。
      ``None`` 且出现 video 段 → 报错（参考实现不绑定视频解码器，P3b §6.3；
      e2e 测试自带合成帧 stub）。

    返回 ``vlm_encode_callback(samples: list[dict]) -> list[dict]``。

    编码规则（text/image/audio/video 统一处理）：
    - 按样本 ``segments`` 的 kind 顺序逐段编码：text → 分词器（或恒等映射）；
      image → ``grid_fn(payload, media_meta)`` 个 placeholder；video →
      :func:`sample_frame_indices` 定帧采样 → ``frame_fn`` 解帧 → resize 到
      ``video_resize`` → 每帧 ``ceil(w/patch) × ceil(h/patch)`` 个 placeholder；
    - **媒体段保持源侧 ``segment_order``**（demux / P3c ts_map 的
      ``(sample_id, segment_order)`` 值匹配契约——audio/video 段必须让 pipeline
      按源侧 order 找回字节/时间戳）；caption 前置的 Text 段取不冲突的新 order
      （``_CAPTION_ORDER_BASE`` 起），不顺移既有媒体段 order；
    - 文本段 ``labels`` = token id 自身，``loss_weight=255``；image/video 段
      ``labels`` 给 0（packer 强制覆盖为 image_label），``loss_weight=0``；
    - ``loss_weights=None``：交由 packer 默认（image 0.0 / text 1.0）；
    - ``media_refs``：image 五键 + video 九键（``kind/seg_order/bytes/grid_h/
      grid_w/flags`` + ``n_frames/width/height/fps_milli/start_ms/duration_ms``）；
      ``flags=0``（VALID）；video 的 ``start_ms = round(timestamp × 1000)``
      （缺失 → fail-fast 含 sid，INV-2 终稿 §6.4）；
    - ``total_video_seconds`` 口径：源时长三级优先级（duration_sec →
      n_frames_total/fps → 0 + WARN，P3b §6.4）。

    返回前对每个样本跑 :func:`validate_encoded_sample`（四条不变式快速失败）。
    """
    _grid = grid_fn or (lambda payload, meta: _default_grid(payload, meta, patch_size))
    # caption Text 的新 order 基址（不冲突、非 0 起——媒体段保持源侧 order）
    caption_order = [_CAPTION_ORDER_BASE]
    fps_default_warned = [False]

    def _text_tokens(text: str) -> List[int]:
        if tokenizer is None:
            return _identity_text_tokens(text)
        ids = tokenizer.encode(text)
        if not isinstance(ids, (list, tuple)) or any(not isinstance(i, int) for i in ids):
            raise TypeError("tokenizer.encode(text) must return list[int] "
                            "(the reference callback does not accept exotic encoders)")
        return list(ids)

    def vlm_encode_callback(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in samples:
            if s.get("error_class"):
                # 编排线程已过滤 error 样本（compute 侧 filter(is_ok)）；出现即契约破坏。
                raise ValueError(
                    f"sample {s.get('sample_id')}: error_class={s['error_class']!r} reached the "
                    "modality encoder — the pipeline should have filtered errored samples"
                )
            sid = s["sample_id"]
            meta = s.get("media_meta")
            token_seq: List[int] = []
            labels: List[int] = []
            seg_table: List[Dict[str, Any]] = []
            media_refs: List[Dict[str, Any]] = []
            stats = dict.fromkeys(_STATS_KEYS, 0)
            next_placeholder = image_range_start

            # P2：样本级伴随文本（单图样本的 caption）前置为 Text 段——
            # 单图行 P0 段表契约不变（文本不进 Rust 段表），encoder 在此重排：
            # caption 存在且输入 segments 不以 Text 段开头时，前置一个 Text 段。
            # P3a/P3b：Text 段取 _CAPTION_ORDER_BASE 起的新 order，**媒体段保持
            # 源侧 segment_order**（demux 的 (sid, seg_order) 源侧查找契约）。
            segs_in = s.get("segments") or []
            caption = s.get("text")
            if caption and not (segs_in and segs_in[0].get("kind") == "text"):
                ids = _text_tokens(caption)
                token_seq.extend(ids)
                labels.extend(ids)
                seg_table.append({"kind": "text", "order": caption_order[0], "token_start": 0,
                                  "token_end": len(token_seq), "media_record_id": None,
                                  "loss_weight": _WEIGHT_ON})
                stats["text_tokens"] += len(ids)
                caption_order[0] += 1  # 多样本连续分配，互不冲突

            for seg in segs_in:
                kind = seg["kind"]
                start = len(token_seq)
                order = seg["order"]  # 源侧 segment_order（契约：值匹配）
                if kind == "text":
                    ids = _text_tokens(seg.get("text") or "")
                    token_seq.extend(ids)
                    labels.extend(ids)
                    seg_table.append({"kind": "text", "order": order, "token_start": start,
                                      "token_end": len(token_seq), "media_record_id": None,
                                      "loss_weight": _WEIGHT_ON})
                    stats["text_tokens"] += len(ids)
                elif kind == "image":
                    # 段级字节优先（统一口径）；fallback 样本级 payload（旧形态兼容）
                    img_bytes = seg.get("bytes") or s.get("payload")
                    grid_h, grid_w = _grid(img_bytes, meta)
                    n = grid_h * grid_w
                    ids = list(range(next_placeholder, next_placeholder + n))
                    next_placeholder += n
                    token_seq.extend(ids)
                    labels.extend([0] * n)   # packer 强制覆盖为 image_label
                    rec_id = len(media_refs)
                    media_refs.append({"kind": "image", "seg_order": order,
                                       "bytes": img_bytes, "grid_h": grid_h,
                                       "grid_w": grid_w, "flags": 0})
                    seg_table.append({"kind": "image", "order": order, "token_start": start,
                                      "token_end": len(token_seq), "media_record_id": rec_id,
                                      "loss_weight": _WEIGHT_OFF})
                    stats["image_tokens"] += n
                    stats["images"] += 1
                elif kind == "audio":
                    # P3a：audio 段由 pipeline 在 EncodedSample 层 demux 切窗
                    # （字节来自源侧预提取）——参考回调只产 audio token 序列 +
                    # Audio 段表项（不产 media_ref：音频字节不走 Vlm sidecar）。
                    # token 数 = round(duration_sec × audio_token_rate)（须与构建
                    # 的 audio_window_token_rate 一致——pipeline 按它切窗）；
                    # duration 缺失时用 _AUDIO_FALLBACK_TOKENS（pipeline 回退
                    # token 隐含时长，自洽）。
                    if audio_range_start is None:
                        raise ValueError(
                            f"sample {sid}: audio segment (order {order}) requires \
                             audio_range_start (declare the Audio id range in \
                             modality_id_ranges and pass audio_range_start)"
                        )
                    ts = seg.get("timestamp")
                    if ts is None:
                        raise ValueError(
                            f"sample {sid}: audio segment (order {order}) has no timestamp: \
                             start_ms = round(timestamp*1000) cannot be computed (INV-2 hard \
                             check, alignment final §6.4)"
                        )
                    dur = (meta or {}).get("duration_sec")
                    n = (int(round(float(dur) * audio_token_rate))
                         if dur is not None else _AUDIO_FALLBACK_TOKENS)
                    ids = list(range(audio_range_start, audio_range_start + n))
                    token_seq.extend(ids)
                    labels.extend([0] * n)
                    seg_table.append({"kind": "audio", "order": order, "token_start": start,
                                      "token_end": len(token_seq), "media_record_id": None,
                                      "loss_weight": _WEIGHT_OFF})
                    stats["audio_tokens"] += n
                    stats["audio_clips"] += 1
                elif kind == "video":
                    # P3b：uniform 定帧采样 + resize + per-frame 网格 placeholder；
                    # media_refs 九键（含 VideoMeta 六字段 + start_ms INV-2）。
                    if video_range_start is None:
                        raise ValueError(
                            f"sample {sid}: video segment (order {order}) requires \
                             video_range_start (declare the Video id range in \
                             modality_id_ranges and pass video_range_start)"
                        )
                    ts = seg.get("timestamp")
                    if ts is None:
                        raise ValueError(
                            f"sample {sid}: video segment (order {order}) has no timestamp: \
                             start_ms = round(timestamp*1000) cannot be computed (INV-2 hard \
                             check, alignment final §6.4)"
                        )
                    start_ms = int(round(float(ts) * 1000))
                    if start_ms < 0:
                        raise ValueError(f"sample {sid}: video timestamp {ts} is negative")
                    # fps：源值优先；缺失 → 兜底值 + 限频 WARN（计数由 pipeline 按
                    # 源侧 media_meta 缺失统计：video_fps_defaulted_clips）
                    fps = (meta or {}).get("fps")
                    if fps is None:
                        if not fps_default_warned[0]:
                            import warnings
                            warnings.warn(
                                "video segment lacks MediaMeta.fps; using video_fps_default="
                                f"{video_fps_default} (frame timing downstream will be off)",
                                stacklevel=2,
                            )
                            fps_default_warned[0] = True
                        fps = video_fps_default
                    fps_milli = int(round(float(fps) * 1000))
                    # 总帧数：容器元数据优先；缺失 → fail-fast（不猜，P3b §6.1）
                    n_frames_total = (meta or {}).get("n_frames_total")
                    if n_frames_total is None:
                        raise ValueError(
                            f"sample {sid}: video segment (order {order}) has no \
                             n_frames_total: cannot sample frames deterministically \
                             (P3b §6.1; detect from container metadata upstream)"
                        )
                    indices = sample_frame_indices(int(n_frames_total), video_n_frames, sid)
                    vid_bytes = seg.get("bytes") or s.get("payload")
                    if frame_fn is None:
                        raise ValueError(
                            f"sample {sid}: video segment (order {order}) requires frame_fn \
                             ((payload, indices) -> list[frame bytes]) — the reference \
                             callback does not bundle a video decoder (P3b §6.3)"
                        )
                    frames = frame_fn(vid_bytes, indices)
                    if len(frames) != len(indices):
                        raise ValueError(
                            f"sample {sid}: frame_fn returned {len(frames)} frames for \
                             {len(indices)} indices"
                        )
                    # per-frame 网格：resize 后 ceil(w/patch) × ceil(h/patch)
                    rw, rh = int(video_resize[0]), int(video_resize[1])
                    gh, gw = -(-rw // patch_size), -(-rh // patch_size)
                    n_per_frame = gh * gw
                    n = n_per_frame * len(indices)
                    ids = list(range(next_placeholder, next_placeholder + n))
                    next_placeholder += n
                    token_seq.extend(ids)
                    labels.extend([0] * n)
                    # 源时长三级优先级（P3b §6.4）：duration_sec → n_frames_total/fps → 0
                    dur = (meta or {}).get("duration_sec")
                    if dur is not None:
                        duration_ms = int(round(float(dur) * 1000))
                    elif n_frames_total and fps:
                        duration_ms = int(round(int(n_frames_total) / float(fps) * 1000))
                    else:
                        duration_ms = 0
                    rec_id = len(media_refs)
                    media_refs.append({
                        "kind": "video", "seg_order": order, "bytes": vid_bytes,
                        "grid_h": gh, "grid_w": gw, "flags": 0,
                        "n_frames": len(indices), "width": rw, "height": rh,
                        "fps_milli": fps_milli, "start_ms": start_ms,
                        "duration_ms": duration_ms,
                    })
                    seg_table.append({"kind": "video", "order": order, "token_start": start,
                                      "token_end": len(token_seq), "media_record_id": rec_id,
                                      "loss_weight": _WEIGHT_OFF})
                    stats["video_tokens"] += n
                    stats["video_clips"] += 1
                else:
                    raise ValueError(
                        f"sample {sid}: segment kind {kind!r} not supported by the reference "
                        "callback"
                    )

            encoded = {
                "sample_id": sid,
                "token_seq": token_seq,
                "segment_table": seg_table,
                "labels": labels,
                "loss_weights": None,
                "stats": stats,
                "media_refs": media_refs,
            }
            validate_encoded_sample(encoded)
            out.append(encoded)
        return out

    # 并行化契约随函数对象走（文档见模块 docstring 与 handoff §2.4）。
    vlm_encode_callback.__doc__ = (
        "vlm modality encoder callback (reference implementation).\n\n"
        "Parallelism contract: Rust orchestrates single-threaded (GIL holder) and calls "
        "this callback chunk by chunk; parallelism inside the callback (thread pool over "
        "PIL/numpy decode, or multiprocessing) is the implementer's responsibility. "
        "Never spawn threads that touch Rust objects (DataBuilder/progress) here."
    )
    return vlm_encode_callback
