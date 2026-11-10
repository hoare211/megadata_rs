//! 多模态 Sample 契约（多模态最终方案 §3.1，P0-T2）。
//!
//! **设计纪律**（方案 §1.2 / §3.1 修订说明）：
//! - 所有模态的 payload 统一持**压缩字节**（原始文件字节），解码只在 encoder
//!   回调内瞬时进行，不落 Sample。音频尤其关键：持 PCM（10min@16kHz ≈ 38MB/样本）
//!   会击穿构建期内存（评审 A 问题 10）。
//! - `SampleId` 跨 source/shard/重试/manifest 可追踪，构建期全局分配，写入 manifest。
//! - 错误分类显式建模（评审 A 问题 7 / 评审 B P0）：source / decode / encode
//!   错误各自可配置 `SkipAndCount`（跳过 + 计数）或 `FailFast`，**禁止静默丢样本**。
//! - Audio/Video 变体**只定义枚举**（P0 范围红线：不实现生命周期），P3a/P3b 落地。

use std::sync::atomic::{AtomicU64, Ordering};

/// 跨 source/shard/重试/manifest 可追踪的稳定标识。
///
/// 构建期全局递增分配（分配器见 [`SampleIdAllocator`]），**不重复、不复用**：
/// 被跳过的样本也消耗 id（保留“第 N 个输入样本 = id N”的对应关系，便于从
/// manifest 反查原始行号）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SampleId(pub u64);

impl std::fmt::Display for SampleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sample#{}", self.0)
    }
}

/// SampleId 全局分配器（构建期内单调递增，线程安全）。
///
/// 按**输入顺序**分配（source 读取是单线程串行，见 `source.rs`），保证：
/// - 同一输入两次构建，样本 → id 映射一致（可复现，方案锚点 8）；
/// - 被错误分类跳过的样本也占 id（行号 ↔ id 稳定对应，便于溯源）。
#[derive(Debug, Default)]
pub struct SampleIdAllocator {
    next: AtomicU64,
}

impl SampleIdAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// 分配下一个 id（0 基）。跳过样本同样调用以保持行号对应。
    pub fn next_id(&self) -> SampleId {
        SampleId(self.next.fetch_add(1, Ordering::Relaxed))
    }

    /// 已分配的总数（含被跳过样本），供 manifest 无漏无重断言。
    pub fn allocated(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}

/// 原始资源稳定引用（多图/交错场景每段各自持 ref，方案 §3.1）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceRef {
    /// 远程 URL（http/https/s3/gs），构建期经 io 层下载（D5）。
    Url(String),
    /// 本地文件路径（JSONL 内 `image_path` / `audio_path` 等字段）。
    LocalPath(String),
    /// tar 包内成员（webdataset，P4 落地；P0 只定义）。
    #[allow(dead_code)]
    TarMember { tar: String, member: String },
    /// parquet 文件行号（P4 落地；P0 只定义）。
    #[allow(dead_code)]
    ParquetRow { file: String, row: u64 },
    /// base64 内联（P0 过渡形态）。
    Inline,
}

/// 编解码标识（影响 decode 策略；P0 至少覆盖图像容器）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Jpeg,
    Png,
    Webp,
    Gif,
    Bmp,
    /// FLAC/Opus/MP3 等音频压缩（P3a）。
    AudioFlac,
    AudioOpus,
    AudioMp3,
    /// MP4/MKV 等视频容器（P3b）。
    VideoMp4,
    VideoMkv,
    /// 容器未知/未探测（reader 侧按内容探测）。
    Unknown,
}

impl Codec {
    /// 由文件扩展名推断（大小写不敏感）；未知扩展名返回 `Codec::Unknown`。
    pub fn from_extension(path: &str) -> Self {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "jpg" | "jpeg" => Codec::Jpeg,
            "png" => Codec::Png,
            "webp" => Codec::Webp,
            "gif" => Codec::Gif,
            "bmp" => Codec::Bmp,
            "flac" => Codec::AudioFlac,
            "opus" => Codec::AudioOpus,
            "mp3" => Codec::AudioMp3,
            "mp4" | "m4v" => Codec::VideoMp4,
            "mkv" => Codec::VideoMkv,
            _ => Codec::Unknown,
        }
    }

    /// data URI 的媒体类型推断（`data:image/jpeg;base64,...` 形式）。
    pub fn from_media_type(media_type: &str) -> Self {
        match media_type.trim().to_ascii_lowercase().as_str() {
            "image/jpeg" => Codec::Jpeg,
            "image/png" => Codec::Png,
            "image/webp" => Codec::Webp,
            "image/gif" => Codec::Gif,
            "image/bmp" => Codec::Bmp,
            "audio/flac" => Codec::AudioFlac,
            "audio/opus" | "audio/ogg" => Codec::AudioOpus,
            "audio/mpeg" | "audio/mp3" => Codec::AudioMp3,
            "video/mp4" => Codec::VideoMp4,
            "video/x-matroska" | "video/webm" => Codec::VideoMkv,
            _ => Codec::Unknown,
        }
    }
}

/// 压缩字节载荷（统一形态，**不持解码后数据**，方案 §3.1）。
#[derive(Debug, Clone, PartialEq)]
pub struct Payload {
    /// JPEG/PNG/FLAC/Opus/MP4… 原始压缩字节。
    ///
    /// 用 `bytes::Bytes` 而非 `Vec<u8>`：引用计数共享，切片/克隆 O(1)，
    /// 与 P1 期 blob sidecar 的零拷贝读取路径同构（P1/D8 对齐 `OwnedBytes` 衔接）。
    pub bytes: bytes::Bytes,
    /// 编解码标识，影响 decode 策略。
    pub codec: Codec,
    /// 原始字节数（字节预算估算用，D5 chunk 双约束）。
    pub raw_len: u64,
}

impl Payload {
    pub fn new(bytes: Vec<u8>, codec: Codec) -> Self {
        Self {
            raw_len: bytes.len() as u64,
            bytes: bytes::Bytes::from(bytes),
            codec,
        }
    }

    /// P4/诊断预留：公开契约方法当前无库内调用方。
    #[allow(dead_code)]
    pub fn from_bytes(bytes: bytes::Bytes, codec: Codec) -> Self {
        let raw_len = bytes.len() as u64;
        Self {
            bytes,
            codec,
            raw_len,
        }
    }
}

/// 影响视觉/音频 token 数的元数据（构建期已知，入 manifest 指纹，方案 §3.1/D9）。
///
/// P0 只使用图像字段（width/height，从 JSONL 元数据带出，缺失时为 `None`）；
/// 音频/视频字段为 P3a/P3b 预留（`skip` 语义：不适用模态为 None）。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MediaMeta {
    // 图像
    pub width: Option<u32>,
    pub height: Option<u32>,
    // 音频（P3a）
    pub sample_rate: Option<u32>,
    pub channels: Option<u8>,
    pub duration_sec: Option<f32>,
    // 视频（P3b）
    pub fps: Option<f32>,
    pub n_frames_total: Option<u64>,
}

/// 解码策略（方案 §3.1）。P0 只落地 `DecodeInEncoder`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecodePolicy {
    /// 默认：encoder 回调内瞬时解码（不落 Sample）。
    #[default]
    DecodeInEncoder,
    /// 跳过（仅记录引用；用于 fail-open 引用场景）。
    Skip,
}

/// 样本内单段：顺序、类型、监督属性、原始资源引用（方案 §3.1）。
///
/// P2 读取侧扩展（vlm encoder 数据来源）：
/// - `text`：Text 段内容。P0 只保留 `len_hint`（文本在 Sample 生命周期内即被
///   丢弃）；vlm 的 modality encoder 必须拿到真文本才能编码 → Text 段随段携带。
/// - `payload`：段级压缩载荷。单模态样本为 None（字节在 `Sample.payload`，
///   避免双份持有）；多图/交错样本字节挂在段上（`Sample.payload = None`）。
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub kind: SegmentKind,
    /// 样本内顺序（0 基递增；多图/交错场景与 `SegmentEntry.segment_order` 对应）。
    pub order: u32,
    /// token/帧数提示（可选；构建期已知时由 encoder 复核）。
    pub len_hint: Option<u32>,
    /// 监督属性（P2 期 packer 消费；P0 只承载）。
    pub supervise: Supervise,
    /// 该段的原始资源（多图/交错用；文本段为 `Inline`）。
    pub source_ref: SourceRef,
    /// 音视频时间戳（对齐用，P3 前置专项设计；P0 只定义）。
    pub timestamp: Option<f64>,
    /// Text 段内容（P2 vlm encoder 需要；非文本段恒 None）。
    pub text: Option<String>,
    /// 段级压缩载荷（多图/交错样本；单模态样本 None——字节在 Sample.payload）。
    pub payload: Option<Payload>,
}

/// 段类型（方案 §3.1）。**P0 范围红线**：Audio/Video 仅定义枚举，不实现生命周期。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    Text,
    Image,
    Audio,
    Video,
}

/// 监督属性（方案 §3.1）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::enum_variant_names)] // LossOn/LossOff/LossWeight 是冻结契约命名
#[allow(dead_code)] // LossWeight 由 SFT 权重策略构造（当前构建路径未产生）
pub enum Supervise {
    #[default]
    LossOn,
    LossOff,
    LossWeight(u8),
}

/// 构建期错误分类（方案 §3.1 / 评审 A 问题 7）。
///
/// **禁止静默丢样本**：每个被跳过的样本必须带 `error_class`，并由失败策略
/// 计数入 manifest `global_stats.skipped_samples_by_class`（P0-T5）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)] // Decode/Encode/IdConflict 由 P4 解码链路与策略矩阵引用
pub enum ErrorClass {
    /// 源不存在（404 / 文件缺失）。
    SourceNotFound,
    /// 源损坏（下载完整但内容校验失败 / 截断）。
    SourceCorrupt,
    /// 解码失败（encoder 回调内 decode 抛错）。
    DecodeFailed,
    /// 编码失败（encoder 回调整体抛错）。
    EncodeFailed,
    /// id 冲突（同一样本被分配两个 id，内部不变式破坏）。
    IdConflict,
}

impl ErrorClass {
    /// manifest / 日志中的稳定字符串键（BTreeMap 序列化可复现）。
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorClass::SourceNotFound => "source_not_found",
            ErrorClass::SourceCorrupt => "source_corrupt",
            ErrorClass::DecodeFailed => "decode_failed",
            ErrorClass::EncodeFailed => "encode_failed",
            ErrorClass::IdConflict => "id_conflict",
        }
    }

    /// 全部错误分类（迭代器，供测试与汇总使用）。
    #[allow(dead_code)]
    pub fn all() -> [ErrorClass; 5] {
        [
            ErrorClass::SourceNotFound,
            ErrorClass::SourceCorrupt,
            ErrorClass::DecodeFailed,
            ErrorClass::EncodeFailed,
            ErrorClass::IdConflict,
        ]
    }
}

/// 单模态 / 交错样本的统一样体（方案 §3.1）。
///
/// - 单模态样本（含纯文本）：`payload = Some`，`segments` 用**单段**表达；
/// - 交错样本：`payload = None`，每段各自持 `source_ref`（P2 期消费）。
///
/// P2（mm）`text`：**伴随文本**——单图样本（`{"text": ..., "image_b64": ...}`）
/// 的文本内容。P0 单图行保持"单 Image 段 + payload=Some"形态逐字节不变，
/// 文本不进段表（避免破坏 P0 段表契约），挂样本级由 vlm encoder 取用。
/// 纯文本样本的文本在 `Segment.text`（Text 段），此处置 None 避免双份。
#[derive(Debug, Clone)]
pub struct Sample {
    pub sample_id: SampleId,
    /// 样本级主引用（单模态样本的源）。
    pub source_ref: SourceRef,
    /// 单模态样本的压缩字节载荷；交错样本为 None。
    pub payload: Option<Payload>,
    /// 段表（单模态样本也用单段表达，保证下游统一走段表）。
    pub segments: Vec<Segment>,
    pub media_meta: Option<MediaMeta>,
    #[allow(dead_code)] // 冻结源契约字段（P4 解码阶段消费；当前构建只透传）
    pub decode_policy: DecodePolicy,
    /// 构建期错误分类（None = 正常；Some = 该样本被跳过，携带原因）。
    pub error_class: Option<ErrorClass>,
    /// 伴随文本（单图样本的 caption；纯文本样本 None——文本在 Text 段）。
    pub text: Option<String>,
}

impl Sample {
    /// 纯文本样本（无 payload，单 Text 段；与旧 JSONL 路径的记录一一对应）。
    ///
    /// P2：`content` 为 `Some` 时段携带文本内容（vlm encoder 需要）；
    /// `None` 保持 P0 旧行为（只有 len_hint）。样本级 `text` 置 None
    ///（文本在 Text 段，避免双份持有）。
    pub fn text(sample_id: SampleId, content: Option<String>, text_len_hint: Option<u32>) -> Self {
        Self {
            sample_id,
            source_ref: SourceRef::Inline,
            payload: None,
            segments: vec![Segment {
                kind: SegmentKind::Text,
                order: 0,
                len_hint: text_len_hint,
                supervise: Supervise::LossOn,
                source_ref: SourceRef::Inline,
                timestamp: None,
                text: content,
                payload: None,
            }],
            media_meta: None,
            decode_policy: DecodePolicy::DecodeInEncoder,
            error_class: None,
            text: None,
        }
    }

    /// 单图像样本（payload = 压缩图像字节，单 Image 段）。
    ///
    /// P2：`caption` 为 `Some` 时作为伴随文本挂样本级（不进段表——P0 单图
    /// 段表契约保持不变；vlm encoder 取样本级 text 前置为 Text 段编码）。
    pub fn image(
        sample_id: SampleId,
        source_ref: SourceRef,
        payload: Payload,
        media_meta: Option<MediaMeta>,
        caption: Option<String>,
    ) -> Self {
        Self {
            sample_id,
            source_ref: source_ref.clone(),
            payload: Some(payload),
            segments: vec![Segment {
                kind: SegmentKind::Image,
                order: 0,
                len_hint: None,
                supervise: Supervise::LossOff,
                source_ref,
                timestamp: None,
                text: None,
                payload: None,
            }],
            media_meta,
            decode_policy: DecodePolicy::DecodeInEncoder,
            error_class: None,
            text: caption,
        }
    }

    /// 交错样本（P2）：payload=None，段各自持字节/引用（多图）。
    ///
    /// `segments` 非空且 order 连续（`validate_segments` 自检）；图像字节挂
    /// `Segment.payload`（内联 b64 解码产物）。一期不支持多图 + 伴随文本
    ///（JSONL 单行无多 text 字段；text 留 None）。
    pub fn interleaved(
        sample_id: SampleId,
        source_ref: SourceRef,
        segments: Vec<Segment>,
        media_meta: Option<MediaMeta>,
    ) -> Self {
        Self {
            sample_id,
            source_ref,
            payload: None,
            segments,
            media_meta,
            decode_policy: DecodePolicy::DecodeInEncoder,
            error_class: None,
            text: None,
        }
    }

    /// 单音频样本（P3a）：payload = 压缩音频字节（FLAC/Opus/MP3），单 Audio 段。
    ///
    /// `timestamp` = 段起点（相对样本起点、秒；对齐终稿 DP2——audio 段对齐轴的
    /// 唯一 build 侧时间输入）。`caption` 为伴随文本时挂样本级（同单图样本，
    /// 不进段表——P0 段表契约保持）。
    pub fn audio(
        sample_id: SampleId,
        source_ref: SourceRef,
        payload: Payload,
        media_meta: Option<MediaMeta>,
        caption: Option<String>,
        timestamp: Option<f64>,
    ) -> Self {
        Self {
            sample_id,
            source_ref: source_ref.clone(),
            payload: Some(payload),
            segments: vec![Segment {
                kind: SegmentKind::Audio,
                order: 0,
                len_hint: None,
                supervise: Supervise::LossOn,
                source_ref,
                timestamp,
                text: None,
                payload: None,
            }],
            media_meta,
            decode_policy: DecodePolicy::DecodeInEncoder,
            error_class: None,
            text: caption,
        }
    }

    /// 单视频样本（P3b）：payload = 压缩视频字节（MP4/MKV），单 Video 段。
    ///
    /// `timestamp` = 段起点（相对样本起点、秒；DP2——video 段对齐轴的 build 侧
    /// 时间输入，`start_ms = round(timestamp × 1000)`）。`caption` 挂样本级
    /// （同单图/单音频样本，不进段表）。
    pub fn video(
        sample_id: SampleId,
        source_ref: SourceRef,
        payload: Payload,
        media_meta: Option<MediaMeta>,
        caption: Option<String>,
        timestamp: Option<f64>,
    ) -> Self {
        Self {
            sample_id,
            source_ref: source_ref.clone(),
            payload: Some(payload),
            segments: vec![Segment {
                kind: SegmentKind::Video,
                order: 0,
                len_hint: None,
                supervise: Supervise::LossOn,
                source_ref,
                timestamp,
                text: None,
                payload: None,
            }],
            media_meta,
            decode_policy: DecodePolicy::DecodeInEncoder,
            error_class: None,
            text: caption,
        }
    }

    /// 错误样本（被跳过，携带分类；不进入后续 encode/pack 流）。
    pub fn errored(sample_id: SampleId, error_class: ErrorClass) -> Self {
        Self {
            sample_id,
            source_ref: SourceRef::Inline,
            payload: None,
            segments: Vec::new(),
            media_meta: None,
            decode_policy: DecodePolicy::Skip,
            error_class: Some(error_class),
            text: None,
        }
    }

    /// 该样本是否正常（未携带错误分类）。
    pub fn is_ok(&self) -> bool {
        self.error_class.is_none()
    }

    /// 字节预算当量（D5 chunk 双约束）：payload 压缩字节 + 源引用长度小项。
    ///
    /// 文本/引用型样本按源 JSON 长度由调用方另行计入（见 `source.rs` 的
    /// `byte_cost_of_line`）；这里只覆盖 payload 主导的图像/音频/视频。
    pub fn payload_bytes(&self) -> u64 {
        self.payload.as_ref().map_or(0, |p| p.raw_len)
    }

    /// 全部压缩字节当量（D5 chunk 字节预算的完整口径）：样本级 payload +
    /// **各段级 payload**（交错样本字节挂段，`payload_bytes` 覆盖不到——
    /// P3a 起音频/视频字节量级大，漏计会击穿 chunk 预算，R2）。
    pub fn total_payload_bytes(&self) -> u64 {
        let seg_bytes: u64 = self
            .segments
            .iter()
            .filter_map(|s| s.payload.as_ref())
            .map(|p| p.raw_len)
            .sum();
        self.payload_bytes() + seg_bytes
    }

    /// 段表不变式自检（方案 §3.2：Σ segment.token_len == token_seq.len() 的
    /// P0 侧镜像——段 order 唯一且连续递增）。
    /// 段表自检（测试/诊断用；构建主链路在 encoder 返回处调用 EncodedSample::validate）。
    #[allow(dead_code)]
    pub fn validate_segments(&self) -> Result<(), String> {
        for (i, seg) in self.segments.iter().enumerate() {
            if seg.order as usize != i {
                return Err(format!(
                    "sample {} segment order is not contiguous from 0: segment {} has order {}",
                    self.sample_id, i, seg.order
                ));
            }
        }
        Ok(())
    }
}

/// 失败策略（方案 §3.1 错误语义 / D5 下载失败域）。
///
/// - `SkipAndCount`（默认）：跳过样本 + 计数入 manifest
///   `global_stats.skipped_samples_by_class`；
/// - `FailFast`：立即中止构建（源损坏场景的严格模式）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FailurePolicy {
    #[default]
    SkipAndCount,
    FailFast,
}

/// 按错误分类的失败策略配置（P0-T4）。
///
/// 默认：**图像下载失败 = SkipAndCount**（方案 §3.1）；其余同样默认
/// SkipAndCount（与旧路径“脏数据丢弃但记日志”的行为对齐，区别是 P0 起
/// **计数入 manifest**，不再只有日志）。
#[derive(Debug, Clone, Copy)]
pub struct FailurePolicyConfig {
    pub source_not_found: FailurePolicy,
    pub source_corrupt: FailurePolicy,
    pub decode_failed: FailurePolicy,
    pub encode_failed: FailurePolicy,
}

impl Default for FailurePolicyConfig {
    fn default() -> Self {
        Self {
            source_not_found: FailurePolicy::SkipAndCount,
            source_corrupt: FailurePolicy::SkipAndCount,
            decode_failed: FailurePolicy::SkipAndCount,
            encode_failed: FailurePolicy::FailFast,
        }
    }
}

impl FailurePolicyConfig {
    pub fn policy_for(&self, class: ErrorClass) -> FailurePolicy {
        match class {
            ErrorClass::SourceNotFound => self.source_not_found,
            ErrorClass::SourceCorrupt => self.source_corrupt,
            ErrorClass::DecodeFailed => self.decode_failed,
            ErrorClass::EncodeFailed | ErrorClass::IdConflict => self.encode_failed,
        }
    }
}

/// P0-T6 encoder 占位（"透传/下载缓存"语义）：消费 `Vec<Sample>` chunk，跨 chunk
/// 校验 `sample_id` 单调连续（无漏无重），不做任何解码——真正解码/编码在 P2
/// （`ModalityAwarePacker` / `PyModalityEncoder`）。
///
/// 校验前提（由 `read_samples` 保证）：
/// - 分配器从 0 起单调递增；**被跳过的样本也占 id**（行号 ↔ id 对应）；
/// - SkipAndCount 把错误样本作为 `Sample::errored` 推入通道（占 id 且进入 chunk），
///   故 chunk 内 id 连续；FailFast 在违规样本处返回 `Err` 中止（该样本未推入），
///   chunk 内仍是前缀连续的。
///
/// 因此正常路径下本 sink 看到的 id 序列恰为 `0..count`；任何 gap/重复都意味着
/// 读取链路内部不变式被破坏（应 FailFast）。本结构是 P0 端到端"全链不重复不
/// 丢失"断言的承载点，也是 P2 真正 encoder 落地前的占位实现。
#[derive(Debug)]
pub struct MmSampleSink {
    next_expected: u64,
    count: u64,
}

impl Default for MmSampleSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MmSampleSink {
    pub fn new() -> Self {
        Self {
            next_expected: 0,
            count: 0,
        }
    }

    /// 推入一个 chunk，逐条校验 id == `next_expected`，发现 gap/重复立即报错。
    pub fn push_chunk(&mut self, samples: &[Sample]) -> Result<(), String> {
        for s in samples {
            if s.sample_id.0 != self.next_expected {
                return Err(format!(
                    "sample_id discontinuity: expected sample#{} got sample#{} (gap or duplicate); \
                     read path invariant violated",
                    self.next_expected, s.sample_id.0
                ));
            }
            self.next_expected += 1;
            self.count += 1;
        }
        Ok(())
    }

    /// 已校验通过的样本数（== 最大连续前缀长度）。
    pub fn count(&self) -> u64 {
        self.count
    }

    /// 下一个期望 id（== 已校验前缀长度；正常结束时 == `allocated`）。
    #[allow(dead_code)] // 测试/诊断断言用
    pub fn next_expected(&self) -> u64 {
        self.next_expected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_id_allocator_is_monotonic_and_shared() {
        let alloc = SampleIdAllocator::new();
        let ids: Vec<SampleId> = (0..100).map(|_| alloc.next_id()).collect();
        // 单调递增、无重复（无漏无重的构造级保证）
        for i in 1..ids.len() {
            assert_eq!(ids[i].0, ids[i - 1].0 + 1);
        }
        assert_eq!(alloc.allocated(), 100);
        // 跨线程安全
        let alloc = std::sync::Arc::new(alloc);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let a = alloc.clone();
                std::thread::spawn(move || {
                    let mut seen = Vec::new();
                    for _ in 0..50 {
                        seen.push(a.next_id().0);
                    }
                    seen
                })
            })
            .collect();
        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 200, "200 ids across 4 threads must be unique");
    }

    #[test]
    fn test_codec_from_extension_and_media_type() {
        assert_eq!(Codec::from_extension("a.JPG"), Codec::Jpeg);
        assert_eq!(Codec::from_extension("a.png"), Codec::Png);
        assert_eq!(Codec::from_extension("a.webp"), Codec::Webp);
        assert_eq!(Codec::from_extension("noext"), Codec::Unknown);
        assert_eq!(Codec::from_media_type("image/JPEG"), Codec::Jpeg);
        assert_eq!(Codec::from_media_type("audio/flac"), Codec::AudioFlac);
        assert_eq!(Codec::from_media_type("video/mp4"), Codec::VideoMp4);
    }

    #[test]
    fn test_sample_constructors_and_segment_invariant() {
        let sid = SampleId(7);
        let t = Sample::text(sid, None, Some(10));
        assert!(t.payload.is_none());
        assert_eq!(t.segments.len(), 1);
        assert_eq!(t.segments[0].kind, SegmentKind::Text);
        assert!(t.is_ok());
        assert_eq!(t.validate_segments(), Ok(()));

        let img = Sample::image(
            sid,
            SourceRef::LocalPath("/x/a.png".into()),
            Payload::new(vec![1, 2, 3], Codec::Png),
            Some(MediaMeta {
                width: Some(64),
                height: Some(64),
                ..Default::default()
            }),
            None,
        );
        assert_eq!(img.payload_bytes(), 3);
        assert_eq!(img.segments[0].kind, SegmentKind::Image);
        // 图像段默认不监督（P2 规则在 packer 侧落地，契约层默认值先行）
        assert_eq!(img.segments[0].supervise, Supervise::LossOff);

        let bad = Sample::errored(SampleId(8), ErrorClass::SourceNotFound);
        assert!(!bad.is_ok());
        assert_eq!(bad.error_class, Some(ErrorClass::SourceNotFound));
        assert!(bad.segments.is_empty());
    }

    #[test]
    fn test_segment_order_must_be_contiguous() {
        let mut s = Sample::text(SampleId(1), None, None);
        s.segments.push(Segment {
            kind: SegmentKind::Image,
            order: 5, // 跳号 → 违反不变式
            len_hint: None,
            supervise: Supervise::LossOff,
            source_ref: SourceRef::Inline,
            timestamp: None,
            text: None,
            payload: None,
        });
        assert!(s.validate_segments().is_err());
    }

    #[test]
    fn test_failure_policy_config_defaults() {
        let cfg = FailurePolicyConfig::default();
        // 方案 §3.1：默认图像下载失败 = SkipAndCount
        assert_eq!(
            cfg.policy_for(ErrorClass::SourceNotFound),
            FailurePolicy::SkipAndCount
        );
        assert_eq!(
            cfg.policy_for(ErrorClass::SourceCorrupt),
            FailurePolicy::SkipAndCount
        );
        assert_eq!(
            cfg.policy_for(ErrorClass::DecodeFailed),
            FailurePolicy::SkipAndCount
        );
        // encode 失败默认 FailFast（编码器整体故障不该静默丢数据）
        assert_eq!(
            cfg.policy_for(ErrorClass::EncodeFailed),
            FailurePolicy::FailFast
        );
    }

    #[test]
    fn test_error_class_keys_are_unique() {
        let keys: Vec<&str> = ErrorClass::all().iter().map(|c| c.as_str()).collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(keys.len(), sorted.len(), "error class keys must be unique");
    }

    /// P0-T6 encoder 占位：跨 chunk 校验 sample_id 连续（无漏无重）。
    #[test]
    fn test_mm_sample_sink_validates_id_continuity() {
        // 正常：两 chunk，id 0..4 连续
        let mut sink = MmSampleSink::new();
        let c0 = vec![
            Sample::text(SampleId(0), None, None),
            Sample::text(SampleId(1), None, None),
        ];
        let c1 = vec![
            Sample::errored(SampleId(2), ErrorClass::SourceNotFound), // 跳过样本也占 id
            Sample::text(SampleId(3), None, None),
        ];
        assert!(sink.push_chunk(&c0).is_ok());
        assert!(sink.push_chunk(&c1).is_ok());
        assert_eq!(sink.count(), 4);
        assert_eq!(sink.next_expected(), 4);

        // gap：先建连续前缀 0,1（next_expected=2），再推 id 3（缺 2）
        let mut bad = MmSampleSink::new();
        assert!(
            bad.push_chunk(&[
                Sample::text(SampleId(0), None, None),
                Sample::text(SampleId(1), None, None)
            ])
            .is_ok()
        );
        let err = bad
            .push_chunk(&[Sample::text(SampleId(3), None, None)])
            .unwrap_err();
        assert!(err.contains("discontinuity"), "{err}");
        assert!(err.contains("expected sample#2 got sample#3"), "{err}");

        // 重复：同 id 出现两次（0 → 0）
        let mut dup = MmSampleSink::new();
        let repeated = vec![
            Sample::text(SampleId(0), None, None),
            Sample::text(SampleId(0), None, None),
        ];
        assert!(
            dup.push_chunk(&repeated).is_err(),
            "duplicate id must be rejected"
        );

        // 非零起点：第一个 id 不是 0
        let mut start = MmSampleSink::new();
        let off = vec![Sample::text(SampleId(2), None, None)];
        let err = start.push_chunk(&off).unwrap_err();
        assert!(err.contains("expected sample#0 got sample#2"), "{err}");
    }
}
