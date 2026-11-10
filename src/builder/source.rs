use sonic_rs::{JsonContainerTrait, JsonValueTrait};
use std::io::BufRead;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::builder::pipeline::BuildProgress;
use crate::builder::sample::{
    Codec, ErrorClass, FailurePolicy, FailurePolicyConfig, MediaMeta, Payload, Sample, SampleId,
    SampleIdAllocator, Segment, SegmentKind, SourceRef, Supervise,
};

// ★ 0.4 Tier-0 快赢项：去 tokio::fs（spawn_blocking-per-line 结构性浪费）。
//   `tokio::fs::File` 本质是 `spawn_blocking` 包同步调用，每行一次 blocking 任务派发，
//   847 万行 = 847 万次调度；本地逐行读无 IO 等待可让出，async 层纯开销。改 `std::fs`
//   + 4MB BufReader + 同步 send，syscall 粒度对齐写端（8MB BufWriter）。
//
//   **诚实注记**：读端当前被背压（23.8 MB/s 远低于其 100–500 MB/s 能力上限），
//   现阶段吞吐收益 ≈0——属浪费消除，定位为「不是性能改动」。

/// 读端缓冲大小（4MB，对齐写端 8MB BufWriter 的量级；旧 tokio BufReader 默认 8KB）。
const READ_BUF_SIZE: usize = 4 * 1024 * 1024;

/// 按块读取 JSONL 文件，防止 OOM。同时更新进度计数器。
///
/// **同步函数**（短期项 #9）：旧名 `read_jsonl_chunks_async` 带 `async` 却内部
/// 没有任何 `.await`——名字会让调用方误以为存在让出点（在阻塞读里"等待"），
/// 属名不副实。改名并去掉 `async`，签名与行为一致。
/// 实现：`std::fs::File` + `read_line` 大块缓冲 + `flume::Sender::send`（同步）。
pub fn read_jsonl_chunks(
    filepath: &str,
    chunk_size: usize,
    tx_raw: flume::Sender<Vec<String>>,
    progress: Arc<BuildProgress>,
) -> std::io::Result<()> {
    let file = std::fs::File::open(filepath)?;
    let mut reader = std::io::BufReader::with_capacity(READ_BUF_SIZE, file);
    let mut chunk = Vec::with_capacity(chunk_size);
    // line 经 clear() 复用（amortized 零分配），每行实际只有 trim_end().to_string() 一次分配
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        progress
            .bytes_read
            .fetch_add(line.len() as u64, Ordering::Relaxed);
        let trimmed = line.trim_end().to_string();
        if !trimmed.is_empty() {
            progress.docs_read.fetch_add(1, Ordering::Relaxed);
            chunk.push(trimmed);
        }

        if chunk.len() >= chunk_size {
            // 同步 send：flume 的 Sender::send 在通道满时阻塞直到有空间（正确的背压）。
            // 单线程 runtime 下无并发可让出，阻塞等价于旧 send_async 的 await。
            if tx_raw.send(std::mem::take(&mut chunk)).is_err() {
                break;
            }
            chunk.reserve(chunk_size);
        }
    }

    if !chunk.is_empty() {
        let _ = tx_raw.send(chunk);
    }

    Ok(())
}

// ============================================================================
// 多模态 source 读取（多模态最终方案 D5 / P0-T3~T6）
// ============================================================================

/// chunk 分块双约束（D5）：条数 + 字节预算。
///
/// 修订原因（评审 A 问题 2）：原方案沿用条数分块，base64 图像单条几 MB，
/// chunk 8192 条 = 数十 GB，P0 自己就 OOM。`payload.raw_len` 计入当量，
/// 超额**提前切 chunk**。
#[derive(Debug, Clone, Copy)]
pub struct ChunkLimit {
    /// 单 chunk 最大样本条数（0 = 不限制条数，仅受字节预算约束）。
    pub max_samples: usize,
    /// 单 chunk 最大字节数（含 payload 压缩字节 + 源 JSON 行长度当量；0 = 不限制）。
    pub max_bytes: u64,
}

impl ChunkLimit {
    pub fn new(max_samples: usize, max_bytes: u64) -> Self {
        Self {
            max_samples,
            max_bytes,
        }
    }
}

impl Default for ChunkLimit {
    /// 默认：4096 条 + 64MiB 字节预算（对齐 read_jsonl_chunks 的 5000 条量级，
    /// 字节预算兜底防大图场景 OOM）。
    fn default() -> Self {
        Self {
            max_samples: 4096,
            max_bytes: 64 * 1024 * 1024,
        }
    }
}

/// 单行源 JSON 的字节当量：行字节数 + 该行携带的 payload 估算字节。
///
/// base64 内联图像的压缩字节 ≈ base64 长度 × 3/4；引用型（URL/path）按行长度计
/// （真实下载字节未知，行长度是上界的保守替代；下载在 encoder 前完成，
/// 由 [`resolve_payload`] 的真实 `raw_len` 校正下一轮预算）。
fn byte_cost_of_line(line_len: usize, inline_media_b64_len: usize) -> u64 {
    line_len as u64 + (inline_media_b64_len as u64 * 3) / 4
}

/// P4：源字段 → Sample 角色的映射（tar 用扩展名，parquet 用列名；同一抽象）。
///
/// 动机：webdataset 扩展名与 parquet 列名千变万化，不能硬编码——提供可配置
/// 映射 + 贴合生态的默认值（P4 方案 §5）。
#[derive(Debug, Clone)]
pub struct FieldMap {
    /// 图像字节字段（扩展名/列名）→ Image 段。可多个（多图）。
    pub image_fields: Vec<String>,
    /// 文本字段 → caption（单值）或 Text 段。
    pub text_fields: Vec<String>,
    /// 音频字节字段 → Audio 段（P3a 消费，P4 只装载）。
    pub audio_fields: Vec<String>,
    /// 视频字节字段 → Video 段（P3b 消费）。
    pub video_fields: Vec<String>,
    /// JSON/struct 元数据字段 → MediaMeta（width/height/...）来源。
    pub meta_fields: Vec<String>,
    /// 时间戳字段（P3a/P3b 联动，P4 方案 §5.5）：`None` = 自动探测
    /// `timestamp`/`start_time`/`start`/`offset`（按序首个命中）；探测不到 →
    /// `Segment.timestamp = None`（P4 不 fail-fast；P3a/P3b INV-2 前置检查会
    /// fail-fast 含 sid——"只装载不验证"纪律）。
    pub timestamp_field: Option<String>,
    /// parquet 列投影：仅读这些列（None = 读全部；含大字节的列务必显式投影）。
    pub projection: Option<Vec<String>>,
}

impl Default for FieldMap {
    /// 默认映射（贴合 webdataset 惯例，P4 方案 §5.3）。
    ///
    /// P4-2（评审）：同时覆盖 tar 扩展名形态与 parquet 列名形态——FieldMap 是
    /// tar/parquet 共用单列表设计（§5.2），T4 落地时直接消费当前默认即可正确
    /// 识别 parquet 图像列（`image`/`image_bytes`/`image_png` 等）。
    fn default() -> Self {
        Self {
            image_fields: vec![
                "jpg".to_string(),
                "jpeg".to_string(),
                "png".to_string(),
                "webp".to_string(),
                "image".to_string(),
                "image_bytes".to_string(),
                "image_png".to_string(),
                "image_jpg".to_string(),
                "image_jpeg".to_string(),
                "image_webp".to_string(),
            ],
            text_fields: vec![
                "txt".to_string(),
                "caption".to_string(),
                "text".to_string(),
                "sentence".to_string(),
            ],
            audio_fields: vec![
                "flac".to_string(),
                "mp3".to_string(),
                "opus".to_string(),
                "wav".to_string(),
                "audio".to_string(),
                "audio_bytes".to_string(),
            ],
            video_fields: vec![
                "mp4".to_string(),
                "mkv".to_string(),
                "webm".to_string(),
                "video".to_string(),
                "video_bytes".to_string(),
            ],
            meta_fields: vec![
                "json".to_string(),
                "meta".to_string(),
                "metadata".to_string(),
            ],
            timestamp_field: None,
            projection: None,
        }
    }
}

/// 多模态 source 读取的配置（P0-T3/T4 汇总，绑定层构造）。
#[derive(Debug, Clone)]
pub struct ReadSamplesOptions {
    /// chunk 分块双约束（D5）。
    pub chunk_limit: ChunkLimit,
    /// 失败策略（显式配置；默认见 FailurePolicyConfig::default）。
    pub failure_policy: FailurePolicyConfig,
    /// 是否下载 URL 引用的媒体字节（false = 仅记录引用，payload 为 None）。
    ///
    /// P0 默认 true：图像字节需要在 encode 前到位（D1 的 visual placeholder
    /// 在 P2 才消费；P0 的 encoder 是"下载并缓存"的占位实现）。
    pub download_url_media: bool,
    /// URL 下载重试次数（复用 io 层退避原语；默认 3）。
    pub download_attempts: u32,
    /// P4：源字段 → 角色映射（**仅 tar/parquet reader 消费**——JSONL reader
    /// 不读取，行内字段解析仍走 P0 固定键，避免语义混淆）。
    pub field_map: FieldMap,
    /// P2-6（评审修复）：构建取消令牌。媒体 URL 下载在每次尝试前检查，
    /// 取消后不再发起新请求（配合 HTTP 超时避免无限等待）。
    pub cancel: Option<crate::cancel::CancellationToken>,
}

impl Default for ReadSamplesOptions {
    fn default() -> Self {
        Self {
            chunk_limit: ChunkLimit::default(),
            failure_policy: FailurePolicyConfig::default(),
            download_url_media: true,
            download_attempts: 3,
            field_map: FieldMap::default(),
            cancel: None,
        }
    }
}

/// 读取期的错误计数（P0-T5：入 manifest `global_stats.skipped_samples_by_class`，
/// 按 ErrorClass 分类）。
#[derive(Debug, Default)]
pub struct SourceErrorCounts {
    /// 按错误分类的跳过计数（键为 `ErrorClass::as_str()`，BTreeMap 保证可复现序列化）。
    pub skipped_by_class: std::collections::BTreeMap<String, u64>,
    /// FailFast 触发次数（每次触发都中止构建，正常应为 0 或 1）。
    pub fail_fast_triggered: u64,
}

impl SourceErrorCounts {
    pub fn record(&mut self, class: ErrorClass) {
        *self
            .skipped_by_class
            .entry(class.as_str().to_string())
            .or_insert(0) += 1;
    }

    #[allow(dead_code)] // 测试/诊断汇总用
    pub fn total(&self) -> u64 {
        self.skipped_by_class.values().sum()
    }
}

/// P4：源格式路由（按扩展名；扩展名统一转小写后匹配）。
///
/// - `.jsonl` / `.json` → JSONL（现状兼容：现实现对任意扩展名都按 JSONL 行读，
///   显式枚举保持这一兼容面）；
/// - `.tar` / `.tar.gz` / `.tgz` → webdataset tar（是否 gzip 由打开后的魔数判定）；
/// - `.parquet` → parquet；
/// - 其他 → **硬报错**（`InvalidInput`，错误信息列出实际扩展名与受支持集合——
///   防止把非 JSONL 文件静默按行读，消灭静默错数据纪律）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceFormat {
    Jsonl,
    Tar,
    Parquet,
}

impl SourceFormat {
    pub fn detect(path: &str) -> std::io::Result<Self> {
        let lower = path.to_ascii_lowercase();
        let format = if lower.ends_with(".jsonl") || lower.ends_with(".json") {
            Self::Jsonl
        } else if lower.ends_with(".tar") || lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
            Self::Tar
        } else if lower.ends_with(".parquet") {
            Self::Parquet
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "unsupported source format for {path}: expected one of .jsonl / .json / \
                     .tar / .tar.gz / .tgz / .parquet (P4 source routing; rename the file or \
                     convert it — refusing to silently read a non-JSONL file line by line)"
                ),
            ));
        };
        Ok(format)
    }
}

/// P4：三 reader 共用的"攒块 + 字节预算 + 背压 send"发射器（抽自原
/// `read_samples` 主体）。保证 JSONL/tar/parquet 三条路径的 chunk 语义、
/// id 连续性、取消行为逐字一致（P4 方案 §3.1）。
pub struct SampleChunkEmitter<'a> {
    options: &'a ReadSamplesOptions,
    id_alloc: &'a SampleIdAllocator,
    tx: flume::Sender<Vec<Sample>>,
    chunk: Vec<Sample>,
    chunk_bytes: u64,
}

impl<'a> SampleChunkEmitter<'a> {
    pub fn new(
        options: &'a ReadSamplesOptions,
        id_alloc: &'a SampleIdAllocator,
        tx: flume::Sender<Vec<Sample>>,
    ) -> Self {
        Self {
            options,
            id_alloc,
            tx,
            chunk: Vec::with_capacity(options.chunk_limit.max_samples.min(4096)),
            chunk_bytes: 0,
        }
    }

    /// 分配下一个 sample_id（所有 reader 必须按"源内出现顺序"调用；跳过样本也
    /// 占用 id，忽略组不占用——忽略 ≠ 跳过，P0-T6）。
    pub fn next_id(&self) -> SampleId {
        self.id_alloc.next_id()
    }

    /// 推入一个已解析样本 + 其字节当量；超额则切块 send。
    /// `byte_cost` = payload 压缩字节 + 源侧结构开销（JSONL 用行长当量，
    /// tar/parquet 用成员/列字节）。返回 Err 仅当通道关闭（消费端退出）。
    pub fn push(&mut self, sample: Sample, byte_cost: u64) -> Result<(), ()> {
        self.chunk_bytes += byte_cost;
        self.chunk.push(sample);
        let over_samples = self.options.chunk_limit.max_samples > 0
            && self.chunk.len() >= self.options.chunk_limit.max_samples;
        let over_bytes = self.options.chunk_limit.max_bytes > 0
            && self.chunk_bytes >= self.options.chunk_limit.max_bytes;
        if over_samples || over_bytes {
            // 同步 send：通道满时阻塞直到有空间（与 read_jsonl_chunks 一致的背压语义）
            if self.tx.send(std::mem::take(&mut self.chunk)).is_err() {
                return Err(());
            }
            self.chunk_bytes = 0;
        }
        Ok(())
    }

    /// 收尾：flush 残余 chunk。通道已关闭时静默（与旧实现 `let _ =` 一致）。
    pub fn flush(self) {
        if !self.chunk.is_empty() {
            let _ = self.tx.send(self.chunk);
        }
    }
}

/// 读取 JSONL 多模态样本 → `Vec<Sample>`，经 flume 背压传出。
///
/// **与 `read_jsonl_chunks` 的关系**：旧路径一行不动（PT/SFT 零改动承诺）；
/// 本函数是多模态新入口，两者按 `mode` 分派（绑定层 vlm 模式走本函数）。
///
/// 支持的行格式（P0 最小集 + P3a audio / P3b video 内联）：
/// ```json
/// {"text": "...", "image_url": "https://...", "image_b64": "...", "image_path": "...", "image_meta": {"width": 64, "height": 64}}
/// {"text": "...", "audio_b64": "<flac>", "audio_media_type": "audio/flac", "audio_meta": {"sample_rate": 16000, "channels": 1, "duration_sec": 4.0}, "timestamp": 1.5}
/// {"text": "...", "video_b64": "<mp4>", "video_media_type": "video/mp4", "video_meta": {"fps": 30.0, "n_frames_total": 900, "duration_sec": 30.0}, "timestamp": 0.0}
/// {"text": "...", "image_b64": "<png>", "audio_b64": "<flac>", "audio_meta": {...}, "timestamp": 0.0}
/// {"text": "...", "image_b64": "<png>", "video_b64": "<mp4>", "video_meta": {...}, "timestamp": 0.0}
/// ```
/// - 纯文本行（无任何图像/音频/视频字段）→ 单 Text 段样本；
/// - 图像行按 `image_url` → `image_b64` → `image_path` 优先级解析；
/// - **audio 行（P3a）**：`audio_b64` 内联 → 单 Audio 段样本（`timestamp` =
///   段起点/相对样本起点/秒，DP2；`audio_meta` 带出 sr/channels/duration）。
///   `audio_url`/`audio_path` 引用与 `audio_b64` 数组形态**暂不支持**（显式
///   SourceCorrupt，不静默降级；引用形态随 P4 source 扩展落地）；
/// - **video 行（P3b）**：`video_b64` 内联 → 单 Video 段样本（同 timestamp 口径；
///   `video_meta` 带出 fps/n_frames_total/duration）。引用/数组形态暂不支持；
/// - **图+音 / 图+视同行（P3a/P3b）**：内联单图 + 内联 AV → 交错样本
///   `[Image, Audio|Video]`（text 挂样本级 caption）；图像引用/多图 + AV、
///   **audio + video 同行**组合暂不支持（显式报错）；
/// - data URI（`data:image/jpeg;base64,...`）按 `Codec::from_media_type` 识别；
/// - 图像下载失败按 `FailurePolicy` 处置：SkipAndCount → 错误样本 + 计数；
///   FailFast → 返回 Err 中止构建（**禁止静默丢样本**）。
///
/// `id_alloc`：sample_id 全局分配器（P0-T6；被跳过样本也占 id，保持行号 ↔ id 对应）。
pub fn read_samples(
    filepath: &str,
    options: &ReadSamplesOptions,
    id_alloc: &SampleIdAllocator,
    error_counts: &mut SourceErrorCounts,
    tx_samples: flume::Sender<Vec<Sample>>,
    progress: Arc<BuildProgress>,
) -> std::io::Result<()> {
    match SourceFormat::detect(filepath)? {
        SourceFormat::Jsonl => read_jsonl_samples(
            filepath,
            options,
            id_alloc,
            error_counts,
            tx_samples,
            progress,
        ),
        SourceFormat::Tar => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("webdataset tar source is not implemented yet (P4-T3 pending): {filepath}"),
        )),
        SourceFormat::Parquet => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("parquet source is not implemented yet (P4-T4 pending): {filepath}"),
        )),
    }
}

fn read_jsonl_samples(
    filepath: &str,
    options: &ReadSamplesOptions,
    id_alloc: &SampleIdAllocator,
    error_counts: &mut SourceErrorCounts,
    tx_samples: flume::Sender<Vec<Sample>>,
    progress: Arc<BuildProgress>,
) -> std::io::Result<()> {
    let file = std::fs::File::open(filepath)?;
    let mut reader = std::io::BufReader::with_capacity(READ_BUF_SIZE, file);
    let mut emitter = SampleChunkEmitter::new(options, id_alloc, tx_samples);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        progress
            .bytes_read
            .fetch_add(line.len() as u64, Ordering::Relaxed);
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            continue;
        }
        progress.docs_read.fetch_add(1, Ordering::Relaxed);

        // sample_id 按输入顺序分配（含被跳过样本：行号 ↔ id 稳定对应）
        let sample_id = emitter.next_id();
        let parsed = parse_sample_line(trimmed, sample_id, options, error_counts)?;
        let cost = byte_cost_of_line(
            trimmed.len(),
            // P3a：段级 payload 也计入（交错样本字节挂段；音频/视频字节量级大，
            // 漏计会击穿 chunk 预算——D5/R2）
            parsed
                .as_ref()
                .map_or(0, |s| s.total_payload_bytes() as usize * 4 / 3),
        );
        if let Some(s) = parsed
            && emitter.push(s, cost).is_err()
        {
            break; // 通道关闭（消费端退出）
        }
    }

    emitter.flush();
    Ok(())
}

/// P3c：构造携带 timestamp 的 Text 段（三模态对齐轴的 text 侧输入）。
fn text_segment(order: u32, text: &str, timestamp: Option<f64>) -> Segment {
    Segment {
        kind: SegmentKind::Text,
        order,
        len_hint: Some(text.chars().count().min(u32::MAX as usize) as u32),
        supervise: Supervise::LossOn,
        source_ref: SourceRef::Inline,
        timestamp,
        text: Some(text.to_string()),
        payload: None,
    }
}

/// P2-5（评审修复）：媒体字段**存在但类型不受支持**（既非 string，也不是调用方
/// 已显式处理/拒绝的 array，也不是表示"未提供"的 null）→ 返回错误消息。
///
/// 旧实现统一 `and_then(as_str)`：`{"video_b64":{"nested":"value"}}` 静默得到
/// `None`，整行按纯文本处理且不计 skipped/error——上游 schema 漂移时模态数据
/// 静默消失、manifest 无法反映。现在按 `SourceCorrupt` + 既有 failure policy 上报。
fn media_field_type_error(val: &sonic_rs::Value, key: &str) -> Option<String> {
    let v = val.get(key)?;
    if v.is_null() || v.as_str().is_some() || v.is_array() {
        return None;
    }
    Some(format!(
        "{key} must be an inline base64 string but got an unsupported JSON type \
         (upstream schema drift must not be silently treated as an absent media field)"
    ))
}

/// 解析一行多模态 JSONL → `Option<Sample>`。
///
/// `None` = 该行被跳过（错误分类已计入 `error_counts`；FailFast 时返回 `Err`）。
/// 纯文本行返回单 Text 段样本（不跳过）。
fn parse_sample_line(
    line: &str,
    sample_id: SampleId,
    options: &ReadSamplesOptions,
    error_counts: &mut SourceErrorCounts,
) -> std::io::Result<Option<Sample>> {
    let val: sonic_rs::Value = match sonic_rs::from_str(line) {
        Ok(v) => v,
        Err(e) => {
            // JSON 解析失败：按 SourceCorrupt 分类（沿用旧路径"脏数据丢弃"语义，
            // 但 P0 起计数入 manifest，不再只有日志）
            return skip_or_fail(
                sample_id,
                ErrorClass::SourceCorrupt,
                options,
                error_counts,
                &format!("JSON parsing failed: {e}"),
            )
            .map(Some);
        }
    };

    let text = val.get("text").and_then(|t| t.as_str());
    // 图像引用优先级：image_url > image_b64 > image_path（方案 P0）
    let image_url = val.get("image_url").and_then(|v| v.as_str());
    let image_b64 = val.get("image_b64").and_then(|v| v.as_str());
    let image_path = val.get("image_path").and_then(|v| v.as_str());
    // P2（mm）：image_b64 支持数组形态（多图 → 交错样本，字节挂段）；url/path
    // 数组不支持（显式报错）。数组检测必须在单值提取之后——as_str 对数组返回
    // None，先检测会把它误判成纯文本行。
    let b64_multi: Option<Vec<String>> =
        if val.get("image_b64").map(|v| v.is_array()).unwrap_or(false) {
            let arr = val
                .get("image_b64")
                .and_then(|v| v.as_array())
                .expect("is_array checked");
            match arr
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_string)
                        .ok_or_else(|| "image_b64 array items must be strings".to_string())
                })
                .collect::<Result<Vec<String>, String>>()
            {
                Ok(items) if !items.is_empty() => Some(items),
                Ok(_) => {
                    return skip_or_fail(
                        sample_id,
                        ErrorClass::SourceCorrupt,
                        options,
                        error_counts,
                        "image_b64 array must not be empty",
                    )
                    .map(Some);
                }
                Err(e) => {
                    return skip_or_fail(
                        sample_id,
                        ErrorClass::SourceCorrupt,
                        options,
                        error_counts,
                        &e,
                    )
                    .map(Some);
                }
            }
        } else {
            None
        };
    for key in ["image_url", "image_path"] {
        if val.get(key).map(|v| v.is_array()).unwrap_or(false) {
            return skip_or_fail(
                sample_id, ErrorClass::SourceCorrupt, options, error_counts,
                &format!("{key} array form is not supported (multi-image requires inline image_b64 array)"),
            ).map(Some);
        }
    }
    let image_meta = parse_image_meta(val.get("image_meta"));

    // P3a：audio 字段提取（audio_b64 内联；引用/数组形态显式报错）。
    // 数组检测必须在单值提取之后（as_str 对数组返回 None，先检测会误判成纯文本行）。
    let audio_b64 = val.get("audio_b64").and_then(|v| v.as_str());
    for key in ["audio_url", "audio_path"] {
        if val.get(key).is_some() {
            return skip_or_fail(
                sample_id,
                ErrorClass::SourceCorrupt,
                options,
                error_counts,
                &format!(
                    "{key} references are not supported yet (inline audio_b64 only; \
                          audio reference forms land with the P4 source extension)"
                ),
            )
            .map(Some);
        }
    }
    if val.get("audio_b64").map(|v| v.is_array()).unwrap_or(false) {
        return skip_or_fail(
            sample_id,
            ErrorClass::SourceCorrupt,
            options,
            error_counts,
            "audio_b64 array form is not supported yet (multi-audio needs per-segment timestamps)",
        )
        .map(Some);
    }
    let audio_meta = parse_audio_meta(val.get("audio_meta"));
    // 段起点（相对样本起点、秒；DP2）——P3a 挂 audio 段、P3b 挂 video 段
    let audio_ts = val.get("timestamp").and_then(|v| v.as_f64());
    let has_audio = audio_b64.is_some();

    // P3b：video 字段提取（镜像 audio——video_b64 内联；引用/数组形态显式报错）
    let video_b64 = val.get("video_b64").and_then(|v| v.as_str());
    for key in ["video_url", "video_path"] {
        if val.get(key).is_some() {
            return skip_or_fail(
                sample_id,
                ErrorClass::SourceCorrupt,
                options,
                error_counts,
                &format!(
                    "{key} references are not supported yet (inline video_b64 only; \
                          video reference forms land with the P4 source extension)"
                ),
            )
            .map(Some);
        }
    }
    if val.get("video_b64").map(|v| v.is_array()).unwrap_or(false) {
        return skip_or_fail(
            sample_id,
            ErrorClass::SourceCorrupt,
            options,
            error_counts,
            "video_b64 array form is not supported yet (multi-video needs per-segment timestamps)",
        )
        .map(Some);
    }
    // P2-5：key 存在但值为非字符串/非数组（对象、数字、布尔）→ 显式 SourceCorrupt，
    // 不得静默当作字段不存在（否则该模态数据整条丢失且统计无痕）。
    for key in [
        "image_url",
        "image_b64",
        "image_path",
        "audio_b64",
        "video_b64",
    ] {
        if let Some(msg) = media_field_type_error(&val, key) {
            return skip_or_fail(
                sample_id,
                ErrorClass::SourceCorrupt,
                options,
                error_counts,
                &msg,
            )
            .map(Some);
        }
    }
    let video_meta = parse_video_meta(val.get("video_meta"));
    let has_video = video_b64.is_some();
    // audio + video 同行 → 显式报错（双 AV 模态组合一期不支持；与多图+caption
    // 同族的"暂不支持组合显式报错"纪律）
    if has_audio && has_video {
        return skip_or_fail(
            sample_id,
            ErrorClass::SourceCorrupt,
            options,
            error_counts,
            "audio + video in one row is not supported yet (a row carries at most one \
             AV modality; use separate rows or the P4 tar/parquet source)",
        )
        .map(Some);
    }
    let has_image =
        image_url.is_some() || image_b64.is_some() || image_path.is_some() || b64_multi.is_some();

    // P3c：text + AV + timestamp → text 升为 **Text 段**（order 0，携带 timestamp）
    // 打头的交错样本——三模态对齐轴（text↔window）的输入形态；无 timestamp 时
    // 保持 caption 语义（Sample.text，向后兼容——caption 无时间信息）。
    let text_as_segment = text.is_some()
        && (has_audio || has_video)
        && audio_ts.is_some()
        && b64_multi.is_none()
        && image_url.is_none()
        && image_path.is_none();

    if !has_image && !has_audio && !has_video {
        // 纯文本行：单 Text 段（len_hint 用文本字符数作粗提示，P2 encoder 复核）。
        // P2：文本内容随段携带（vlm encoder 需要真文本；pt/sft 不走 read_samples，
        // 旧路径不受影响）。
        let hint = text.map(|t| t.chars().count().min(u32::MAX as usize) as u32);
        return Ok(Some(Sample::text(
            sample_id,
            text.map(str::to_string),
            hint,
        )));
    }

    // P3a：audio 内联解码（audio-only 早退；audio+inline-image 走下面的交错分支）
    let decode_audio = || -> Result<(Vec<u8>, Codec), String> {
        let b64 = audio_b64.expect("has_audio checked");
        let bytes = decode_base64(b64)
            .map_err(|e| format!("base64 audio payload could not be decoded: {e}"))?;
        let codec = Codec::from_media_type(
            val.get("audio_media_type")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
        );
        Ok((bytes, codec))
    };
    if has_audio && !has_image {
        let (bytes, codec) = match decode_audio() {
            Ok(v) => v,
            Err(e) => {
                return skip_or_fail(
                    sample_id,
                    ErrorClass::SourceCorrupt,
                    options,
                    error_counts,
                    &e,
                )
                .map(Some);
            }
        };
        // P3c：text+audio+timestamp → [Text(0,ts), Audio(1,ts)] 交错（字节挂段）；
        // 否则单模态 audio 样本（payload=Some + 单 Audio 段；伴随文本挂样本级，
        // 同单图 caption 契约，不进段表）
        if text_as_segment {
            let segs = vec![
                text_segment(0, text.expect("text_as_segment checked"), audio_ts),
                Segment {
                    kind: SegmentKind::Audio,
                    order: 1,
                    len_hint: None,
                    supervise: Supervise::LossOn,
                    source_ref: SourceRef::Inline,
                    timestamp: audio_ts,
                    text: None,
                    payload: Some(Payload::new(bytes, codec)),
                },
            ];
            return Ok(Some(Sample::interleaved(
                sample_id,
                SourceRef::Inline,
                segs,
                audio_meta,
            )));
        }
        return Ok(Some(Sample::audio(
            sample_id,
            SourceRef::Inline,
            Payload::new(bytes, codec),
            audio_meta,
            text.map(str::to_string),
            audio_ts,
        )));
    }
    // audio + 图像组合：仅支持内联单图（其余组合显式报错，不静默降级）
    if has_audio && (b64_multi.is_some() || image_url.is_some() || image_path.is_some()) {
        return skip_or_fail(
            sample_id,
            ErrorClass::SourceCorrupt,
            options,
            error_counts,
            "audio + image reference/multi-image interleave is not supported yet \
             (use inline image_b64 + audio_b64 for a mixed sample)",
        )
        .map(Some);
    }
    // P3b：video 内联解码 + video-only 早退（图像引用不参与）
    let decode_video = || -> Result<(Vec<u8>, Codec), String> {
        let b64 = video_b64.expect("has_video checked");
        let bytes = decode_base64(b64)
            .map_err(|e| format!("base64 video payload could not be decoded: {e}"))?;
        let codec = Codec::from_media_type(
            val.get("video_media_type")
                .and_then(|v| v.as_str())
                .unwrap_or(""),
        );
        Ok((bytes, codec))
    };
    if has_video && !has_image {
        let (bytes, codec) = match decode_video() {
            Ok(v) => v,
            Err(e) => {
                return skip_or_fail(
                    sample_id,
                    ErrorClass::SourceCorrupt,
                    options,
                    error_counts,
                    &e,
                )
                .map(Some);
            }
        };
        // P3c：text+video+timestamp → [Text(0,ts), Video(1,ts)] 交错（字节挂段）；
        // 否则单模态 video 样本（payload=Some + 单 Video 段；伴随文本挂样本级）
        if text_as_segment {
            let segs = vec![
                text_segment(0, text.expect("text_as_segment checked"), audio_ts),
                Segment {
                    kind: SegmentKind::Video,
                    order: 1,
                    len_hint: None,
                    supervise: Supervise::LossOn,
                    source_ref: SourceRef::Inline,
                    timestamp: audio_ts,
                    text: None,
                    payload: Some(Payload::new(bytes, codec)),
                },
            ];
            return Ok(Some(Sample::interleaved(
                sample_id,
                SourceRef::Inline,
                segs,
                video_meta,
            )));
        }
        return Ok(Some(Sample::video(
            sample_id,
            SourceRef::Inline,
            Payload::new(bytes, codec),
            video_meta,
            text.map(str::to_string),
            audio_ts,
        )));
    }
    // video + 图像组合：仅支持内联单图（其余组合显式报错）
    if has_video && (b64_multi.is_some() || image_url.is_some() || image_path.is_some()) {
        return skip_or_fail(
            sample_id,
            ErrorClass::SourceCorrupt,
            options,
            error_counts,
            "video + image reference/multi-image interleave is not supported yet \
             (use inline image_b64 + video_b64 for a mixed sample)",
        )
        .map(Some);
    }

    // 图像行：解析引用 → 载荷。
    // (source_ref, codec, inline_base64)：inline_base64 = Some 时立即解码为压缩字节，
    // None = 引用型（URL / 本地路径），走 resolve_payload。
    // 数组 b64（多图）也归 Inline（字节挂段）。
    let (source_ref, codec_hint, b64_inline): (SourceRef, Codec, Option<String>) =
        if b64_multi.is_some() {
            (SourceRef::Inline, Codec::Unknown, None)
        } else if let Some(b64) = image_b64 {
            let codec = Codec::from_media_type(
                val.get("image_media_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or(""),
            );
            (SourceRef::Inline, codec, Some(b64.to_string()))
        } else if let Some(url) = image_url {
            if let Some(rest) = url.strip_prefix("data:") {
                // data URI：data:[<mediatype>][;base64],<data>
                let (meta_part, data_part) = rest.split_once(',').unwrap_or((rest, ""));
                let media_type = meta_part.split(';').next().unwrap_or("");
                let codec = Codec::from_media_type(media_type);
                (SourceRef::Inline, codec, Some(data_part.to_string()))
            } else {
                (
                    SourceRef::Url(url.to_string()),
                    Codec::from_extension(url),
                    None,
                )
            }
        } else {
            let path = image_path.expect("one of the three image refs must exist");
            (
                SourceRef::LocalPath(path.to_string()),
                Codec::from_extension(path),
                None,
            )
        };

    // P2（mm）：单图行的伴随文本挂样本级 caption（不进段表——P0 单图
    // 段表契约保持不变；vlm encoder 取样本级 text 前置为 Text 段编码）。
    let caption = text.map(str::to_string);

    // base64 内联（image_b64 单值/数组、或 data URI 内嵌）：立即解码为压缩字节。
    // - 单图 → P0 原样（Sample::image，payload=Some，单 Image 段；caption 挂样本级）；
    // - 多图数组 → 交错形态（Sample.payload=None，字节挂段——P2 vlm 编码需要逐图字节；
    //   一期不支持多图 + 伴随文本，caption 存在时报错避免静默丢文本）。
    if b64_inline.is_some() || b64_multi.is_some() {
        let items: Vec<String> =
            b64_multi.unwrap_or_else(|| vec![b64_inline.expect("b64 checked")]);
        if items.len() == 1 {
            // 单图：P0 原样（caption 挂样本级，段表不变）
            let bytes = match decode_base64(&items[0]) {
                Ok(b) => b,
                Err(e) => {
                    return skip_or_fail(
                        sample_id,
                        ErrorClass::SourceCorrupt,
                        options,
                        error_counts,
                        &format!("base64 image payload could not be decoded: {e}"),
                    )
                    .map(Some);
                }
            };
            let codec = if codec_hint == Codec::Unknown {
                sniff_image_codec(&bytes).unwrap_or(Codec::Unknown)
            } else {
                codec_hint
            };
            // P3a：图+音同行 → 交错样本 [Image, Audio]（text 挂样本级 caption，
            // 由 encoder 前置为 Text 段编码——同单图 caption 契约）
            if has_audio {
                let (a_bytes, a_codec) = match decode_audio() {
                    Ok(v) => v,
                    Err(e) => {
                        return skip_or_fail(
                            sample_id,
                            ErrorClass::SourceCorrupt,
                            options,
                            error_counts,
                            &e,
                        )
                        .map(Some);
                    }
                };
                // P3c：text_as_segment 时 Text 段打头（order 0），Image/Audio 顺移
                let (base_order, prefix) = if text_as_segment {
                    (
                        1u32,
                        vec![text_segment(0, text.expect("checked"), audio_ts)],
                    )
                } else {
                    (0u32, Vec::new())
                };
                let mut segs = prefix;
                segs.push(Segment {
                    kind: SegmentKind::Image,
                    order: base_order,
                    len_hint: None,
                    supervise: Supervise::LossOff,
                    source_ref: SourceRef::Inline,
                    timestamp: None,
                    text: None,
                    payload: Some(Payload::new(bytes, codec)),
                });
                segs.push(Segment {
                    kind: SegmentKind::Audio,
                    order: base_order + 1,
                    len_hint: None,
                    supervise: Supervise::LossOn,
                    source_ref: SourceRef::Inline,
                    timestamp: audio_ts,
                    text: None,
                    payload: Some(Payload::new(a_bytes, a_codec)),
                });
                let mut s = Sample::interleaved(
                    sample_id,
                    source_ref,
                    segs,
                    merge_media_meta(image_meta, audio_meta, None),
                );
                s.text = if text_as_segment { None } else { caption };
                return Ok(Some(s));
            }
            // P3b：图+视同行 → 交错样本 [Image, Video]（text 挂样本级 caption，
            // 由 encoder 前置为 Text 段编码——同单图 caption 契约）
            if has_video {
                let (v_bytes, v_codec) = match decode_video() {
                    Ok(v) => v,
                    Err(e) => {
                        return skip_or_fail(
                            sample_id,
                            ErrorClass::SourceCorrupt,
                            options,
                            error_counts,
                            &e,
                        )
                        .map(Some);
                    }
                };
                // P3c：text_as_segment 时 Text 段打头（order 0），Image/Video 顺移
                let (base_order, prefix) = if text_as_segment {
                    (
                        1u32,
                        vec![text_segment(0, text.expect("checked"), audio_ts)],
                    )
                } else {
                    (0u32, Vec::new())
                };
                let mut segs = prefix;
                segs.push(Segment {
                    kind: SegmentKind::Image,
                    order: base_order,
                    len_hint: None,
                    supervise: Supervise::LossOff,
                    source_ref: SourceRef::Inline,
                    timestamp: None,
                    text: None,
                    payload: Some(Payload::new(bytes, codec)),
                });
                segs.push(Segment {
                    kind: SegmentKind::Video,
                    order: base_order + 1,
                    len_hint: None,
                    supervise: Supervise::LossOn,
                    source_ref: SourceRef::Inline,
                    timestamp: audio_ts,
                    text: None,
                    payload: Some(Payload::new(v_bytes, v_codec)),
                });
                let mut s = Sample::interleaved(
                    sample_id,
                    source_ref,
                    segs,
                    merge_media_meta(image_meta, None, video_meta),
                );
                s.text = if text_as_segment { None } else { caption };
                return Ok(Some(s));
            }
            return Ok(Some(Sample::image(
                sample_id,
                source_ref,
                Payload::new(bytes, codec),
                image_meta,
                caption,
            )));
        }
        // 多图：交错（payload=None，字节挂段）
        if caption.is_some() {
            return skip_or_fail(
                sample_id,
                ErrorClass::SourceCorrupt,
                options,
                error_counts,
                "multi-image (image_b64 array) with text caption is not supported yet \
                 (single-text + multi-image interleave needs per-segment text; P3)",
            )
            .map(Some);
        }
        let mut segs: Vec<Segment> = Vec::with_capacity(items.len());
        for b64 in &items {
            let bytes = match decode_base64(b64) {
                Ok(b) => b,
                Err(e) => {
                    return skip_or_fail(
                        sample_id,
                        ErrorClass::SourceCorrupt,
                        options,
                        error_counts,
                        &format!("base64 image payload could not be decoded: {e}"),
                    )
                    .map(Some);
                }
            };
            let codec = if codec_hint == Codec::Unknown {
                sniff_image_codec(&bytes).unwrap_or(Codec::Unknown)
            } else {
                codec_hint
            };
            segs.push(Segment {
                kind: SegmentKind::Image,
                order: segs.len() as u32,
                len_hint: None,
                supervise: Supervise::LossOff,
                source_ref: SourceRef::Inline,
                timestamp: None,
                text: None,
                payload: Some(Payload::new(bytes, codec)),
            });
        }
        return Ok(Some(Sample::interleaved(
            sample_id, source_ref, segs, image_meta,
        )));
    }

    // URL / 本地路径引用：P0 下载（或仅记录引用）。
    // 单图引用 + caption：P0 形态（单 Image 段），caption 挂样本级。
    let payload = match resolve_payload(&source_ref, options) {
        Ok(Some(p)) => p,
        Ok(None) => {
            return Ok(Some(reference_only_sample(
                sample_id, source_ref, image_meta, caption,
            )));
        }
        Err(e) => {
            let class = classify_source_error(&e);
            return skip_or_fail(sample_id, class, options, error_counts, &e.to_string()).map(Some);
        }
    };
    Ok(Some(Sample::image(
        sample_id, source_ref, payload, image_meta, caption,
    )))
}

/// 仅记录引用的样本（download_url_media=false）：payload 为 None，
/// Image 段保留 source_ref（P1 sidecar 落引用，P2 encoder 按需下载）。
/// P2：伴随文本挂样本级 caption（不进段表——P0 单图段表契约不变）。
fn reference_only_sample(
    sample_id: SampleId,
    source_ref: SourceRef,
    media_meta: Option<MediaMeta>,
    caption: Option<String>,
) -> Sample {
    Sample {
        sample_id,
        source_ref: source_ref.clone(),
        payload: None,
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
        decode_policy: crate::builder::sample::DecodePolicy::DecodeInEncoder,
        error_class: None,
        text: caption,
    }
}

/// 失败策略处置：SkipAndCount → 错误样本（计数）；FailFast → Err 中止构建。
fn skip_or_fail(
    sample_id: SampleId,
    class: ErrorClass,
    options: &ReadSamplesOptions,
    error_counts: &mut SourceErrorCounts,
    detail: &str,
) -> std::io::Result<Sample> {
    let policy = options.failure_policy.policy_for(class);
    match policy {
        FailurePolicy::SkipAndCount => {
            error_counts.record(class);
            tracing::debug!(
                "Skipping sample {} ({}) per SkipAndCount policy: {detail}",
                sample_id,
                class.as_str()
            );
            Ok(Sample::errored(sample_id, class))
        }
        FailurePolicy::FailFast => {
            error_counts.fail_fast_triggered += 1;
            let msg = format!(
                "Sample {} failed with {} under FailFast policy: {detail}. \
                 Set the failure policy to SkipAndCount to skip and count such samples.",
                sample_id,
                class.as_str()
            );
            tracing::error!("{msg}");
            Err(std::io::Error::other(msg))
        }
    }
}

/// 把 io 错误归类为 ErrorClass（P0-T5 分类的来源）。
fn classify_source_error(err: &std::io::Error) -> ErrorClass {
    match err.kind() {
        std::io::ErrorKind::NotFound => ErrorClass::SourceNotFound,
        std::io::ErrorKind::InvalidData => ErrorClass::SourceCorrupt,
        _ => ErrorClass::SourceNotFound,
    }
}

/// 解析 `image_meta`（width/height，构建期已知，入 manifest 指纹）。
fn parse_image_meta(v: Option<&sonic_rs::Value>) -> Option<MediaMeta> {
    let v = v?;
    let width = v.get("width").and_then(|x| x.as_u64()).map(|x| x as u32);
    let height = v.get("height").and_then(|x| x.as_u64()).map(|x| x as u32);
    if width.is_none() && height.is_none() {
        return None;
    }
    Some(MediaMeta {
        width,
        height,
        ..Default::default()
    })
}

/// P3a：解析 `audio_meta`（sample_rate / channels / duration_sec；全缺返回 None
/// ——demux 侧对缺失 sr/channels/timestamp fail-fast，不静默写 0）。
fn parse_audio_meta(v: Option<&sonic_rs::Value>) -> Option<MediaMeta> {
    let v = v?;
    let sample_rate = v
        .get("sample_rate")
        .and_then(|x| x.as_u64())
        .map(|x| x as u32);
    let channels = v.get("channels").and_then(|x| x.as_u64()).map(|x| x as u8);
    let duration_sec = v
        .get("duration_sec")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32);
    if sample_rate.is_none() && channels.is_none() && duration_sec.is_none() {
        return None;
    }
    Some(MediaMeta {
        sample_rate,
        channels,
        duration_sec,
        ..Default::default()
    })
}

/// P3b：解析 `video_meta`（fps / n_frames_total / duration_sec；全缺返回 None
/// ——`total_video_seconds` 口径与 fps 兜底统计的源，缺失由 P3b encoder 侧
/// WARN + 默认 fps 记账，source 层不猜）。
fn parse_video_meta(v: Option<&sonic_rs::Value>) -> Option<MediaMeta> {
    let v = v?;
    let fps = v.get("fps").and_then(|x| x.as_f64()).map(|x| x as f32);
    let n_frames_total = v.get("n_frames_total").and_then(|x| x.as_u64());
    let duration_sec = v
        .get("duration_sec")
        .and_then(|x| x.as_f64())
        .map(|x| x as f32);
    if fps.is_none() && n_frames_total.is_none() && duration_sec.is_none() {
        return None;
    }
    Some(MediaMeta {
        fps,
        n_frames_total,
        duration_sec,
        ..Default::default()
    })
}

/// P3a/P3b：合并图像/音频/视频 meta 为一个 `MediaMeta`（同行的交错样本需要
/// 各模态字段；audio 与 video 不同行共存，字段不冲突）。
fn merge_media_meta(
    image: Option<MediaMeta>,
    audio: Option<MediaMeta>,
    video: Option<MediaMeta>,
) -> Option<MediaMeta> {
    let mut out = image.unwrap_or_default();
    let mut any = image.is_some();
    if let Some(a) = audio {
        out.sample_rate = a.sample_rate;
        out.channels = a.channels;
        out.duration_sec = a.duration_sec;
        any = true;
    }
    if let Some(v) = video {
        out.fps = v.fps;
        out.n_frames_total = v.n_frames_total;
        out.duration_sec = v.duration_sec;
        any = true;
    }
    if any { Some(out) } else { None }
}

/// 解析引用型源的载荷：下载 URL / 读本地文件；失败返回 Err（携带可分类的 io::Error）。
///
/// **复用 io 层重试原语**（D5 / P0-T4）：指数退避 + 抖动与云端 shard 下载同一套
/// 语义（见 `io.rs` 的退避常量）。P0 用 `reqwest::blocking`（构建期同步上下文）。
fn resolve_payload(
    source_ref: &SourceRef,
    options: &ReadSamplesOptions,
) -> Result<Option<Payload>, std::io::Error> {
    match source_ref {
        SourceRef::Url(url) => {
            if !options.download_url_media {
                return Ok(None);
            }
            let bytes =
                download_with_retry(url, options.download_attempts, options.cancel.as_ref())?;
            Ok(Some(Payload::new(bytes, Codec::from_extension(url))))
        }
        SourceRef::LocalPath(path) => match std::fs::read(path) {
            Ok(bytes) => Ok(Some(Payload::new(bytes, Codec::from_extension(path)))),
            Err(e) => Err(std::io::Error::new(
                e.kind(),
                format!("failed to read local media file {path}: {e}"),
            )),
        },
        // P4 才落地的源类型：P0 显式报错（不静默跳过）
        SourceRef::TarMember { .. } | SourceRef::ParquetRow { .. } => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!(
                "source type {:?} is not supported until P4 (webdataset/parquet readers)",
                source_ref
            ),
        )),
        SourceRef::Inline => Ok(None),
    }
}

/// URL 下载 + 指数退避重试（复用 io.rs 的退避语义：base=500ms × 2^n + 抖动）。
///
/// **P2-6（评审修复）**：显式 `reqwest::blocking::Client` 配置连接/总超时
/// （旧 `reqwest::blocking::get` 无超时——对端接受连接后不返回响应头/体时
/// 构建可无限等待）；每次重试前检查取消令牌（已取消不再发起新请求）。
fn download_with_retry(
    url: &str,
    attempts: u32,
    cancel: Option<&crate::cancel::CancellationToken>,
) -> Result<Vec<u8>, std::io::Error> {
    /// 连接超时（秒）；对端 TCP 可达但不回包时快速失败。
    const MEDIA_HTTP_CONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    /// 单次尝试总超时（秒）；防止响应体持续但永不结束的挂死。
    const MEDIA_HTTP_TOTAL_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

    fn client() -> &'static reqwest::blocking::Client {
        static CLIENT: std::sync::OnceLock<reqwest::blocking::Client> = std::sync::OnceLock::new();
        CLIENT.get_or_init(|| {
            reqwest::blocking::Client::builder()
                .connect_timeout(MEDIA_HTTP_CONNECT_TIMEOUT)
                .timeout(MEDIA_HTTP_TOTAL_TIMEOUT)
                .build()
                .expect("failed to build blocking HTTP client with timeouts")
        })
    }

    let attempts = attempts.max(1);
    let mut last_err: Option<std::io::Error> = None;
    for attempt in 0..attempts {
        // P2-6：取消检查先于等待/请求——训练中止后不再无谓重试下载。
        if let Some(token) = cancel
            && token.is_cancelled()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                format!("media download canceled: {url}"),
            ));
        }
        if attempt > 0 {
            // 指数退避 + 抖动（与 io.rs RETRY_BACKOFF_BASE 同语义，防止千卡同时重试打爆存储后端）
            let base_ms = 500u64 * (1u64 << (attempt - 1).min(5));
            let jitter = rand::random::<u64>() % base_ms.max(1);
            std::thread::sleep(std::time::Duration::from_millis(base_ms + jitter));
        }
        let result = client()
            .get(url)
            .send()
            .map_err(|e| std::io::Error::other(format!("HTTP download failed: {e}")))
            .and_then(|resp| {
                resp.error_for_status()
                    .map_err(|e| std::io::Error::other(format!("HTTP status error: {e}")))
            })
            .and_then(|ok| {
                ok.bytes().map(|b| b.to_vec()).map_err(|e| {
                    std::io::Error::other(format!("failed to read response body: {e}"))
                })
            });
        match result {
            Ok(bytes) => return Ok(bytes),
            Err(e) => {
                tracing::warn!(
                    "Media download failed (attempt {}/{}): {url} — {e}",
                    attempt + 1,
                    attempts
                );
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| std::io::Error::other("download failed")))
}

/// base64 解码（标准字母表 + 容忍缺失 padding；图像载荷不需要 URL 安全字母表）。
fn decode_base64(s: &str) -> Result<Vec<u8>, String> {
    const INVALID: u8 = 0xFF;
    let mut table = [INVALID; 256];
    for (i, &c) in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        .iter()
        .enumerate()
    {
        table[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for ch in s.bytes() {
        if ch == b'=' || ch.is_ascii_whitespace() {
            continue;
        }
        let v = table[ch as usize];
        if v == INVALID {
            return Err(format!("invalid base64 character {:?}", ch as char));
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

/// 魔数嗅探图像容器（media_type 缺失时的兜底，与 Codec 枚举对应）。
fn sniff_image_codec(bytes: &[u8]) -> Option<Codec> {
    if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some(Codec::Jpeg)
    } else if bytes.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']) {
        Some(Codec::Png)
    } else if bytes.starts_with(b"RIFF") && bytes.len() >= 12 && &bytes[8..12] == b"WEBP" {
        Some(Codec::Webp)
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some(Codec::Gif)
    } else if bytes.starts_with(b"BM") {
        Some(Codec::Bmp)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 收集 read_samples 的全部输出 chunk（同步收通道）。
    fn drain(rx: &flume::Receiver<Vec<Sample>>) -> Vec<Vec<Sample>> {
        let mut out = Vec::new();
        while let Ok(chunk) = rx.try_recv() {
            out.push(chunk);
        }
        out
    }

    /// 写测试用 JSONL（单调序号命名，避免不同测试同行长文件互撞）。
    fn write_tmp(lines: &[String]) -> String {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let path = tmp_dir().join(format!("src_{seq}.jsonl"));
        std::fs::write(&path, lines.join("\n")).unwrap();
        path.to_string_lossy().to_string()
    }

    /// 测试临时目录（同进程内所有测试共享）。
    fn tmp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("megadata_mm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn base_options() -> ReadSamplesOptions {
        ReadSamplesOptions {
            download_url_media: false, // 测试不打网络
            ..Default::default()
        }
    }

    /// P0-T3：base64 大图场景的 chunk 字节预算——8192 条 × 4MB 不得合成一个 chunk。
    #[test]
    fn test_chunk_byte_budget_splits_large_base64_images() {
        // 4MB 压缩字节的 base64 内联图（每条约 5.6MB 源行）
        let payload = vec![0xA5u8; 4 * 1024 * 1024];
        let b64 = base64_encode_for_test(&payload);
        let line = format!(r#"{{"text":"t","image_b64":"{b64}","image_media_type":"image/png"}}"#);
        let n_lines = 100; // 100 × ~4MB = 400MB 远超默认 64MiB 预算
        let lines: Vec<String> = (0..n_lines).map(|_| line.clone()).collect();
        let path = write_tmp(&lines);

        let mut opts = base_options();
        opts.chunk_limit = ChunkLimit::new(8192, 64 * 1024 * 1024); // 条数不设限，只压字节
        let (tx, rx) = flume::bounded(16);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let chunks = drain(&rx);
        let total_samples: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total_samples, n_lines, "所有样本都必须读出（无漏）");
        assert!(
            chunks.len() > 1,
            "64MiB 预算下 100×4MB 必须切成多个 chunk（实测 {} 个）",
            chunks.len()
        );
        // 每个 chunk 的估算字节都不超预算（payload raw_len + 行长度当量）
        for (i, c) in chunks.iter().enumerate() {
            let bytes: u64 = c.iter().map(|s| s.payload_bytes()).sum();
            assert!(
                bytes <= opts.chunk_limit.max_bytes + 4 * 1024 * 1024,
                "chunk {i} 字节 {bytes} 超预算（允许最后一个 chunk 的单条溢出）"
            );
        }
        assert_eq!(errs.total(), 0);
    }

    /// P0-T3：条数约束仍然生效（与旧 read_jsonl_chunks 语义对齐）。
    #[test]
    fn test_chunk_sample_count_limit() {
        let lines: Vec<String> = (0..50)
            .map(|i| format!(r#"{{"text":"doc-{i}"}}"#))
            .collect();
        let path = write_tmp(&lines);
        let mut opts = base_options();
        opts.chunk_limit = ChunkLimit::new(10, 0); // 只限条数
        let (tx, rx) = flume::bounded(16);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let chunks = drain(&rx);
        assert_eq!(chunks.len(), 5, "50 条 / 每 chunk 10 条 = 5 个 chunk");
        for c in &chunks {
            assert_eq!(c.len(), 10);
        }
        // 纯文本行 → Text 段样本
        assert_eq!(chunks[0][0].segments[0].kind, SegmentKind::Text);
    }

    /// P0-T3：三种图像引用形态（base64 / data URI / 本地路径）都能产出 payload。
    #[test]
    fn test_image_ref_forms_all_produce_payload() {
        // 真实 PNG 头的 base64（解码后由 sniff 识别为 Png）
        let png = base64_encode_for_test(&[
            0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n', 0x00, 0x01,
        ]);
        // 预置本地图像文件
        let img_path = tmp_dir().join("img_local.png");
        std::fs::write(&img_path, [0x89u8, b'P', b'N', b'G', 0x00]).unwrap();

        let lines = vec![
            format!(r#"{{"text":"a","image_b64":"{png}","image_media_type":"image/png"}}"#),
            format!(r#"{{"text":"b","image_url":"data:image/jpeg;base64,{png}"}}"#),
            format!(
                r#"{{"text":"c","image_path":"{}","image_meta":{{"width":2,"height":3}}}}"#,
                img_path.display()
            ),
        ];

        let path = write_tmp(&lines);
        let opts = base_options();
        let (tx, rx) = flume::bounded(8);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let chunks = drain(&rx);
        let all: Vec<Sample> = chunks.into_iter().flatten().collect();
        assert_eq!(all.len(), 3);
        // 三种形态 payload 都必须到位（P0 期下载/读取在 source 层完成）
        for s in &all {
            assert!(s.payload.is_some(), "sample {} 应有 payload", s.sample_id);
            assert_eq!(s.segments[0].kind, SegmentKind::Image);
            assert!(s.is_ok());
        }
        assert_eq!(all[0].payload.as_ref().unwrap().codec, Codec::Png);
        assert_eq!(all[1].payload.as_ref().unwrap().codec, Codec::Jpeg); // data URI media_type
        assert_eq!(all[2].payload.as_ref().unwrap().codec, Codec::Png); // 扩展名
        // image_meta 透传
        let meta = all[2].media_meta.expect("image_meta should parse");
        assert_eq!(meta.width, Some(2));
        assert_eq!(meta.height, Some(3));
        assert_eq!(errs.total(), 0);
    }

    /// P0-T4：下载失败 SkipAndCount（默认）→ 错误样本 + 计数，不中止。
    #[test]
    fn test_failed_local_path_skip_and_count() {
        let lines = vec![
            r#"{"text":"ok"}"#.to_string(),
            r#"{"text":"bad","image_path":"/nonexistent/megadata/missing.png"}"#.to_string(),
            r#"{"text":"ok2"}"#.to_string(),
        ];
        let path = write_tmp(&lines);
        let opts = base_options();
        let (tx, rx) = flume::bounded(8);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let all: Vec<Sample> = drain(&rx).into_iter().flatten().collect();
        assert_eq!(all.len(), 3, "跳过样本也占 chunk 位置（error_class 标记）");
        assert!(all[0].is_ok() && all[2].is_ok());
        assert!(!all[1].is_ok());
        assert_eq!(all[1].error_class, Some(ErrorClass::SourceNotFound));
        // 计数入 SourceErrorCounts（P0-T5 的 manifest 来源）
        assert_eq!(errs.skipped_by_class.get("source_not_found"), Some(&1));
        assert_eq!(errs.total(), 1);
    }

    /// P0-T4：FailFast 策略 → 立即 Err 中止构建（不静默继续）。
    #[test]
    fn test_fail_fast_policy_aborts() {
        let lines = vec![
            r#"{"text":"ok"}"#.to_string(),
            r#"{"text":"bad","image_path":"/nonexistent/megadata/missing.png"}"#.to_string(),
        ];
        let path = write_tmp(&lines);
        let mut opts = base_options();
        opts.failure_policy.source_not_found = FailurePolicy::FailFast;
        let (tx, _rx) = flume::bounded(8);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        let result = read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress);
        assert!(result.is_err(), "FailFast 必须返回 Err");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("FailFast"), "错误信息应指明策略: {msg}");
        assert_eq!(errs.fail_fast_triggered, 1);
    }

    /// P0-T4：坏 base64 → SourceCorrupt 分类（不是 NotFound）。
    #[test]
    fn test_corrupt_base64_classified_as_source_corrupt() {
        let lines = vec![r#"{"text":"bad","image_b64":"!!!not-base64!!!"}"#.to_string()];
        let path = write_tmp(&lines);
        let opts = base_options();
        let (tx, rx) = flume::bounded(8);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let all: Vec<Sample> = drain(&rx).into_iter().flatten().collect();
        assert_eq!(all[0].error_class, Some(ErrorClass::SourceCorrupt));
        assert_eq!(errs.skipped_by_class.get("source_corrupt"), Some(&1));
    }

    /// P0-T6：sample_id 按输入行号分配——跳过的样本也占 id（行号 ↔ id 稳定对应）。
    #[test]
    fn test_sample_id_tracks_input_row_order() {
        let lines = vec![
            r#"{"text":"row0"}"#.to_string(),
            r#"{"text":"bad","image_path":"/nonexistent/x.png"}"#.to_string(),
            r#"{"text":"row2"}"#.to_string(),
        ];
        let path = write_tmp(&lines);
        let opts = base_options();
        let (tx, rx) = flume::bounded(8);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let all: Vec<Sample> = drain(&rx).into_iter().flatten().collect();
        let ids: Vec<u64> = all.iter().map(|s| s.sample_id.0).collect();
        assert_eq!(ids, vec![0, 1, 2], "行号 ↔ id 严格对应（含被跳过行）");
        // 无漏无重：allocated == 行数
        assert_eq!(id_alloc.allocated(), 3);
    }

    /// P0-T6：多输入文件场景 id 连续不重复（跨文件全局递增）。
    #[test]
    fn test_sample_id_continuous_across_files() {
        let p1 = write_tmp(&[r#"{"text":"a"}"#.to_string(), r#"{"text":"b"}"#.to_string()]);
        let p2 = write_tmp(&[r#"{"text":"c"}"#.to_string()]);
        let opts = base_options();
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();
        let (tx, rx) = flume::bounded(8);

        read_samples(
            &p1,
            &opts,
            &id_alloc,
            &mut errs,
            tx.clone(),
            progress.clone(),
        )
        .unwrap();
        read_samples(&p2, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let all: Vec<Sample> = drain(&rx).into_iter().flatten().collect();
        let ids: Vec<u64> = all.iter().map(|s| s.sample_id.0).collect();
        assert_eq!(ids, vec![0, 1, 2], "跨文件 id 连续（同一分配器）");
    }

    /// download_url_media=false：URL 引用不下载，产出 reference-only 样本。
    #[test]
    fn test_reference_only_mode_skips_download() {
        let lines = vec![r#"{"text":"ref","image_url":"https://example.com/a.png"}"#.to_string()];
        let path = write_tmp(&lines);
        let opts = base_options(); // download_url_media=false
        let (tx, rx) = flume::bounded(8);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();

        read_samples(&path, &opts, &id_alloc, &mut errs, tx, progress).unwrap();

        let all: Vec<Sample> = drain(&rx).into_iter().flatten().collect();
        assert_eq!(all.len(), 1);
        assert!(all[0].payload.is_none(), "reference-only 模式不下载");
        assert_eq!(
            all[0].segments[0].source_ref,
            SourceRef::Url("https://example.com/a.png".to_string()),
            "引用必须保留在段上（P1 sidecar 落引用）"
        );
        assert!(all[0].is_ok());
        assert_eq!(errs.total(), 0);
    }

    /// base64 解码器：标准字母表 + padding 容忍 + 非法字符报错。
    #[test]
    fn test_base64_decoder() {
        // "hello" → "aGVsbG8="
        assert_eq!(decode_base64("aGVsbG8=").unwrap(), b"hello");
        // 缺 padding 容忍
        assert_eq!(decode_base64("aGVsbG8").unwrap(), b"hello");
        // 空白容忍
        assert_eq!(decode_base64("aGV sbG8=").unwrap(), b"hello");
        // 非法字符
        assert!(decode_base64("aGV*bG8=").is_err());
        // 空串
        assert_eq!(decode_base64("").unwrap(), Vec::<u8>::new());
    }

    /// 图像魔数嗅探（media_type 缺失兜底）。
    #[test]
    fn test_sniff_image_codec() {
        assert_eq!(
            sniff_image_codec(&[0xFF, 0xD8, 0xFF, 0xE0]),
            Some(Codec::Jpeg)
        );
        assert_eq!(
            sniff_image_codec(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']),
            Some(Codec::Png)
        );
        assert_eq!(sniff_image_codec(b"RIFF????WEBPVP8 "), Some(Codec::Webp));
        assert_eq!(sniff_image_codec(b"GIF89a..."), Some(Codec::Gif));
        assert_eq!(sniff_image_codec(&[0, 1, 2, 3]), None);
    }

    /// 测试辅助：标准 base64 编码（只用于造测试数据）。
    fn base64_encode_for_test(bytes: &[u8]) -> String {
        const TABLE: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in bytes.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
            out.push(TABLE[(n >> 18) as usize & 63] as char);
            out.push(TABLE[(n >> 12) as usize & 63] as char);
            if chunk.len() > 1 {
                out.push(TABLE[(n >> 6) as usize & 63] as char);
            } else {
                out.push('=');
            }
            if chunk.len() > 2 {
                out.push(TABLE[n as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
        out
    }

    // ------------------------------------------------------------------
    // P3a：audio 行解析（audio_b64 内联 + timestamp + audio_meta）
    // ------------------------------------------------------------------

    /// audio-only 行 → 单 Audio 段样本（payload=Some、timestamp 挂段、meta 带出）。
    #[test]
    fn test_parse_audio_only_line() {
        let payload = vec![0xF0u8; 300];
        let b64 = base64_encode_for_test(&payload);
        let line = format!(
            r#"{{"text":"caption","audio_b64":"{b64}","audio_media_type":"audio/flac","audio_meta":{{"sample_rate":16000,"channels":1,"duration_sec":2.5}},"timestamp":1.5}}"#
        );
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(0), &base_options(), &mut errs)
            .unwrap()
            .expect("audio line must parse");
        // P3c：text+audio+timestamp → Text 段打头的交错样本
        assert!(s.payload.is_none(), "交错样本字节挂段");
        assert_eq!(s.segments.len(), 2);
        assert_eq!(s.segments[0].kind, SegmentKind::Text);
        assert_eq!(
            s.segments[0].timestamp,
            Some(1.5),
            "timestamp 挂 Text 段（P3c 对齐轴）"
        );
        assert_eq!(s.segments[0].text.as_deref(), Some("caption"));
        let seg = &s.segments[1];
        assert_eq!(seg.kind, SegmentKind::Audio);
        assert_eq!(seg.timestamp, Some(1.5), "timestamp 挂 audio 段（DP2）");
        assert_eq!(seg.payload.as_ref().unwrap().raw_len, 300);
        assert_eq!(seg.payload.as_ref().unwrap().codec, Codec::AudioFlac);
        assert!(s.text.is_none(), "text 已升为段，不再挂样本级 caption");
        let meta = s.media_meta.expect("audio_meta 必须带出");
        assert_eq!(meta.sample_rate, Some(16000));
        assert_eq!(meta.channels, Some(1));
        assert_eq!(meta.duration_sec, Some(2.5));
        assert_eq!(errs.total(), 0);
        // total_payload_bytes 覆盖单模态 payload
        assert_eq!(s.total_payload_bytes(), 300);
    }

    /// 图+音同行 → 交错样本 [Image, Audio]；caption 挂样本级；两侧 meta 合并。
    #[test]
    fn test_parse_image_plus_audio_interleaved() {
        let img = vec![0x89u8; 64]; // 伪 PNG 字节（嗅探失败 → Unknown，无妨）
        let aud = vec![0x11u8; 128];
        let img_b64 = base64_encode_for_test(&img);
        let aud_b64 = base64_encode_for_test(&aud);
        let line = format!(
            r#"{{"text":"mixed","image_b64":"{img_b64}","audio_b64":"{aud_b64}","audio_media_type":"audio/flac","image_meta":{{"width":16,"height":16}},"audio_meta":{{"sample_rate":8000,"channels":2,"duration_sec":1.0}},"timestamp":0.25}}"#
        );
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(1), &base_options(), &mut errs)
            .unwrap()
            .expect("mixed line must parse");
        assert!(s.payload.is_none(), "交错样本字节挂段");
        assert_eq!(s.segments.len(), 3, "Text(0) + Image(1) + Audio(2)");
        assert_eq!(s.segments[0].kind, SegmentKind::Text);
        assert_eq!(s.segments[0].timestamp, Some(0.25));
        assert_eq!(s.segments[0].text.as_deref(), Some("mixed"));
        assert_eq!(s.segments[1].kind, SegmentKind::Image);
        assert_eq!(s.segments[2].kind, SegmentKind::Audio);
        assert_eq!(s.segments[2].timestamp, Some(0.25));
        assert_eq!(s.segments[2].payload.as_ref().unwrap().raw_len, 128);
        assert!(s.text.is_none(), "text 已升为段");
        let meta = s.media_meta.expect("merged meta");
        assert_eq!(meta.width, Some(16), "图像侧字段保留");
        assert_eq!(meta.sample_rate, Some(8000), "音频侧字段合并");
        assert_eq!(meta.channels, Some(2));
        assert_eq!(meta.duration_sec, Some(1.0));
        assert_eq!(errs.total(), 0);
        // total_payload_bytes 覆盖段级字节（D5 预算不漏计）
        assert_eq!(s.total_payload_bytes(), 64 + 128);
        assert!(s.validate_segments().is_ok());
    }

    /// P4：SourceFormat::detect 路由（扩展名大小写不敏感；未知扩展名硬报错）。
    #[test]
    fn test_source_format_detect_routing() {
        assert_eq!(
            SourceFormat::detect("a.jsonl").unwrap(),
            SourceFormat::Jsonl
        );
        assert_eq!(SourceFormat::detect("A.JSON").unwrap(), SourceFormat::Jsonl);
        assert_eq!(SourceFormat::detect("s.tar").unwrap(), SourceFormat::Tar);
        assert_eq!(SourceFormat::detect("s.TAR.GZ").unwrap(), SourceFormat::Tar);
        assert_eq!(SourceFormat::detect("s.tgz").unwrap(), SourceFormat::Tar);
        assert_eq!(
            SourceFormat::detect("s.parquet").unwrap(),
            SourceFormat::Parquet
        );
        let e = SourceFormat::detect("data.txt").unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
        assert!(e.to_string().contains("unsupported source format"), "{e}");
        // read_samples 分派：未知扩展名 → 显式错误（不静默按 JSONL 读）
        let dir = tmp_dir();
        let p = dir.join("data.txt");
        std::fs::write(&p, "{\"text\":\"x\"}\n").unwrap();
        let (tx, _rx) = flume::bounded(4);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();
        let e = read_samples(
            p.to_str().unwrap(),
            &base_options(),
            &id_alloc,
            &mut errs,
            tx,
            progress,
        )
        .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }

    /// P4：read_samples 对 JSONL 的行为逐字保持（分派器重构回归锚点）。
    #[test]
    fn test_read_samples_jsonl_dispatch_unchanged() {
        let dir = tmp_dir();
        let p = dir.join("ok.jsonl");
        std::fs::write(&p, "{\"text\":\"hello\"}\n{\"text\":\"world\"}\n").unwrap();
        let (tx, rx) = flume::bounded(4);
        let id_alloc = SampleIdAllocator::new();
        let mut errs = SourceErrorCounts::default();
        let progress = crate::builder::pipeline::BuildProgress::new();
        read_samples(
            p.to_str().unwrap(),
            &base_options(),
            &id_alloc,
            &mut errs,
            tx,
            progress,
        )
        .unwrap();
        let chunks = drain(&rx);
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, 2);
        assert_eq!(chunks[0][0].sample_id, SampleId(0));
        assert_eq!(chunks[0][1].sample_id, SampleId(1));
        assert_eq!(errs.total(), 0);
    }

    /// audio 引用/数组形态 → 显式 SourceCorrupt（不静默降级）+ 计数 + 占 id。
    #[test]
    fn test_parse_audio_unsupported_forms_fail_loudly() {
        let mut errs = SourceErrorCounts::default();
        let cases = [
            r#"{"audio_url":"https://x/a.flac"}"#.to_string(),
            r#"{"audio_path":"/x/a.flac"}"#.to_string(),
            r#"{"audio_b64":["aaaa","bbbb"]}"#.to_string(),
            // 图 + 音频引用组合
            r#"{"image_url":"https://x/a.png","audio_b64":"aaaa"}"#.to_string(),
            // 多图 + 音频组合
            r#"{"image_b64":["aaaa","bbbb"],"audio_b64":"cccc"}"#.to_string(),
        ];
        for (i, line) in cases.iter().enumerate() {
            let s =
                parse_sample_line(line, SampleId(i as u64), &base_options(), &mut errs).unwrap();
            let s = s.expect("跳过样本也推入通道（占 id）");
            assert!(!s.is_ok(), "case {i} must be skipped: {line}");
            assert_eq!(s.error_class, Some(ErrorClass::SourceCorrupt));
        }
        assert_eq!(errs.total(), cases.len() as u64, "每种不合规形态都计数");
    }

    /// audio_meta 缺失 → meta 为 None（demux 侧 fail-fast，source 不静默补默认值）。
    #[test]
    fn test_parse_audio_missing_meta_is_none() {
        let b64 = base64_encode_for_test(&[0xAAu8; 16]);
        let line = format!(r#"{{"audio_b64":"{b64}","timestamp":0.0}}"#);
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(0), &base_options(), &mut errs)
            .unwrap()
            .unwrap();
        assert!(
            s.media_meta.is_none(),
            "audio_meta 缺失 → None（fail-fast 留给 demux）"
        );
        assert_eq!(
            errs.total(),
            0,
            "解析层不报错（缺 meta 是 demux 期 INV-2 硬校验）"
        );
    }

    // ------------------------------------------------------------------
    // P3b：video 行解析（video_b64 内联 + timestamp + video_meta）
    // ------------------------------------------------------------------

    /// video-only 行 → 单 Video 段样本（payload=Some、timestamp 挂段、meta 带出）。
    #[test]
    fn test_parse_video_only_line() {
        let payload = vec![0x00u8, 0x00, 0x00, 0x18]; // 伪 mp4 头（内容无关）
        let b64 = base64_encode_for_test(&payload);
        let line = format!(
            r#"{{"text":"clip","video_b64":"{b64}","video_media_type":"video/mp4","video_meta":{{"fps":30.0,"n_frames_total":900,"duration_sec":30.0}},"timestamp":2.5}}"#
        );
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(0), &base_options(), &mut errs)
            .unwrap()
            .expect("video line must parse");
        // P3c：text+video+timestamp → Text 段打头的交错样本
        assert!(s.payload.is_none(), "交错样本字节挂段");
        assert_eq!(s.segments.len(), 2);
        assert_eq!(s.segments[0].kind, SegmentKind::Text);
        assert_eq!(s.segments[0].timestamp, Some(2.5));
        assert_eq!(s.segments[0].text.as_deref(), Some("clip"));
        let seg = &s.segments[1];
        assert_eq!(seg.kind, SegmentKind::Video);
        assert_eq!(seg.timestamp, Some(2.5), "timestamp 挂 video 段（DP2）");
        assert_eq!(seg.payload.as_ref().unwrap().codec, Codec::VideoMp4);
        assert!(s.text.is_none(), "text 已升为段");
        let meta = s.media_meta.expect("video_meta 必须带出");
        assert_eq!(meta.fps, Some(30.0));
        assert_eq!(meta.n_frames_total, Some(900));
        assert_eq!(meta.duration_sec, Some(30.0));
        assert_eq!(errs.total(), 0);
        assert_eq!(s.total_payload_bytes(), 4);
    }

    /// 图+视同行 → 交错样本 [Image, Video]；caption 挂样本级；两侧 meta 合并。
    #[test]
    fn test_parse_image_plus_video_interleaved() {
        let img = vec![0x89u8; 64];
        let vid = vec![0x00u8; 256];
        let img_b64 = base64_encode_for_test(&img);
        let vid_b64 = base64_encode_for_test(&vid);
        let line = format!(
            r#"{{"text":"mixed","image_b64":"{img_b64}","video_b64":"{vid_b64}","video_media_type":"video/mp4","image_meta":{{"width":16,"height":16}},"video_meta":{{"fps":29.97,"n_frames_total":600,"duration_sec":20.0}},"timestamp":0.75}}"#
        );
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(1), &base_options(), &mut errs)
            .unwrap()
            .expect("mixed line must parse");
        assert!(s.payload.is_none(), "交错样本字节挂段");
        assert_eq!(s.segments.len(), 3, "Text(0) + Image(1) + Video(2)");
        assert_eq!(s.segments[0].kind, SegmentKind::Text);
        assert_eq!(s.segments[0].timestamp, Some(0.75));
        assert_eq!(s.segments[1].kind, SegmentKind::Image);
        assert_eq!(s.segments[2].kind, SegmentKind::Video);
        assert_eq!(s.segments[2].timestamp, Some(0.75));
        assert_eq!(s.segments[2].payload.as_ref().unwrap().raw_len, 256);
        assert!(s.text.is_none(), "text 已升为段");
        let meta = s.media_meta.expect("merged meta");
        assert_eq!(meta.width, Some(16), "图像侧字段保留");
        assert_eq!(meta.fps, Some(29.97), "视频侧字段合并");
        assert_eq!(meta.n_frames_total, Some(600));
        assert_eq!(meta.duration_sec, Some(20.0));
        assert_eq!(errs.total(), 0);
        assert_eq!(s.total_payload_bytes(), 64 + 256);
        assert!(s.validate_segments().is_ok());
    }

    /// video 引用/数组、audio+video 同行 → 显式 SourceCorrupt（不静默降级）。
    #[test]
    fn test_parse_video_unsupported_forms_fail_loudly() {
        let mut errs = SourceErrorCounts::default();
        let cases = [
            r#"{"video_url":"https://x/a.mp4"}"#.to_string(),
            r#"{"video_path":"/x/a.mp4"}"#.to_string(),
            r#"{"video_b64":["aaaa","bbbb"]}"#.to_string(),
            // audio + video 同行（双 AV 模态一期不支持）
            r#"{"audio_b64":"aaaa","video_b64":"bbbb","timestamp":0.0}"#.to_string(),
            // 视频引用 + 内联图组合
            r#"{"image_url":"https://x/a.png","video_b64":"aaaa"}"#.to_string(),
        ];
        for (i, line) in cases.iter().enumerate() {
            let s =
                parse_sample_line(line, SampleId(i as u64), &base_options(), &mut errs).unwrap();
            let s = s.expect("跳过样本也推入通道（占 id）");
            assert!(!s.is_ok(), "case {i} must be skipped: {line}");
            assert_eq!(s.error_class, Some(ErrorClass::SourceCorrupt));
        }
        assert_eq!(errs.total(), cases.len() as u64, "每种不合规形态都计数");
    }

    /// P2-5：媒体字段存在但 JSON 类型不受支持（对象/数字/布尔）→ 显式
    /// SourceCorrupt + 计数；旧实现静默当作字段不存在，整行按纯文本处理。
    #[test]
    fn test_parse_media_field_wrong_type_fails_loudly() {
        let mut errs = SourceErrorCounts::default();
        let cases = [
            r#"{"text":"t","video_b64":{"nested":"value"}}"#.to_string(),
            r#"{"text":"t","audio_b64":123}"#.to_string(),
            r#"{"text":"t","image_b64":{"nested":"value"}}"#.to_string(),
            r#"{"text":"t","image_url":123}"#.to_string(),
            r#"{"text":"t","image_path":false}"#.to_string(),
        ];
        for (i, line) in cases.iter().enumerate() {
            let s =
                parse_sample_line(line, SampleId(i as u64), &base_options(), &mut errs).unwrap();
            let s = s.expect("跳过样本也推入通道（占 id）");
            assert!(!s.is_ok(), "case {i} must be skipped: {line}");
            assert_eq!(s.error_class, Some(ErrorClass::SourceCorrupt), "case {i}");
        }
        assert_eq!(
            errs.total(),
            cases.len() as u64,
            "类型漂移必须计数（不得静默）"
        );
    }

    /// P2-5 边界：JSON null 表示"未提供"，保持既有语义（不报错、按纯文本行）。
    #[test]
    fn test_parse_media_field_null_is_absent() {
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(
            r#"{"text":"t","video_b64":null}"#,
            SampleId(0),
            &base_options(),
            &mut errs,
        )
        .unwrap()
        .expect("null = 未提供，不是损坏");
        assert!(s.is_ok());
        assert_eq!(s.segments.len(), 1, "按纯文本行解析");
        assert_eq!(errs.total(), 0);
    }

    /// text + AV **无 timestamp** → 保持 caption 语义（伴随文本挂样本级，
    /// 不进段表——向后兼容；caption 无时间信息故不入对齐轴）。
    #[test]
    fn test_parse_text_av_without_timestamp_keeps_caption() {
        let b64 = base64_encode_for_test(&[0xF0u8; 64]);
        let line = format!(
            r#"{{"text":"cap","audio_b64":"{b64}","audio_meta":{{"sample_rate":16000,"channels":1}}}}"#
        );
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(0), &base_options(), &mut errs)
            .unwrap()
            .unwrap();
        assert_eq!(
            s.segments.len(),
            1,
            "无 timestamp → 单 Audio 段（caption 不进段表）"
        );
        assert_eq!(s.segments[0].kind, SegmentKind::Audio);
        assert_eq!(s.segments[0].timestamp, None);
        assert_eq!(s.text.as_deref(), Some("cap"), "caption 挂样本级（旧语义）");
        assert_eq!(errs.total(), 0);
    }

    /// video_meta 缺失 → meta 为 None（P3b encoder 侧 WARN + 默认 fps 记账，
    /// source 层不猜）。
    #[test]
    fn test_parse_video_missing_meta_is_none() {
        let b64 = base64_encode_for_test(&[0xBBu8; 16]);
        let line = format!(r#"{{"video_b64":"{b64}","timestamp":0.0}}"#);
        let mut errs = SourceErrorCounts::default();
        let s = parse_sample_line(&line, SampleId(0), &base_options(), &mut errs)
            .unwrap()
            .unwrap();
        assert!(s.media_meta.is_none(), "video_meta 缺失 → None");
        assert_eq!(errs.total(), 0);
    }
}
