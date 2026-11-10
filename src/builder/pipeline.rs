use crate::builder::TokenizedDoc;
use crate::builder::encoded_sample::{
    EncodedSample, EncodedStats, IdRange, MediaRecord, SegmentEntry, VideoMeta,
    validate_modality_id_ranges,
};
use crate::builder::metadata::ModalityIdRange;
use crate::builder::metadata::{DatasetManifest, ShardMeta, StorageFormat, TokenizerFingerprint};
use crate::builder::sample::{
    Codec, Sample, SampleId, SampleIdAllocator, SegmentKind, SourceRef, Supervise,
};
use crate::builder::sidecar::{
    AudioPackedBatch, AudioRecordInput, AudioRecordInputOwned, AudioSidecarWriter,
    ImageRecordInput, MmSidecarWriter, VideoRecordInput, VideoSidecarWriter, aud_flags,
};
use crate::builder::source::{
    FieldMap, ReadSamplesOptions, SourceErrorCounts, SourceFormat, read_jsonl_chunks, read_samples,
};
use crate::builder::template::TemplateEngine;
use crate::builder::writer::{
    ArrowIpcWriter, ArrowShardWriter, IpcCompression, MemmapShardHashes, MemmapWriter,
    ZstdCompression, sft_fixed_width_columns,
};
use crate::cancel::CancellationToken;
use crate::packing::document_aware::{DocumentAwarePacker, PackedBatch};
use crate::packing::greedy::GreedyPacker;
use bbpe_rs::tokenizer::BBPETokenizer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use rayon::prelude::*;
use sonic_rs::{JsonContainerTrait, JsonValueTrait};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// 自动聚合模式下的目标 shard 文件大小（**约 128MiB**，借鉴 MDS 的 shard 粒度，
/// 解决 1 batch = 1 文件导致的文件数爆炸：20 万行 → 5 万+ 个 .bin）。
///
/// 注：历史注释曾写"约 100MB 量级"，与本常量矛盾，已于 20261007 评审统一为 128MiB
/// （README / docs/使用指南 一直是 128MB，以常量为唯一事实来源）。
///
/// ⚠️ PT + zstd 时本阈值按**未压缩**字节计算（`resolve_batches_per_shard` 不感知
/// 压缩率），而实际磁盘占用约为其42%（真实语料实测，见 README）。若需要精确的磁盘
/// 容量规划，PT 数据集请按 0.42 倍估算。
const TARGET_SHARD_BYTES: u64 = 128 * 1024 * 1024;
/// 自动聚合模式下单 shard 的 batch 数上限
const MAX_BATCHES_PER_SHARD: usize = 4096;
/// F11：构建期 shard 数告警的"常见 world_size"参照——shard 数少于它时，
/// world_size ≥ 该值的 DDP 训练会有 rank 分到 0 个 shard（读取端会显式报错）。
const F11_COMMON_WORLD_SIZE: usize = 8;

/// 构建进度（跨线程共享的原子计数器）。Python 端通过 get_progress() 拉取快照。
pub struct BuildProgress {
    pub bytes_read: AtomicU64,
    pub docs_read: AtomicU64,
    pub docs_tokenized: AtomicU64,
    pub shards_written: AtomicU64,
    pub bytes_written: AtomicU64,
    /// F2：SFT 截断可见性——被尾部截断的文档数 / 丢弃 token 数
    pub docs_truncated: AtomicU64,
    pub tokens_dropped_to_truncation: AtomicU64,
    /// F2：labels 全为 -100 的"废样本"文档数（user 段占满 / assistant 段被截掉）
    pub all_masked_docs: AtomicU64,
    /// F7：疑似 turn 标记注入的文档数（token 流中轮次标记数 ≠ 源数据轮次数）
    pub docs_flagged_injection: AtomicU64,
    /// F2-PT：被跨 batch 切分的文档数 / 这些文档的 token 总数
    /// （PT 不截断，但超长文档分段后 varlen 训练会丢跨 batch 上下文，
    /// 详见 20261007 评审 §3.5）。与 SFT 的 `docs_truncated` 语义不同，不可混用。
    pub docs_split: AtomicU64,
    pub tokens_in_split_docs: AtomicU64,
    /// P2-T2：VLM 视觉预算截断的图像段数（max_visual_tokens_per_sample 超限，TruncateAndCount）。
    pub visual_truncated: AtomicU64,
    /// P2-T2：VLM 因行宽/原子性丢弃的 token 数（图像段不跨 row + text 尾部截断）。
    pub vlm_tokens_dropped: AtomicU64,
    /// P3b §6.1：源缺 `MediaMeta.fps` 的 video 段数（encoder 采用构建默认 fps；
    /// 入 manifest `video_fps_defaulted_clips`）。
    pub video_fps_defaulted: AtomicU64,
    /// P3a-2（评审）：空 payload / 零 token 的 audio 段数（入 manifest
    /// `audio_empty_segments`）。
    pub audio_empty_segments: AtomicU64,
    pub start: Instant,
}

impl BuildProgress {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            bytes_read: AtomicU64::new(0),
            docs_read: AtomicU64::new(0),
            docs_tokenized: AtomicU64::new(0),
            shards_written: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            docs_truncated: AtomicU64::new(0),
            tokens_dropped_to_truncation: AtomicU64::new(0),
            all_masked_docs: AtomicU64::new(0),
            docs_flagged_injection: AtomicU64::new(0),
            docs_split: AtomicU64::new(0),
            tokens_in_split_docs: AtomicU64::new(0),
            visual_truncated: AtomicU64::new(0),
            vlm_tokens_dropped: AtomicU64::new(0),
            video_fps_defaulted: AtomicU64::new(0),
            audio_empty_segments: AtomicU64::new(0),
            start: Instant::now(),
        })
    }
}

/// 计算每个 shard 聚合的 batch 数。
/// `requested == 0` 表示自动：按 TARGET_SHARD_BYTES 与单个 batch 的字节量推算，
/// 并 clamp 到 [1, MAX_BATCHES_PER_SHARD]。
fn resolve_batches_per_shard(
    requested: usize,
    batch_size: usize,
    max_seq_len: usize,
    is_sft: bool,
    extra_columns: bool,
) -> usize {
    if requested > 0 {
        return requested.max(1);
    }
    // SFT 定宽列数从 writer 的 schema 定义推导（消除 magic number 与 schema 的
    // 隐式耦合——schema 加列时自动跟随）；PT Memmap 裸流只有 input_ids 一个定宽列。
    let cols = if is_sft { sft_fixed_width_columns() } else { 1 };
    let mut bytes_per_batch = (batch_size
        .saturating_mul(max_seq_len)
        .saturating_mul(4 * cols)) as u64;
    if !is_sft && extra_columns {
        // 文档级旁路列：doc_ids 每 token 4 字节 + loss_weights 每 token 2 字节
        bytes_per_batch += (batch_size.saturating_mul(max_seq_len).saturating_mul(6)) as u64;
    }
    ((TARGET_SHARD_BYTES / bytes_per_batch.max(1)) as usize).clamp(1, MAX_BATCHES_PER_SHARD)
}

/// 真实有效 token 数（不含 padding）：cu_seqlens 末值即 batch 内有效长度；
/// cu_seqlens 缺失时退回满载计数（旧行为，保证不崩）。
fn valid_tokens_of_batch(batch: &PackedBatch) -> u64 {
    match batch.cu_seqlens.last() {
        Some(&valid_len) if valid_len >= 0 => valid_len as u64,
        _ => (batch.batch_size * batch.max_seq_len) as u64,
    }
}

/// shard 收尾：flush 文件 → 计算 sha256 → 记入 manifest。
///
/// 0.3 Tier-0 快赢项：PT 路径走写盘时增量哈希（`MemmapWriter::finish` 直接产出
/// `MemmapShardHashes`），不再回读 5 个旁路文件；Arrow 路径仍回读单个 `.arrow`
/// 文件（1 文件成本可接受，且 IPC 写入含 seek/覆盖，增量哈希 ≠ 落盘哈希）。
///
/// **语义注记**（仅 PT 增量路径）：校验的是"写出的字节"而非"落盘后回读的字节"——
/// 不再覆盖磁盘写坏场景（原回读走的也是 page cache，测不出真实磁盘错误）。
fn finish_shard(
    filename: &str,
    primary_path: &str,
    shard_idx: usize,
    num_batches: usize,
    valid_tokens: u64,
    manifest: &mut DatasetManifest,
    progress: &Arc<BuildProgress>,
    pt_hashes: Option<MemmapShardHashes>,
) -> Result<(), String> {
    let size = std::fs::metadata(primary_path)
        .map_err(|e| format!("Failed to read file metadata (shard {shard_idx}): {e}"))?
        .len();
    let mut hashes = BTreeMap::new();
    match pt_hashes {
        // PT：复用写盘时增量哈希（manifest checksum 值与旧回读一致）
        Some(h) => {
            hashes.insert("sha256".to_string(), h.bin);
            // .cu / .idx 在 MemmapWriter 中恒创建，哈希必然存在
            hashes.insert("sha256:cu".to_string(), h.cu);
            hashes.insert("sha256:idx".to_string(), h.idx);
            // 文档级旁路列：仅在列扩展开启时存在
            if let Some(v) = h.doc_ids {
                hashes.insert("sha256:col_doc_ids".to_string(), v);
            }
            if let Some(v) = h.loss_weights {
                hashes.insert("sha256:col_loss_weights".to_string(), v);
            }
        }
        // Arrow：回读单个 .arrow 文件（无旁路文件）
        None => {
            let sha256 = crate::io::sha256_file_hex(std::path::Path::new(primary_path))
                .map_err(|e| format!("Failed to compute sha256 (shard {shard_idx}): {e}"))?;
            hashes.insert("sha256".to_string(), sha256);
        }
    }

    progress.shards_written.fetch_add(1, Ordering::Relaxed);
    progress.bytes_written.fetch_add(size, Ordering::Relaxed);
    manifest.append_shard(
        ShardMeta {
            shard_id: shard_idx,
            filename: filename.to_string(),
            num_batches,
            size_bytes: size,
            hashes,
            align_aud_range: None,
        },
        valid_tokens,
    );
    Ok(())
}

// ============================================================================
// P2：VLM 构建配置 + 通道包装（vlm 分派接通 run_offline_builder_pipeline）
// ============================================================================

/// VLM 构建配置（P2-T2/T3，由绑定层从 Python 透传）。
///
/// `mode="vlm"` 时生效；`mode` 为 pt/sft 时 `run_offline_builder_pipeline` 不消费本配置。
#[derive(Debug, Clone)]
pub struct VlmBuildConfig {
    pub max_visual_tokens_per_sample: u32,
    /// 图像段默认 label（T3，默认 -100）。
    pub image_label: i32,
    /// 视觉预算超限策略。
    pub visual_budget_policy: crate::packing::modality_aware::BudgetPolicy,
    // ---- P3a：audio 流（R1 路由：GreedyPacker 变长装箱 + ashard_ 独立序列）----
    /// 对齐窗长 `W_ms`（整数毫秒；对齐终稿 DP1 / P3c §3.1 默认 2000）。
    /// **含 audio 段的构建**缺此值（0）= 启动期 fail-fast（终稿 §6.4“对齐参数
    /// 缺失”行，禁静默降级为 G1）；纯文本/图像构建不落 align_fingerprint。
    pub w_ms: u32,
    /// 音频窗内 token 率（tokens/s，Whisper 50 tok/s 类）——切窗的 token↔时间
    /// 映射参数（P3a §4.2）。含 audio 段的构建缺此值（0）→ fail-fast。
    pub audio_window_token_rate: u32,
    /// audio 流 `max_seq_len`（0 = 同 Vlm 流）。
    pub audio_max_seq_len: usize,
    /// audio 流 `batch_size`（0 = 同 Vlm 流）。
    pub audio_batch_size: usize,
    /// audio 流每 shard 的 batch 数（0 = 同 Vlm 流 `batches_per_shard`）。
    pub audio_batches_per_shard: usize,
    /// INV-2 软校验升级开关：同 `(sid, Audio)` 内 seg_order 递增但 start_ms
    /// 倒退时，缺省只 WARN（乱序时间戳在合法语料中真实存在）；true = fail-fast。
    pub audio_strict_monotonic: bool,
    // ---- P3b：video 定帧采样确定性（入 align_fingerprint.video）----
    /// 采样帧数 N（uniform 定帧采样；D9 定宽 visual token 预算）。
    pub video_n_frames: u16,
    /// resize 目标分辨率 `(w, h)`（以指纹为准、rec 透传，P3b §6.2）。
    pub video_resize: (u16, u16),
    /// 采样器版本占位（uniform 策略不消费 seed；未来切换 seeded 随机采样时启用
    /// 且改变指纹，P3b §6.1）。
    pub video_seed: u64,
    /// fps 兜底值（`MediaMeta.fps` 缺失时限频 WARN + 采用此值并计入
    /// `video_fps_defaulted_clips`，P3b §6.1）。
    pub video_fps_default: f32,
    /// Vlm 流 shard 字节均衡上限（`vid_bytes_in_shard >= 此值` 时滚动；
    /// 0 = 不限，回退纯条数判据；默认 512 MiB，P3b §9）。
    pub video_shard_max_bytes: u64,
}

/// 写线程通道项：pt/sft 走 `Legacy`（行为零变更），vlm 走 `Vlm`（主 batch + sidecar 记录）。
///
/// 引入本枚举是为 vlm 通道携带 `VlmPackedBatch`（含图像 sidecar 记录），同时**不改动**
/// `PackedBatch` 结构定义（`greedy.rs`/`document_aware.rs` 一行不动）。pt/sft 路径仅在外层
/// 多包一层 `Legacy(...)`，写入语义逐字节不变。
///
/// `Vlm` 变体承载模态感知装箱的 batch。VR-2 设计修正后不再附带 audio 水位计数——
/// `align_aud_range` 改为 audio 流整体收尾后据 per-batch span 构造性导出，无需逐
/// shard 水位屏障（旧 P3c-1 屏障随游标快照语义一并移除）。
pub enum PackedItem {
    Legacy(PackedBatch),
    Vlm(crate::packing::modality_aware::VlmPackedBatch),
}

// ============================================================================
// P3a：audio demux（EncodedSample 层分流 + 样本时间轴锚定切窗 + rec↔batch 配对）
// ============================================================================

// ============================================================================
// P3c：时间戳 token（ts token）插入——text↔window 对齐的 build 侧落地
// ============================================================================

/// P3c：ts token id 区间预留宽度（对齐终稿 §3.4：最大样本时长 1h / 默认桶宽
/// 2s = 1800 桶，向上取 2 的幂留余量）。超预算 → fail-fast（§6.4）。
pub(crate) const TS_RANGE_SIZE: u32 = 2048;
/// P3c：样本时长上界（毫秒；1 小时）。已知 `duration_sec` 且超此值 →
/// 构建期硬失败（§3.4 预校验层；缺 duration_sec 跳过预检 + WARN）。
pub(crate) const TS_MAX_SAMPLE_DURATION_MS: u64 = 3_600_000;

/// P3c §4.2：text 段 timestamp 暂存映射 + A/V 轨样本集 + 样本时长
/// （encode 前从 `ok_samples` 预提取——encode 后 `Sample` 已消费，且
/// `SegmentEntry` 无 timestamp 字段，终稿 F7）。
///
/// 键值匹配（非位置匹配）：`(sample_id, segment_order) -> start_ms`——
/// 复用 packer 的 `seg_order` 匹配契约（`modality_aware.rs` media_refs 匹配
/// 同源），不新增 encoder 保序假设。
pub(crate) struct TsPreExtract {
    /// Text 段的 `(sid, seg_order) -> start_ms`（round(ts×1000)，DP2/R7）
    pub ts_map: HashMap<(u64, u32), i64>,
    /// 有 Audio/Video 段的样本 sid 集（INV：无 A/V 轨样本不插 ts token）
    pub av_track: HashSet<u64>,
    /// 样本时长（秒；`MediaMeta.duration_sec`，Option——缺失跳过预校验）
    pub durations: HashMap<u64, Option<f32>>,
}

/// P3c §4.2：encode 前预提取（与 `extract_audio_sources` 同点）。
pub(crate) fn extract_ts_pre(samples: &[Sample]) -> TsPreExtract {
    let mut ts_map = HashMap::new();
    let mut av_track = HashSet::new();
    let mut durations = HashMap::new();
    for s in samples {
        if s.segments
            .iter()
            .any(|g| matches!(g.kind, SegmentKind::Audio | SegmentKind::Video))
        {
            av_track.insert(s.sample_id.0);
        }
        durations.insert(s.sample_id.0, s.media_meta.and_then(|m| m.duration_sec));
        for seg in s.segments.iter().filter(|g| g.kind == SegmentKind::Text) {
            if let Some(ts) = seg.timestamp {
                ts_map.insert((s.sample_id.0, seg.order), (ts * 1000.0).round() as i64);
            }
        }
    }
    TsPreExtract {
        ts_map,
        av_track,
        durations,
    }
}

/// P3c §7 软校验 WARN 限流器（数据质量开关：text 段缺 timestamp、duration_sec
/// 缺失——缺省 WARN，可配置升格 fail-fast 的严格模式留 P3c 后续参数）。
#[derive(Default)]
pub(crate) struct TsSoftWarns {
    text_no_ts: AtomicU64,
    duration_missing: AtomicU64,
}

impl TsSoftWarns {
    /// 限频 WARN：同类告警第 1 / 第 100 / 第 10000 … 次各发一次（防刷屏）。
    fn warn_rate_limited(counter: &AtomicU64, msg: &str) {
        let n = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n == 100 || n.is_multiple_of(10_000) {
            tracing::warn!("{msg} (occurrence #{n})");
        }
    }
}

/// P3c §4.3–4.5：在 EncodedSample 层插入 ts token（demux 后、pack 前）。
///
/// 规则：
/// - **样本级硬规则**（终稿 §6.4 R4）：`sid ∉ av_track`（无 A/V 轨）→ 全样本
///   不插（text-only 样本的 sid 在 Vlm 流不可恢复，插了必成孤儿 token）；
/// - **段级规则**：样本的每个 text 段查 `ts_map`，命中 → 在段**第一个 text
///   token 之前**插入 `token_id = base_id + floor(start_ms / bucket_width_ms)`；
/// - 无 timestamp 的 text 段：不插（缺失即语义，§6.3）→ 限频 WARN；
/// - ts token 并入所属 text 段（`token_start == pos` 的段 `token_end += 1`），
///   `label = -100`、loss 权重 0.0（结构 token 不参与监督）；
/// - 预算（§4.5）：恒等层逐 token 校验 `start_ms >= 0` 且
///   `bucket_id < TS_RANGE_SIZE`；预校验层仅 `duration_sec` 已知时校验
///   `T ≤ TS_MAX_SAMPLE_DURATION_MS`（缺失 → WARN + 跳过）。
///
/// 装箱算法零改动：ts token 是 text 段成员，经 packer Text 分支逐 token 复制
/// 自然入行（§4.4 机制论证，F12）。
pub(crate) fn insert_ts_tokens(
    es: &mut EncodedSample,
    pre: &TsPreExtract,
    base_id: i32,
    bucket_width_ms: u32,
    soft_warns: &TsSoftWarns,
) -> Result<(), String> {
    let sid = es.sample_id.0;
    if !pre.av_track.contains(&sid) {
        return Ok(()); // 无 A/V 轨样本：不插（R4 硬规则）
    }
    // 预校验层：样本时长上界（仅 duration_sec 已知时执行）
    if let Some(Some(dur)) = pre.durations.get(&sid) {
        let t_ms = (dur * 1000.0).round() as u64;
        if t_ms > TS_MAX_SAMPLE_DURATION_MS {
            return Err(format!(
                "P3c: sample {sid} duration {t_ms}ms exceeds the max sample duration \
                 {TS_MAX_SAMPLE_DURATION_MS}ms (ts token id budget would overflow)"
            ));
        }
    } else {
        TsSoftWarns::warn_rate_limited(
            &soft_warns.duration_missing,
            &format!(
                "P3c: sample {sid} has no duration_sec: skipping the sample-duration \
                 precheck for the ts token budget (alignment keys remain self-describing)"
            ),
        );
    }
    // 收集插入点（pos = 段首 token 下标, start_ms）——text 段、有 timestamp
    let mut inserts: Vec<(usize, i64)> = Vec::new();
    for seg in es
        .segment_table
        .iter()
        .filter(|e| e.kind == SegmentKind::Text)
    {
        match pre.ts_map.get(&(sid, seg.segment_order)) {
            Some(&start_ms) => inserts.push((seg.token_start as usize, start_ms)),
            None => TsSoftWarns::warn_rate_limited(
                &soft_warns.text_no_ts,
                &format!(
                    "P3c: text segment (sample {sid}, seg_order {}) has no timestamp: \
                     no ts token inserted (missing = valid semantics)",
                    seg.segment_order
                ),
            ),
        }
    }
    if inserts.is_empty() {
        return Ok(());
    }
    // 降序处理：各插入点互不扰动（后插的不影响先插的下标）
    inserts.sort_by_key(|x| std::cmp::Reverse(x.0));
    // loss_weights 物化（None → 按段默认：Text 1.0 / Image|Video 0.0）
    // ——packer Text 分支逐 token 复制，物化后 "None → 默认 1.0" 分支对含 ts
    // token 的样本不可能触发（§4.4 机制论证）。
    if es.loss_weights.is_none() {
        let mut lw = Vec::with_capacity(es.token_seq.len() * 2);
        for seg in &es.segment_table {
            let fill: &[u8] = if seg.kind == SegmentKind::Text {
                &crate::packing::document_aware::F16_ONE_LE
            } else {
                &crate::packing::document_aware::F16_ZERO_LE
            };
            for _ in 0..seg.token_len() {
                lw.extend_from_slice(fill);
            }
        }
        es.loss_weights = Some(lw);
    }
    for (pos, start_ms) in inserts {
        // 恒等层预算校验（§4.5）
        if start_ms < 0 {
            return Err(format!(
                "P3c: text segment timestamp is negative ({start_ms}ms) for sample {sid}"
            ));
        }
        let bucket_id = (start_ms as u64 / bucket_width_ms as u64) as u32;
        if bucket_id >= TS_RANGE_SIZE {
            return Err(format!(
                "P3c: sample {sid} exceeds the ts token id budget: bucket {bucket_id} \
                 >= range size {TS_RANGE_SIZE} (base_id={base_id})"
            ));
        }
        let token_id = base_id + bucket_id as i32;
        es.token_seq.insert(pos, token_id);
        es.labels.insert(pos, -100);
        if let Some(lw) = es.loss_weights.as_mut() {
            // f16 LE 2 字节/token；ts 位打 0.0（splice 一次性插入保序）
            lw.splice(
                pos * 2..pos * 2,
                crate::packing::document_aware::F16_ZERO_LE,
            );
        }
        // 段表重算（F13 连续覆盖不变式）：后移段 +1；pos 命中段 token_end +1
        // （ts token 并入所属 text 段）
        for seg in es.segment_table.iter_mut() {
            if seg.token_start as usize > pos {
                seg.token_start += 1;
                seg.token_end += 1;
            } else if seg.token_start as usize == pos {
                seg.token_end += 1;
            }
        }
    }
    // 复检连续覆盖/长度不变式（失败即 bug，fail-fast）
    es.validate().map_err(|e| {
        format!("P3c: ts token insertion broke the segment invariant (sample {sid}): {e}")
    })
}

/// P3a：audio 段 build 侧元数据（encode 前从 `Sample` 预提取）。
///
/// `start_ms` / `duration_ms` / `sr` / `channels` 只在 Sample 侧可见
/// （`Segment.timestamp` + `MediaMeta`；encode 后 `Sample` 已消费，且
/// `SegmentEntry` 无 timestamp——对齐终稿 F7）⇒ demux 前必须暂存。
/// 键 = `(sample_id, segment_order)`（与 packer 的 `seg_order` 匹配契约同源：
/// 值匹配而非位置匹配，不依赖 encoder 保序）。
struct AudioSegSource {
    /// 整段压缩字节（FLAC/Opus/MP3；不可按毫秒随机切分，P3a §4.4）。
    bytes: Vec<u8>,
    /// 压缩编码标识（入 `align_fingerprint.audio.codec` 集合，DP5）。
    codec: Codec,
    sample_rate: Option<u32>,
    channels: Option<u8>,
    duration_sec: Option<f32>,
    /// 段起点（相对样本起点，秒；对齐终稿 DP2）。
    timestamp: Option<f64>,
}

/// P3a：从 `ok_samples` 预提取 audio 段元数据（demux 前的暂存映射）。
fn extract_audio_sources(samples: &[Sample]) -> HashMap<(u64, u32), AudioSegSource> {
    let mut map = HashMap::new();
    for s in samples {
        for seg in &s.segments {
            if seg.kind != SegmentKind::Audio {
                continue;
            }
            // 段级字节优先，回落样本级 payload（与 sample_to_dict 的段级 bytes 口径一致）
            let bytes = seg.payload.as_ref().or(s.payload.as_ref());
            map.insert(
                (s.sample_id.0, seg.order),
                AudioSegSource {
                    bytes: bytes.map(|p| p.bytes.to_vec()).unwrap_or_default(),
                    codec: bytes.map(|p| p.codec).unwrap_or(Codec::Unknown),
                    sample_rate: s.media_meta.and_then(|m| m.sample_rate),
                    channels: s.media_meta.and_then(|m| m.channels),
                    duration_sec: s.media_meta.and_then(|m| m.duration_sec),
                    timestamp: seg.timestamp,
                },
            );
        }
    }
    map
}

/// P3a：切窗产出的窗口 doc（token 子序列 + 配对 rec）。
#[derive(Debug)]
struct AudioWindowDoc {
    tokens: Vec<i32>,
    rec: AudioRecordInputOwned,
}

/// P3a 软校验（数据质量开关）的限频 WARN 计数器——跨 chunk 共享、compute 线程
/// 独占。缺省 WARN（含 sid + 段定位），每类限频 [`AUDIO_SOFT_WARN_LIMIT`] 条，
/// 其余只计数（脏数据多时不刷屏，项目限频 WARN 纪律）。
#[derive(Debug, Default)]
struct AudioSoftWarns {
    duration_fallback: AtomicU64,
    start_ms_regression: AtomicU64,
    /// P3a-3（评审）：W_ms×rate/1000 < 1 → 所有窗口零 token、audio 静默丢弃。
    zero_token_window: AtomicU64,
}

/// 软 WARN 限频条数（每类）。
const AUDIO_SOFT_WARN_LIMIT: u64 = 5;

impl AudioSoftWarns {
    /// `duration_sec` 缺失 → token 隐含时长回退（终稿 §6.4 软处置，不 fail-fast）。
    fn warn_duration_fallback(&self, sid: u64, seg_order: u32) {
        if self.duration_fallback.fetch_add(1, Ordering::Relaxed) < AUDIO_SOFT_WARN_LIMIT {
            tracing::warn!(
                "INV-2 soft: audio segment (sample {sid}, seg_order {seg_order}) has no duration_sec; \
                 end_ms falls back to the token-implied duration (n_tokens × 1000 / window_token_rate). \
                 Window assignment stays self-consistent, but track-duration upper-bound checks are skipped."
            );
        }
    }

    /// 同 `(sid, Audio)` 内 seg_order 递增但 start_ms 倒退（乱序时间戳在合法语料中
    /// 真实存在——交错对话转写/多说话人字幕；单调性仅具可读性价值）。
    fn warn_start_ms_regression(
        &self,
        sid: u64,
        prev_order: u32,
        prev_start: u32,
        order: u32,
        start: u32,
    ) {
        if self.start_ms_regression.fetch_add(1, Ordering::Relaxed) < AUDIO_SOFT_WARN_LIMIT {
            tracing::warn!(
                "INV-2 soft: start_ms regression {prev_start} → {start} (sample {sid}, \
                 seg_order {prev_order} → {order}). Alignment keys are computed per record and stay \
                 correct; enable strict monotonic mode to fail-fast instead."
            );
        }
    }

    /// P3a-3（评审）：W_ms×rate/1000 < 1 → 所有窗口零 token、audio 全部静默丢弃。
    /// 方案 §4.2 末条要求 WARN（不 fail-fast：极端配置可能合法用于纯占位）。
    fn warn_zero_token_window(&self, sid: u64, seg_order: u32, w_ms: u32, rate: u32) {
        if self.zero_token_window.fetch_add(1, Ordering::Relaxed) < AUDIO_SOFT_WARN_LIMIT {
            tracing::warn!(
                "P3a: W_ms({w_ms}) × window_token_rate({rate}) / 1000 = 0 < 1 token per window \
                 (sample {sid}, seg_order {seg_order}). Every window yields zero tokens — audio will be \
                 silently dropped. Increase W_ms or window_token_rate (or remove audio segments)."
            );
        }
    }
}

/// P3a demux 运行参数（compute 线程构造一次，跨 chunk 复用）。
struct AudioDemuxParams<'a> {
    /// 对齐窗长（整数毫秒）。
    w_ms: u32,
    /// 音频窗内 token 率（tokens/s）。
    window_token_rate: u32,
    /// audio GreedyPacker 容量（= audio_batch_size × audio_max_seq_len）——超长窗
    /// fail-fast 与准入校验的判据（§4.3）。
    capacity: usize,
    /// 单调性倒退是否升格 fail-fast（数据质量开关，缺省 WARN）。
    strict_monotonic: bool,
    /// 窗口 doc uid 分配器（复用全局 `uid_counter`，F13）。
    uid_counter: &'a AtomicU32,
    /// 软校验限频 WARN 计数器。
    soft_warns: &'a AudioSoftWarns,
}

/// P3a §4.1/§4.2：把一段 audio token 序列按**样本时间轴锚定**切窗（每子段恒属
/// 单窗 k：`floor(sub_start/W) == k` 且 `sub_end ≤ (k+1)·W`——join 键
/// `window_index = floor(rec.start_ms / W_ms)` 无歧义）。
///
/// 公式（P3a §4.1，评审 P1-1 修正后的样本时间轴锚定版）：
/// ```text
/// k_min = floor(start_ms / W_ms),  k_max = ceil(end_ms / W_ms) - 1
/// 子段 k: [max(start_ms, k·W), min(end_ms, (k+1)·W))
/// ```
/// token 映射（§4.2）：`tokens_per_ms = window_token_rate / 1000`，
/// `sub_token_start = floor((sub_start - start_ms) × tokens_per_ms)`；
/// **末窗吸收尾差**（`sub_token_end = n_tokens`）；相邻子段共享边界毫秒 ⇒
/// floor 映射相等 ⇒ token 区间首尾相接（拼接全部子段 == 原序列，A9）。
/// 零 token 子段 → 不发射（自然空窗，终稿 §6.4）。
///
/// blob 归属（§4.4）：**首个发射窗** = VALID 携整段压缩字节；后续窗 =
/// `AUDIO_TAIL_WINDOW` 以 `(offset, len)` 引用同组首窗（writer 侧解析）。
/// 注：零 token 的首窗（k=k_min）不发射时，首个发射窗承接字节归属——
/// 否则尾窗将引用不到首窗（writer `seg_ref` 构造性检测会 fail-fast）。
///
/// 校验（P3a §4.3 / §7）：
/// - **超长窗 fail-fast**：单窗 token 数 + eos（GreedyPacker 追加）> `capacity`
///   → 硬失败（含 sid + 窗号 + token 数 + capacity）——跨 batch 切分会使
///   rec↔batch 归属二义（评审 R3 反例 2 的正解）；
/// - **准入条件**：`W_ms × rate / 1000 + 1 ≤ capacity`（构建期配置校验，+1 即
///   eos——恰好等于 capacity 的窗加 eos 后会跨 batch，故按 eos 口径收紧）；
/// - **INV-2 硬校验**：`rec.start_ms ≤ T×1000`（T = 段 duration_sec；缺失 →
///   跳过上界校验 + WARN，终稿 §6.4 软处置）。
fn split_audio_segment_to_windows(
    sid: u64,
    seg_order: u32,
    tokens: &[i32],
    src: &AudioSegSource,
    params: &AudioDemuxParams<'_>,
) -> Result<Vec<AudioWindowDoc>, String> {
    let n_tokens = tokens.len();
    // ---- 前置检查（P3a §7：缺一即无法构造自足的 rec，禁静默）----
    let ts = src.timestamp.ok_or_else(|| {
        format!(
            "audio segment (sample {sid}, seg_order {seg_order}) has no timestamp: \
             start_ms cannot be computed (INV-2 hard — rec 缺/非法 start_ms fail-fast, 终稿 §6.4)"
        )
    })?;
    if params.w_ms == 0 || params.window_token_rate == 0 {
        return Err(format!(
            "audio segment (sample {sid}, seg_order {seg_order}) present but alignment config missing \
             (w_ms={}, audio_window_token_rate={}): builds containing audio must \
             declare W_ms and window_token_rate (P3a §6.3 / 终稿 §6.4 对齐参数缺失 fail-fast)",
            params.w_ms, params.window_token_rate
        ));
    }
    // 准入条件（§4.3）：每窗 token 上界（W×rate/1000）+ eos 必须 ≤ capacity。
    let max_window_tokens = (params.w_ms as u64 * params.window_token_rate as u64) / 1000;
    // P3a-3（评审）：W_ms×rate/1000 < 1 时所有窗口零 token、audio 全部静默丢弃。
    // 方案 §4.2 末条要求 WARN（不 fail-fast：极端配置可能合法用于纯占位）。
    if max_window_tokens == 0 {
        params.soft_warns.warn_zero_token_window(
            sid,
            seg_order,
            params.w_ms,
            params.window_token_rate,
        );
    }
    if max_window_tokens + 1 > params.capacity as u64 {
        return Err(format!(
            "audio window budget exceeds stream capacity: W_ms={} × window_token_rate={} / 1000 + 1(eos) \
             = {} > audio stream capacity {} (sample {sid}, seg_order {seg_order}). \
             Raise audio_max_seq_len/audio_batch_size or lower W_ms/window_token_rate.",
            params.w_ms,
            params.window_token_rate,
            max_window_tokens + 1,
            params.capacity
        ));
    }
    // 空 payload / 零 token 段：SkipAndCount（源数据质量分类，§7）——返回空
    // （不产 rec、不进流）；计数由调用方按 skipped 语义处置。
    if n_tokens == 0 || src.bytes.is_empty() {
        return Ok(Vec::new());
    }
    // N-1 定稿裁定：col_aud_rec 的 sr u32 / channels u16 需落具体值，缺失 =
    // 静默写 0（“静默错数据”）→ 与 timestamp 同类 fail-fast 含 sid。
    let sample_rate = src.sample_rate.ok_or_else(|| {
        format!(
            "audio segment (sample {sid}, seg_order {seg_order}) has no sample_rate: \
             col_aud_rec needs a concrete sr value (fail-fast, no silent zero)"
        )
    })?;
    let channels = src.channels.ok_or_else(|| {
        format!(
            "audio segment (sample {sid}, seg_order {seg_order}) has no channels: \
             col_aud_rec needs a concrete channels value (fail-fast, no silent zero)"
        )
    })?;
    if !ts.is_finite() || ts < 0.0 {
        return Err(format!(
            "audio segment (sample {sid}, seg_order {seg_order}) has invalid timestamp {ts} \
             (must be finite and >= 0, seconds relative to sample start)"
        ));
    }
    // start_ms = round(timestamp × 1000)（DP2/R7：统一整数毫秒，消解浮点边界二义）
    let start_ms = (ts * 1000.0).round() as u64;
    if start_ms > u32::MAX as u64 {
        return Err(format!(
            "audio segment (sample {sid}, seg_order {seg_order}) start_ms {start_ms} overflows u32"
        ));
    }
    let start_ms = start_ms as u32;
    // end_ms 真源（§4.2）：duration_sec 存在 → 段时长；缺失 → token 隐含时长
    // 回退（ceil(n_tokens × 1000 / rate)；终稿 §6.4 软处置，不 fail-fast）。
    let (end_ms, duration_known) = match src.duration_sec {
        Some(d) if d.is_finite() && d >= 0.0 => {
            (start_ms as u64 + (d * 1000.0).round() as u64, true)
        }
        _ => {
            params.soft_warns.warn_duration_fallback(sid, seg_order);
            (
                start_ms as u64
                    + (n_tokens as u64 * 1000).div_ceil(params.window_token_rate as u64),
                false,
            )
        }
    };
    if end_ms > u32::MAX as u64 || end_ms <= start_ms as u64 {
        return Err(format!(
            "audio segment (sample {sid}, seg_order {seg_order}) has invalid duration: \
             end_ms={end_ms} <= start_ms={start_ms} or overflows u32 (INV-2 hard)"
        ));
    }
    // INV-2 硬校验上界：rec.start_ms ≤ T×1000（T = duration_sec；缺失即跳过）
    let t_ms: Option<u64> = if duration_known {
        Some((src.duration_sec.unwrap() * 1000.0).round() as u64)
    } else {
        None
    };

    let k_min = (start_ms / params.w_ms) as u64;
    let k_max = end_ms.div_ceil(params.w_ms as u64) - 1; // ceil(end/W) - 1
    let tokens_per_ms = params.window_token_rate as f64 / 1000.0;
    let mut docs: Vec<AudioWindowDoc> = Vec::new();
    for k in k_min..=k_max {
        let sub_start = (start_ms as u64).max(k * params.w_ms as u64);
        let sub_end = end_ms.min((k + 1) * params.w_ms as u64);
        let mut tok_start = ((sub_start - start_ms as u64) as f64 * tokens_per_ms).floor() as usize;
        let mut tok_end = ((sub_end - start_ms as u64) as f64 * tokens_per_ms).floor() as usize;
        if k == k_max {
            tok_end = n_tokens; // 末窗吸收尾差（§4.2）
        }
        tok_start = tok_start.min(n_tokens);
        tok_end = tok_end.min(n_tokens);
        if tok_end <= tok_start {
            continue; // 零 token 子段 → 自然空窗，不发射 doc/rec
        }
        // 超长窗 fail-fast（§4.3）：单窗 token 数 + eos > capacity → 跨 batch
        // 切分（rec↔batch 归属二义）——硬失败含 sid + 窗号 + token 数 + capacity。
        let window_tokens = tok_end - tok_start;
        if window_tokens as u64 + 1 > params.capacity as u64 {
            return Err(format!(
                "oversized audio window: sample {sid}, seg_order {seg_order}, window k={k} has \
                 {window_tokens} tokens (+1 eos) exceeding audio stream capacity {} — \
                 long-audio windows must not span batches (P3a §4.3 fail-fast)",
                params.capacity
            ));
        }
        if let Some(t) = t_ms
            && sub_start > t
        {
            return Err(format!(
                "INV-2 hard: start_ms {sub_start} > T×1000 {t} (sample {sid}, seg_order {seg_order})"
            ));
        }
        // 首个发射窗 = VALID（携整段字节）；其余 = 尾窗引用
        let is_first = docs.is_empty();
        docs.push(AudioWindowDoc {
            tokens: tokens[tok_start..tok_end].to_vec(),
            rec: AudioRecordInputOwned {
                bytes: if is_first {
                    Some(src.bytes.clone())
                } else {
                    None
                },
                flags: if is_first {
                    aud_flags::VALID
                } else {
                    aud_flags::AUDIO_TAIL_WINDOW
                },
                sample_rate,
                duration_ms: (sub_end - sub_start) as u32,
                channels: channels as u16,
                sample_id: sid,
                seg_order,
                start_ms: sub_start as u32,
            },
        });
    }
    Ok(docs)
}

/// P3a demux 产出（compute 线程消费）。
#[derive(Debug)]
struct AudioDemuxOutput {
    /// Vlm 流侧样本（audio 段已剔除；纯 audio 样本不入列）。
    vlm_samples: Vec<EncodedSample>,
    /// audio 窗口 doc `(tokens, uid, rec)`——喂 GreedyPacker 并维护 `pending`
    /// 配对表（§5.4）。
    window_docs: Vec<(Vec<i32>, u32, AudioRecordInputOwned)>,
    /// 本批进入 audio 流的源段总时长（秒；§6.5 `total_audio_seconds` 计数点）。
    audio_seconds: f64,
    /// P3a-2（评审）：本批空 payload / 零 token 的 audio 段数（SkipAndCount 语义）。
    empty_segments: u64,
}

/// P3a §3：demux 驱动。对一批 `EncodedSample`：audio 段切窗成窗口 doc + rec，
/// 其余段经 [`EncodedSample::filter_audio_segments`] 进 Vlm 流。
///
/// 软校验（§7）：同 `(sid, Audio)` 内按 seg_order 稳定排序后 start_ms 应单调
/// 不减；倒退 → 缺省 WARN（限频），`strict_monotonic` 时升格 fail-fast。
fn demux_audio_samples(
    encoded: Vec<EncodedSample>,
    audio_sources: &HashMap<(u64, u32), AudioSegSource>,
    params: &AudioDemuxParams<'_>,
) -> Result<AudioDemuxOutput, String> {
    let mut out = AudioDemuxOutput {
        vlm_samples: Vec::with_capacity(encoded.len()),
        window_docs: Vec::new(),
        audio_seconds: 0.0,
        empty_segments: 0,
    };
    for es in encoded {
        let sid = es.sample_id.0;
        let audio_segs: Vec<&SegmentEntry> = es
            .segment_table
            .iter()
            .filter(|e| e.kind == SegmentKind::Audio)
            .collect();
        if !audio_segs.is_empty() {
            // 软校验基准：每段首个发射窗的 (seg_order, start_ms)
            let mut seg_starts: Vec<(u32, u32)> = Vec::with_capacity(audio_segs.len());
            for seg in &audio_segs {
                let src = audio_sources
                    .get(&(sid, seg.segment_order))
                    .ok_or_else(|| {
                        format!(
                            "audio segment (sample {sid}, seg_order {}) not found in pre-extracted \
                         sources: the modality encoder must preserve sample_id / segment_order \
                         (P3a pairing contract)",
                            seg.segment_order
                        )
                    })?;
                let tokens = &es.token_seq[seg.token_start as usize..seg.token_end as usize];
                let windows =
                    split_audio_segment_to_windows(sid, seg.segment_order, tokens, src, params)?;
                if windows.is_empty() {
                    out.empty_segments += 1; // P3a-2（评审）：空段计数入 manifest
                    continue; // 跳过段（SkipAndCount 语义：无 doc/rec）
                }
                seg_starts.push((seg.segment_order, windows[0].rec.start_ms));
                // total_audio_seconds：原始段总时长（§6.5；duration_sec 存在用之，
                // 缺失用 token 隐含时长）——每段计一次，不每窗一次。
                out.audio_seconds += match src.duration_sec {
                    Some(d) if d.is_finite() && d >= 0.0 => d as f64,
                    _ => tokens.len() as f64 / params.window_token_rate as f64,
                };
                for w in windows {
                    let uid = params.uid_counter.fetch_add(1, Ordering::Relaxed);
                    out.window_docs.push((w.tokens, uid, w.rec));
                }
            }
            // 软校验：按 seg_order 稳定排序后 start_ms 单调不减（组内切窗发射序
            // 天然有序；段间乱序在合法语料中真实存在 → 缺省只 WARN）。
            seg_starts.sort_by_key(|(order, _)| *order);
            for w in seg_starts.windows(2) {
                let (prev_order, prev_start) = w[0];
                let (order, start) = w[1];
                if start < prev_start {
                    if params.strict_monotonic {
                        return Err(format!(
                            "INV-2 soft (strict): start_ms regression {prev_start} → {start} \
                             (sample {sid}, seg_order {prev_order} → {order})"
                        ));
                    }
                    params
                        .soft_warns
                        .warn_start_ms_regression(sid, prev_order, prev_start, order, start);
                }
            }
        }
        // 剩余段 → Vlm 流（纯 audio 样本 → None，不进 vlm packer）
        if let Some(filtered) = es.filter_audio_segments()? {
            out.vlm_samples.push(filtered);
        }
    }
    Ok(out)
}

/// P3a §5.4：rec↔batch 配对（**容忍 doc 跨 batch 切分**，见下）。
///
/// GreedyPacker 按 PT 语义在 batch 边界切分 doc（"GreedyPacker 一行不动"纪律下
/// 无法避免——方案 §4.3 原设想"超长窗 fail-fast 保证单 doc 不跨 batch"对**边界
/// 切分**不成立：小于 capacity 的 doc 落在批次剩余空间不足时同样被切）。因此
/// rec 归属裁定为：**doc 首段所在的 batch 拥有该 rec**；后续 batch 中同 uid 的
/// 延续段不重复产 rec——`col_aud_cu` 前缀和寻址仍严格成立（每条 rec 有唯一的
/// 起始 batch），blob 字节只写一次（首窗 VALID），P3c 的 `(sid, window_index)`
/// join 只依赖 rec 自描述的 `start_ms`，与 token 跨批无关。
///
/// uid 从未见过 → fail-fast（配对 bug：doc 没进任何 batch 或 doc_ids 缺失）。
fn pair_audio_records(
    batch: &PackedBatch,
    pending: &mut HashMap<u32, AudioRecordInputOwned>,
    claimed: &mut HashSet<u32>,
) -> Result<Vec<AudioRecordInputOwned>, String> {
    let mut recs = Vec::with_capacity(batch.cu_seqlens.len().saturating_sub(1));
    for w in batch.cu_seqlens.windows(2) {
        let (s, e) = (w[0] as usize, w[1] as usize);
        if e <= s {
            continue;
        }
        let uid = batch.doc_ids.get(s).map(|&d| d as u32).ok_or_else(|| {
            format!(
                "audio batch doc_ids missing at token {s}: the audio GreedyPacker must run with \
                 extra_columns=true (doc_ids carries the window doc uid for rec pairing)"
            )
        })?;
        if let Some(rec) = pending.remove(&uid) {
            claimed.insert(uid);
            recs.push(rec);
        } else if claimed.contains(&uid) {
            // 跨 batch 延续段：rec 已归首段所在 batch，不重复产出
            continue;
        } else {
            return Err(format!(
                "audio rec pairing failed: doc uid {uid} (token {s}) not found in pending and never \
                 claimed — the window doc never reached a batch (pairing bug)"
            ));
        }
    }
    Ok(recs)
}

// ============================================================================
// Python 分词回调路径（第三方分词器接入方案 §4.2 / §4.3）
// ============================================================================

/// 相位 T 的批量编码器：持有一个 Python callable，把"整块文本 → token 化文档"的
/// 工作交给第三方分词器（HF fast / remote_code / tiktoken / 自研）。
///
/// **GIL 红线**（§4.2）：本类型是**唯一**获取 GIL 的地方，且只由**编排线程**
/// （pipeline 的计算线程）调用。rayon / tokio 工作线程永远不触碰 Python ——
/// 工作窃取 + GIL 等待 = 死锁（pyo3 官方警告）。
///
/// 相位 R（模板渲染）与相位 P（pack/压缩/写盘）是纯 Rust，运行在 par_iter 里，
/// 天然不持 GIL；因此编排线程在相位 R/P 期间处于"未持 GIL"状态，其它 Python
/// 线程（含主线程轮询进度 / 响应 Ctrl+C）不会被饿死。
pub struct PyBatchEncoder {
    /// `callback(texts: list[str]) -> (list[list[int]], list[list[str]] | None)`
    /// SFT 模式第二项为 tokens；PT 模式为 None。
    callback: Py<PyAny>,
    /// 模式已烘焙进回调（builder.py 侧适配），此处只用于校验返回值。
    is_sft: bool,
    /// manifest 指纹用：Python 侧报告的标识（`name_or_path` / 类名），无则 "opaque"。
    pub identity: Option<String>,
    /// manifest 指纹用：词表大小（不可得则 None）。
    pub vocab_size: Option<u32>,
    /// 自增 chunk 序号，仅用于错误信息定位。
    chunk_seq: AtomicU64,
}

impl PyBatchEncoder {
    pub fn new(
        callback: Py<PyAny>,
        is_sft: bool,
        identity: Option<String>,
        vocab_size: Option<u32>,
    ) -> Self {
        Self {
            callback,
            is_sft,
            identity,
            vocab_size,
            chunk_seq: AtomicU64::new(0),
        }
    }

    /// **相位 T**：单点获取 GIL → 调用 Python 分词器 → 校验返回值 → 释放 GIL。
    ///
    /// 只应由编排线程调用。返回前已完成全部校验，调用方拿到的是干净的所有权数据
    /// （不再引用任何 Python 对象）。
    pub fn encode_batch(&self, texts: Vec<String>) -> Result<Vec<TokenizedDoc>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let want = texts.len();
        let chunk_idx = self.chunk_seq.fetch_add(1, Ordering::Relaxed);
        Python::attach(|py| {
            let inner = || -> PyResult<Vec<TokenizedDoc>> {
                // 相位 T 边界查信号：Ctrl+C 能在此处打断长构建（§4.2 红线 2）。
                // 抛出的 KeyboardInterrupt 会随 Err 上抛 → 现有取消机制接管。
                py.check_signals()?;
                let ret = self.callback.bind(py).call1((texts,))?;
                let (ids_batch, tokens_batch): (Vec<Vec<i64>>, Option<Vec<Vec<String>>>) = ret
                    .extract()
                    .map_err(|e| PyTypeError::new_err(format!(
                        "Python tokenizer callback returned a value of the wrong type (chunk {chunk_idx}): expected \
                         (list[list[int]], list[list[str]] | None), but it cannot be parsed: {e}"
                    )))?;

                if ids_batch.len() != want {
                    return Err(PyValueError::new_err(format!(
                        "Python tokenizer callback returned {} results, expected {} (chunk {chunk_idx})",
                        ids_batch.len(),
                        want
                    )));
                }
                let mut tokens_batch = match (self.is_sft, tokens_batch) {
                    (true, None) => {
                        return Err(PyValueError::new_err(format!(
                            "in SFT mode the Python tokenizer callback must return a second item tokens (chunk {chunk_idx}):\
                             the packer relies on token strings to match turn markers; missing tokens would silently make all labels -100"
                        )));
                    }
                    (_, Some(t)) if t.len() != want => {
                        return Err(PyValueError::new_err(format!(
                            "Python tokenizer callback returned {} tokens, expected {} (chunk {chunk_idx})",
                            t.len(),
                            want
                        )));
                    }
                    (_, Some(t)) => t,
                    (false, None) => vec![Vec::new(); want],
                };

                let mut docs = Vec::with_capacity(want);
                for (doc_idx, ids) in ids_batch.into_iter().enumerate() {
                    let mut out = Vec::with_capacity(ids.len());
                    for v in ids {
                        if v < 0 || v > i32::MAX as i64 {
                            return Err(PyValueError::new_err(format!(
                                "Python tokenizer callback returned an invalid token id {v} (chunk {chunk_idx} / doc {doc_idx}):\
                                 must be non-negative and <= i32::MAX ({max})",
                                max = i32::MAX
                            )));
                        }
                        out.push(v as i32);
                    }
                    docs.push(TokenizedDoc {
                        ids: out,
                        tokens: std::mem::take(&mut tokens_batch[doc_idx]),
                        // uid 由 pack_phase 统一分配（跨 chunk 唯一）
                        uid: 0,
                    });
                }
                Ok(docs)
            };
            // 在仍持有 GIL 时格式化错误（py_err_to_string 需要 GIL 读 traceback）：
            // 避免退出 GIL 后再为错误路径重新 attach 一次。
            inner().map_err(|e| py_err_to_string(py, e))
        })
    }
}

// ============================================================================
// P2-T5：PyModalityEncoder（多模态 Python 编码器，复用 GIL 红线）
// ============================================================================

/// 多模态 Python 编码器（最终方案 §3.2 / D7，P2-T5）。
///
/// callback 签名：`(samples: list[dict]) -> list[dict]`
/// - 输入 dict = [`Sample`] 的 Python 投影（见 [`sample_to_dict`]）；
/// - 输出 dict = [`EncodedSample`] 的 Python 投影（见 [`encoded_sample_from_dict`]）。
///
/// Rust 侧负责 Sample↔dict 与 EncodedSample↔dict 转换 + 不变式校验（含 D6 id 区间
/// 校验：token 必须落入文本 vocab 或某个 modality id 区间，冲突硬失败）。
///
/// **GIL 红线**（与 [`PyBatchEncoder`] 完全一致）：本类型是**唯一**获取 GIL 的点，
/// 且只由编排线程调用。rayon / tokio 工作线程永远不触碰 Python。
///
/// **多进程 fan-out**（R1 最高风险，方案 D7 必选项）：由 Python 侧 callback 实现者
/// 负责——callback 内部可用线程池（PIL decode/resize 释放 GIL，线程池有效）或
/// 多进程（每进程独立 GIL）。Rust 侧单点编排、逐 chunk 调用，不做工作线程 fan-out
/// （工作窃取 + GIL 等待 = 死锁，pyo3 官方警告）。本契约在 builder.py 侧文档化。
pub struct PyModalityEncoder {
    callback: Py<PyAny>,
    pub vocab_size: Option<u32>,
    /// encoder 声明的专属 id 区间（D6）；pipeline 校验与文本 vocab 不相交。
    pub id_ranges: Vec<IdRange>,
    chunk_seq: AtomicU64,
}

impl PyModalityEncoder {
    /// 构造。调用前应由 pipeline 调用 [`validate_modality_id_ranges`] 校验
    /// `id_ranges` 与文本 vocab 不相交（冲突硬失败）。
    pub fn new(callback: Py<PyAny>, vocab_size: Option<u32>, id_ranges: Vec<IdRange>) -> Self {
        Self {
            callback,
            vocab_size,
            id_ranges,
            chunk_seq: AtomicU64::new(0),
        }
    }

    /// 透出 id 区间的 manifest 序列化形态（pipeline 写 manifest 时调用）。
    pub fn manifest_id_ranges(&self) -> Vec<ModalityIdRange> {
        self.id_ranges
            .iter()
            .map(|r| ModalityIdRange {
                start: r.start,
                end: r.end,
                kind: segment_kind_str(r.kind).to_string(),
            })
            .collect()
    }

    /// **相位 T**（vlm）：单点获取 GIL → Sample→dict → 调 Python callback →
    /// dict→EncodedSample → 不变式校验 + D6 id 区间校验 → 释放 GIL。
    ///
    /// 只应由编排线程调用。返回的所有权数据不再引用任何 Python 对象。
    pub fn encode_samples(&self, samples: Vec<Sample>) -> Result<Vec<EncodedSample>, String> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let want = samples.len();
        let chunk_idx = self.chunk_seq.fetch_add(1, Ordering::Relaxed);
        let id_ranges = self.id_ranges.clone();
        let vocab = self.vocab_size;
        Python::attach(|py| {
            let inner = || -> PyResult<Vec<EncodedSample>> {
                py.check_signals()?;
                // Sample → list[dict]
                let py_inputs: Vec<Py<PyDict>> = samples
                    .iter()
                    .map(|s| sample_to_dict(py, s))
                    .collect::<PyResult<_>>()?;
                let py_list = PyList::new(py, py_inputs)?;
                let ret = self.callback.bind(py).call1((py_list,))?;
                let list: Vec<Py<PyDict>> = ret
                    .extract()
                    .map_err(|e| PyTypeError::new_err(format!(
                        "modality encoder callback returned a non-list value (chunk {chunk_idx}): expected list[dict], got: {e}"
                    )))?;
                if list.len() != want {
                    return Err(PyValueError::new_err(format!(
                        "modality encoder callback returned {} samples, expected {} (chunk {chunk_idx})",
                        list.len(),
                        want
                    )));
                }
                let mut out = Vec::with_capacity(want);
                for (i, d) in list.into_iter().enumerate() {
                    let d = d.bind(py);
                    let es =
                        encoded_sample_from_dict(d, chunk_idx, i).map_err(PyValueError::new_err)?;
                    // 不变式自检（Σ segment.token_len == token_seq.len() 等）
                    es.validate().map_err(|e| PyValueError::new_err(format!(
                        "modality encoder output failed invariant (chunk {chunk_idx} / sample {i}): {e}"
                    )))?;
                    // D6 id 区间校验：每个 token 落入文本 vocab 或某 modality 区间，否则硬失败。
                    validate_token_id_ranges(&es.token_seq, &id_ranges, vocab, chunk_idx, i)?;
                    out.push(es);
                }
                Ok(out)
            };
            inner().map_err(|e| py_err_to_string(py, e))
        })
    }
}

/// D6 token id 区间校验（PyModalityEncoder 返回值校验链扩展，P2-T4）：
/// 每个 token id 必须落入 `[0, text_vocab_size)` 或某个 modality id 区间；
/// 既不在文本 vocab 也不在已声明 modality 区间 → 硬失败（不静默放过非法 id）。
fn validate_token_id_ranges(
    tokens: &[i32],
    ranges: &[IdRange],
    text_vocab_size: Option<u32>,
    chunk_idx: u64,
    sample_idx: usize,
) -> PyResult<()> {
    for (t_idx, &tok) in tokens.iter().enumerate() {
        if tok < 0 {
            return Err(PyValueError::new_err(format!(
                "modality encoder returned negative token id {tok} (chunk {chunk_idx} / sample {sample_idx} / token {t_idx})"
            )));
        }
        let in_text = text_vocab_size.is_none_or(|vsz| (tok as u32) < vsz);
        if in_text {
            continue;
        }
        let in_modality = ranges.iter().any(|r| r.contains(tok));
        if !in_modality {
            return Err(PyValueError::new_err(format!(
                "modality encoder returned token id {tok} that is neither in text vocab [0,{vsz}) \
                 nor in any declared modality id range (chunk {chunk_idx} / sample {sample_idx} / token {t_idx}) \
                 — D6 硬失败：所有 visual placeholder id 必须在已声明区间内",
                vsz = text_vocab_size.map_or("?".to_string(), |v| v.to_string())
            )));
        }
    }
    Ok(())
}

/// `SegmentKind` → 稳定字符串键（manifest / Python dict 共用）。
fn segment_kind_str(k: SegmentKind) -> &'static str {
    match k {
        SegmentKind::Text => "text",
        SegmentKind::Image => "image",
        SegmentKind::Audio => "audio",
        SegmentKind::Video => "video",
    }
}
/// 稳定字符串键 → `SegmentKind`（未知键硬失败）。
fn segment_kind_from_str(s: &str) -> Result<SegmentKind, String> {
    match s {
        "text" => Ok(SegmentKind::Text),
        "image" => Ok(SegmentKind::Image),
        "audio" => Ok(SegmentKind::Audio),
        "video" => Ok(SegmentKind::Video),
        other => Err(format!(
            "unknown segment kind {other:?} (expected text/image/audio/video)"
        )),
    }
}

/// `Sample` → Python dict（PyModalityEncoder callback 输入契约）。
///
/// 字段：`sample_id`(int), `source_ref`(str|None), `payload`(bytes|None),
/// `codec`(str|None), `segments`(list[dict]), `media_meta`(dict|None),
/// `error_class`(str|None)。
///
/// P2（mm）段级数据（回调统一取用，避免区分单模态/交错两种形态）：
/// - `segments[].text`：Text 段内容（str|None）；
/// - `segments[].bytes` / `segments[].codec`：段级压缩字节（交错样本挂段；
///   单模态媒体段同步透出样本级 payload 的一份拷贝）。
fn sample_to_dict(py: Python, s: &Sample) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    d.set_item("sample_id", s.sample_id.0)?;
    d.set_item(
        "source_ref",
        match &s.source_ref {
            SourceRef::Url(u) => Some(u.as_str()),
            SourceRef::LocalPath(p) => Some(p.as_str()),
            SourceRef::TarMember { tar, .. } => Some(tar.as_str()),
            SourceRef::ParquetRow { file, .. } => Some(file.as_str()),
            SourceRef::Inline => None,
        },
    )?;
    if let Some(p) = &s.payload {
        d.set_item("payload", PyBytes::new(py, &p.bytes))?;
        d.set_item("codec", codec_str(p.codec))?;
    } else {
        d.set_item("payload", py.None())?;
        d.set_item("codec", py.None())?;
    }
    // P2（mm）：样本级伴随文本（单图样本的 caption；纯文本样本 None——文本在 Text 段）。
    // vlm encoder 取此字段前置为 Text 段编码（不进 Rust 段表，P0 单图契约不变）。
    d.set_item("text", s.text.as_deref())?;
    let segs: Vec<Py<PyDict>> = s
        .segments
        .iter()
        .map(|seg| {
            let sd = PyDict::new(py);
            sd.set_item("kind", segment_kind_str(seg.kind))?;
            sd.set_item("order", seg.order)?;
            sd.set_item("len_hint", seg.len_hint)?;
            sd.set_item("supervise", supervise_str(seg.supervise))?;
            // P3b（方案 §4.3 v2.1）：段级 timestamp 投影——`Segment.timestamp`
            // （段起点/相对样本起点/秒）是 encoder 计算 `start_ms = round(ts×1000)`
            // 的唯一真源（缺失 → fail-fast 含 sid，INV-2）。P3a audio 段同键消费。
            sd.set_item("timestamp", seg.timestamp)?;
            // P2（mm）：段级 text 与压缩字节。单模态样本的媒体段共享样本级
            // payload（一份拷贝），交错样本字节挂段——回调统一读段级 bytes。
            sd.set_item("text", seg.text.as_deref())?;
            let seg_bytes = seg.payload.as_ref().or_else(|| {
                (seg.kind != SegmentKind::Text)
                    .then_some(s.payload.as_ref())
                    .flatten()
            });
            match seg_bytes {
                Some(p) => {
                    sd.set_item("bytes", PyBytes::new(py, &p.bytes))?;
                    sd.set_item("codec", codec_str(p.codec))?;
                }
                None => {
                    sd.set_item("bytes", py.None())?;
                    sd.set_item("codec", py.None())?;
                }
            }
            Ok::<_, PyErr>(sd.into())
        })
        .collect::<PyResult<_>>()?;
    let seg_list = PyList::new(py, segs)?;
    d.set_item("segments", seg_list)?;
    if let Some(mm) = s.media_meta {
        let md = PyDict::new(py);
        md.set_item("width", mm.width)?;
        md.set_item("height", mm.height)?;
        md.set_item("sample_rate", mm.sample_rate)?;
        md.set_item("channels", mm.channels)?;
        md.set_item("duration_sec", mm.duration_sec)?;
        md.set_item("fps", mm.fps)?;
        md.set_item("n_frames_total", mm.n_frames_total)?;
        d.set_item("media_meta", md)?;
    } else {
        d.set_item("media_meta", py.None())?;
    }
    d.set_item("error_class", s.error_class.map(|c| c.as_str()))?;
    Ok(d.into())
}

/// `EncodedSample` ← Python dict（PyModalityEncoder callback 输出契约，PyModalityEncoder 消费）。
fn encoded_sample_from_dict(
    d: &Bound<PyDict>,
    chunk_idx: u64,
    sample_idx: usize,
) -> Result<EncodedSample, String> {
    let get = |key: &str| -> Result<Bound<PyAny>, String> {
        d.get_item(key)
            .map_err(|e| format!("dict access error for {key:?}: {e}"))?
            .ok_or_else(|| format!("modality encoder output missing key {key:?} (chunk {chunk_idx} / sample {sample_idx})"))
    };
    let sample_id = get("sample_id")?
        .extract::<u64>()
        .map_err(|e| format!("sample_id not int: {e}"))?;
    let token_seq = get("token_seq")?
        .extract::<Vec<i64>>()
        .map_err(|e| format!("token_seq not list[int]: {e}"))?
        .into_iter()
        .map(|v| {
            if v < 0 || v > i32::MAX as i64 {
                return Err(format!(
                    "token id {v} out of i32 range (chunk {chunk_idx} / sample {sample_idx})"
                ));
            }
            Ok(v as i32)
        })
        .collect::<Result<Vec<i32>, String>>()?;
    let labels = get("labels")?
        .extract::<Vec<i64>>()
        .map_err(|e| format!("labels not list[int]: {e}"))?
        .into_iter()
        .map(|v| v as i32)
        .collect();
    // segment_table
    let seg_list = get("segment_table")?;
    let seg_iter: Vec<Bound<PyDict>> = seg_list
        .extract()
        .map_err(|e| format!("segment_table not list[dict]: {e}"))?;
    let mut segment_table = Vec::with_capacity(seg_iter.len());
    for sd in &seg_iter {
        let kind = segment_kind_from_str(
            &sd.get_item("kind")
                .map_err(|e| format!("{e}"))?
                .ok_or("segment missing kind")?
                .extract::<String>()
                .map_err(|e| format!("{e}"))?,
        )?;
        let order = sd
            .get_item("order")
            .map_err(|e| format!("{e}"))?
            .ok_or("segment missing order")?
            .extract::<u32>()
            .map_err(|e| format!("{e}"))?;
        let token_start = sd
            .get_item("token_start")
            .map_err(|e| format!("{e}"))?
            .ok_or("segment missing token_start")?
            .extract::<u32>()
            .map_err(|e| format!("{e}"))?;
        let token_end = sd
            .get_item("token_end")
            .map_err(|e| format!("{e}"))?
            .ok_or("segment missing token_end")?
            .extract::<u32>()
            .map_err(|e| format!("{e}"))?;
        let media_record_id = sd
            .get_item("media_record_id")
            .map_err(|e| format!("{e}"))?
            .and_then(|v| v.extract::<u64>().ok());
        let loss_weight = sd
            .get_item("loss_weight")
            .map_err(|e| format!("{e}"))?
            .ok_or("segment missing loss_weight")?
            .extract::<u8>()
            .map_err(|e| format!("{e}"))?;
        segment_table.push(SegmentEntry {
            sample_id: SampleId(sample_id),
            segment_order: order,
            kind,
            token_start,
            token_end,
            media_record_id,
            loss_weight,
        });
    }
    // loss_weights（可选 bytes；契约 `bytes | None`——键存在但值为 Python None
    // 同样表示"未提供"（None = packer 默认），不能落进 extract 报 TypeError）
    let loss_weights = match d.get_item("loss_weights").map_err(|e| format!("{e}"))? {
        Some(v) if !v.is_none() => {
            let b: Vec<u8> = v
                .extract()
                .map_err(|e| format!("loss_weights not bytes: {e}"))?;
            if b.is_empty() { None } else { Some(b) }
        }
        _ => None,
    };
    // stats（可选；缺失则默认 0）
    let stats = match d.get_item("stats").map_err(|e| format!("{e}"))? {
        Some(v) => stats_from_dict(&v),
        None => Ok(EncodedStats::default()),
    }?;
    // media_refs
    let mr_list = get("media_refs")?;
    let mr_iter: Vec<Bound<PyDict>> = mr_list
        .extract()
        .map_err(|e| format!("media_refs not list[dict]: {e}"))?;
    let mut media_refs = Vec::with_capacity(mr_iter.len());
    for md in &mr_iter {
        let kind = segment_kind_from_str(
            &md.get_item("kind")
                .map_err(|e| format!("{e}"))?
                .ok_or("media_ref missing kind")?
                .extract::<String>()
                .map_err(|e| format!("{e}"))?,
        )?;
        let seg_order = md
            .get_item("seg_order")
            .map_err(|e| format!("{e}"))?
            .ok_or("media_ref missing seg_order")?
            .extract::<u32>()
            .map_err(|e| format!("{e}"))?;
        // bytes：契约 `bytes | None`（Python None 同样合法 = 无字节）
        let bytes = match md.get_item("bytes").map_err(|e| format!("{e}"))? {
            Some(v) if !v.is_none() => {
                let b: Vec<u8> = v
                    .extract()
                    .map_err(|e| format!("media_ref bytes not bytes: {e}"))?;
                if b.is_empty() { None } else { Some(b) }
            }
            _ => None,
        };
        let grid_h = md
            .get_item("grid_h")
            .map_err(|e| format!("{e}"))?
            .ok_or("media_ref missing grid_h")?
            .extract::<u16>()
            .map_err(|e| format!("{e}"))?;
        let grid_w = md
            .get_item("grid_w")
            .map_err(|e| format!("{e}"))?
            .ok_or("media_ref missing grid_w")?
            .extract::<u16>()
            .map_err(|e| format!("{e}"))?;
        let flags = md
            .get_item("flags")
            .map_err(|e| format!("{e}"))?
            .ok_or("media_ref missing flags")?
            .extract::<u8>()
            .map_err(|e| format!("{e}"))?;
        // P3b（方案 §4.3）：kind=="video" 时逐键 extract 六个 int 字段构造
        // VideoMeta；任一键缺失/类型错 → 构建失败（错误含 sample 定位，与该
        // 函数现有错误风格一致）。其余 kind 不得携带 video 元数据。
        let video = if kind == SegmentKind::Video {
            let vid_get = |k: &str| -> Result<u32, String> {
                md.get_item(k)
                    .map_err(|e| format!("{e}"))?
                    .ok_or_else(|| format!("video media_ref missing {k} (sample {sample_id})"))?
                    .extract::<u32>()
                    .map_err(|e| format!("video media_ref {k} not int (sample {sample_id}): {e}"))
            };
            Some(VideoMeta {
                n_frames: vid_get("n_frames")? as u16,
                width: vid_get("width")? as u16,
                height: vid_get("height")? as u16,
                fps_milli: vid_get("fps_milli")?,
                start_ms: vid_get("start_ms")?,
                duration_ms: vid_get("duration_ms")?,
            })
        } else {
            None
        };
        media_refs.push(MediaRecord {
            sample_id: SampleId(sample_id),
            seg_order,
            kind,
            bytes,
            grid_h,
            grid_w,
            flags,
            video,
        });
    }
    Ok(EncodedSample {
        sample_id: SampleId(sample_id),
        token_seq,
        segment_table,
        labels,
        loss_weights,
        stats,
        media_refs,
    })
}

fn stats_from_dict(d: &Bound<PyAny>) -> Result<EncodedStats, String> {
    let dd = d
        .extract::<Bound<PyDict>>()
        .map_err(|e| format!("stats not dict: {e}"))?;
    let get = |k: &str| -> u64 {
        dd.get_item(k)
            .ok()
            .and_then(|o| o)
            .and_then(|v| v.extract::<u64>().ok())
            .unwrap_or(0)
    };
    Ok(EncodedStats {
        text_tokens: get("text_tokens"),
        image_tokens: get("image_tokens"),
        audio_tokens: get("audio_tokens"),
        video_tokens: get("video_tokens"),
        images: get("images"),
        audio_clips: get("audio_clips"),
        video_clips: get("video_clips"),
    })
}

fn codec_str(c: Codec) -> &'static str {
    match c {
        Codec::Jpeg => "jpeg",
        Codec::Png => "png",
        Codec::Webp => "webp",
        Codec::Gif => "gif",
        Codec::Bmp => "bmp",
        Codec::AudioFlac => "audio/flac",
        Codec::AudioOpus => "audio/opus",
        Codec::AudioMp3 => "audio/mp3",
        Codec::VideoMp4 => "video/mp4",
        Codec::VideoMkv => "video/mkv",
        Codec::Unknown => "unknown",
    }
}

fn supervise_str(s: Supervise) -> &'static str {
    match s {
        Supervise::LossOn => "on",
        Supervise::LossOff => "off",
        Supervise::LossWeight(_) => "weight",
    }
}

/// 把 `PyErr` 渲染成人类可读文本，**保留 Python 异常类型与 traceback**。
///
/// 回调里用户分词器抛的任何异常都应当可见（"错误信息保留 Python 栈"，§4.3），
/// 否则只剩一句 "构建任务异常: ..." 无从排查。
fn py_err_to_string(py: Python<'_>, e: PyErr) -> String {
    let tb = e
        .traceback(py)
        .and_then(|tb| tb.format().ok())
        .unwrap_or_default();
    let tb = tb.trim_end();
    if tb.is_empty() {
        format!("Python tokenizer callback raised an exception: {e}")
    } else {
        format!("Python tokenizer callback raised an exception: {e}\n{tb}")
    }
}

/// **相位 R**（render，PT）：模板渲染 / 文本抽取。rayon 并行，纯 Rust，不持 GIL。
///
/// PT 模式的"渲染"就是从原始 JSON 取 `text` 字段；保证"文本抽取"这一单一事实源
/// 留在 Rust（F1），Python 路径只是把结果交出去 tokenize，不做双语言复刻。
fn render_phase(raw_chunk: Vec<String>) -> Vec<String> {
    raw_chunk
        .into_par_iter()
        .filter_map(|json_str| {
            let val: sonic_rs::Value = match sonic_rs::from_str(&json_str) {
                Ok(v) => v,
                Err(e) => {
                    tracing::error!("Discarding dirty data (JSON parsing failed): {}", e);
                    return None;
                }
            };
            val.get("text")?.as_str().map(str::to_string)
        })
        .collect()
}

/// **相位 R**（render，SFT 变体）：模板渲染 + 记录源数据轮次数。
///
/// 轮次数是 F7 注入检测的比对基准（token 流中 turn 起始标记的出现次数应与
/// 源数据的轮次数一致），必须在渲染时从原始 JSON 一并取出（渲染后只剩拼接
/// 文本，无法再恢复"源数据有几轮"）。
fn render_phase_sft(engine: &TemplateEngine, raw_chunk: Vec<String>) -> Vec<(String, usize)> {
    raw_chunk
        .into_par_iter()
        .filter_map(|json_str| match engine.render_chat(&json_str, false) {
            Ok(rendered) => Some((rendered, expected_turn_count(&json_str))),
            Err(e) => {
                tracing::error!("Discarding dirty data: {}", e);
                None
            }
        })
        .collect()
}

/// F7：源数据的轮次数 = `messages` 或 `conversations` 数组长度（与 template.rs
/// 的变量映射一致：`messages` 变量兼容两种 key；两者都缺失时为 0）。
fn expected_turn_count(json_str: &str) -> usize {
    let Ok(v) = sonic_rs::from_str::<sonic_rs::Value>(json_str) else {
        return 0;
    };
    let turns = v.get("messages").or_else(|| v.get("conversations"));
    turns
        .and_then(|m| m.as_array())
        .map(|a| a.len())
        .unwrap_or(0)
}

/// F7 WARN 限频：逐条 WARN 最多给这么多条，其余只计数（脏数据多时不刷屏）。
const INJECTION_WARN_LIMIT: u64 = 5;

/// F7 注入检测（SFT）：token 流中 turn 起始标记的出现次数与源数据轮次数不一致
/// → 疑似"用户内容里嵌入了 turn 标记"（分隔符注入：掩码状态机会把它当真实
/// 轮次监督，造成监督泄漏）。
///
/// **不丢弃样本**（误报会伤合法数据——例如模板并非每轮都输出 turn 标记），
/// 只计数（`progress.docs_flagged_injection`）+ 限频 WARN，让风险可见。
fn check_turn_injection(
    tokens: &[String],
    expected_turns: usize,
    turn_start_str: &str,
    flagged: &AtomicU64,
    warned: &AtomicU64,
) -> bool {
    if turn_start_str.is_empty() || expected_turns == 0 {
        return false;
    }
    let seen = tokens.iter().filter(|t| t.contains(turn_start_str)).count();
    if seen == expected_turns {
        return false;
    }
    flagged.fetch_add(1, Ordering::Relaxed);
    if warned.fetch_add(1, Ordering::Relaxed) < INJECTION_WARN_LIMIT {
        tracing::warn!(
            "Possible turn-marker injection in an SFT doc: found {seen} turn-start marker(s) ({turn_start_str:?}) \
             but the source has {expected_turns} turn(s). User content containing the marker string is treated as a \
             real turn and supervised (label leakage). Sanitize/escape such content or tighten the template."
        );
    }
    true
}

/// **相位 P**（pack）：装箱。rayon 之外（贪心串行，与 bbpe 路径一致），纯 Rust，不持 GIL。
fn pack_phase(
    sft_packer: &DocumentAwarePacker,
    pt_packer: &mut GreedyPacker,
    is_sft: bool,
    mut docs: Vec<TokenizedDoc>,
    batch_size: usize,
    extra_columns: bool,
    uid_counter: &AtomicU32,
    progress: &Arc<BuildProgress>,
) -> Vec<PackedBatch> {
    // 文档 uid：构建内全局递增（跨 chunk 唯一），是 `doc_ids` 旁路列的取值来源。
    // 未启用列扩展时 uid 只是走一遍分配流程（成本可忽略），writer 不落盘该列。
    let batches = if is_sft {
        for d in docs.iter_mut() {
            d.uid = uid_counter.fetch_add(1, Ordering::Relaxed);
        }
        sft_packer.pack_chunks(docs, batch_size, extra_columns)
    } else {
        // F3：GreedyPacker 已有状态化——残余缓冲跨 chunk 保留，管线在输入流
        // 结束时统一调用 finish() flush（每个数据集至多一个未满 batch）。
        let pairs: Vec<(Vec<i32>, u32)> = docs
            .into_iter()
            .map(|d| {
                let uid = uid_counter.fetch_add(1, Ordering::Relaxed);
                (d.ids, uid)
            })
            .collect();
        let out = pt_packer.push_docs(pairs);
        // F2-PT（20261007 评审 §3.5）：把本 chunk 的切分统计汇入进度
        // （构建结束统一 WARN + 写入 manifest.global_stats.split_*）。
        // SFT 的截断统计走上面的 `take_truncation_stats`，两者语义不同、不可合并。
        let (split_docs, split_tokens) = pt_packer.take_split_stats();
        if split_docs > 0 {
            progress.docs_split.fetch_add(split_docs, Ordering::Relaxed);
            progress
                .tokens_in_split_docs
                .fetch_add(split_tokens, Ordering::Relaxed);
        }
        out
    };
    // F2：把本 chunk 的截断统计汇入进度（构建结束统一 WARN + 写入 manifest）
    let (truncated_docs, truncated_tokens, all_masked) = sft_packer.take_truncation_stats();
    if truncated_docs > 0 {
        progress
            .docs_truncated
            .fetch_add(truncated_docs, Ordering::Relaxed);
    }
    if truncated_tokens > 0 {
        progress
            .tokens_dropped_to_truncation
            .fetch_add(truncated_tokens, Ordering::Relaxed);
    }
    if all_masked > 0 {
        progress
            .all_masked_docs
            .fetch_add(all_masked, Ordering::Relaxed);
    }
    batches
}

/// 构建期 turn 标记校验（§4.4 / D8）：标记字符串经**当前生效**的分词器编码必须
/// **恰为 1 个 token**；显式传入 id 时必须一致。违反 → 报错。
///
/// 只用于 bbpe 路径（有 tokenizer 在手）；Python 路径由 builder.py 在发起构建
/// 前用 adapter 做同样的校验。
fn validate_turn_markers_bbpe(
    tokenizer: &BBPETokenizer,
    turn_start_str: &str,
    turn_end_str: &str,
    turn_start_id: i32,
    turn_end_id: i32,
) -> Result<(), String> {
    for (marker, explicit) in [(turn_start_str, turn_start_id), (turn_end_str, turn_end_id)] {
        if marker.is_empty() {
            continue;
        }
        let enc = tokenizer.encode_with_types(marker, &HashSet::new(), true);
        if enc.ids.len() != 1 {
            return Err(format!(
                "turn marker {marker:?} encodes to {} tokens ({:?}) via the tokenizer, but it must be exactly 1.\
                 This marker identifies conversation-turn boundaries: multiple tokens would mis-identify assistant spans and\
                 corrupt supervision labels. Check that the marker string exactly matches a tokenizer special symbol.",
                enc.ids.len(),
                enc.tokens
            ));
        }
        if explicit >= 0 && enc.ids[0] != explicit {
            return Err(format!(
                "turn marker {marker:?} encodes to id {} (token {:?}), which does not match the explicitly given id {explicit}.\
                 Check turn_start_id / turn_end_id, or omit them to derive automatically.",
                enc.ids[0], enc.tokens
            ));
        }
    }
    Ok(())
}

/// 把 `tokenizer_path` 解析成实际模型文件路径（目录 → 目录下 `tokenizer.json`）。
pub fn resolve_tokenizer_file(tokenizer_path: &str) -> String {
    if std::fs::metadata(tokenizer_path)
        .map(|m| m.is_dir())
        .unwrap_or(false)
    {
        std::path::Path::new(tokenizer_path)
            .join("tokenizer.json")
            .to_string_lossy()
            .into_owned()
    } else {
        tokenizer_path.to_string()
    }
}

/// bbpe 路径的 manifest 指纹：`sha256(文件)[:16] + "@v" + format_version`。
fn bbpe_tokenizer_fingerprint(
    tokenizer_file: &str,
    tokenizer: &BBPETokenizer,
) -> TokenizerFingerprint {
    let digest = crate::io::sha256_file_hex(std::path::Path::new(tokenizer_file))
        .unwrap_or_else(|_| "unknown".to_string());
    let short: String = digest.chars().take(16).collect();
    let version = tokenizer
        .state
        .format_version
        .map(|v| v.to_string())
        .unwrap_or_else(|| "legacy".to_string());
    TokenizerFingerprint {
        backend: "bbpe".to_string(),
        identity: Some(format!("{short}@v{version}")),
        vocab_size: Some(tokenizer.state.vocab.len() as u32),
    }
}

/// 离线构建流水线。返回 Result<(), String>：任何 IO / 序列化 / panic 失败都映射成
/// P4 §8：云路径预解析——`s3://` / `gs://` / `http(s)://` 前缀的源分片先经
/// io 层落成本地缓存路径（下载/缓存/单飞/sha256 校验全部复用既有原语），
/// `read_samples` 保持纯同步、只读本地路径（与 JSONL 现状一致）。
///
/// **实现约束**：
/// 1. `ensure_local_file_verified` 是 `(base_dir_or_url, filename)` **拼接式**
///    签名（本地 `Path::join`、云端 URL 拼接），而 `input_paths` 是完整路径/URL
///    ——必须按最后一个 `/` 拆分为 base + filename（URL 语义 rsplit，勿用
///    `Path::join` 语义）。
/// 2. 该函数内部 `rt.block_on`——本调用点运行在 tokio 任务内（`io_read_task`），
///    禁止在当前 runtime 上 block_on；用一次性 mini runtime 执行（每个远端源
///    一次，开销可忽略）。
fn resolve_source_to_local(
    io_mgr: &crate::io::CloudIoManager,
    path: &str,
) -> Result<String, String> {
    let is_remote = path.starts_with("s3://")
        || path.starts_with("gs://")
        || path.starts_with("http://")
        || path.starts_with("https://");
    if !is_remote {
        return Ok(path.to_string());
    }
    let (base, filename) = match path.rsplit_once('/') {
        Some((b, f)) if !f.is_empty() => (b.to_string(), f.to_string()),
        _ => {
            return Err(format!(
                "invalid remote source path {path}: cannot split into base and filename"
            ));
        }
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to build runtime for remote source {path}: {e}"))?;
    io_mgr
        .ensure_local_file_verified(&rt, &base, &filename, None)
        .map_err(|e| format!("failed to materialize remote source {path}: {e}"))
}

/// 人类可读的 String，由 Python 绑定层转成 PyIOError 抛出，避免整个 Python 进程 panic。
///
/// `batches_per_shard`：单个 shard 文件聚合的 batch 数（0 = 自动，约 128MB/shard）。
///   解决旧版 1 batch = 1 文件的文件数爆炸；聚合后 shard 数仍由读取端校验
///   `shards >= world_size`（大集群 rank 空转防护，见 stream.rs）。
/// `compression`：Some("zstd"|"zstd:N"|"lz4") 时启用压缩。
///   - PT 模式：per-batch zstd frame（Zstd-MDS 格式）
///   - SFT 模式：Arrow IPC 原生压缩（零格式改动）
///
/// `encoder`：`Some` 时走 **Python 分词回调路径**（第三方分词器）——不再加载
/// bbpe 分词器，tokenize 由回调承担，其余（读取 / 渲染 / pack / 压缩 / 写盘 /
/// 索引）与 bbpe 路径共享。`None` 时为 bbpe 默认路径，行为与性能零改动（D1）。
#[allow(clippy::too_many_arguments)]
pub async fn run_offline_builder_pipeline(
    input_paths: Vec<String>,
    output_dir: String,
    mode: String,
    tokenizer_path: &str,
    template_str: &str,
    max_seq_len: usize,
    batch_size: usize,
    pad_id: i32,
    eos_id: i32,
    turn_start_id: i32,
    turn_end_id: i32,
    turn_start_str: String,
    turn_end_str: String,
    assistant_str: String,
    batches_per_shard: usize,
    compression: Option<String>,
    cancel_token: CancellationToken,
    progress: Arc<BuildProgress>,
    encoder: Option<PyBatchEncoder>,
    bbpe: Option<Arc<BBPETokenizer>>,
    // 是否落盘文档级旁路列（`doc_ids` / `loss_weights`）。关闭时磁盘/内存零额外开销。
    extra_columns: bool,
    // P2：vlm 模式专用——多模态 Python 编码器 + 视觉预算/标签配置。
    // `mode != "vlm"` 时必须为 `None`（pt/sft 零行为变更）。
    modality_encoder: Option<PyModalityEncoder>,
    vlm_config: Option<VlmBuildConfig>,
    // P4：源字段映射（tar/parquet reader 消费；JSONL 不读）。pt/sft 不用。
    field_map: FieldMap,
) -> Result<(), String> {
    tracing::info!(
        "🚀 Starting megadata_rs fast build pipeline [{}] mode...",
        mode.to_uppercase()
    );
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output directory: {e}"))?;

    let is_sft = mode == "sft";
    let is_vlm = mode == "vlm";
    // vlm 模式必须配齐 modality_encoder + vlm_config；非 vlm 模式不得传入（防误用）。
    if is_vlm {
        if modality_encoder.is_none() {
            return Err(
                "vlm mode requires a modality_encoder callback (PyModalityEncoder)".to_string(),
            );
        }
        if vlm_config.is_none() {
            return Err("vlm mode requires vlm_config (max_visual_tokens_per_sample / image_label / budget_policy)".to_string());
        }
    } else if modality_encoder.is_some() || vlm_config.is_some() {
        return Err(format!(
            "modality_encoder / vlm_config only valid in vlm mode (got mode={mode:?}); \
             pt/sft paths must pass None to preserve zero-change behavior"
        ));
    }
    // P2-T4（D6）：modality encoder 的 id 区间与文本 vocab 不相交校验（冲突硬失败）。
    if let Some(me) = &modality_encoder {
        validate_modality_id_ranges(&me.id_ranges, me.vocab_size)
            .map_err(|e| format!("modality id range validation failed (D6): {e}"))?;
    }
    // P2：提前抽取 manifest 透出用 id 区间 + 统计累加器（encoder/config 将 move 进 compute 线程）。
    let modality_id_ranges_for_manifest =
        modality_encoder.as_ref().map(|me| me.manifest_id_ranges());
    // P3c：对齐启用判定 = 声明了 audio/video 专属 id 区间（D6）。启用才落
    // `col_row_sids` 列 + 完整 `align_fingerprint`；不启用（纯文本/图像）产物
    // 与 P2/P3a/P3b 期逐字节一致（A6 兼容锚点）。
    let align_enabled = modality_encoder.as_ref().is_some_and(|me| {
        me.id_ranges
            .iter()
            .any(|r| matches!(r.kind, SegmentKind::Audio | SegmentKind::Video))
    });
    // P3c：ts token 区间基址 = max(模态区间 end) 对齐 256 边界；宽度 2048
    // （对齐终稿 §3.2/§3.4）。与文本 vocab 相交 → 构建期硬失败。
    let ts_base_id: i32 = if align_enabled {
        let me = modality_encoder.as_ref().expect("align_enabled ⇒ encoder");
        let max_end = me.id_ranges.iter().map(|r| r.end).max().unwrap_or(0);
        let base = (max_end + 255) / 256 * 256;
        if let Some(vsz) = me.vocab_size
            && (base as u32) < vsz
        {
            return Err(format!(
                "P3c: alignment-marker ts token base_id {base} overlaps text vocab [0,{vsz}) \
                 (D6 hard fail): raise the modality id ranges above the vocab size"
            ));
        }
        base
    } else {
        0
    };
    // P3c：id 区间快照提前克隆（`modality_id_ranges_for_manifest` 将在 vlm 统计
    // 块 move 进 manifest；指纹的 modality_id_ranges 成员用快照）
    let modality_id_ranges_snapshot = modality_id_ranges_for_manifest.clone();
    let vlm_stats_acc = Arc::new(Mutex::new(EncodedStats::default()));

    // 分词后端二选一：有回调 = Python 路径（不加载 bbpe，也不要求 tokenizer 文件存在）。
    // bbpe 路径的分词器由绑定层加载并缓存后传入，构建期不再二次加载。
    // vlm：编码全在 modality_encoder 回调内（文本+视觉），不需要分词器——
    // 绑定层不加载 bbpe（golden e2e 可完全不依赖 tokenizer）。
    let tokenizer: Option<Arc<BBPETokenizer>> = match (&encoder, is_vlm) {
        (Some(_), _) => None,
        (None, true) => None,
        (None, false) => {
            let tok = bbpe.ok_or_else(|| {
                "internal error: bbpe path did not receive a loaded tokenizer (should be passed in by the binding layer)".to_string()
            })?;
            tracing::info!(
                "🔤 Tokenizer backend: bbpe_rs (native Rust, file {})",
                resolve_tokenizer_file(tokenizer_path)
            );
            Some(tok)
        }
    };

    // 构建期 turn 标记校验（D8）：bbpe 路径在此（手上有 tokenizer）；
    // Python 路径由 builder.py 用 adapter 发起前校验。
    if is_sft && let Some(tok) = &tokenizer {
        validate_turn_markers_bbpe(
            tok,
            &turn_start_str,
            &turn_end_str,
            turn_start_id,
            turn_end_id,
        )?;
    }

    // manifest 分词器指纹（D7）：构建期记录，读取端不校验。
    let tokenizer_fp: Option<TokenizerFingerprint> = match (&tokenizer, &encoder) {
        (Some(tok), _) => {
            let file = resolve_tokenizer_file(tokenizer_path);
            Some(bbpe_tokenizer_fingerprint(&file, tok))
        }
        (None, Some(enc)) => {
            tracing::info!("🔤 Tokenizer backend: python callback (third-party tokenizer)");
            Some(TokenizerFingerprint {
                backend: "python".to_string(),
                identity: enc.identity.clone(),
                vocab_size: enc.vocab_size,
            })
        }
        (None, None) => None,
    };

    let template_engine = Arc::new(
        TemplateEngine::new(template_str)
            .map_err(|e| format!("Template engine initialization failed: {e}"))?,
    );

    let sft_packer = Arc::new(DocumentAwarePacker::new(
        max_seq_len,
        pad_id,
        turn_start_id,
        turn_end_id,
        turn_start_str.clone(),
        turn_end_str,
        assistant_str,
    ));

    // F3：PT packer 有状态（跨 chunk 缓冲），只被 compute 线程独占使用 →
    // 直接 move 进线程，无需 Arc。
    let mut pt_packer = GreedyPacker::new(max_seq_len, eos_id, batch_size, extra_columns);
    // F7 注入检测的 WARN 限频计数器（跨 chunk 共享，只由 compute 线程触及）
    let injection_warned = Arc::new(AtomicU64::new(0));

    let (tx_raw, rx_raw) = flume::bounded::<Vec<String>>(100);
    let (tx_packed, rx_packed) = flume::bounded::<PackedItem>(100);
    // vlm 模式的样本通道（Vec<Sample>，经 read_samples 产出；pt/sft 不用）。
    let (tx_samples, rx_samples) = flume::bounded::<Vec<Sample>>(8);
    // P3a：audio 流独立通道（AudioPackedBatch = 主 batch + 配对 rec；**不混入
    // tx_packed**——单通道混流会污染 Vlm 流 batches_in_shard 计数、破坏 shard
    // 边界，F4）。pt/sft 恒 None。
    let (tx_audio_packed, rx_audio_packed) = flume::bounded::<AudioPackedBatch>(100);
    // VR-2 / P3c 设计修正（取代旧"游标快照 + 水位屏障"）：audio 写线程每写一批
    // 追加一条 `AudioBatchSpan`（本批 rec 的 [min_sid,max_sid] + (ashard,batch)
    // 坐标）。Vlm 写线程在 `aud_stream_done` 后据此为每个 shard **构造性导出**
    // `align_aud_range`（= 本 shard 样本 rec 所在的最小/最大 audio batch），统一
    // 消除旧快照语义的 `first` 非超集（§10.6(3)）与 VR-2 的 `last` 漏覆盖两缺口。
    let aud_spans: Arc<Mutex<Vec<AudioBatchSpan>>> = Arc::new(Mutex::new(Vec::new()));
    let aud_spans_for_audio = aud_spans.clone();
    let aud_spans_for_write = aud_spans.clone();
    // P3c：audio 流整体收尾标志（finish flush 后置位）。deferred 导出**只在主循环
    // 外等它一次**——此时 `rx_packed` 已断流 ⇒ compute 结束并 drop `tx_audio_packed`
    // ⇒ audio 写线程 drain 完必置位（P3c-2：Ok/Err/panic 三出口均置位），无死锁。
    let aud_stream_done = Arc::new(AtomicBool::new(false));
    let aud_done_for_write = aud_stream_done.clone();
    // audio 写线程闭包内的 done 句柄别名（收尾置位用）
    let audio_done_for_body = aud_stream_done.clone();
    // P3a：audio 源段总时长累加器（demux 侧计数点，§6.5；manifest 合并点读取）
    let audio_seconds_acc = Arc::new(Mutex::new(0.0f64));

    // P4 §11：源路径清单克隆（io_read_task 会 move 原件；manifest 透出用）
    let input_paths_for_manifest = input_paths.clone();
    // P4 §11：field_map 克隆（io_read_task move 原件；manifest 透出用）
    let field_map_for_manifest = field_map.clone();
    // 1. IO 读取
    let read_token = cancel_token.clone();
    let read_progress = progress.clone();
    // P2：vlm 的 skipped 计数（SkipAndCount 入 manifest；pt/sft 不用，保持 None）。
    let vlm_error_counts = Arc::new(Mutex::new(SourceErrorCounts::default()));
    let vlm_error_counts_for_read = vlm_error_counts.clone();
    // P4 §8：云源物化管理器（vlm 读取任务用；本地路径零成本直通）
    let io_mgr = crate::io::CloudIoManager::new();
    let io_read_task: tokio::task::JoinHandle<Result<(), String>> = tokio::spawn(async move {
        if is_vlm {
            // vlm 源读取：read_samples 产出 Vec<Sample>（含 payload 压缩字节 + 段表）。
            // chunk 双约束（条数 + 字节预算）防 OOM（D5/P0-T3）；失败按 FailurePolicy 处置。
            let id_alloc = SampleIdAllocator::new();
            let mut error_counts = SourceErrorCounts::default();
            // P4：field_map 经构建配置线程化传入（替换原硬编码 default——
            // 否则只在 debug_read_samples 生效、生产构建永远用默认映射）
            let options = ReadSamplesOptions {
                field_map: field_map.clone(),
                // P2-6：媒体 URL 下载接入构建取消令牌（中止后不再发起下载）。
                cancel: Some(read_token.clone()),
                ..Default::default()
            };
            for path in input_paths {
                if read_token.is_cancelled() {
                    break;
                }
                // P4 §8：云路径预解析（async 上下文内完成；read_samples 保持同步本地读）
                let local_path = match resolve_source_to_local(&io_mgr, &path) {
                    Ok(p) => p,
                    Err(e) => return Err(e),
                };
                if let Err(e) = read_samples(
                    &local_path,
                    &options,
                    &id_alloc,
                    &mut error_counts,
                    tx_samples.clone(),
                    read_progress.clone(),
                ) {
                    return Err(format!("Failed to read samples {path}: {e}"));
                }
            }
            // skipped 计数回填共享槽（构建结束写 manifest global_stats.skipped_*）。
            *vlm_error_counts_for_read.lock().unwrap() = error_counts;
            // tx_samples drop → compute 线程 recv 结束
            drop(tx_samples);
        } else {
            for path in input_paths {
                if read_token.is_cancelled() {
                    break;
                }
                // ★ 读源错误必须上报：旧实现 `let _ =` 静默吞掉，输入文件不存在 / 读到中途
                //   出错时会产出"看起来完整"的部分数据集（实测非法 UTF-8 会静默少 16.7% 数据）。
                //   这里 return 后 tx_raw 被 drop → 计算/写入线程随即结束。
                // 同步读（短期项 #9：read_jsonl_chunks 已去 async，调用点不再需要 .await）
                if let Err(e) =
                    read_jsonl_chunks(&path, 5000, tx_raw.clone(), read_progress.clone())
                {
                    return Err(format!("Failed to read input {path}: {e}"));
                }
            }
        }
        Ok(())
    });

    // 2. CPU 计算层 (动态分流)。脏数据按 filter_map 丢弃并记录日志，不让单条脏数据崩掉整个构建。
    //
    //    **bbpe 路径**：render + encode 混合 per_doc par_iter —— 原样保留（D1 零改动承诺）。
    //    **Python 路径**（有 encoder）：重排为编排线程逐 chunk 驱动的三相位
    //    （§4.2）——相位 R（渲染，par_iter 无 GIL）→ 相位 T（tokenize，单点 GIL）
    //    → 相位 P（pack，无 GIL）。回调绝不能出现在 rayon 并行闭包内。
    let compute_token = cancel_token.clone();
    let compute_progress = progress.clone();
    // 文档 uid 分配器：整个构建内全局递增（rayon 收集保序 → 分配结果确定可复现）
    let uid_counter = Arc::new(AtomicU32::new(0));
    let vlm_stats_acc_for_compute = vlm_stats_acc.clone();
    // P3a：audio 流尺寸配置（0 = 同 Vlm 流）——compute 线程建 packer、audio 写线程
    // 定 shard 容量共用；pt/sft 不消费。
    let audio_stream_dims: (usize, usize, usize) = vlm_config
        .as_ref()
        .map(|c| {
            (
                c.audio_max_seq_len,
                c.audio_batch_size,
                c.audio_batches_per_shard,
            )
        })
        .unwrap_or((0, 0, 0));
    // P3a：align_fingerprint 最小集参数（W_ms / window_token_rate；manifest 合并点写）
    let align_fp_params: (u32, u32) = vlm_config
        .as_ref()
        .map(|c| (c.w_ms, c.audio_window_token_rate))
        .unwrap_or((0, 0));
    // P3b：align_fingerprint.video 子键参数 + Vlm 流 shard 字节均衡上限
    // （video_* 由构建配置层已定，P3c 统一写指纹；此处只取写线程消费项）
    let video_shard_max_bytes: u64 = vlm_config
        .as_ref()
        .map(|c| c.video_shard_max_bytes)
        .unwrap_or(0);
    // P3b：align_fingerprint.video 子键参数（n_frames / fps 兜底 / resize / seed；
    // P3b §6.5——只落 video 子键，ts_token/text_in_axis 由 P3c 增补）
    let align_fp_video: (u16, (u16, u16), u64, f32) = vlm_config
        .as_ref()
        .map(|c| {
            (
                c.video_n_frames,
                c.video_resize,
                c.video_seed,
                c.video_fps_default,
            )
        })
        .unwrap_or((16, (336, 336), 0, 30.0));
    // P3b：video 源时长累加器（写侧累计，manifest 合并点读取；同 audio_seconds_acc 模式）
    let video_seconds_acc = Arc::new(Mutex::new(0.0f64));
    let video_seconds_acc_for_write = video_seconds_acc.clone();
    // P3a：audio 段实际 codec 集合（demux 侧收集，BTreeSet 排序可复现）
    let audio_codecs_acc = Arc::new(Mutex::new(BTreeSet::new()));
    let audio_seconds_acc_for_compute = audio_seconds_acc.clone();
    let audio_codecs_acc_for_compute = audio_codecs_acc.clone();
    let compute_task = std::thread::spawn(move || -> Result<(), String> {
        // P2：vlm 分支——read_samples → PyModalityEncoder.encode_samples → ModalityAwarePacker.pack。
        // 与 pt/sft 完全分流（不触碰既有 render/encode/pack_phase 三相位，零行为变更）。
        if is_vlm {
            let me = modality_encoder.expect("vlm validated: encoder present");
            let cfg = vlm_config.expect("vlm validated: config present");
            let mut vlm_packer = crate::packing::modality_aware::ModalityAwarePacker::new(
                max_seq_len,
                pad_id,
                batch_size,
                true, /*落 loss_weights 监督掩码*/
                cfg.max_visual_tokens_per_sample,
                cfg.visual_budget_policy,
                cfg.image_label,
            );
            // P3a：build 级 audio GreedyPacker（recv 循环外创建、循环外 finish——
            // F11 先例：chunk 内创建会每 chunk flush 一个非满批，破坏 A5）。
            // extra_columns=true：doc_ids 承载窗口 doc uid（rec↔batch 配对）+
            // loss_weights 落盘；eos 复用文本 eos_id（模型侧忽略，P3-1）。
            let audio_msl = if cfg.audio_max_seq_len > 0 {
                cfg.audio_max_seq_len
            } else {
                max_seq_len
            };
            let audio_bs = if cfg.audio_batch_size > 0 {
                cfg.audio_batch_size
            } else {
                batch_size
            };
            let mut audio_packer = GreedyPacker::new(audio_msl, eos_id, audio_bs, true);
            // §5.4 配对表：窗口 doc uid → rec（push/finish 出批时按 doc_ids 取回）；
            // claimed = 已归首段 batch 的 uid（跨 batch 延续段不重复产 rec）
            let mut audio_pending: HashMap<u32, AudioRecordInputOwned> = HashMap::new();
            let mut audio_claimed: HashSet<u32> = HashSet::new();
            let audio_soft_warns = AudioSoftWarns::default();
            // P3c：ts token 软校验限频器 + 区间基址（编排线程计算，Copy 捕获）
            let ts_soft_warns = TsSoftWarns::default();
            let ts_base_id = ts_base_id;
            let demux_params = AudioDemuxParams {
                w_ms: cfg.w_ms,
                window_token_rate: cfg.audio_window_token_rate,
                capacity: audio_bs * audio_msl,
                strict_monotonic: cfg.audio_strict_monotonic,
                uid_counter: uid_counter.as_ref(),
                soft_warns: &audio_soft_warns,
            };
            while let Ok(samples) = rx_samples.recv() {
                if compute_token.is_cancelled() {
                    break;
                }
                // error 样本（SkipAndCount）不进 encoder（source 已计数 + 占 id）。
                let ok_samples: Vec<Sample> = samples.into_iter().filter(|s| s.is_ok()).collect();
                if ok_samples.is_empty() {
                    continue;
                }
                // P3a：encode 前预提取 audio 段 build 侧元数据（timestamp/MediaMeta/
                // 压缩字节——encode 后 Sample 已消费，SegmentEntry 无 timestamp）。
                let audio_sources = extract_audio_sources(&ok_samples);
                // P3c §4.2：text 段 timestamp 暂存 + A/V 轨样本集（同点预提取）
                let ts_pre = extract_ts_pre(&ok_samples);
                // P3b §6.1：源缺 fps 的 video 段计数（encoder 侧限频 WARN + 采用
                // video_fps_default；计数入 manifest video_fps_defaulted_clips）
                for s in &ok_samples {
                    let n_video = s
                        .segments
                        .iter()
                        .filter(|g| g.kind == SegmentKind::Video)
                        .count();
                    if n_video > 0 && s.media_meta.and_then(|m| m.fps).is_none() {
                        compute_progress
                            .video_fps_defaulted
                            .fetch_add(n_video as u64, Ordering::Relaxed);
                    }
                }
                for src in audio_sources.values() {
                    audio_codecs_acc_for_compute
                        .lock()
                        .unwrap()
                        .insert(codec_str(src.codec).to_string());
                }
                // 相位 T（vlm）：单点 GIL 调 Python modality encoder（D7 GIL 红线）。
                let encoded = me.encode_samples(ok_samples)?;
                compute_progress
                    .docs_tokenized
                    .fetch_add(encoded.len() as u64, Ordering::Relaxed);
                if compute_token.is_cancelled() {
                    break;
                }
                // P3a demux：audio 段切窗入 GreedyPacker（变长装箱），其余段进
                // ModalityAwarePacker（定宽行）——R1 路由，两 packer 一行不动。
                let AudioDemuxOutput {
                    mut vlm_samples,
                    window_docs,
                    audio_seconds,
                    empty_segments,
                } = demux_audio_samples(encoded, &audio_sources, &demux_params)?;
                // P3c §4.1：ts 插入在 demux **之后**、pack 之前（只触碰 Vlm 流
                // 子序列，坐标重算局部化；audio 通路零交互）。
                for es in vlm_samples.iter_mut() {
                    insert_ts_tokens(es, &ts_pre, ts_base_id, cfg.w_ms, &ts_soft_warns)?;
                }
                if audio_seconds > 0.0 {
                    *audio_seconds_acc_for_compute.lock().unwrap() += audio_seconds;
                }
                // P3a-2（评审）：空 audio 段计数入 manifest（不静默丢段）
                if empty_segments > 0 {
                    compute_progress
                        .audio_empty_segments
                        .fetch_add(empty_segments, Ordering::Relaxed);
                }
                if !window_docs.is_empty() {
                    let mut audio_docs: Vec<(Vec<i32>, u32)> =
                        Vec::with_capacity(window_docs.len());
                    for (tokens, uid, rec) in window_docs {
                        audio_pending.insert(uid, rec);
                        audio_docs.push((tokens, uid));
                    }
                    for ab in audio_packer.push_docs(audio_docs) {
                        let recs = pair_audio_records(&ab, &mut audio_pending, &mut audio_claimed)?;
                        if tx_audio_packed
                            .send(AudioPackedBatch {
                                batch: ab,
                                audio_records: recs,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                }
                // 相位 P（vlm）：模态感知装箱（不持 GIL）。
                let batches = vlm_packer.pack(vlm_samples)?;
                // 截断统计汇入 BuildProgress（构建结束 WARN + manifest）。
                let (vt, td) = vlm_packer.take_stats();
                if vt > 0 {
                    compute_progress
                        .visual_truncated
                        .fetch_add(vt, Ordering::Relaxed);
                }
                if td > 0 {
                    compute_progress
                        .vlm_tokens_dropped
                        .fetch_add(td, Ordering::Relaxed);
                }
                // 模态统计：取 packer **实际入包**计数（post-drop），作为 manifest
                // total_text_tokens / total_image_tokens / total_images 真源。
                // 不能用 encoder 的 es.stats（pre-drop）——原子性丢弃 / 视觉预算截断 /
                // text 尾部截断会让 Σ 模态 != total_tokens(packed) 而破坏守恒。
                {
                    let (pt, pi, pimgs, pvt, pvc) = vlm_packer.take_packed_modality_stats();
                    let packed = EncodedStats {
                        text_tokens: pt,
                        image_tokens: pi,
                        images: pimgs,
                        video_tokens: pvt,
                        video_clips: pvc,
                        ..Default::default()
                    };
                    vlm_stats_acc_for_compute.lock().unwrap().add(&packed);
                }
                for vb in batches {
                    if tx_packed.send(PackedItem::Vlm(vb)).is_err() {
                        break;
                    }
                }
            }
            // flush vlm 残余行（全数据集至多一个未满批）。
            if !compute_token.is_cancelled() {
                for vb in vlm_packer.finish() {
                    if tx_packed.send(PackedItem::Vlm(vb)).is_err() {
                        break;
                    }
                }
                // P3a：flush audio 残余（A5 全数据集口径：build 级 packer，至多 1 个
                // 非满批，属最后一个 audio shard）。
                for ab in audio_packer.finish() {
                    let recs = pair_audio_records(&ab, &mut audio_pending, &mut audio_claimed)?;
                    if tx_audio_packed
                        .send(AudioPackedBatch {
                            batch: ab,
                            audio_records: recs,
                        })
                        .is_err()
                    {
                        break;
                    }
                    // 注意：flush 阶段冲出的尾部 audio batch 与 push 阶段的 batch 一样
                    // 由 audio 写线程记录 span；deferred 导出在 aud_stream_done 后统一
                    // 覆盖（VR-2：旧水位屏障恰恰会漏掉这些尾部 flush batch）。
                }
                // 配对表必须清空：有 doc 没进任何 batch = 配对 bug（构造性检测）
                if !audio_pending.is_empty() {
                    return Err(format!(
                        "audio rec pairing leak: {} window doc(s) never reached a batch \
                         (pairing bug or oversized window not intercepted)",
                        audio_pending.len()
                    ));
                }
            }
            return Ok(());
        }
        while let Ok(raw_chunk) = rx_raw.recv() {
            if compute_token.is_cancelled() {
                break;
            }

            let batches: Vec<PackedBatch> = if let Some(enc) = &encoder {
                // ---- Python 路径 ----
                // 相位 R：模板渲染（rayon，不持 GIL）。SFT 额外带出源数据轮次数
                // （F7 注入检测的比对基准）。
                let (texts, expected_turns): (Vec<String>, Vec<usize>) = if is_sft {
                    render_phase_sft(&template_engine, raw_chunk)
                        .into_iter()
                        .unzip()
                } else {
                    (render_phase(raw_chunk), Vec::new())
                };
                if texts.is_empty() {
                    continue;
                }
                // 相位 T：单点 GIL，调用 Python encode_batch（HF fast 引擎内部自并行）
                let docs = enc.encode_batch(texts)?;
                compute_progress
                    .docs_tokenized
                    .fetch_add(docs.len() as u64, Ordering::Relaxed);
                if is_sft {
                    for (doc, &expected) in docs.iter().zip(&expected_turns) {
                        check_turn_injection(
                            &doc.tokens,
                            expected,
                            &turn_start_str,
                            &compute_progress.docs_flagged_injection,
                            &injection_warned,
                        );
                    }
                }
                if compute_token.is_cancelled() {
                    break;
                }
                // 相位 P：装箱（不持 GIL）
                pack_phase(
                    &sft_packer,
                    &mut pt_packer,
                    is_sft,
                    docs,
                    batch_size,
                    extra_columns,
                    &uid_counter,
                    &compute_progress,
                )
            } else if is_sft {
                let tokenizer = tokenizer.as_ref().expect("bbpe path must have a tokenizer");
                let docs: Vec<TokenizedDoc> = raw_chunk
                    .into_par_iter()
                    .filter_map(|json_str| {
                        match template_engine.render_chat(&json_str, false) {
                            Ok(rendered) => {
                                let enc =
                                    tokenizer.encode_with_types(&rendered, &HashSet::new(), true);
                                // F7 注入检测（只计数 + 限频 WARN，不丢弃样本）
                                check_turn_injection(
                                    &enc.tokens,
                                    expected_turn_count(&json_str),
                                    &turn_start_str,
                                    &compute_progress.docs_flagged_injection,
                                    &injection_warned,
                                );
                                Some(TokenizedDoc::from(enc))
                            }
                            Err(e) => {
                                tracing::error!("Discarding dirty data: {}", e);
                                None
                            }
                        }
                    })
                    .collect();
                compute_progress
                    .docs_tokenized
                    .fetch_add(docs.len() as u64, Ordering::Relaxed);
                pack_phase(
                    &sft_packer,
                    &mut pt_packer,
                    true,
                    docs,
                    batch_size,
                    extra_columns,
                    &uid_counter,
                    &compute_progress,
                )
            } else {
                let tokenizer = tokenizer.as_ref().expect("bbpe path must have a tokenizer");
                let ids_lists: Vec<_> = raw_chunk
                    .into_par_iter()
                    .filter_map(|json_str| {
                        let val: sonic_rs::Value = match sonic_rs::from_str(&json_str) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::error!(
                                    "Discarding dirty data (JSON parsing failed): {}",
                                    e
                                );
                                return None;
                            }
                        };
                        let text = val.get("text")?.as_str()?;
                        Some(tokenizer.encode_ordinary(text))
                    })
                    .collect();
                compute_progress
                    .docs_tokenized
                    .fetch_add(ids_lists.len() as u64, Ordering::Relaxed);
                pack_phase(
                    &sft_packer,
                    &mut pt_packer,
                    false,
                    ids_lists.into_iter().map(TokenizedDoc::from).collect(),
                    batch_size,
                    extra_columns,
                    &uid_counter,
                    &compute_progress,
                )
            };

            for packed_batch in batches {
                if tx_packed.send(PackedItem::Legacy(packed_batch)).is_err() {
                    break;
                }
            }
        }
        // F3：整个输入流结束——flush PT 残余缓冲（跨 chunk 保留的未满 batch，
        // 全数据集至多一个）。取消时不 flush（构建反正会以取消收场）。
        if !compute_token.is_cancelled() && !is_sft {
            for packed_batch in pt_packer.finish() {
                if tx_packed.send(PackedItem::Legacy(packed_batch)).is_err() {
                    break;
                }
            }
        }
        Ok(())
    });

    // 3. IO 落盘层 (双轨写入)。任何写盘失败都映射成 String 上抛，不再 panic。
    //    index.json **不在这里落盘**：由外层在"读/算/写全部成功"后保存，使
    //    "目录里有 index.json" 等价于"构建完整"（旧实现边写边存，读源失败时会留下
    //    看似完整、实则缺数据的产物）。
    let write_token = cancel_token.clone();
    let write_progress = progress.clone();
    let output_dir_for_task = output_dir.clone();
    let write_tokenizer_fp = tokenizer_fp.clone();
    // 落盘是纯阻塞 IO（文件写 + 写完回读 sha256，可能几十 MB/shard）：用普通线程而不是
    // async 任务，避免"在 async 里做阻塞 IO"（复核问题 15）——也就让 runtime 可以单线程。
    // ★ 评审 §3.7：半成品 shard 登记槽位（与写线程内的 state 共享）。写线程 panic 时
    //   state 随 unwind 析构、显式的 abort_current_shard 被跳过，只能靠它找到
    //   "正在写、未收尾"的 shard 文件——那个文件不在 manifest 里（finish 才登记），
    //   纯 manifest 驱动清理必然漏它。
    let in_progress_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let slot_for_state = in_progress_slot.clone();
    // P3a：audio 写线程共用同一 compression 配置（.aud_bin 随构建 compression，
    // 格式冻结 §1）——在 io_write_thread 闭包 move 之前克隆。
    let compression_for_audio = compression.clone();
    let io_write_thread = std::thread::spawn(move || {
        let output_dir = output_dir_for_task;
        let is_sft = mode == "sft";
        let pt_compression = if is_sft {
            None
        } else {
            ZstdCompression::parse(compression.as_deref())
        };
        let ipc_compression = if is_sft {
            IpcCompression::parse(compression.as_deref())
        } else {
            None
        };
        let storage_format = match (is_sft, pt_compression) {
            (true, _) => StorageFormat::ArrowIpc,
            (false, Some(_)) => StorageFormat::ZstdMds,
            (false, None) => StorageFormat::Memmap,
        };
        let batches_per_shard = resolve_batches_per_shard(
            batches_per_shard,
            batch_size,
            max_seq_len,
            is_sft,
            extra_columns,
        );
        tracing::info!(
            "📦 Shard aggregation granularity: {batches_per_shard} batch/shard{}",
            if pt_compression.is_some() {
                ", PT per-batch zstd frame compression enabled"
            } else {
                ""
            }
        );

        // ★ 评审 §3.7：manifest 声明在 catch_unwind **外面**。panic 时 unwind 只
        //   析构内层闭包的局部量（state 及其 writer 句柄），外层 manifest 原样
        //   保留——外层清理才能按它删掉"已收尾"的 shard。旧实现 panic 直接冲出
        //   线程 → join 得 Err → write_outcome=None → `.unwrap_or(0)` 一个文件
        //   都不删，半成品 shard（无 index.json 的孤儿）永远留在磁盘上。
        let mut manifest =
            DatasetManifest::new(storage_format, max_seq_len).with_tokenizer(write_tokenizer_fp);
        manifest.global_stats.batch_size = batch_size; // ★ 记录构建期 batch_size 供加载期校验
        if is_sft {
            // A4 数据自描述：SFT 的 Arrow 落盘含行级 `row_lens` 列（每行有效长度，
            // 不含行内 padding）。读取端据此直接把 (B, S) 还原成连续流，
            // 无需在运行期复刻 DocumentAwarePacker 的装箱算法。
            // 旧数据集无该列时读取端回退到重算（向后兼容）。
            manifest
                .schema
                .insert("row_lens".to_string(), "list<int32>".to_string());
        } else if extra_columns {
            // PT Memmap 的文档级旁路列：读取端按 schema key 判定"该数据集有这一列"
            // （旧数据集没有这些 key → 不打开任何旁路列，零兼容成本）。
            manifest
                .schema
                .insert("doc_ids".to_string(), "list<int32>".to_string());
            manifest
                .schema
                .insert("loss_weights".to_string(), "list<f16>".to_string());
        }

        let arrow_writer = ArrowIpcWriter::with_compression(max_seq_len, ipc_compression);
        let write_body = std::panic::AssertUnwindSafe(|| {
            let mut state = ShardWriteState {
                extra_columns,
                is_sft,
                output_dir: &output_dir,
                pt_compression,
                arrow_writer: &arrow_writer,
                shard_idx: 0,
                batches_in_shard: 0,
                valid_tokens_in_shard: 0,
                shard_filename: String::new(),
                arrow_shard: None,
                memmap_shard: None,
                mm_sidecar: None,
                vid_sidecar: None,
                vid_bytes_in_shard: 0,
                video_seconds_acc: video_seconds_acc_for_write.clone(),
                align_enabled,
                aud_spans: aud_spans_for_write.clone(),
                shard_min_sid: u64::MAX,
                shard_max_sid: 0,
                shard_sid_ranges: Vec::new(),
                aud_stream_done: aud_done_for_write.clone(),
                is_vlm,
                in_progress_slot: slot_for_state.clone(),
            };

            let mut write_result: Result<(), String> = Ok(());
            while let Ok(item) = rx_packed.recv() {
                if write_token.is_cancelled() {
                    break;
                }
                let r = match item {
                    PackedItem::Legacy(pb) => state.write_batch(&pb),
                    PackedItem::Vlm(vb) => state.write_vlm_batch(&vb),
                };
                if let Err(e) = r {
                    write_result = Err(e);
                    break;
                }
                // P3b §9：条数 / vid blob 字节双约束滚动（video_shard_max_bytes=0
                // 时回退纯条数判据；vid_bytes_in_shard 在 finish 时归零）
                let over_bytes =
                    video_shard_max_bytes > 0 && state.vid_bytes_in_shard >= video_shard_max_bytes;
                if (state.batches_in_shard >= batches_per_shard || over_bytes)
                    && let Err(e) = state.finish_current_shard(&mut manifest, &write_progress)
                {
                    write_result = Err(e);
                    break;
                }
            }
            let cancelled = write_token.is_cancelled();
            // 只有正常写完（非中断、无错误）才收尾最后一个未满的 shard
            if write_result.is_ok() && !cancelled && state.batches_in_shard > 0 {
                write_result = state.finish_current_shard(&mut manifest, &write_progress);
            }
            // VR-2 / P3c 设计修正：`align_aud_range` 改为**deferred 精确导出**——
            // 等 audio 流整体收尾（`aud_stream_done`）后，据 audio 写线程记录的
            // per-batch 样本 id 跨度 + 每个 Vlm shard 的样本 id 区间，构造性算出
            // “本 shard 样本 rec 所在的最小/最大 audio batch”。这统一消除了旧
            // “游标快照”语义的两个同族缺口：`first` 非超集（§10.6(3)，已 xfail）
            // 与 VR-2 的 `last` 漏覆盖尾部 flush。无死锁：`rx_packed` 断流 ⇒
            // compute 已结束并 drop `tx_audio_packed` ⇒ audio 写线程必 drain 后置位。
            if write_result.is_ok() && !cancelled && state.align_enabled {
                state.finalize_align_aud_ranges(&mut manifest);
            }
            if write_result.is_err() || cancelled {
                // 失败/取消：丢弃未收尾的 shard 文件，不留半成品（已收尾的由外层按 manifest 清理）
                state.abort_current_shard();
            }
            (write_result, cancelled)
        });
        match std::panic::catch_unwind(write_body) {
            Ok((write_result, cancelled)) => WriteOutcome {
                manifest,
                result: write_result,
                cancelled,
            },
            Err(panic) => {
                // panic 兜底（§3.7）：state 已随 unwind 析构（writer 句柄先关），
                // 按槽位删除"正在写"的 shard；已收尾的由外层按 manifest 删。
                let cleaned = cleanup_in_progress_shard(&in_progress_slot, &output_dir);
                let detail = panic_detail(&panic);
                tracing::error!(
                    "❌ IO write thread panicked{detail}; cleaned up {cleaned} in-progress shard file(s) \
                     (finished shards are cleaned by the caller via the manifest)"
                );
                WriteOutcome {
                    manifest,
                    result: Err(format!("IO write thread panicked{detail}")),
                    cancelled: false,
                }
            }
        }
    });

    // P3a：audio 流第二写线程（整体镜像 io_write_thread：独立消费循环 + 独立聚合
    // 状态机 + 独立半成品槽位 + panic 兜底）。audio batch **不混入** Vlm 流写线程
    // （F4：单通道混流会污染 batches_in_shard 计数、破坏 Vlm shard 边界）。
    // pt/sft 不生成（无 audio 流，通道无发送方即退出）。
    let audio_write_token = cancel_token.clone();
    let audio_output_dir = output_dir.clone();
    let audio_in_progress_slot: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let audio_slot_for_state = audio_in_progress_slot.clone();
    let io_audio_thread: Option<std::thread::JoinHandle<AudioWriteOutcome>> = if is_vlm {
        Some(std::thread::spawn(move || {
            let output_dir = audio_output_dir;
            let pt_compression = ZstdCompression::parse(compression_for_audio.as_deref());
            let (audio_msl_cfg, audio_bs_cfg, audio_bps_cfg) = audio_stream_dims;
            let audio_msl = if audio_msl_cfg > 0 {
                audio_msl_cfg
            } else {
                max_seq_len
            };
            let audio_bs = if audio_bs_cfg > 0 {
                audio_bs_cfg
            } else {
                batch_size
            };
            let audio_batches_per_shard = resolve_batches_per_shard(
                audio_bps_cfg,
                audio_bs,
                audio_msl,
                false,
                true, /*audio 恒落 doc_ids + loss_weights*/
            );
            tracing::info!(
                "🎵 Audio stream shard aggregation: {audio_batches_per_shard} batch/shard \
                 (msl={audio_msl}, bs={audio_bs}, capacity={})",
                audio_bs * audio_msl
            );
            let mut aggregates = AudioWriteOutcome::empty();
            let write_body = std::panic::AssertUnwindSafe(|| {
                let mut state = AudioShardWriteState {
                    output_dir: &output_dir,
                    pt_compression,
                    ashard_idx: 0,
                    batches_in_shard: 0,
                    valid_tokens_in_shard: 0,
                    shard_filename: String::new(),
                    memmap_shard: None,
                    aud_sidecar: None,
                    valid_recs_in_shard: 0,
                    in_progress_slot: audio_slot_for_state.clone(),
                    aud_spans: aud_spans_for_audio.clone(),
                };
                let mut result: Result<(), String> = Ok(());
                while let Ok(ab) = rx_audio_packed.recv() {
                    if audio_write_token.is_cancelled() {
                        break;
                    }
                    // P2-1（评审修复）：rollover 判定放在**写入本 batch 之前**。
                    // 若本 batch 仍在续写当前 shard 内未收尾的窗口组（尾窗引用
                    // 本 shard 已写首窗），延后切 shard——保证同组首尾窗落在同一
                    // shard（新 writer 的 seg_ref 为空，跨 shard 尾窗必然构建失败）。
                    if state.batches_in_shard >= audio_batches_per_shard
                        && !state.continues_open_group(&ab)
                        && let Err(e) = state.finish_current_audio_shard(&mut aggregates)
                    {
                        result = Err(e);
                        break;
                    }
                    if let Err(e) = state.write_audio_batch(&ab) {
                        result = Err(e);
                        break;
                    }
                }
                let cancelled = audio_write_token.is_cancelled();
                if result.is_ok() && !cancelled && state.batches_in_shard > 0 {
                    result = state.finish_current_audio_shard(&mut aggregates);
                }
                if result.is_err() || cancelled {
                    state.abort_current_audio_shard();
                }
                (result, cancelled)
            });
            match std::panic::catch_unwind(write_body) {
                Ok((result, cancelled)) => {
                    aggregates.result = result;
                    aggregates.cancelled = cancelled;
                    // P3c 屏障：audio 流全部写出（Vlm 侧 shard 收尾等待释放）
                    audio_done_for_body.store(true, Ordering::Release);
                    aggregates
                }
                Err(panic) => {
                    // P3c-2 修复：panic 路径也必须置 done——否则 Vlm 写线程在屏障上
                    // 永久自旋（is_poisoned 防御无效：audio 线程持的是 audio 槽位锁，
                    // 毒化传导不到 Vlm 检查的 in_progress_slot）。
                    audio_done_for_body.store(true, Ordering::Release);
                    // panic 兜底（§3.7 同构）：按 audio 槽位删"正在写"的 shard
                    let cleaned =
                        cleanup_in_progress_audio_shard(&audio_in_progress_slot, &output_dir);
                    let detail = panic_detail(&panic);
                    tracing::error!(
                        "❌ IO audio write thread panicked{detail}; cleaned up {cleaned} in-progress \
                         audio shard file(s) (finished audio shards are cleaned by the caller)"
                    );
                    AudioWriteOutcome {
                        shards: aggregates.shards,
                        total_tokens: aggregates.total_tokens,
                        total_clips: aggregates.total_clips,
                        result: Err(format!("IO audio write thread panicked{detail}")),
                        cancelled: false,
                    }
                }
            }
        }))
    } else {
        None
    };

    // 读任务（async，驱动 tokio::fs）与写线程（阻塞 IO）并行跑，各自收尾后再判定
    let read_res = io_read_task.await;
    let write_res = io_write_thread
        .join()
        .map_err(|_| "IO write thread panicked".to_string());
    // P3a：audio 写线程 join（compute 线程结束 → tx_audio_packed drop → 本线程退出）
    let (audio_outcome, audio_join_err) = match io_audio_thread {
        Some(h) => match h.join() {
            Ok(o) => (o, None),
            Err(_) => (
                AudioWriteOutcome::empty(),
                Some("IO audio write thread panicked".to_string()),
            ),
        },
        None => (AudioWriteOutcome::empty(), None),
    };
    let read_err = match read_res {
        Ok(Ok(())) => None,
        Ok(Err(e)) => Some(e),
        Err(e) => Some(format!("IO read task failed: {e}")),
    };
    // 写线程的 JoinError（panic）单独记；正常时拿回 manifest（失败清理要用）
    let (write_outcome, write_join_err) = match write_res {
        Ok(w) => (Some(w), None),
        Err(e) => (None, Some(e)),
    };
    // compute_task 是 std::thread::spawn，join 返回 thread::Result<Result<(), String>>
    let compute_err = compute_task
        .join()
        .map_err(|_| "compute thread panicked".to_string())
        .and_then(|r| r)
        .err();

    // ---- 构建结束汇总（F2 / F4 / F7）：让"静默损失"在收尾时集中可见 ----
    let docs_read_total = progress.docs_read.load(Ordering::Relaxed);
    let docs_tokenized_total = progress.docs_tokenized.load(Ordering::Relaxed);
    // F4 措施 2：渲染失败汇总（此前只有逐条 error 日志，无汇总）——帮助发现"部分脏数据"场景
    if docs_read_total > docs_tokenized_total {
        tracing::warn!(
            "⚠️ {}/{} input rows were discarded during rendering/tokenization (dirty rows); \
             see the per-row error logs above for causes.",
            docs_read_total - docs_tokenized_total,
            docs_read_total
        );
    }
    // F2 措施 1：截断汇总（含比例）——消灭"静默截断"
    let docs_truncated = progress.docs_truncated.load(Ordering::Relaxed);
    let all_masked = progress.all_masked_docs.load(Ordering::Relaxed);
    if docs_truncated > 0 || all_masked > 0 {
        let tokens_dropped = progress
            .tokens_dropped_to_truncation
            .load(Ordering::Relaxed);
        let pct = docs_truncated as f64 * 100.0 / docs_tokenized_total.max(1) as f64;
        tracing::warn!(
            "⚠️ SFT truncation summary: {docs_truncated} doc(s) ({pct:.2}%) exceeded max_seq_len and were tail-truncated \
             ({tokens_dropped} tokens dropped); {all_masked} doc(s) have all -100 labels (assistant span cut off). \
             Increase max_seq_len or shorten conversations."
        );
    }
    // F7：注入检测汇总
    let flagged_injection = progress.docs_flagged_injection.load(Ordering::Relaxed);
    if flagged_injection > 0 {
        tracing::warn!(
            "⚠️ Turn-marker injection summary: {flagged_injection} SFT doc(s) have turn-start marker counts inconsistent \
             with their source turn count (possible label leakage via injected markers). Sanitize the data or tighten the template."
        );
    }
    // F2-PT 切分汇总（20261007 评审 §3.5）：PT 不截断，但超长文档被跨 batch 切分，
    //   varlen 训练下每段都是 position 从 0 起的独立序列 → 跨 batch 上下文静默丢失。
    //   与 SFT 的截断告警并列输出，让用户一眼看清"数据到底损失了什么"。
    let docs_split = progress.docs_split.load(Ordering::Relaxed);
    if docs_split > 0 {
        let tokens_in_split = progress.tokens_in_split_docs.load(Ordering::Relaxed);
        let pct = docs_split as f64 * 100.0 / docs_tokenized_total.max(1) as f64;
        let avg_seg = tokens_in_split as f64 / docs_split as f64;
        tracing::warn!(
            "⚠️ PT long-document split summary: {docs_split} doc(s) ({pct:.2}%) are longer than one batch \
             ({tokens_in_split} tokens total, avg {avg_seg:.0} tokens/doc) and were split across batches. \
             No data was dropped, but with FlashAttention varlen (cu_seqlens) each segment becomes an independent \
             sequence starting at position 0, so cross-batch context inside one document is lost. \
             Increase max_seq_len so most documents fit in a single batch."
        );
    }

    // ★ 三方（读 / 计算 / 写）全部成功才落盘 index.json；任何失败都清理本次写出的
    //   shard 文件，让"有 index.json" == "构建完整"（否则部分数据集会被误当完整使用）。
    let mut first_err = read_err
        .or(compute_err)
        .or(write_join_err)
        .or(audio_join_err);
    if first_err.is_none() {
        first_err = write_outcome
            .as_ref()
            .and_then(|w| w.result.as_ref().err().cloned());
    }
    if first_err.is_none() {
        first_err = audio_outcome.result.as_ref().err().cloned();
    }
    if let Some(msg) = first_err {
        // ★ 评审 §3.7：写线程 panic 现在也产出 WriteOutcome（catch_unwind 兜底，
        //   manifest 保留），因此这条清理路径对 panic 同样生效：
        //   - "已收尾"的 shard → 下面按 manifest 删；
        //   - "正在写、未收尾"的 shard → 写线程 panic 分支里已按 in_progress_slot 删。
        //   `unwrap_or(0)` 只剩"理论上不可达"的防御（panic=abort 时 join 才 Err）。
        let removed = write_outcome
            .as_ref()
            .map(|w| remove_partial_shards(&w.manifest, &output_dir))
            .unwrap_or(0)
            + remove_partial_audio_shards(&audio_outcome.shards, &output_dir);
        return Err(format!(
            "{msg} (cleaned up {removed} partial shard file(s); index.json not written)"
        ));
    }
    let Some(mut outcome) = write_outcome else {
        return Err(
            "internal error: write task produced no output (index.json not written)".to_string(),
        );
    };
    if outcome.cancelled || audio_outcome.cancelled {
        let removed = remove_partial_shards(&outcome.manifest, &output_dir)
            + remove_partial_audio_shards(&audio_outcome.shards, &output_dir);
        tracing::warn!(
            "⏹ Build canceled: cleaned up {removed} partial shard files, index.json not written"
        );
        return Ok(());
    }
    if outcome.manifest.shards.is_empty() && audio_outcome.shards.is_empty() {
        // F4：0 产出从"静默成功"改为显式失败——schema 与模板变量不匹配（如
        // `messages` 数据配 `conversations` 模板）时，旧行为不落 index.json 也不
        // 报错，Python 侧只看到 manifest=None，问题被拖到训练期才暴露。
        // P3a：两流均 0 产出才算空（纯 audio 数据集只有 audio 流 shards）。
        let _ = std::fs::remove_dir(&output_dir); // 尽力清掉空目录（非空则保留原内容）
        return Err(format!(
            "Build produced 0 shards: no input rows survived the pipeline (docs_read={docs_read_total}, \
             docs_tokenized={docs_tokenized_total}). Common causes: the input files are empty; every row failed \
             template rendering (check that template variables match the data keys — the `messages` template \
             variable accepts both `messages` and `conversations` keys, while the `conversations` template variable \
             only reads the `conversations` key); or PT rows lack the `text` field."
        ));
    }
    // F11：构建期 shard 数告警——读取端只有拿到具体 world_size 才能精确校验
    // （< world_size 报错），构建期用"常见 world_size"提前暴露分片不足：shard 数少于
    // 8 时，world_size ≥ 8 的 DDP 训练会有 rank 分到 0 个 shard。更小的集群是否可接受
    // 由使用者结合拓扑判断（读取端有精确校验兜底）。
    if outcome.manifest.shards.len() < F11_COMMON_WORLD_SIZE {
        tracing::warn!(
            "⚠️ Dataset has only {} shard(s) (< {F11_COMMON_WORLD_SIZE}): when world_size >= {F11_COMMON_WORLD_SIZE}, \
             some ranks receive 0 shards and the reader errors out. Increase batches_per_shard, or merge small \
             datasets, before multi-rank training.",
            outcome.manifest.shards.len()
        );
    }
    // F2 措施 2：截断统计写入 manifest（为 0 时序列化省略，保持旧格式兼容）
    outcome.manifest.global_stats.truncated_docs = docs_truncated;
    outcome.manifest.global_stats.truncated_tokens = progress
        .tokens_dropped_to_truncation
        .load(Ordering::Relaxed);
    // F2-PT 措施 2：切分统计写入 manifest（为 0 时序列化省略，保持旧格式兼容）
    outcome.manifest.global_stats.split_docs = progress.docs_split.load(Ordering::Relaxed);
    outcome.manifest.global_stats.split_tokens =
        progress.tokens_in_split_docs.load(Ordering::Relaxed);
    // 多模态 P1-T3：模态拆分一步到位（sidecar 格式冻结 §8）。文本/纯文本构建
    // image/audio/video 恒 0；P2 vlm 管线落地时由 packer 分模态累加。
    // total_tokens 口径不含 padding（v2.0 语义），拆分字段同口径 → 守恒严格成立。
    outcome.manifest.global_stats.stats_version = 2;
    outcome.manifest.global_stats.total_text_tokens = outcome.manifest.global_stats.total_tokens;
    // P2：vlm 模态统计 + id 区间 + skipped 计数入 manifest（pt/sft 跳过，保持 0/None）。
    if is_vlm {
        let stats = *vlm_stats_acc.lock().unwrap();
        outcome.manifest.global_stats.total_text_tokens = stats.text_tokens;
        outcome.manifest.global_stats.total_image_tokens = stats.image_tokens;
        outcome.manifest.global_stats.total_images = stats.images;
        // P3b：video 模态统计（packer 五元组 post-drop 口径；无 video 数据集恒 0）
        outcome.manifest.global_stats.total_video_tokens = stats.video_tokens;
        outcome.manifest.global_stats.total_video_clips = stats.video_clips;
        // P3b §6.4：total_video_seconds = 写侧累计的源视频时长（Σ duration_ms/1000）
        outcome.manifest.global_stats.total_video_seconds = *video_seconds_acc.lock().unwrap();
        // P3b §6.1：fps 兜底计数（源缺 fps 的 video 段数；消费侧以此为信号核查源数据）
        outcome.manifest.global_stats.video_fps_defaulted_clips =
            progress.video_fps_defaulted.load(Ordering::Relaxed);
        let vfd = progress.video_fps_defaulted.load(Ordering::Relaxed);
        if vfd > 0 {
            tracing::warn!(
                "⚠️ Video fps defaulted: {vfd} video segment(s) lacked MediaMeta.fps and used the \
                 build default (see global_stats.video_fps_defaulted_clips in index.json). Frame \
                 timing computed downstream (start_ms + frame_idx/fps) will be off for those clips — \
                 check the source data."
            );
        }
        // P3a-2（评审）：空 audio 段计数入 manifest（不静默丢段）
        outcome.manifest.global_stats.audio_empty_segments =
            progress.audio_empty_segments.load(Ordering::Relaxed);
        // total_tokens 由 append_shard 按 cu_seqlens 末值累加（含 visual placeholder），
        // 与 Σ 模态（text+image）应守恒；text_tokens 取 encoder 统计（不含 padding）。
        let ec = vlm_error_counts.lock().unwrap();
        if !ec.skipped_by_class.is_empty() {
            outcome.manifest.global_stats.skipped_samples_by_class = ec.skipped_by_class.clone();
        }
        outcome.manifest.global_stats.fail_fast_triggered = ec.fail_fast_triggered;
        outcome.manifest.modality_id_ranges = modality_id_ranges_for_manifest;
        // 视觉预算/行宽截断汇总 WARN（消灭静默损失）。
        let vt = progress.visual_truncated.load(Ordering::Relaxed);
        let td = progress.vlm_tokens_dropped.load(Ordering::Relaxed);
        if vt > 0 || td > 0 {
            tracing::warn!(
                "⚠️ VLM truncation summary: {vt} image segment(s) dropped by visual budget \
                 (max_visual_tokens_per_sample); {td} token(s) dropped to row width / image atomicity."
            );
        }
    }
    // P3a：manifest 合并点——两写线程 join 后、index.json 落盘前，编排线程**串行**
    // 合并（§6.3/§6.5；任一写线程失败已在上面整体失败 + 双侧清理，不会走到这）：
    // audio_shards + 两流统计 + schema 键 + align_fingerprint 最小集。
    // 无 audio 流（纯文本/图像构建）时整体跳过——产物与 P2 期逐字节一致。
    let has_audio_stream = !audio_outcome.shards.is_empty();
    if has_audio_stream {
        outcome.manifest.audio_shards = audio_outcome.shards;
        // total_tokens 扩为两流之和（Vlm 流 append_shard 口径 + audio 流
        // cu_seqlens 末值口径）；漏加 audio 会被守恒断言构造性捕获。
        outcome.manifest.global_stats.total_tokens += audio_outcome.total_tokens;
        outcome.manifest.global_stats.total_audio_tokens = audio_outcome.total_tokens;
        // total_audio_clips = VALID（首窗）rec 数（尾窗 flags=4 不计，格式冻结 §9.9）
        outcome.manifest.global_stats.total_audio_clips = audio_outcome.total_clips;
        // total_audio_seconds = demux 侧累加器（duration_sec 存在用之，缺失用
        // token 隐含时长，§6.5）
        outcome.manifest.global_stats.total_audio_seconds = *audio_seconds_acc.lock().unwrap();
        // schema 键（格式冻结 §5 v1.3）：audio sidecar 三 key + audio 流主文件两列
        outcome
            .manifest
            .schema
            .insert("aud_blob".to_string(), "bytes".to_string());
        outcome
            .manifest
            .schema
            .insert("aud_rec".to_string(), "audrec40".to_string());
        outcome
            .manifest
            .schema
            .insert("aud_cu".to_string(), "audcu".to_string());
        outcome
            .manifest
            .schema
            .insert("aud_doc_ids".to_string(), "list<int32>".to_string());
        outcome
            .manifest
            .schema
            .insert("aud_loss_weights".to_string(), "list<f16>".to_string());
        tracing::info!(
            "🎵 Audio stream: {} shard(s) / {} token(s) / {} clip(s) merged into manifest",
            outcome.manifest.audio_shards.len(),
            outcome.manifest.global_stats.total_audio_tokens,
            outcome.manifest.global_stats.total_audio_clips,
        );
    }
    // P3c §3.2：对齐启用时写**完整** align_fingerprint（终稿 DP5 最小充分集）——
    // W_ms / granularity / text_in_axis / ts_token / modality_id_ranges 快照恒写；
    // audio / video 子键仅在有相应内容时写（P3a/P3b 口径不变）。
    // 不存在 = 对齐未启用（旧/纯文本图像数据集）；存在但自相矛盾 → 读取端 fail-fast。
    if align_enabled {
        let (w_ms, window_token_rate) = align_fp_params;
        let mut fp = BTreeMap::new();
        fp.insert("W_ms".to_string(), serde_json::json!(w_ms));
        fp.insert("granularity".to_string(), serde_json::json!("G2"));
        fp.insert("text_in_axis".to_string(), serde_json::json!(true));
        fp.insert(
            "ts_token".to_string(),
            serde_json::json!({
                "base_id": ts_base_id,
                "bucket_width_ms": w_ms,
                "encoding": "contiguous",
                "range_size": TS_RANGE_SIZE,
            }),
        );
        if let Some(ranges) = &modality_id_ranges_snapshot {
            let mut snapshot: Vec<serde_json::Value> = ranges
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "start": r.start, "end": r.end, "kind": r.kind,
                    })
                })
                .collect();
            // P3c-4（评审）：ts token 区间与模态区间同步登记——方案 §3.2/P2-6
            // 要求；缺登记使 A7"range_size 与登记不符"校验失去依据。
            snapshot.push(serde_json::json!({
                "start": ts_base_id,
                "end": ts_base_id + TS_RANGE_SIZE as i32,
                "kind": "timestamp",
            }));
            fp.insert(
                "modality_id_ranges".to_string(),
                serde_json::json!(snapshot),
            );
        }
        if !outcome.manifest.audio_shards.is_empty() {
            let codecs: Vec<String> = audio_codecs_acc.lock().unwrap().iter().cloned().collect();
            fp.insert(
                "audio".to_string(),
                serde_json::json!({
                    "window_token_rate": window_token_rate,
                    "codec": codecs,
                }),
            );
        }
        if outcome.manifest.global_stats.total_video_clips > 0 {
            let (n_frames, resize, seed, fps) = align_fp_video;
            fp.insert(
                "video".to_string(),
                serde_json::json!({
                    "n_frames": n_frames,
                    "fps": fps,
                    "resize": [resize.0, resize.1],
                    "seed": seed,
                }),
            );
        }
        outcome.manifest.align_fingerprint = Some(fp);
    }
    // P4 §11：源格式与字段映射透出（仅 vlm——pt/sft 不落新字段，旧 manifest 兼容）
    if is_vlm {
        let mut formats: Vec<String> = Vec::new();
        for p in &input_paths_for_manifest {
            let f = match SourceFormat::detect(p) {
                Ok(SourceFormat::Jsonl) => "jsonl",
                Ok(SourceFormat::Tar) => "webdataset",
                Ok(SourceFormat::Parquet) => "parquet",
                Err(_) => continue, // 读取期已显式报错（未知扩展名）
            };
            let s = f.to_string();
            if !formats.contains(&s) {
                formats.push(s);
            }
        }
        formats.sort();
        outcome.manifest.source_formats = formats;
        let mut fm_summary: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let fm = &field_map_for_manifest;
        fm_summary.insert("image_fields".to_string(), fm.image_fields.clone());
        fm_summary.insert("text_fields".to_string(), fm.text_fields.clone());
        fm_summary.insert("audio_fields".to_string(), fm.audio_fields.clone());
        fm_summary.insert("video_fields".to_string(), fm.video_fields.clone());
        fm_summary.insert("meta_fields".to_string(), fm.meta_fields.clone());
        // P4-4（评审）：timestamp_field 是 Option<String>——配置了才落单元素列表，
        // 未配置则**省略该键**（而非空列表），消除 “[] = 未配置 vs [] = 空候选” 的序列化歧义。
        if let Some(v) = fm.timestamp_field.clone() {
            fm_summary.insert("timestamp_field".to_string(), vec![v]);
        }
        outcome.manifest.source_field_map = Some(fm_summary);
    }
    outcome
        .manifest
        .global_stats
        .check_modality_conservation()?;
    outcome
        .manifest
        .save_to_file(&format!("{}/index.json", output_dir))
        .map_err(|e| format!("Failed to save index.json: {e}"))?;
    tracing::info!(
        "✅ Dataset build complete: {}/index.json ({} shards / {} batches)",
        output_dir,
        outcome.manifest.global_stats.total_shards,
        outcome.manifest.global_stats.total_batches
    );
    Ok(())
}

/// 写任务产出：manifest 总是带出（失败时用于清理本次已写出的 shard 文件）。
///
/// ★ 评审 §3.7：写线程 panic 时也产出（catch_unwind 兜底）——否则 join 得 Err、
/// 本结构为 None，外层清理 `.unwrap_or(0)` 一个文件都不删。manifest 声明在
/// catch_unwind 之外，panic 后仍带着"已收尾 shard 清单"。
struct WriteOutcome {
    manifest: DatasetManifest,
    result: Result<(), String>,
    cancelled: bool,
}

/// P3a：audio 写线程产出（manifest 合并点 + 失败清理消费）。
///
/// 镜像 `WriteOutcome` 的"总是带出"语义：panic 兜底路径也返回（catch_unwind），
/// 已收尾 audio shard 清单不丢——否则外层清理一个文件都删不到。
struct AudioWriteOutcome {
    /// 已收尾的 audio shard（`ashard_` 序列；并入 manifest.audio_shards）
    shards: Vec<ShardMeta>,
    /// audio 流有效 token 总数（各 shard cu_seqlens 末值累加；不含 padding）
    total_tokens: u64,
    /// VALID（首窗）rec 总数 = `total_audio_clips` 口径（尾窗不计，格式冻结 §9.9）
    total_clips: u64,
    result: Result<(), String>,
    cancelled: bool,
}

impl AudioWriteOutcome {
    /// 空产出（pt/sft 无 audio 流 / 理论上不可达的 join 失败路径）。
    fn empty() -> Self {
        Self {
            shards: Vec::new(),
            total_tokens: 0,
            total_clips: 0,
            result: Ok(()),
            cancelled: false,
        }
    }
}

/// VR-2 / P3c 设计修正：一个 audio batch 的**样本 id 跨度** + 其 `(ashard,
/// batch_in_shard)` 坐标。audio 写线程每写一批追加一条（按写入序）；Vlm 写线程
/// 在 `aud_stream_done` 后据此为每个 Vlm shard **构造性导出** `align_aud_range`
/// （= 本 shard 样本 rec 所在的最小/最大 audio batch）。
///
/// 为何能取代旧的“游标快照”语义（评审 VR-2 + §10.6(3) `first` 缺口）：
/// - audio doc 按 sample_id 序 push、GreedyPacker FIFO 出批 ⇒ 各 batch 的 rec
///   sample_id 跨 batch **单调不减**（`max_sid(G) ≤ min_sid(G+1)`）；
/// - sample_id 按输入序单调分配（source.rs），Vlm shard = 连续 batch 区间
///   ⇒ 每 shard 的 sample_id 构成连续区间 `[lo,hi]`；
/// - 于是“与 `[lo,hi]` 重叠的 audio batch”在 span 序列上是**连续一段**，
///   两次 `partition_point` 二分即得 `[i_lo,i_hi]`，first/last = 两端坐标。
///   该区间是“本 shard 样本 rec 所在 batch”的**最小超集**（不漏不多扫）。
#[derive(Debug, Clone, Copy)]
struct AudioBatchSpan {
    /// 本 batch 全部 rec 的最小 sample_id（无 rec 的 batch 不记录）。
    min_sid: u64,
    /// 本 batch 全部 rec 的最大 sample_id。
    max_sid: u64,
    /// 坐标：audio shard 序号（= `AlignAudRange.first/last.0`）。
    ashard: u32,
    /// 坐标：shard 内 batch 序（= `AlignAudRange.first/last.1`）。
    batch: u32,
}

/// P3a：audio 流 shard 写入状态（独立 `ashard_` 序列 + 第二写线程，镜像
/// `ShardWriteState`——两流 batch 序列各自聚合计数，互不污染）。
struct AudioShardWriteState<'a> {
    output_dir: &'a str,
    pt_compression: Option<ZstdCompression>,
    /// audio 流独立 shard 序号（0,1,2…；与 Vlm 流 shard_idx 无关）
    ashard_idx: usize,
    batches_in_shard: usize,
    valid_tokens_in_shard: u64,
    shard_filename: String,
    memmap_shard: Option<MemmapWriter>,
    /// audio sidecar 三件套（col_aud_blob/rec/cu；与主 shard 同生命周期）
    aud_sidecar: Option<AudioSidecarWriter>,
    /// 该 shard 内 VALID（首窗）rec 数——`total_audio_clips` 计数点（§6.5）
    valid_recs_in_shard: u64,
    /// 半成品登记槽（panic 兜底清理的唯一依据；同 §3.7 纪律）
    in_progress_slot: Arc<Mutex<Option<String>>>,
    /// VR-2：每写完一个含 rec 的 audio batch 追加一条 span（[min_sid,max_sid] +
    /// (ashard,batch) 坐标）；Vlm 写线程收尾后据此构造性导出 align_aud_range。
    /// （audio 流收尾标志 `aud_stream_done` 由写线程闭包的 `audio_done_for_body`
    /// 句柄置位，不经本 state。）
    aud_spans: Arc<Mutex<Vec<AudioBatchSpan>>>,
}

impl AudioShardWriteState<'_> {
    /// P2-1（评审修复）：本 batch 是否仍在续写**当前 shard 内**未收尾的窗口组。
    ///
    /// 判据 = batch 内任一 `AUDIO_TAIL_WINDOW` rec 的 `(sample_id, seg_order)`
    /// 已由当前 writer 写过首窗 VALID。命中 ⇒ 若此刻切 shard，新 writer 的
    /// `seg_ref` 为空，尾窗会报 `has no VALID record in its group`。延后 rollover
    /// 直到该组窗口全部写完（shard 可略超 batch 数上限，正确性优先）。
    fn continues_open_group(&self, ab: &AudioPackedBatch) -> bool {
        let Some(sc) = self.aud_sidecar.as_ref() else {
            return false;
        };
        ab.audio_records.iter().any(|r| {
            r.flags == aud_flags::AUDIO_TAIL_WINDOW && sc.has_valid_group(r.sample_id, r.seg_order)
        })
    }

    /// 追加一个 audio batch：主五件套（Memmap：aud_bin/idx/cu + aud_doc_ids/
    /// aud_loss_weights 旁路列）+ sidecar 三件套。同 batch 顺序配对（§5.4 载体 =
    /// `AudioPackedBatch.audio_records`）。
    fn write_audio_batch(&mut self, ab: &AudioPackedBatch) -> Result<(), String> {
        if self.batches_in_shard == 0 {
            let idx = self.ashard_idx;
            self.shard_filename = format!("ashard_{idx:05}.aud_bin");
            let bin_path = format!("{}/{}", self.output_dir, self.shard_filename);
            let idx_path = format!("{}/ashard_{idx:05}.aud_idx", self.output_dir);
            let cu_path = format!("{}/ashard_{idx:05}.aud_cu", self.output_dir);
            let base = std::path::Path::new(&bin_path);
            // audio 流恒落两列：aud_doc_ids（窗口 doc uid，同源窗口标记）+
            // aud_loss_weights（监督权重；格式冻结 §1 v1.3）
            let doc_ids_path = Some(
                base.with_extension("aud_doc_ids")
                    .to_string_lossy()
                    .into_owned(),
            );
            let loss_weights_path = Some(
                base.with_extension("aud_loss_weights")
                    .to_string_lossy()
                    .into_owned(),
            );
            self.memmap_shard = Some(
                MemmapWriter::with_columns(
                    &bin_path,
                    &idx_path,
                    &cu_path,
                    doc_ids_path.as_deref(),
                    loss_weights_path.as_deref(),
                    None, // audio 流无 col_row_sids（P3c join 索引只服务 Vlm 流）
                    self.pt_compression,
                )
                .map_err(|e| format!("Failed to create MemmapWriter (audio shard {idx}): {e}"))?,
            );
            self.aud_sidecar = Some(
                AudioSidecarWriter::create(
                    std::path::Path::new(self.output_dir),
                    &format!("ashard_{idx:05}"),
                )
                .map_err(|e| {
                    format!("Failed to create AudioSidecarWriter (audio shard {idx}): {e}")
                })?,
            );
            if let Ok(mut slot) = self.in_progress_slot.lock() {
                *slot = Some(self.shard_filename.clone());
            }
        }

        let pb = &ab.batch;
        self.valid_tokens_in_shard += valid_tokens_of_batch(pb);
        let writer = self.memmap_shard.as_mut().unwrap();
        writer
            .append_batch(&pb.input_ids, &pb.cu_seqlens)
            .map_err(|e| {
                format!(
                    "Memmap append failed (audio shard {}): {e}",
                    self.ashard_idx
                )
            })?;
        writer
            .append_columns(&pb.doc_ids, &pb.loss_weights)
            .map_err(|e| {
                format!(
                    "Memmap column append failed (audio shard {}): {e}",
                    self.ashard_idx
                )
            })?;
        // sidecar：owned → 借用；一次调用 = blob+rec+aud_cu 原子成对更新
        let recs: Vec<AudioRecordInput> = ab
            .audio_records
            .iter()
            .map(|r| AudioRecordInput {
                bytes: r.bytes.as_deref(),
                flags: r.flags,
                sample_rate: r.sample_rate,
                duration_ms: r.duration_ms,
                channels: r.channels,
                sample_id: r.sample_id,
                seg_order: r.seg_order,
                start_ms: r.start_ms,
            })
            .collect();
        self.aud_sidecar
            .as_mut()
            .unwrap()
            .append_audio_batch(&recs)
            .map_err(|e| {
                format!(
                    "AudioSidecar append failed (audio shard {}): {e}",
                    self.ashard_idx
                )
            })?;
        // total_audio_clips 口径：只计 VALID（首窗）rec（尾窗 flags=4 不计）
        self.valid_recs_in_shard += ab
            .audio_records
            .iter()
            .filter(|r| r.flags == aud_flags::VALID)
            .count() as u64;
        // VR-2：记录本 batch 的样本 id 跨度（无 rec 的 batch 不记录——其样本的 rec
        // 已归入更早的首现 batch，span 序列的 sid 单调性不受影响）。坐标 batch =
        // 本批写入前的 batches_in_shard（0-based，与旧游标 `batches_in_shard-1` 同义）。
        if !ab.audio_records.is_empty() {
            let mut min_sid = u64::MAX;
            let mut max_sid = 0u64;
            for r in &ab.audio_records {
                if r.sample_id < min_sid {
                    min_sid = r.sample_id;
                }
                if r.sample_id > max_sid {
                    max_sid = r.sample_id;
                }
            }
            if let Ok(mut spans) = self.aud_spans.lock() {
                spans.push(AudioBatchSpan {
                    min_sid,
                    max_sid,
                    ashard: self.ashard_idx as u32,
                    batch: self.batches_in_shard as u32,
                });
            }
        }
        self.batches_in_shard += 1;
        Ok(())
    }

    /// 收尾一个 audio shard：finish 主五件套（增量哈希）+ sidecar 三件套，登记
    /// `ashard` ShardMeta 进聚合产出。hash 键按 audio 流自描述前缀
    /// （`sha256:aud_bin` 等，格式冻结 §5 v1.3）。
    fn finish_current_audio_shard(&mut self, out: &mut AudioWriteOutcome) -> Result<(), String> {
        if self.batches_in_shard == 0 {
            return Ok(());
        }
        let idx = self.ashard_idx;
        let mut writer = self.memmap_shard.take().ok_or_else(|| {
            format!("internal error: Memmap writer for audio shard {idx} does not exist")
        })?;
        let h = writer
            .finish()
            .map_err(|e| format!("Memmap finish failed (audio shard {idx}): {e}"))?;
        let bin_path = format!("{}/{}", self.output_dir, self.shard_filename);
        let size = std::fs::metadata(&bin_path)
            .map_err(|e| format!("Failed to read file metadata (audio shard {idx}): {e}"))?
            .len();
        let mut hashes = BTreeMap::new();
        hashes.insert("sha256:aud_bin".to_string(), h.bin);
        hashes.insert("sha256:aud_idx".to_string(), h.idx);
        hashes.insert("sha256:aud_cu".to_string(), h.cu);
        if let Some(v) = h.doc_ids {
            hashes.insert("sha256:aud_doc_ids".to_string(), v);
        }
        if let Some(v) = h.loss_weights {
            hashes.insert("sha256:aud_loss_weights".to_string(), v);
        }
        if let Some(mut sc) = self.aud_sidecar.take() {
            for (k, v) in sc
                .finish()
                .map_err(|e| format!("AudioSidecar finish failed (audio shard {idx}): {e}"))?
            {
                hashes.insert(k, v);
            }
        }
        out.shards.push(ShardMeta {
            shard_id: idx,
            filename: self.shard_filename.clone(),
            num_batches: self.batches_in_shard,
            size_bytes: size,
            hashes,
            align_aud_range: None,
        });
        out.total_tokens += self.valid_tokens_in_shard;
        out.total_clips += self.valid_recs_in_shard;
        self.ashard_idx += 1;
        self.batches_in_shard = 0;
        self.valid_tokens_in_shard = 0;
        self.valid_recs_in_shard = 0;
        if let Ok(mut slot) = self.in_progress_slot.lock() {
            *slot = None;
        }
        Ok(())
    }

    /// 丢弃当前未收尾的 audio shard：关闭句柄 + 删全部 8 个文件（不留崩溃残留）。
    fn abort_current_audio_shard(&mut self) {
        self.memmap_shard = None;
        self.aud_sidecar = None;
        if self.shard_filename.is_empty() {
            return;
        }
        for name in audio_sidecar_names(&self.shard_filename) {
            let _ = std::fs::remove_file(format!("{}/{}", self.output_dir, name));
        }
        if let Ok(mut slot) = self.in_progress_slot.lock() {
            *slot = None;
        }
    }
}

/// P3a：audio shard 主体 + 旁路文件名（`ashard_00000.aud_bin` → 自身 + 同 stem 的
/// aud_idx/aud_cu/aud_doc_ids/aud_loss_weights + col_aud_blob/rec/cu 三件套，
/// 共 8 个；失败/取消/panic 清理必须全覆盖）。
fn audio_sidecar_names(shard_filename: &str) -> Vec<String> {
    let stem = shard_filename
        .strip_suffix(".aud_bin")
        .unwrap_or(shard_filename);
    [
        ".aud_bin",
        ".aud_idx",
        ".aud_cu",
        ".aud_doc_ids",
        ".aud_loss_weights",
        ".col_aud_blob",
        ".col_aud_rec",
        ".col_aud_cu",
    ]
    .iter()
    .map(|s| format!("{stem}{s}"))
    .collect()
}

/// 从 `catch_unwind` 的 `Box<dyn Any + Send>` 提取 panic 消息（拼进错误文案）。
/// panic 载荷通常是 `&str` / `String`；其它类型（`panic_any` 自定义值）只做标注。
fn panic_detail(panic: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        format!(": {s}")
    } else if let Some(s) = panic.downcast_ref::<String>() {
        format!(": {s}")
    } else {
        " (non-string panic payload)".to_string()
    }
}

/// ★ 评审 §3.7：按 `in_progress_slot` 删除"正在写、未收尾"的 shard 文件
/// （写线程 panic 兜底路径）。返回删除数量。
///
/// 该文件**不在 manifest 里**（`finish_current_shard` 才登记），纯 manifest 驱动
/// 的清理覆盖不到它；槽位为空 / 锁毒化时返回 0——本函数跑在 panic 兜底路径上，
/// 自己绝不能 panic。
fn cleanup_in_progress_shard(slot: &Mutex<Option<String>>, output_dir: &str) -> usize {
    let Ok(guard) = slot.lock() else { return 0 };
    let Some(name) = guard.as_ref() else { return 0 };
    remove_named_files(&sidecar_names(name), output_dir)
}

/// P3a：audio 流版"正在写、未收尾"清理（按 audio 槽位删 ashard_ 8 件套）。
fn cleanup_in_progress_audio_shard(slot: &Mutex<Option<String>>, output_dir: &str) -> usize {
    let Ok(guard) = slot.lock() else { return 0 };
    let Some(name) = guard.as_ref() else { return 0 };
    remove_named_files(&audio_sidecar_names(name), output_dir)
}

/// 删除一组文件（不存在即跳过），返回删除数量。跑在 panic 兜底/失败清理路径上，
/// 自己绝不能 panic（`remove_file` 的 Result 已吞）。
fn remove_named_files(names: &[String], output_dir: &str) -> usize {
    let mut removed = 0;
    for f in names {
        if std::fs::remove_file(format!("{output_dir}/{f}")).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// 删除本次构建已收尾的 shard 文件（含 .cu/.idx 旁路），返回删除数量。
/// 仅在构建失败/取消时调用。
fn remove_partial_shards(manifest: &DatasetManifest, output_dir: &str) -> usize {
    let mut removed = 0;
    for shard in &manifest.shards {
        removed += remove_named_files(&sidecar_names(&shard.filename), output_dir);
    }
    removed
}

/// P3a：删除本次构建已收尾的 **audio 流** shard 文件（ashard_ 8 件套）。
/// 与 `remove_partial_shards` 并列调用（两流批次独立，文件族不同）。
fn remove_partial_audio_shards(shards: &[ShardMeta], output_dir: &str) -> usize {
    let mut removed = 0;
    for shard in shards {
        removed += remove_named_files(&audio_sidecar_names(&shard.filename), output_dir);
    }
    removed
}

/// shard 主体 + 旁路文件名（`shard_00000.bin` → 自身 / `.cu` / `.idx`；Arrow 无旁路）。
///
/// 列扩展的 `.col_doc_ids` / `.col_loss_weights` 也一并列出——构建取消/失败清理时
/// 不能留下这些半成品文件（否则读取端会按 schema key 打开到半截列文件）。
///
/// ★ 评审 §3.7：返回 `Vec` 并**去重**。旧实现固定返回 `[String; 5]`，而 Arrow
///   shard 名是 `shard_00000.arrow`，4 次 `.replace(".bin", ...)` 全部是no-op →
///   返回 5 个**完全相同**的名字 → `remove_partial_shards` 对同一路径调用 5 次
///   `remove_file`（后4 次必然失败），删除计数虚高、无意义。
///   （清理动作本身仍能完成，故此前未被发现。）
fn sidecar_names(shard_filename: &str) -> Vec<String> {
    let mut names: Vec<String> = vec![shard_filename.to_string()];
    // P3b：.col_vid_blob/.col_vid_rec 惰性创建——列在此处保证 abort/panic 清理
    // 覆盖（文件不存在时 remove_file 静默跳过，无副作用）。
    // P3c：.col_row_sids 对齐启用才建——同理列入清理覆盖。
    for suffix in [
        ".cu",
        ".idx",
        ".col_doc_ids",
        ".col_loss_weights",
        ".col_img_blob",
        ".col_img_rec",
        ".mm_cu",
        ".col_vid_blob",
        ".col_vid_rec",
        ".col_row_sids",
    ] {
        let candidate = shard_filename.replace(".bin", suffix);
        if !names.contains(&candidate) {
            names.push(candidate);
        }
    }
    names
}

/// 单个 shard 文件的写入状态（跨多个 batch 复用同一文件句柄，实现 shard 聚合）。
struct ShardWriteState<'a> {
    /// 是否落盘文档级旁路列（`.col_doc_ids` / `.col_loss_weights`）
    extra_columns: bool,
    is_sft: bool,
    output_dir: &'a str,
    pt_compression: Option<ZstdCompression>,
    arrow_writer: &'a ArrowIpcWriter,
    shard_idx: usize,
    batches_in_shard: usize,
    valid_tokens_in_shard: u64,
    shard_filename: String,
    arrow_shard: Option<ArrowShardWriter>,
    memmap_shard: Option<MemmapWriter>,
    /// P2：vlm 模式的图像 sidecar 写入器（col_img_blob/rec/mm_cu；与主 shard 同生命周期）。
    /// pt/sft 恒为 None。打开/收尾/丢弃与主 writer 同步（格式冻结 §7 调用顺序契约）。
    mm_sidecar: Option<MmSidecarWriter>,
    /// P3b：vlm 模式的视频 sidecar 写入器（col_vid_blob/rec；与主 shard 同生命周期）。
    /// pt/sft/无 video 数据集恒为 None。per-batch 计数经 `mm_cu.vid_cnt` 透出
    /// （video 走 Vlm 流，无需独立 .cu）。
    vid_sidecar: Option<VideoSidecarWriter>,
    /// P3b：当前 shard 内已写入的 video blob 字节（shard 字节双约束滚动判据，
    /// §9；仅 vid blob，img blob 既有行为不变）。
    vid_bytes_in_shard: u64,
    /// P3b：video 源时长累加器（demux/写侧共享；manifest 合并点读取，
    /// `total_video_seconds` = Σ duration_ms / 1000，§6.4）。
    video_seconds_acc: Arc<Mutex<f64>>,
    /// P3c：对齐启用（声明了 audio/video id 区间）——决定 col_row_sids 列
    /// 是否落盘 + schema key `row_sids` 是否写入。
    align_enabled: bool,
    /// VR-2：audio 写线程记录的 per-batch span（[min_sid,max_sid] + 坐标）。
    /// `finalize_align_aud_ranges` 在 audio 流收尾后读它，为每个 shard 二分导出区间。
    aud_spans: Arc<Mutex<Vec<AudioBatchSpan>>>,
    /// VR-2：本 shard 已见样本的最小 sample_id（sentinel `u64::MAX` = 尚无真实样本）。
    shard_min_sid: u64,
    /// VR-2：本 shard 已见样本的最大 sample_id（sentinel `0`）。
    shard_max_sid: u64,
    /// VR-2：每个已收尾 Vlm shard 的样本 id 区间 `[lo,hi]`（与 `manifest.shards`
    /// 同序等长；`lo>hi` = 该 shard 无真实样本）。deferred 导出据此二分匹配 span。
    shard_sid_ranges: Vec<(u64, u64)>,
    /// P3c：audio 流全部写完的标志（deferred 导出等它一次，判 spans 完整）。
    aud_stream_done: Arc<AtomicBool>,
    /// P2：是否 vlm 模式（决定 write_vlm_batch 走 sidecar 路径；与 is_sft 互斥）。
    is_vlm: bool,
    /// ★ 评审 §3.7：当前"已打开、未收尾"shard 的文件名（与写线程共享的槽位）。
    ///   写线程 panic 时 `state` 随 unwind 析构、显式的 `abort_current_shard`
    ///   被跳过，外层只能靠这个槽位找到半成品文件并删除——manifest 里只有
    ///   `finish_current_shard` 登记过的 shard，覆盖不到正在写的这个。
    ///   锁毒化安全（`if let Ok`）：本槽位写在 panic 兜底路径上。
    in_progress_slot: Arc<Mutex<Option<String>>>,
}

impl ShardWriteState<'_> {
    /// 追加一个 batch； shard 满员时由调用方调用 finish_current_shard 收尾。
    fn write_batch(&mut self, packed_batch: &PackedBatch) -> Result<(), String> {
        // 开启新 shard 文件
        if self.batches_in_shard == 0 {
            let shard_idx = self.shard_idx;
            if self.is_sft {
                self.shard_filename = format!("shard_{:05}.arrow", shard_idx);
                self.arrow_shard = Some(
                    self.arrow_writer
                        .open(&format!("{}/{}", self.output_dir, self.shard_filename))
                        .map_err(|e| format!("Arrow write failed (shard {shard_idx}): {e}"))?,
                );
            } else {
                self.shard_filename = format!("shard_{:05}.bin", shard_idx);
                let bin_path = format!("{}/{}", self.output_dir, self.shard_filename);
                let idx_path = format!("{}/shard_{:05}.idx", self.output_dir, shard_idx);
                let cu_path = format!("{}/shard_{:05}.cu", self.output_dir, shard_idx);
                // 文档级旁路列（列扩展开启时才有）：shard_00000.bin → .col_doc_ids / .col_loss_weights
                let (doc_ids_path, loss_weights_path) = if self.extra_columns {
                    let base = std::path::Path::new(&bin_path);
                    (
                        Some(
                            base.with_extension("col_doc_ids")
                                .to_string_lossy()
                                .into_owned(),
                        ),
                        Some(
                            base.with_extension("col_loss_weights")
                                .to_string_lossy()
                                .into_owned(),
                        ),
                    )
                } else {
                    (None, None)
                };
                self.memmap_shard = Some(
                    MemmapWriter::with_columns(
                        &bin_path,
                        &idx_path,
                        &cu_path,
                        doc_ids_path.as_deref(),
                        loss_weights_path.as_deref(),
                        None,
                        self.pt_compression,
                    )
                    .map_err(|e| {
                        format!("Failed to create MemmapWriter (shard {shard_idx}): {e}")
                    })?,
                );
            }
            // ★ §3.7：登记"正在写"的 shard（panic 清理的唯一依据；finish/abort 时清除）
            if let Ok(mut slot) = self.in_progress_slot.lock() {
                *slot = Some(self.shard_filename.clone());
            }
        }

        self.valid_tokens_in_shard += valid_tokens_of_batch(packed_batch);
        if self.is_sft {
            self.arrow_shard
                .as_mut()
                .unwrap()
                .write_batch(packed_batch)
                .map_err(|e| format!("Arrow write failed (shard {}): {e}", self.shard_idx))?;
        } else {
            let writer = self.memmap_shard.as_mut().unwrap();
            writer
                .append_batch(&packed_batch.input_ids, &packed_batch.cu_seqlens)
                .map_err(|e| format!("Memmap append failed (shard {}): {e}", self.shard_idx))?;
            writer
                .append_columns(&packed_batch.doc_ids, &packed_batch.loss_weights)
                .map_err(|e| {
                    format!(
                        "Memmap column append failed (shard {}): {e}",
                        self.shard_idx
                    )
                })?;
        }
        self.batches_in_shard += 1;
        Ok(())
    }

    /// P2：追加一个 vlm batch——主 token 流走 Memmap（与 PT 同 writer），图像记录走
    /// `MmSidecarWriter`（col_img_blob/rec/mm_cu）。同 batch 顺序配对（格式冻结 §7）。
    ///
    /// VR-2：顺带追踪本 shard 的样本 id 区间（`shard_min_sid/max_sid`），收尾时登记
    /// 进 `shard_sid_ranges`，供 audio 流收尾后构造性导出 `align_aud_range`。
    fn write_vlm_batch(
        &mut self,
        vb: &crate::packing::modality_aware::VlmPackedBatch,
    ) -> Result<(), String> {
        // 开启新 shard（vlm 走 Memmap 主文件 + sidecar）
        if self.batches_in_shard == 0 {
            let shard_idx = self.shard_idx;
            self.shard_filename = format!("shard_{:05}.bin", shard_idx);
            let bin_path = format!("{}/{}", self.output_dir, self.shard_filename);
            let idx_path = format!("{}/shard_{:05}.idx", self.output_dir, shard_idx);
            let cu_path = format!("{}/shard_{:05}.cu", self.output_dir, shard_idx);
            // vlm 总是落 loss_weights 旁路（图像段 0.0 / 文本段 1.0 = 监督掩码）
            let base = std::path::Path::new(&bin_path);
            let doc_ids_path = Some(
                base.with_extension("col_doc_ids")
                    .to_string_lossy()
                    .into_owned(),
            );
            let loss_weights_path = Some(
                base.with_extension("col_loss_weights")
                    .to_string_lossy()
                    .into_owned(),
            );
            // P3c：对齐启用（声明了 audio/video id 区间）时才落 col_row_sids
            // ——行→sid 承载；不启用则不产该文件（产物与 P2/P3a/P3b 期逐字节一致）
            let row_sids_path = if self.align_enabled {
                Some(
                    base.with_extension("col_row_sids")
                        .to_string_lossy()
                        .into_owned(),
                )
            } else {
                None
            };
            self.memmap_shard = Some(
                MemmapWriter::with_columns(
                    &bin_path,
                    &idx_path,
                    &cu_path,
                    doc_ids_path.as_deref(),
                    loss_weights_path.as_deref(),
                    row_sids_path.as_deref(),
                    self.pt_compression,
                )
                .map_err(|e| {
                    format!("Failed to create MemmapWriter (vlm shard {shard_idx}): {e}")
                })?,
            );
            // sidecar：与主 shard 同 stem（col_img_blob / col_img_rec / mm_cu）
            let stem = format!("shard_{:05}", shard_idx);
            self.mm_sidecar = Some(
                MmSidecarWriter::create(std::path::Path::new(self.output_dir), &stem).map_err(
                    |e| format!("Failed to create MmSidecarWriter (vlm shard {shard_idx}): {e}"),
                )?,
            );
            // P3b：video sidecar **惰性创建**（该 shard 首个 video 记录时才开文件）
            // ——无 video 的 shard 不产 col_vid_* 文件、不写 schema key，图像路径
            // 产物逐字节不变（P3b §8.4 回归锚点）。
            if let Ok(mut slot) = self.in_progress_slot.lock() {
                *slot = Some(self.shard_filename.clone());
            }
        }

        let pb = &vb.batch;
        self.valid_tokens_in_shard += valid_tokens_of_batch(pb);
        let writer = self.memmap_shard.as_mut().unwrap();
        writer
            .append_batch(&pb.input_ids, &pb.cu_seqlens)
            .map_err(|e| format!("Memmap append failed (vlm shard {}): {e}", self.shard_idx))?;
        // doc_ids 为空（VLM 用 .cu + rec.sample_id 表达行↔样本）；loss_weights 落盘
        writer
            .append_columns(&pb.doc_ids, &pb.loss_weights)
            .map_err(|e| {
                format!(
                    "Memmap column append failed (vlm shard {}): {e}",
                    self.shard_idx
                )
            })?;
        // 图像记录 → sidecar（owned → 借用）。一次调用 = blob+rec+mm_cu 原子成对更新。
        let recs: Vec<ImageRecordInput> = vb
            .image_records
            .iter()
            .map(|r| ImageRecordInput {
                bytes: r.bytes.as_deref(),
                grid_h: r.grid_h,
                grid_w: r.grid_w,
                flags: r.flags,
                sample_id: r.sample_id,
                seg_order: r.seg_order,
            })
            .collect();
        // mm_cu.vid_cnt = 同 batch 的 video 记录数（P3b §3；vid_cnt=0 时 mm_cu
        // 字节与 P2 期逐字节一致）
        self.mm_sidecar
            .as_mut()
            .unwrap()
            .append_mm_batch(&recs, vb.video_records.len() as u32)
            .map_err(|e| {
                format!(
                    "MmSidecar append failed (vlm shard {}): {e}",
                    self.shard_idx
                )
            })?;
        // VR-2：追踪本 shard 的样本 id 区间（排除 padding 行的 NO_SAMPLE_SID）。
        // sample_id 按输入序单调分配、Vlm shard = 连续 batch 区间 ⇒ 每 shard 的
        // sid 构成连续区间 [lo,hi]，收尾时登记，供 deferred 二分匹配 audio span。
        for &sid in &vb.row_sample_ids {
            if sid == crate::packing::modality_aware::NO_SAMPLE_SID {
                continue;
            }
            if sid < self.shard_min_sid {
                self.shard_min_sid = sid;
            }
            if sid > self.shard_max_sid {
                self.shard_max_sid = sid;
            }
        }
        // P3c：行→sid 列（对齐启用才落；writer 未开列时空切片 = 不写字节）
        if self.align_enabled {
            let writer = self.memmap_shard.as_mut().unwrap();
            writer.append_row_sids(&vb.row_sample_ids).map_err(|e| {
                format!(
                    "col_row_sids append failed (vlm shard {}): {e}",
                    self.shard_idx
                )
            })?;
        }
        // P3b：video 记录 → VideoSidecarWriter（owned → 借用；blob/rec 成对更新）
        if !vb.video_records.is_empty() {
            // 惰性创建：该 shard 首个 video 记录时才开 col_vid_blob/rec 两文件
            if self.vid_sidecar.is_none() {
                let stem = format!("shard_{:05}", self.shard_idx);
                self.vid_sidecar = Some(
                    VideoSidecarWriter::create(std::path::Path::new(self.output_dir), &stem)
                        .map_err(|e| {
                            format!(
                                "Failed to create VideoSidecarWriter (vlm shard {}): {e}",
                                self.shard_idx
                            )
                        })?,
                );
            }
            let vrecs: Vec<VideoRecordInput> = vb
                .video_records
                .iter()
                .map(|r| VideoRecordInput {
                    bytes: r.bytes.as_deref(),
                    n_frames: r.n_frames,
                    width: r.width,
                    height: r.height,
                    flags: r.flags,
                    sample_id: r.sample_id,
                    seg_order: r.seg_order,
                    fps_milli: r.fps_milli,
                    start_ms: r.start_ms,
                })
                .collect();
            self.vid_sidecar
                .as_mut()
                .unwrap()
                .append_vid_batch(&vrecs)
                .map_err(|e| {
                    format!(
                        "VideoSidecar append failed (vlm shard {}): {e}",
                        self.shard_idx
                    )
                })?;
            // shard 字节账本（仅 vid blob；§9 双约束滚动判据）
            let blob_bytes: u64 = vrecs
                .iter()
                .map(|v| v.bytes.map_or(0, |b| b.len() as u64))
                .sum();
            self.vid_bytes_in_shard += blob_bytes;
            // total_video_seconds：源视频时长口径（Σ duration_ms / 1000，§6.4）
            let secs: f64 = vb
                .video_records
                .iter()
                .map(|r| r.duration_ms as f64 / 1000.0)
                .sum();
            *self.video_seconds_acc.lock().unwrap() += secs;
        }
        self.batches_in_shard += 1;
        Ok(())
    }
    fn finish_current_shard(
        &mut self,
        manifest: &mut DatasetManifest,
        progress: &Arc<BuildProgress>,
    ) -> Result<(), String> {
        if self.batches_in_shard == 0 {
            return Ok(()); // 没有未收尾的数据
        }
        let shard_idx = self.shard_idx;
        // 0.3 Tier-0：PT 路径由 MemmapWriter::finish 直接产出增量哈希，免 5 文件回读；
        // Arrow 路径传 None，finish_shard 内回读单个 .arrow 文件。
        let pt_hashes = if self.is_sft {
            let writer = self.arrow_shard.take().ok_or_else(|| {
                format!("internal error: Arrow writer for shard {shard_idx} does not exist")
            })?;
            writer
                .finish()
                .map_err(|e| format!("Arrow finish failed (shard {shard_idx}): {e}"))?;
            None
        } else {
            let mut writer = self.memmap_shard.take().ok_or_else(|| {
                format!("internal error: Memmap writer for shard {shard_idx} does not exist")
            })?;
            Some(
                writer
                    .finish()
                    .map_err(|e| format!("Memmap finish failed (shard {shard_idx}): {e}"))?,
            )
        };
        // P3c：col_row_sids 哈希在 pt_hashes move 前取出（对齐启用才落）
        let row_sids_hash_opt = pt_hashes.as_ref().and_then(|h| h.row_sids.clone());
        finish_shard(
            &self.shard_filename,
            &format!("{}/{}", self.output_dir, self.shard_filename),
            shard_idx,
            self.batches_in_shard,
            self.valid_tokens_in_shard,
            manifest,
            progress,
            pt_hashes,
        )?;
        // P2：vlm 收尾 sidecar——finish 三文件哈希，并入该 shard 的 hashes +
        // manifest schema 三 key（img_blob/img_rec/mm_cu）。格式冻结 §5。
        if self.is_vlm
            && let Some(mut sc) = self.mm_sidecar.take()
        {
            let mm_hashes = sc
                .finish()
                .map_err(|e| format!("MmSidecar finish failed (shard {shard_idx}): {e}"))?;
            if let Some(shard) = manifest.shards.last_mut() {
                for (k, v) in mm_hashes {
                    shard.hashes.insert(k, v);
                }
            }
            manifest
                .schema
                .insert("img_blob".to_string(), "bytes".to_string());
            manifest
                .schema
                .insert("img_rec".to_string(), "imgrec32".to_string());
            manifest
                .schema
                .insert("mm_cu".to_string(), "mmcu".to_string());
            // vlm 总是落 loss_weights 旁路列（图像段 0.0 / 文本段 1.0 = 监督掩码，
            // T3 "image 段默认 -100" 的持久化形态）；doc_ids 恒空不落盘、不写 key。
            manifest
                .schema
                .insert("loss_weights".to_string(), "list<f16>".to_string());
        }
        // P3b：video sidecar 收尾——finish 两文件哈希并入该 shard 的 hashes +
        // manifest schema 两 key（vid_blob/vid_rec；格式冻结 §5 v1.4）。
        // 打开判定三 key = vid_rec + vid_blob + mm_cu（mm_cu 由上面 img 分支写入）。
        if self.is_vlm
            && let Some(mut vsc) = self.vid_sidecar.take()
        {
            let vid_hashes = vsc
                .finish()
                .map_err(|e| format!("VideoSidecar finish failed (shard {shard_idx}): {e}"))?;
            if let Some(shard) = manifest.shards.last_mut() {
                for (k, v) in vid_hashes {
                    shard.hashes.insert(k, v);
                }
            }
            manifest
                .schema
                .insert("vid_blob".to_string(), "bytes".to_string());
            manifest
                .schema
                .insert("vid_rec".to_string(), "vidrec40".to_string());
        }
        // P3c：col_row_sids 哈希并入 + schema key（对齐启用才落）
        if self.align_enabled {
            let row_sids_hash = row_sids_hash_opt.ok_or_else(|| {
                    format!(
                        "internal error: col_row_sids hash missing for vlm shard {shard_idx}                          (writer not opened with the row_sids column?)"
                    )
                })?;
            if let Some(shard) = manifest.shards.last_mut() {
                shard
                    .hashes
                    .insert("sha256:col_row_sids".to_string(), row_sids_hash);
            }
            manifest
                .schema
                .insert("row_sids".to_string(), "list<uint64>".to_string());
            // VR-2：登记本 shard 的样本 id 区间（与 manifest.shards 同序）；
            // align_aud_range 留待 audio 流收尾后由 finalize_align_aud_ranges 统一导出。
            self.shard_sid_ranges
                .push((self.shard_min_sid, self.shard_max_sid));
        }
        self.shard_idx += 1;
        self.batches_in_shard = 0;
        self.valid_tokens_in_shard = 0;
        self.vid_bytes_in_shard = 0; // P3b：字节账本随 shard 归零
        self.shard_min_sid = u64::MAX; // VR-2：sid 区间随 shard 重置
        self.shard_max_sid = 0;
        // ★ §3.7：已登记进 manifest → 半成品槽位交权（后续清理走 manifest 路径）
        if let Ok(mut slot) = self.in_progress_slot.lock() {
            *slot = None;
        }
        Ok(())
    }

    /// VR-2 / P3c 设计修正：audio 流整体收尾后，据 audio 写线程记录的 per-batch
    /// span + 每个 Vlm shard 的样本 id 区间，**构造性导出**每个 shard 的
    /// `align_aud_range`（= 本 shard 样本 rec 所在的最小/最大 audio batch）。
    ///
    /// 取代旧"游标快照"语义，一并消除两个同族缺口：
    /// - `first` 非超集（§10.6(3)，原 xfail）：旧 `first = 上一 shard 快照游标`，
    ///   audio 写线程领先 Vlm 写线程时本 shard 样本的 rec 可能落在 `first` 之前；
    /// - VR-2 `last` 漏覆盖：末 shard 由滚动触发收尾时只等到水位，漏掉尾部 flush。
    ///
    /// 正确性依据：① audio doc 按 sample_id 序 push + GreedyPacker FIFO 出批 ⇒
    /// span 序列 `[min_sid,max_sid]` 跨 batch 单调不减（`max_sid(G) ≤ min_sid(G+1)`）；
    /// ② sample_id 按输入序单调、Vlm shard = 连续 batch 区间 ⇒ 每 shard sid 构成
    /// 连续区间 `[lo,hi]`；③ 于是"与 `[lo,hi]` 重叠的 span"在序列上是连续一段，
    /// 两次 `partition_point` 二分即得 `[i_lo,i_hi]`，端点坐标 = first/last。该区间
    /// 是"本 shard 样本 rec 所在 batch"的最小超集（不漏不多扫；无重叠 ⇒ None）。
    ///
    /// 无死锁：调用点在主循环外，`rx_packed` 已断流 ⇒ compute 结束并 drop
    /// `tx_audio_packed` ⇒ audio 写线程 drain 完必置 `aud_stream_done`（P3c-2：
    /// Ok/Err/panic 三出口均置位）；audio 写线程从不等待 Vlm 写线程，不构成环等。
    fn finalize_align_aud_ranges(&mut self, manifest: &mut DatasetManifest) {
        // 等 audio 流整体收尾（spans 完整）。
        while !self.aud_stream_done.load(Ordering::Acquire) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let spans = self.aud_spans.lock().map(|g| g.clone()).unwrap_or_default();
        if spans.is_empty() {
            return; // 无任何 audio 写入 ⇒ 所有 shard align_aud_range 保持 None
        }
        // shard_sid_ranges 与 manifest.shards 同序等长（每次 finish_current_shard
        // 各追加一条）；逐 shard 二分匹配重叠 span 区间。
        for (i, &(lo, hi)) in self.shard_sid_ranges.iter().enumerate() {
            if lo > hi {
                continue; // 无真实样本的 shard ⇒ None
            }
            // i_lo = 首个 max_sid ≥ lo 的 span（之前的都在 lo 之前结束，不重叠）
            let i_lo = spans.partition_point(|s| s.max_sid < lo);
            // i_hi_excl = 首个 min_sid > hi 的 span（之前的都在 hi 之前开始，可能重叠）
            let i_hi_excl = spans.partition_point(|s| s.min_sid <= hi);
            if i_lo >= spans.len() || i_hi_excl == 0 || i_lo >= i_hi_excl {
                continue; // 无重叠 span ⇒ 本 shard 样本无 audio rec ⇒ None
            }
            let i_hi = i_hi_excl - 1;
            let range = crate::builder::metadata::AlignAudRange {
                first: (spans[i_lo].ashard, spans[i_lo].batch),
                last: (spans[i_hi].ashard, spans[i_hi].batch),
            };
            if let Some(shard) = manifest.shards.get_mut(i) {
                shard.align_aud_range = Some(range);
            }
        }
    }

    /// 丢弃当前未收尾的 shard：关闭写入器并删除已写出的文件（含 .idx/.cu 旁路）。
    /// 构建失败/取消时调用，避免在 output_dir 留下半成品文件。
    fn abort_current_shard(&mut self) {
        // 先 drop 写入器（关闭文件句柄），再删文件
        self.arrow_shard = None;
        self.memmap_shard = None;
        self.mm_sidecar = None; // P2：丢弃 sidecar 句柄（关闭三文件）
        self.vid_sidecar = None; // P3b：丢弃 video sidecar 句柄（关闭两文件）
        if self.shard_filename.is_empty() {
            return;
        }
        for name in sidecar_names(&self.shard_filename) {
            let _ = std::fs::remove_file(format!("{}/{}", self.output_dir, name));
        }
        // ★ §3.7：显式删除已完成，清掉半成品槽位（panic 兜底路径不再重复删）
        if let Ok(mut slot) = self.in_progress_slot.lock() {
            *slot = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P3c：ts token 插入单测（§4.3–4.5 规则 + §4.4 机制）。
    mod ts_token_tests {
        use super::*;
        use crate::builder::encoded_sample::{EncodedSample, MediaRecord, SegmentEntry};
        use crate::builder::sample::SegmentKind;

        fn sid(v: u64) -> SampleId {
            SampleId(v)
        }

        /// 样本：Text(0, ts=2500ms) + Image(1)——Image 段带 VideoMeta-less rec。
        fn mk_sample(_with_ts: bool) -> EncodedSample {
            EncodedSample {
                sample_id: sid(0),
                token_seq: vec![10, 11, 12, 20, 21],
                segment_table: vec![
                    SegmentEntry {
                        sample_id: sid(0),
                        segment_order: 0,
                        kind: SegmentKind::Text,
                        token_start: 0,
                        token_end: 3,
                        media_record_id: None,
                        loss_weight: 255,
                    },
                    SegmentEntry {
                        sample_id: sid(0),
                        segment_order: 1,
                        kind: SegmentKind::Image,
                        token_start: 3,
                        token_end: 5,
                        media_record_id: Some(0),
                        loss_weight: 0,
                    },
                ],
                labels: vec![10, 11, 12, -100, -100],
                loss_weights: None,
                stats: EncodedStats::default(),
                media_refs: vec![MediaRecord {
                    sample_id: sid(0),
                    seg_order: 1,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![1]),
                    grid_h: 1,
                    grid_w: 2,
                    flags: 0,
                    video: None,
                }],
            }
        }

        fn pre(av: bool, text_ts: Option<i64>, dur: Option<f32>) -> TsPreExtract {
            let mut ts_map = HashMap::new();
            if let Some(v) = text_ts {
                ts_map.insert((0u64, 0u32), v);
            }
            let mut av_track = HashSet::new();
            if av {
                av_track.insert(0u64);
            }
            let mut durations = HashMap::new();
            durations.insert(0u64, dur);
            TsPreExtract {
                ts_map,
                av_track,
                durations,
            }
        }

        /// 基本插入：ts token 落在 Text 段首、并入该段；labels=-100、权重 0.0；
        /// 后续段坐标 +1；validate 通过。
        #[test]
        fn insert_basic_merges_into_text_segment() {
            let mut es = mk_sample(true);
            let warns = TsSoftWarns::default();
            // 源侧 ts=2.5s → start_ms=2500；W=2000 → bucket 1 → id = base+1
            let pre = pre(true, Some(2500), Some(4.0));
            insert_ts_tokens(&mut es, &pre, 32000, 2000, &warns).unwrap();
            assert_eq!(es.token_seq, vec![32001, 10, 11, 12, 20, 21]);
            assert_eq!(es.labels[0], -100);
            // ts token 并入 Text(0)：段长 3→4，Image 段坐标 +1
            assert_eq!(es.segment_table[0].token_len(), 4);
            assert_eq!(es.segment_table[1].token_start, 4);
            assert_eq!(es.segment_table[1].token_end, 6);
            // loss_weights 物化：Text 位 1.0（0x3C00）、ts 位 0.0、Image 位 0.0
            let lw = es.loss_weights.as_ref().unwrap();
            assert_eq!(lw.len(), 12);
            assert_eq!(&lw[0..2], &[0x00, 0x00], "ts 位权重 0.0");
            assert_eq!(&lw[2..4], &[0x00, 0x3C], "text 位权重 1.0");
            assert_eq!(&lw[10..12], &[0x00, 0x00], "image 位权重 0.0");
            es.validate().unwrap();
        }

        /// 无 A/V 轨样本（R4 硬规则）→ 不插（text-only 样本的 sid 不可恢复）。
        #[test]
        fn no_av_track_sample_never_inserts() {
            let mut es = mk_sample(true);
            let pre = pre(false, Some(2500), None);
            insert_ts_tokens(&mut es, &pre, 32000, 2000, &TsSoftWarns::default()).unwrap();
            assert_eq!(es.token_seq, vec![10, 11, 12, 20, 21], "无 A/V 轨 → 零插入");
        }

        /// Text 段无 timestamp → 不插（缺失即语义）+ 软 WARN 不失败。
        #[test]
        fn text_without_timestamp_skipped_with_warn() {
            let mut es = mk_sample(false);
            let pre = pre(true, None, Some(4.0));
            insert_ts_tokens(&mut es, &pre, 32000, 2000, &TsSoftWarns::default()).unwrap();
            assert_eq!(es.token_seq.len(), 5, "无 timestamp → 零插入");
        }

        /// 负 timestamp / 超 id 区间预算 → fail-fast（§4.5 恒等层）。
        #[test]
        fn negative_or_over_budget_fail_fast() {
            let mut es = mk_sample(true);
            let e = insert_ts_tokens(
                &mut es,
                &pre(true, Some(-5), Some(4.0)),
                32000,
                2000,
                &TsSoftWarns::default(),
            )
            .unwrap_err();
            assert!(e.contains("negative"), "{e}");
            // 桶 5000 ≥ range_size 2048
            let e = insert_ts_tokens(
                &mut es,
                &pre(true, Some(10_000_000), Some(4.0)),
                32000,
                2000,
                &TsSoftWarns::default(),
            )
            .unwrap_err();
            assert!(e.contains("exceeds the ts token id budget"), "{e}");
        }

        /// 样本时长超上界（已知 duration_sec）→ fail-fast（预校验层）。
        #[test]
        fn over_duration_fail_fast() {
            let mut es = mk_sample(true);
            let e = insert_ts_tokens(
                &mut es,
                &pre(true, Some(0), Some(7200.0)),
                32000,
                2000,
                &TsSoftWarns::default(),
            )
            .unwrap_err();
            assert!(e.contains("exceeds the max sample duration"), "{e}");
        }

        /// 两点插入互不扰动（降序处理）：两个 Text 段各插一个 ts token。
        #[test]
        fn multi_insert_descending_positions() {
            let mut es = EncodedSample {
                sample_id: sid(0),
                token_seq: vec![10, 11, 20, 21, 22],
                segment_table: vec![
                    SegmentEntry {
                        sample_id: sid(0),
                        segment_order: 0,
                        kind: SegmentKind::Text,
                        token_start: 0,
                        token_end: 2,
                        media_record_id: None,
                        loss_weight: 255,
                    },
                    SegmentEntry {
                        sample_id: sid(0),
                        segment_order: 2,
                        kind: SegmentKind::Text,
                        token_start: 2,
                        token_end: 5,
                        media_record_id: None,
                        loss_weight: 255,
                    },
                ],
                labels: vec![10, 11, 20, 21, 22],
                loss_weights: None,
                stats: EncodedStats::default(),
                media_refs: vec![],
            };
            let mut ts_map = HashMap::new();
            ts_map.insert((0u64, 0u32), 2500i64); // bucket 1
            ts_map.insert((0u64, 2u32), 4500i64); // bucket 2
            let mut av_track = HashSet::new();
            av_track.insert(0u64);
            let pre = TsPreExtract {
                ts_map,
                av_track,
                durations: HashMap::from([(0u64, Some(6.0))]),
            };
            insert_ts_tokens(&mut es, &pre, 32000, 2000, &TsSoftWarns::default()).unwrap();
            assert_eq!(es.token_seq, vec![32001, 10, 11, 32002, 20, 21, 22]);
            assert_eq!(es.segment_table[0].token_len(), 3);
            assert_eq!(es.segment_table[1].token_len(), 4);
            es.validate().unwrap();
        }
    }

    /// ★ 评审 §5.1：构建管线此前**零 Rust 单测**（48KB 核心代码只靠 Python e2e 覆盖）。
    ///   20260927 评审已把 `resolve_batches_per_shard` 列为 P1-6 并建议加单测，
    ///   至今未落地 —— 本模块补上。
    mod resolve_batches_per_shard_tests {
        use super::super::resolve_batches_per_shard;
        use crate::builder::pipeline::TARGET_SHARD_BYTES;
        use crate::builder::writer::sft_fixed_width_columns;

        /// 自动档位必须真的产出 ~128MiB 的 shard（而不是某个魔数）。
        #[test]
        fn auto_targets_128mib_per_shard() {
            // PT + 无列扩展：每 batch = bs*msl*4 字节
            let (bs, msl) = (16usize, 8192usize);
            let n = resolve_batches_per_shard(0, bs, msl, false, false);
            let bytes_per_batch = (bs * msl * 4) as u64;
            let total = n as u64 * bytes_per_batch;
            assert!(
                (total as i64 - TARGET_SHARD_BYTES as i64).abs() < bytes_per_batch as i64,
                "auto shard size {total}B should be within one batch of {TARGET_SHARD_BYTES}B (got {n} batches)"
            );
        }

        /// SFT 的定宽列数从 schema 推导（不是 magic number）——这个契约必须成立，
        /// 否则 schema 加列时`resolve_batches_per_shard` 会静默算错 shard 粒度。
        #[test]
        fn sft_column_count_comes_from_schema() {
            let (bs, msl) = (16usize, 8192usize);
            let cols = sft_fixed_width_columns();
            assert!(
                cols >= 3,
                "SFT schema should have >=3 fixed-width columns, got {cols}"
            );
            // SFT 自动档位按 cols 个定宽列计算，且必然小于 PT（同样 batch 下字节更多）
            let sft = resolve_batches_per_shard(0, bs, msl, true, false);
            let pt = resolve_batches_per_shard(0, bs, msl, false, false);
            assert!(
                sft < pt,
                "SFT ({sft} batches/shard) must be smaller than PT ({pt}) for the same shape"
            );
        }

        /// 列扩展让每batch 变大 → 每 shard 的 batch 数变少（否则 shard 会超128MiB）。
        #[test]
        fn extra_columns_reduce_batches_per_shard() {
            let (bs, msl) = (16usize, 8192usize);
            let plain = resolve_batches_per_shard(0, bs, msl, false, false);
            let with_cols = resolve_batches_per_shard(0, bs, msl, false, true);
            assert!(
                with_cols < plain,
                "extra_columns (6 bytes/token) must shrink batches/shard: {with_cols} < {plain}"
            );
        }

        /// 显式请求值原样透传（只做 max(1) 下限保护）。
        #[test]
        fn explicit_request_is_respected() {
            assert_eq!(resolve_batches_per_shard(7, 16, 8192, false, false), 7);
            assert_eq!(resolve_batches_per_shard(1, 16, 8192, true, true), 1);
            // 0 → 自动；下界保护：任何形状都至少 1 个 batch
            for (bs, msl) in [(1usize, 1usize), (1, 1_000_000), (4096, 8192)] {
                let n = resolve_batches_per_shard(0, bs, msl, false, false);
                assert!(
                    (1..=super::super::MAX_BATCHES_PER_SHARD).contains(&n),
                    "shape ({bs},{msl}) -> {n} out of clamp range"
                );
            }
        }
    }

    /// ★ 评审 §3.7：旧实现固定返回 `[String; 5]`，Arrow shard（`shard_00000.arrow`）
    ///   的 4 次 `.replace(".bin", ..)` 全是 no-op → 返回 5 个同名 →
    ///   `remove_partial_shards` 对同一路径删 5 次（后 4 次必然失败），计数虚高。
    #[test]
    fn sidecar_names_are_deduplicated() {
        // PT：主体 + 4 个文本旁路 + 3 个 vlm img sidecar + 2 个 P3b vid sidecar
        // + 1 个 P3c row_sids
        let pt = sidecar_names("shard_00000.bin");
        assert_eq!(
            pt.len(),
            11,
            "PT shard should have 11 distinct names (incl. vlm sidecar): {pt:?}"
        );
        assert!(pt.contains(&"shard_00000.cu".to_string()));
        assert!(pt.contains(&"shard_00000.idx".to_string()));
        assert!(pt.contains(&"shard_00000.col_doc_ids".to_string()));
        assert!(pt.contains(&"shard_00000.col_loss_weights".to_string()));
        assert!(
            pt.contains(&"shard_00000.col_img_blob".to_string()),
            "vlm img blob must be cleaned: {pt:?}"
        );
        assert!(
            pt.contains(&"shard_00000.col_img_rec".to_string()),
            "vlm img rec must be cleaned: {pt:?}"
        );
        assert!(
            pt.contains(&"shard_00000.mm_cu".to_string()),
            "vlm mm_cu must be cleaned: {pt:?}"
        );
        assert!(
            pt.contains(&"shard_00000.col_vid_blob".to_string()),
            "P3b vid blob must be cleaned: {pt:?}"
        );
        assert!(
            pt.contains(&"shard_00000.col_vid_rec".to_string()),
            "P3b vid rec must be cleaned: {pt:?}"
        );

        // Arrow：无旁路，`.replace(".bin", ..)` 全是 no-op → 必须去重成 1 个
        let arrow = sidecar_names("shard_00000.arrow");
        assert_eq!(
            arrow,
            vec!["shard_00000.arrow".to_string()],
            "Arrow shard must not produce duplicate sidecar names"
        );

        // 去重后不可能有重复项（通用性质）：排序去重后的长度必须等于原长度
        for name in [
            "shard_00000.bin",
            "shard_00000.arrow",
            "weird.name.with.dots",
        ] {
            let names = sidecar_names(name);
            let mut sorted = names.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(
                names.len(),
                sorted.len(),
                "sidecar_names({name}) returned duplicates: {names:?}"
            );
        }
    }

    /// ★ 评审 §3.7：panic 兜底清理必须真的删掉"正在写"的 shard（含旁路），
    /// 且槽位为空 / 锁毒化时安静返回 0（本路径自己绝不能 panic）。
    #[test]
    fn cleanup_in_progress_shard_removes_half_written_files() {
        let dir =
            std::env::temp_dir().join(format!("megadata_pipe_inprogress_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // 造一个"写了一半"的 PT shard：主体 + .cu/.idx 旁路 + 旁路列
        for f in [
            "shard_00000.bin",
            "shard_00000.cu",
            "shard_00000.idx",
            "shard_00000.col_doc_ids",
            "shard_00000.col_loss_weights",
        ] {
            std::fs::write(dir.join(f), vec![0u8; 16]).unwrap();
        }
        // 一个不属于本次构建的文件（模拟同目录旧产物）：不应被误删
        std::fs::write(dir.join("keep_me.bin"), vec![0u8; 16]).unwrap();
        std::fs::write(dir.join("index.json"), b"{}").unwrap();

        let slot: Mutex<Option<String>> = Mutex::new(Some("shard_00000.bin".to_string()));
        let removed = cleanup_in_progress_shard(&slot, dir.to_str().unwrap());
        assert_eq!(removed, 5, "PT shard 主体+4 旁路应全部删除，实际 {removed}");
        for f in [
            "shard_00000.bin",
            "shard_00000.cu",
            "shard_00000.idx",
            "shard_00000.col_doc_ids",
            "shard_00000.col_loss_weights",
        ] {
            assert!(!dir.join(f).exists(), "{f} 应被删除");
        }
        assert!(
            dir.join("keep_me.bin").exists(),
            "非 shard_ 前缀文件不应被误删"
        );
        assert!(
            dir.join("index.json").exists(),
            "index.json 不在本函数职责内"
        );

        // 槽位为空 → 0（正常收尾后不应再删任何东西）
        let empty: Mutex<Option<String>> = Mutex::new(None);
        assert_eq!(cleanup_in_progress_shard(&empty, dir.to_str().unwrap()), 0);

        // 锁毒化 → 0 且不 panic（本函数跑在 panic 兜底路径上）
        let poisoned: Mutex<Option<String>> = Mutex::new(Some("shard_00000.bin".to_string()));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = poisoned.lock().unwrap();
            panic!("poison the slot lock on purpose");
        }));
        assert_eq!(
            cleanup_in_progress_shard(&poisoned, dir.to_str().unwrap()),
            0
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P3a：audio shard 文件名族（ashard_ 8 件套）——失败/取消/panic 清理的覆盖
    /// 面；去重且以 .aud_bin 为 stem 基准。
    #[test]
    fn audio_sidecar_names_cover_eight_files() {
        let names = audio_sidecar_names("ashard_00003.aud_bin");
        assert_eq!(names.len(), 8);
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            names.len(),
            sorted.len(),
            "audio shard names must be unique: {names:?}"
        );
        for expect in [
            "ashard_00003.aud_bin",
            "ashard_00003.aud_idx",
            "ashard_00003.aud_cu",
            "ashard_00003.aud_doc_ids",
            "ashard_00003.aud_loss_weights",
            "ashard_00003.col_aud_blob",
            "ashard_00003.col_aud_rec",
            "ashard_00003.col_aud_cu",
        ] {
            assert!(
                names.contains(&expect.to_string()),
                "missing {expect}: {names:?}"
            );
        }
        // Vlm 流 shard 名传入不得崩（stem 基准容错）：按原名 + 8 后缀展开
        let odd = audio_sidecar_names("shard_00000.bin");
        assert_eq!(odd.len(), 8);
        assert!(odd[0].starts_with("shard_00000.bin"));
    }

    /// `panic_detail`：str / String 载荷提取消息，其它载荷只做标注（不 panic）。
    #[test]
    fn panic_detail_extracts_message() {
        let s: Box<dyn std::any::Any + Send> = Box::new("boom");
        assert_eq!(panic_detail(&*s), ": boom");
        let owned: Box<dyn std::any::Any + Send> = Box::new(String::from("disk full"));
        assert_eq!(panic_detail(&*owned), ": disk full");
        let other: Box<dyn std::any::Any + Send> = Box::new(42u8);
        assert_eq!(panic_detail(&*other), " (non-string panic payload)");
    }

    /// `valid_tokens_of_batch`：cu_seqlens 缺失/非法时退回满载计数（旧行为，保证不崩）。
    #[test]
    fn valid_tokens_falls_back_when_cu_seqlens_unusable() {
        use crate::packing::document_aware::PackedBatch;
        let mk = |cu: Vec<i32>| PackedBatch {
            input_ids: vec![0; 8 * 4],
            labels: vec![],
            position_ids: vec![],
            cu_seqlens: cu,
            row_lens: vec![],
            doc_ids: vec![],
            loss_weights: vec![],
            batch_size: 8,
            max_seq_len: 4,
        };
        assert_eq!(
            valid_tokens_of_batch(&mk(vec![0, 10, 20])),
            20,
            "last cu value is the valid token count"
        );
        assert_eq!(
            valid_tokens_of_batch(&mk(vec![])),
            32,
            "empty cu_seqlens falls back to batch_size*max_seq_len"
        );
        assert_eq!(
            valid_tokens_of_batch(&mk(vec![0, -5])),
            32,
            "negative cu value falls back"
        );
    }

    /// ★ F2-PT（评审 §3.5）：PT 超长文档必须被计入切分统计——这是本次修复的核心不变量。
    #[test]
    fn pt_long_documents_are_counted_as_split() {
        use crate::packing::greedy::GreedyPacker;
        // capacity = 2*8 = 16 token
        let mut packer = GreedyPacker::new(8, 9, 2, false);
        // 一篇 40token 的文档 >> 16 → 必须被计为split
        let out = packer.push_docs(vec![(vec![1; 39], 0u32)]); // +eos = 40
        assert!(
            !out.is_empty(),
            "40 tokens must produce at least one full batch"
        );
        let (split_docs, split_tokens) = packer.take_split_stats();
        assert_eq!(
            split_docs, 1,
            "the 40-token doc (capacity 16) must be counted as split"
        );
        assert_eq!(
            split_tokens, 40,
            "split_tokens must record the doc's full token count (incl. eos)"
        );

        // 短文档不计入
        let mut packer2 = GreedyPacker::new(8, 9, 2, false);
        let _ = packer2.push_docs(vec![(vec![1; 4], 0u32)]); // +eos = 5 < 16
        assert_eq!(
            packer2.take_split_stats(),
            (0, 0),
            "a short doc must not be counted as split"
        );

        // take_split_stats 是"换出"语义：第二次取应为 0
        let mut packer3 = GreedyPacker::new(8, 9, 2, false);
        let _ = packer3.push_docs(vec![(vec![1; 39], 0u32)]);
        assert_ne!(packer3.take_split_stats(), (0, 0));
        assert_eq!(
            packer3.take_split_stats(),
            (0, 0),
            "take_split_stats must drain (swap to 0)"
        );
    }

    // ------------------------------------------------------------------
    // P3a：audio demux（提取 / 切窗 / 配对）单测
    // ------------------------------------------------------------------

    use crate::builder::encoded_sample::{EncodedSample, EncodedStats, SegmentEntry};
    use crate::builder::sample::{DecodePolicy, MediaMeta, Payload, SampleId, Segment};

    fn aud_src(
        bytes: Vec<u8>,
        sr: Option<u32>,
        ch: Option<u8>,
        dur: Option<f32>,
        ts: Option<f64>,
    ) -> AudioSegSource {
        AudioSegSource {
            bytes,
            codec: Codec::AudioFlac,
            sample_rate: sr,
            channels: ch,
            duration_sec: dur,
            timestamp: ts,
        }
    }

    /// 测试用 demux 参数（W=2000ms / rate=50 tok/s；capacity 由用例指定）。
    fn demux_params<'a>(
        uid: &'a AtomicU32,
        warns: &'a AudioSoftWarns,
        capacity: usize,
    ) -> AudioDemuxParams<'a> {
        AudioDemuxParams {
            w_ms: 2000,
            window_token_rate: 50,
            capacity,
            strict_monotonic: false,
            uid_counter: uid,
            soft_warns: warns,
        }
    }

    /// P3a §4.1 反例对照（评审 P1-1 回归锚点 A11）：W=2000ms、段 [1500,5500)
    /// → 子段 [1500,2000)/[2000,4000)/[4000,5500)，各属 w0/w1/w2——锁定
    /// **样本时间轴锚定**（段起点锚定的旧公式会产跨窗子段，已废弃）。
    #[test]
    fn test_split_windows_sample_timeline_anchored() {
        // rate=50 tok/s → tokens_per_ms=0.05；n=200 token ⇒ token 隐含时长 4000ms
        let tokens: Vec<i32> = (0..200).collect();
        let src = aud_src(vec![0xAA; 300], Some(16000), Some(1), Some(4.0), Some(1.5));
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let docs = split_audio_segment_to_windows(7, 2, &tokens, &src, &params).unwrap();
        assert_eq!(docs.len(), 3, "[1500,5500) W=2000 → 3 子段");
        // 子段真实起点 / 时长 / 归属窗
        assert_eq!((docs[0].rec.start_ms, docs[0].rec.duration_ms), (1500, 500));
        assert_eq!(
            (docs[1].rec.start_ms, docs[1].rec.duration_ms),
            (2000, 2000)
        );
        assert_eq!(
            (docs[2].rec.start_ms, docs[2].rec.duration_ms),
            (4000, 1500)
        );
        for (k, d) in docs.iter().enumerate() {
            assert_eq!(d.rec.start_ms / 2000, k as u32, "子段恒属单窗 k");
        }
        // 首窗 VALID 携整段字节；尾窗引用（demux 侧 bytes=None，writer 侧解析）
        assert_eq!(docs[0].rec.flags, aud_flags::VALID);
        assert_eq!(docs[0].rec.bytes.as_ref().unwrap().len(), 300);
        for t in &docs[1..] {
            assert_eq!(t.rec.flags, aud_flags::AUDIO_TAIL_WINDOW);
            assert!(t.rec.bytes.is_none());
            assert_eq!(t.rec.seg_order, 2, "seg_order 继承原段");
        }
        // token 区间首尾相接、无重无丢（A9）：25 + 100 + 75 = 200
        assert_eq!(docs.iter().map(|d| d.tokens.len()).sum::<usize>(), 200);
        assert_eq!(docs[0].tokens, (0..25).collect::<Vec<i32>>());
        assert_eq!(docs[1].tokens, (25..125).collect::<Vec<i32>>());
        assert_eq!(docs[2].tokens, (125..200).collect::<Vec<i32>>());
    }

    /// 零 token 的首窗（k=k_min）不发射时，**首个发射窗**承接 VALID 字节归属——
    /// 否则尾窗引用不到首窗（writer seg_ref 构造性检测会 fail-fast）。
    #[test]
    fn test_split_windows_first_emitted_carries_bytes() {
        // start_ms=1999（W=2000）：首窗 [1999,2000) 仅 1ms → floor(1×0.05)=0 token → 跳过
        let tokens: Vec<i32> = (0..100).collect();
        let src = aud_src(vec![0xBB; 64], Some(8000), Some(2), None, Some(1.999));
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let docs = split_audio_segment_to_windows(1, 0, &tokens, &src, &params).unwrap();
        assert_eq!(docs.len(), 1, "仅第二个窗有 token");
        assert_eq!(docs[0].rec.flags, aud_flags::VALID, "首个发射窗必须 VALID");
        assert_eq!(docs[0].rec.start_ms, 2000);
        assert_eq!(docs[0].rec.bytes.as_ref().unwrap().len(), 64);
        // duration_sec 缺失 → token 隐含时长回退（end_ms = 1999+2000=3999）仍正确切窗
        assert_eq!(docs[0].tokens.len(), 100, "末窗吸收尾差：全部 token");
        // 软 WARN 计数：duration 回退记一次
        assert_eq!(warns.duration_fallback.load(Ordering::Relaxed), 1);
    }

    /// INV-2 前置检查：timestamp 缺失 / sr 缺失 / channels 缺失 → fail-fast 含 sid；
    /// 零 token 段 → 静默跳过（SkipAndCount 语义，不产 rec）。
    #[test]
    fn test_split_windows_precondition_fail_fast() {
        let tokens: Vec<i32> = (0..50).collect();
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        // timestamp 缺失
        let e = split_audio_segment_to_windows(
            9,
            0,
            &tokens,
            &aud_src(vec![1], Some(1), Some(1), Some(1.0), None),
            &params,
        )
        .unwrap_err();
        assert!(e.contains("no timestamp") && e.contains("sample 9"), "{e}");
        // sr 缺失（N-1 定稿裁定）
        let e = split_audio_segment_to_windows(
            9,
            0,
            &tokens,
            &aud_src(vec![1], None, Some(1), Some(1.0), Some(0.0)),
            &params,
        )
        .unwrap_err();
        assert!(e.contains("sample_rate"), "{e}");
        // channels 缺失
        let e = split_audio_segment_to_windows(
            9,
            0,
            &tokens,
            &aud_src(vec![1], Some(1), None, Some(1.0), Some(0.0)),
            &params,
        )
        .unwrap_err();
        assert!(e.contains("channels"), "{e}");
        // 零 token → 空（不进流）
        let docs = split_audio_segment_to_windows(
            9,
            0,
            &[],
            &aud_src(vec![1], Some(1), Some(1), Some(1.0), Some(0.0)),
            &params,
        )
        .unwrap();
        assert!(docs.is_empty());
        // 对齐参数缺失（w_ms=0）→ fail-fast
        let p0 = AudioDemuxParams {
            w_ms: 0,
            ..demux_params(&uid, &warns, 4096)
        };
        let e = split_audio_segment_to_windows(
            9,
            0,
            &tokens,
            &aud_src(vec![1], Some(1), Some(1), Some(1.0), Some(0.0)),
            &p0,
        )
        .unwrap_err();
        assert!(e.contains("alignment config missing"), "{e}");
    }

    /// 非正时长（duration_sec=0）→ INV-2 硬校验 fail-fast。
    #[test]
    fn test_split_windows_non_positive_duration_fails() {
        let tokens: Vec<i32> = (0..50).collect();
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let e = split_audio_segment_to_windows(
            3,
            1,
            &tokens,
            &aud_src(vec![1], Some(1), Some(1), Some(0.0), Some(1.0)),
            &params,
        )
        .unwrap_err();
        assert!(e.contains("invalid duration"), "{e}");
    }

    /// 准入条件（§4.3）：W×rate/1000 + 1(eos) > capacity → 配置错误 fail-fast。
    #[test]
    fn test_split_windows_budget_admission_fails() {
        // W=2000 × rate=50 / 1000 = 100 token/窗；+1 eos = 101 > capacity=100
        let tokens: Vec<i32> = (0..50).collect();
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 100);
        let e = split_audio_segment_to_windows(
            4,
            0,
            &tokens,
            &aud_src(vec![1], Some(1), Some(1), Some(1.0), Some(0.0)),
            &params,
        )
        .unwrap_err();
        assert!(e.contains("exceeds stream capacity"), "{e}");
        // capacity=101：100+1 ≤ 101 → 准入通过（W 整数窗恰满时不再误杀）
        let params_ok = demux_params(&uid, &warns, 101);
        assert!(
            split_audio_segment_to_windows(
                4,
                0,
                &tokens,
                &aud_src(vec![1], Some(1), Some(1), Some(1.0), Some(0.0)),
                &params_ok
            )
            .is_ok()
        );
    }

    /// 超长窗 fail-fast（§4.3）：末窗吸收尾差后单窗 token + eos > capacity
    /// （duration 短但 token 多的不一致段）→ 硬失败含 sid + 窗号 + token 数。
    #[test]
    fn test_split_windows_oversized_window_fails() {
        // 准入：100+1=101 ≤ 128 通过；但 duration=1s（单窗）而 n_tokens=200
        // → 末窗 tok_end=200，+1 > 128 → fail
        let tokens: Vec<i32> = (0..200).collect();
        let src = aud_src(vec![1; 16], Some(16000), Some(1), Some(1.0), Some(0.0));
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 128);
        let e = split_audio_segment_to_windows(6, 1, &tokens, &src, &params).unwrap_err();
        assert!(
            e.contains("oversized audio window") && e.contains("sample 6"),
            "{e}"
        );
        assert!(e.contains("200 tokens"), "{e}");
    }

    /// INV-2 硬校验：start_ms > T×1000（duration_sec 已知）→ fail-fast 含 sid。
    #[test]
    fn test_split_windows_start_ms_upper_bound_fails() {
        let tokens: Vec<i32> = (0..50).collect();
        // T = 1.0s ⇒ t_ms=1000；timestamp=2.0 ⇒ start_ms=2000 > 1000
        let src = aud_src(vec![1; 16], Some(16000), Some(1), Some(1.0), Some(2.0));
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let e = split_audio_segment_to_windows(11, 0, &tokens, &src, &params).unwrap_err();
        assert!(
            e.contains("INV-2 hard") && e.contains("start_ms 2000 > T×1000 1000"),
            "{e}"
        );
    }

    /// demux 驱动：text+audio+text → Vlm 侧过滤样本 + 窗口 doc（uid 连续）；
    /// audio_seconds 按源段计一次（不每窗一次）。
    #[test]
    fn test_demux_audio_samples_separates_streams() {
        // text[0..3) + audio[3..53) + text[53..56)：audio 50 token @50tok/s = 1000ms = 1 窗
        let es = EncodedSample {
            sample_id: SampleId(5),
            token_seq: vec![7; 56],
            segment_table: vec![
                SegmentEntry {
                    sample_id: SampleId(5),
                    segment_order: 0,
                    kind: SegmentKind::Text,
                    token_start: 0,
                    token_end: 3,
                    media_record_id: None,
                    loss_weight: 255,
                },
                SegmentEntry {
                    sample_id: SampleId(5),
                    segment_order: 1,
                    kind: SegmentKind::Audio,
                    token_start: 3,
                    token_end: 53,
                    media_record_id: None,
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: SampleId(5),
                    segment_order: 2,
                    kind: SegmentKind::Text,
                    token_start: 53,
                    token_end: 56,
                    media_record_id: None,
                    loss_weight: 255,
                },
            ],
            labels: vec![100; 56],
            loss_weights: None,
            stats: EncodedStats {
                audio_tokens: 50,
                ..Default::default()
            },
            media_refs: vec![],
        };
        let mut sources = HashMap::new();
        sources.insert(
            (5u64, 1u32),
            aud_src(vec![0xCC; 128], Some(16000), Some(1), Some(1.0), Some(0.5)),
        );
        let uid = AtomicU32::new(100);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let out = demux_audio_samples(vec![es], &sources, &params).unwrap();
        // Vlm 侧：audio 剔除后剩两段 text（6 token）
        assert_eq!(out.vlm_samples.len(), 1);
        let f = &out.vlm_samples[0];
        assert_eq!(f.token_seq.len(), 6);
        assert_eq!(f.segment_table.len(), 2);
        assert!(f.validate().is_ok());
        // audio 侧：1 个窗口 doc（[500,1500) ∈ w0），uid 从 100 起
        assert_eq!(out.window_docs.len(), 1);
        let (tokens, doc_uid, rec) = &out.window_docs[0];
        assert_eq!(tokens.len(), 50);
        assert_eq!(*doc_uid, 100);
        assert_eq!(rec.flags, aud_flags::VALID);
        assert_eq!(rec.start_ms, 500);
        assert_eq!(rec.sample_id, 5);
        // audio_seconds：源段 1.0s（一次，不每窗）
        assert!((out.audio_seconds - 1.0).abs() < 1e-9);
    }

    /// 纯 audio 样本 → 不进 Vlm 流（vlm_samples 空），窗口 doc 照常。
    #[test]
    fn test_demux_pure_audio_skips_vlm_stream() {
        let es = EncodedSample {
            sample_id: SampleId(8),
            token_seq: vec![7; 10],
            segment_table: vec![SegmentEntry {
                sample_id: SampleId(8),
                segment_order: 0,
                kind: SegmentKind::Audio,
                token_start: 0,
                token_end: 10,
                media_record_id: None,
                loss_weight: 0,
            }],
            labels: vec![-100; 10],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let mut sources = HashMap::new();
        sources.insert(
            (8u64, 0u32),
            aud_src(vec![0xDD; 32], Some(16000), Some(1), Some(0.2), Some(0.0)),
        );
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let out = demux_audio_samples(vec![es], &sources, &params).unwrap();
        assert!(out.vlm_samples.is_empty(), "纯 audio 样本不进 vlm packer");
        assert_eq!(out.window_docs.len(), 1);
    }

    /// demux 时 audio 源缺失（encoder 未保持 sample_id/seg_order）→ fail-fast。
    #[test]
    fn test_demux_missing_source_fails() {
        let es = EncodedSample {
            sample_id: SampleId(8),
            token_seq: vec![7; 10],
            segment_table: vec![SegmentEntry {
                sample_id: SampleId(8),
                segment_order: 3,
                kind: SegmentKind::Audio,
                token_start: 0,
                token_end: 10,
                media_record_id: None,
                loss_weight: 0,
            }],
            labels: vec![-100; 10],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let e = demux_audio_samples(vec![es], &HashMap::new(), &params).unwrap_err();
        assert!(e.contains("not found in pre-extracted"), "{e}");
    }

    /// INV-2 软校验（§7）：同 (sid, Audio) 内 seg_order 递增但 start_ms 倒退
    /// → 缺省 WARN（限频计数）；strict_monotonic → fail-fast。
    #[test]
    fn test_demux_start_ms_regression_soft_and_strict() {
        // seg_order 0 @5000ms、seg_order 1 @1000ms → 倒退
        let es = EncodedSample {
            sample_id: SampleId(12),
            token_seq: vec![7; 100],
            segment_table: vec![
                SegmentEntry {
                    sample_id: SampleId(12),
                    segment_order: 0,
                    kind: SegmentKind::Audio,
                    token_start: 0,
                    token_end: 50,
                    media_record_id: None,
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: SampleId(12),
                    segment_order: 1,
                    kind: SegmentKind::Audio,
                    token_start: 50,
                    token_end: 100,
                    media_record_id: None,
                    loss_weight: 0,
                },
            ],
            labels: vec![-100; 100],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let mut sources = HashMap::new();
        // dur=5.0s ⇒ T×1000=5000 ≥ 各段 start_ms（INV-2 硬上界自洽）；
        // 两段各只首窗有 token（token 隐含时长 1000ms 远小于声明时长，其余窗零 token 跳过）
        sources.insert(
            (12u64, 0u32),
            aud_src(vec![1; 16], Some(16000), Some(1), Some(5.0), Some(5.0)),
        );
        sources.insert(
            (12u64, 1u32),
            aud_src(vec![2; 16], Some(16000), Some(1), Some(5.0), Some(1.0)),
        );

        // 缺省：WARN（软校验计数 +1），不 fail
        let uid = AtomicU32::new(0);
        let warns = AudioSoftWarns::default();
        let params = demux_params(&uid, &warns, 4096);
        let out = demux_audio_samples(vec![es.clone()], &sources, &params).unwrap();
        assert_eq!(out.window_docs.len(), 2, "两段各一窗（均 1000ms ≤ W）");
        assert_eq!(warns.start_ms_regression.load(Ordering::Relaxed), 1);

        // strict：fail-fast 含 sid
        let uid2 = AtomicU32::new(0);
        let warns2 = AudioSoftWarns::default();
        let strict = AudioDemuxParams {
            strict_monotonic: true,
            ..demux_params(&uid2, &warns2, 4096)
        };
        let e = demux_audio_samples(vec![es], &sources, &strict).unwrap_err();
        assert!(
            e.contains("INV-2 soft (strict)") && e.contains("sample 12"),
            "{e}"
        );
    }

    /// extract_audio_sources：段级 payload 优先、回落样本级；meta/timestamp 取值。
    #[test]
    fn test_extract_audio_sources_payload_fallback() {
        // 交错样本：audio 段字节挂 Segment.payload
        let inter = Sample::interleaved(
            SampleId(1),
            SourceRef::Inline,
            vec![Segment {
                kind: SegmentKind::Audio,
                order: 0,
                len_hint: None,
                supervise: Supervise::LossOn,
                source_ref: SourceRef::Inline,
                timestamp: Some(2.5),
                text: None,
                payload: Some(Payload::new(vec![1, 2, 3], Codec::AudioFlac)),
            }],
            Some(MediaMeta {
                sample_rate: Some(44100),
                channels: Some(2),
                duration_sec: Some(3.0),
                ..Default::default()
            }),
        );
        // 单模态样本：字节在 Sample.payload（音频单模态样本无专用构造器，
        // 结构体字面量构造——P0 起 audio 段只经 interleaved / 源读取器产出）
        let single = Sample {
            sample_id: SampleId(2),
            source_ref: SourceRef::Inline,
            payload: Some(Payload::new(vec![9, 9], Codec::AudioMp3)),
            segments: vec![Segment {
                kind: SegmentKind::Audio,
                order: 0,
                len_hint: None,
                supervise: Supervise::LossOn,
                source_ref: SourceRef::Inline,
                timestamp: None,
                text: None,
                payload: None,
            }],
            media_meta: Some(MediaMeta {
                sample_rate: Some(8000),
                channels: Some(1),
                duration_sec: None,
                ..Default::default()
            }),
            decode_policy: DecodePolicy::DecodeInEncoder,
            error_class: None,
            text: None,
        };
        let map = extract_audio_sources(&[inter, single]);
        let a = map.get(&(1, 0)).unwrap();
        assert_eq!(a.bytes, vec![1, 2, 3], "段级 payload 优先");
        assert_eq!(a.timestamp, Some(2.5));
        assert_eq!(a.sample_rate, Some(44100));
        assert_eq!(a.duration_sec, Some(3.0));
        let b = map.get(&(2, 0)).unwrap();
        assert_eq!(b.bytes, vec![9, 9], "回落样本级 payload");
        assert_eq!(b.timestamp, None, "image 构造器无 timestamp");
    }

    /// pair_audio_records：按 cu_seqlens 段边界 + doc_ids uid 配对；uid 缺失 → fail-fast。
    #[test]
    fn test_pair_audio_records_by_uid() {
        // 两个 doc（uid 10/11）拼成一个 batch：cu [0,4,9]，doc_ids 逐 token
        let batch = PackedBatch {
            input_ids: vec![0; 9],
            labels: vec![],
            position_ids: vec![],
            cu_seqlens: vec![0, 4, 9],
            row_lens: vec![],
            doc_ids: vec![10, 10, 10, 10, 11, 11, 11, 11, 11],
            loss_weights: vec![],
            batch_size: 1,
            max_seq_len: 9,
        };
        let mk = |uid: u32, flags: u8| AudioRecordInputOwned {
            bytes: if flags == aud_flags::VALID {
                Some(vec![uid as u8])
            } else {
                None
            },
            flags,
            sample_rate: 16000,
            duration_ms: 100,
            channels: 1,
            sample_id: uid as u64,
            seg_order: 0,
            start_ms: 0,
        };
        let mut pending = HashMap::new();
        pending.insert(10u32, mk(10, aud_flags::VALID));
        pending.insert(11u32, mk(11, aud_flags::AUDIO_TAIL_WINDOW));
        let mut claimed = HashSet::new();
        let recs = pair_audio_records(&batch, &mut pending, &mut claimed).unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].sample_id, 10);
        assert_eq!(recs[1].flags, aud_flags::AUDIO_TAIL_WINDOW);
        assert!(pending.is_empty(), "配对后 pending 应清空");
        assert_eq!(claimed.len(), 2);

        // doc 跨 batch 切分（GreedyPacker PT 语义，边界切分无法避免）：uid 11 的
        // doc 跨两个 batch → rec 归首段 batch，延续段不重复产 rec
        let cont = PackedBatch {
            cu_seqlens: vec![0, 3], // uid 11 的延续段（3 token）
            doc_ids: vec![11, 11, 11],
            ..batch.clone()
        };
        let mut pending_c = HashMap::new();
        pending_c.insert(10u32, mk(10, aud_flags::VALID));
        pending_c.insert(11u32, mk(11, aud_flags::AUDIO_TAIL_WINDOW));
        let mut claimed_c = HashSet::new();
        let recs_first = pair_audio_records(&batch, &mut pending_c, &mut claimed_c).unwrap();
        assert_eq!(recs_first.len(), 2, "首段 batch 拥有两条 rec");
        let recs_cont = pair_audio_records(&cont, &mut pending_c, &mut claimed_c).unwrap();
        assert!(
            recs_cont.is_empty(),
            "延续段不重复产 rec（rec 归首段 batch）"
        );
        assert!(pending_c.is_empty());

        // uid 从未见过（配对 bug）→ fail-fast
        let mut pending2 = HashMap::new();
        pending2.insert(10u32, mk(10, aud_flags::VALID));
        let mut claimed2 = HashSet::new();
        let e = pair_audio_records(&batch, &mut pending2, &mut claimed2).unwrap_err();
        assert!(
            e.contains("doc uid 11") && e.contains("never claimed"),
            "{e}"
        );

        // extra_columns 关闭（doc_ids 空）→ fail-fast（audio 流恒开）
        let no_cols = PackedBatch {
            doc_ids: vec![],
            ..batch
        };
        let e = pair_audio_records(&no_cols, &mut HashMap::new(), &mut HashSet::new()).unwrap_err();
        assert!(e.contains("extra_columns=true"), "{e}");
    }

    /// P4-1（评审）：`resolve_source_to_local` 本地路径直通 + 无效远程路径报错。
    /// 远程路径物化需真实云基建（不在单测覆盖）。
    #[test]
    fn resolve_source_to_local_passes_local_path() {
        let io_mgr = crate::io::CloudIoManager::new();
        // 本地路径直通（不触碰 io_mgr）
        assert_eq!(
            resolve_source_to_local(&io_mgr, "/tmp/data.jsonl").unwrap(),
            "/tmp/data.jsonl"
        );
        assert_eq!(
            resolve_source_to_local(&io_mgr, "relative/path.jsonl").unwrap(),
            "relative/path.jsonl"
        );
        // 非 s3/gs/http(s) 前缀的路径视为本地
        assert_eq!(
            resolve_source_to_local(&io_mgr, "foo://bar/baz.jsonl").unwrap(),
            "foo://bar/baz.jsonl"
        );
    }

    #[test]
    fn resolve_source_to_local_errors_on_bad_remote_path() {
        let io_mgr = crate::io::CloudIoManager::new();
        // 远程前缀但无法拆分 base/filename → 报错
        let e = resolve_source_to_local(&io_mgr, "s3://").unwrap_err();
        assert!(e.contains("cannot split"), "{e}");
    }

    /// P4-1（评审）：manifest source_formats + source_field_map 透出逻辑测试。
    /// 验证 `SourceFormat::detect` 对混合路径的去重排序 + FieldMap 摘要构建。
    #[test]
    fn source_formats_and_field_map_summary() {
        use crate::builder::source::{FieldMap, SourceFormat};
        // 去重排序
        let paths = vec![
            "a.jsonl".to_string(),
            "b.tar".to_string(),
            "c.jsonl".to_string(),
        ];
        let mut formats: Vec<String> = Vec::new();
        for p in &paths {
            let f = match SourceFormat::detect(p) {
                Ok(SourceFormat::Jsonl) => "jsonl",
                Ok(SourceFormat::Tar) => "webdataset",
                Ok(SourceFormat::Parquet) => "parquet",
                Err(_) => continue,
            };
            let s = f.to_string();
            if !formats.contains(&s) {
                formats.push(s);
            }
        }
        formats.sort();
        assert_eq!(formats, vec!["jsonl", "webdataset"]);

        // FieldMap 摘要（含 parquet 默认列名）
        let fm = FieldMap::default();
        let mut summary: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        summary.insert("image_fields".to_string(), fm.image_fields.clone());
        summary.insert("text_fields".to_string(), fm.text_fields.clone());
        summary.insert("audio_fields".to_string(), fm.audio_fields.clone());
        summary.insert("video_fields".to_string(), fm.video_fields.clone());
        summary.insert("meta_fields".to_string(), fm.meta_fields.clone());
        // P4-4：timestamp_field 缺省 → 省略该键（与生产逻辑一致）
        if let Some(v) = fm.timestamp_field.clone() {
            summary.insert("timestamp_field".to_string(), vec![v]);
        }
        // parquet 列名默认应存在（P4-2）
        assert!(
            summary["image_fields"].iter().any(|f| f == "image_bytes"),
            "parquet image_bytes default missing"
        );
        assert!(
            summary["image_fields"].iter().any(|f| f == "image_png"),
            "parquet image_png default missing"
        );
        // timestamp_field 缺省 → 省略该键（P4-4：不再用空列表表示 None）
        assert!(!summary.contains_key("timestamp_field"));
    }
}
