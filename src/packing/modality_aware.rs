//! VLM 模态感知装箱器（最终方案 §4.4 D4 / D1 / D7，P2-T2/T3）。
//!
//! **净新增**：`greedy.rs` / `document_aware.rs` 一行不动；VLM 走 `mode="vlm"` 分派
//! 到本 packer。图像/视频 → SFT 式定宽行模型（单样本 = 单行，`bs × msl` 恒定）。
//!
//! **核心不变式（D1 + 可追溯链）**：
//! - `cu_seqlens` 记**样本边界**（每行一个样本 → cu_seqlens 即行边界），不按模态段切。
//! - 模态段边界走旁路：本 packer 消费 `EncodedSample.segment_table` 设置 labels /
//!   loss_weights，并产出图像 sidecar 记录（与主 batch 同 batch 顺序配对落盘）。
//! - **图像段原子性**（D4）：一个图像段的 placeholder token 不跨 row 劈开——
//!   单样本=单行下，图像段要么整段进当前行、要么整段丢弃（不计入 token 流）。
//! - **样本级总视觉预算**（D7/T2）：`max_visual_tokens_per_sample` 硬约束每样本
//!   image 段 token 总数；超限按 `BudgetPolicy` 截断+计数 或 硬失败（禁止静默超限）。
//! - **VLM 标签规则**（T3）：image 段 label 默认 -100（`image_label` 可配置）；
//!   loss_weights：image 段 0.0、text 段 1.0（或 encoder 提供的权重）。
//!
//! 输出 `VlmPackedBatch` = 兼容现有 writer 的 `PackedBatch` + 该 batch 的图像 sidecar
//! 记录 + 截断统计。pipeline 将 `batch` 喂 `MemmapWriter`、`image_records` 喂
//! `MmSidecarWriter::append_mm_batch`（同 batch 顺序配对，格式冻结 §7 契约）。

use std::sync::atomic::{AtomicU64, Ordering};

use crate::builder::encoded_sample::EncodedSample;
use crate::builder::sample::SegmentKind;
use crate::packing::document_aware::{F16_ONE_LE, F16_ZERO_LE, PackedBatch};

/// 视觉预算超限策略（D7 / T2）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPolicy {
    /// 截断多余图像段 + 计数入 manifest（默认）。
    TruncateAndCount,
    /// 硬失败（构建中止；用于严格场景）。
    HardFail,
}

/// sidecar 图像记录的 owned 形态（跨 flume 通道传输；落盘时由 pipeline 转
/// `ImageRecordInput<'a>` 借用给 `MmSidecarWriter`）。
#[derive(Debug, Clone)]
pub struct ImageRecordInputOwned {
    /// `Some(非空)` = VALID 载荷；`None` = 占位（NO_IMAGE/EMPTY/ERRORED）。
    pub bytes: Option<Vec<u8>>,
    pub grid_h: u16,
    pub grid_w: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
}

/// sidecar 视频记录的 owned 形态（P3b；镜像 `ImageRecordInputOwned` 先例——
/// 跨 flume 通道传输，落盘时由 pipeline 转 `VideoRecordInput<'a>` 借用给
/// `VideoSidecarWriter`）。`duration_ms` 仅统计口径用（`total_video_seconds`），
/// **不落 rec**（格式冻结 §2.3 无该字段）。
#[derive(Debug, Clone)]
pub struct VideoRecordInputOwned {
    pub bytes: Option<Vec<u8>>,
    pub n_frames: u16,
    pub width: u16,
    pub height: u16,
    pub flags: u8,
    pub sample_id: u64,
    pub seg_order: u32,
    pub fps_milli: u32,
    pub start_ms: u32,
    pub duration_ms: u32,
}

/// VLM 单次装箱产出：主 batch + 配对图像/视频 sidecar 记录 + 截断统计。
#[derive(Debug)]
pub struct VlmPackedBatch {
    /// 与现有 writer 兼容的主 batch（input_ids/labels/position_ids/cu_seqlens/
    /// row_lens + doc_ids/loss_weights 旁路列）。
    pub batch: PackedBatch,
    /// 该 batch 的图像 sidecar 记录（顺序 = batch 内 row 顺序，row 内按 seg_order）。
    /// 由 `MmSidecarWriter::append_mm_batch` 落盘，与 `MemmapWriter::append_batch`
    /// 同 batch 顺序配对（格式冻结 §7 调用顺序契约）。
    pub image_records: Vec<ImageRecordInputOwned>,
    /// 该 batch 的视频 sidecar 记录（P3b；同 batch 配对，由
    /// `VideoSidecarWriter::append_vid_batch` 落盘；per-batch 计数经
    /// `mm_cu.vid_cnt` 透出）。
    pub video_records: Vec<VideoRecordInputOwned>,
    /// P3c：每行的样本 id（行长序；`col_row_sids` 列的落盘载体）。
    /// packer 逐样本 `build_row`（一行 = 一样本，F14）⇒ 行序 = 样本序；
    /// 装箱逻辑零改动（batch 元数据面增量，P3c §4.6）。
    pub row_sample_ids: Vec<u64>,
}

/// VLM 模态感知装箱器（**有状态**：跨 chunk 累积行直到满一个 batch）。
pub struct ModalityAwarePacker {
    max_seq_len: usize,
    pad_id: i32,
    batch_size: usize,
    extra_columns: bool,
    /// 样本级总视觉预算（D7）：每样本 image 段 token 总数上限（≤ msl）。
    max_visual_tokens_per_sample: u32,
    /// 超预算策略。
    visual_budget_policy: BudgetPolicy,
    /// 图像段默认 label（T3：默认 -100，可配置）。
    image_label: i32,

    // ---- 跨 chunk 缓冲（攒满 batch_size 行才出批）----
    rows: Vec<RowBuffer>,
    // ---- 截断统计（跨 chunk 原子累计，pipeline 经 take_stats 汇入 BuildProgress）----
    visual_truncated: AtomicU64,
    tokens_dropped_to_msl: AtomicU64,
    // ---- 实际入包的 per-modality 计数（manifest 模态拆分真源；post-drop）----
    // 必须取 pack 后的值：encoder 的 es.stats 是 pre-drop，原子性丢弃 / 视觉预算
    // 截断 / text 尾部截断会让 total_tokens(packed) != Σ 模态(encoder) 而破坏守恒。
    packed_text_tokens: AtomicU64,
    packed_image_tokens: AtomicU64,
    packed_images: AtomicU64,
    packed_video_tokens: AtomicU64,
    packed_video_clips: AtomicU64,
}

/// 单行的中间缓冲（攒满 batch_size 行 → 一个 PackedBatch）。
struct RowBuffer {
    input_ids: Vec<i32>,
    labels: Vec<i32>,
    position_ids: Vec<i32>,
    loss_weights: Vec<u8>,
    image_records: Vec<ImageRecordInputOwned>,
    video_records: Vec<VideoRecordInputOwned>,
    valid_len: usize,
    /// P3c：本行样本的 sid（一行 = 一样本；padding 空行 = `NO_SAMPLE_SID`）
    sample_id: u64,
}

/// P3c：padding 空行的 sentinel sid（无样本；reader 侧该行不产出对齐条目）
pub const NO_SAMPLE_SID: u64 = u64::MAX;

impl RowBuffer {
    fn new(max_seq_len: usize) -> Self {
        Self {
            input_ids: Vec::with_capacity(max_seq_len),
            labels: Vec::with_capacity(max_seq_len),
            position_ids: Vec::with_capacity(max_seq_len),
            loss_weights: Vec::with_capacity(max_seq_len * 2),
            image_records: Vec::new(),
            video_records: Vec::new(),
            valid_len: 0,
            sample_id: NO_SAMPLE_SID,
        }
    }
}

impl ModalityAwarePacker {
    pub fn new(
        max_seq_len: usize,
        pad_id: i32,
        batch_size: usize,
        extra_columns: bool,
        max_visual_tokens_per_sample: u32,
        visual_budget_policy: BudgetPolicy,
        image_label: i32,
    ) -> Self {
        Self {
            max_seq_len,
            pad_id,
            batch_size,
            extra_columns,
            max_visual_tokens_per_sample,
            visual_budget_policy,
            image_label,
            rows: Vec::with_capacity(batch_size),
            visual_truncated: AtomicU64::new(0),
            tokens_dropped_to_msl: AtomicU64::new(0),
            packed_text_tokens: AtomicU64::new(0),
            packed_image_tokens: AtomicU64::new(0),
            packed_images: AtomicU64::new(0),
            packed_video_tokens: AtomicU64::new(0),
            packed_video_clips: AtomicU64::new(0),
        }
    }

    /// 取走累计的截断统计（原子换出 0，供管线逐 chunk 汇入 BuildProgress）。
    /// 返回 `(visual_truncated, tokens_dropped_to_msl)`。
    pub fn take_stats(&self) -> (u64, u64) {
        (
            self.visual_truncated.swap(0, Ordering::Relaxed),
            self.tokens_dropped_to_msl.swap(0, Ordering::Relaxed),
        )
    }

    /// 取走实际入包的 per-modality 统计（manifest 模态拆分真源；post-drop）。
    /// 返回 `(text_tokens, image_tokens, images, video_tokens, video_clips)`——
    /// P3b 起 video 二元组为真值（此前恒 0）。pipeline 用它替代 encoder 的
    /// `es.stats` 累加——保证 manifest 守恒 `total_tokens == Σ 模态` 在有丢弃
    /// （原子性 / 预算 / 尾截断）时仍成立。
    pub fn take_packed_modality_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.packed_text_tokens.swap(0, Ordering::Relaxed),
            self.packed_image_tokens.swap(0, Ordering::Relaxed),
            self.packed_images.swap(0, Ordering::Relaxed),
            self.packed_video_tokens.swap(0, Ordering::Relaxed),
            self.packed_video_clips.swap(0, Ordering::Relaxed),
        )
    }

    /// 装箱一批 `EncodedSample` → 尽可能多的满载 `VlmPackedBatch`。
    ///
    /// 调用约定：任意多次 `pack` → 恰好一次 [`finish`](Self::finish)。
    /// 残余不满一个 batch 的行保留到 `finish`（全数据集至多一个未满 batch，F3 同构）。
    pub fn pack(&mut self, samples: Vec<EncodedSample>) -> Result<Vec<VlmPackedBatch>, String> {
        let mut out = Vec::new();
        for s in samples {
            // 契约自检（P2-T4 返回值校验链的一环；encoder 产出已由 pipeline.validate 过，
            // 此处防御性二次校验，错则硬失败——不让坏数据进 token 流）。
            s.validate()
                .map_err(|e| format!("EncodedSample invariant violated: {e}"))?;
            let row = self.build_row(s)?;
            self.rows.push(row);
            while self.rows.len() >= self.batch_size {
                let rows: Vec<RowBuffer> = self.rows.drain(..self.batch_size).collect();
                out.push(self.flush_batch(rows));
            }
        }
        Ok(out)
    }

    /// 输入流结束：把残余行 pad 成最后一个（未满）batch。无残余则不产出空 batch。
    pub fn finish(&mut self) -> Vec<VlmPackedBatch> {
        if self.rows.is_empty() {
            return Vec::new();
        }
        let rows: Vec<RowBuffer> = std::mem::take(&mut self.rows);
        vec![self.flush_batch(rows)]
    }

    /// 把一个样本压成一个行（经视觉预算 + 行宽 + 原子性约束）。
    fn build_row(&self, mut s: EncodedSample) -> Result<RowBuffer, String> {
        // 1. 视觉预算（D7/T2）：每样本 image 段 token 总数 ≤ max_visual_tokens_per_sample。
        self.enforce_visual_budget(&mut s)?;

        // 2. 行宽 + 原子性：贪心纳入段，image 段整段进或整段丢，text 段可尾部截断。
        let mut row = RowBuffer::new(self.max_seq_len);
        row.sample_id = s.sample_id.0; // P3c：行→sid 承载（col_row_sids）
        // 按 token_start 排序的段（validate 已保证连续，此处再保序）。
        let mut segs = s.segment_table.clone();
        segs.sort_by_key(|e| e.token_start);

        let mut pos = 0i32;
        'seg: for seg in &segs {
            if row.valid_len >= self.max_seq_len {
                break;
            }
            let seg_len = seg.token_len() as usize;
            let remain = self.max_seq_len - row.valid_len;

            match seg.kind {
                SegmentKind::Image | SegmentKind::Video => {
                    // 原子性：整段进或整段丢（D4）。
                    if seg_len > remain {
                        self.tokens_dropped_to_msl
                            .fetch_add(seg_len as u64, Ordering::Relaxed);
                        // 整段丢弃：不进 token 流，不计 position；继续后续段
                        // （后续 text 段可能仍能纳入_remain，符合"贪心填充"）。
                        continue 'seg;
                    }
                    // 整段纳入：image 段 label = image_label（T3），权重 0.0。
                    row.input_ids.extend_from_slice(
                        &s.token_seq[seg.token_start as usize..seg.token_end as usize],
                    );
                    row.labels
                        .extend(std::iter::repeat_n(self.image_label, seg_len));
                    row.position_ids
                        .extend((pos..pos + seg_len as i32).collect::<Vec<_>>());
                    row.loss_weights
                        .extend(std::iter::repeat_n(F16_ZERO_LE, seg_len).flatten());
                    // 取对应 media_ref（按 seg_order）产出 sidecar 记录。
                    // P3b：按 kind 分流——image → col_img_rec（32B），
                    // video → col_vid_rec（40B 含 start_ms/fps_milli）。
                    //
                    // P1-5（评审修复）：找不到 = 契约违规硬失败。旧实现 `if let Some`
                    // 静默跳过 rec 产出，却仍累计下方 packed_image/video 统计。
                    let rec = s
                        .media_refs
                        .iter()
                        .find(|r| r.seg_order == seg.segment_order)
                        .ok_or_else(|| {
                            format!(
                                "visual segment (sample {}, seg_order {}, kind {:?}) has no \
                                 matching media_ref in the packer (P1-5 contract: every image/video \
                                 segment must carry exactly one sidecar record)",
                                seg.sample_id, seg.segment_order, seg.kind
                            )
                        })?;
                    match seg.kind {
                        SegmentKind::Image => {
                            row.image_records.push(ImageRecordInputOwned {
                                bytes: rec.bytes.clone(),
                                grid_h: rec.grid_h,
                                grid_w: rec.grid_w,
                                flags: rec.flags,
                                sample_id: seg.sample_id.0,
                                seg_order: seg.segment_order,
                            });
                        }
                        SegmentKind::Video => {
                            // validate 已保证 kind==Video 时 rec.video 为 Some；
                            // 双保险：缺失即契约违规，报错不 panic。
                            let vm = rec.video.ok_or_else(|| {
                                format!(
                                    "video media_ref (sample {}, seg_order {}) carries no VideoMeta \
                                     (P3b: col_vid_rec requires n_frames/w/h/fps_milli/start_ms)",
                                    seg.sample_id, seg.segment_order
                                )
                            })?;
                            row.video_records.push(VideoRecordInputOwned {
                                bytes: rec.bytes.clone(),
                                n_frames: vm.n_frames,
                                width: vm.width,
                                height: vm.height,
                                flags: rec.flags,
                                sample_id: seg.sample_id.0,
                                seg_order: seg.segment_order,
                                fps_milli: vm.fps_milli,
                                start_ms: vm.start_ms,
                                duration_ms: vm.duration_ms,
                            });
                        }
                        _ => unreachable!("outer match already filtered Image|Video"),
                    }
                    // per-modality 入包计数（manifest 模态统计真源；仅整段纳入时计）。
                    // P3b：video 段不得错计为 image（§8.2 统计分流）。
                    match seg.kind {
                        SegmentKind::Image => {
                            self.packed_image_tokens
                                .fetch_add(seg_len as u64, Ordering::Relaxed);
                            self.packed_images.fetch_add(1, Ordering::Relaxed);
                        }
                        SegmentKind::Video => {
                            self.packed_video_tokens
                                .fetch_add(seg_len as u64, Ordering::Relaxed);
                            self.packed_video_clips.fetch_add(1, Ordering::Relaxed);
                        }
                        _ => unreachable!("outer match already filtered Image|Video"),
                    }
                    pos += seg_len as i32;
                    row.valid_len += seg_len;
                }
                SegmentKind::Audio => {
                    // P3a 复用 GreedyPacker 变长模型；VLM packer 不处理 audio 段。
                    // 出现即视为契约违规（encoder 不该把 audio 段喂给定宽行 packer）。
                    return Err(format!(
                        "audio segment reached VLM (fixed-width-row) packer (sample {}): \
                         audio must use the GreedyPacker variable-length path (P3a)",
                        s.sample_id
                    ));
                }
                SegmentKind::Text => {
                    // text 段可尾部截断（与 SFT tail-cut 同构）。
                    let take = std::cmp::min(seg_len, remain);
                    let ts = seg.token_start as usize;
                    row.input_ids.extend_from_slice(&s.token_seq[ts..ts + take]);
                    // text 段 label 用 encoder 提供的（按段 superv 策略；image 边界由 encoder 定）。
                    row.labels.extend_from_slice(&s.labels[ts..ts + take]);
                    row.position_ids
                        .extend((pos..pos + take as i32).collect::<Vec<_>>());
                    // loss_weights：encoder 提供则用其，否则 text 段默认 1.0。
                    match &s.loss_weights {
                        Some(lw) => row
                            .loss_weights
                            .extend_from_slice(&lw[ts * 2..(ts + take) * 2]),
                        None => row
                            .loss_weights
                            .extend(std::iter::repeat_n(F16_ONE_LE, take).flatten()),
                    }
                    pos += take as i32;
                    row.valid_len += take;
                    // per-modality 入包计数：text 段按实际纳入量（尾部截断部分不计）。
                    self.packed_text_tokens
                        .fetch_add(take as u64, Ordering::Relaxed);
                    if take < seg_len {
                        // text 段被尾部截断 → 行满，本样本剩余段全部丢弃。
                        self.tokens_dropped_to_msl
                            .fetch_add((seg_len - take) as u64, Ordering::Relaxed);
                        break 'seg;
                    }
                }
            }
        }
        Ok(row)
    }

    /// 视觉预算执行（D7/T2）：每样本 image/video 段 token 总数 ≤
    /// `max_visual_tokens_per_sample`（P3b §8.1 裁定 (a)：video 计入样本级总视觉
    /// 预算——与 D7"样本级总视觉预算"字面精神一致；video clip token 数远大于单图，
    /// 启用 video 的数据集须按 `n_frames × grid` 重估预算）。
    /// 超限按策略：截断多余视觉段（highest seg_order）+ 计数；或硬失败。
    fn enforce_visual_budget(&self, s: &mut EncodedSample) -> Result<(), String> {
        let mut total_visual: u32 = s
            .segment_table
            .iter()
            .filter(|e| e.kind == SegmentKind::Image || e.kind == SegmentKind::Video)
            .map(|e| e.token_len())
            .sum();
        if total_visual <= self.max_visual_tokens_per_sample {
            return Ok(());
        }
        match self.visual_budget_policy {
            BudgetPolicy::HardFail => Err(format!(
                "visual budget exceeded: sample {} has {total_visual} visual tokens but limit is {} \
                 (policy=HardFail; switch to TruncateAndCount or raise max_visual_tokens_per_sample)",
                s.sample_id, self.max_visual_tokens_per_sample
            )),
            BudgetPolicy::TruncateAndCount => {
                // 按 seg_order 降序逐个丢弃 image/video 段，直到 total ≤ budget。
                // 丢弃 = 从 token_seq/labels/loss_weights 中切除该段字节、从 segment_table
                // 与 media_refs 中移除、重排后续段 token 坐标。
                let mut drop_orders: Vec<u32> = s
                    .segment_table
                    .iter()
                    .filter(|e| e.kind == SegmentKind::Image || e.kind == SegmentKind::Video)
                    .map(|e| e.segment_order)
                    .collect();
                drop_orders.sort_by(|a, b| b.cmp(a)); // 降序（丢靠后的图）
                for order in drop_orders {
                    if total_visual <= self.max_visual_tokens_per_sample {
                        break;
                    }
                    // 找该 image/video 段并切除（seg_order 样本内唯一、不依赖 kind，
                    // P3b §8.1：video 切除路径与 image 同构——retain 对 video rec
                    // 同样正确，被切除的 video 段不产出 rec）
                    let pos = s.segment_table.iter().position(|e| {
                        e.segment_order == order
                            && (e.kind == SegmentKind::Image || e.kind == SegmentKind::Video)
                    });
                    let Some(pos) = pos else { continue };
                    let seg = s.segment_table[pos].clone();
                    let len = seg.token_len() as usize;
                    let ts = seg.token_start as usize;
                    let te = seg.token_end as usize;
                    // 切除 token_seq/labels/loss_weights 中 [ts..te) 切片
                    s.token_seq.drain(ts..te);
                    s.labels.drain(ts..te);
                    if let Some(lw) = s.loss_weights.as_mut() {
                        lw.drain(ts * 2..te * 2);
                    }
                    s.segment_table.remove(pos);
                    s.media_refs.retain(|r| r.seg_order != order);
                    total_visual = total_visual.saturating_sub(len as u32);
                    self.visual_truncated.fetch_add(1, Ordering::Relaxed);
                    // 重排后续段 token 坐标（切除后整体左移 len）
                    for e in s.segment_table.iter_mut() {
                        if e.token_start >= te as u32 {
                            e.token_start = e.token_start.saturating_sub(len as u32);
                            e.token_end = e.token_end.saturating_sub(len as u32);
                        }
                    }
                }
                // 切除后重新自检（坐标重排可能引入不一致，及时发现）
                s.validate().map_err(|e| {
                    format!(
                        "visual budget truncation broke segment invariant (sample {}): {e}",
                        s.sample_id
                    )
                })?;
                Ok(())
            }
        }
    }

    /// 把 batch_size 行攒成一个 PackedBatch + 配对 sidecar 记录。
    /// 不足 batch_size 行时（仅 finish 路径）补空行到 batch_size，保证 `bs × msl`
    /// 定宽恒定（与 SFT/PT 同纪律：全数据集至多一个未满批但形状恒定）。
    fn flush_batch(&self, mut rows: Vec<RowBuffer>) -> VlmPackedBatch {
        let capacity = self.batch_size * self.max_seq_len;
        let mut input_ids = Vec::with_capacity(capacity);
        let mut labels = Vec::with_capacity(capacity);
        let mut position_ids = Vec::with_capacity(capacity);
        let mut cu_seqlens: Vec<i32> = Vec::with_capacity(self.batch_size + 1);
        let mut row_lens: Vec<i32> = Vec::with_capacity(self.batch_size);
        let mut loss_weights = Vec::with_capacity(capacity * 2);
        let mut image_records: Vec<ImageRecordInputOwned> = Vec::new();
        let mut video_records: Vec<VideoRecordInputOwned> = Vec::new();
        let mut row_sample_ids: Vec<u64> = Vec::with_capacity(self.batch_size);

        cu_seqlens.push(0);
        let mut acc: i32 = 0;
        for _ in 0..self.batch_size {
            let r = if !rows.is_empty() {
                rows.remove(0)
            } else {
                RowBuffer::new(self.max_seq_len)
            };
            row_sample_ids.push(r.sample_id);
            // pad 到 max_seq_len
            let pad = self.max_seq_len - r.valid_len;
            r_input_ids_extend(&mut input_ids, &r.input_ids, self.pad_id, pad);
            r_labels_extend(&mut labels, &r.labels, pad);
            r_position_extend(&mut position_ids, &r.position_ids, pad);
            if self.extra_columns {
                loss_weights.extend_from_slice(&r.loss_weights);
                loss_weights.extend(std::iter::repeat_n(F16_ZERO_LE, pad).flatten());
            }
            acc += r.valid_len as i32;
            cu_seqlens.push(acc);
            row_lens.push(r.valid_len as i32);
            image_records.extend(r.image_records);
            video_records.extend(r.video_records);
        }
        VlmPackedBatch {
            batch: PackedBatch {
                input_ids,
                labels,
                position_ids,
                cu_seqlens,
                row_lens,
                doc_ids: Vec::new(), // VLM 行↔样本↔rec 用 .cu + rec.sample_id 表达，无需 per-token doc id
                loss_weights,
                batch_size: self.batch_size,
                max_seq_len: self.max_seq_len,
            },
            image_records,
            video_records,
            row_sample_ids,
        }
    }
}

/// 行 input_ids 追加：实际 token + pad_id 填充（pad 数 = pad）。
fn r_input_ids_extend(out: &mut Vec<i32>, src: &[i32], pad_id: i32, pad: usize) {
    out.extend_from_slice(src);
    out.extend(std::iter::repeat_n(pad_id, pad));
}
/// 行 labels 追加：实际 label + (-100) 填充。
fn r_labels_extend(out: &mut Vec<i32>, src: &[i32], pad: usize) {
    out.extend_from_slice(src);
    out.extend(std::iter::repeat_n(-100, pad));
}
/// 行 position_ids 追加：实际 position + 0 填充。
fn r_position_extend(out: &mut Vec<i32>, src: &[i32], pad: usize) {
    out.extend_from_slice(src);
    out.extend(std::iter::repeat_n(0, pad));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::encoded_sample::{
        EncodedSample, EncodedStats, IdRange, MediaRecord, SegmentEntry,
    };
    use crate::builder::sample::{SampleId, SegmentKind};

    fn sid(n: u64) -> SampleId {
        SampleId(n)
    }

    /// 构造"文本+图像+文本"交错样本（最小纵向闭环用例，P2-T1 golden case 核心）。
    /// text[0..5) + image[5..13) + text[13..20)，image grid 2×4=8 token。
    fn make_interleaved(sample_id: u64) -> EncodedSample {
        let text_len = 5u32;
        let img_tokens = 8u32;
        let total = text_len * 2 + img_tokens;
        EncodedSample {
            sample_id: sid(sample_id),
            token_seq: vec![7i32; total as usize],
            segment_table: vec![
                SegmentEntry {
                    sample_id: sid(sample_id),
                    segment_order: 0,
                    kind: SegmentKind::Text,
                    token_start: 0,
                    token_end: text_len,
                    media_record_id: None,
                    loss_weight: 255,
                },
                SegmentEntry {
                    sample_id: sid(sample_id),
                    segment_order: 1,
                    kind: SegmentKind::Image,
                    token_start: text_len,
                    token_end: text_len + img_tokens,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: sid(sample_id),
                    segment_order: 2,
                    kind: SegmentKind::Text,
                    token_start: text_len + img_tokens,
                    token_end: total,
                    media_record_id: None,
                    loss_weight: 255,
                },
            ],
            labels: vec![100; total as usize], // text 段 label（encoder 提供；image 段会被 packer 覆盖为 -100）
            loss_weights: None,
            stats: EncodedStats {
                text_tokens: 10,
                image_tokens: 8,
                images: 1,
                ..Default::default()
            },
            media_refs: vec![MediaRecord {
                sample_id: sid(sample_id),
                seg_order: 1,
                kind: SegmentKind::Image,
                bytes: Some(vec![1, 2, 3, 4]),
                grid_h: 2,
                grid_w: 4,
                flags: 0,
                video: None,
            }],
        }
    }

    /// T1 纵向闭环核心断言：单图单文本 → 单 batch → token range / image record /
    /// labels / loss_weights / blob bytes 五者对应关系逐项验证。
    #[test]
    fn test_t1_vertical_loop_single_image_single_text() {
        let mut packer =
            ModalityAwarePacker::new(32, 0, 1, true, 64, BudgetPolicy::TruncateAndCount, -100);
        let sample = make_interleaved(0);
        let media_bytes = sample.media_refs[0].bytes.clone().unwrap();
        // batch_size=1：1 个样本即满批，pack 阶段就吐出
        let out = packer.pack(vec![sample]).unwrap();
        assert_eq!(out.len(), 1);
        assert!(packer.finish().is_empty());
        let vb = &out[0];
        let b = &vb.batch;

        // 定宽性：bs × msl 恒定
        assert_eq!(b.input_ids.len(), 32);
        assert_eq!(b.labels.len(), 32);
        // cu_seqlens = 样本边界：[0, valid_len]
        assert_eq!(b.cu_seqlens, vec![0, 18]);
        // row_lens = 行有效长度
        assert_eq!(b.row_lens, vec![18]);
        // image 段 token range [5..13)：labels 全 -100（image_label）
        for i in 5..13 {
            assert_eq!(b.labels[i], -100, "image token at {i} must be -100");
        }
        // text 段 labels = encoder 提供的 (100)
        for i in 0..5 {
            assert_eq!(
                b.labels[i], 100,
                "text token at {i} must keep encoder label"
            );
        }
        // padding 位 -100
        for i in 18..32 {
            assert_eq!(b.labels[i], -100);
        }
        // loss_weights：image 段 0.0、text 段 1.0、padding 0.0
        let lw = &b.loss_weights;
        assert_eq!(lw.len(), 32 * 2);
        assert_eq!(&lw[5 * 2..13 * 2], &[0x00u8, 0x00].repeat(8)); // image=0.0
        assert_eq!(&lw[0..5 * 2], &[0x00u8, 0x3C].repeat(5)); // text=1.0
        // 可追溯链：一条 image record，sample_id/seg_order 与 segment 对应
        assert_eq!(vb.image_records.len(), 1);
        let rec = &vb.image_records[0];
        assert_eq!(rec.sample_id, 0);
        assert_eq!(rec.seg_order, 1);
        assert_eq!(rec.grid_h, 2);
        assert_eq!(rec.grid_w, 4);
        assert_eq!(rec.bytes.as_ref().unwrap(), &media_bytes);
        // position_ids：行内 0..valid 连续
        assert_eq!(b.position_ids[0], 0);
        assert_eq!(b.position_ids[17], 17);
        assert_eq!(b.position_ids[18], 0); // padding
    }

    /// 图像段原子性：图像段超过行剩余容量 → 整段丢弃（不进 token 流），后续 text 仍可纳入。
    #[test]
    fn test_image_segment_atomicity_drop_when_not_fit() {
        // msl=10：text[0..3) + image[3..11)（8 token，超 remain=7）→ image 整段丢，
        // 但这里 image 后无 text，故行有效长=3，image 计入 dropped。
        let mut packer =
            ModalityAwarePacker::new(10, 0, 1, false, 64, BudgetPolicy::TruncateAndCount, -100);
        let s = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 11],
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
                    token_end: 11,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
            ],
            labels: vec![100; 11],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Image,
                bytes: Some(vec![1]),
                grid_h: 2,
                grid_w: 4,
                flags: 0,
                video: None,
            }],
        };
        let out = packer.pack(vec![s]).unwrap();
        let out2 = packer.finish();
        assert!(
            out2.is_empty(),
            "batch_size=1 → full batch emitted in pack, finish empty"
        );
        let vb = &out[0];
        assert_eq!(
            vb.batch.cu_seqlens,
            vec![0, 3],
            "only text survives, image dropped atomically"
        );
        assert!(
            vb.image_records.is_empty(),
            "dropped image must not emit sidecar record"
        );
        // dropped 统计：image 整段 8 token（take_stats 在原实例上累计）
        let (vt, td) = packer.take_stats();
        assert_eq!(td, 8, "8 image tokens dropped to msl (atomic drop)");
        let _ = vt;
    }

    /// 视觉预算 TruncateAndCount：多图超限 → 丢弃靠后的图 + 计数 + 坐标重排。
    #[test]
    fn test_visual_budget_truncate_and_count() {
        // 两张图各 8 token = 16 visual，budget=8 → 丢弃第二张（seg_order=2）
        let mut packer =
            ModalityAwarePacker::new(64, 0, 1, false, 8, BudgetPolicy::TruncateAndCount, -100);
        // text[0..2) + image[2..10) + image[10..18) + text[18..20)
        let s = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 20],
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
                    segment_order: 1,
                    kind: SegmentKind::Image,
                    token_start: 2,
                    token_end: 10,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: sid(0),
                    segment_order: 2,
                    kind: SegmentKind::Image,
                    token_start: 10,
                    token_end: 18,
                    media_record_id: Some(1),
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: sid(0),
                    segment_order: 3,
                    kind: SegmentKind::Text,
                    token_start: 18,
                    token_end: 20,
                    media_record_id: None,
                    loss_weight: 255,
                },
            ],
            labels: vec![100; 20],
            loss_weights: None,
            stats: EncodedStats {
                images: 2,
                image_tokens: 16,
                text_tokens: 4,
                ..Default::default()
            },
            media_refs: vec![
                MediaRecord {
                    sample_id: sid(0),
                    seg_order: 1,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![1]),
                    grid_h: 2,
                    grid_w: 4,
                    flags: 0,
                    video: None,
                },
                MediaRecord {
                    sample_id: sid(0),
                    seg_order: 2,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![2]),
                    grid_h: 2,
                    grid_w: 4,
                    flags: 0,
                    video: None,
                },
            ],
        };
        let out = packer.pack(vec![s]).unwrap();
        assert!(
            packer.finish().is_empty(),
            "batch_size=1 → full batch in pack"
        );
        let vb = &out[0];
        // 第二张图被丢弃：只剩 1 条 sidecar 记录（seg_order=1）
        assert_eq!(vb.image_records.len(), 1);
        assert_eq!(vb.image_records[0].seg_order, 1);
        // token_seq 切除第二张图 8 token → 12 token；valid_len=12
        assert_eq!(vb.batch.cu_seqlens, vec![0, 12]);
        // 截断统计
        let (vt, _td) = packer.take_stats();
        assert_eq!(vt, 1, "one image truncated by budget");
    }

    /// per-modality 入包统计（manifest 模态拆分真源）：必须取 **post-drop** 值，
    /// 否则原子性丢弃 / 视觉预算截断 / text 尾截断会让 Σ 模态 != total_tokens(packed)
    /// 而破坏 manifest 守恒（pipeline 用本方法替代 encoder 的 es.stats 累加）。
    #[test]
    fn test_packed_modality_stats_reflect_drops() {
        // ---- 原子性丢弃：image 整段不进 → packed_image=0、packed_text=入包 text ----
        // msl=10：text[0..3) + image[3..11)（8 token 超 remain=7）→ image 整段丢。
        let mut packer =
            ModalityAwarePacker::new(10, 0, 1, false, 64, BudgetPolicy::TruncateAndCount, -100);
        let s = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 11],
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
                    token_end: 11,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
            ],
            labels: vec![100; 11],
            loss_weights: None,
            stats: EncodedStats {
                text_tokens: 3,
                image_tokens: 8,
                images: 1,
                ..Default::default()
            },
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Image,
                bytes: Some(vec![1]),
                grid_h: 2,
                grid_w: 4,
                flags: 0,
                video: None,
            }],
        };
        let _ = packer.pack(vec![s]).unwrap();
        let (pt, pi, pimgs, pvt, pvc) = packer.take_packed_modality_stats();
        assert_eq!(pt, 3, "packed text = 3 (surviving text tokens)");
        assert_eq!(
            pi, 0,
            "packed image = 0 (atomically dropped image not counted)"
        );
        assert_eq!(
            pimgs, 0,
            "packed images = 0 (no sidecar record for dropped image)"
        );
        assert_eq!((pvt, pvc), (0, 0), "P3b: 无 video 段时 video 统计恒 0");

        // ---- 视觉预算截断：第二张图被截 → packed 只计留存图 ----
        // text[0..2) + image[2..10) + image[10..18) + text[18..20)，budget=8 → 丢第二张图
        let mut packer =
            ModalityAwarePacker::new(64, 0, 1, false, 8, BudgetPolicy::TruncateAndCount, -100);
        let s = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 20],
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
                    segment_order: 1,
                    kind: SegmentKind::Image,
                    token_start: 2,
                    token_end: 10,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: sid(0),
                    segment_order: 2,
                    kind: SegmentKind::Image,
                    token_start: 10,
                    token_end: 18,
                    media_record_id: Some(1),
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: sid(0),
                    segment_order: 3,
                    kind: SegmentKind::Text,
                    token_start: 18,
                    token_end: 20,
                    media_record_id: None,
                    loss_weight: 255,
                },
            ],
            labels: vec![100; 20],
            loss_weights: None,
            stats: EncodedStats {
                images: 2,
                image_tokens: 16,
                text_tokens: 4,
                ..Default::default()
            },
            media_refs: vec![
                MediaRecord {
                    sample_id: sid(0),
                    seg_order: 1,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![1]),
                    grid_h: 2,
                    grid_w: 4,
                    flags: 0,
                    video: None,
                },
                MediaRecord {
                    sample_id: sid(0),
                    seg_order: 2,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![2]),
                    grid_h: 2,
                    grid_w: 4,
                    flags: 0,
                    video: None,
                },
            ],
        };
        let _ = packer.pack(vec![s]).unwrap();
        let (pt, pi, pimgs, pvt, pvc) = packer.take_packed_modality_stats();
        // 留存：text 2+2=4、image 8（第一张 seg_order=1）、1 张图
        assert_eq!(pt, 4, "packed text = 4 (both text segments survive)");
        assert_eq!(pi, 8, "packed image = 8 (only first image survives budget)");
        assert_eq!(
            pimgs, 1,
            "packed images = 1 (second image truncated by budget)"
        );
        assert_eq!(
            (pvt, pvc),
            (0, 0),
            "P3b: 纯 image 样本 video 统计恒 0（不串统计）"
        );
        // 对照：encoder es.stats 仍是 pre-drop（image_tokens=16/images=2）——若 pipeline
        // 误用 es.stats，Σ 模态=20 != total_tokens(packed)=12 守恒破坏。本测试锁住
        // "取 post-drop"语义。
    }

    /// 视觉预算 HardFail：超限 → 构建失败。
    #[test]
    fn test_visual_budget_hard_fail() {
        let mut packer = ModalityAwarePacker::new(64, 0, 1, false, 4, BudgetPolicy::HardFail, -100);
        let s = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 10],
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
                    segment_order: 1,
                    kind: SegmentKind::Image,
                    token_start: 2,
                    token_end: 10,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
            ],
            labels: vec![100; 10],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Image,
                bytes: Some(vec![1]),
                grid_h: 2,
                grid_w: 4,
                flags: 0,
                video: None,
            }],
        };
        let err = packer.pack(vec![s]).unwrap_err();
        assert!(err.contains("visual budget exceeded"), "{err}");
        assert!(err.contains("HardFail"), "{err}");
    }

    /// 多 batch 累积：攒满 batch_size 行才出批，残余留 finish。
    #[test]
    fn test_batch_accumulation_and_residual() {
        let mut packer =
            ModalityAwarePacker::new(32, 0, 2, false, 64, BudgetPolicy::TruncateAndCount, -100);
        // 3 个样本 → 1 个满批（2行）+ 1 行残余
        let samples = vec![
            make_interleaved(0),
            make_interleaved(1),
            make_interleaved(2),
        ];
        let out = packer.pack(samples).unwrap();
        assert_eq!(out.len(), 1, "one full batch of 2 rows");
        assert_eq!(out[0].batch.cu_seqlens.len(), 3); // [0, 18, 36]
        let fin = packer.finish();
        assert_eq!(
            fin.len(),
            1,
            "residual 1 row flushed (padded to batch_size)"
        );
        // 残余批补齐空行：cu_seqlens [0, 18, 18]（空行 valid_len=0，定宽 bs*msl 恒定）
        assert_eq!(fin[0].batch.cu_seqlens, vec![0, 18, 18]);
        assert_eq!(
            fin[0].batch.input_ids.len(),
            2 * 32,
            "residual batch still bs*msl"
        );
    }

    /// 定宽性恒定：bs × msl（验收锚点 3）。
    #[test]
    fn test_fixed_width_invariant() {
        let mut packer =
            ModalityAwarePacker::new(16, 0, 4, false, 64, BudgetPolicy::TruncateAndCount, -100);
        let samples: Vec<EncodedSample> = (0..5).map(make_interleaved).collect();
        let mut out = packer.pack(samples).unwrap();
        out.extend(packer.finish());
        for vb in &out {
            assert_eq!(vb.batch.input_ids.len(), 4 * 16, "bs*msl constant");
            // cu_seqlens 末值 ≤ bs*msl
            let last = *vb.batch.cu_seqlens.last().unwrap() as usize;
            assert!(last <= 4 * 16);
            // row_lens 恒为 batch_size 长
            assert_eq!(vb.batch.row_lens.len(), 4);
        }
    }

    /// id 区间校验集成（D6）：合法区间通过。
    #[test]
    fn test_id_ranges_validate() {
        let ranges = [IdRange {
            start: 32000,
            end: 32100,
            kind: SegmentKind::Image,
        }];
        assert!(
            crate::builder::encoded_sample::validate_modality_id_ranges(&ranges, Some(32000))
                .is_ok()
        );
    }

    /// P3b（§8.2）：Image|Video 同分支内的 **records 分流 + 统计分流**——
    /// video 段必须产 col_vid_rec 记录并计 video 二元组，不得错计为 image。
    #[test]
    fn test_video_records_and_stats_split() {
        use crate::builder::encoded_sample::VideoMeta;
        let mut packer =
            ModalityAwarePacker::new(128, 0, 1, false, 128, BudgetPolicy::TruncateAndCount, -100);
        // text[0..2) + image[2..6) + video[6..14)：video 段 8 token
        let s = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 14],
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
                    segment_order: 1,
                    kind: SegmentKind::Image,
                    token_start: 2,
                    token_end: 6,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
                SegmentEntry {
                    sample_id: sid(0),
                    segment_order: 2,
                    kind: SegmentKind::Video,
                    token_start: 6,
                    token_end: 14,
                    media_record_id: Some(1),
                    loss_weight: 0,
                },
            ],
            labels: vec![100; 14],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![
                MediaRecord {
                    sample_id: sid(0),
                    seg_order: 1,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![1]),
                    grid_h: 2,
                    grid_w: 2,
                    flags: 0,
                    video: None,
                },
                MediaRecord {
                    sample_id: sid(0),
                    seg_order: 2,
                    kind: SegmentKind::Video,
                    bytes: Some(vec![2, 2]),
                    grid_h: 0,
                    grid_w: 0,
                    flags: 0,
                    video: Some(VideoMeta {
                        n_frames: 16,
                        width: 336,
                        height: 336,
                        fps_milli: 30000,
                        start_ms: 250,
                        duration_ms: 20000,
                    }),
                },
            ],
        };
        let batches = packer.pack(vec![s]).unwrap();
        let vb = &batches[0];
        // records 分流：image 1 条、video 1 条（VideoMeta 六字段透传）
        assert_eq!(vb.image_records.len(), 1);
        assert_eq!(vb.image_records[0].seg_order, 1);
        assert_eq!(vb.video_records.len(), 1);
        let vr = &vb.video_records[0];
        assert_eq!(vr.seg_order, 2);
        assert_eq!(vr.n_frames, 16);
        assert_eq!(vr.width, 336);
        assert_eq!(vr.fps_milli, 30000);
        assert_eq!(
            vr.start_ms, 250,
            "start_ms = round(timestamp×1000) 由 encoder 计算"
        );
        assert_eq!(
            vr.duration_ms, 20000,
            "duration_ms 仅统计口径用（不落 rec）"
        );
        // 统计分流：video 不计入 image 二元组
        let (pt, pi, pimgs, pvt, pvc) = packer.take_packed_modality_stats();
        assert_eq!(pt, 2);
        assert_eq!(pi, 4);
        assert_eq!(pimgs, 1);
        assert_eq!(pvt, 8, "video token 数");
        assert_eq!(pvc, 1, "video clip 数");
    }

    /// P3b（§8.1）：video 段计入 `max_visual_tokens_per_sample`——超预算时
    /// TruncateAndCount 切除 video 段（highest seg_order）、无 rec 产出；
    /// HardFail 策略中止。
    #[test]
    fn test_visual_budget_covers_video_segments() {
        use crate::builder::encoded_sample::VideoMeta;
        let vm = || VideoMeta {
            n_frames: 16,
            width: 336,
            height: 336,
            fps_milli: 30000,
            start_ms: 0,
            duration_ms: 1000,
        };
        let mk = |video_len: u32| EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; (2 + video_len) as usize],
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
                    segment_order: 1,
                    kind: SegmentKind::Video,
                    token_start: 2,
                    token_end: 2 + video_len,
                    media_record_id: Some(0),
                    loss_weight: 0,
                },
            ],
            labels: vec![100; (2 + video_len) as usize],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Video,
                bytes: Some(vec![3, 3]),
                grid_h: 0,
                grid_w: 0,
                flags: 0,
                video: Some(vm()),
            }],
        };
        // budget=8：video 段 16 token 超限 → 整段切除（仅剩 text）
        let mut packer =
            ModalityAwarePacker::new(128, 0, 1, false, 8, BudgetPolicy::TruncateAndCount, -100);
        let batches = packer.pack(vec![mk(16)]).unwrap();
        assert!(
            batches[0].video_records.is_empty(),
            "被切除的 video 段不产出 rec"
        );
        assert!(batches[0].image_records.is_empty());
        let (pt, pi, pimgs, pvt, pvc) = packer.take_packed_modality_stats();
        assert_eq!(pt, 2, "只剩 text 2 token");
        assert_eq!((pi, pimgs, pvt, pvc), (0, 0, 0, 0));
        // HardFail：超限 → 构建失败
        let mut packer2 =
            ModalityAwarePacker::new(128, 0, 1, false, 8, BudgetPolicy::HardFail, -100);
        let e = packer2.pack(vec![mk(16)]).unwrap_err();
        assert!(e.contains("visual budget exceeded"), "{e}");
        // video 段不超预算（8 ≤ 8）→ 正常入包
        let mut packer3 =
            ModalityAwarePacker::new(128, 0, 1, false, 8, BudgetPolicy::TruncateAndCount, -100);
        let batches = packer3.pack(vec![mk(8)]).unwrap();
        assert_eq!(batches[0].video_records.len(), 1);
    }
}
