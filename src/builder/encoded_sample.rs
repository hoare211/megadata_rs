//! 多模态编码器契约（最终方案 §3.2，P2-T4/T5）。
//!
//! 冻结 `EncodedSample` + `segment_table` + **可追溯链**：
//! `sample_id -> segment_order -> token_range -> media_record_id`。
//! 该链保证 token 段与 sidecar blob 记录一一对应，**不依赖数组下标**
//! （多图/packing/batch 组合后仍可证明不错位）。
//!
//! - `IdRange` + `validate_modality_id_ranges`：D6 专属 token id 区间管理
//!   （visual placeholder id 与文本词表不相交，冲突硬失败）。
//! - 本模块只定义契约与不变式自检；encoder 实现见 `pipeline::PyModalityEncoder`，
//!   packer 消费见 `packing::modality_aware::ModalityAwarePacker`。
//!
//! **纪律**：本模块是冻结契约，与 `20261016-多模态sidecar-格式冻结.md`
//! 及方案 §3.2 任何出入都按"实现错误"处理；字段增删须先改方案/格式文档。

use crate::builder::sample::{SampleId, SegmentKind};

/// 专属 token id 区间（D6：visual placeholder id 与文本词表不相交）。
///
/// 各模态使用互不相交的 placeholder id 区间（仿 Qwen-VL `<|image_pad|>`）。
/// encoder 声明本区间 → pipeline 校验与文本 vocab 不相交（冲突硬失败，见
/// [`validate_modality_id_ranges`]）→ 汇总进 manifest `modality_id_ranges`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdRange {
    /// 区间起始（含）。
    pub start: i32,
    /// 区间结束（不含）。
    pub end: i32,
    /// 该区间所属模态（用于 manifest 透出与冲突归类）。
    pub kind: SegmentKind,
}

impl IdRange {
    /// 该 id 是否落入本区间。
    pub fn contains(&self, id: i32) -> bool {
        id >= self.start && id < self.end
    }
    /// 区间长度（token 数当量）。
    #[allow(dead_code)] // 公开契约方法（manifest/诊断方可用）
    pub fn len(&self) -> i32 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

/// 校验 modality id 区间集合（D6 硬失败语义）：
/// - 各区间非空（`start < end`）；
/// - 区间两两不相交（按 start 排序后相邻不重叠）；
/// - 与文本词表 `[0, text_vocab_size)` 不相交（visual placeholder 必须在文本 vocab 之外）。
///
/// 任一冲突返回 `Err`（由 pipeline 转为构建失败）。空区间集合法（纯文本构建无 modality
/// encoder，不调用本函数）。
pub fn validate_modality_id_ranges(
    ranges: &[IdRange],
    text_vocab_size: Option<u32>,
) -> Result<(), String> {
    let mut sorted: Vec<IdRange> = ranges.to_vec();
    sorted.sort_by_key(|r| r.start);

    // 1. 非空
    for (i, r) in sorted.iter().enumerate() {
        if r.is_empty() {
            return Err(format!(
                "modality id range #{i} is empty: start={} end={} kind={:?}",
                r.start, r.end, r.kind
            ));
        }
    }

    // 2. 两两不相交（排序后只需查相邻）
    for w in sorted.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a.end > b.start {
            return Err(format!(
                "modality id ranges overlap: [{},{}) ({:?}) and [{},{}) ({:?}) — \
                 visual placeholder 区间必须互不相交",
                a.start, a.end, a.kind, b.start, b.end, b.kind
            ));
        }
    }

    // 3. 与文本 vocab 不相交
    if let Some(vsz) = text_vocab_size {
        for r in &sorted {
            if r.start < vsz as i32 {
                return Err(format!(
                    "modality id range [{},{}) ({:?}) conflicts with text vocab [0,{vsz}) \
                     (D6 硬失败)：visual placeholder id 必须在文本词表之外",
                    r.start, r.end, r.kind
                ));
            }
        }
    }
    Ok(())
}

/// 一段在主 token 序列中的范围 + 它指向的 sidecar 记录（可追溯链的中间节点）。
///
/// `token_start/token_end` 是**样本内坐标**（在 `EncodedSample.token_seq` 内）；
/// packing 拼接后由 [`crate::packing::modality_aware::ModalityAwarePacker`]
/// 平移为 batch 内坐标并产出旁路 `segment_table`（见 D1：模态段边界走旁路）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentEntry {
    pub sample_id: SampleId,
    pub segment_order: u32,
    pub kind: SegmentKind,
    /// 在样本 token_seq 内的起始（含）。
    pub token_start: u32,
    /// 在样本 token_seq 内的结束（不含）。
    pub token_end: u32,
    /// 指向 sidecar record 的序号（写入端由 writer 按 batch 顺序分配；
    /// 文本段为 `None`；图像/视频段在 packer 侧必为 `Some`）。
    pub media_record_id: Option<u64>,
    /// 该段的默认 loss 权重（逻辑值 0..=255；255 = 1.0）。
    pub loss_weight: u8,
}

impl SegmentEntry {
    pub fn token_len(&self) -> u32 {
        self.token_end - self.token_start
    }
}

/// 编码统计量（入 `GlobalStats`，按模态分别计数；方案 §3.2/D3）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EncodedStats {
    pub text_tokens: u64,
    pub image_tokens: u64,
    pub audio_tokens: u64,
    pub video_tokens: u64,
    pub images: u64,
    pub audio_clips: u64,
    pub video_clips: u64,
}

impl EncodedStats {
    /// 累加到本 stats（packer 逐样本累加）。
    pub fn add(&mut self, other: &EncodedStats) {
        self.text_tokens += other.text_tokens;
        self.image_tokens += other.image_tokens;
        self.audio_tokens += other.audio_tokens;
        self.video_tokens += other.video_tokens;
        self.images += other.images;
        self.audio_clips += other.audio_clips;
        self.video_clips += other.video_clips;
    }
}

/// video 段专属元数据（P3b；`kind == Video` 的 `MediaRecord` 必填 `Some`）。
///
/// 冻结契约修订声明（P3b 方案 §4.2）：本结构为 P3b 范围内的契约扩展，
/// 随 P3b 实现方案评审通过生效；sidecar 格式冻结（`col_vid_rec` 布局）
/// 不受影响（v1.2 已冻结，P3b 不改字节布局）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoMeta {
    /// 采样帧数（= 构建配置 video_n_frames；非源总帧数，见 P3b §4.4）。
    pub n_frames: u16,
    /// resize 后目标分辨率宽（以指纹 resize 参数为准，rec 透传；见 P3b §6.2）。
    pub width: u16,
    /// resize 后目标分辨率高。
    pub height: u16,
    /// fps × 1000 取整（显式公式：`(fps_f32 × 1000.0).round() as u32`）。
    pub fps_milli: u32,
    /// 段起点（相对样本起点，整数毫秒；终稿 DP4(a′)）。
    pub start_ms: u32,
    /// 源视频时长（毫秒；统计口径用，见 P3b §6.4；缺失 = 0）。
    pub duration_ms: u32,
}

/// 待写入 sidecar 的媒体记录（encoder 产出，writer 落盘）。
///
/// `bytes` 为压缩字节（VALID 时）；占位记录（NO_IMAGE/EMPTY/ERRORED）为 `None`。
/// `grid_h/grid_w` 由 encoder 冻结（D7 定宽预算：`tokens = grid_h * grid_w`）。
/// P3b：`kind == Video` 时 `video` 必为 `Some`（`col_vid_rec` 六字段承载），
/// 其余 kind 必为 `None`（`validate` 双向校验）。
#[derive(Debug, Clone)]
pub struct MediaRecord {
    pub sample_id: SampleId,
    pub seg_order: u32,
    pub kind: SegmentKind,
    pub bytes: Option<Vec<u8>>,
    pub grid_h: u16,
    pub grid_w: u16,
    pub flags: u8,
    /// P3b：video 段专属元数据（kind==Video 必填 Some；其余 kind 必为 None）。
    pub video: Option<VideoMeta>,
}

/// encoder 回调返回的单样本编码结果（冻结契约，方案 §3.2）。
///
/// 不变式（[`Self::validate`]）：
/// - `segment_table` 各段按 `token_start` 排序后**连续覆盖**整个 `token_seq`（无重叠、无空洞）；
/// - `Σ segment.token_len == token_seq.len()`；
/// - `labels.len() == token_seq.len()`；
/// - `loss_weights.len() == 2 * token_seq.len()`（若提供）。
#[derive(Debug, Clone)]
pub struct EncodedSample {
    pub sample_id: SampleId,
    /// 主 token 序列（visual placeholder / text token 混合流，进 input_ids）。
    pub token_seq: Vec<i32>,
    /// 显式段表：token range -> media record 的可追溯映射。
    pub segment_table: Vec<SegmentEntry>,
    /// 默认 labels（文本段按监督策略，图像段默认 -100，可由 encoder 覆盖）。
    pub labels: Vec<i32>,
    /// 逐 token 权重（f16 LE 字节流，长度 = 2 * token_seq.len()）。
    /// `None` = 由 packer 按段默认填充（文本=1.0、图像=-100 段=0.0）。
    pub loss_weights: Option<Vec<u8>>,
    pub stats: EncodedStats,
    /// 待写入 sidecar 的媒体记录（与 segment_table 的媒体段一一对应，按 seg_order 排序）。
    pub media_refs: Vec<MediaRecord>,
}

impl EncodedSample {
    /// 段表守恒不变式自检（方案 §3.2：Σ segment.token_len == token_seq.len()）。
    ///
    /// 由 pipeline 在 encoder 返回后、packer 消费前调用（P2-T4 返回值校验链的一环）。
    pub fn validate(&self) -> Result<(), String> {
        let n = self.token_seq.len();
        let mut segs = self.segment_table.clone();
        segs.sort_by_key(|s| s.token_start);
        let mut cursor = 0u32;
        for (i, seg) in segs.iter().enumerate() {
            if seg.token_start != cursor {
                return Err(format!(
                    "segment_table not contiguous at index {i}: expected start {cursor}, \
                     got {} (sample {})",
                    seg.token_start, self.sample_id
                ));
            }
            if seg.token_end <= seg.token_start {
                return Err(format!(
                    "segment {i} has non-positive length (start={} end={}) (sample {})",
                    seg.token_start, seg.token_end, self.sample_id
                ));
            }
            cursor = seg.token_end;
        }
        if cursor as usize != n {
            return Err(format!(
                "segment_table covers {cursor} tokens but token_seq has {n} (sample {})",
                self.sample_id
            ));
        }
        if self.labels.len() != n {
            return Err(format!(
                "labels length {} != token_seq length {n} (sample {})",
                self.labels.len(),
                self.sample_id
            ));
        }
        if let Some(lw) = &self.loss_weights
            && lw.len() != n * 2
        {
            return Err(format!(
                "loss_weights length {} != 2*token_seq length {} (sample {})",
                lw.len(),
                n * 2,
                self.sample_id
            ));
        }
        // P1-5（评审修复）：Image/Video 段 ↔ media_ref **严格双向映射**。
        //
        // 旧实现只校验了 VideoMeta 有无；packer 用 `if let Some(rec) = find(...)`
        // 找不到就静默跳过 rec 产出，却仍无条件累计 packed_image_tokens /
        // packed_video_clips——构建成功、manifest 显示视觉统计，实际 sidecar
        // 没有对应字节（sample_id → segment → token_range → media_record 契约断裂）。
        // 现在构造期即硬失败：每个视觉段必须恰有一条同 sample_id / seg_order / kind
        // 的 media_ref；每条视觉 media_ref 必须恰对应一个视觉段；seg_order 不得重复。
        {
            use std::collections::HashMap;
            let mut seg_orders: HashMap<u32, SegmentKind> = HashMap::new();
            for (i, seg) in self.segment_table.iter().enumerate() {
                if !matches!(seg.kind, SegmentKind::Image | SegmentKind::Video) {
                    continue;
                }
                if seg.sample_id != self.sample_id {
                    return Err(format!(
                        "segment {i} (seg_order {}) belongs to sample {} but EncodedSample is \
                         sample {} (sample {})",
                        seg.segment_order, seg.sample_id.0, self.sample_id.0, self.sample_id
                    ));
                }
                if let Some(prev) = seg_orders.insert(seg.segment_order, seg.kind) {
                    return Err(format!(
                        "duplicate seg_order {} among visual segments ({prev:?} and {:?}) \
                         (sample {})",
                        seg.segment_order, seg.kind, self.sample_id
                    ));
                }
            }
            let mut refs_by_order: HashMap<u32, &MediaRecord> = HashMap::new();
            for (i, r) in self.media_refs.iter().enumerate() {
                if !matches!(r.kind, SegmentKind::Image | SegmentKind::Video) {
                    continue;
                }
                if r.sample_id != self.sample_id {
                    return Err(format!(
                        "media_ref #{i} (seg_order {}) belongs to sample {} but EncodedSample is \
                         sample {} (sample {})",
                        r.seg_order, r.sample_id.0, self.sample_id.0, self.sample_id
                    ));
                }
                if refs_by_order.insert(r.seg_order, r).is_some() {
                    return Err(format!(
                        "duplicate seg_order {} among image/video media_refs (sample {})",
                        r.seg_order, self.sample_id
                    ));
                }
            }
            for (order, kind) in &seg_orders {
                match refs_by_order.get(order) {
                    None => {
                        return Err(format!(
                            "visual segment seg_order {order} ({kind:?}) has no matching \
                             media_ref (sample {}) — sidecar record would be missing while \
                             modality stats still count it",
                            self.sample_id
                        ));
                    }
                    Some(r) if r.kind != *kind => {
                        return Err(format!(
                            "media_ref seg_order {order} kind {:?} does not match segment kind \
                             {kind:?} (sample {})",
                            r.kind, self.sample_id
                        ));
                    }
                    Some(_) => {}
                }
            }
            for (order, r) in &refs_by_order {
                if !seg_orders.contains_key(order) {
                    return Err(format!(
                        "image/video media_ref seg_order {order} ({:?}) has no matching visual \
                         segment (sample {}) — orphan sidecar record",
                        r.kind, self.sample_id
                    ));
                }
            }
        }

        // P3b（方案 §4.2）：MediaRecord 与 kind 的双向绑定——kind==Video 必须
        // 携带 VideoMeta（col_vid_rec 六字段承载），其余 kind 不得携带；
        // n_frames == 0 非法（采样帧数至少 1）。
        for (i, r) in self.media_refs.iter().enumerate() {
            match (r.kind, &r.video) {
                (SegmentKind::Video, None) => {
                    return Err(format!(
                        "media_ref #{i}: kind=Video but video metadata is missing (sample {}) — \
                         P3b requires n_frames/width/height/fps_milli/start_ms/duration_ms",
                        self.sample_id
                    ));
                }
                (SegmentKind::Video, Some(v)) => {
                    if v.n_frames == 0 {
                        return Err(format!(
                            "media_ref #{i}: VideoMeta.n_frames must be >= 1 (sample {})",
                            self.sample_id
                        ));
                    }
                }
                (_, Some(_)) => {
                    return Err(format!(
                        "media_ref #{i}: video metadata present on kind={:?} record (sample {}) — \
                         only kind=Video may carry VideoMeta",
                        r.kind, self.sample_id
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// P3a：剔除 audio 段，构造 Vlm 流侧过滤样本（demux 在 `EncodedSample` 层，
    /// P3a §3.1——tokenize 后按 `segment_table.kind` 分流，不触碰 encoder 逻辑）。
    ///
    /// 返回：
    /// - `None` = **纯 audio 样本**（无剩余段）：不进 `vlm_packer`（空样本跳过），
    ///   audio 段照常走 audio 流（GreedyPacker 变长装箱）；
    /// - `Some(es)` = 剩余段样本：token 子序列拼接 + **全部剩余段 token 坐标重算**
    ///   + **重过一遍构造校验**（连续覆盖不变式，P2-5 显式验收项）。
    ///
    /// 无 audio 段的样本零动作（原样返回，Vlm 流行为逐字节不变）。
    /// `media_record_id` 链原值保留（可追溯性不破坏）；audio 段的 media_ref
    /// （若有）随之剔除——Vlm packer 只按 Image/Video 段消费 rec。
    pub fn filter_audio_segments(self) -> Result<Option<EncodedSample>, String> {
        let has_audio = self
            .segment_table
            .iter()
            .any(|e| e.kind == SegmentKind::Audio);
        if !has_audio {
            return Ok(Some(self));
        }
        // 按 token_start 排序（validate 已保证连续覆盖；此处保序拼接）。
        let mut segs: Vec<&SegmentEntry> = self.segment_table.iter().collect();
        segs.sort_by_key(|e| e.token_start);

        let mut token_seq = Vec::with_capacity(self.token_seq.len());
        let mut labels = Vec::with_capacity(self.labels.len());
        let mut loss_weights: Option<Vec<u8>> = self
            .loss_weights
            .as_ref()
            .map(|lw| Vec::with_capacity(lw.len()));
        let mut segment_table = Vec::with_capacity(segs.len());
        let mut offset = 0u32;
        let mut any_remaining = false;

        for seg in segs {
            if seg.kind == SegmentKind::Audio {
                continue; // audio 段：不进 Vlm 流 token 序列
            }
            any_remaining = true;
            let ts = seg.token_start as usize;
            let te = seg.token_end as usize;
            let len = te - ts;
            token_seq.extend_from_slice(&self.token_seq[ts..te]);
            labels.extend_from_slice(&self.labels[ts..te]);
            if let (Some(lw), Some(out)) = (self.loss_weights.as_ref(), loss_weights.as_mut()) {
                out.extend_from_slice(&lw[ts * 2..te * 2]);
            }
            segment_table.push(SegmentEntry {
                sample_id: seg.sample_id,
                segment_order: seg.segment_order,
                kind: seg.kind,
                token_start: offset,
                token_end: offset + len as u32,
                // 可追溯链原值保留：media_record_id 与 rec 的对应关系不受剔除影响
                media_record_id: seg.media_record_id,
                loss_weight: seg.loss_weight,
            });
            offset += len as u32;
        }
        if !any_remaining {
            return Ok(None); // 纯 audio 样本
        }
        let mut media_refs = self.media_refs;
        media_refs.retain(|r| r.kind != SegmentKind::Audio);
        let filtered = EncodedSample {
            sample_id: self.sample_id,
            token_seq,
            segment_table,
            labels,
            loss_weights,
            stats: self.stats,
            media_refs,
        };
        // 重过构造校验（坐标重排可能引入不一致，及时发现——P3a §3.1 第 3 步）
        filtered.validate().map_err(|e| {
            format!(
                "filter_audio_segments broke segment invariant (sample {}): {e}",
                filtered.sample_id
            )
        })?;
        Ok(Some(filtered))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::sample::{Sample, SegmentKind, SourceRef, Supervise};

    fn sid(n: u64) -> SampleId {
        SampleId(n)
    }

    fn text_seg(order: u32, start: u32, end: u32) -> SegmentEntry {
        SegmentEntry {
            sample_id: sid(0),
            segment_order: order,
            kind: SegmentKind::Text,
            token_start: start,
            token_end: end,
            media_record_id: None,
            loss_weight: 255,
        }
    }

    fn img_seg(order: u32, start: u32, end: u32, rec: u64) -> SegmentEntry {
        SegmentEntry {
            sample_id: sid(0),
            segment_order: order,
            kind: SegmentKind::Image,
            token_start: start,
            token_end: end,
            media_record_id: Some(rec),
            loss_weight: 0,
        }
    }

    #[test]
    fn test_validate_modality_id_ranges_passes_disjoint() {
        let ranges = [
            IdRange {
                start: 32000,
                end: 32100,
                kind: SegmentKind::Image,
            },
            IdRange {
                start: 32100,
                end: 32200,
                kind: SegmentKind::Image,
            },
        ];
        // 相邻不重叠 + 在文本 vocab 之外 → 合法
        assert!(validate_modality_id_ranges(&ranges, Some(32000)).is_ok());
    }

    #[test]
    fn test_validate_modality_id_ranges_rejects_overlap() {
        let ranges = [
            IdRange {
                start: 32000,
                end: 32100,
                kind: SegmentKind::Image,
            },
            IdRange {
                start: 32050,
                end: 32200,
                kind: SegmentKind::Image,
            },
        ];
        let err = validate_modality_id_ranges(&ranges, None).unwrap_err();
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    fn test_validate_modality_id_ranges_rejects_vocab_conflict() {
        let ranges = [IdRange {
            start: 100,
            end: 200,
            kind: SegmentKind::Image,
        }];
        let err = validate_modality_id_ranges(&ranges, Some(1000)).unwrap_err();
        assert!(err.contains("conflicts with text vocab"), "{err}");
        assert!(err.contains("D6"), "{err}");
    }

    #[test]
    fn test_validate_modality_id_ranges_rejects_empty() {
        let ranges = [IdRange {
            start: 500,
            end: 500,
            kind: SegmentKind::Image,
        }];
        let err = validate_modality_id_ranges(&ranges, None).unwrap_err();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn test_encoded_sample_validate_passes_contiguous() {
        // text[0..5) + image[5..13) + text[13..20) = 20 tokens，段表连续覆盖
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 20],
            segment_table: vec![text_seg(0, 0, 5), img_seg(1, 5, 13, 0), text_seg(2, 13, 20)],
            labels: vec![-100; 20],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Image,
                bytes: Some(vec![1, 2]),
                grid_h: 2,
                grid_w: 4,
                flags: 0,
                video: None,
            }],
        };
        assert!(es.validate().is_ok());
    }

    /// P1-5 负例：视觉段缺失对应 media_ref → validate 硬失败（旧实现构建成功
    /// 且 manifest 仍累计视觉统计）。
    #[test]
    fn test_encoded_sample_validate_rejects_missing_media_ref() {
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 20],
            segment_table: vec![text_seg(0, 0, 5), img_seg(1, 5, 13, 0), text_seg(2, 13, 20)],
            labels: vec![-100; 20],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let err = es.validate().unwrap_err();
        assert!(err.contains("no matching media_ref"), "{err}");
        assert!(
            err.contains("P1-5") || err.contains("modality stats"),
            "{err}"
        );
    }

    /// P1-5 负例：同 seg_order 重复 media_ref → 无法一一对应，硬失败。
    #[test]
    fn test_encoded_sample_validate_rejects_duplicate_media_ref_order() {
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 20],
            segment_table: vec![text_seg(0, 0, 5), img_seg(1, 5, 13, 0), text_seg(2, 13, 20)],
            labels: vec![-100; 20],
            loss_weights: None,
            stats: EncodedStats::default(),
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
                    seg_order: 1,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![2]),
                    grid_h: 2,
                    grid_w: 4,
                    flags: 0,
                    video: None,
                },
            ],
        };
        let err = es.validate().unwrap_err();
        assert!(err.contains("duplicate seg_order"), "{err}");
    }

    /// P1-5 负例：段 kind 与 media_ref kind 不一致 → 硬失败。
    #[test]
    fn test_encoded_sample_validate_rejects_media_ref_kind_mismatch() {
        use crate::builder::encoded_sample::VideoMeta;
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 20],
            segment_table: vec![text_seg(0, 0, 5), img_seg(1, 5, 13, 0), text_seg(2, 13, 20)],
            labels: vec![-100; 20],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Video,
                bytes: Some(vec![1]),
                grid_h: 0,
                grid_w: 0,
                flags: 0,
                video: Some(VideoMeta {
                    n_frames: 1,
                    width: 2,
                    height: 2,
                    fps_milli: 1000,
                    start_ms: 0,
                    duration_ms: 0,
                }),
            }],
        };
        let err = es.validate().unwrap_err();
        assert!(err.contains("does not match segment kind"), "{err}");
    }

    /// P1-5 负例：孤儿 media_ref（无对应视觉段 → 会写无主 sidecar 字节）→ 硬失败。
    #[test]
    fn test_encoded_sample_validate_rejects_orphan_media_ref() {
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 20],
            segment_table: vec![text_seg(0, 0, 5), img_seg(1, 5, 13, 0), text_seg(2, 13, 20)],
            labels: vec![-100; 20],
            loss_weights: None,
            stats: EncodedStats::default(),
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
                    seg_order: 99,
                    kind: SegmentKind::Image,
                    bytes: Some(vec![2]),
                    grid_h: 2,
                    grid_w: 4,
                    flags: 0,
                    video: None,
                },
            ],
        };
        let err = es.validate().unwrap_err();
        assert!(err.contains("no matching visual segment"), "{err}");
    }

    #[test]
    fn test_encoded_sample_validate_rejects_gap() {
        // 段表覆盖 [0..5) 与 [8..13)，缺 [5..8) → 不连续
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 13],
            segment_table: vec![text_seg(0, 0, 5), img_seg(1, 8, 13, 0)],
            labels: vec![-100; 13],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let err = es.validate().unwrap_err();
        assert!(err.contains("not contiguous"), "{err}");
    }

    #[test]
    fn test_encoded_sample_validate_rejects_labels_length_mismatch() {
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![1; 10],
            segment_table: vec![text_seg(0, 0, 10)],
            labels: vec![-100; 9], // 长度不匹配
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let err = es.validate().unwrap_err();
        assert!(err.contains("labels length"), "{err}");
    }

    /// 构造合法交错样本（文本+图像+文本）的辅助测试，验证 segment_table 与 token_seq
    /// 对齐——这是可追溯链的基础，ModalityAwarePacker 的黄金断言同样依赖此不变式。
    #[test]
    fn test_interleaved_sample_segment_table_covers_token_seq() {
        let _s = Sample::text(sid(0), None, Some(5)); // 纯文本样本不在此测；测 EncodedSample
        let text_tokens = 5u32;
        let img_tokens = 8u32; // grid_h=2, grid_w=4 → 8 visual tokens
        let total = text_tokens + img_tokens + text_tokens;
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; total as usize],
            segment_table: vec![
                text_seg(0, 0, text_tokens),
                img_seg(1, text_tokens, text_tokens + img_tokens, 0),
                text_seg(2, text_tokens + img_tokens, total),
            ],
            labels: vec![-100; total as usize],
            loss_weights: None,
            stats: EncodedStats {
                text_tokens: 10,
                image_tokens: 8,
                images: 1,
                ..Default::default()
            },
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Image,
                bytes: Some(vec![1, 2, 3]),
                grid_h: 2,
                grid_w: 4,
                flags: 0,
                video: None,
            }],
        };
        assert!(es.validate().is_ok());
        // 可追溯链：segment.media_record_id -> media_refs 索引存在
        let img_seg = &es.segment_table[1];
        let rec_idx = img_seg.media_record_id.unwrap() as usize;
        assert_eq!(es.media_refs[rec_idx].seg_order, img_seg.segment_order);
        // 确认 SourceRef 可构造（编译期类型健全性）
        let _ = SourceRef::Inline;
        let _ = Supervise::LossOff;
    }

    // ------------------------------------------------------------------
    // P3a：filter_audio_segments（demux 的 Vlm 流侧，§3.1）
    // ------------------------------------------------------------------

    fn audio_seg(order: u32, start: u32, end: u32) -> SegmentEntry {
        SegmentEntry {
            sample_id: sid(0),
            segment_order: order,
            kind: SegmentKind::Audio,
            token_start: start,
            token_end: end,
            media_record_id: None,
            loss_weight: 0,
        }
    }

    /// text[0..5) + audio[5..13) + text[13..20)：剔除 audio 段后剩余两段
    /// 坐标重算为 [0..5) / [5..12)，连续覆盖不变式保持，可追溯链不断。
    #[test]
    fn test_filter_audio_segments_removes_and_reindexes() {
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 20],
            segment_table: vec![text_seg(0, 0, 5), audio_seg(1, 5, 13), text_seg(2, 13, 20)],
            labels: vec![100; 20],
            loss_weights: None,
            stats: EncodedStats {
                text_tokens: 10,
                audio_tokens: 8,
                audio_clips: 1,
                ..Default::default()
            },
            // encoder 若给 audio 段也发了 media_ref，必须随之剔除
            media_refs: vec![MediaRecord {
                sample_id: sid(0),
                seg_order: 1,
                kind: SegmentKind::Audio,
                bytes: Some(vec![9, 9]),
                grid_h: 0,
                grid_w: 0,
                flags: 0,
                video: None,
            }],
        };
        let filtered = es
            .filter_audio_segments()
            .unwrap()
            .expect("text+audio sample keeps text");
        assert_eq!(filtered.token_seq.len(), 12, "8 audio tokens removed");
        assert_eq!(filtered.segment_table.len(), 2);
        assert_eq!(filtered.segment_table[0].token_start, 0);
        assert_eq!(filtered.segment_table[0].token_end, 5);
        assert_eq!(filtered.segment_table[1].token_start, 5);
        assert_eq!(filtered.segment_table[1].token_end, 12);
        // 可追溯链原值保留：seg_order / media_record_id 不变
        assert_eq!(filtered.segment_table[1].segment_order, 2);
        assert!(filtered.media_refs.is_empty(), "audio media_ref 剔除");
        assert!(filtered.validate().is_ok());
    }

    /// 纯 audio 样本（无剩余段）→ None（不进 vlm packer；audio 走 GreedyPacker）。
    #[test]
    fn test_filter_audio_segments_pure_audio_returns_none() {
        let es = EncodedSample {
            sample_id: sid(3),
            token_seq: vec![7; 8],
            segment_table: vec![audio_seg(0, 0, 8)],
            labels: vec![-100; 8],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        assert!(es.filter_audio_segments().unwrap().is_none());
    }

    /// 无 audio 段样本零动作（Vlm 流行为逐字节不变的回归锚点）。
    #[test]
    fn test_filter_audio_segments_no_audio_is_noop() {
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 5],
            segment_table: vec![text_seg(0, 0, 5)],
            labels: vec![100; 5],
            loss_weights: None,
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let filtered = es
            .filter_audio_segments()
            .unwrap()
            .expect("text sample passes through");
        assert_eq!(filtered.token_seq, vec![7; 5]);
        assert_eq!(filtered.segment_table, vec![text_seg(0, 0, 5)]);
    }

    /// loss_weights 随段按 token 区间切片搬运（2 字节/token），长度守恒。
    #[test]
    fn test_filter_audio_segments_carries_loss_weights() {
        // text[0..3) + audio[3..7) + text[7..10)：f16 权重每 token 2 字节
        let es = EncodedSample {
            sample_id: sid(0),
            token_seq: vec![7; 10],
            segment_table: vec![text_seg(0, 0, 3), audio_seg(1, 3, 7), text_seg(2, 7, 10)],
            labels: vec![100; 10],
            loss_weights: Some([0x00u8, 0x3C].repeat(10)), // 全 1.0
            stats: EncodedStats::default(),
            media_refs: vec![],
        };
        let filtered = es.filter_audio_segments().unwrap().unwrap();
        assert_eq!(filtered.token_seq.len(), 6);
        let lw = filtered
            .loss_weights
            .as_ref()
            .expect("loss_weights 保持 Some");
        assert_eq!(lw.len(), 6 * 2, "按剩余 token 数重切");
        assert!(lw.iter().all(|&b| b == 0x00 || b == 0x3C));
        assert!(filtered.validate().is_ok());
    }
}
