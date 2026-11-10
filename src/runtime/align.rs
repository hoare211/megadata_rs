//! P3c：三模态对齐 join（读侧）——`(sample_id, window_index)` 归并两流 +
//! ts token 解码 + `av_align` 条目构造（对齐终稿 §5/§6，P3c 方案 §5–§7）。
//!
//! 本模块只做**纯函数式计算**（输入 = Vlm batch 的 input_ids / 行 sids /
//! video rec 列表 / 音频范围 rec 列表 + 指纹参数；输出 = `Vec<AlignEntry>`）——
//! 文件读取、句柄管理、预算账本在 `stream.rs`；Python 透出在 `bindings`。
//!
//! 正确性前提 = INV-2 单条 rec 自足：`window_index = floor(start_ms / W_ms)`
//! 由每条 rec 独立计算（无全局可见性前提、无空洞占位、不依赖跨 shard 枚举）。

use crate::builder::sidecar::AudioRecord;

/// audio 流内 rec 引用（P3c §6.1 v1.2 修正：一窗可含多条 rec、下标未必连续、
/// 可跨 audio batch ⇒ 用引用列表而非连续区间，防 last-write-wins 丢 rec）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioRecRef {
    /// 音频流 batch 序号（`align_aud_range` 范围内）
    pub aud_batch: u32,
    /// 该 batch `col_aud_rec` 内的下标
    pub idx: u32,
}

/// 一个 G2 窗口的对齐条目（三侧配对；缺键 = 空窗，合法——终稿 §6.4）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignEntry {
    pub sample_id: u64,
    /// `floor(rec.start_ms / W_ms)`（A/V 侧）；text 侧经 ts token 桶换算
    pub window_index: u32,
    /// 该窗 video rec 下标列表（Vlm 流内 col_vid_rec 下标，升序；空 = 空窗）
    pub video_rec_range: Vec<u32>,
    /// 该窗 audio rec 引用列表（可跨 audio batch，按 (aud_batch, idx) 升序）
    pub audio_rec_range: Vec<AudioRecRef>,
    /// 该窗 text token 区间列表（Vlm batch input_ids 内 `[start, end)`，ts token
    /// 界定；**空 = 无文本**）。按 start 升序；一行 = 一样本 ⇒ 同窗可有多段
    /// （text → 视觉段 → text 的交错样本，P1-1/P3-3 评审修复：旧单区间 API 会把
    /// 后续 Image/Video placeholder 当文本，或静默只保留第一段）。
    pub text_token_range: Vec<(u32, u32)>,
}

/// P3c §5.1：扫描 `input_ids` 命中 ts 区间的 token，返回 `(pos, bucket_id)`。
///
/// `base_id / range_size` 来自 manifest `align_fingerprint.ts_token`
/// （缺指纹 = 对齐未启用，调用方不得进入本函数）。
pub fn decode_ts_tokens(input_ids: &[i32], base_id: i32, range_size: u32) -> Vec<(usize, u32)> {
    let mut out = Vec::new();
    for (pos, &tid) in input_ids.iter().enumerate() {
        if tid >= base_id {
            let bucket = (tid - base_id) as u32;
            if bucket < range_size {
                out.push((pos, bucket));
            }
        }
    }
    out
}

/// P3c §5.2：桶 → 窗两级换算（终稿 §6.3 N7 细粒度档 join 规则）。
///
/// - 默认档 `bucket_width_ms == W_ms`：恒等（`window_index = bucket_id`）；
/// - 细粒度档：`window_index = floor(bucket_id × bucket_width_ms / W_ms)`。
pub fn bucket_to_window(bucket_id: u32, bucket_width_ms: u32, w_ms: u32) -> u32 {
    if w_ms == 0 {
        return bucket_id; // 防御：W_ms 缺失不应到达（指纹门槛已挡）
    }
    (bucket_id as u64 * bucket_width_ms as u64 / w_ms as u64) as u32
}

/// P3c §4.3/§5.2：text↔桶归属——ts token 之后、下一 ts token 之前的 text
/// token 归属该桶（一期段级归属：段跨桶不切分，长段整段归首桶）。
///
/// **P1-1（评审修复）**：读侧没有文本段边界，旧实现的区间末端直接取到
/// "下一 ts token 或行尾"，会把行内后续的 Image/Video placeholder token 纳入
/// "文本区间"（下游取文本嵌入/算文本损失即把视觉 token 当文本）。现在区间在
/// **首个非文本 token** 处终止——非文本由 `visual_ranges`（manifest
/// `modality_id_ranges` 的 image/audio/video 区间，ts token 区间不计入）
/// 判定；细粒度档下一行可有多个区间（P3-3）。
///
/// 输入 = 该行 input_ids 切片 + 行内 ts token 位置列表（已按 pos 升序）+
/// 非文本 id 区间（`(start, end)` 闭开区间列表）；
/// 输出 = `(window_index, token_range)` 列表（`token_range` 为行内 `[start, end)`，
/// 起于 ts token 之后、止于下一 ts token / 首个非文本 token / 行尾）。
pub fn text_bucket_attribution(
    row_ids: &[i32],
    ts_positions: &[(usize, u32)],
    bucket_width_ms: u32,
    w_ms: u32,
    visual_ranges: &[(i32, i32)],
) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for (i, &(pos, bucket)) in ts_positions.iter().enumerate() {
        let start = (pos + 1) as u32;
        let mut end = match ts_positions.get(i + 1) {
            Some(&(next, _)) => next as u32,
            None => row_ids.len() as u32,
        };
        // P1-1：视觉 placeholder（image/audio/video id）不是文本——区间在首个
        // 非文本 token 处截断（读侧无段表，按 manifest modality id ranges 判定）。
        for (k, &tid) in row_ids.iter().enumerate().skip(pos + 1) {
            if (k as u32) >= end {
                break;
            }
            if visual_ranges.iter().any(|&(s, e)| tid >= s && tid < e) {
                end = k as u32;
                break;
            }
        }
        if end > start {
            out.push((bucket_to_window(bucket, bucket_width_ms, w_ms), start, end));
        }
    }
    out
}

/// P3c §6.2：join 主函数——按 `(sample_id, window_index)` 归并三侧。
///
/// 输入：
/// - `row_sids`：该 batch 每行的 sid（`col_row_sids`；padding 行 = `u64::MAX`，
///   不产条目）；
/// - `batch_ids`：该 batch 的 input_ids（行主序，`bs × msl`）；
/// - `row_lens`：每行有效 token 数（text 区间据此裁到有效长度，不含 padding）；
/// - `video_recs`：该 batch 的 video rec（`(sample_id, start_ms)` 序列；Vlm 流内
///   下标 = 数组下标——`video_rec_range` 的元素即此下标）；
/// - `audio_recs`：音频范围 rec（`(aud_batch, idx_in_batch, AudioRecord)`；
///   `aud_batch` 为音频流 batch 序号、`idx_in_batch` 为该 batch `col_aud_rec`
///   内的下标——两者构成 `AudioRecRef`）；
/// - `visual_ranges`：非文本（image/audio/video）token id 区间列表——text 区间
///   在首个非文本 token 处终止（P1-1 评审修复）；
/// - 指纹参数：`base_id / range_size / bucket_width_ms / w_ms`。
///
/// 输出：按 `(sample_id, window_index)` 归并的 `AlignEntry` 列表（按键排序稳定）。
/// 守恒：所有有 `start_ms` 的 rec 均出现在某条目中且恰一次。
pub fn join_align_entries(
    row_sids: &[u64],
    batch_ids: &[i32],
    row_lens: &[i32],
    max_seq_len: usize,
    video_recs: &[(u64, u32)],
    audio_recs: &[(u32, u32, AudioRecord)],
    visual_ranges: &[(i32, i32)],
    base_id: i32,
    range_size: u32,
    bucket_width_ms: u32,
    w_ms: u32,
) -> Vec<AlignEntry> {
    // key = (sid, window) → (video idx 列表, audio ref 列表, text 区间列表)
    type Sides = (Vec<u32>, Vec<AudioRecRef>, Vec<(u32, u32)>);
    let mut map: std::collections::BTreeMap<(u64, u32), Sides> = std::collections::BTreeMap::new();

    // 1. text 侧：逐行扫 ts token（一行 = 一样本 ⇒ sid 单值）
    for (row, &sid) in row_sids.iter().enumerate() {
        if sid == u64::MAX {
            continue; // padding 行无样本
        }
        let base = row * max_seq_len;
        let valid = (row_lens.get(row).copied().unwrap_or(0).max(0) as usize).min(max_seq_len);
        let row_ids = &batch_ids[base..base + valid];
        let ts = decode_ts_tokens(row_ids, base_id, range_size);
        for (window, start, end) in
            text_bucket_attribution(row_ids, &ts, bucket_width_ms, w_ms, visual_ranges)
        {
            map.entry((sid, window))
                .or_default()
                .2
                .push((base as u32 + start, base as u32 + end));
        }
    }

    // 2. video 侧：Vlm 流内 rec（下标 = 数组下标）
    for (idx, &(sample_id, start_ms)) in video_recs.iter().enumerate() {
        let window = (start_ms as u64 / w_ms.max(1) as u64) as u32;
        map.entry((sample_id, window))
            .or_default()
            .0
            .push(idx as u32);
    }

    // 3. audio 侧：音频范围 rec（可跨 batch；ref 携带 (batch, idx)）
    for (aud_batch, idx_in_batch, rec) in audio_recs {
        let window = (rec.start_ms as u64 / w_ms.max(1) as u64) as u32;
        map.entry((rec.sample_id, window))
            .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()))
            .1
            .push(AudioRecRef {
                aud_batch: *aud_batch,
                idx: *idx_in_batch,
            });
    }

    // 4. 汇总（BTreeMap 按键排序 ⇒ 输出稳定可复现）
    map.into_iter()
        .map(|((sample_id, window_index), (mut v, mut a, mut t))| {
            v.sort_unstable();
            a.sort_by_key(|x| (x.aud_batch, x.idx));
            // text：一行 = 一样本 ⇒ 该 (sid, window) 可有多个区间（text → 视觉段
            // → text 交错；P1-1/P3-3）。按 start 升序输出，绝不静默取第一段。
            t.sort_unstable();
            AlignEntry {
                sample_id,
                window_index,
                video_rec_range: v,
                audio_rec_range: a,
                text_token_range: t,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn vid_rec(sid: u64, start_ms: u32) -> (u64, u32) {
        (sid, start_ms)
    }

    fn aud_rec(sid: u64, start_ms: u32) -> AudioRecord {
        AudioRecord {
            offset: 0,
            len: 10,
            sample_rate: 16000,
            duration_ms: 100,
            channels: 1,
            flags: 0,
            sample_id: sid,
            seg_order: 0,
            start_ms,
        }
    }

    #[test]
    fn ts_decode_and_bucket_to_window() {
        // base=32000, range=2048：命中区间内/外
        let ids = vec![31999, 32000, 32005, 34047, 34048, 5];
        let ts = decode_ts_tokens(&ids, 32000, 2048);
        assert_eq!(ts, vec![(1, 0), (2, 5), (3, 2047)]);
        // 默认档恒等；细粒度档 floor(bucket×100/2000)
        assert_eq!(bucket_to_window(5, 2000, 2000), 5);
        assert_eq!(bucket_to_window(5, 100, 2000), 0);
        assert_eq!(bucket_to_window(39, 100, 2000), 1);
    }

    #[test]
    fn text_attribution_segment_level() {
        // 行内：[t0] a b [t2] c  ⇒ (w0, [1,3)),(w2, [4,5))
        let ids = vec![32000, 10, 11, 32002, 12];
        let ts = vec![(0usize, 0u32), (3, 2)];
        let got = text_bucket_attribution(&ids, &ts, 2000, 2000, &[]);
        assert_eq!(got, vec![(0, 1, 3), (2, 4, 5)]);
    }

    /// **P1-1 回归锁**：text 区间必须在首个 Image/Video placeholder token 处终止，
    /// 不得把视觉 token 纳入"文本区间"（旧实现末端取行尾 ⇒ Counter 含 IMG）。
    #[test]
    fn text_attribution_stops_at_visual_placeholder() {
        // 行内：[ts w0] text text IMG×3 [ts w2] text  ⇒ w0 区间止于 IMG 起点
        let img = 32000;
        let ids = vec![32000, 10, 11, img, img + 1, img + 2, 32002, 12];
        let ts = vec![(0usize, 0u32), (6, 2)];
        let visual = [(img, img + 100)];
        let got = text_bucket_attribution(&ids, &ts, 2000, 2000, &visual);
        assert_eq!(
            got,
            vec![(0, 1, 3), (2, 7, 8)],
            "w0 区间必须在首个 IMG token（pos 3）处截断；IMG token 不是文本"
        );
    }

    /// P1-1 边界：ts token 之后紧跟视觉 placeholder ⇒ 该窗无 text 区间。
    #[test]
    fn text_attribution_empty_when_only_visual_after_ts() {
        let img = 32000;
        let ids = vec![32000, img, img + 1];
        let ts = vec![(0usize, 0u32)];
        let got = text_bucket_attribution(&ids, &ts, 2000, 2000, &[(img, img + 100)]);
        assert!(
            got.is_empty(),
            "ts 后纯视觉段 ⇒ 无 text 区间，不得产出空/反向区间"
        );
    }

    #[test]
    fn join_merges_three_sides_by_sid_window() {
        // 2 行 batch（msl=4）：row0 sid=7（ts w1 + video w1），row1 sid=8（纯文本无 ts）
        let ids = vec![32001, 10, 11, 0, 20, 21, 22, 0];
        let row_sids = vec![7u64, 8u64];
        let vrecs = vec![vid_rec(7, 2500u32)]; // w1
        let arecs = vec![(0u32, 0u32, aud_rec(7, 2500))]; // w1
        let entries = join_align_entries(
            &row_sids,
            &ids,
            &[3, 3],
            4,
            &vrecs,
            &arecs,
            &[],
            32000,
            2048,
            2000,
            2000,
        );
        assert_eq!(entries.len(), 1, "sid=8 无 A/V 无 ts → 不进对齐视图");
        let e = &entries[0];
        assert_eq!((e.sample_id, e.window_index), (7, 1));
        assert_eq!(e.video_rec_range, vec![0]);
        assert_eq!(
            e.audio_rec_range,
            vec![AudioRecRef {
                aud_batch: 0,
                idx: 0
            }]
        );
        // text 区间：row0 内 [1,3)（ts token 之后到行尾）
        assert_eq!(e.text_token_range, vec![(1, 3)]);
    }

    /// P1-1 回归锁（join 层）：text + 视觉 placeholder + 后续 text 的交错行，
    /// text_token_range 必须排除视觉 token，且同窗多段全部保留（P3-3）。
    #[test]
    fn join_excludes_visual_tokens_and_keeps_multi_ranges() {
        // msl=8：row0 = [ts w0] t t IMG×2 [ts w2] t（视觉段后又有文本段）
        let img = 32000;
        let ids = vec![32000, 10, 11, img, img + 1, 32002, 12, 0];
        let row_sids = vec![7u64];
        let vrecs = vec![vid_rec(7, 0u32)]; // video w0（同窗）
        let arecs = vec![];
        let entries = join_align_entries(
            &row_sids,
            &ids,
            &[7],
            8,
            &vrecs,
            &arecs,
            &[(img, img + 100)],
            32000,
            2048,
            2000,
            2000,
        );
        assert_eq!(entries.len(), 2, "w0 与 w2 各一条");
        let e0 = &entries[0];
        assert_eq!((e0.sample_id, e0.window_index), (7, 0));
        assert_eq!(e0.video_rec_range, vec![0]);
        // w0 只有 [1,3) 一段——IMG token（pos 3..5）不是文本
        assert_eq!(e0.text_token_range, vec![(1, 3)]);
        let e2 = &entries[1];
        assert_eq!((e2.sample_id, e2.window_index), (7, 2));
        assert_eq!(e2.text_token_range, vec![(6, 7)]);
    }

    #[test]
    fn join_empty_window_and_multi_rec() {
        // 同窗多 rec（video 2 条 + audio 跨 batch 2 条）+ padding 行
        let ids = vec![0i32; 8];
        let row_sids = vec![u64::MAX, 9u64];
        let vrecs = vec![vid_rec(9, 0u32), vid_rec(9, 100u32)];
        let arecs = vec![(0u32, 0u32, aud_rec(9, 0)), (1u32, 0u32, aud_rec(9, 100))];
        let entries = join_align_entries(
            &row_sids,
            &ids,
            &[0, 0],
            4,
            &vrecs,
            &arecs,
            &[],
            32000,
            2048,
            2000,
            2000,
        );
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.sample_id, 9);
        assert_eq!(e.video_rec_range, vec![0, 1]);
        assert_eq!(
            e.audio_rec_range,
            vec![
                AudioRecRef {
                    aud_batch: 0,
                    idx: 0
                },
                AudioRecRef {
                    aud_batch: 1,
                    idx: 0
                }
            ]
        );
        assert!(e.text_token_range.is_empty(), "无 ts token → 无 text 区间");
    }
}
