use crate::packing::document_aware::{F16_ONE_LE, F16_ZERO_LE, PackedBatch};
use std::sync::atomic::{AtomicU64, Ordering};

/// PT 预训练的纯 ID 贪心流式拼接器（**有状态**，修复 F3）。
///
/// 旧实现 `pack_chunks` 无状态、按读取块（每 5000 行）独立装箱：每个块末尾都
/// 强制 flush 一个未满 batch（eos 填充），导致"非满批数 == 读取块数"——既损失
/// 填充效率，又让 eos 填充位在整张量 shift-loss 下被监督。现改为**跨读取块连续
/// 填充**：`push_docs` 只在缓冲满载时出批，残余缓冲保留到 `finish`（整个输入流
/// 结束时才 flush 一次），全数据集至多一个未满 batch。
pub struct GreedyPacker {
    max_seq_len: usize,
    eos_id: i32,
    batch_size: usize,
    extra_columns: bool,

    // ---- 跨 chunk 缓冲（有状态部分，F3）----
    input_ids: Vec<i32>,
    /// 文档级旁路列（与 input_ids 同步推进）
    doc_ids: Vec<i32>,
    loss_weights: Vec<u8>,
    /// 当前 batch 内每个文档段的起始位置。初始 [0]。
    cu_seqlens: Vec<i32>,

    // ---- F2-PT 切分审计（对齐 SFT 的截断统计，见 20261007 评审 §3.5）----
    /// 因长度**严格超过一个 batch 容量**（`doc_len + eos > batch_size × max_seq_len`）
    /// 而被跨 batch 铺开的文档数。
    ///
    /// PT 与 SFT 的关键差异：SFT 超长文档**丢弃尾部**（数据损失），
    /// PT 超长文档**完整保留但分段**（数据不丢，但段间注意力被切断）。
    /// 后者在 varlen 训练下是"语义隐形损失"——原先完全没有可见性，
    /// 用户只能在训练收敛异常时反向猜测原因。
    ///
    /// **口径说明**（20261007 复核 §3.4，刻意选择的判据）：只统计"一个文档就装不进
    /// 一个 batch"的极端情况。精确判据（`batch 内已有 token + 本文档 > capacity`，
    /// 即"落点非 batch 开头也会被切"）在数学上更准，但会把**几乎所有跨 batch
    /// 边界的普通短文档**都计入（期望比例 ≈ 平均文档长 / batch 容量，随容量增大
    /// 趋近 100%）——WARN 会在每个真实数据集上触发，告警噪声化，真正需要关注的
    /// 超长文档反而被淹没。短文档跨边界是顺序装箱的固有几何现象（数据不丢、
    /// 段边界完整记录在 cu_seqlens），不构成"本文档超长"的信号，故刻意不计入。
    split_docs: AtomicU64,
    /// 被切分文档的原始 token 总数（含被分到后续 batch 的部分）。
    split_tokens: AtomicU64,
}

impl GreedyPacker {
    pub fn new(max_seq_len: usize, eos_id: i32, batch_size: usize, extra_columns: bool) -> Self {
        let capacity = batch_size * max_seq_len;
        Self {
            max_seq_len,
            eos_id,
            batch_size,
            extra_columns,
            input_ids: Vec::with_capacity(capacity),
            doc_ids: Vec::with_capacity(capacity),
            loss_weights: Vec::with_capacity(capacity * 2),
            cu_seqlens: vec![0],
            split_docs: AtomicU64::new(0),
            split_tokens: AtomicU64::new(0),
        }
    }

    /// 取走累计的切分统计（原子换出 0，供管线逐 chunk 汇入 BuildProgress）。
    /// 返回 `(split_docs, split_tokens)`。
    pub fn take_split_stats(&self) -> (u64, u64) {
        (
            self.split_docs.swap(0, Ordering::Relaxed),
            self.split_tokens.swap(0, Ordering::Relaxed),
        )
    }

    /// PT 预训练的纯 ID 贪心流式拼接。
    ///
    /// 将海量变长文档无缝拼接，产出尽可能多的满载 Batch。
    ///
    /// 与 SFT 的 DocumentAwarePacker 不同，PT 模式的设计取舍：
    /// - **不预生成 labels**：PT 的 label = input_ids（因果 LM 标准做法），
    ///   训练循环自行 `logits[:,:-1]` / `labels[:,1:]` shift。PT 的目标是
    ///   预测所有下一个 token（含跨文档），这与 SFT 的"-100 mask 非目标段"本质不同。
    /// - **不预生成 position_ids**：训练循环自行生成 `[0..seq_len]`，
    ///   或基于 cu_seqlens 做 varlen position（每文档从 0 重新计）。
    /// - **产出 cu_seqlens**：记录 batch 内每个文档段的边界，让 PT 也能选配
    ///   FlashAttention varlen（避免跨文档 attend）。语义与 SFT 一致：
    ///   `[0, seg1_end, seg2_end, ..., valid_len]`。若训练循环用标准 attention，
    ///   忽略 cu_seqlens 即可（空列表）。
    ///
    /// 参数：
    /// - `doc_ids_list`：`(文档 token ids, 文档序号)`。`doc_uids`（即这里的序号）是
    ///   列扩展 `doc_ids` 旁路列的取值来源，由构建管线在装箱前全局分配。
    /// - `extra_columns`：关闭时 `doc_ids` / `loss_weights` 恒为空 Vec（不落盘该列）。
    ///
    /// 调用约定：任意多次 `push_docs`（每次一个读取块）→ 恰好一次 `finish`。
    pub fn push_docs(&mut self, mut doc_ids_list: Vec<(Vec<i32>, u32)>) -> Vec<PackedBatch> {
        let capacity = self.batch_size * self.max_seq_len;
        let mut out = Vec::new();

        for (doc_tokens, doc_uid) in doc_ids_list.iter_mut() {
            doc_tokens.push(self.eos_id); // 强制追加结束符
            // ★ F2-PT（20261007 评审 §3.5）：超长文档的可见性。
            //   PT 不截断，但长度超过 capacity 的文档会被**连续铺到多个 batch**，
            //   每段在 `cu_seqlens` 里各自成为一个独立段。若训练侧按 README 的建议
            //   走FlashAttention varlen，则每段都是 position 从 0 开始的独立序列
            //   ——**跨 batch 的文档上下文被静默切断**。原先这里既无统计也无告警，
            //   与 SFT 完整的截断审计体系形成明显不对称。
            //   现在计入统计（由管线汇总 WARN + 写进 manifest），使其可审计。
            //
            //   ★ 口径（20261007 复核 §3.4，刻意为之，详见 split_docs 字段文档）：
            //   `doc_tokens.len()` 已含上面追加的 eos，判据是"本文档（含 eos）
            //   单独就超过一个 batch 容量"。注意这与"本文档恰好等于 capacity 且
            //   落点非 batch 开头"的极端边角不同——那种情况同样会被切分，但属
            //   普通装箱几何（信号被噪声淹没），刻意不计入。
            if doc_tokens.len() > capacity {
                self.split_docs.fetch_add(1, Ordering::Relaxed);
                self.split_tokens
                    .fetch_add(doc_tokens.len() as u64, Ordering::Relaxed);
            }
            let mut pointer = 0;

            while pointer < doc_tokens.len() {
                let remain_in_doc = doc_tokens.len() - pointer;
                let remain_in_batch = capacity - self.input_ids.len();
                let take_len = std::cmp::min(remain_in_doc, remain_in_batch);

                // 记录当前文档段在 batch 内的起始位置（与上一段不同时才记录，避免重复边界）
                let seg_start = self.input_ids.len() as i32;
                if *self.cu_seqlens.last().unwrap() != seg_start {
                    self.cu_seqlens.push(seg_start);
                }

                self.input_ids
                    .extend_from_slice(&doc_tokens[pointer..pointer + take_len]);
                if self.extra_columns {
                    // doc_ids：本文档贡献的每个 token 标同一 uid；权重 1.0（有效 token）
                    self.doc_ids
                        .extend(std::iter::repeat_n(*doc_uid as i32, take_len));
                    self.loss_weights
                        .extend(std::iter::repeat_n(F16_ONE_LE, take_len).flatten());
                }
                pointer += take_len;

                // 满了一个 Batch，立刻吐出（这是 push 阶段唯一出批的时机）
                if self.input_ids.len() == capacity {
                    self.cu_seqlens.push(capacity as i32);
                    out.push(PackedBatch {
                        labels: vec![], // PT 不预生成 label（训练循环自行 shift）
                        input_ids: std::mem::take(&mut self.input_ids),
                        position_ids: vec![], // PT 不预生成 position
                        cu_seqlens: std::mem::take(&mut self.cu_seqlens),
                        row_lens: vec![], // PT 无行概念（文档跨行连续拼接）
                        doc_ids: std::mem::take(&mut self.doc_ids),
                        loss_weights: std::mem::take(&mut self.loss_weights),
                        batch_size: self.batch_size,
                        max_seq_len: self.max_seq_len,
                    });
                    self.input_ids.reserve(capacity);
                    self.doc_ids.reserve(capacity);
                    self.loss_weights.reserve(capacity * 2);
                    self.cu_seqlens = vec![0];
                }
            }
        }

        out
    }

    /// 整个输入流结束：把最后一点残余数据用 EOS 填满并打包。
    ///
    /// 全数据集**至多**产出这一个未满 batch（F3 修复点：旧实现每个读取块末尾都
    /// flush 一个）。输入恰好填满时（无残余）不产出空 batch。
    pub fn finish(&mut self) -> Vec<PackedBatch> {
        let capacity = self.batch_size * self.max_seq_len;
        let mut out = Vec::new();
        if self.input_ids.is_empty() {
            return out;
        }

        let valid_len = self.input_ids.len() as i32;
        // cu_seqlens 最后一个值 = 有效 token 数（不含 EOS padding），与 SFT 语义一致
        if *self.cu_seqlens.last().unwrap() != valid_len {
            self.cu_seqlens.push(valid_len);
        }
        let pad_len = capacity - self.input_ids.len();
        self.input_ids
            .extend(std::iter::repeat_n(self.eos_id, pad_len));
        if self.extra_columns {
            // pad 位置不属于任何文档：doc_ids = -1、权重 0.0
            self.doc_ids.extend(std::iter::repeat_n(-1, pad_len));
            self.loss_weights
                .extend(std::iter::repeat_n(F16_ZERO_LE, pad_len).flatten());
        }

        out.push(PackedBatch {
            labels: vec![],
            input_ids: std::mem::take(&mut self.input_ids),
            position_ids: vec![],
            cu_seqlens: std::mem::take(&mut self.cu_seqlens),
            row_lens: vec![], // PT 无行概念（文档跨行连续拼接）
            doc_ids: std::mem::take(&mut self.doc_ids),
            loss_weights: std::mem::take(&mut self.loss_weights),
            batch_size: self.batch_size,
            max_seq_len: self.max_seq_len,
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(len: usize, uid: u32, fill: i32) -> (Vec<i32>, u32) {
        (vec![fill; len], uid)
    }

    /// F3 回归：跨 push_docs 保留未满 batch——push 阶段只出满载 batch，
    /// 残余保留到 finish；eos 填充只出现在最后的未满 batch。
    #[test]
    fn test_push_keeps_partial_batch_until_finish() {
        let mut packer = GreedyPacker::new(16, 9, 2, false); // capacity = 32
        // chunk 1：10+1 + 15+1 = 27 < 32 → 旧实现会 flush 未满批，新实现不出批
        let b1 = packer.push_docs(vec![doc(10, 0, 1), doc(15, 1, 2)]);
        assert!(
            b1.is_empty(),
            "push must not flush a partial batch (the old implementation flushed one per read chunk)"
        );
        // chunk 2：27 + 5+1 = 33 → 1 个满载批 + 残余 1
        let b2 = packer.push_docs(vec![doc(5, 2, 3)]);
        assert_eq!(b2.len(), 1);
        assert_eq!(b2[0].input_ids.len(), 32);
        assert_eq!(*b2[0].cu_seqlens.last().unwrap(), 32);
        // finish：残余 1 token → 恰好一个未满 batch，pad 位全部是 eos
        let b3 = packer.finish();
        assert_eq!(b3.len(), 1);
        let valid = *b3[0].cu_seqlens.last().unwrap() as usize;
        assert!(valid < 32, "the final batch must be a non-full one");
        assert!(
            b3[0].input_ids[valid..].iter().all(|&t| t == 9),
            "padding must be eos_id"
        );
    }

    /// F3 回归：分块推送与一次性推送产出**完全一致**（有状态不改变装箱结果，
    /// 有效流不丢不重）。
    #[test]
    fn test_chunked_push_equals_one_shot() {
        let docs: Vec<(Vec<i32>, u32)> = (0..40)
            .map(|d| (vec![(d % 7 + 1) as i32; 3 + d % 11], d as u32))
            .collect();
        let mut streamed = GreedyPacker::new(16, 9, 2, true);
        let mut out = Vec::new();
        for chunk in docs.chunks(7) {
            out.extend(streamed.push_docs(chunk.to_vec()));
        }
        out.extend(streamed.finish());

        let mut oneshot = GreedyPacker::new(16, 9, 2, true);
        let mut expected = oneshot.push_docs(docs.clone());
        expected.extend(oneshot.finish());

        assert_eq!(out.len(), expected.len());
        for (a, b) in out.iter().zip(&expected) {
            assert_eq!(a.input_ids, b.input_ids);
            assert_eq!(a.cu_seqlens, b.cu_seqlens);
            assert_eq!(a.doc_ids, b.doc_ids);
            assert_eq!(a.loss_weights, b.loss_weights);
        }
    }

    /// 输入恰好填满（无残余）→ finish 不产出空 batch。
    #[test]
    fn test_finish_without_residual_emits_nothing() {
        let mut packer = GreedyPacker::new(8, 9, 2, false); // capacity 16
        let b = packer.push_docs(vec![doc(7, 0, 1), doc(7, 1, 2)]); // 8+8=16 恰满
        assert_eq!(b.len(), 1);
        assert!(packer.finish().is_empty());
    }
}
