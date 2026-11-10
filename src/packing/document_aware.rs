use crate::builder::TokenizedDoc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct PackedBatch {
    pub input_ids: Vec<i32>,
    pub labels: Vec<i32>,
    pub position_ids: Vec<i32>,
    pub cu_seqlens: Vec<i32>,
    /// 每行的**有效长度**（不含行内 padding），长度恒为 batch_size。
    ///
    /// SFT（DocumentAwarePacker）专有：文档不跨行、padding 分散在每行末尾，
    /// `cu_seqlens` 是跨行逻辑累计，无法直接推出各行物理边界；
    /// `row_lens` 把该信息**自描述**地写进数据集，读取端无需在运行期复刻装箱算法
    /// （A4：数据自描述优先于运行期反推）。
    /// PT（GreedyPacker）无行概念，恒为空 Vec。
    pub row_lens: Vec<i32>,
    /// **文档级旁路列**（DLPack 列扩展落地层）：
    ///
    /// - `doc_ids`：与 `input_ids` 同构的逐 token 文档序号（`batch_size*max_seq_len`），
    ///   pad 位置填 `-1`。用途：文档级 loss 加权 / 课程学习 / 训练样本追溯——
    ///   这些是"文档身份"需求，`cu_seqlens` 只给边界不给身份，覆盖不了。
    /// - `loss_weights`：逐 token 权重，f16 LE 字节流（每 token 2 字节）。
    ///   默认有效 token = 1.0、pad = 0.0（后续可按 doc 广播真实权重）。
    ///
    /// 两者**仅在构建期开启列扩展时填充**；否则恒为空 Vec（writer 跳过落盘、
    /// manifest 不记 schema key、读取端按"无该列"降级）。
    pub doc_ids: Vec<i32>,
    pub loss_weights: Vec<u8>,
    pub batch_size: usize,
    pub max_seq_len: usize,
}

/// f16 常量：1.0（有效 token 的默认权重）与 0.0（padding 权重）。
/// IEEE 754 half 位型的编译期常量（LE 字节序）：
/// 1.0 → 0x3C00（指数 15、尾数 0）；0.0 → 0x0000。
/// 用字面量而非 `half::f16::from_f32` —— 后者非 const fn，无法用于常量初始化。
pub(crate) const F16_ONE_LE: [u8; 2] = [0x00, 0x3C];
pub(crate) const F16_ZERO_LE: [u8; 2] = [0x00, 0x00];

pub struct DocumentAwarePacker {
    max_seq_len: usize,
    pad_id: i32,

    // 动态传入的控制符配置（兼容各种大模型）
    turn_start_id: i32,
    turn_end_id: i32,
    turn_start_str: String,
    turn_end_str: String,
    assistant_str: String,

    // F2 截断统计（跨 chunk 原子累计；构建结束时由管线经 take_truncation_stats
    // 汇入 BuildProgress → WARN + manifest global_stats，消灭"静默截断"）。
    /// 因超过 max_seq_len 被尾部截断的文档数
    truncated_docs: AtomicU64,
    /// 截断丢弃的 token 数（doc_len - safe_len 之和）
    truncated_tokens: AtomicU64,
    /// labels 全为 -100 的"废样本"文档数（user 段占满 / assistant 段被截掉时出现）
    all_masked_docs: AtomicU64,
}

impl DocumentAwarePacker {
    pub fn new(
        max_seq_len: usize,
        pad_id: i32,
        turn_start_id: i32,
        turn_end_id: i32,
        turn_start_str: String,
        turn_end_str: String,
        assistant_str: String,
    ) -> Self {
        Self {
            max_seq_len,
            pad_id,
            turn_start_id,
            turn_end_id,
            turn_start_str,
            turn_end_str,
            assistant_str,
            truncated_docs: AtomicU64::new(0),
            truncated_tokens: AtomicU64::new(0),
            all_masked_docs: AtomicU64::new(0),
        }
    }

    /// 取走累计的截断统计（原子换出 0，供管线逐 chunk 汇入 BuildProgress）。
    /// 返回 `(truncated_docs, truncated_tokens, all_masked_docs)`。
    pub fn take_truncation_stats(&self) -> (u64, u64, u64) {
        (
            self.truncated_docs.swap(0, Ordering::Relaxed),
            self.truncated_tokens.swap(0, Ordering::Relaxed),
            self.all_masked_docs.swap(0, Ordering::Relaxed),
        )
    }

    /// 核心逻辑 1：泛用型掩码状态机 (支持 ChatML, Llama-3, Gemma 等一切格式)
    ///
    /// 【修复 next-token shift】原实现 labels[t] = encoding.ids[t]，等价于让模型学
    /// "恒等拷贝"——loss 会迅速塌到 ~0。SFT 标准是 next-token prediction：labels[t]
    /// 应该是 encoding.ids[t+1]。末尾位置没有下一个 token，用 -100 掩掉。
    ///
    /// assistant 段识别后，labels 设为 "下一个 token"；非 assistant 段保持 -100。
    /// 这样非 assistant 段后的第一个 token（即 assistant 段的 BOS / 角色行）也
    /// 自动是 -100，符合标准做法。
    /// 【前置条件】`encoding` 必须是**单篇文档**（调用方 pack_chunks 逐文档调用）。
    /// `is_assistant` 状态机跨整个 encoding 维护、依赖文档完整性——若文档在
    /// assistant 段内被截断（无 turn_end），状态只影响本篇末尾（不会泄漏到
    /// 下一篇）；未来若改为批量处理，需在每个文档边界显式重置。
    fn generate_labels(&self, encoding: &TokenizedDoc) -> Vec<i32> {
        let n = encoding.ids.len();
        // 先按原规则生成每个位置是否是监督位的掩码
        let mut supervised = vec![false; n];
        // 显式从 false 开始（单文档前置条件下的确定状态，防未来批量化的状态泄漏）
        let mut is_assistant = false;

        for (i, supervised_i) in supervised.iter_mut().enumerate() {
            let id = encoding.ids[i];
            let tok = encoding.tokens.get(i).map(String::as_str).unwrap_or("");

            // 1. 判断是否是对话轮次的起点
            let is_turn_start = (self.turn_start_id >= 0 && id == self.turn_start_id)
                || (!self.turn_start_str.is_empty() && tok.contains(&self.turn_start_str));

            if is_turn_start {
                let mut role_text = String::new();
                for j in (i + 1)..std::cmp::min(i + 8, n) {
                    let next_tok = encoding.tokens.get(j).map(String::as_str).unwrap_or("");
                    if next_tok.contains('\n')
                        || next_tok.contains(&self.turn_end_str)
                        || next_tok.contains(&self.turn_start_str)
                    {
                        break;
                    }
                    role_text.push_str(next_tok);
                }
                let role_trim = role_text.trim();
                is_assistant = role_trim.starts_with(&self.assistant_str);
            }

            if is_assistant {
                *supervised_i = true;
            }

            // 2. 判断是否是对话轮次的终点
            let is_turn_end = (self.turn_end_id >= 0 && id == self.turn_end_id)
                || (!self.turn_end_str.is_empty() && tok.contains(&self.turn_end_str));

            if is_turn_end && is_assistant {
                *supervised_i = true;
                is_assistant = false;
            }
        }

        // 3. next-token shift：labels[t] = ids[t+1] 当 supervised[t] 为 true 时
        //    末位没有下一个 token → label = -100
        let mut labels = vec![-100; n];
        for i in 0..n.saturating_sub(1) {
            if supervised[i] {
                labels[i] = encoding.ids[i + 1];
            }
        }

        labels
    }

    /// 核心逻辑 2：多文档装箱 (Packing) 保持不变
    ///
    /// `row_lens` 记录与装箱同步进行：`current_bin_count` 即"当前行号"，
    /// 文档放入时若行号已越过 `row_lens` 末尾则开新行（push），否则累加当前行。
    /// 换行 pad 不改变任何 `row_lens`（行有效长度在 pad 前已定稿）。
    pub fn pack_chunks(
        &self,
        documents: Vec<TokenizedDoc>,
        batch_size: usize,
        extra_columns: bool,
    ) -> Vec<PackedBatch> {
        let mut out = Vec::new();
        let mut input_ids: Vec<i32> = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut labels: Vec<i32> = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut position_ids: Vec<i32> = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut cu_seqlens: Vec<i32> = vec![0];
        let mut row_lens: Vec<i32> = Vec::with_capacity(batch_size);
        // 文档级旁路列（开启时与 input_ids 同步推进）
        let mut doc_ids: Vec<i32> = Vec::with_capacity(batch_size * self.max_seq_len);
        let mut loss_weights: Vec<u8> = Vec::with_capacity(batch_size * self.max_seq_len * 2);
        let mut current_bin_len = 0usize;
        let mut current_bin_count = 0usize;

        let flush_current = |input_ids: &mut Vec<i32>,
                             labels: &mut Vec<i32>,
                             position_ids: &mut Vec<i32>,
                             cu_seqlens: &mut Vec<i32>,
                             row_lens: &mut Vec<i32>,
                             doc_ids: &mut Vec<i32>,
                             loss_weights: &mut Vec<u8>,
                             out: &mut Vec<PackedBatch>| {
            if input_ids.len() < batch_size * self.max_seq_len {
                let need = batch_size * self.max_seq_len - input_ids.len();
                input_ids.extend(std::iter::repeat_n(self.pad_id, need));
                labels.extend(std::iter::repeat_n(-100, need));
                position_ids.extend(std::iter::repeat_n(0, need));
                // pad 位置：doc_ids = -1（无文档归属），权重 = 0.0
                doc_ids.extend(std::iter::repeat_n(-1, need));
                loss_weights.extend(std::iter::repeat_n(F16_ZERO_LE, need).flatten());
            }
            // 尾部未使用行补 0：row_lens 恒为 batch_size 长（读取端按行切片）
            if row_lens.len() < batch_size {
                row_lens.resize(batch_size, 0);
            }
            out.push(PackedBatch {
                input_ids: std::mem::take(input_ids),
                labels: std::mem::take(labels),
                position_ids: std::mem::take(position_ids),
                cu_seqlens: std::mem::take(cu_seqlens),
                row_lens: std::mem::take(row_lens),
                doc_ids: std::mem::take(doc_ids),
                loss_weights: std::mem::take(loss_weights),
                batch_size,
                max_seq_len: self.max_seq_len,
            });
        };

        for doc in documents {
            let doc_len = doc.ids.len();
            let safe_len = std::cmp::min(doc_len, self.max_seq_len);
            if safe_len == 0 {
                continue;
            }

            // F2：超长文档不再静默——截断事件与丢弃 token 数计入统计
            if doc_len > safe_len {
                self.truncated_docs.fetch_add(1, Ordering::Relaxed);
                self.truncated_tokens
                    .fetch_add((doc_len - safe_len) as u64, Ordering::Relaxed);
            }

            if current_bin_len + safe_len > self.max_seq_len {
                let pad_len = self.max_seq_len - current_bin_len;
                input_ids.extend(std::iter::repeat_n(self.pad_id, pad_len));
                labels.extend(std::iter::repeat_n(-100, pad_len));
                position_ids.extend(std::iter::repeat_n(0, pad_len));
                if extra_columns {
                    doc_ids.extend(std::iter::repeat_n(-1, pad_len));
                    loss_weights.extend(std::iter::repeat_n(F16_ZERO_LE, pad_len).flatten());
                }
                current_bin_count += 1;

                if current_bin_count >= batch_size {
                    flush_current(
                        &mut input_ids,
                        &mut labels,
                        &mut position_ids,
                        &mut cu_seqlens,
                        &mut row_lens,
                        &mut doc_ids,
                        &mut loss_weights,
                        &mut out,
                    );
                    cu_seqlens.push(0);
                    current_bin_len = 0;
                    current_bin_count = 0;
                } else {
                    current_bin_len = 0;
                }
            }

            input_ids.extend_from_slice(&doc.ids[..safe_len]);
            let doc_labels = self.generate_labels(&doc);
            // F2：全 -100"废样本"计数（user 段占满、assistant 段被整段截掉时出现）
            if doc_labels[..safe_len].iter().all(|&l| l == -100) {
                self.all_masked_docs.fetch_add(1, Ordering::Relaxed);
            }
            labels.extend_from_slice(&doc_labels[..safe_len]);
            let doc_positions: Vec<i32> = (0..safe_len as i32).collect();
            position_ids.extend(doc_positions);
            if extra_columns {
                // doc_ids：本文档的每个 token 都标同一 uid（文档身份，非段边界）
                doc_ids.extend(std::iter::repeat_n(doc.uid as i32, safe_len));
                loss_weights.extend(std::iter::repeat_n(F16_ONE_LE, safe_len).flatten());
            }

            let last_cu = *cu_seqlens.last().unwrap();
            cu_seqlens.push(last_cu + safe_len as i32);
            current_bin_len += safe_len;

            // 与装箱同步记录行有效长度：current_bin_count 即当前行号
            if row_lens.len() == current_bin_count {
                row_lens.push(safe_len as i32);
            } else {
                *row_lens
                    .last_mut()
                    .expect("row_lens is non-empty (already pushed)") += safe_len as i32;
            }
        }

        if !input_ids.is_empty() {
            flush_current(
                &mut input_ids,
                &mut labels,
                &mut position_ids,
                &mut cu_seqlens,
                &mut row_lens,
                &mut doc_ids,
                &mut loss_weights,
                &mut out,
            );
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        // 告诉 proptest 疯狂生成以下随机数作为测试用例：
        // 1. 生成 1 到 500 篇假文档，每篇文档长度在 0 到 20000 之间浮动 (模拟极端长文和空文)
        // 2. 随机的 batch_size (1 ~ 16)
        // 3. 随机的 max_seq_len (128 ~ 8192)
        #[test]
        fn fuzz_document_aware_packing_never_panics(
            doc_lengths in prop::collection::vec(0usize..20000, 1..500),
            batch_size in 1usize..16,
            max_seq_len in 128usize..8192
        ) {
            let packer = DocumentAwarePacker::new(max_seq_len, 0, 1, 2, "start".into(), "end".into(), "assist".into());

            // 构造畸形的随机假文档
            let mut docs = Vec::new();
            for (uid, len) in doc_lengths.iter().copied().enumerate() {
                docs.push(TokenizedDoc {
                    ids: vec![1; len], // 用全 1 填充假 ID
                    tokens: vec![],
                    uid: uid as u32,
                });
            }

            // 暴力执行打包算法 (如果算法有任何数组越界，这里会直接引发 Panic，测试将失败)
            let batches = packer.pack_chunks(docs, batch_size, true);

            // 如果没 Panic，接下来校验物理定量的绝对正确性
            for batch in batches {
                // 1. 形状必须绝对对齐
                assert_eq!(batch.input_ids.len(), batch_size * max_seq_len);
                assert_eq!(batch.labels.len(), batch_size * max_seq_len);
                assert_eq!(batch.position_ids.len(), batch_size * max_seq_len);

                // 2. FlashAttention 拓扑的极性校验
                // cu_seqlens 的最后一个元素（文档总长）绝不能超过 input_ids 的总容量
                let total_len_in_cu = *batch.cu_seqlens.last().unwrap() as usize;
                assert!(total_len_in_cu <= batch.input_ids.len());

                // 3. cu_seqlens 必须是单调递增的
                for i in 1..batch.cu_seqlens.len() {
                    assert!(batch.cu_seqlens[i] >= batch.cu_seqlens[i-1]);
                }

                // 4. row_lens：每行有效长度非负、不超过 max_seq_len，总和 == cu_seqlens 末值
                assert_eq!(batch.row_lens.len(), batch_size, "row_lens must always be batch_size long");
                assert!(batch.row_lens.iter().all(|&l| l >= 0 && (l as usize) <= max_seq_len));
                let sum_rows: i32 = batch.row_lens.iter().sum();
                assert_eq!(sum_rows as usize, total_len_in_cu, "sum(row_lens) must equal the number of valid tokens");
            }
        }
    }

    /// F2 回归：截断统计——超长文档计入 truncated_docs/truncated_tokens；
    /// 无监督内容的文档（tokens 为空，状态机找不到任何 assistant 轮次）计入
    /// all_masked_docs（"labels 全 -100"废样本场景）。
    #[test]
    fn test_truncation_stats_counted() {
        let packer =
            DocumentAwarePacker::new(8, 0, -1, -1, String::new(), String::new(), String::new());
        // 文档 0：长度 20 > max_seq_len 8 → 截断（丢弃 12 token）；全 -100
        // 文档 1：长度 5 → 不截断；全 -100
        let docs = vec![
            TokenizedDoc {
                ids: vec![7; 20],
                tokens: vec![],
                uid: 0,
            },
            TokenizedDoc {
                ids: vec![7; 5],
                tokens: vec![],
                uid: 1,
            },
        ];
        let batches = packer.pack_chunks(docs, 1, false);
        // batch_size=1：每篇文档独占一行 → 两个 batch（统计断言不受影响）
        assert_eq!(batches.len(), 2);
        assert_eq!(
            batches[0].input_ids.len(),
            8,
            "the over-long doc is tail-truncated to max_seq_len"
        );
        let (truncated_docs, truncated_tokens, all_masked) = packer.take_truncation_stats();
        assert_eq!(
            truncated_docs, 1,
            "only the over-long doc counts as truncated"
        );
        assert_eq!(
            truncated_tokens, 12,
            "20 - 8 = 12 tokens dropped to truncation"
        );
        assert_eq!(
            all_masked, 2,
            "both docs have no supervised span → all labels -100"
        );
        // take 是"取走"语义：再次读取应为 0
        assert_eq!(packer.take_truncation_stats(), (0, 0, 0));
    }
}
