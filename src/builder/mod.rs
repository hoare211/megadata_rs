pub(crate) mod encoded_sample;
pub(crate) mod metadata;
pub(crate) mod pipeline;
pub(crate) mod sample;
pub(crate) mod sidecar;
pub(crate) mod source;
mod template;
mod writer;

use bbpe_rs::tokenizer::SftEncoding;

/// 构建管线内流转的 token 化文档（方案 §4.1 / D5：中间契约最小集）。
///
/// 只包含 packer / writer 真正消费的字段：
/// - `ids`：PT 与 SFT 都要（最终写进 shard 的 input_ids）
/// - `tokens`：仅 SFT 需要（`DocumentAwarePacker` 用字符串 `contains` 匹配 turn 标记）
///
/// bbpe 侧同时产出的 `types` / `offsets` 无任何消费方（F2），在此止步——
/// 不进入 packing / writer 的签名。消除其**计算**需要 bbpe_rs 提供精简 encode
/// API，属已知项，本期不改（方案 §1 非目标）。
#[derive(Debug, Clone, Default)]
pub struct TokenizedDoc {
    pub ids: Vec<i32>,
    /// PT 路径为空 Vec；SFT 路径供 packer 匹配 turn 标记。
    pub tokens: Vec<String>,
    /// **文档唯一序号**（列扩展 v3.1：`doc_ids` 旁路列的取值来源）。
    /// 由构建管线在文档进入装箱前按处理顺序分配（0 基递增），同一构建内唯一；
    /// 未启用列扩展时恒为 0（不参与任何逻辑，writer 也不会落盘该列）。
    pub uid: u32,
}

impl From<SftEncoding> for TokenizedDoc {
    fn from(enc: SftEncoding) -> Self {
        Self {
            ids: enc.ids,
            tokens: enc.tokens,
            uid: 0,
        }
    }
}

impl From<Vec<i32>> for TokenizedDoc {
    fn from(ids: Vec<i32>) -> Self {
        Self {
            ids,
            tokens: Vec::new(),
            uid: 0,
        }
    }
}

/// 管道中流转的核心数据包
pub struct ChunkPayload {
    pub raw_json_lines: Vec<String>,  // 阶段1：原始文本
    pub rendered_texts: Vec<String>,  // 阶段2：Jinja渲染后的字符串
    pub encodings: Vec<TokenizedDoc>, // 阶段3：切词结果（bbpe 直连或 Python 回调产出）
    pub packed_arrays: arrow::record_batch::RecordBatch, // 阶段4：Arrow 内存结构
}

// ★ 评审 §3.8(d)：原先此处有一个 `#[allow(dead_code)] pub struct DataBuilder`
//   （注释自承"历史遗留的早期原型结构，保留供参考"）。它有 3 个字段
//   （tokenizer / jinja_env / max_seq_len）但**无任何 impl 块、无构造、无调用方**，
//   是纯遗留壳——与本文件其他 `#[allow(dead_code)]`（"旧版兼容 API，有真实调用方或
//   明确的对外 API 意图"）性质完全不同。已删除，避免误导后续维护者以为它是可用入口。
//   真正的构建入口是 `pipeline::run_offline_builder_pipeline`，绑定层见
//   `bindings::python::PyDataBuilder`。
