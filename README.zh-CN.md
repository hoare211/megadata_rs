# megadata_rs: The Ultimate LLM Data Pipeline Engine

[English](README.md) | **简体中文**

[![Rust](https://img.shields.io/badge/Rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyO3](https://img.shields.io/badge/Binding-PyO3-red.svg)](https://github.com/PyO3/pyo3)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

`megadata_rs` 是专为万卡级大规模语言模型（LLM）与多模态模型（VLM）训练打造的 **Rust 原生极速数据流转引擎**。
它旨在彻底终结大模型训练中因 `JSON 解析、Python GIL、多进程 IPC 拷贝、磁盘 I/O 拥塞` 而导致的 GPU 空转与算力浪费问题，提供从海量原始数据到 GPU Tensor 的**绝对零内存拷贝（Zero-Copy）**极限吞吐解决方案。

## 核心特性

* **双模打包引擎**
  * **SFT 模式**：原生支持 HuggingFace ChatML 等模板渲染。使用 Document-Aware 二维装箱算法，精准生成 `-100` 掩码，并输出 `cu_seqlens` 供 FlashAttention 2 无缝调用，消除交叉 Attention 幻觉。
  * **PT 模式**：采用 Greedy Packing 贪心算法无缝拼接海量预训练语料，极限压榨 MFU（算力利用率）。PT 的有效 token 流跨文档无缝、不丢不重；但若训练循环按整张 `(B,S)` 算 loss，须用 `cu_seqlens` 把 eos 填充位（收尾未满 batch）截出监督，或改用 varlen attention。
  * ⚠️ **PT 长文档会被跨 batch 切分**：超过 `batch_size × max_seq_len` 的文档被连续铺到多个 batch，每段在 `cu_seqlens` 中各自成段——数据不丢，但 varlen 训练下**跨 batch 上下文被静默切断**。构建期会汇总 WARN 并把切分统计写入 `index.json`（`global_stats.split_docs` / `split_tokens`），加载时再 WARN 一次；对策是把 `max_seq_len` 设到覆盖绝大多数文档（详见使用指南 §4.5）。
* **极限零拷贝（Zero-Copy FFI）**
  * 通过自研 **DLPack C ABI** 和 **Arrow C Data Interface**，将 Rust 堆内存在 $O(1)$ 时间内转移给 PyTorch Tensor。无论多大的 Batch，Python 端都不发生哪怕 1 Byte 的数据拷贝。
* **动态多源打碟机（Temperature Mixer）**
  * 告别静态混洗数据。支持运行时挂载多个数据流（如 Code/Math/Web），基于**温度采样算法（Temperature Scaling）**在底层异步并发抽样，平滑控制模型知识配比。
* **确定性分片与秒级断点续训（MMP Protocol）**
  * 无需中心化 RPC 通信，千卡集群基于确定性 PRNG + 轮询分配（round-robin）自动划分 Shard（分片），同 epoch 内所有 rank 看到的分片序列完全一致、可复现。
  * 劫持底层 PRNG 随机数发生器与文件偏移量，支持**毫秒级恢复宕机状态**（100% 还原跨源抽取顺序）。
* **生产级控制面**
  * **Shard 聚合**：单个 shard 文件聚合 N 个 batch（默认按 ~128MB/shard 自动），文件数从"每 batch 一个"降到几百个；加载端强校验 `shard 数 >= world_size`，杜绝大集群 rank 空转。
  * **完整性校验**：每个 shard 写完回读 sha256 记入 `index.json`；云端下载原子写（`.tmp` + rename）+ 下载后强制校验 + 失败删缓存重下 + 指数退避重试（带抖动）+ 跨进程单飞（flock）+ LRU 有界缓存（`MEGADATA_CACHE_LIMIT`）。**本地数据集默认不校验**（零开销），可用 `MEGADATA_VERIFY_LOCAL=1` 开启全量、`=sample:P` 按比例抽样（结果按 size+mtime 缓存，同一文件只 hash 一次）。
  * **错误不静默**：IO / 解压 / 数据损坏一律抛 `OSError`（消息含 shard 名 + batch 下标），不会被当成"读完"或无限重试挂住训练；本地读带**有限次重试**（2 次，50/200ms 退避，瞬时半截文件可自愈），重试用尽才上报；EOF 与用户取消仍按正常结束处理，且 EOF 后 reader 保持存活以支撑续训 Seek。
  * **预取线程 panic 也会上报**（不降级成"读完"）：后台预取线程体包 `catch_unwind`，任何 panic（含锁中毒）都会被记为致命错误抛给消费端，而不是静默截断本轮数据。
  * **可选压缩**：PT 裸流 per-batch zstd frame（`Zstd-MDS`，保留 O(1) 随机寻址）——**默认开启**（真实语料 benchmark：151k 词表压缩到 42%，解压 629MB/s，随机访问 +103µs；`compression="none"` 关闭）；SFT 走 Arrow IPC 原生 zstd/lz4（显式开启）。
  * ⚠️ `batches_per_shard` 自动档位按**未压缩**字节算（PT+zstd 实际落盘约为其 42%），做磁盘容量规划时按 0.42 倍估算。
  * **预取滑窗节流**：`predownload` 参数限制在途预取 batch 数（默认 16），削平 IO 突刺并约束预取内存上限。
  * **置换表 Shuffle**：Rust 侧按 (seed, epoch) 生成 batch 级确定性置换表，`state_dict` 从 GB 级 buffer 缩到几十字节，天然跨进程可复现。
  * **弹性扩缩容**：`relaxed` 分区 + `num_canonical_nodes` 弹性锚点，全局样本序与集群拓扑无关，换节点数可续训。
  * **三维 World 拓扑**：node × rank × worker，`num_workers>0` 时每个 worker 自动拿到互斥 shard 子集（reader 按 fork 惰性重建）。
  * **开箱即用生态**：`StreamingDataset(IterableDataset)` + `MegadataDataLoader(DataLoader)`，HF Trainer / Lightning 直接可用。

## 安装与编译

需要 Rust 1.85+（`edition = "2024"`）、Python 3.8+ 与 `maturin`。
`bbpe_rs`（分词器）是 crates.io 上的正式依赖，`cargo` 自动拉取，无需手动放置：

```bash
pip install maturin pytest torch pyarrow
maturin develop --release
```

想用 `bbpe_tokenizer` 训练/导出自己的分词器时：

```bash
pip install "git+https://github.com/hoare211/bbpe_rs@v0.1.0"
```

## 极速上手

### 1. 离线构建极速数据集（`DataBuilder`）

`DataBuilder` 使用 Tokio 异步 IO 和 Rayon 多线程池，瞬间吞噬 JSONL 语料并落盘为极限格式。

```python
from megadata import DataBuilder

builder = DataBuilder(tokenizer_path="./model_tokenizer", max_seq_len=8192)

# 构建 SFT 对话数据集（落盘为 Arrow IPC 格式）
builder.build_dataset(
    input_paths=["data/sft/raw.jsonl"],
    output_dir="data/sft_processed",
    mode="sft",
    template_str="{% for msg in messages %}<|im_start|>{{ msg.role }}\n{{ msg.content }}<|im_end|>\n{% endfor %}",
    batch_size=16,
)

# 构建 PT 预训练数据集（落盘为 Megatron 风格的裸流 Memmap .bin/.idx）
builder.build_dataset(
    input_paths=["data/pt/raw.jsonl"],
    output_dir="data/pt_processed",
    mode="pt",
    batch_size=16,
    # 可选控制面参数：
    # batches_per_shard=0   # shard 聚合粒度：0=自动(~128MB/shard)；加载端校验 shard 数 >= world_size
    # compression="zstd"    # PT 默认 zstd（per-batch frame，Zstd-MDS）；"none" 关闭
)
```

### 2. 多源零拷贝加载与训练

```python
import torch
from megadata import MixBatchStream

# 温度控制的多源混采（Code/Math/Web 配比、课程学习、退火）
loader = MixBatchStream(
    [
        {"path": "data/pt_processed/index.json", "weight": 0.7},
        {"path": "data/sft_processed/index.json", "weight": 0.3},
    ],
    batch_size=16,
    max_seq_len=8192,
    world_size=1024,
    rank=0,                  # DDP 拓扑直接交给 megadata，无需 DistributedSampler
    steps_per_epoch=1000,    # DDP 下必须给确定性轮长
)
loader.update_temperature(2.0)   # 温度平滑：提升小语料曝光率

# 断点恢复（毫秒级 Jump）
# loader.load_state_dict(checkpoint["megadata_state"])

for batch in loader:
    # 这里的 Tensor 内存直接映射自 Rust，0 开销
    input_ids = batch["input_ids"].cuda()

    loss = model(input_ids).loss
    loss.backward()

# 导出极其精确的（文件游标 + 随机数）状态快照
# checkpoint["megadata_state"] = loader.state_dict()
```

> 单源训练循环用 `BatchStream`；评估/抽查用 `RandomAccessBatchReader`；
> HF Trainer / Lightning 集成用 `StreamingDataset` + `MegadataDataLoader`。详见使用指南。

## 文档

- **[`docs/guide/使用指南.md`](docs/guide/使用指南.md)** —— 权威使用说明（构建 → 加载 → 训练集成 → 断点续训 → 调参 → 排障）
- **[`docs/guide/usage.md`](docs/guide/usage.md)** —— 英文精简版上手文档
- **[`MIGRATION.md`](MIGRATION.md)** —— v1/v2 → v3 的类名、快照与行为差异迁移表
- **[`docs/README.md`](docs/README.md)** —— 文档导航

## CI（GitHub Actions）

`.github/workflows/ci.yml` 四个 job：`lint`（ruff check python/ tests/ tools/）、`docs`
（`docs_lint.py check` + `self-test`，纯标准库）、`rust-test`（`cargo clippy --all-targets
-p megadata_rs -- -D warnings` + `cargo test --lib`）、`python-test`（maturin 构建 + pytest）。
`bench.yml` 手动触发吞吐基准，`nightly.yml` 每晚跑全量 proptest fuzz。

无需配置任何仓库变量：`bbpe_rs` 走 crates.io 依赖，`cargo` 自动拉取。

## 工业级质量保证

本项目配备严苛的四维测试矩阵（`tests/` 目录）：

1. **端到端对齐（E2E Parity）**：证明 FlashAttention 拓扑不漏一条数据。
2. **分布式隔离（Distributed Integrity）**：证明多 Rank 下数据严格互斥。
3. **断点续训时空检验（Time-Travel）**：证明多源混采打断后重启，数据字节与伪随机数抽样顺序 $100\%$ 对齐。
4. **内存泄漏压测（Memory Leak Test）**：经过高频 C FFI 跨语言通信与 Python GC 压力测试，证明无任何内存泄漏。

## 贡献

欢迎 issue 与 PR，提交前请读 [`CONTRIBUTING.md`](CONTRIBUTING.md)（含本地自检命令）。
安全问题请走 [`SECURITY.md`](SECURITY.md) 的私下渠道。

## 许可证

本项目以 [MIT](LICENSE) 许可发布。

## 致谢

分词能力由姊妹项目 [bbpe_rs](https://github.com/hoare211/bbpe_rs)（Rust BPE 分词器训练器/编码器）提供；
`object_store` / `arrow` / `pyo3` / `tokio` / `rayon` 等上游项目构成了本引擎的基础设施。
