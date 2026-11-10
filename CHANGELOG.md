# 更新日志

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [0.2.0] - 2026-11-10

首个公开版本。

### Added

- **双模打包引擎**：SFT（Document-Aware 二维装箱 + ChatML 模板渲染 + `-100` 掩码）、
  PT（Greedy Packing 无缝拼接），统一输出 `cu_seqlens` 供 varlen attention 使用。
- **零拷贝 FFI**：自研 DLPack C ABI 与 Arrow C Data Interface，Rust 堆内存在 O(1) 内
  转交 PyTorch Tensor，Python 端零字节拷贝。
- **多源温度混采**：`MixBatchStream` 按温度采样在底层并发抽样，运行时调 `update_temperature()`。
- **确定性分片与秒级续训**：确定性 PRNG + 轮询分配划分 shard；劫持 PRNG 与文件偏移，
  断点恢复可 100% 还原跨源抽取顺序（`state_dict` 仅几十字节）。
- **生产级控制面**：shard 聚合（默认按 ~128MB）、`index.json` 记录 sha256、
  云端原子下载 + 重试 + 跨进程单飞 + LRU 有界缓存、预取滑窗节流、置换表 shuffle、
  弹性扩缩容（`relaxed` + `num_canonical_nodes`）、三维 world 拓扑。
- **可选压缩**：PT 裸流 per-batch zstd frame（`Zstd-MDS`，保留 O(1) 随机寻址，默认开启）；
  SFT 走 Arrow IPC 原生 zstd/lz4。
- **Tier-1 生态接入**：`StreamingDataset(IterableDataset)` + `MegadataDataLoader(DataLoader)`，
  HF Trainer / Lightning 可直接使用。
- **多模态扩展**：图像/音频/视频 sidecar（`col_img_blob` / `col_img_rec` / `mm_cu`），
  音视频时间对齐，tar/parquet 源读取。
- **文档级旁路列**：`doc_ids` / `loss_weights` 零拷贝透出（i32 / f16）。
- **工具链**：`tools/verify/` 九阶段端到端验收脚本、`bench/throughput.py`、
  `bench/compression_bench.py`、`bench/compare_baselines.py`、`tools/docs_lint.py`。

### Notes

- 接口协议为 **v3**（相对 v1/v2 完全重置，旧类名不保留别名）。
  迁移映射见 [MIGRATION.md](MIGRATION.md)。
- 构建依赖 `bbpe_rs` v0.1.0（crates.io），cargo 自动拉取。
