# 文档导航（docs/）

对外文档只保留"怎么用"这类长期有效的内容；设计与过程记录不随仓库发布。

## 入口

| 文档 | 内容 |
|---|---|
| [`guide/使用指南.md`](guide/使用指南.md) | 权威使用说明：构建 → 加载 → 训练集成 → 断点续训 → 调参 → 排障 |
| [`../README.md`](../README.md) | 项目定位、核心特性、安装与快速上手 |
| [`../MIGRATION.md`](../MIGRATION.md) | v1/v2 → v3 的类名、快照与行为差异迁移表 |

## 其它可读入口

- `tests/`：可运行的用法示例（`test_control_plane.py`、`test_simulation.py` 最值得先读）
- `bench/`：吞吐 / 压缩 / 第三方分词器基准脚本
- `tools/verify/`：九阶段端到端验收脚本，`phase0_build.py` 起按序执行

## 文档索引

<!-- DOCS-INDEX:BEGIN 由 tools/docs_lint.py index 自动生成，勿手工编辑 -->
| 文档 | 状态 | 一句话定位（取文档标题） |
|---|---|---|
| `guide/使用指南.md` | 现行权威（随代码更新；最近逐条核对 2026-09-14） | megadata_rs 使用指南 |
<!-- DOCS-INDEX:END -->

## 约定

- 使用指南随代码更新；与实现不一致时以代码与测试为准，欢迎提 PR 修正。
- 文档体检（CI 常驻）：`python3 tools/docs_lint.py check`；改过文档头后先跑
  `python3 tools/docs_lint.py index`。
