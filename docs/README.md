# Documentation map (docs/)

Only long-lived, "how to use it" material ships with the repository; design
discussions and review archives stay internal.

## Guides

| Document | Language | Content |
|---|---|---|
| [`guide/usage.md`](guide/usage.md) | English | Condensed quickstart: install → build → load → train → resume → tune → troubleshoot |
| [`guide/使用指南.md`](guide/使用指南.md) | 中文 | Full reference: every parameter, edge case and benchmark table |
| [`../README.md`](../README.md) | English | Positioning, key features, install, quick start |
| [`../README.zh-CN.md`](../README.zh-CN.md) | 中文 | 项目定位、核心特性、安装与快速上手 |
| [`../MIGRATION.md`](../MIGRATION.md) | 中文 | v1/v2 → v3 rename, snapshot and behaviour changes |

## Other useful entry points

- `tests/` — runnable usage examples (`test_control_plane.py`, `test_simulation.py`
  are the best starting points)
- `bench/` — throughput / compression / third-party tokenizer benchmarks
- `tools/verify/` — nine-stage end-to-end acceptance scripts, run in order from
  `phase0_build.py`

## Documentation index

<!-- DOCS-INDEX:BEGIN 由 tools/docs_lint.py index 自动生成，勿手工编辑 -->
| 文档 | 状态 | 一句话定位（取文档标题） |
|---|---|---|
| `guide/usage.md` | 现行英文上手文档（随代码更新；完整参考见中文版 `使用指南.md`） | megadata_rs — Usage Guide (English quickstart) |
| `guide/使用指南.md` | 现行权威（随代码更新；最近逐条核对 2026-09-14） | megadata_rs 使用指南 |
<!-- DOCS-INDEX:END -->

## Conventions

- The Chinese full guide tracks the code; when they disagree, the code and tests
  win — PRs correcting the text are welcome.
- Documentation lint runs in CI: `python3 tools/docs_lint.py check`. After adding
  or renaming a document, run `python3 tools/docs_lint.py index` first.
