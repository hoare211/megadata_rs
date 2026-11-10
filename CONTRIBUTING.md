# 贡献指南

感谢关注 megadata_rs。提交前请先读下面几节，可以让 PR 一次过。

## 开发环境

```bash
# Rust 1.85+（edition 2024）、Python 3.8+、maturin
uv venv --python 3.12 .venv
env -u CONDA_PREFIX VIRTUAL_ENV=$PWD/.venv uv pip install maturin "torch>=2.0" "pyarrow>=14" \
    pytest "moto[server]>=5" zstandard numpy ruff tokenizers
env -u CONDA_PREFIX VIRTUAL_ENV=$PWD/.venv maturin develop --release
```

`bbpe_rs` 由 `Cargo.toml` 的 git tag 依赖自动拉取，无需手工放置。
需要用自己的分词器跑测试时，设置 `BBPE_TOKENIZER_DIR=/path/to/tokenizer`；
不设置时 `tests/conftest.py` 会现训一个迷你分词器用来兜底。

## 提交前自检

```bash
uvx ruff check python/ tests/ tools/
cargo clippy --all-targets -p megadata_rs -- -D warnings
cargo test --lib -- --skip fuzz_document_aware
.venv/bin/python -m pytest tests/
python3 tools/docs_lint.py check
```

CI（`.github/workflows/ci.yml`）执行的就是这五条；`nightly.yml` 额外跑全量 fuzz。

## 约定

- **提交信息**：`<type>(<scope>): <摘要>`，type 取 `feat|fix|refactor|docs|test|bench|chore`。
- **数据契约优先**：改动落盘格式（`index.json` 字段、shard 布局、列文件后缀）必须同步改
  读取端、写测试、并在 PR 描述里说明兼容性（老数据集是否需要重建）。
- **性能改动需带基准**：涉及热路径（打包、编码、预取、DLPack 透出）的 PR，请附
  `bench/throughput.py` 或 `bench/compression_bench.py` 前后对比。
- **注释与文档**：不要在注释里写死个人绝对路径或外部仓库路径；仓库内文档用相对路径引用。
- **测试**：新行为默认要有回归测试；纯文档改动可只跑 `docs_lint`。

## 报告问题

请附：`megadata.__version__`、Python/Rust 版本、构建参数（`mode` / `batch_size` /
`max_seq_len` / `compression`）、能复现的最小配置，以及完整报错栈。
安全问题请走 [SECURITY.md](SECURITY.md) 的私下渠道，不要开公开 issue。
