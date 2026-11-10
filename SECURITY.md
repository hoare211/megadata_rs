# 安全策略

## 支持范围

只对最新发布版本（见 `CHANGELOG.md`）与 `main` 分支提供安全修复。

## 报告漏洞

请**不要**通过公开 issue 报告安全问题。请使用 GitHub 的
[Private vulnerability reporting](https://github.com/hoare211/megadata_rs/security/advisories/new)，
或直接邮件联系维护者（见仓库 `Cargo.toml` / `pyproject.toml` 的 maintainer 字段）。

请在报告里包含：

- 受影响版本、平台、Python/Rust 版本；
- 复现步骤或最小复现脚本；
- 影响评估（数据泄露 / 任意文件读写 / 拒绝服务 / 依赖供应链）。

我们会在 7 天内响应，并在修复发布后于 `CHANGELOG.md` 致谢（如你愿意署名）。

## 已知的安全相关设计

- **不内置任何凭据**：云存储访问统一走标准环境变量（`AWS_ACCESS_KEY_ID` /
  `AWS_SECRET_ACCESS_KEY` / `AWS_ENDPOINT_URL` 等），代码中不含任何密钥或内网地址。
- **路径与长度校验先于内存分配**：sidecar / shard 读取路径先校验结构再分配，
  长度不符或越界一律 fail-fast 抛 `OSError`，不会静默返回空数据。
- **完整性校验**：每个 shard 写完回读 sha256 记入 `index.json`；
  云端下载走 `.tmp` + rename 原子写，下载后强制校验，失败删缓存重下。
- **反序列化边界**：只解析本库自己产出的 Arrow IPC / Megatron idx / 自定义 sidecar 布局，
  不加载 pickle 或其它可执行格式。
