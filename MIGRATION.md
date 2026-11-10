# megadata v1/v2 → v3 迁移指南

> v3 是**协议重置**：旧类名直接消失、不留别名/shim，旧 `state_dict` 不兼容。
> 迁移策略：按下表机械替换即可；语义差异见"行为变化"一节。

## 1. 类名与构造签名

| v1 / v2 写法 | v3 写法 |
|---|---|
| `SequentialStreamLoader(p, bs, msl)` | `BatchStream(p, batch_size=bs, max_seq_len=msl, shuffle=False)` |
| `ShuffledStreamLoader(p, bs, msl, shuffle="permutation")` | `BatchStream(p, batch_size=bs, max_seq_len=msl)`（默认即 permutation） |
| `ShuffledStreamLoader(..., shuffle="buffer")` | `BatchStream(..., shuffle=True)`（buffer 已移除，permutation 是打散质量超集） |
| `MultiSourceMixLoader(cfgs, bs, msl, temperature=t)` | `MixBatchStream(cfgs, batch_size=bs, max_seq_len=msl)`（`temperature` 仅保留为 `update_temperature()`） |
| `LLMMapBatchLoader(p, bs, msl)`; `ds[10]` | `RandomAccessBatchReader(p, batch_size=bs, max_seq_len=msl)`; `r.get_batch(10)` |
| `StreamingDataset(p, bs, msl, ...)`（位置参数） | `StreamingDataset(p, batch_size=bs, max_seq_len=msl, ...)`（路径之后全 keyword-only） |
| `loader(num_epochs=N)` | 移除参数；轮数归外层：`for epoch in range(N): for b in loader:` |
| `loader(num_epochs=0)`（无限流） | `while True: for b in loader:`（无 `sys.maxsize` 哨兵） |
| `builder.build_dataset(..., compression=None)` | `compression="auto"`（PT→zstd，SFT→不压缩；`"none"` 显式关闭） |

> **2026-09-16 追加（表达式清理）**：
> v3 目标名又经历一次净化——`StreamLoader` → `BatchStream`、`MixLoader` → `MixBatchStream`。
> 原因不是重复实现，而是**旧名的 `Loader` 后缀会诱导"torch `DataLoader` 协议"联想，而这三个类刻意
> 不是 `Dataset` 子类**（套 `DataLoader` 会直接 `TypeError`）。新名带 `Batch` = 裸 batch 迭代器，
> 与 Tier-1 的 `StreamingDataset` / `MixStreamingDataset` / `MegadataDataLoader` 形成机械可判的分层；
> `RandomAccessBatchReader` 亦保持不变（已含 `Batch` 且语义最准）。
> **同样不留别名/shim**。

## 2. 迭代与 `len()`

| 语义 | v1 / v2 | v3 |
|---|---|---|
| 一次 `for` | `num_epochs` 轮（或直到子流耗尽） | **恰好一个 epoch** |
| `len(loader)` | × 轮数 / `sys.maxsize` | **当前轮的真实 batch 数** |
| `next(loader)` | loader 自身是 Iterator | **loader 是 Iterable**：`next(iter(loader))` |
| `break` 后重新 `for` | 视实现而异 | **重跑当前轮**（同置换表、同数据） |
| 中途 `load_state_dict` 后首次 `for` | 视实现而异 | 从 `(epoch, in_epoch_pos)` **续位**（不重复） |
| 并发两个 `for` | 读头交错（静默错数据） | `RuntimeError`（响亮失败） |
| 随机访问 | `SequentialStreamLoader.seek_to_global_batch(i)` | `RandomAccessBatchReader.get_batch(i)`（单轮语义 + `IndexError`） |

## 3. 数据契约

| 项 | v1 / v2 | v3 |
|---|---|---|
| `cu_seqlens` | Python list | **`torch.Tensor`（int32）**；Arrow 路径同契约 |
| `format` | 无 | `loader.format`（`"pt"`/`"sft"`）；`MixBatchStream.source_formats` 逐子流；**异构混采时 batch 携带 `"format"` 键** |
| 异构混采分支 | `if "labels" in batch:` | 同上（`format` 键更明确；`labels` 分支仍然可用） |
| `flatten_and_unpad` | 仅 docstring 警告 | 新增 `packer=` 参数：`"sft"` 时**抛错**；建议传 `packer=batch.get("format")` |

## 4. 快照（`state_dict` / `load_state_dict`）

v3 只认 `schema == 3`：

```python
# BatchStream
{"schema": 3, "epoch": int, "in_epoch_pos": int,   # 下一个待产出位置
 "seed": int, "manifest_fingerprint": {...}, "topology": {...}}

# MixBatchStream 增补
{"weights": [...], "steps_yielded": int,
 "sub_stream_cursors": [[epoch, in_epoch], ...],  # 逐子流
 "rng_word_pos": int}
```

| 旧字段 | v3 |
|---|---|
| `consumed`（全局游标） | `epoch` + `in_epoch_pos` |
| `mode: "permutation"/"buffer"` | 无（buffer 已移除） |
| `num_epochs` | 无（轮数归外层） |
| `sub_stream_consumed`（逐子流计数） | `sub_stream_cursors`（逐子流 `(epoch, in_epoch)`） |
| 无指纹 | `manifest_fingerprint`（身份字段不一致 → 拒绝；`index_path` 变化 → 仅告警） |

- 旧 state（无 `schema`）→ `ValueError`：用 v3 重新开始训练；
- `load_state_dict` 会**覆盖构造期的 `epoch=`**；
- 活跃迭代会话中调用 `load_state_dict` → `RuntimeError`。

## 5. 行为变化（容易踩的点）

1. **每轮由外层驱动**：`for epoch in range(3): for b in loader:`。`BatchStream` 自己
   也会在轮末 `StopIteration`（Rust `at_eof` 信号），不会自旋挂死。
2. **Tier 1（HF/Lightning）必须经 `MegadataDataLoader` 迭代**（`for b in ds:` 不推进轮号）；
   `num_workers>0` 时主进程拿不到 dataset 进度（`state_dict()["dataset"] == {}`，
   会告警）——精确续训用引擎层或 `num_workers=0`。
3. **MixBatchStream 两模式**：
   - `steps_per_epoch=None`（自然）：被采样的子流耗尽即轮末；`len()` 是上界；
     **DDP 下必须显式 `steps_per_epoch`**（否则构造期跨 rank 校验不一致会报错）；
   - `steps_per_epoch=N`（确定性）：恰好 N 步，DDP 安全。
4. **取消**：`loader.cancel()` → `MegadataInterrupted`；`Ctrl+C` 同样。
   全局 `request_cancel()` / `is_cancel_requested()` / `raise_if_canceled()` 已移除；
   2026-09-27 起全局 `reset_cancel()` 与 Rust 侧 `CANCEL_FLAG` 一并退役
   （取消 = 每实例终态，重建实例即可继续）。
5. **`BatchStream` 不是 torch `Dataset`**：多 worker 只能走 `StreamingDataset + MegadataDataLoader`。
6. **无限流**：`while True:` 包 for；每个轮边界有一次 seek（预取队列会重新预热）。

## 6. 常用片段

```python
from megadata import BatchStream, MixBatchStream, RandomAccessBatchReader, DataBuilder

# 单源训练（自定义循环）
loader = BatchStream("data/index.json")          # batch_size/max_seq_len 从 manifest 自动读取
for epoch in range(3):
    for batch in loader:                          # 一次 for = 一轮
        step(batch)                               # batch: input_ids / cu_seqlens(int32 Tensor)
    ckpt = loader.state_dict()                    # 几十字节，schema 3

# 断点续训
loader2 = BatchStream("data/index.json")
loader2.load_state_dict(ckpt)

# 多源混采（确定性轮长，DDP 安全）
mix = MixBatchStream([{"path": "a/index.json", "weight": 0.9},
                 {"path": "b/index.json"}],        # 省略 weight → 按 manifest token 数配比
                steps_per_epoch=1000)
for batch in mix:
    step(batch)
    if batch.get("format") == "sft":              # 异构混采时携带来源格式
        ...

# 评估/抽查（随机访问，无 torch 协议）
reader = RandomAccessBatchReader("data/index.json")
print(len(reader), reader.format)
batch = reader.get_batch(10)

# 构建
builder = DataBuilder(tokenizer_path="tok", max_seq_len=4096)
builder.build_dataset(["raw.jsonl"], "out/", mode="pt", batch_size=8,
                      compression="auto")         # PT → zstd
```
