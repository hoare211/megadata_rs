"""全场景测试共用工具（文件名以 `_` 开头，pytest 不收集为测试模块）。

统一"用仓库自带的 bbpe 分词器构建小数据集"的姿势，避免每个场景各写一套。

F-34：本文件是**被 pytest 用例 import 的共享工具**（`test_project_journey` /
`test_io_cache_env_scenarios` / `test_reader_scenarios` 均依赖），故保留在 `tests/`；
其余 `_diag*` / `_prerun*` / `_probe*` / `_test_*` 一次性诊断脚本已移至 `tools/`。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Set

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TOKENIZER = os.environ.get("BBPE_TOKENIZER_DIR") or os.path.join(REPO_ROOT, "tokenizer")

MAX_SEQ_LEN = 64
BATCH_SIZE = 4

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
)


def tokenizer_path() -> str:
    """仓库自带分词器目录；缺失则跳过（与既有测试口径一致）。"""
    path = os.path.abspath(os.environ.get("TOKENIZER_PATH") or DEFAULT_TOKENIZER)
    if not os.path.exists(path):
        pytest.skip(f"未找到分词器：{path}（可用 TOKENIZER_PATH 覆盖）")
    return path


def write_jsonl(path: str, records: Sequence[Any]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec if isinstance(rec, str) else json.dumps(rec, ensure_ascii=False))
            f.write("\n")
    return path


def pt_records(n: int, *, tag: str = "pt", offset: int = 0) -> List[Dict[str, str]]:
    """唯一性可验证的 PT 语料（每条文本含唯一编号）。"""
    return [{"text": f"{tag}-doc-{offset + i} 中文语料内容 abc {offset + i} 😀"} for i in range(n)]


def sft_records(n: int, *, tag: str = "sft", offset: int = 0) -> List[Dict[str, Any]]:
    """唯一性可验证的 SFT 语料。"""
    return [
        {"messages": [
            {"role": "user", "content": f"{tag}-问-{offset + i}"},
            {"role": "assistant", "content": f"{tag}-答-{offset + i}"},
        ]}
        for i in range(n)
    ]


def build_dataset(
    out_dir: str,
    records: Sequence[Any],
    mode: str,
    *,
    max_seq_len: int = MAX_SEQ_LEN,
    batch_size: int = BATCH_SIZE,
    batches_per_shard: int = 4,
    compression: str = "auto",
    template: str = CHAT_TEMPLATE,
    progress_cb=None,
    extra_columns: bool = False,
    **kwargs: Any,
) -> str:
    """构建数据集，返回 `index.json` 路径。"""
    from megadata import DataBuilder

    os.makedirs(out_dir, exist_ok=True)
    raw = write_jsonl(os.path.join(out_dir, "raw.jsonl"), records)
    ds = os.path.join(out_dir, "ds")
    builder = DataBuilder(tokenizer_path=tokenizer_path(), max_seq_len=max_seq_len)
    builder.build_dataset(
        input_paths=[raw], output_dir=ds, mode=mode, template_str=template,
        batch_size=batch_size, pad_id=0, batches_per_shard=batches_per_shard,
        compression=compression, progress_cb=progress_cb,
        extra_columns=extra_columns, **kwargs,
    )
    return os.path.join(ds, "index.json")


def read_batches(index_path: str, *, batch_size: int = BATCH_SIZE,
                 max_seq_len: int = MAX_SEQ_LEN, **kwargs: Any) -> List[Dict[str, Any]]:
    """一次性读完一轮（world_size=1 / shuffle=False，确定性顺序）。"""
    from megadata import BatchStream

    loader = BatchStream(index_json_path=index_path, batch_size=batch_size,
                          max_seq_len=max_seq_len, world_size=1, rank=0,
                          shuffle=False, **kwargs)
    return [dict(b) for b in loader]


def batch_key(batch: Dict[str, Any]) -> bytes:
    """batch 内容指纹：用于"无重叠 / 无遗漏"这类集合判定。

    数据本身带唯一编号，因此相同 fingerprint ⟺ 同一批数据。
    """
    h = hashlib.sha1()
    for name in ("input_ids", "labels", "cu_seqlens"):
        if name in batch:
            t = batch[name]
            h.update(name.encode())
            h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.digest()


def batch_keys(batches: Iterable[Dict[str, Any]]) -> List[bytes]:
    return [batch_key(b) for b in batches]


def load_index(index_path: str) -> Dict[str, Any]:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def total_batches(index_path: str) -> int:
    return int(load_index(index_path)["global_stats"]["total_batches"])


def unique_keys(keys: Iterable[bytes]) -> Set[bytes]:
    return set(keys)
