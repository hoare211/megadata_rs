# -*- coding: utf-8 -*-
"""20261007 代码评审修复的 Python 侧回归测试。

本文件只覆盖**不需要 TOKENIZER_PATH / 真实数据集**的轻量不变量，因此可以在
只装了编译产物（.so）、没有分词器/数据的 CI 环境里跑起来。这正面回应了评审
§5.3 指出的测试环境缺陷：最关键的不变量不应被一个环境变量门控掉。

覆盖的评审项：
- §3.1（P0）预取线程必须包 catch_unwind —— 用 Rust 源码结构断言。
- §3.5``validate`` 把 PT 长文档切分 / SFT 截断作为告警项报出。
- §3.8(d)``simulation.py`` 的 ``predownload`` 默认值与 Rust ``DEFAULT_PREDOWNLOAD``
  必须一致（防止两份事实源漂移）。
- §3.9(a)``MixBatchStream.epoch`` 的 except 收窄 + 降级留痕。
- §3.9(b)``BatchStream`` / ``MixBatchStream`` 的会话标志在迭代器被丢弃后仍释放。
- §3.9(c)``MixBatchStream`` 的 wrap 无进展守卫存在且触发时抛错。

实现说明：涉及"源码结构"的断言一律用 **AST 解析**而非字符串包含——评审 §5.2
指出字符串匹配类断言容易被注释/文档字符串里的同名词绕过（本文件开发过程中就
真实踩到过：注释里写"原为裸 except Exception"导致 `except Exception not in src`
误判失败）。
"""
from __future__ import annotations

import ast
import json
import pathlib

import pytest

from megadata import simulation

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE_PY = REPO_ROOT / "python" / "megadata" / "engine.py"
STREAM_RS = REPO_ROOT / "src" / "runtime" / "stream.rs"


# ===========================================================================
# AST 辅助（结构断言，避免字符串匹配的脆弱性）
# ===========================================================================
def _func_node(path: pathlib.Path, dotted: str):
    """按 `Class.method` / `func` 取出模块级 AST 节点（只解析，不执行）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = tree
    for part in dotted.split("."):
        found = None
        for child in ast.iter_child_nodes(node):
            if getattr(child, "name", None) == part:
                found = child
                break
        assert found is not None, f"在 {path} 中找不到 {part!r}（查找 {dotted}）"
        node = found
    return node


def _assigned_names(node) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def _except_type_names(node) -> list[str]:
    names: list[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.ExceptHandler) and n.type is not None:
            if isinstance(n.type, ast.Tuple):
                names.extend(e.id for e in n.type.elts if isinstance(e, ast.Name))
            elif isinstance(n.type, ast.Name):
                names.append(n.type.id)
    return names


def _calls_named(node, attr: str) -> list[ast.Call]:
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]


def _raises(node, exc_name: str) -> bool:
    for n in ast.walk(node):
        if (
            isinstance(n, ast.Raise)
            and isinstance(n.exc, ast.Call)
            and isinstance(n.exc.func, ast.Name)
            and n.exc.func.id == exc_name
        ):
            return True
    return False


# ===========================================================================
# §3.1（P0）预取线程 panic 不得被降级成 EOF
# ===========================================================================
def test_prefetch_thread_catches_unwind() -> None:
    """预取线程体必须包 ``catch_unwind``。

    没有它，线程体任何 panic 都会让 ``fatal_error`` 保持 None，
    ``exit_reason()`` 按 Eof 处理，绑定层转成 ``("exhausted", ...)``，
    **训练数据被静默截断**（20261007 评审 §3.1，P0）。
    """
    if not STREAM_RS.exists():
        pytest.skip("Rust 源码不可用（仅安装了编译产物）")
    text = STREAM_RS.read_text(encoding="utf-8")
    assert "catch_unwind" in text, "预取线程体必须包 catch_unwind，否则 panic 会静默降级为 EOF"
    assert "record_fatal_msg" in text, "panic 分支必须写 fatal 槽位"


def test_record_fatal_msg_is_poisoning_safe() -> None:
    """``record_fatal_msg`` 是 panic 兜底路径的最后一道防线，内部不得 `.lock().unwrap()`。

    否则形成"panic 套 panic"：锁中毒时兜底函数自己再 panic，连降级成EOF 的
    机会都没有（Python 侧会看到 PanicException 而非数据被截断，更难定位）。
    """
    if not STREAM_RS.exists():
        pytest.skip("Rust 源码不可用（仅安装了编译产物）")
    text = STREAM_RS.read_text(encoding="utf-8")
    idx = text.find("fn record_fatal_msg")
    assert idx > 0, "应存在 record_fatal_msg（panic 兜底路径）"
    # 取函数体到下一个顶层 `fn` 之前
    body = text[idx: text.find("\nfn ", idx + 10) if text.find("\nfn ", idx + 10) > 0 else len(text)]
    assert ".lock().unwrap()" not in body, (
        "record_fatal_msg 内不得有 .lock().unwrap()（锁中毒会二次 panic）"
    )
    assert "if let Ok(mut guard) = slot.lock()" in body, (
        "record_fatal_msg 应显式处理锁毒化（if let Ok(mut guard) = slot.lock()）"
    )


def test_local_verify_has_no_lock_unwrap() -> None:
    """``verify_local_file`` 里的 ``local_verified.lock().unwrap()`` 必须消除。

    该函数经 CloudIoManager 随 SyncInnerReader move 进预取线程，是 P0 失效链上
    一个**已确认的真实触发源**（Mutex 中毒后每次本地校验都 panic）。
    """
    io_rs = REPO_ROOT / "src" / "io.rs"
    if not io_rs.exists():
        pytest.skip("Rust 源码不可用")
    text = io_rs.read_text(encoding="utf-8")
    idx = text.find("fn verify_local_file")
    assert idx > 0
    end = text.find("\n    fn ", idx + 10)
    body = text[idx: end if end > 0 else len(text)]
    assert "local_verified.lock().unwrap()" not in body, (
        "verify_local_file 内的 local_verified.lock().unwrap() 必须改为毒化安全"
    )


# ===========================================================================
# §3.5 validate 报出 PT 长文档切分 / SFT 截断
# ===========================================================================
def _minimal_manifest(**stats) -> dict:
    """构造一个能通过前几项检查的最小 manifest（版本/格式取真实枚举值）。"""
    base = {
        "version": "2.0",
        "format": "Memmap",
        "max_seq_len": 2048,
        "global_stats": {
            "total_tokens": 1000000,
            "batch_size": 8,
            "total_batches": 10,
            "total_shards": 1,
        },
        "schema": {},
        "shards": [
            {
                "shard_id": 0,
                "filename": "shard_00000.bin",
                "num_batches": 10,
                "size_bytes": 1,
                "hashes": {},
            }
        ],
    }
    base["global_stats"].update(stats)
    return base


def _write(tmp_path, stats) -> str:
    p = tmp_path / "index.json"
    p.write_text(json.dumps(_minimal_manifest(**stats)), encoding="utf-8")
    return str(p)


def _checks_text(report: dict) -> str:
    return " | ".join(f"{c['check']}: {c['detail']}" for c in report["checks"])


def test_validate_reports_pt_split_docs(tmp_path) -> None:
    """``split_docs > 0`` 时 validate 必须报出告警并解释后果。

    PT 长文档被跨 batch 切分**不丢数据**，但 varlen 训练下每段都是 position 从0
    起的独立序列 → 跨 batch 上下文丢失。属"必须在训练前看到"的类别。
    """
    from megadata.validate import validate

    report = validate(_write(tmp_path, {"split_docs": 5, "split_tokens": 50000}))
    joined = _checks_text(report).lower()
    assert "single batch" in joined, f"应出现 PT 切分告警，实际: {joined}"
    assert "lost" in joined, "告警必须说明后果（varlen 下上下文丢失）"
    assert report["ok"] is False, "存在切分告警时整体结论不应是 OK"


def test_validate_clean_dataset_has_no_data_loss_warnings(tmp_path) -> None:
    """干净数据集不应产生切分/截断告警（避免噪声淹没真问题）。"""
    from megadata.validate import validate

    report = validate(_write(tmp_path, {}))
    names = [c["check"] for c in report["checks"]]
    assert not any("single batch" in n for n in names), f"干净数据集不应报切分告警: {names}"
    assert not any("truncat" in n for n in names), f"干净数据集不应报截断告警: {names}"


def test_validate_reports_sft_truncation(tmp_path) -> None:
    """SFT 截断是真丢数据，也必须报出，且文案要与 PT 切分明确区分。"""
    from megadata.validate import validate

    report = validate(_write(tmp_path, {"truncated_docs": 2, "truncated_tokens": 999}))
    joined = _checks_text(report)
    assert "truncat" in joined.lower(), f"应报出 SFT 截断，实际: {joined}"
    assert "dropped" in joined.lower(), "截断告警必须说明 token 被丢弃"


# ===========================================================================
# §3.8(d) simulation.predownload 与 Rust 常量一致性
# ===========================================================================
def test_predownload_default_matches_rust() -> None:
    """仿真器的预取窗口默认值必须来自Rust 侧常量，而非独立硬编码。

    两份各自维护时，改 Rust 的 ``DEFAULT_PREDOWNLOAD`` 不会同步仿真器 →
    容量规划报告给出与真实引擎不符的结论（"16 够用"，实际不是）。
    """
    try:
        from megadata._rust import DEFAULT_PREDOWNLOAD as RUST_VALUE
    except ImportError:
        pytest.skip("扩展未重建（旧 .so 无该导出），无法校验一致性")

    assert simulation.DEFAULT_PREDOWNLOAD == RUST_VALUE, (
        f"simulation.DEFAULT_PREDOWNLOAD={simulation.DEFAULT_PREDOWNLOAD} "
        f"!= Rust DEFAULT_PREDOWNLOAD={RUST_VALUE}；两者必须共用同一事实源"
    )
    assert simulation.SimulationConfig("index.json", world_size=1).predownload == RUST_VALUE


def test_simulation_config_accepts_zero_predownload() -> None:
    cfg = simulation.SimulationConfig("index.json", world_size=1, predownload=0)
    assert cfg.predownload == 0  # 0 = 不预取（退化为按需拉取），合法


def test_simulation_module_has_no_bare_exception() -> None:
    """simulation.py 不应有裸 except Exception（与 §3.9(a) 同款收窄精神）。"""
    tree = ast.parse((REPO_ROOT / "python" / "megadata" / "simulation.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            assert not (
                node.type is None
                or (isinstance(node.type, ast.Name) and node.type.id == "Exception")
            ), f"simulation.py:{node.lineno} 存在裸 except Exception"


# ===========================================================================
# §3.9(a) epoch 属性异常收窄
# ===========================================================================
def test_epoch_property_narrows_broad_except_and_warns() -> None:
    """``.epoch`` 不得再用裸 ``except Exception``；降级路径必须发 RuntimeWarning。

    真正的编程错误（AttributeError / NameError 等）不再被静默吞掉；
    可容忍的引擎异常仍降级返回，但必须留下痕迹。
    """
    prop = _func_node(ENGINE_PY, "MixBatchStream.epoch")

    names = _except_type_names(prop)
    assert "Exception" not in names, f"except 子句里不得含宽泛的 Exception: {names}"
    for expected in ("RuntimeError", "OSError"):
        assert expected in names, f"应显式捕获 {expected}，实际: {names}"

    warns = _calls_named(prop, "warn")
    assert warns, "降级路径必须调用 warnings.warn（不能完全静默）"
    assert any(
        len(c.args) >= 2
        and isinstance(c.args[1], ast.Name)
        and c.args[1].id == "RuntimeWarning"
        for c in warns
    ), "必须用 RuntimeWarning 标记这是可容忍的降级"


# ===========================================================================
# §3.9(b) 会话标志在迭代器被丢弃后仍能释放
# ===========================================================================
def test_session_iterator_releases_on_del() -> None:
    """``_SessionIterator`` 必须在被 GC 时释放会话标志。

    裸生成器解决不了"从未启动即丢弃"：CPython 对未启动生成器调 ``close()``
    不会执行函数体，``finally`` 不跑 → 实例被永久卡在活动会话。
    """
    from megadata.engine import _SessionIterator

    released = []

    def _gen():
        yield 1
        yield 2

    it = _SessionIterator(_gen(), lambda: released.append(True))
    assert not released, "刚创建时不应已释放"
    del it  # 触发 __del__
    assert released == [True], "__del__ 必须释放会话标志，否则实例被永久卡死"


def test_session_iterator_releases_on_stopiteration_and_explicit_close() -> None:
    from megadata.engine import _SessionIterator

    released = []
    it = _SessionIterator(iter([1]), lambda: released.append(True))
    assert next(it) == 1
    with pytest.raises(StopIteration):
        next(it)
    assert released, "StopIteration 路径必须释放会话"

    released2 = []
    it2 = _SessionIterator(_gen_two(), lambda: released2.append(True))
    it2.close()
    assert released2 == [True], "显式 close 必须释放会话"
    it2.close()  # 幂等
    assert released2 == [True], "重复 close 必须幂等"


def _gen_two():
    yield 1
    yield 2


def test_session_iterator_is_iterable_and_context_manager() -> None:
    from megadata.engine import _SessionIterator

    released = []
    it = _SessionIterator(iter([1, 2]), lambda: released.append(True))
    assert iter(it) is it, "必须是可迭代对象（否则 for x in stream 直接失败）"
    with it as handle:
        assert list(handle) == [1, 2]
    assert released, "with 退出时必须释放会话"


def test_batchstream_uses_session_iterator() -> None:
    """``BatchStream.__iter__`` 必须返回 ``_SessionIterator``，而不是裸生成器。"""
    fn = _func_node(ENGINE_PY, "BatchStream.__iter__")
    returned = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
    ]
    assert any(
        isinstance(c.func, ast.Name) and c.func.id == "_SessionIterator"
        for r in returned
        for c in [r.value]
    ), "BatchStream.__iter__ 应返回 _SessionIterator(...)，否则会话标志会卡死"


def test_mixstream_uses_session_iterator() -> None:
    fn = _func_node(ENGINE_PY, "MixBatchStream.__iter__")
    returned = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
    ]
    assert any(
        isinstance(r.value.func, ast.Name) and r.value.func.id == "_SessionIterator"
        for r in returned
    ), "MixBatchStream.__iter__ 应返回 _SessionIterator(...)"


def test_batchstream_seek_happens_in_dunder_iter() -> None:
    """seek 必须在 ``__iter__`` 内完成（不等第一次 ``__next__``）。

    否则生成器从未启动时游标没动，而 ``_session_start`` / ``_produced`` 已被改写，
    状态与实际游标不一致。
    """
    fn = _func_node(ENGINE_PY, "BatchStream.__iter__")
    calls = {
        n.func.attr
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert "seek_to_epoch_batch" in calls, "seek 必须在 __iter__ 内立即发生"


# ===========================================================================
# §3.9(c) wrap 无进展守卫
# ===========================================================================
def test_mixstream_iterate_has_no_progress_guard() -> None:
    """确定性模式的 wrap 循环必须有"无进展"守卫，否则空子流会忙等。

    失效形态：某子流本轮0 个 batch → ``exhausted → wrap → 立即又 exhausted → …``
    形成永不产出的单核忙循环，训练看起来就是"卡住"。
    """
    it = _func_node(ENGINE_PY, "MixBatchStream._iterate")
    assigned = _assigned_names(it)
    assert "no_progress_wraps" in assigned, "缺少 wrap 无进展守卫"
    assert "max_no_progress_wraps" in assigned, "守卫必须有一个明确的阈值"
    assert _raises(it, "RuntimeError"), "无进展超阈值时必须抛 RuntimeError（否则变成忙等）"