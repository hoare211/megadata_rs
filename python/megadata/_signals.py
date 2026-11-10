# python/megadata/_signals.py
"""中断异常 + 每实例取消的活跃实例注册表 + 惰性链式信号安装。

F-36：从 `dataloader.py` 抽出——`builder.py` 只需要 `_register`，
而 `dataloader.py` 顶层强依赖 torch/pyarrow；抽出后 `import megadata` 的
构建路径（DataBuilder）不再被拖入 torch 依赖链。

设计要点：
1. 不在 import 时安装信号 handler（避免副作用）；首个 megadata 实例构造时惰性安装。
2. 链式回调：保存前一个 handler 并在触发时调用，保证框架自身的信号处理不受影响。
3. env `MEGADATA_DISABLE_SIGNAL_HANDLER=1` 可彻底关闭（让框架托管信号）。
"""
import os
import signal
import threading
import warnings
import weakref
from typing import Any, Dict


class MegadataInterrupted(KeyboardInterrupt):
    """megadata 流被取消时抛出（KeyboardInterrupt 子类，可用 except KeyboardInterrupt 捕获）。"""


_ACTIVE_INSTANCES: "weakref.WeakSet[Any]" = weakref.WeakSet()
_SIGNAL_INSTALLED = False
_SIGNAL_LOCK = threading.Lock()
_ORIGINAL_HANDLERS: Dict[int, Any] = {}


def _cancel_all_active() -> None:
    for inst in list(_ACTIVE_INSTANCES):
        try:
            inst._cancel()
        except Exception:
            pass


def _megadata_signal_handler(signum, frame):
    _cancel_all_active()
    prev = _ORIGINAL_HANDLERS.get(signum)
    if callable(prev):
        prev(signum, frame)
        return
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    # SIGTERM 等：恢复默认行为并重新触发，保证进程仍可被终止
    # （只对 SIGINT 抛 KeyboardInterrupt、其余"什么都不做"等于把 SIGTERM 静默吞掉，
    #   `timeout` / k8s 的 pod 终止都杀不掉进程，只能 SIGKILL）
    try:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    except Exception:
        raise SystemExit(128 + int(signum))


def install_signal_handlers() -> None:
    """惰性、幂等地安装 Ctrl+C handler。首个 megadata 实例构造时自动调用。"""
    global _SIGNAL_INSTALLED
    if os.environ.get("MEGADATA_DISABLE_SIGNAL_HANDLER", "") in ("1", "true", "True"):
        return
    with _SIGNAL_LOCK:
        if _SIGNAL_INSTALLED:
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                _ORIGINAL_HANDLERS[sig] = signal.signal(sig, _megadata_signal_handler)
            except (ValueError, OSError):
                pass
        _SIGNAL_INSTALLED = True


def uninstall_signal_handlers() -> None:
    """恢复原始 signal handler。

    F-28：逐信号检查**当前 handler 是否仍是 megadata 的**——若不是（说明安装之后
    框架又装了自己的 handler），跳过恢复并告警，避免把框架的 handler 冲掉。
    """
    global _SIGNAL_INSTALLED
    with _SIGNAL_LOCK:
        for sig, handler in _ORIGINAL_HANDLERS.items():
            try:
                current = signal.getsignal(sig)
            except (ValueError, OSError):
                current = None
            if current is not _megadata_signal_handler:
                warnings.warn(
                    f"the current handler for signal {sig} is no longer the one installed by megadata "
                    "(installed later by a framework?): skipping restore to avoid clobbering it; "
                    "to fully hand back control, call this after the framework uninstalls its handler.",
                    stacklevel=2,
                )
                continue
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass
        _ORIGINAL_HANDLERS.clear()
        _SIGNAL_INSTALLED = False
        # 旧版在此调用全局 `reset_cancel()`（Rust CANCEL_FLAG 已于 2026-09-27
        # 评审清理中退役——每实例令牌才是唯一取消入口，该调用本就是 no-op）


def _register(inst: Any) -> None:
    """注册一个可取消实例（暴露 _cancel() 方法）。"""
    install_signal_handlers()
    _ACTIVE_INSTANCES.add(inst)
