"""pytest 配置：把 tests/ 放进 sys.path，使共享工具 `_scenario_kit` 可被导入。

pytest 默认（importmode=prepend）也会插入该目录，但显式声明后不依赖 import-mode
与 rootdir 的推断，`pytest tests/xxx.py` / `pytest /abs/path/xxx.py`
（从任意 cwd 调用）都能工作。

迷你分词器兜底：bbpe 分词器模型是 `bbpe_tokenizer` 的训练产物、不入库——全新克隆
环境下 mixer / resumption / cloud_io / s3_io / memory_leak 等核心域的测试会因
TOKENIZER_PATH 缺失整片 pytest.skip，覆盖率直接归零。这里在收集期**现训一个
几 KB 的迷你分词器**并注入 TOKENIZER_PATH，把"skip"降级为"全量执行"；用户通过
`BBPE_TOKENIZER_DIR` 指向自己的分词器目录时优先使用，训练失败时保留原 skip 行为（不硬失败）。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_TOKENIZER = os.environ.get("BBPE_TOKENIZER_DIR") or os.path.join(_REPO_ROOT, "tokenizer")

_MINI_TOKENIZER_SENTINELS = ["<|im_start|>", "<|im_end|>", "<pad>"]


def _train_mini_tokenizer():
    """用 bbpe_tokenizer 现训一个迷你分词器，返回其目录；失败返回 None。

    产物缓存于系统临时目录（按语料指纹命名）：语料不变时跨 pytest 会话复用，
    训练一次约 1 秒。
    """
    try:
        import bbpe_tokenizer
    except ImportError:
        return None

    import hashlib
    import tempfile

    corpus = "".join(
        f"mini tokenizer corpus line {i} the quick brown fox jumps over "
        f"中文测试语料内容 ab dc {i % 97} 😀\n"
        for i in range(300)
    )
    fingerprint = hashlib.sha1(corpus.encode("utf-8")).hexdigest()[:12]
    out_dir = os.path.join(tempfile.gettempdir(), f"megadata_mini_tokenizer_{fingerprint}")
    model_file = os.path.join(out_dir, "tokenizer.json")
    if os.path.exists(model_file):
        return out_dir

    try:
        os.makedirs(out_dir, exist_ok=True)
        corpus_file = os.path.join(out_dir, "corpus.txt")
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write(corpus)
        # vocab 600：ASCII 可见字符 + 常用中文 + 哨兵后留少量 merge 空间；
        # min_frequency=1 保证小语料也产出足够 merge。
        cfg = (
            bbpe_tokenizer.TrainerConfig(600)
            .with_pattern(bbpe_tokenizer.PatternFlavor.gpt4())
            .with_min_frequency(1)
            .with_special_tokens(_MINI_TOKENIZER_SENTINELS)
        )
        trainer = bbpe_tokenizer.BBPETrainer(cfg, None)
        trainer.train(bbpe_tokenizer.TextFileReader(corpus_file))
        trainer.save(model_file)
        return out_dir if os.path.exists(model_file) else None
    except Exception:  # noqa: BLE001 - 训练失败不阻塞测试收集，回退到既有 skip 行为
        return None


def _ensure_tokenizer_path() -> None:
    """TOKENIZER_PATH 未显式提供且仓库自带分词器缺失时，注入迷你分词器。"""
    explicit = os.environ.get("TOKENIZER_PATH")
    if explicit and os.path.exists(explicit):
        return
    if not explicit and os.path.isdir(_DEFAULT_TOKENIZER):
        # 与各测试文件的默认解析一致：让"仓库自带分词器"继续生效
        os.environ.setdefault("TOKENIZER_PATH", os.path.abspath(_DEFAULT_TOKENIZER))
        return
    mini = _train_mini_tokenizer()
    if mini:
        os.environ["TOKENIZER_PATH"] = mini
        # 标记：部分用例（如 e2e 对齐）依赖真实 ChatML 分词器的 token id 语义，
        # 迷你兜底分词器下无法成立，据此显式 skip 而不是给出误导性的失败。
        os.environ.setdefault("MEGADATA_MINI_TOKENIZER", "1")


_ensure_tokenizer_path()


def pytest_configure(config):
    """注册自定义 marker（避免 unknown-marker 告警；CI 可 `-m "not slow"` 跳过）。"""
    config.addinivalue_line("markers", "slow: 吞吐/基线类耗时测试，可在快速 CI 跳过")
    config.addinivalue_line(
        "markers",
        "real_tokenizer: 依赖真实 ChatML 分词器的 token id/特殊符号语义；"
        "仅有迷你兜底分词器时自动 skip",
    )


def pytest_collection_modifyitems(config, items):
    """迷你兜底分词器下，跳过依赖真实分词器语义的用例（避免误导性失败）。"""
    if os.environ.get("MEGADATA_MINI_TOKENIZER") != "1":
        return
    skip = pytest.mark.skip(
        reason="当前使用 conftest 现训的迷你分词器兜底；本用例依赖真实 ChatML 分词器语义，"
        "请设置 BBPE_TOKENIZER_DIR 指向已训练的分词器目录"
    )
    for item in items:
        if item.get_closest_marker("real_tokenizer"):
            item.add_marker(skip)
