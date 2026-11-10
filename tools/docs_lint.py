#!/usr/bin/env python3
"""文档体检与索引生成（docs/ 结构守卫，仅标准库）。

用法:
    python tools/docs_lint.py check          # 链接 + 顶层白名单 + 索引同步（CI / 提交前）
    python tools/docs_lint.py index          # 重新生成 docs/README.md 的文档索引表
    python tools/docs_lint.py index --check  # 只校验索引是否最新（不写回）
    python tools/docs_lint.py self-test      # 代码引用正则的正/负向用例（CI 常驻回归）

约定:
    活跃区 = docs/guide/**, docs/README.md, 根 README.md, MIGRATION.md, python/**/*.md
    冻结区 = 无（对外发布版不保留过程档案）
    索引区间 = docs/README.md 中 <!-- DOCS-INDEX:BEGIN --> 与 <!-- DOCS-INDEX:END -->
    状态字段 = 文件前 15 行的 `状态：` / `**状态**：` / `- 状态：`，取到第一个「。」或行尾
    索引收录 = guide/*.md

退出码: check 有错误为 1，否则 0。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
INDEX_FILE = DOCS / "README.md"

BEGIN = "<!-- DOCS-INDEX:BEGIN 由 tools/docs_lint.py index 自动生成，勿手工编辑 -->"
END = "<!-- DOCS-INDEX:END -->"

LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
# 正文里的正则/代码片段（如 [^.]+[.]([^/]*)$）会被 LINK_RE 误判，按元字符过滤
REGEX_CHARS = set("*^$+[]{}|\\")

# 代码/配置中的仓库内文档路径（示例：docs/design/<主题>/xxx.md，占位符不参与自检）
# 前后对称 ASCII 类 + 惰性量词：拦 `xdocs/`/`.mdx`/`v3docs/` 粘连；放行 CJK 紧贴（`详见docs/…`）
DOC_REF_RE = re.compile(r"(?<![A-Za-z0-9_])docs/[\w\-./]+?\.md(?![A-Za-z0-9_])")
DOC_REF_CASES = (
    ("xdocs/foo.md", ()),  # docref: ignore
    ("docs/foo.mdx", ()),  # docref: ignore
    ("v3docs/foo.md", ()),  # docref: ignore
    ("见docs/foo.md说明", ("docs/foo.md",)),  # docref: ignore
    ("docs/a.md到docs/b.md", ("docs/a.md", "docs/b.md")),  # docref: ignore
    ("docs/design/文档治理/x.md。", ("docs/design/文档治理/x.md",)),  # docref: ignore
    ("`docs/design/多模态/handoff.md`", ("docs/design/多模态/handoff.md",)),  # docref: ignore
)
# 单行白名单：行内含此标记时不参与代码引用检查（示例路径/自检夹具用）
CODE_REF_IGNORE = "docref: ignore"
CODE_SCAN_DIRS = ("src", "python", "tests", "tools")
CODE_SCAN_EXTS = ("*.rs", "*.py", "*.yml", "*.yaml")
SKIP_PATH_PARTS = {".venv", "__pycache__", "node_modules", ".git", ".mypy_cache", ".pytest_cache"}

STATUS_PATTERNS = (
    re.compile(r"^\s*>\s*\*{0,2}状态\*{0,2}\s*[：:]\s*(.+)$"),
    re.compile(r"^\s*-\s*\*{0,2}状态\*{0,2}\s*[：:]\s*(.+)$"),
)

TOP_ALLOWED_FILES = {"README.md"}
TOP_ALLOWED_DIRS = {"guide"}

SKIP_LINK_PREFIXES = ("http://", "https://", "mailto:", "#", "file:", "ftp:")


def _md_files(*dirs: Path) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out.extend(sorted(d.rglob("*.md")))
    return out


def active_md_files() -> list[Path]:
    files = _md_files(DOCS / "guide")
    files += [
        INDEX_FILE,
        ROOT / "README.md",
        ROOT / "MIGRATION.md",
    ]
    files += _md_files(ROOT / "python")
    return [f for f in files if f.exists()]


def frozen_md_files() -> list[Path]:
    """冻结档案区（本仓库对外发布版不保留，函数保留以便结构扩展）。"""
    return []


def resolve_links(path: Path) -> tuple[int, list[tuple[str, str]]]:
    """返回 (链接总数, 断链列表[(target, 链接文本)])。"""
    text = path.read_text(encoding="utf-8")
    total = 0
    broken: list[tuple[str, str]] = []
    for m in LINK_RE.finditer(text):
        target = m.group(2).strip()
        if target.startswith(SKIP_LINK_PREFIXES):
            continue
        if any(ch in target for ch in REGEX_CHARS):
            continue  # 正则/占位符误报
        target = target.split("#")[0].strip()
        if not target:
            continue
        total += 1
        cand1 = (path.parent / target).resolve()
        cand2 = (ROOT / target).resolve()
        if not (cand1.exists() or cand2.exists()):
            broken.append((target, m.group(1)[:40]))
    return total, broken


def check_top_level() -> list[str]:
    bad: list[str] = []
    for entry in sorted(DOCS.iterdir()):
        if entry.is_dir() and entry.name not in TOP_ALLOWED_DIRS:
            bad.append(entry.name + "/")
        elif entry.is_file() and entry.name not in TOP_ALLOWED_FILES:
            bad.append(entry.name)
    return bad


def code_ref_files() -> list[Path]:
    """代码/配置中可能出现文档路径引用的文件集合。"""
    files: list[Path] = []
    for d in CODE_SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for ext in CODE_SCAN_EXTS:
            files.extend(sorted(base.rglob(ext)))
    gh = ROOT / ".github"
    if gh.exists():
        for ext in ("*.yml", "*.yaml"):
            files.extend(sorted(gh.rglob(ext)))
    return [f for f in files if not (set(f.parts) & SKIP_PATH_PARTS)]


def iter_code_refs(text: str) -> list[tuple[int, str]]:
    """返回文本中的 (行号, 文档路径)；跳过 `docref: ignore` 白名单行与冻结评审档案。"""
    refs: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if CODE_REF_IGNORE in line:
            continue  # 示例路径/自检夹具：单行白名单
        for m in DOC_REF_RE.finditer(line):
            target = m.group(0)
            if target.startswith("docs/reviews/"):
                continue
            refs.append((lineno, target))
    return refs


def check_code_refs() -> tuple[int, list[str]]:
    """扫描代码注释/字符串中的仓库内文档路径（docs/**.md）。

    返回 (引用总数, 错误列表)。docs/reviews/** 为冻结档案，允许保留历史引用。
    """
    total = 0
    errors: list[str] = []
    for f in code_ref_files():
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, target in iter_code_refs(text):
            total += 1
            if not (ROOT / target).exists():
                errors.append(f"{f.relative_to(ROOT)}:{lineno}: 文档路径不存在 -> {target}")
    return total, errors


def extract_status(path: Path) -> str | None:
    lines = path.read_text(encoding="utf-8").splitlines()[:15]
    for ln in lines:
        for pat in STATUS_PATTERNS:
            m = pat.match(ln)
            if not m:
                continue
            s = m.group(1).replace("**", "").strip()
            s = re.split(r"[。\n]", s)[0].strip()
            if len(s) > 50:
                s = s[:49] + "…"
            return s or "（空）"
    return None


def extract_title(path: Path) -> str:
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("# "):
            return ln[2:].strip()
    return path.stem


PREFIX_RANK = (("guide/", 0),)


def _sort_key(rel: str) -> tuple[int, str]:
    for prefix, rank in PREFIX_RANK:
        if rel == prefix or rel.startswith(prefix):
            return rank, rel
    return 8, rel


def index_entries() -> list[tuple[str, str | None, str]]:
    files = _md_files(DOCS / "guide")
    entries: list[tuple[str, str | None, str]] = []
    for f in files:
        if not f.exists():
            continue
        entries.append((f.relative_to(DOCS).as_posix(), extract_status(f), extract_title(f)))
    entries.sort(key=lambda e: _sort_key(e[0]))
    return entries


def render_index() -> str:
    rows = ["| 文档 | 状态 | 一句话定位（取文档标题） |", "|---|---|---|"]
    for rel, status, title in index_entries():
        rows.append(f"| `{rel}` | {status or '⚠ 缺状态字段'} | {title} |")
    return "\n".join(rows)


def replace_index(content: str, table: str) -> str:
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pattern.search(content):
        raise SystemExit(f"docs/README.md 缺少索引标记区间：\n{BEGIN}\n...\n{END}")
    return pattern.sub(f"{BEGIN}\n{table}\n{END}", content)


def cmd_index(write: bool) -> int:
    content = INDEX_FILE.read_text(encoding="utf-8")
    expected = replace_index(content, render_index())
    if write:
        INDEX_FILE.write_text(expected, encoding="utf-8")
        print(f"已更新 {INDEX_FILE.relative_to(ROOT)}（{len(index_entries())} 篇）")
        return 0
    if expected != content:
        print("ERROR docs/README.md 文档索引已过期：运行 python tools/docs_lint.py index")
        return 1
    print("文档索引已是最新")
    return 0


def cmd_check() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    total = 0
    for f in active_md_files():
        n, broken = resolve_links(f)
        total += n
        for target, text in broken:
            errors.append(f"{f.relative_to(ROOT)}: 断链 -> {target}（{text}）")

    frozen_links = frozen_total = 0
    for f in frozen_md_files():
        n, broken = resolve_links(f)
        frozen_total += n
        frozen_links += len(broken)

    for name in check_top_level():
        errors.append(f"docs/ 顶层出现非白名单项：{name}（移动文件或更新白名单）")

    code_refs, code_errors = check_code_refs()
    errors.extend(code_errors)

    for rel, status, _ in index_entries():
        if not status:
            warnings.append(f"{rel}: 缺状态字段（头部加 `> **状态**：…。`）")

    content = INDEX_FILE.read_text(encoding="utf-8")
    if replace_index(content, render_index()) != content:
        errors.append("docs/README.md 文档索引已过期：运行 python tools/docs_lint.py index")

    print(f"活跃区：{len(active_md_files())} 篇，链接 {total} 条")
    print(f"冻结区：{len(frozen_md_files())} 篇，链接 {frozen_total} 条，断链 {frozen_links} 条（冻结档案允许，仅提示）")
    print(f"代码内文档引用：{code_refs} 条（src/python/tests/tools/.github）")
    for w in warnings:
        print("WARN", w)
    for e in errors:
        print("ERROR", e)
    print(f"结果：{len(errors)} error / {len(warnings)} warning")
    return 1 if errors else 0


def cmd_self_test() -> int:
    """代码引用正则/白名单的正负向用例（不读仓库文件）。"""
    failures = 0
    for text, expected in DOC_REF_CASES:
        got = tuple(m.group(0) for m in DOC_REF_RE.finditer(text))
        if got != expected:
            failures += 1
            print(f"FAIL {text!r}: 期望 {expected}，实际 {got}")

    extras = (
        ("白名单行跳过（docref: ignore）", "x = 'docs/missing.md'  # docref: ignore\n", ()),
        ("非白名单行命中", "见 docs/a.md\n", ((1, "docs/a.md"),)),  # docref: ignore
        ("冻结评审路径豁免", "docs/reviews/code/x.md\n", ()),
    )
    for name, text, expected in extras:
        got = tuple(iter_code_refs(text))
        if got != expected:
            failures += 1
            print(f"FAIL {name}: 期望 {expected}，实际 {got}")

    total = len(DOC_REF_CASES) + len(extras)
    if failures:
        print(f"self-test：{failures} 项失败")
        return 1
    print(f"self-test：{total} 项全部通过")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["check"]:
        return cmd_check()
    if args[:1] == ["index"]:
        return cmd_index(write="--check" not in args)
    if args[:1] == ["self-test"]:
        return cmd_self_test()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
