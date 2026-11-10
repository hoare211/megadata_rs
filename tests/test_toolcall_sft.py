"""多轮对话 + tool-call 场景的 SFT 数据正确性测试（2026-09-30 验证项 3）。

覆盖场景（数据全部为本测试确定性构造）：
- 多轮 user/assistant 交替：所有 assistant 轮监督、user 轮不监督；
- role="tool" 轮（函数结果回传）：**不监督**——labels 只进 assistant 段；
- assistant 空 content（tool_calls 结构里 assistant 常只带 tool_calls 字段）：
  轮标记本身仍被监督 → 该文档 labels 非全 -100（不产生无声废样本）；
- assistant 的 tool-call JSON 内容：属于 assistant 段 → 被监督；
- 顶层 tools 字段经模板渲染进 system 段：内容可见（进 input_ids）但不监督；
- 无漏无重：cu_seqlens 还原的文档数 == 输入行数，文档序列 == 独立复算；
- next-token 不变式 / row_lens+padding 契约 / position_ids 文档内相对位置。

真值策略：复用 `test_tokenizer_integration` 的"同一分词器双文件形态"——
bbpe 模型走默认构建路径（被测），官方 `tokenizers.Tokenizer` 独立实例复算
ids/tokens（真值），两条路径不共享分词实现。
"""
from __future__ import annotations

import json
import os

import pytest

import test_tokenizer_integration as T  # noqa: F401  (模块级 importorskip("tokenizers"))

from tokenizers import Tokenizer  # noqa: E402

from megadata import BatchStream, DataBuilder  # noqa: E402
from megadata.builder import BuildConfig, Marker  # noqa: E402

MAX_SEQ_LEN = 128
BATCH_SIZE = 4

# 泛化 ChatML：任意 role 透传（tool / system 也按 <|im_start|>role 渲染）
TOOL_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
)
# 带顶层 tools 字段的变体：工具定义渲染进 system 段（不进 messages）
TOOLSYS_TEMPLATE = (
    "{% if tools %}<|im_start|>system\n{{ tools | length }} tools available<|im_end|>\n{% endif %}"
    + TOOL_TEMPLATE
)


@pytest.fixture(scope="module")
def tok_pair(tmp_path_factory):
    """`(bbpe_model_path, official_tokenizer, im_id, em_id)`：同一分词器的两种形态。"""
    d = tmp_path_factory.mktemp("toolcall_tok")
    hf = T.build_hf_tokenizer()
    bbpe_path = T.write_bbpe_model(hf, str(d / "model.bbpe"))
    hf_path = T.write_hf_tokenizer(hf, str(d / "tokenizer.json"))
    official = Tokenizer.from_file(hf_path)
    return (bbpe_path, official,
            official.token_to_id("<|im_start|>"), official.token_to_id("<|im_end|>"))


def _records():
    """确定性 tool-call 对话集（下标即用例编号，见模块 docstring）。"""
    return [
        # 0 普通 user/assistant
        {"messages": [{"role": "user", "content": "q0"},
                       {"role": "assistant", "content": "a0"}]},
        # 1 三轮 user/assistant 交替
        {"messages": [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q1b"}, {"role": "assistant", "content": "a1b"},
            {"role": "user", "content": "q1c"}, {"role": "assistant", "content": "a1c"},
        ]},
        # 2 tool-call 链：assistant 发起（JSON 内容）→ tool 回传 → assistant 总结
        {"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant",
             "content": '{"name": "get_weather", "arguments": {"city": "Hangzhou"}}'},
            {"role": "tool", "content": "TOOLRESULT sunny 28C"},
            {"role": "assistant", "content": "Hangzhou is sunny."},
        ]},
        # 3 空 assistant content（tool_calls 常见形态）+ tool 回传
        {"messages": [
            {"role": "user", "content": "files?"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "TOOLRESULT [a.txt, b.txt]"},
            {"role": "assistant", "content": "two files"},
        ]},
    ]


def _render(records, template_globals=None):
    """与模板等价的朴素渲染（真值用，不依赖 minijinja）。"""
    out = []
    for rec in records:
        parts = []
        if template_globals and rec.get("tools"):
            parts.append(f"<|im_start|>system\n{len(rec['tools'])} tools available<|im_end|>\n")
        for m in rec["messages"]:
            parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        out.append("".join(parts))
    return out


def _expected_supervision(ids, tokstrs, tu, te):
    """独立复现监督状态机：assistant 段（含轮标记）监督，其余 role 不监督。"""
    n = len(ids)
    sup = [False] * n
    in_asst = False
    for i in range(n):
        tok = tokstrs[i] if i < len(tokstrs) else ""
        if ids[i] == tu or (tok and "<|im_start|>" in tok):
            role = []
            for j in range(i + 1, min(i + 8, n)):
                nx = tokstrs[j] if j < len(tokstrs) else ""
                if "\n" in nx or "<|im_end|>" in nx or "<|im_start|>" in nx:
                    break
                role.append(nx)
            in_asst = "".join(role).strip().startswith("assistant")
        if in_asst:
            sup[i] = True
        if ids[i] == te or (tok and "<|im_end|>" in tok):
            if in_asst:
                sup[i] = True
                in_asst = False
    return sup


def _expected_labels(ids, sup):
    lab = [-100] * len(ids)
    for i in range(len(ids) - 1):
        if sup[i]:
            lab[i] = ids[i + 1]
    return lab


def _build(tok_pair, tmp_path, records, template):
    """bbpe 默认路径构建，返回 index.json 路径。"""
    bbpe_path, _official, im_id, em_id = tok_pair
    raw = os.path.join(tmp_path, "raw.jsonl")
    with open(raw, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ds = os.path.join(tmp_path, "ds")
    cfg = BuildConfig(
        input_paths=[raw], output_dir=ds, mode="sft", template_str=template,
        batch_size=BATCH_SIZE, pad_id=0,
        turn_start=Marker(id=im_id), turn_end=Marker(id=em_id),
    )
    DataBuilder(tokenizer_path=bbpe_path, max_seq_len=MAX_SEQ_LEN).build_dataset(cfg)
    return os.path.join(ds, "index.json")


def _read_docs_with_labels(index_path):
    """确定性读全量（单 shard 顺序），按 row_lens/cu 还原 [(ids, labels, pos)]。"""
    loader = BatchStream(index_json_path=index_path, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                          shuffle=False, shard_order="sequential")
    docs = []
    for b in loader:
        ids_rows = b["input_ids"].tolist()
        lab_rows = b["labels"].tolist()
        pos_rows = b["position_ids"].tolist()
        row_lens = b["row_lens"].tolist()
        cu = b["cu_seqlens"].tolist()
        di = 0
        for r, rl in enumerate(row_lens):
            pos = 0
            while di < len(cu) - 1 and pos < rl:
                L = cu[di + 1] - cu[di]
                if L <= 0:
                    di += 1
                    continue
                docs.append((ids_rows[r][pos:pos + L],
                             lab_rows[r][pos:pos + L],
                             pos_rows[r][pos:pos + L]))
                pos += L
                di += 1
    return docs


def test_toolcall_sft_masking_and_lossless(tok_pair, tmp_path):
    """tool-call 全场景：无漏无重 + 监督掩码逐位 == 独立状态机 + 工具段不监督。"""
    _bbpe_path, official, im_id, em_id = tok_pair
    records = _records()
    index = _build(tok_pair, tmp_path, records, TOOL_TEMPLATE)

    # 真值：官方库独立复算渲染 + 编码
    rendered = _render(records)
    exp_ids, exp_tokens = [], []
    for text in rendered:
        e = official.encode(text, add_special_tokens=False)
        exp_ids.append(list(e.ids))
        exp_tokens.append(list(e.tokens))

    docs = _read_docs_with_labels(index)

    # 无漏无重：还原文档数 == 输入行数（cu_seqlens 文档数契约）
    assert len(docs) == len(records), f"docs={len(docs)} != inputs={len(records)}"
    # 文档 token 序列逐条 == 独立复算（渲染/顺序/内容）
    assert [d[0] for d in docs] == exp_ids, "文档序列与独立复算不一致（漏/重/错位）"

    bad_lab = [i for i, (ids, lab, _p) in enumerate(docs)
               if lab != _expected_labels(ids, _expected_supervision(ids, exp_tokens[i], im_id, em_id))]
    assert not bad_lab, f"监督掩码与独立状态机不一致的文档: {bad_lab}"

    # next-token 不变式 + position_ids 相对位置 + padding 契约
    loader = BatchStream(index_json_path=index, batch_size=BATCH_SIZE,
                          max_seq_len=MAX_SEQ_LEN, world_size=1, rank=0,
                          shuffle=False, shard_order="sequential")
    for b in loader:
        ids_rows, lab_rows = b["input_ids"].tolist(), b["labels"].tolist()
        row_lens = b["row_lens"].tolist()
        cu = b["cu_seqlens"].tolist()
        assert sum(row_lens) == cu[-1]
        for r, rl in enumerate(row_lens):
            assert all(t == 0 for t in ids_rows[r][rl:]), "padding 区 input_ids 必须 = pad_id"
            assert all(t == -100 for t in lab_rows[r][rl:]), "padding 区 labels 必须 = -100"
            for t in range(rl - 1):
                if lab_rows[r][t] != -100:
                    assert lab_rows[r][t] == ids_rows[r][t + 1], "next-token 不变式破坏"


def test_toolcall_tool_turn_never_supervised(tok_pair, tmp_path):
    """语义级：tool 轮内容绝不进监督段；assistant 段（含 JSON 参数与最终回答）被监督。"""
    _bbpe_path, official, _im, _em = tok_pair
    records = _records()
    index = _build(tok_pair, tmp_path, records, TOOL_TEMPLATE)
    docs = _read_docs_with_labels(index)

    for gdi, (ids, lab, _p) in enumerate(docs):
        sup_next = [ids[i + 1] for i in range(len(ids) - 1) if lab[i] != -100]
        sup_text = official.decode(sup_next)
        assert "TOOLRESULT" not in sup_text, \
            f"doc{gdi}: tool 回传内容泄漏进监督段: {sup_text!r}"
        if gdi in (2, 3):
            assert "Hangzhou is sunny." in sup_text or "two files" in sup_text, \
                f"doc{gdi}: assistant 最终回答未被监督: {sup_text!r}"
    # doc2 的 tool-call JSON 参数属于 assistant 段 → 被监督
    sup2 = official.decode([docs[2][0][i + 1] for i in range(len(docs[2][0]) - 1)
                            if docs[2][1][i] != -100])
    assert "get_weather" in sup2, f"tool-call JSON 应被监督: {sup2!r}"
    # doc3 空 assistant content：轮标记仍被监督 → labels 非全 -100（非无声废样本）
    assert any(x != -100 for x in docs[3][1]), "空 content 的 assistant 轮标记应被监督"


def test_toolcall_tools_field_rendered_but_not_supervised(tok_pair, tmp_path):
    """顶层 tools 字段经模板渲染进 system 段：内容进 input_ids，但 system 角色不监督。"""
    _bbpe_path, official, im_id, em_id = tok_pair
    rec = {"tools": [{"type": "function",
                      "function": {"name": "get_weather", "description": "query weather"}}],
           "messages": [{"role": "user", "content": "weather again?"},
                         {"role": "assistant", "content": "call tool then answer"}]}
    index = _build(tok_pair, tmp_path, [rec], TOOLSYS_TEMPLATE)
    docs = _read_docs_with_labels(index)

    exp_text = _render([rec], template_globals=True)[0]
    e = official.encode(exp_text, add_special_tokens=False)
    assert len(docs) == 1
    assert docs[0][0] == list(e.ids), "tools 渲染进 system 段后文档序列应与独立复算一致"

    tk = [official.id_to_token(i) for i in docs[0][0]]
    sup = _expected_supervision(docs[0][0], tk, im_id, em_id)
    assert docs[0][1] == _expected_labels(docs[0][0], sup), "掩码与独立状态机不一致"
    sup_next = [docs[0][0][i + 1] for i in range(len(docs[0][0]) - 1) if docs[0][1][i] != -100]
    sup_text = official.decode(sup_next)
    assert "tools available" not in sup_text, "system 段（工具定义）不得被监督"
    assert "call tool then answer" in sup_text, "assistant 段应被监督"
