# Step 7 技术代码路径方案（依赖补齐 + 自动化测试）

目标：在 Step1~6 已可运行的基础上，建立“可重复执行、可回归”的测试网，覆盖配置层、分块层、检索层、工具集成层。

对应主文档：`docs/RAG_IMPLEMENTATION.md` 的 Step 7（3.7）。

---

## 1. 本阶段范围

本阶段新增内容：

- 测试依赖与执行入口（`pytest`）
- 单测：`rag.config`、`rag.chunking`、`rag.retriever`
- 集成测：`search_knowledge_base` 工具调用到回答输出
- 测试目录与命名规范

本阶段不做：

- 修改 RAG 业务逻辑（除非为可测性做极小接口暴露）
- 引入真实外部 API/向量服务做在线测试

---

## 2. 目录与文件规划

建议新增（仓库根目录）：

```text
tests/
  rag/
    test_config.py
    test_chunking.py
    test_retriever.py
  integration/
    test_rag_tool_flow.py
```

建议新增（仓库根目录）：

```text
pytest.ini
```

`pytest.ini` 建议内容：

```ini
[pytest]
pythonpath = src
testpaths = tests
```

这样可长期解决 `src/tests` 执行时的模块导入路径问题（`ModuleNotFoundError`）。

---

## 3. 依赖补齐方案

当前 `requirements.txt` 已含 RAG 运行依赖；Step7 仅补测试依赖即可。

建议补充：

- `pytest`
- （可选）`pytest-cov`

示例：

```txt
pytest>=8.0.0
pytest-cov>=5.0.0
```

---

## 4. 文件级实现方案

## 4.1 `tests/rag/test_config.py`

覆盖目标：`build_rag_config`、`sanitize_rag_config`、`validate_rag_config` 三段链路。

建议用例：

1. `test_build_sanitize_validate_config`
   - 构造最小 `Settings`（`rag_enabled=True` + 合法 embedding key）
   - 断言 build 后字段映射正确
   - 断言 sanitize 后 `retrieval_k >= top_k`
   - 断言 validate 不抛异常
2. `test_validate_rag_config_raise_when_missing_api_key`
   - `enabled=True` 且 `embedding_api_key=None`
   - 断言抛 `ValueError`

代码模板：

```python
from config.settings import Settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config


def make_settings(**kwargs) -> Settings:
    base = Settings(
        openai_api_key="dummy",
        rag_enabled=True,
        rag_source_dirs=["/tmp/not_used"],
        rag_embedding_api_key="test-key",
    )
    for k, v in kwargs.items():
        setattr(base, k, v)
    return base


def test_build_sanitize_validate_config():
    s = make_settings(rag_top_k=4, rag_retrieval_k=2)
    cfg = build_rag_config(s)
    cfg = sanitize_rag_config(cfg)

    assert cfg.top_k == 4
    assert cfg.retrieval_k >= cfg.top_k

    validate_rag_config(cfg)  # should not raise


def test_validate_rag_config_raise_when_missing_api_key():
    s = make_settings(rag_embedding_api_key=None)
    cfg = sanitize_rag_config(build_rag_config(s))

    try:
        validate_rag_config(cfg)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "embedding_api_key" in str(exc)
```

---

## 4.2 `tests/rag/test_chunking.py`

覆盖目标：`ParentChildChunker.build_parent_child` 的 parent-child 映射正确性和分块数量稳定性。

建议用例：

1. `test_parent_child_mapping_and_chunk_count`
   - 构造 2 个 parent 文档
   - 断言 `parent_map` 数量等于 parent 数量
   - 断言每个 child 都有 `child_id/parent_id/chunk_index/doc_type=child`
   - 断言 `child_parent[child_id] == child.metadata["parent_id"]`
2. `test_split_long_text_respects_overlap`
   - 长文本触发多 chunk
   - 断言 chunk 数 > 1 且没有空 chunk

代码模板：

```python
from langchain_core.documents import Document
from rag.chunking import ParentChildChunker


def _mk_parent(pid: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "parent_id": pid,
            "source": f"/tmp/{pid}.md",
            "title": pid,
            "doc_type": "parent",
        },
    )


def test_parent_child_mapping_and_chunk_count():
    parents = [
        _mk_parent("p1", "# 标题1\n" + "a" * 400),
        _mk_parent("p2", "# 标题2\n" + "b" * 400),
    ]
    chunker = ParentChildChunker(chunk_size=120, chunk_overlap=20)
    children, parent_map, child_parent = chunker.build_parent_child(parents)

    assert len(parent_map) == 2
    assert len(children) > 0
    assert len(child_parent) == len(children)

    for c in children:
        md = c.metadata
        assert md["doc_type"] == "child"
        assert md.get("child_id")
        assert md.get("parent_id") in {"p1", "p2"}
        assert child_parent[md["child_id"]] == md["parent_id"]
```

---

## 4.3 `tests/rag/test_retriever.py`

覆盖目标：`rrf_fuse` 融合 + `child_to_parent` 去重逻辑。

实现策略：不依赖真实 FAISS，使用最小 fake vectorstore。

建议用例：

1. `test_rrf_fusion_and_parent_dedup`
   - 构造 child 文档（两个 child 属于同一 parent）
   - fake vector 检索与 bm25 检索返回交叉结果
   - 断言 fused 结果包含 `rrf_score`
   - 断言映射 parent 后按 parent 去重

代码模板：

```python
from langchain_core.documents import Document
from rag.retriever import HybridRetriever


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
    def invoke(self, query):
        return self._docs


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs
    def as_retriever(self, search_kwargs=None):
        return _FakeRetriever(self._docs)


def _mk_child(child_id: str, parent_id: str, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={"child_id": child_id, "parent_id": parent_id, "chunk_index": 0},
    )


def test_rrf_fusion_and_parent_dedup():
    c1 = _mk_child("c1", "p1", "宫保鸡丁 做法")
    c2 = _mk_child("c2", "p1", "宫保鸡丁 步骤")
    c3 = _mk_child("c3", "p2", "尖椒炒牛肉 做法")
    children = [c1, c2, c3]

    parent_map = {
        "p1": Document(page_content="P1", metadata={"source": "/tmp/p1.md"}),
        "p2": Document(page_content="P2", metadata={"source": "/tmp/p2.md"}),
    }
    child_parent = {"c1": "p1", "c2": "p1", "c3": "p2"}

    retriever = HybridRetriever(
        vectorstore=_FakeVectorStore([c1, c2, c3]),
        children=children,
        parent_map=parent_map,
        child_parent=child_parent,
        rrf_k=60,
    )

    # 只校验核心行为，不依赖语义效果
    result = retriever.hybrid_search("宫保鸡丁怎么做", retrieval_k=3, top_k=2)
    assert len(result.parents) <= 2
    assert len(result.sources) == len(set(result.sources))
    assert "fused_hits" in result.debug
```

---

## 4.4 `tests/integration/test_rag_tool_flow.py`

覆盖目标：工具 `search_knowledge_base` 从“绑定 service -> 调用 answer -> 返回文本（含来源）”的端到端链路。

建议用例：

1. `test_tool_to_answer_end_to_end`
   - 使用 fake service（仅实现 `answer()`）
   - 调 `set_rag_service(fake_service)`
   - 调 `search_knowledge_base.invoke({"query": "..."})`
   - 断言回答文本包含主回答与“来源：”
2. `test_tool_returns_not_initialized_when_service_missing`
   - `set_rag_service(None)`
   - 断言返回“知识检索服务未初始化。”

代码模板：

```python
from rag.types import AnswerResult
from actions.basic_tools import set_rag_service, search_knowledge_base


class _FakeService:
    def answer(self, query: str) -> AnswerResult:
        return AnswerResult(
            query=query,
            route="detail",
            answer="宫保鸡丁做法：先腌制鸡丁，再爆香后翻炒。",
            sources=["/tmp/宫保鸡丁.md"],
            debug={"ok": True},
        )


def test_tool_to_answer_end_to_end():
    set_rag_service(_FakeService())
    out = search_knowledge_base.invoke({"query": "宫保鸡丁怎么做"})
    assert "宫保鸡丁做法" in out
    assert "来源：" in out


def test_tool_returns_not_initialized_when_service_missing():
    set_rag_service(None)
    out = search_knowledge_base.invoke({"query": "宫保鸡丁怎么做"})
    assert "未初始化" in out
```

---

## 5. 执行与分层测试策略

建议分层执行，便于快速定位故障：

1. 配置层：`pytest tests/rag/test_config.py -q`
2. 分块层：`pytest tests/rag/test_chunking.py -q`
3. 检索层：`pytest tests/rag/test_retriever.py -q`
4. 集成层：`pytest tests/integration/test_rag_tool_flow.py -q`
5. 全量：`pytest -q`

---

## 6. 验收标准（DoD）

- [ ] 可在仓库根目录直接执行 `pytest -q`。
- [ ] `test_config` 覆盖 build/sanitize/validate 三段链路。
- [ ] `test_chunking` 覆盖 parent-child 映射与 child 元数据。
- [ ] `test_retriever` 覆盖 RRF 融合与 parent 去重。
- [ ] `test_rag_tool_flow` 覆盖工具端到端返回格式与未初始化降级。

---

## 7. 风险与规避

- **外部依赖波动**：Step7 单测不依赖真实 embedding API；优先 fake/mocking。
- **导入路径失败**：使用 `pytest.ini` 固化 `pythonpath=src`。
- **用例脆弱**：检索层断言结构与关键字段，不对排名做过度强约束。
- **全局状态污染**：`set_rag_service` 用例结束前可显式重置为 `None`。

---

## 8. 与后续阶段的衔接

Step7 完成后，可在 CI 中接入：

- `pytest -q`
- （可选）`pytest --cov=src --cov-report=term-missing`

后续若引入 Step8/GraphRAG，可在 `tests/integration/` 增加新链路测试，保持“旧能力回归不退化”。

