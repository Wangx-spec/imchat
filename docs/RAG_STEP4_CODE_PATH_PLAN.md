# Step 4 技术代码路径方案（generation_router + service）

目标：在 Step3（索引与检索）基础上，完成“可回答闭环”：

- 查询路由（`list/detail/general`）
- 查询规则化改写（轻量、无 LLM 依赖）
- 基于检索结果生成可读回答
- 服务层统一编排（初始化、检索、回答、降级）

对应主文档：`docs/RAG_IMPLEMENTATION.md` 的 Step 4。

---

## 1. 本阶段范围

仅实现：

- `src/rag/generation_router.py`
- `src/rag/service.py`

依赖输入（来自 Step1~3）：

- `RAGConfig` 与配置校验链路
- `MarkdownDataLoader` / `ParentChildChunker`
- `LocalFAISSIndexStore` / `HybridRetriever`

不在本阶段实现：

- Agent 工具接入（Step5）
- CLI/Web 启动接入（Step6）
- 完整自动化测试收口（Step7）

---

## 2. 代码路径（调用关系）

Step4 完成后的最小调用链：

1. `service = RAGService(cfg)`
2. `service.initialize(force_rebuild=False)`
3. `result = service.answer(query)`
4. `answer_text = result.answer`（并附 `sources/debug`）

初始化内部顺序：

1. `validate_rag_config`
2. `loader.load_documents`
3. `chunker.build_parent_child`
4. `index_store.load / build+save`
5. `retriever = HybridRetriever(...)`
6. `router = GenerationRouter()`
7. `ready = True`

---

## 3. 文件级实现方案

## 3.1 `src/rag/generation_router.py`

### 3.1.1 推荐类结构

- `class GenerationRouter:`
  - `route_query(query: str) -> str`
  - `rewrite_query(query: str, route: str) -> str`
  - `build_answer(query: str, route: str, parents: list[Document]) -> str`
  - `_build_list_answer(query: str, parents: list[Document]) -> str`
  - `_build_detail_answer(query: str, parents: list[Document]) -> str`
  - `_build_general_answer(query: str, parents: list[Document]) -> str`

### 3.1.2 路由规则（先规则版）

- `list`：包含“推荐/有哪些/来几道/列出”等词。
- `detail`：包含“怎么做/步骤/做法/需要什么”等词。
- 其他归为 `general`。

### 3.1.3 改写规则（轻量）

- `list/detail`：默认不改写，保持原 query。
- `general`：可做轻度清洗（去多余标点、空白归一）。

### 3.1.4 回答构造策略

- `list`：提取去重标题，按条目返回。
- `detail`：优先返回首个高相关 parent 的结构化摘要（标题 + 要点）。
- `general`：返回 top parents 的简要汇总。

---

## 3.2 `src/rag/service.py`

### 3.2.1 推荐类结构

- `class RAGService:`
  - `__init__(cfg: RAGConfig) -> None`
  - `initialize(force_rebuild: bool = False) -> None`
  - `is_ready() -> bool`
  - `retrieve(query: str) -> RetrievalResult`
  - `answer(query: str) -> AnswerResult`
  - `stats() -> dict`（可选）

### 3.2.2 `initialize()` 职责

- 配置校验：`sanitize + validate`
- 构建文档与分块：parents/children/mapping
- 索引加载/构建：`load` 失败则 `build+save`
- 初始化检索器与路由器
- 设置 `ready=True`

### 3.2.3 `retrieve()` 与 `answer()` 职责

- `retrieve(query)`：只做检索，返回 `RetrievalResult`
- `answer(query)`：
  1) route  
  2) rewrite  
  3) retrieve  
  4) build_answer  
  5) 组装 `AnswerResult`

### 3.2.4 降级策略（必须有）

- `enabled=False`：返回可解释提示，不报错。
- `ready=False`：返回“服务未初始化”提示。
- 任一步异常：捕获并返回“降级回答 + debug 错误信息”。

---

## 4. 关键伪代码（可直接映射）

```python
# src/rag/generation_router.py
class GenerationRouter:
    def route_query(self, query: str) -> str:
        q = query.strip().lower()
        if any(x in q for x in ["推荐", "有哪些", "来几道", "列出"]):
            return "list"
        if any(x in q for x in ["怎么做", "做法", "步骤", "需要什么"]):
            return "detail"
        return "general"

    def rewrite_query(self, query: str, route: str) -> str:
        if route in {"list", "detail"}:
            return query
        return " ".join(query.split()).strip()

    def build_answer(self, query: str, route: str, parents: list[Document]) -> str:
        if not parents:
            return "未检索到相关内容，请尝试换个问法。"
        if route == "list":
            return self._build_list_answer(query, parents)
        if route == "detail":
            return self._build_detail_answer(query, parents)
        return self._build_general_answer(query, parents)
```

```python
# src/rag/service.py
class RAGService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ready = False
        self.loader = MarkdownDataLoader()
        self.chunker = ParentChildChunker(cfg.chunk_size, cfg.chunk_overlap)
        self.index_store = LocalFAISSIndexStore(cfg)
        self.retriever = None
        self.router = GenerationRouter()

    def initialize(self, force_rebuild: bool = False) -> None:
        if not self.cfg.enabled:
            self.ready = False
            return
        validate_rag_config(self.cfg)
        parents = self.loader.load_documents(self.cfg.source_dirs)
        children, parent_map, child_parent = self.chunker.build_parent_child(parents)
        loaded = False if force_rebuild else self.index_store.load()
        if not loaded:
            self.index_store.build(children)
            self.index_store.save()
        self.retriever = HybridRetriever(
            vectorstore=self.index_store.vectorstore,
            children=children,
            parent_map=parent_map,
            child_parent=child_parent,
            rrf_k=self.cfg.rrf_k,
        )
        self.ready = True

    def retrieve(self, query: str) -> RetrievalResult:
        if not self.ready or self.retriever is None:
            return RetrievalResult(query=query, parents=[], sources=[], debug={"error": "not_ready"})
        return self.retriever.hybrid_search(query, self.cfg.retrieval_k, self.cfg.top_k)

    def answer(self, query: str) -> AnswerResult:
        if not self.cfg.enabled:
            return AnswerResult(query=query, route="disabled", answer="RAG 未启用。")
        if not self.ready:
            return AnswerResult(query=query, route="not_ready", answer="RAG 尚未初始化。")
        route = self.router.route_query(query)
        rewritten = self.router.rewrite_query(query, route)
        ret = self.retrieve(rewritten)
        answer = self.router.build_answer(query, route, ret.parents)
        return AnswerResult(query=query, route=route, answer=answer, sources=ret.sources, debug=ret.debug)
```

---

## 5. 最小可运行自测（Step4）

## 5.1 建议脚本

`src/tests/tmp_step4_smoke.py`

## 5.2 自测流程

1. 初始化 `RAGService(cfg)`。
2. 执行 `initialize()`（观察 loaded/build 分支）。
3. 连续调用：
   - `answer("推荐几道家常菜")` -> `list`
   - `answer("宫保鸡丁怎么做")` -> `detail`
   - `answer("什么是食材相克")` -> `general`
4. 打印 `route/sources/debug/answer 预览`。

## 5.3 运行命令

```bash
PYTHONPATH=src python src/tests/tmp_step4_smoke.py
```

---

## 6. 验收标准（DoD）

- [ ] `initialize()` 可在无索引时构建并保存索引。
- [ ] `initialize()` 可在有索引时快速加载并就绪。
- [ ] `answer()` 能返回 `AnswerResult` 且 `route` 合理。
- [ ] `enabled=False`、`ready=False` 路径有可解释降级文本。
- [ ] `debug` 字段有可用诊断信息（至少包含检索统计或错误原因）。

---

## 7. 风险与规避

- **初始化过重**：首次构建慢，建议保留索引复用并打印耗时。
- **路由误判**：规则法简单但有边界，先保证可解释，后续再升级。
- **答案模板过弱**：先可读可追踪，Step5 后再结合 Agent 提示优化。
- **异常传播中断**：`answer()` 内统一兜底，避免影响主对话链路。

---

## 8. 与 Step5 的接口约定

Step5 只依赖：

- `RAGService.initialize(...)`
- `RAGService.answer(query) -> AnswerResult`

因此 Step4 必须保证：

- `AnswerResult.answer/sources/debug` 结构稳定
- 降级路径不抛未捕获异常

---

## 9. 当前代码改造建议（对齐你现状）

当前文件状态：

- `src/rag/generation_router.py`：3 个方法均为 `pass`
- `src/rag/service.py`：关键方法未实现

建议落地顺序：

1. 先补 `GenerationRouter.route_query/rewrite_query/build_answer`
2. 再补 `RAGService.initialize/retrieve/answer`
3. 写 `tmp_step4_smoke.py` 跑 3 条 query 验证路由分支
4. 最后再做样式优化与错误文案统一

