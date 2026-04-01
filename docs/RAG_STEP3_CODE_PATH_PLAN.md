# Step 3 技术代码路径方案（index_store + retriever）

目标：在 Step2（`data_loader + chunking`）基础上，完成“可检索闭环”：

- 建立本地向量索引（构建/保存/加载）
- 建立 BM25 检索器
- 实现向量检索 + BM25 + RRF 融合
- 输出统一 `RetrievalResult`，供 Step4 服务层调用

对应主文档：`docs/RAG_IMPLEMENTATION.md` 的 Step 3。

---

## 1. 本阶段范围

仅实现：

- `src/rag/index_store.py`
- `src/rag/retriever.py`

依赖输入（来自 Step2）：

- `children: list[Document]`
- `parent_map: dict[str, Document]`
- `child_parent: dict[str, str]`

不在本阶段实现：

- 路由器与答案生成（Step4）
- Agent 工具接入（Step5）
- Web/CLI 启动接入（Step6）

---

## 2. 代码路径（调用关系）

Step3 完成后的最小调用链：

1. `settings = load_settings()`
2. `cfg = build_rag_config(settings)` -> `sanitize` -> `validate`
3. `parents = loader.load_documents(cfg.source_dirs)`
4. `children, parent_map, child_parent = chunker.build_parent_child(parents)`
5. `store = LocalFAISSIndexStore(cfg)`
6. `loaded = store.load()`（可选）
7. `if not loaded: store.build(children); store.save()`
8. `retriever = HybridRetriever(store.vectorstore, children, parent_map, child_parent, cfg.rrf_k)`
9. `result = retriever.hybrid_search(query, cfg.retrieval_k, cfg.top_k)`

输出（供 Step4）：

- `RetrievalResult.query`
- `RetrievalResult.parents`
- `RetrievalResult.sources`
- `RetrievalResult.debug`

---

## 3. Step3 数据与分数规范

## 3.1 检索文档标识

建议在 child metadata 中优先使用 `child_id` 作为唯一键；若缺失可回退 `source + chunk_index`。

## 3.2 RRF 融合规则

RRF 公式（单文档）：

`score = sum(1 / (rrf_k + rank_i + 1))`

其中：

- `rank_i` 为该文档在某一路检索结果中的名次（从 0 开始）
- `rrf_k` 来自配置（默认 `60`）

## 3.3 回填 parent 规则

1. 对融合后的 child 排序。
2. 用 `child_parent` 映射到 `parent_id`。
3. parent 去重保序。
4. 截断为 `top_k` 个 parent。

---

## 4. 文件级实现方案

## 4.1 `src/rag/index_store.py`

### 4.1.1 推荐类结构

- `class LocalFAISSIndexStore:`
  - `__init__(cfg: RAGConfig) -> None`
  - `_build_embeddings()`
  - `exists() -> bool`
  - `build(children: list[Document]) -> None`
  - `save() -> None`
  - `load() -> bool`
  - `as_retriever(k: int)`

### 4.1.2 Embedding 接入建议（DashScope）

优先使用 `langchain_openai.OpenAIEmbeddings`（OpenAI 兼容接口）：

- `api_key=cfg.embedding_api_key`
- `base_url=cfg.embedding_base_url`
- `model=cfg.embedding_model`（如 `text-embedding-v4`）
- `dimensions=cfg.embedding_dimensions`（如 `1024`）
- `check_embedding_ctx_length=False`（兼容某些 OpenAI 兼容 embedding 网关）
- `chunk_size=10`（适配当前 DashScope 单批大小限制）

### 4.1.3 最小实现顺序

1. `__init__`：初始化路径与 embeddings。
2. `build`：`FAISS.from_documents(children, embeddings)`。
3. `save`：`vectorstore.save_local(index_dir)`。
4. `load`：`FAISS.load_local(...)` 并返回 `True/False`。
5. `as_retriever(k)`：输出向量检索器。

### 4.1.4 失败策略

- 索引不存在：`load()` 返回 `False`，不抛错。
- embeddings 初始化失败：抛异常交给上层（Step4 决定降级）。
- 索引存在但配置已变化：建议通过“索引签名”判定并强制重建。

---

## 4.2 `src/rag/retriever.py`

### 4.2.1 推荐类结构

- `class HybridRetriever:`
  - `__init__(vectorstore, children, parent_map, child_parent, rrf_k=60)`
  - `vector_search(query: str, k: int) -> list[Document]`
  - `bm25_search(query: str, k: int) -> list[Document]`
  - `rrf_fuse(vector_docs, bm25_docs) -> list[Document]`
  - `child_to_parent(fused_children: list[Document], top_k: int) -> list[Document]`
  - `_collect_sources(parents: list[Document]) -> list[str]`
  - `hybrid_search(query: str, retrieval_k: int, top_k: int) -> RetrievalResult`

类型建议：

- `parent_map` 建议使用 `dict[str, Document]`，便于静态检查与后续维护。

### 4.2.2 最小实现顺序

1. 在 `__init__` 里构造 BM25（`BM25Retriever.from_documents(children)`）。
2. 实现 `vector_search` 与 `bm25_search`。
3. 实现 `rrf_fuse`（按唯一 doc key 聚合分数）。
4. 实现 `child_to_parent`（去重回填）。
5. 在 `hybrid_search` 输出 `RetrievalResult`。

### 4.2.3 Debug 字段建议

`RetrievalResult.debug` 建议至少包含：

- `vector_hits`
- `bm25_hits`
- `fused_hits`
- `parent_hits`
- `rrf_k`

---

## 5. 关键伪代码（可直接映射）

```python
# src/rag/index_store.py
class LocalFAISSIndexStore:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_dir = Path(cfg.index_dir)
        self.embeddings = OpenAIEmbeddings(
            api_key=cfg.embedding_api_key,
            base_url=cfg.embedding_base_url,
            model=cfg.embedding_model,
            dimensions=cfg.embedding_dimensions,
        )
        self.vectorstore = None

    def build(self, children):
        self.vectorstore = FAISS.from_documents(children, self.embeddings)

    def save(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(self.index_dir))

    def load(self) -> bool:
        if not self.index_dir.exists():
            return False
        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return True
```

```python
# src/rag/retriever.py
class HybridRetriever:
    def hybrid_search(self, query, retrieval_k, top_k):
        vector_docs = self.vector_search(query, retrieval_k)
        bm25_docs = self.bm25_search(query, retrieval_k)

        fused_children = self.rrf_fuse(vector_docs, bm25_docs)
        parents = self.child_to_parent(fused_children, top_k)
        sources = self._collect_sources(parents)

        return RetrievalResult(
            query=query,
            parents=parents,
            sources=sources,
            debug={
                "vector_hits": len(vector_docs),
                "bm25_hits": len(bm25_docs),
                "fused_hits": len(fused_children),
                "parent_hits": len(parents),
                "rrf_k": self.rrf_k,
            },
        )
```

---

## 6. 最小可运行自测（Step3）

## 6.1 建议脚本

`src/tests/tmp_step3_smoke.py`

## 6.2 自测流程

1. 跑 Step1 配置链路。
2. 跑 Step2 生成 `children/parent_map/child_parent`。
3. 跑 Step3 索引构建（或加载）。
4. 执行 `hybrid_search("宫保鸡丁怎么做", retrieval_k=8, top_k=3)`。
5. 打印：
   - `len(result.parents)`
   - `result.sources[:3]`
   - `result.debug`

## 6.3 运行命令

```bash
PYTHONPATH=src python src/tests/tmp_step3_smoke.py
```

---

## 7. 验收标准（DoD）

- [ ] `index_store.load()` 在无索引时返回 `False`。
- [ ] `index_store.build()+save()+load()` 可闭环。
- [ ] `hybrid_search()` 能返回非空 `RetrievalResult`（给定常见 query）。
- [ ] `parents` 数量不超过 `top_k`，且去重生效。
- [ ] `sources` 可追踪到原文档路径。
- [ ] `debug` 字段包含检索统计。
- [ ] 菜名类 query（如“宫保鸡丁怎么做”）结果中至少有 1 条同名或高相关文档。

---

## 8. 风险与规避

- **依赖缺失**：确保安装 `langchain-community`、`langchain-openai`、`faiss-cpu`、`rank-bm25`。
- **Embedding 接口不兼容**：固定使用 DashScope OpenAI 兼容 `base_url`。
- **索引加载失败**：统一返回 `False`，避免异常中断全流程。
- **索引陈旧风险**：仅按文件存在加载会复用旧索引，建议增加签名校验。
- **doc 唯一键不稳定**：优先使用 `child_id` 聚合 RRF 分数。
- **检索偏题风险**：菜名明确问题仍可能命中泛文档，需做 query 规则化或轻过滤。
- **IDE 依赖告警**：终端可运行但 IDE 报导入错误，多为解释器未对齐。

---

## 9. 与 Step4 的接口约定

Step4 只依赖以下稳定接口：

- `LocalFAISSIndexStore.load/build/save`
- `HybridRetriever.hybrid_search(...) -> RetrievalResult`

请在 Step3 内保持 `RetrievalResult` 字段语义不变，避免 Step4 对接成本上升。

---

## 10. 当前实现对齐状态（基于最近 smoke）

已达成：

- Step1 -> Step2 -> Step3 全链路可执行。
- `tmp_step3_smoke.py` 成功输出 `STEP3_SMOKE_OK`。
- 索引可加载复用（`index_loaded=True`）。
- `hybrid_search` 返回 `RetrievalResult` 且 `debug` 字段齐全。

待收口（进入 Step4 前建议完成）：

- [ ] 增加索引签名机制，避免配置变更后误用旧索引。
- [ ] 修正 `HybridRetriever` 中 `parent_map` 的类型标注。
- [ ] 增加菜名 query 的相关性兜底策略。

---

## 11. Step3 收尾清单（10 分钟版）

1. **索引一致性**
   - [ ] 增加 `index.meta.json`（记录数据源/分块/模型摘要）。
   - [ ] `load()` 前校验摘要，不一致则 `build+save`。

2. **检索质量**
   - [ ] 固定 5 条 query 回归（菜名/推荐/知识问答各类）。
   - [ ] 检查 `result.sources[:3]` 是否语义匹配。

3. **代码稳态**
   - [ ] 统一类型标注：`parent_map: dict[str, Document]`。
   - [ ] 保持 `RetrievalResult.debug` 字段稳定，便于 Step4 接入。

