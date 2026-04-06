# wx-langchain RAG 实施文档（重整版）

本文档按固定顺序编排：

1. 开发需求
2. 开发顺序
3. 每个开发顺序中的简要代码实现（伪代码）

---

## 1. 开发需求

## 1.1 背景与目标

在当前 `cook-proj` 中接入 RAG 能力，支持基于本地知识库（如 `chapter8`、`source_dir/HowToCook`）的问答，且不破坏现有 Agent 的 CLI/Web 主流程。

目标：

- 保持现有 `create_agent` 架构不变，以工具方式接入 RAG。
- 先实现轻量可用版本（本地索引 + 混合检索）。
- 故障可降级，不影响基础聊天功能。
- 为后续图增强（Neo4j/Milvus）预留演进空间。

## 1.2 范围（一期）

包含：

- 配置扩展（Settings + RAGConfig）
- Markdown 数据加载
- Parent/Child 分块
- 向量检索 + BM25 + RRF（后续步骤）
- RAG 服务编排
- Agent 工具接入
- CLI/Web 启动链路接入

不包含（一期外）：

- 图 RAG 多跳推理
- 多模态检索
- 高级重排模型

## 1.3 关键约束

- `RAG_ENABLED=false` 时，系统行为与当前一致。
- 启动失败可降级，不阻塞主进程。
- 配置与业务规则分层：`Settings` 负责读取，`rag/config.py` 负责映射/校验。
- Agent 运行时支持 `langgraph/langchain` 切换，不影响 RAG 工具接口约定。

## 1.4 技术路线（一句话）

先做“轻量 RAG 闭环”（Step 1-7），再按收益决定图增强。

---

## 2. 开发顺序

按以下顺序开发：

1. **Step 1**：`settings.py` + `rag/types.py` + `rag/config.py`
2. **Step 2**：`rag/data_loader.py` + `rag/chunking.py`
3. **Step 3**：`rag/index_store.py` + `rag/retriever.py`
4. **Step 4**：`rag/generation_router.py` + `rag/service.py`
5. **Step 5**：`actions/basic_tools.py` + `agents/dialog_agent.py`
6. **Step 6**：`main.py` + `web/app.py`
7. **Step 7**：`requirements.txt` + `tests/*`

推荐执行原则：

- 每步先“最小可运行”，再补优化。
- 每步都做 smoke 验证，避免问题堆积。
- 先保证链路打通，再做质量和性能调优。

---

## 3. 每个开发顺序中的简要代码实现（伪代码）

## 3.1 Step 1：配置与类型基础

目标：完成配置读取、RAG 配置对象、映射与校验。

```python
# src/config/settings.py
@dataclass
class Settings:
    openai_api_key: str
    ...
    rag_enabled: bool = False
    rag_source_dirs: list[str] = field(default_factory=list)
    rag_index_dir: str = "data/rag_index"
    rag_top_k: int = 4
    rag_retrieval_k: int = 12
    rag_embedding_provider: str = "dashscope"
    rag_embedding_api_key: str | None = None
    rag_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rag_embedding_model: str = "text-embedding-v4"
    rag_embedding_dimensions: int = 1024
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_rrf_k: int = 60
    rag_rebuild: bool = False

def load_settings() -> Settings:
    load_dotenv()
    # parse bool/int/csv/path
    return Settings(...)
```

```python
# src/rag/types.py
@dataclass
class RAGConfig: ...

@dataclass
class RetrievalResult: ...

@dataclass
class AnswerResult: ...
```

```python
# src/rag/config.py
def build_rag_config(settings: Settings) -> RAGConfig:
    return RAGConfig(...)

def sanitize_rag_config(cfg: RAGConfig) -> RAGConfig:
    # overlap/size, retrieval_k/top_k 等可修正项
    return fixed_cfg

def validate_rag_config(cfg: RAGConfig) -> None:
    # 不可修复项抛 ValueError
    return
```

---

## 3.2 Step 2：数据加载与分块

目标：把知识库变成可检索子块，并建立父子映射。

```python
# src/rag/data_loader.py
class MarkdownDataLoader:
    def scan_markdown_files(self, source_dirs: list[str]) -> list[Path]:
        return all_md_files

    def load_documents(self, source_dirs: list[str]) -> list[Document]:
        docs = []
        for p in self.scan_markdown_files(source_dirs):
            text = self._read_text(p)
            meta = self._build_parent_metadata(p, text)
            docs.append(Document(page_content=text, metadata=meta))
        return docs
```

```python
# src/rag/chunking.py
class ParentChildChunker:
    def build_parent_child(self, parents: list[Document]):
        children = []
        parent_map = {}      # parent_id -> parent doc
        child_parent = {}    # child_id -> parent_id
        for parent in parents:
            chunks = self._split_by_markdown_headers_then_length(parent.page_content)
            for i, chunk_text in enumerate(chunks):
                child = self._make_child_doc(parent, chunk_text, i)
                children.append(child)
                child_parent[child.metadata["child_id"]] = child.metadata["parent_id"]
            parent_map[parent.metadata["parent_id"]] = parent
        return children, parent_map, child_parent
```

---

## 3.3 Step 3：索引与检索

目标：实现向量检索 + BM25 + RRF 融合。

```python
# src/rag/index_store.py
class LocalFAISSIndexStore:
    def build(self, children: list[Document]) -> None:
        # OpenAIEmbeddings (DashScope compatible)
        # FAISS.from_documents(...)
        return

    def save(self) -> None: ...
    def load(self) -> bool: ...
```

```python
# src/rag/retriever.py
class HybridRetriever:
    def vector_search(self, query: str, k: int) -> list[Document]: ...
    def bm25_search(self, query: str, k: int) -> list[Document]: ...
    def rrf_fuse(self, vector_docs, bm25_docs, rrf_k: int) -> list[Document]: ...

    def hybrid_search(self, query: str, retrieval_k: int, top_k: int) -> RetrievalResult:
        v = self.vector_search(query, retrieval_k)
        b = self.bm25_search(query, retrieval_k)
        fused_children = self.rrf_fuse(v, b, self.rrf_k)
        parents = self.child_to_parent_dedup(fused_children, top_k)
        return RetrievalResult(query=query, parents=parents, sources=self._collect_sources(parents))
```

---

## 3.4 Step 4：路由与服务编排

目标：统一初始化、检索、回答入口。

```python
# src/rag/generation_router.py
class GenerationRouter:
    def route_query(self, query: str) -> str:
        # list/detail/general
        return route

    def rewrite_query(self, query: str, route: str) -> str:
        # Step4 先规则化，不上复杂 LLM
        return query_or_rewritten

    def build_answer(self, query: str, route: str, parents: list[Document]) -> str:
        # 先模板拼接，后续可升级
        return answer_text
```

```python
# src/rag/service.py
class RAGService:
    def initialize(self, force_rebuild: bool = False) -> None:
        # config -> load docs -> chunk -> load/build index -> init retriever
        self.ready = True

    def retrieve(self, query: str) -> RetrievalResult:
        return self.retriever.hybrid_search(query, self.cfg.retrieval_k, self.cfg.top_k)

    def answer(self, query: str) -> AnswerResult:
        route = self.router.route_query(query)
        q2 = self.router.rewrite_query(query, route)
        ret = self.retrieve(q2)
        ans = self.router.build_answer(query, route, ret.parents)
        return AnswerResult(query=query, route=route, answer=ans, sources=ret.sources, debug=ret.debug)
```

---

## 3.5 Step 5：工具接入 Agent

目标：新增 `search_knowledge_base` 并让 Agent 能调用。

```python
# src/actions/basic_tools.py
_rag_service = None

def set_rag_service(service): ...

@tool
def search_knowledge_base(query: str) -> str:
    if _rag_service is None:
        return "知识检索服务未初始化。"
    result = _rag_service.answer(query)
    return format_tool_output(result.answer, result.sources)

def get_actions():
    return [get_current_time, calculate, search_knowledge_base]
```

```python
# src/agents/dialog_agent.py
def build_dialog_agent(settings):
    return create_agent(
        model=...,
        tools=get_actions(),
        system_prompt=(
            "文档/章节/做法问题优先调用 search_knowledge_base；"
            "时间问题优先 get_current_time；数学问题优先 calculate。"
        ),
    )
```

---

## 3.6 Step 6：CLI/Web 启动链路接入

目标：应用启动时初始化 RAG，失败可降级。

```python
# src/main.py / src/web/app.py
def bootstrap_rag(settings):
    cfg = sanitize_rag_config(build_rag_config(settings))
    validate_rag_config(cfg)
    service = RAGService(cfg)
    service.initialize(force_rebuild=cfg.rebuild)
    set_rag_service(service)

def startup():
    settings = load_settings()
    try:
        if settings.rag_enabled:
            bootstrap_rag(settings)
    except Exception as exc:
        log_warning("rag_bootstrap_failed", error=str(exc))
    # continue normal app flow
```

---

## 3.7 Step 7：依赖与测试

目标：补齐依赖并建立基础测试网。

```python
# tests/rag/test_config.py
def test_build_sanitize_validate_config(): ...

# tests/rag/test_chunking.py
def test_parent_child_mapping_and_chunk_count(): ...

# tests/rag/test_retriever.py
def test_rrf_fusion_and_parent_dedup(): ...

# tests/integration/test_rag_tool_flow.py
def test_tool_to_answer_end_to_end(): ...
```

---

## 附：里程碑验收建议

- Step 1 完成：配置与类型 smoke 通过。
- Step 2 完成：可输出 parents/children/mapping。
- Step 3 完成：输入 query 可返回稳定检索结果。
- Step 4 完成：`RAGService.answer()` 可单独调用。
- Step 5-6 完成：CLI/Web 中可真实触发 RAG 工具。
- Step 7 完成：关键测试可重复执行。

> 实施建议：每个 Step 单独 commit，提交信息如 `feat(rag): complete step2 loader and chunking`。
# wx-langchain RAG 实施方案（重构版）

## 1. 文档目的

本文档用于指导在当前 `cook-proj` 中落地可维护、可扩展的 RAG 能力。

重构目标：

- 对齐当前仓库真实结构与代码入口。
- 明确一期与二期边界，避免过度设计。
- 给出可直接执行的模块拆分、配置设计、实施步骤和验收标准。

---

## 2. 项目现状与约束

### 2.1 当前仓库现状

当前项目是轻量 Agent 架构，主要入口如下：

- `src/agents/dialog_agent.py`：`create_agent` 组装入口。
- `src/actions/basic_tools.py`：工具注册与实现。
- `src/main.py`：CLI 对话入口。
- `src/web/app.py`：FastAPI Web 入口。
- `src/config/settings.py`：配置加载。
- `src/rag/`：已预留目录但模块尚未实现。

### 2.2 参考项目定位

- `source_dir/HowToCook`：菜谱 Markdown 数据源，不是 RAG 应用代码。
- `source_dir/What-to-eat-today`：图 RAG 参考实现（Neo4j + Milvus + 路由器等），可借鉴思想但不建议一期全量迁移。
- `chapter8`：轻量 RAG 实施思路，和当前仓库兼容度高。

### 2.3 关键约束

- 不破坏现有 CLI/Web 主链路。
- 保持 Agent 工具化接入方式不变。
- RAG 故障时可降级，不影响基础问答。
- 优先本地可运行、低依赖、低运维成本。

---

## 3. 建设策略（核心决策）

### 3.1 一期：轻量 RAG（推荐先落地）

采用 chapter8 路线：

- Markdown 知识库加载
- Parent/Child 分块
- 向量检索 + BM25 + RRF
- 工具化接入 Agent
- 本地索引缓存

### 3.2 二期：图增强 RAG（可选演进）

在一期稳定后，引入 Neo4j 图检索能力，作为可插拔增强检索器，仅在复杂关系问题触发。

### 3.3 为什么不一期直接图 RAG

- 当前仓库是单体 Agent 项目，图 RAG 的外部依赖和运维复杂度较高。
- 一期目标是“快速可用 + 可解释 + 易维护”，轻量方案更匹配。

---

## 4. 目标能力范围

### 4.1 一期能力

- 回答教程与菜谱类问题（如“某道菜怎么做”“某章节讲了什么”）。
- 优先返回基于知识库的答案，减少幻觉。
- 支持来源可追踪（至少保留 source/title）。
- 支持启动时索引复用（秒级启动）。

### 4.2 暂不纳入一期

- 图谱推理、多跳路径检索
- 多模态检索（图文）
- 复杂重排模型（ColBERT/RankLLM）

---

## 5. 技术栈（预计）

| 层级 | 技术/组件 | 一期 | 二期 |
|---|---|---|---|
| 语言与运行时 | Python 3.10+ | 必选 | 必选 |
| Agent 框架 | LangChain + 现有 `create_agent` | 必选 | 必选 |
| LLM 接入 | 当前 `openai` 兼容链路 | 必选 | 必选 |
| Embedding | `sentence-transformers` + `BAAI/bge-small-zh-v1.5` | 必选 | 必选 |
| 向量索引 | `faiss-cpu` | 必选 | 可保留 |
| 关键词检索 | `rank-bm25` | 必选 | 必选 |
| 分块工具 | `langchain-text-splitters` | 必选 | 必选 |
| 图数据库 | Neo4j | 不用 | 可选 |
| 向量数据库服务化 | Milvus | 不用 | 可选 |
| Web 接口 | FastAPI（现有） | 必选 | 必选 |
| 日志 | `logging` | 必选 | 必选 |
| 测试 | `pytest` | 建议 | 必选 |

---

## 6. 架构设计（一期）

### 6.1 系统形态

- RAG 作为 Agent 的一个工具 `search_knowledge_base`。
- Agent 根据问题类型决定是否调用该工具。
- RAG 返回结构化检索结果，LLM 基于结果生成最终回答。

### 6.2 总体流程

1. 启动时初始化 `RAGService`。
2. 检查索引签名：
   - 命中：加载本地索引。
   - 未命中：重建索引并保存。
3. 用户提问时，Agent 判断是否调用 RAG 工具。
4. 工具执行混合检索并返回高相关上下文。
5. Agent 基于工具返回生成回答。

---

## 7. 目录与模块职责

建议目录：

```text
src/rag/
├─ __init__.py
├─ types.py
├─ config.py
├─ data_loader.py
├─ chunking.py
├─ index_store.py
├─ retriever.py
├─ generation_router.py
└─ service.py
```

职责说明：

- `types.py`：定义 `RAGConfig`、`RetrievalResult`、`AnswerResult`。
- `config.py`：`Settings -> RAGConfig` 映射与校验。
- `data_loader.py`：递归加载 Markdown，补全元数据。
- `chunking.py`：Parent/Child 分块与映射关系维护。
- `index_store.py`：Embedding/FAISS 构建、保存、加载、签名管理。
- `retriever.py`：向量检索、BM25、RRF 融合与去重。
- `generation_router.py`：查询类型判断（list/detail/general）和可选重写。
- `service.py`：统一初始化、检索、返回结果入口。

---

## 8. 配置设计（Settings 扩展）

建议新增环境变量：

- `RAG_ENABLED=true`
- `RAG_SOURCE_DIRS=chapter8,source_dir/HowToCook`
- `RAG_INDEX_DIR=data/rag_index`
- `RAG_TOP_K=4`
- `RAG_RETRIEVAL_K=12`
- `RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`
- `RAG_CHUNK_SIZE=800`
- `RAG_CHUNK_OVERLAP=120`
- `RAG_RRF_K=60`
- `RAG_REBUILD=false`

规则：

- `RAG_SOURCE_DIRS` 支持逗号分隔，启动时转绝对路径。
- 数值参数设最小阈值。
- `chunk_overlap >= chunk_size` 自动修正并记录告警。
- `RAG_ENABLED=false` 时全链路跳过 RAG 初始化。

---

## 9. 数据准备与分块策略

### 9.1 数据加载

- 递归读取 `*.md`。
- 跳过图片、二进制、超大文本（按阈值过滤）。
- 元数据最小集合：
  - `source`
  - `title`
  - `category`（由路径推断）
  - `parent_id`
  - `doc_type`

### 9.2 Parent/Child 设计

- Parent：完整文档（生成阶段使用）。
- Child：细粒度块（检索阶段使用）。

推荐流程：

1. 先按 Markdown 标题切分（`#`、`##`、`###`）。
2. 过长片段再按长度切分（`chunk_size/chunk_overlap`）。
3. 维护 `child_id -> parent_id` 映射。

### 9.3 去重策略

- 检索返回 child 后映射回 parent。
- 按 parent 命中次数和融合分数排序。
- 最终返回去重后的 parent 文档。

---

## 10. 索引与检索策略

### 10.1 向量检索

- 使用 `HuggingFaceEmbeddings`。
- 使用 `FAISS` 本地索引。
- 支持持久化和热加载。

### 10.2 BM25 检索

- 基于 child 文本构建 BM25 检索器。

### 10.3 融合重排（RRF）

- 分别执行向量检索和 BM25 检索。
- 使用 RRF 融合排名，参数 `rrf_k` 配置化。
- 输出统一相关性分数，便于日志和调试。

### 10.4 索引签名

签名建议包含：

- 数据目录列表
- 文档数量/mtime 摘要
- embedding 模型名
- chunk 参数

签名不一致则触发重建。

---

## 11. 生成策略与查询路由

查询类型：

- `list`：推荐/列表型问题
- `detail`：步骤/做法型问题
- `general`：概念/说明型问题

策略建议：

- `list/detail` 直接检索并生成。
- `general` 可选规则化 rewrite（先规则，后续再 LLM 化）。
- 如未检索到结果，返回可解释降级文本。

---

## 12. 与现有系统接入点

### 12.1 工具层

修改 `src/actions/basic_tools.py`：

- 新增工具 `search_knowledge_base(query: str) -> str`
- 工具内部调用 `RAGService.retrieve()` 或 `RAGService.answer()`
- 保持原 `get_current_time` 与 `calculate` 不变

### 12.2 Agent 层

修改 `src/agents/dialog_agent.py`：

- 注册新工具
- 更新 system prompt：
  - 文档/章节/做法类问题优先调用 `search_knowledge_base`
  - 时间问题优先 `get_current_time`
  - 数学问题优先 `calculate`

### 12.3 启动链路

修改 `src/main.py` 与 `src/web/app.py`：

- 启动时尝试初始化 RAG（可开关）
- 初始化失败时记录日志并继续服务
- 提供运行时状态日志（是否命中索引缓存）

---

## 13. 依赖清单

在现有 `requirements.txt` 基础上增加：

- `langchain-community`
- `langchain-text-splitters`
- `faiss-cpu`
- `sentence-transformers`
- `rank-bm25`

二期可选：

- `neo4j`
- `pymilvus`

---

## 14. 分阶段实施计划

### Phase A（MVP，建议 2-4 天）

1. 完成 `Settings` 与 RAG 配置扩展。
2. 完成 `data_loader/chunking/index_store/retriever/service`。
3. 接入 `search_knowledge_base` 到 Agent 工具链。
4. CLI 端跑通端到端问答。

### Phase B（可用版，建议 2-3 天）

1. 接入 Web 初始化链路。
2. 完善索引签名与缓存。
3. 完善日志、异常处理、降级文本。
4. 增加最小测试集与回归校验。

### Phase C（优化版，建议 3-5 天）

1. 加入 `generation_router` 和 rewrite。
2. 参数调优（`chunk/top_k/retrieval_k/rrf_k`）。
3. 构建离线评测样本集并记录指标。

### Phase D（二期图增强，可选）

1. 引入图数据与图检索器接口。
2. 复杂查询路由到图检索，简单查询仍走轻量检索。
3. 形成可插拔双引擎架构。

---

## 15. 验收标准（Definition of Done）

- [ ] `RAG_ENABLED=true` 时，可回答知识库问题且明显优于纯模型直答。
- [ ] `RAG_ENABLED=false` 时，不影响原有 Agent 功能。
- [ ] 首次构建索引成功，二次启动可命中缓存。
- [ ] 工具调用日志可追踪（查询、命中数、耗时、数据源）。
- [ ] Web 与 CLI 均可稳定运行。
- [ ] 核心模块具备基础单元测试。

---

## 16. 最小测试集（建议）

### 16.1 功能测试

- “chapter8 第三节讲什么？”
- “宫保鸡丁怎么做？”
- “推荐 3 道简单素菜”

### 16.2 降级测试

- 数据目录不存在
- 索引目录无权限
- embedding 模型加载失败

### 16.3 回归测试

- 时间问题是否仍调用 `get_current_time`
- 数学问题是否仍调用 `calculate`
- 普通闲聊是否未被 RAG 干扰

---

## 17. 风险与规避

- **依赖冲突**：新增依赖后先锁定可用组合并做最小回归。
- **启动变慢**：通过索引缓存和延迟初始化控制。
- **错误降级缺失**：所有 RAG 异常必须转为可解释文本。
- **检索偏移**：通过 RRF + 参数调优 + 测试集持续修正。

---

## 18. FAQ

### Q1：为什么一期不用 Neo4j/Milvus？

当前项目目标是低侵入快速落地，先以本地 FAISS 跑通核心价值，再按收益引入图增强。

### Q2：HowToCook 和 What-to-eat-today 的关系？

`HowToCook` 是知识数据源，`What-to-eat-today` 是实现参考，二者角色不同。

### Q3：RAG 会不会影响现有 Agent？

不会。RAG 作为工具可开关接入，且失败可降级，不应阻断主流程。

---

## 19. 执行建议（推荐顺序）

1. 先实现 `data_loader + chunking + index_store`。
2. 再实现 `retriever + service`。
3. 接入工具与 prompt。
4. 打通 CLI，再打通 Web。
5. 最后做参数调优与测试补齐。

---

## 20. 一句话总结

本方案以“一期轻量可用、二期图增强演进”为主线，确保在当前 `cook-proj` 中快速落地 RAG，并保留后续升级到图 RAG 的清晰路径。
# wx-langchain RAG 大型实施方案（整合版）

## 1. 项目目标

在当前项目中以最小侵入方式接入 RAG 能力，通过工具化检索支持 `chapter8` 文档问答，并可扩展至任意 Markdown 知识库目录。

核心目标：

- 不改动现有 CLI/Web 交互主流程。
- 保持 `create_agent` 架构不变，通过工具扩展能力。
- 采用可解释、可追踪、可降级的工程实现。
- 首次建索引，后续复用本地缓存。

---

## 2. 总体设计思路

### 2.1 架构原则

- **工具化接入**：新增 `search_knowledge_base`，由 Agent 决策是否调用。
- **小块检索，大块生成**：使用 Parent/Child 结构提升检索与回答质量。
- **双路检索融合**：向量检索 + BM25，使用 RRF 融合排序。
- **工程可用性优先**：支持索引缓存、日志、降级、配置化。

### 2.2 现有接入点

- `src/agents/dialog_agent.py`：Agent 组装与系统提示词。
- `src/actions/basic_tools.py`：工具注册与工具实现。
- `src/main.py`：CLI 启动和调用链。
- `src/web/app.py`：Web 服务启动和调用链。
- `src/config/settings.py`：配置加载与校验。

### 2.3 预计技术栈

| 层级 | 技术/组件 | 预计用途 |
|---|---|---|
| 语言与运行时 | Python 3.10+ | 主体业务实现、脚本与服务运行 |
| 核心应用框架 | LangChain / wx-langchain 现有 Agent 架构 | 工具化接入 RAG、保持现有 Agent 编排方式 |
| LLM 接入 | 项目现有大模型提供方（与当前 `create_agent` 一致） | 对话生成、检索增强回答 |
| 向量表示 | `sentence-transformers` + `BAAI/bge-small-zh-v1.5` | 中文语义向量化 |
| 向量检索 | `FAISS`（`faiss-cpu`） | 本地向量索引构建、持久化与查询 |
| 稀疏检索 | `rank-bm25` / `BM25Retriever` | 关键词召回，补充语义检索 |
| 文本处理 | `langchain-text-splitters` | Parent/Child 分块与切片策略实现 |
| 数据源 | Markdown 文档目录（如 `chapter8`） | RAG 知识库输入 |
| 接口形态 | CLI + Web（现有 `src/main.py` / `src/web/app.py`） | 统一对外交互入口 |
| 配置管理 | `Settings` + 环境变量（`.env`） | RAG 开关、路径、检索参数与模型参数管理 |
| 日志与可观测性 | Python `logging`（可扩展到结构化日志） | 初始化、检索、降级路径可追踪 |
| 测试与质量 | `pytest`（建议） | 核心模块单测、回归与可用性验证 |

---

## 3. 目标目录结构

建议新增：

```text
src/rag/
├─ __init__.py
├─ types.py
├─ config.py
├─ data_loader.py
├─ chunking.py
├─ index_store.py
├─ retriever.py
├─ generation_router.py
└─ service.py
```

职责划分：

- `types.py`：RAG 核心数据结构定义。
- `config.py`：从 `Settings` 映射 RAG 配置并校验。
- `data_loader.py`：Markdown 文档加载与元数据增强。
- `chunking.py`：Parent/Child 分块与映射维护。
- `index_store.py`：Embedding + FAISS 构建、保存、加载。
- `retriever.py`：BM25 + 向量检索 + RRF 融合。
- `generation_router.py`：查询路由与回答模式。
- `service.py`：统一初始化、检索、回答入口。

---

## 4. 配置方案（Settings 扩展）

建议新增环境变量：

- `RAG_ENABLED=true`
- `RAG_SOURCE_DIRS=chapter8`
- `RAG_INDEX_DIR=data/rag_index`
- `RAG_TOP_K=4`
- `RAG_RETRIEVAL_K=12`
- `RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`
- `RAG_CHUNK_SIZE=800`
- `RAG_CHUNK_OVERLAP=120`
- `RAG_RRF_K=60`

配置规则：

- `RAG_SOURCE_DIRS` 支持逗号分隔目录。
- 启动时统一转换绝对路径并做存在性检查。
- 数值参数做最小值保护。
- 若 `chunk_overlap >= chunk_size` 自动修正并记录告警。
- `RAG_ENABLED=false` 时全链路降级，不影响原有对话。

---

## 5. 数据准备与分块策略

### 5.1 文档加载

- 递归扫描 `source_dirs` 下的 `*.md`。
- 每篇文档生成一个 `parent_id`。
- Parent 元数据最小集合：
  - `source`
  - `title`
  - `category`
  - `parent_id`
  - `doc_type=parent`

### 5.2 Parent/Child 分块

- Parent：完整文档（用于回答上下文）。
- Child：小块文档（用于检索召回）。
- 推荐流程：
  1. 先按 Markdown 标题（`#`, `##`, `###`）拆段。
  2. 再按长度切块（`chunk_size/chunk_overlap`）。
- Child 元数据最小集合：
  - `child_id`
  - `parent_id`
  - `chunk_index`
  - `doc_type=child`

### 5.3 映射结构

- `parent_id -> parent_document`
- `child_id -> parent_id`

---

## 6. 索引与检索策略

### 6.1 向量检索

- Embedding：`HuggingFaceEmbeddings`
- 向量库：`FAISS`
- 支持：
  - 首次构建并保存到 `RAG_INDEX_DIR`
  - 后续启动加载索引缓存

### 6.2 稀疏检索

- 使用 `BM25Retriever` 对 child chunks 建立稀疏检索器。

### 6.3 融合检索（RRF）

- 分别执行向量检索和 BM25 检索。
- 使用 RRF 融合排序，兼顾语义匹配与关键词匹配。
- 将融合后的 child 命中映射回 parent，并去重排序。
- 最终返回 `top_k` 个 parent 用于生成。

---

## 7. 生成策略与路由

查询路由分三类：

- `list`：推荐/列表类问题，返回条目式结果。
- `detail`：步骤/做法类问题，返回结构化详细内容。
- `general`：概念/解释类问题，返回概述型答案。

优化策略：

- `list`、`detail` 可保持原 query。
- `general` 可选 query rewrite（先规则化，后续可升级到 LLM）。

---

## 8. 服务层统一编排（RAGService）

建议接口：

- `initialize(force_rebuild: bool = False) -> None`
- `retrieve(query: str) -> RetrievalResult`
- `answer(query: str) -> AnswerResult`

初始化流程：

1. 配置校验。
2. 文档加载与分块。
3. 索引加载或构建并保存。
4. 检索器初始化。
5. 就绪状态标记。

异常策略：

- 初始化失败不应导致主程序退出。
- 检索失败返回可解释降级文本。
- 全程记录日志并带关键上下文。

---

## 9. 工具化接入设计

改造 `src/actions/basic_tools.py`：

- 新增注入函数：`set_rag_service(service)`。
- 新增工具函数：`search_knowledge_base(query: str)`。
- 更新 `get_actions()` 返回 RAG 工具。

建议日志：

- `[RAG_CALL] query=...`
- `[RAG_HIT] vec=... bm25=... parent=...`
- `[RAG_RESULT] sources=...`
- `[RAG_ERROR] ...`

---

## 10. 启动链路接入

### 10.1 CLI（`src/main.py`）

- 启动加载 settings 后构建 RAG config。
- 若 `RAG_ENABLED=true`，初始化 `RAGService` 并注入工具层。
- 初始化失败仅 warning，不中断 CLI 会话。

### 10.2 Web（`src/web/app.py`）

- 在应用启动阶段执行与 CLI 同等初始化。
- 确保 `/api/chat` 可在 RAG 失败时继续服务。

---

## 11. Agent 提示词策略

改造 `src/agents/dialog_agent.py` 的 system prompt：

- 遇到文档/章节/实现细节问题，优先调用 `search_knowledge_base`。
- 时间问题优先 `get_current_time`。
- 数学问题优先 `calculate`。
- 无把握时先检索后作答，减少幻觉。

---

## 12. 依赖清单

`requirements.txt` 增加：

- `langchain-community`
- `langchain-text-splitters`
- `faiss-cpu`
- `sentence-transformers`
- `rank-bm25`

---

## 13. 分阶段实施计划

### Phase A（MVP）

1. 扩展 `Settings` 与环境变量。
2. 新建 `src/rag` 基础模块。
3. 接入工具层并在 CLI 跑通。

### Phase B（可用版）

1. 接入 Web 初始化链路。
2. 完成索引缓存闭环。
3. 完善日志与异常降级。

### Phase C（优化版）

1. 路由策略与 query rewrite 优化。
2. 参数调优（chunk/top_k/retrieval_k/rrf_k）。
3. 增加评测样本并做回归测试。

---

## 14. 文件级开发清单

### 14.1 `src/config/settings.py`

- [ ] 扩展 RAG 字段。
- [ ] 增加布尔/整数/CSV 解析辅助函数。
- [ ] 完成数值容错与路径规范化。

### 14.2 `src/rag/types.py`

- [ ] 定义 `RAGConfig`、`RetrievalResult`、`AnswerResult` 等结构。

### 14.3 `src/rag/config.py`

- [ ] `build_rag_config(settings)`。
- [ ] `validate_rag_config(config)`。

### 14.4 `src/rag/data_loader.py`

- [ ] `load_documents(source_dirs)`。
- [ ] `_build_parent_metadata(...)`。
- [ ] `_infer_category(...)`。

### 14.5 `src/rag/chunking.py`

- [ ] `build_parent_child(parents)`。
- [ ] 标题切分 + 长度切分 + 映射维护。

### 14.6 `src/rag/index_store.py`

- [ ] `load()` / `build()` / `save()` / `as_retriever()`。

### 14.7 `src/rag/retriever.py`

- [ ] `_vector_search(...)`。
- [ ] `_bm25_search(...)`。
- [ ] `_rrf_merge(...)`。
- [ ] `hybrid_search(...)`。

### 14.8 `src/rag/generation_router.py`

- [ ] `route_query(...)`。
- [ ] `rewrite_query(...)`。
- [ ] `build_answer(...)`。

### 14.9 `src/rag/service.py`

- [ ] `initialize(...)`。
- [ ] `retrieve(...)`。
- [ ] `answer(...)`。

### 14.10 `src/actions/basic_tools.py`

- [ ] `set_rag_service(...)`。
- [ ] `search_knowledge_base(...)`。
- [ ] 更新 `get_actions()`。

### 14.11 `src/agents/dialog_agent.py`

- [ ] 更新 system prompt 的工具调用策略。

### 14.12 `src/main.py` / `src/web/app.py`

- [ ] 启动初始化 RAG 并注入工具。
- [ ] 降级兜底与日志记录。

### 14.13 `README.md` / `.env.example`

- [ ] 增加 RAG 使用说明、配置说明、常见问题。

---

## 15. 函数级骨架（接口建议）

> 下列为建议接口，便于快速实现与多人协作。

- `rag/config.py`
  - `build_rag_config(settings) -> RAGConfig`
  - `validate_rag_config(cfg) -> None`
- `rag/data_loader.py`
  - `load_documents(source_dirs) -> list[Document]`
- `rag/chunking.py`
  - `build_parent_child(parents) -> (child_docs, parent_map, child_to_parent)`
- `rag/index_store.py`
  - `load() -> bool`
  - `build(chunks) -> None`
  - `save() -> None`
- `rag/retriever.py`
  - `hybrid_search(query, retrieval_k, top_k) -> RetrievalResult`
- `rag/generation_router.py`
  - `route_query(query) -> str`
  - `rewrite_query(query, route) -> str`
  - `build_answer(query, route, parents) -> str`
- `rag/service.py`
  - `initialize(force_rebuild=False) -> None`
  - `retrieve(query) -> RetrievalResult`
  - `answer(query) -> AnswerResult`

---

## 16. 验收标准（Definition of Done）

满足以下即视为完成：

1. 文档问题可触发 `search_knowledge_base` 并返回正确答案。
2. 回答可附带来源（至少文件路径/文件名）。
3. 日志可见完整调用链路（CALL -> HIT -> RESULT）。
4. CLI 与 Web 均可正常使用。
5. `RAG_ENABLED=false` 时系统可无缝降级。
6. 热启动较冷启动明显提速。

---

## 17. 最小测试集

- `chapter8 里 RAG 的核心模块有哪些？`
- `为什么要做 parent/child 分块？`
- `向量检索和 BM25 各自的作用是什么？`
- `RRF 在这个系统里解决了什么问题？`
- 非文档问题（如天气、闲聊）验证工具触发是否合理。

---

## 18. 回归测试矩阵

### 功能

- [ ] 文档问答准确性
- [ ] 工具调用稳定性
- [ ] 来源返回正确性

### 性能

- [ ] 冷启动构建耗时
- [ ] 热启动加载耗时
- [ ] 单次问答平均耗时

### 鲁棒性

- [ ] 空目录/坏文件/编码异常
- [ ] 依赖缺失时的提示与降级
- [ ] 索引损坏时的重建流程

---

## 19. 风险与规避

- 依赖兼容风险：优先锁定当前项目可用版本组合。
- 首次建库耗时：启动日志提示进度并持久化索引。
- 中文检索效果波动：Embedding 模型配置化，可快速替换。
- 工具误触发：通过系统提示词和返回格式约束进行收敛。

---

## 20. 常见问题排查（FAQ）

- `ModuleNotFoundError: langchain_community`
  - 安装 `langchain-community`。
- `No module named faiss`
  - 安装 `faiss-cpu` 并检查平台兼容。
- 检索结果为空
  - 检查 `RAG_SOURCE_DIRS` 路径是否正确，文档是否存在。
- 工具不触发
  - 调整 system prompt，明确文档问题优先走 RAG 工具。

---

## 21. 执行建议（推荐节奏）

Day 1：

1. 配置扩展 + types/config/data_loader。
2. chunking/index_store/retriever，完成可检索闭环。

Day 2：

1. generation_router/service。
2. 工具接入 + CLI 跑通。

Day 3：

1. Web 接入 + 文档更新 + 回归测试。
2. 调参与问题收敛。

---

## 22. 总结

该方案在不破坏现有项目结构的前提下，给出从配置、数据、索引、检索、生成到接入和验收的完整工程路径。建议先按清单实现 MVP，再通过路由、重写和参数调优逐步提升效果与稳定性。

---

## 23. 伪代码方案（可直接映射实现）

本节提供模块级伪代码，目标是让你按“接口 -> 流程 -> 关键分支”快速落地编码。

### 23.1 `settings.py` 配置扩展伪代码

```python
def parse_bool(raw: str, default: bool) -> bool:
    if raw is None or raw.strip() == "":
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def parse_int(raw: str, default: int, min_value: int) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(value, min_value)


def parse_dirs(raw: str) -> list[str]:
    # "chapter8,docs,knowledge/base" -> ["chapter8", "docs", "knowledge/base"]
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts


def load_settings() -> Settings:
    # 1) 读取现有 OpenAI 配置
    # 2) 读取 RAG 环境变量
    # 3) 做数值校验和路径标准化
    # 4) overlap >= chunk_size 时自动修正
    # 5) 返回完整 Settings
    pass
```

### 23.2 `rag/config.py` 映射伪代码

```python
def build_rag_config(settings: Settings) -> RAGConfig:
    cfg = RAGConfig(
        enabled=settings.rag_enabled,
        source_dirs=to_abs_paths(settings.rag_source_dirs),
        index_dir=to_abs_path(settings.rag_index_dir),
        embedding_model=settings.rag_embedding_model,
        top_k=max(1, settings.rag_top_k),
        retrieval_k=max(settings.rag_top_k, settings.rag_retrieval_k),
        chunk_size=max(100, settings.rag_chunk_size),
        chunk_overlap=max(0, settings.rag_chunk_overlap),
        rrf_k=max(1, settings.rag_rrf_k),
    )
    if cfg.chunk_overlap >= cfg.chunk_size:
        cfg.chunk_overlap = cfg.chunk_size // 4
    return cfg
```

### 23.3 `rag/data_loader.py` 伪代码

```python
class MarkdownDataLoader:
    def load_documents(self, source_dirs: list[str]) -> list[Document]:
        docs = []
        for source_dir in source_dirs:
            if not exists(source_dir):
                log_warning("dir_not_found", source_dir)
                continue

            for md_file in rglob(source_dir, "*.md"):
                text = safe_read_utf8(md_file)
                if not text:
                    continue

                parent_id = uuid4()
                metadata = {
                    "source": abs_path(md_file),
                    "title": stem(md_file),
                    "category": infer_category(md_file, source_dir),
                    "parent_id": parent_id,
                    "doc_type": "parent",
                }
                docs.append(Document(page_content=text, metadata=metadata))

        log_info("loaded_docs", len(docs))
        return docs
```

### 23.4 `rag/chunking.py` 伪代码（Parent/Child）

```python
class ParentChildChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def build_parent_child(self, parents: list[Document]):
        child_docs = []
        parent_map = {}       # parent_id -> parent_doc
        child_to_parent = {}  # child_id -> parent_id

        header_splitter = MarkdownHeaderTextSplitter(headers=["#", "##", "###"])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        for parent_doc in parents:
            parent_id = parent_doc.metadata["parent_id"]
            parent_map[parent_id] = parent_doc

            header_blocks = header_splitter.split_text(parent_doc.page_content)
            chunk_index = 0
            for block in header_blocks:
                pieces = text_splitter.split_text(block.page_content)
                for piece in pieces:
                    child_id = uuid4()
                    metadata = copy(parent_doc.metadata)
                    metadata.update({
                        "doc_type": "child",
                        "child_id": child_id,
                        "parent_id": parent_id,
                        "chunk_index": chunk_index,
                    })

                    child_doc = Document(page_content=piece, metadata=metadata)
                    child_docs.append(child_doc)
                    child_to_parent[child_id] = parent_id
                    chunk_index += 1

        return child_docs, parent_map, child_to_parent
```

### 23.5 `rag/index_store.py` 伪代码（索引缓存）

```python
class VectorIndexStore:
    def __init__(self, model_name: str, index_dir: str):
        self.model_name = model_name
        self.index_dir = index_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = None

    def load(self) -> bool:
        if not exists(self.index_dir):
            return False
        self.vectorstore = FAISS.load_local(self.index_dir, self.embeddings)
        return True

    def build(self, child_docs: list[Document]) -> None:
        texts = [d.page_content for d in child_docs]
        metas = [d.metadata for d in child_docs]
        self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metas)

    def save(self) -> None:
        ensure_dir(self.index_dir)
        self.vectorstore.save_local(self.index_dir)
```

### 23.6 `rag/retriever.py` 伪代码（BM25 + Vector + RRF）

```python
class HybridRetriever:
    def __init__(self, vectorstore, chunks, parent_map, rrf_k=60):
        self.vectorstore = vectorstore
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.parent_map = parent_map
        self.rrf_k = rrf_k

    def hybrid_search(self, query: str, retrieval_k: int, top_k: int) -> RetrievalResult:
        # 1) 双路召回
        vec_docs = vector_retrieve(self.vectorstore, query, k=retrieval_k)
        bm25_docs = bm25_retrieve(self.bm25, query, k=retrieval_k)

        # 2) RRF 融合排序
        merged_child_docs = self.rrf_merge(vec_docs, bm25_docs, k=self.rrf_k)

        # 3) 子块映射回父文档并去重
        parent_scores = {}  # parent_id -> {hit_count, score}
        for rank, child_doc in enumerate(merged_child_docs):
            parent_id = child_doc.metadata["parent_id"]
            score = 1 / (rank + 1)
            update_parent_score(parent_scores, parent_id, score)

        ranked_parent_ids = sort_by_hit_and_score(parent_scores)[:top_k]
        parent_docs = [self.parent_map[pid] for pid in ranked_parent_ids]

        # 4) 封装结果
        return build_retrieval_result(
            query=query,
            children=merged_child_docs[:retrieval_k],
            parents=parent_docs,
            debug={
                "vec_hits": len(vec_docs),
                "bm25_hits": len(bm25_docs),
                "top_k": top_k,
                "retrieval_k": retrieval_k,
            }
        )

    def rrf_merge(self, vec_docs, bm25_docs, k: int):
        # score(doc) = Σ 1 / (k + rank + 1)
        # 返回融合后的去重文档序列
        pass
```

### 23.7 `rag/generation_router.py` 伪代码

```python
class GenerationRouter:
    def route_query(self, query: str) -> str:
        if is_list_query(query):
            return "list"
        if is_detail_query(query):
            return "detail"
        return "general"

    def rewrite_query(self, query: str, route: str) -> str:
        if route in {"list", "detail"}:
            return query
        # 初期规则重写，后期可替换为 LLM 改写
        return normalize_query(query)

    def build_answer(self, query: str, route: str, parent_docs: list[Document]) -> str:
        if not parent_docs:
            return "未检索到相关资料，请尝试更具体的问题。"

        if route == "list":
            return format_list_answer(parent_docs)
        if route == "detail":
            return format_step_by_step_answer(parent_docs[0])
        return format_general_answer(parent_docs)
```

### 23.8 `rag/service.py` 伪代码（总编排）

```python
class RAGService:
    def __init__(self, rag_config: RAGConfig):
        self.cfg = rag_config
        self.loader = MarkdownDataLoader()
        self.chunker = ParentChildChunker(rag_config.chunk_size, rag_config.chunk_overlap)
        self.index_store = VectorIndexStore(rag_config.embedding_model, rag_config.index_dir)
        self.router = GenerationRouter()
        self.retriever = None
        self.ready = False

    def initialize(self, force_rebuild=False):
        if not self.cfg.enabled:
            log_info("rag_disabled")
            return

        parents = self.loader.load_documents(self.cfg.source_dirs)
        children, parent_map, _ = self.chunker.build_parent_child(parents)

        loaded = False
        if not force_rebuild:
            loaded = self.index_store.load()
        if not loaded:
            self.index_store.build(children)
            self.index_store.save()

        self.retriever = HybridRetriever(
            vectorstore=self.index_store.vectorstore,
            chunks=children,
            parent_map=parent_map,
            rrf_k=self.cfg.rrf_k,
        )
        self.ready = True
        log_info("rag_ready")

    def answer(self, query: str) -> AnswerResult:
        if not self.cfg.enabled:
            return AnswerResult(query=query, route="disabled", answer="RAG 未启用。")
        if not self.ready:
            return AnswerResult(query=query, route="not_ready", answer="RAG 尚未初始化完成。")

        route = self.router.route_query(query)
        rewritten = self.router.rewrite_query(query, route)
        retrieval = self.retriever.hybrid_search(rewritten, self.cfg.retrieval_k, self.cfg.top_k)
        answer = self.router.build_answer(query, route, retrieval.parents)

        return AnswerResult(
            query=query,
            route=route,
            answer=answer,
            sources=retrieval.sources,
            debug=retrieval.debug,
        )
```

### 23.9 `actions/basic_tools.py` 伪代码（工具注入）

```python
_rag_service = None


def set_rag_service(service):
    global _rag_service
    _rag_service = service


@tool
def search_knowledge_base(query: str) -> str:
    log_info("[RAG_CALL]", query=query)

    if _rag_service is None:
        return "知识检索服务未初始化。"

    try:
        result = _rag_service.answer(query)
        log_info("[RAG_RESULT]", route=result.route, sources=result.sources)
        return build_tool_text(result.answer, result.sources)
    except Exception as exc:
        log_error("[RAG_ERROR]", error=str(exc))
        return f"知识检索失败：{exc}"
```

### 23.10 `main.py` 与 `web/app.py` 启动伪代码

```python
def bootstrap_rag(settings: Settings):
    rag_cfg = build_rag_config(settings)
    rag_service = RAGService(rag_cfg)
    rag_service.initialize(force_rebuild=False)
    set_rag_service(rag_service)
    return rag_service


def app_startup():
    settings = load_settings()
    try:
        if settings.rag_enabled:
            bootstrap_rag(settings)
    except Exception as exc:
        # 降级运行，不阻塞主服务
        log_warning("rag_bootstrap_failed", error=str(exc))
```

### 23.11 端到端执行伪代码

```python
user_query -> agent.invoke(messages)
    -> agent decides tool call?
        -> yes: search_knowledge_base(query)
            -> rag_service.answer(query)
                -> route_query(query)
                -> rewrite_query_if_needed(query)
                -> hybrid_retrieval(vector + bm25 + rrf)
                -> child_to_parent_dedup
                -> build_answer_by_route
            -> tool returns "answer + sources"
        -> no: normal llm answer
    -> final assistant response
```

> 实施建议：先按伪代码跑通最小链路（不做复杂 prompt），再逐步替换路由器和回答模板。

---

## 24. 逐文件开发任务拆解（函数级 TODO）

说明：

- 优先按 `P0 -> P1 -> P2` 顺序开发。
- 每个文件先完成“接口骨架 + 最小可运行”，再补优化。
- 所有函数先写 docstring 和异常语义，再写实现。

### 24.1 `src/config/settings.py`

**P0（必须）**

- [ ] 扩展 `Settings` 字段：
  - [ ] `rag_enabled: bool`
  - [ ] `rag_source_dirs: list[str]`
  - [ ] `rag_index_dir: str`
  - [ ] `rag_top_k: int`
  - [ ] `rag_retrieval_k: int`
  - [ ] `rag_embedding_model: str`
  - [ ] `rag_chunk_size: int`
  - [ ] `rag_chunk_overlap: int`
  - [ ] `rag_rrf_k: int`
  - [ ] `rag_rebuild: bool`
- [ ] 新增解析函数：
  - [ ] `_parse_bool(name: str, default: bool) -> bool`
  - [ ] `_parse_int(name: str, default: int, minimum: int) -> int`
  - [ ] `_parse_csv(name: str, default: str) -> list[str]`
  - [ ] `_normalize_paths(paths: list[str]) -> list[str]`
- [ ] 在 `load_settings()` 中注入上述字段并做容错。

**P1（建议）**

- [ ] 为路径字段增加存在性检查与 warning 日志（不直接抛错）。

---

### 24.2 `src/rag/types.py`

**P0（必须）**

- [ ] 定义 `RAGConfig` dataclass（与 settings 字段一一对应）。
- [ ] 定义 `RetrievalResult`：
  - [ ] `query: str`
  - [ ] `parents: list[Document]`
  - [ ] `sources: list[str]`
  - [ ] `debug: dict`
- [ ] 定义 `AnswerResult`：
  - [ ] `query: str`
  - [ ] `route: str`
  - [ ] `answer: str`
  - [ ] `sources: list[str]`
  - [ ] `debug: dict`

**P1（建议）**

- [ ] 定义轻量错误类型：`RAGNotReadyError`、`RAGConfigError`。

---

### 24.3 `src/rag/config.py`

**P0（必须）**

- [ ] 实现 `build_rag_config(settings: Settings) -> RAGConfig`
- [ ] 实现 `validate_rag_config(cfg: RAGConfig) -> None`
- [ ] 实现 `sanitize_rag_config(cfg: RAGConfig) -> RAGConfig`
  - [ ] 修正 `chunk_overlap >= chunk_size`。
  - [ ] 修正 `retrieval_k < top_k`。

**P1（建议）**

- [ ] 实现 `describe_rag_config(cfg: RAGConfig) -> dict`（用于启动日志）。

---

### 24.4 `src/rag/data_loader.py`

**P0（必须）**

- [ ] 实现类 `MarkdownDataLoader`。
- [ ] 实现 `scan_markdown_files(source_dirs: list[str]) -> list[Path]`。
- [ ] 实现 `load_documents(source_dirs: list[str]) -> list[Document]`。
- [ ] 实现 `_build_parent_metadata(path: Path, content: str) -> dict`：
  - [ ] `source`
  - [ ] `title`
  - [ ] `category`
  - [ ] `parent_id`
  - [ ] `doc_type=parent`

**P1（建议）**

- [ ] 实现 `_infer_category(path: Path) -> str`（按目录映射分类）。
- [ ] 实现 `_extract_title(content: str, fallback: str) -> str`。

**P2（优化）**

- [ ] 增加文件过滤：超大文件、空文件、非法编码兜底。

---

### 24.5 `src/rag/chunking.py`

**P0（必须）**

- [ ] 实现类 `ParentChildChunker`。
- [ ] 实现 `build_parent_child(parents: list[Document]) -> tuple[list[Document], dict[str, Document], dict[str, str]]`。
- [ ] 实现 `_split_by_markdown_headers(text: str) -> list[str]`。
- [ ] 实现 `_split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]`。
- [ ] 实现 `_make_child_doc(parent: Document, text: str, idx: int) -> Document`。

**P1（建议）**

- [ ] 在 child metadata 增加：
  - [ ] `child_id`
  - [ ] `parent_id`
  - [ ] `chunk_index`
  - [ ] `doc_type=child`

---

### 24.6 `src/rag/index_store.py`

**P0（必须）**

- [ ] 实现类 `LocalFAISSIndexStore`。
- [ ] 实现 `build(children: list[Document]) -> None`。
- [ ] 实现 `save() -> None`。
- [ ] 实现 `load() -> bool`。
- [ ] 实现 `exists() -> bool`。

**P1（建议）**

- [ ] 实现 `compute_signature(...) -> str`。
- [ ] 实现 `save_signature(signature: str) -> None`。
- [ ] 实现 `is_signature_match(signature: str) -> bool`。

**P2（优化）**

- [ ] 增加 `health_check() -> dict`（向量条数、模型名、签名）。

---

### 24.7 `src/rag/retriever.py`

**P0（必须）**

- [ ] 实现类 `HybridRetriever`。
- [ ] 实现 `vector_search(query: str, k: int) -> list[Document]`。
- [ ] 实现 `bm25_search(query: str, k: int) -> list[Document]`。
- [ ] 实现 `rrf_fuse(vector_docs: list[Document], bm25_docs: list[Document], rrf_k: int) -> list[Document]`。
- [ ] 实现 `child_to_parent(fused_children: list[Document], top_k: int) -> list[Document]`。
- [ ] 实现 `hybrid_search(query: str, retrieval_k: int, top_k: int) -> RetrievalResult`。

**P1（建议）**

- [ ] 实现 `_collect_sources(parents: list[Document]) -> list[str]`。
- [ ] 实现 `_build_debug_info(...) -> dict`（命中数、耗时、分数）。

---

### 24.8 `src/rag/generation_router.py`

**P0（必须）**

- [ ] 实现类 `GenerationRouter`。
- [ ] 实现 `route_query(query: str) -> str`（`list/detail/general`）。
- [ ] 实现 `rewrite_query(query: str, route: str) -> str`（先规则版）。
- [ ] 实现 `build_answer(query: str, route: str, parents: list[Document]) -> str`。

**P1（建议）**

- [ ] 实现 `_build_list_answer(...)`。
- [ ] 实现 `_build_detail_answer(...)`。
- [ ] 实现 `_build_general_answer(...)`。

**P2（优化）**

- [ ] 增加 LLM 路由开关（默认关闭，仅实验时启用）。

---

### 24.9 `src/rag/service.py`

**P0（必须）**

- [ ] 实现类 `RAGService`。
- [ ] 实现 `initialize(force_rebuild: bool = False) -> None`：
  - [ ] 配置校验
  - [ ] 文档加载与分块
  - [ ] 索引加载/重建
  - [ ] 检索器初始化
  - [ ] `ready=True`
- [ ] 实现 `retrieve(query: str) -> RetrievalResult`。
- [ ] 实现 `answer(query: str) -> AnswerResult`。
- [ ] 实现 `is_ready() -> bool`。

**P1（建议）**

- [ ] 实现 `stats() -> dict`（文档数、chunk 数、索引状态）。
- [ ] 实现 `close() -> None`（预留资源回收）。

---

### 24.10 `src/actions/basic_tools.py`

**P0（必须）**

- [ ] 增加模块级变量 `_rag_service: Optional[RAGService] = None`。
- [ ] 实现 `set_rag_service(service: RAGService) -> None`。
- [ ] 新增工具函数：
  - [ ] `search_knowledge_base(query: str) -> str`
- [ ] 更新 `get_actions()`，注册新工具。

**P1（建议）**

- [ ] 实现 `_format_rag_tool_result(answer: str, sources: list[str]) -> str`。

---

### 24.11 `src/agents/dialog_agent.py`

**P0（必须）**

- [ ] 更新 system prompt：
  - [ ] 文档/章节/做法/教程问题优先 `search_knowledge_base`
  - [ ] 时间问题优先 `get_current_time`
  - [ ] 数学问题优先 `calculate`

**P1（建议）**

- [ ] 在 `debug=True` 下输出路由提示信息（仅日志，不暴露给用户）。

---

### 24.12 `src/main.py`

**P0（必须）**

- [ ] 在启动流程增加 `bootstrap_rag(settings)`。
- [ ] 调用 `set_rag_service(rag_service)` 注入工具层。
- [ ] RAG 初始化失败时降级运行（不中断 CLI）。

**P1（建议）**

- [ ] 启动时打印 RAG 状态摘要（enabled/ready/index_loaded）。

---

### 24.13 `src/web/app.py`

**P0（必须）**

- [ ] 在应用启动阶段执行与 CLI 同等 RAG 初始化。
- [ ] 确保失败降级不影响 `/api/chat`。

**P1（建议）**

- [ ] 新增只读状态接口（可选）：
  - [ ] `GET /api/rag/status`

---

### 24.14 `requirements.txt`

**P0（必须）**

- [ ] 增加依赖：
  - [ ] `langchain-community`
  - [ ] `langchain-text-splitters`
  - [ ] `faiss-cpu`
  - [ ] `sentence-transformers`
  - [ ] `rank-bm25`

**P1（建议）**

- [ ] 二期候选依赖单独注释分组，避免一期强依赖污染。

---

### 24.15 测试文件建议（新增）

**P0（必须）**

- [ ] `tests/rag/test_config.py`
  - [ ] 配置解析与容错测试
- [ ] `tests/rag/test_chunking.py`
  - [ ] 父子映射与切块稳定性测试
- [ ] `tests/rag/test_retriever.py`
  - [ ] RRF 融合与去重测试

**P1（建议）**

- [ ] `tests/integration/test_rag_tool_flow.py`
  - [ ] 工具调用到回答输出的端到端链路

---

### 24.16 建议开发顺序（可直接照做）

- [ ] Step 1：`settings.py` + `rag/types.py` + `rag/config.py`
- [ ] Step 2：`rag/data_loader.py` + `rag/chunking.py`
- [ ] Step 3：`rag/index_store.py` + `rag/retriever.py`
- [ ] Step 4：`rag/generation_router.py` + `rag/service.py`
- [ ] Step 5：`basic_tools.py` + `dialog_agent.py`
- [ ] Step 6：`main.py` + `web/app.py`
- [ ] Step 7：`requirements.txt` + `tests/*`

