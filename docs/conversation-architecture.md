# 对话架构总览

---

## 一、项目对话的基本流程

项目有 **两个入口**，分别对应 CLI 和 Web，但核心调用链一致：

```
用户输入
  │
  ├── CLI 入口 (src/main.py::run_chat)
  │     └── 直接调用 dialog_runner.invoke / .stream
  │
  └── Web 入口 (src/web/app.py → controller → chat_service)
        └── chat_service.invoke() / chat_service.stream()
              └── 调用同一个 dialog_runner（即 LangGraph ReAct Agent）
```

### 完整调用链（以 Web 端 SSE 流式为例）

```
前端 index.html
  → POST /api/chat/stream (JSON: {session_id, message})
    → chat_controller.chat_stream()
      → chat_service.stream(session_id, message)
        → save_message(user)              ← 写入用户消息到 DB
        → dialog_runner.stream()          ← LangGraph ReAct Agent
          → LLM 决策是否调用 Tool
            → search_knowledge_base()     ← RAG 知识库检索（如需要）
            → get_current_time()          ← 时间工具
            → calculate()                 ← 计算工具
          → LLM 生成最终回答
        → SSE yield chunk/done 给前端
        → save_message(assistant)         ← 写入助手消息到 DB
        → update_conversation()           ← 更新对话时间戳
        → set_title()                     ← 首轮自动设标题
```

### Agent 运行时

| 运行时 | 构建方式 | 说明 |
|--------|---------|------|
| `langgraph`（默认） | `create_react_agent()` + Checkpointer | 支持流式、自带多轮状态管理 |
| `langchain`（回退） | `create_agent()` | LangGraph 初始化失败时的降级方案 |

LLM 通过 system prompt（`src/prompts/system_prompts.py`）被指导按优先级使用三个工具：

1. `search_knowledge_base` — 文档/知识库类问题
2. `get_current_time` — 时间问题
3. `calculate` — 数学表达式

---

## 二、项目是如何保存对话的

对话持久化分 **两层**，职责不同：

### 第一层：LangGraph Checkpointer（Agent 内部多轮记忆）

- 配置了 `postgres_uri` 时，使用 `PostgresSaver` 把 LangGraph 的完整状态（含 messages 列表、tool 调用记录等）写入 PostgreSQL 的 LangGraph 专用表。
- 未配置时回退到 `MemorySaver`（纯内存，重启丢失）。
- **作用**：让 Agent 的 `invoke/stream` 在同一个 `thread_id`（即 `session_id`）下自动携带历史消息上下文，实现多轮对话。

相关代码：`src/graphs/dialog_graph.py` 中的 `_build_checkpointer()`。

### 第二层：业务层 conversations + messages 表（应用自管理）

```sql
-- conversations 表：一个 session 一行
conversations (session_id PK, user_id, title, created_at, updated_at)

-- messages 表：每条用户/助手消息一行
messages (id PK, session_id FK, role, content, created_at)
```

写入时机：

| 操作 | 触发位置 | 说明 |
|------|---------|------|
| `create_conversation` | 新建对话时（CLI `/new`、Web `POST /conversations`） | 插入 conversations 行 |
| `save_message(user)` | agent 调用**前** | 记录用户消息 |
| `save_message(assistant)` | agent 调用**后** | 记录助手回复 |
| `update_conversation` | 每轮对话结束 | 刷新 `updated_at` |
| `update_title` | 首轮（messages <= 2 条时） | 用用户输入前 30 字符作标题 |

**两层的关系**：Checkpointer 保证 Agent 运行时的多轮上下文连贯；messages 表保证前端能独立查询/展示完整的聊天历史，两者互不依赖。

---

## 三、项目如何通过问题检索知识库进行问答

当 LLM 判断用户问题属于"文档/知识库"类问题时，会调用 `search_knowledge_base` 工具，其内部流程：

```
用户问题
  │
  ▼
search_knowledge_base(query)         ← src/actions/knowledge_base_tools.py
  │
  ▼
RAGService.answer(query)             ← src/rag/service.py
  │
  ├─ 1. 路由判断 (GenerationRouter.route_query)
  │     → "detail" / "list" / "general"
  │
  ├─ 2. 查询改写 (GenerationRouter.rewrite_query)
  │
  ├─ 3. 检索 (RAGService.retrieve → HybridRetriever.hybrid_search)
  │     ├─ LLM Query Planner → 生成 normalized_query + core_terms + variants
  │     ├─ 多变体向量检索 (FAISS)
  │     ├─ 多变体 BM25 稀疏检索
  │     ├─ RRF 融合排序
  │     ├─ child → parent 映射
  │     ├─ fallback parent recall（关键词回退补召回）
  │     ├─ Qwen Rerank 重排（可选）
  │     └─ 置信度评分 (confidence_score)
  │
  ├─ 4. 置信度门控
  │     → 若 route=detail 且 confidence < 0.45 且无强证据 → 返回"证据不足"拦截回答
  │
  └─ 5. 答案生成 (GenerationRouter.build_answer)
        → 拼装文本 + 参考文档引用
```

最终结果以 JSON 字符串返回给 LLM，包含 `ok`, `answer`, `citations`, `sources` 等字段。LLM 再根据 system prompt 中的 `KB_GROUNDING_RULES` 来决定如何呈现给用户（引用来源、标注是否为知识库结果等）。

### 兜底机制

`chat_service.py` 中的 `sanitize_ungrounded_kb_claim()`：如果 LLM 回复中出现了"根据知识库"之类的话术，但实际并未调用 `search_knowledge_base` 工具，则会被拦截并加上"本次回答未调用知识库"的提示。

---

## 四、项目切分和检索文档的原理

### 4.1 文档加载

`MarkdownDataLoader`（`src/rag/data_loader.py`）：

- 递归扫描 `source_dirs` 下所有 `.md` 文件
- 每个 `.md` 文件 = 一个 **父文档（parent）**
- 提取 metadata：`source`（文件路径）、`title`（首个 `#` 标题）、`category`（目录推断）、`parent_id`（UUID）

### 4.2 文档切分（Parent-Child 模式）

`ParentChildChunker`（`src/rag/chunking.py`）：

```
Parent（完整 .md 文件）
  │
  ├─ Step 1: 按 Markdown 标题 (# / ## / ###) 切分为 sections
  │
  └─ Step 2: 每个 section 若超过 chunk_size(800字符)
              → 按 chunk_size 滑动窗口切分，overlap = 120 字符
              → 每个切片 = 一个 Child 文档
```

- **Parent** 保留完整内容，用于最终展示给用户
- **Child** 是检索单元，体积小、语义集中，用于向量/BM25 检索
- 通过 `parent_map[parent_id] → parent_doc` 和 `child_parent[child_id] → parent_id` 维护双向映射

### 4.3 索引构建

`LocalFAISSIndexStore`（`src/rag/index_store.py`）：

- 使用 DashScope `text-embedding-v4` 模型对所有 child 文档生成 1024 维向量
- 构建 FAISS 向量索引，持久化到 `data/cook_rag_index/`
- 同时保存 `index.meta.json` 签名（包含 source_dirs、embedding_model、chunk_size、children_count 等）
- 启动时校验签名一致才复用缓存索引，否则重建

### 4.4 检索流程（Hybrid Search）

`HybridRetriever.hybrid_search()`（`src/rag/retriever.py`）核心流程：

```
原始 query
  │
  ▼
LLM Query Planner (可选)
  → normalized_query, core_terms, query_variants (2~5 个变体)
  │
  ▼  对每个 variant 分别执行：
  ├── FAISS 向量检索 (top k_per)
  └── BM25 稀疏检索 (top k_per)
        │
        ▼
    RRF 融合排序 (Reciprocal Rank Fusion, k=60)
        │
        ▼
    Child → Parent 映射去重
        │
        ▼
    Fallback Parent Recall（关键词直接在 parent 标题/内容匹配补召回）
        │
        ▼
    合并去重
        │
        ▼
    Qwen Rerank 重排（可选，调用 DashScope qwen3-rerank API）
        │
        ▼
    置信度评分 → confidence_score, is_confident
        │
        ▼
    截取 top_k(4) 个 parent 返回
```

BM25 的分词策略：英文词 + 中文单字 + 中文双字（bigram），兼顾中英文场景。

---

## 五、前后端通过哪些交互控制 session_id 进行对话

### API 接口一览

| 方法 | 路径 | 功能 | session_id 来源 |
|------|------|------|----------------|
| `POST` | `/api/conversations` | 创建新对话 | 后端生成 UUID 并返回 |
| `GET` | `/api/conversations` | 获取对话列表 | — |
| `GET` | `/api/conversations/{session_id}/messages` | 获取某对话的历史消息 | URL 路径参数 |
| `POST` | `/api/chat/stream` | 发送消息（SSE 流式） | 请求体 `{session_id, message}` |
| `POST` | `/api/chat` | 发送消息（非流式） | 请求体 `{session_id, message}` |
| `POST` | `/api/reset` | 重置对话 | 请求体 `{session_id}` |

### 前端 session_id 生命周期

```
1. 页面初始化
   init()
     → fetchConversations()                    // GET /api/conversations
     → 有历史？取第一条的 session_id
     → 无历史？handleNewChat()
          → POST /api/conversations → 拿到 session_id

2. 新建对话
   newChatBtn.click → handleNewChat()
     → POST /api/conversations → 拿到新 session_id
     → 清空 chatBox，更新 sessionLabel

3. 切换对话
   convList 点击 → switchSession(newId)
     → sessionId = newId
     → GET /api/conversations/{sessionId}/messages  ← 加载历史消息
     → 逐条 appendMessage 渲染

4. 发送消息
   sendBtn.click → sendMessage()
     → POST /api/chat/stream {session_id: sessionId, message}
     → 读取 SSE stream，实时渲染 assistant 回复
     → finally: refreshConvList() 刷新侧边栏
```

**关键设计点**：`session_id` 始终由后端生成（UUID），前端只负责存储和传递，不会自行创建。这保证了 DB 中的 conversations 表和 LangGraph Checkpointer 的 `thread_id` 始终一致。

---

## 六、补充说明

### 6.1 两个入口的代码重复

`src/main.py`（CLI）和 `src/services/chat_service.py`（Web）中存在大量重复逻辑（`extract_text_from_content`, `extract_tool_calls`, 消息保存逻辑等）。`chat_service.py` 是对 `main.py` 中相同逻辑的重构提取，但 `main.py` 并未复用 `chat_service`，而是保留了自己的一套。后续逻辑变更需要同步维护两处。

### 6.2 流式 stream 中消息可能重复保存

在 `chat_service.stream()` 中，如果 streaming 没拿到答案会 fallback 调用 `invoke()`，而 `invoke()` 内部也会 `save_message`。之后 `stream()` 的 `finally` 块又会再次 `save_message`。这可能导致同一轮对话的 user/assistant 消息被写入两次。

### 6.3 数据库方言

`schema.sql` 使用 PostgreSQL 语法（`SERIAL`, `TIMESTAMPTZ`, `JSONB`），项目的连接池也是 `psycopg`（PostgreSQL 驱动）。如果要在 MySQL 上运行，需要全面改写 SQL 和连接层。

### 6.4 RAG 与 Agent 的解耦设计

RAG 并不是直接嵌入 Agent 的固定流程，而是作为一个 **Tool** 注册给 LLM。LLM 自主决定是否调用。这意味着：

- 简单的闲聊问题不会触发 RAG
- 知识库问题由 LLM 判断后主动调用 `search_knowledge_base`
- System Prompt 通过优先级引导保证知识库问题不被遗漏

### 6.5 项目文件结构（核心模块）

```
src/
├── main.py                          ← CLI 入口
├── web/
│   ├── app.py                       ← Web 入口 (FastAPI)
│   └── static/index.html            ← 前端页面
├── agents/dialog_agent.py           ← Agent 构建（langgraph / langchain）
├── graphs/dialog_graph.py           ← LangGraph ReAct Agent + Checkpointer
├── services/chat_service.py         ← Web 端对话业务逻辑
├── controllers/
│   ├── chat_controller.py           ← HTTP API 路由
│   └── system_controller.py         ← 系统状态接口
├── db/
│   ├── connection.py                ← PostgreSQL 连接池
│   ├── conversations.py             ← conversations 表 CRUD
│   ├── messages.py                  ← messages 表 CRUD
│   └── schema.sql                   ← DDL
├── actions/
│   ├── basic_tools.py               ← 工具注册（时间、计算、知识库）
│   └── knowledge_base_tools.py      ← search_knowledge_base 工具实现
├── rag/
│   ├── bootstrap.py                 ← RAG 初始化入口
│   ├── service.py                   ← RAGService（answer / retrieve）
│   ├── retriever.py                 ← HybridRetriever（向量+BM25+RRF+Rerank）
│   ├── chunking.py                  ← Parent-Child 切分
│   ├── data_loader.py               ← Markdown 文档加载
│   ├── index_store.py               ← FAISS 索引管理
│   ├── query_planner.py             ← LLM 查询规划器
│   ├── generation_router.py         ← 路由 + 答案拼装
│   ├── config.py                    ← RAG 配置
│   └── types.py                     ← 数据类型定义
├── prompts/
│   ├── system_prompts.py            ← Agent System Prompt
│   └── knowledge_base_prompt.py     ← KB 引用规则 + Query Planner Prompt
├── llms/openai_chat.py              ← LLM 客户端构建
├── memory/session_memory.py         ← 内存会话记忆（CLI 备用）
└── config/
    ├── settings.py                  ← 环境变量 → Settings 配置
    └── logging_setup.py             ← 日志配置
```
