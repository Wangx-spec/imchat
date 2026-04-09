# 重构方案：接口独立 + 去 langchain 化

## 概述

本文档对应 `memo.md` 中的两项待办：

1. 将接口从 webapp 中独立出来
2. 去 langchain 化

---

## 任务 1：将接口从 webapp 中独立出来

### 1.1 现状问题

`src/web/app.py`（282 行）是一个混合文件，包含：

- FastAPI 路由定义（HTTP 层）
- Agent 调用逻辑（`_invoke_chat`、流式处理）
- 结果提取工具函数（`_extract_text_from_content` 等）
- 防幻觉后处理（`_sanitize_ungrounded_kb_claim`）
- 模块级初始化（settings / agent / RAG bootstrap）

同时 `src/main.py` 与 `src/web/app.py` 存在大量**重复函数**：

- `_extract_text_from_content`
- `_extract_tool_calls_from_message`
- `_extract_tool_calls`
- `_extract_text_from_result`

### 1.2 目标

- HTTP 路由层只负责参数校验与响应封装；
- Agent 调用、结果提取、后处理集中到一个 service 层；
- CLI 和 Web 共享同一套 service，消除重复代码。

### 1.3 拆分方案

采用类似 Spring Boot 的分层结构：**controller（路由层）+ service（业务层）+ app（组装层）**。

```text
src/
├─ controllers/                # 路由层（类比 Spring Boot @RestController）
│  ├─ __init__.py
│  ├─ chat_controller.py       # /api/chat, /api/chat/stream
│  └─ system_controller.py     # /health, /api/reset, /（首页）
├─ services/                   # 业务层（类比 Spring Boot @Service）
│  ├─ __init__.py
│  └─ chat_service.py          # Agent 调用 + 结果提取 + 后处理
├─ web/
│  └─ app.py                   # 应用组装（类比 Spring Boot Application）
└─ main.py                     # CLI 入口（调用 chat_service）
```

设计理念对照：

| Spring Boot | 本项目（FastAPI） |
|---|---|
| `@RestController` + `@RequestMapping` | `controllers/*.py` + `APIRouter` |
| `@Service` | `services/chat_service.py` |
| `Application.java` + `@SpringBootApplication` | `web/app.py` + `create_app()` |
| `application.yml` | `config/settings.py` + `.env` |

### 1.4 各文件职责

#### `src/controllers/chat_controller.py`

纯 HTTP 层，只负责参数校验、调用 service、封装响应：

```python
from fastapi import APIRouter
from services.chat_service import invoke_chat, stream_chat

router = APIRouter(prefix="/api", tags=["chat"])

@router.post("/chat")              # 同步对话
@router.post("/chat/stream")       # SSE 流式对话
```

#### `src/controllers/system_controller.py`

系统级接口：

```python
from fastapi import APIRouter

router = APIRouter(tags=["system"])

@router.get("/health")             # 健康检查
@router.post("/api/reset")         # 会话重置（兼容接口）
@router.get("/")                   # 首页静态文件
```

#### `src/services/chat_service.py`

从 `web/app.py` + `main.py` 提取并去重，CLI 和 Web 共享：

```python
# 结果提取（去重后统一）
def extract_text_from_content(content) -> str
def extract_tool_calls_from_message(message) -> list[dict]
def extract_tool_calls(result: dict) -> list[dict]
def extract_text_from_result(result: dict) -> str

# 防幻觉后处理（从 web/app.py 移入）
def has_kb_tool_call(tool_calls: list[dict]) -> bool
def sanitize_ungrounded_kb_claim(answer: str, tool_calls: list[dict]) -> str

# Agent 调用（统一入口）
def invoke_chat(agent, session_id: str, message: str) -> tuple[str, list[dict]]
def stream_chat(agent, session_id: str, message: str) -> Generator
```

#### `src/web/app.py`（精简后）

只负责组装，类似 Spring Boot 的 Application 入口：

```python
from controllers.chat_controller import router as chat_router
from controllers.system_controller import router as system_router

def create_app() -> FastAPI:
    app = FastAPI(title="Chat UI")
    # startup: settings / agent / RAG bootstrap
    app.include_router(chat_router)
    app.include_router(system_router)
    return app

app = create_app()
```

#### `src/main.py`（精简后）

删除所有重复的 `_extract_*` 函数，直接调用 `chat_service`：

```python
from services.chat_service import invoke_chat, stream_chat

def run_chat():
    # ... bootstrap ...
    while True:
        answer, tool_calls = invoke_chat(agent, session_id, user_input)
        # 或 stream_chat(...)
```

### 1.5 请求流转示意

```text
HTTP Request
  -> controllers/chat_controller.py    参数校验、响应封装
    -> services/chat_service.py        Agent 调用、结果提取、防幻觉
      -> graphs/dialog_graph.py        LangGraph Agent 执行
        -> actions/*                   工具调用（KB/时间/计算）
```

### 1.6 迁移步骤

1. 创建 `src/controllers/__init__.py`、`src/services/__init__.py`；
2. 创建 `src/services/chat_service.py`，将公共函数从 `web/app.py` 和 `main.py` 移入；
3. 创建 `src/controllers/chat_controller.py`，将 `/api/chat` 和 `/api/chat/stream` 路由移入；
4. 创建 `src/controllers/system_controller.py`，将 `/health`、`/api/reset`、`/` 移入；
5. 精简 `web/app.py` 为 `create_app()` 工厂 + `include_router`；
6. 精简 `main.py`，改为调用 `chat_service`；
7. 验证 CLI + Web 功能正常。

---

## 任务 2：去 langchain 化

### 2.1 当前 langchain 依赖全景

| 包 | 引用位置 | 用途 |
|---|---|---|
| `langchain` | `agents/dialog_agent.py` | `create_agent`（仅 langchain fallback 路径） |
| `langchain_core` | `actions/basic_tools.py`、`actions/knowledge_base_tools.py` | `@tool` 装饰器 |
| `langchain_core` | `rag/data_loader.py`、`rag/chunking.py`、`rag/retriever.py`、`rag/index_store.py`、`rag/generation_router.py` | `Document` 数据类 |
| `langchain_core` | `memory/session_memory.py` | `HumanMessage/AIMessage`（已废弃未引用） |
| `langchain_openai` | `llms/openai_chat.py`、`rag/query_planner.py`、`rag/index_store.py` | `ChatOpenAI`、`OpenAIEmbeddings` |
| `langchain_community` | `rag/index_store.py` | `FAISS` 向量库封装 |
| `langchain_community` | `rag/retriever.py` | `BM25Retriever` |
| `langgraph` | `graphs/dialog_graph.py` | `create_react_agent`、`MemorySaver` |

### 2.2 关键约束

`langgraph` 的 `create_react_agent` 内部要求：

- model 是 `BaseChatModel`（来自 `langchain_core`）
- tools 是 `BaseTool`（来自 `langchain_core`）

因此**只要继续使用 `create_react_agent`，就无法完全移除 `langchain_core`**。

### 2.3 分阶段方案

#### 第一阶段：删除显式 langchain 依赖（低风险）

| 改动 | 文件 | 说明 |
|---|---|---|
| 删除 langchain fallback 路径 | `src/agents/dialog_agent.py` | 移除 `build_dialog_agent_langchain()`、`from langchain.agents import create_agent`，`build_dialog_runtime` 只保留 langgraph 路径 |
| 删除废弃的会话内存文件 | `src/memory/session_memory.py` | 已无引用，可直接删除 |
| 移除 `requirements.txt` 中的 `langchain>=0.3.0` | `requirements.txt` | 仅 fallback 用到，删除后不影响主链路 |

改动后 `dialog_agent.py` 变为：

```python
from graphs.dialog_graph import build_dialog_graph
from config.settings import Settings

def build_dialog_runtime(settings: Settings):
    return build_dialog_graph(settings), "langgraph"
```

可移除的依赖：`langchain>=0.3.0`

#### 第二阶段：RAG 层去 langchain 化（中等风险）

| 改动 | 替代方案 |
|---|---|
| `langchain_core.documents.Document` | 自定义 `dataclass`（如 `rag/types.py` 中新增 `DocChunk`），字段兼容 `page_content + metadata` |
| `langchain_openai.OpenAIEmbeddings` | 直接用 `openai` SDK 的 `client.embeddings.create(...)` |
| `langchain_community.vectorstores.FAISS` | 直接用 `faiss` 原生 API（`faiss.IndexFlatIP` + 手动管理 id 映射） |
| `langchain_community.retrievers.BM25Retriever` | 直接用 `rank_bm25.BM25Okapi`（已在 requirements 中） |
| `langchain_openai.ChatOpenAI` | 直接用 `openai` SDK 的 `client.chat.completions.create(...)` |

涉及文件：

- `src/rag/types.py`：新增 `DocChunk` dataclass
- `src/rag/data_loader.py`：返回 `DocChunk` 替代 `Document`
- `src/rag/chunking.py`：同上
- `src/rag/index_store.py`：用原生 faiss + openai SDK
- `src/rag/retriever.py`：用原生 BM25Okapi + 适配 `DocChunk`
- `src/rag/generation_router.py`：接收 `DocChunk`
- `src/rag/query_planner.py`：用 openai SDK
- `src/llms/openai_chat.py`：用 openai SDK

可移除的依赖：`langchain-openai`、`langchain-community`

新增依赖：`openai`

#### 第三阶段：自建 ReAct 替代 langgraph（高风险，可选）

- 自己实现 tool 调度循环（ReAct loop）
- 自己实现消息协议与 checkpointer 接口
- 移除 `langgraph` 和 `langchain_core`

> 此阶段代价最大、收益边际递减，建议仅在有明确需求时推进。

### 2.4 各阶段依赖变化

```text
当前 requirements.txt:
  langchain>=0.3.0              # 第一阶段移除
  langgraph>=0.2.0              # 保留（第三阶段才可移除）
  langchain-openai>=0.2.0       # 第二阶段移除（换 openai SDK）
  langchain-core>=0.3.0         # 保留（langgraph 隐式依赖）
  langchain-community>=0.3.0    # 第二阶段移除
  faiss-cpu>=1.10.0             # 保留（第二阶段改为直接调用）
  rank_bm25>=0.2.0              # 保留（第二阶段改为直接调用）
  python-dotenv>=1.0.1          # 保留
  fastapi>=0.115.0              # 保留
  uvicorn>=0.32.0               # 保留
  numpy>=1.26.0                 # 保留

第二阶段完成后新增：
  openai>=1.0.0                 # 替代 langchain-openai
```

---

## 建议执行顺序

```text
任务1（接口独立）
  └─> 任务2 第一阶段（删 langchain fallback + 废弃文件）
        └─> 任务2 第二阶段（RAG 层去 langchain 化）
              └─> [可选] 任务2 第三阶段（自建 ReAct）
```

理由：
- 任务 1 是纯重构，不涉及依赖变更，风险最低；
- 拆完后各模块边界清晰，任务 2 的改动更集中、更安全；
- 第一阶段立竿见影（删一个文件 + 简化一个函数）；
- 第二阶段逐文件替换，可分多次 PR 推进。

---

## 验收标准

- [ ] `controllers/` 目录包含 `chat_controller.py` 和 `system_controller.py`
- [ ] `services/chat_service.py` 包含所有公共业务逻辑
- [ ] `web/app.py` 精简到 ≤40 行，只含 app 组装
- [ ] `main.py` 无重复的 `_extract_*` 函数，直接调用 service
- [ ] CLI + Web 功能回归通过
- [ ] `langchain>=0.3.0` 从 `requirements.txt` 移除（第一阶段）
- [ ] `memory/session_memory.py` 已删除（第一阶段）
- [ ] RAG 不依赖 `langchain-community`（第二阶段）
- [ ] LLM 调用不依赖 `langchain-openai`（第二阶段）
