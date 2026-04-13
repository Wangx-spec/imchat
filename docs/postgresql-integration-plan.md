# 接入 PostgreSQL 方案（单用户阶段）

## 1. 背景与目标

当前系统使用 `MemorySaver`（纯内存 checkpointer），服务重启后所有会话状态丢失。本方案将：

1. **会话持久化**：用 `PostgresSaver` 替换 `MemorySaver`，重启不丢对话上下文；
2. **多对话管理**：每次"新建对话"生成独立 `session_id`，支持对话列表、切换、重置；
3. **预留多用户扩展**：schema 中预留 `user_id` 字段，当前固定为 `default-user`。

---

## 2. 改动范围

```
改动文件（按依赖顺序）：
──────────────────────────────────────────
1. requirements.txt                    ← 新增依赖
2. .env                                ← 新增 PG 连接串
3. src/config/settings.py              ← 读取 PG 配置
4. src/db/__init__.py                  ← 新建：包初始化
5. src/db/connection.py                ← 新建：连接池管理
6. src/db/schema.sql                   ← 新建：DDL
7. src/db/conversations.py             ← 新建：对话 CRUD
8. src/graphs/dialog_graph.py          ← MemorySaver → PostgresSaver
9. src/controllers/chat_controller.py  ← 新增对话管理接口
10. src/services/chat_service.py       ← 对话创建/列表逻辑
11. src/main.py                        ← CLI 会话管理改造
12. src/web/static/index.html          ← 前端多对话 UI
```

---

## 3. 新增依赖

`requirements.txt` 追加：

```
langgraph-checkpoint-postgres>=2.0.0
psycopg[binary]>=3.1.0
```

- `langgraph-checkpoint-postgres`：LangGraph 官方 PostgreSQL checkpointer，内部使用 `psycopg3`；
- `.setup()` 自动建 checkpoint 表，不需要手动建。

---

## 4. 环境配置

### 4.1 `.env` 新增

```bash
# PostgreSQL
POSTGRES_URI=postgresql://cook:cook123@localhost:5432/cook_proj
```

### 4.2 Docker 快速起 PG（可选）

```bash
docker run -d \
  --name cook-pg \
  -e POSTGRES_USER=cook \
  -e POSTGRES_PASSWORD=cook123 \
  -e POSTGRES_DB=cook_proj \
  -p 5432:5432 \
  postgres:16-alpine
```

---

## 5. Settings 改动

`src/config/settings.py`：

### 5.1 `Settings` dataclass 新增字段

```python
@dataclass
class Settings:
    # ... 现有字段 ...
    postgres_uri: str | None = None       # 新增
```

### 5.2 `load_settings()` 读取

```python
postgres_uri = os.getenv("POSTGRES_URI", "").strip() or None
```

传入 `Settings(... postgres_uri=postgres_uri ...)`。

---

## 6. 数据库连接管理

新建 `src/db/__init__.py`（空文件）。

新建 `src/db/connection.py`：

```python
from __future__ import annotations
import logging
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None

def init_pool(postgres_uri: str, min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    _pool = ConnectionPool(
        conninfo=postgres_uri,
        min_size=min_size,
        max_size=max_size,
    )
    logger.info("PG connection pool created (min=%d, max=%d)", min_size, max_size)
    return _pool

def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("PG pool not initialized. Call init_pool() first.")
    return _pool
```

---

## 7. 数据库 Schema

新建 `src/db/schema.sql`。

> checkpoint 相关表由 `PostgresSaver.setup()` 自动创建（幂等），无需手动管理。

```sql
CREATE TABLE IF NOT EXISTS conversations (
    session_id   TEXT PRIMARY KEY,            -- 即 LangGraph thread_id
    user_id      TEXT NOT NULL DEFAULT 'default-user',  -- 预留多用户，当前固定值
    title        TEXT,                         -- 对话标题（取首条消息摘要）
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);
```

未来可选（RAG 索引元数据，当前阶段不需要）：

```sql
CREATE TABLE IF NOT EXISTS rag_index_meta (
    id              SERIAL PRIMARY KEY,
    source_dirs     JSONB NOT NULL,
    embedding_model TEXT NOT NULL,
    children_count  INT NOT NULL,
    built_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 8. 对话 CRUD 层

新建 `src/db/conversations.py`：

```python
from __future__ import annotations
import logging
from datetime import datetime
from db.connection import get_pool

logger = logging.getLogger(__name__)

DEFAULT_USER = "default-user"

def create_conversation(session_id: str, title: str | None = None) -> dict:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations (session_id, user_id, title)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (session_id) DO NOTHING
                   RETURNING session_id, user_id, title, created_at, updated_at""",
                (session_id, DEFAULT_USER, title),
            )
            row = cur.fetchone()
            conn.commit()
    if row is None:
        return {"session_id": session_id, "exists": True}
    return _row_to_dict(row)

def list_conversations(limit: int = 50) -> list[dict]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT session_id, user_id, title, created_at, updated_at
                   FROM conversations
                   WHERE user_id = %s
                   ORDER BY updated_at DESC
                   LIMIT %s""",
                (DEFAULT_USER, limit),
            )
            return [_row_to_dict(r) for r in cur.fetchall()]

def touch_conversation(session_id: str) -> None:
    """每条新消息时更新 updated_at。"""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET updated_at = now() WHERE session_id = %s",
                (session_id,),
            )
            conn.commit()

def update_title(session_id: str, title: str) -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET title = %s WHERE session_id = %s",
                (title, session_id),
            )
            conn.commit()

def _row_to_dict(row) -> dict:
    return {
        "session_id": row[0],
        "user_id": row[1],
        "title": row[2],
        "created_at": row[3].isoformat() if isinstance(row[3], datetime) else row[3],
        "updated_at": row[4].isoformat() if isinstance(row[4], datetime) else row[4],
    }
```

---

## 9. 核心改动：Checkpointer 替换

改动 `src/graphs/dialog_graph.py`。

### 9.1 改动前（当前代码）

```python
from langgraph.checkpoint.memory import MemorySaver

_CHECKPOINTER = MemorySaver()

def build_dialog_graph(settings: Settings) -> Any:
    llm = build_openai_chat_model(settings)
    actions = get_actions()
    return create_react_agent(
        model=llm,
        tools=actions,
        prompt=SYSTEM_PROMPT,
        debug=settings.verbose,
        checkpointer=_CHECKPOINTER
    )
```

### 9.2 改动后

```python
from typing import Any
from langgraph.prebuilt import create_react_agent
from actions.basic_tools import get_actions
from config.settings import Settings
from llms.openai_chat import build_openai_chat_model
from prompts.system_prompts import SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)

_CHECKPOINTER = None
_PG_CONN = None

def _build_checkpointer(settings: Settings):
    global _CHECKPOINTER, _PG_CONN
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    if settings.postgres_uri:
        from psycopg import Connection
        from langgraph.checkpoint.postgres import PostgresSaver
        _PG_CONN = Connection.connect(
            settings.postgres_uri,
            autocommit=True,
            prepare_threshold=0,
        )
        _CHECKPOINTER = PostgresSaver(_PG_CONN)
        _CHECKPOINTER.setup()   # 自动建 checkpoint 表（幂等）
        logger.info("Checkpointer: PostgresSaver")
    else:
        from langgraph.checkpoint.memory import MemorySaver
        _CHECKPOINTER = MemorySaver()
        logger.info("Checkpointer: MemorySaver (in-memory fallback)")
    return _CHECKPOINTER

def build_dialog_graph(settings: Settings) -> Any:
    checkpointer = _build_checkpointer(settings)
    llm = build_openai_chat_model(settings)
    actions = get_actions()
    return create_react_agent(
        model=llm,
        tools=actions,
        prompt=SYSTEM_PROMPT,
        debug=settings.verbose,
        checkpointer=checkpointer,
    )
```

设计要点：

- `PostgresSaver.from_conn_string()` 在 v3.x 是 `@contextmanager`，不能直接赋值。需用 `psycopg.Connection.connect()` 手动建连接，传给 `PostgresSaver(conn)` 构造；
- 连接设置 `autocommit=True, prepare_threshold=0` 是 LangGraph 内部要求；
- `_PG_CONN` 持有连接引用，避免被 GC 回收；
- `.setup()` 幂等，重复调用不报错；
- 未配置 `POSTGRES_URI` 时降级为 `MemorySaver`，保证向后兼容。

---

## 10. Web 层改动

### 10.1 `chat_controller.py` — 新增对话管理接口

```python
from db.conversations import (
    create_conversation,
    list_conversations,
    touch_conversation,
)
import uuid

class NewConversationResponse(BaseModel):
    session_id: str

@router.post("/conversations", response_model=NewConversationResponse)
def new_conversation() -> NewConversationResponse:
    session_id = str(uuid.uuid4())
    create_conversation(session_id)
    return NewConversationResponse(session_id=session_id)

@router.get("/conversations")
def get_conversations() -> list[dict]:
    return list_conversations()
```

### 10.2 `reset` 端点改造

```python
@router.post("/reset")
def reset(payload: ResetRequest) -> dict:
    session_id = payload.session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    new_id = str(uuid.uuid4())
    create_conversation(new_id)
    logger.info("[CHAT_RESET] old=%s new=%s", session_id, new_id)
    return {"ok": True, "new_session_id": new_id}
```

### 10.3 `chat_service.py` — invoke/stream 时更新 updated_at

```python
from db.conversations import touch_conversation, create_conversation

def invoke(session_id: str, message: str) -> tuple[str, list[dict]]:
    create_conversation(session_id)  # 幂等，首次自动创建
    result = _agent.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": session_id}},
    )
    touch_conversation(session_id)
    return extract_text_from_result(result), extract_tool_calls(result)
```

`stream()` 同理，在 `finally` 块中加 `touch_conversation(session_id)`。

### 10.4 `web/app.py` — startup 初始化连接池

```python
from db.connection import init_pool

@app.on_event("startup")
def on_startup() -> None:
    if _settings.postgres_uri:
        init_pool(_settings.postgres_uri)

    # ... 原有 RAG bootstrap 逻辑不变 ...
```

---

## 11. CLI 改动

改造 `src/main.py` 的 `run_chat()`，支持会话命令：

```python
import uuid
from db.connection import init_pool
from db.conversations import create_conversation, list_conversations

def run_chat() -> None:
    setup_logging()
    settings = load_settings()

    if settings.postgres_uri:
        init_pool(settings.postgres_uri)

    # ... 原有 RAG bootstrap ...

    dialog_runner, runtime = build_dialog_runtime(settings)

    session_id = str(uuid.uuid4())
    create_conversation(session_id)

    print(f"Chat runtime: {runtime}. Session: {session_id[:8]}...")
    print("Commands: /new (new chat) | /list (history) | /quit")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit", "/quit"}:
            print("Bye!")
            break
        if user_input == "/new":
            session_id = str(uuid.uuid4())
            create_conversation(session_id)
            print(f"New session: {session_id[:8]}...")
            continue
        if user_input == "/list":
            for c in list_conversations(limit=10):
                print(f"  {c['session_id'][:8]}  {c['title'] or '(untitled)'}  {c['updated_at']}")
            continue
        if not user_input:
            continue

        # ... 原有 invoke/stream 逻辑不变 ...
```

---

## 12. 前端改动要点

`src/web/static/index.html` JavaScript 部分：

1. 启动时调 `GET /api/conversations` 获取历史对话列表；
2. "新建对话"按钮调 `POST /api/conversations`，拿到新 `session_id`；
3. `sessionId` 不再存 localStorage 写死，而是跟随当前选中的对话动态切换；
4. "Reset"按钮调 `POST /api/reset`，用返回的 `new_session_id` 替换当前会话。

---

## 13. 启动流程（改动后）

```
服务启动
  │
  ├─ load_settings()          读取 POSTGRES_URI
  │
  ├─ init_pool(postgres_uri)  建立 PG 业务连接池
  │
  ├─ build_dialog_graph()
  │    └─ PostgresSaver.from_conn_string(...)  建立 checkpoint 专用连接
  │    └─ .setup()                             自动建 checkpoint 表
  │
  ├─ bootstrap_rag()          RAG 初始化（不变）
  │
  └─ uvicorn.run()            开始服务
```

---

## 14. 实施步骤

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| **1** | Docker 起 PG，配 `.env` | `psql` 连接确认 |
| **2** | 加依赖，改 `settings.py` | `python -c "from config.settings import load_settings; print(load_settings().postgres_uri)"` |
| **3** | 建 `db/` 模块 + 执行 `schema.sql` | 手动执行 DDL，确认表建成 |
| **4** | 改 `dialog_graph.py`（checkpointer 替换） | 启动服务，发消息，重启，追问，验证上下文恢复 |
| **5** | 加对话管理接口 + `chat_service.py` 改动 | `curl POST /api/conversations`，确认 PG 有记录 |
| **6** | 改 CLI | `/new` → `/list` 验证 |
| **7** | 改前端 | 浏览器 F12 确认接口调用正常 |

> **步骤 4 是最关键的里程碑** — 完成后即实现"服务重启不丢会话"的核心目标。步骤 5-7 是体验优化。

---

## 15. 注意事项

1. **两套连接不冲突**：`PostgresSaver` 用自己的 `psycopg` 连接（`from_conn_string`），业务 CRUD 用 `ConnectionPool`，各自管理生命周期。

2. **`ChatSessionMemory` 可清理**：`src/memory/session_memory.py` 已不在主流程中使用，接入 PG 后可删除。

3. **对话标题自动生成**：首条消息返回后，取用户消息前 30 字符作为 title，调 `update_title()` 写入，提升对话列表辨识度。

4. **checkpoint 数据量**：LangGraph 的 checkpoint 存储完整消息历史，长对话占用空间较大。后续可加定期清理策略（保留最近 N 轮）。

5. **向后兼容**：`postgres_uri` 为空时自动降级为 `MemorySaver`，不配置 PG 也能正常运行。
