# 新增 messages 表存储纯文本对话记录

## 1. 背景

当前对话内容由 LangGraph 的 `PostgresSaver` 以二进制 checkpoint 形式存储，用于 agent 上下文恢复。但 checkpoint 数据不可读、不方便查询和展示。

本方案在 checkpoint 之外，**额外维护一张 `messages` 表**，以纯文本形式记录每轮 user / assistant 消息，用于：

- 前端切换对话时加载历史消息
- 对话列表显示标题（自动取首条消息前 30 字符）
- 未来的审计、导出、搜索

两条线互不干扰：

| 层 | 存什么 | 谁管 | 用途 |
|---|---|---|---|
| **checkpoint 表** | 完整 graph 状态（二进制） | LangGraph 自动 | 上下文恢复、追问 |
| **messages 表** | 纯文本对话记录 | 业务代码 | UI 展示、审计、导出 |

---

## 2. 改动范围

```
改动文件（按依赖顺序）：
──────────────────────────────────────────
1. src/db/schema.sql                   ← 追加 messages 表 DDL
2. src/db/messages.py                  ← 新建：消息 CRUD
3. src/services/chat_service.py        ← invoke/stream 前后写入消息
4. src/controllers/chat_controller.py  ← 新增消息查询接口
5. src/main.py                         ← CLI 对话循环写入消息
6. src/web/static/index.html           ← 切换对话时加载历史消息
```

---

## 3. DDL — `src/db/schema.sql` 追加

```sql
CREATE TABLE IF NOT EXISTS messages (
    id          SERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,          -- 'user' / 'assistant'
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id, created_at);
```

> 需在数据库中手动执行一次，或通过启动脚本自动执行。

---

## 4. 消息 CRUD — `src/db/messages.py`（新建）

```python
from __future__ import annotations
import logging
from datetime import datetime
from db.connection import get_postgres_pool

logger = logging.getLogger(__name__)


def save_message(session_id: str, role: str, content: str) -> None:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
            conn.commit()


def list_messages(session_id: str, limit: int = 200) -> list[dict]:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, session_id, role, content, created_at
                   FROM messages
                   WHERE session_id = %s
                   ORDER BY created_at ASC
                   LIMIT %s""",
                (session_id, limit),
            )
            return [_row_to_dict(r) for r in cur.fetchall()]


def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "session_id": row[1],
        "role": row[2],
        "content": row[3],
        "created_at": row[4].isoformat() if isinstance(row[4], datetime) else row[4],
    }
```

---

## 5. 改动 `src/services/chat_service.py`

### 5.1 新增 import

```python
from db.messages import save_message
from db.conversations import update_conversation, update_title
```

### 5.2 新增辅助函数 `_maybe_set_title()`

```python
def _maybe_set_title(session_id: str, user_message: str) -> None:
    """首条消息时自动设置对话标题（取前 30 字符）。"""
    try:
        from db.messages import list_messages
        msgs = list_messages(session_id, limit=2)
        if len(msgs) <= 2:
            title = user_message[:30].strip()
            if title:
                update_title(session_id, title)
    except Exception:
        pass
```

### 5.3 改动 `invoke()` 函数

```python
def invoke(session_id: str, message: str) -> tuple[str, list[dict]]:
    save_message(session_id, "user", message)
    result = _agent.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": session_id}},
    )
    answer = extract_text_from_result(result)
    tool_calls = extract_tool_calls(result)
    save_message(session_id, "assistant", answer)
    update_conversation(session_id)
    _maybe_set_title(session_id, message)
    return answer, tool_calls
```

### 5.4 改动 `stream()` 函数

在函数体最前面（`try` 之前）加入 `save_message(session_id, "user", message)`。

在 `finally` 块中加入：

```python
    finally:
        if latest_answer:
            save_message(session_id, "assistant", latest_answer)
            update_conversation(session_id)
            _maybe_set_title(session_id, message)
            logger.info("[CHAT_RESULT] session=%s answer=%s", session_id, latest_answer)
```

---

## 6. 改动 `src/controllers/chat_controller.py`

追加一个 GET 接口，供前端按 session_id 查询历史消息：

```python
from db.messages import list_messages

@router.get("/conversations/{session_id}/messages")
def get_messages(session_id: str) -> list[dict]:
    return list_messages(session_id)
```

---

## 7. 改动 `src/main.py`

### 7.1 新增 import

```python
from db.messages import save_message
from db.conversations import update_conversation, update_title
```

### 7.2 对话循环 `try` 块内改动

在 agent 调用前写入 user 消息，返回后写入 assistant 消息：

```python
        try:
            save_message(session_id, "user", user_input)

            # ... 原有 invoke/stream 逻辑不变 ...

            save_message(session_id, "assistant", answer)
            update_conversation(session_id)

            # 自动设置对话标题
            from db.messages import list_messages
            msgs = list_messages(session_id, limit=2)
            if len(msgs) <= 2:
                title = user_input[:30].strip()
                if title:
                    update_title(session_id, title)

            _log_tool_calls(tool_calls)
            # ...
```

---

## 8. 改动 `src/web/static/index.html`

修改 `switchSession` 函数，切换对话时从后端加载历史消息：

```javascript
    async function switchSession(newId) {
      sessionId = newId;
      sessionLabel.textContent = sessionId ? sessionId.slice(0, 8) + "..." : "";
      chatBox.innerHTML = "";

      try {
        const res = await fetch(`/api/conversations/${sessionId}/messages`);
        if (res.ok) {
          const messages = await res.json();
          for (const m of messages) {
            appendMessage(m.role, m.content);
          }
        }
      } catch {}

      if (!chatBox.children.length) {
        appendMessage("assistant", "Switched to conversation " + sessionId.slice(0, 8) + "...");
      }
      messageInput.focus();
      await refreshConvList();
    }
```

---

## 9. 数据流

```
用户发消息
  │
  ├─ save_message(session_id, "user", message)     ← 写 messages 表
  │
  ├─ _agent.invoke / .stream(...)                   ← LangGraph 自动写 checkpoint 表
  │
  ├─ save_message(session_id, "assistant", answer)  ← 写 messages 表
  │
  ├─ update_conversation(session_id)                ← 更新 conversations.updated_at
  │
  └─ _maybe_set_title(session_id, message)          ← 首条消息自动设标题
```

---

## 10. 实施步骤

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| **1** | 在 PG 中执行 messages 表 DDL | `\d messages` 确认表结构 |
| **2** | 新建 `src/db/messages.py` | `python -c "from db.messages import save_message"` |
| **3** | 改 `chat_service.py` | 发消息后 `SELECT * FROM messages` 确认有记录 |
| **4** | 加 controller 接口 | `curl GET /api/conversations/{id}/messages` |
| **5** | 改 `main.py` | CLI 发消息后查数据库确认 |
| **6** | 改前端 `switchSession` | 切换对话看到历史消息 |

---

## 11. 注意事项

1. **不替换 checkpoint**：messages 表是 checkpoint 的补充，不是替代。LangGraph 上下文恢复仍依赖 checkpoint。
2. **写入时机**：user 消息在 agent 调用前写入，assistant 消息在收到完整回复后写入，保证时序正确。
3. **自动标题**：首条消息后自动取前 30 字符作为 title，提升对话列表辨识度。
4. **性能**：每轮多两次 INSERT，对 PG 来说开销极小。
5. **ON DELETE CASCADE**：删除 conversation 时自动清理关联消息。
