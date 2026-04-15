# 对话列表与历史消息展示改造方案

---

## 一、背景

当前项目的 Web 端已具备以下基础能力：

- 会话列表展示
- 新建会话
- 切换会话
- 发送消息（SSE 流式）
- 后端持久化 `conversations` / `messages`

但前端交互仍存在两个明显问题：

1. **缺少删除会话能力**
   - 前端每个会话对应一个 `session_id`
   - 当前没有删除按钮，也没有删除接口
   - 用户无法清理历史对话

2. **默认恢复会话时未展示完整历史**
   - 页面初始化时只恢复了最近的 `session_id`
   - 但没有加载该会话的 `messages`
   - 导致用户刷新页面后，界面中看不到该会话的完整用户问题与助手回答

本方案目标是：**补齐会话删除闭环，并统一前端历史消息加载逻辑，让对话恢复行为与切换行为一致。**

---

## 二、现状分析

### 2.1 前端会话列表现状

当前 `src/web/static/index.html` 中的会话列表只支持点击切换：

- `renderConvList(conversations)` 负责渲染列表项
- 每个 `conv-item` 只绑定 `switchSession(c.session_id)`

没有：

- 删除按钮
- 删除确认
- 删除后切换逻辑

### 2.2 后端接口现状

当前 `src/controllers/chat_controller.py` 已有：

- `POST /api/conversations`：新建会话
- `GET /api/conversations`：列出会话
- `GET /api/conversations/{session_id}/messages`：获取会话消息

没有：

- `DELETE /api/conversations/{session_id}`：删除会话

### 2.3 数据库现状

当前 `src/db/schema.sql` 中：

- `conversations.session_id` 是主键
- `messages.session_id` 外键引用 `conversations(session_id)`
- 并配置了 `ON DELETE CASCADE`

即：

```sql
session_id  TEXT NOT NULL REFERENCES conversations(session_id) ON DELETE CASCADE
```

这意味着：

- **只要删除 `conversations` 表中的一条会话记录**
- **该会话对应的所有 `messages` 会自动被数据库级联删除**

因此不需要额外手工删除消息表。

### 2.4 历史消息展示现状

当前 `switchSession(newId)` 会调用：

- `GET /api/conversations/{session_id}/messages`
- 并逐条执行 `appendMessage(m.role, m.content)`

这说明：

- 切换会话时，历史消息展示逻辑是存在的

但在页面初始化 `init()` 中：

- 若已有历史会话，只是设置了 `sessionId`
- 显示了一条 `"Welcome back. Resumed last conversation."`
- **没有调用 `switchSession()`**

因此页面首次进入时不会真正加载历史消息。

---

## 三、改造目标

本次改造分为两个目标：

### 目标 1：支持删除会话

要求：

- 前端会话列表支持删除某条会话
- 删除动作应有确认提示
- 后端提供删除接口
- 删除 `conversations` 后自动级联删除 `messages`
- 若删除的是当前会话，前端需要自动切换到其他会话或创建新会话

### 目标 2：默认恢复时显示完整历史

要求：

- 页面初始化若已有历史会话，应自动调用与“切换会话”一致的消息加载逻辑
- 页面展示中应包含用户问题与助手回答
- 不再插入伪造的 assistant 欢迎语污染真实历史

---

## 四、涉及文件

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| 修改 | `src/db/conversations.py` | 增加删除会话 DAO |
| 修改 | `src/controllers/chat_controller.py` | 增加删除会话接口 |
| 修改 | `src/web/static/index.html` | 增加删除按钮、删除逻辑、初始化历史加载逻辑 |
| 无需修改 | `src/db/schema.sql` | 已有 `ON DELETE CASCADE`，可直接复用 |
| 无需修改 | `src/db/messages.py` | 已有按 `session_id` 查询消息能力 |

---

## 五、后端改造方案

### 5.1 `src/db/conversations.py`：新增删除 DAO

#### 目标

新增一个删除会话的方法：

- 输入：`session_id`
- 输出：是否成功删除

#### 建议新增函数

```python
def delete_conversation(session_id: str) -> bool:
    pool = get_postgres_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE session_id = %s",
                (session_id,),
            )
            deleted = cur.rowcount > 0
            conn.commit()
    return deleted
```

#### 说明

- `DELETE conversations` 即可
- `messages` 会通过数据库外键自动级联删除
- 返回 `bool` 方便 controller 决定是否返回 `404`

---

### 5.2 `src/controllers/chat_controller.py`：新增删除接口

#### 目标

提供标准 RESTful 删除接口：

- `DELETE /api/conversations/{session_id}`

#### 建议新增接口

```python
@router.delete("/conversations/{session_id}")
def remove_conversation(session_id: str) -> dict:
    session_id = session_id.strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    ok = delete_conversation(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="conversation not found")

    return {"ok": True}
```

#### 需要新增 import

在文件顶部导入：

```python
from db.conversations import delete_conversation
```

#### 接口语义

- `400`：路径参数为空
- `404`：该 `session_id` 不存在
- `200`：删除成功

---

## 六、前端改造方案

### 6.1 会话列表增加删除按钮

#### 当前问题

当前列表项只有点击切换，没有操作按钮。

#### 改造目标

每个会话项应包含：

- 左侧：标题 + 更新时间
- 右侧：删除按钮

#### 建议结构

把当前：

```html
<div class="conv-item">
  <div>标题</div>
  <div class="conv-time">时间</div>
</div>
```

调整为类似：

```html
<div class="conv-item">
  <div class="conv-main">
    <div>标题</div>
    <div class="conv-time">时间</div>
  </div>
  <button class="conv-delete-btn">Delete</button>
</div>
```

#### 注意事项

- 删除按钮点击时必须 `stopPropagation()`
- 否则会同时触发父元素的 `switchSession()`

---

### 6.2 新增删除会话前端方法

#### 建议新增函数

```javascript
async function deleteConversationById(id) {
  try {
    const res = await fetch(`/api/conversations/${id}`, {
      method: "DELETE"
    });
    return res.ok;
  } catch {
    return false;
  }
}
```

#### 交互建议

点击删除按钮后：

1. 弹确认框
2. 若确认，则调用删除接口
3. 删除成功后刷新会话列表
4. 若删的是当前会话，自动切换或新建

#### 建议删除确认逻辑

```javascript
if (!confirm("Delete this conversation?")) return;
```

---

### 6.3 删除当前会话后的前端行为

这是整个交互里最需要定义清楚的一点。

#### 建议规则

### 情况 A：删除的是非当前会话

- 不改变当前聊天窗口
- 只刷新列表

### 情况 B：删除的是当前会话

删除成功后：

1. 重新拉取会话列表
2. 若仍有剩余会话：
   - 切换到列表第一条会话
3. 若没有任何剩余会话：
   - 自动调用 `handleNewChat()`

#### 建议实现逻辑

```javascript
const deletedCurrent = id === sessionId;
const ok = await deleteConversationById(id);
if (!ok) {
  appendMessage("assistant", "Error: failed to delete conversation.");
  return;
}

const list = await fetchConversations();
renderConvList(list);

if (deletedCurrent) {
  if (list.length > 0) {
    await switchSession(list[0].session_id);
  } else {
    await handleNewChat();
  }
}
```

---

### 6.4 初始化时改为加载真实历史

#### 当前问题

初始化时只设置 `sessionId`，但没有加载消息：

- 没有调用 `switchSession()`
- 反而插入了一条假的 assistant 提示

#### 当前逻辑的问题表现

页面刷新后，用户只看到：

- `"Welcome back. Resumed last conversation."`

而不是该会话真正的：

- 用户问题
- 助手回答

#### 正确目标

页面初始化时，若已有历史会话：

- 直接走 `switchSession(firstSessionId)`

这样可以统一复用：

- 拉取历史消息
- 渲染 user / assistant 消息
- 刷新列表状态

#### 建议修改

当前：

```javascript
(async function init() {
  const conversations = await fetchConversations();
  if (conversations.length > 0) {
    sessionId = conversations[0].session_id;
    renderConvList(conversations);
    sessionLabel.textContent = sessionId.slice(0, 8) + "...";
    appendMessage("assistant", "Welcome back. Resumed last conversation.");
  } else {
    await handleNewChat();
  }
  messageInput.focus();
})();
```

建议改为：

```javascript
(async function init() {
  const conversations = await fetchConversations();
  if (conversations.length > 0) {
    renderConvList(conversations);
    await switchSession(conversations[0].session_id);
  } else {
    await handleNewChat();
  }
  messageInput.focus();
})();
```

#### 说明

- 不要手动 `appendMessage("assistant", "Welcome back...")`
- 历史消息应只展示真实消息

---

### 6.5 消息展示行为说明

当前 `sendMessage()` 已经会在用户发送当下执行：

```javascript
appendMessage("user", message);
```

因此：

- 实时发送阶段，用户消息本来就是会显示的

真正缺失的是：

- **页面重新加载后**
- **默认恢复旧会话时**

没有把该会话历史回放出来。

所以本次改造不需要调整：

- `appendMessage("user", message)`

只需要修复初始化恢复逻辑即可。

---

## 七、前端样式建议

### 新增样式项

为了支持删除按钮，建议给 `index.html` 增加样式：

- `.conv-item`
  - 改为 `display: flex`
  - `justify-content: space-between`
  - `align-items: center`

- `.conv-main`
  - 容纳标题和时间
  - `min-width: 0`

- `.conv-delete-btn`
  - 小号按钮
  - hover 高亮为危险色

#### 示例方向

```css
.conv-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.conv-main {
  flex: 1;
  min-width: 0;
}

.conv-delete-btn {
  border: none;
  background: transparent;
  color: #64748b;
  cursor: pointer;
  font-size: 12px;
}

.conv-delete-btn:hover {
  color: #f87171;
}
```

---

## 八、完整数据流

### 8.1 删除会话

```text
前端点击删除
  → confirm 确认
  → DELETE /api/conversations/{session_id}
    → chat_controller.remove_conversation()
      → db.conversations.delete_conversation(session_id)
        → DELETE FROM conversations WHERE session_id = ?
          → messages 通过 ON DELETE CASCADE 自动删除
  → 前端拉取最新会话列表
  → 若删除的是当前会话：
       ├─ 有剩余会话 → switchSession(first)
       └─ 无剩余会话 → handleNewChat()
```

### 8.2 页面初始化恢复历史

```text
页面加载
  → GET /api/conversations
  → 若有会话：
       → renderConvList(conversations)
       → switchSession(first_session_id)
           → GET /api/conversations/{session_id}/messages
           → appendMessage("user", ...)
           → appendMessage("assistant", ...)
  → 若无会话：
       → POST /api/conversations
       → 创建新会话
```

---

## 九、实现顺序建议

### 第一阶段：先修复历史消息展示

原因：

- 改动小
- 风险低
- 用户体验提升立竿见影

执行步骤：

1. 修改 `init()`，改为调用 `switchSession()`
2. 删除 `"Welcome back. Resumed last conversation."`

### 第二阶段：补删除会话闭环

执行步骤：

1. `conversations.py` 增加 `delete_conversation`
2. `chat_controller.py` 增加 `DELETE /api/conversations/{session_id}`
3. `index.html` 增加删除按钮与交互
4. 处理删除当前会话后的自动切换/新建

---

## 十、验收清单

### 历史消息展示

- 页面首次加载已有历史会话时，能看到完整 user / assistant 历史消息
- 切换会话后，能正确看到该会话完整历史
- 页面中不再出现伪造的 `"Welcome back..."` assistant 消息

### 删除会话

- 每个会话列表项都可删除
- 删除非当前会话时，当前窗口不跳转
- 删除当前会话时，自动切换到剩余第一条或新建会话
- 删除后刷新页面，被删会话不再出现

### 数据一致性

- 删除某条 `session_id` 后，`conversations` 中对应记录消失
- `messages` 中该 `session_id` 的所有消息同步消失
- 无孤儿消息残留

---

## 十一、补充建议

### 1. 删除按钮可以后续换成更多操作菜单

若未来会话操作变多（如重命名、置顶、导出），建议把右侧按钮升级为：

- `...` 菜单

当前阶段直接用 `Delete` / `×` 即可。

### 2. 可考虑增加“软删除”而不是物理删除

如果未来你有以下需求：

- 恢复已删除对话
- 审计
- 回收站

可把删除改成：

- `conversations.deleted_at`

当前项目还不需要，物理删除更简单直接。

### 3. 可增加删除保护

比如：

- 当会话只有一条时，删除后直接创建新会话
- 避免界面出现空白状态

---

## 十二、结论

本次方案不涉及复杂架构改造，主要是补齐现有对话系统的 UI 与 API 闭环：

- 历史消息展示问题：**前端初始化未复用 `switchSession()`**
- 删除会话问题：**缺少前端删除入口 + 后端删除接口**

当前数据库已经具备 `ON DELETE CASCADE`，因此实现成本较低，推荐优先完成。

