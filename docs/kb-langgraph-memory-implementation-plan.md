# 知识库问答短期记忆改造方案（方案A + LangGraph原生记忆）

## 1. 目标与结论

本方案实现两个目标：

1. **统一链路（方案A）**：所有对话（含知识库问答）都走同一条 `LangGraph ReAct Agent` 链路，不再走 Web 层知识库强制捷径。
2. **原生短期记忆**：使用 `langgraph checkpointer + thread_id(session_id)` 保存并恢复会话状态，实现“追问上一个问题”在知识库场景下同样可用。

---

## 2. 当前问题（简述）

当前知识库问答存在“像没有短期记忆”的根因：

- `web/app.py` 在 `rag_force_tool_route` 命中时，直接 `call_kb(message)`，只传当前用户句子，不包含上下文。
- 这条旁路绕开了 Agent 的完整回合（模型推理 + 工具决策 + 消息状态演进）。
- `create_react_agent` 当前未显式配置 checkpointer，`langgraph` 原生会话状态没有启用。

---

## 3. 目标架构（改造后）

统一为：

`用户消息 -> LangGraph(agent) -> (可选) search_knowledge_base -> LLM生成最终回答 -> checkpointer按session保存 -> 下一轮按session恢复`

关键点：

- 会话键：`session_id`（映射为 `configurable.thread_id`）。
- 持久载体：`MemorySaver`（开发环境）或 `SqliteSaver/Postgres`（生产建议）。
- Web 层不再直接调用 `call_kb/forced_route`。

---

## 4. 文件级改造清单

## 4.1 `src/graphs/dialog_graph.py`

### 改造目标

- 为 `create_react_agent` 注入 checkpointer。
- 返回可复用的 graph runner（保持现有接口风格）。

### 实现建议

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
        checkpointer=_CHECKPOINTER,
    )
```

> 说明：`MemorySaver` 为进程内存级持久，服务重启会丢失。若要重启后保留，可替换为 sqlite/postgres saver。

---

## 4.2 `src/web/app.py`

### 改造目标

1. 去除“知识库强制路由旁路”；
2. 所有 invoke/stream 都带 `thread_id=session_id`；
3. 优先让 graph checkpointer 维护短期记忆；
4. 逐步下线 `ChatSessionMemory`（推荐在本次一起完成，避免双状态源）。

### 必改点 A：统一调用入口

将 `_invoke_chat(history)` 改为 `_invoke_chat(session_id, message)`：

```python
def _invoke_chat(session_id: str, message: str) -> tuple[str, list[dict]]:
    result = _agent.invoke(
        {"messages": [("user", message)]},
        config={"configurable": {"thread_id": session_id}},
    )
    return _extract_text_from_result(result), _extract_tool_calls(result)
```

### 必改点 B：流式调用同样传 thread_id

```python
for update in _agent.stream(
    {"messages": [("user", message)]},
    config={"configurable": {"thread_id": session_id}},
    stream_mode="updates",
):
    ...
```

### 必改点 C：删除 Web 层强制知识库捷径

删除以下逻辑及其依赖：

- `if _settings.rag_force_tool_route and should_force_kb(message): ...`
- `from rag.forced_route import ...`

保留系统提示词中“KB问题优先调用工具”的约束，让 Agent 自主调用 `search_knowledge_base`。

### 必改点 D：移除/简化 `_memory_map`

推荐改为只保留“session reset接口”语义：

- `/api/reset` 不再操作 `_memory_map`，改为调用 `checkpointer` 的会话清理（若后端 saver 支持）；
- 如果 `MemorySaver` 不支持按线程删，可接受“reset=前端生成新 session_id”策略。

---

## 4.3 `src/main.py`（CLI）

### 改造目标

CLI 与 Web 保持一致：也用 `thread_id` 驱动 LangGraph 记忆。

### 实现建议

- 启动时生成一个 `session_id`（例如固定 `"cli-default"`，或用户可输入）。
- `invoke/stream` 都改为：

```python
result = dialog_runner.invoke(
    {"messages": [("user", user_input)]},
    config={"configurable": {"thread_id": session_id}},
)
```

```python
for update in dialog_runner.stream(
    {"messages": [("user", user_input)]},
    config={"configurable": {"thread_id": session_id}},
    stream_mode="updates",
):
    ...
```

- `ChatSessionMemory` 可移除；若暂时保留，避免再把完整 history 传给 graph（否则会重复堆叠消息）。

---

## 4.4 `src/config/settings.py`

### 改造目标

- 为迁移期增加显式开关，降低一次性切换风险。

### 建议新增配置

- `AGENT_USE_LANGGRAPH_MEMORY=true`（默认 true）

### 建议清理配置

- 删除 `RAG_FORCE_TOOL_ROUTE`
- 删除 `RAG_FORCE_TOOL_POLISH`

如果要彻底清理旧路径，可直接删除 `rag_force_tool_route/rag_force_tool_polish` 及相关配置读取。

---

## 4.5 `src/rag/forced_route.py`

### 改造目标

- 方案A生效后，该文件逻辑不再是主路径。

### 处理建议

两种选择：

1. **平滑迁移**：保留文件但不在 Web 主链路引用；
2. **彻底清理**：删除文件及引用（推荐第二阶段进行）。

---

## 5. 迁移顺序（建议按阶段）

### 阶段1：先打通主链路

1. `dialog_graph.py` 接入 checkpointer；
2. `web/app.py` 所有调用传 `thread_id`；
3. 去掉强制 KB 旁路；
4. 保留旧配置但默认关闭。

### 阶段2：清理双记忆源

1. 移除 `_memory_map/ChatSessionMemory`；
2. CLI 同步改造；
3. 清理 forced_route 相关代码。

### 阶段3：持久化增强（可选）

- 将 `MemorySaver` 升级为 sqlite/postgres saver，支持重启后会话恢复。

---

## 6. 核心验收用例

## 6.1 多轮知识库追问

1. 用户：`宫保鸡丁怎么做？`
2. 用户：`这个菜大概几人份？`
3. 用户：`上一步火候再详细一点`

预期：

- 第二、三轮能正确继承第一轮上下文实体“宫保鸡丁”；
- Tool 调用日志显示 `search_knowledge_base` 在需要时被调用；
- 回答不再因“只看当前句子”而丢主题。

## 6.2 非知识库连续对话不回归

1. 用户：`现在几点`
2. 用户：`再说一遍`

预期：`get_current_time` 调用与短期上下文行为正常。

## 6.3 Session 隔离

- 同时创建 `sessionA/sessionB`，分别问不同菜谱；
- 交叉追问不应串会话内容。

---

## 7. 风险与规避

- **风险1：重复消息**  
  若同时传全量 history 且启用 checkpointer，会导致上下文重复。  
  **规避**：启用原生记忆后，每轮只传当前用户消息。

- **风险2：reset 语义变化**  
  原来 reset 清空 `_memory_map`，现在需通过“换 session_id”或后端删除 thread state。  
  **规避**：前端 reset 生成新 session_id，后端兼容保留接口。

- **风险3：重启丢失会话**（MemorySaver）  
  **规避**：生产切 sqlite/postgres saver。

---

## 8. 完成定义（Definition of Done）

- [ ] Web 普通/流式接口全部带 `thread_id=session_id`；
- [ ] Web 层无 KB 强制旁路；
- [ ] graph 已配置 checkpointer；
- [ ] KB 多轮追问通过；
- [ ] 多 session 隔离通过；
- [ ] 无重复消息堆叠；
- [ ] 相关日志字段可定位 tool 调用与 session 轨迹。

---

## 9. 可选增强（后续）

- 增加“对话改写子工具”：在 KB tool 内对省略问句做 query rewrite（不是必须，方案A通常已足够）。
- 在 observability 中增加：
  - `thread_id`
  - `tool_name`
  - `kb_hit/blocked`
  - `latency_ms`
- 对 `search_knowledge_base` 增加结构化返回校验与统一错误码。

