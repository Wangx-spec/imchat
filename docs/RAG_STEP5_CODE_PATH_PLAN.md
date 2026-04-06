# Step 5 技术代码路径方案（basic_tools + dialog_agent）

目标：在 Step4（`RAGService` 可回答闭环）基础上，完成“Agent 工具化接入闭环”：

- 在工具层新增 `search_knowledge_base`
- 通过 `set_rag_service(...)` 注入已初始化的 RAG 服务
- 将新工具注册到 `get_actions()`
- 在 `dialog_agent` 提示词中明确工具选择策略

对应主文档：`docs/RAG_IMPLEMENTATION.md` 的 Step 5。

---

## 1. 本阶段范围

仅实现：

- `src/actions/basic_tools.py`
- `src/agents/dialog_agent.py`

依赖输入（来自 Step4）：

- `RAGService`
- `AnswerResult`（至少包含 `answer/sources/debug`）

不在本阶段实现：

- 启动链路初始化（Step6）
- 依赖与测试收口（Step7）

---

## 2. 代码路径（调用关系）

Step5 完成后的最小调用链：

1. 启动阶段（由 Step6 完成）调用 `set_rag_service(service)` 注入实例
2. Agent 构建时通过 `get_actions()` 获取工具清单
3. 用户提问触发 Agent 规划
4. 命中文档/教程/做法类问题时调用 `search_knowledge_base(query)`
5. 工具内调用 `_rag_service.answer(query)` 并格式化返回
6. Agent 基于工具返回组织最终回复

核心目标：即使 RAG 未注入或执行异常，也不影响 Agent 基础对话能力。

---

## 3. 文件级实现方案

## 3.1 `src/actions/basic_tools.py`

### 3.1.1 推荐结构

- 模块级变量：`_rag_service = None`
- 注入函数：`set_rag_service(service) -> None`
- 工具函数：`@tool def search_knowledge_base(query: str) -> str`
- 注册函数：`get_actions() -> list`

### 3.1.2 `search_knowledge_base` 最小行为

1. 校验 `query` 非空（空字符串直接提示用户补充问题）。
2. 若 `_rag_service is None`，返回“知识检索服务未初始化”。
3. 调用 `result = _rag_service.answer(query)`。
4. 将 `result.answer` 与 `result.sources` 合并为可读文本返回。
5. 出现异常时返回降级提示，避免向 Agent 抛未捕获异常。

### 3.1.3 建议输出格式

建议在工具返回中保留来源，便于 Agent 引用与解释：

- 第一段：回答正文（`result.answer`）
- 第二段：来源列表（最多前 3 条）
- 可选第三段：调试摘要（仅开发态）

---

## 3.2 `src/agents/dialog_agent.py`

### 3.2.1 推荐接入点

- 在 `build_dialog_agent(settings)` 中继续使用 `create_agent(...)`
- 工具列表使用 `tools=get_actions()`
- 在系统提示词中增加“优先选用工具”的规则

### 3.2.2 工具选择规则建议

- 文档/章节/做法/知识类问题：优先 `search_knowledge_base`
- 时间类问题：优先 `get_current_time`
- 数学计算类问题：优先 `calculate`
- 非工具可解问题：允许直接回答

### 3.2.3 失败与兜底

- 当 `search_knowledge_base` 返回“未初始化/异常”文案时，Agent 应继续给出自然语言兜底，不中断对话。

---

## 4. 关键伪代码（可直接映射）

```python
# src/actions/basic_tools.py
_rag_service = None

def set_rag_service(service) -> None:
    global _rag_service
    _rag_service = service

@tool
def search_knowledge_base(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "请提供要检索的问题。"

    if _rag_service is None:
        return "知识检索服务未初始化。"

    try:
        result = _rag_service.answer(q)
        lines = [result.answer]
        if getattr(result, "sources", None):
            top_sources = result.sources[:3]
            lines.append("来源：")
            lines.extend([f"- {s}" for s in top_sources])
        return "\n".join(lines)
    except Exception as exc:
        return f"知识检索暂时不可用：{exc}"

def get_actions():
    return [get_current_time, calculate, search_knowledge_base]
```

```python
# src/agents/dialog_agent.py
def build_dialog_agent(settings):
    return create_agent(
        model=build_chat_model(settings),
        tools=get_actions(),
        system_prompt=(
            "遇到文档、章节、做法、知识库相关问题时，优先调用 search_knowledge_base；"
            "时间相关问题优先调用 get_current_time；"
            "数学计算优先调用 calculate。"
        ),
    )
```

---

## 5. 最小可运行自测（Step5）

## 5.1 建议脚本

`src/tests/tmp_step5_tool_smoke.py`

## 5.2 自测流程

1. 构造或初始化 `RAGService`，并 `set_rag_service(service)`。
2. 直接调用 `search_knowledge_base("宫保鸡丁怎么做")`。
3. 检查返回是否包含回答正文与来源段。
4. 清空注入服务（或不注入）再次调用，确认返回“未初始化”提示。
5. 通过 `build_dialog_agent(settings)` 构建 Agent，观察工具列表含 `search_knowledge_base`。

## 5.3 运行命令

```bash
PYTHONPATH=src python src/tests/tmp_step5_tool_smoke.py
```

---

## 6. 验收标准（DoD）

- [ ] `get_actions()` 包含 `search_knowledge_base`。
- [ ] `set_rag_service(...)` 注入后，工具可正常调用 `RAGService.answer(...)`。
- [ ] 未注入服务时工具可返回可解释提示，不抛异常。
- [ ] 工具返回中包含可读回答，且可附来源信息。
- [ ] `dialog_agent` 系统提示词明确工具优先级规则。

---

## 7. 风险与规避

- **工具误用风险**：提示词不明确会导致 LLM 不调用新工具，需强化规则描述。
- **输出过长风险**：来源过多会污染上下文，建议工具内限制来源条数（如前 3 条）。
- **异常外溢风险**：工具内必须 `try/except`，避免一次检索失败中断整轮 Agent 执行。
- **全局状态风险**：`_rag_service` 为模块级变量，后续若并发增大可考虑依赖注入容器化。

---

## 8. 与 Step6 的接口约定

Step6 启动链路需要确保在 Agent 真正开始处理请求前完成：

- `RAGService(...)` 构建与 `initialize(...)`
- `set_rag_service(service)` 注入

Step5 对 Step6 的最小要求：

- 若初始化失败，允许跳过注入，工具保持“未初始化”降级可用。

