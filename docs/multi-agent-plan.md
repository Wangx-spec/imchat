# Multi-Agent 多智能体协作方案

---

## 一、现状盘点

当前项目是**单 Agent 架构 + Skill 模块化**：

```
用户消息 → 唯一的 ReAct Agent（承载所有 Skill 的 tools + prompt）→ 回复
```

已有基础设施对 Multi-Agent 的支撑：

| 层 | 现状 | 对 Multi-Agent 的价值 |
|---|------|----------------------|
| **Skill 注册表** (`prompts/skills/__init__.py`) | `SkillDef` 按名映射 tools + prompt | 天然就是按 Agent 分配 Skill 的基础 |
| **Tool 注册表** (`basic_tools.py` `_ALL_TOOLS`) | 全量 tool 字典 + `get_actions(enabled_skills)` 按需筛选 | 已支持"给不同 Agent 不同 tool 子集" |
| **Prompt 组装** (`system_prompts.py` `build_system_prompt`) | 按 `enabled_skills` 动态拼装 | 已支持"给不同 Agent 不同 prompt" |
| **Checkpointer** (`dialog_graph.py`) | PostgresSaver / MemorySaver | 整个 StateGraph 共享一个即可 |
| **Chat Service** (`chat_service.py`) | `invoke()` / `stream()` 调用 `_agent` | 只需把 `_agent` 换成 multi-agent graph |

**关键结论：现有 Skill 系统已为 Multi-Agent 打好 80% 的地基。** `get_actions(enabled_skills)` 和 `build_system_prompt(enabled_skills)` 已支持按 skill 子集构建 Agent，只是目前所有 skill 都塞给了同一个 Agent。

---

## 二、目标架构

### 2.1 总体架构

```
用户消息
  │
  ▼
┌─────────────────────────────────────────────┐
│            StateGraph (LangGraph)            │
│                                             │
│  ┌─────────┐                                │
│  │Supervisor│ ── 判断意图，选择 specialist ──┐ │
│  └─────────┘                               │ │
│       │                                    │ │
│       ├──→ KnowledgeAgent                  │ │
│       │    skills: [knowledge_base]        │ │
│       │                                    │ │
│       ├──→ RecommendAgent                  │ │
│       │    skills: [dish_recommend]        │ │
│       │                                    │ │
│       ├──→ ChatAgent                       │ │
│       │    skills: [time, calculator]      │ │
│       │                                    │ │
│       └──→ (未来新 Agent...)               │ │
│                                            │ │
│            ◄─── 结果返回 ──────────────────┘ │
└─────────────────────────────────────────────┘
  │
  ▼
chat_service → SSE/invoke → 前端
```

### 2.2 设计原则

1. **Supervisor 只路由，不回答**：纯分发，token 消耗极低
2. **Specialist 复用现有构建方式**：每个 specialist 用 `create_react_agent`，与当前单 Agent 完全一致，只是 skills 子集不同
3. **Skill 系统零改动**：`get_actions(skills)` + `build_system_prompt(skills)` 直接复用
4. **向后兼容**：通过 `AGENT_MODE=single|multi` 切换，不设置时行为与改造前完全一致
5. **共享 Checkpointer**：整个 StateGraph compile 时传入顶层 checkpointer，由 LangGraph 统一管理状态持久化

---

## 三、涉及文件

| 操作 | 文件路径 | 说明 |
|------|----------|------|
| **新建** | `src/agents/agent_registry.py` | Agent 定义 + 注册表 |
| **新建** | `src/graphs/multi_agent_graph.py` | Supervisor + Specialist 编排图 |
| **改** | `src/graphs/dialog_graph.py` | 新增 `build_multi_agent()` 入口 |
| **改** | `src/agents/dialog_agent.py` | 按配置选择 single/multi 模式 |
| **改** | `src/config/settings.py` | 新增 `agent_mode` 配置 |
| **微调** | `src/services/chat_service.py` | stream 中过滤 supervisor 节点输出 |

> **前置条件**：`dish_recommend` skill 的接线工作需先完成（见 [dish-recommend-impl-guide.md](./dish-recommend-impl-guide.md)），确保 Skill 系统端到端正常。

---

## 四、核心组件详解

### 4.1 Agent 定义与注册 — `src/agents/agent_registry.py`（新建）

复用现有 Skill 体系，将"哪些 Skill 属于哪个 Agent"声明出来。

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class AgentDef:
    name: str                          # "knowledge", "recommend", "chat"
    description: str                   # 给 Supervisor 看的能力描述
    skills: list[str]                  # 该 Agent 拥有的 skill 名称
    model_override: str | None = None  # 可选：给该 Agent 用不同的 LLM

_AGENT_REGISTRY: dict[str, AgentDef] = {}

def register_agent(agent: AgentDef) -> None:
    _AGENT_REGISTRY[agent.name] = agent

def get_agent(name: str) -> AgentDef | None:
    return _AGENT_REGISTRY.get(name)

def all_agents() -> list[AgentDef]:
    return list(_AGENT_REGISTRY.values())
```

默认 Agent 定义：

```python
register_agent(AgentDef(
    name="knowledge",
    description="回答知识库/菜谱/教程类问题，提供具体做法和步骤",
    skills=["knowledge_base"],
))

register_agent(AgentDef(
    name="recommend",
    description="推荐菜品、列菜单、按条件筛选菜品",
    skills=["dish_recommend"],
))

register_agent(AgentDef(
    name="chat",
    description="通用对话：闲聊、时间查询、数学计算等非知识库问题",
    skills=["time", "calculator"],
))
```

设计要点：

- `AgentDef` 和 `SkillDef` 是平行关系：`SkillDef` 映射 tool + prompt，`AgentDef` 映射 skill 集合 → Agent
- 一个 Agent 可拥有多个 Skill，一个 Skill 也可被分配给多个 Agent
- `description` 字段给 Supervisor 路由决策用，不进入 specialist 的 system prompt
- `model_override` 预留不同 Agent 用不同模型的能力

### 4.2 Multi-Agent Graph — `src/graphs/multi_agent_graph.py`（新建）

编排层，用 LangGraph 的 `StateGraph` 把 Supervisor 和各 Specialist 连起来。

#### 4.2.1 Supervisor 节点

```python
import json
import logging
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import SystemMessage

logger = logging.getLogger(__name__)

def _build_supervisor_node(llm, agent_defs):
    agent_descriptions = "\n".join(
        f'- "{a.name}": {a.description}'
        for a in agent_defs
    )
    agent_names = [a.name for a in agent_defs]

    router_prompt = f"""\
你是一个路由器，负责将用户消息分派给合适的专家处理。

可用专家：
{agent_descriptions}

规则：
1. 分析用户最新一条消息的意图
2. 选择最匹配的专家
3. 只返回 JSON: {{"agent": "<专家名>", "reason": "<一句话理由>"}}
4. 不要回答用户的问题，只做路由判断

可选的 agent 值: {json.dumps(agent_names)}"""

    def supervisor_node(state: MessagesState) -> dict:
        messages = [
            SystemMessage(content=router_prompt),
            *state["messages"],
        ]
        response = llm.invoke(messages)
        try:
            parsed = json.loads(response.content)
            chosen = parsed.get("agent", agent_names[-1])
            if chosen not in agent_names:
                chosen = agent_names[-1]
        except (json.JSONDecodeError, AttributeError):
            chosen = agent_names[-1]

        logger.info("[SUPERVISOR] routed to=%s", chosen)
        return {"next": chosen}

    return supervisor_node
```

Supervisor 是纯路由，不回答问题，token 消耗极低。路由失败时 fallback 到最后一个 Agent（即 `chat`）。

#### 4.2.2 Specialist 节点

```python
from langgraph.prebuilt import create_react_agent
from actions.basic_tools import get_actions
from prompts.system_prompts import build_system_prompt

def _build_specialist_node(name, llm, agent_def):
    tools = get_actions(agent_def.skills)
    prompt = build_system_prompt(agent_def.skills)

    sub_agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
    )

    def specialist_node(state: MessagesState) -> dict:
        result = sub_agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    return specialist_node
```

每个 specialist 与当前单 Agent 用完全相同的构建方式，只是传入不同的 skill 子集。Sub-agent 不需要自己的 checkpointer，由顶层 StateGraph 统一管理。

#### 4.2.3 图编排

```python
from agents.agent_registry import all_agents
from llms.openai_chat import build_openai_chat_model

def build_multi_agent_graph(settings, checkpointer=None):
    llm = build_openai_chat_model(settings)
    agent_defs = all_agents()
    agent_names = [a.name for a in agent_defs]

    supervisor = _build_supervisor_node(llm, agent_defs)

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor)

    for agent_def in agent_defs:
        node = _build_specialist_node(agent_def.name, llm, agent_def)
        builder.add_node(agent_def.name, node)

    builder.add_edge(START, "supervisor")

    builder.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next", agent_names[-1]),
        {name: name for name in agent_names},
    )

    for name in agent_names:
        builder.add_edge(name, END)

    return builder.compile(checkpointer=checkpointer)
```

数据流：

```
START → supervisor → (conditional edge based on "next") → specialist → END
```

### 4.3 `dialog_graph.py` — 新增构建入口

在现有文件末尾新增一个函数，原有 `build_dialog_graph` 完全不动：

```python
def build_multi_agent(settings: Settings) -> Any:
    checkpointer = _build_checkpointer(settings)
    from graphs.multi_agent_graph import build_multi_agent_graph
    return build_multi_agent_graph(settings, checkpointer=checkpointer)
```

### 4.4 `dialog_agent.py` — 按配置选模式

在 `build_dialog_runtime` 中增加 `agent_mode` 分支：

```python
def build_dialog_runtime(settings: Settings) -> Tuple[Any, str]:
    runtime = (settings.agent_runtime or "langgraph").strip().lower()
    if runtime == "langchain":
        return build_dialog_agent_langchain(settings), "langchain"

    try:
        mode = (settings.agent_mode or "single").strip().lower()
        if mode == "multi":
            from graphs.dialog_graph import build_multi_agent
            return build_multi_agent(settings), "langgraph-multi"
        return build_dialog_graph(settings), "langgraph"
    except Exception as exc:
        logger.warning(
            "LangGraph init failed, fallback to LangChain runtime: %s",
            exc,
            exc_info=settings.verbose,
        )
        return build_dialog_agent_langchain(settings), "langchain"
```

### 4.5 `settings.py` — 新增配置

Settings dataclass 新增字段：

```python
agent_mode: str = "single"  # "single" | "multi"
```

`load_settings()` 中解析：

```python
agent_mode = os.getenv("AGENT_MODE", "single").strip().lower()
if agent_mode not in {"single", "multi"}:
    logger.warning("Invalid AGENT_MODE=%r, fallback to single", agent_mode)
    agent_mode = "single"
```

### 4.6 `chat_service.py` — stream 兼容性微调

Multi-agent graph 的 `invoke()` 返回结构与单 Agent 一致（`{"messages": [...]}`），`extract_text_from_result` 和 `extract_tool_calls` 无需改动。

`stream()` 中的 `stream_mode="updates"` 会额外包含 supervisor 节点的输出，需过滤：

```python
for update in _agent.stream(..., stream_mode="updates"):
    for node_name, node_state in update.items():
        if node_name == "supervisor":
            continue  # supervisor 只路由，不产生用户可见内容
        # ... 现有的消息提取逻辑不变
```

---

## 五、启动后数据流

### 5.1 Single 模式（不变）

```
app.py → build_dialog_runtime(settings)
  → settings.agent_mode == "single"
  → build_dialog_graph(settings)
    → create_react_agent(model, tools=ALL, prompt=ALL_SKILLS, checkpointer)
```

### 5.2 Multi 模式

```
app.py → build_dialog_runtime(settings)
  → settings.agent_mode == "multi"
  → build_multi_agent(settings)
    → build_multi_agent_graph(settings, checkpointer)
      │
      ├─ _build_supervisor_node(llm, agent_defs)
      │    → 生成 router_prompt（包含各 agent description）
      │
      ├─ _build_specialist_node("knowledge", llm, AgentDef(skills=["knowledge_base"]))
      │    → get_actions(["knowledge_base"]) → [search_knowledge_base]
      │    → build_system_prompt(["knowledge_base"]) → KB 专用 prompt
      │    → create_react_agent(model, tools, prompt)
      │
      ├─ _build_specialist_node("recommend", llm, AgentDef(skills=["dish_recommend"]))
      │    → get_actions(["dish_recommend"]) → [recommend_dishes]
      │    → build_system_prompt(["dish_recommend"]) → 推荐专用 prompt
      │    → create_react_agent(model, tools, prompt)
      │
      └─ _build_specialist_node("chat", llm, AgentDef(skills=["time", "calculator"]))
           → get_actions(["time", "calculator"]) → [get_current_time, calculate]
           → build_system_prompt(["time", "calculator"]) → 通用对话 prompt
           → create_react_agent(model, tools, prompt)
```

### 5.3 运行时调用链示例

```
用户: "推荐几道难的菜"
  │
  ▼
supervisor_node
  → LLM 判断意图 → {"agent": "recommend", "reason": "用户要求推荐菜品"}
  │
  ▼
recommend specialist
  → ReAct Agent (tools=[recommend_dishes], prompt=dish_recommend skill)
  → 调用 recommend_dishes(query="推荐几道难的菜")
  → 返回候选菜品列表
  → LLM 根据 Skill Prompt 约束组织回复
  │
  ▼
chat_service → SSE → 前端
```

```
用户: "宫保鸡丁怎么做"
  │
  ▼
supervisor_node
  → LLM 判断意图 → {"agent": "knowledge", "reason": "用户询问具体菜品做法"}
  │
  ▼
knowledge specialist
  → ReAct Agent (tools=[search_knowledge_base], prompt=KB skill)
  → 调用 search_knowledge_base(query="宫保鸡丁怎么做")
  → 返回做法详情 + 引用来源
```

---

## 六、环境变量配置

```bash
# .env

# Agent 模式：single（默认，单 Agent）或 multi（多 Agent）
AGENT_MODE=multi

# 以下配置与 single 模式完全兼容，不受影响
AGENT_RUNTIME=langgraph
AGENT_STREAMING=true
ENABLED_SKILLS=knowledge_base,dish_recommend,time,calculator
```

---

## 七、实施步骤

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| **0** | 完成 dish_recommend 接线（前置） | 参考 [dish-recommend-impl-guide.md](./dish-recommend-impl-guide.md) |
| **1** | 新建 `src/agents/agent_registry.py` | `python -c "from agents.agent_registry import all_agents; print([a.name for a in all_agents()])"` |
| **2** | 新建 `src/graphs/multi_agent_graph.py` | 单元测试：构建 graph，检查节点和边 |
| **3** | 改 `dialog_graph.py`：加 `build_multi_agent()` | import 不报错 |
| **4** | 改 `dialog_agent.py`：加 `agent_mode` 分支 | `AGENT_MODE=multi` 时返回 `"langgraph-multi"` runtime |
| **5** | 改 `settings.py`：加 `agent_mode` 字段 | `python -c "from config.settings import load_settings; print(load_settings().agent_mode)"` |
| **6** | 改 `chat_service.py`：stream 过滤 supervisor | SSE 流不输出路由 JSON |
| **7** | 端到端测试 | 推荐/做法/闲聊 三类问题分别命中正确的 specialist |

---

## 八、注意事项

1. **Supervisor 路由准确率**：初期建议 Supervisor 用与 specialist 相同的 LLM。如果路由错误率高，可考虑在 router_prompt 中添加 few-shot 示例，或用 Embedding 分类替代 LLM 路由。

2. **Fallback 机制**：如果 Supervisor 解析 JSON 失败，默认 fallback 到 `chat` agent（注册表中的最后一个），保证不中断对话。

3. **与 single 模式的兼容**：`AGENT_MODE=single` 时走原有路径，所有行为不变。两种模式共享同一套 Skill 注册表和 Tool 注册表。

4. **Checkpointer 共享**：multi-agent graph 的 compile 传入的 checkpointer 管理整个图的状态，包含 supervisor 路由决策和各 specialist 的对话上下文。

5. **messages 表不受影响**：`chat_service.py` 的 `save_message` 只关心最终输出文本，对 multi-agent 内部流转透明。

---

## 九、后续扩展方向

| 方向 | 说明 |
|------|------|
| **Supervisor 路由优化** | 从纯 LLM JSON 路由 → Embedding 分类 → 规则 + LLM 混合，逐步降本提速 |
| **Agent 间协作** | specialist 完成后回到 supervisor 决定是否需要二次分发（如"推荐菜后查做法"两步任务） |
| **异构 LLM** | `AgentDef.model_override` 让不同 Agent 用不同模型（如 recommend 用轻量模型，knowledge 用强推理模型） |
| **动态 Agent 加载** | 把 `agent_registry.py` 中的硬编码改为配置文件或数据库驱动 |
| **Critic 节点** | 在 specialist → END 之间插入校验节点，检查幻觉/引用合规性 |
| **并行分发** | 用户问题可能涉及多个 agent 时，supervisor 并行调用多个 specialist 再合并结果 |

---

## 相关文档

- [Skill 模块化架构](./skill-architecture.md)：Skill 注册表、Tool 注册表、Prompt 组装机制
- [菜品推荐 Skill 实现方案](./dish-recommend-skill-plan.md)：`recommend_dishes` 设计
- [菜品推荐逐文件实现指南](./dish-recommend-impl-guide.md)：接线工作（Multi-Agent 前置）
- [对话架构总览](./conversation-architecture.md)：CLI/Web 入口、Checkpointer、messages 表
