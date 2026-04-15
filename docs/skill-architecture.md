# Skill 模块化架构

---

## 一、概述

Skill 是本项目对"工具能力"的模块化封装。每个 Skill 包含两部分：

| 组成 | 位置 | 作用 |
|------|------|------|
| **Tool 实现** | `src/actions/xxx_tools.py` | `@tool` 装饰的函数，定义能力"怎么执行" |
| **Skill Prompt** | `src/prompts/skills/xxx.py` | 告诉 LLM 这个能力"什么时候用、怎么用好" |

两者通过 **注册表** 关联：Skill Prompt 模块在 import 时自动注册到全局 `_REGISTRY`，系统根据配置动态组装 System Prompt 和 Tool 列表。

### 核心设计原则

1. **新增 Skill 不改已有代码**——只需新建文件 + 加注册
2. **Tool 和 Prompt 同步启停**——由同一个 `enabled_skills` 配置控制
3. **向后兼容**——不设置 `ENABLED_SKILLS` 环境变量时，行为与改造前完全一致

---

## 二、配置

### 环境变量

| 变量名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ENABLED_SKILLS` | 逗号分隔字符串 | 空（加载全部） | 要启用的 Skill 名称列表 |

### 示例

```bash
# .env

# 加载全部 Skill（默认行为，等同于不设置）
# ENABLED_SKILLS=

# 只加载知识库和时间两个 Skill
ENABLED_SKILLS=knowledge_base,time

# 加载全部（显式写法）
ENABLED_SKILLS=knowledge_base,time,calculator
```

### Settings 字段

```python
# src/config/settings.py

@dataclass
class Settings:
    # ...
    enabled_skills: list[str] | None = None
```

- `None`：加载全部已注册的 Skill
- `["knowledge_base", "time"]`：只加载指定的 Skill

---

## 三、架构与数据流

### 3.1 文件结构

```
src/
├── prompts/
│   ├── system_prompts.py                ← 动态组装 System Prompt
│   ├── knowledge_base_prompt.py         ← RAG 内部 prompt（非 Skill）
│   └── skills/
│       ├── __init__.py                  ← SkillDef 定义 + 注册表
│       ├── knowledge_base.py            ← KB Skill Prompt
│       ├── time.py                      ← 时间 Skill Prompt
│       └── calculator.py               ← 计算器 Skill Prompt
├── actions/
│   ├── basic_tools.py                   ← Tool 注册表 + get_actions()
│   └── knowledge_base_tools.py          ← search_knowledge_base 实现
├── config/
│   └── settings.py                      ← enabled_skills 配置
├── graphs/
│   └── dialog_graph.py                  ← 构建 Agent 时传入 enabled_skills
└── agents/
    └── dialog_agent.py                  ← 同上（langchain 回退路径）
```

### 3.2 启动时数据流

```
1. dialog_graph.py / dialog_agent.py 被 import
     │
2. import system_prompts
     │
3. system_prompts.py 顶部 import prompts.skills.xxx 模块
     │  → 每个模块在 import 时调用 register(SkillDef(...))
     │  → _REGISTRY 被填充
     │
4. build_dialog_graph(settings) 被调用
     │
     ├── get_actions(settings.enabled_skills)
     │     → get_tool_names_for_skills() 从 _REGISTRY 查出需要的 tool 名
     │     → 从 _ALL_TOOLS 字典中筛选出对应的 @tool 函数
     │
     └── build_system_prompt(settings.enabled_skills)
           → get_prompts_for_skills() 从 _REGISTRY 查出需要的 prompt
           → 拼装 角色 + 全局规则 + 优先级列表 + Skill 使用规范
           │
           ▼
     create_react_agent(model, tools, prompt, checkpointer)
```

### 3.3 组装后的 System Prompt 结构

```
你是一名有帮助的智能烹饪助手。

最终回复必须与用户最新一条消息保持同一语言。

工具调用优先级（从高到低）：
1. 文档/菜谱/教程/知识库类问题 → search_knowledge_base
2. 时间/日期问题 → get_current_time
3. 数学表达式 → calculate

以下是每个工具的详细使用规范：

【Skill: search_knowledge_base】
触发条件：...
输入要求：...
使用规则：...

【Skill: get_current_time】
触发条件：...

【Skill: calculate】
触发条件：...
```

---

## 四、核心组件详解

### 4.1 Skill 注册表 — `src/prompts/skills/__init__.py`

```python
@dataclass(frozen=True)
class SkillDef:
    name: str              # Skill 唯一标识，如 "knowledge_base"
    tool_names: list[str]  # 关联的 @tool 函数名，如 ["search_knowledge_base"]
    prompt: str            # LLM 使用规范文本

_REGISTRY: dict[str, SkillDef] = {}

def register(skill: SkillDef) -> None           # 注册
def get_skill(name: str) -> SkillDef | None      # 按名查询
def all_skills() -> list[SkillDef]               # 获取全部
def get_prompts_for_skills(names) -> list[str]   # 批量取 prompt
def get_tool_names_for_skills(names) -> set[str] # 批量取 tool 名
```

### 4.2 Skill Prompt 模块 — `src/prompts/skills/xxx.py`

每个 Skill Prompt 文件遵循统一模式：

```python
from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: <tool_function_name>】
触发条件：<何时调用>
输入要求：<参数说明>
输出格式：<返回值格式>
使用规则：
- <规则1>
- <规则2>\
"""

register(SkillDef(
    name="<skill_name>",
    tool_names=["<tool_function_name>"],
    prompt=SKILL_PROMPT,
))
```

### 4.3 Tool 注册表 — `src/actions/basic_tools.py`

```python
_ALL_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
}

def get_actions(enabled_skills: list[str] | None = None) -> list:
    if enabled_skills is None:
        return list(_ALL_TOOLS.values())
    needed = get_tool_names_for_skills(enabled_skills)
    return [t for name, t in _ALL_TOOLS.items() if name in needed]
```

### 4.4 Prompt 组装器 — `src/prompts/system_prompts.py`

```python
def build_system_prompt(enabled_skills: list[str] | None = None) -> str:
    # enabled_skills=None 时使用 _DEFAULT_SKILLS 加载全部
    # 否则只组装指定 skill 的 prompt
```

关键常量：

| 常量 | 作用 |
|------|------|
| `_DEFAULT_SKILLS` | 默认加载的 Skill 名称列表 |
| `_PRIORITY_MAP` | 每个 Skill 在优先级列表中的描述文本 |

---

## 五、现有 Skill 一览

| Skill 名称 | Tool 函数 | Prompt 文件 | Tool 文件 |
|------------|-----------|-------------|-----------|
| `knowledge_base` | `search_knowledge_base` | `skills/knowledge_base.py` | `actions/knowledge_base_tools.py` |
| `time` | `get_current_time` | `skills/time.py` | `actions/basic_tools.py` |
| `calculator` | `calculate` | `skills/calculator.py` | `actions/basic_tools.py` |

---

## 六、新增 Skill 标准流程

以新增"食材推荐"功能为例，完整流程如下：

### 第 1 步：编写 Tool 实现

新建 `src/actions/recipe_recommend_tools.py`：

```python
import logging
from langchain_core.tools import tool

logger = logging.getLogger("chat.tools")

@tool
def recommend_by_ingredients(ingredients: str) -> str:
    """Recommend recipes based on available ingredients.
    Input: comma-separated ingredient names like '鸡蛋,番茄,豆腐'."""
    logger.info("[TOOL_CALL] name=recommend_by_ingredients ingredients=%s", ingredients)
    # 实现检索/推荐逻辑...
    result = "..."
    logger.info("[TOOL_RESULT] name=recommend_by_ingredients ok")
    return result
```

**要求：**
- 函数必须用 `@tool` 装饰
- docstring 用英文写（LLM 识别效果更稳定）
- 参数用 type hint 标注
- 添加 `[TOOL_CALL]` 和 `[TOOL_RESULT]` 日志

### 第 2 步：编写 Skill Prompt

新建 `src/prompts/skills/recipe_recommend.py`：

```python
from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: recommend_by_ingredients】
触发条件：用户提到手头有什么食材、问"能做什么菜"、"有什么推荐"时调用。
输入要求：将用户提到的食材提取为逗号分隔的字符串传入。
输出格式：返回匹配菜谱列表的 JSON 字符串。
使用规则：
- 食材不足 2 种时，建议用户补充更多食材以获得更精准推荐。
- 推荐结果应注明匹配度和缺少的食材。\
"""

register(SkillDef(
    name="recipe_recommend",
    tool_names=["recommend_by_ingredients"],
    prompt=SKILL_PROMPT,
))
```

**Prompt 编写要点：**
- `触发条件`：明确告诉 LLM 什么场景下该调用
- `输入要求`：说明如何从用户话语中提取参数
- `输出格式`：让 LLM 知道如何解读返回值
- `使用规则`：边界情况的处理指导

### 第 3 步：注册到系统

修改 3 处，每处各加 1-2 行：

**a) `src/prompts/system_prompts.py`** — 加 import 触发注册 + 优先级描述

```python
import prompts.skills.recipe_recommend   # noqa: F401  ← 新增

_PRIORITY_MAP: dict[str, str] = {
    "knowledge_base": "文档/菜谱/教程/知识库类问题 → search_knowledge_base",
    "recipe_recommend": "食材推荐/能做什么菜 → recommend_by_ingredients",  # ← 新增
    "time": "时间/日期问题 → get_current_time",
    "calculator": "数学表达式 → calculate",
}

_DEFAULT_SKILLS = ["knowledge_base", "recipe_recommend", "time", "calculator"]  # ← 更新
```

**b) `src/actions/basic_tools.py`** — 注册 Tool 到 `_ALL_TOOLS`

```python
from actions.recipe_recommend_tools import recommend_by_ingredients  # ← 新增

_ALL_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
    "recommend_by_ingredients": recommend_by_ingredients,  # ← 新增
}
```

**c) `.env`**（可选）— 如果设置了 `ENABLED_SKILLS`，追加新 skill 名

```bash
ENABLED_SKILLS=knowledge_base,recipe_recommend,time,calculator
```

### 验证清单

| 验证项 | 方法 |
|--------|------|
| Skill 注册成功 | `python -c "from prompts.system_prompts import SYSTEM_PROMPT; print(SYSTEM_PROMPT)"` 看到新 Skill 区块 |
| Tool 注册成功 | `python -c "from actions.basic_tools import get_actions; print([t.name for t in get_actions()])"` 看到新 tool 名 |
| 按需加载 | 设 `ENABLED_SKILLS=knowledge_base,time` → 新 tool 不出现 |
| 端到端 | 启动服务，发送触发新 Skill 的问题，确认 Tool 被调用 |

---

## 七、Skill 与 RAG 的关系

Skill 系统管理的是 **Agent 层面的工具和 prompt**。RAG 管道（检索、索引、重排等）是 `knowledge_base` Skill 的**内部实现**，不受 Skill 框架管理。

```
Skill 层（本文档范围）
├── knowledge_base Skill
│   ├── Prompt: 何时调用、如何解读结果
│   └── Tool: search_knowledge_base()
│              └── 内部调用 RAGService.answer()
│                       └── RAG 管道（不在 Skill 管辖范围）
│                           ├── data_loader
│                           ├── chunking
│                           ├── index_store (FAISS)
│                           ├── retriever (Hybrid)
│                           ├── query_planner
│                           └── generation_router
├── time Skill
├── calculator Skill
└── ... 未来新增
```

如果未来新增的 Skill 需要独立的知识库（如"营养数据库"），可以：
1. 构建独立的数据源和检索管道
2. 封装为独立的 `@tool` 函数
3. 注册为新 Skill

不需要修改现有 RAG 管道或 `knowledge_base` Skill。

---

## 八、注意事项

1. **Skill 名称唯一**：`SkillDef.name` 作为注册表的 key，不能重复。
2. **tool_names 与 _ALL_TOOLS 一致**：`SkillDef.tool_names` 中的名称必须与 `_ALL_TOOLS` 字典的 key 完全一致，否则 `get_actions()` 无法筛选到对应的 Tool。
3. **import 触发注册**：Skill Prompt 模块必须在 `system_prompts.py` 顶部被 import，否则不会注册到 `_REGISTRY`。
4. **一个 Skill 可关联多个 Tool**：`tool_names` 是列表，如果某个业务需要多个 Tool 协作（如推荐 + 详情查询），可以放在同一个 Skill 下。
5. **Prompt 不宜过长**：每个 Skill Prompt 控制在 200 字以内，过长会浪费 token 且可能干扰 LLM 决策。

---

## 相关文档

- [菜品推荐 Skill 实现方案](./dish-recommend-skill-plan.md)：`recommend_dishes` 工具与 Skill Prompt 的设计、与 `search_knowledge_base` 的分工、测试清单（实施前可读）。
