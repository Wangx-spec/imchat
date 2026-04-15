# 菜品推荐 Skill —— 逐文件代码实现指南

> 基于 [dish-recommend-skill-plan.md](./dish-recommend-skill-plan.md) 的落地方案。
> 本文档给出**每个文件的完整代码或精确 diff**，可按顺序逐步执行。

---

## 〇、涉及文件总览

| 序号 | 操作 | 文件路径 | 说明 |
|:----:|------|----------|------|
| 1 | **改** | `src/actions/knowledge_base_tools.py` | 增加 `get_rag_service()` |
| 2 | **新建** | `src/actions/dishes/recommend_dishes_tools.py` | `recommend_dishes` 工具实现 |
| 3 | **新建** | `src/prompts/skills/dish_recommend.py` | Skill Prompt + 注册 |
| 4 | **改** | `src/actions/basic_tools.py` | `_ALL_TOOLS` 挂入新工具 |
| 5 | **改** | `src/prompts/system_prompts.py` | import / `_DEFAULT_SKILLS` / `_PRIORITY_MAP` |
| 6 | **改** | `src/services/chat_service.py` | `sanitize_ungrounded_kb_claim` 兼容新工具 |
| 7 | **改** | `docs/skill-architecture.md` | Skill 一览表补一行 |

---

## 一、`src/actions/knowledge_base_tools.py` — 增加 `get_rag_service()`

### 1.1 改动说明

当前文件已有模块级变量 `_rag_service` 及 `set_rag_service()`。推荐工具需要读取同一实例，增加一个配对的 getter，避免外部直接访问私有变量。

### 1.2 精确 diff

在 `set_rag_service` 函数之后（约第 15 行）新增：

```python
def get_rag_service() -> Any | None:
    return _rag_service
```

改动后该区域完整代码：

```python
_rag_service: Any = None


def set_rag_service(service: Any) -> None:
    global _rag_service
    _rag_service = service
    logger.info("[RAG_BIND] service=%s", type(service).__name__ if service else "None")


def get_rag_service() -> Any | None:
    return _rag_service
```

无需修改 `rag/bootstrap.py`——`bootstrap_rag` 仍只调用 `set_rag_service`，推荐工具通过 `get_rag_service()` 读同一引用即可。

---

## 二、`src/actions/dishes/recommend_dishes_tools.py` — 新建

### 2.1 设计要点

| 要点 | 做法 |
|------|------|
| 走 `retrieve()` 而非 `answer()` | 只拿 parent 列表，不走 GenerationRouter 的拼装逻辑 |
| 按 parent 去重 | 用 `parent_id`（或 `source`）做 seen-set，保留首次出现顺序 |
| 轻量规则扩展 | 「高难度/星/星星」类意图时追加 `starsystem` 等关键词 |
| 返回结构化 JSON 字符串 | 与 `search_knowledge_base` 风格一致，方便 Skill Prompt 约束输出 |
| docstring 用英文 | LangChain `@tool` 的 description 来自 docstring，英文对 LLM 函数调用更稳定 |

### 2.2 完整代码

```python
import json
import logging
import re
from typing import Any

from langchain_core.tools import tool

from actions.knowledge_base_tools import get_rag_service

logger = logging.getLogger("chat.tools")

_RECOMMEND_LIMIT = 15

_EXPAND_RULES: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"(难|高难度|[4-5]\s*[星⭐])"), ["starsystem", "4星", "5星"]),
    (re.compile(r"(简单|入门|新手|[1-2]\s*[星⭐])"), ["starsystem", "1星", "2星"]),
    (re.compile(r"(三星|3\s*[星⭐]|中等)"), ["starsystem", "3星"]),
]


def _expand_query(query: str) -> str:
    extras: list[str] = []
    for pat, terms in _EXPAND_RULES:
        if pat.search(query):
            extras.extend(terms)
    if not extras:
        return query
    return query + " " + " ".join(dict.fromkeys(extras))


def _dedup_parents(parents: list[Any], limit: int) -> list[dict]:
    seen: set[str] = set()
    candidates: list[dict] = []
    for p in parents:
        meta = getattr(p, "metadata", {}) or {}
        pid = meta.get("parent_id") or meta.get("source") or ""
        if pid in seen:
            continue
        seen.add(pid)
        title = meta.get("title", "")
        source = meta.get("source", "")
        if not title and not source:
            continue
        candidates.append({"title": title, "source": source})
        if len(candidates) >= limit:
            break
    return candidates


def _build_payload(
    query: str,
    ok: bool,
    candidates: list[dict] | None = None,
    sources_distinct: list[str] | None = None,
    error: str | None = None,
    debug: dict[str, Any] | None = None,
) -> str:
    payload = {
        "type": "dish_recommendation_result",
        "ok": ok,
        "query": query,
        "candidates": candidates or [],
        "sources_distinct": sources_distinct or [],
        "limit": _RECOMMEND_LIMIT,
        "error": error,
        "debug": debug or {},
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def recommend_dishes(query: str) -> str:
    """Recommend dish names from the local knowledge base.
    Input: user's original request describing preferences such as difficulty,
    cuisine style, occasion, dietary restrictions, or general 'what to cook'.
    Returns a JSON string with a list of candidate dish titles."""
    q = (query or "").strip()
    logger.info("[TOOL_CALL] name=recommend_dishes query=%s", q)

    if not q:
        return _build_payload(query=q, ok=False, error="empty_query")

    svc = get_rag_service()
    if svc is None:
        return _build_payload(query=q, ok=False, error="service_not_initialized")

    try:
        expanded = _expand_query(q)
        logger.info("[RECOMMEND] expanded_query=%s", expanded)

        ret = svc.retrieve(expanded)

        candidates = _dedup_parents(ret.parents, _RECOMMEND_LIMIT)
        sources = list(dict.fromkeys(c["source"] for c in candidates if c["source"]))

        text = _build_payload(
            query=q,
            ok=True,
            candidates=candidates,
            sources_distinct=sources,
            debug={
                "expanded_query": expanded,
                "parent_hits": len(ret.parents),
                "deduped": len(candidates),
            },
        )
        logger.info(
            "[TOOL_RESULT] name=recommend_dishes ok candidates=%d",
            len(candidates),
        )
        return text
    except Exception as exc:
        logger.exception("[TOOL_RESULT] name=recommend_dishes error=%s", exc)
        return _build_payload(query=q, ok=False, error=f"tool_exception:{exc}")
```

### 2.3 关键实现说明

1. **`_expand_query`**：轻量规则扩展。用正则匹配用户语句中的「高难度/星」等意图词，追加 `starsystem` / `4星` 等关键词，提高命中星级索引页的概率。规则以 `_EXPAND_RULES` 列表形式维护，后续可方便追加。

2. **`_dedup_parents`**：从 `RetrievalResult.parents`（LangChain `Document` 列表）中按 `parent_id` 或 `source` 去重，保留首次出现顺序，截取前 `_RECOMMEND_LIMIT` 条。每条仅提取 `title` 和 `source`，不传完整正文。

3. **`_build_payload`**：统一 JSON 输出结构，与计划文档第四节契约一致：`type`、`ok`、`candidates`、`sources_distinct`、`limit`、`error`、`debug`。

4. **`svc.retrieve()`**：不走 `svc.answer()`。`retrieve()` 内部经过 QueryPlanner + HybridRetriever（向量 + BM25 + RRF + Rerank），返回 `RetrievalResult`，其 `.parents` 已按相关度排序。

5. **`_RECOMMEND_LIMIT = 15`**：第一期写死常量。若后续需要 `RAG_RECOMMEND_TOP_K` 环境变量控制，在 `Settings` 和 `RAGConfig` 中新增字段，再把这里改为从 service 配置中读取即可。

---

## 三、`src/prompts/skills/dish_recommend.py` — 新建

### 3.1 设计要点

- Prompt 必须明确**触发条件**、**与 `search_knowledge_base` 的分工**、**输出约束**（菜名只能来自 `candidates`）。
- 模式完全对齐已有 Skill（`knowledge_base.py`、`time.py`、`calculator.py`）。

### 3.2 完整代码

```python
from prompts.skills import SkillDef, register

SKILL_PROMPT = """\
【Skill: recommend_dishes】
触发条件：用户想要多道菜推荐、不知道做什么、按难度/口味/场景/人数/忌口筛选菜品、问"有什么菜""推荐几道""想吃啥"等名单式需求时调用。
与 search_knowledge_base 的分工：
- 用户要某道菜的具体做法、步骤、食材用量、单菜详解 → 用 search_knowledge_base。
- 用户要推荐多道菜、列菜单、按条件选菜 → 用 recommend_dishes。
- 若不确定，优先尝试 recommend_dishes；用户追问做法时再切 search_knowledge_base。
输入要求：必须传入用户原始查询文本，不得翻译或改写。
输出格式：返回 JSON 字符串，包含 ok、candidates、sources_distinct、error 字段。
使用规则：
- ok=true 时，从 candidates 列表中挑选菜名呈现给用户；可适当分组（如按来源、难度）但菜名必须来自 candidates，禁止自造菜名。
- candidates 为空或极少时，如实告知"知识库未找到符合条件的菜品"，可建议用户换一种描述方式，禁止编造菜名补充。
- 未调用此工具时，禁止使用"根据知识库推荐""为您从菜谱库中筛选"等暗示已检索的表述。
- 回复语言与用户最新消息保持一致。\
"""

register(SkillDef(
    name="dish_recommend",
    tool_names=["recommend_dishes"],
    prompt=SKILL_PROMPT,
))
```

### 3.3 Prompt 设计说明

| 段落 | 目的 |
|------|------|
| 触发条件 | 罗列典型用户表述，让 LLM 精准命中推荐意图 |
| 与 search_knowledge_base 的分工 | 防止两工具互抢；"不确定时先推荐"降低误路由成本 |
| 输入要求 | 与 KB Skill 一致——不翻译、不改写 |
| 使用规则第 1 条 | **核心防幻觉条款**：菜名只能来自 `candidates` |
| 使用规则第 2 条 | 空结果兜底：不硬编 |
| 使用规则第 3 条 | 与 KB Skill 对齐的"未调用则不伪装" |

---

## 四、`src/actions/basic_tools.py` — 修改

### 4.1 改动说明

新增 import + `_ALL_TOOLS` 加一项。`get_actions` 无需改动——它已通过 `get_tool_names_for_skills` 按 skill 过滤。

### 4.2 精确 diff

**a) 头部 import 区新增一行：**

```python
# 现有
from actions.knowledge_base_tools import search_knowledge_base
# 新增（紧跟其后）
from actions.dishes.recommend_dishes_tools import recommend_dishes
```

**b) `_ALL_TOOLS` 字典新增一项：**

```python
_ALL_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
    "recommend_dishes": recommend_dishes,                  # ← 新增
}
```

改动后 `basic_tools.py` 完整代码（省略 `_safe_eval_expr` 等不变部分）：

```python
import ast
import logging
import operator as op
from datetime import datetime

from langchain_core.tools import tool

from actions.knowledge_base_tools import search_knowledge_base
from actions.dishes.recommend_dishes_tools import recommend_dishes
from prompts.skills import get_tool_names_for_skills

# ... _ALLOWED_OPERATORS / _safe_eval_expr / get_current_time / calculate 不变 ...

_ALL_TOOLS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_knowledge_base": search_knowledge_base,
    "recommend_dishes": recommend_dishes,
}


def get_actions(enabled_skills: list[str] | None = None) -> list:
    if enabled_skills is None:
        return list(_ALL_TOOLS.values())
    needed = get_tool_names_for_skills(enabled_skills)
    return [t for name, t in _ALL_TOOLS.items() if name in needed]
```

---

## 五、`src/prompts/system_prompts.py` — 修改

### 5.1 改动说明

三处改动：import 触发注册、`_DEFAULT_SKILLS` 加入新 skill、`_PRIORITY_MAP` 加描述。

### 5.2 精确 diff

**a) 顶部 import 区新增一行（位于其他 skill import 之后）：**

```python
import prompts.skills.knowledge_base  # noqa: F401 — 触发注册
import prompts.skills.time            # noqa: F401
import prompts.skills.calculator      # noqa: F401
import prompts.skills.dish_recommend  # noqa: F401          # ← 新增
```

**b) `_DEFAULT_SKILLS` — 将 `dish_recommend` 插入 `knowledge_base` 之后：**

```python
_DEFAULT_SKILLS = ["knowledge_base", "dish_recommend", "time", "calculator"]
```

> `dish_recommend` 排在 `knowledge_base` 之后而非之前，因为"具体做法查询"场景更高频，保持 `knowledge_base` 最高优先级。如果产品需要推荐优先（即引导用户先选菜再看做法），可将 `dish_recommend` 挪到首位。

**c) `_PRIORITY_MAP` 新增一项：**

```python
_PRIORITY_MAP: dict[str, str] = {
    "knowledge_base": "文档/菜谱/教程/知识库类问题 → search_knowledge_base",
    "dish_recommend": "多菜推荐/想吃啥/按条件选菜 → recommend_dishes",  # ← 新增
    "time": "时间/日期问题 → get_current_time",
    "calculator": "数学表达式 → calculate",
}
```

### 5.3 改动后完整代码

```python
from __future__ import annotations

import prompts.skills.knowledge_base  # noqa: F401 — 触发注册
import prompts.skills.time            # noqa: F401
import prompts.skills.calculator      # noqa: F401
import prompts.skills.dish_recommend  # noqa: F401

from prompts.skills import get_prompts_for_skills

_ROLE = "你是一名有帮助的智能烹饪助手。"

_GLOBAL_RULES = "最终回复必须与用户最新一条消息保持同一语言。"

_DEFAULT_SKILLS = ["knowledge_base", "dish_recommend", "time", "calculator"]

_PRIORITY_MAP: dict[str, str] = {
    "knowledge_base": "文档/菜谱/教程/知识库类问题 → search_knowledge_base",
    "dish_recommend": "多菜推荐/想吃啥/按条件选菜 → recommend_dishes",
    "time": "时间/日期问题 → get_current_time",
    "calculator": "数学表达式 → calculate",
}


def build_system_prompt(enabled_skills: list[str] | None = None) -> str:
    skills = enabled_skills or _DEFAULT_SKILLS

    priority_lines = []
    for i, s in enumerate(skills, 1):
        desc = _PRIORITY_MAP.get(s, s)
        priority_lines.append(f"{i}. {desc}")
    priority_section = "工具调用优先级（从高到低）：\n" + "\n".join(priority_lines)

    skill_prompts = get_prompts_for_skills(skills)
    skill_section = "\n\n".join(skill_prompts) if skill_prompts else ""

    parts = [_ROLE, _GLOBAL_RULES, priority_section]
    if skill_section:
        parts.append("以下是每个工具的详细使用规范：\n\n" + skill_section)

    return "\n\n".join(parts)


SYSTEM_PROMPT = build_system_prompt()
```

---

## 六、`src/services/chat_service.py` — 修改 `sanitize_ungrounded_kb_claim`

### 6.1 改动说明

当前的 `has_kb_tool_call` 只检测 `search_knowledge_base`。如果用户这轮调用的是 `recommend_dishes`（同样基于知识库检索），回复中出现「根据知识库」类话术也应视为 grounded，不应被改写。

### 6.2 精确 diff

将 `has_kb_tool_call` 扩展为同时匹配两个工具名：

```python
# ── 改动前 ──
def has_kb_tool_call(tool_calls: list[dict]) -> bool:
    for c in tool_calls:
        if str(c.get("name", "")).strip() == "search_knowledge_base":
            return True
    return False

# ── 改动后 ──
_KB_GROUNDED_TOOLS = {"search_knowledge_base", "recommend_dishes"}

def has_kb_tool_call(tool_calls: list[dict]) -> bool:
    for c in tool_calls:
        if str(c.get("name", "")).strip() in _KB_GROUNDED_TOOLS:
            return True
    return False
```

`sanitize_ungrounded_kb_claim` 函数本身无需修改——它调用 `has_kb_tool_call` 的逻辑不变，只是现在两个工具都能让它提前 return。

---

## 七、`docs/skill-architecture.md` — 更新 Skill 一览表

### 7.1 精确 diff

在「五、现有 Skill 一览」的表格中补一行：

```markdown
## 五、现有 Skill 一览

| Skill 名称 | Tool 函数 | Prompt 文件 | Tool 文件 |
|------------|-----------|-------------|-----------|
| `knowledge_base` | `search_knowledge_base` | `skills/knowledge_base.py` | `actions/knowledge_base_tools.py` |
| `dish_recommend` | `recommend_dishes` | `skills/dish_recommend.py` | `actions/dishes/recommend_dishes_tools.py` |
| `time` | `get_current_time` | `skills/time.py` | `actions/basic_tools.py` |
| `calculator` | `calculate` | `skills/calculator.py` | `actions/basic_tools.py` |
```

同时更新文件结构树（3.1 节）和组装示例（3.3 节）中的对应部分，使其包含新文件和新 Skill 块。

---

## 八、启动后数据流验证

改动完成后，系统启动时数据流如下：

```
app.py
  │
  ├─ bootstrap_rag(settings)
  │     └─ set_rag_service(service)          ← _rag_service 被设置
  │
  └─ build_dialog_graph(settings) / build_dialog_agent(settings)
        │
        ├─ import system_prompts
        │     ├─ import skills.knowledge_base  → register("knowledge_base")
        │     ├─ import skills.dish_recommend  → register("dish_recommend")   ← 新增
        │     ├─ import skills.time            → register("time")
        │     └─ import skills.calculator      → register("calculator")
        │
        ├─ build_system_prompt(enabled_skills)
        │     → _DEFAULT_SKILLS = [..., "dish_recommend", ...]
        │     → 组装出包含 dish_recommend Skill Prompt 的 System Prompt
        │
        └─ get_actions(enabled_skills)
              → get_tool_names_for_skills(["...", "dish_recommend", "..."])
              → 返回 {"recommend_dishes", ...}
              → 从 _ALL_TOOLS 取出 recommend_dishes 函数
              │
              ▼
        create_react_agent(model, tools=[..., recommend_dishes, ...], prompt, ...)
```

运行时调用链：

```
用户: "推荐几道难的菜"
  │
  ▼
LLM 看到 System Prompt 中 dish_recommend 的触发条件 → 决定调用 recommend_dishes
  │
  ▼
recommend_dishes(query="推荐几道难的菜")
  ├─ get_rag_service() → 拿到 RAGService 实例
  ├─ _expand_query("推荐几道难的菜")
  │     → "推荐几道难的菜 starsystem 4星 5星"
  ├─ svc.retrieve(expanded)
  │     └─ HybridRetriever.hybrid_search(...)
  │           → RetrievalResult(parents=[...], sources=[...])
  ├─ _dedup_parents(ret.parents, 15)
  │     → [{"title": "宫保鸡丁", "source": "..."}, ...]
  └─ 返回 JSON 字符串 → LLM
         │
         ▼
LLM 根据 Skill Prompt 约束，只展开 candidates 中的菜名
```

---

## 九、`.env` 配置说明

如果项目中设置了 `ENABLED_SKILLS` 环境变量（白名单模式），需追加 `dish_recommend`：

```bash
# .env（白名单模式示例）
ENABLED_SKILLS=knowledge_base,dish_recommend,time,calculator
```

如果未设置 `ENABLED_SKILLS`（默认加载全部），无需修改——`_DEFAULT_SKILLS` 已包含。

---

## 十、验证清单

| 序号 | 场景 | 期望结果 | 验证方式 |
|:----:|------|----------|----------|
| 1 | Skill 注册 | System Prompt 中出现 `【Skill: recommend_dishes】` | `python -c "from prompts.system_prompts import SYSTEM_PROMPT; print(SYSTEM_PROMPT)"` |
| 2 | Tool 注册 | `get_actions()` 返回列表中包含 `recommend_dishes` | `python -c "from actions.basic_tools import get_actions; print([t.name for t in get_actions()])"` |
| 3 | 推荐路由 | "推荐几道难的菜" → 调用 `recommend_dishes` | 查看日志 `[TOOL_CALL] name=recommend_dishes` |
| 4 | 做法路由 | "宫保鸡丁怎么做" → 调用 `search_knowledge_base` | 查看日志 `[TOOL_CALL] name=search_knowledge_base` |
| 5 | 防幻觉 | 回复中的菜名均可在 `candidates` JSON 中找到 | 对比日志中 `[TOOL_RESULT]` 与最终回复 |
| 6 | RAG 未启用 | `recommend_dishes` 返回 `ok: false, error: "service_not_initialized"` | 关闭 `RAG_ENABLED` 后测试 |
| 7 | 空结果 | 回复承认"知识库未找到"，不编造菜名 | 传入极偏僻查询 |
| 8 | sanitize 兼容 | 调用 `recommend_dishes` 后，回复中的"根据知识库"不被误拦截 | 查看日志中无 sanitize 改写 |
| 9 | 白名单模式 | `ENABLED_SKILLS` 不含 `dish_recommend` 时，工具不可用 | 设 `ENABLED_SKILLS=knowledge_base,time` 后验证 |

---

## 十一、二期扩展备忘

以下不在本次实现范围内，记录于此供后续参考：

| 项目 | 说明 |
|------|------|
| `RAG_RECOMMEND_TOP_K` | 在 `Settings` / `RAGConfig` 中新增字段，`recommend_dishes_tools.py` 从 service 配置读取上限，而非写死 `_RECOMMEND_LIMIT = 15` |
| 列表页 Markdown 解析 | 若命中 `4Star.md` 等索引页，工具层用正则解析 `- [菜名](./path)` 链接列表，直接输出菜名，不依赖 parent title |
| 候选白名单校验 | 在 `sanitize_ungrounded_kb_claim` 中解析 `recommend_dishes` 返回的 JSON，与最终回复中的菜名做交叉校验 |
| Critic 节点 | 独立后处理节点，校验最终回复中的菜名是否全部来源于工具返回的 `candidates` |

---

## 相关文档

- [菜品推荐 Skill 方案设计](./dish-recommend-skill-plan.md)
- [Skill 模块化架构](./skill-architecture.md)
- [对话系统架构](./conversation-architecture.md)
- [KB 端到端技术流程](./kb-end-to-end-tech-flow.md)
