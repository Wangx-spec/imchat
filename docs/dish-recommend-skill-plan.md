# 菜品推荐 Skill 实现方案

---

## 一、目标与原则

| 目标 | 说明 |
|------|------|
| 新 Skill | 根据用户问题和本地知识库，**优先推荐菜谱名**（及可选来源路径） |
| 与 `search_knowledge_base` 分工 | **详情 / 做法 / 单菜详解** → `search_knowledge_base`；**多道菜推荐、筛选、想吃啥类** → `recommend_dishes` |
| 防幻觉 | 工具返回**结构化候选列表**（全部来自检索到的 parent 文档）；Skill Prompt 要求最终回复**只展开列表内条目**，禁止自造菜名 |

---

## 二、架构选择

新增独立 Tool：**`recommend_dishes(query: str)`**，内部复用已有 `RAGService`，但**走 `retrieve()`，不走 `answer()`**。

**原因：** `answer()` 会经过 `GenerationRouter` 的 list/detail/general 拼装；推荐场景更需要「多 parent、以 title 为主」的可控输出。`retrieve()` 直接返回 parent 列表，便于只抽取 `metadata.title` / `source`，与星级索引页（如 `starsystem/4Star.md`）等大列表类文档对齐。

可选：若希望召回更偏列表类文档，可对服务端传入的 `query` 做轻量规则扩展（例如用户问「高难度」时自动附带 `starsystem`、`4星` 等关键词），提高命中星级索引页的概率。

---

## 三、涉及文件（落地时）

| 操作 | 路径 |
|------|------|
| 新建 | `src/actions/dishes/recommend_dishes_tools.py` |
| 新建 | `src/prompts/skills/dish_recommend.py` |
| 修改 | `src/actions/basic_tools.py`（`_ALL_TOOLS` + import） |
| 修改 | `src/prompts/system_prompts.py`（import skill、`_PRIORITY_MAP`、`_DEFAULT_SKILLS`） |
| 修改 | `docs/skill-architecture.md`（现有 Skill 一览表补一行） |
| 可选 | `src/actions/knowledge_base_tools.py` | 增加 `get_rag_service()`，供推荐工具读取同一 `RAGService` 实例 |
| 可选 | `src/services/chat_service.py` | 扩展 `sanitize_ungrounded_kb_claim`，覆盖「推荐」话术与工具调用关系（见下文） |

`rag/bootstrap.py` 已通过 `set_rag_service` 绑定服务。推荐工具需要访问同一实例，推荐做法：

1. 在 `knowledge_base_tools.py` 增加 `get_rag_service() -> Any | None`，与 `set_rag_service` 配对；**或**
2. 在 `bootstrap` 里对推荐模块再绑一次同一 `service` 引用（避免直接读私有变量）。

---

## 四、Tool 契约

```text
recommend_dishes(query: str) -> str   # 返回 JSON 字符串
```

### 输入

- `query`：用户原意，可包含难度、口味、人数、忌口等；**要求传入用户语言的原始表述，不要随意翻译**（与知识库 Skill 的输入规范一致）。

### 输出 JSON 建议字段

```json
{
  "type": "dish_recommendation_result",
  "ok": true,
  "query": "...",
  "candidates": [
    {"title": "宫保鸡丁", "source": "source_dir/file_data/..."},
    "..."
  ],
  "sources_distinct": ["..."],
  "limit": 15,
  "error": null,
  "debug": {}
}
```

- `candidates`：**仅从本次 `retrieve()` 返回的 `parents` 去重得到**，每条至少包含 `title`（来自 `Document.metadata["title"]`），`source` 可选。
- 条数：可与 `rag_top_k` 区分，例如在 `Settings` / 环境变量中增加 `RAG_RECOMMEND_TOP_K`（略高于一般问答的 `top_k`），检索时传给 `HybridRetriever`。

### 内部逻辑（伪流程）

1. `svc = get_rag_service()`；未就绪则返回 `ok: false`，`error: "service_not_initialized"` 等。
2. `ret = svc.retrieve(expanded_query)`，其中 `expanded_query` 可对「推荐 / 高难度 / 星星」类意图做服务端规则扩展。
3. 从 `ret.parents` 聚合文档，按 parent 去重，截取前 `N` 条。
4. 将 `title`、`source` 写入 `candidates`。

### 进阶（可选，二期）

若命中的 parent 是 `4Star.md` 这类**列表页**，正文为 Markdown 链接列表 `- [菜名](./path)`，可在工具层用正则解析出**菜名列表**，输出仍只含解析结果，不调用 LLM 生成新菜名。这样「高难度推荐」与源文件逐行一致，grounding 最强。

---

## 五、Skill Prompt（`src/prompts/skills/dish_recommend.py`）

编写要点：

1. **触发条件**：推荐多道菜、想吃啥、有啥菜、按难度/口味/场景筛选、不知道做啥等。
2. **与 `search_knowledge_base` 的边界**：要问**具体步骤、单菜详解、章节教程** → 用 `search_knowledge_base`；只要**名单式推荐** → 用 `recommend_dishes`。
3. **输出约束**：最终回复中的菜名**必须来自** `candidates`（或工具明确解析出的列表）；若 `candidates` 为空或很少，如实说明并引导换问法，**禁止编造菜名**。
4. **语言**：与用户最新消息一致。

---

## 六、`system_prompts.py` 调整清单

1. 增加一行：`import prompts.skills.dish_recommend  # noqa: F401`，触发注册。
2. `_DEFAULT_SKILLS` 中加入 `dish_recommend`（若希望「推荐类」优先路由，可将其排在 `knowledge_base` 之前）。
3. `_PRIORITY_MAP["dish_recommend"] = "多菜推荐/想吃啥/按条件选菜 → recommend_dishes"`（文案可按产品微调）。

---

## 七、与 `sanitize_ungrounded_kb_claim` 的关系

当前逻辑：若未调用 `search_knowledge_base` 却出现「根据知识库」类话术，则改写。

建议后续扩展：

- 若声称「来自知识库的推荐」但**既未调用 `search_knowledge_base` 也未调用 `recommend_dishes`**，同样视为 ungrounded。
- **更强约束（二期）**：若调用了 `recommend_dishes`，对最终回复中出现的菜名与 `candidates` 做白名单校验；未命中则 prepend 警示或删改——需解析 tool 返回的 JSON 与助手文本，实现成本高于仅 Prompt。

**第一期**可只做 Tool + Skill Prompt + 优先级；**第二期**再加强后处理。

---

## 八、配置

| 配置 | 说明 |
|------|------|
| `ENABLED_SKILLS` | 白名单模式下追加 `dish_recommend` |
| 可选 `RAG_RECOMMEND_TOP_K` | 推荐场景拉取的 parent 数量上限（默认可略高于 `RAG_TOP_K`） |

---

## 九、测试清单

| 场景 | 期望 |
|------|------|
| 「推荐几道难的菜」 | 调用 `recommend_dishes`，回复中的菜名应可追溯至工具返回的 `candidates`（或二期解析列表） |
| 「宫保鸡丁怎么做」 | 走 `search_knowledge_base`，不走推荐 Tool |
| RAG 未启用 | `recommend_dishes` 返回 `ok: false`，模型不伪装已有知识库 |
| 命中 `4Star.md` | 若实现二期 Markdown 解析，菜名应与文件内链一致 |

---

## 十、与 Multi-Agent / Critic 的关系

- **不需要** Supervisor：单 ReAct Agent + 新 Tool 即可。
- **Critic** 可作为后续增强；本方案通过「结构化候选 + Prompt 约束」先降低幻觉，与独立 Critic 节点不冲突。

---

## 十一、参考

- 通用 Skill 注册与配置：见 [skill-architecture.md](./skill-architecture.md)。
- 星级索引页示例：`source_dir/file_data/starsystem/4Star.md`、`5Star.md`。
