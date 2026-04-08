# RAG LLM Query Planner 重构方案

## 目标

- 让 LLM 成为 query 信息提取与检索规划的唯一入口。
- 减少 `retriever` 中规则化 query 处理逻辑，降低冗余与维护成本。
- 保持 RAG 主流程稳定，LLM 失败时可自动降级。

## 当前痛点

- `exact_match_hit` 依赖字符串包含，容易出现“已召回但未命中”的假阴性。
- `retriever` 同时存在规则扩展、规则关键词提取、规则重排，与 LLM 规划能力重叠。
- 日志中难以直接判断 planner 是否生效（调用成功、输出质量、降级原因）。

## 目标架构

`User Query -> LLMQueryPlanner -> HybridRetriever(执行层) -> Rerank -> Evidence/Confidence -> AnswerGuard`

原则：

- Planner 负责“理解问题”；Retriever 负责“执行检索”。
- Guard 使用证据分数，不使用单一 `exact_match_hit` 硬门槛。
- 全链路输出结构化 debug 字段，支持快速定位。

## 结构化输出协议（QueryPlan）

建议 `QueryPlan` 包含：

- `original_query: str`
- `normalized_query: str`
- `core_terms: list[str]`
- `query_variants: list[str]`
- `intent_hint: str`（`detail/list/general`）
- `entities: list[str]`（菜名/食材/人数/场景）
- `constraints: dict[str, str | int | float]`
- `confidence: float`（0~1）
- `used_llm: bool`
- `error: str | None`

要求：

- `query_variants` 2~5 条（首条最接近主语义）。
- `core_terms` 2~6 条，不允许整句原样返回。
- 输出必须是 JSON，无解释文本。

## 按文件改造清单

### 1) `src/prompts/knowledge_base_prompt.py`

- 升级 `build_query_prompt()` 的输出协议说明，覆盖新增字段。
- 增加严格约束：仅输出 JSON、字段类型/范围要求。

### 2) `src/rag/query_planner.py`

- 扩展 `QueryPlan` dataclass 字段。
- `plan()` 增加 schema 级清洗与校验：
  - 空值填充、类型校验、长度截断、去重。
  - `confidence` 非法值自动夹逼到 `[0,1]`。
- 降级策略：
  - `used_llm=False`
  - `normalized_query=query`
  - `query_variants=[query]`
  - `core_terms` 使用轻量拆词兜底（非整句）。

### 3) `src/rag/retriever.py`

- 保留执行层能力：
  - `vector_search()`
  - `bm25_search()`
  - `rrf_fuse()`
  - `child_to_parent()`
  - `_rerank_parents_by_qwen()`
- 下线或降级为 fallback-only 的规则逻辑：
  - `_expand_queries()`
  - `_extract_query_tokens()`
  - `_score_parent_match()`
  - `_rerank_parents_by_query()`
  - `_extract_core_phrase()`
- 将命中判定从 `exact_match_hit` 改为证据分数：
  - `confidence_score`
  - `is_confident`
  - `confidence_reasons`

### 4) `src/rag/service.py`

- `retrieve()` 统一先拿 `query_plan`，再传给 `hybrid_search()`。
- `answer()` 的 detail 门控改为：
  - 基于 `confidence_score` + 证据项（如 direct hits/top title overlap）
  - 不再只依赖 `exact_match_hit`。

### 5) `src/actions/knowledge_base_tools.py`

- `TOOL_KB_DEBUG` 增加关键字段打印：
  - `query_plan_used_llm`
  - `query_plan_error`
  - `query_normalized`
  - `query_core_terms`
  - `query_variants_count`
  - `confidence_score`
  - `confidence_reasons`

## 分阶段实施

### Phase A（低风险）

- 扩展 QueryPlan 协议与 prompt。
- retriever 先“消费新字段”，暂不删除旧规则函数。
- 增加 debug 字段与日志。

验收：

- 查询日志中可看到 planner 是否生效及错误原因。
- `query_variants` 不再长期只有 1 条原句。

### Phase B（核心切换）

- Guard 切到 `confidence_score`。
- 旧 `exact_match_hit` 仅作为辅助观测，不参与阻断。

验收：

- 出现 `direct_hit_titles` 时，不再高频误判未命中。
- detail 问答的 false negative 显著下降。

### Phase C（减冗余）

- 删除 retriever 中重复规则逻辑（或仅保留兜底）。
- 文档化新链路，补充回归测试用例。

验收：

- 代码复杂度下降，行为更稳定可解释。

## 测试与验证清单

- 用例 1：口语 query（如“白菜猪肉炖粉条咋做”）应被规范化并命中正确文档。
- 用例 2：同义表达（“做法/怎么做/步骤”）命中结果一致。
- 用例 3：查询无关问题应正确触发低置信度保护。
- 用例 4：LLM planner 超时/异常时系统仍可返回可控结果。
- 用例 5：查看 payload/debug，确认关键字段完整输出。

## 风险与回滚

- 风险：LLM 输出不稳定导致计划质量波动。
  - 处理：强校验 + 降级兜底 + rerank 保底。
- 风险：切换期间线上表现抖动。
  - 处理：分阶段开关（先观测再切换 guard）。

回滚策略：

- 保留旧 guard 路径与旧规则函数一个版本周期。
- 若新逻辑异常，可通过配置开关切回旧路径。
