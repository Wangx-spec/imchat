# 医疗助手亮点迁移缺口清单

> 基于 `docs/medical-assistant-reference-analysis.md` 的 10 项吸收建议，对当前代码现状的补齐清单。  
> 结论快照：已实现 6 项，部分实现 1 项，未实现 3 项。

---

## 1) 总览状态

- [x] 1. Supervisor JSON + confidence + fallback
- [x] 2. Agent-to-Agent handoff（已补文本兜底探测）
- [x] 3. Input/Output Guardrails 双闸
- [x] 4. Per-agent LLM override（已在 specialist 构建时生效）
- [ ] 5. LLM-based 语义分块
- [ ] 6. Qdrant Hybrid 替换 FAISS
- [x] 7. Query Expansion（LLM 查询扩写）
- [x] 8. CrossEncoder Rerank
- [~] 9. Response 附带源文档 + 引用图片（仅文档来源，缺图片引用反解）
- [ ] 10. Human-in-the-loop Validation

---

## 2) P1 优先补齐（建议先做）

### [x] A. Handoff 增加文本兜底探测（补齐第 2 项）

- 目标：在 `medical_kb -> web_search` 条件中，除了置信度，还要识别回答文本中的“信息不足”信号。
- 改动点：
  - `src/graphs/multi_agent_graph.py`
    - 在 `_build_specialist_node()` 里读取 `kb_result.answer`
    - 增加 `insufficient_info` 文本探测函数（中英文短语表）
    - `should_handoff = low_confidence or (not is_confident and confidence_score < 0.45) or insufficient_info`
- 验收标准：
  - 低置信度问题仍可触发 handoff
  - 即使置信度字段异常，只要出现“信息不足”文本也会 handoff
  - 日志能区分 `handoff_reason`（如 `medical_kb_insufficient_info`）
  - 当前状态：已完成，落地在 `src/graphs/multi_agent_graph.py`

### [x] B. Per-agent LLM override 真正生效（补齐第 4 项）

- 目标：让 `AgentDef` 的覆盖配置在 specialist 构建时实际使用。
- 改动点：
  - `src/agents/agent_registry.py`
    - 将 `model_override` 扩展为 `llm_override`（建议含 `model/temperature/top_p`）
  - `src/graphs/multi_agent_graph.py`
    - `_build_specialist_node()` 内按 agent 覆写默认 LLM
  - `src/llms/openai_chat.py`
    - 增加一个“按参数构建 ChatOpenAI”的辅助方法，避免复制逻辑
- 验收标准：
  - 不同 agent 在日志中可看到不同模型或温度
  - 未配置 override 的 agent 保持现有默认行为
  - 当前状态：已完成，落地在 `src/agents/agent_registry.py`、`src/llms/openai_chat.py`、`src/graphs/multi_agent_graph.py`

### [ ] G. Output Guardrail 改为 PASS/FAIL 两阶段（稳定性修复）

- 目标：避免输出护栏把“审核说明”当作最终答案返回给用户。
- 改动点：
  - `src/agents/guardrails/local_guardrails.py`
    - 第一阶段只输出 `PASS` 或 `FAIL: reason`
    - 只有 `FAIL` 时才进入第二阶段重写
- 验收标准：
  - 合格回答直接原样返回
  - 不再出现“合格/不合格说明”覆盖正文
  - 日志中可看到 `fail reason=...`

---

## 3) P2 质量提升项

### [ ] C. LLM-based semantic chunking（补齐第 5 项）

- 目标：从固定窗口切分升级为“标题粗分 + LLM 决策合并/切分”。
- 改动点：
  - `src/rag/ingestion/chunking.py`
    - 保留当前 header split 作为第一步
    - 新增可选 LLM 分块策略（配置开关）
  - `src/rag/core/config.py` / `src/config/settings.py`
    - 增加分块策略配置（如 `RAG_CHUNKING_MODE=rule|llm`）
- 验收标准：
  - 关闭开关时行为与现状一致
  - 开启后 chunk 边界更贴合语义段落，且不明显增大空块/超长块比例

### [ ] D. Response 引用增强（补齐第 9 项剩余部分）

- 目标：在现有“参考文档”基础上，补充图片引用能力与可点击来源链接。
- 改动点：
  - `src/rag/ingestion/data_loader.py`
    - 为图片占位符保留反解信息（如 `picture_counter_n -> path/url`）
    - 优化 `source` 为可访问 URL（而非本地绝对路径）
  - `src/rag/generation/generation_router.py`
    - 追加 `Reference images` 区块（有图时展示）
- 验收标准：
  - 回答中出现图片引用时，尾部有对应图片链接
  - `参考文档` 链接在前端可直接点击访问

---

## 4) P3 架构级补齐

### [ ] E. FAISS -> Qdrant Hybrid（补齐第 6 项）

- 目标：改为 dense + sparse 原生混合检索，减少手工融合复杂度。
- 改动点：
  - `src/rag/retrieval/index_store.py`
    - 新建 Qdrant store 实现并保留 FAISS 兼容路径
  - `src/rag/retrieval/retriever.py`
    - 统一接口调用，确保可切换后端
  - `requirements.txt`
    - 增加 `qdrant-client`（如需本地 sparse 还需相应依赖）
- 验收标准：
  - 配置切换后可在同一套 API 下运行
  - 索引构建、加载、检索链路均通过 smoke test

### [ ] F. Human-in-the-loop Validation（补齐第 10 项）

- 目标：增加可选人工确认流程（主要用于高风险医疗问答）。
- 改动点：
  - `src/controllers/` 新增验证路由（如 `/validate`）
  - `src/graphs/multi_agent_graph.py` 增加 `needs_human_validation` 分支
  - `src/services/chat_service.py` 支持同 `thread_id` 续跑
- 验收标准：
  - 可在一次会话中发起“待人工确认”并续跑返回修订结果
  - 默认路径不启用该分支，避免影响现有体验

---

## 5) 建议执行顺序

1. G（output guardrail PASS/FAIL 稳定性修复）
2. C（semantic chunking）
3. D（引用增强）
4. E（Qdrant hybrid）
5. F（human validation）

---

## 6) 回归检查清单（每个阶段都做）

- [ ] 单轮多智能体问答不出现重复拼接
- [ ] `TAVILY_ENABLED=false` 时 `web_search` 不实际调用 Tavily
- [x] `medical_kb` 低置信/信息不足能稳定 handoff 到 `web_search`
- [ ] 输出护栏不再把“审核说明”替换成最终回答
- [ ] RAG 检索 debug 字段完整（query plan / rerank / confidence）

