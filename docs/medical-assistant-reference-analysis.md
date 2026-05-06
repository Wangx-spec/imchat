# Multi-Agent-Medical-Assistant 参考项目分析与吸收方案

> 参考项目路径：[`Multi-Agent-Medical-Assistant/`](../Multi-Agent-Medical-Assistant)
> 目标：把本项目改造成一个生产级的多智能体生物医疗 agent，本文梳理参考项目的架构，并在 **不改变当前编程语言、不替换 LangGraph 框架** 的前提下，列出值得吸收的亮点与落地优先级。

---

## 一、参考项目代码架构梳理

### 1.1 整体分层

```
Multi-Agent-Medical-Assistant/
├── app.py                 # FastAPI 入口：聊天/上传图/人工校验/语音 5 个路由
├── config.py              # Config 聚合类：每个子 Agent 各自的 LLM/参数配置
├── ingest_rag_data.py     # RAG 文档入库脚本（--file / --dir）
├── agents/
│   ├── agent_decision.py  # 核心：LangGraph StateGraph 编排 + Supervisor 路由
│   ├── guardrails/        # 输入/输出安全过滤
│   ├── rag_agent/         # 6 个模块化组件：parse → summarize → chunk → index → rerank → generate
│   ├── web_search_processor_agent/   # Tavily + PubMed
│   └── image_analysis_agent/         # 3 个 CV 专家（脑肿瘤/胸片/皮损）
├── data/
│   ├── raw/               # 源 PDF（医学论文/指南）
│   ├── parsed_docs/       # Docling 解析出的 page/picture/table PNG
│   ├── qdrant_db/         # Qdrant 本地向量库（稠密 + 稀疏）
│   └── docs_db/           # LocalFileStore（父文档原文）
└── templates/ + uploads/  # Jinja2 前端 + 图片/语音临时目录
```

### 1.2 运行时 Graph 拓扑

入口 `process_query` → `create_agent_graph()`，**每次 invoke 都重新 compile** 一次图，但共享同一个 module 级 `MemorySaver`，所以状态靠 checkpointer 维护。

```
START
  ↓
analyze_input ── (guardrails 拦截) ──┐
  ↓ (正常)                          ↓
route_to_agent (LLM + JsonOutputParser 输出 {agent, reasoning, confidence})
  ↓ conditional
  ├─ CONVERSATION_AGENT ─┐
  ├─ RAG_AGENT ───┐      │
  │               │ 低置信度/答不出 → WEB_SEARCH_PROCESSOR_AGENT
  ├─ WEB_SEARCH_PROCESSOR_AGENT ─┤
  ├─ BRAIN_TUMOR_AGENT ──┤
  ├─ CHEST_XRAY_AGENT ───┤
  └─ SKIN_LESION_AGENT ──┤
                         ↓
                  check_validation (CV 结果需人工 validate)
                         ↓
                  human_validation (在 output 后追加"Human Validation Required"提示)
                         ↓
                  apply_guardrails (输出安全检查)
                         ↓
                        END
```

关键设计点：

- **AgentState** 继承 `MessagesState`，额外扩展：
  `agent_name / has_image / image_type / retrieval_confidence / bypass_routing / insufficient_info / needs_human_validation`
- **Supervisor 带置信度**：路由器输出 `{agent, reasoning, confidence}`，`confidence < 0.85` 时走 fallback（强制到 RAG）。与本项目 `docs/multi-agent-plan.md` 中"Supervisor 只路由不回答"的思路一致，但多了一个**置信度兜底**机制。
- **Agent-to-Agent Handoff**：RAG → Web Search 的 handoff 用两个维度判定：
  1. `retrieval_confidence < min_retrieval_confidence (0.40)`
  2. `insufficient_info`：扫描答案文本是否出现 "don't have enough information / cannot answer" 等兜底话术
  任一触发即转交 Web Search。
- **Human-in-the-Loop**：`/validate` 端点接收 `yes/no + comments`，以同一 `thread_id` 再次 invoke 整个 graph，靠 checkpointer 续上状态。

### 1.3 RAG 子系统（`agents/rag_agent/`）—— 6 步流水线

| 步骤 | 文件 | 关键细节 |
|------|------|----------|
| 1. Parse | `doc_parser.py` | Docling 解析 PDF：TableFormer(ACCURATE) + OCR + formula enrichment + 抽 picture/table 保存为 PNG |
| 2. Image summary | `content_processor.summarize_images` | 多模态 LLM 把每张图 → 一段医学文字描述；无关图返回 `non-informative` |
| 3. Markdown 化 + 嵌入图摘要 | `content_processor.format_document_with_images` | 用 `<!-- image_placeholder -->` 定位，替换成 `picture_counter_N + <摘要>`（后续反查图片路径） |
| 4. LLM 语义分块 | `content_processor.chunk_document` | 先按 `\n#` 粗分 → 标记 `<\|start_chunk_X\|>` → 让 LLM 返回 `split_after: 3, 5`，按 LLM 决策合并 |
| 5. Hybrid 索引 | `vectorstore_qdrant.py` | Qdrant Collection 同时建 dense（Azure ada-002）+ sparse（`Qdrant/bm25` FastEmbed），`RetrievalMode.HYBRID` 融合；父文档原文存 `LocalFileStore` |
| 6. Query-side | `query_expander.py` + `reranker.py` + `response_generator.py` | LLM 扩写 query → CrossEncoder (`ms-marco-TinyBERT-L-6`) rerank → 回复时把 `picture_counter_N` 反解为图片 URL 附在结尾 |

### 1.4 其他辅助

- **Guardrails**（`agents/guardrails/local_guardrails.py`）：两条 LLM prompt chain，一进一出。输入检查 47 条规则（防 prompt injection / 非医疗话题 / PII / 代码执行等），输出检查 10 条（幻觉/免责声明/法律）。**无外部依赖**，纯 LangChain。
- **Web 搜索**：`Tavily` + `PubMed`（后者已注释），很轻。
- **Per-agent 温度**：`config.py` 为每个 agent 单独实例化 LLM，仅差 `temperature`：
  - decision 路由 `0.1`
  - conversation `0.7`
  - RAG 检索后回答 `0.3`
  - chunker（语义分块）`0.0`
  - summarizer（图摘要）`0.5`

---

## 二、数据源与知识库构建

### 2.1 数据源

**全部是公开学术 PDF**，作者手工收集到 `data/raw/` 和 `data/raw_extras/`，4 个主题：

| 主题 | 示例文件 |
|---|---|
| 脑肿瘤综述 | `brain_tumor_2023.pdf / brain_tumor_2024.pdf / brain_tumors_ucni.pdf`（Mayfield Clinic 科普） |
| COVID 胸片 | `covid_chest_xray_2020.pdf ... 2024.pdf`（BMJ/Sci Rep/Appl Intell 等） |
| 皮肤病变 | `skin_lesion_2016.pdf / 2023.pdf`（ISIC Challenge + Medical Image Analysis survey） |
| 糖尿病（少量） | `diabetes.pdf` |

**来源**：`agents/README.md` §citations 列出了 12 篇论文全部带 DOI，直接从 Nature / Springer / BMJ / arXiv 下载 PDF。**没有外部 API 拉数据**（运行时才会调 Tavily 做 web search）。

### 2.2 知识库构建流程

运行 `python ingest_rag_data.py --dir ./data/raw`，触发 `MedicalRAG.ingest_directory`：

```
for 每个 PDF:
  1. docling → 解析得到：
     ├─ 结构化 markdown
     ├─ 每页 page_N.png
     ├─ 每张 picture-N.png、table-N.png 保存到 data/parsed_docs/
     └─ 图的 base64/uri 收集起来

  2. 多模态 LLM → 逐图生成文字摘要

  3. 把 markdown 里 <!-- image_placeholder --> 换成
     "picture_counter_3 本图展示脑肿瘤 T1 加权 MRI..."

  4. LLM 语义分块（先按 # 标题粗分，再让 LLM 合并/切分）

  5. 对每个 chunk：
     - 生成 UUID
     - 稠密向量（Azure ada-002 → 1536 维）+ 稀疏 BM25 一起入 Qdrant
     - metadata: {source: 文件名, doc_id, source_path: "http://localhost:8000/data/raw/xxx.pdf"}
     - 完整原文 chunk encode 成 bytes 存进 LocalFileStore (data/docs_db/)
```

两份存储职责分离：**Qdrant 只存向量 + 轻量 metadata → 检索时先用向量召回拿 `doc_id` → 再从 `docs_db` 取完整原文喂给 LLM**（小号 parent-document retriever pattern）。

---

## 三、当前项目完成度复盘

本节按原吸收方案重新对照当前代码。结论是：最初列为 P0/P1 的多智能体骨架能力已经基本落地，剩下的重点从“搭出 multi-agent”转向“默认启用、RAG 质量、可观测验证、持久化与引用体验”。

### 3.1 已完成目标

| 目标 | 当前状态 | 证据 |
|---|---|---|
| Supervisor 输出 JSON + confidence | 已完成 | `src/graphs/multi_agent_graph.py` 的 `_build_supervisor_node` 要求返回 `agent/reason/confidence`，并在低于 `SUPERVISOR_CONFIDENCE_THRESHOLD` 时 fallback |
| Agent-to-Agent handoff | 已完成 | `medical_kb` 根据 `low_confidence_blocked`、`insufficient_info`、时效性问题低置信等条件转交 `web_search` |
| Input/Output Guardrails | 已完成 | `src/agents/guardrails/local_guardrails.py` 提供输入拦截、输出复核和输出修复；multi graph 首尾已接入 guardrail 节点 |
| Per-agent LLM override | 已完成 | `src/agents/agent_registry.py` 已有 `LLMSpec(model/temperature/top_p)`，`web_search` 已配置独立温度 |
| Web Search specialist | 已完成但默认关闭 | `src/actions/web_search_tools.py` 已接入 Tavily；受 `TAVILY_ENABLED` 和 `TAVILY_API_KEY` 控制 |
| Parent-child RAG + 混合检索 | 已完成 | `RAGService` 用 markdown parent、child chunk、FAISS dense、BM25 sparse、RRF 合并 |
| Query expansion / query planning | 部分完成 | `LLMQueryPlanner` 已实现 query variants / core terms / entities，但只有配置 `RAG_QUERY_PLAN_API_KEY` 后启用 |
| CrossEncoder / API rerank | 部分完成 | `HybridRetriever` 已支持 Qwen rerank 和本地 `sentence_transformers.CrossEncoder`，但 `RAG_RERANK_ENABLED` 默认是 `false` |
| 回答附带参考文档 | 部分完成 | `GenerationRouter` 会追加“参考文档”，但目前主要是文本 source，不是统一可点击 URL，也没有图片引用反解 |

### 3.2 未完成或仍有风险的目标

| 缺口 | 当前表现 | 影响 |
|---|---|---|
| Multi-agent 默认未启用 | `AGENT_MODE` 默认仍是 `single` | 用户可能实际跑不到 supervisor / handoff / guardrails 完整图 |
| RAG 默认未启用 | `RAG_ENABLED` 默认是 `false` | 知识库能力依赖部署配置，开发/演示容易误判能力缺失 |
| Rerank 默认未启用 | `RAG_RERANK_ENABLED=false` | 已有质量增强能力没有默认进入主路径 |
| LLM-based semantic chunking 未实现 | 当前 `ParentChildChunker` 是 markdown header + 固定窗口 | 医学指南、论文综述等结构化长文可能被切断语义段 |
| Qdrant hybrid 未迁移 | 当前是本地 FAISS + BM25 | 短期可用，但不如 Qdrant 适合持久化、远程服务、多 collection、dense+sparse 统一检索 |
| 原文 docstore 未独立持久化 | parent docs 运行期加载，向量库只保存 FAISS index | 换索引/换 embedding 后的可恢复性、可追溯性不如参考项目 |
| source_path / 图片引用链路未统一 | 回答引用不是稳定 HTTP URL，也没有 `picture_counter_N` → 图片链接 | 生产级引用体验不足 |
| Collection-per-domain 未实现 | 目前没有按疾病/指南/文档域拆 collection | 后续扩多个医学领域时，召回边界和权限边界不清晰 |
| Human-in-the-loop validation 未实现 | 没有 `/validate` 类接口或 graph 中断/恢复流程 | 对高风险诊疗建议、影像判断类场景仍缺人工复核闭环 |
| 自动化测试不足 | 主要是 `src/tests/tmp_*` smoke 脚本 | multi-agent routing、handoff、guardrails、RAG 质量缺回归保护 |

---

## 四、参考项目仍值得吸收的点

排除伤筋动骨的大改动：不整体替换编程语言，不替换 LangGraph，不照搬 Azure OpenAI 绑定，也不把医疗 CV 模型强行接入当前主路径。

### 4.1 已经吸收完成，可保留为设计依据

1. **Supervisor confidence + fallback**
   参考项目的 `{agent, reasoning, confidence}` 已在当前 `_build_supervisor_node` 中落地。当前项目选择低置信 fallback 到 `conversation`，而不是参考项目中的 RAG，符合本项目“安全兜底先澄清/说明”的定位。

2. **RAG → Web Search handoff**
   参考项目用 `retrieval_confidence` + `insufficient_info` 双条件触发 web search。当前项目已经扩展为三类触发：RAG 低置信拦截、答案文本不足、时效性问题低置信。

3. **Guardrails 双闸**
   当前项目已经实现输入过滤、输出复核、输出修复，并接入 multi graph。后续重点不是“有没有”，而是规则调优、误杀率评估和测试。

4. **Per-agent LLM 配置**
   当前 `AgentDef.llm_override` 已经覆盖 `model/temperature/top_p`，可以继续作为 agent registry 的长期扩展点。

### 4.2 仍建议吸收或强化

1. **Web search query reframing**
   参考项目的 `web_search_processor.py` 会先把多轮对话和用户问题压缩成搜索查询，再调用 Tavily。当前项目已有 web search tool，但可以增加“搜索查询改写”步骤，减少直接拿原始口语问题搜索导致的召回噪声。

2. **PubMed provider 作为可选搜索源**
   参考项目保留了 `pubmed_search.py`。如果本项目定位是生物医学 agent，而不是泛健康问答，PubMed/NCBI 这类 provider 比普通网页搜索更适合作为高可信补充源。

3. **LLM-based semantic chunking**
   参考项目 `content_processor.chunk_document` 的思路仍值得迁移：先按 markdown 标题粗分，再让 LLM 决定合并/切分边界。这个点对医学指南、论文综述、疾病条目尤其有价值。

4. **Qdrant hybrid + docstore 分离**
   参考项目的 `vectorstore_qdrant.py` 同时存 dense/sparse，并用 `LocalFileStore` 保存原文。当前 FAISS + BM25 可继续作为轻量本地方案，但生产化和多领域扩展更适合 Qdrant。

5. **Response source/image contract**
   参考项目在回答尾部统一输出 `Source documents` 和 `Reference images`。当前项目已经有“参考文档”，下一步应统一 `source_path` 为可点击 URL；如果引入 PDF/图表解析，再追加图片引用。

6. **Ingest CLI、healthcheck、Docker/CI**
   参考项目有 `ingest_rag_data.py`、`/health`、Dockerfile 和 docker build workflow。当前项目已有 FastAPI，但可以把 RAG 入库、健康检查和容器化流程文档化/脚本化，降低部署成本。

7. **Human validation 作为高风险能力预留**
   参考项目的 `/validate` 和 `needs_human_validation` 对 CV 诊断类 agent 很重要。当前可以暂不实现，但如果未来做影像、用药、慢病管理建议，应把 HITL 作为准入条件。

### 4.3 仍不建议吸收

- **整套 Docling PDF 解析链路**：除非确定知识源以 PDF 论文/指南为主，否则依赖和系统成本偏高。
- **脑肿瘤 / 胸片 / 皮损 CV 子 agent**：当前项目主线是生物医疗问答/RAG，不应先引入重模型。
- **ElevenLabs 语音**：不是 multi-agent 或 RAG 质量的核心瓶颈。
- **全套 Azure OpenAI 配置**：当前已经有 OpenAI-compatible/Qwen 配置，保留现状即可。

---

## 五、知识库构建思路的迁移建议

从 Medical 项目的做法对照本项目 `src/rag/`，核心迁移建议调整为四个层级：

1. **先把现有能力默认化和可验证化**
   multi-agent、RAG、rerank、query planner 都已有代码路径，但部分依赖 env 开关。下一步应先明确默认运行形态、README 配置和 smoke/test 覆盖。

2. **增强 chunk 与 query 两端质量**
   当前检索链路已经比参考项目早期版本更完整：有 FAISS dense、BM25 sparse、RRF、query planner、rerank。短期收益最高的是启用 rerank、调 query planner、补 semantic chunking，而不是马上换向量库。

3. **双 store 分离（向量库 + 原文 docstore）**
   参考项目的向量库只负责召回 id，原文单独存。这个模式仍值得迁移，尤其是后续要重建索引、换 embedding 模型、或做引用审计时。

4. **source_path 与 collection-per-domain**
   source 应统一为可点击 URL 或稳定资源 id。未来若拆 “医学指南 / 论文综述 / 患者科普 / 药物说明” 多领域，可让 Supervisor 决定 domain，再映射到不同 collection 或 index。

---

## 六、未完成目标优先级

下面按“对用户可见能力提升 + 实现风险 + 对后续工作的依赖关系”排序。

| 优先级 | 未完成目标 | 为什么排这里 | 建议落点 | 预计工作量 |
|---|---|---|---|---|
| P0 | 明确默认运行形态：`AGENT_MODE=multi`、`RAG_ENABLED`、`TAVILY_ENABLED` 的开发/生产配置 | 现有核心能力已写好，但默认配置可能让用户实际跑不到 | `.env.example`、README、启动脚本、`settings.py` 默认策略 | 0.5 天 |
| P0 | 为 supervisor / handoff / guardrails / RAG 低置信补 smoke 或单元测试 | 已完成能力需要回归保护，否则后续调 prompt 很容易破坏路由 | `src/tests/` 或正式 `tests/` | 1 天 |
| P1 | 默认启用或配置化启用 rerank，并完成一组 RAG 质量样例 | 这是现有 RAG 质量增益最大的低成本项 | `settings.py`、`retriever.py`、测试样例 | 0.5-1 天 |
| P1 | Web search query reframing + PubMed provider 预留 | 提升联网补充质量，适合医学 agent 定位 | `src/actions/web_search_tools.py` 或新增 web search processor | 1 天 |
| P1 | 统一 source_path / 可点击引用 | 直接改善回答可信度和产品体验，也为审计做准备 | `data_loader.py`、`generation_router.py`、FastAPI static mount | 0.5-1 天 |
| P2 | LLM-based semantic chunking | 改善长医学文档召回，依赖入库成本和 prompt 稳定性 | `src/rag/ingestion/chunking.py` | 1-2 天 |
| P2 | 独立 docstore（原文持久化） | 为后续 Qdrant、多 collection、索引重建打基础 | `src/rag/retrieval/` 或新增 `docstore.py` | 1 天 |
| P3 | Qdrant hybrid 替换/并存 FAISS | 生产化收益高，但涉及依赖、迁移和部署 | `src/rag/retrieval/index_store.py` 抽象化后新增 Qdrant store | 2-3 天 |
| P3 | Collection-per-domain / domain routing | 需要知识源规模变大后再做，否则容易过早设计 | agent registry + RAG config + retriever | 2 天 |
| P4 | Human-in-the-loop validation | 只有进入高风险诊疗、影像、用药建议时才必要 | graph state、API `/validate`、checkpointer resume | 2-3 天 |
| P4 | 图片引用 / 图表解析 / CV specialist | 当前不是主线，除非知识源转向 PDF 图表或影像诊断 | ingestion + generation + specialist agent | 3 天以上 |

**下一步最建议先做 P0 + P1 前三项**：把现有 multi-agent 能力稳定跑起来，补最小测试，再打开/验证 rerank。这样能最快把当前项目从“功能已实现但靠配置触发”推进到“默认路径稳定可演示”。

---

## 七、相关文档

- [多智能体协作方案](./multi-agent-plan.md)：当前规划的 single → multi 改造路径
- [Skill 模块化架构](./skill-architecture.md)：Skill 注册表、Tool 注册表、Prompt 组装机制
- [知识库端到端技术流](./kb-end-to-end-tech-flow.md)：当前 RAG 实现细节
- [对话架构总览](./conversation-architecture.md)：CLI/Web 入口、Checkpointer、messages 表
