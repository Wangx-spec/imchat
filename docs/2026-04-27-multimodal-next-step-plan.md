# 2026-04-27 多模态医疗助手下一步实现方案

> 目标：在当前已具备 multi-agent、RAG、guardrails、handoff 能力的基础上，先稳定主路径，再以低风险方式引入多模态能力。

---

## 一、当前判断

当前项目已经不是“从零搭 multi-agent”的阶段，而是进入“把已有能力稳定变成默认主路径”的阶段。

已具备的关键基础：

| 能力 | 当前状态 | 说明 |
|---|---|---|
| Multi-agent graph | 已实现 | `src/graphs/multi_agent_graph.py` 已有 supervisor、specialist、guardrails、handoff |
| Supervisor confidence | 已实现 | supervisor 返回 `agent/reason/confidence`，低置信 fallback |
| KB → Web Search handoff | 已实现 | RAG 低置信、答案不足、时效性低置信可转交 `web_search` |
| Guardrails | 已实现 | 输入拦截、输出复核、输出修复已接入 multi graph |
| Per-agent LLM override | 已实现 | `AgentDef.llm_override` 支持 `model/temperature/top_p` |
| RAG 混合检索 | 已实现 | FAISS dense + BM25 sparse + RRF |
| Query planner / rerank | 部分实现 | 代码已有，但依赖配置开关和 API key |
| Qwen-VL 调用方式 | 已确认 | 采用 DashScope OpenAI-compatible `/chat/completions` 接口，模型先定为 `qwen3.6-plus` |

因此下一步不建议马上重构 Qdrant、Docling 或 CV 专家模型，而应先完成稳定化和可验证化。

---

## 二、下一步优先级

### P0：稳定当前 multi-agent 主路径

目标：确保本项目默认能跑到已经实现的 multi-agent 能力，而不是停留在 single-agent 或未启用 RAG 的状态。

建议任务：

| 任务 | 说明 | 预计工作量 |
|---|---|---|
| 明确默认运行配置 | 梳理 `AGENT_MODE=multi`、`RAG_ENABLED=true`、`TAVILY_ENABLED` 的开发/生产推荐配置 | 0.5 天 |
| 更新 `.env.example` 和 README | 明确多智能体、RAG、联网搜索、rerank 的启用方式 | 0.5 天 |
| 增加启动自检 | Web 启动时输出当前 agent mode、RAG ready、Tavily enabled、rerank enabled | 0.5 天 |

验收标准：

- 开发环境可以明确以 multi-agent 模式启动。
- 日志能看出 supervisor、guardrails、KB、web_search 是否启用。
- 用户不会因为默认配置而误以为 multi-agent/RAG 不存在。

### P0：补最小回归测试

目标：保护已经完成的核心能力，避免后续接多模态时破坏路由、handoff 和安全边界。

建议测试覆盖：

| 测试点 | 验证内容 |
|---|---|
| Supervisor routing | 医学知识问题走 `medical_kb`，最新指南/近期研究走 `web_search`，闲聊走 `conversation` |
| Low-confidence handoff | KB 返回低置信或“信息不足”时转交 `web_search` |
| Input guardrail | 非医疗、prompt injection、危险请求能被拦截 |
| Output guardrail | 缺免责声明或过度诊断性输出能被修复 |
| RAG answer guard | 医学问题低置信时不硬答，返回安全兜底 |

建议先做轻量 smoke / unit test，不必一开始追求完整 E2E。

### P1：启用并验证 rerank

目标：用最小代价提升 RAG 召回后的排序质量。

当前 `HybridRetriever` 已支持两种 rerank：

- Qwen rerank API
- 本地 `sentence_transformers.CrossEncoder`

建议路线：

1. 先保留默认关闭，但补一组明确的配置样例。
2. 准备 5-10 个医学知识库问题作为质量样例。
3. 对比 rerank 前后的 top source、confidence score、最终回答。
4. 若收益稳定，再考虑在开发/演示环境默认开启。

验收标准：

- 能用固定样例说明 rerank 是否提升命中质量。
- rerank 失败时系统可降级，不影响基础 RAG。

---

## 三、多模态需求是否现在可以做

可以做，但建议先做“轻量多模态 RAG / Qwen-VL 辅助理解”，不要一上来做医学影像诊断 CV 专家。

本项目多模态 LLM 明确采用 Qwen-VL 路线，优先使用 DashScope 的 OpenAI-compatible 调用方式：

| 配置项 | 建议值 | 说明 |
|---|---|---|
| `MULTIMODAL_PROVIDER` | `dashscope` | 多模态供应商，第一版固定为 DashScope 即可 |
| `MULTIMODAL_MODEL` | `qwen3.6-plus` | 当前已确认可用的 Qwen-VL 模型 |
| `MULTIMODAL_API_KEY` | 为空时回退 `DASHSCOPE_API_KEY` | 避免和文本模型 key 强绑定 |
| `MULTIMODAL_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible base url |
| `MULTIMODAL_TIMEOUT_MS` | `10000` | 图片理解比文本调用更慢，建议独立超时 |

先用 curl 验证 key、模型名、图片输入格式都可用，再接入代码：

```bash
curl --location 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions' \
  --header "Authorization: Bearer $DASHSCOPE_API_KEY" \
  --header 'Content-Type: application/json' \
  --data '{
    "model": "qwen3.6-plus",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"}},
          {"type": "text", "text": "图中描绘的是什么景象?"}
        ]
      }
    ]
  }'
```

代码接入时不建议复用 `src/llms/openai_chat.py` 里的文本 LLM builder。更稳妥的做法是在 `src/llms/` 新增 `qwen_vl.py` 或 `multimodal_chat.py`，只负责图片描述、图片类型识别和非诊断性视觉摘要。

原因：

1. 当前项目已经有 multi-agent graph 和 agent registry，足够承接新的 `multimodal` agent。
2. 当前 RAG 已经可以把文本摘要入库，多模态第一阶段可以把图片理解结果转成文本摘要复用现有链路。
3. Qwen-VL 已有明确调用方式，可以先作为图片摘要器接入，不影响现有文本 LLM 和 RAG 链路。
4. 医学影像诊断风险高，需要模型验证、人工复核、免责声明和可能的合规约束，不适合作为第一阶段。

---

## 四、多模态 MVP 路线

> 阶段 1 + 阶段 2 的完整代码实现方案见：[`docs/2026-04-27-multimodal-mvp-impl-plan.md`](./2026-04-27-multimodal-mvp-impl-plan.md)。

### 阶段 1：图片摘要入库 + 图片引用回答

这是最推荐的第一阶段。

目标：让知识库支持“文档里的图片/图表/示意图”，并在回答中能引用相关图片。

实现思路：

1. 在 ingestion 阶段支持图片资源：
   - 保存图片到稳定目录，例如 `data/rag_assets/images/`
   - 为每张图片生成 `image_id`、`image_path`、`source_doc`、`page` 等 metadata

2. 用 Qwen-VL 生成图片摘要：
   - 对图表、医学影像示意图、流程图生成结构化描述
   - 对无信息图片标记为 `non-informative`
   - prompt 固定要求“只描述图片中可见内容，不确定则说明不确定，不输出诊断结论”
   - 调用失败时记录错误并跳过该图片，不阻塞整批 ingestion

3. 把图片摘要作为文本 chunk 入库：
   - chunk 内容包含 `image_id` 和摘要
   - metadata 保留 `image_path`，用于回答时反查

4. 回答时附带参考图片：
   - `GenerationRouter` 在参考文档后追加“参考图片”
   - 图片路径应是可点击 URL 或稳定静态资源路径

建议输出格式：

```markdown
参考文档：
[1] 文档标题 | source_path

参考图片：
[1] 图 2：脑肿瘤 MRI 影像示意 | /static/rag_assets/images/xxx.png
```

适合场景：

- 医学指南中的流程图
- 论文中的图表
- 疾病科普中的解剖/影像示意图
- 检查报告或图像截图的辅助理解

不做事项：

- 不直接输出诊断结论
- 不做“模型判断是良性/恶性”
- 不替代医生阅片

### 阶段 2：用户上传图片问答

目标：用户可以上传图片并提问，系统先做图片描述，再结合 RAG 或 Web Search 回答。

建议流程：

```text
用户上传图片 + 问题
  ↓
输入 guardrail：判断是否允许处理
  ↓
Qwen-VL image caption：生成非诊断性图片描述
  ↓
image type gate：识别普通图 / 医学图表 / 影像类 / 不支持
  ↓
把“图片描述 + 用户问题”交给 supervisor
  ↓
medical_kb / web_search / conversation
  ↓
output guardrail：补免责声明、避免过度诊断
```

需要新增能力：

| 能力 | 建议落点 |
|---|---|
| 图片上传接口 | `src/web/` 下新增 upload endpoint 或扩展 chat endpoint |
| 图片临时存储 | `data/uploads/` 或 `data/tmp_uploads/` |
| Qwen-VL client | `src/llms/` 新增 `qwen_vl.py` 或 `multimodal_chat.py` |
| 多模态 agent | `src/agents/agent_registry.py` 注册 `multimodal` 或 `image_understanding` |
| 图片安全规则 | 扩展 `LocalGuardrails` 输入规则 |

输出原则：

- 可以描述图片中可见内容。
- 可以解释相关医学知识。
- 可以建议用户咨询专业医生。
- 不给确定诊断，不给处方，不声称替代影像科/皮肤科医生。

### 阶段 3：专科图像 agent

只有满足以下条件后再做：

- 明确领域：例如脑 MRI、胸片、皮损之一。
- 有可验证模型或 API。
- 有测试集或人工标注样例。
- 有 human-in-the-loop validation。
- 有强免责声明和拒答策略。

可参考 `Multi-Agent-Medical-Assistant`：

| 参考能力 | 借鉴方式 |
|---|---|
| `image_classifier.py` | 先用 Qwen-VL 做图像类型 gate，再决定是否调用专科模型 |
| `brain_tumor_agent` / `chest_xray_agent` / `skin_lesion_agent` | 只作为结构参考，不建议直接接入当前主线 |
| `needs_human_validation` | 高风险图像结论必须进入人工复核 |
| `/validate` | 可作为后续 HITL API 参考 |

---

## 五、推荐实现顺序

综合当前状态和多模态目标，建议顺序如下：

| 顺序 | 任务 | 类型 | 原因 |
|---|---|---|---|
| 1 | 稳定 multi-agent/RAG 默认配置 | 基础设施 | 先确保已有能力真实进入主路径 |
| 2 | 补 supervisor、handoff、guardrails、RAG 低置信测试 | 质量保障 | 多模态会增加分支，先保护旧路径 |
| 3 | 启用并验证 rerank | RAG 质量 | 低成本提升知识回答质量 |
| 4 | 统一 source_path 可点击引用 | RAG 产品体验 | 为图片引用打基础 |
| 5 | Qwen-VL client + curl smoke test | 多模态基础设施 | 先确认 key、模型名、图片输入格式和超时策略 |
| 6 | 图片摘要入库 + 图片引用回答 | 多模态 MVP | 风险最低，复用现有 RAG |
| 7 | 用户上传图片问答 | 多模态交互 | 需要新增 API、存储、Qwen-VL client |
| 8 | 专科图像 agent + HITL | 高风险扩展 | 等前面能力稳定后再做 |

---

## 六、建议的第一版多模态数据结构

图片 metadata 可以先保持简单：

```json
{
  "asset_type": "image",
  "image_id": "uuid",
  "image_path": "/static/rag_assets/images/uuid.png",
  "source": "document-name.md",
  "source_path": "/static/docs/document-name.md",
  "page": 3,
  "caption": "Qwen-VL 生成的图片摘要",
  "is_medical": true,
  "is_diagnostic": false
}
```

图片摘要 chunk 可以这样写入 RAG：

```markdown
图像摘要：brain_mri_figure_02
来源：xxx 文档，第 3 页
内容：该图展示了脑部 MRI T1/T2 加权图像中病灶区域的示意，用于说明影像表现与肿瘤位置关系。
注意：该图片摘要仅用于知识检索，不构成诊断结论。
```

这样现有 dense/BM25/RRF/rerank 链路都可以直接复用。

---

## 七、风险与边界

1. **医学安全风险**
   多模态图片很容易被用户理解为“AI 帮我看片”。第一阶段必须定位为图片内容解释和知识检索辅助，不输出诊断结论。

2. **Qwen-VL 幻觉风险**
   Qwen-VL 可能看错图、编造图中不存在的信息。图片摘要 prompt 需要要求“只描述可见内容，不确定就说明不确定”。

3. **引用链路风险**
   如果图片路径、source_path 不稳定，回答引用会失去可追溯性。应先统一资源路径，再做图片引用。

4. **测试风险**
   多模态引入后，普通文本问题不应被误路由到 image agent；无图片请求也不应触发 Qwen-VL。

---

## 八、阶段性验收清单

### 文本主路径稳定化

- [ ] `AGENT_MODE=multi` 下可正常聊天
- [ ] 医学知识问题能调用 `medical_kb`
- [ ] KB 低置信能 handoff 到 `web_search`
- [ ] Guardrails 能拦截非医疗/危险/prompt injection 请求
- [ ] RAG rerank 可配置启用并可降级

### 多模态 MVP

- [ ] ingestion 能保存图片 asset
- [ ] `DASHSCOPE_API_KEY` 或 `MULTIMODAL_API_KEY` 可通过 curl 调通 `qwen3.6-plus`
- [ ] Qwen-VL client 能生成非诊断性图片摘要
- [ ] 图片摘要能进入 RAG index
- [ ] 命中图片摘要时回答能展示参考图片
- [ ] 输出包含医学免责声明
- [ ] 无图片问题不影响原有文本问答路径

---

## 九、结论

当前可以开始做多模态，且多模态 LLM 已明确采用 Qwen-VL。仍不建议直接做高风险影像诊断，最合适的第一步是：

> 先稳定 multi-agent + RAG 主路径，再接入 Qwen-VL client，并实现“图片摘要入库 + 图片引用回答”的多模态 RAG MVP。

这条路线能最大化复用现有系统，避免过早引入重依赖和高风险诊断模型，也为后续用户上传图片问答、专科图像 agent、human-in-the-loop validation 留出清晰演进路径。
