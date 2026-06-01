# medix-agent-swarm 技术架构与主要流程

## 1. 项目定位

`medix-agent-swarm` 是一个医疗问答多智能体系统，核心是 **Skills-Agent 两层架构**：

- **Skill 层**：原子能力（知识检索、风险评估、症状分析、指南检索、深度研究等）
- **Agent 层**：通过 LLM + Agent Loop 自主选择并调用 Skills 完成任务
- **Swarm 层**：复杂问题下进行多 Agent 协作（任务分解、并行执行、结果汇总）

入口在 `medix-agent-swarm/main.py`，运行后进入交互式会话，调用 `process_with_swarm()` 处理用户问题。

---

## 2. 总体架构分层

### 2.1 分层视图

1. **交互入口层**
   - `main.py`
   - 提供 CLI 会话、打印模式（单 Agent / Swarm）与耗时

2. **编排与路由层**
   - `swarm/swarm_coordinator.py`：统一入口与路由
   - `swarm/lead_agent.py`：任务分解与结果汇总
   - `swarm/shared_context.py`：共享上下文（黑板/信息素模型）
   - `swarm/events_20260428_231035.py`：事件模型定义（代码中以 `swarm.events` 方式引用）

3. **智能体执行层**
   - `agents/base_agent.py`：Agent 抽象基类
   - `agents/consultation_agent.py`
   - `agents/diagnostic_agent.py`
   - `agents/research_agent.py`
   - `core/agent_loop.py`：Think-Act-Observe 循环

4. **Skills 与工具层**
   - `core/skill_loader.py`：自动发现 `.claude/skills/*`
   - `core/skill_registry.py`：注册、执行、转 OpenAI Function Calling 格式
   - `agents/skill_registry_mixin.py`：统一给 Worker Agent 注册全部 Skills

5. **记忆与约束层**
   - `memory/short_term.py`：短期记忆（会话）
   - `memory/long_term.py`：长期记忆（Mem0）
   - `constraints/validator.py`：约束校验
   - `validation/auto_fixer_20260428_231043.py`：输出自动修复

6. **知识与研究层**
   - `knowledge/milvus_kb.py`：Milvus Lite + Embedding 检索
   - `research/deep_research_workflow.py`：Web + KB 并行深度研究流程

---

## 3. 核心模块职责

## 3.1 SwarmCoordinator（系统总入口）

`swarm/swarm_coordinator.py` 负责：

- 检索并注入记忆上下文（短期 recent history + 长期 similar sessions）
- 调用 LeadAgent 进行任务分解
- 路由：
  - 1 个子任务 -> 单 Agent 执行
  - >=2 子任务 -> Swarm 并行执行
  - 异常/无任务 -> 回退 ConsultationAgent
- 统一保存长期记忆（Mem0）

它是“路由器 + 生命周期管理器”，不是细粒度编排器。

## 3.2 LeadAgent（分解与汇总）

`swarm/lead_agent.py` 负责两件事：

- **前置分解**：将用户问题转为 `subtasks`（每个子任务指定 `assigned_agent`）
- **后置汇总**：读取各 Agent 的贡献，生成统一最终答复

即：**前分解 + 后综合**，中间执行交给 Worker 并行处理。

## 3.3 SharedContext（协作黑板）

`swarm/shared_context.py` 提供协作共享状态：

- 子任务状态：`pending -> in_progress -> completed/failed`
- Agent 贡献：按 `agent_id` 聚合结果
- 事件流：任务分解、子任务开始/完成、上下文变更、Swarm 启停

这部分实现了类似“信息素”机制的间接通信。

## 3.4 AgentLoop（LLM 驱动执行内核）

`core/agent_loop.py` 是每个 Worker Agent 的执行循环：

1. 初始化消息（系统提示词 + 历史对话 + 当前问题）
2. 调用 `llm_client.chat_with_tools(...)`
3. 如模型返回 `tool_calls`，执行 Skill 并回写 tool message
4. 如返回最终文本，则结束
5. 达到迭代/工具调用上限后强制收束总结

同时内置：

- 短期记忆写入（user/assistant/tool）
- 约束校验与自动修复（Harness）

## 3.5 SkillRegistry 与动态发现

- `core/skill_loader.py`：扫描 `.claude/skills`，解析 `SKILL.md` + script 函数
- `agents/skill_registry_mixin.py`：根据函数签名自动推断参数并注册
- `core/skill_registry.py`：转换为 OpenAI Function Calling schema，供 LLM 直接调用

这让 Skill 层具备“热插拔/自动发现”特性。

---

## 4. 主要流程

## 4.1 主入口流程（CLI）

`main.py` 交互循环：

1. 用户输入问题
2. 调用 `process_with_swarm(question, session_id)`
3. 输出回答、建议、免责声明、耗时与模式信息

## 4.2 单 Agent 流程（简单问题）

触发条件：LeadAgent 分解后只有 1 个子任务。

流程：

1. `SwarmCoordinator.process()` 检索并注入记忆上下文
2. 选择目标 Agent（consultation/diagnostic/research）
3. Agent 运行 `AgentLoop`
4. LLM 按需调用 1~N 个 Skills
5. 结果后处理（提取建议/免责声明）
6. 保存长期记忆并返回

## 4.3 Swarm 流程（复杂问题）

触发条件：子任务数量 >=2 且启用 swarm。

流程：

1. 创建 `SharedContext`
2. LeadAgent 创建子任务并写入共享上下文
3. Worker 池并行执行各自子任务（`asyncio.gather`）
4. Worker 完成后写入 contribution
5. LeadAgent 汇总贡献生成最终答案
6. 生成会话摘要并写入长期记忆
7. 返回协作元数据（参与 Agent、完成任务数、耗时、超时状态）

## 4.4 Skill 调用流程（通用）

1. Agent 提供可用 tools（OpenAI 格式）
2. 模型决定是否调用 function
3. `execute_tool()` -> `SkillRegistry.execute()`
4. Skill 运行（同步/异步均支持）
5. 结果以 tool message 回注 LLM 上下文
6. 模型基于检索结果输出最终回答

## 4.5 记忆读写流程

**读取（请求开始）**

- 短期：`ShortTermMemory.get_recent_messages(session_id, limit=10)`
- 长期：`LongTermMemory.search_similar_sessions(query, limit=3)`

**写入（执行中 + 结束后）**

- 执行中：AgentLoop 持续写入 user/assistant/tool 消息到短期记忆
- 结束后：SwarmCoordinator 写入 session summary 到 Mem0 长期记忆

## 4.6 DeepResearch 流程（复杂研究型 Skill）

`deep_research` Skill 会调用 `research/deep_research_workflow.py`：

1. LLM 规划 2~3 个子查询
2. 并行执行 Web 搜索 + Milvus 检索
3. EvidenceSynthesizer 综合证据
4. 输出 `ResearchReport`（关键发现、证据等级、置信度、建议）

---

## 5. 数据与依赖栈

## 5.1 关键依赖（见 `medix-agent-swarm/requirements.txt`）

- LLM/API：`openai`
- Embedding/RAG：`sentence-transformers`, `transformers`, `torch`
- 向量库：`pymilvus`（Milvus Lite）
- 记忆：`mem0ai`, `redis`
- 搜索抓取：`duckduckgo-search`, `requests`, `beautifulsoup4`
- 工程能力：`loguru`, `pyyaml`, `aiohttp`, `httpx`

## 5.2 关键数据对象

- `SubTask`：子任务单元（目标 Agent、状态、结果）
- `Contribution`：Worker 对子任务的贡献记录
- `ConversationHistory`：短期会话消息集合
- `ResearchReport`：深度研究结果对象

---

## 6. 架构特点与扩展点

## 6.1 架构特点

- **两层解耦**：Agent 不绑定具体 Skill 实现，只依赖注册表
- **路由前置**：先分解再执行，简单问题避免过度协作
- **并发友好**：Swarm 下 Worker 并行执行，提升复杂问题吞吐
- **记忆增强**：短期上下文 + 长期相似案例，支持多轮连续问答
- **安全约束**：输出约束校验与自动修复，降低医疗建议风险

## 6.2 可扩展方向

- 新增 Skill：在 `.claude/skills/<name>/script/` 增加函数并补充 `SKILL.md`
- 新增 Agent：继承 `BaseAgent`，复用 `AgentLoop` 与 SkillRegistry
- 新增路由策略：增强 LeadAgent 的分解规则与 Coordinator 路由决策
- 新增知识源：扩展 `DeepResearchWorkflow` 的检索后端

---

## 7. 一句话总结

`medix-agent-swarm` 通过 **“LLM 自主 Skill 调用 + 复杂任务多 Agent 协作 + 记忆与约束护栏”** 形成了一个可扩展的医疗智能体系统：简单问题快答，复杂问题分解并行，最终由 Lead 统一综合输出。
