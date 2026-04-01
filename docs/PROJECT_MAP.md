# Lagent Project Map

## 项目定位

`lagent` 是一个用于构建智能体（Agent）的 Python 框架。  
核心思路是分层解耦：`Agent` 负责流程，`LLM` 负责生成，`Action` 负责工具调用，`Memory/Prompt/Hook` 负责上下文与扩展，`distributed` 负责服务化部署。

## 仓库顶层结构

- `lagent/`：核心库代码（主框架）
- `examples/`：示例脚本（同步、异步、多后端、服务化）
- `tests/`：测试代码（主要覆盖 actions，含部分 agent 测试）
- `docs/`：中英文文档源文件
- `cases/`：轻量实验脚本（如 `case-1.py`）
- `README.md`：项目入口说明与快速使用
- `requirements/`：依赖拆分（runtime/optional/docs）

## 核心模块职责（lagent/）

### 1. `lagent/agents/`

- `agent.py`：基础 `Agent`、异步与流式变体，串联 memory、hook、LLM 调用
- `react.py`：`ReAct` 智能体，循环执行“思考 -> 选工具 -> 执行 -> 反馈”
- `stream.py`：流式相关 Agent 实现（如 InternLM 场景）
- `aggregator/`：将 memory 聚合成模型输入（如 OpenAI 风格 message 列表）

### 2. `lagent/actions/`

- `base_action.py`：Action 抽象与工具 API 装饰
- `action_executor.py`：统一工具执行入口（注册、派发、错误处理）
- 具体工具实现：Python/IPython 解释器、搜索、网页浏览等

### 3. `lagent/llms/`

- 统一 LLM 封装接口（`chat` 等）
- 多后端适配：OpenAI、vLLM、LMDeploy、HuggingFace、Anthropic

### 4. `lagent/prompts/`

- Prompt 模板与输出解析
- 包含 JSON、Tool、Plugin 等 parser

### 5. `lagent/memory/`

- 会话记忆抽象与管理（如按 `session_id` 管理多会话）

### 6. `lagent/hooks/`

- 生命周期钩子：`before/after agent`、`before/after action`
- 支持 Action 前后处理与日志扩展

### 7. `lagent/distributed/`

- HTTP 服务化能力（server/client）
- Ray 运行包装（分布式部署）

### 8. 其他

- `lagent/schema.py`：跨模块共享的数据结构与状态码
- `lagent/utils/`：工具函数（对象创建等）
- `lagent/version.py`：版本信息

## Examples 使用导图

- 同步本地：`examples/run_agent_lmdeploy.py`
- CLI 演示：`examples/model_cli_demo.py`
- 异步链路：`examples/run_async_agent_*.py`
- HTTP 服务：`examples/run_agent_services.py`
- Ray 部署：`examples/run_ray_async_agent_lmdeploy.py`

## 简化执行链路

### 普通 Agent

1. 读取/更新 memory
2. aggregator 组装模型输入
3. 调用 `llm.chat`
4. parser 解析输出
5. 回写 memory 并返回结果

### ReAct Agent

1. LLM 输出下一步动作（工具名 + 参数）
2. `ActionExecutor` 执行对应 Action
3. 将执行结果回注入上下文
4. 循环至结束条件或最大轮次

## 建议阅读顺序

1. `README.md`
2. `lagent/schema.py`
3. `lagent/agents/agent.py`
4. `lagent/agents/aggregator/default_aggregator.py`
5. `lagent/agents/react.py`
6. `lagent/actions/action_executor.py`
7. 选择一个 `examples/` 脚本跑通

## 维护建议

- 新增能力优先按分层放置：LLM 放 `llms/`，工具放 `actions/`，流程放 `agents/`
- 新增示例脚本时，命名体现后端和执行模式（sync/async/service）
- 为新增 Action 与 Agent 能力补充对应 `tests/` 用例

