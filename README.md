# LangChain Agent 基础项目

这是一个按标准分层组织的 LangChain Agent 示例项目，支持：

- 基本多轮对话
- 简单工具调用（当前时间、数学表达式计算）

## 目录结构

```text
wx-langchain/
├─ requirements.txt
├─ .env.example
├─ README.md
└─ src/
   ├─ main.py
   ├─ agents/
   │  └─ dialog_agent.py
   ├─ actions/
   │  └─ basic_tools.py
   ├─ llms/
   │  └─ openai_chat.py
   ├─ memory/
   │  └─ session_memory.py
   └─ config/
      └─ settings.py
```

## 1) 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
# OPENAI_BASE_URL=https://api.openai.com/v1
```

可选：

- `AGENT_VERBOSE=true` 开启 LangChain 执行日志
- `LOG_LEVEL=INFO` 控制应用日志级别（如 `DEBUG/INFO/WARNING`）

## 3) 运行项目

```bash
python src/main.py
```

## 4) 启动 Web 测试界面

```bash
uvicorn web.app:app --app-dir src --reload
```

打开浏览器访问：

- `http://127.0.0.1:8000`

## 5) 示例提问

- `你好，介绍一下你自己`
- `现在几点了？`
- `请帮我计算 (12 + 8) * 3 / 2`
- `我上一个问题问了什么？`

## 实现说明

- `config/settings.py`：加载 `.env` 配置
- `llms/openai_chat.py`：模型初始化
- `actions/basic_tools.py`：对话可调用工具
- `agents/dialog_agent.py`：组装 Agent
- `memory/session_memory.py`：会话记忆管理
- `main.py`：命令行对话入口
- `web/app.py` + `web/static/index.html`：Web 对话测试页
