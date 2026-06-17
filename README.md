# 医学多 Agent 助手（Biomedical Multi-Agent Assistant）

基于 `LangGraph`、`LangChain/LangChain Core` 与 `FastAPI` 的医学多智能体助手。项目当前使用统一的 `langgraph-swarm` 运行路径：输入先经过图像/文本护栏与分诊，再由领域 Agent 或 Swarm 并行协作生成回答，最后经过输出 guardrails 与可选 Harness 约束复核。

> 免责声明：本系统输出仅供学习、信息检索与健康知识参考，不构成医疗诊断或治疗建议。任何临床决策请遵循执业医师意见。

## 当前能力

- 统一 LangGraph Swarm 编排：`triage -> single/image/swarm -> hitl -> output_guardrail`。
- 领域 Agent：医学咨询、诊断辅助、医学研究，基于 `create_react_agent` 自主调用 Skill。
- 文档驱动 Skill：`src/skills/defs/*/SKILL.md` + `script/*.py` 自动加载。
- RAG：FAISS 默认向量库，支持 BM25、RRF、Query Planner、rerank、低置信拦截；可选切换 Qdrant。
- 多模态：支持最多多张图片上传，使用 VLM 生成图像摘要与医学相关性提示。
- HITL：可选 LangGraph interrupt 暂停/恢复，用于人工复核诊断类输出。
- Guardrails/Harness：输入、图像、输出安全复核；可选确定性约束与自动修复。
- Web 前端：Vue 3 + Vite，支持会话列表、SSE 流式、多图预览、HITL 卡片、STT/TTS。
- 服务分层：`controllers`、`services`、`graphs`、`rag`、`skills`、`db` 分层维护。

## 技术栈

- Python 3.10+
- FastAPI + uvicorn
- LangGraph + LangChain Core + LangChain OpenAI
- Vue 3 + Vite + TypeScript + Pinia
- FAISS / Qdrant（可选）
- PostgreSQL checkpointer / in-memory checkpointer
- ElevenLabs STT/TTS（可选）
- Mem0 长期记忆（可选）

## 目录结构

```text
cook-proj/
├─ frontend/                 # Vue 3 前端源码
├─ src/
│  ├─ actions/               # RAG/Web Search 等后端工具适配
│  ├─ agents/                # Agent registry、领域 Agent、guardrails
│  ├─ config/                # Settings 与日志
│  ├─ controllers/           # FastAPI API 路由
│  ├─ db/                    # 会话与消息持久化
│  ├─ graphs/                # LangGraph swarm 图与公共节点
│  ├─ llms/                  # LLM/VLM 客户端构造
│  ├─ memory/                # Mem0 等长期记忆封装
│  ├─ rag/                   # RAG ingestion/retrieval/generation
│  ├─ services/              # Chat/Speech 服务
│  ├─ skills/                # 文档驱动 Skill loader 与 defs
│  ├─ tests/                 # pytest 测试
│  ├─ main.py                # CLI 入口
│  └─ web/
│     ├─ app.py              # FastAPI app 初始化
│     └─ static/             # Vue 构建产物
├─ source_dir/file_data/     # 当前 RAG Markdown 语料目录
├─ data/                     # RAG index、上传图片等运行数据
├─ docs/                     # 历史方案与融合方案
├─ .env.example              # 脱敏配置模板
├─ requirements.txt
└─ README.md
```

## 环境准备

### 1. Python 依赖

```bash
cd /Users/aimiaomiao/cook-proj
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 前端依赖

只有需要修改 Vue 源码或重新构建前端时才需要执行：

```bash
npm install --prefix frontend
npm run build --prefix frontend
```

构建产物会输出到 `src/web/static/`，由 FastAPI 直接托管。

### 3. 配置环境变量

复制模板并填写真实密钥：

```bash
cp .env.example .env
```

至少需要配置：

```env
OPENAI_API_KEY=your_openai_or_dashscope_key
OPENAI_MODEL=qwen-plus
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

RAG_ENABLED=true
RAG_SOURCE_DIRS=source_dir/file_data
RAG_INDEX_DIR=data/medical_rag_index
RAG_EMBEDDING_API_KEY=your_embedding_key
```

`.env` 已加入 `.gitignore`，不要提交真实密钥。`.env.example` 可以提交。

## 启动方式

### Web

```bash
source .venv/bin/activate
uvicorn web.app:app --app-dir src --reload
```

访问：

```text
http://127.0.0.1:8000
```

### Vue 前端开发模式

后端仍需运行在 `127.0.0.1:8000`，前端开发服务器会代理 `/api` 与 `/health`：

```bash
npm run dev --prefix frontend
```

通常访问：

```text
http://127.0.0.1:5173
```

### CLI

```bash
source .venv/bin/activate
python src/main.py
```

## Web API

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | Vue 前端页面 |
| `GET` | `/health` | 健康检查，返回 runtime 与 RAG 状态 |
| `POST` | `/api/conversations` | 新建会话 |
| `GET` | `/api/conversations` | 获取会话列表 |
| `GET` | `/api/conversations/{session_id}/messages` | 获取会话消息 |
| `POST` | `/api/chat` | 同步文本问答 |
| `POST` | `/api/chat/stream` | 文本 SSE 流式问答 |
| `POST` | `/api/chat/multimodal/stream` | 图片 + 文本多模态 SSE |
| `POST` | `/api/chat/hitl/resume` | HITL 人工复核后恢复图执行 |
| `POST` | `/api/speech/stt` | 语音转文本 |
| `POST` | `/api/speech/tts` | 文本转语音 |
| `POST` | `/api/reset` | 创建新会话 ID |

SSE 事件：

- `chunk`：增量文本，形如 `{"text": "...", "replace": false}`。
- `done`：最终答案，形如 `{"answer": "..."}`。
- `error`：错误信息，形如 `{"detail": "..."}`。
- `hitl`：人工复核中断，形如 `{"interrupt": {...}}`。

## RAG 说明

当前默认语料目录为：

```env
RAG_SOURCE_DIRS=source_dir/file_data
```

RAG 初始化由 `src/rag/core/bootstrap.py` 在 Web 启动阶段触发。若配置不完整或索引构建失败，`/health` 会返回 `rag_ready=false` 与失败原因；聊天工具会降级处理。

向量后端默认是 FAISS：

```env
VECTOR_DB_PROVIDER=faiss
```

如需使用 Qdrant，配置：

```env
VECTOR_DB_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=rag_documents
```

## 可选功能开关

默认均可关闭，避免缺外部服务时影响主流程。

```env
MULTIMODAL_ENABLED=true
HITL_ENABLED=false
ELEVENLABS_ENABLED=false
MEM0_ENABLED=false
HARNESS_ENABLED=false
TAVILY_ENABLED=false
```

- `MULTIMODAL_ENABLED`：图片上传与 VLM 摘要。
- `HITL_ENABLED`：LangGraph interrupt 人工复核。
- `ELEVENLABS_ENABLED`：STT/TTS 语音能力。
- `MEM0_ENABLED`：长期记忆。
- `HARNESS_ENABLED`：确定性安全约束与自动修复。
- `TAVILY_ENABLED`：Web Search。

## 开发与验证

Python 语法检查示例：

```bash
python3 -m py_compile src/web/app.py src/services/chat_service.py src/graphs/swarm_graph.py
```

前端构建：

```bash
npm run build --prefix frontend
```

测试（依赖安装完整后）：

```bash
pytest -q src/tests
```

## 常见问题

- `ModuleNotFoundError`：确认已激活 `.venv` 并执行 `pip install -r requirements.txt`。
- 启动时报 `Missing OPENAI_API_KEY`：检查 `.env` 是否存在且已填写真实 key。
- Web 页面静态资源 404：执行 `npm run build --prefix frontend`，确认 `src/web/static/frontend/assets/` 存在。
- RAG 未就绪：访问 `/health` 查看 `rag_reason`，优先检查 `RAG_SOURCE_DIRS`、embedding key 与索引目录。
- 语音按钮不可用：确认 `ELEVENLABS_ENABLED=true` 且 `ELEVENLABS_API_KEY` 已配置。
- HITL 不触发：确认 `HITL_ENABLED=true`，且当前回答路径设置了 `needs_human_validation`。
