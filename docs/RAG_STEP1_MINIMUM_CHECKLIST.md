# Step 1 最小可运行实现清单

目标：在不接入检索和工具层的前提下，先完成配置与类型基础设施，确保后续 Step 2-7 可以无缝衔接。

范围（仅 Step 1）：

- `src/config/settings.py`
- `src/rag/types.py`
- `src/rag/config.py`

非目标（本阶段不做）：

- 文档加载/分块/索引/检索
- Agent 工具注入
- CLI/Web 启动接入

---

## 1. 完成定义（DoD）

- `load_settings()` 能成功解析 RAG 相关环境变量并返回 `Settings`。
- `build_rag_config(settings)` 能返回合法 `RAGConfig`。
- `validate_rag_config(config)` 能对明显错误配置抛出可解释异常。
- `sanitize_rag_config(config)` 能修正可自动纠正的问题（如 overlap >= size）。
- 本地最小自测脚本通过（见文末“自测步骤”）。

---

## 2. 文件级实现顺序

建议顺序：

1. `src/rag/types.py`
2. `src/config/settings.py`
3. `src/rag/config.py`

原因：先定义结构，再接入配置解析，最后做映射与校验。

---

## 3. 函数级 TODO（精确到先写什么、先返回什么）

## 3.1 `src/rag/types.py`

### A. 定义 `RAGConfig`（先做）

先写字段（全部给默认值，便于最小运行）：

- `enabled: bool = False`
- `source_dirs: list[str] = field(default_factory=list)`
- `index_dir: str = "data/rag_index"`
- `top_k: int = 4`
- `retrieval_k: int = 12`
- `embedding_provider: str = "dashscope"`
- `embedding_api_key: str | None = None`
- `embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"`
- `embedding_model: str = "text-embedding-v4"`
- `embedding_dimensions: int = 1024`
- `chunk_size: int = 800`
- `chunk_overlap: int = 120`
- `rrf_k: int = 60`
- `rebuild: bool = False`

先返回什么：

- 无函数返回；`RAGConfig()` 可直接实例化成功即可。

### B. 定义 `RetrievalResult`（最小字段）

先写字段：

- `query: str`
- `parents: list[Any] = field(default_factory=list)`
- `sources: list[str] = field(default_factory=list)`
- `debug: dict[str, Any] = field(default_factory=dict)`

先返回什么：

- `RetrievalResult(query="x")` 能构造成功。

### C. 定义 `AnswerResult`（最小字段）

先写字段：

- `query: str`
- `route: str`
- `answer: str`
- `sources: list[str] = field(default_factory=list)`
- `debug: dict[str, Any] = field(default_factory=dict)`

先返回什么：

- `AnswerResult(query="q", route="general", answer="ok")` 能构造成功。

---

## 3.2 `src/config/settings.py`

### A. 扩展 `Settings` 字段（先做）

新增字段：

- `rag_enabled: bool = False`
- `rag_source_dirs: list[str] = field(default_factory=list)`
- `rag_index_dir: str = "data/rag_index"`
- `rag_top_k: int = 4`
- `rag_retrieval_k: int = 12`
- `rag_embedding_provider: str = "dashscope"`
- `rag_embedding_api_key: str | None = None`
- `rag_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"`
- `rag_embedding_model: str = "text-embedding-v4"`
- `rag_embedding_dimensions: int = 1024`
- `rag_chunk_size: int = 800`
- `rag_chunk_overlap: int = 120`
- `rag_rrf_k: int = 60`
- `rag_rebuild: bool = False`

先返回什么：

- 旧配置不变时，`load_settings()` 依然可运行。

### B. 新增解析函数（再做）

1) `_parse_bool(name: str, default: bool) -> bool`  
先写最小逻辑：

- 环境变量为空 -> 返回 `default`
- `1/true/yes/on` -> `True`
- `0/false/no/off` -> `False`
- 其他值 -> 返回 `default`（并记录 warning）

2) `_parse_int(name: str, default: int, minimum: int) -> int`  
先写最小逻辑：

- 为空 -> `default`
- 转换失败 -> `default`（warning）
- 小于 `minimum` -> `minimum`

3) `_parse_csv(name: str, default: str = "") -> list[str]`  
先写最小逻辑：

- 读取 `env` 或 `default`
- 按 `,` 切分
- `strip` + 去空串

4) `_normalize_paths(paths: list[str]) -> list[str]`  
先写最小逻辑：

- 用 `Path(p).expanduser()` + `resolve()` 转绝对路径
- 不做存在性过滤（Step 1 只规范化，不拦截）

5) `_resolve_rag_embedding_api_key() -> str | None`  
先写最小逻辑：

- 优先读 `RAG_EMBEDDING_API_KEY`
- 若为空，则回退读 `DASHSCOPE_API_KEY`
- 再为空返回 `None`（Step 1 不强制抛错）

### C. 在 `load_settings()` 接线（最后做）

先写法：

- 使用上述解析函数取值
- 构造并返回 `Settings(...)`

先返回什么：

- 当 `.env` 只含 OpenAI 字段时：RAG 字段走默认值
- 当补充 RAG 字段时：能正确生效

---

## 3.3 `src/rag/config.py`

### A. `build_rag_config(settings) -> RAGConfig`（先做）

先写最小映射（1:1 字段复制）：

- `enabled <- settings.rag_enabled`
- `source_dirs <- settings.rag_source_dirs`
- `embedding_provider <- settings.rag_embedding_provider`
- `embedding_api_key <- settings.rag_embedding_api_key`
- `embedding_base_url <- settings.rag_embedding_base_url`
- `embedding_model <- settings.rag_embedding_model`
- `embedding_dimensions <- settings.rag_embedding_dimensions`
- ...

先返回什么：

- 对任何合法 `Settings`，都返回 `RAGConfig`。

### B. `sanitize_rag_config(cfg) -> RAGConfig`（再做）

先写最小修正：

- `chunk_overlap >= chunk_size` 时：
  - `chunk_overlap = max(0, chunk_size // 5)`
- `retrieval_k < top_k` 时：
  - `retrieval_k = top_k`
- `top_k <= 0` 时：
  - `top_k = 1`

先返回什么：

- 返回“已修正”的新配置对象或原对象（保持一致风格即可）。

### C. `validate_rag_config(cfg) -> None`（最后做）

先做硬校验（失败抛 `ValueError`）：

- `index_dir` 不能为空
- `embedding_provider` 不能为空
- `embedding_model` 不能为空
- `embedding_base_url` 不能为空
- 当 `enabled=true` 时，`embedding_api_key` 不能为空
- `embedding_dimensions >= 128`（建议阈值，默认 1024）
- `top_k >= 1`
- `retrieval_k >= top_k`
- `chunk_size >= 100`（建议阈值）
- `chunk_overlap >= 0`

先返回什么：

- 合法配置：返回 `None`
- 非法配置：抛出带明确字段名的信息

---

## 4. 最小自测步骤（本地手工）

说明：本项目的 `load_settings()` 使用 `python-dotenv` 加载环境变量，因此 **RAG 参数可以直接写在项目根目录 `.env`**。推荐优先使用 `.env` 管理，终端 `export` 仅用于临时覆盖。

## 4.1 快速脚本自测

创建临时脚本（例如 `tmp_step1_smoke.py`）并执行：

```python
from config.settings import load_settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config

settings = load_settings()
cfg = build_rag_config(settings)
cfg = sanitize_rag_config(cfg)
validate_rag_config(cfg)
print("STEP1_SMOKE_OK", cfg)
```

运行：

```bash
python -m src.main  # 可先验证旧链路没坏
python src/tmp_step1_smoke.py
```

预期：

- 输出 `STEP1_SMOKE_OK`
- 无异常抛出

## 4.2 `.env` 配置测试（推荐）

在项目根目录 `.env` 中增加（示例）：

```env
RAG_ENABLED=true
RAG_SOURCE_DIRS=chapter8,source_dir/HowToCook
RAG_INDEX_DIR=data/rag_index
RAG_TOP_K=3
RAG_RETRIEVAL_K=8

RAG_EMBEDDING_PROVIDER=dashscope
RAG_EMBEDDING_API_KEY=your_dashscope_key
RAG_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_EMBEDDING_DIMENSIONS=1024

RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=80
RAG_RRF_K=60
RAG_REBUILD=false
```

保存后重新运行 smoke，预期配置值正确映射。

## 4.3 终端临时覆盖（可选）

如果你不想改 `.env`，可以临时用 `export` 覆盖：

```bash
export RAG_ENABLED=true
export RAG_SOURCE_DIRS=chapter8,source_dir/HowToCook
export RAG_TOP_K=3
export RAG_RETRIEVAL_K=8
export RAG_EMBEDDING_PROVIDER=dashscope
export RAG_EMBEDDING_API_KEY=your_dashscope_key
export RAG_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export RAG_EMBEDDING_MODEL=text-embedding-v4
export RAG_EMBEDDING_DIMENSIONS=1024
export RAG_CHUNK_SIZE=500
export RAG_CHUNK_OVERLAP=80
```

## 4.4 异常值测试

设置异常值：

```bash
export RAG_TOP_K=-1
export RAG_RETRIEVAL_K=0
export RAG_CHUNK_OVERLAP=9999
```

预期：

- `sanitize_rag_config` 自动修正可修正值
- `validate_rag_config` 对不可接受值抛可解释异常

---

## 5. 交付检查清单（提交前）

- [ ] `Settings` 新字段存在并有默认值
- [ ] 4 个解析辅助函数可用
- [ ] `RAGConfig` / `RetrievalResult` / `AnswerResult` 已定义
- [ ] `build/sanitize/validate` 三函数已实现
- [ ] 手工 smoke 通过
- [ ] 旧 CLI 主流程未受影响

---

## 6. 常见坑位提醒

- 不要在 Step 1 引入检索库初始化（会增加调试噪音）。
- `rag_source_dirs` 先只做路径规范化，不要在 Step 1 强校验路径存在。
- 不要直接改动现有聊天链路的 `OPENAI_BASE_URL` 来复用 embedding；建议使用 `RAG_EMBEDDING_*` 独立配置，避免影响聊天模型。
- `sanitize` 和 `validate` 职责分离：
  - `sanitize` 修正可修正项
  - `validate` 拦截不可修复项

