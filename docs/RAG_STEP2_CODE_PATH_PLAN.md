# Step 2 技术代码路径方案（data_loader + chunking）

目标：基于已完成的 Step 1（`settings/types/config`），实现可复用的数据加载与 Parent/Child 分块能力，为 Step 3 的索引和检索提供标准输入。

对应主文档：`docs/RAG_IMPLEMENTATION.md` 中 `24.16` 的 Step 2。

---

## 1. 本阶段范围

只实现这两个文件：

- `src/rag/data_loader.py`
- `src/rag/chunking.py`

不在本阶段实现：

- 向量化、FAISS、BM25、RRF
- 工具接入与 Agent 调用
- Web/CLI 启动接入

---

## 2. 代码路径（调用关系）

Step 2 完成后的最小调用链：

1. `settings = load_settings()`
2. `cfg = build_rag_config(settings)` -> `sanitize` -> `validate`
3. `loader = MarkdownDataLoader()`
4. `parents = loader.load_documents(cfg.source_dirs)`
5. `chunker = ParentChildChunker(cfg.chunk_size, cfg.chunk_overlap)`
6. `children, parent_map, child_parent = chunker.build_parent_child(parents)`

输出供 Step 3 使用：

- `children`: 子文档列表（检索输入）
- `parent_map`: `parent_id -> parent Document`
- `child_parent`: `child_id -> parent_id`

---

## 3. 数据结构与元数据规范

## 3.1 Parent 文档 metadata（最低要求）

- `source`: 原始文件绝对路径
- `title`: 文档标题（优先首个 `#`）
- `category`: 由路径推断（如 `chapter8`、`HowToCook` 子目录）
- `parent_id`: 唯一 ID（推荐 UUID）
- `doc_type`: 固定 `parent`

## 3.2 Child 文档 metadata（最低要求）

- 继承 Parent 关键 metadata
- `child_id`: 唯一 ID
- `parent_id`: 对应父文档 ID
- `chunk_index`: 在父文档内的序号
- `doc_type`: 固定 `child`
- `chunk_size`: 当前 chunk 字符长度

---

## 4. 文件级实现方案

## 4.1 `src/rag/data_loader.py`

### 4.1.1 建议类与函数

- `class MarkdownDataLoader:`
  - `scan_markdown_files(source_dirs: list[str]) -> list[Path]`
  - `load_documents(source_dirs: list[str]) -> list[Document]`
  - `_read_text(path: Path) -> str`
  - `_extract_title(text: str, fallback: str) -> str`
  - `_infer_category(path: Path) -> str`
  - `_build_parent_metadata(path: Path, text: str) -> dict`

### 4.1.2 最小实现顺序

1. `scan_markdown_files`: 递归找 `*.md`，去重排序。
2. `_read_text`: `utf-8` 读取，异常时 warning + 跳过。
3. `_extract_title`: 正则取首个 Markdown 一级标题，否则用文件名。
4. `_build_parent_metadata`: 产出标准 metadata。
5. `load_documents`: 组合以上逻辑，返回 `Document` 列表。

### 4.1.3 错误处理策略

- 单文件读取失败：跳过，不中断全量加载。
- 全部失败：返回空列表，由上层决定是否抛错。

---

## 4.2 `src/rag/chunking.py`

### 4.2.1 建议类与函数

- `class ParentChildChunker:`
  - `__init__(chunk_size: int, chunk_overlap: int) -> None`
  - `build_parent_child(parents: list[Document]) -> tuple[list[Document], dict[str, Document], dict[str, str]]`
  - `_split_by_markdown_headers(text: str) -> list[str]`
  - `_split_long_text(text: str) -> list[str]`
  - `_make_child_doc(parent: Document, chunk_text: str, idx: int) -> Document`

### 4.2.2 最小分块策略（建议）

1. 先按 Markdown 标题切段（`#`, `##`, `###`）。
2. 每段若超过 `chunk_size`，再做长度分块（带 `chunk_overlap`）。
3. 段内空白块过滤。
4. 为每个 chunk 补齐 child metadata。

### 4.2.3 三个输出的构建方式

- `children`: append 每个 child 文档
- `parent_map`: `parent_id` 映射回父文档（用于后续回填）
- `child_parent`: `child_id -> parent_id`

---

## 5. 函数签名建议（可直接照着写）

```python
# src/rag/data_loader.py
from pathlib import Path
from langchain_core.documents import Document

class MarkdownDataLoader:
    def scan_markdown_files(self, source_dirs: list[str]) -> list[Path]: ...
    def load_documents(self, source_dirs: list[str]) -> list[Document]: ...
```

```python
# src/rag/chunking.py
from langchain_core.documents import Document

class ParentChildChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None: ...
    def build_parent_child(
        self, parents: list[Document]
    ) -> tuple[list[Document], dict[str, Document], dict[str, str]]: ...
```

---

## 6. 最小可运行自测（Step 2）

## 6.1 临时脚本位置

建议放：`src/tests/tmp_step2_smoke.py`

## 6.2 脚本逻辑

1. 读取 settings -> rag config。
2. 调 `MarkdownDataLoader.load_documents(cfg.source_dirs)`。
3. 调 `ParentChildChunker(...).build_parent_child(parents)`。
4. 打印统计：
   - parent 数量
   - child 数量
   - 前 3 条 source
   - 任意 1 条 child 的 metadata

## 6.3 运行命令（推荐）

```bash
PYTHONPATH=src python src/tests/tmp_step2_smoke.py
```

---

## 7. 验收标准（DoD）

- 能从 `cfg.source_dirs` 成功加载 Markdown 父文档。
- `parent.metadata` 字段齐全（`source/title/category/parent_id/doc_type`）。
- 能生成子文档，且 `chunk_overlap < chunk_size` 生效。
- `children`、`parent_map`、`child_parent` 三者数量与映射关系一致。
- 异常文件不会导致流程中断（有 warning，最终仍能返回可用结果）。

---

## 8. 风险与规避

- **路径基准错误**：统一用 Step 1 规范化后的绝对路径。
- **超大文档切分慢**：Step 2 先实现简单切分，后续再优化。
- **标题缺失**：标题提取失败时回退文件名。
- **空 chunk**：切分后统一 `strip` 过滤空块。

---

## 9. 与 Step 3 的接口约定

Step 3 依赖 Step 2 输出，不要变更这三个对象的语义：

- `children`：作为向量化与 BM25 的输入
- `parent_map`：检索命中 child 后回填 parent
- `child_parent`：可用于调试/解释与召回统计

若后续需要扩展字段，优先在 metadata 中追加，避免改动现有键名。

