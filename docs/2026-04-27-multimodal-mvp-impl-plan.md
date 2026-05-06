# 2026-04-27 多模态 MVP（阶段 1 + 阶段 2）代码实现方案

> 配套文档：`docs/2026-04-27-multimodal-next-step-plan.md` 第四章。
> 目标：在不破坏现有 multi-agent / RAG / guardrails 主路径的前提下，把 “图片摘要入库 + 图片引用回答” 和 “用户上传图片问答” 一次落地。
> VLM 固定为 DashScope Qwen-VL（`qwen3.6-plus`），通过 OpenAI-compatible `/chat/completions` 调用。

---

## 一、总体设计

```
┌────────────────────────────── 阶段 1：ingestion + 引用 ─────────────────────────────┐
│                                                                                    │
│   PDF/图文 MD                                                                       │
│      │                                                                              │
│      ▼                                                                              │
│   pdf_parser.parse_pdf()  ──► 抽取图片到 data/rag_assets/images/{stem}/...           │
│      │                                                                              │
│      ▼                                                                              │
│   QwenVLClient.summarize_image()  ──► 非诊断性 caption                              │
│      │                                                                              │
│      ▼                                                                              │
│   pdf_to_markdown.render_markdown()  ──► 在每页 page 后追加 “### 图片：xxx” 块       │
│      │                                                                              │
│      ▼                                                                              │
│   ParentChildChunker  ──► 把 “### 图片：xxx” 块的 image metadata 写到 child.meta    │
│      │                                                                              │
│      ▼                                                                              │
│   HybridRetriever / Rerank  ──► 复用现有链路                                         │
│      │                                                                              │
│      ▼                                                                              │
│   GenerationRouter.build_answer()  ──► 在 “参考文档” 后追加 “参考图片”               │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────── 阶段 2：用户上传图片 ────────────────────────────────┐
│                                                                                    │
│   POST /api/chat/multimodal/stream  (multipart: text + images[])                   │
│      │                                                                              │
│      ▼                                                                              │
│   chat_service.stream_multimodal()                                                  │
│      │   - 保存图片到 data/uploads/{session_id}/{uuid}.{ext}                        │
│      │   - 构造 HumanMessage(content=[text, image_url])                             │
│      ▼                                                                              │
│   multi_agent_graph                                                                 │
│      START                                                                           │
│       └─► image_input_guardrail  (mime/size/数量预检)                               │
│            └─► image_caption       (Qwen-VL 生成 caption，注入新 HumanMessage)      │
│                 └─► input_guardrail (文本 + caption 一起过 LLM 安全检查)             │
│                      └─► supervisor (路由到 medical_kb / web_search / conversation) │
│                           └─► specialist                                            │
│                                └─► output_guardrail                                 │
│                                     └─► END                                         │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

设计要点：

1. 阶段 1 通过 “PDF→Markdown 多了一段图片摘要文本” 的方式接入，对 ingestion / retrieval 改动极小，dense + BM25 + RRF + rerank 全部复用。
2. 阶段 2 通过新增 `image_caption` 节点把图片转成文本并注入消息，下游 supervisor / KB / output guardrail 不需要懂图片。
3. 图片相关 metadata 全程随 Document 走，最终由 `GenerationRouter` 输出 “参考图片：” 段落。
4. Qwen-VL 失败必须降级（跳过该图、记录错误），不阻塞文本主路径。

---

## 二、新增 / 修改文件清单

| 类型 | 路径 | 说明 |
|---|---|---|
| 新增 | `src/llms/qwen_vl.py` | DashScope Qwen-VL OpenAI-compatible client |
| 新增 | `src/agents/guardrails/image_guardrails.py` | 图片 mime / size / 数量 / 越权请求预检 |
| 新增 | `docs/2026-04-27-multimodal-mvp-impl-plan.md` | 本文档 |
| 修改 | `src/config/settings.py` | `multimodal_*` 配置字段 |
| 修改 | `.env`（用户侧） | 多模态相关变量 |
| 修改 | `src/rag/ingestion/pdf_parser.py` | 抽图到磁盘，`ParsedPdfDoc.images` 改成 `list[list[str]]` |
| 修改 | `src/rag/ingestion/pdf_to_markdown.py` | 在每个 page 后追加 “### 图片：xxx” 块 |
| 修改 | `src/rag/ingestion/chunking.py` | 解析 `### 图片：xxx`，把 image metadata 写进 child.meta |
| 修改 | `src/rag/generation/generation_router.py` | 输出 `参考图片：` 段落 |
| 修改 | `src/scripts/ingest_pdf.py` | 增加 `--vlm-enabled / --vlm-skip-existing` 开关，加载 settings 构造 client |
| 修改 | `src/web/app.py` | 静态目录 mount：`/static/rag_assets/images`、`/static/uploads` |
| 修改 | `src/controllers/chat_controller.py` | 新增 `POST /api/chat/multimodal/stream` |
| 修改 | `src/services/chat_service.py` | `stream_multimodal()` |
| 修改 | `src/graphs/multi_agent_graph.py` | 新增 `image_input_guardrail`、`image_caption` 节点 + state 字段 |
| 修改 | `src/web/static/index.html` | 上传按钮 + 多模态请求路径 |
| 修改 | `requirements.txt` | 显式声明 `httpx`、`python-multipart` |

---

## 三、配置层

### 3.1 `src/config/settings.py`

`Settings` 增加：

```python
# 多模态
multimodal_enabled: bool = False
multimodal_provider: str = "dashscope"
multimodal_model: str = "qwen3.6-plus"
multimodal_api_key: str | None = None
multimodal_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
multimodal_timeout_ms: int = 10000
multimodal_max_image_bytes: int = 8 * 1024 * 1024
multimodal_max_images_per_request: int = 3
multimodal_allowed_mime: list[str] = field(default_factory=lambda: [
    "image/jpeg", "image/png", "image/webp",
])
multimodal_assets_dir: str = "data/rag_assets/images"
multimodal_uploads_dir: str = "data/uploads"
multimodal_assets_url_prefix: str = "/static/rag_assets/images"
multimodal_uploads_url_prefix: str = "/static/uploads"
multimodal_min_image_bytes: int = 4096  # 跳过过小的图标
```

`load_settings()` 增加（关键的几行）：

```python
multimodal_enabled = _parse_bool("MULTIMODAL_ENABLED", False)
multimodal_model = os.getenv("MULTIMODAL_MODEL", "qwen3.6-plus").strip() or "qwen3.6-plus"
multimodal_api_key = (
    os.getenv("MULTIMODAL_API_KEY", "").strip()
    or os.getenv("DASHSCOPE_API_KEY", "").strip()
    or os.getenv("RAG_EMBEDDING_API_KEY", "").strip()  # 已知项目里这一项就是 DashScope key
    or None
)
multimodal_base_url = os.getenv(
    "MULTIMODAL_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
).strip()
multimodal_timeout_ms = _parse_int("MULTIMODAL_TIMEOUT_MS", 10000, 1000)
multimodal_max_image_bytes = _parse_int("MULTIMODAL_MAX_IMAGE_BYTES", 8 * 1024 * 1024, 1024)
multimodal_max_images_per_request = _parse_int("MULTIMODAL_MAX_IMAGES_PER_REQUEST", 3, 1)
multimodal_assets_dir = os.getenv("MULTIMODAL_ASSETS_DIR", "data/rag_assets/images").strip()
multimodal_uploads_dir = os.getenv("MULTIMODAL_UPLOADS_DIR", "data/uploads").strip()
```

### 3.2 `.env` 追加

```bash
# =========================
# Multimodal (Qwen-VL)
# =========================
MULTIMODAL_ENABLED=true
MULTIMODAL_MODEL=qwen3.6-plus
MULTIMODAL_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
MULTIMODAL_TIMEOUT_MS=10000
MULTIMODAL_MAX_IMAGE_BYTES=8388608
MULTIMODAL_MAX_IMAGES_PER_REQUEST=3
# MULTIMODAL_API_KEY=  # 留空时回退 DASHSCOPE_API_KEY / RAG_EMBEDDING_API_KEY
MULTIMODAL_ASSETS_DIR=data/rag_assets/images
MULTIMODAL_UPLOADS_DIR=data/uploads
```

### 3.3 `requirements.txt`

显式声明：

```
httpx>=0.27,<0.29
python-multipart>=0.0.9
```

---

## 四、Qwen-VL Client（`src/llms/qwen_vl.py`）

```python
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)


SUMMARY_SYSTEM_PROMPT = """\
你是一个医学知识库的图片摘要器。
任务：基于用户提供的图片，生成一段用于知识检索的非诊断性摘要。
要求：
1. 只描述图片中可见的内容，禁止推断不可见的信息。
2. 不输出诊断结论、不判断良恶性、不替代医生阅片。
3. 若是医学影像/示意图/图表，简述类型、关键标注、明显特征。
4. 若图片信息不足或与医学无关，明确说明。
5. caption 控制在 200 字以内。
返回 JSON：
{
  "caption": "<不超过 200 字的中文描述>",
  "image_type": "medical_imaging | medical_diagram | chart | general | unsupported",
  "is_medical": true,
  "is_diagnostic_request": false,
  "uncertain_points": ["<可选：图中难以辨识的点>"]
}
只返回 JSON，不要输出其他文字。"""


@dataclass(frozen=True)
class VLMConfig:
    model: str
    api_key: str
    base_url: str
    timeout_ms: int = 10000


@dataclass
class ImageCaption:
    ok: bool
    caption: str
    image_type: str            # medical_imaging | medical_diagram | chart | general | unsupported
    is_medical: bool
    is_diagnostic_request: bool
    uncertain_points: list[str]
    error: str | None = None


class QwenVLClient:
    """DashScope OpenAI-compatible Qwen-VL Client."""

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self._endpoint = f"{cfg.base_url.rstrip('/')}/chat/completions"

    def summarize_image(
        self,
        *,
        image_path: str | None = None,
        image_url: str | None = None,
        user_question: str | None = None,
    ) -> ImageCaption:
        if not image_path and not image_url:
            return _failed("no_image_input")

        try:
            content = self._build_user_content(image_path, image_url, user_question)
            payload = {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
            }
            headers = {
                "Authorization": f"Bearer {self.cfg.api_key}",
                "Content-Type": "application/json",
            }
            with httpx.Client(timeout=self.cfg.timeout_ms / 1000.0) as client:
                resp = client.post(self._endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_lenient(text)
            return ImageCaption(
                ok=True,
                caption=str(parsed.get("caption", "")).strip(),
                image_type=str(parsed.get("image_type", "general")).strip().lower() or "general",
                is_medical=bool(parsed.get("is_medical", False)),
                is_diagnostic_request=bool(parsed.get("is_diagnostic_request", False)),
                uncertain_points=[str(x) for x in (parsed.get("uncertain_points") or []) if x],
            )
        except Exception as exc:
            logger.exception("[QWEN_VL] summarize_image failed: %s", exc)
            return _failed(f"error:{exc}")

    def caption_many(
        self,
        items: Iterable[dict],
    ) -> list[ImageCaption]:
        # items: [{image_path?, image_url?, user_question?}, ...]
        return [self.summarize_image(**item) for item in items]

    def _build_user_content(
        self,
        image_path: str | None,
        image_url: str | None,
        user_question: str | None,
    ) -> list[dict]:
        content: list[dict] = []
        if image_url:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_path:
            content.append({
                "type": "image_url",
                "image_url": {"url": _local_image_to_data_url(image_path)},
            })
        text = (user_question or "请生成图片摘要。").strip()
        content.append({"type": "text", "text": text})
        return content


def build_qwen_vl_client(settings) -> QwenVLClient | None:
    if not getattr(settings, "multimodal_enabled", False):
        return None
    api_key = (settings.multimodal_api_key or "").strip()
    if not api_key:
        logger.warning("[QWEN_VL] no api key, multimodal disabled")
        return None
    return QwenVLClient(VLMConfig(
        model=settings.multimodal_model,
        api_key=api_key,
        base_url=settings.multimodal_base_url,
        timeout_ms=settings.multimodal_timeout_ms,
    ))


def _failed(error: str) -> ImageCaption:
    return ImageCaption(
        ok=False, caption="", image_type="unsupported",
        is_medical=False, is_diagnostic_request=False,
        uncertain_points=[], error=error,
    )


def _local_image_to_data_url(path_str: str) -> str:
    p = Path(path_str)
    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _parse_json_lenient(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]+\}", raw)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
```

> 备注：`httpx.Client` 默认不会重试。生产环境可叠加简单重试（最多 1 次，仅对 5xx / 网络错误），但第一版可先省略。

---

## 五、阶段 1：ingestion 改造

### 5.1 `src/rag/ingestion/pdf_parser.py`

`ParsedPdfDoc` 调整：

```python
@dataclass
class ParsedPdfDoc:
    title: str
    source_path: str
    raw_text: str
    pages: list[str]
    images: list[list[str]]  # 每页对应一组图片绝对路径，长度 == len(pages)
```

`parse_pdf()` 增加图片抽取：

```python
def parse_pdf(pdf_path: Path, image_dir: Path, *, min_image_bytes: int = 4096) -> ParsedPdfDoc:
    image_dir = image_dir.resolve()
    image_dir.mkdir(parents=True, exist_ok=True)
    out_dir = image_dir / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    pages: list[str] = []
    images_per_page: list[list[str]] = []

    for page_idx, page in enumerate(doc, start=1):
        text = _clean_page_text(page.get_text("text"))
        pages.append(text)

        page_imgs: list[str] = []
        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image", b"")
                ext = base_image.get("ext", "png")
                if len(image_bytes) < min_image_bytes:
                    continue
                target = out_dir / f"page_{page_idx}_fig_{img_idx}.{ext}"
                target.write_bytes(image_bytes)
                page_imgs.append(str(target.resolve()))
            except Exception as exc:
                logger.warning("extract_image_failed page=%d img=%d err=%s", page_idx, img_idx, exc)
        images_per_page.append(page_imgs)

    raw_text = "\n\n".join([p for p in pages if p]).strip()
    return ParsedPdfDoc(
        title=pdf_path.stem,
        source_path=str(pdf_path.resolve()),
        raw_text=raw_text,
        pages=pages,
        images=images_per_page,
    )
```

### 5.2 `src/rag/ingestion/pdf_to_markdown.py`

签名升级：

```python
def render_markdown(
    parsed_doc: ParsedPdfDoc,
    topic: str,
    *,
    vlm: QwenVLClient | None = None,
    static_url_prefix: str = "/static/rag_assets/images",
    assets_root: Path | None = None,
) -> str:
```

逻辑：

1. 先按现有规则输出 `# title`、`> Topic`、`## Page n` + 文本。
2. 在每个 `## Page n` 之后，遍历 `parsed_doc.images[n-1]`，调用 `vlm.summarize_image(image_path=...)`。
3. 仅保留 `caption_obj.ok=True` 且 `image_type != "unsupported"` 的结果。
4. 输出格式：

```markdown
### 图片：{image_id}

- 图片路径：{static_url_prefix}/{stem}/page_{n}_fig_{i}.{ext}
- 类型：{image_type}
- is_medical：true/false
- 摘要：{caption}

> 注：该摘要由 Qwen-VL 生成，仅用于知识检索，不构成诊断结论。
```

`image_id` 推荐：`{stem}_p{page}_f{idx}`，避免长 uuid 污染检索。

`static_url_prefix` 推导：传入 `assets_root=Path("data/rag_assets/images")`，把图片绝对路径相对化后拼前缀。

### 5.3 `src/rag/ingestion/chunking.py`

新增正则提取，并在 `_make_child_doc` 中写入 metadata：

```python
import re

_IMAGE_BLOCK_RE = re.compile(
    r"^###\s+图片：(?P<image_id>[\w\-\.]+)\s*\n"
    r"(?:\s*\n)?"
    r"(?:- 图片路径：(?P<path>.+?)\s*\n)?"
    r"(?:- 类型：(?P<type>.+?)\s*\n)?"
    r"(?:- is_medical：(?P<medical>.+?)\s*\n)?"
    r"(?:- 摘要：(?P<caption>.+?)\s*\n)?",
    flags=re.MULTILINE,
)


def _extract_image_meta(chunk_text: str) -> list[dict]:
    out: list[dict] = []
    for m in _IMAGE_BLOCK_RE.finditer(chunk_text or ""):
        out.append({
            "image_id": m.group("image_id"),
            "image_path": (m.group("path") or "").strip(),
            "image_type": (m.group("type") or "general").strip(),
            "is_medical": (m.group("medical") or "false").strip().lower() == "true",
            "caption": (m.group("caption") or "").strip(),
        })
    return out
```

`_make_child_doc` 写入：

```python
meta["images"] = _extract_image_meta(chunk_text)
meta["has_image"] = bool(meta["images"])
```

> 这一步保证 `Document.metadata.images` 是结构化列表，下游 `GenerationRouter` 直接消费。

### 5.4 `src/rag/generation/generation_router.py`

新增辅助：

```python
def _collect_image_refs(self, parents: list[Document], max_imgs: int = 5) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for p in parents:
        for img in (p.metadata.get("images") or []):
            key = img.get("image_id") or img.get("image_path")
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(img)
            if len(out) >= max_imgs:
                return out
    return out


def _append_image_references(self, answer: str, parents: list[Document], max_imgs: int = 5) -> str:
    imgs = self._collect_image_refs(parents, max_imgs=max_imgs)
    if not imgs:
        return answer
    lines = [answer, "", "参考图片："]
    for i, img in enumerate(imgs, start=1):
        cap = (img.get("caption") or "").strip()
        path = (img.get("image_path") or "").strip()
        if cap and path:
            lines.append(f"[{i}] {cap} | {path}")
        elif path:
            lines.append(f"[{i}] {path}")
        elif cap:
            lines.append(f"[{i}] {cap}")
    return "\n".join(lines)
```

`build_answer` 末尾追加：

```python
body = self._append_references(body, parents, max_refs=5)
body = self._append_image_references(body, parents, max_imgs=5)
return body
```

### 5.5 `src/scripts/ingest_pdf.py`

CLI 新增开关并构造 client：

```python
ap.add_argument("--vlm-enabled", action="store_true", help="Use Qwen-VL to summarize PDF images")
ap.add_argument("--vlm-skip-existing", action="store_true", help="Skip image summarization if md exists")

# main():
from config.settings import load_settings
from llms.qwen_vl import build_qwen_vl_client

settings = load_settings()
vlm = build_qwen_vl_client(settings) if args.vlm_enabled else None
if args.vlm_enabled and vlm is None:
    logger.warning("vlm_enabled requested but client unavailable, fallback to text-only ingestion")

# ingest_one_pdf 内：
parsed = parse_pdf(pdf_path=pdf_path, image_dir=image_dir, min_image_bytes=settings.multimodal_min_image_bytes)
markdown = render_markdown(
    parsed_doc=parsed,
    topic=topic,
    vlm=vlm,
    static_url_prefix=settings.multimodal_assets_url_prefix,
    assets_root=Path(settings.multimodal_assets_dir),
)
```

> 重要：阶段 1 的 ingest 是离线脚本。运行后会重建 markdown，再通过 `RAG_REBUILD=true` 触发索引重建。

---

## 六、阶段 2：用户上传图片问答

### 6.1 graph state 扩展（`src/graphs/multi_agent_graph.py`）

```python
class MultiAgentState(MessagesState):
    next: str | None
    blocked: bool
    supervisor_reason: str | None
    supervisor_confidence: float
    handoff_to: str | None
    handoff_reason: str | None
    # 多模态
    attachments: list[dict]   # [{image_id, image_path, public_url, mime, size}]
    had_image: bool
    image_captions: list[dict]
```

### 6.2 图片输入预检节点

```python
from agents.guardrails.image_guardrails import ImageGuardrailResult, check_image_attachments


def _build_image_input_guardrail_node(settings):
    def node(state: MultiAgentState) -> dict:
        if not state.get("had_image"):
            return {}
        result: ImageGuardrailResult = check_image_attachments(
            attachments=state.get("attachments") or [],
            allowed_mime=settings.multimodal_allowed_mime,
            max_image_bytes=settings.multimodal_max_image_bytes,
            max_images=settings.multimodal_max_images_per_request,
        )
        if result.ok:
            return {}
        return {
            "blocked": True,
            "messages": [AIMessage(content=result.user_message)],
        }
    return node
```

`src/agents/guardrails/image_guardrails.py`：

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImageGuardrailResult:
    ok: bool
    user_message: str = ""
    reason: str = ""


def check_image_attachments(
    *,
    attachments: list[dict],
    allowed_mime: list[str],
    max_image_bytes: int,
    max_images: int,
) -> ImageGuardrailResult:
    if not attachments:
        return ImageGuardrailResult(ok=True)

    if len(attachments) > max_images:
        return ImageGuardrailResult(
            ok=False,
            reason="too_many_images",
            user_message=f"一次最多支持 {max_images} 张图片，请分多次上传。",
        )

    bad_mime: list[str] = []
    too_large: list[str] = []
    for att in attachments:
        mime = (att.get("mime") or "").strip().lower()
        size = int(att.get("size") or 0)
        if mime not in allowed_mime:
            bad_mime.append(att.get("image_id") or mime or "unknown")
        if size > max_image_bytes:
            too_large.append(att.get("image_id") or "unknown")

    if bad_mime:
        return ImageGuardrailResult(
            ok=False,
            reason="bad_mime",
            user_message=f"以下图片格式不支持：{bad_mime}。仅支持 jpg/png/webp。",
        )
    if too_large:
        return ImageGuardrailResult(
            ok=False,
            reason="too_large",
            user_message=f"以下图片超过 {max_image_bytes // 1024 // 1024}MB 限制：{too_large}。",
        )
    return ImageGuardrailResult(ok=True)
```

### 6.3 `image_caption` 节点

```python
def _build_image_caption_node(vlm):
    def node(state: MultiAgentState) -> dict:
        if not state.get("had_image"):
            return {}
        attachments = state.get("attachments") or []
        if vlm is None or not attachments:
            return {
                "image_captions": [],
                "messages": [HumanMessage(content="[图片处理] 多模态服务不可用，本轮请改为文字描述。")],
            }

        captions: list[dict] = []
        for att in attachments:
            cap = vlm.summarize_image(image_path=att.get("image_path"))
            captions.append({
                "image_id": att.get("image_id"),
                "image_path": att.get("image_path"),
                "public_url": att.get("public_url"),
                "ok": cap.ok,
                "caption": cap.caption,
                "image_type": cap.image_type,
                "is_medical": cap.is_medical,
                "is_diagnostic_request": cap.is_diagnostic_request,
                "error": cap.error,
            })

        original_text = _latest_user_message_text(state)
        captions_block = _format_captions_block(captions)
        new_text = (
            f"{original_text.strip()}\n\n[随附图片摘要]\n{captions_block}"
            if original_text.strip()
            else f"用户上传了图片，请基于以下图片摘要回答。\n\n[随附图片摘要]\n{captions_block}"
        )
        logger.info(
            "[IMAGE_CAPTION] count=%d ok_count=%d",
            len(captions),
            sum(1 for c in captions if c["ok"]),
        )
        return {
            "image_captions": captions,
            "messages": [HumanMessage(content=new_text)],
        }
    return node


def _format_captions_block(captions: list[dict]) -> str:
    lines = []
    for i, c in enumerate(captions, start=1):
        if not c.get("ok"):
            lines.append(f"图片[{i}] 处理失败：{c.get('error') or 'unknown'}")
            continue
        lines.append(
            f"图片[{i}] 类型={c.get('image_type','general')} "
            f"is_medical={c.get('is_medical',False)}\n"
            f"  摘要：{c.get('caption','').strip()}"
        )
    return "\n".join(lines) if lines else "（无）"
```

> 关键：节点返回 `messages: [HumanMessage(...)]`，LangGraph 会**追加**而不是替换。supervisor 用的 `_latest_user_message_text` 取最后一条 human，因此自动看到的是 “原文 + 图片摘要”。
> 不直接修改原始消息，可以保留多模态原始上下文，便于 debug。

### 6.4 graph 拓扑接线

`build_multi_agent_graph()` 调整：

```python
vlm = build_qwen_vl_client(settings)  # 顶部导入

builder.add_node("image_input_guardrail", _build_image_input_guardrail_node(settings))
builder.add_node("image_caption", _build_image_caption_node(vlm))

# 入口接线：
if settings.guardrails_enabled:
    builder.add_edge(START, "image_input_guardrail")
    builder.add_conditional_edges(
        "image_input_guardrail",
        lambda s: "blocked" if s.get("blocked") else "caption",
        {"blocked": END, "caption": "image_caption"},
    )
    builder.add_edge("image_caption", "input_guardrail")
else:
    builder.add_edge(START, "image_input_guardrail")
    builder.add_conditional_edges(
        "image_input_guardrail",
        lambda s: "blocked" if s.get("blocked") else "caption",
        {"blocked": END, "caption": "image_caption"},
    )
    builder.add_edge("image_caption", "supervisor")
```

> `had_image=False` 时，`image_input_guardrail` 和 `image_caption` 都直接 `return {}`，等价于透传，不影响纯文本路径。

### 6.5 chat service 增加多模态入口

`src/services/chat_service.py`：

```python
from langchain_core.messages import HumanMessage
import base64, mimetypes


def _attachment_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b}"


def _build_multimodal_input(message: str, attachments: list[dict]) -> dict:
    parts: list[dict] = []
    if message.strip():
        parts.append({"type": "text", "text": message.strip()})
    for att in attachments:
        url = att.get("public_url") or _attachment_to_data_url(att["image_path"])
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return {
        "messages": [HumanMessage(content=parts)],
        "attachments": attachments,
        "had_image": bool(attachments),
    }


def stream_multimodal(
    session_id: str,
    message: str,
    attachments: list[dict],
) -> Generator[str, None, None]:
    note = f"[已附带 {len(attachments)} 张图片]" if attachments else ""
    saved_text = (message + ("\n" + note if note else "")).strip() or note
    save_message(session_id, "user", saved_text)

    latest_answer = ""
    sent_answer = ""
    collected_tool_calls: list[dict] = []
    inputs = _build_multimodal_input(message, attachments)

    try:
        if _runtime.startswith("langgraph") and _settings.agent_streaming:
            stream_multi = _runtime == "langgraph-multi"
            for update in _agent.stream(
                inputs,
                config={"configurable": {"thread_id": session_id}},
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue
                for node_name, node_state in update.items():
                    if node_name in {"supervisor", "image_input_guardrail", "image_caption"}:
                        continue
                    if not isinstance(node_state, dict):
                        continue
                    turn_messages = _current_turn_messages(node_state.get("messages", []), saved_text)
                    for msg in turn_messages:
                        _extend_tool_calls_unique(collected_tool_calls, extract_tool_calls_from_message(msg))
                        if getattr(msg, "type", "") == "ai":
                            text = extract_text_from_content(getattr(msg, "content", "")).strip()
                            if not text:
                                continue
                            latest_answer = text
                            if stream_multi:
                                continue
                            if text != sent_answer:
                                is_prefix = text.startswith(sent_answer)
                                delta = text[len(sent_answer):] if is_prefix else text
                                sent_answer = text
                                if delta:
                                    payload: dict[str, Any] = {"text": delta}
                                    if not is_prefix:
                                        payload["replace"] = True
                                    yield sse_event("chunk", payload)
            if not latest_answer:
                latest_answer, collected_tool_calls = _invoke_agent_with_inputs(session_id, inputs)
                yield sse_event("chunk", {"text": latest_answer})
        else:
            latest_answer, collected_tool_calls = _invoke_agent_with_inputs(session_id, inputs)
            yield sse_event("chunk", {"text": latest_answer})

        log_tool_calls(session_id, collected_tool_calls)
        latest_answer = sanitize_ungrounded_kb_claim(latest_answer, collected_tool_calls)
        yield sse_event("done", {"answer": latest_answer})
    except Exception as exc:
        logger.exception("[CHAT_MM_STREAM_ERROR] session=%s error=%s", session_id, exc)
        yield sse_event("error", {"detail": str(exc)})
    finally:
        if latest_answer:
            save_message(session_id, "assistant", latest_answer)
            update_conversation(session_id)
            set_title(session_id, message or "（图片对话）")


def _invoke_agent_with_inputs(session_id: str, inputs: dict) -> tuple[str, list[dict]]:
    result = _agent.invoke(inputs, config={"configurable": {"thread_id": session_id}})
    return extract_text_from_result(result), extract_tool_calls(result)
```

> `stream` 和 `stream_multimodal` 共享 90% 逻辑，可抽公共函数。第一版优先正确性。

### 6.6 chat controller 增加 multipart 接口

`src/controllers/chat_controller.py`：

```python
import os
import uuid
import shutil
from pathlib import Path
from fastapi import UploadFile, File, Form
from services.chat_service import stream_multimodal


@router.post("/chat/multimodal/stream")
async def chat_multimodal_stream(
    session_id: str = Form(...),
    message: str = Form(""),
    images: list[UploadFile] = File(default_factory=list),
) -> StreamingResponse:
    sid = session_id.strip()
    if not sid:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not message.strip() and not images:
        raise HTTPException(status_code=400, detail="message or image is required")
    if not try_acquire_session(sid):
        raise HTTPException(status_code=409, detail="This conversation is already processing another request.")

    from config.settings import load_settings  # 已经在启动时加载过；此处用注入更佳，如已有 settings 单例则直接引用
    settings = load_settings()
    if images and not settings.multimodal_enabled:
        release_session(sid)
        raise HTTPException(status_code=400, detail="multimodal disabled by config")

    attachments = await _persist_uploaded_images(sid, images, settings)

    def guarded_stream():
        try:
            yield from stream_multimodal(sid, message, attachments)
        finally:
            release_session(sid)

    return StreamingResponse(guarded_stream(), media_type="text/event-stream")


async def _persist_uploaded_images(session_id: str, files: list[UploadFile], settings) -> list[dict]:
    if not files:
        return []
    base_dir = Path(settings.multimodal_uploads_dir) / session_id
    base_dir.mkdir(parents=True, exist_ok=True)

    out: list[dict] = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower() or ".bin"
        if ext.lstrip(".") not in {"jpg", "jpeg", "png", "webp"}:
            continue
        image_id = uuid.uuid4().hex
        target = base_dir / f"{image_id}{ext}"
        with target.open("wb") as out_f:
            shutil.copyfileobj(f.file, out_f)
        size = target.stat().st_size
        public_url = f"{settings.multimodal_uploads_url_prefix}/{session_id}/{target.name}"
        out.append({
            "image_id": image_id,
            "image_path": str(target.resolve()),
            "public_url": public_url,
            "mime": f.content_type or "",
            "size": size,
            "filename": f.filename,
        })
    return out
```

> 注意：依赖 `python-multipart`。`settings` 不要每次都 `load_settings()`，应该用 app 启动时的单例。文档里这样写只是为了示意。

### 6.7 web/app.py 静态资源 mount

```python
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ...
_assets_dir = Path(_settings.multimodal_assets_dir)
_uploads_dir = Path(_settings.multimodal_uploads_dir)
_assets_dir.mkdir(parents=True, exist_ok=True)
_uploads_dir.mkdir(parents=True, exist_ok=True)

app.mount(
    _settings.multimodal_assets_url_prefix,
    StaticFiles(directory=str(_assets_dir)),
    name="rag_assets_images",
)
app.mount(
    _settings.multimodal_uploads_url_prefix,
    StaticFiles(directory=str(_uploads_dir)),
    name="user_uploads",
)
```

### 6.8 前端 `src/web/static/index.html`

最小改动思路（不改样式骨架）：

1. 输入框旁加上传按钮 `<input id="imgInput" type="file" accept="image/*" multiple hidden />`，触发 `imgInput.click()`。
2. 选中图片后在输入框上方展示缩略图（最多 3 张），可移除。
3. 提交时：

```js
async function sendMultimodal(sessionId, message, files) {
  const fd = new FormData();
  fd.append("session_id", sessionId);
  fd.append("message", message);
  for (const f of files) fd.append("images", f);
  const resp = await fetch("/api/chat/multimodal/stream", { method: "POST", body: fd });
  // 解析 SSE 同 /api/chat/stream，复用现有解析逻辑
}
```

4. 没选图片就走老的 `/api/chat/stream`；选了图就走 multimodal endpoint。

---

## 七、Prompt 与安全约束

1. `src/prompts/skills/medical_kb.py` 在已有 prompt 后追加一段：

```
- 若上下文中包含 “[随附图片摘要]”，请把这些摘要视为用户已经描述过的图片信息：
  - 可以基于摘要做知识检索和解释。
  - 不得宣称自己看到了图片，不得做诊断结论。
  - 若摘要标注 image_type=unsupported 或 ok=false，应主动询问用户重新上传。
```

2. 新增 `src/prompts/skills/multimodal.py`（仅当未来注册独立 multimodal agent 时启用，第一版可不启用）。

3. `LocalGuardrails.input_check_prompt` 不需要改，因为 image_caption 节点已经把图片信息以文本形式拼进了消息，原 LLM 文本审核流程依然有效。可以选择性增加一条规则：

```
- 若用户基于图片要求“给我具体诊断/这是什么病/告诉我严重程度”，应判为 UNSAFE: 越权诊断请求。
```

4. `output_check_prompt` 增加：

```
- 若回复中出现“根据您上传的图片可以确诊…”等绝对诊断表述，必须 FAIL。
```

---

## 八、回归与验收

### 8.1 不回归文本主路径

- 保留 `AGENT_MODE=multi`、`MULTIMODAL_ENABLED=false` 的组合，行为应与现状完全一致。
- `had_image=False` 时，`image_input_guardrail` 和 `image_caption` 都返回 `{}`。
- 现有 5 个 smoke test（`tmp_step1~4_smoke`、`tmp_validate_kungpao`）保持通过。

### 8.2 阶段 1 验收

- [ ] 任意一个 PDF 在 `--vlm-enabled` 下重新生成 markdown，文件里出现 `### 图片：xxx` 块。
- [ ] `data/rag_assets/images/{stem}/page_*_fig_*.{png|jpg}` 实际存在。
- [ ] `RAG_REBUILD=true` 重建索引后，知识库查询命中含图章节，回答末尾出现 “参考图片：” 段落。
- [ ] Web UI 中 “参考图片” 的 URL 可以通过 `/static/rag_assets/images/...` 直接打开。
- [ ] Qwen-VL 故意失效（key 错）时，ingestion 仍能完成，但跳过图片摘要，回答不含 “参考图片”。

### 8.3 阶段 2 验收

- [ ] `POST /api/chat/multimodal/stream` 上传 1 张医学示意图 + “这张图想说明什么？”，返回非诊断性的描述。
- [ ] 上传图片 + “请总结相关知识库内容”，supervisor 路由到 `medical_kb`，回答中能引用 KB 文档。
- [ ] 上传非 jpg/png/webp 文件，被 image_input_guardrail 拦截，返回友好错误。
- [ ] 单次上传 4 张图（超过 `MULTIMODAL_MAX_IMAGES_PER_REQUEST=3`）被拦截。
- [ ] 上传图片 + “直接告诉我这是哪种肿瘤”，被 input_guardrail / output_guardrail 任一层兜住，不输出确定诊断。
- [ ] `MULTIMODAL_ENABLED=false` 时调用 multimodal endpoint 返回 400。

### 8.4 简单调用样例

```bash
curl -X POST http://127.0.0.1:8000/api/chat/multimodal/stream \
  -F "session_id=test-mm-1" \
  -F "message=这张图能说明什么医学知识？" \
  -F "images=@/path/to/sample.png"
```

---

## 九、推荐落地顺序（建议 4 个 PR）

| PR | 范围 | 目的 |
|---|---|---|
| 1 | `qwen_vl.py` + `settings.py` + `.env` + `requirements.txt` + curl smoke test | 多模态基础设施，与业务隔离 |
| 2 | ingestion + chunking + generation_router + `ingest_pdf.py` 改造 | 阶段 1：文档侧图片摘要入库 + 引用 |
| 3 | graph + chat_service + chat_controller + web/app static mount + image_guardrails | 阶段 2 后端：上传图片到回答闭环 |
| 4 | `index.html` 上传 UI + 前端 SSE 兼容 + 验收用例 | 阶段 2 前端 + 联调验收 |

每个 PR 都保证：

1. `MULTIMODAL_ENABLED=false` 时退化到现状。
2. Qwen-VL 不可用时全链路降级，文本路径不受影响。
3. 失败原因落进 `logger.info`，便于排查。

---

## 十、不做事项

- 不引入 Qdrant / Milvus 等向量库迁移。
- 不引入专科 CV 模型（脑 MRI / 胸片 / 皮损分类器）。
- 不输出医学影像的诊断结论或病灶定性。
- 不在 prompt 中诱导模型 “补足细节”，所有图片信息只来自实际可见内容。
- 不在阶段 1 / 2 引入跨轮多模态记忆（后续阶段再评估）。
