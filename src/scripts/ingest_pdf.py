from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.settings import load_settings
from llms.qwen_vl import QwenVLClient, build_qwen_vl_client

from rag.ingestion.pdf_parser import parse_pdf
from rag.ingestion.pdf_to_markdown import render_markdown

logger = logging.getLogger("scripts.ingest_pdf")

TOPIC_MAP: dict[str, tuple[str, ...]] = {
    "brain_tumor": ("brain_tumor", "brain_tumors", "glioma", "meningioma"),
    "chest_xray": ("chest_xray", "covid", "xray", "pneumonia"),
    "skin_lesion": ("skin_lesion", "melanoma", "dermoscopy", "nevus"),
    "diabetes": ("diabetes", "glucose", "insulin", "retinopathy"),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert medical PDFs into markdown corpus.")
    ap.add_argument("--src", required=True, help="PDF input directory")
    ap.add_argument("--dst", required=True, help="Markdown output root directory")
    ap.add_argument("--image-dir", default="data/parsed_docs", help="Extracted image output directory")
    ap.add_argument("--topic", default="", help="Optional fixed topic override")
    ap.add_argument("--force", action="store_true", help="Overwrite existing markdown files")
    ap.add_argument("--vlm-enabled", action="store_true", help="Use Qwen-VL to summarize PDF images")
    ap.add_argument("--vlm-skip-existing", action="store_true", help="Skip image summarization if md exists")
    return ap.parse_args()


def infer_topic(pdf_path: Path, topic_override: str = "") -> str:
    if topic_override:
        return topic_override.strip().lower()

    parts = [p.lower() for p in pdf_path.parts]
    name = pdf_path.stem.lower()

    for topic, keywords in TOPIC_MAP.items():
        if topic in parts:
            return topic
        if any(keyword in name for keyword in keywords):
            return topic

    return "general"


def iter_pdfs(src_dir: Path) -> list[Path]:
    return sorted(p for p in src_dir.rglob("*.pdf") if p.is_file())


def build_output_path(dst_root: Path, topic: str, pdf_path: Path) -> Path:
    return dst_root / topic / f"{pdf_path.stem}.md"


def ingest_one_pdf(
    pdf_path: Path,
    dst_root: Path,
    image_dir: Path,
    settings,
    vlm: QwenVLClient | None,
    *,
    force: bool = False,
    topic_override: str = "",
) -> bool:
    topic = infer_topic(pdf_path, topic_override)
    out_path = build_output_path(dst_root, topic, pdf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        logger.info("skip existing: %s", out_path)
        return False

    parsed = parse_pdf(
        pdf_path=pdf_path,
        image_dir=image_dir,
        min_image_bytes=settings.multimodal_min_image_bytes,
    )
    markdown = render_markdown(
        parsed_doc=parsed,
        topic=topic,
        vlm=vlm,
        static_url_prefix=settings.multimodal_assets_url_prefix,
        assets_root=Path(settings.multimodal_assets_dir),
    )
    out_path.write_text(markdown, encoding="utf-8")

    logger.info("ok: %s -> %s", pdf_path.name, out_path)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    src_dir = Path(args.src)
    dst_root = Path(args.dst)
    image_dir = Path(args.image_dir)

    settings = load_settings()
    vlm = build_qwen_vl_client(settings) if args.vlm_enabled else None
    if args.vlm_enabled and vlm is None:
        logger.warning("vlm_enabled requested but client unavailable, fallback to text-only ingestion")

    if not src_dir.exists():
        raise FileNotFoundError(f"source dir not found: {src_dir}")

    dst_root.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdfs = iter_pdfs(src_dir)
    logger.info("found %d pdf(s) under %s", len(pdfs), src_dir)

    created = 0
    skipped = 0
    failed = 0

    for pdf_path in pdfs:
        try:
            changed = ingest_one_pdf(
                pdf_path=pdf_path,
                dst_root=dst_root,
                image_dir=image_dir,
                settings=settings,
                vlm=vlm,
                force=args.force,
                topic_override=args.topic,
            )
            if changed:
                created += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            logger.exception("failed: %s (%s)", pdf_path, exc)

    logger.info("done: created=%d skipped=%d failed=%d", created, skipped, failed)


if __name__ == "__main__":
    main()