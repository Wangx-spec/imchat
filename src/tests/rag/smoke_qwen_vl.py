from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import importlib
from pathlib import Path

# Support running via `python src/tests/rag/smoke_qwen_vl.py`.
if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from llms.qwen_vl import QwenVLClient, QwenVLConfig


logger = logging.getLogger("scripts.smoke_qwen_vl")


def _load_project_dotenv() -> None:
    """Best-effort load .env from repo root for local smoke tests."""
    dotenv_mod = importlib.util.find_spec("dotenv")
    if dotenv_mod is None:
        logger.warning("python-dotenv not installed, skip .env autoload")
        return
    load_dotenv = importlib.import_module("dotenv").load_dotenv

    repo_root = Path(__file__).resolve().parents[3]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    else:
        load_dotenv(override=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Smoke test Qwen-VL with local image path or image URL."
    )
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-path", help="Local image file path")
    group.add_argument("--image-url", help="Public image URL")

    ap.add_argument(
        "--question",
        default="请描述图中可见内容，并返回结构化摘要。",
        help="Prompt text passed to Qwen-VL",
    )
    ap.add_argument("--model", default="", help="Override model name")
    ap.add_argument("--base-url", default="", help="Override base URL")
    ap.add_argument("--api-key", default="", help="Override API key")
    ap.add_argument(
        "--timeout-ms",
        type=int,
        default=0,
        help="Override request timeout in ms (0 means use env/default)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logs",
    )
    return ap.parse_args()


def _resolve_env(args: argparse.Namespace) -> tuple[str, str, str, int]:
    model = (
        args.model.strip()
        or os.getenv("MULTIMODAL_MODEL", "").strip()
        or "qwen3.6-plus"
    )
    base_url = (
        args.base_url.strip()
        or os.getenv("MULTIMODAL_BASE_URL", "").strip()
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    api_key = (
        args.api_key.strip()
        or os.getenv("MULTIMODAL_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or os.getenv("RAG_EMBEDDING_API_KEY", "").strip()
    )
    timeout_ms = (
        args.timeout_ms
        if args.timeout_ms > 0
        else int(os.getenv("MULTIMODAL_TIMEOUT_MS", "10000") or "10000")
    )
    return model, base_url, api_key, timeout_ms


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    _load_project_dotenv()

    model, base_url, api_key, timeout_ms = _resolve_env(args)
    if not api_key:
        print("ERROR: missing API key. Set MULTIMODAL_API_KEY or DASHSCOPE_API_KEY.")
        return 2

    if args.image_path:
        image_path = Path(args.image_path).expanduser().resolve()
        if not image_path.exists():
            print(f"ERROR: image not found: {image_path}")
            return 2
        if not image_path.is_file():
            print(f"ERROR: not a file: {image_path}")
            return 2
    else:
        image_path = None

    client = QwenVLClient(
        QwenVLConfig(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_ms=timeout_ms,
        )
    )

    logger.info(
        "calling qwen-vl model=%s base_url=%s timeout_ms=%s via=%s",
        model,
        base_url,
        timeout_ms,
        "image_path" if image_path else "image_url",
    )

    result = client.summarize_image(
        image_path=str(image_path) if image_path else None,
        image_url=args.image_url,
        user_question=args.question.strip() or "请描述图中可见内容。",
    )

    print(
        json.dumps(
            {
                "ok": result.ok,
                "caption": result.caption,
                "image_type": result.image_type,
                "is_medical": result.is_medical,
                "is_diagnostic_request": result.is_diagnostic_request,
                "uncertain_points": result.uncertain_points,
                "error": result.error,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
