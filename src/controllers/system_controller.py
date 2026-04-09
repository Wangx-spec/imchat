from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["system"])

_runtime: str = ""
_index_file: Path | None = None
_rag_status: dict = {"ok": False, "reason": "not_bootstrapped"}


def init(runtime: str, index_file: Path, rag_status: dict) -> None:
    """由 app.py 启动时调一次，注入状态。"""
    global _runtime, _index_file, _rag_status
    _runtime = runtime
    _index_file = index_file
    _rag_status = rag_status  

@router.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "rag_ready": _rag_status["ok"],
        "rag_reason": _rag_status["reason"],
        "runtime": _runtime,
    }
@router.get("/")
def index() -> FileResponse:
    return FileResponse(_index_file)

