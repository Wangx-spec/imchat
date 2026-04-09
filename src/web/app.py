import logging
from pathlib import Path
import sys
# Support running via `python web/app.py` from `src/`.
if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from config.settings import load_settings
from agents.dialog_agent import build_dialog_runtime
from config.logging_setup import setup_logging
from services.chat_service import init as chat_service_init

from rag.bootstrap import bootstrap_rag
from controllers.chat_controller import router as chat_router
from controllers.system_controller import router as system_router, init as system_init
from fastapi import FastAPI



setup_logging()
logger = logging.getLogger("web.app")

_settings = load_settings()
_agent, _runtime = build_dialog_runtime(_settings)

chat_service_init(_agent, _settings, _runtime)

app = FastAPI(title="Chat UI")
app.include_router(chat_router)
app.include_router(system_router)

_index_file = Path(__file__).resolve().parent / "static" / "index.html"
_rag_status = {"ok": False, "reason": "not_bootstrapped"}

system_init(_runtime, _index_file, _rag_status)

@app.on_event("startup")
def on_startup() -> None:
    if not _settings.rag_enabled:
        _rag_status["ok"] = False                 # 直接改字典
        _rag_status["reason"] = "disabled"
        logger.info("rag_bootstrap_skipped reason=disabled")
        return
    try:
        ok, reason = bootstrap_rag(_settings)
        _rag_status["ok"] = ok
        _rag_status["reason"] = reason
        if ok:
            logger.info("rag_bootstrap_ok reason=%s", reason)
        else:
            logger.warning("rag_bootstrap_failed reason=%s", reason)
    except Exception as exc:
        _rag_status["ok"] = False
        _rag_status["reason"] = f"init_error:{exc}"




if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
