import logging
from typing import Any, Tuple
from config.settings import Settings
from graphs.dialog_graph import build_swarm_runtime


logger = logging.getLogger("chat.agent")


def build_dialog_runtime(settings: Settings) -> Tuple[Any, str]:
    runtime = (settings.agent_runtime or "langgraph").strip().lower()
    mode = (settings.agent_mode or "single").strip().lower()
    logger.info("[RUNTIME_SELECT] requested_runtime=%s agent_mode=%s", runtime, mode)
    if runtime != "langgraph":
        logger.warning("[RUNTIME_SELECT] unsupported_runtime=%s fallback=langgraph-swarm", runtime)
    if mode != "swarm":
        logger.info("[RUNTIME_SELECT] forcing unified swarm graph for legacy mode=%s", mode)

    logger.info(
        "[RUNTIME_SELECT] selected_runtime=langgraph-swarm max_workers=%s timeout_s=%s complexity_threshold=%s",
        settings.swarm_max_workers,
        settings.swarm_timeout_s,
        settings.complexity_threshold,
    )
    return build_swarm_runtime(settings), "langgraph-swarm"


def build_dialog_agent(settings: Settings) -> Any:
    runner, _runtime = build_dialog_runtime(settings)
    return runner
