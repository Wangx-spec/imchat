from typing import Any
from config.settings import Settings
from graphs.checkpointer import build_checkpointer
import logging

logger = logging.getLogger(__name__)

def build_swarm_runtime(settings: Settings) -> Any:
    checkpointer = build_checkpointer(settings)
    from graphs.swarm_graph import build_swarm_graph
    logger.info(
        "[AGENT_BUILD] mode=swarm runtime=langgraph checkpointer=%s",
        type(checkpointer).__name__,
    )
    return build_swarm_graph(settings, checkpointer=checkpointer)