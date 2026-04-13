from typing import Any
from langgraph.prebuilt import create_react_agent
from actions.basic_tools import get_actions
from config.settings import Settings
from llms.openai_chat import build_openai_chat_model
from prompts.system_prompts import build_system_prompt
import logging

logger = logging.getLogger(__name__)

_CHECKPOINTER = None
_PG_CONN = None

def _build_checkpointer(settings: Settings):
    global _CHECKPOINTER, _PG_CONN
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    if settings.postgres_uri:
        from psycopg import Connection
        from langgraph.checkpoint.postgres import PostgresSaver
        _PG_CONN = Connection.connect(
            settings.postgres_uri,
            autocommit=True,
            prepare_threshold=0,
        )
        _CHECKPOINTER = PostgresSaver(_PG_CONN)
        _CHECKPOINTER.setup()
        logger.info("Checkpointer: PostgresSaver")
    else:
        from langgraph.checkpoint.memory import MemorySaver
        _CHECKPOINTER = MemorySaver()
        logger.info("Checkpointer: MemorySaver (in-memory fallback)")
    return _CHECKPOINTER

def build_dialog_graph(settings: Settings) -> Any:
    checkpointer = _build_checkpointer(settings)
    llm = build_openai_chat_model(settings)
    actions = get_actions(settings.enabled_skills)
    prompt = build_system_prompt(settings.enabled_skills)
    return create_react_agent(
        model=llm,
        tools=actions,
        prompt=prompt,
        debug=settings.verbose,
        checkpointer=checkpointer
    )
