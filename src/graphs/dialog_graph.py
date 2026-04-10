from typing import Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from actions.basic_tools import get_actions
from config.settings import Settings
from llms.openai_chat import build_openai_chat_model
from prompts.system_prompts import SYSTEM_PROMPT
import logging

logger = logging.getLogger(__name__)

# 进程内短期记忆（按 thread_id 分桶）
# 注意：服务重启后会丢失
_CHECKPOINTER = None

def _build_checkpointer(settings: Settings):
    global _CHECKPOINTER
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER
    
    if settings.postgres_uri:
        from langgraph.checkpoint.postgres import PostgresSaver
        _CHECKPOINTER = PostgresSaver.from_conn_string(settings.postgres_uri)
        _CHECKPOINTER.setup()
        logger.info("Checkpointer: PostgresSaver")
    else:
        # 未配置 postgres_uri 时降级为 MemorySaver
        from langgraph.checkpoint.memory import MemorySaver
        _CHECKPOINTER = MemorySaver()
        logger.info("Checkpointer: MemorySaver (in-memory fallback)")
    return _CHECKPOINTER

def build_dialog_graph(settings: Settings) -> Any:
    checkpointer = _build_checkpointer(settings)
    llm = build_openai_chat_model(settings)
    actions = get_actions()
    return create_react_agent(
        model=llm,
        tools=actions,
        prompt=SYSTEM_PROMPT,
        debug=settings.verbose,
        checkpointer=checkpointer
    )
