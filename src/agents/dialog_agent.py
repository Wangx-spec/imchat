import logging
from typing import Any, Tuple

from langchain.agents import create_agent
from prompts.system_prompts import SYSTEM_PROMPT
from actions.basic_tools import get_actions
from config.settings import Settings
from graphs.dialog_graph import build_dialog_graph
from llms.openai_chat import build_openai_chat_model


logger = logging.getLogger("chat.agent")



def build_dialog_agent_langchain(settings: Settings) -> Any:
    llm = build_openai_chat_model(settings)
    actions = get_actions()
    return create_agent(
        model=llm,
        tools=actions,
        system_prompt=SYSTEM_PROMPT,
        debug=settings.verbose,
    )


def build_dialog_runtime(settings: Settings) -> Tuple[Any, str]:
    runtime = (settings.agent_runtime or "langgraph").strip().lower()
    if runtime == "langchain":
        return build_dialog_agent_langchain(settings), "langchain"

    try:
        return build_dialog_graph(settings), "langgraph"
    except Exception as exc:
        logger.warning(
            "LangGraph init failed, fallback to LangChain runtime: %s",
            exc,
            exc_info=settings.verbose,
        )
        return build_dialog_agent_langchain(settings), "langchain"


def build_dialog_agent(settings: Settings) -> Any:
    runner, _runtime = build_dialog_runtime(settings)
    return runner
