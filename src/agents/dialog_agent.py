import logging
from typing import Any, Tuple
from langchain.agents import create_agent
from prompts.system_prompts import build_system_prompt
from actions.basic_tools import get_actions
from config.settings import Settings
from graphs.dialog_graph import build_dialog_graph
from llms.openai_chat import build_openai_chat_model


logger = logging.getLogger("chat.agent")


def build_dialog_agent_langchain(settings: Settings) -> Any:
    llm = build_openai_chat_model(settings)
    actions = get_actions(settings.enabled_skills)
    prompt = build_system_prompt(settings.enabled_skills)
    logger.info(
        "[AGENT_BUILD] mode=%s runtime=langchain skills=%s tool_count=%d",
        settings.agent_mode,
        settings.enabled_skills,
        len(actions),
    )
    return create_agent(
        model=llm,
        tools=actions,
        system_prompt=prompt,
        debug=settings.verbose,
    )

def build_dialog_runtime(settings: Settings) -> Tuple[Any, str]:
    runtime = (settings.agent_runtime or "langgraph").strip().lower()
    mode = (settings.agent_mode or "single").strip().lower()
    logger.info("[RUNTIME_SELECT] requested_runtime=%s agent_mode=%s", runtime, mode)
    if runtime == "langchain":
        return build_dialog_agent_langchain(settings), "langchain"

    try:
        if mode == "multi":
            from graphs.dialog_graph import build_multi_agent
            logger.info("[RUNTIME_SELECT] selected_runtime=langgraph-multi")
            return build_multi_agent(settings), "langgraph-multi"
        logger.info("[RUNTIME_SELECT] selected_runtime=langgraph")
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
