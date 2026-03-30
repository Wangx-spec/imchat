from typing import Any

from langchain.agents import create_agent

from actions.basic_tools import get_actions
from config.settings import Settings
from llms.openai_chat import build_openai_chat_model


def build_dialog_agent(settings: Settings) -> Any:
    llm = build_openai_chat_model(settings)
    actions = get_actions()

    return create_agent(
        model=llm,
        tools=actions,
        system_prompt=(
            "You are a helpful assistant. "
            "Use tools when needed, especially for math and current time questions."
        ),
        debug=settings.verbose,
    )
