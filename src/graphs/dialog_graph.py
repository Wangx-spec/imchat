from typing import Any

from langgraph.prebuilt import create_react_agent

from actions.basic_tools import get_actions
from config.settings import Settings
from llms.openai_chat import build_openai_chat_model


def build_dialog_graph(settings: Settings) -> Any:
    llm = build_openai_chat_model(settings)
    actions = get_actions()
    prompt = (
        "You are a helpful assistant. "
        "Use tools when needed, especially for math and current time questions."
    )
    return create_react_agent(model=llm, tools=actions, prompt=prompt, debug=settings.verbose)

