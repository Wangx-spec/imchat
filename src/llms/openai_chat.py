from langchain_openai import ChatOpenAI

from config.settings import Settings


def build_openai_chat_model(settings: Settings) -> ChatOpenAI:
    llm_kwargs = {
        "api_key": settings.openai_api_key,
        "model": settings.openai_model,
        "temperature": 0,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url

    return ChatOpenAI(**llm_kwargs)
