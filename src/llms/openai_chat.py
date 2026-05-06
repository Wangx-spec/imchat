from langchain_openai import ChatOpenAI
from config.settings import Settings
from agents.agent_registry import LLMSpec

def build_openai_chat_model(settings: Settings) -> ChatOpenAI:
    return build_openai_chat_model_with_override(settings, None)
    
def build_openai_chat_model_with_override(
    settings: Settings,
    override: LLMSpec | None,
) -> ChatOpenAI:
    model = (override.model if override and override.model else settings.openai_model)
    temperature = (
        override.temperature if (override and override.temperature is not None) else 0
    )
    llm_kwargs = {
        "api_key": settings.openai_api_key,
        "model": model,
        "temperature": temperature,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    if override and override.top_p is not None:
        llm_kwargs["top_p"] = override.top_p
    return ChatOpenAI(**llm_kwargs)