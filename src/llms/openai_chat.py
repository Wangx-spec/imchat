from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from config.settings import Settings
from agents.agent_registry import LLMSpec
from llms.glm_async_chat import GLMAsyncChatModel

def build_openai_chat_model(settings: Settings) -> BaseChatModel:
    return build_chat_model(settings, None)


def build_router_chat_model(settings: Settings) -> BaseChatModel:
    if getattr(settings, "text_model_provider", "glm_sync") in {"glm_async", "glm_sync"}:
        return build_chat_model(
            settings,
            LLMSpec(
                model=getattr(settings, "glm_router_model", None) or getattr(settings, "glm_model", "glm-4-flashx-250414"),
                temperature=0,
            ),
        )
    return build_chat_model(
        settings,
        LLMSpec(model=getattr(settings, "router_model", "qwen-max"), temperature=0),
    )


def build_chat_model(
    settings: Settings,
    override: LLMSpec | None,
) -> BaseChatModel:
    model = (override.model if override and override.model else settings.openai_model)
    temperature = (
        override.temperature if (override and override.temperature is not None) else 0
    )
    provider = getattr(settings, "text_model_provider", "glm_sync")
    if provider == "glm_async":
        glm_api_key = getattr(settings, "glm_api_key", None)
        if not glm_api_key:
            raise ValueError("Missing GLM_API_KEY for glm_async text model provider")
        glm_model = override.model if override and override.model else getattr(settings, "glm_model", "glm-4-flashx-250414")
        glm_temperature = (
            override.temperature
            if (override and override.temperature is not None)
            else getattr(settings, "glm_temperature", 1.0)
        )
        return GLMAsyncChatModel(
            api_key=glm_api_key,
            model=glm_model,
            base_url=getattr(settings, "glm_base_url", "https://open.bigmodel.cn/api/paas/v4"),
            async_submit_path=getattr(settings, "glm_async_submit_path", "/async/chat/completions"),
            async_result_path=getattr(settings, "glm_async_result_path", "/async-result/{id}"),
            poll_interval_s=getattr(settings, "glm_poll_interval_s", 1.0),
            max_poll_s=getattr(settings, "glm_max_poll_s", 30.0),
            temperature=glm_temperature,
            request_timeout_s=getattr(settings, "agent_request_timeout_s", 30.0),
        )
    if provider == "glm_sync":
        glm_api_key = getattr(settings, "glm_api_key", None)
        if not glm_api_key:
            raise ValueError("Missing GLM_API_KEY for glm_sync text model provider")
        glm_model = override.model if override and override.model else getattr(settings, "glm_model", "glm-4-flashx-250414")
        glm_temperature = (
            override.temperature
            if (override and override.temperature is not None)
            else getattr(settings, "glm_temperature", 1.0)
        )
        llm_kwargs = {
            "api_key": glm_api_key,
            "model": glm_model,
            "temperature": glm_temperature,
            "base_url": getattr(settings, "glm_base_url", "https://open.bigmodel.cn/api/paas/v4"),
            "timeout": getattr(settings, "agent_request_timeout_s", 30.0),
            "max_retries": 1,
        }
        if override and override.top_p is not None:
            llm_kwargs["top_p"] = override.top_p
        return ChatOpenAI(**llm_kwargs)

    llm_kwargs = {
        "api_key": settings.openai_api_key,
        "model": model,
        "temperature": temperature,
        "timeout": getattr(settings, "agent_request_timeout_s", 30.0),
        "max_retries": 1,
    }
    if settings.openai_base_url:
        llm_kwargs["base_url"] = settings.openai_base_url
    if override and override.top_p is not None:
        llm_kwargs["top_p"] = override.top_p
    return ChatOpenAI(**llm_kwargs)