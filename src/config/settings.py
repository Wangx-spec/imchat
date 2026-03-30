import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    verbose: bool = False


def load_settings() -> Settings:
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Please create a .env file based on .env.example."
        )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    verbose = os.getenv("AGENT_VERBOSE", "false").lower() in {"1", "true", "yes"}

    return Settings(
        openai_api_key=api_key,
        openai_model=model,
        openai_base_url=base_url,
        verbose=verbose,
    )
