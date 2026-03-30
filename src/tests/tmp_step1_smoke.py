from config.settings import load_settings
from rag.config import build_rag_config, sanitize_rag_config, validate_rag_config

settings = load_settings()
cfg = build_rag_config(settings)
cfg = sanitize_rag_config(cfg)
validate_rag_config(cfg)
print("STEP1_SMOKE_OK", cfg)