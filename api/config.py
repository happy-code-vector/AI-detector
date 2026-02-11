"""API configuration."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings

from shared_config import (
    get_model_name,
    get_checkpoint_dir,
    get_device,
    get_max_sentences,
    get_max_words_per_sentence,
    get_api_config,
)


class Settings(BaseSettings):
    """API settings."""

    # Model settings (from shared config)
    model_path: str = str(get_checkpoint_dir())
    model_name: str = get_model_name()

    # API settings (from shared config)
    max_sentences: int = get_max_sentences()
    max_words_per_sentence: int = get_max_words_per_sentence()
    batch_size: int = get_api_config().get("batch_size", 8)

    # Server settings (from shared config)
    host: str = get_api_config().get("host", "0.0.0.0")
    port: int = get_api_config().get("port", 8000)
    reload: bool = get_api_config().get("reload", True)

    # Device settings (from shared config)
    device: str = get_device()

    # API metadata (from shared config)
    api_title: str = get_api_config().get("title", "AI Text Detector API")
    api_version: str = get_api_config().get("version", "1.0.0")
    api_description: str = get_api_config().get(
        "description", "Word-level AI-generated text detection API"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Re-export for convenience
__all__ = ["settings", "get_device"]

