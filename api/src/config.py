"""API configuration."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API settings."""

    # Model settings
    model_path: str = "models/checkpoint-best"
    model_name: str = "microsoft/deberta-v3-base"

    # API settings
    max_sentences: int = 120
    max_words_per_sentence: int = 50
    batch_size: int = 8

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    # Device settings
    device: str = "auto"  # auto, cuda, cpu

    # API metadata
    api_title: str = "AI Text Detector API"
    api_version: str = "1.0.0"
    api_description: str = "Word-level AI-generated text detection API"

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_device() -> str:
    """Get the device to use for inference."""
    if settings.device == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return settings.device
