"""Shared configuration utilities for AI Detector project."""

import os
from pathlib import Path
from typing import Dict, Any

import yaml


def get_shared_config_path() -> Path:
    """Get the path to the shared configuration file."""
    # This file is in training/, so go up one level to project root
    return Path(__file__).parent.parent / "shared_config.yaml"


def load_shared_config() -> Dict[str, Any]:
    """
    Load the shared configuration file.

    Returns:
        Dictionary with configuration sections: model, data, training, api
    """
    config_path = get_shared_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Shared configuration not found at {config_path}. "
            "Please create shared_config.yaml at project root."
        )

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config() -> Dict[str, Any]:
    """Get model configuration from shared config."""
    config = load_shared_config()
    return config.get("model", {})


def get_data_config() -> Dict[str, Any]:
    """Get data configuration from shared config."""
    config = load_shared_config()
    return config.get("data", {})


def get_training_config() -> Dict[str, Any]:
    """Get training configuration from shared config."""
    config = load_shared_config()
    return config.get("training", {})


def get_api_config() -> Dict[str, Any]:
    """Get API configuration from shared config."""
    config = load_shared_config()
    return config.get("api", {})


def get_model_name() -> str:
    """Get the base model name from shared config."""
    return get_model_config().get("name", "microsoft/deberta-v3-base")


def get_checkpoint_dir() -> Path:
    """Get the checkpoint directory from shared config (as absolute path)."""
    project_root = Path(__file__).parent.parent
    checkpoint_rel = get_model_config().get("checkpoint_dir", "api/models/checkpoint-best")
    return project_root / checkpoint_rel


def get_device() -> str:
    """Get the device from shared config (with auto-detection)."""
    device = get_model_config().get("device", "auto")

    if device == "auto":
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    return device
