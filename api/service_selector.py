"""
Model service selector.

Returns the PyTorch model service for GPU inference (BF16).

Usage:
    from service_selector import get_model_service
    service = get_model_service()
"""

from model_service import get_model_service

__all__ = ["get_model_service"]
