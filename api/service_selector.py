"""
Smart model service selector.

Automatically detects and uses ONNX Runtime when available,
falls back to PyTorch otherwise.

Usage:
    from service_selector import get_model_service
    service = get_model_service()  # Returns ONNX or PyTorch service
"""

from pathlib import Path

from config import settings


def get_model_service():
    """
    Get the appropriate model service based on available model format.

    Returns:
        ONNXModelService if ONNX model is found
        ModelService (PyTorch) as fallback
    """
    model_path = Path(settings.model_path)

    # Check if ONNX model exists
    onnx_file = model_path / "model.onnx"
    onnx_quantized = model_path / "model_int8.onnx"

    # Use ONNX if available
    if onnx_file.exists() or onnx_quantized.exists():
        try:
            from model_service_onnx import get_onnx_model_service

            print("🚀 Using ONNX Runtime for optimized CPU inference")
            return get_onnx_model_service()
        except ImportError:
            print("⚠️  ONNX Runtime not available, using PyTorch")
            from model_service import get_model_service as get_pt_service
            return get_pt_service()

    # Fall back to PyTorch
    print("📦 Using PyTorch for inference")
    from model_service import get_model_service as get_pt_service
    return get_pt_service()


# Convenience re-export
__all__ = ["get_model_service"]
