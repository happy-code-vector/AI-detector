"""GPU detection and auto-configuration utilities."""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class GPUConfig:
    """GPU-specific training configuration."""
    name: str
    vram_gb: int
    batch_size: int
    gradient_accumulation_steps: int
    attention_implementation: str
    torch_dtype: str
    fp8_available: bool
    recommended_epochs: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_name": self.name,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "attention_implementation": self.attention_implementation,
            "torch_dtype": self.torch_dtype,
            "fp8_available": self.fp8_available,
        }


# GPU presets with optimal configurations
GPU_PRESETS = {
    # Cloud/Colab GPUs
    "T4": GPUConfig(
        name="T4",
        vram_gb=16,
        batch_size=4,
        gradient_accumulation_steps=4,
        attention_implementation="sdpa",
        torch_dtype="float16",  # T4 doesn't support BF16 well
        fp8_available=False,
        recommended_epochs=3,
    ),
    # Consumer GPUs
    "RTX 3060": GPUConfig(
        name="RTX 3060",
        vram_gb=12,
        batch_size=2,
        gradient_accumulation_steps=8,
        attention_implementation="sdpa",
        torch_dtype="bfloat16",
        fp8_available=False,
        recommended_epochs=3,
    ),
    "RTX 3090": GPUConfig(
        name="RTX 3090",
        vram_gb=24,
        batch_size=8,
        gradient_accumulation_steps=2,
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        fp8_available=False,
        recommended_epochs=3,
    ),
    "RTX 4090": GPUConfig(
        name="RTX 4090",
        vram_gb=24,
        batch_size=8,
        gradient_accumulation_steps=2,
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        fp8_available=False,
        recommended_epochs=3,
    ),
    # Datacenter GPUs
    "A100": GPUConfig(
        name="A100",
        vram_gb=40,  # or 80GB variant
        batch_size=12,
        gradient_accumulation_steps=1,
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        fp8_available=False,
        recommended_epochs=3,
    ),
    "H100": GPUConfig(
        name="H100",
        vram_gb=80,
        batch_size=16,
        gradient_accumulation_steps=1,
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        fp8_available=True,
        recommended_epochs=3,
    ),
    "H200": GPUConfig(
        name="H200",
        vram_gb=141,
        batch_size=24,
        gradient_accumulation_steps=1,
        attention_implementation="flash_attention_2",
        torch_dtype="bfloat16",
        fp8_available=True,
        recommended_epochs=3,
    ),
    "B200": GPUConfig(
        name="B200",
        vram_gb=192,
        batch_size=32,
        gradient_accumulation_steps=1,
        attention_implementation="flash_attention_2",
        torch_dtype="float8",  # FP8 training
        fp8_available=True,
        recommended_epochs=3,
    ),
}

# Default fallback for unknown GPUs
DEFAULT_CONFIG = GPUConfig(
    name="Unknown",
    vram_gb=8,
    batch_size=2,
    gradient_accumulation_steps=8,
    attention_implementation="eager",
    torch_dtype="float32",
    fp8_available=False,
    recommended_epochs=3,
)


def detect_gpu() -> Optional[str]:
    """
    Detect the GPU type.

    Returns:
        GPU name string or None if no GPU detected
    """
    if not torch.cuda.is_available():
        return None

    gpu_name = torch.cuda.get_device_name(0)

    # Map detected name to preset key
    for preset_name in GPU_PRESETS.keys():
        if preset_name.lower() in gpu_name.lower():
            return preset_name

    # Check for partial matches
    gpu_lower = gpu_name.lower()
    if "tesla t4" in gpu_lower or "t4" in gpu_lower:
        return "T4"
    elif "rtx 3060" in gpu_lower or "rtx3060" in gpu_lower:
        return "RTX 3060"
    elif "rtx 3090" in gpu_lower or "rtx3090" in gpu_lower:
        return "RTX 3090"
    elif "rtx 4090" in gpu_lower or "rtx4090" in gpu_lower:
        return "RTX 4090"
    elif "a100" in gpu_lower:
        return "A100"
    elif "h100" in gpu_lower:
        return "H100"
    elif "h200" in gpu_lower:
        return "H200"
    elif "b200" in gpu_lower:
        return "B200"

    return gpu_name  # Return actual name if no match


def get_gpu_config(gpu_name: Optional[str] = None) -> GPUConfig:
    """
    Get the optimal configuration for a GPU.

    Args:
        gpu_name: GPU name (auto-detected if None)

    Returns:
        GPUConfig with optimal settings
    """
    if gpu_name is None:
        gpu_name = detect_gpu()

    if gpu_name and gpu_name in GPU_PRESETS:
        return GPU_PRESETS[gpu_name]

    # Try to estimate based on VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if vram_gb >= 150:
            return GPU_PRESETS["B200"]
        elif vram_gb >= 100:
            return GPU_PRESETS["H200"]
        elif vram_gb >= 60:
            return GPU_PRESETS["H100"]
        elif vram_gb >= 20:
            return GPU_PRESETS["RTX 4090"]
        elif vram_gb >= 10:
            return GPU_PRESETS["RTX 3060"]

    return DEFAULT_CONFIG


def get_optimal_dtype(gpu_config: GPUConfig) -> torch.dtype:
    """Get the optimal torch dtype for the GPU."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float8": torch.bfloat16,  # FP8 training saves as BF16
    }
    return dtype_map.get(gpu_config.torch_dtype, torch.bfloat16)


def print_gpu_info(gpu_config: GPUConfig) -> None:
    """Print GPU configuration info."""
    print("\n" + "=" * 50)
    print(f"GPU Detected: {gpu_config.name}")
    print("=" * 50)
    print(f"  VRAM: {gpu_config.vram_gb} GB")
    print(f"  Batch Size: {gpu_config.batch_size}")
    print(f"  Gradient Accumulation: {gpu_config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {gpu_config.batch_size * gpu_config.gradient_accumulation_steps}")
    print(f"  Attention: {gpu_config.attention_implementation}")
    print(f"  Precision: {gpu_config.torch_dtype}")
    print(f"  FP8 Available: {'Yes' if gpu_config.fp8_available else 'No'}")
    print("=" * 50 + "\n")


def get_training_config_override(gpu_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get training config overrides based on GPU.

    Args:
        gpu_name: GPU name (auto-detected if None)

    Returns:
        Dictionary of config overrides
    """
    gpu_config = get_gpu_config(gpu_name)
    print_gpu_info(gpu_config)

    return {
        "batch_size": gpu_config.batch_size,
        "gradient_accumulation_steps": gpu_config.gradient_accumulation_steps,
        "gpu_config": gpu_config.to_dict(),
    }


# List all supported GPUs
def list_supported_gpus() -> None:
    """Print all supported GPU configurations."""
    print("\nSupported GPU Configurations:")
    print("-" * 70)
    print(f"{'GPU':<12} {'VRAM':>8} {'Batch':>8} {'Grad Acc':>10} {'Attention':>18}")
    print("-" * 70)
    for name, config in GPU_PRESETS.items():
        print(f"{name:<12} {config.vram_gb:>6}GB {config.batch_size:>8} {config.gradient_accumulation_steps:>10} {config.attention_implementation:>18}")
    print("-" * 70)


if __name__ == "__main__":
    # Test GPU detection
    print("Testing GPU Detection...")
    detected = detect_gpu()
    print(f"Detected GPU: {detected}")

    if detected:
        config = get_gpu_config(detected)
        print_gpu_info(config)

    list_supported_gpus()
