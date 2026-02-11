"""
Simple ONNX conversion script for AI Detector models.

This version uses the basic torch.onnx.export without dynamic axes
to avoid compatibility issues with onnxscript.
"""

import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoModelForSequenceClassification

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from shared_config import get_checkpoint_dir, get_model_name


def convert_to_onnx_simple(
    model_path: Path,
    output_path: Path,
) -> None:
    """
    Convert model to ONNX format using simple export.

    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
    """
    print(f"Loading model from: {model_path}")

    # Load model and tokenizer
    try:
        from peft import PeftModel
        print("🔧 PEFT/LoRA model detected.")
        print("   Loading and merging adapters...")

        # Load base model
        base_model = AutoModelForTokenClassification.from_pretrained(
            str(model_path),
            num_labels=2,
        )

        # Load PEFT adapters and merge
        model = PeftModel.from_pretrained(base_model, str(model_path))
        model = model.merge_and_unload()
        print("   ✓ Adapters merged successfully")

    except ImportError:
        # Load regular model
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model.eval()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dummy input
    dummy_input = tokenizer(
        "This is a sample sentence for ONNX export.",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    # Export to ONNX (simple version)
    print("Exporting to ONNX...")
    try:
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            f=str(output_path / "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=14,
            do_constant_folding=True,
        )
        print(f"✓ ONNX model saved to: {output_path / 'model.onnx'}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        raise

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path))
    print(f"✓ Tokenizer saved to: {output_path}")


def quantize_model(onnx_path: Path, output_path: Path) -> None:
    """Quantize ONNX model to INT8."""
    print("\nApplying INT8 quantization...")

    try:
        from onnxruntime import quantize_dynamic
        quantized_path = output_path / "model_int8.onnx"

        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=onnxruntime.quantization.QuantType.QInt8,
        )

        print(f"✓ Quantized model saved to: {quantized_path}")
        print("\nNote: Use model_int8.onnx for production (faster, slightly less accurate)")
        print("      Use model.onnx for testing (baseline accuracy)")

    except ImportError:
        print("⚠️  onnxruntime not installed. Skipping quantization.")
        print("   Install with: pip install onnxruntime")


def main():
    """Main conversion function."""
    # Get paths from shared config
    checkpoint_dir = get_checkpoint_dir()
    model_name = get_model_name()

    print("=" * 60)
    print("Simple ONNX Conversion for AI Detector")
    print("=" * 60)
    print(f"Source model: {checkpoint_dir}")
    print(f"Base model: {model_name}")
    print()

    # Output path
    onnx_dir = checkpoint_dir.parent / "model_onnx"

    # Check if source exists
    if not checkpoint_dir.exists():
        print(f"❌ Model not found at: {checkpoint_dir}")
        print("   Please train the model first.")
        return

    # Ask user about quantization
    print("\nFor 350-word sentences, INT8 quantization is recommended")
    print("  1. FP32 ONNX (best accuracy, slower)")
    print("  2. FP32 ONNX + INT8 quantization (faster, slight accuracy drop)")
    print()

    choice = input("Select option (1 or 2) [default: 2 for INT8]: ").strip()
    quantize = choice != "1"

    try:
        # Step 1: Convert to ONNX FP32
        convert_to_onnx_simple(checkpoint_dir, onnx_dir)

        # Step 2: Quantize to INT8 if requested
        if quantize:
            quantize_model(onnx_dir / "model.onnx", onnx_dir)

        print("\n" + "=" * 60)
        print("Conversion complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Update shared_config.yaml:")
        print(f"   model:")
        print(f"     checkpoint_dir: \"api/models/model_onnx\"")
        print("2. Set environment variable:")
        print("   export OMP_NUM_THREADS=12")
        print("3. Restart API server")

    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
