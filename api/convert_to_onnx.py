"""
Convert trained model to ONNX format for optimized CPU inference.

ONNX Runtime provides 2-4x speedup on CPU compared to PyTorch.

Usage:
    python convert_to_onnx.py

Requirements:
    pip install onnx onnxruntime transformers optimum
"""

import os
import sys
from pathlib import Path

import torch
import torch.onnx  # Use standard ONNX export (not onnxscript)
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
from shared_config import get_checkpoint_dir, get_model_name


def convert_to_onnx(
    model_path: Path,
    output_path: Path,
    quantize: bool = False,
) -> None:
    """
    Convert model to ONNX format.

    Args:
        model_path: Path to trained PyTorch model
        output_path: Path to save ONNX model
        quantize: Whether to apply INT8 quantization
    """
    print(f"Loading model from: {model_path}")

    # Check if it's a PEFT model
    adapter_config = model_path / "adapter_config.json"

    if adapter_config.exists():
        print("🔧 PEFT/LoRA model detected.")
        print("   Loading and merging adapters...")

        try:
            from peft import PeftModel
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
            print("❌ PEFT library not installed.")
            print("   Install with: pip install peft")
            print("   Or retrain model without PEFT for ONNX conversion")
            return
    else:
        # Load regular model
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    model.eval()

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Export to ONNX
    print("Exporting to ONNX...")

    # Sample inputs for export (use max_length from training config)
    # For 350-word policy, we use 512 to accommodate longer sequences
    dummy_input = tokenizer(
        ["This is a sample sentence for ONNX export with longer max length."],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,  # Increased from 128 for 350-word support
    )

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }
    output_names = ["logits"]

    # Export
    print("Attempting ONNX export (this may take a moment)...")
    try:
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            f=str(output_path / "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,  # Use ONNX opset 14 for transformers
            do_constant_folding=True,  # Optimize for better inference
        )
        print(f"✓ ONNX model saved to: {output_path / 'model.onnx'}")
    except Exception as export_error:
        print(f"❌ ONNX export failed: {export_error}")
        print("\nTrying alternative export method...")
        # Alternative: Use simpler export without dynamic axes
        try:
            torch.onnx.export(
                model,
                {"input_ids": dummy_input["input_ids"], "attention_mask": dummy_input["attention_mask"]},
                f=str(output_path / "model.onnx"),
                input_names=input_names,
                output_names=output_names,
                opset_version=14,
                do_constant_folding=True,
            )
            print(f"✓ ONNX model saved to: {output_path / 'model.onnx'}")
        except Exception as e2:
            raise RuntimeError(f"Both ONNX export methods failed. Consider updating PyTorch: {e2}") from e2

    print(f"✓ ONNX model saved to: {output_path / 'model.onnx'}")

    # Save tokenizer
    tokenizer.save_pretrained(str(output_path))
    print(f"✓ Tokenizer saved to: {output_path}")

    # Optional: Quantize to INT8
    if quantize:
        print("\nApplying INT8 quantization...")
        try:
            import onnxruntime
            from onnxruntime.quantization import quantize_dynamic

            quantized_path = output_path / "model_int8.onnx"
            quantize_dynamic(
                str(output_path / "model.onnx"),
                str(quantized_path),
                weight_type=onnxruntime.quantization.QuantType.QInt8,
            )
            print(f"✓ Quantized model saved to: {quantized_path}")

            # Also create a symlink/note about which to use
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
    print("ONNX Conversion for AI Detector")
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

    # Ask user if they want quantization
    print("Conversion options:")
    print("  1. FP32 ONNX (best accuracy, slower)")
    print("  2. FP32 ONNX + INT8 quantization (fastest, slight accuracy drop)")
    print()
    print("NOTE: For 350-word sentences, INT8 quantization is recommended")
    print("      to meet < 20 second processing time targets.")

    choice = input("\nSelect option (1 or 2) [default: 2 for INT8]: ").strip()

    # Default to INT8 (option 2) for 350-word policy
    quantize = choice != "1"

    try:
        convert_to_onnx(checkpoint_dir, onnx_dir, quantize=quantize)
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
