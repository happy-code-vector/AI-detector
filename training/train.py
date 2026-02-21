"""Training script for AI text detection model."""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import random_split
from transformers import Trainer, TrainingArguments

from data_loader import AIDetectionDataset, load_custom_dataset
from model import AIDetectorModel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from shared_config import load_shared_config, get_checkpoint_dir, get_test_subset_size

# PEFT imports
try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    # Filter out special tokens (label = -100)
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def load_config(config_path: Optional[str] = None, mode: str = "full") -> Dict:
    """
    Load training configuration.

    Args:
        config_path: Optional path to config file
        mode: "test" or "full" training mode

    Returns:
        Configuration dictionary
    """
    # Load shared config as base
    shared = load_shared_config()

    # Flatten shared config sections into a single dict
    base_config = {}
    for section in ["model", "data", "training"]:
        if section in shared:
            base_config.update(shared[section])

    # Load specific config overrides if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        config = {**base_config, **overrides}
    else:
        config = base_config

    # Set mode
    config["mode"] = mode

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent

    if "output_dir" in config:
        config["output_dir"] = str(project_root / config["output_dir"])

    # Set data path based on mode (use passed mode, not config file mode)
    if mode == "test":
        data_path = config.get("test_data_path", "training/data/custom/test_subset.json")
    else:
        data_path = config.get("full_data_path", "training/data/custom/all_datasets_combined.json")
    config["custom_data_path"] = str(project_root / data_path)

    # Ensure numeric values are properly typed
    numeric_fields = [
        "epochs", "batch_size", "learning_rate", "max_length",
        "warmup_ratio", "weight_decay", "logging_steps", "eval_steps", "save_steps",
        "train_split", "eval_split", "test_split",
        "lora_r", "lora_alpha", "lora_dropout"
    ]

    for field in numeric_fields:
        if field in config and isinstance(config[field], str):
            try:
                if "." in config[field] or "e" in config[field].lower():
                    config[field] = float(config[field])
                else:
                    config[field] = int(config[field])
            except ValueError:
                pass

    # Ensure boolean values are properly typed
    boolean_fields = ["use_peft"]

    for field in boolean_fields:
        if field in config and isinstance(config[field], str):
            config[field] = config[field].lower() in ("true", "1", "yes", "on")

    return config


def create_test_subset(source_path: str, dest_path: str, subset_size: int) -> None:
    """
    Create a test subset from the full dataset.

    Args:
        source_path: Path to full dataset
        dest_path: Path to save subset
        subset_size: Number of samples to include
    """
    import json

    print(f"Creating test subset with {subset_size} samples...")

    with open(source_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]

    # Take first N samples (or random sample)
    if len(sentences) > subset_size:
        # Use fixed seed for reproducibility
        torch.manual_seed(42)
        indices = torch.randperm(len(sentences))[:subset_size].tolist()
        subset = [sentences[i] for i in indices]
    else:
        subset = sentences

    subset_data = {"sentences": subset}

    # Ensure directory exists
    Path(dest_path).parent.mkdir(parents=True, exist_ok=True)

    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(subset_data, f, ensure_ascii=False)

    print(f"Test subset saved to: {dest_path}")


def prepare_datasets(config: Dict, tokenizer) -> tuple:
    """
    Prepare train and evaluation datasets.

    Args:
        config: Training configuration
        tokenizer: Pre-trained tokenizer

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    custom_data_path = config.get("custom_data_path")
    mode = config.get("mode", "full")

    if custom_data_path and Path(custom_data_path).exists():
        print(f"Loading {'test' if mode == 'test' else 'full'} dataset from: {custom_data_path}")
        data = load_custom_dataset(custom_data_path)
        texts = data["texts"]
        labels = data["labels"]

        # Create combined dataset
        full_dataset = AIDetectionDataset(
            texts=texts,
            labels=labels,
            tokenizer=tokenizer,
            max_length=config["max_length"],
        )

        # Split into train/eval/test
        total_size = len(full_dataset)
        train_size = int(total_size * config["train_split"])
        eval_size = int(total_size * config["eval_split"])
        test_size = total_size - train_size - eval_size

        train_dataset, eval_dataset, _ = random_split(
            full_dataset,
            [train_size, eval_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        print(f"Dataset size: {total_size} samples")
        print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    else:
        raise FileNotFoundError(
            f"Dataset not found at {custom_data_path}. "
            "Please ensure the dataset exists or run with --mode test to create a test subset."
        )

    return train_dataset, eval_dataset


def train_model(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: str = "full",
):
    """
    Train the AI detection model.

    Args:
        config_path: Optional path to configuration file
        output_dir: Override output directory from config
        mode: "test" for quick testing, "full" for production training
    """
    # Load configuration
    config = load_config(config_path, mode)

    if output_dir:
        config["output_dir"] = output_dir

    # Use checkpoint_dir from shared config if output_dir not set
    if "output_dir" not in config:
        config["output_dir"] = str(get_checkpoint_dir())

    use_peft = config.get("use_peft", True)

    print("=" * 60)
    print(f"Training Mode: {mode.upper()}")
    print("=" * 60)
    print("\nConfiguration:")
    print(yaml.dump({k: v for k, v in config.items() if not k.startswith("_")}, default_flow_style=False))

    if use_peft and not PEFT_AVAILABLE:
        print("\n⚠️  PEFT requested but not installed. Install with: pip install peft")
        use_peft = False

    # Initialize model and tokenizer
    model_name = config.get("name", "microsoft/deberta-v3-large")
    print(f"\nLoading model: {model_name}")
    model_wrapper = AIDetectorModel(model_name=model_name)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # Apply PEFT/LoRA
    if use_peft:
        print("\n🔧 Applying LoRA adapters for efficient training...")

        lora_config = LoraConfig(
            r=config.get("lora_r", 16),
            lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=config.get("lora_dropout", 0.1),
            target_modules=["query_proj", "key_proj", "value_proj"],
            modules_to_save=["classifier"],
            task_type=TaskType.TOKEN_CLS,
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config["learning_rate"],
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=3,
        max_grad_norm=config.get("max_grad_norm", 1.0),
        bf16=True,  # BF16 for RTX 3090/5090/H100
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to: {config['output_dir']}")
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # Final evaluation
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    print(yaml.dump(metrics, default_flow_style=False))

    print("\nTraining complete!")
    return trainer


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train AI text detection model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "full"],
        default="full",
        help="Training mode: 'test' for quick local testing (small subset), 'full' for production training",
    )
    parser.add_argument(
        "--create-test-subset",
        action="store_true",
        help="Create test subset from full dataset",
    )

    args = parser.parse_args()

    # Change to training directory
    training_dir = Path(__file__).parent
    os.chdir(training_dir)

    # Create test subset if requested or if in test mode and subset doesn't exist
    project_root = training_dir.parent
    test_subset_path = project_root / "training/data/custom/test_subset.json"
    full_data_path = project_root / "training/data/custom/all_datasets_combined.json"

    if args.create_test_subset or (args.mode == "test" and not test_subset_path.exists()):
        if full_data_path.exists():
            create_test_subset(
                str(full_data_path),
                str(test_subset_path),
                get_test_subset_size()
            )
        else:
            print(f"Warning: Full dataset not found at {full_data_path}")

    # Train model
    train_model(args.config, args.output, args.mode)


if __name__ == "__main__":
    main()
