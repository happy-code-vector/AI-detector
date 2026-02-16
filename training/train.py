"""Training script for AI text detection model."""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from data_loader import AIDetectionDataset, load_custom_dataset
from model import AIDetectorModel
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from shared_config import load_shared_config, get_checkpoint_dir

# PEFT imports (optional)
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


def load_config(config_path: str) -> Dict:
    """
    Load training configuration from YAML file.

    Merges shared_config.yaml (base) with the specific config file (overrides).
    Specific config values override shared config values.
    """
    # Load shared config as base
    shared = load_shared_config()

    # Flatten shared config sections into a single dict
    base_config = {}
    for section in ["model", "data", "training"]:
        if section in shared:
            base_config.update(shared[section])

    # Load specific config overrides
    with open(config_path, "r") as f:
        overrides = yaml.safe_load(f) or {}

    # Merge: overrides take precedence
    config = {**base_config, **overrides}

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent

    if "output_dir" in config:
        config["output_dir"] = str(project_root / config["output_dir"])

    if "custom_data_path" in config:
        config["custom_data_path"] = str(project_root / config["custom_data_path"])

    # Ensure numeric values are properly typed
    numeric_fields = [
        "epochs", "batch_size", "learning_rate", "max_length",
        "warmup_steps", "logging_steps", "eval_steps", "save_steps",
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
    boolean_fields = [
        "load_in_8bit", "load_in_4bit", "use_peft"
    ]

    for field in boolean_fields:
        if field in config and isinstance(config[field], str):
            config[field] = config[field].lower() in ("true", "1", "yes", "on")

    return config


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

    if custom_data_path and Path(custom_data_path).exists():
        print(f"Loading custom dataset from: {custom_data_path}")
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

        print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

    else:
        # Load public dataset
        print("Using public datasets (not implemented - add dataset names to config)")
        # TODO: Implement public dataset loading
        raise NotImplementedError(
            "Please provide custom_data_path in config or implement public dataset loading"
        )

    return train_dataset, eval_dataset


def train_model(
    config_path: str = "configs/default.yaml",
    output_dir: Optional[str] = None,
):
    """
    Train the AI detection model.

    Args:
        config_path: Path to configuration file
        output_dir: Override output directory from config
    """
    # Load configuration
    config = load_config(config_path)
    if output_dir:
        config["output_dir"] = output_dir
    # Use checkpoint_dir from shared config if output_dir not set
    if "output_dir" not in config and "checkpoint_dir" in config:
        config["output_dir"] = config["checkpoint_dir"]

    # Get quantization settings
    load_in_8bit = config.get("load_in_8bit", False)
    load_in_4bit = config.get("load_in_4bit", False)
    use_peft = config.get("use_peft", False)

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    if load_in_8bit or load_in_4bit:
        mode = "8-bit" if load_in_8bit else "4-bit"
        print(f"\n⚡ Loading model in {mode} mode for memory efficiency")

    if use_peft and not PEFT_AVAILABLE:
        print("\n⚠️  PEFT requested but not installed. Install with: pip install peft")
        use_peft = False

    # Initialize model and tokenizer
    model_name = config.get("name", "microsoft/deberta-v3-base")
    print(f"\nLoading model: {model_name}")
    model_wrapper = AIDetectorModel(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # Apply PEFT/LoRA if enabled
    if use_peft:
        print("\n🔧 Applying LoRA adapters for efficient training...")

        # For quantized models, don't use modules_to_save (causes bitsandbytes conflicts)
        # The classifier will be trained separately
        if load_in_8bit or load_in_4bit:
            lora_config = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=config.get("lora_dropout", 0.1),
                target_modules=["query_proj", "key_proj", "value_proj"],
                task_type=TaskType.TOKEN_CLS,
            )
        else:
            # For non-quantized models, we can save the classifier
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
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",  # Changed from evaluation_strategy (transformers >= 4.20)
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",  # Disable wandb unless configured
        save_total_limit=3,
        max_grad_norm=config.get("max_grad_norm", 1.0),  # Gradient clipping to prevent NaN
        fp16=True,  # Use BFloat16 (better than FP16, supported on RTX 3090/5090)
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
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample dataset for testing",
    )

    args = parser.parse_args()

    if args.create_sample:
        from data_loader import create_sample_dataset

        sample_path = Path(__file__).parent / "data" / "custom" / "sample.json"
        create_sample_dataset(sample_path)
        print(f"\nUpdate your config with:")
        print(f"custom_data_path: {sample_path}")
        return

    # Change to training directory
    training_dir = Path(__file__).parent
    os.chdir(training_dir)

    # Train model
    train_model(args.config, args.output)


if __name__ == "__main__":
    main()
