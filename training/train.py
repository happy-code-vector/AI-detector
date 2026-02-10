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
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Initialize model and tokenizer
    print(f"\nLoading model: {config['model']}")
    model_wrapper = AIDetectorModel(model_name=config["model"])
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",  # Disable wandb unless configured
        save_total_limit=3,
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
