"""Data loading utilities for AI detection training."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import torch
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class AIDetectionDataset(Dataset):
    """Dataset for word-level AI text detection with pre-tokenization."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        """
        Initialize dataset with pre-tokenization.

        Args:
            texts: List of sentences
            labels: List of word-level label lists (0=human, 1=AI)
            tokenizer: Pre-trained tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Pre-tokenizing {len(texts)} samples...")

        # Batch tokenize all texts at once (Rust-parallelized, very fast)
        batch_size = 10000  # Process in chunks to show progress
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing"):
            batch_texts = texts[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            # Batch tokenization (Rust-parallelized)
            encodings = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            # Align labels for each sample in batch
            for j, (text, word_labels) in enumerate(zip(batch_texts, batch_labels)):
                offset_mapping = encodings["offset_mapping"][j].tolist()
                token_labels = self._align_labels_to_tokens(
                    offset_mapping, word_labels, text
                )
                all_input_ids.append(encodings["input_ids"][j])
                all_attention_masks.append(encodings["attention_mask"][j])
                all_labels.append(torch.tensor(token_labels, dtype=torch.long))

        # Stack all tensors
        self.input_ids = torch.stack(all_input_ids)
        self.attention_masks = torch.stack(all_attention_masks)
        self.labels_list = all_labels

        print(f"Pre-tokenization complete! Dataset ready with {len(texts)} samples")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels_list[idx],
        }

    def _align_labels_to_tokens(
        self, offset_mapping, word_labels: List[int], text: str
    ) -> List[int]:
        """
        Align word-level labels to subword tokens.

        Args:
            offset_mapping: Token character offsets
            word_labels: Word-level labels
            text: Original text

        Returns:
            Token-level labels (with -100 for special tokens)
        """
        token_labels = []
        words = text.split()

        for offset in offset_mapping:
            # Special tokens ([CLS], [SEP], [PAD])
            if offset[0] == 0 and offset[1] == 0:
                token_labels.append(-100)
                continue

            # Find which word this token belongs to
            char_start, char_end = offset
            current_pos = 0

            for i, word in enumerate(words):
                word_start = text.find(word, current_pos)
                word_end = word_start + len(word)

                if word_start <= char_start < word_end:
                    token_labels.append(word_labels[i] if i < len(word_labels) else 0)
                    break
                current_pos = word_end
            else:
                token_labels.append(0)

        return token_labels


def load_public_dataset(
    dataset_name: str,
    split: Optional[str] = None,
    text_field: str = "text",
    label_field: Optional[str] = None,
) -> HFDataset:
    """
    Load a public dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset on Hugging Face Hub
        split: Dataset split to load (train/validation/test)
        text_field: Field name for text data
        label_field: Field name for labels (if applicable)

    Returns:
        Hugging Face dataset
    """
    try:
        dataset = load_dataset(dataset_name, split=split)
        return dataset
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {e}")


def load_custom_dataset(
    file_path: Union[str, Path],
) -> Dict[str, List]:
    """
    Load custom dataset from JSON or CSV.

    Expected JSON format:
    {
        "sentences": [
            {
                "text": "The quick brown fox.",
                "labels": [0, 0, 0, 0]
            }
        ]
    }

    Expected CSV format:
    text,labels
    "The quick brown fox.", "0,0,0,0"

    Args:
        file_path: Path to dataset file

    Returns:
        Dictionary with 'texts' and 'labels' keys
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "sentences" not in data:
            raise ValueError("JSON must contain 'sentences' key")

        texts = [item["text"] for item in data["sentences"]]
        labels = [item["labels"] for item in data["sentences"]]

    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)

        if "text" not in df.columns or "labels" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'labels' columns")

        texts = df["text"].tolist()
        labels = [
            [int(x) for x in labels_str.split(",")]
            for labels_str in df["labels"].tolist()
        ]
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return {"texts": texts, "labels": labels}


def create_sample_dataset(output_path: Union[str, Path]) -> None:
    """
    Create a sample dataset for testing.

    Args:
        output_path: Path to save the sample dataset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_data = {
        "sentences": [
            {
                "text": "The quick brown fox jumps over the lazy dog in the forest.",
                "labels": [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            },
            {
                "text": "Artificial intelligence is transforming modern technology rapidly.",
                "labels": [1, 1, 1, 1, 1, 0, 0, 0],
            },
            {
                "text": "Machine learning models require large datasets for training effectively.",
                "labels": [1, 1, 0, 0, 0, 0, 0, 0],
            },
            {
                "text": "Natural language processing enables computers to understand human language.",
                "labels": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            },
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    print(f"Sample dataset created at: {output_path}")
