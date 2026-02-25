"""Data loading utilities for AI detection training."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class AIDetectionDataset(Dataset):
    """Dataset for word-level AI text detection."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        """
        Initialize dataset.

        Args:
            texts: List of sentences
            labels: List of word-level label lists (0=human, 1=AI)
            tokenizer: Pre-trained tokenizer
            max_length: Maximum sequence length
        """
        self.max_length = max_length

        # Pre-tokenize all data for faster training
        print(f"Pre-tokenizing {len(texts)} samples...")
        self.encodings = []
        for i, (text, word_labels) in enumerate(zip(texts, labels)):
            if i % 50000 == 0:
                print(f"  Tokenized {i}/{len(texts)} samples...")

            encoding = tokenizer(
                text,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True,
            )

            # Remove batch dimension
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}

            # Align word labels to tokens
            token_labels = self._align_labels_to_tokens(
                encoding["offset_mapping"].tolist(), word_labels, text
            )

            # Remove offset_mapping (not needed for training)
            encoding.pop("offset_mapping")
            encoding["labels"] = token_labels

            self.encodings.append(encoding)

        print(f"Pre-tokenization complete!")

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, idx: int) -> Dict:
        return self.encodings[idx]

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
        word_idx = 0
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
