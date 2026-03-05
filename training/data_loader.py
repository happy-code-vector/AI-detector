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
    """Dataset for word-level AI text detection with pre-tokenization and chunking."""

    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset with pre-tokenization and chunking.

        Long texts are split into multiple chunks at word boundaries,
        ensuring all data is used for training.

        Args:
            texts: List of sentences
            labels: List of word-level label lists (0=human, 1=AI)
            tokenizer: Pre-trained tokenizer
            max_length: Maximum sequence length per chunk
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        print(f"Pre-tokenizing {len(texts)} samples with chunking (max_length={max_length})...")

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        original_count = len(texts)
        chunks_created = 0

        # Process each text
        for text, word_labels in tqdm(zip(texts, labels), total=len(texts), desc="Tokenizing"):
            chunks = self._create_chunks(text, word_labels, tokenizer, max_length)
            chunks_created += len(chunks)

            for chunk_text, chunk_labels in chunks:
                # Tokenize chunk
                encoding = tokenizer(
                    chunk_text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )

                # Align labels to tokens
                offset_mapping = encoding["offset_mapping"][0].tolist()
                token_labels = self._align_labels_to_tokens(
                    offset_mapping, chunk_labels, chunk_text
                )

                all_input_ids.append(encoding["input_ids"][0])
                all_attention_masks.append(encoding["attention_mask"][0])
                all_labels.append(torch.tensor(token_labels, dtype=torch.long))

        # Stack all tensors
        self.input_ids = torch.stack(all_input_ids)
        self.attention_masks = torch.stack(all_attention_masks)
        self.labels_list = all_labels

        print(f"Pre-tokenization complete!")
        print(f"  Original samples: {original_count}")
        print(f"  Chunks created: {chunks_created}")
        print(f"  Increase: {chunks_created - original_count} ({(chunks_created/original_count - 1)*100:.1f}% more)")

    def _create_chunks(
        self,
        text: str,
        word_labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> List[tuple]:
        """
        Split text into chunks at word boundaries if it exceeds max_length.

        Returns list of (chunk_text, chunk_labels) tuples.
        """
        words = text.split()

        if len(word_labels) < len(words):
            # Pad labels if needed
            word_labels = word_labels + [0] * (len(words) - len(word_labels))
        elif len(word_labels) > len(words):
            word_labels = word_labels[:len(words)]

        # Check if text fits in one chunk (reserve 2 for [CLS] and [SEP])
        test_encoding = tokenizer(
            text,
            max_length=None,
            truncation=False,
            return_tensors="pt",
        )
        total_tokens = test_encoding["input_ids"].shape[1]

        if total_tokens <= max_length:
            # Fits in single chunk
            return [(text, word_labels)]

        # Need to split into multiple chunks
        chunks = []
        current_word_start = 0

        while current_word_start < len(words):
            # Find how many words fit in this chunk
            chunk_end = self._find_chunk_end(
                words, current_word_start, tokenizer, max_length
            )

            # Extract chunk
            chunk_words = words[current_word_start:chunk_end]
            chunk_labels = word_labels[current_word_start:chunk_end]
            chunk_text = " ".join(chunk_words)

            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append((chunk_text, chunk_labels))

            current_word_start = chunk_end

            # Safety check to avoid infinite loop
            if chunk_end <= current_word_start:
                break

        return chunks

    def _find_chunk_end(
        self,
        words: List[str],
        start: int,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
    ) -> int:
        """
        Find the word index where chunk should end to fit within max_length.
        Uses binary search for efficiency.
        """
        # Reserve tokens for [CLS], [SEP], and safety margin
        target_tokens = max_length - 4

        # Binary search for optimal chunk size
        low, high = start + 1, len(words)
        best_end = start + 1

        while low <= high:
            mid = (low + high) // 2
            chunk_text = " ".join(words[start:mid])

            encoding = tokenizer(
                chunk_text,
                max_length=None,
                truncation=False,
                return_tensors="pt",
            )
            token_count = encoding["input_ids"].shape[1]

            if token_count <= target_tokens:
                best_end = mid
                low = mid + 1
            else:
                high = mid - 1

        # Ensure at least one word per chunk
        return max(best_end, start + 1)

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

    def save(self, path: str) -> None:
        """
        Save pre-tokenized dataset to disk.

        Args:
            path: Directory path to save tensors
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        print(f"Saving pre-tokenized dataset to {path}...")

        # Save tensors
        torch.save(self.input_ids, path / "input_ids.pt")
        torch.save(self.attention_masks, path / "attention_masks.pt")
        torch.save(self.labels_list, path / "labels.pt")

        # Save metadata
        metadata = {
            "num_samples": len(self),
            "max_length": self.max_length,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Saved {len(self)} samples to {path}")

    @classmethod
    def load(cls, path: str) -> "AIDetectionDataset":
        """
        Load pre-tokenized dataset from disk.

        Args:
            path: Directory path containing saved tensors

        Returns:
            AIDetectionDataset instance with loaded data
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pre-tokenized data not found at {path}")

        print(f"Loading pre-tokenized dataset from {path}...")

        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.tokenizer = None  # Not needed for pre-tokenized data
        instance.max_length = metadata["max_length"]

        # Load tensors
        instance.input_ids = torch.load(path / "input_ids.pt", weights_only=True)
        instance.attention_masks = torch.load(path / "attention_masks.pt", weights_only=True)
        instance.labels_list = torch.load(path / "labels.pt", weights_only=True)

        print(f"Loaded {len(instance)} samples from {path}")

        return instance


def pretokenize_and_save(
    data_path: str,
    output_path: str,
    tokenizer_name: str = "microsoft/deberta-v3-large",
    max_length: int = 512,
) -> None:
    """
    Pre-tokenize dataset and save to disk for later use on GPU.

    Run this locally on CPU to save GPU time.

    Args:
        data_path: Path to raw JSON dataset
        output_path: Path to save pre-tokenized tensors
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
    """
    from transformers import AutoTokenizer

    print("=" * 60)
    print("PRE-TOKENIZATION MODE (CPU)")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load raw data
    print(f"\nLoading raw data from: {data_path}")
    data = load_custom_dataset(data_path)

    # Create and save dataset
    dataset = AIDetectionDataset(
        texts=data["texts"],
        labels=data["labels"],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    dataset.save(output_path)

    print("\n" + "=" * 60)
    print("PRE-TOKENIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput saved to: {output_path}")
    print(f"\nTo use on GPU server:")
    print(f"  1. Copy '{output_path}' folder to GPU server")
    print(f"  2. Run: python train.py --mode full --pretokenized {output_path}")


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pre-tokenize dataset for training")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to raw JSON dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/pretokenized",
        help="Output directory for pre-tokenized tensors",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="microsoft/deberta-v3-large",
        help="Tokenizer name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    pretokenize_and_save(
        data_path=args.data,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
    )
