"""
Split long texts into multiple entries based on token count.

If a text has >512 tokens, it becomes 2 entries.
If >1024 tokens, it becomes 3 entries, etc.

Usage:
    python split_long_texts.py --input input.json --output output.json --max-tokens 512
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer


def split_long_texts(
    texts: list,
    labels: list,
    tokenizer,
    max_tokens: int = 512,
) -> tuple:
    """
    Split texts that exceed max_tokens into multiple entries.

    Args:
        texts: List of text strings
        labels: List of word-level labels
        tokenizer: Tokenizer to use for counting tokens
        max_tokens: Maximum tokens per entry (default 512)

    Returns:
        Tuple of (split_texts, split_labels)
    """
    split_texts = []
    split_labels = []

    for text, word_labels in tqdm(zip(texts, labels), total=len(texts), desc="Splitting"):
        words = text.split()

        # Ensure labels match word count
        if len(word_labels) < len(words):
            word_labels = word_labels + [0] * (len(words) - len(word_labels))
        elif len(word_labels) > len(words):
            word_labels = word_labels[:len(words)]

        # Count tokens in full text
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            # Text fits, keep as is
            split_texts.append(text)
            split_labels.append(word_labels)
        else:
            # Need to split - use word boundaries
            # Estimate words per chunk (roughly 1.5 tokens per word for DeBERTa)
            words_per_chunk = max(1, int(max_tokens / 1.5))

            start_idx = 0
            while start_idx < len(words):
                end_idx = min(start_idx + words_per_chunk, len(words))

                # Get chunk words and labels
                chunk_words = words[start_idx:end_idx]
                chunk_labels = word_labels[start_idx:end_idx]
                chunk_text = " ".join(chunk_words)

                # Verify chunk fits
                chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)

                # If still too long, reduce chunk size
                while len(chunk_tokens) > max_tokens and len(chunk_words) > 1:
                    end_idx -= 1
                    chunk_words = words[start_idx:end_idx]
                    chunk_labels = word_labels[start_idx:end_idx]
                    chunk_text = " ".join(chunk_words)
                    chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)

                if chunk_text.strip():
                    split_texts.append(chunk_text)
                    split_labels.append(chunk_labels)

                start_idx = end_idx

    return split_texts, split_labels


def main():
    parser = argparse.ArgumentParser(description="Split long texts into multiple entries")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens per entry")
    parser.add_argument("--tokenizer", type=str, default="microsoft/deberta-v3-large", help="Tokenizer name")

    args = parser.parse_args()

    print("=" * 60)
    print("SPLIT LONG TEXTS")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load input JSON
    print(f"\nLoading: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data["sentences"]]
    labels = [item["labels"] for item in data["sentences"]]

    print(f"Loaded {len(texts)} entries")

    # Split long texts
    print(f"\nSplitting texts that exceed {args.max_tokens} tokens...")
    split_texts, split_labels = split_long_texts(texts, labels, tokenizer, args.max_tokens)

    print(f"\nOriginal entries: {len(texts)}")
    print(f"After splitting: {len(split_texts)}")
    print(f"New entries added: {len(split_texts) - len(texts)}")

    # Save output
    output_data = {
        "sentences": [
            {"text": t, "labels": l}
            for t, l in zip(split_texts, split_labels)
        ]
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to: {args.output}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
