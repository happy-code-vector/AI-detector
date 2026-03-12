"""
Quick verification: compare total words/tokens between original and split files.
"""

import json
import sys
from transformers import AutoTokenizer

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

def count_words_tokens(filepath, tokenizer):
    """Count total words and tokens in a file."""
    total_words = 0
    total_tokens = 0
    entry_count = 0

    if HAS_IJSON:
        with open(filepath, "rb") as f:
            for item in ijson.items(f, "sentences.item"):
                entry_count += 1
                text = item.get("text", "")
                words = text.split()
                total_words += len(words)
                tokens = tokenizer.encode(text, add_special_tokens=False)
                total_tokens += len(tokens)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("sentences", []):
            entry_count += 1
            text = item.get("text", "")
            words = text.split()
            total_words += len(words)
            tokens = tokenizer.encode(text, add_special_tokens=False)
            total_tokens += len(tokens)

    return entry_count, total_words, total_tokens


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_content.py original.json split.json")
        sys.exit(1)

    original_path = sys.argv[1]
    split_path = sys.argv[2]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

    print(f"\nAnalyzing original: {original_path}")
    orig_entries, orig_words, orig_tokens = count_words_tokens(original_path, tokenizer)
    print(f"  Entries: {orig_entries:,}")
    print(f"  Total words: {orig_words:,}")
    print(f"  Total tokens: {orig_tokens:,}")

    print(f"\nAnalyzing split: {split_path}")
    split_entries, split_words, split_tokens = count_words_tokens(split_path, tokenizer)
    print(f"  Entries: {split_entries:,}")
    print(f"  Total words: {split_words:,}")
    print(f"  Total tokens: {split_tokens:,}")

    print(f"\n{'='*50}")
    print("COMPARISON")
    print(f"{'='*50}")
    print(f"Entry count: {orig_entries:,} → {split_entries:,} (+{split_entries - orig_entries:,})")
    print(f"Word count:  {orig_words:,} → {split_words:,} ({'+' if split_words >= orig_words else ''}{split_words - orig_words:,})")
    print(f"Token count: {orig_tokens:,} → {split_tokens:,} ({'+' if split_tokens >= orig_tokens else ''}{split_tokens - orig_tokens:,})")

    if split_words == orig_words:
        print("\n✓ All words preserved!")
    else:
        diff_pct = abs(split_words - orig_words) / orig_words * 100
        print(f"\n✗ Word count differs by {abs(split_words - orig_words):,} ({diff_pct:.2f}%)")


if __name__ == "__main__":
    main()
