"""
Split long texts into multiple entries based on token count.

Uses ijson for memory-efficient streaming of large JSON files.

Usage:
    python split_long_texts.py --input input.json --output output.json --max-tokens 512
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("WARNING: ijson not installed. Install with: pip install ijson")
    print("Falling back to standard json.load (may use more memory)")


def split_single_text(text: str, word_labels: list, tokenizer, max_tokens: int) -> list:
    """
    Split a single text if it exceeds max_tokens.
    Returns list of (text, labels) tuples.
    """
    words = text.split()

    # Ensure labels match word count
    if len(word_labels) < len(words):
        word_labels = word_labels + [0] * (len(words) - len(word_labels))
    elif len(word_labels) > len(words):
        word_labels = word_labels[:len(words)]

    # Count tokens in full text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return [(text, word_labels)]

    # Need to split - use word boundaries
    words_per_chunk = max(1, int(max_tokens / 1.5))

    chunks = []
    start_idx = 0

    while start_idx < len(words):
        end_idx = min(start_idx + words_per_chunk, len(words))

        chunk_words = words[start_idx:end_idx]
        chunk_labels = word_labels[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)

        while len(chunk_tokens) > max_tokens and len(chunk_words) > 1:
            end_idx -= 1
            chunk_words = words[start_idx:end_idx]
            chunk_labels = word_labels[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)
            chunk_tokens = tokenizer.encode(chunk_text, add_special_tokens=False)

        if chunk_text.strip():
            chunks.append((chunk_text, chunk_labels))

        start_idx = end_idx

    return chunks


def count_entries(filepath: str) -> int:
    """Count total entries in the JSON file."""
    if HAS_IJSON:
        count = 0
        with open(filepath, "rb") as f:
            for _ in ijson.items(f, "sentences.item"):
                count += 1
        return count
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("sentences", []))


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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_count = 0
    split_count = 0
    needs_split_count = 0
    error_count = 0

    # Count entries first for progress bar
    print(f"\nCounting entries in: {args.input}")
    total_entries = count_entries(args.input)
    print(f"Found {total_entries} entries")

    # Process with streaming
    print(f"\nProcessing (max_tokens={args.max_tokens})...")

    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write('{"sentences": [\n')
        first_entry = True

        if HAS_IJSON:
            # Use ijson for streaming
            with open(args.input, "rb") as fin:
                for item in tqdm(ijson.items(fin, "sentences.item"), total=total_entries, desc="Splitting"):
                    try:
                        if "text" not in item or "labels" not in item:
                            error_count += 1
                            continue

                        original_count += 1
                        text = item["text"]
                        labels = item["labels"]

                        chunks = split_single_text(text, labels, tokenizer, args.max_tokens)

                        if len(chunks) > 1:
                            needs_split_count += 1

                        for chunk_text, chunk_labels in chunks:
                            split_count += 1
                            if not first_entry:
                                fout.write(',\n')
                            first_entry = False
                            json.dump({"text": chunk_text, "labels": chunk_labels}, fout, ensure_ascii=False)

                    except Exception as e:
                        error_count += 1
                        continue
        else:
            # Fallback to json.load
            with open(args.input, "r", encoding="utf-8") as fin:
                data = json.load(fin)

            for item in tqdm(data.get("sentences", []), desc="Splitting"):
                try:
                    if "text" not in item or "labels" not in item:
                        error_count += 1
                        continue

                    original_count += 1
                    text = item["text"]
                    labels = item["labels"]

                    chunks = split_single_text(text, labels, tokenizer, args.max_tokens)

                    if len(chunks) > 1:
                        needs_split_count += 1

                    for chunk_text, chunk_labels in chunks:
                        split_count += 1
                        if not first_entry:
                            fout.write(',\n')
                        first_entry = False
                        json.dump({"text": chunk_text, "labels": chunk_labels}, fout, ensure_ascii=False)

                except Exception as e:
                    error_count += 1
                    continue

        fout.write('\n]}')

    # Verify output file is valid JSON
    print("\nVerifying output file...")
    try:
        with open(args.output, "r", encoding="utf-8") as f:
            result = json.load(f)
        output_count = len(result.get("sentences", []))
        print(f"Output file contains {output_count} entries")
    except json.JSONDecodeError as e:
        print(f"WARNING: Output file may be corrupted: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Original entries processed: {original_count}")
    print(f"Entries needing split: {needs_split_count}")
    print(f"Total after splitting: {split_count}")
    print(f"New entries added: {split_count - original_count}")
    print(f"Errors/skipped: {error_count}")
    print(f"{'='*60}")
    print(f"\nSaved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
