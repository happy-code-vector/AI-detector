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
        # Text fits, keep as is
        return [(text, word_labels)]

    # Need to split - use word boundaries
    # Estimate words per chunk (roughly 1.5 tokens per word for DeBERTa)
    words_per_chunk = max(1, int(max_tokens / 1.5))

    chunks = []
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
            chunks.append((chunk_text, chunk_labels))

        start_idx = end_idx

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Split long texts into multiple entries")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens per entry")
    parser.add_argument("--tokenizer", type=str, default="microsoft/deberta-v3-large", help="Tokenizer name")

    args = parser.parse_args()

    print("=" * 60)
    print("SPLIT LONG TEXTS (Streaming Mode)")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Count total entries first (fast)
    print(f"\nCounting entries in: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        # Find the start of sentences array
        for line in f:
            if '"sentences"' in line:
                break
        # Count entries by counting opening braces
        total = 0
        in_string = False
        escape = False
        for line in f:
            for char in line:
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                elif char == '{' and not in_string:
                    total += 1

    print(f"Found {total} entries")

    # Process with streaming
    print(f"\nProcessing and splitting (max_tokens={args.max_tokens})...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_count = 0
    split_count = 0
    needs_split_count = 0

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        # Start JSON output
        fout.write('{"sentences": [\n')
        first_entry = True

        # Parse line by line
        current_item = ""
        brace_depth = 0
        in_sentences = False

        for line in tqdm(fin, total=total * 2, desc="Processing"):
            if '"sentences"' in line:
                in_sentences = True
                continue

            if not in_sentences:
                continue

            # Track braces to find complete JSON objects
            for char in line:
                current_item += char
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    if brace_depth == 0 and current_item.strip().startswith('{'):
                        # Complete object found, try to parse it
                        try:
                            item = json.loads(current_item.strip().rstrip(','))
                            if "text" in item and "labels" in item:
                                original_count += 1

                                # Split if needed
                                chunks = split_single_text(
                                    item["text"],
                                    item["labels"],
                                    tokenizer,
                                    args.max_tokens
                                )

                                if len(chunks) > 1:
                                    needs_split_count += 1

                                for text, labels in chunks:
                                    split_count += 1
                                    if not first_entry:
                                        fout.write(',\n')
                                    first_entry = False
                                    json.dump({"text": text, "labels": labels}, fout, ensure_ascii=False)

                        except json.JSONDecodeError:
                            pass
                        current_item = ""
                        break

        # Close JSON
        fout.write('\n]}')

    print(f"\nOriginal entries: {original_count}")
    print(f"Entries needing split: {needs_split_count}")
    print(f"Total after splitting: {split_count}")
    print(f"New entries added: {split_count - original_count}")
    print(f"\nSaved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
