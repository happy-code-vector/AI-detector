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


def extract_json_objects(content: str) -> list:
    """Extract all JSON objects from a string."""
    objects = []
    i = 0

    while i < len(content):
        # Find next '{'
        brace_idx = content.find('{', i)
        if brace_idx == -1:
            break

        # Find matching '}'
        depth = 0
        in_string = False
        escape = False
        j = brace_idx

        while j < len(content):
            char = content[j]
            if escape:
                escape = False
                j += 1
                continue
            if char == '\\':
                escape = True
                j += 1
                continue
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        # Found complete object
                        obj_str = content[brace_idx:j+1]
                        objects.append(obj_str)
                        i = j + 1
                        break
            j += 1
        else:
            break

    return objects


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

    # Read file in chunks to find the sentences array
    print(f"\nReading: {args.input}")

    with open(args.input, "r", encoding="utf-8") as fin:
        # Read until we find "sentences": [
        buffer = ""
        found_start = False

        while True:
            chunk = fin.read(1000000)  # Read 1MB at a time
            if not chunk:
                break
            buffer += chunk

            if not found_start and '"sentences"' in buffer:
                # Find the opening bracket
                idx = buffer.find('"sentences"')
                bracket_idx = buffer.find('[', idx)
                if bracket_idx != -1:
                    buffer = buffer[bracket_idx + 1:]
                    found_start = True
                    break

        if not found_start:
            print("ERROR: Could not find 'sentences' array!")
            return

        # Now read the rest of the file
        print("Loading remaining content...")
        buffer += fin.read()

    print(f"Loaded {len(buffer) / 1e9:.2f} GB of content")

    # Find the closing bracket of the array
    print("Finding array boundaries...")
    depth = 0
    in_string = False
    escape = False
    end_idx = len(buffer)

    for i, char in enumerate(buffer):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
        elif not in_string:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break

    array_content = buffer[:end_idx]
    print(f"Array content size: {len(array_content) / 1e9:.2f} GB")

    # Extract all JSON objects
    print("\nExtracting JSON objects...")
    objects = extract_json_objects(array_content)
    print(f"Found {len(objects)} objects")

    # Process and write output
    print(f"\nProcessing (max_tokens={args.max_tokens})...")

    with open(args.output, "w", encoding="utf-8") as fout:
        fout.write('{"sentences": [\n')
        first_entry = True

        for obj_str in tqdm(objects, desc="Splitting"):
            try:
                item = json.loads(obj_str)
                if "text" in item and "labels" in item:
                    original_count += 1

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

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed entry")
                continue

        fout.write('\n]}')

    print(f"\n{'='*40}")
    print(f"Original entries: {original_count}")
    print(f"Entries needing split: {needs_split_count}")
    print(f"Total after splitting: {split_count}")
    print(f"New entries added: {split_count - original_count}")
    print(f"{'='*40}")
    print(f"\nSaved to: {args.output}")
    print("Done!")


if __name__ == "__main__":
    main()
