"""Preprocess OpenTuringBench dataset with DIFF-based labeling.

This script pairs human and AI texts by URL and creates word-level labels
based on differences (0=human, 1=AI-changed).

Key difference from preprocess_open_turing_bench.py:
- Uses diff-based labeling like preprocess_ai_polished.py
- Matches human and AI texts by URL
- Creates mixed labels (0s and 1s) based on text differences
"""

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)
    # Replace HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Normalize quotes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace("-", "-").replace("-", "-")
    return text


def get_word_labels_from_diff(original: str, ai_text: str) -> List[int]:
    """
    Generate word-level labels by comparing human and AI texts.

    Args:
        original: Human-written text
        ai_text: AI-generated text

    Returns:
        List of word-level labels (0=human, 1=AI)
    """
    original = normalize_text(original)
    ai_text = normalize_text(ai_text)

    original_words = original.split()
    ai_words = ai_text.split()

    if len(original_words) == 0:
        # Empty original, mark all as AI
        return [1] * len(ai_words)

    if len(ai_words) == 0:
        return []

    # Use SequenceMatcher to find differences at character level
    matcher = SequenceMatcher(None, original, ai_text)

    # Track which character positions in AI text are changes
    ai_changes = [False] * len(ai_text)

    # Mark changed regions
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        # Mark all characters in this region as changed
        for j in range(j1, min(j2, len(ai_text))):
            ai_changes[j] = True

    # Convert character changes to word labels
    labels = []
    current_pos = 0
    for word_idx, word in enumerate(ai_words):
        word_start = current_pos
        word_end = current_pos + len(word)

        # Check if any character in this word was changed
        word_changed = False
        changed_chars = 0
        for j in range(word_start, min(word_end, len(ai_changes))):
            if ai_changes[j]:
                changed_chars += 1

        # Mark as AI if significant portion changed (50%+ or any change for short words)
        if changed_chars > 0:
            if len(word) <= 2 or changed_chars / len(word) >= 0.5:
                word_changed = True

        labels.append(1 if word_changed else 0)
        current_pos = word_end + 1  # +1 for space

    return labels


def load_and_pair_texts(
    human_json_path: Path,
    ai_json_path: Path,
    sample_size: int = None,
) -> Tuple[List[Dict], Dict]:
    """
    Load human and AI texts, pair them by URL.

    Args:
        human_json_path: Path to train_human.json
        ai_json_path: Path to in-domain/train.json
        sample_size: Optional limit on number of pairs

    Returns:
        List of paired sentences with diff-based labels
        Statistics dictionary
    """
    print(f"Loading human texts from: {human_json_path}")
    with open(human_json_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    print(f"Loading AI texts from: {ai_json_path}")
    with open(ai_json_path, "r", encoding="utf-8") as f:
        ai_data = json.load(f)

    # Create URL -> text mapping for human data
    human_by_url = {}
    for entry in tqdm(human_data, desc="Indexing human texts"):
        url = entry.get("url", "")
        text = entry.get("content", "")
        if url and text:
            human_by_url[url] = entry

    print(f"  Indexed {len(human_by_url):,} human texts by URL")

    # Pair AI texts with human texts by URL
    sentences = []
    stats = {
        "total_pairs": 0,
        "matched_pairs": 0,
        "unmatched_ai": 0,
        "total_words": 0,
        "total_ai_labels": 0,
        "models": defaultdict(int),
    }

    for entry in tqdm(ai_data, desc="Creating diff-based labels"):
        url = entry.get("url", "")
        ai_text = entry.get("content", "")
        model = entry.get("model", "unknown")

        if not url or not ai_text:
            continue

        stats["models"][model] += 1

        # Find matching human text
        if url in human_by_url:
            human_entry = human_by_url[url]
            human_text = human_entry.get("content", "")

            # Generate diff-based labels
            labels = get_word_labels_from_diff(human_text, ai_text)

            # Validate label count matches word count
            ai_words = ai_text.split()
            if len(labels) != len(ai_words):
                if len(labels) < len(ai_words):
                    labels.extend([0] * (len(ai_words) - len(labels)))
                else:
                    labels = labels[:len(ai_words)]

            sentences.append({
                "text": ai_text,
                "labels": labels,
                "metadata": {
                    "source": "open_turing_bench_diff",
                    "type": "paired",
                    "model": model,
                    "url": url,
                    "id": entry.get("id", -1),
                }
            })

            stats["matched_pairs"] += 1
            stats["total_words"] += len(labels)
            stats["total_ai_labels"] += sum(labels)
        else:
            stats["unmatched_ai"] += 1

        stats["total_pairs"] += 1

        if sample_size and stats["matched_pairs"] >= sample_size:
            break

    print(f"\n  Matched {stats['matched_pairs']:,} pairs")
    print(f"  Unmatched AI texts: {stats['unmatched_ai']:,}")

    return sentences, stats


def process_dataset(
    input_dir: Path,
    output_path: Path,
    sample_size: int = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Process entire OpenTuringBench dataset with diff-based labeling.

    Args:
        input_dir: Path to OpenTuringBench directory
        output_path: Path to output JSON file
        sample_size: Optional limit on number of pairs

    Returns:
        Statistics dictionary and sentences list
    """
    human_json = input_dir / "train_human.json"
    ai_json = input_dir / "in-domain" / "train.json"

    if not human_json.exists():
        raise FileNotFoundError(f"Human data not found: {human_json}")
    if not ai_json.exists():
        raise FileNotFoundError(f"AI data not found: {ai_json}")

    # Load and pair texts
    sentences, stats = load_and_pair_texts(
        human_json_path=human_json,
        ai_json_path=ai_json,
        sample_size=sample_size,
    )

    # Calculate statistics
    stats["total_sentences"] = len(sentences)
    stats["avg_ai_label_ratio"] = (
        stats["total_ai_labels"] / stats["total_words"]
        if stats["total_words"] > 0
        else 0
    )

    return stats, sentences


def print_statistics(stats: Dict):
    """Print processing statistics."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total pairs checked: {stats['total_pairs']:,}")
    print(f"Matched pairs:       {stats['matched_pairs']:,}")
    print(f"Unmatched AI texts:   {stats['unmatched_ai']:,}")
    print(f"Total sentences:      {stats['total_sentences']:,}")
    print(f"Total words:          {stats['total_words']:,}")
    print(f"Total AI labels:      {stats['total_ai_labels']:,}")
    print(f"Avg AI label ratio:   {stats['avg_ai_label_ratio']:.4f}")

    if stats["models"]:
        print("\nAI Models:")
        for model, count in sorted(stats["models"].items(), key=lambda x: -x[1])[:10]:
            print(f"  {model:30s}: {count:6d}")

    print("=" * 60)


def validate_output(output_path: Path):
    """Validate output file."""
    print(f"\nValidating output: {output_path}")

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]
    errors = []

    # Check each sentence
    for i, sent in enumerate(sentences[:100]):  # Check first 100
        text = sent["text"]
        labels = sent["labels"]
        words = text.split()

        if len(labels) != len(words):
            errors.append(f"Sentence {i}: label count {len(labels)} != word count {len(words)}")

        # Check label values
        for j, label in enumerate(labels):
            if label not in [0, 1]:
                errors.append(f"Sentence {i}, word {j}: invalid label {label}")

        # Check for mixed labels (should have both 0 and 1)
        if len(labels) > 10:
            has_zero = any(l == 0 for l in labels)
            has_one = any(l == 1 for l in labels)
            if has_zero and has_one:
                # Good! This has mixed labels
                pass

    if errors:
        print("  Validation errors found:")
        for error in errors[:10]:
            print(f"    - {error}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
    else:
        print("  Validation passed!")


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Preprocess OpenTuringBench dataset with diff-based labeling"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to OpenTuringBench directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/custom/open_turing_bench_diff.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample size (default: all data)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip output validation",
    )

    args = parser.parse_args()

    # Resolve input path
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = script_dir / "datasets" / "OpenTuringBench"

    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # Process dataset
    stats, sentences = process_dataset(
        input_dir=input_dir,
        output_path=output_path,
        sample_size=args.sample,
    )

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    print(f"\nSaving to: {output_path}")
    output_data = {
        "sentences": sentences,
        "stats": stats,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print statistics
    print_statistics(stats)

    # Validate output
    if not args.no_validate:
        validate_output(output_path)

    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput file size: {file_size_mb:.1f} MB")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
