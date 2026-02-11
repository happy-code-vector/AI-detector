"""Preprocess AI-polished dataset for AI detection training.

This script converts the AI-polished dataset into the format expected
by the training pipeline, with word-level labels (0=human, 1=AI).
"""

import argparse
import json
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
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


def get_word_labels_from_diff(original: str, polished: str) -> List[int]:
    """
    Generate word-level labels by comparing original and polished texts.

    Args:
        original: Original human-written text
        polished: AI-polished text

    Returns:
        List of word-level labels (0=human, 1=AI)
    """
    original = normalize_text(original)
    polished = normalize_text(polished)

    original_words = original.split()
    polished_words = polished.split()

    if len(original_words) == 0:
        # Empty original, mark all as AI
        return [1] * len(polished_words)

    if len(polished_words) == 0:
        return []

    # Use SequenceMatcher to find differences at character level
    matcher = SequenceMatcher(None, original, polished)

    # Track which character positions in the polished text are changes
    polished_changes = [False] * len(polished)

    # Map character positions to word positions
    def char_to_word_pos(char_pos: int, words: List[str]) -> int:
        """Find which word contains this character position."""
        current_pos = 0
        for i, word in enumerate(words):
            word_start = current_pos
            word_end = current_pos + len(word)
            # Account for space after word
            if word_start <= char_pos < word_end:
                return i
            current_pos = word_end + 1  # +1 for space
        return len(words) - 1  # Last word if out of bounds

    # Mark changed regions
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        # Mark all characters in this region as changed
        for j in range(j1, min(j2, len(polished))):
            polished_changes[j] = True

    # Convert character changes to word labels
    labels = []
    current_pos = 0
    for word_idx, word in enumerate(polished_words):
        word_start = current_pos
        word_end = current_pos + len(word)

        # Check if any character in this word was changed
        word_changed = False
        changed_chars = 0
        for j in range(word_start, min(word_end, len(polished_changes))):
            if polished_changes[j]:
                changed_chars += 1

        # Mark as AI if significant portion changed (50%+ or any change for short words)
        if changed_chars > 0:
            if len(word) <= 2 or changed_chars / len(word) >= 0.5:
                word_changed = True

        labels.append(1 if word_changed else 0)
        current_pos = word_end + 1  # +1 for space

    return labels


def load_human_texts(csv_path: Path) -> List[Dict]:
    """
    Load human-written texts from CSV.

    Args:
        csv_path: Path to HWT_original_data.csv

    Returns:
        List of sentence dictionaries with text and labels
    """
    df = pd.read_csv(csv_path)

    sentences = []
    for _, row in df.iterrows():
        text = row.get("generation", "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        words = text.split()
        labels = [0] * len(words)  # All human

        sentences.append({
            "text": text,
            "labels": labels,
            "metadata": {
                "source": "human",
                "model": "human",
                "polish_level": None,
                "domain": row.get("domain", "unknown"),
                "original_id": row.get("id", -1),
            }
        })

    return sentences


def load_ai_polished_texts(json_path: Path) -> List[Dict]:
    """
    Load AI-polished texts from JSON.

    Args:
        json_path: Path to polished JSON file

    Returns:
        List of sentence dictionaries with text and labels
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract model and polish level from filename
    # Format: polished_texts_1_gpt.json, polished_texts_major_llama.json
    filename = json_path.stem  # Remove .json
    parts = filename.replace("polished_texts_", "").split("_")

    if len(parts) >= 2:
        polish_level = parts[0]
        model = parts[1]
    else:
        polish_level = "unknown"
        model = "unknown"

    sentences = []
    for entry in data:
        original = entry.get("original", "")
        polished = entry.get("polished", "")

        if not isinstance(polished, str) or len(polished.strip()) == 0:
            continue

        # Generate labels from diff
        labels = get_word_labels_from_diff(original, polished)

        # Validate label count matches word count
        words = polished.split()
        if len(labels) != len(words):
            # Fallback: if mismatch, try to adjust
            if len(labels) < len(words):
                labels.extend([0] * (len(words) - len(labels)))
            else:
                labels = labels[:len(words)]

        sentences.append({
            "text": polished,
            "labels": labels,
            "metadata": {
                "source": "ai_polished",
                "model": model,
                "polish_level": polish_level,
                "domain": entry.get("domain", "unknown"),
                "original_id": entry.get("id", -1),
                "polish_ratio": entry.get("polish_ratio", 0),
                "levenshtein_distance": entry.get("levenshtein_distance", 0),
                "jaccard_distance": entry.get("jaccard_distance", 0),
            }
        })

    return sentences


def process_dataset(
    input_dir: Path,
    output_path: Path,
    include_human: bool = True,
    sample_only: bool = False,
    sample_size: int = 100,
) -> Dict:
    """
    Process the entire AI-polished dataset.

    Args:
        input_dir: Path to AI-polished directory
        output_path: Path to output JSON file
        include_human: Whether to include human-written texts
        sample_only: Only process a small sample for testing
        sample_size: Number of samples per variant

    Returns:
        Statistics dictionary
    """
    human_csv = input_dir / "HWT_original_data.csv"
    polished_json_dir = input_dir / "polished_json"

    all_sentences = []
    stats = {
        "human_only": 0,
        "ai_polished": 0,
        "variants_processed": 0,
        "total_ai_labels": 0,
        "total_words": 0,
    }

    # Load human texts
    if include_human and human_csv.exists():
        print(f"Loading human texts from: {human_csv}")
        human_sentences = load_human_texts(human_csv)

        if sample_only:
            human_sentences = human_sentences[:sample_size]

        all_sentences.extend(human_sentences)
        stats["human_only"] = len(human_sentences)
        print(f"  Loaded {len(human_sentences)} human sentences")

    # Load AI-polished texts
    if polished_json_dir.exists():
        json_files = sorted(polished_json_dir.glob("*.json"))
        print(f"\nLoading AI-polished texts from: {polished_json_dir}")
        print(f"  Found {len(json_files)} JSON files")

        for json_file in tqdm(json_files, desc="Processing variants"):
            polished_sentences = load_ai_polished_texts(json_file)

            if sample_only:
                polished_sentences = polished_sentences[:sample_size]

            all_sentences.extend(polished_sentences)
            stats["ai_polished"] += len(polished_sentences)
            stats["variants_processed"] += 1

        print(f"  Loaded {stats['ai_polished']} AI-polished sentences")

    # Calculate statistics
    for sent in all_sentences:
        stats["total_words"] += len(sent["labels"])
        stats["total_ai_labels"] += sum(sent["labels"])

    stats["total_sentences"] = len(all_sentences)
    stats["avg_ai_label_ratio"] = (
        stats["total_ai_labels"] / stats["total_words"]
        if stats["total_words"] > 0
        else 0
    )

    # Calculate per-polish-level stats
    polish_level_stats = {}
    for sent in all_sentences:
        level = sent["metadata"].get("polish_level", "unknown")
        if level not in polish_level_stats:
            polish_level_stats[level] = {"sentences": 0, "ai_labels": 0, "words": 0}

        polish_level_stats[level]["sentences"] += 1
        polish_level_stats[level]["words"] += len(sent["labels"])
        polish_level_stats[level]["ai_labels"] += sum(sent["labels"])

    for level in polish_level_stats:
        if polish_level_stats[level]["words"] > 0:
            polish_level_stats[level]["ai_ratio"] = (
                polish_level_stats[level]["ai_labels"]
                / polish_level_stats[level]["words"]
            )

    stats["by_polish_level"] = polish_level_stats

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    print(f"\nSaving to: {output_path}")
    output_data = {
        "sentences": all_sentences,
        "stats": stats,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return stats


def print_statistics(stats: Dict):
    """Print processing statistics."""
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total sentences:     {stats['total_sentences']:,}")
    print(f"Human only:          {stats['human_only']:,}")
    print(f"AI-polished:         {stats['ai_polished']:,}")
    print(f"Variants processed:  {stats['variants_processed']}")
    print(f"Total words:         {stats['total_words']:,}")
    print(f"Total AI labels:     {stats['total_ai_labels']:,}")
    print(f"Avg AI label ratio:  {stats['avg_ai_label_ratio']:.4f}")

    if "by_polish_level" in stats:
        print("\nBy polish level:")
        # Sort with None values handled
        def sort_key(item):
            level, _ = item
            if level is None or level == "None" or level == "unknown":
                return ("zzz_unknown", 0)
            return (level, 0)

        for level, level_stats in sorted(stats["by_polish_level"].items(), key=sort_key):
            if level is None or level == "None":
                continue
            print(
                f"  {str(level):20s}: {level_stats['sentences']:6d} sentences, "
                f"AI ratio: {level_stats.get('ai_ratio', 0):.4f}"
            )

    print("=" * 60)


def validate_output(output_path: Path):
    """Validate the output file."""
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
    # Get script directory for relative paths
    script_dir = Path(__file__).parent.parent  # training/datasets -> training

    parser = argparse.ArgumentParser(
        description="Preprocess AI-polished dataset for training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to AI-polished directory (default: training/datasets/AI-polished)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/custom/ai_polished_full.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--include-human",
        action="store_true",
        default=True,
        help="Include human-written texts",
    )
    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Exclude human-written texts",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create a small sample for testing",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Sample size per variant (default: 100)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip output validation",
    )

    args = parser.parse_args()

    # Resolve input path (default: training/datasets/AI-polished)
    if args.input:
        input_dir = Path(args.input)
    else:
        input_dir = script_dir / "datasets" / "AI-polished"

    output_path = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    include_human = args.include_human and not args.no_human

    # Process dataset
    stats = process_dataset(
        input_dir=input_dir,
        output_path=output_path,
        include_human=include_human,
        sample_only=args.sample,
        sample_size=args.sample_size,
    )

    # Print statistics
    print_statistics(stats)

    # Validate output
    if not args.no_validate:
        validate_output(output_path)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
