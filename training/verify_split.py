"""
Verify that split_data.json contains all data from all_datasets_combined.json.

Usage:
    python verify_split.py --original original.json --split split.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def extract_text_segments(text: str, labels: list, max_words: int = 340) -> list:
    """Split text into segments for comparison (same logic as split script)."""
    words = text.split()

    # Ensure labels match word count
    if len(labels) < len(words):
        labels = labels + [0] * (len(words) - len(labels))
    elif len(labels) > len(words):
        labels = labels[:len(words)]

    segments = []
    start_idx = 0

    while start_idx < len(words):
        end_idx = min(start_idx + max_words, len(words))
        segment_words = words[start_idx:end_idx]
        segment_labels = labels[start_idx:end_idx]
        segments.append((" ".join(segment_words), tuple(segment_labels)))
        start_idx = end_idx

    return segments


def main():
    parser = argparse.ArgumentParser(description="Verify split data integrity")
    parser.add_argument("--original", type=str, required=True, help="Original JSON file")
    parser.add_argument("--split", type=str, required=True, help="Split JSON file")

    args = parser.parse_args()

    print("=" * 60)
    print("VERIFY SPLIT DATA INTEGRITY")
    print("=" * 60)

    # Load original file
    print(f"\nLoading original: {args.original}")
    with open(args.original, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    original_entries = original_data["sentences"]
    print(f"Original entries: {len(original_entries)}")

    # Load split file
    print(f"\nLoading split: {args.split}")
    with open(args.split, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    split_entries = split_data["sentences"]
    print(f"Split entries: {len(split_entries)}")

    # Create a set of all text segments from original (split the same way)
    print("\nBuilding comparison sets...")

    original_segments = Counter()
    for entry in original_entries:
        text = entry["text"]
        labels = entry["labels"]
        segments = extract_text_segments(text, labels)
        for seg_text, seg_labels in segments:
            original_segments[(seg_text, seg_labels)] += 1

    # Create a set of all entries from split
    split_segments = Counter()
    for entry in split_entries:
        key = (entry["text"], tuple(entry["labels"]))
        split_segments[key] += 1

    print(f"Original unique segments: {len(original_segments)}")
    print(f"Split unique segments: {len(split_segments)}")

    # Compare
    print("\nComparing...")

    # Check what's in original but not in split
    missing = original_segments - split_segments
    if missing:
        print(f"\nMISSING in split file: {sum(missing.values())} segments ({len(missing)} unique)")
        # Show a few examples
        print("Examples of missing segments:")
        for i, (key, count) in enumerate(list(missing.items())[:5]):
            print(f"  {i+1}. Text (first 100 chars): {key[0][:100]}...")
    else:
        print("\n✓ All original segments are present in split file")

    # Check what's in split but not in original (shouldn't happen)
    extra = split_segments - original_segments
    if extra:
        print(f"\nEXTRA in split file (should not exist): {sum(extra.values())} segments")
    else:
        print("✓ No extra segments in split file")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Original entries: {len(original_entries)}")
    print(f"Split entries: {len(split_entries)}")
    print(f"Expected increase: {len(original_segments) - len(original_entries)}")
    print(f"Actual increase: {len(split_entries) - len(original_entries)}")

    if not missing and not extra:
        print("\n✓✓✓ VERIFICATION PASSED - All data preserved ✓✓✓")
    else:
        print("\n✗✗✗ VERIFICATION FAILED - Data loss detected ✗✗✗")
        if missing:
            print(f"   Missing: {sum(missing.values())} segments")
        if extra:
            print(f"   Extra: {sum(extra.values())} segments")


if __name__ == "__main__":
    main()
