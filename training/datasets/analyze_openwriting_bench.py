"""
Script to analyze OpenTuringBench dataset and detect human vs AI-generated text.

This script:
1. Scans train.json for "model" attribute values
2. Identifies which entries are human-written vs AI-generated
3. Provides statistics on the dataset composition
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List


def analyze_json_file(file_path: Path) -> Dict:
    """
    Analyze a JSON file from OpenTuringBench dataset.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {file_path.name}")
    print(f"{'='*60}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[!] Unexpected data format: {type(data)}")
        return {}

    total_entries = len(data)
    model_counter: Counter = Counter()
    human_entries: List[Dict] = []
    ai_entries: Dict[str, List[Dict]] = {}

    print(f"Total entries: {total_entries:,}")

    # Analyze first 1000 entries to understand the structure
    sample_size = min(1000, total_entries)
    print(f"\nAnalyzing first {sample_size} entries...\n")

    for idx, entry in enumerate(data[:sample_size]):
        if not isinstance(entry, dict):
            continue

        model = entry.get("model", "unknown")

        # Count model occurrences
        model_counter[model] += 1

        # Categorize as human or AI
        if model.lower() in ["human", "human_written", "written"]:
            human_entries.append(entry)
        elif model.lower() not in ["unknown", ""]:
            if model not in ai_entries:
                ai_entries[model] = []
            ai_entries[model].append(entry)

    # Display results
    print(f"\n{'='*60}")
    print("MODEL DISTRIBUTION")
    print(f"{'='*60}")

    # Show top models
    top_models = model_counter.most_common(10)
    for model, count in top_models:
        percentage = (count / sample_size) * 100
        print(f"{model[:50]:50} {count:6,} ({percentage:5.2f}%)")

    # Check for human data
    print(f"\n{'='*60}")
    print("HUMAN vs AI DETECTION")
    print(f"{'='*60}")

    human_count = len(human_entries)
    ai_count = sum(len(entries) for entries in ai_entries.values())

    print(f"Human-written entries:  {human_count:6,} ({human_count/sample_size*100:5.2f}%)")
    print(f"AI-generated entries:   {ai_count:6,} ({ai_count/sample_size*100:5.2f}%)")

    if human_count > 0:
        print(f"\n[OK] Found {human_count} human-written entries in this file!")
        print("\nExample human entry:")
        if human_entries:
            example = human_entries[0]
            print(f"  URL: {example.get('url', 'N/A')[:50]}")
            print(f"  Content preview: {example.get('content', 'N/A')[:100]}...")

    # Show AI models found
    if ai_entries:
        print(f"\n[OK] Found {len(ai_entries)} different AI models:")
        for model, entries in sorted(ai_entries.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            print(f"  - {model}: {len(entries):,} entries")

    return {
        "file": str(file_path),
        "total_entries": total_entries,
        "sample_size": sample_size,
        "model_distribution": dict(model_counter),
        "human_count": human_count,
        "ai_count": ai_count,
        "ai_models": list(ai_entries.keys()),
    }


def main():
    """Main function to analyze OpenTuringBench dataset."""

    # Define paths
    base_path = Path(__file__).parent / "datasets" / "OpenTuringBench"

    if not base_path.exists():
        print(f"[X] OpenTuringBench directory not found at: {base_path}")
        return

    print(f"\n[*] OpenTuringBench Dataset Analyzer")
    print(f"Base path: {base_path}\n")

    # Files to analyze
    files_to_check = [
        base_path / "train_human.json",
        base_path / "in-domain" / "train.json",
        base_path / "in-domain" / "val.json",
        base_path / "in-domain" / "test.json",
    ]

    all_results = []

    for file_path in files_to_check:
        if file_path.exists():
            try:
                result = analyze_json_file(file_path)
                all_results.append(result)
            except Exception as e:
                print(f"[X] Error analyzing {file_path.name}: {e}")
        else:
            print(f"[!]️  File not found: {file_path.relative_to(base_path)}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    if all_results:
        total_human = sum(r["human_count"] for r in all_results)
        total_ai = sum(r["ai_count"] for r in all_results)

        print(f"Total human entries across all files:  {total_human:,}")
        print(f"Total AI entries across all files:    {total_ai:,}")

        if total_human > 0:
            print(f"\n[OK] Human data IS present in the dataset!")
        else:
            print(f"\n[!]️  No human data found - check if 'model' attribute uses different label")


if __name__ == "__main__":
    main()
