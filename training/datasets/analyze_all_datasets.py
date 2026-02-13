"""
Comprehensive script to analyze all OpenTuringBench datasets and their URL overlaps.

This script analyzes:
- Human-written data (train_human.json)
- in-domain (AI-generated on same topics as human)
- in-domain-variations (AI variations on same topics)
- out-of-distribution (AI on different topics)

Checks for URL overlaps between all datasets.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter


def load_urls_from_json(file_path: Path) -> Tuple[Set[str], int]:
    """
    Load all URLs from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Tuple of (set of URLs, total entries)
    """
    print(f"Loading: {file_path.relative_to(file_path.parent.parent)}")

    if not file_path.exists():
        print(f"  [!] File not found")
        return set(), 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  [!] Unexpected data format")
            return set(), 0

        urls = set()
        models = Counter()

        for entry in data:
            if isinstance(entry, dict):
                url = entry.get("url")
                if url:
                    urls.add(url)
                model = entry.get("model")
                if model:
                    models[model] += 1

        print(f"  [*] {len(data):,} entries, {len(urls):,} unique URLs")
        if models:
            print(f"      Models: {len(models)} different")
            for model, count in models.most_common(5):
                print(f"        - {model[:40]:40} ({count})")

        return urls, len(data)

    except Exception as e:
        print(f"  [X] Error: {e}")
        return set(), 0


def compare_all_datasets(base_path: Path):
    """
    Compare all datasets in OpenTuringBench.

    Args:
        base_path: Base path to OpenTuringBench directory
    """
    print(f"\n{'='*80}")
    print("OPEN TURING BENCH - COMPREHENSIVE DATASET ANALYZER")
    print(f"{'='*80}")
    print(f"Base path: {base_path}\n")

    # Find all JSON files in subdirectories
    all_json_files = {}
    for json_file in base_path.glob("**/*.json"):
            # Skip if it's in a subdirectory we'll process separately
            if json_file.name == "train_human.json":
                all_json_files["train_human.json"] = json_file
            elif json_file.parent.name in ["in-domain", "in-domain-variations", "out-of-distribution"]:
                key = f"{json_file.parent.name}/{json_file.name}"
                all_json_files[key] = json_file

    if not all_json_files:
        print("[X] No JSON files found!")
        return

    print(f"Found {len(all_json_files)} datasets to analyze\n")
    print(f"{'='*80}")

    # Load all URLs
    print("STEP 1: Loading all datasets\n")
    all_urls = {}
    all_counts = {}

    for name, file_path in sorted(all_json_files.items()):
        urls, count = load_urls_from_json(file_path)
        all_urls[name] = urls
        all_counts[name] = count

    # Create comparison matrix
    print(f"\n{'='*80}")
    print("STEP 2: URL Overlap Matrix")
    print(f"{'='*80}")
    print("(Number of common URLs between datasets)\n")

    dataset_names = list(all_urls.keys())
    matrix = []

    for i, name1 in enumerate(dataset_names):
        row = [name1]
        for j, name2 in enumerate(dataset_names):
            if i == j:
                row.append(f"{all_counts[name1]:,}")
            else:
                common = len(all_urls[name1].intersection(all_urls[name2]))
                pct = (common / len(all_urls[name1]) * 100) if len(all_urls[name1]) > 0 else 0
                if common > 0:
                    row.append(f"{common:,} ({pct:.1f}%)")
                else:
                    row.append("-")
        matrix.append(row)

    # Print table
    separator = '=' * 100
    print(separator)
    header_parts = ['Dataset']
    for name in dataset_names[:5]:  # Show first 5 columns to fit
        header_parts.append(name[:20])
    header = ' | '.join(header_parts)
    print(header)
    print(separator)

    for i, name1 in enumerate(dataset_names):
        row_parts = [name1[:50]]
        for j, name2 in enumerate(dataset_names[:5]):  # Show first 5 columns
            if i == j:
                row_parts.append(str(all_counts[name1]))
            else:
                common = len(all_urls[name1].intersection(all_urls[name2]))
                pct = (common / len(all_urls[name1]) * 100) if len(all_urls[name1]) > 0 else 0
                if common > 0:
                    row_parts.append(f"{common} ({pct:.1f}%)")
                else:
                    row_parts.append("-")
        row = ' | '.join(row_parts)
        print(row)
    print(separator)

    # Detailed findings
    print(f"\n{'='*80}")
    print("STEP 3: Key Findings")
    print(f"{'='*80}")

    findings = []

    # Check human vs in-domain
    if "train_human.json" in all_urls:
        human_urls = all_urls["train_human.json"]
        human_count = all_counts["train_human.json"]

        # Find in-domain overlaps
        for name in dataset_names:
            if "in-domain/" in name or "in-domain-variations/" in name:
                common = human_urls.intersection(all_urls[name])
                if common:
                    pct = (len(common) / human_count) * 100
                    findings.append({
                        "type": "CRITICAL",
                        "finding": f"Human vs {name}",
                        "overlap": f"{len(common):,} URLs ({pct:.1f}%)"
                    })

        # Find out-of-distribution overlaps (should be minimal)
        for name in dataset_names:
            if "out-of-distribution/" in name:
                common = human_urls.intersection(all_urls[name])
                if common:
                    pct = (len(common) / human_count) * 100
                    findings.append({
                        "type": "UNEXPECTED",
                        "finding": f"Human vs {name}",
                        "overlap": f"{len(common):,} URLs ({pct:.1f}%)"
                    })
                else:
                    findings.append({
                        "type": "GOOD",
                        "finding": f"Human vs {name}",
                        "overlap": "No overlap (clean!)"
                    })

    # Print findings
    if findings:
        for finding in findings:
            icon = {
                "CRITICAL": "[!]",
                "UNEXPECTED": "[?]",
                "GOOD": "[OK]"
            }.get(finding["type"], "[*]")
            print(f"{icon} {finding['finding']:60} {finding['overlap']}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    total_human = all_counts.get("train_human.json", 0)
    total_in_domain = sum(all_counts[name] for name in dataset_names if "in-domain/" in name)
    total_in_domain_var = sum(all_counts[name] for name in dataset_names if "in-domain-variations/" in name)
    total_out_of_dist = sum(all_counts[name] for name in dataset_names if "out-of-distribution/" in name)

    print(f"train_human.json (human):              {total_human:15,} entries")
    print(f"in-domain/* (same topics, AI):         {total_in_domain:15,} entries")
    print(f"in-domain-variations/* (variations, AI): {total_in_domain_var:15,} entries")
    print(f"out-of-distribution/* (diff topics, AI): {total_out_of_dist:15,} entries")

    print(f"\nTotal datasets analyzed: {len(dataset_names)}")
    print(f"Total entries across all datasets: {sum(all_counts.values()):15,}")


def main():
    """Main function."""
    script_path = Path(__file__).parent
    base_path = script_path / "OpenTuringBench"

    if not base_path.exists():
        # Try alternate path
        base_path = Path(__file__).parent.parent / "datasets" / "OpenTuringBench"

    if not base_path.exists():
        print(f"[X] OpenTuringBench directory not found at:")
        print(f"    {base_path}")
        print(f"\nPlease ensure OpenTuringBench dataset is in the correct location.")
        return

    compare_all_datasets(base_path)


if __name__ == "__main__":
    main()
