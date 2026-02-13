"""
Script to compare URLs between human.json and in-domain JSON files.

This script checks if there are any common URLs between:
- Human-written data (train_human.json)
- AI-generated data (in-domain/train.json, val.json, test.json)

Common URLs would indicate data leakage or overlapping samples.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from collections import Counter


def load_urls_from_json(file_path: Path) -> Set[str]:
    """
    Load all URLs from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Set of URLs
    """
    print(f"Loading URLs from: {file_path.name}")

    if not file_path.exists():
        print(f"  [!] File not found")
        return set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  [!] Unexpected data format")
            return set()

        urls = set()
        for entry in data:
            if isinstance(entry, dict):
                url = entry.get("url")
                if url:
                    urls.add(url)

        print(f"  [*] Loaded {len(urls):,} unique URLs")
        return urls

    except Exception as e:
        print(f"  [X] Error: {e}")
        return set()


def find_common_urls(urls1: Set[str], urls2: Set[str], name1: str, name2: str) -> Set[str]:
    """
    Find common URLs between two sets.

    Args:
        urls1: First set of URLs
        urls2: Second set of URLs
        name1: Name of first set
        name2: Name of second set

    Returns:
        Set of common URLs
    """
    common = urls1.intersection(urls2)

    print(f"\n{'='*60}")
    print(f"Comparing: {name1} vs {name2}")
    print(f"{'='*60}")
    print(f"Common URLs: {len(common)}")

    if common:
        print(f"\n[!] FOUND {len(common)} COMMON URLS!")
        print("\nCommon URLs (first 20):")
        for i, url in enumerate(sorted(common)[:20]):
            print(f"  {i+1:2}. {url}")

        if len(common) > 20:
            print(f"\n  ... and {len(common) - 20} more")
    else:
        print(f"\n[OK] No common URLs found - datasets are disjoint!")

    return common


def analyze_url_overlap(base_path: Path):
    """
    Analyze URL overlap between human and AI-generated datasets.

    Args:
        base_path: Base path to OpenTuringBench directory
    """
    print(f"\n{'='*60}")
    print("OPEN TURING BENCH - URL OVERLAP ANALYZER")
    print(f"{'='*60}")
    print(f"Base path: {base_path}\n")

    # Define file paths
    human_file = base_path / "train_human.json"
    in_domain_files = {
        "train": base_path / "in-domain" / "train.json",
        "val": base_path / "in-domain" / "val.json",
        "test": base_path / "in-domain" / "test.json",
    }

    # Load human URLs
    human_urls = load_urls_from_json(human_file)

    if not human_urls:
        print(f"\n[X] No human URLs loaded. Cannot proceed.")
        return

    # Compare with each in-domain file
    all_common: Dict[str, Set[str]] = {}
    total_common_count = 0

    for name, file_path in in_domain_files.items():
        ai_urls = load_urls_from_json(file_path)

        if ai_urls:
            common = find_common_urls(human_urls, ai_urls, "Human", f"in-domain/{name}.json")
            all_common[name] = common
            total_common_count += len(common)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Human URLs:       {len(human_urls):,}")
    print(f"Total in-domain files checked: {len(in_domain_files)}")
    print(f"Total common URLs found:    {total_common_count:,}")

    if total_common_count > 0:
        print(f"\n[!] DATA LEAKAGE DETECTED!")
        print(f"    Found {total_common_count} URLs that appear in BOTH human and AI datasets.")
        print(f"    This indicates overlap between training and test sets.")

        # Show breakdown by file
        print(f"\nBreakdown by file:")
        for name, common_urls in all_common.items():
            if common_urls:
                print(f"  - in-domain/{name}.json: {len(common_urls)} common URLs")
    else:
        print(f"\n[OK] NO DATA LEAKAGE")
        print(f"    Human and AI datasets are completely disjoint (no shared URLs).")


def main():
    """Main function."""
    # Determine base path
    script_path = Path(__file__).parent
    base_path = script_path.parent / "OpenTuringBench"

    if not base_path.exists():
        # Try alternate path
        base_path = Path(__file__).parent.parent / "datasets" / "OpenTuringBench"

    if not base_path.exists():
        print(f"[X] OpenTuringBench directory not found at:")
        print(f"    {base_path}")
        print(f"\nPlease ensure OpenTuringBench dataset is in the correct location.")
        return

    analyze_url_overlap(base_path)


if __name__ == "__main__":
    main()
