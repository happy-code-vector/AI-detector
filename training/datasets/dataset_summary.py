"""
Clean summary of OpenTuringBench datasets structure.
"""

import json
from pathlib import Path
from collections import Counter


def analyze_directory(base_path: Path):
    """Analyze and summarize all datasets."""
    print("\n" + "="*80)
    print("OPEN TURING BENCH - DATASET STRUCTURE SUMMARY")
    print("="*80 + "\n")

    # Define datasets
    datasets = {
        "Human Data": [
            base_path / "train_human.json",
        ],
        "In-Domain (AI, same topics as human)": [
            base_path / "in-domain" / "train.json",
            base_path / "in-domain" / "val.json",
            base_path / "in-domain" / "test.json",
        ],
        "In-Domain Variations (AI, variations)": [
            base_path / "in-domain-variations" / "high_temperature.json",
            base_path / "in-domain-variations" / "human_continue.json",
            base_path / "in-domain-variations" / "human_revise.json",
            base_path / "in-domain-variations" / "larger_models.json",
            base_path / "in-domain-variations" / "mid_temperature.json",
            base_path / "in-domain-variations" / "model_rewrite.json",
        ],
        "Out-of-Distribution (AI, different topics)": [
            base_path / "out-of-distribution" / "essay.json",
            base_path / "out-of-distribution" / "unseen_model.json",
        ],
    }

    all_data = {}

    for category, files in datasets.items():
        print(f"Category: {category}")
        print("-" * 80)

        category_data = {
            "files": [],
            "total_entries": 0,
            "unique_urls": set(),
            "models": Counter(),
        }

        for file_path in files:
            if not file_path.exists():
                print(f"  [!] {file_path.name} - NOT FOUND")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                urls = set()
                models = Counter()

                for entry in data:
                    url = entry.get("url")
                    model = entry.get("model")
                    if url:
                        urls.add(url)
                    if model:
                        models[model] += 1

                file_info = {
                    "name": file_path.name,
                    "path": file_path,
                    "entries": len(data),
                    "urls": urls,
                    "models": dict(models),
                }

                category_data["files"].append(file_info)
                category_data["total_entries"] += len(data)
                category_data["unique_urls"].update(urls)

                for model, count in models.items():
                    category_data["models"][model] += count

                # Print file info
                model_preview = list(models.keys())[:3]
                model_str = ", ".join(model_preview)
                if len(models) > 3:
                    model_str += f" (+{len(models) - 3} more)"

                print(f"  [*] {file_path.name}")
                print(f"      Entries: {len(data):,}")
                print(f"      Unique URLs: {len(urls):,}")
                print(f"      Models: {len(models)} different")
                print(f"      Preview: {model_str}")
                print()

            except Exception as e:
                print(f"  [X] {file_path.name}: {e}")

        all_data[category] = category_data

    # Check URL overlaps
    print("\n" + "="*80)
    print("URL OVERLAP ANALYSIS")
    print("="*80 + "\n")

    human_urls = None
    if "Human Data" in all_data:
        human_urls = all_data["Human Data"]["unique_urls"]
        print(f"Human URLs: {len(human_urls):,} unique")

    for category, data in all_data.items():
        if category == "Human Data":
            continue

        overlap = human_urls.intersection(data["unique_urls"]) if human_urls else set()
        pct = (len(overlap) / len(human_urls) * 100) if human_urls and len(human_urls) > 0 else 0

        if len(overlap) > 0:
            print(f"[!] {category}")
            print(f"    Common URLs: {len(overlap):,} ({pct:.1f}% of human)")
        else:
            print(f"[OK] {category}")
            print(f"    No common URLs (clean!)")

    # Final summary
    print("\n" + "="*80)
    print("TOTAL SUMMARY")
    print("="*80 + "\n")

    for category, data in all_data.items():
        print(f"{category}:")
        print(f"  Files: {len(data['files'])}")
        print(f"  Total entries: {data['total_entries']:,}")
        print(f"  Unique URLs: {len(data['unique_urls']):,}")
        print(f"  Different models: {len(data['models'])}")
        print()


def main():
    script_path = Path(__file__).parent
    base_path = script_path / "OpenTuringBench"

    if not base_path.exists():
        base_path = Path(__file__).parent.parent / "datasets" / "OpenTuringBench"

    if not base_path.exists():
        print("[X] OpenTuringBench directory not found")
        return

    analyze_directory(base_path)


if __name__ == "__main__":
    main()
