import json
import os
from pathlib import Path
from typing import Dict, List


def combine_datasets(data_dir: str = "training/data", output_file: str = "training/data/custom/all_datasets_combined.json") -> None:
    """
    Combine all JSON datasets in the specified directory into a single file.

    Args:
        data_dir: Directory containing the dataset files
        output_file: Path where the combined dataset will be saved
    """
    data_path = Path(data_dir)

    # Find all JSON files in the directory and subdirectories
    json_files = list(data_path.rglob("*.json"))

    # Filter out the output file if it already exists
    json_files = [f for f in json_files if f != Path(output_file)]

    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return

    print(f"Found {len(json_files)} dataset(s) to combine:")
    for f in json_files:
        print(f"  - {f}")

    # Combine all datasets
    combined_sentences = []
    total_sentences = 0

    for json_file in json_files:
        try:
            print(f"\nLoading {json_file}...")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract sentences from the dataset
            if "sentences" in data:
                sentences = data["sentences"]
                combined_sentences.extend(sentences)
                sentence_count = len(sentences)
                total_sentences += sentence_count
                print(f"  Added {sentence_count:,} sentences")
            else:
                print(f"  Warning: No 'sentences' key found in {json_file}")

        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
            continue

    # Create the combined dataset
    combined_dataset = {
        "sentences": combined_sentences
    }

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the combined dataset
    print(f"\nSaving combined dataset to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Successfully combined {total_sentences:,} sentences from {len(json_files)} dataset(s)")
    print(f"  Output saved to: {output_file}")

    # Print file size info
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine all datasets in training/data directory")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="training/data",
        help="Directory containing the dataset files (default: training/data)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/custom/all_datasets_combined.json",
        help="Output file path for the combined dataset (default: training/data/custom/all_datasets_combined.json)"
    )

    args = parser.parse_args()

    combine_datasets(args.data_dir, args.output)
