"""Analyze token length distribution in the dataset."""

import json
from collections import Counter
from transformers import AutoTokenizer

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")

# Load dataset
print("Loading dataset...")
with open("training/data/custom/all_datasets_combined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [item["text"] for item in data["sentences"]]
print(f"Total samples: {len(texts)}")

# Analyze token lengths
print("\nAnalyzing token lengths (this may take a minute)...")
token_lengths = []
word_counts = []

for i, text in enumerate(texts):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_lengths.append(len(tokens))
    word_counts.append(len(text.split()))

    if (i + 1) % 50000 == 0:
        print(f"  Processed {i + 1}/{len(texts)} samples...")

# Calculate statistics
import statistics

print("\n" + "="*60)
print("TOKEN LENGTH DISTRIBUTION")
print("="*60)

print(f"\nTotal samples: {len(token_lengths)}")
print(f"\nToken length stats:")
print(f"  Min: {min(token_lengths)}")
print(f"  Max: {max(token_lengths)}")
print(f"  Mean: {statistics.mean(token_lengths):.1f}")
print(f"  Median: {statistics.median(token_lengths):.1f}")
print(f"  Std dev: {statistics.stdev(token_lengths):.1f}")

print(f"\nWord count stats:")
print(f"  Min: {min(word_counts)}")
print(f"  Max: {max(word_counts)}")
print(f"  Mean: {statistics.mean(word_counts):.1f}")
print(f"  Median: {statistics.median(word_counts):.1f}")

# Distribution by ranges
print("\n" + "-"*60)
print("DISTRIBUTION BY TOKEN LENGTH RANGES")
print("-"*60)

ranges = [
    (0, 128),
    (128, 256),
    (256, 384),
    (384, 512),
    (512, 768),
    (768, 1024),
    (1024, float('inf'))
]

total = len(token_lengths)
cumulative = 0

for low, high in ranges:
    count = sum(1 for l in token_lengths if low <= l < high)
    pct = count / total * 100
    cumulative += pct
    high_str = str(int(high)) if high != float('inf') else "+"
    bar = "█" * int(pct / 2)
    print(f"  {low:4d}-{high_str:>4}: {count:7d} ({pct:5.1f}%) {bar} [cum: {cumulative:.1f}%]")

# Key thresholds
print("\n" + "-"*60)
print("SAMPLES EXCEEDING KEY THRESHOLDS")
print("-"*60)

thresholds = [128, 256, 384, 512, 768, 1024]
for t in thresholds:
    count = sum(1 for l in token_lengths if l > t)
    pct = count / total * 100
    print(f"  > {t:4d} tokens: {count:7d} ({pct:5.1f}%)")

# Recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)

over_512 = sum(1 for l in token_lengths if l > 512)
over_768 = sum(1 for l in token_lengths if l > 768)
over_1024 = sum(1 for l in token_lengths if l > 1024)

print(f"\n  Samples > 512 tokens: {over_512} ({over_512/total*100:.1f}%)")
print(f"  Samples > 768 tokens: {over_768} ({over_768/total*100:.1f}%)")
print(f"  Samples > 1024 tokens: {over_1024} ({over_1024/total*100:.1f}%)")

if over_512 / total < 0.05:
    print(f"\n  ✅ max_length=512 is fine (only {over_512/total*100:.1f}% truncated)")
elif over_768 / total < 0.01:
    print(f"\n  💡 Consider max_length=768 (covers {(1-over_768/total)*100:.1f}% of data)")
else:
    print(f"\n  ⚠️  Consider max_length=1024 for better coverage")
