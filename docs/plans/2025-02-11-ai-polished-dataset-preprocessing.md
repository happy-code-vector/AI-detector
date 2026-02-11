# AI-Polished Dataset Preprocessing Design

**Date**: 2025-02-11
**Status**: Approved

## Overview

Convert the AI-polished dataset into the training format expected by the AI detection model. The dataset contains:
- Human-written texts (HWT_original_data.csv)
- AI-polished variants (55 different model/polish level combinations)
- Goal: Generate word-level labels (0=human, 1=AI) based on text differences

## Dataset Structure

```
datasets/AI-polished/
├── HWT_original_data.csv          # Human-written texts
├── polished/                       # AI-polished CSV files
│   ├── polished_texts_1_gpt.csv
│   ├── polished_texts_5_llama.csv
│   └── ... (55 variants total)
└── polished_json/                  # AI-polished JSON files
    ├── polished_texts_1_gpt.json
    ├── polished_texts_5_llama.json
    └── ...
```

## Training Format Requirements

The model expects a JSON file with:
```json
{
  "sentences": [
    {
      "text": "The quick brown fox.",
      "labels": [0, 0, 0, 0]
    }
  ]
}
```

Labels are word-level: 0 = human, 1 = AI.

## Preprocessing Pipeline

### 1. Human Baseline Loader
- Read `HWT_original_data.csv`
- Extract the `generation` column (actual text)
- Assign all words label `0`
- Include metadata: source="human", model="human", domain

### 2. AI-Polished Diff Processor
For each JSON file in `polished_json/`:

1. Parse JSON with `original` and `polished` text pairs
2. Perform word-level diffing
3. Assign label `1` to modified words, `0` to unchanged words
4. Include metadata: source="ai_polished", model, polish_level, domain

### 3. Word-Level Diffing Algorithm

**Tokenization**: Split both texts into words using `.split()`

**Character-to-word alignment**:
- Use `difflib.SequenceMatcher` to compare original vs polished at character level
- Map character changes to word positions
- Any word containing changed characters gets label `1`

**Label rules**:
- If 50%+ of a word's characters changed → label `1`
- If word is completely new → label `1`
- If word is unchanged → label `0`

**Example**:
```
Original:  "i am going to the"
Polished:  "I am going to the"
Labels:    [1, 0, 0, 0, 0, 0]   # Only "I" marked as AI
```

### 4. Output Format

```json
{
  "sentences": [
    {
      "text": "Another week just flew by me...",
      "labels": [0, 0, 0, 0, 0, 0, 1, 0, ...],
      "metadata": {
        "source": "ai_polished",
        "model": "gpt-4o",
        "polish_level": "polish_1",
        "domain": "blog",
        "polish_ratio": 0.01
      }
    }
  ],
  "stats": {
    "total_sentences": 56000,
    "human_only": 1000,
    "ai_polished": 55000,
    "avg_ai_label_ratio": 0.15
  }
}
```

### 5. Implementation Details

**Memory efficiency**: Process files incrementally, stream to output

**Edge cases**:
- Unicode normalization (e.g., `&nbsp;`, smart quotes)
- Empty text fields → skip
- JSON parse errors → log and continue
- Diff failures → fallback to full AI label

**Validation**:
- Human texts have `ai_label_ratio = 0`
- Higher polish levels have higher AI ratios
- Label counts match word counts

## Usage

```bash
python training/datasets/preprocess_ai_polished.py \
  --input datasets/AI-polished \
  --output training/data/custom/ai_polished_full.json \
  --include-human \
  --all-variants
```

## Files

- `training/datasets/preprocess_ai_polished.py` - Main preprocessing script
- `training/data/custom/ai_polished_full.json` - Output (all variants)
- `training/data/custom/ai_polished_sample.json` - Sample subset for testing
