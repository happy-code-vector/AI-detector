# AI Text Detector Design

**Date:** 2025-02-10
**Author:** Claude Sonnet 4.5

## Overview

Word-level AI-generated text detection system with two subprojects:
1. **Training** - Fine-tune DeBERTa-v3 for token-level classification
2. **API** - FastAPI service for inference on sentence arrays

## Requirements

**Input:** Array of sentences (up to 120, each up to 50 words)
**Output:** 2D array of probabilities (AI-generated likelihood per word)
**Processing:** One sentence at a time, async/batch support

## Project Structure

```
ai-detector/
├── training/                 # Subproject 1: Model fine-tuning
│   ├── src/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   └── loader.py     # Dataset loading (public + custom)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── model.py      # DeBERTa-v3 token classification
│   │   ├── __init__.py
│   │   ├── train.py          # Training script
│   │   └── utils.py          # Helpers
│   ├── data/
│   │   ├── public/           # Public datasets
│   │   └── custom/           # Custom datasets (JSON/CSV)
│   ├── configs/
│   │   └── default.yaml      # Training config
│   ├── tests/
│   │   └── ...
│   ├── requirements.txt
│   └── outputs/              # Checkpoints (gitignored)
│
├── api/                      # Subproject 2: FastAPI service
│   ├── src/
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes.py     # API endpoints
│   │   │   └── dependencies.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py    # Pydantic models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── model_service.py  # Model loading & inference
│   │   ├── config.py         # Configuration
│   │   ├── main.py           # FastAPI app entry
│   │   └── __init__.py
│   ├── models/               # Saved model files (gitignored)
│   ├── tests/
│   │   └── ...
│   └── requirements.txt
│
└── shared/                   # Shared utilities
    └── src/
        └── __init__.py
```

## Training Subproject

### Model

- **Base:** `microsoft/deberta-v3-base`
- **Task:** Token classification (binary: human=0, AI=1)
- **Framework:** Hugging Face Transformers + PyTorch

### Data Loading

Supports two data sources:

1. **Public datasets** (Hugging Face):
   - Auto-load with `datasets.load_dataset()`
   - Format: text with token/word-level labels

2. **Custom datasets** (JSON/CSV):
   ```json
   {
     "sentences": [
       {
         "text": "The quick brown fox jumps.",
         "labels": [0, 0, 0, 0, 1]
       }
     ]
   }
   ```

### Training Pipeline

```
1. Load dataset (public + custom)
2. Tokenize with DeBERTa tokenizer
3. Align subword tokens to word labels
4. Fine-tune with Trainer API
5. Evaluate (token-level F1, accuracy)
6. Save best checkpoint
```

### Configuration

```yaml
model: microsoft/deberta-v3-base
epochs: 3
batch_size: 16
learning_rate: 2e-5
max_length: 128
warmup_steps: 500
output_dir: outputs/checkpoint-best
```

## API Subproject

### Architecture

```
ModelService (singleton)
  ├─ Lazy load model on first request
  ├─ predict_single(sentence) -> List[float]
  └─ predict_batch(sentences) -> List[List[float]]

FastAPI Routes
  ├─ POST /predict/sentences  # Main endpoint
  ├─ POST /predict/batch      # Async background
  ├─ GET /health              # Health check
  └─ GET /jobs/{id}           # Batch job status
```

### Endpoints

#### POST /predict/sentences

**Request:**
```json
{
  "sentences": [
    "The quick brown fox",
    "jumps over the lazy dog"
  ]
}
```

**Response:**
```json
{
  "probabilities": [
    [0.1, 0.2, 0.15, 0.8],  // Sentence 1 word probabilities
    [0.9, 0.3, 0.2, 0.1, 0.05]  // Sentence 2
  ],
  "metadata": {
    "model_version": "deberta-v3-base-v1.0",
    "processing_time_ms": 150
  }
}
```

#### POST /predict/batch

Async processing for large arrays. Returns job ID.

### Validation

- Max 120 sentences per request
- Max 50 words per sentence
- Non-empty strings required

### Performance

- Lazy model loading (singleton)
- Async batch processing with `asyncio.gather()`
- Auto GPU/CPU detection
- Configurable batch size for parallel processing

## Dependencies

### Training
```
transformers>=4.30.0
torch>=2.0.0
datasets>=2.12.0
scikit-learn
pandas
tqdm
numpy
pyyaml
wandb  # optional
```

### API
```
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
pydantic>=2.0
torch>=2.0.0
transformers>=4.30.0
python-multipart
pytest
httpx
```

## Error Handling

| Error | Status | Description |
|-------|--------|-------------|
| ValidationError | 422 | Invalid input format |
| ModelNotLoadedError | 503 | Model failed to load |
| SentenceTooLongError | 400 | >50 words |
| TooManySentencesError | 400 | >120 sentences |
| InternalServerError | 500 | Inference failure |

## Deployment

1. **Train model:**
   ```bash
   cd training
   python src/train.py --config configs/default.yaml
   ```

2. **Copy model to API:**
   ```bash
   cp -r training/outputs/checkpoint-best api/models/
   ```

3. **Run API:**
   ```bash
   cd api
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Testing Strategy

- **Training tests:** Mock data loading, token alignment
- **API tests:** Endpoint testing with mock model service
- **Integration tests:** Full pipeline with small model
