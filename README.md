# AI Text Detector

Word-level AI-generated text detection system using fine-tuned DeBERTa-v3 model.

## Configuration

This project uses a **unified configuration system**. All common settings are defined in `shared_config.yaml` at the project root. Training and API both read from this shared configuration, ensuring consistency.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for detailed configuration guide.

**Quick config changes:**
- Change model: Edit `model.name` in `shared_config.yaml`
- Training parameters: Edit `training.*` in `shared_config.yaml`
- Experiment overrides: Edit specific `training/configs/*.yaml` files

## Project Structure

```
ai-detector/
├── training/          # Model fine-tuning subproject
│   ├── src/
│   │   ├── data/     # Data loading utilities
│   │   ├── models/   # Model definition
│   │   ├── train.py  # Training script
│   │   └── utils.py  # Helper functions
│   ├── configs/      # Training configurations
│   ├── requirements.txt
│   └── outputs/      # Model checkpoints (gitignored)
│
├── api/              # FastAPI service subproject
│   ├── src/
│   │   ├── api/      # API routes
│   │   ├── models/   # Pydantic schemas
│   │   ├── services/ # Model service
│   │   ├── config.py # Configuration
│   │   └── main.py   # Application entry
│   ├── requirements.txt
│   └── .env.example
│
└── docs/
    └── plans/        # Design documents
```

## Quick Start

### 1. Setup Training Environment

```bash
cd training
pip install -r requirements.txt
```

### 2. Prepare Training Data

Create a sample dataset:

```bash
python src/train.py --create-sample
```

This creates `data/custom/sample.json`. Update `configs/default.yaml`:

```yaml
custom_data_path: data/custom/sample.json
```

Or provide your own dataset in JSON format:

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

### 3. Train the Model

```bash
python src/train.py --config configs/default.yaml
```

The model will be saved to `../api/models/checkpoint-best/`.

### 4. Run the API

```bash
cd api
pip install -r requirements.txt
cp .env.example .env  # Optional: customize settings
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Make Predictions

```bash
curl -X POST "http://localhost:8000/api/v1/predict/sentences" \
  -H "Content-Type: application/json" \
  -d '{
    "sentences": [
      "The quick brown fox jumps over the lazy dog",
      "Machine learning is transforming technology"
    ]
  }'
```

Response:

```json
{
  "probabilities": [
    [0.1, 0.2, 0.15, 0.8, 0.9, 0.3, 0.2, 0.1, 0.05],
    [0.95, 0.92, 0.3, 0.2, 0.85]
  ],
  "metadata": {
    "model_version": "checkpoint-best",
    "processing_time_ms": 150.5,
    "total_words": 14
  }
}
```

## API Endpoints

### POST /api/v1/predict/sentences
Predict AI-generated probability for each word in sentences.

**Request:**
```json
{
  "sentences": ["text to analyze", "another sentence"]
}
```

**Constraints:**
- Max 120 sentences per request
- Max 50 words per sentence

### POST /api/v1/predict/batch
Submit batch job for async processing.

### GET /api/v1/jobs/{job_id}
Get batch job status and results.

### GET /api/v1/health
Health check endpoint.

## Configuration

### Training Configuration (`training/configs/default.yaml`)

```yaml
model: microsoft/deberta-v3-base
epochs: 3
batch_size: 16
learning_rate: 2e-5
max_length: 128
output_dir: ../api/models/checkpoint-best
```

### API Configuration (`api/.env`)

```
MODEL_PATH=models/checkpoint-best
DEVICE=auto
MAX_SENTENCES=120
MAX_WORDS_PER_SENTENCE=50
```

## Dependencies

### Training
- transformers>=4.30.0
- torch>=2.0.0
- datasets>=2.12.0
- scikit-learn

### API
- fastapi>=0.100.0
- uvicorn[standard]>=0.22.0
- pydantic>=2.0
- torch>=2.0.0

## License

MIT
