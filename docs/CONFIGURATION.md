# Configuration Guide

## Overview

The AI Detector project uses a **unified configuration system** to ensure consistency between training and inference. All common settings are defined in a single shared configuration file.

## Configuration Structure

```
AI-detector/
├── shared_config.yaml          # Single source of truth
├── training/
│   ├── configs/                # Training-specific overrides only
│   └── shared_config.py        # Utilities to load shared config
└── api/
    ├── config.py               # Uses shared config by default
    └── shared_config.py        # Utilities to load shared config
```

## Shared Configuration (`shared_config.yaml`)

The `shared_config.yaml` file at project root contains:

### Section: `model`
- `name`: Base HuggingFace model name (e.g., `microsoft/deberta-v3-base`)
- `checkpoint_dir`: Where trained models are saved/loaded
- `device`: `auto`, `cuda`, or `cpu`

### Section: `data`
- `max_sentences`: Maximum sentences per request
- `max_words_per_sentence`: Text processing limit
- `train_split`, `eval_split`, `test_split`: Dataset split ratios

### Section: `training`
- `batch_size`, `learning_rate`, `epochs`
- `max_length`, `max_grad_norm`, `warmup_steps`
- LoRA settings: `use_peft`, `lora_r`, `lora_alpha`, `lora_dropout`
- Quantization: `load_in_8bit`, `load_in_4bit`

### Section: `api`
- `batch_size`: Inference batch size (can differ from training)
- `host`, `port`, `reload`: Server settings
- API metadata: `title`, `version`, `description`

## Training Configuration Files

Files in `training/configs/*.yaml` now only contain **overrides** to the shared config.

### Example: `stable.yaml`
```yaml
# Uses shared_config.yaml as base - only specify overrides
model: microsoft/deberta-v3-base
```

If no overrides are needed, the file can be empty or just have comments:
```yaml
# No overrides needed - uses all defaults from shared_config.yaml
```

### How Config Merging Works

When you run training:
1. **Shared config** is loaded first (base defaults)
2. **Config file** is loaded second (overrides)
3. **Environment variables** can override both (via `.env`)

Example:
```bash
python train.py --config configs/stable.yaml
```

This merges:
- `shared_config.yaml` (base) + `configs/stable.yaml` (overrides)

## API Configuration

The API automatically reads from `shared_config.yaml`. You can still override via:

### Environment Variables (`.env`)

```bash
# Override shared config defaults
MODEL_PATH=models/custom-checkpoint
BATCH_SIZE=16
PORT=9000
```

### Programmatic Overrides

```python
from config import settings

# Access values (already loaded from shared config)
model_path = settings.model_path
batch_size = settings.batch_size
```

## Benefits of Unified Configuration

1. **Single Source of Truth**: Model name, paths, and parameters defined once
2. **Consistency**: Training and API always use compatible settings
3. **Simplified Config Files**: Training configs only contain experiment-specific overrides
4. **Easy Updates**: Change model/data settings in one place
5. **No Duplication**: Eliminates mismatched parameters between components

## Common Tasks

### Change the Base Model

Edit `shared_config.yaml`:
```yaml
model:
  name: "microsoft/deberta-v3-large"  # Changed from base
```

Both training and API will now use `deberta-v3-large`.

### Change Training Batch Size Only

Edit `shared_config.yaml`:
```yaml
training:
  batch_size: 32
```

API will still use its own `api.batch_size` (default: 8).

### Use a Different Model for Specific Experiment

Create `training/configs/experiment.yaml`:
```yaml
model: google/flan-t5-base
learning_rate: 5e-5
```

Run:
```bash
python train.py --config configs/experiment.yaml
```

This overrides only the specified values; everything else comes from `shared_config.yaml`.

## Configuration Validation

The API validates configuration on startup:
- Checks if model checkpoint exists
- Logs warnings on mismatched settings
- Auto-detects CUDA availability

## Troubleshooting

### "Shared configuration not found"

Ensure `shared_config.yaml` exists at project root (same level as `training/` and `api/`).

### Model Not Found After Training

Check that:
1. `output_dir` in training config matches `checkpoint_dir` in shared config
2. Training completed successfully
3. Model files are in the expected location

### API Loading Wrong Model

1. Check `shared_config.yaml` → `model.name`
2. Check `.env` file for `MODEL_NAME` override
3. Verify checkpoint contains expected model files
