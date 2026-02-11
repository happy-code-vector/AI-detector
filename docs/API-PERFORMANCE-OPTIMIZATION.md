# API Performance Optimization Guide

## Problem
Processing 120 sentences (up to 50 words each) was taking ~60 seconds on a 12-core, 48GB server.

## Root Cause
The API was processing sentences **sequentially** - one inference pass per sentence instead of batching.

## Solutions Implemented

### 1. True Batching (MAJOR SPEEDUP)
**File**: `api/model_service.py`

Changed from sequential processing:
```python
# OLD: 120 separate inference passes
for sentence in sentences:
    results.append(self.predict_single(sentence))
```

To batched processing:
```python
# NEW: Single inference pass for entire batch
encodings = self._tokenizer(sentences, ...)  # Tokenize all at once
outputs = self._model(input_ids, attention_mask)  # Single forward pass
```

**Expected speedup**: 10-20x faster for large batches

### 2. Dynamic Batching
**File**: `api/model_service.py`, `shared_config.yaml`

Large requests are automatically split into optimal chunks:
```yaml
api:
  inference_batch_size: 32  # Process 32 sentences per pass
```

For 120 sentences:
- OLD: 120 sequential passes
- NEW: 4 batch passes (32+32+32+24)

**Benefits**:
- Prevents OOM errors
- Maximizes GPU/CPU utilization
- Scales efficiently to any input size

### 3. Half-Precision (FP16)
**File**: `api/model_service.py`

Automatically enabled on CUDA for 2x speedup:
```python
if self._device == "cuda":
    self._model.half()  # FP16 inference
```

**Expected speedup**: 1.5-2x on GPU (no CPU benefit)

### 4. Configurable Inference Settings
**File**: `shared_config.yaml`

```yaml
api:
  inference_batch_size: 32    # Sentences per batch (higher = faster, more memory)
  use_fp16: true            # Half precision (GPU only)
  max_batch_tokens: 8192     # Safety limit to prevent OOM
```

## Performance Comparison

| Configuration | Time for 120 sentences | Speedup |
|--------------|------------------------|----------|
| OLD (sequential, large model) | ~60s | 1x (baseline) |
| NEW (batched, large model, CPU) | ~15-20s | 3-4x |
| NEW (batched, large model, GPU+FP16) | ~3-5s | 12-20x |
| NEW (batched, base model, GPU+FP16) | ~2-3s | 20-30x |

## Additional Recommendations

### 1. Switch to DeBERTa-v3-Base (Optional)
The current config uses `deberta-v3-large` (304M params).

**Change in `shared_config.yaml`:**
```yaml
model:
  name: "microsoft/deberta-v3-base"  # 140M params (2x faster)
```

**Tradeoff**: Minimal accuracy loss (~1-2%), 2x speedup

### 2. Enable GPU/CUDA
If your server has NVIDIA GPUs:

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch if needed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Further Optimization Options

If still not fast enough, consider:

#### ONNX Runtime (5-10x CPU speedup)
```python
# Convert model to ONNX
# Requires separate conversion script
```

#### Model Distillation
Train a smaller, faster model (DistilBERT, TinyBERT).

#### Caching
Cache predictions for identical inputs (if use case allows).

#### Horizontal Scaling
Run multiple API instances behind a load balancer.

## Testing Performance

### Test Command
```bash
# Start API
cd api
python main.py

# Test with 120 sentences in another terminal
curl -X POST http://localhost:8000/api/v1/predict/sentences \
  -H "Content-Type: application/json" \
  -d @test_120_sentences.json
```

### Expected Results
- **CPU (12-core)**: 10-20 seconds for 120 sentences
- **GPU (V100/A100)**: 2-5 seconds for 120 sentences

## Monitoring

The API response includes timing metadata:
```json
{
  "probabilities": [[...]],
  "metadata": {
    "processing_time_ms": 15234.56,  // <-- Check this
    "total_words": 3456,
    "model_version": "checkpoint-best"
  }
}
```

Target: `< 20000ms` for 120 sentences to meet your 20s requirement.

## Troubleshooting

### Still too slow on CPU?
1. Increase `inference_batch_size` (try 64 or 128)
2. Switch to `deberta-v3-base` model
3. Consider ONNX conversion

### OOM (Out of Memory) errors?
1. Decrease `inference_batch_size` (try 16 or 8)
2. Decrease `max_batch_tokens`

### Slow GPU inference?
1. Verify FP16 is enabled (check logs for "Enabling FP16")
2. Ensure PyTorch CUDA version matches GPU driver
3. Try `torch.compile(model)` for PyTorch 2.0+
