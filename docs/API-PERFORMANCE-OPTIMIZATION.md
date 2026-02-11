# API Performance Optimization Guide

## Quick Start for 12-Core CPU Server

**Updated policy: 350 words per sentence (7x increase from 50)**

To achieve **< 20 seconds** for 120 sentences (350 words each):

```bash
# 1. Install ONNX Runtime
cd api
pip install onnxruntime onnx

# 2. Convert model to ONNX
python convert_to_onnx.py
# Select option 2 (INT8 quantization) for best performance

# 3. Update config
# Edit shared_config.yaml:
#   model.checkpoint_dir = "api/models/model_onnx"

# 4. Set thread optimization
export OMP_NUM_THREADS=12

# 5. Start API
python main.py
```

**Expected result:**
- **With ONNX FP32: ~35-56 seconds** (may exceed 20s limit)
- **With ONNX INT8: ~15-25 seconds** ✓ (within 20s limit)

**IMPORTANT: The 7× increase in text length means processing time increases proportionally.**

---

## Problem
Processing 120 sentences (up to 350 words each) on a 12-core, 48GB server with original sequential implementation would take ~420 seconds (7× 60s).

## Root Cause
The API was processing sentences **sequentially** - one inference pass per sentence instead of batching.

## Solutions Implemented

### 0. ONNX Runtime (RECOMMENDED FOR CPU)
**Files**: `api/convert_to_onnx.py`, `api/model_service_onnx.py`, `api/service_selector.py`

**The most impactful optimization for CPU inference.**

ONNX Runtime is specifically optimized for CPU performance:
- 2-4x faster than PyTorch on CPU
- Optimized matrix operations (Intel MKL, OpenBLAS)
- Better thread parallelization

#### Setup Instructions

1. **Convert model to ONNX:**
```bash
cd api
python convert_to_onnx.py
```

Select option 1 (FP32) for best accuracy, or option 2 (INT8) for maximum speed.

2. **Update config to use ONNX:**
```yaml
# shared_config.yaml
model:
  checkpoint_dir: "api/models/model_onnx"  # Changed from checkpoint-best
```

3. **Install dependencies:**
```bash
pip install -r api/requirements.txt  # Includes onnxruntime
```

4. **Restart API - it will auto-detect and use ONNX:**
```bash
python main.py
# Look for: "🚀 Using ONNX Runtime for optimized CPU inference"
```

**Expected Performance with ONNX (12-core CPU):**

| Sentence Length | FP32 ONNX | INT8 ONNX |
|--------------|-------------|-------------|
| 50 words (old) | 5-8 seconds | 2-4 seconds |
| 350 words (new) | 35-56 seconds | 15-25 seconds |

**For 350-word policy, INT8 quantization is recommended to meet < 20s target.**

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

### For 50 words per sentence (old policy):

| Configuration | Time for 120 sentences | Speedup |
|--------------|------------------------|----------|
| OLD (sequential, deberta-v3-large, CPU) | ~60s | 1x (baseline) |
| NEW (batched, deberta-v3-large, PyTorch CPU) | ~15-20s | 3-4x |
| **ONNX Runtime FP32 (deberta-v3-large, CPU)** | **~5-8s** | **8-12x** |
| **ONNX Runtime INT8 (deberta-v3-large, CPU)** | **~2-4s** | **15-30x** |

### For 350 words per sentence (new policy - 7× increase):

| Configuration | Time for 120 sentences | Within 20s limit? |
|--------------|------------------------|-------------------|
| OLD (sequential, deberta-v3-large, CPU) | ~420s (7× 60s) | ❌ No |
| NEW (batched, deberta-v3-large, PyTorch CPU) | ~105-140s | ❌ No |
| **ONNX Runtime FP32 (deberta-v3-large, CPU)** | **~35-56s** | **❌ No** |
| **ONNX Runtime INT8 (deberta-v3-large, CPU)** | **~15-25s** | **✓ Sometimes** |
| **Switch to deberta-v3-base + ONNX INT8** | **~10-18s** | **✓ Yes** |

**Recommended for 350-word policy:**
1. Use **ONNX INT8** quantization (required for < 20s)
2. Consider **smaller model** (deberta-v3-base) if still too slow

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

If ONNX Runtime is still not fast enough:

#### Switch to DeBERTa-v3-Base
Smaller model (140M vs 304M params) with minimal accuracy loss.

#### Model Distillation
Train a smaller, faster model (DistilBERT, TinyBERT).
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

For **50 words per sentence** (old policy):
- CPU (12-core) PyTorch: 15-20 seconds for 120 sentences
- CPU (12-core) ONNX FP32: 5-8 seconds for 120 sentences ✓
- CPU (12-core) ONNX INT8: 2-4 seconds for 120 sentences ✓✓
- GPU (V100/A100): 2-5 seconds for 120 sentences

For **350 words per sentence** (new policy - 7× longer):
- CPU (12-core) PyTorch: 105-140 seconds ❌
- CPU (12-core) ONNX FP32: 35-56 seconds ❌
- CPU (12-core) ONNX INT8: 15-25 seconds ✓
- GPU (V100/A100): 10-20 seconds

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

### 350-word policy causing > 20s processing time?

With 7× longer text (350 vs 50 words), processing time increases proportionally.

**To meet < 20s target with 350-word sentences:**

1. **Use ONNX INT8** (required - FP32 will be ~35-56s):
   ```bash
   cd api
   python convert_to_onnx.py
   # Select option 2 (INT8 quantization)
   ```

2. **Switch to smaller model** (deberta-v3-base):
   ```yaml
   # shared_config.yaml
   model:
     name: "microsoft/deberta-v3-base"  # 2× faster
   ```
   This brings ONNX INT8 to ~10-18s ✓

3. **Consider GPU** if available:
   GPU inference: ~10-20s with deberta-v3-large

4. **Reduce request size** - process fewer sentences per batch if possible

### Still too slow on CPU (with 50-word policy)?

### ONNX not loading?
1. Check logs for "ONNX model not found"
2. Ensure `convert_to_onnx.py` completed successfully
3. Verify `checkpoint_dir` in shared_config.yaml points to ONNX model
4. Check onnxruntime is installed: `pip list | grep onnx`
5. **For 350-word policy**: Ensure INT8 quantization was selected during conversion

### OOM (Out of Memory) errors?
1. Decrease `inference_batch_size` (try 16 or 8)
2. Decrease `max_batch_tokens`

### Slow GPU inference?
1. Verify FP16 is enabled (check logs for "Enabling FP16")
2. Ensure PyTorch CUDA version matches GPU driver
3. Try `torch.compile(model)` for PyTorch 2.0+
