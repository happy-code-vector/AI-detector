# 350-Word Policy Implementation Guide

## Policy Change

**Old**: 50 words per sentence
**New**: 350 words per sentence
**Impact**: 7× increase in text length

## Impact on Processing Time

Processing time scales linearly with text length. With 7× longer text:

| Configuration | Old (50 words) | New (350 words) |
|--------------|-------------------|--------------------|
| Sequential PyTorch | ~60s | ~420s (7×) |
| Batched PyTorch | ~15-20s | ~105-140s (7×) |
| ONNX FP32 | ~5-8s | ~35-56s (7×) |
| **ONNX INT8** | ~2-4s | **~15-25s (7×)** |

## Recommendations for < 20 Second Target

### Option 1: ONNX INT8 (Minimum Changes)

**Processing time: ~15-25 seconds**

This is the minimum-change solution that might meet the target.

```bash
cd api
pip install onnxruntime onnx
python convert_to_onnx.py  # Select option 2 (INT8)
```

Update `shared_config.yaml`:
```yaml
model:
  checkpoint_dir: "api/models/model_onnx"
```

**Pros**:
- Minimal code changes
- Uses existing trained model
- 15-25s processing time

**Cons**:
- May still exceed 20s depending on server load
- Slight accuracy drop from INT8 quantization (~1%)

---

### Option 2: Switch to deberta-v3-base + ONNX INT8 (RECOMMENDED)

**Processing time: ~10-18 seconds** ✓

This is the recommended approach to reliably meet the < 20s target.

```yaml
# shared_config.yaml
model:
  name: "microsoft/deberta-v3-base"  # Smaller, faster model
```

Then convert with ONNX INT8 as shown in Option 1.

**Pros**:
- Reliably under 20s
- 2× faster inference (140M vs 304M params)
- Lower memory usage

**Cons**:
- Slight accuracy drop from smaller model (~1-2%)
- Need to retrain if accuracy is critical

---

### Option 3: GPU Inference (If Available)

**Processing time: ~10-20 seconds** ✓

If your server has NVIDIA GPUs, enable CUDA:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If True, CUDA will be used automatically
# No code changes needed
```

**Pros**:
- Fastest option
- No accuracy loss
- Uses existing model

**Cons**:
- Requires GPU hardware
- Higher cloud/operational costs

---

## Quick Decision Matrix

| Scenario | Recommended Option | Expected Time |
|-----------|-------------------|----------------|
| Minimal changes, accept INT8 accuracy loss | ONNX INT8 (large) | 15-25s |
| Need < 20s reliably, accept 1-2% accuracy loss | ONNX INT8 (base) | 10-18s ✓ |
| Need < 20s, no accuracy loss acceptable | GPU inference | 10-20s ✓ |
| Can retrain for optimal performance | Train deberta-v3-base | Same as ONNX INT8 (base) |

## Configuration Updates Applied

### Max Sequence Length Increased

```yaml
# shared_config.yaml - training section
training:
  max_length: 512  # Increased from 128 to support 350 words
```

**Rationale**: 350 words × ~1.3 tokens/word ≈ 455 tokens → rounded to 512

### Words Per Sentence Limit Increased

```yaml
# shared_config.yaml - data section
data:
  max_words_per_sentence: 350  # Increased from 50
```

## Testing

Test your configuration with 120 sentences at 350 words each:

```bash
# Start API
cd api
python main.py

# In another terminal, test with realistic data
curl -X POST http://localhost:8000/api/v1/predict/sentences \
  -H "Content-Type: application/json" \
  -d @test_350_words.json
```

Check `processing_time_ms` in response metadata. Target: `< 20000ms`.
