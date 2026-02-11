"""
ONNX Runtime model service for optimized CPU inference.

Provides 2-4x speedup on CPU compared to PyTorch.

Usage:
    1. Convert model: python convert_to_onnx.py
    2. Set in config: model.checkpoint_dir = "api/models/model_onnx"
    3. API will automatically use ONNX Runtime
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
from transformers import AutoTokenizer

from config import get_device
from shared_config import get_api_config, get_model_config

# ONNX Runtime imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not installed. Falling back to PyTorch.")
    print("   Install with: pip install onnxruntime")


# Global regex for word splitting (compiled once for performance)
_WORD_PATTERN = re.compile(r'\S+')


def _align_to_words_optimized_onnx(
    text: str,
    token_probs: np.ndarray,
    offset_mapping: np.ndarray,
) -> List[float]:
    """
    Optimized word alignment using pre-computed positions (ONNX version).

    Args:
        text: Original input text
        token_probs: Token-level probabilities (numpy array)
        offset_mapping: Token character offsets (numpy array)

    Returns:
        Word-level probabilities
    """
    # Find all word positions in one pass using regex
    word_boundaries = []  # List of (start, end) tuples

    for match in _WORD_PATTERN.finditer(text):
        word_boundaries.append((match.start(), match.end()))

    if not word_boundaries:
        return []

    # Build token intervals from offset_mapping
    token_intervals = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:  # Skip special tokens
            continue
        if i < len(token_probs):
            token_intervals.append((start, end, float(token_probs[i])))

    word_probs = []

    # For each word, find all tokens that fall within its boundaries
    for word_start, word_end in word_boundaries:
        matching_probs = []

        for token_start, token_end, prob in token_intervals:
            # Token belongs to this word if it's fully contained within word boundaries
            if token_start >= word_start and token_end <= word_end:
                matching_probs.append(prob)

        # Average probabilities for the word
        if matching_probs:
            avg_prob = sum(matching_probs) / len(matching_probs)
            word_probs.append(round(avg_prob, 4))
        else:
            # No tokens found for this word (shouldn't happen with proper tokenization)
            word_probs.append(0.0)

    return word_probs


class ONNXModelService:
    """
    ONNX Runtime-based model service for optimized CPU inference.

    Key optimizations:
    - Uses ONNX Runtime (2-4x faster than PyTorch on CPU)
    - Dynamic batching (processes all sentences at once)
    - Thread control (OMP_NUM_THREADS)
    - Memory-efficient tokenization
    - Optimized word alignment with regex
    - Parallel word alignment processing
    """

    _instance: Optional["ONNXModelService"] = None
    _session = None
    _tokenizer = None
    _device = None
    _model_version = None
    _inference_batch_size = None
    _alignment_workers = None  # Number of parallel workers for word alignment

    def __new__(cls) -> "ONNXModelService":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize ONNX model service (lazy loading)."""
        # Set optimal thread count for CPU inference from config
        # Use configured number of threads or default to CPU count
        if not os.environ.get("OMP_NUM_THREADS"):
            model_config = get_model_config()
            num_threads = model_config.get("omp_num_threads", os.cpu_count())
            os.environ["OMP_NUM_THREADS"] = str(num_threads)
            print(f"Set OMP_NUM_THREADS={num_threads} for optimal CPU performance")

        # Get number of workers for parallel word alignment
        if self._alignment_workers is None:
            api_config = get_api_config()
            # Use configured threads or default to CPU count - 1
            self._alignment_workers = api_config.get("omp_num_threads", os.cpu_count() or 4) - 1
            if self._alignment_workers < 1:
                self._alignment_workers = 1

    def load_model(self) -> None:
        """Load ONNX model and tokenizer."""
        if self._session is not None:
            return  # Already loaded

        from config import settings

        model_path = Path(settings.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {settings.model_path}. "
                "Please train model first or update model_path in config."
            )

        try:
            print(f"Loading ONNX model from: {model_path}")

            # Check for quantized model
            onnx_path = model_path / "model_int8.onnx"
            if not onnx_path.exists():
                onnx_path = model_path / "model.onnx"

            if not onnx_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {onnx_path}. "
                    "Run convert_to_onnx.py first."
                )

            # Configure ONNX Runtime for CPU optimization
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            # Enable all CPU optimizations
            available_providers = ort.get_available_providers()
            print(f"Available ONNX providers: {available_providers}")

            # Use CPU provider (optimized for x86_64)
            providers = ["CPUExecutionProvider"]

            # Create inference session
            self._session = ort.InferenceSession(
                str(onnx_path),
                sess_options=so,
                providers=providers,
            )

            # Load tokenizer from same directory
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            # Extract version from path
            self._model_version = model_path.name

            # Get input/output names from model
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]

            print(f"Model inputs: {self._input_names}")
            print(f"Model outputs: {self._output_names}")
            print(f"✓ ONNX model loaded successfully")

        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._session is not None

    @property
    def model_version(self) -> Optional[str]:
        """Get model version."""
        return self._model_version

    @property
    def device(self) -> str:
        """Get device being used."""
        return "cpu"  # ONNX Runtime on CPU

    def predict_batch(self, sentences: List[str]) -> List[List[float]]:
        """
        Predict AI probability for multiple sentences using ONNX Runtime.

        Optimized for CPU with dynamic batching and efficient padding.

        Args:
            sentences: List of input sentences

        Returns:
            2D list of word-level AI probabilities
        """
        if not self.is_loaded:
            self.load_model()

        if not sentences:
            return []

        # Get inference batch size from config
        api_config = get_api_config()
        batch_size = api_config.get("inference_batch_size", 32)

        # Filter out empty sentences
        non_empty_sentences = [(i, s) for i, s in enumerate(sentences) if s.strip()]

        if not non_empty_sentences:
            return [[] for _ in sentences]

        try:
            # Process in chunks to optimize memory
            results = [None] * len(sentences)

            for chunk_start in range(0, len(non_empty_sentences), batch_size):
                chunk = non_empty_sentences[chunk_start : chunk_start + batch_size]
                chunk_results = self._process_batch_chunk(chunk)
                for orig_idx, word_probs in chunk_results:
                    results[orig_idx] = word_probs

            # Fill empty sentences
            for i in range(len(sentences)):
                if results[i] is None:
                    results[i] = []

            return results

        except Exception as e:
            raise RuntimeError(f"ONNX batch prediction failed: {e}") from e

    def _process_batch_chunk(
        self, indexed_sentences: List[tuple]
    ) -> List[tuple]:
        """
        Process a batch of sentences through ONNX model with parallel word alignment.

        Args:
            indexed_sentences: List of (original_index, sentence) tuples

        Returns:
            List of (original_index, word_probabilities) tuples
        """
        texts = [s for _, s in indexed_sentences]

        # Tokenize with DYNAMIC padding (only pad to max in batch)
        encodings = self._tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            padding="longest",  # Dynamic padding!
            return_offsets_mapping=True,
        )

        # Prepare inputs for ONNX
        onnx_inputs = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
        }

        # Run inference
        start_time = time.time()
        logits = self._session.run(
            self._output_names,
            onnx_inputs,
        )[0]  # [batch_size, seq_len, num_labels]
        inference_time = (time.time() - start_time) * 1000

        if len(indexed_sentences) > 10:  # Only log for larger batches
            print(f"ONNX inference: {len(texts)} sentences in {inference_time:.1f}ms")

        # Get AI probabilities (class 1)
        # Use softmax along last dimension
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        ai_probs = probs[:, :, 1]  # [batch_size, seq_len]

        # Prepare arguments for parallel word alignment
        alignment_args = []
        for batch_idx, (orig_idx, sentence) in enumerate(indexed_sentences):
            offset_mapping = encodings["offset_mapping"][batch_idx]
            token_probs = ai_probs[batch_idx]
            alignment_args.append((sentence, token_probs, offset_mapping, orig_idx))

        # Use parallel word alignment for better performance
        results = []

        # Only use parallel processing if we have enough sentences
        if len(alignment_args) > 4:
            with ThreadPoolExecutor(max_workers=min(self._alignment_workers, len(alignment_args))) as executor:
                aligned_results = list(executor.map(
                    lambda args: _align_to_words_optimized_onnx(args[0], args[1], args[2]),
                    alignment_args
                ))

            # Pair results with original indices
            for idx, (_, _, _, orig_idx) in enumerate(alignment_args):
                results.append((orig_idx, aligned_results[idx]))
        else:
            # Sequential processing for small batches
            for sentence, token_probs, offset_mapping, orig_idx in alignment_args:
                word_probs = _align_to_words_optimized_onnx(sentence, token_probs, offset_mapping)
                results.append((orig_idx, word_probs))

        return results


# Global ONNX service instance
onnx_model_service = ONNXModelService()


def get_onnx_model_service() -> ONNXModelService:
    """Get global ONNX model service instance."""
    return onnx_model_service
