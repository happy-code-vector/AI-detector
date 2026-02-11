"""
ONNX Runtime model service for optimized CPU inference.

Provides 2-4x speedup on CPU compared to PyTorch.

Usage:
    1. Convert model: python convert_to_onnx.py
    2. Set in config: model.checkpoint_dir = "api/models/model_onnx"
    3. API will automatically use ONNX Runtime
"""

import os
import time
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


class ONNXModelService:
    """
    ONNX Runtime-based model service for optimized CPU inference.

    Key optimizations:
    - Uses ONNX Runtime (2-4x faster than PyTorch on CPU)
    - Dynamic batching (processes all sentences at once)
    - Thread control (OMP_NUM_THREADS)
    - Memory-efficient tokenization
    """

    _instance: Optional["ONNXModelService"] = None
    _session = None
    _tokenizer = None
    _device = None
    _model_version = None
    _inference_batch_size = None

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
        Process a batch of sentences through ONNX model.

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

        # Align to words
        results = []
        for batch_idx, (orig_idx, sentence) in enumerate(indexed_sentences):
            offset_mapping = encodings["offset_mapping"][batch_idx]
            token_probs = ai_probs[batch_idx]
            word_probs = self._align_to_words(sentence, token_probs, offset_mapping)
            results.append((orig_idx, word_probs))

        return results

    def _align_to_words(
        self,
        text: str,
        token_probs: np.ndarray,
        offset_mapping: np.ndarray,
    ) -> List[float]:
        """
        Align token-level probabilities to words.

        Args:
            text: Original input text
            token_probs: Token-level probabilities
            offset_mapping: Token character offsets

        Returns:
            Word-level probabilities
        """
        words = text.split()
        word_probs = []
        text_lower = text.lower()

        for word in words:
            # Find word position in text
            word_start = text_lower.find(word.lower())
            if word_start == -1:
                word_probs.append(0.0)
                continue

            word_end = word_start + len(word)

            # Find tokens for this word
            token_values = []
            for i, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:  # Special token
                    continue
                if start >= word_start and end <= word_end:
                    if i < len(token_probs):
                        token_values.append(token_probs[i])

            # Average probabilities for word
            if token_values:
                avg_prob = sum(token_values) / len(token_values)
                word_probs.append(round(float(avg_prob), 4))
            else:
                word_probs.append(0.0)

        return word_probs


# Global ONNX service instance
onnx_model_service = ONNXModelService()


def get_onnx_model_service() -> ONNXModelService:
    """Get global ONNX model service instance."""
    return onnx_model_service
