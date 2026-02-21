"""Model service for AI text detection inference."""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import get_device, settings
from shared_config import get_api_config

# PEFT support (optional - for loading models trained with LoRA)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# Global regex for word splitting (compiled once for performance)
_WORD_PATTERN = re.compile(r'\S+')
_SPACE_PATTERN = re.compile(r'\s+')


def _align_to_words_optimized(
    text: str,
    token_probs: List[float],
    offset_mapping,
) -> List[float]:
    """
    Optimized word alignment using pre-computed positions.

    Args:
        text: Original input text
        token_probs: Token-level probabilities
        offset_mapping: Token character offsets

    Returns:
        Word-level probabilities
    """
    # Find all word positions in one pass using regex
    words = []
    word_boundaries = []  # List of (start, end) tuples

    for match in _WORD_PATTERN.finditer(text):
        words.append(match.group())
        word_boundaries.append((match.start(), match.end()))

    if not words:
        return []

    # Build token -> probability mapping for O(1) lookup
    # Create intervals from offset_mapping for efficient matching
    token_intervals = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == 0 and end == 0:  # Skip special tokens
            continue
        if i < len(token_probs):
            token_intervals.append((start, end, token_probs[i]))

    word_probs = []

    # For each word, find all tokens that fall within its boundaries
    for word_start, word_end in word_boundaries:
        matching_probs = []

        # Binary search would be ideal, but linear scan is fast enough
        # since we have ~300-500 tokens per sentence
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


def _align_words_parallel(args: Tuple[str, List[float], object]) -> List[float]:
    """
    Worker function for parallel word alignment.

    Args:
        args: Tuple of (text, token_probs, offset_mapping)

    Returns:
        Word-level probabilities
    """
    text, token_probs, offset_mapping = args
    return _align_to_words_optimized(text, token_probs, offset_mapping)


class ModelService:
    """Singleton service for model loading and inference."""

    _instance: Optional["ModelService"] = None
    _model = None
    _tokenizer = None
    _device = None
    _model_version = None
    _inference_batch_size = None
    _use_fp16 = None
    _alignment_workers = None  # Number of parallel workers for word alignment

    def __new__(cls) -> "ModelService":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model service (lazy loading)."""
        if self._device is None:
            self._device = get_device()

        # Get number of workers for parallel word alignment
        if self._alignment_workers is None:
            import os
            # Default to CPU count - 1 for parallel processing
            self._alignment_workers = max(1, (os.cpu_count() or 4) - 1)

    def load_model(self) -> None:
        """Load the fine-tuned model and tokenizer."""
        if self._model is not None:
            return  # Already loaded

        model_path = Path(settings.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {settings.model_path}. "
                "Please train the model first or update model_path in config."
            )

        try:
            print(f"Loading model from: {model_path}")

            # Check if this is a PEFT model (has adapter_config.json)
            adapter_config_path = model_path / "adapter_config.json"
            is_peft_model = adapter_config_path.exists()

            if is_peft_model:
                if not PEFT_AVAILABLE:
                    raise ImportError(
                        "Model was trained with PEFT/LoRA but peft is not installed. "
                        "Install with: pip install peft"
                    )
                print("🔧 Detected PEFT/LoRA model - loading with adapters...")

                # Load base model first
                base_model_name = settings.model_name  # or read from adapter_config
                base_model = AutoModelForTokenClassification.from_pretrained(
                    base_model_name,
                    num_labels=2,
                )

                # Load PEFT adapters
                self._model = PeftModel.from_pretrained(base_model, str(model_path))

                # Merge adapters for faster inference (optional but recommended)
                print("Merging LoRA adapters...")
                self._model = self._model.merge_and_unload()
            else:
                # Regular model loading
                self._model = AutoModelForTokenClassification.from_pretrained(
                    str(model_path),
                )

            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            self._model.to(self._device)
            self._model.eval()

            # Use BF16 for faster inference (if CUDA available and supported)
            if self._device == "cuda" and torch.cuda.is_bf16_supported():
                print("Enabling BF16 for faster inference...")
                self._model = self._model.to(torch.bfloat16)

            # Extract version from path or config
            self._model_version = model_path.name

            print(f"Model loaded successfully on {self._device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def model_version(self) -> Optional[str]:
        """Get model version."""
        return self._model_version

    @property
    def device(self) -> str:
        """Get device being used."""
        return self._device

    def predict_single(self, text: str) -> List[float]:
        """
        Predict AI probability for each word in a sentence.

        Args:
            text: Input sentence

        Returns:
            List of word-level AI probabilities
        """
        if not self.is_loaded:
            self.load_model()

        try:
            with torch.no_grad():
                # Tokenize
                encoding = self._tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    return_offsets_mapping=True,
                )

                input_ids = encoding["input_ids"].to(self._device)
                offset_mapping = encoding["offset_mapping"][0]

                # Get predictions
                outputs = self._model(input_ids)
                logits = outputs.logits[0]  # [seq_len, num_labels]

                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                ai_probs = probs[:, 1].cpu().tolist()  # AI probability

                # Align to words using optimized method
                word_probs = _align_to_words_optimized(text, ai_probs, offset_mapping)

            return word_probs

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}") from e

    def predict_batch(self, sentences: List[str]) -> List[List[float]]:
        """
        Predict AI probability for multiple sentences.

        Uses dynamic batching - splits large requests into optimal chunks
        to maximize throughput while avoiding OOM errors.

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

        # Process in chunks to avoid OOM and maximize throughput
        results = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            results.extend(self._process_batch(batch))

        return results

    def _process_batch(self, sentences: List[str]) -> List[List[float]]:
        """Process a single batch of sentences with parallel word alignment."""
        # Filter out empty sentences
        indexed_sentences = [(i, s) for i, s in enumerate(sentences) if s.strip()]

        if not indexed_sentences:
            return [[] for _ in sentences]

        try:
            with torch.no_grad():
                # Tokenize all sentences at once
                texts = [s for _, s in indexed_sentences]
                encodings = self._tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    return_offsets_mapping=True,
                )

                input_ids = encodings["input_ids"].to(self._device)
                attention_mask = encodings["attention_mask"].to(self._device)
                offset_mappings = encodings["offset_mapping"]

                # Batch inference - single forward pass
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [batch_size, seq_len, num_labels]

                # Get probabilities for AI class (index 1)
                probs = torch.softmax(logits, dim=-1)[:, :, 1]  # [batch_size, seq_len]

                # Move to CPU for processing
                probs = probs.cpu().tolist()
                offset_mappings = offset_mappings.tolist()

                # Use parallel word alignment for better performance
                # Prepare arguments for parallel processing
                alignment_args = []
                for batch_idx, (orig_idx, sentence) in enumerate(indexed_sentences):
                    offset_mapping = offset_mappings[batch_idx]
                    token_probs = probs[batch_idx]
                    alignment_args.append((sentence, token_probs, offset_mapping, orig_idx))

                # Parallel word alignment using ThreadPoolExecutor
                # Using ThreadPool instead of ProcessPool to avoid pickling overhead
                # and because this is CPU-bound with GIL-released operations
                results = [None] * len(sentences)

                # Only use parallel processing if we have enough sentences
                if len(alignment_args) > 4:
                    with ThreadPoolExecutor(max_workers=min(self._alignment_workers, len(alignment_args))) as executor:
                        aligned_results = list(executor.map(
                            lambda args: _align_to_words_optimized(args[0], args[1], args[2]),
                            alignment_args
                        ))

                    # Place results in correct positions
                    for idx, (_, _, _, orig_idx) in enumerate(alignment_args):
                        results[orig_idx] = aligned_results[idx]
                else:
                    # Sequential processing for small batches
                    for sentence, token_probs, offset_mapping, orig_idx in alignment_args:
                        results[orig_idx] = _align_to_words_optimized(sentence, token_probs, offset_mapping)

                # Fill empty sentences
                for i in range(len(sentences)):
                    if results[i] is None:
                        results[i] = []

                return results

        except Exception as e:
            raise RuntimeError(f"Batch prediction failed: {e}") from e

    async def predict_batch_async(
        self, sentences: List[str]
    ) -> List[List[float]]:
        """
        Predict AI probability for multiple sentences asynchronously.

        Args:
            sentences: List of input sentences

        Returns:
            2D list of word-level AI probabilities
        """
        import asyncio

        tasks = [asyncio.to_thread(self.predict_single, s) for s in sentences]
        return await asyncio.gather(*tasks)


# Global model service instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """Get the global model service instance."""
    return model_service
