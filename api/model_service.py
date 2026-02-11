"""Model service for AI text detection inference."""

import time
import traceback
from pathlib import Path
from typing import List, Optional

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


class ModelService:
    """Singleton service for model loading and inference."""

    _instance: Optional["ModelService"] = None
    _model = None
    _tokenizer = None
    _device = None
    _model_version = None
    _inference_batch_size = None
    _use_fp16 = None

    def __new__(cls) -> "ModelService":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize model service (lazy loading)."""
        if self._device is None:
            self._device = get_device()

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

            # Use half precision for faster inference (if CUDA available)
            if self._device == "cuda":
                print("Enabling FP16 for faster inference...")
                self._model.half()

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

                # Align to words
                word_probs = self._align_to_words(text, ai_probs, offset_mapping)

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
        """Process a single batch of sentences."""
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

                # Align to words for each sentence
                results = [None] * len(sentences)
                for batch_idx, (orig_idx, sentence) in enumerate(indexed_sentences):
                    offset_mapping = offset_mappings[batch_idx]
                    token_probs = probs[batch_idx]
                    word_probs = self._align_to_words(sentence, token_probs, offset_mapping)
                    results[orig_idx] = word_probs

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

    def _align_to_words(
        self,
        text: str,
        token_probs: List[float],
        offset_mapping,
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

            # Average probabilities for the word
            if token_values:
                avg_prob = sum(token_values) / len(token_values)
                word_probs.append(round(avg_prob, 4))
            else:
                word_probs.append(0.0)

        return word_probs


# Global model service instance
model_service = ModelService()


def get_model_service() -> ModelService:
    """Get the global model service instance."""
    return model_service
