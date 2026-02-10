"""Model service for AI text detection inference."""

import time
import traceback
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import get_device, settings


class ModelService:
    """Singleton service for model loading and inference."""

    _instance: Optional["ModelService"] = None
    _model = None
    _tokenizer = None
    _device = None
    _model_version = None

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
            self._model = AutoModelForTokenClassification.from_pretrained(
                str(model_path),
            )
            self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))

            self._model.to(self._device)
            self._model.eval()

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

        Args:
            sentences: List of input sentences

        Returns:
            2D list of word-level AI probabilities
        """
        results = []
        for sentence in sentences:
            results.append(self.predict_single(sentence))
        return results

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
