"""Model definition for AI text detection."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)


class AIDetectorModel(nn.Module):
    """Wrapper for DeBERTa-v3 token classification model."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-large",
        num_labels: int = 2,
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Initialize AI detector model.

        Args:
            model_name: Hugging Face model name
            num_labels: Number of classification labels (2: human=0, AI=1)
            cache_dir: Directory to cache downloaded models
            load_in_8bit: Load model in 8-bit mode (requires bitsandbytes)
            load_in_4bit: Load model in 4-bit mode (requires bitsandbytes)
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Load model configuration
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
        )

        # Prepare quantization config if needed
        quantization_config = None
        if load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir,
            quantization_config=quantization_config,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        )

    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs)

    def predict(
        self,
        texts: List[str],
        device: str = "cpu",
        return_probabilities: bool = True,
    ) -> List[List[float]]:
        """
        Predict AI probability for each word in input texts.

        Args:
            texts: List of input sentences
            device: Device to run inference on
            return_probabilities: Return probabilities (True) or labels (False)

        Returns:
            List of word-level probabilities or labels
        """
        self.model.eval()
        self.model.to(device)

        results = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                )
                encoding = {k: v.to(device) for k, v in encoding.items()}

                # Get predictions
                outputs = self.model(**encoding)
                logits = outputs.logits[0]  # [seq_len, num_labels]

                # Get probabilities (softmax)
                probs = torch.softmax(logits, dim=-1)
                ai_probs = probs[:, 1].cpu().tolist()  # AI probability for each token

                # Align to words
                word_probs = self._align_to_words(text, ai_probs)
                results.append(word_probs)

        return results

    def _align_to_words(
        self,
        text: str,
        token_probs: List[float],
    ) -> List[float]:
        """
        Align token-level probabilities to words.

        Args:
            text: Original input text
            token_probs: Token-level probabilities

        Returns:
            Word-level probabilities
        """
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offset_mapping = encoding["offset_mapping"][0].tolist()
        words = text.split()
        word_probs = []

        for word in words:
            word_start = text.find(word)
            word_end = word_start + len(word)

            # Find all tokens for this word
            token_values = []
            for i, (start, end) in enumerate(offset_mapping):
                if start == 0 and end == 0:  # Special token
                    continue
                if start >= word_start and end <= word_end:
                    if i < len(token_probs):
                        token_values.append(token_probs[i])

            # Average probabilities for the word
            if token_values:
                word_probs.append(sum(token_values) / len(token_values))
            else:
                word_probs.append(0.0)

        return word_probs

    def save(self, save_directory: str) -> None:
        """
        Save model and tokenizer.

        Args:
            save_directory: Directory to save model
        """
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "AIDetectorModel":
        """
        Load a fine-tuned model from directory.

        Args:
            model_path: Path to saved model directory

        Returns:
            Loaded AIDetectorModel instance
        """
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.num_labels = 2

        instance.model = AutoModelForTokenClassification.from_pretrained(model_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.config = instance.model.config

        return instance


def get_model(
    model_name: str = "microsoft/deberta-v3-large",
    pretrained_path: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> AIDetectorModel:
    """
    Get AI detector model.

    Args:
        model_name: Base model name (for new model)
        pretrained_path: Path to fine-tuned model (optional)
        load_in_8bit: Load model in 8-bit mode
        load_in_4bit: Load model in 4-bit mode

    Returns:
        AIDetectorModel instance
    """
    if pretrained_path:
        return AIDetectorModel.from_pretrained(pretrained_path)
    return AIDetectorModel(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
    )
