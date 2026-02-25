"""Model definition for AI text detection."""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

# Check for Flash Attention 2 availability
def _check_flash_attn_available() -> bool:
    """Check if Flash Attention 2 is available."""
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn
        # Check if GPU architecture is supported (Ampere+ for FA2)
        major, _ = torch.cuda.get_device_capability()
        if major < 8:  # Pre-Ampere (T4, etc.) don't support FA2 well
            return False
        return True
    except ImportError:
        return False


def _get_attention_implementation():
    """Get the best available attention implementation for DeBERTa."""
    if not torch.cuda.is_available():
        return "eager"

    # Try Flash Attention 2 first (best for H100, RTX 3090)
    if _check_flash_attn_available():
        return "flash_attention_2"

    # DeBERTa-v2 doesn't support SDPA, fall back to eager
    # See: https://github.com/huggingface/transformers/issues/28005
    return "eager"


class AIDetectorModel(nn.Module):
    """Wrapper for DeBERTa-v3 token classification model."""

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 2,
        cache_dir: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ):
        """
        Initialize AI detector model.

        Args:
            model_name: Hugging Face model name
            num_labels: Number of classification labels (2: human=0, AI=1)
            cache_dir: Directory to cache downloaded models
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")
        """
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels

        # Auto-detect best attention implementation if not specified
        if attn_implementation is None:
            attn_implementation = _get_attention_implementation()
        else:
            # Validate requested attention implementation is available
            if attn_implementation == "flash_attention_2":
                if not _check_flash_attn_available():
                    print("⚠️  Flash Attention 2 requested but not available (install flash-attn or GPU not supported)")
                    print("   Falling back to eager implementation")
                    attn_implementation = "eager"
            elif attn_implementation == "sdpa":
                # DeBERTa-v2 doesn't support SDPA
                print("⚠️  SDPA requested but DeBERTa-v2 doesn't support it")
                print("   Falling back to eager implementation")
                attn_implementation = "eager"

        print(f"Using attention implementation: {attn_implementation}")

        # Load model configuration
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            cache_dir=cache_dir,
        )

        # Load model with optimized attention
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            config=self.config,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
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
        device: str = "cuda",
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
    def from_pretrained(cls, model_path: str, attn_implementation: Optional[str] = None) -> "AIDetectorModel":
        """
        Load a fine-tuned model from directory.

        Args:
            model_path: Path to saved model directory
            attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")

        Returns:
            Loaded AIDetectorModel instance
        """
        instance = cls.__new__(cls)
        instance.model_name = model_path
        instance.num_labels = 2

        # Auto-detect best attention implementation if not specified
        if attn_implementation is None:
            attn_implementation = _get_attention_implementation()

        instance.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        instance.tokenizer = AutoTokenizer.from_pretrained(model_path)
        instance.config = instance.model.config

        return instance


def get_model(
    model_name: str = "microsoft/deberta-v3-base",
    pretrained_path: Optional[str] = None,
    attn_implementation: Optional[str] = None,
) -> AIDetectorModel:
    """
    Get AI detector model.

    Args:
        model_name: Base model name (for new model)
        pretrained_path: Path to fine-tuned model (optional)
        attn_implementation: Attention implementation ("flash_attention_2", "sdpa", "eager")

    Returns:
        AIDetectorModel instance
    """
    if pretrained_path:
        return AIDetectorModel.from_pretrained(pretrained_path, attn_implementation)
    return AIDetectorModel(model_name=model_name, attn_implementation=attn_implementation)
