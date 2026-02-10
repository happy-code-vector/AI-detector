"""Pydantic schemas for API requests and responses."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request for AI text prediction."""

    sentences: List[str] = Field(
        ...,
        description="List of sentences to analyze (max 120)",
        min_length=1,
        max_length=120,
    )

    @field_validator("sentences")
    @classmethod
    def validate_sentences(cls, v: List[str]) -> List[str]:
        """Validate sentence constraints."""
        from config import settings

        if len(v) > settings.max_sentences:
            raise ValueError(
                f"Too many sentences. Maximum {settings.max_sentences} allowed."
            )

        for i, sentence in enumerate(v):
            if not isinstance(sentence, str):
                raise ValueError(f"Sentence {i} must be a string.")
            if not sentence.strip():
                raise ValueError(f"Sentence {i} is empty.")
            word_count = len(sentence.split())
            if word_count > settings.max_words_per_sentence:
                raise ValueError(
                    f"Sentence {i} exceeds maximum word count of {settings.max_words_per_sentence}."
                )

        return v


class WordProbabilities(BaseModel):
    """Word-level probabilities for a single sentence."""

    probabilities: List[float] = Field(
        ...,
        description="AI-generated probability for each word (0.0 to 1.0)",
    )


class PredictResponse(BaseModel):
    """Response for AI text prediction."""

    probabilities: List[List[float]] = Field(
        ...,
        description="2D array of word-level AI probabilities",
    )
    metadata: Optional["ResponseMetadata"] = None


class ResponseMetadata(BaseModel):
    """Metadata about the prediction response."""

    model_version: str = Field(..., description="Version of the model used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    total_words: int = Field(..., description="Total number of words processed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="API status (healthy/unhealthy)")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Model version if loaded")
    device: str = Field(..., description="Device being used (cuda/cpu)")


class BatchJobRequest(BaseModel):
    """Request for batch processing."""

    sentences: List[str] = Field(
        ...,
        description="List of sentences for batch processing",
        min_length=1,
    )
    callback_url: Optional[str] = Field(
        None,
        description="Optional webhook URL for completion notification",
    )


class BatchJobResponse(BaseModel):
    """Response for batch job creation."""

    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status (pending/processing/complete/failed)")
    total_sentences: int = Field(..., description="Number of sentences to process")


class BatchJobStatus(BaseModel):
    """Status of a batch job."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    progress: float = Field(..., description="Progress (0.0 to 1.0)")
    result: Optional[PredictResponse] = Field(None, description="Results if complete")
    error: Optional[str] = Field(None, description="Error message if failed")


# Update forward references
PredictResponse.model_rebuild()
