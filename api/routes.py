"""FastAPI routes for AI text detection."""

import time
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from schemas import (
    BatchJobRequest,
    BatchJobResponse,
    BatchJobStatus,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ResponseMetadata,
)
from service_selector import get_model_service

# Batch job storage (in-memory for now)
batch_jobs: Dict[str, BatchJobStatus] = {}

router = APIRouter(prefix="/api/v1", tags=["prediction"])


@router.post(
    "/predict/sentences",
    response_model=PredictResponse,
    summary="Predict AI-generated probability for sentences",
    description="Accepts an array of sentences and returns word-level AI probabilities.",
)
async def predict_sentences(request: PredictRequest) -> PredictResponse:
    """
    Predict AI-generated probability for each word in the input sentences.

    Args:
        request: PredictRequest with sentences array

    Returns:
        PredictResponse with 2D array of probabilities
    """
    service = get_model_service()

    if not service.is_loaded:
        try:
            service.load_model()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e),
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {e}",
            )

    start_time = time.time()

    try:
        # Process sentences
        probabilities = service.predict_batch(request.sentences)

        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        total_words = sum(len(s.split()) for s in request.sentences)

        metadata = ResponseMetadata(
            model_version=service.model_version or "unknown",
            processing_time_ms=round(processing_time, 2),
            total_words=total_words,
        )

        return PredictResponse(probabilities=probabilities, metadata=metadata)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchJobResponse,
    summary="Submit batch job for async processing",
    description="Submit a large batch of sentences for asynchronous processing.",
)
async def submit_batch_job(request: BatchJobRequest) -> BatchJobResponse:
    """
    Submit a batch job for asynchronous processing.

    Args:
        request: BatchJobRequest with sentences and optional callback URL

    Returns:
        BatchJobResponse with job ID
    """
    job_id = str(uuid.uuid4())

    # Create job entry
    batch_jobs[job_id] = BatchJobStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
    )

    # Process in background (simplified - use Celery/ARQ for production)
    # For now, just process synchronously and mark complete
    try:
        service = get_model_service()
        if not service.is_loaded:
            service.load_model()

        probabilities = service.predict_batch(request.sentences)

        # Update job status
        total_words = sum(len(s.split()) for s in request.sentences)
        metadata = ResponseMetadata(
            model_version=service.model_version or "unknown",
            processing_time_ms=0.0,
            total_words=total_words,
        )

        batch_jobs[job_id] = BatchJobStatus(
            job_id=job_id,
            status="complete",
            progress=1.0,
            result=PredictResponse(probabilities=probabilities, metadata=metadata),
        )

    except Exception as e:
        batch_jobs[job_id] = BatchJobStatus(
            job_id=job_id,
            status="failed",
            progress=0.0,
            error=str(e),
        )

    return BatchJobResponse(
        job_id=job_id,
        status=batch_jobs[job_id].status,
        total_sentences=len(request.sentences),
    )


@router.get(
    "/jobs/{job_id}",
    response_model=BatchJobStatus,
    summary="Get batch job status",
    description="Retrieve the status and results of a batch job.",
)
async def get_job_status(job_id: str) -> BatchJobStatus:
    """
    Get the status of a batch job.

    Args:
        job_id: Job identifier

    Returns:
        BatchJobStatus with current status and results if complete
    """
    if job_id not in batch_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return batch_jobs[job_id]


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health and model status.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns:
        HealthResponse with API status and model information
    """
    service = get_model_service()

    return HealthResponse(
        status="healthy",
        model_loaded=service.is_loaded,
        model_version=service.model_version,
        device=service.device,
    )


# Note: Exception handlers should be added to main FastAPI app, not router
# FastAPI already handles HTTPExceptions by default
