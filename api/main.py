"""FastAPI application entry point for AI Text Detector API."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import router
from config import settings
from shared_config import get_model_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Set thread count for optimal CPU performance
    if not os.environ.get("OMP_NUM_THREADS"):
        model_config = get_model_config()
        num_threads = model_config.get("omp_num_threads", os.cpu_count() or 12)
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        print(f"Set OMP_NUM_THREADS={num_threads} for optimal CPU performance")

    # Startup
    print("Starting AI Text Detector API...")
    print(f"Model path: {settings.model_path}")
    print(f"Device: {settings.device}")

    # Preload model (optional - can also lazy load)
    # service = get_model_service()
    # service.load_model()

    yield

    # Shutdown
    print("Shutting down AI Text Detector API...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI Text Detector API",
        "version": settings.api_version,
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
    )
