"""
FastAPI backend for FocusPlaces.

Provides REST API endpoints for Streamlit frontend to:
- Search for places using Google Places API
- Process reviews and compute focus scores
- Return ranked results

Environment requirements:
- GOOGLE_PLACES_API_KEY: Set in .env file
- BACKEND_HOST: Defaults to "localhost" (optional)
- BACKEND_PORT: Defaults to 8000 (optional)

Startup:
    uvicorn main:app --reload
    # Then access: http://localhost:8000
    # API docs: http://localhost:8000/docs
"""

import os
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import existing modules (we'll use these when building endpoints)
from places_api import geocode_address
from nlp_review_processor import process_places

# Load environment variables from .env
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", 8000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validate required environment variables
if not API_KEY:
    raise RuntimeError(
        "Missing GOOGLE_PLACES_API_KEY in .env file. "
        "Please set GOOGLE_PLACES_API_KEY=your_api_key in .env"
    )

# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Request/Response Models
# ============================================================================


class SearchRequest(BaseModel):
    """Request model for place search endpoint."""

    queries: list[str]
    location: Optional[str] = None
    radius_miles: float = 7.5
    recent_days: int = 900
    min_recent_reviews: int = 3
    max_candidates_per_query: int = 5
    max_reviews_per_place: int = 5

    class Config:
        json_schema_extra = {
            "example": {
                "queries": ["coffee shop", "library"],
                "location": "State College, PA",
                "radius_miles": 7.5,
                "recent_days": 900,
                "min_recent_reviews": 3,
                "max_candidates_per_query": 5,
                "max_reviews_per_place": 5,
            }
        }


class ReviewKeyword(BaseModel):
    """A keyword detected in a review."""

    keyword: str
    weight: float


class ReviewHighlight(BaseModel):
    """Single review that contributed to a place's focus score."""

    text: str
    author: str
    score: float
    keywords: list[ReviewKeyword]


class PlaceResult(BaseModel):
    """A place result with focus score and key review."""

    name: str
    place_id: str
    location: dict  # {"lat": float, "lng": float}
    focus_score: float
    num_reviews: int
    rating: Optional[float] = None
    top_review: Optional[ReviewHighlight] = None
    url: Optional[str] = None


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    success: bool
    results: list[PlaceResult]
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str
    api_key_loaded: bool


# ============================================================================
# Custom Exception Handlers
# ============================================================================


class APIConfigError(Exception):
    """Raised when API is misconfigured."""

    pass


# ============================================================================
# Lifespan Context (startup/shutdown)
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    # Startup
    logger.info("FocusPlaces API starting up...")
    logger.info(f"GOOGLE_PLACES_API_KEY loaded: {bool(API_KEY)}")
    logger.info(f"Listening on http://{BACKEND_HOST}:{BACKEND_PORT}")
    logger.info(f"API docs available at http://{BACKEND_HOST}:{BACKEND_PORT}/docs")

    yield

    # Shutdown
    logger.info("FocusPlaces API shutting down...")


# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="FocusPlaces API",
    description="Backend API for FocusPlaces Streamlit application",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS to allow requests from Streamlit frontend
ALLOWED_ORIGINS = [
    "http://localhost:8501",      # Streamlit default port
    "http://127.0.0.1:8501",
    "http://localhost:3000",      # Alternative frontend port
    "http://127.0.0.1:3000",
]

if os.getenv("ENVIRONMENT") == "production":
    # In production, restrict to your actual frontend domain
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"CORS configured for origins: {ALLOWED_ORIGINS}")

# ============================================================================
# Root and Health Check Endpoints (examples)
# ============================================================================


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {
        "message": "FocusPlaces API is running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        api_key_loaded=bool(API_KEY),
    )


# ============================================================================
# TODO: Search Endpoint (to be implemented)
# ============================================================================
# Once you finalize the Streamlit frontend structure, add the search endpoint:
#
# @app.post("/search", response_model=SearchResponse, tags=["Search"])
# async def search(request: SearchRequest):
#     """
#     Search for places and compute focus scores.
#
#     This endpoint will:
#     1. Geocode the provided location (or use current location)
#     2. Call Google Places API for each query
#     3. Fetch reviews for each place
#     4. Compute focus scores using NLP
#     5. Return ranked results
#     """
#     # TODO: Implement search logic here
#     pass


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(APIConfigError)
async def api_config_error_handler(request, exc):
    """Handle API configuration errors."""
    logger.error(f"API configuration error: {exc}")
    return {
        "error": "API Configuration Error",
        "detail": str(exc),
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
