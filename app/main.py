"""FastAPI application."""
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before importing config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.routers import predict, feedback, admin

# Setup logging for production
setup_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)
app.include_router(feedback.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """API root."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    import os
    from datetime import datetime

    health_status = {
        "status": "healthy",
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENV", "development"),
    }

    # Optional: Check Supabase connectivity
    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if url and key:
            client = create_client(url, key)
            response = client.table("feedback").select("count", count="exact").execute()
            health_status["database"] = "connected"
            health_status["feedback_count"] = response.count
        else:
            health_status["database"] = "not configured"
    except Exception as e:
        health_status["database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"

    return health_status


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
