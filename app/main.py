from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import sys
from pathlib import Path

from app.api.routes import router
from app.config import get_settings

# Configure logging with UTF-8 encoding
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Force UTF-8 encoding for file handler
file_handler = logging.FileHandler('app.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[stream_handler, file_handler]
)

# Set encoding for stdout on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Social Media Sentiment Analyzer",
    description="API for analyzing sentiments and detecting trends on Reddit and Twitter",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Mount static files directory for frontend
static_dir = Path(__file__).parent.parent / "frontend"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML page"""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "Frontend not found. Please check the frontend directory."}
else:
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Social Media Sentiment Analyzer API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Social Media Sentiment Analyzer API")
    settings = get_settings()
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Database: PostgreSQL (async)")
    
    # Initialize PostgreSQL tables
    from app.services.database_service import database_service
    logger.info("Creating PostgreSQL tables if they don't exist...")
    await database_service.create_tables()
    
    # Check database availability
    if await database_service.is_available():
        logger.info("✅ PostgreSQL connection successful")
    else:
        logger.warning("⚠️ PostgreSQL connection failed - running without database")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Social Media Sentiment Analyzer API")
    from app.services.database_service import database_service
    await database_service.close()
    logger.info("✅ PostgreSQL connection closed")


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
