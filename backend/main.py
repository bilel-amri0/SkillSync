"""
SkillSync Main Application - Modular Architecture
Version: 2.2.0 (10/10 Release)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging

# Import logging configuration
from utils.logging_config import setup_logging, get_logger

# Setup structured logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "skillsync.log")
JSON_LOGGING = os.getenv("JSON_LOGGING", "true").lower() == "true"

setup_logging(
    log_level=LOG_LEVEL,
    log_file=LOG_FILE,
    json_format=JSON_LOGGING
)

logger = get_logger(__name__)

# Import routers
from routers import cv_analysis, recommendations, dashboard, portfolio
from auth.router import router as auth_router

# Import middleware
from middleware.logging_middleware import RequestLoggingMiddleware

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    logger.warning("slowapi not installed. Rate limiting disabled.")

# Initialize database
try:
    from database import init_db
    init_db()
    logger.info(" Database initialized successfully")
except Exception as e:
    logger.error(f" Database initialization failed: {e}", exc_info=True)

# Create FastAPI app
app = FastAPI(
    title="SkillSync API",
    description="Enterprise-Grade AI-Powered Career Development Platform with Authentication",
    version="2.2.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Rate limiting setup
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info(" Rate limiting enabled (100 req/min)")

# CORS configuration
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
    max_age=600
)

logger.info(f" CORS configured for origins: {ALLOWED_ORIGINS}")

# Include routers
app.include_router(auth_router)
app.include_router(cv_analysis.router)
app.include_router(recommendations.router)
app.include_router(dashboard.router)
app.include_router(portfolio.router)

logger.info(" All routers registered (including portfolio generation)")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info(" SkillSync API v2.2.0 starting...")
    logger.info(f" Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f" Database: {os.getenv('DATABASE_URL', 'sqlite:///./skillsync.db')}")
    logger.info(f" Logging: JSON={JSON_LOGGING}, Level={LOG_LEVEL}")
    logger.info(" SkillSync API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info(" SkillSync API shutting down...")
    logger.info(" Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    logger.info(f" Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
