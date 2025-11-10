"""
SkillSync Main API Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print("🔑 Environment variables loaded from .env file")
    else:
        print("⚠️ .env file not found at", dotenv_path)
except ImportError:
    # If python-dotenv is not installed, try to load manually
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("🔑 Environment variables loaded manually from .env file")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import routers
from routers.interview_router import router as interview_router

# Create FastAPI application
app = FastAPI(
    title="SkillSync API",
    description="AI-Powered Career Development Platform with Interview Practice",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(interview_router)

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SkillSync API",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "AI-Powered Interview Practice",
            "CV Analysis",
            "Job Matching",
            "Experience Translation",
            "Portfolio Generation"
        ]
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
