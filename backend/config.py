"""
Configuration settings for SkillSync backend
"""

import os
from pathlib import Path
from typing import List

class Settings:
    """Application settings"""
    
    # API Settings
    API_TITLE: str = "SkillSync API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-Powered Job Search Revolution"
    
    # Server Settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ['.pdf', '.docx', '.doc']
    UPLOAD_DIRECTORY: Path = Path("uploads")
    
    # AI Model Settings
    SEMANTIC_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPACY_MODEL: str = "en_core_web_sm"
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    
    # Portfolio Settings
    PORTFOLIO_OUTPUT_DIR: Path = Path("generated_portfolios")
    PORTFOLIO_TEMPLATES_DIR: Path = Path("templates")
    
    # Database Settings (for production)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///skillsync.db")
    
    # Performance Settings
    CV_ANALYSIS_TIMEOUT: int = 60  # seconds
    MAX_CONCURRENT_ANALYSES: int = 5
    
    # XAI Settings
    EXPLANATION_CONFIDENCE_THRESHOLD: float = 0.8
    MAX_EXPLANATION_FEATURES: int = 10
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "skillsync-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Adzuna API Settings
    ADZUNA_APP_ID: str = os.getenv("c008d169", "")
    ADZUNA_APP_KEY: str = os.getenv("2f303dff525ab15b04cc310e9a15c985", "")
    ADZUNA_BASE_URL: str = "http://api.adzuna.com/v1/api/jobs"
    
    # Job Matching Settings
    MAX_JOB_RESULTS: int = 20
    DEFAULT_COUNTRY: str = "fr"  # France par défaut
    JOB_SEARCH_TIMEOUT: int = 30  # secondes
    
    def __init__(self):
        """Initialize settings and create directories"""
        self.UPLOAD_DIRECTORY.mkdir(exist_ok=True)
        self.PORTFOLIO_OUTPUT_DIR.mkdir(exist_ok=True)
        self.PORTFOLIO_TEMPLATES_DIR.mkdir(exist_ok=True)

# Global settings instance
settings = Settings()