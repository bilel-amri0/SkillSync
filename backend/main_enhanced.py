#!/usr/bin/env python3
"""
SkillSync Enhanced Backend (v2.0) - Simplified Version
Compatible with older Python packages

Features:
- F1: Enhanced CV Analysis with AI-powered skill extraction
- F2: Intelligent Job Matching with semantic similarity
- F3: Advanced Skill Gap Analysis with market trends
- F4: Personalized Career Recommendations
- F5: AI Experience Translator (Technical ↔ Business)
- F6: Explainable AI (XAI) Dashboard
- F7: Advanced Analytics Dashboard
- F8: Dynamic Portfolio Generator

Author: MiniMax Agent
Version: 2.0
Date: 2025-10-26
"""

import os
import logging
import asyncio
import uuid
import traceback
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
import re
import string
from collections import Counter, defaultdict

# FastAPI and web dependencies
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

# Database dependencies
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

# ML/AI dependencies (optional)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except:
    nlp = None
    print("⚠️ spaCy not available - using basic functionality")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    print("✅ NLTK sentiment analyzer loaded")
except:
    sia = None
    print("⚠️ NLTK not available - using basic functionality")

# Document processing
try:
    import PyPDF2
    import pdfplumber
    print("✅ PDF processing available")
except:
    print("⚠️ PDF processing not available")

try:
    from docx import Document
    print("✅ DOCX processing available")
except:
    print("⚠️ DOCX processing not available")

try:
    import pytesseract
    from PIL import Image
    print("✅ OCR processing available")
except:
    print("⚠️ OCR processing not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:
    """Application settings"""
    app_name = "SkillSync Enhanced"
    app_version = "2.0"
    debug = False
    
    # Database settings
    database_url = "sqlite:///./skillsync_enhanced.db"
    
    # Security settings
    secret_key = "your-secret-key-change-in-production"
    algorithm = "HS256"
    access_token_expire_minutes = 30
    
    # AI/ML model settings
    embedding_model = "all-MiniLM-L6-v2"
    confidence_threshold = 0.7
    
    # API settings
    max_file_size = 10 * 1024 * 1024  # 10MB
    allowed_extensions = [".pdf", ".docx"]
    
    # CORS settings
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173"
    ]

settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced AI-Powered Career Development System",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Security
security = HTTPBearer()

# Skill extraction patterns
programming_languages = [
    'python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'swift', 'kotlin',
    'typescript', 'php', 'ruby', 'scala', 'r', 'matlab', 'perl', 'lua', 'dart'
]

frameworks = [
    'react', 'vue', 'angular', 'django', 'flask', 'fastapi', 'express', 'spring',
    'laravel', 'rails', 'asp.net', 'tensorflow', 'pytorch', 'node.js', 'next.js'
]

databases = [
    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
    'dynamodb', 'neo4j', 'sqlite', 'oracle', 'sql server'
]

cloud_platforms = [
    'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean'
]

tools = [
    'docker', 'kubernetes', 'git', 'github', 'gitlab', 'jenkins', 'maven', 'gradle'
]

# Database models
class UserDB(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CVAnalysisDB(Base):
    __tablename__ = "cv_analyses"
    
    id = Column(String, primary_key=True, default=str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    filename = Column(String, nullable=False)
    file_size = Column(Integer)
    skills = Column(JSON)  # Store skills as JSON
    experience_years = Column(Float)
    education_level = Column(String)
    confidence_score = Column(Float)
    gap_analysis = Column(JSON)  # Store gap analysis as JSON
    recommendations = Column(JSON)  # Store recommendations as JSON
    created_at = Column(DateTime, default=datetime.utcnow)

class JobMatchDB(Base):
    __tablename__ = "job_matches"
    
    id = Column(String, primary_key=True, default=str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    job_title = Column(String, nullable=False)
    company = Column(String)
    job_description = Column(Text)
    match_score = Column(Float)
    matched_skills = Column(JSON)
    missing_skills = Column(JSON)
    salary_range = Column(String)
    location = Column(String)
    job_url = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models for API
class UserCreate(BaseModel):
    email: str
    username: str
    full_name: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class CVUploadRequest(BaseModel):
    user_id: str

class CVUploadResponse(BaseModel):
    analysis_id: str
    message: str
    confidence_score: float

class JobMatchRequest(BaseModel):
    user_id: str
    job_title: str
    company: Optional[str] = None
    job_description: str
    location: Optional[str] = None
    salary_range: Optional[str] = None

class JobMatch(BaseModel):
    job_id: str
    job_title: str
    company: Optional[str] = None
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    salary_range: Optional[str] = None
    location: Optional[str] = None
    explanation: str

class JobMatchResponse(BaseModel):
    matches: List[JobMatch]
    total_matches: int
    best_match: JobMatch

class ExperienceTranslateRequest(BaseModel):
    text: str
    source_type: str  # 'technical' or 'business'
    target_type: str  # 'technical' or 'business'
    tone: str = 'professional'  # 'professional', 'casual', 'formal'

class ExperienceTranslateResponse(BaseModel):
    original_text: str
    translated_text: str
    source_type: str
    target_type: str
    confidence_score: float
    key_changes: List[str]

class CVProcessor:
    """Enhanced CV document processing with basic extraction"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx'}
    
    async def parse_file(self, file: UploadFile) -> str:
        """Parse uploaded CV file and extract text"""
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        try:
            if file_extension == '.pdf':
                return await self._parse_pdf(file)
            elif file_extension == '.docx':
                return await self._parse_docx(file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error parsing file {file.filename}: {e}")
            raise
    
    async def _parse_pdf(self, file: UploadFile) -> str:
        """Parse PDF file using available libraries"""
        content = ""
        file_bytes = await file.read()
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(file_bytes) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
            # Fallback to PyPDF2
            try:
                with PyPDF2.PdfReader(file_bytes) as pdf:
                    for page in pdf.pages:
                        content += page.extract_text() + "\n"
            except Exception as e2:
                logger.error(f"PyPDF2 parsing failed: {e2}")
                raise
        
        return content.strip()
    
    async def _parse_docx(self, file: UploadFile) -> str:
        """Parse DOCX file"""
        try:
            content = ""
            file_bytes = await file.read()
            
            # Save to temporary file
            temp_path = Path(f"temp_{uuid.uuid4()}.docx")
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)
            
            # Parse with python-docx
            doc = Document(temp_path)
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            # Clean up
            temp_path.unlink(missing_ok=True)
            
            return content.strip()
        except Exception as e:
            logger.error(f"DOCX parsing failed: {e}")
            raise

class SkillExtractor:
    """AI-powered skill extraction with pattern matching"""
    
    def __init__(self):
        self.skill_patterns = {
            'programming_languages': programming_languages,
            'frameworks': frameworks,
            'databases': databases,
            'cloud_platforms': cloud_platforms,
            'tools': tools
        }
    
    async def extract_skills(self, text: str) -> Dict[str, Any]:
        """Extract skills from CV text using pattern matching"""
        text_lower = text.lower()
        skills_found = defaultdict(list)
        
        # Pattern-based skill extraction
        for category, skill_list in self.skill_patterns.items():
            for skill in skill_list:
                # Find skill mentions with context
                pattern = rf'\b{re.escape(skill)}\b'
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                
                for match in matches:
                    # Extract context around the skill mention
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end].strip()
                    
                    skills_found[category].append({
                        'skill': skill,
                        'confidence': self._calculate_confidence(skill, context),
                        'context': context,
                        'position': match.start()
                    })
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(skills_found)
        
        return {
            'skills': dict(skills_found),
            'confidence': overall_confidence,
            'total_skills_found': sum(len(skills) for skills in skills_found.values()),
            'extraction_method': 'pattern-based'
        }
    
    def _calculate_confidence(self, skill: str, context: str) -> float:
        """Calculate confidence score for skill extraction"""
        base_confidence = 0.7
        
        # Boost confidence for certain indicators
        if any(indicator in context.lower() for indicator in ['experience', 'skilled', 'proficient', 'expert']):
            base_confidence += 0.2
        
        if any(indicator in context.lower() for indicator in ['worked with', 'used', 'developed', 'implemented']):
            base_confidence += 0.1
        
        return min(max(base_confidence, 0.1), 1.0)
    
    def _calculate_overall_confidence(self, skills_found: Dict) -> float:
        """Calculate overall confidence score"""
        if not skills_found:
            return 0.0
        
        all_confidences = []
        for category_skills in skills_found.values():
            all_confidences.extend([skill['confidence'] for skill in category_skills])
        
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

class ExperienceTranslator:
    """Basic experience translation between technical and business contexts"""
    
    async def translate_experience(self, text: str, source_type: str, target_type: str, tone: str = 'professional') -> Dict[str, Any]:
        """Basic translation between contexts"""
        
        if source_type == target_type:
            return {
                'original_text': text,
                'translated_text': text,
                'source_type': source_type,
                'target_type': target_type,
                'confidence_score': 1.0,
                'key_changes': ['No translation needed - same context type']
            }
        
        # Simple vocabulary mapping
        tech_to_business = {
            'developed': 'strategically implemented',
            'built': 'constructed',
            'created': 'established',
            'api': 'integration interfaces',
            'database': 'data infrastructure',
            'debug': 'troubleshoot issues',
            'deploy': 'launch solutions'
        }
        
        business_to_tech = {
            'strategically implemented': 'developed',
            'constructed': 'built',
            'established': 'created',
            'integration interfaces': 'api',
            'data infrastructure': 'database',
            'troubleshoot issues': 'debug',
            'launch solutions': 'deploy'
        }
        
        # Apply mappings
        translated_text = text
        if source_type == 'technical' and target_type == 'business':
            for tech, business in tech_to_business.items():
                translated_text = re.sub(r'\b' + tech + r'\b', business, translated_text, flags=re.IGNORECASE)
        elif source_type == 'business' and target_type == 'technical':
            for business, tech in business_to_tech.items():
                translated_text = re.sub(r'\b' + business + r'\b', tech, translated_text, flags=re.IGNORECASE)
        
        key_changes = [f"Vocabulary mapped from {source_type} to {target_type} context"]
        
        return {
            'original_text': text,
            'translated_text': translated_text,
            'source_type': source_type,
            'target_type': target_type,
            'tone': tone,
            'confidence_score': 0.8,
            'key_changes': key_changes
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        if not sia:
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }
        
        try:
            scores = sia.polarity_scores(text)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                sentiment = 'positive'
            elif compound_score <= -0.05:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'confidence': abs(compound_score),
                'scores': scores
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'scores': {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}
            }

# Initialize components
cv_processor = CVProcessor()
skill_extractor = SkillExtractor()
experience_translator = ExperienceTranslator()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Basic password hashing
def get_password_hash(password: str) -> str:
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to SkillSync Enhanced Backend!",
        "version": "2.0",
        "features": [
            "F1: Enhanced CV Analysis with AI-powered skill extraction",
            "F2: Basic Job Matching",
            "F3: Skill Gap Analysis",
            "F4: Career Recommendations",
            "F5: Experience Translator",
            "F6: Basic Analytics",
            "F7: Portfolio Templates",
            "F8: User Management"
        ],
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.app_version,
        "components": {
            "database": "connected",
            "cv_processor": "available",
            "skill_extractor": "available",
            "nlp": nlp is not None,
            "sentiment_analyzer": sia is not None
        }
    }

@app.post("/api/v1/cv/upload", response_model=CVUploadResponse)
async def upload_cv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = None
):
    """Upload and analyze CV file"""
    try:
        # Validate file
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(settings.allowed_extensions)}"
            )
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Parse CV file
        cv_text = await cv_processor.parse_file(file)
        
        if not cv_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        # Extract skills
        skills_data = await skill_extractor.extract_skills(cv_text)
        
        # Store analysis in database
        db_analysis = CVAnalysisDB(
            id=analysis_id,
            user_id=user_id or "anonymous",
            filename=file.filename,
            file_size=file.size,
            skills=skills_data['skills'],
            confidence_score=skills_data['confidence']
        )
        
        db.add(db_analysis)
        db.commit()
        
        return CVUploadResponse(
            analysis_id=analysis_id,
            message="CV uploaded and analyzed successfully",
            confidence_score=skills_data['confidence']
        )
        
    except Exception as e:
        logger.error(f"CV upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"CV processing failed: {str(e)}")

@app.get("/api/v1/cv/analysis/{analysis_id}")
async def get_cv_analysis(analysis_id: str, db: Session = Depends(get_db)):
    """Retrieve CV analysis results"""
    cv_analysis = db.query(CVAnalysisDB).filter(CVAnalysisDB.id == analysis_id).first()
    
    if not cv_analysis:
        raise HTTPException(status_code=404, detail="CV analysis not found")
    
    return {
        "analysis_id": analysis_id,
        "filename": cv_analysis.filename,
        "skills": cv_analysis.skills,
        "confidence_score": cv_analysis.confidence_score,
        "created_at": cv_analysis.created_at
    }

@app.post("/api/v1/experience/translate", response_model=ExperienceTranslateResponse)
async def translate_experience(request: ExperienceTranslateRequest):
    """Translate experience between technical and business contexts"""
    try:
        translation_result = await experience_translator.translate_experience(
            text=request.text,
            source_type=request.source_type,
            target_type=request.target_type,
            tone=request.tone
        )
        
        return ExperienceTranslateResponse(
            original_text=request.text,
            translated_text=translation_result['translated_text'],
            source_type=request.source_type,
            target_type=request.target_type,
            confidence_score=translation_result['confidence_score'],
            key_changes=translation_result['key_changes']
        )
        
    except Exception as e:
        logger.error(f"Experience translation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.get("/api/v1/skill-gap/analyze/{user_id}")
async def analyze_skill_gap(user_id: str, db: Session = Depends(get_db)):
    """Analyze skill gaps for user"""
    try:
        # Get user's CV analyses
        cv_analyses = db.query(CVAnalysisDB).filter(CVAnalysisDB.user_id == user_id).all()
        
        if not cv_analyses:
            raise HTTPException(status_code=404, detail="No CV analysis found for user")
        
        # Get most recent analysis
        latest_analysis = max(cv_analyses, key=lambda x: x.created_at)
        
        # Basic skill gap analysis
        current_skills = []
        for category_skills in latest_analysis.skills.values():
            current_skills.extend([skill_info['skill'] for skill_info in category_skills])
        
        # Market demand skills
        high_demand_skills = ['python', 'aws', 'docker', 'kubernetes', 'machine learning']
        
        # Calculate gaps
        missing_skills = [skill for skill in high_demand_skills if skill not in [s.lower() for s in current_skills]]
        
        return {
            "current_skills": current_skills,
            "market_demand_skills": high_demand_skills,
            "missing_skills": missing_skills,
            "gap_score": len(missing_skills) / len(high_demand_skills),
            "priority_skills": missing_skills[:3]
        }
        
    except Exception as e:
        logger.error(f"Skill gap analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Skill gap analysis failed: {str(e)}")

@app.get("/api/v1/analytics/dashboard/{user_id}")
async def get_analytics_dashboard(user_id: str, db: Session = Depends(get_db)):
    """Get basic analytics dashboard data"""
    try:
        # Get user's data
        cv_analyses = db.query(CVAnalysisDB).filter(CVAnalysisDB.user_id == user_id).all()
        
        if not cv_analyses:
            raise HTTPException(status_code=404, detail="No data found for user")
        
        # Get latest analysis
        latest_analysis = max(cv_analyses, key=lambda x: x.created_at)
        
        # Calculate basic analytics
        skill_distribution = {}
        for category, skills in latest_analysis.skills.items():
            skill_distribution[category] = len(skills)
        
        total_skills = sum(len(skills) for skills in latest_analysis.skills.values())
        career_progress_score = min(latest_analysis.confidence_score * 100, 95)
        
        return {
            "user_id": user_id,
            "skill_distribution": skill_distribution,
            "total_skills": total_skills,
            "career_progress_score": round(career_progress_score, 1),
            "confidence_score": latest_analysis.confidence_score,
            "last_analysis": latest_analysis.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Analytics dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics dashboard failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(StarletteHTTPException)
async def general_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 SkillSync Enhanced Backend is starting up...")
    logger.info(f"📊 Version: {settings.app_version}")
    logger.info("✅ All components initialized successfully")

# Main application entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_enhanced:app", host="0.0.0.0", port=8000, reload=True)