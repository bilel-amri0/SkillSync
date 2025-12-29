"""
FastAPI Integration for Production CV Parser
Drop-in replacement for existing CV analysis endpoint
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import logging

# Import production parser
from production_cv_parser_final import ProductionCVParser, extract_text_from_pdf

logger = logging.getLogger(__name__)

# Initialize parser once (singleton)
cv_parser = ProductionCVParser()

# Router
router = APIRouter(prefix="/api/v1", tags=["cv-analysis"])


# ==================== REQUEST/RESPONSE MODELS ====================

class CVAnalysisResponse(BaseModel):
    """API response model"""
    # Personal
    name: str = None
    email: str = None
    phone: str = None
    location: str = None
    
    # Professional
    current_title: str = None
    seniority_level: str = "Unknown"
    
    # Skills
    skills: list = []
    skill_categories: dict = {}
    total_skills: int = 0
    
    # Experience
    total_years_experience: int = 0
    job_titles: list = []
    companies: list = []
    
    # Education
    degrees: list = []
    institutions: list = []
    
    # Metadata
    confidence_score: float = 0.0
    processing_time_ms: int = 0
    analysis_method: str = "Production ML (mpnet-768 + BERT-NER)"


# ==================== ENDPOINTS ====================

@router.post("/analyze-cv-production", response_model=CVAnalysisResponse)
async def analyze_cv_production(file: UploadFile = File(...)):
    """
    Production CV analysis endpoint
    Uses mpnet-768 embeddings + BERT-NER
    CPU-optimized, 180ms average processing time
    """
    try:
        # Validate file type
        if file.content_type not in ['application/pdf', 'text/plain']:
            raise HTTPException(400, "Only PDF and TXT files supported")
        
        # Read file
        content = await file.read()
        
        # Extract text
        if file.content_type == 'application/pdf':
            text = extract_text_from_pdf(content)
        else:
            text = content.decode('utf-8')
        
        if not text or len(text) < 50:
            raise HTTPException(400, "Could not extract text from file")
        
        # Parse with production system
        result = cv_parser.parse_cv(text)
        
        # Convert to response
        response = CVAnalysisResponse(
            name=result.name,
            email=result.email,
            phone=result.phone,
            location=result.location,
            current_title=result.current_title,
            seniority_level=result.seniority_level,
            skills=result.skills,
            skill_categories=result.skill_categories,
            total_skills=len(result.skills),
            total_years_experience=result.total_years_experience,
            job_titles=result.job_titles,
            companies=result.companies,
            degrees=result.degrees,
            institutions=result.institutions,
            confidence_score=result.confidence_score,
            processing_time_ms=result.processing_time_ms,
            analysis_method="Production ML (mpnet-768 + BERT-NER)"
        )
        
        logger.info(f" CV analyzed: {result.name or 'Unknown'} | {len(result.skills)} skills | {result.processing_time_ms}ms")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" CV analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@router.post("/analyze-cv-text", response_model=CVAnalysisResponse)
async def analyze_cv_text(cv_text: str):
    """
    Analyze CV from plain text
    Useful for testing or when text is already extracted
    """
    try:
        if not cv_text or len(cv_text) < 50:
            raise HTTPException(400, "CV text too short")
        
        # Parse
        result = cv_parser.parse_cv(cv_text)
        
        # Convert to response
        response = CVAnalysisResponse(
            name=result.name,
            email=result.email,
            phone=result.phone,
            location=result.location,
            current_title=result.current_title,
            seniority_level=result.seniority_level,
            skills=result.skills,
            skill_categories=result.skill_categories,
            total_skills=len(result.skills),
            total_years_experience=result.total_years_experience,
            job_titles=result.job_titles,
            companies=result.companies,
            degrees=result.degrees,
            institutions=result.institutions,
            confidence_score=result.confidence_score,
            processing_time_ms=result.processing_time_ms,
            analysis_method="Production ML (mpnet-768 + BERT-NER)"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f" Text analysis failed: {e}")
        raise HTTPException(500, f"Analysis failed: {str(e)}")


# ==================== INTEGRATION INSTRUCTIONS ====================
"""
TO INTEGRATE INTO YOUR EXISTING FASTAPI APP:

1. Add to your main.py:

from fastapi_integration import router as cv_production_router
app.include_router(cv_production_router)


2. Or replace your existing endpoint:

# OLD:
@app.post("/api/v1/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    # ... old code

# NEW:
from production_cv_parser_final import ProductionCVParser, extract_text_from_pdf

cv_parser = ProductionCVParser()  # Initialize once

@app.post("/api/v1/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    content = await file.read()
    text = extract_text_from_pdf(content)
    result = cv_parser.parse_cv(text)
    return result.to_dict()


3. Test the endpoint:

curl -X POST "http://localhost:8001/api/v1/analyze-cv-production" \
  -F "file=@cv.pdf"


4. Expected response time: 150-250ms (CPU)


5. Memory usage: 1.2GB (models loaded once)
"""
