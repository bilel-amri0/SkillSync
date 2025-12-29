"""CV Analysis Router - Uses CVProcessor core module"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
import uuid
import logging

# Import auth dependencies if available
try:
    from auth.dependencies import optional_auth, get_current_user
    from models import User
    AUTH_AVAILABLE = True
except ImportError:
    AUTH_AVAILABLE = False
    User = None

# Import CVProcessor core module
try:
    from services.cv_processor import CVProcessor
    CV_PROCESSOR_AVAILABLE = True
except ImportError:
    CV_PROCESSOR_AVAILABLE = False
    logging.warning("CVProcessor not available")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["CV Analysis"])

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Initialize CV Processor (singleton pattern)
cv_processor: Optional[CVProcessor] = None


class CVAnalysisRequest(BaseModel):
    """Request model for CV analysis"""
    cv_content: str
    format: str = "text"


class CVAnalysisResponse(BaseModel):
    """Response model for CV analysis"""
    analysis_id: str
    skills: List[dict]
    experience_years: int
    job_titles: List[str]
    education: List[dict]
    summary: str
    personal_info: dict
    analyzed_at: str


def create_cv_analysis(cv_text: str) -> CVAnalysisResponse:
    """Create CV analysis from text content"""
    import re
    
    # Simple skill extraction
    skills = []
    skill_keywords = [
        'python', 'java', 'javascript', 'typescript', 'react', 'angular', 'vue',
        'node', 'django', 'flask', 'fastapi', 'docker', 'kubernetes', 'aws',
        'azure', 'gcp', 'sql', 'mongodb', 'postgresql', 'git', 'ci/cd',
        'machine learning', 'ai', 'data science', 'tensorflow', 'pytorch'
    ]
    
    cv_lower = cv_text.lower()
    for skill in skill_keywords:
        if skill in cv_lower:
            skills.append(skill.title())
    
    # Extract experience years
    experience_years = 0
    exp_matches = re.findall(r'(\d+)\s*(?:years?|ans?)', cv_lower)
    if exp_matches:
        experience_years = max(int(year) for year in exp_matches)
    
    # Extract job titles (simple heuristic)
    job_titles = []
    title_keywords = ['developer', 'engineer', 'analyst', 'manager', 'designer', 'architect']
    for title in title_keywords:
        if title in cv_lower:
            job_titles.append(title.title())
    
    # Extract education
    education = []
    edu_keywords = ['bachelor', 'master', 'phd', 'diploma', 'degree', 'university', 'college']
    for edu in edu_keywords:
        if edu in cv_lower:
            education.append(edu.title())
    
    analysis_id = str(uuid.uuid4())
    
    response = CVAnalysisResponse(
        analysis_id=analysis_id,
        skills=list(set(skills)),
        experience_years=experience_years,
        job_titles=list(set(job_titles)),
        education=list(set(education)),
        summary=f"Found {len(set(skills))} skills and {experience_years} years of experience",
        analyzed_at=datetime.utcnow().isoformat()
    )
    
    return response


@router.post("/analyze")
async def analyze_cv(
    file: UploadFile = File(...),
    extract_skills: bool = True
):
    """
    Analyze uploaded CV file using production CVProcessor.
    
    **Parameters:**
    - **file**: CV file (PDF, DOCX, or TXT)
    - **extract_skills**: Whether to extract skills using NLP (default: true)
    
    **Returns:**
    - Structured CV data with personal info, sections, skills, experience, education
    """
    logger.info(f"📄 Received CV analysis request: {file.filename}")
    
    try:
        # Step 1: Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_extension}. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        
        # Step 2: Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f} MB"
            )
        
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided"
            )
        
        logger.info(f"✅ File validated: {file.filename} ({len(file_content)} bytes)")
        
        # Step 3: Process CV using our core module
        processor = get_cv_processor()
        parsed_cv = await processor.process_cv(
            file_content=file_content,
            filename=file.filename,
            extract_skills=extract_skills
        )
        
        logger.info(f"✅ CV processed successfully: {len(parsed_cv.skills)} skills extracted")
        
        # Step 4: Convert to dict for JSON response
        response_data = {
            "raw_text": parsed_cv.raw_text[:500] + "..." if len(parsed_cv.raw_text) > 500 else parsed_cv.raw_text,
            "personal_info": parsed_cv.personal_info,
            "sections": {
                section_name: {
                    "content": section.content[:200] + "..." if len(section.content) > 200 else section.content,
                    "confidence": section.confidence
                }
                for section_name, section in parsed_cv.sections.items()
            },
            "skills": [
                {
                    "skill": skill.skill,
                    "category": skill.category,
                    "confidence": skill.confidence,
                    "source": skill.source
                }
                for skill in parsed_cv.skills
            ],
            "experience": parsed_cv.experience,
            "education": parsed_cv.education,
            "metadata": parsed_cv.metadata
        }
        
        return {
            "success": True,
            "message": "CV analyzed successfully",
            "data": response_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing CV: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process CV: {str(e)}"
        )


@router.get("/analyze/health")
async def cv_analysis_health():
    """
    Check CV analysis service health.
    
    **Returns:**
    - Service status and capabilities
    """
    try:
        processor = get_cv_processor()
        
        return {
            "status": "healthy",
            "service": "CV Analysis",
            "supported_formats": list(SUPPORTED_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE / (1024*1024),
            "capabilities": {
                "pdf_parsing": True,
                "docx_parsing": True,
                "ocr_fallback": True,
                "skill_extraction": True,
                "section_detection": True,
                "personal_info_extraction": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
