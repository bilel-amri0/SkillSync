"""CV Analysis Router - Dynamic ingestion backed by the database."""

from fastapi import APIRouter, UploadFile, File, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from pathlib import Path
import re
from sqlalchemy.orm import Session

from database import get_db, CVAnalysisService

# Import CV Processor core module
try:
    from services.cv_processor import CVProcessor
    CV_PROCESSOR_AVAILABLE = True
except ImportError:
    CV_PROCESSOR_AVAILABLE = False
    logging.warning("CVProcessor not available")

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1", tags=["CV Analysis"])

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Initialize CV Processor (singleton pattern)
cv_processor: Optional['CVProcessor'] = None


class CVTextAnalysisRequest(BaseModel):
    """Payload for text-based CV analysis"""
    cv_content: str = Field(..., min_length=20)
    format: Optional[str] = "text"
    user_id: Optional[str] = None


class CVTextAnalysisResponse(BaseModel):
    """Response returned for CV text analysis"""
    analysis_id: str
    skills: List[str]
    experience_years: int
    job_titles: List[str]
    education: List[str]
    summary: str
    confidence_score: float
    timestamp: str
    learning_focus: List[str] = Field(default_factory=list)
    raw_text_length: int


class CVAnalysisSummary(BaseModel):
    analysis_id: str
    skills: List[str]
    experience_years: int
    summary: Optional[str] = None
    created_at: str


class CVAnalysisListResponse(BaseModel):
    analyses: List[CVAnalysisSummary]
    total: int


def get_cv_processor() -> 'CVProcessor':
    """
    Dependency injection for CVProcessor.
    Creates singleton instance on first call.
    """
    global cv_processor
    
    if not CV_PROCESSOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="CV processing service is not available"
        )
    
    if cv_processor is None:
        cv_processor = CVProcessor()
        logger.info("✅ CVProcessor initialized")
    
    return cv_processor


@router.post("/analyze")
async def analyze_cv(
    file: UploadFile = File(...),
    extract_skills: bool = True,
    db: Session = Depends(get_db)
):
    """
    Analyze uploaded CV file using production CVProcessor.
    
    **Parameters:**
    - **file**: CV file (PDF, DOCX, or TXT)
    - **extract_skills**: Whether to extract skills using NLP (default: true)
    
    **Returns:**
    - Structured CV data with personal info, sections, skills, experience, education
    
    **Example Response:**
    ```json
    {
        "success": true,
        "data": {
            "personal_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "+1234567890"
            },
            "skills": [
                {
                    "skill": "Python",
                    "category": "programming_languages",
                    "confidence": 0.95
                }
            ],
            "experience": [...],
            "education": [...]
        }
    }
    ```
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
        
        analysis = CVAnalysisService.create_analysis(
            db=db,
            filename=file.filename,
            original_text=parsed_cv.raw_text,
            analysis_data=response_data,
        )
        response_data["analysis_id"] = analysis.id
        
        return {
            "success": True,
            "message": "CV analyzed successfully",
            "data": response_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
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


@router.post("/analyze-cv", response_model=CVTextAnalysisResponse)
async def analyze_cv_text(
    request: CVTextAnalysisRequest,
    db: Session = Depends(get_db)
):
    """Analyze raw CV text and persist the result in the database."""
    cv_content = request.cv_content.strip()
    if not cv_content:
        raise HTTPException(status_code=400, detail="cv_content must not be empty")
    
    analysis_payload = _build_text_analysis_payload(cv_content)
    analysis = CVAnalysisService.create_analysis(
        db=db,
        filename="inline-text",
        original_text=cv_content,
        analysis_data=analysis_payload,
        user_id=request.user_id
    )
    analysis_payload["analysis_id"] = analysis.id
    return CVTextAnalysisResponse(**analysis_payload)


@router.get("/cv-analyses", response_model=CVAnalysisListResponse)
async def list_cv_analyses(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List recent CV analyses from the dynamic ingestion database."""
    limit = max(1, min(limit, 100))
    analyses = CVAnalysisService.get_recent_analyses(db, limit)
    summaries: List[CVAnalysisSummary] = []
    for analysis in analyses:
        payload = analysis.analysis_data or {}
        summaries.append(
            CVAnalysisSummary(
                analysis_id=analysis.id,
                skills=_normalize_skills(payload.get("skills")),
                experience_years=int(payload.get("experience_years", 0) or 0),
                summary=payload.get("summary"),
                created_at=analysis.created_at.isoformat(),
            )
        )
    return CVAnalysisListResponse(analyses=summaries, total=len(summaries))


# ---------------------------------------------------------------------------
# Helper functions for lightweight text analysis
# ---------------------------------------------------------------------------
SKILL_KEYWORDS: Dict[str, List[str]] = {
    'python': ['python', 'django', 'flask', 'fastapi'],
    'javascript': ['javascript', 'node.js', 'nodejs', 'typescript', 'react', 'vue', 'angular'],
    'java': ['java', 'spring'],
    'cloud': ['aws', 'azure', 'gcp', 'google cloud', 'heroku'],
    'devops': ['docker', 'kubernetes', 'terraform', 'ansible', 'ci/cd'],
    'data science': ['data science', 'machine learning', 'ml', 'pandas', 'numpy', 'tensorflow', 'pytorch'],
    'sql': ['sql', 'postgresql', 'mysql', 'sqlite'],
    'leadership': ['leadership', 'mentored', 'managed team', 'led'],
}


def _build_text_analysis_payload(cv_text: str) -> Dict[str, Any]:
    skills = _extract_skills(cv_text)
    job_titles = _extract_job_titles(cv_text)
    education = _extract_education(cv_text)
    experience_years = _estimate_experience_years(cv_text, skills)
    summary = _build_summary(cv_text)
    confidence = min(0.96, 0.6 + (len(skills) * 0.03))
    learning_focus = _derive_learning_focus(skills)
    timestamp = datetime.utcnow().isoformat()
    return {
        "analysis_id": "pending",
        "skills": skills,
        "experience_years": experience_years,
        "job_titles": job_titles,
        "education": education,
        "summary": summary,
        "confidence_score": round(confidence, 2),
        "timestamp": timestamp,
        "learning_focus": learning_focus,
        "raw_text_length": len(cv_text)
    }


def _extract_skills(cv_text: str) -> List[str]:
    text_lower = cv_text.lower()
    found = []
    for canonical, keywords in SKILL_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            found.append(canonical.title())
    if not found:
        found.append("Professional Skills")
    return found


def _estimate_experience_years(cv_text: str, skills: List[str]) -> int:
    match = re.search(r"(\d{1,2})\s*(?:\+)?\s*(years|yrs)", cv_text.lower())
    if match:
        return max(int(match.group(1)), 1)
    return max(len(skills) // 2 + 2, 1)


def _extract_job_titles(cv_text: str) -> List[str]:
    titles = []
    patterns = [
        r"senior [a-z\s]+developer",
        r"[a-z\s]+engineer",
        r"data scientist",
        r"product manager",
        r"devops engineer",
        r"full[-\s]?stack developer",
    ]
    text_lower = cv_text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            cleaned = match.strip().title()
            if cleaned not in titles:
                titles.append(cleaned)
    if not titles:
        titles.append("Professional")
    return titles


def _extract_education(cv_text: str) -> List[str]:
    education = []
    keywords = [
        "bachelor",
        "master",
        "phd",
        "b.sc",
        "m.sc",
        "engineer",
    ]
    sentences = re.split(r"[\.;\n]", cv_text)
    for sentence in sentences:
        lowered = sentence.lower()
        if any(keyword in lowered for keyword in keywords):
            education.append(sentence.strip().title())
    return education[:5]


def _build_summary(cv_text: str) -> str:
    snippet = cv_text.strip().split("\n")[0]
    if len(snippet) > 200:
        snippet = snippet[:200] + "..."
    return snippet


def _derive_learning_focus(skills: List[str]) -> List[str]:
    focus = []
    for skill in skills[:3]:
        focus.append(f"Deepen expertise in {skill}")
    return focus or ["Clarify next-step learning goals"]


def _normalize_skills(raw_skills: Any) -> List[str]:
    if not raw_skills:
        return []
    normalized = []
    for skill in raw_skills:
        if isinstance(skill, str):
            normalized.append(skill)
        elif isinstance(skill, dict):
            normalized.append(skill.get("skill") or skill.get("name") or "Skill")
    return normalized
