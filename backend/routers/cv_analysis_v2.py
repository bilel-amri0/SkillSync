"""
CV Analysis Router - Uses production CVProcessor core module
Handles CV upload, parsing, and skill extraction endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Optional
import logging
from datetime import datetime
from pathlib import Path

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
    extract_skills: bool = True
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
