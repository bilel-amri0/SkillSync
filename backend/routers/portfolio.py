"""
Portfolio Generator Router
Handles portfolio generation and download endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import io

# Import Portfolio Generator
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from services.portfolio_generator_v2 import PortfolioGenerator, PortfolioConfig
    PORTFOLIO_GENERATOR_AVAILABLE = True
except ImportError:
    PORTFOLIO_GENERATOR_AVAILABLE = False
    logging.warning("PortfolioGenerator not available")

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter()

# Initialize Portfolio Generator (singleton pattern)
portfolio_generator: Optional[PortfolioGenerator] = None


# Pydantic models
class PersonalInfo(BaseModel):
    """Personal information model."""
    name: str = Field(..., description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location/Address")


class Skill(BaseModel):
    """Skill model."""
    skill: str = Field(..., description="Skill name")
    category: str = Field(default="other", description="Skill category")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class Experience(BaseModel):
    """Work experience model."""
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    start_year: str = Field(..., description="Start date")
    end_year: str = Field(default="Present", description="End date")
    context: str = Field(default="", description="Job description")


class Education(BaseModel):
    """Education model."""
    degree: str = Field(..., description="Degree name")
    institution: str = Field(..., description="Institution name")
    year: Optional[str] = Field(None, description="Graduation year")
    location: Optional[str] = Field(None, description="Institution location")


class PortfolioGenerationRequest(BaseModel):
    """Request model for portfolio generation."""
    cv_data: Dict[str, Any] = Field(..., description="Parsed CV data from /analyze endpoint")
    template_id: str = Field(
        default="modern",
        description="Template identifier (modern, classic, creative, minimal, tech)"
    )
    color_scheme: str = Field(
        default="blue",
        description="Color scheme (blue, green, purple, red, orange)"
    )
    include_photo: bool = Field(default=True, description="Include photo section")
    include_projects: bool = Field(default=True, description="Include projects section")
    include_contact_form: bool = Field(default=True, description="Include contact form")
    dark_mode: bool = Field(default=False, description="Enable dark mode")
    
    class Config:
        schema_extra = {
            "example": {
                "cv_data": {
                    "personal_info": {
                        "name": "John Doe",
                        "email": "john@example.com",
                        "phone": "+1234567890"
                    },
                    "skills": [
                        {"skill": "Python", "category": "programming_languages", "confidence": 0.95}
                    ],
                    "experience": [
                        {
                            "title": "Software Engineer",
                            "company": "Tech Corp",
                            "start_year": "2020",
                            "end_year": "Present",
                            "context": "Developed web applications"
                        }
                    ],
                    "education": [
                        {
                            "degree": "BS Computer Science",
                            "institution": "University",
                            "year": "2020"
                        }
                    ],
                    "sections": {
                        "summary": "Experienced software engineer..."
                    }
                },
                "template_id": "modern",
                "color_scheme": "blue"
            }
        }


def get_portfolio_generator() -> PortfolioGenerator:
    """
    Dependency injection for PortfolioGenerator.
    Creates singleton instance on first call.
    """
    global portfolio_generator
    
    if not PORTFOLIO_GENERATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Portfolio generation service is not available"
        )
    
    if portfolio_generator is None:
        portfolio_generator = PortfolioGenerator()
        logger.info("✅ PortfolioGenerator initialized")
    
    return portfolio_generator


@router.post("/generate-portfolio")
async def generate_portfolio(request: PortfolioGenerationRequest):
    """
    Generate professional portfolio website from CV data.
    
    **Parameters:**
    - **cv_data**: Parsed CV data (from /analyze endpoint)
    - **template_id**: Template choice (modern, classic, creative, minimal, tech)
    - **color_scheme**: Color scheme (blue, green, purple, red, orange)
    - **include_photo**: Include photo section
    - **include_projects**: Include projects section
    - **include_contact_form**: Include contact form
    - **dark_mode**: Enable dark mode
    
    **Returns:**
    - ZIP file download containing complete portfolio website
    
    **Response Headers:**
    - Content-Type: application/zip
    - Content-Disposition: attachment; filename="portfolio_{name}.zip"
    """
    logger.info(f"🎨 Received portfolio generation request: template={request.template_id}, colors={request.color_scheme}")
    
    try:
        # Step 1: Validate inputs
        generator = get_portfolio_generator()
        
        # Validate template
        available_templates = generator.list_available_templates()
        if request.template_id not in available_templates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid template_id: {request.template_id}. Available: {', '.join(available_templates.keys())}"
            )
        
        # Validate color scheme
        available_colors = generator.list_color_schemes()
        if request.color_scheme not in available_colors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid color_scheme: {request.color_scheme}. Available: {', '.join(available_colors)}"
            )
        
        # Validate CV data
        if not request.cv_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="cv_data cannot be empty"
            )
        
        logger.info("✅ Input validation passed")
        
        # Step 2: Create portfolio configuration
        config = PortfolioConfig(
            template_name=request.template_id,
            color_scheme=request.color_scheme,
            include_photo=request.include_photo,
            include_projects=request.include_projects,
            include_contact_form=request.include_contact_form,
            dark_mode=request.dark_mode
        )
        
        # Step 3: Generate portfolio
        result = await generator.generate_portfolio(
            cv_data=request.cv_data,
            config=config
        )
        
        logger.info(f"✅ Portfolio generated: {result.file_size} bytes, {len(result.files_included)} files")
        
        # Step 4: Prepare filename
        name = request.cv_data.get('personal_info', {}).get('name', 'portfolio')
        # Sanitize filename
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"portfolio_{safe_name}.zip"
        
        # Step 5: Return ZIP as streaming response
        return StreamingResponse(
            io.BytesIO(result.zip_bytes),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(result.file_size),
                "X-Portfolio-ID": result.portfolio_id,
                "X-Template-Used": result.template_used,
                "X-Color-Scheme": result.color_scheme
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"❌ Error generating portfolio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate portfolio: {str(e)}"
        )


@router.get("/templates")
async def list_templates():
    """
    List available portfolio templates.
    
    **Returns:**
    - Dictionary of template IDs and descriptions
    """
    generator = get_portfolio_generator()
    templates = generator.list_available_templates()
    
    return {
        "success": True,
        "templates": templates,
        "count": len(templates),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/color-schemes")
async def list_color_schemes():
    """
    List available color schemes.
    
    **Returns:**
    - List of color scheme names
    """
    generator = get_portfolio_generator()
    color_schemes = generator.list_color_schemes()
    
    return {
        "success": True,
        "color_schemes": color_schemes,
        "count": len(color_schemes),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/portfolio/health")
async def portfolio_health():
    """
    Check portfolio generation service health.
    
    **Returns:**
    - Service status and capabilities
    """
    generator = get_portfolio_generator()
    
    return {
        "status": "healthy",
        "service": "Portfolio Generation",
        "available_templates": list(generator.list_available_templates().keys()),
        "available_color_schemes": generator.list_color_schemes(),
        "capabilities": {
            "template_rendering": True,
            "custom_colors": True,
            "zip_generation": True,
            "responsive_design": True
        },
        "timestamp": datetime.utcnow().isoformat()
    }
