"""
FastAPI router for portfolio generation endpoints
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/portfolio",
    tags=["portfolio"],
)


# Pydantic Models
class PortfolioGenerateRequest(BaseModel):
    """Request to generate a portfolio"""
    cv_data: Dict[str, Any] = Field(..., description="CV analysis data")
    template: Optional[str] = Field(default="modern", description="Template name")
    style: Optional[str] = Field(default="professional", description="Style preference")


class PortfolioResponse(BaseModel):
    """Response after generating portfolio"""
    portfolio_id: str
    html_content: str
    template: str
    generated_at: str
    status: str


class TemplateInfo(BaseModel):
    """Portfolio template information"""
    id: str
    name: str
    description: str
    preview_url: str


class PortfolioListItem(BaseModel):
    """Portfolio list item"""
    id: str
    name: str
    template: str
    created_at: str


def generate_portfolio_html(cv_data: Dict[str, Any], template: str) -> str:
    """Generate HTML portfolio from CV data"""
    
    # Extract basic info
    name = cv_data.get('personal_info', {}).get('name', 'Professional')
    title = cv_data.get('job_titles', ['Developer'])[0] if cv_data.get('job_titles') else 'Professional'
    skills = cv_data.get('skills', [])
    experience = cv_data.get('experience_years', 0)
    summary = cv_data.get('summary', 'Professional with expertise in various technologies.')
    
    # Color schemes based on template
    color_schemes = {
        'modern': {'primary': '#2563eb', 'secondary': '#1e40af', 'bg': '#f8fafc'},
        'classic': {'primary': '#1f2937', 'secondary': '#374151', 'bg': '#f9fafb'},
        'creative': {'primary': '#7c3aed', 'secondary': '#6d28d9', 'bg': '#faf5ff'},
        'minimal': {'primary': '#0f172a', 'secondary': '#334155', 'bg': '#ffffff'},
        'tech': {'primary': '#059669', 'secondary': '#047857', 'bg': '#f0fdf4'}
    }
    
    colors = color_schemes.get(template, color_schemes['modern'])
    
    # Generate HTML
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{name} - Portfolio</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                background: {colors['bg']};
                color: #333;
            }}
            .container {{
                max-width: 900px;
                margin: 40px auto;
                background: white;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 3px solid {colors['primary']};
                padding-bottom: 30px;
                margin-bottom: 40px;
            }}
            .header h1 {{
                font-size: 2.5em;
                color: {colors['primary']};
                margin-bottom: 10px;
            }}
            .header h2 {{
                font-size: 1.5em;
                color: {colors['secondary']};
                font-weight: 400;
                margin-bottom: 10px;
            }}
            .header p {{
                color: #666;
                font-size: 1.1em;
            }}
            .section {{
                margin-bottom: 40px;
            }}
            .section h3 {{
                font-size: 1.8em;
                color: {colors['primary']};
                margin-bottom: 20px;
                border-left: 4px solid {colors['primary']};
                padding-left: 15px;
            }}
            .skills {{
                display: flex;
                flex-wrap: wrap;
                gap: 12px;
                margin-top: 20px;
            }}
            .skill {{
                background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']});
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .summary {{
                background: {colors['bg']};
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid {colors['primary']};
                line-height: 1.8;
                font-size: 1.05em;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #e5e7eb;
                color: #666;
            }}
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    box-shadow: none;
                    margin: 0;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{name}</h1>
                <h2>{title}</h2>
                <p>💼 {experience} years of professional experience</p>
            </div>
            
            <div class="section">
                <h3>About Me</h3>
                <div class="summary">
                    <p>{summary}</p>
                </div>
            </div>
            
            <div class="section">
                <h3>Core Skills</h3>
                <div class="skills">
                    {''.join([f'<span class="skill">{skill}</span>' for skill in (skills[:15] if skills else ['Python', 'JavaScript', 'React'])])}
                </div>
            </div>
            
            <div class="footer">
                <p>Generated with SkillSync Portfolio Generator</p>
                <p style="font-size: 0.9em; margin-top: 10px;">Template: {template.title()} | Generated: {datetime.now().strftime('%B %Y')}</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return html_content


@router.get("/templates", response_model=List[TemplateInfo])
async def get_templates():
    """
    Get available portfolio templates
    
    Returns list of available templates with descriptions
    """
    templates = [
        {
            "id": "modern",
            "name": "Modern",
            "description": "Clean and contemporary design with gradient accents",
            "preview_url": "/templates/modern/preview.png"
        },
        {
            "id": "classic",
            "name": "Classic",
            "description": "Traditional professional layout with elegant typography",
            "preview_url": "/templates/classic/preview.png"
        },
        {
            "id": "creative",
            "name": "Creative",
            "description": "Bold and colorful design for creative professionals",
            "preview_url": "/templates/creative/preview.png"
        },
        {
            "id": "minimal",
            "name": "Minimal",
            "description": "Simple and clean with focus on content",
            "preview_url": "/templates/minimal/preview.png"
        },
        {
            "id": "tech",
            "name": "Tech",
            "description": "Tech-focused design with modern accents",
            "preview_url": "/templates/tech/preview.png"
        }
    ]
    
    return [TemplateInfo(**t) for t in templates]


@router.post("/generate", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
async def generate_portfolio(request: PortfolioGenerateRequest):
    """
    Generate a portfolio from CV data
    
    Args:
        request: PortfolioGenerateRequest with cv_data, template, and style
    
    Returns:
        PortfolioResponse with portfolio_id and HTML content
    """
    try:
        logger.info(f"🎨 Portfolio generation requested with template: {request.template}")
        
        # Extract data from request
        cv_data = request.cv_data
        template = request.template or "modern"
        
        # Generate portfolio HTML
        portfolio_html = generate_portfolio_html(cv_data, template)
        
        # Create response
        portfolio_id = str(uuid.uuid4())
        response = PortfolioResponse(
            portfolio_id=portfolio_id,
            html_content=portfolio_html,
            template=template,
            generated_at=datetime.now().isoformat(),
            status="success"
        )
        
        logger.info(f"✅ Portfolio {portfolio_id} generated successfully")
        return response
        
    except Exception as e:
        logger.error(f"❌ Portfolio generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Portfolio generation failed: {str(e)}"
        )


@router.get("/list", response_model=List[PortfolioListItem])
async def get_portfolios():
    """
    Get list of user's portfolios
    
    Returns list of portfolios (currently returns demo data)
    """
    # Demo data - in production, this would fetch from database
    demo_portfolios = [
        {
            "id": str(uuid.uuid4()),
            "name": "Software Developer Portfolio",
            "template": "modern",
            "created_at": datetime.now().isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "Data Scientist Portfolio",
            "template": "tech",
            "created_at": datetime.now().isoformat()
        }
    ]
    
    return [PortfolioListItem(**p) for p in demo_portfolios]


@router.get("/export/{portfolio_id}")
async def export_portfolio(portfolio_id: str, format: str = "html"):
    """
    Export portfolio in specified format
    
    Args:
        portfolio_id: Portfolio ID
        format: Export format (html or pdf)
    
    Returns:
        Exported portfolio data
    """
    # Mock export functionality
    return {
        "portfolio_id": portfolio_id,
        "format": format,
        "download_url": f"/downloads/{portfolio_id}.{format}",
        "message": f"Portfolio exported as {format.upper()}"
    }
