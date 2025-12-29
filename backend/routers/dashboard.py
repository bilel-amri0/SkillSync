"""Dashboard Router backed by dynamic CV analysis data."""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
from collections import Counter
from sqlalchemy.orm import Session

from auth.dependencies import optional_auth
from models import User
from database import get_db, CVAnalysisService

router = APIRouter(prefix="/api/v1", tags=["Dashboard"])


class DashboardResponse(BaseModel):
    """Response model for dashboard data"""
    recent_analyses: List[Dict[str, Any]]
    job_match_count: int
    skills_summary: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    portfolio_status: str
    last_updated: str


@router.get("/dashboard/latest", response_model=DashboardResponse)
async def get_dashboard(
    current_user: User = Depends(optional_auth),
    db: Session = Depends(get_db)
):
    """Get dashboard data backed by the ingestion database."""
    analyses = CVAnalysisService.get_recent_analyses(db, limit=5)
    recent_analyses: List[Dict[str, Any]] = []
    skill_counter: Counter[str] = Counter()
    recommendations_preview: List[Dict[str, Any]] = []
    
    for analysis in analyses:
        payload = analysis.analysis_data or {}
        skills = _normalize_skills(payload.get('skills'))
        skill_counter.update(skills)
        recent_analyses.append({
            "analysis_id": analysis.id,
            "filename": payload.get('personal_info', {}).get('name') or analysis.filename,
            "skills_count": len(skills),
            "experience_years": payload.get('experience_years'),
            "analyzed_at": analysis.created_at.isoformat()
        })
        if not recommendations_preview and payload.get('learning_focus'):
            for focus in payload['learning_focus'][:3]:
                recommendations_preview.append({
                    "action": focus,
                    "source": "learning_focus"
                })
    
    skills_summary = [
        {"skill": skill, "count": count}
        for skill, count in skill_counter.most_common(10)
    ]
    
    return DashboardResponse(
        recent_analyses=recent_analyses,
        job_match_count=len(recent_analyses),
        skills_summary=skills_summary,
        recommendations=recommendations_preview,
        portfolio_status="dynamic",
        last_updated=datetime.utcnow().isoformat()
    )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }

def _normalize_skills(skills: Any) -> List[str]:
    if not skills:
        return []
    normalized: List[str] = []
    for skill in skills:
        if isinstance(skill, str):
            normalized.append(skill)
        elif isinstance(skill, dict):
            normalized.append(skill.get('skill') or skill.get('name') or 'Skill')
    return normalized
