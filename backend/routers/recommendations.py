"""Recommendations Router - uses persisted CV analyses."""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from auth.dependencies import optional_auth
from models import User
from database import get_db, CVAnalysisService

router = APIRouter(prefix="/api/v1", tags=["Recommendations"])


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    analysis_id: str
    recommendations: Dict[str, Any]
    generated_at: str
    user_profile: Dict[str, Any]
    global_confidence: float
    engine_info: Dict[str, str]


def generate_recommendations(analysis_id: str, cv_data: Dict[str, Any]) -> RecommendationResponse:
    """Generate personalized recommendations based on CV analysis"""
    
    skills = cv_data.get('skills', [])
    experience_years = cv_data.get('experience_years', 0)
    job_titles = cv_data.get('job_titles', [])
    
    # Determine career level
    if experience_years < 2:
        career_level = "junior"
    elif experience_years < 5:
        career_level = "intermediate"
    else:
        career_level = "senior"
    
    # Determine career path
    career_path = "Software Engineer"
    if any(title.lower() in ['manager', 'lead'] for title in job_titles):
        career_path = "Engineering Manager"
    elif any(title.lower() in ['data', 'analyst'] for title in job_titles):
        career_path = "Data Analyst"
    
    recommendations = {
        "LEARNING_RESOURCES": {
            "free_resources": [
                {
                    "title": "freeCodeCamp",
                    "url": "https://freecodecamp.org",
                    "description": "Comprehensive free coding bootcamp",
                    "category": "comprehensive",
                    "time_commitment": "Self-paced",
                    "score": 0.9
                },
                {
                    "title": "MDN Web Docs",
                    "url": "https://developer.mozilla.org",
                    "description": "Authoritative web development documentation",
                    "category": "reference",
                    "time_commitment": "As needed",
                    "score": 0.85
                }
            ],
            "paid_courses": [
                {
                    "platform": "Pluralsight",
                    "description": "Professional technology learning platform",
                    "cost": "$29/month",
                    "category": "professional",
                    "score": 0.88
                },
                {
                    "platform": "Udemy",
                    "description": "Affordable specialized programming courses",
                    "cost": "$10-200",
                    "category": "specialized",
                    "score": 0.82
                }
            ],
            "practice_platforms": [
                {
                    "name": "LeetCode",
                    "url": "https://leetcode.com",
                    "description": "Algorithm and data structure practice",
                    "focus": "Problem solving",
                    "category": "algorithms",
                    "score": 0.9
                },
                {
                    "name": "GitHub",
                    "url": "https://github.com",
                    "description": "Open source contribution and portfolio",
                    "focus": "Real projects",
                    "category": "collaboration",
                    "score": 0.95
                }
            ]
        },
        "CAREER_ROADMAP": {
            "career_path": career_path,
            "current_level": career_level,
            "immediate_steps": [
                {
                    "action": f"Master {skills[0] if skills else 'core'} technology",
                    "timeline": "1-3 months",
                    "priority": "high",
                    "description": f"Essential immediate action for {career_path.lower()} growth",
                    "score": 0.85
                } for _ in range(3)
            ],
            "mid_term_goals": [
                {
                    "goal": "Build portfolio projects",
                    "timeline": "3-6 months",
                    "priority": "medium",
                    "description": "Demonstrate practical skills",
                    "score": 0.75
                } for _ in range(3)
            ],
            "long_term_vision": [
                {
                    "vision": f"Become Senior {career_path}",
                    "timeline": "1-3 years",
                    "priority": "low",
                    "description": "Long-term career milestone",
                    "score": 0.65
                } for _ in range(3)
            ]
        },
        "IMMEDIATE_ACTIONS": [
            {
                "action": "Update LinkedIn profile",
                "category": "profile_optimization",
                "description": "Update profile with industry-trending skills, achievements, and quantified results",
                "estimated_time": "2-3 hours",
                "impact": "Increases recruiter visibility by 40%"
            },
            {
                "action": "Build portfolio website",
                "category": "portfolio",
                "description": "Create professional portfolio showcasing projects with comprehensive documentation",
                "estimated_time": "3-5 days",
                "impact": "Demonstrates real-world capabilities"
            },
            {
                "action": "Practice coding challenges",
                "category": "skill_maintenance",
                "description": "Solve 2-3 problems daily to master patterns and interview preparation",
                "estimated_time": "45-60 min/day",
                "impact": "Interview readiness"
            }
        ],
        "CERTIFICATION_ROADMAP": []
    }
    
    return RecommendationResponse(
        analysis_id=analysis_id,
        recommendations=recommendations,
        generated_at=datetime.utcnow().isoformat(),
        user_profile={
            "skills_count": len(skills),
            "experience_years": experience_years,
            "career_level": career_level,
            "primary_focus": career_path
        },
        global_confidence=0.85,
        engine_info={
            "version": "2.0",
            "mode": "rule-based"
        }
    )


@router.get("/recommendations/{analysis_id}", response_model=RecommendationResponse)
async def get_recommendations(
    analysis_id: str,
    current_user: User = Depends(optional_auth),
    db: Session = Depends(get_db)
):
    """Get personalized recommendations for a CV analysis"""
    analysis = CVAnalysisService.get_analysis(db, analysis_id)
    if not analysis or not analysis.analysis_data:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    recommendations = generate_recommendations(analysis.id, analysis.analysis_data)
    
    return recommendations
