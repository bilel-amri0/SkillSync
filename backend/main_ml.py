"""
SkillSync ML-Integrated Backend
================================
FastAPI backend with real ML pipeline for CV analysis, job matching,
XAI explanations, NLG translation, and ML recommendations.

Replaces rule-based logic with:
- SBERT semantic embeddings
- SpaCy NER skill extraction
- Cosine similarity matching
- SHAP explainability
- T5 text generation
- Embedding-based recommendations
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime
import base64
import uuid
import re
import os

# Environment setup
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(" Environment variables loaded")
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import ML pipeline
try:
    from ml import (
        get_embedding_engine,
        get_ner_extractor,
        get_scoring_engine,
        get_xai_explainer,
        get_translator,
        get_recommender,
        EmbeddingResult,
        NERResult,
        MatchScore,
        XAIExplanation,
        TranslationResult,
        RecommendationSet
    )
    ML_AVAILABLE = True
    logger.info(" ML Pipeline loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f" ML Pipeline not available: {e}")

# Import job service
from services.multi_job_api_service import get_job_service, JobResult

# Import interview system
try:
    from skillsync.interviews import realtime as interview_realtime
    from skillsync.interviews import routes as interview_routes
    INTERVIEWS_AVAILABLE = True
except ImportError:
    INTERVIEWS_AVAILABLE = False
    logger.warning(" Interview system not available")

# FastAPI app
app = FastAPI(
    title="SkillSync ML API",
    description="AI-Powered CV Analysis & Job Matching with Real Machine Learning",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (TODO: Replace with PostgreSQL + pgvector)
cv_analysis_storage: Dict[str, Dict] = {}
job_match_storage: Dict[str, Dict] = {}
xai_explanation_storage: Dict[str, Dict] = {}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class CVAnalysisRequest(BaseModel):
    cv_content: str

class CVAnalysisResponse(BaseModel):
    analysis_id: str
    skills: List[str]
    skills_by_category: Dict[str, List[str]]
    experience_years: int
    job_titles: List[str]
    summary: str
    education: List[str]
    embedding_generated: bool
    ner_confidence: Optional[float] = None

class JobSearchRequest(BaseModel):
    what: str
    where: str = ""
    remote_only: bool = False
    num_pages: int = 1

class JobMatchRequest(BaseModel):
    analysis_id: str
    job_id: str

class JobMatchResponse(BaseModel):
    match_id: str
    analysis_id: str
    job_id: str
    overall_score: float
    semantic_similarity: float
    skill_overlap: float
    experience_match: float
    matched_skills: List[str]
    missing_skills: List[str]
    explanation: str
    confidence: float

class XAIExplanationResponse(BaseModel):
    match_id: str
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]
    feature_importance: Dict[str, float]
    explanation_text: str
    confidence: float

class TranslationRequest(BaseModel):
    text: str
    style: str = "professional"  # professional, technical, creative, concise, impactful, executive

class TranslationResponse(BaseModel):
    original: str
    translated: str
    style: str
    model_used: str
    confidence: float

class RecommendationsResponse(BaseModel):
    analysis_id: str
    projects: List[Dict[str, Any]]
    certifications: List[Dict[str, Any]]
    courses: List[Dict[str, Any]]
    skills_to_learn: List[str]
    learning_path: List[Dict[str, str]]


# ============================================================================
# ML-POWERED CV ANALYSIS
# ============================================================================

@app.post("/api/v1/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv_text(request: CVAnalysisRequest):
    """
    Analyze CV with real ML pipeline:
    1. Generate SBERT embedding (384-dim vector)
    2. Extract skills using SpaCy NER
    3. Store results for matching
    """
    try:
        content_length = len(request.cv_content)
        logger.info(f" CV analysis request: {content_length} characters")
        
        if not ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML pipeline not available")
        
        analysis_id = str(uuid.uuid4())
        
        # 1. Generate embedding
        embedding_engine = get_embedding_engine()
        cv_data = {"text": request.cv_content}
        embedding_result = embedding_engine.encode_cv(cv_data)
        
        # 2. Extract skills with NER
        ner_extractor = get_ner_extractor()
        ner_result = ner_extractor.extract_from_cv(cv_data)
        
        # 3. Extract metadata (experience, titles, education)
        experience_years = _extract_experience(request.cv_content)
        job_titles = _extract_titles(request.cv_content)
        education = _extract_education(request.cv_content)
        summary = _generate_summary(request.cv_content, ner_result.all_skills[:5])
        
        # 4. Store analysis
        cv_analysis = {
            "analysis_id": analysis_id,
            "raw_text": request.cv_content,
            "embedding": embedding_result.embedding.tolist(),
            "skills": ner_result.all_skills,
            "skills_by_category": ner_result.skills_by_category,
            "experience_years": experience_years,
            "job_titles": job_titles,
            "education": education,
            "summary": summary,
            "ner_confidence": ner_result.average_confidence,
            "created_at": datetime.now().isoformat()
        }
        cv_analysis_storage[analysis_id] = cv_analysis
        
        logger.info(f" CV analyzed: {len(ner_result.all_skills)} skills found")
        
        return CVAnalysisResponse(
            analysis_id=analysis_id,
            skills=ner_result.all_skills,
            skills_by_category=ner_result.skills_by_category,
            experience_years=experience_years,
            job_titles=job_titles,
            summary=summary,
            education=education,
            embedding_generated=True,
            ner_confidence=ner_result.average_confidence
        )
        
    except Exception as e:
        logger.error(f" CV analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/upload-cv", response_model=CVAnalysisResponse)
async def upload_cv_file(file: UploadFile = File(...)):
    """Upload and analyze CV file"""
    try:
        content = await file.read()
        
        # Extract text from file
        if file.filename.endswith('.txt'):
            cv_text = content.decode('utf-8')
        elif file.filename.endswith('.pdf'):
            # TODO: Add PDF parsing
            cv_text = content.decode('utf-8', errors='ignore')
        else:
            cv_text = content.decode('utf-8', errors='ignore')
        
        # Reuse analyze_cv_text logic
        request = CVAnalysisRequest(cv_content=cv_text)
        return await analyze_cv_text(request)
        
    except Exception as e:
        logger.error(f" CV upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# JOB SEARCH & MATCHING
# ============================================================================

@app.get("/api/v1/jobs/search")
async def search_jobs(
    what: str = Query(..., description="Job title or keywords"),
    where: str = Query("", description="Location"),
    remote_only: bool = Query(False),
    num_pages: int = Query(1, ge=1, le=5)
):
    """Search jobs from multiple APIs"""
    try:
        job_service = get_job_service()
        
        results = await job_service.search_multi_sources(
            what=what,
            where=where,
            remote_only=remote_only,
            num_pages=num_pages
        )
        
        logger.info(f" Found {len(results)} jobs from {len(set(r.source for r in results))} sources")
        
        return {
            "total": len(results),
            "jobs": [r.model_dump() for r in results]
        }
        
    except Exception as e:
        logger.error(f" Job search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/match", response_model=JobMatchResponse)
async def match_cv_to_job(request: JobMatchRequest):
    """
    Match CV to job using ML scoring:
    - 50% Semantic similarity (SBERT embeddings)
    - 35% Skill overlap (NER)
    - 15% Experience match
    """
    try:
        if not ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML pipeline not available")
        
        # Get CV analysis
        cv_analysis = cv_analysis_storage.get(request.analysis_id)
        if not cv_analysis:
            raise HTTPException(status_code=404, detail="CV analysis not found")
        
        # Get job (TODO: Fetch from job_match_storage or API)
        # For now, create mock job data
        job_data = {
            "job_id": request.job_id,
            "title": "Senior Full Stack Developer",
            "description": "Looking for experienced developer with Python, React, AWS skills",
            "skills_required": ["Python", "React", "AWS", "Docker", "PostgreSQL"],
            "min_experience_years": 5
        }
        
        # Score match
        scoring_engine = get_scoring_engine()
        cv_data = {
            "text": cv_analysis["raw_text"],
            "skills": cv_analysis["skills"],
            "experience_years": cv_analysis["experience_years"],
            "embedding": cv_analysis["embedding"]
        }
        
        match_score = scoring_engine.score_cv_job_match(cv_data, job_data)
        
        # Store match
        match_id = str(uuid.uuid4())
        job_match = {
            "match_id": match_id,
            "analysis_id": request.analysis_id,
            "job_id": request.job_id,
            "overall_score": match_score.overall_score,
            "semantic_similarity": match_score.semantic_similarity,
            "skill_overlap": match_score.skill_overlap,
            "experience_match": match_score.experience_match,
            "matched_skills": match_score.matched_skills,
            "missing_skills": match_score.missing_skills,
            "explanation": match_score.explanation,
            "confidence": match_score.confidence,
            "factors": match_score.factors,
            "created_at": datetime.now().isoformat()
        }
        job_match_storage[match_id] = job_match
        
        logger.info(f" Match scored: {match_score.overall_score:.1f}%")
        
        return JobMatchResponse(**job_match)
        
    except Exception as e:
        logger.error(f" Matching failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# XAI EXPLAINABILITY
# ============================================================================

@app.get("/api/v1/xai/{match_id}", response_model=XAIExplanationResponse)
async def get_xai_explanation(match_id: str):
    """
    Get SHAP-based explanation for match score:
    - Feature importance (which factors matter most)
    - Top positive factors (why score is high)
    - Top negative factors (why score is not higher)
    """
    try:
        if not ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML pipeline not available")
        
        # Get match
        job_match = job_match_storage.get(match_id)
        if not job_match:
            raise HTTPException(status_code=404, detail="Match not found")
        
        # Get CV and job data
        cv_analysis = cv_analysis_storage.get(job_match["analysis_id"])
        
        # Generate XAI explanation
        xai_explainer = get_xai_explainer()
        explanation = xai_explainer.explain_match_score(
            match_score_data=job_match,
            cv_data=cv_analysis,
            job_data={"job_id": job_match["job_id"]}  # TODO: Get full job data
        )
        
        # Store explanation
        xai_data = {
            "match_id": match_id,
            "top_positive_factors": [f.__dict__ for f in explanation.top_positive_factors],
            "top_negative_factors": [f.__dict__ for f in explanation.top_negative_factors],
            "feature_importance": explanation.feature_importance,
            "explanation_text": explanation.explanation_text,
            "confidence": explanation.confidence
        }
        xai_explanation_storage[match_id] = xai_data
        
        logger.info(f" XAI explanation generated for match {match_id}")
        
        return XAIExplanationResponse(**xai_data)
        
    except Exception as e:
        logger.error(f" XAI explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# NLG TRANSLATION
# ============================================================================

@app.post("/api/v1/translate", response_model=TranslationResponse)
async def translate_experience(request: TranslationRequest):
    """
    Translate/rewrite text using T5 NLG:
    - Styles: professional, technical, creative, concise, impactful, executive
    - Uses google/flan-t5-base for generation
    """
    try:
        if not ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML pipeline not available")
        
        translator = get_translator()
        result = translator.translate(
            text=request.text,
            style=request.style
        )
        
        logger.info(f" Translated to {request.style} style")
        
        return TranslationResponse(
            original=result.original_text,
            translated=result.translated_text,
            style=result.style,
            model_used=result.model_used,
            confidence=result.confidence
        )
        
    except Exception as e:
        logger.error(f" Translation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ML RECOMMENDATIONS
# ============================================================================

@app.get("/api/v1/recommendations/{analysis_id}", response_model=RecommendationsResponse)
async def get_recommendations(analysis_id: str, target_role: str = "Full Stack Developer"):
    """
    Get ML-powered recommendations:
    - Projects (embedding similarity)
    - Certifications (skill gap analysis)
    - Courses (learning path)
    - Skills to learn (NER + target role analysis)
    """
    try:
        if not ML_AVAILABLE:
            raise HTTPException(status_code=503, detail="ML pipeline not available")
        
        # Get CV analysis
        cv_analysis = cv_analysis_storage.get(analysis_id)
        if not cv_analysis:
            raise HTTPException(status_code=404, detail="CV analysis not found")
        
        # Generate recommendations
        recommender = get_recommender()
        cv_data = {
            "text": cv_analysis["raw_text"],
            "skills": cv_analysis["skills"],
            "embedding": cv_analysis["embedding"]
        }
        
        recommendations = recommender.recommend_for_cv(cv_data, target_role)
        
        logger.info(f" Generated {len(recommendations.projects)} projects, {len(recommendations.certifications)} certs")
        
        return RecommendationsResponse(
            analysis_id=analysis_id,
            projects=[p.__dict__ for p in recommendations.projects],
            certifications=[c.__dict__ for c in recommendations.certifications],
            courses=[c.__dict__ for c in recommendations.courses],
            skills_to_learn=recommendations.skills_to_learn,
            learning_path=recommendations.learning_path
        )
        
    except Exception as e:
        logger.error(f" Recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/status")
async def get_status():
    """System status"""
    ml_status = {}
    if ML_AVAILABLE:
        try:
            ml_status = {
                "embeddings": get_embedding_engine() is not None,
                "ner": get_ner_extractor() is not None,
                "scoring": get_scoring_engine() is not None,
                "xai": get_xai_explainer() is not None,
                "translator": get_translator() is not None,
                "recommender": get_recommender() is not None
            }
        except Exception as e:
            ml_status = {"error": str(e)}
    
    return {
        "version": "2.0.0",
        "ml_enabled": ML_AVAILABLE,
        "ml_components": ml_status,
        "job_apis": get_job_service().get_enabled_apis(),
        "cv_analyses_stored": len(cv_analysis_storage),
        "job_matches_stored": len(job_match_storage),
        "interviews_available": INTERVIEWS_AVAILABLE
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_experience(text: str) -> int:
    """Extract years of experience from CV text"""
    matches = re.findall(r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', text, re.IGNORECASE)
    if matches:
        return max(int(x) for x in matches)
    return 3  # Default

def _extract_titles(text: str) -> List[str]:
    """Extract job titles from CV"""
    patterns = [
        r'(senior|junior|lead|principal)?\s*(software|web|full stack|backend|frontend|data)\s*(developer|engineer|analyst)',
        r'(project|product|technical)\s*manager',
        r'(devops|system)\s*engineer'
    ]
    titles = []
    for line in text.split('\n')[:30]:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                titles.append(match.group(0))
                break
    return titles[:3] if titles else ["Software Developer"]

def _extract_education(text: str) -> List[str]:
    """Extract education from CV"""
    education = []
    keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
    for line in text.split('\n'):
        if any(kw in line.lower() for kw in keywords) and len(line.strip()) > 10:
            education.append(line.strip())
            if len(education) >= 3:
                break
    return education

def _generate_summary(text: str, top_skills: List[str]) -> str:
    """Generate CV summary"""
    skills_str = ", ".join(top_skills[:5])
    return f"Professional with expertise in {skills_str}. {text[:200]}..."


# ============================================================================
# MOUNT INTERVIEW ROUTES (if available)
# ============================================================================

if INTERVIEWS_AVAILABLE:
    app.include_router(interview_routes.router, prefix="/api/interviews", tags=["Interviews"])
    logger.info(" Interview routes mounted")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
