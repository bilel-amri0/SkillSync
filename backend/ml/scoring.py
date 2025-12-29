"""
SkillSync ML Pipeline - ML Scoring Engine
==========================================
Real ML scoring using cosine similarity of embeddings.
NO random numbers, NO templates - pure semantic matching.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .embeddings import SemanticEmbeddingEngine, EmbeddingResult, get_embedding_engine
from .ner import SkillNERExtractor, NERResult, get_ner_extractor

logger = logging.getLogger(__name__)


@dataclass
class MatchScore:
    """Individual match scoring result"""
    overall_score: float  # 0-100
    semantic_similarity: float  # 0-1
    skill_overlap: float  # 0-1
    experience_match: float  # 0-1
    confidence: float  # 0-1
    explanation: str
    factors: Dict[str, float]  # Contributing factors


@dataclass
class JobMatchResult:
    """Complete job matching result"""
    job_id: str
    job_title: str
    match_score: MatchScore
    matched_skills: List[str]
    missing_skills: List[str]
    timestamp: str


class MLScoringEngine:
    """
    Real ML-based scoring using semantic embeddings.
    
    Algorithm:
    1. Generate embeddings for CV and job
    2. Calculate cosine similarity
    3. Analyze skill overlap via NER
    4. Weight factors and compute final score
    5. Provide explainability
    """
    
    def __init__(
        self,
        embedding_engine: Optional[SemanticEmbeddingEngine] = None,
        ner_extractor: Optional[SkillNERExtractor] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ML scoring engine.
        
        Args:
            embedding_engine: Semantic embedding engine
            ner_extractor: NER skill extractor
            weights: Scoring weights (semantic, skills, experience)
        """
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.ner_extractor = ner_extractor or get_ner_extractor()
        
        # Default weights (must sum to 1.0)
        self.weights = weights or {
            "semantic_similarity": 0.50,  # 50% weight on semantic match
            "skill_overlap": 0.35,        # 35% weight on skill overlap
            "experience_match": 0.15      # 15% weight on experience level
        }
        
        logger.info("✅ MLScoringEngine initialized")
    
    def score_cv_job_match(
        self,
        cv_data: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> MatchScore:
        """
        Score how well a CV matches a job using ML.
        
        Args:
            cv_data: CV data dictionary
            job_data: Job data dictionary
            
        Returns:
            MatchScore with detailed scoring
        """
        factors = {}
        
        # 1. Semantic Similarity (embedding-based)
        try:
            cv_embedding = self.embedding_engine.encode_cv(cv_data)
            job_embedding = self.embedding_engine.encode_job(job_data)
            
            semantic_sim = self.embedding_engine.cosine_similarity(
                cv_embedding,
                job_embedding
            )
            factors["semantic_similarity"] = semantic_sim
            logger.info(f"Semantic similarity: {semantic_sim:.3f}")
            
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")
            semantic_sim = 0.0
            factors["semantic_similarity"] = 0.0
        
        # 2. Skill Overlap (NER-based)
        try:
            # Extract skills from CV
            cv_ner_result = self.ner_extractor.extract_from_cv(cv_data)
            cv_skills = set(s.skill_name for s in cv_ner_result.skills)
            
            # Extract skills from job
            job_text = f"{job_data.get('title', '')} {job_data.get('description', '')}"
            if job_data.get('skills_required'):
                skills_req = job_data['skills_required']
                if isinstance(skills_req, list):
                    job_text += " " + " ".join(skills_req)
                elif isinstance(skills_req, str):
                    job_text += " " + skills_req
            
            job_skills_extracted = self.ner_extractor.extract_from_text(job_text, source="job")
            job_skills = set(s.skill_name for s in job_skills_extracted)
            
            # Calculate overlap
            if len(job_skills) > 0:
                matched = cv_skills.intersection(job_skills)
                skill_overlap = len(matched) / len(job_skills)
            else:
                skill_overlap = 0.5  # Neutral if no job skills
            
            factors["skill_overlap"] = skill_overlap
            factors["matched_skills_count"] = len(cv_skills.intersection(job_skills))
            factors["missing_skills_count"] = len(job_skills - cv_skills)
            
            logger.info(f"Skill overlap: {skill_overlap:.3f} ({len(matched)}/{len(job_skills)})")
            
        except Exception as e:
            logger.error(f"Skill overlap failed: {e}")
            skill_overlap = 0.0
            factors["skill_overlap"] = 0.0
        
        # 3. Experience Match (years/seniority)
        try:
            cv_years = cv_data.get("years_experience", 0)
            job_years_min = job_data.get("min_experience_years", 0)
            job_years_max = job_data.get("max_experience_years", job_years_min + 5)
            
            if cv_years >= job_years_min and cv_years <= job_years_max:
                experience_match = 1.0
            elif cv_years < job_years_min:
                # Penalty for under-qualified
                diff = job_years_min - cv_years
                experience_match = max(0.0, 1.0 - (diff * 0.15))
            else:
                # Slight penalty for overqualified
                diff = cv_years - job_years_max
                experience_match = max(0.5, 1.0 - (diff * 0.05))
            
            factors["experience_match"] = experience_match
            factors["cv_years"] = cv_years
            factors["job_years_required"] = job_years_min
            
            logger.info(f"Experience match: {experience_match:.3f}")
            
        except Exception as e:
            logger.error(f"Experience match failed: {e}")
            experience_match = 0.7  # Neutral
            factors["experience_match"] = 0.7
        
        # 4. Compute weighted score
        overall_score = (
            semantic_sim * self.weights["semantic_similarity"] +
            skill_overlap * self.weights["skill_overlap"] +
            experience_match * self.weights["experience_match"]
        )
        
        # Convert to 0-100 scale
        overall_score_pct = overall_score * 100.0
        
        # Calculate confidence (based on data availability)
        confidence = 1.0
        if not cv_data.get("text") and not cv_data.get("experience"):
            confidence *= 0.7
        if not job_data.get("description"):
            confidence *= 0.8
        factors["confidence"] = confidence
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_score_pct,
            semantic_sim,
            skill_overlap,
            experience_match,
            factors
        )
        
        logger.info(f"✅ Final score: {overall_score_pct:.2f}%")
        
        return MatchScore(
            overall_score=overall_score_pct,
            semantic_similarity=semantic_sim,
            skill_overlap=skill_overlap,
            experience_match=experience_match,
            confidence=confidence,
            explanation=explanation,
            factors=factors
        )
    
    def score_cv_jobs_batch(
        self,
        cv_data: Dict[str, Any],
        jobs_data: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[JobMatchResult]:
        """
        Score CV against multiple jobs (efficient batching).
        
        Args:
            cv_data: CV data
            jobs_data: List of job data
            top_n: Return only top N matches
            
        Returns:
            List of JobMatchResult, sorted by score
        """
        results = []
        
        for job_data in jobs_data:
            try:
                match_score = self.score_cv_job_match(cv_data, job_data)
                
                # Get matched/missing skills
                cv_ner_result = self.ner_extractor.extract_from_cv(cv_data)
                cv_skills = set(s.skill_name for s in cv_ner_result.skills)
                
                job_text = f"{job_data.get('title', '')} {job_data.get('description', '')}"
                job_skills_extracted = self.ner_extractor.extract_from_text(job_text, source="job")
                job_skills = set(s.skill_name for s in job_skills_extracted)
                
                matched_skills = list(cv_skills.intersection(job_skills))
                missing_skills = list(job_skills - cv_skills)
                
                results.append(JobMatchResult(
                    job_id=job_data.get("id", "unknown"),
                    job_title=job_data.get("title", "Unknown Position"),
                    match_score=match_score,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    timestamp=datetime.utcnow().isoformat()
                ))
                
            except Exception as e:
                logger.error(f"Failed to score job {job_data.get('id')}: {e}")
                continue
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.match_score.overall_score, reverse=True)
        
        # Return top N if specified
        if top_n:
            results = results[:top_n]
        
        logger.info(f"✅ Scored {len(results)} jobs")
        return results
    
    def _generate_explanation(
        self,
        overall_score: float,
        semantic_sim: float,
        skill_overlap: float,
        experience_match: float,
        factors: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation of the score.
        
        Args:
            overall_score: Final score (0-100)
            semantic_sim: Semantic similarity (0-1)
            skill_overlap: Skill overlap (0-1)
            experience_match: Experience match (0-1)
            factors: Additional factors
            
        Returns:
            Explanation string
        """
        parts = []
        
        # Overall assessment
        if overall_score >= 85:
            parts.append("Excellent match!")
        elif overall_score >= 70:
            parts.append("Strong match.")
        elif overall_score >= 50:
            parts.append("Moderate match.")
        else:
            parts.append("Weak match.")
        
        # Semantic similarity
        if semantic_sim >= 0.8:
            parts.append("CV and job are semantically very similar.")
        elif semantic_sim >= 0.6:
            parts.append("CV and job have good semantic overlap.")
        elif semantic_sim >= 0.4:
            parts.append("CV and job have some semantic similarity.")
        else:
            parts.append("CV and job are semantically different.")
        
        # Skill overlap
        matched = factors.get("matched_skills_count", 0)
        missing = factors.get("missing_skills_count", 0)
        
        if matched > 0:
            parts.append(f"Matched {matched} required skills.")
        if missing > 0:
            parts.append(f"Missing {missing} skills.")
        
        # Experience
        cv_years = factors.get("cv_years", 0)
        job_years = factors.get("job_years_required", 0)
        
        if experience_match >= 0.9:
            parts.append(f"Experience level ({cv_years}y) matches requirements.")
        elif cv_years < job_years:
            parts.append(f"Under-qualified by {job_years - cv_years} years.")
        else:
            parts.append(f"Overqualified by {cv_years - job_years} years.")
        
        return " ".join(parts)


# Global singleton
_global_scoring_engine: Optional[MLScoringEngine] = None


def get_scoring_engine() -> MLScoringEngine:
    """
    Get or create global scoring engine (singleton).
    
    Returns:
        MLScoringEngine instance
    """
    global _global_scoring_engine
    
    if _global_scoring_engine is None:
        _global_scoring_engine = MLScoringEngine()
    
    return _global_scoring_engine
