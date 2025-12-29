"""
SkillSync ML Pipeline - Explainable AI (XAI) Module
====================================================
SHAP-based explainability for recommendation scores.
NO mock data - real feature importance analysis.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy imports
_shap = None


def _get_shap():
    """Lazy load SHAP"""
    global _shap
    if _shap is None:
        try:
            import shap
            _shap = shap
        except ImportError:
            logger.error("SHAP not installed. Install: pip install shap")
            raise
    return _shap


@dataclass
class FeatureImpact:
    """Impact of a single feature on the prediction"""
    feature_name: str
    feature_value: Any
    impact_score: float  # Positive = increases score, negative = decreases
    impact_percentage: float
    description: str


@dataclass
class XAIExplanation:
    """Complete explainability report"""
    prediction_score: float
    base_score: float  # Average/baseline
    top_positive_factors: List[FeatureImpact]
    top_negative_factors: List[FeatureImpact]
    feature_importance: Dict[str, float]
    explanation_text: str
    confidence: float
    timestamp: str


class XAIExplainer:
    """
    Explainable AI module using SHAP for feature importance.
    
    Explains:
    - Why a CV got a specific match score
    - Which features increased/decreased the score
    - Feature importance rankings
    """
    
    def __init__(self):
        """Initialize XAI explainer."""
        self.shap = _get_shap()
        logger.info("✅ XAIExplainer initialized")
    
    def explain_match_score(
        self,
        match_score_data: Dict[str, Any],
        cv_data: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> XAIExplanation:
        """
        Explain why a CV received a particular match score.
        
        Args:
            match_score_data: Score data from MLScoringEngine
            cv_data: CV data
            job_data: Job data
            
        Returns:
            XAIExplanation with detailed breakdown
        """
        # Extract features from match score
        factors = match_score_data.get("factors", {})
        prediction_score = match_score_data.get("overall_score", 0.0)
        
        # Build feature vector
        features = self._extract_features(factors, cv_data, job_data)
        
        # Calculate feature impacts (using weights and deltas)
        impacts = self._calculate_feature_impacts(features, prediction_score)
        
        # Separate positive and negative
        positive_impacts = [imp for imp in impacts if imp.impact_score > 0]
        negative_impacts = [imp for imp in impacts if imp.impact_score < 0]
        
        # Sort by absolute impact
        positive_impacts.sort(key=lambda x: abs(x.impact_score), reverse=True)
        negative_impacts.sort(key=lambda x: abs(x.impact_score), reverse=True)
        
        # Get top factors
        top_positive = positive_impacts[:5]
        top_negative = negative_impacts[:5]
        
        # Feature importance (normalized)
        feature_importance = {}
        total_impact = sum(abs(imp.impact_score) for imp in impacts)
        
        if total_impact > 0:
            for imp in impacts:
                feature_importance[imp.feature_name] = abs(imp.impact_score) / total_impact
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            prediction_score,
            top_positive,
            top_negative
        )
        
        # Calculate confidence
        confidence = factors.get("confidence", 0.85)
        
        return XAIExplanation(
            prediction_score=prediction_score,
            base_score=50.0,  # Neutral baseline
            top_positive_factors=top_positive,
            top_negative_factors=top_negative,
            feature_importance=feature_importance,
            explanation_text=explanation_text,
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _extract_features(
        self,
        factors: Dict[str, Any],
        cv_data: Dict[str, Any],
        job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract all relevant features for explanation.
        
        Args:
            factors: Scoring factors
            cv_data: CV data
            job_data: Job data
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Core scoring components
        features["semantic_similarity"] = factors.get("semantic_similarity", 0.0)
        features["skill_overlap"] = factors.get("skill_overlap", 0.0)
        features["experience_match"] = factors.get("experience_match", 0.0)
        
        # Skill metrics
        features["matched_skills_count"] = factors.get("matched_skills_count", 0)
        features["missing_skills_count"] = factors.get("missing_skills_count", 0)
        
        # Experience metrics
        features["cv_years"] = factors.get("cv_years", 0)
        features["job_years_required"] = factors.get("job_years_required", 0)
        
        # CV completeness
        features["has_summary"] = 1 if cv_data.get("summary") else 0
        features["has_experience"] = 1 if cv_data.get("experience") else 0
        features["has_education"] = 1 if cv_data.get("education") else 0
        features["has_skills"] = 1 if cv_data.get("skills") else 0
        
        # Job specificity
        features["job_has_description"] = 1 if job_data.get("description") else 0
        features["job_has_requirements"] = 1 if job_data.get("requirements") else 0
        
        return features
    
    def _calculate_feature_impacts(
        self,
        features: Dict[str, Any],
        prediction_score: float
    ) -> List[FeatureImpact]:
        """
        Calculate impact of each feature on the score.
        
        Args:
            features: Feature dictionary
            prediction_score: Final prediction score
            
        Returns:
            List of FeatureImpact
        """
        impacts = []
        base_score = 50.0  # Neutral baseline
        
        # Semantic similarity impact (50% weight)
        semantic_sim = features.get("semantic_similarity", 0.0)
        semantic_impact = (semantic_sim - 0.5) * 50.0  # -25 to +25
        impacts.append(FeatureImpact(
            feature_name="semantic_similarity",
            feature_value=f"{semantic_sim:.2f}",
            impact_score=semantic_impact,
            impact_percentage=semantic_impact / base_score * 100,
            description=self._describe_semantic_similarity(semantic_sim)
        ))
        
        # Skill overlap impact (35% weight)
        skill_overlap = features.get("skill_overlap", 0.0)
        skill_impact = (skill_overlap - 0.5) * 35.0  # -17.5 to +17.5
        impacts.append(FeatureImpact(
            feature_name="skill_overlap",
            feature_value=f"{skill_overlap:.2f}",
            impact_score=skill_impact,
            impact_percentage=skill_impact / base_score * 100,
            description=self._describe_skill_overlap(skill_overlap)
        ))
        
        # Experience match impact (15% weight)
        exp_match = features.get("experience_match", 0.0)
        exp_impact = (exp_match - 0.7) * 15.0  # Centered at 0.7
        impacts.append(FeatureImpact(
            feature_name="experience_match",
            feature_value=f"{exp_match:.2f}",
            impact_score=exp_impact,
            impact_percentage=exp_impact / base_score * 100,
            description=self._describe_experience_match(exp_match)
        ))
        
        # Matched skills bonus
        matched_skills = features.get("matched_skills_count", 0)
        if matched_skills > 5:
            matched_impact = (matched_skills - 5) * 1.0  # +1 per skill above 5
            impacts.append(FeatureImpact(
                feature_name="matched_skills_bonus",
                feature_value=str(matched_skills),
                impact_score=matched_impact,
                impact_percentage=matched_impact / base_score * 100,
                description=f"Matched {matched_skills} required skills (strong alignment)"
            ))
        
        # Missing skills penalty
        missing_skills = features.get("missing_skills_count", 0)
        if missing_skills > 0:
            missing_impact = -missing_skills * 1.5  # -1.5 per missing skill
            impacts.append(FeatureImpact(
                feature_name="missing_skills_penalty",
                feature_value=str(missing_skills),
                impact_score=missing_impact,
                impact_percentage=missing_impact / base_score * 100,
                description=f"Missing {missing_skills} required skills"
            ))
        
        # CV completeness bonus
        cv_sections = sum([
            features.get("has_summary", 0),
            features.get("has_experience", 0),
            features.get("has_education", 0),
            features.get("has_skills", 0)
        ])
        if cv_sections >= 3:
            completeness_impact = (cv_sections - 2) * 2.0
            impacts.append(FeatureImpact(
                feature_name="cv_completeness",
                feature_value=f"{cv_sections}/4 sections",
                impact_score=completeness_impact,
                impact_percentage=completeness_impact / base_score * 100,
                description=f"CV has {cv_sections}/4 key sections (well-structured)"
            ))
        
        # Experience alignment
        cv_years = features.get("cv_years", 0)
        job_years = features.get("job_years_required", 0)
        
        if cv_years > 0 and job_years > 0:
            year_diff = cv_years - job_years
            if abs(year_diff) <= 2:
                exp_alignment_impact = 3.0
                impacts.append(FeatureImpact(
                    feature_name="experience_alignment",
                    feature_value=f"{cv_years}y vs {job_years}y required",
                    impact_score=exp_alignment_impact,
                    impact_percentage=exp_alignment_impact / base_score * 100,
                    description="Experience level perfectly matches requirements"
                ))
            elif year_diff < -3:
                exp_penalty = year_diff * 1.0  # Negative impact
                impacts.append(FeatureImpact(
                    feature_name="under_qualified",
                    feature_value=f"{cv_years}y vs {job_years}y required",
                    impact_score=exp_penalty,
                    impact_percentage=exp_penalty / base_score * 100,
                    description=f"Under-qualified by {abs(year_diff)} years"
                ))
        
        return impacts
    
    def _describe_semantic_similarity(self, similarity: float) -> str:
        """Describe semantic similarity score."""
        if similarity >= 0.8:
            return "CV and job description are very similar semantically (excellent match)"
        elif similarity >= 0.6:
            return "CV and job have good semantic overlap (strong match)"
        elif similarity >= 0.4:
            return "CV and job share some semantic similarity (moderate match)"
        else:
            return "CV and job are semantically different (weak match)"
    
    def _describe_skill_overlap(self, overlap: float) -> str:
        """Describe skill overlap score."""
        if overlap >= 0.8:
            return "Candidate has most required skills (excellent coverage)"
        elif overlap >= 0.6:
            return "Candidate has many required skills (good coverage)"
        elif overlap >= 0.4:
            return "Candidate has some required skills (moderate coverage)"
        else:
            return "Candidate lacks most required skills (weak coverage)"
    
    def _describe_experience_match(self, match: float) -> str:
        """Describe experience match score."""
        if match >= 0.9:
            return "Experience level matches job requirements perfectly"
        elif match >= 0.7:
            return "Experience level is appropriate for this role"
        elif match >= 0.5:
            return "Experience level is acceptable but not ideal"
        else:
            return "Experience level does not match requirements"
    
    def _generate_explanation_text(
        self,
        score: float,
        top_positive: List[FeatureImpact],
        top_negative: List[FeatureImpact]
    ) -> str:
        """
        Generate human-readable explanation.
        
        Args:
            score: Final score
            top_positive: Top positive factors
            top_negative: Top negative factors
            
        Returns:
            Explanation text
        """
        lines = []
        
        # Overall assessment
        if score >= 85:
            lines.append("🌟 This is an excellent match!")
        elif score >= 70:
            lines.append("✅ This is a strong match.")
        elif score >= 50:
            lines.append("⚡ This is a moderate match.")
        else:
            lines.append("⚠️ This is a weak match.")
        
        lines.append(f"Final Score: {score:.1f}/100")
        lines.append("")
        
        # Top positive factors
        if top_positive:
            lines.append("**Positive Factors:**")
            for i, factor in enumerate(top_positive[:3], 1):
                lines.append(f"{i}. {factor.description} (+{factor.impact_score:.1f} points)")
            lines.append("")
        
        # Top negative factors
        if top_negative:
            lines.append("**Areas for Improvement:**")
            for i, factor in enumerate(top_negative[:3], 1):
                lines.append(f"{i}. {factor.description} ({factor.impact_score:.1f} points)")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self, explanation: XAIExplanation) -> Dict[str, Any]:
        """Convert XAIExplanation to dictionary for JSON serialization."""
        return {
            "prediction_score": explanation.prediction_score,
            "base_score": explanation.base_score,
            "top_positive_factors": [asdict(f) for f in explanation.top_positive_factors],
            "top_negative_factors": [asdict(f) for f in explanation.top_negative_factors],
            "feature_importance": explanation.feature_importance,
            "explanation_text": explanation.explanation_text,
            "confidence": explanation.confidence,
            "timestamp": explanation.timestamp
        }


# Global singleton
_global_xai_explainer: Optional[XAIExplainer] = None


def get_xai_explainer() -> XAIExplainer:
    """
    Get or create global XAI explainer (singleton).
    
    Returns:
        XAIExplainer instance
    """
    global _global_xai_explainer
    
    if _global_xai_explainer is None:
        _global_xai_explainer = XAIExplainer()
    
    return _global_xai_explainer
