"""
SkillSync ML Pipeline Package
==============================
Real machine learning infrastructure for AI-powered recruiting.

Modules:
- embeddings: Semantic embeddings using SBERT
- ner: Named Entity Recognition for skill extraction
- scoring: ML-based CV-job matching scores
- xai: Explainable AI with SHAP
- translator: NLG experience rewriting with T5
- recommender: ML-based recommendations
"""

from .embeddings import (
    SemanticEmbeddingEngine,
    EmbeddingResult,
    get_embedding_engine
)

from .ner import (
    SkillNERExtractor,
    ExtractedSkill,
    NERResult,
    get_ner_extractor
)

from .scoring import (
    MLScoringEngine,
    MatchScore,
    JobMatchResult,
    get_scoring_engine
)

from .xai import (
    XAIExplainer,
    XAIExplanation,
    FeatureImpact,
    get_xai_explainer
)

from .translator import (
    NLGExperienceTranslator,
    TranslationResult,
    get_translator
)

from .recommender import (
    MLRecommender,
    Recommendation,
    RecommendationSet,
    get_recommender
)

__version__ = "1.0.0"

__all__ = [
    # Embeddings
    "SemanticEmbeddingEngine",
    "EmbeddingResult",
    "get_embedding_engine",
    
    # NER
    "SkillNERExtractor",
    "ExtractedSkill",
    "NERResult",
    "get_ner_extractor",
    
    # Scoring
    "MLScoringEngine",
    "MatchScore",
    "JobMatchResult",
    "get_scoring_engine",
    
    # XAI
    "XAIExplainer",
    "XAIExplanation",
    "FeatureImpact",
    "get_xai_explainer",
    
    # Translation
    "NLGExperienceTranslator",
    "TranslationResult",
    "get_translator",
    
    # Recommendations
    "MLRecommender",
    "Recommendation",
    "RecommendationSet",
    "get_recommender",
]
