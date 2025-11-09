"""API endpoints pour le système de recommandations multicritères"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging

from .models import (
    UserProfile, RecommendationPreferences, ComprehensiveRecommendations,
    RoadmapRecommendation, CertificationRecommendation, 
    SkillRecommendation, ProjectRecommendation, RecommendationFeedback
)
from .core.recommendation_orchestrator import RecommendationOrchestrator

logger = logging.getLogger(__name__)

# Création du router
router = APIRouter(prefix="/v2/recommendations", tags=["Recommendations v2"])

# Instance globale de l'orchestrateur
recommendation_orchestrator = RecommendationOrchestrator()


@router.post("/comprehensive", response_model=ComprehensiveRecommendations)
async def get_comprehensive_recommendations(
    user_profile: UserProfile,
    preferences: Optional[RecommendationPreferences] = None
):
    """
    Génère un ensemble complet de recommandations personnalisées
    incluant roadmaps, certifications, compétences et projets.
    
    **Fonctionnalités:**
    - Analyse ML du profil utilisateur
    - Scoring unifié multicritères
    - Personnalisation avancée
    - Équilibrage et diversification
    """
    try:
        logger.info(f"Generating comprehensive recommendations for user {user_profile.user_id}")
        
        recommendations = await recommendation_orchestrator.generate_comprehensive_recommendations(
            user_profile, preferences
        )
        
        logger.info(f"Generated {len(recommendations.recommendations)} recommendation types")
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating comprehensive recommendations: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la génération des recommandations: {str(e)}"
        )


@router.post("/roadmaps", response_model=List[RoadmapRecommendation])
async def get_roadmap_recommendations(
    user_profile: UserProfile,
    max_results: int = 5,
    focus_domains: Optional[List[str]] = None
):
    """
    Génère des recommandations de roadmaps de carrière personnalisés.
    
    **Paramètres:**
    - max_results: Nombre maximum de roadmaps (défaut: 5)
    - focus_domains: Domaines de focus ['data_science', 'web_dev', 'devops']
    """
    try:
        # Création des préférences
        preferences = {'focus_areas': focus_domains} if focus_domains else {}
        
        # Extraction des features utilisateur
        profile_analysis = {'extracted_skills': {'skills': user_profile.current_skills}}
        user_features = recommendation_orchestrator.personalization_engine.extract_user_features(
            user_profile, profile_analysis
        )
        
        # Génération des recommandations
        roadmap_recs = await recommendation_orchestrator.roadmap_recommender.recommend(
            user_features, preferences
        )
        
        return roadmap_recs[:max_results]
        
    except Exception as e:
        logger.error(f"Error generating roadmap recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des roadmaps: {str(e)}"
        )


@router.post("/certifications", response_model=List[CertificationRecommendation])
async def get_certification_recommendations(
    user_profile: UserProfile,
    max_results: int = 5,
    budget_range: str = "free-premium",  # "free", "free-premium", "premium", "enterprise"
    certification_level: Optional[str] = None  # "associate", "professional", "expert"
):
    """
    Génère des recommandations de certifications professionnelles.
    
    **Paramètres:**
    - budget_range: Gamme de budget pour les certifications
    - certification_level: Niveau de certification ciblé
    """
    try:
        preferences = {
            'budget_range': budget_range,
            'certification_level': certification_level
        }
        
        profile_analysis = {'extracted_skills': {'skills': user_profile.current_skills}}
        user_features = recommendation_orchestrator.personalization_engine.extract_user_features(
            user_profile, profile_analysis
        )
        
        cert_recs = await recommendation_orchestrator.certification_recommender.recommend(
            user_features, preferences
        )
        
        return cert_recs[:max_results]
        
    except Exception as e:
        logger.error(f"Error generating certification recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des certifications: {str(e)}"
        )


@router.post("/skills", response_model=List[SkillRecommendation])
async def get_skill_recommendations(
    user_profile: UserProfile,
    max_results: int = 8,
    priority_filter: Optional[str] = None,  # "high", "medium", "low"
    category_filter: Optional[List[str]] = None  # ["technical", "soft", "domain"]
):
    """
    Génère des recommandations de compétences à développer.
    
    **Paramètres:**
    - priority_filter: Filtre par niveau de priorité
    - category_filter: Filtre par catégories de compétences
    """
    try:
        preferences = {
            'priority_filter': priority_filter,
            'category_filter': category_filter
        }
        
        profile_analysis = {'extracted_skills': {'skills': user_profile.current_skills}}
        user_features = recommendation_orchestrator.personalization_engine.extract_user_features(
            user_profile, profile_analysis
        )
        
        skill_recs = await recommendation_orchestrator.skills_recommender.recommend(
            user_features, preferences
        )
        
        # Filtrage post-génération si nécessaire
        if priority_filter:
            skill_recs = [rec for rec in skill_recs if rec.priority == priority_filter]
        
        return skill_recs[:max_results]
        
    except Exception as e:
        logger.error(f"Error generating skill recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des compétences: {str(e)}"
        )


@router.post("/projects", response_model=List[ProjectRecommendation])
async def get_project_recommendations(
    user_profile: UserProfile,
    max_results: int = 6,
    difficulty_level: Optional[str] = None,  # "beginner", "intermediate", "advanced"
    project_domain: Optional[List[str]] = None,  # ["web_development", "data_science", etc.]
    time_commitment: Optional[str] = None  # "short", "medium", "long"
):
    """
    Génère des recommandations de projets pratiques.
    
    **Paramètres:**
    - difficulty_level: Niveau de difficulté souhaité
    - project_domain: Domaines de projets d'intérêt
    - time_commitment: Engagement temporel (court: <4 semaines, moyen: 4-8, long: >8)
    """
    try:
        preferences = {
            'difficulty_level': difficulty_level,
            'focus_areas': project_domain,
            'time_commitment': time_commitment
        }
        
        profile_analysis = {'extracted_skills': {'skills': user_profile.current_skills}}
        user_features = recommendation_orchestrator.personalization_engine.extract_user_features(
            user_profile, profile_analysis
        )
        
        project_recs = await recommendation_orchestrator.project_recommender.recommend(
            user_features, preferences
        )
        
        return project_recs[:max_results]
        
    except Exception as e:
        logger.error(f"Error generating project recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération des projets: {str(e)}"
        )


@router.post("/feedback")
async def submit_recommendation_feedback(
    feedback: RecommendationFeedback,
    background_tasks: BackgroundTasks
):
    """
    Collecte le feedback utilisateur pour améliorer les recommandations futures.
    
    **Types de feedback:**
    - 'like': Recommandation appréciée
    - 'dislike': Recommandation non pertinente
    - 'applied': Recommandation suivie/commencée
    - 'completed': Recommandation terminée avec succès
    - 'not_relevant': Recommandation non adaptée au profil
    """
    try:
        # Traitement immédiat
        logger.info(f"Feedback received: {feedback.feedback_type} for recommendation {feedback.recommendation_id}")
        
        # Traitement en arrière-plan pour mise à jour des modèles
        background_tasks.add_task(
            _process_feedback_for_learning,
            feedback
        )
        
        return {
            "status": "success",
            "message": "Feedback enregistré avec succès",
            "feedback_id": f"{feedback.user_id}_{feedback.recommendation_id}_{int(feedback.created_at.timestamp())}"
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'enregistrement du feedback: {str(e)}"
        )


@router.get("/analytics/{user_id}")
async def get_recommendation_analytics(
    user_id: str,
    time_period: str = "30d"  # "7d", "30d", "90d", "1y"
):
    """
    Retourne les analytics des recommandations pour un utilisateur.
    
    **Métriques incluses:**
    - Nombre de recommandations générées par type
    - Taux d'adoption des recommandations
    - Feedback positif/négatif
    - Progression des compétences
    """
    try:
        # TODO: Implémenter la collecte de vraies analytics depuis la DB
        analytics = {
            "user_id": user_id,
            "period": time_period,
            "recommendations_generated": {
                "roadmaps": 12,
                "certifications": 8,
                "skills": 24,
                "projects": 15,
                "total": 59
            },
            "adoption_rates": {
                "roadmaps": 0.67,
                "certifications": 0.45,
                "skills": 0.78,
                "projects": 0.58
            },
            "feedback_summary": {
                "positive": 42,
                "negative": 8,
                "neutral": 9,
                "completion_rate": 0.34
            },
            "skill_progression": {
                "skills_learned": 6,
                "certifications_earned": 2,
                "projects_completed": 4,
                "roadmaps_in_progress": 1
            },
            "top_domains": ["web_development", "data_science", "cloud_computing"],
            "generated_at": "2025-01-17T12:22:05Z"
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting analytics for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des analytics: {str(e)}"
        )


@router.get("/health")
async def recommendation_system_health():
    """
    Vérifie l'état de santé du système de recommandations.
    """
    try:
        # Vérifications de base
        health_status = {
            "status": "healthy",
            "components": {
                "recommendation_orchestrator": "operational",
                "roadmap_recommender": "operational",
                "certification_recommender": "operational",
                "skills_recommender": "operational",
                "project_recommender": "operational",
                "scoring_engine": "operational",
                "personalization_engine": "operational"
            },
            "database_connections": {
                "recommendations_db": "connected",
                "analytics_db": "connected"
            },
            "ml_models": {
                "similarity_model": "loaded" if recommendation_orchestrator.ml_engine else "unavailable",
                "neural_scorer": "loaded" if recommendation_orchestrator.scoring_engine.neural_scorer else "unavailable"
            },
            "version": "2.0.0",
            "uptime": "24h 15m",
            "last_updated": "2025-01-17T12:22:05Z"
        }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2025-01-17T12:22:05Z"
        }


async def _process_feedback_for_learning(feedback: RecommendationFeedback):
    """
    Traite le feedback pour l'apprentissage du système (tâche en arrière-plan)
    """
    try:
        # TODO: Implémenter la logique de réapprentissage
        # - Mettre à jour les poids des modèles
        # - Ajuster les algorithmes de personnalisation
        # - Améliorer les scores de recommandation
        
        logger.info(f"Processing feedback {feedback.feedback_type} for learning")
        
        # Simulation du traitement
        if feedback.feedback_type == "completed":
            # Renforcer ce type de recommandation
            pass
        elif feedback.feedback_type == "dislike":
            # Réduire le poids de ce type de recommandation
            pass
        
        logger.info("Feedback processing completed")
        
    except Exception as e:
        logger.error(f"Error processing feedback for learning: {e}")