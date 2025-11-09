"""Orchestrateur principal du système de recommandations multicritères"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# CORRECTION: Imports relatifs corrigés
from ..models import (
    UserProfile, RecommendationPreferences, ComprehensiveRecommendations,
    UserFeatures, BaseRecommendation, RecommendationType
)
from .scoring_engine import ScoringEngine
from .personalization_engine import PersonalizationEngine
from ..recommenders.roadmap_recommender import RoadmapRecommender
from ..recommenders.certification_recommender import CertificationRecommender
from ..recommenders.skills_recommender import SkillsRecommender
from ..recommenders.project_recommender import ProjectRecommender

# Import ML engine existant
try:
    from ...ml_models.advanced_recommendation_engine import AdvancedRecommendationEngine
except ImportError:
    logging.warning("ML models not available, using fallback")
    AdvancedRecommendationEngine = None

logger = logging.getLogger(__name__)


class RecommendationOrchestrator:
    """
    Orchestrateur principal gérant tous les types de recommandations
    avec scoring unifié et personnalisation avancée
    """
    
    def __init__(self, models_path: Optional[str] = None):
        # Moteur ML existant
        if AdvancedRecommendationEngine:
            self.ml_engine = AdvancedRecommendationEngine(models_path)
        else:
            self.ml_engine = None
            logger.warning("ML engine not available")
        
        # Nouveaux moteurs
        self.scoring_engine = ScoringEngine()
        self.personalization_engine = PersonalizationEngine()
        
        # Recommandeurs spécialisés
        self.roadmap_recommender = RoadmapRecommender()
        self.certification_recommender = CertificationRecommender()
        self.skills_recommender = SkillsRecommender()
        self.project_recommender = ProjectRecommender()
        
        logger.info("Recommendation Orchestrator initialized")
    
    async def generate_comprehensive_recommendations(
        self, 
        user_profile: UserProfile, 
        preferences: Optional[RecommendationPreferences] = None
    ) -> ComprehensiveRecommendations:
        """
        Génère un ensemble complet de recommandations personnalisées
        """
        try:
            if preferences is None:
                preferences = RecommendationPreferences()
            
            logger.info(f"Generating recommendations for user {user_profile.user_id}")
            
            # 1. Analyse du profil utilisateur avec ML existant
            profile_analysis = await self._analyze_user_profile(user_profile)
            
            # 2. Extraction des caractéristiques utilisateur
            user_features = self.personalization_engine.extract_user_features(
                user_profile, profile_analysis
            )
            
            # 3. Génération des recommandations par type
            recommendations = await self._generate_typed_recommendations(
                user_features, preferences
            )
            
            # 4. Scoring unifié et personnalisation
            scored_recommendations = await self._score_and_personalize(
                recommendations, user_features, preferences
            )
            
            # 5. Équilibrage et diversification
            balanced_recommendations = self._balance_recommendations(
                scored_recommendations, preferences
            )
            
            # 6. Génération de l'explication globale
            global_explanation = self._generate_global_explanation(
                user_features, balanced_recommendations
            )
            
            # 7. Calcul de la confiance globale
            confidence = self._calculate_global_confidence(
                balanced_recommendations
            )
            
            return ComprehensiveRecommendations(
                user_profile=user_profile,
                recommendations=balanced_recommendations,
                global_explanation=global_explanation,
                confidence=confidence,
                generated_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=7)  # Expiration 7 jours
            )
            
        except Exception as e:
            logger.error(f"Error generating comprehensive recommendations: {e}")
            return ComprehensiveRecommendations(
                user_profile=user_profile,
                recommendations={},
                global_explanation={'error': str(e)},
                confidence=0.0,
                generated_at=datetime.now()
            )
    
    async def _analyze_user_profile(self, user_profile: UserProfile) -> Dict[str, Any]:
        """
        Analyse le profil utilisateur avec le moteur ML existant
        """
        if self.ml_engine:
            try:
                return self.ml_engine.analyze_cv_profile(user_profile.cv_data)
            except Exception as e:
                logger.error(f"ML analysis failed: {e}")
        
        # Fallback analysis
        return {
            'extracted_skills': {'skills': user_profile.current_skills},
            'categorized_skills': self._categorize_skills_fallback(user_profile.current_skills),
            'profile_analysis': {
                'experience_level': user_profile.level.value,
                'domain': user_profile.industry
            }
        }
    
    async def _generate_typed_recommendations(
        self, 
        user_features: UserFeatures, 
        preferences: RecommendationPreferences
    ) -> Dict[str, List[BaseRecommendation]]:
        """
        Génère les recommandations par type
        """
        recommendations = {}
        
        try:
            # Roadmaps de carrière
            if 'roadmaps' not in preferences.focus_areas or 'roadmaps' in preferences.focus_areas:
                recommendations['roadmaps'] = await self.roadmap_recommender.recommend(
                    user_features, preferences.roadmap_preferences
                )
            
            # Certifications
            if 'certifications' not in preferences.focus_areas or 'certifications' in preferences.focus_areas:
                recommendations['certifications'] = await self.certification_recommender.recommend(
                    user_features, preferences.cert_preferences
                )
            
            # Compétences
            if 'skills' not in preferences.focus_areas or 'skills' in preferences.focus_areas:
                recommendations['skills'] = await self.skills_recommender.recommend(
                    user_features, preferences.skill_preferences
                )
            
            # Projets pratiques
            if 'projects' not in preferences.focus_areas or 'projects' in preferences.focus_areas:
                recommendations['projects'] = await self.project_recommender.recommend(
                    user_features, preferences.project_preferences
                )
                
        except Exception as e:
            logger.error(f"Error generating typed recommendations: {e}")
        
        return recommendations
    
    async def _score_and_personalize(
        self,
        recommendations: Dict[str, List[BaseRecommendation]],
        user_features: UserFeatures,
        preferences: RecommendationPreferences
    ) -> Dict[str, List[BaseRecommendation]]:
        """
        Applique le scoring unifié et la personnalisation
        """
        scored = {}
        
        for rec_type, rec_list in recommendations.items():
            scored_list = []
            for rec in rec_list:
                try:
                    # Scoring unifié
                    scoring_result = self.scoring_engine.calculate_unified_score(
                        rec, user_features
                    )
                    
                    # Mise à jour des scores
                    rec.scores.update(scoring_result.individual_scores)
                    rec.scores['unified'] = scoring_result.combined_score
                    rec.confidence = scoring_result.confidence
                    
                    # Personnalisation
                    personalized_rec = self.personalization_engine.personalize_recommendation(
                        rec, user_features, preferences
                    )
                    
                    scored_list.append(personalized_rec)
                    
                except Exception as e:
                    logger.error(f"Error scoring recommendation {rec.id}: {e}")
                    scored_list.append(rec)  # Garde la recommandation sans scoring
            
            # Tri par score unifié
            scored_list.sort(key=lambda x: x.scores.get('unified', 0), reverse=True)
            scored[rec_type] = scored_list
        
        return scored
    
    def _balance_recommendations(
        self,
        recommendations: Dict[str, List[BaseRecommendation]],
        preferences: RecommendationPreferences
    ) -> Dict[str, List[BaseRecommendation]]:
        """
        Équilibre les recommandations pour éviter la surspécialisation
        """
        balanced = {}
        total_slots = preferences.max_recommendations
        
        # Allocation dynamique des slots par type
        allocation = {
            'skills': max(3, int(total_slots * 0.3)),      # 30% pour skills
            'projects': max(2, int(total_slots * 0.25)),   # 25% pour projets
            'certifications': max(2, int(total_slots * 0.25)), # 25% pour certifications
            'roadmaps': max(1, int(total_slots * 0.2))     # 20% pour roadmaps
        }
        
        for rec_type, max_count in allocation.items():
            if rec_type in recommendations:
                rec_list = recommendations[rec_type][:max_count]
                
                # Diversification au sein du type
                diversified = self._diversify_recommendations(rec_list)
                balanced[rec_type] = diversified
        
        return balanced
    
    def _diversify_recommendations(
        self, 
        recommendations: List[BaseRecommendation]
    ) -> List[BaseRecommendation]:
        """
        Diversifie les recommandations au sein d'un type
        """
        if len(recommendations) <= 3:
            return recommendations
        
        # Algorithme simple de diversification basé sur les domaines
        diversified = []
        used_domains = set()
        
        # Première passe : une recommandation par domaine
        for rec in recommendations:
            domain = getattr(rec, 'domain', 'general')
            if domain not in used_domains:
                diversified.append(rec)
                used_domains.add(domain)
        
        # Deuxième passe : compléter avec les meilleures
        remaining_slots = len(recommendations) - len(diversified)
        for rec in recommendations:
            if rec not in diversified and remaining_slots > 0:
                diversified.append(rec)
                remaining_slots -= 1
        
        return diversified
    
    def _generate_global_explanation(
        self,
        user_features: UserFeatures,
        recommendations: Dict[str, List[BaseRecommendation]]
    ) -> Dict[str, Any]:
        """
        Génère une explication globale des recommandations
        """
        total_recs = sum(len(recs) for recs in recommendations.values())
        
        explanation = {
            'summary': f"Nous avons généré {total_recs} recommandations personnalisées pour votre profil.",
            'approach': "Nos recommandations combinent l'IA avancée et votre profil unique.",
            'breakdown': {},
            'next_steps': []
        }
        
        # Breakdown par type
        for rec_type, rec_list in recommendations.items():
            if rec_list:
                top_rec = rec_list[0]  # Meilleure recommandation du type
                explanation['breakdown'][rec_type] = {
                    'count': len(rec_list),
                    'top_recommendation': top_rec.title,
                    'reason': top_rec.explanation.get('main_reason', 'Excellent match pour votre profil')
                }
        
        # Actions recommandées
        if 'skills' in recommendations and recommendations['skills']:
            top_skill = recommendations['skills'][0]
            explanation['next_steps'].append(
                f"🎯 Priorité : Développer '{top_skill.title}' pour maximiser vos opportunités"
            )
        
        if 'projects' in recommendations and recommendations['projects']:
            top_project = recommendations['projects'][0]
            explanation['next_steps'].append(
                f"🛠️ Action : Démarrer le projet '{top_project.title}' pour appliquer vos compétences"
            )
        
        return explanation
    
    def _calculate_global_confidence(
        self, 
        recommendations: Dict[str, List[BaseRecommendation]]
    ) -> float:
        """
        Calcule la confiance globale des recommandations
        """
        if not recommendations:
            return 0.0
        
        all_confidences = []
        for rec_list in recommendations.values():
            for rec in rec_list:
                all_confidences.append(rec.confidence)
        
        if not all_confidences:
            return 0.5  # Confiance moyenne par défaut
        
        return sum(all_confidences) / len(all_confidences)
    
    def _categorize_skills_fallback(self, skills: List[str]) -> Dict[str, List[str]]:
        """
        Catégorisation de base des compétences (fallback)
        """
        # Mapping simple par mots-clés
        categories = {
            'programming': [],
            'frameworks': [],
            'cloud': [],
            'tools': [],
            'soft_skills': []
        }
        
        programming_keywords = ['python', 'javascript', 'java', 'c++', 'react', 'vue', 'angular']
        cloud_keywords = ['aws', 'azure', 'gcp', 'docker', 'kubernetes']
        
        for skill in skills:
            skill_lower = skill.lower()
            if any(keyword in skill_lower for keyword in programming_keywords):
                categories['programming'].append(skill)
            elif any(keyword in skill_lower for keyword in cloud_keywords):
                categories['cloud'].append(skill)
            else:
                categories['tools'].append(skill)
        
        return categories
