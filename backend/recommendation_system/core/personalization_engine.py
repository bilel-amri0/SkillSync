"""Moteur de personnalisation pour les recommandations"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..models import (
    UserProfile, UserFeatures, BaseRecommendation, 
    RecommendationPreferences, ExperienceLevel
)

logger = logging.getLogger(__name__)


class PersonalizationEngine:
    """
    Moteur de personnalisation pour adapter les recommandations au profil utilisateur
    """
    
    def __init__(self):
        logger.info("PersonalizationEngine initialized")
    
    def extract_user_features(self, user_profile: UserProfile, profile_analysis: Dict[str, Any]) -> UserFeatures:
        """
        Extrait les features utilisateur pour le ML et la personnalisation
        """
        try:
            # Features d'expérience
            experience_features = {
                'years': user_profile.experience_years,
                'level_numeric': self._level_to_numeric(user_profile.level),
                'industry_score': self._industry_score(user_profile.industry)
            }
            
            # Features de carrière
            career_features = {
                'skills': user_profile.current_skills,
                'goals': user_profile.career_goals,
                'role_score': self._role_score(user_profile.current_role)
            }
            
            # Features de préférences
            preference_features = {
                'time_availability_numeric': self._time_to_numeric(user_profile.time_availability),
                'budget_score': self._budget_score(user_profile.budget_constraints),
                'learning_style_score': self._learning_style_score(user_profile.learning_preferences)
            }
            
            # Features contextuelles
            contextual_features = {
                'urgency_score': 0.7,  # Par défaut
                'market_alignment': 0.8,
                'growth_potential': 0.9
            }
            
            # Vecteur de compétences (simplifié)
            skill_vector = self._create_skill_vector(user_profile.current_skills)
            
            return UserFeatures(
                skill_vector=skill_vector,
                experience_features=experience_features,
                career_features=career_features,
                preference_features=preference_features,
                contextual_features=contextual_features
            )
            
        except Exception as e:
            logger.error(f"Error extracting user features: {e}")
            # Retourne des features par défaut
            return self._default_user_features()
    
    def personalize_recommendation(
        self, 
        recommendation: BaseRecommendation, 
        user_features: UserFeatures,
        preferences: RecommendationPreferences
    ) -> BaseRecommendation:
        """
        Personnalise une recommandation selon le profil utilisateur
        """
        try:
            # Ajustement du score selon les préférences
            personalization_boost = self._calculate_personalization_boost(
                recommendation, user_features, preferences
            )
            
            # Mise à jour du score personnalisé
            if 'unified' in recommendation.scores:
                recommendation.scores['personalized'] = min(
                    recommendation.scores['unified'] + personalization_boost, 1.0
                )
            
            # Enrichissement de l'explication
            recommendation.explanation.update({
                'personalization_reason': self._generate_personalization_reason(
                    recommendation, user_features
                )
            })
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error personalizing recommendation: {e}")
            return recommendation
    
    def _level_to_numeric(self, level: ExperienceLevel) -> float:
        """Convertit le niveau d'expérience en valeur numérique"""
        mapping = {
            ExperienceLevel.JUNIOR: 0.2,
            ExperienceLevel.MID: 0.5,
            ExperienceLevel.SENIOR: 0.8,
            ExperienceLevel.LEAD: 0.9,
            ExperienceLevel.EXECUTIVE: 1.0
        }
        return mapping.get(level, 0.4)
    
    def _industry_score(self, industry: str) -> float:
        """Score d'industrie basé sur la demande du marché"""
        industry_scores = {
            'tech': 0.9,
            'finance': 0.8,
            'healthcare': 0.7,
            'education': 0.6,
            'government': 0.5
        }
        return industry_scores.get(industry.lower(), 0.6)
    
    def _role_score(self, role: str) -> float:
        """Score de rôle basé sur la demande"""
        if any(keyword in role.lower() for keyword in ['senior', 'lead', 'architect']):
            return 0.9
        elif any(keyword in role.lower() for keyword in ['developer', 'engineer', 'analyst']):
            return 0.7
        else:
            return 0.5
    
    def _time_to_numeric(self, time_availability: str) -> float:
        """Convertit la disponibilité en score numérique"""
        if '15' in time_availability or '20' in time_availability:
            return 0.9
        elif '10' in time_availability:
            return 0.7
        elif '5' in time_availability:
            return 0.5
        else:
            return 0.6
    
    def _budget_score(self, budget_constraints: Dict[str, Any]) -> float:
        """Score de budget"""
        monthly_budget = budget_constraints.get('monthly_budget', 0)
        if monthly_budget > 200:
            return 0.9
        elif monthly_budget > 100:
            return 0.7
        elif monthly_budget > 50:
            return 0.5
        else:
            return 0.3
    
    def _learning_style_score(self, learning_preferences: Dict[str, Any]) -> float:
        """Score de style d'apprentissage"""
        style = learning_preferences.get('style', [])
        if 'hands-on' in style:
            return 0.8
        elif 'project-based' in style:
            return 0.9
        else:
            return 0.6
    
    def _create_skill_vector(self, skills: List[str]) -> List[float]:
        """Crée un vecteur de compétences simplifié"""
        # Vecteur de 10 dimensions pour les catégories principales
        vector = [0.0] * 10
        
        skill_categories = {
            'programming': [0, ['python', 'javascript', 'java', 'c++', 'react']],
            'data': [1, ['sql', 'excel', 'powerbi', 'pandas', 'numpy']],
            'cloud': [2, ['aws', 'azure', 'docker', 'kubernetes']],
            'web': [3, ['html', 'css', 'react', 'vue', 'angular']],
            'ml': [4, ['machine learning', 'ai', 'tensorflow', 'pytorch']],
            'devops': [5, ['git', 'jenkins', 'terraform', 'ansible']],
            'mobile': [6, ['android', 'ios', 'flutter', 'react native']],
            'database': [7, ['postgresql', 'mongodb', 'mysql', 'redis']],
            'security': [8, ['cybersecurity', 'encryption', 'firewall']],
            'management': [9, ['leadership', 'project management', 'agile']]
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            for category, (index, keywords) in skill_categories.items():
                if any(keyword in skill_lower for keyword in keywords):
                    vector[index] += 1.0
        
        # Normalisation
        max_val = max(vector) if max(vector) > 0 else 1
        return [v / max_val for v in vector]
    
    def _calculate_personalization_boost(
        self, 
        recommendation: BaseRecommendation,
        user_features: UserFeatures,
        preferences: RecommendationPreferences
    ) -> float:
        """Calcule le boost de personnalisation"""
        boost = 0.0
        
        # Boost basé sur l'alignement avec les préférences
        if hasattr(recommendation, 'domain'):
            domain = getattr(recommendation, 'domain', '')
            if any(area in domain for area in preferences.focus_areas):
                boost += 0.1
        
        # Boost basé sur le niveau d'expérience
        experience_level = user_features.experience_features.get('level_numeric', 0.5)
        if hasattr(recommendation, 'difficulty'):
            difficulty = getattr(recommendation, 'difficulty', None)
            if difficulty and self._is_difficulty_appropriate(difficulty, experience_level):
                boost += 0.05
        
        return min(boost, 0.2)  # Maximum 20% de boost
    
    def _is_difficulty_appropriate(self, difficulty, experience_level: float) -> bool:
        """Vérifie si la difficulté est appropriée au niveau"""
        difficulty_mapping = {
            'beginner': (0.0, 0.4),
            'intermediate': (0.3, 0.7),
            'advanced': (0.6, 0.9),
            'expert': (0.8, 1.0)
        }
        
        if hasattr(difficulty, 'value'):
            difficulty_str = difficulty.value
        else:
            difficulty_str = str(difficulty).lower()
        
        if difficulty_str in difficulty_mapping:
            min_level, max_level = difficulty_mapping[difficulty_str]
            return min_level <= experience_level <= max_level
        
        return True
    
    def _generate_personalization_reason(
        self, 
        recommendation: BaseRecommendation,
        user_features: UserFeatures
    ) -> str:
        """Génère une raison de personnalisation"""
        reasons = []
        
        # Raison basée sur l'expérience
        experience_years = user_features.experience_features.get('years', 0)
        if experience_years < 2:
            reasons.append("Adapté à votre profil débutant")
        elif experience_years < 5:
            reasons.append("Parfait pour votre niveau intermédiaire")
        else:
            reasons.append("Correspond à votre expertise avancée")
        
        # Raison basée sur les compétences
        skills = user_features.career_features.get('skills', [])
        if len(skills) > 5:
            reasons.append("Complète vos compétences existantes")
        else:
            reasons.append("Développe votre base de compétences")
        
        return "; ".join(reasons) if reasons else "Recommandation personnalisée"
    
    def _default_user_features(self) -> UserFeatures:
        """Retourne des features par défaut en cas d'erreur"""
        return UserFeatures(
            skill_vector=[0.5] * 10,
            experience_features={'years': 2, 'level_numeric': 0.4, 'industry_score': 0.6},
            career_features={'skills': [], 'goals': [], 'role_score': 0.5},
            preference_features={'time_availability_numeric': 0.6, 'budget_score': 0.5, 'learning_style_score': 0.7},
            contextual_features={'urgency_score': 0.7, 'market_alignment': 0.8, 'growth_potential': 0.9}
        )
