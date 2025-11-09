"""Moteur de scoring unifié pour les recommandations"""

import logging
from typing import Dict, List, Any, Optional
import random

from ..models import (
    UserFeatures, BaseRecommendation, ScoringResult
)

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Moteur de scoring unifié combinant heuristiques et IA
    """
    
    def __init__(self):
        # Poids pour les différents facteurs de scoring
        self.scoring_weights = {
            'relevance': 0.3,
            'feasibility': 0.2,
            'market_value': 0.2,
            'user_fit': 0.2,
            'growth_potential': 0.1
        }
        logger.info("ScoringEngine initialized")
    
    def calculate_unified_score(
        self, 
        recommendation: BaseRecommendation, 
        user_features: UserFeatures
    ) -> ScoringResult:
        """
        Calcule le score unifié d'une recommandation
        """
        try:
            # 1. Calcul des scores individuels
            individual_scores = self._calculate_individual_scores(
                recommendation, user_features
            )
            
            # 2. Score pondéré (heuristique)
            weighted_score = self._calculate_weighted_score(individual_scores)
            
            # 3. Score neural (simulé)
            neural_score = self._calculate_neural_score(
                recommendation, user_features
            )
            
            # 4. Score combiné
            combined_score = (weighted_score * 0.7 + neural_score * 0.3)
            
            # 5. Calcul de la confiance
            confidence = self._calculate_confidence(
                individual_scores, weighted_score, neural_score
            )
            
            # 6. Génération de l'explication
            explanation = self._generate_scoring_explanation(
                individual_scores, recommendation, user_features
            )
            
            return ScoringResult(
                individual_scores=individual_scores,
                weighted_score=weighted_score,
                neural_score=neural_score,
                combined_score=combined_score,
                confidence=confidence,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error calculating unified score: {e}")
            return self._default_scoring_result()
    
    def _calculate_individual_scores(
        self, 
        recommendation: BaseRecommendation, 
        user_features: UserFeatures
    ) -> Dict[str, float]:
        """
        Calcule les scores pour chaque facteur individuel
        """
        scores = {}
        
        # Score de pertinence
        scores['relevance'] = self._score_relevance(recommendation, user_features)
        
        # Score de faisabilité
        scores['feasibility'] = self._score_feasibility(recommendation, user_features)
        
        # Score de valeur marché
        scores['market_value'] = self._score_market_value(recommendation)
        
        # Score d'adéquation utilisateur
        scores['user_fit'] = self._score_user_fit(recommendation, user_features)
        
        # Score de potentiel de croissance
        scores['growth_potential'] = self._score_growth_potential(recommendation)
        
        return scores
    
    def _score_relevance(self, recommendation: BaseRecommendation, user_features: UserFeatures) -> float:
        """Score de pertinence basé sur l'alignement avec le profil"""
        base_score = 0.6  # Score par défaut
        
        # Bonus pour l'alignement avec les compétences
        user_skills = user_features.career_features.get('skills', [])
        if hasattr(recommendation, 'skills_to_learn') or hasattr(recommendation, 'skill'):
            # Logique d'alignement des compétences
            base_score += 0.2
        
        # Bonus pour l'alignement avec les objectifs
        career_goals = user_features.career_features.get('goals', [])
        if career_goals and hasattr(recommendation, 'domain'):
            domain = getattr(recommendation, 'domain', '')
            if any(goal.lower() in domain.lower() for goal in career_goals):
                base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _score_feasibility(self, recommendation: BaseRecommendation, user_features: UserFeatures) -> float:
        """Score de faisabilité basé sur les contraintes utilisateur"""
        base_score = 0.7
        
        # Pénalité pour le temps requis
        time_availability = user_features.preference_features.get('time_availability_numeric', 0.6)
        if hasattr(recommendation, 'estimated_time') or hasattr(recommendation, 'duration'):
            # Logique de temps - simplifié
            if time_availability > 0.7:
                base_score += 0.2
            elif time_availability < 0.4:
                base_score -= 0.2
        
        # Pénalité pour le budget
        budget_score = user_features.preference_features.get('budget_score', 0.6)
        if hasattr(recommendation, 'cost') and hasattr(recommendation, 'certification'):
            cost = getattr(recommendation.certification, 'cost', 0) if hasattr(recommendation, 'certification') else 0
            if cost > 500 and budget_score < 0.5:
                base_score -= 0.3
        
        return max(min(base_score, 1.0), 0.0)
    
    def _score_market_value(self, recommendation: BaseRecommendation) -> float:
        """Score de valeur marché"""
        base_score = 0.6
        
        # Bonus pour la demande du marché
        if hasattr(recommendation, 'market_demand'):
            market_demand = getattr(recommendation, 'market_demand', 0)
            base_score += market_demand * 0.3
        elif hasattr(recommendation, 'roadmap') and hasattr(recommendation.roadmap, 'market_demand'):
            market_demand = getattr(recommendation.roadmap, 'market_demand', 0)
            base_score += market_demand * 0.3
        
        # Bonus pour l'impact salarial
        if hasattr(recommendation, 'salary_impact'):
            salary_impact = getattr(recommendation, 'salary_impact', 0)
            base_score += salary_impact * 0.2
        elif hasattr(recommendation, 'skill') and hasattr(recommendation.skill, 'average_salary_impact'):
            salary_impact = getattr(recommendation.skill, 'average_salary_impact', 0)
            base_score += salary_impact * 0.2
        
        return min(base_score, 1.0)
    
    def _score_user_fit(self, recommendation: BaseRecommendation, user_features: UserFeatures) -> float:
        """Score d'adéquation avec le profil utilisateur"""
        base_score = 0.5
        
        # Bonus pour l'expérience appropriée
        experience_level = user_features.experience_features.get('level_numeric', 0.5)
        if hasattr(recommendation, 'difficulty'):
            difficulty = getattr(recommendation, 'difficulty', None)
            if self._is_difficulty_appropriate(difficulty, experience_level):
                base_score += 0.3
        
        # Bonus pour l'alignement avec l'industrie
        industry_score = user_features.experience_features.get('industry_score', 0.6)
        base_score += industry_score * 0.2
        
        return min(base_score, 1.0)
    
    def _score_growth_potential(self, recommendation: BaseRecommendation) -> float:
        """Score de potentiel de croissance"""
        base_score = 0.6
        
        # Bonus pour les technologies émergentes
        if hasattr(recommendation, 'future_relevance'):
            future_relevance = getattr(recommendation, 'future_relevance', 0)
            base_score += future_relevance * 0.3
        elif hasattr(recommendation, 'skill') and hasattr(recommendation.skill, 'future_relevance'):
            future_relevance = getattr(recommendation.skill, 'future_relevance', 0)
            base_score += future_relevance * 0.3
        
        # Bonus pour l'innovation
        if any(keyword in recommendation.title.lower() for keyword in ['ai', 'machine learning', 'cloud', 'blockchain']):
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _calculate_weighted_score(self, individual_scores: Dict[str, float]) -> float:
        """Calcule le score pondéré"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor, score in individual_scores.items():
            weight = self.scoring_weights.get(factor, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_neural_score(
        self, 
        recommendation: BaseRecommendation, 
        user_features: UserFeatures
    ) -> float:
        """Simule un score neural (à remplacer par un vrai modèle)"""
        # Simulation basée sur des features clés
        features = [
            sum(user_features.skill_vector) / len(user_features.skill_vector),
            user_features.experience_features.get('level_numeric', 0.5),
            user_features.preference_features.get('budget_score', 0.5),
            len(user_features.career_features.get('skills', [])) / 10.0,
            random.random() * 0.2  # Facteur de variabilité
        ]
        
        # Simulation d'un réseau de neurones simple
        weighted_features = sum(f * w for f, w in zip(features, [0.3, 0.25, 0.2, 0.15, 0.1]))
        return min(max(weighted_features, 0.0), 1.0)
    
    def _calculate_confidence(
        self, 
        individual_scores: Dict[str, float], 
        weighted_score: float, 
        neural_score: float
    ) -> float:
        """Calcule la confiance du scoring"""
        # Variance des scores individuels
        scores = list(individual_scores.values())
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        
        # Cohérence entre scores pondéré et neural
        coherence = 1.0 - abs(weighted_score - neural_score)
        
        # Confiance basée sur la variance (faible variance = haute confiance)
        variance_confidence = max(0.5, 1.0 - variance)
        
        # Confiance combinée
        confidence = (variance_confidence * 0.6 + coherence * 0.4)
        return min(max(confidence, 0.3), 0.95)  # Entre 30% et 95%
    
    def _generate_scoring_explanation(
        self, 
        individual_scores: Dict[str, float],
        recommendation: BaseRecommendation,
        user_features: UserFeatures
    ) -> Dict[str, Any]:
        """Génère une explication du scoring"""
        strengths = []
        considerations = []
        
        # Analyse des points forts
        for factor, score in individual_scores.items():
            if score > 0.8:
                strengths.append(f"Excellent {factor} ({score:.0%})")
            elif score > 0.6:
                strengths.append(f"Bon {factor} ({score:.0%})")
        
        # Analyse des points d'attention
        for factor, score in individual_scores.items():
            if score < 0.4:
                considerations.append(f"{factor} pourrait être amélioré ({score:.0%})")
        
        # Points forts par défaut
        if not strengths:
            strengths.append("Recommandation bien alignée avec votre profil")
        
        return {
            'strengths': strengths,
            'considerations': considerations,
            'top_factor': max(individual_scores.items(), key=lambda x: x[1])[0],
            'improvement_area': min(individual_scores.items(), key=lambda x: x[1])[0]
        }
    
    def _is_difficulty_appropriate(self, difficulty, experience_level: float) -> bool:
        """Vérifie si la difficulté est appropriée"""
        if not difficulty:
            return True
        
        difficulty_str = difficulty.value if hasattr(difficulty, 'value') else str(difficulty).lower()
        
        difficulty_ranges = {
            'beginner': (0.0, 0.4),
            'intermediate': (0.3, 0.7),
            'advanced': (0.6, 0.9),
            'expert': (0.8, 1.0)
        }
        
        if difficulty_str in difficulty_ranges:
            min_level, max_level = difficulty_ranges[difficulty_str]
            return min_level <= experience_level <= max_level
        
        return True
    
    def _default_scoring_result(self) -> ScoringResult:
        """Retourne un résultat de scoring par défaut en cas d'erreur"""
        default_scores = {
            'relevance': 0.6,
            'feasibility': 0.7,
            'market_value': 0.6,
            'user_fit': 0.5,
            'growth_potential': 0.6
        }
        
        return ScoringResult(
            individual_scores=default_scores,
            weighted_score=0.6,
            neural_score=0.6,
            combined_score=0.6,
            confidence=0.5,
            explanation={
                'strengths': ['Recommandation standard'],
                'considerations': ['Scoring par défaut appliqué'],
                'top_factor': 'feasibility',
                'improvement_area': 'user_fit'
            }
        )
