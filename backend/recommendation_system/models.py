"""Modèles de données pour le système de recommandations étendu"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class RecommendationType(Enum):
    """Types de recommandations supportés"""
    ROADMAP = "roadmap"
    CERTIFICATION = "certification"
    SKILL = "skill"
    PROJECT = "project"
    JOB = "job"


class DifficultyLevel(Enum):
    """Niveaux de difficulté"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ExperienceLevel(Enum):
    """Niveaux d'expérience"""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    EXECUTIVE = "executive"


@dataclass
class UserProfile:
    """Profil utilisateur complet"""
    user_id: str
    cv_data: Dict[str, Any]
    current_skills: List[str] = field(default_factory=list)
    experience_years: int = 0
    current_role: str = ""
    industry: str = ""
    level: ExperienceLevel = ExperienceLevel.JUNIOR
    career_goals: List[str] = field(default_factory=list)
    learning_preferences: Dict[str, Any] = field(default_factory=dict)
    time_availability: str = "5-10h/week"  # heures par semaine
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class RecommendationPreferences:
    """Préférences pour les recommandations"""
    max_recommendations: int = 20
    focus_areas: List[str] = field(default_factory=list)
    difficulty_preference: Optional[DifficultyLevel] = None
    time_commitment: str = "moderate"  # "low", "moderate", "high"
    budget_range: str = "free-premium"
    learning_style: List[str] = field(default_factory=lambda: ["hands-on"])
    
    # Préférences par type
    roadmap_preferences: Dict[str, Any] = field(default_factory=dict)
    cert_preferences: Dict[str, Any] = field(default_factory=dict)
    skill_preferences: Dict[str, Any] = field(default_factory=dict)
    project_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoadmapStep:
    """Étape d'un roadmap de carrière"""
    step_number: int
    title: str
    description: str
    duration: str  # "2-4 weeks"
    skills_to_learn: List[str]
    resources: List[Dict[str, str]]
    projects: List[str]  # Références vers projets
    certifications: List[str]  # Certifications recommandées
    prerequisites: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)


@dataclass
class CareerRoadmap:
    """Roadmap de carrière complet"""
    id: str
    title: str
    domain: str  # "data_science", "web_dev", "devops"
    level_progression: List[ExperienceLevel]
    estimated_duration: str  # "6-12 months"
    difficulty: DifficultyLevel
    
    # Étapes détaillées
    steps: List[RoadmapStep]
    
    # Métadonnées
    required_base_skills: List[str] = field(default_factory=list)
    target_roles: List[str] = field(default_factory=list)
    industry_relevance: List[str] = field(default_factory=list)
    popularity_score: float = 0.0
    market_demand: float = 0.0
    avg_salary_impact: float = 0.0
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class Certification:
    """Certification professionnelle"""
    id: str
    title: str
    provider: str  # "AWS", "Google", "Microsoft", "Coursera"
    domain: str
    level: str  # "associate", "professional", "expert"
    
    # Détails pratiques
    cost: float
    duration_prep: str  # "2-3 months"
    exam_format: str  # "multiple_choice", "hands_on", "mixed"
    validity_period: str  # "3 years"
    
    # Prérequis et compétences
    required_skills: List[str] = field(default_factory=list)
    skills_gained: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Impact carrière
    market_value: float = 0.0
    salary_impact: float = 0.0
    job_opportunities: int = 0
    industry_recognition: float = 0.0
    
    # Métadonnées
    pass_rate: float = 0.0
    difficulty_rating: float = 0.0
    updated_date: datetime = field(default_factory=datetime.now)


@dataclass
class PracticalProject:
    """Projet pratique pour développer des compétences"""
    id: str
    title: str
    description: str
    domain: str
    difficulty: DifficultyLevel
    
    # Spécifications techniques
    technologies: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    
    # Apprentissage
    skills_practiced: List[str] = field(default_factory=list)
    skills_gained: List[str] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    
    # Métadonnées
    estimated_time: str = "2-4 weeks"
    complexity_score: float = 0.0
    portfolio_value: float = 0.0  # Impact sur le portfolio
    industry_relevance: List[str] = field(default_factory=list)
    
    # Ressources
    tutorial_links: List[str] = field(default_factory=list)
    github_templates: List[str] = field(default_factory=list)
    demo_links: List[str] = field(default_factory=list)


@dataclass
class Skill:
    """Compétence à développer"""
    id: str
    name: str
    category: str  # "technical", "soft", "domain"
    subcategory: str  # "programming", "cloud", "communication"
    
    # Caractéristiques
    difficulty_to_learn: float = 0.0
    market_demand: float = 0.0
    future_relevance: float = 0.0
    average_salary_impact: float = 0.0
    
    # Relations
    prerequisites: List[str] = field(default_factory=list)
    complementary_skills: List[str] = field(default_factory=list)
    career_paths: List[str] = field(default_factory=list)
    
    # Apprentissage
    learning_resources: List[Dict[str, str]] = field(default_factory=list)
    estimated_learning_time: str = "4-8 weeks"
    practical_projects: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)


@dataclass
class ScoringResult:
    """Résultat de scoring pour une recommandation"""
    individual_scores: Dict[str, float]
    weighted_score: float
    neural_score: float
    combined_score: float
    confidence: float
    explanation: Dict[str, Any]


@dataclass
class BaseRecommendation:
    """Recommandation de base - CORRECTION: suppression des valeurs par défaut"""
    id: str
    recommendation_type: RecommendationType
    title: str
    description: str
    scores: Dict[str, float]
    explanation: Dict[str, Any]
    confidence: float
    created_at: datetime


@dataclass
class RoadmapRecommendation(BaseRecommendation):
    """Recommandation de roadmap"""
    roadmap: CareerRoadmap
    match_reason: str
    progression_fit: float
    next_steps: List[str] = field(default_factory=list)


@dataclass
class CertificationRecommendation(BaseRecommendation):
    """Recommandation de certification"""
    certification: Certification
    match_reason: str
    preparation_estimate: str
    roi_estimate: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRecommendation(BaseRecommendation):
    """Recommandation de compétence"""
    skill: Skill
    priority: str  # "high", "medium", "low"
    learning_path: List[str] = field(default_factory=list)
    immediate_benefits: List[str] = field(default_factory=list)


@dataclass
class ProjectRecommendation(BaseRecommendation):
    """Recommandation de projet"""
    project: PracticalProject
    learning_value: float
    portfolio_impact: float
    skill_development: List[str] = field(default_factory=list)


@dataclass
class ComprehensiveRecommendations:
    """Ensemble complet de recommandations"""
    user_profile: UserProfile
    recommendations: Dict[str, List[BaseRecommendation]]
    global_explanation: Dict[str, Any]
    confidence: float
    generated_at: datetime
    expires_at: Optional[datetime] = None


@dataclass
class UserFeatures:
    """Features extraites du profil utilisateur pour le ML"""
    skill_vector: List[float]
    experience_features: Dict[str, float]
    career_features: Dict[str, float]
    preference_features: Dict[str, float]
    contextual_features: Dict[str, float]


@dataclass
class RecommendationFeedback:
    """Feedback utilisateur sur une recommandation"""
    user_id: str
    recommendation_id: str
    feedback_type: str  # 'like', 'dislike', 'applied', 'completed', 'not_relevant'
    created_at: datetime
    rating: Optional[int] = None  # 1-5
    comment: Optional[str] = None
