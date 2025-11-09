# Système de Recommandations Multicritères SkillSync
"""Système avancé de recommandations pour carrière complète"""

__version__ = "2.0.0"
__author__ = "SkillSync Team"

# Types de recommandations supportés
RECOMMENDATION_TYPES = [
    'roadmaps',
    'certifications', 
    'skills',
    'projects',
    'jobs'
]

# Configuration par défaut
DEFAULT_CONFIG = {
    'max_recommendations_per_type': 5,
    'min_confidence_threshold': 0.6,
    'diversity_weight': 0.3,
    'personalization_weight': 0.4
}