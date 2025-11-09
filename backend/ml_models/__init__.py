"""
ML Models Package - Advanced Machine Learning Components for SkillSync

Contains:
- BERT-based NER for skills extraction
- Sentence-Transformers for semantic similarity
- Neural scoring models for recommendations
"""

from .skills_extractor import SkillsExtractorModel
from .similarity_engine import SimilarityEngine
from .neural_scorer import NeuralScorer
from .ml_trainer import MLTrainer

__all__ = [
    'SkillsExtractorModel',
    'SimilarityEngine', 
    'NeuralScorer',
    'MLTrainer'
]
