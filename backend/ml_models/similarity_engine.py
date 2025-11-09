"""
Semantic Similarity Engine using Sentence-Transformers
Adapted from notebook for production use
"""

import os
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging
import pickle
import json
from datetime import datetime

# Sentence Transformers with error handling
try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    from torch.utils.data import DataLoader
except ImportError as e:
    logging.warning(f"Sentence-transformers not available: {e}")
    SentenceTransformer = None

logger = logging.getLogger(__name__)

class SimilarityEngine:
    """
    Production-ready semantic similarity engine for CV-Job matching
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.cache = {}
        
        # Initialize model if available
        if SentenceTransformer is not None:
            self._initialize_model()
        else:
            logger.warning("Sentence-transformers not available, using fallback similarity")
    
    def _initialize_model(self):
        """Initialize Sentence-Transformers model"""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading fine-tuned model from {self.model_path}")
                self.model = SentenceTransformer(self.model_path)
            else:
                logger.info(f"Loading base model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
            
            logger.info("Similarity model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading similarity model: {e}")
            self.model = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector
        """
        if self.model is None:
            return self._fallback_encoding(text)
        
        try:
            # Check cache first
            cache_key = hash(text)
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Encode text
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Cache result
            self.cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return self._fallback_encoding(text)
    
    def _fallback_encoding(self, text: str) -> np.ndarray:
        """
        Simple fallback encoding using TF-IDF-like approach
        """
        # Simple bag of words encoding
        words = text.lower().split()
        vocab_size = 1000  # Fixed vocabulary size
        encoding = np.zeros(vocab_size)
        
        for word in words:
            # Simple hash to vocabulary index
            idx = hash(word) % vocab_size
            encoding[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm
            
        return encoding
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        """
        try:
            embedding1 = self.encode_text(text1)
            embedding2 = self.encode_text(text2)
            
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def calculate_cv_job_similarity(self, cv_data: Dict, job_data: Dict) -> Dict[str, float]:
        """
        Calculate comprehensive similarity between CV and job
        """
        similarities = {}
        
        try:
            # Extract text representations
            cv_text = self._extract_cv_text(cv_data)
            job_text = self._extract_job_text(job_data)
            
            # Overall similarity
            similarities['overall'] = self.calculate_similarity(cv_text, job_text)
            
            # Skills similarity
            cv_skills = cv_data.get('skills', [])
            job_skills = job_data.get('required_skills', [])
            if cv_skills and job_skills:
                cv_skills_text = ' '.join(cv_skills)
                job_skills_text = ' '.join(job_skills)
                similarities['skills'] = self.calculate_similarity(cv_skills_text, job_skills_text)
            
            # Experience similarity
            cv_experience = cv_data.get('experience_text', '')
            job_requirements = job_data.get('requirements', '')
            if cv_experience and job_requirements:
                similarities['experience'] = self.calculate_similarity(cv_experience, job_requirements)
            
            # Industry/domain similarity
            cv_industry = cv_data.get('industry', '')
            job_industry = job_data.get('industry', '')
            if cv_industry and job_industry:
                similarities['industry'] = self.calculate_similarity(cv_industry, job_industry)
                
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating CV-job similarity: {e}")
            return {'overall': 0.0}
    
    def _extract_cv_text(self, cv_data: Dict) -> str:
        """
        Extract comprehensive text from CV data
        """
        text_parts = []
        
        # Add skills
        skills = cv_data.get('skills', [])
        if skills:
            text_parts.append(' '.join(skills))
        
        # Add experience
        experience = cv_data.get('experience', [])
        for exp in experience:
            if isinstance(exp, dict):
                text_parts.append(exp.get('description', ''))
                text_parts.append(exp.get('role', ''))
            elif isinstance(exp, str):
                text_parts.append(exp)
        
        # Add education
        education = cv_data.get('education', [])
        for edu in education:
            if isinstance(edu, dict):
                text_parts.append(edu.get('degree', ''))
                text_parts.append(edu.get('field', ''))
            elif isinstance(edu, str):
                text_parts.append(edu)
        
        # Add summary/objective
        summary = cv_data.get('summary', '') or cv_data.get('objective', '')
        if summary:
            text_parts.append(summary)
        
        return ' '.join(filter(None, text_parts))
    
    def _extract_job_text(self, job_data: Dict) -> str:
        """
        Extract comprehensive text from job data
        """
        text_parts = []
        
        # Add title
        title = job_data.get('title', '')
        if title:
            text_parts.append(title)
        
        # Add description
        description = job_data.get('description', '')
        if description:
            text_parts.append(description)
        
        # Add requirements
        requirements = job_data.get('requirements', '')
        if requirements:
            text_parts.append(requirements)
        
        # Add skills
        skills = job_data.get('required_skills', [])
        if skills:
            text_parts.append(' '.join(skills))
        
        # Add company info
        company = job_data.get('company', '')
        if company:
            text_parts.append(company)
        
        return ' '.join(filter(None, text_parts))
    
    def batch_similarity(self, reference_text: str, candidate_texts: List[str]) -> List[float]:
        """
        Calculate similarity between reference text and multiple candidates
        """
        similarities = []
        reference_embedding = self.encode_text(reference_text)
        
        for text in candidate_texts:
            candidate_embedding = self.encode_text(text)
            similarity = np.dot(reference_embedding, candidate_embedding) / (
                np.linalg.norm(reference_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append(float(similarity))
        
        return similarities
    
    def find_best_matches(self, cv_data: Dict, job_list: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Find best job matches for a CV
        """
        cv_text = self._extract_cv_text(cv_data)
        matches = []
        
        for job in job_list:
            job_text = self._extract_job_text(job)
            similarity = self.calculate_similarity(cv_text, job_text)
            
            match = {
                'job': job,
                'similarity': similarity,
                'similarity_details': self.calculate_cv_job_similarity(cv_data, job)
            }
            matches.append(match)
        
        # Sort by similarity
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return matches[:top_k]
    
    def explain_similarity(self, cv_data: Dict, job_data: Dict) -> Dict[str, Any]:
        """
        Provide detailed explanation of similarity calculation
        """
        similarities = self.calculate_cv_job_similarity(cv_data, job_data)
        
        explanation = {
            'overall_score': similarities.get('overall', 0.0),
            'breakdown': similarities,
            'strengths': [],
            'gaps': [],
            'recommendations': []
        }
        
        # Analyze strengths and gaps
        cv_skills = set(cv_data.get('skills', []))
        job_skills = set(job_data.get('required_skills', []))
        
        matching_skills = cv_skills.intersection(job_skills)
        missing_skills = job_skills - cv_skills
        extra_skills = cv_skills - job_skills
        
        if matching_skills:
            explanation['strengths'].append(f"Matching skills: {', '.join(matching_skills)}")
        
        if missing_skills:
            explanation['gaps'].append(f"Missing skills: {', '.join(missing_skills)}")
            explanation['recommendations'].append(f"Consider learning: {', '.join(list(missing_skills)[:3])}")
        
        if extra_skills:
            explanation['strengths'].append(f"Additional valuable skills: {', '.join(list(extra_skills)[:3])}")
        
        return explanation
