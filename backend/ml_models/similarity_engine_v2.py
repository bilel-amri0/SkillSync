"""
Similarity & Matching Engine (F3) - Production Version
Semantic matching between CVs and Job Descriptions using Sentence Transformers.
This is the enhanced production-ready version with comprehensive matching logic.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install: pip install sentence-transformers")

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of CV-Job matching"""
    score: float  # 0-100
    match_level: str  # Low, Medium, High
    semantic_similarity: float  # 0-1
    keyword_overlap: float  # 0-1
    explanation: str
    matched_keywords: List[str]
    missing_keywords: List[str]
    confidence: float


class SimilarityEngineV2:
    """
    Production-ready Similarity Engine for CV-Job matching.
    Uses Sentence Transformers for semantic embeddings and cosine similarity.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        cache_embeddings: bool = True
    ):
        """
        Initialize Similarity Engine with Sentence Transformer model.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.cache_embeddings = cache_embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Initialize model
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"🔄 Loading Sentence Transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info(f"✅ Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
                self.model = None
        else:
            logger.error("❌ sentence-transformers not installed")
        
        # Matching thresholds
        self.thresholds = {
            'low': 0.0,
            'medium': 0.4,
            'high': 0.7
        }
    
    def calculate_match_score(
        self, 
        cv_text: str, 
        job_text: str,
        cv_skills: Optional[List[str]] = None,
        job_requirements: Optional[List[str]] = None
    ) -> MatchResult:
        """
        Calculate comprehensive match score between CV and Job Description.
        
        Args:
            cv_text: Full CV text
            job_text: Full job description text
            cv_skills: Optional list of extracted CV skills
            job_requirements: Optional list of job requirements
            
        Returns:
            MatchResult with score and detailed breakdown
        """
        logger.info("🎯 Calculating CV-Job match score...")
        
        try:
            # Component 1: Semantic Similarity (60% weight)
            semantic_score = self._calculate_semantic_similarity(cv_text, job_text)
            
            # Component 2: Keyword Overlap (40% weight)
            keyword_score = 0.5  # Default
            matched_keywords = []
            missing_keywords = []
            
            if cv_skills and job_requirements:
                keyword_score, matched_keywords, missing_keywords = self._calculate_keyword_overlap(
                    cv_skills, job_requirements
                )
            
            # Composite score (weighted average)
            final_score = (semantic_score * 0.6) + (keyword_score * 0.4)
            final_score_percentage = final_score * 100
            
            # Determine match level
            match_level = self._determine_match_level(final_score)
            
            # Generate explanation
            explanation = self._generate_explanation(
                final_score, semantic_score, keyword_score,
                len(matched_keywords), len(missing_keywords)
            )
            
            # Calculate confidence (higher if both components agree)
            confidence = 1.0 - abs(semantic_score - keyword_score)
            
            result = MatchResult(
                score=round(final_score_percentage, 2),
                match_level=match_level,
                semantic_similarity=round(semantic_score, 3),
                keyword_overlap=round(keyword_score, 3),
                explanation=explanation,
                matched_keywords=matched_keywords,
                missing_keywords=missing_keywords,
                confidence=round(confidence, 3)
            )
            
            logger.info(f"✅ Match Score: {result.score}% ({result.match_level})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error calculating match score: {str(e)}", exc_info=True)
            # Return fallback result
            return MatchResult(
                score=50.0,
                match_level="Medium",
                semantic_similarity=0.5,
                keyword_overlap=0.5,
                explanation="Unable to calculate precise match score",
                matched_keywords=[],
                missing_keywords=[],
                confidence=0.3
            )
    
    def _calculate_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            text1: First text (CV)
            text2: Second text (Job Description)
            
        Returns:
            Similarity score (0-1)
        """
        if not self.model:
            logger.warning("Model not available, returning default similarity")
            return 0.5
        
        try:
            # Get embeddings (with caching)
            emb1 = self._get_embedding(text1)
            emb2 = self._get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            
            # Ensure in valid range [0, 1]
            similarity = np.clip(similarity, 0.0, 1.0)
            
            logger.debug(f"Semantic similarity: {similarity:.3f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.5
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, with optional caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (numpy array)
        """
        # Create cache key (hash of first 500 chars for efficiency)
        cache_key = hash(text[:500])
        
        # Check cache
        if self.cache_embeddings and cache_key in self.embedding_cache:
            logger.debug("📦 Using cached embedding")
            return self.embedding_cache[cache_key]
        
        # Generate embedding
        # Truncate to 512 tokens for performance
        truncated_text = text[:2000]  # ~512 tokens
        embedding = self.model.encode(
            truncated_text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Cache if enabled
        if self.cache_embeddings:
            self.embedding_cache[cache_key] = embedding
            
            # Limit cache size to 1000 entries
            if len(self.embedding_cache) > 1000:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
        
        return embedding
    
    def _calculate_keyword_overlap(
        self,
        cv_skills: List[str],
        job_requirements: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """
        Calculate keyword overlap between CV skills and job requirements.
        
        Args:
            cv_skills: List of skills from CV
            job_requirements: List of required skills from job
            
        Returns:
            Tuple of (overlap_score, matched_keywords, missing_keywords)
        """
        # Normalize for case-insensitive comparison
        cv_skills_lower = [s.lower().strip() for s in cv_skills]
        job_requirements_lower = [s.lower().strip() for s in job_requirements]
        
        # Find exact matches
        matched = []
        for req in job_requirements_lower:
            if req in cv_skills_lower:
                matched.append(req)
        
        # Find missing requirements
        missing = [req for req in job_requirements_lower if req not in cv_skills_lower]
        
        # Calculate overlap score
        if len(job_requirements_lower) == 0:
            overlap_score = 0.5  # No requirements specified
        else:
            overlap_score = len(matched) / len(job_requirements_lower)
        
        logger.debug(f"Keyword overlap: {len(matched)}/{len(job_requirements_lower)} = {overlap_score:.2f}")
        
        return overlap_score, matched, missing
    
    def _determine_match_level(self, score: float) -> str:
        """
        Determine match level category based on score.
        
        Args:
            score: Match score (0-1)
            
        Returns:
            Match level string: Low, Medium, or High
        """
        if score >= self.thresholds['high']:
            return "High"
        elif score >= self.thresholds['medium']:
            return "Medium"
        else:
            return "Low"
    
    def _generate_explanation(
        self,
        final_score: float,
        semantic_score: float,
        keyword_score: float,
        matched_count: int,
        missing_count: int
    ) -> str:
        """
        Generate human-readable explanation of match score.
        
        Args:
            final_score: Overall match score
            semantic_score: Semantic similarity component
            keyword_score: Keyword overlap component
            matched_count: Number of matched keywords
            missing_count: Number of missing keywords
            
        Returns:
            Explanation string
        """
        match_level = self._determine_match_level(final_score)
        
        explanations = {
            "High": "Strong match! Your CV aligns well with the job requirements.",
            "Medium": "Moderate match. You meet many requirements but have some gaps.",
            "Low": "Limited match. Significant skill gaps exist for this position."
        }
        
        base_explanation = explanations.get(match_level, "Match assessment completed.")
        
        # Add details
        details = []
        
        if semantic_score >= 0.7:
            details.append("Your experience context is highly relevant")
        elif semantic_score >= 0.4:
            details.append("Your experience has moderate relevance")
        else:
            details.append("Your experience shows limited alignment")
        
        if keyword_score >= 0.7:
            details.append(f"you have {matched_count} of the required skills")
        elif keyword_score >= 0.4:
            details.append(f"you have {matched_count} skills but are missing {missing_count} key requirements")
        else:
            details.append(f"you're missing {missing_count} critical requirements")
        
        full_explanation = f"{base_explanation} {details[0]} and {details[1]}."
        
        return full_explanation
    
    def batch_calculate_matches(
        self,
        cv_text: str,
        job_descriptions: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Calculate match scores for multiple jobs (batch processing).
        
        Args:
            cv_text: CV text
            job_descriptions: List of job dicts with 'text' and 'requirements'
            top_k: Number of top matches to return
            
        Returns:
            List of jobs with match scores, sorted by score descending
        """
        logger.info(f"🔄 Batch calculating matches for {len(job_descriptions)} jobs...")
        
        results = []
        
        for i, job in enumerate(job_descriptions):
            try:
                job_text = job.get('text', job.get('description', ''))
                job_requirements = job.get('requirements', [])
                
                match_result = self.calculate_match_score(
                    cv_text=cv_text,
                    job_text=job_text,
                    cv_skills=None,  # Would be extracted from CV
                    job_requirements=job_requirements
                )
                
                results.append({
                    'job': job,
                    'match_score': match_result.score,
                    'match_level': match_result.match_level,
                    'semantic_similarity': match_result.semantic_similarity,
                    'keyword_overlap': match_result.keyword_overlap,
                    'matched_keywords': match_result.matched_keywords,
                    'missing_keywords': match_result.missing_keywords,
                    'explanation': match_result.explanation
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"✅ Processed {i + 1}/{len(job_descriptions)} jobs")
                    
            except Exception as e:
                logger.error(f"Error processing job {i}: {str(e)}")
                continue
        
        # Sort by match score (descending)
        results.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Return top K
        top_results = results[:top_k]
        
        logger.info(f"✅ Batch matching complete. Top score: {top_results[0]['match_score']}%")
        return top_results
    
    def clear_cache(self) -> int:
        """
        Clear the embedding cache.
        
        Returns:
            Number of entries cleared
        """
        count = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"🧹 Cleared {count} cached embeddings")
        return count
