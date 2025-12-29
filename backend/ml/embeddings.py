"""
SkillSync ML Pipeline - Semantic Embeddings Engine
===================================================
Real SBERT embeddings for CV, jobs, and skills semantic matching.
NO templates, NO regex - pure neural embeddings.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Lazy imports to handle missing dependencies gracefully
_sentence_transformer = None
_model = None


def _get_sentence_transformer():
    """Lazy load sentence-transformers"""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers not installed. Install: pip install sentence-transformers")
            raise
    return _sentence_transformer


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: np.ndarray
    model_name: str
    dimension: int
    
    def to_list(self) -> List[float]:
        """Convert embedding to list for JSON serialization"""
        return self.embedding.tolist()


class SemanticEmbeddingEngine:
    """
    Real semantic embedding engine using sentence-transformers.
    
    Models supported:
    - all-MiniLM-L6-v2 (384 dim, fast, lightweight)
    - all-mpnet-base-v2 (768 dim, best quality)
    - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None
    ):
        """
        Initialize semantic embedding engine.
        
        Args:
            model_name: HuggingFace model name
            device: 'cpu' or 'cuda' (auto-detect if None)
            cache_folder: Where to cache downloaded models
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"🚀 Initializing SemanticEmbeddingEngine: {model_name}")
        logger.info(f"📱 Device: {self.device}")
        
        # Load model
        SentenceTransformer = _get_sentence_transformer()
        
        try:
            if cache_folder:
                self.model = SentenceTransformer(model_name, device=self.device, cache_folder=cache_folder)
            else:
                self.model = SentenceTransformer(model_name, device=self.device)
                
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded successfully (dim={self.embedding_dimension})")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            raise
    
    def encode_single(self, text: str, normalize: bool = True) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            EmbeddingResult with embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            zero_vector = np.zeros(self.embedding_dimension)
            return EmbeddingResult(
                text="",
                embedding=zero_vector,
                model_name=self.model_name,
                dimension=self.embedding_dimension
            )
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False
            )
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self.model_name,
                dimension=self.embedding_dimension
            )
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_batch(
        self, 
        texts: List[str], 
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts (efficient batching).
        
        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            List of EmbeddingResult
        """
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                batch_size=batch_size,
                show_progress_bar=show_progress
            )
            
            results = []
            for text, embedding in zip(texts, embeddings):
                results.append(EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model_name=self.model_name,
                    dimension=self.embedding_dimension
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            raise
    
    def encode_cv(self, cv_data: Dict[str, Any]) -> EmbeddingResult:
        """
        Generate embedding for entire CV.
        
        Args:
            cv_data: CV data with text, skills, experience
            
        Returns:
            EmbeddingResult for full CV
        """
        # Combine relevant CV fields into one text
        cv_text_parts = []
        
        # Add summary/objective
        if cv_data.get("summary"):
            cv_text_parts.append(cv_data["summary"])
        
        # Add skills
        if cv_data.get("skills"):
            skills = cv_data["skills"]
            if isinstance(skills, list):
                cv_text_parts.append("Skills: " + ", ".join(skills))
            elif isinstance(skills, str):
                cv_text_parts.append("Skills: " + skills)
        
        # Add experience
        if cv_data.get("experience"):
            exp = cv_data["experience"]
            if isinstance(exp, list):
                for job in exp:
                    if isinstance(job, dict):
                        title = job.get("title", "")
                        description = job.get("description", "")
                        cv_text_parts.append(f"{title}: {description}")
                    elif isinstance(job, str):
                        cv_text_parts.append(job)
            elif isinstance(exp, str):
                cv_text_parts.append(exp)
        
        # Add education
        if cv_data.get("education"):
            edu = cv_data["education"]
            if isinstance(edu, list):
                for degree in edu:
                    if isinstance(degree, dict):
                        cv_text_parts.append(degree.get("degree", "") + " " + degree.get("field", ""))
                    elif isinstance(degree, str):
                        cv_text_parts.append(degree)
            elif isinstance(edu, str):
                cv_text_parts.append(edu)
        
        # Add raw text if available
        if cv_data.get("text"):
            cv_text_parts.append(cv_data["text"])
        
        full_cv_text = " ".join(cv_text_parts).strip()
        
        if not full_cv_text:
            logger.warning("CV has no extractable text")
            full_cv_text = "Empty CV"
        
        logger.info(f"Encoding CV ({len(full_cv_text)} chars)")
        return self.encode_single(full_cv_text)
    
    def encode_job(self, job_data: Dict[str, Any]) -> EmbeddingResult:
        """
        Generate embedding for job description.
        
        Args:
            job_data: Job posting data
            
        Returns:
            EmbeddingResult for job
        """
        job_text_parts = []
        
        # Add title
        if job_data.get("title"):
            job_text_parts.append(job_data["title"])
        
        # Add description
        if job_data.get("description"):
            job_text_parts.append(job_data["description"])
        
        # Add required skills
        if job_data.get("skills_required"):
            skills = job_data["skills_required"]
            if isinstance(skills, list):
                job_text_parts.append("Required skills: " + ", ".join(skills))
            elif isinstance(skills, str):
                job_text_parts.append("Required skills: " + skills)
        
        # Add requirements
        if job_data.get("requirements"):
            req = job_data["requirements"]
            if isinstance(req, list):
                job_text_parts.append(" ".join(req))
            elif isinstance(req, str):
                job_text_parts.append(req)
        
        full_job_text = " ".join(job_text_parts).strip()
        
        if not full_job_text:
            logger.warning("Job has no extractable text")
            full_job_text = "Empty job"
        
        logger.info(f"Encoding job ({len(full_job_text)} chars)")
        return self.encode_single(full_job_text)
    
    def cosine_similarity(
        self, 
        embedding1: Union[np.ndarray, EmbeddingResult],
        embedding2: Union[np.ndarray, EmbeddingResult]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (array or EmbeddingResult)
            embedding2: Second embedding (array or EmbeddingResult)
            
        Returns:
            Similarity score [0.0, 1.0]
        """
        # Extract numpy arrays if EmbeddingResult
        if isinstance(embedding1, EmbeddingResult):
            embedding1 = embedding1.embedding
        if isinstance(embedding2, EmbeddingResult):
            embedding2 = embedding2.embedding
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure in range [0, 1] (normalized embeddings should already be)
        similarity = float(np.clip(similarity, -1.0, 1.0))
        
        # Convert from [-1, 1] to [0, 1]
        similarity = (similarity + 1.0) / 2.0
        
        return similarity
    
    def batch_similarity(
        self,
        query_embedding: Union[np.ndarray, EmbeddingResult],
        candidate_embeddings: List[Union[np.ndarray, EmbeddingResult]]
    ) -> List[float]:
        """
        Calculate similarity between one query and multiple candidates (fast).
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            
        Returns:
            List of similarity scores
        """
        # Extract numpy array
        if isinstance(query_embedding, EmbeddingResult):
            query_embedding = query_embedding.embedding
        
        # Stack candidate embeddings
        candidates = []
        for emb in candidate_embeddings:
            if isinstance(emb, EmbeddingResult):
                candidates.append(emb.embedding)
            else:
                candidates.append(emb)
        
        candidates_matrix = np.vstack(candidates)
        
        # Vectorized cosine similarity
        dot_products = np.dot(candidates_matrix, query_embedding)
        norms = np.linalg.norm(candidates_matrix, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        
        if query_norm == 0:
            return [0.0] * len(candidates)
        
        similarities = dot_products / (norms * query_norm + 1e-8)
        
        # Convert to [0, 1] range
        similarities = np.clip(similarities, -1.0, 1.0)
        similarities = (similarities + 1.0) / 2.0
        
        return similarities.tolist()


# Global singleton instance
_global_embedding_engine: Optional[SemanticEmbeddingEngine] = None


def get_embedding_engine(model_name: str = "all-MiniLM-L6-v2") -> SemanticEmbeddingEngine:
    """
    Get or create global embedding engine (singleton pattern).
    
    Args:
        model_name: Model to use if creating new instance
        
    Returns:
        SemanticEmbeddingEngine instance
    """
    global _global_embedding_engine
    
    if _global_embedding_engine is None:
        _global_embedding_engine = SemanticEmbeddingEngine(model_name=model_name)
    
    return _global_embedding_engine
