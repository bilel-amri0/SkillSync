"""
SkillSync ML Pipeline - ML-Based Recommender
=============================================
Embedding-based recommendations for projects, certifications, learning paths.
NO static dictionaries - dynamic ML recommendations.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from .embeddings import SemanticEmbeddingEngine, get_embedding_engine
from .ner import SkillNERExtractor, get_ner_extractor

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Single recommendation item"""
    id: str
    type: str  # project, certification, course, skill
    title: str
    description: str
    relevance_score: float  # 0-100
    skills_gained: List[str]
    estimated_time: str
    difficulty: str  # beginner, intermediate, advanced
    url: Optional[str] = None
    provider: Optional[str] = None


@dataclass
class RecommendationSet:
    """Complete set of recommendations"""
    projects: List[Recommendation]
    certifications: List[Recommendation]
    courses: List[Recommendation]
    skills_to_learn: List[str]
    learning_path: List[Dict[str, Any]]
    timestamp: str


class MLRecommender:
    """
    ML-based recommender using semantic embeddings.
    
    Recommends:
    - Projects based on skill gaps
    - Certifications aligned with career goals
    - Courses for skill development
    - Personalized learning paths
    """
    
    # Knowledge base of recommendations (will be replaced by vector DB)
    PROJECTS_DB = [
        {
            "id": "proj_001",
            "title": "Build a RESTful API with FastAPI",
            "description": "Create a production-ready REST API with authentication, database integration, and documentation",
            "skills": ["python", "fastapi", "postgresql", "docker"],
            "difficulty": "intermediate",
            "estimated_time": "2-3 weeks"
        },
        {
            "id": "proj_002",
            "title": "Machine Learning Pipeline for Predictive Analytics",
            "description": "Build an end-to-end ML pipeline with data preprocessing, model training, and deployment",
            "skills": ["python", "scikit-learn", "pandas", "machine learning"],
            "difficulty": "advanced",
            "estimated_time": "4-6 weeks"
        },
        {
            "id": "proj_003",
            "title": "React + TypeScript Dashboard Application",
            "description": "Develop a responsive dashboard with charts, real-time updates, and state management",
            "skills": ["react", "typescript", "javascript", "tailwind"],
            "difficulty": "intermediate",
            "estimated_time": "3-4 weeks"
        },
        {
            "id": "proj_004",
            "title": "Kubernetes Microservices Deployment",
            "description": "Deploy a microservices architecture on Kubernetes with CI/CD pipeline",
            "skills": ["kubernetes", "docker", "devops", "ci/cd"],
            "difficulty": "advanced",
            "estimated_time": "4-5 weeks"
        },
        {
            "id": "proj_005",
            "title": "Natural Language Processing Chatbot",
            "description": "Build an intelligent chatbot using transformers and conversation AI",
            "skills": ["python", "nlp", "transformers", "machine learning"],
            "difficulty": "advanced",
            "estimated_time": "5-6 weeks"
        }
    ]
    
    CERTIFICATIONS_DB = [
        {
            "id": "cert_001",
            "title": "AWS Certified Solutions Architect",
            "description": "Validate expertise in designing distributed systems on AWS",
            "skills": ["aws", "cloud", "architecture", "devops"],
            "provider": "Amazon Web Services",
            "difficulty": "intermediate",
            "url": "https://aws.amazon.com/certification/"
        },
        {
            "id": "cert_002",
            "title": "Professional Scrum Master (PSM I)",
            "description": "Demonstrate understanding of Scrum framework and agile principles",
            "skills": ["scrum", "agile", "project management"],
            "provider": "Scrum.org",
            "difficulty": "beginner",
            "url": "https://www.scrum.org/assessments/professional-scrum-master-i-certification"
        },
        {
            "id": "cert_003",
            "title": "TensorFlow Developer Certificate",
            "description": "Prove proficiency in building ML models with TensorFlow",
            "skills": ["tensorflow", "machine learning", "deep learning", "python"],
            "provider": "Google",
            "difficulty": "intermediate",
            "url": "https://www.tensorflow.org/certificate"
        },
        {
            "id": "cert_004",
            "title": "Certified Kubernetes Administrator (CKA)",
            "description": "Validate skills in Kubernetes administration and deployment",
            "skills": ["kubernetes", "docker", "devops", "linux"],
            "provider": "CNCF",
            "difficulty": "advanced",
            "url": "https://www.cncf.io/certification/cka/"
        },
        {
            "id": "cert_005",
            "title": "Microsoft Azure Data Scientist Associate",
            "description": "Demonstrate ability to apply data science and ML on Azure",
            "skills": ["azure", "machine learning", "data science", "python"],
            "provider": "Microsoft",
            "difficulty": "intermediate",
            "url": "https://docs.microsoft.com/en-us/learn/certifications/azure-data-scientist/"
        }
    ]
    
    COURSES_DB = [
        {
            "id": "course_001",
            "title": "Advanced Python Programming",
            "description": "Master advanced Python concepts including decorators, generators, async/await",
            "skills": ["python", "programming"],
            "provider": "Coursera",
            "difficulty": "intermediate",
            "estimated_time": "4 weeks"
        },
        {
            "id": "course_002",
            "title": "Deep Learning Specialization",
            "description": "Comprehensive deep learning course covering CNNs, RNNs, and transformers",
            "skills": ["deep learning", "machine learning", "neural networks", "python"],
            "provider": "Coursera",
            "difficulty": "advanced",
            "estimated_time": "12 weeks"
        },
        {
            "id": "course_003",
            "title": "Full Stack Web Development with React",
            "description": "Build complete web applications with React, Node.js, and MongoDB",
            "skills": ["react", "javascript", "node.js", "mongodb"],
            "provider": "Udemy",
            "difficulty": "intermediate",
            "estimated_time": "8 weeks"
        }
    ]
    
    def __init__(
        self,
        embedding_engine: Optional[SemanticEmbeddingEngine] = None,
        ner_extractor: Optional[SkillNERExtractor] = None
    ):
        """
        Initialize ML recommender.
        
        Args:
            embedding_engine: Semantic embedding engine
            ner_extractor: NER skill extractor
        """
        self.embedding_engine = embedding_engine or get_embedding_engine()
        self.ner_extractor = ner_extractor or get_ner_extractor()
        
        # Pre-compute embeddings for recommendation items
        self._precompute_embeddings()
        
        logger.info("✅ MLRecommender initialized")
    
    def _precompute_embeddings(self):
        """Pre-compute embeddings for all recommendation items."""
        logger.info("Pre-computing embeddings for recommendation database...")
        
        # Projects
        self.project_embeddings = []
        for project in self.PROJECTS_DB:
            text = f"{project['title']} {project['description']} {' '.join(project['skills'])}"
            emb = self.embedding_engine.encode_single(text)
            self.project_embeddings.append(emb)
        
        # Certifications
        self.cert_embeddings = []
        for cert in self.CERTIFICATIONS_DB:
            text = f"{cert['title']} {cert['description']} {' '.join(cert['skills'])}"
            emb = self.embedding_engine.encode_single(text)
            self.cert_embeddings.append(emb)
        
        # Courses
        self.course_embeddings = []
        for course in self.COURSES_DB:
            text = f"{course['title']} {course['description']} {' '.join(course['skills'])}"
            emb = self.embedding_engine.encode_single(text)
            self.course_embeddings.append(emb)
        
        logger.info("✅ Embeddings pre-computed")
    
    def recommend_for_cv(
        self,
        cv_data: Dict[str, Any],
        target_role: Optional[str] = None,
        max_recommendations: int = 10
    ) -> RecommendationSet:
        """
        Generate personalized recommendations for a CV.
        
        Args:
            cv_data: CV data
            target_role: Optional target role for recommendations
            max_recommendations: Max items per category
            
        Returns:
            RecommendationSet with all recommendations
        """
        # Extract current skills
        ner_result = self.ner_extractor.extract_from_cv(cv_data)
        current_skills = set(s.skill_name for s in ner_result.skills)
        
        # Generate CV embedding
        cv_embedding = self.embedding_engine.encode_cv(cv_data)
        
        # If target role provided, incorporate it
        if target_role:
            target_text = f"{cv_data.get('summary', '')} aiming for {target_role}"
            cv_embedding = self.embedding_engine.encode_single(target_text)
        
        # Recommend projects
        project_recs = self._recommend_projects(
            cv_embedding,
            current_skills,
            max_recommendations
        )
        
        # Recommend certifications
        cert_recs = self._recommend_certifications(
            cv_embedding,
            current_skills,
            max_recommendations
        )
        
        # Recommend courses
        course_recs = self._recommend_courses(
            cv_embedding,
            current_skills,
            max_recommendations
        )
        
        # Identify skill gaps
        skills_to_learn = self.ner_extractor.suggest_related_skills(
            list(current_skills),
            top_n=10
        )
        
        # Generate learning path
        learning_path = self._generate_learning_path(
            current_skills,
            skills_to_learn,
            target_role
        )
        
        return RecommendationSet(
            projects=project_recs,
            certifications=cert_recs,
            courses=course_recs,
            skills_to_learn=skills_to_learn,
            learning_path=learning_path,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _recommend_projects(
        self,
        cv_embedding,
        current_skills: set,
        max_n: int
    ) -> List[Recommendation]:
        """Recommend projects using semantic similarity."""
        similarities = self.embedding_engine.batch_similarity(
            cv_embedding,
            self.project_embeddings
        )
        
        # Sort by similarity
        ranked = sorted(
            zip(self.PROJECTS_DB, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = []
        for project, similarity in ranked[:max_n]:
            # Check skill overlap
            project_skills = set(project['skills'])
            new_skills = project_skills - current_skills
            
            # Higher score if it teaches new skills
            relevance = similarity * 100
            if new_skills:
                relevance += len(new_skills) * 2
            
            relevance = min(relevance, 100.0)
            
            recommendations.append(Recommendation(
                id=project['id'],
                type='project',
                title=project['title'],
                description=project['description'],
                relevance_score=relevance,
                skills_gained=list(new_skills),
                estimated_time=project['estimated_time'],
                difficulty=project['difficulty']
            ))
        
        return recommendations
    
    def _recommend_certifications(
        self,
        cv_embedding,
        current_skills: set,
        max_n: int
    ) -> List[Recommendation]:
        """Recommend certifications using semantic similarity."""
        similarities = self.embedding_engine.batch_similarity(
            cv_embedding,
            self.cert_embeddings
        )
        
        ranked = sorted(
            zip(self.CERTIFICATIONS_DB, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = []
        for cert, similarity in ranked[:max_n]:
            cert_skills = set(cert['skills'])
            skill_overlap = len(current_skills.intersection(cert_skills))
            
            # Prefer certs where user has most required skills
            relevance = similarity * 70 + (skill_overlap / len(cert_skills)) * 30
            
            recommendations.append(Recommendation(
                id=cert['id'],
                type='certification',
                title=cert['title'],
                description=cert['description'],
                relevance_score=relevance,
                skills_gained=cert['skills'],
                estimated_time="Varies",
                difficulty=cert['difficulty'],
                url=cert.get('url'),
                provider=cert.get('provider')
            ))
        
        return recommendations
    
    def _recommend_courses(
        self,
        cv_embedding,
        current_skills: set,
        max_n: int
    ) -> List[Recommendation]:
        """Recommend courses using semantic similarity."""
        similarities = self.embedding_engine.batch_similarity(
            cv_embedding,
            self.course_embeddings
        )
        
        ranked = sorted(
            zip(self.COURSES_DB, similarities),
            key=lambda x: x[1],
            reverse=True
        )
        
        recommendations = []
        for course, similarity in ranked[:max_n]:
            course_skills = set(course['skills'])
            new_skills = course_skills - current_skills
            
            relevance = similarity * 100
            if new_skills:
                relevance += len(new_skills) * 3
            
            relevance = min(relevance, 100.0)
            
            recommendations.append(Recommendation(
                id=course['id'],
                type='course',
                title=course['title'],
                description=course['description'],
                relevance_score=relevance,
                skills_gained=list(new_skills),
                estimated_time=course['estimated_time'],
                difficulty=course['difficulty'],
                provider=course.get('provider')
            ))
        
        return recommendations
    
    def _generate_learning_path(
        self,
        current_skills: set,
        target_skills: List[str],
        target_role: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate a structured learning path.
        
        Args:
            current_skills: Skills user has
            target_skills: Skills to learn
            target_role: Target job role
            
        Returns:
            Ordered learning path
        """
        path = []
        
        # Categorize target skills
        categorized = self.ner_extractor.categorize_skills(target_skills)
        
        # Phase 1: Foundational skills
        foundational = []
        for skill in target_skills[:3]:
            if skill not in current_skills:
                foundational.append(skill)
        
        if foundational:
            path.append({
                "phase": 1,
                "title": "Build Foundation",
                "duration": "2-4 weeks",
                "skills": foundational,
                "activities": ["Take introductory courses", "Complete tutorials", "Practice with small projects"]
            })
        
        # Phase 2: Intermediate development
        intermediate = target_skills[3:6]
        if intermediate:
            path.append({
                "phase": 2,
                "title": "Develop Core Skills",
                "duration": "4-8 weeks",
                "skills": intermediate,
                "activities": ["Work on real projects", "Contribute to open source", "Build portfolio pieces"]
            })
        
        # Phase 3: Advanced/specialization
        advanced = target_skills[6:10]
        if advanced:
            path.append({
                "phase": 3,
                "title": "Specialize & Master",
                "duration": "8-12 weeks",
                "skills": advanced,
                "activities": ["Pursue certifications", "Lead projects", "Mentor others"]
            })
        
        return path


# Global singleton
_global_recommender: Optional[MLRecommender] = None


def get_recommender() -> MLRecommender:
    """
    Get or create global recommender (singleton).
    
    Returns:
        MLRecommender instance
    """
    global _global_recommender
    
    if _global_recommender is None:
        _global_recommender = MLRecommender()
    
    return _global_recommender
