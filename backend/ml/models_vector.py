"""
SkillSync Database Models with pgvector Support
================================================
SQLAlchemy models with vector embeddings for semantic search.
"""

from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime
import uuid

Base = declarative_base()


class CVAnalysis(Base):
    """CV analysis with semantic embeddings"""
    __tablename__ = "cv_analyses"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    
    # CV content
    raw_text = Column(Text, nullable=False)
    summary = Column(Text)
    
    # Extracted information
    skills = Column(JSON, default=list)  # List of extracted skills
    experience_years = Column(Integer, default=0)
    education_level = Column(String)
    
    # Semantic embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = Column(Vector(384))
    
    # Metadata
    filename = Column(String)
    file_size = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="cv_analyses")
    job_matches = relationship("JobMatch", back_populates="cv_analysis")


class Job(Base):
    """Job postings with semantic embeddings"""
    __tablename__ = "jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Job details
    title = Column(String, nullable=False, index=True)
    company = Column(String)
    description = Column(Text)
    location = Column(String)
    
    # Requirements
    skills_required = Column(JSON, default=list)
    min_experience_years = Column(Integer, default=0)
    max_experience_years = Column(Integer)
    
    # Salary
    salary_min = Column(Float)
    salary_max = Column(Float)
    salary_currency = Column(String, default="USD")
    
    # Semantic embedding (384 dimensions)
    embedding = Column(Vector(384))
    
    # Metadata
    source = Column(String)  # API source (jsearch, adzuna, etc.)
    external_id = Column(String, index=True)
    url = Column(String)
    posted_at = Column(DateTime)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    job_matches = relationship("JobMatch", back_populates="job")


class JobMatch(Base):
    """CV-Job matching results with ML scores"""
    __tablename__ = "job_matches"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    cv_analysis_id = Column(String, ForeignKey("cv_analyses.id"), nullable=False)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False)
    
    # ML scores
    overall_score = Column(Float, nullable=False)  # 0-100
    semantic_similarity = Column(Float)  # 0-1
    skill_overlap = Column(Float)  # 0-1
    experience_match = Column(Float)  # 0-1
    
    # Matched/missing skills
    matched_skills = Column(JSON, default=list)
    missing_skills = Column(JSON, default=list)
    
    # Explanation
    explanation_text = Column(Text)
    factors = Column(JSON, default=dict)
    
    # Metadata
    confidence = Column(Float, default=0.85)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    cv_analysis = relationship("CVAnalysis", back_populates="job_matches")
    job = relationship("Job", back_populates="job_matches")
    xai_explanation = relationship("XAIExplanation", back_populates="job_match", uselist=False)


class XAIExplanation(Base):
    """XAI explainability data for job matches"""
    __tablename__ = "xai_explanations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_match_id = Column(String, ForeignKey("job_matches.id"), nullable=False, unique=True)
    
    # Explanation data
    prediction_score = Column(Float)
    base_score = Column(Float, default=50.0)
    
    top_positive_factors = Column(JSON, default=list)
    top_negative_factors = Column(JSON, default=list)
    feature_importance = Column(JSON, default=dict)
    
    explanation_text = Column(Text)
    confidence = Column(Float)
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    job_match = relationship("JobMatch", back_populates="xai_explanation")


class Skill(Base):
    """Skills taxonomy with embeddings"""
    __tablename__ = "skills"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    name = Column(String, nullable=False, unique=True, index=True)
    category = Column(String, index=True)  # programming, framework, soft_skill, etc.
    description = Column(Text)
    
    # Semantic embedding
    embedding = Column(Vector(384))
    
    # Metadata
    esco_code = Column(String)  # ESCO ontology code
    onet_code = Column(String)  # O*NET code
    popularity_score = Column(Float, default=0.0)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class Recommendation(Base):
    """ML-based recommendations (projects, courses, certs)"""
    __tablename__ = "recommendations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cv_analysis_id = Column(String, ForeignKey("cv_analyses.id"), nullable=False)
    
    # Recommendation details
    type = Column(String, nullable=False)  # project, certification, course, skill
    title = Column(String, nullable=False)
    description = Column(Text)
    
    # Scoring
    relevance_score = Column(Float, nullable=False)  # 0-100
    
    # Skills
    skills_gained = Column(JSON, default=list)
    skills_required = Column(JSON, default=list)
    
    # Metadata
    difficulty = Column(String)  # beginner, intermediate, advanced
    estimated_time = Column(String)
    url = Column(String)
    provider = Column(String)
    
    created_at = Column(DateTime, server_default=func.now())


class ExperienceTranslation(Base):
    """NLG translations of CV experiences"""
    __tablename__ = "experience_translations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    cv_analysis_id = Column(String, ForeignKey("cv_analyses.id"), nullable=False)
    
    # Original and translated text
    original_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    
    # Translation parameters
    style = Column(String, nullable=False)  # professional, technical, creative
    model_used = Column(String)
    confidence = Column(Float)
    
    created_at = Column(DateTime, server_default=func.now())


class User(Base):
    """User accounts"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    
    # Relationships
    cv_analyses = relationship("CVAnalysis", back_populates="user")


# Indexes for vector similarity search
# These will be created by Alembic migrations:
# CREATE INDEX ON cv_analyses USING ivfflat (embedding vector_cosine_ops);
# CREATE INDEX ON jobs USING ivfflat (embedding vector_cosine_ops);
# CREATE INDEX ON skills USING ivfflat (embedding vector_cosine_ops);
