"""Database models for SkillSync - Extensible Option B Architecture"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """User model with authentication support"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    name = Column(String, nullable=True)  # Backward compatibility
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    analyses = relationship("CVAnalysis", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    """Refresh token model for JWT token refresh"""
    __tablename__ = "refresh_tokens"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    token = Column(String, unique=True, index=True, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_revoked = Column(Boolean, default=False)

class CVAnalysis(Base):
    """Core CV Analysis model - Current MVP + Option B ready"""
    __tablename__ = "cv_analyses"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Optional for MVP
    
    # CV Content
    filename = Column(String, nullable=False)
    original_text = Column(Text, nullable=False)
    
    # Analysis Results (Current MVP format)
    analysis_data = Column(JSON, nullable=False)  # Store complete analysis
    
    # Option B Preparation
    cv_embedding = Column(Text, nullable=True)  # Future: CV vector embeddings
    ai_confidence = Column(Float, nullable=True)  # Future: AI confidence score
    processing_version = Column(String, default="v1.0")  # Track analysis version
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="analyses")
    skills = relationship("IdentifiedSkill", back_populates="analysis", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="analysis", cascade="all, delete-orphan")

class IdentifiedSkill(Base):
    """Individual skills identified in CV - Option B ready"""
    __tablename__ = "identified_skills"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("cv_analyses.id"), nullable=False)
    
    # Skill Data
    skill_name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # technical, soft, language, etc.
    level = Column(String, nullable=False)  # beginner, intermediate, advanced
    confidence = Column(Float, default=0.8)  # Current: static, Future: AI confidence
    
    # Option B Extensions
    skill_embedding = Column(Text, nullable=True)  # Future: skill vector
    market_demand = Column(Float, nullable=True)  # Future: job market data
    ai_explanation = Column(Text, nullable=True)  # Future: XAI explanations
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("CVAnalysis", back_populates="skills")

class Recommendation(Base):
    """Recommendations generated for each analysis"""
    __tablename__ = "recommendations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("cv_analyses.id"), nullable=False)
    
    # Recommendation Data
    category = Column(String, nullable=False)  # immediate_actions, skill_development, etc.
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(Integer, default=5)  # 1=highest, 10=lowest
    
    # Option B Extensions
    ai_reasoning = Column(Text, nullable=True)  # Future: XAI explanations
    success_probability = Column(Float, nullable=True)  # Future: AI predictions
    personalization_score = Column(Float, nullable=True)  # Future: personalization
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("CVAnalysis", back_populates="recommendations")

class AnalysisSession(Base):
    """Track analysis sessions - Option B Analytics"""
    __tablename__ = "analysis_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = Column(String, ForeignKey("cv_analyses.id"), nullable=False)
    
    # Session Metadata
    processing_time = Column(Float, nullable=True)  # seconds
    model_version = Column(String, default="simple_v1")
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    # Future Analytics
    user_feedback = Column(JSON, nullable=True)  # Future: user ratings
    a_b_test_group = Column(String, nullable=True)  # Future: A/B testing
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Option B Preparation Models (Empty for now, ready for future)
class JobMarketData(Base):
    """Future: Job market intelligence data"""
    __tablename__ = "job_market_data"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    skill_name = Column(String, nullable=False)
    demand_score = Column(Float, nullable=True)
    average_salary = Column(Float, nullable=True)
    growth_trend = Column(Float, nullable=True)
    data_source = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AIModel(Base):
    """Future: Track different AI models and their performance"""
    __tablename__ = "ai_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False)
    model_type = Column(String, nullable=False)  # embedding, classification, etc.
    version = Column(String, nullable=False)
    performance_metrics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
