"""Database configuration and CRUD operations for SkillSync"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import logging
import os

from models import Base, User, CVAnalysis, IdentifiedSkill, Recommendation, AnalysisSession

# Import interview tables so Base.metadata is aware of them before create_all
from skillsync.interviews import models as interview_models  # noqa: F401

# Configuration - Support PostgreSQL or SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./skillsync.db")
logger = logging.getLogger(__name__)

# Determine if using PostgreSQL or SQLite
is_postgres = DATABASE_URL.startswith("postgresql://")

# Database setup
if is_postgres:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False  # Set True for SQL debugging
    )
else:
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def get_db() -> Session:
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    """Direct database session for scripts"""
    return SessionLocal()

# =============================================================================
# CRUD Operations for Current MVP
# =============================================================================

class CVAnalysisService:
    """Service class for CV Analysis operations"""
    
    @staticmethod
    def create_analysis(
        db: Session,
        filename: str,
        original_text: str,
        analysis_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> CVAnalysis:
        """Create new CV analysis record"""
        try:
            # Store anonymous analyses without forcing fake users
            if not user_id:
                user_id = None
            
            analysis = CVAnalysis(
                user_id=user_id,
                filename=filename,
                original_text=original_text,
                analysis_data=analysis_data,
                processing_version="v1.0"
            )
            
            db.add(analysis)
            db.commit()
            db.refresh(analysis)
            
            # Create session record
            session = AnalysisSession(
                analysis_id=analysis.id,
                model_version="simple_v1",
                success=True
            )
            db.add(session)
            db.commit()
            
            logger.info(f"Created analysis {analysis.id} for file {filename}")
            return analysis
            
        except SQLAlchemyError as e:
            db.rollback()
            logger.error(f"Failed to create analysis: {e}")
            raise
    
    @staticmethod
    def get_analysis(db: Session, analysis_id: str) -> Optional[CVAnalysis]:
        """Get analysis by ID"""
        return db.query(CVAnalysis).filter(CVAnalysis.id == analysis_id).first()
    
    @staticmethod
    def get_latest_analysis(db: Session, user_id: Optional[str] = None) -> Optional[CVAnalysis]:
        """Get most recent analysis for user (or any user if None)"""
        query = db.query(CVAnalysis)
        if user_id:
            query = query.filter(CVAnalysis.user_id == user_id)
        return query.order_by(CVAnalysis.created_at.desc()).first()
    
    @staticmethod
    def get_user_analyses(db: Session, user_id: str, limit: int = 10) -> List[CVAnalysis]:
        """Get all analyses for a user"""
        return db.query(CVAnalysis)\
                 .filter(CVAnalysis.user_id == user_id)\
                 .order_by(CVAnalysis.created_at.desc())\
                 .limit(limit)\
                 .all()

    @staticmethod
    def get_recent_analyses(db: Session, limit: int = 10) -> List[CVAnalysis]:
        """Return most recent analyses regardless of user."""
        return (
            db.query(CVAnalysis)
            .order_by(CVAnalysis.created_at.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def list_all_analyses(db: Session, limit: int = 100, offset: int = 0) -> List[CVAnalysis]:
        """List analyses with pagination support."""
        return (
            db.query(CVAnalysis)
            .order_by(CVAnalysis.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

class UserService:
    """Service class for User operations"""
    
    @staticmethod
    def get_or_create_anonymous_user(db: Session) -> User:
        """Get or create anonymous user for MVP"""
        # Check for existing anonymous user
        user = db.query(User).filter(User.email.is_(None)).first()
        
        if not user:
            user = User(name="Anonymous User")
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"Created anonymous user {user.id}")
        
        return user
    
    @staticmethod
    def create_user(db: Session, email: str, name: str) -> User:
        """Create new user (future authentication)"""
        user = User(email=email, name=name)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

class SkillService:
    """Service class for Skills operations"""
    
    @staticmethod
    def create_skills_from_analysis(
        db: Session,
        analysis_id: str,
        skills_data: List[Dict[str, Any]]
    ) -> List[IdentifiedSkill]:
        """Create skill records from analysis data"""
        skills = []
        
        for skill_data in skills_data:
            skill = IdentifiedSkill(
                analysis_id=analysis_id,
                skill_name=skill_data.get('name', ''),
                category=skill_data.get('category', 'technical'),
                level=skill_data.get('level', 'intermediate'),
                confidence=skill_data.get('confidence', 0.8)
            )
            skills.append(skill)
        
        db.add_all(skills)
        db.commit()
        
        for skill in skills:
            db.refresh(skill)
        
        logger.info(f"Created {len(skills)} skills for analysis {analysis_id}")
        return skills
    
    @staticmethod
    def get_analysis_skills(db: Session, analysis_id: str) -> List[IdentifiedSkill]:
        """Get all skills for an analysis"""
        return db.query(IdentifiedSkill)\
                 .filter(IdentifiedSkill.analysis_id == analysis_id)\
                 .all()

class RecommendationService:
    """Service class for Recommendations operations"""
    
    @staticmethod
    def create_recommendations_from_analysis(
        db: Session,
        analysis_id: str,
        recommendations_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Recommendation]:
        """Create recommendation records from analysis data"""
        recommendations = []
        
        for category, items in recommendations_data.items():
            for i, item in enumerate(items):
                if isinstance(item, str):
                    # Handle simple string recommendations
                    rec = Recommendation(
                        analysis_id=analysis_id,
                        category=category,
                        title=item[:100],  # First 100 chars as title
                        description=item,
                        priority=i + 1
                    )
                else:
                    # Handle structured recommendations
                    rec = Recommendation(
                        analysis_id=analysis_id,
                        category=category,
                        title=item.get('title', item.get('name', 'Recommendation')),
                        description=item.get('description', str(item)),
                        priority=item.get('priority', i + 1)
                    )
                
                recommendations.append(rec)
        
        db.add_all(recommendations)
        db.commit()
        
        for rec in recommendations:
            db.refresh(rec)
        
        logger.info(f"Created {len(recommendations)} recommendations for analysis {analysis_id}")
        return recommendations
    
    @staticmethod
    def get_analysis_recommendations(db: Session, analysis_id: str) -> Dict[str, List[Dict]]:
        """Get recommendations for an analysis, formatted for frontend"""
        recommendations = db.query(Recommendation)\
                           .filter(Recommendation.analysis_id == analysis_id)\
                           .order_by(Recommendation.priority)\
                           .all()
        
        # Group by category
        result = {}
        for rec in recommendations:
            if rec.category not in result:
                result[rec.category] = []
            
            result[rec.category].append({
                'id': rec.id,
                'title': rec.title,
                'description': rec.description,
                'priority': rec.priority,
                'created_at': rec.created_at.isoformat()
            })
        
        return result

# =============================================================================
# Database Utilities
# =============================================================================

def test_connection() -> bool:
    """Test database connection"""
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False

def get_database_stats(db: Session) -> Dict[str, int]:
    """Get basic database statistics"""
    stats = {
        'users': db.query(User).count(),
        'analyses': db.query(CVAnalysis).count(),
        'skills': db.query(IdentifiedSkill).count(),
        'recommendations': db.query(Recommendation).count(),
        'sessions': db.query(AnalysisSession).count()
    }
    return stats

def cleanup_old_data(db: Session, days: int = 30) -> int:
    """Cleanup old anonymous data (future maintenance)"""
    # Implementation for future data retention policies
    # For now, just return 0
    return 0

# =============================================================================
# Migration Helpers (Option B Preparation)
# =============================================================================

def migrate_legacy_data(db: Session) -> bool:
    """Future: Migrate data from old format to new schema"""
    # Placeholder for future migrations
    return True

def add_ai_embeddings(db: Session, analysis_id: str, embeddings_data: Dict) -> bool:
    """Future: Add AI embeddings to existing analysis"""
    # Placeholder for Option B AI features
    return True
