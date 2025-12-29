"""
Alembic Migration: Add pgvector extension and create tables with embeddings
============================================================================
Revision ID: 001_initial_vector_tables
Create Date: 2025-11-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import pgvector.sqlalchemy

# revision identifiers
revision = '001_initial_vector_tables'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Create tables with vector embeddings"""
    
    # 1. Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # 2. Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=False)
    
    # 3. Create cv_analyses table with embeddings
    op.create_table(
        'cv_analyses',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('raw_text', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('skills', sa.JSON(), nullable=True),
        sa.Column('experience_years', sa.Integer(), nullable=True),
        sa.Column('education_level', sa.String(), nullable=True),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(384), nullable=True),
        sa.Column('filename', sa.String(), nullable=True),
        sa.Column('file_size', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 4. Create jobs table with embeddings
    op.create_table(
        'jobs',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('company', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('location', sa.String(), nullable=True),
        sa.Column('skills_required', sa.JSON(), nullable=True),
        sa.Column('min_experience_years', sa.Integer(), nullable=True),
        sa.Column('max_experience_years', sa.Integer(), nullable=True),
        sa.Column('salary_min', sa.Float(), nullable=True),
        sa.Column('salary_max', sa.Float(), nullable=True),
        sa.Column('salary_currency', sa.String(), nullable=True),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(384), nullable=True),
        sa.Column('source', sa.String(), nullable=True),
        sa.Column('external_id', sa.String(), nullable=True),
        sa.Column('url', sa.String(), nullable=True),
        sa.Column('posted_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_jobs_title'), 'jobs', ['title'], unique=False)
    op.create_index(op.f('ix_jobs_external_id'), 'jobs', ['external_id'], unique=False)
    
    # 5. Create job_matches table
    op.create_table(
        'job_matches',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('cv_analysis_id', sa.String(), nullable=False),
        sa.Column('job_id', sa.String(), nullable=False),
        sa.Column('overall_score', sa.Float(), nullable=False),
        sa.Column('semantic_similarity', sa.Float(), nullable=True),
        sa.Column('skill_overlap', sa.Float(), nullable=True),
        sa.Column('experience_match', sa.Float(), nullable=True),
        sa.Column('matched_skills', sa.JSON(), nullable=True),
        sa.Column('missing_skills', sa.JSON(), nullable=True),
        sa.Column('explanation_text', sa.Text(), nullable=True),
        sa.Column('factors', sa.JSON(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['cv_analysis_id'], ['cv_analyses.id'], ),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 6. Create xai_explanations table
    op.create_table(
        'xai_explanations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('job_match_id', sa.String(), nullable=False),
        sa.Column('prediction_score', sa.Float(), nullable=True),
        sa.Column('base_score', sa.Float(), nullable=True),
        sa.Column('top_positive_factors', sa.JSON(), nullable=True),
        sa.Column('top_negative_factors', sa.JSON(), nullable=True),
        sa.Column('feature_importance', sa.JSON(), nullable=True),
        sa.Column('explanation_text', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['job_match_id'], ['job_matches.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_match_id')
    )
    
    # 7. Create skills table with embeddings
    op.create_table(
        'skills',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('category', sa.String(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('embedding', pgvector.sqlalchemy.Vector(384), nullable=True),
        sa.Column('esco_code', sa.String(), nullable=True),
        sa.Column('onet_code', sa.String(), nullable=True),
        sa.Column('popularity_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_skills_name'), 'skills', ['name'], unique=False)
    op.create_index(op.f('ix_skills_category'), 'skills', ['category'], unique=False)
    
    # 8. Create recommendations table
    op.create_table(
        'recommendations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('cv_analysis_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('relevance_score', sa.Float(), nullable=False),
        sa.Column('skills_gained', sa.JSON(), nullable=True),
        sa.Column('skills_required', sa.JSON(), nullable=True),
        sa.Column('difficulty', sa.String(), nullable=True),
        sa.Column('estimated_time', sa.String(), nullable=True),
        sa.Column('url', sa.String(), nullable=True),
        sa.Column('provider', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['cv_analysis_id'], ['cv_analyses.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 9. Create experience_translations table
    op.create_table(
        'experience_translations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('cv_analysis_id', sa.String(), nullable=False),
        sa.Column('original_text', sa.Text(), nullable=False),
        sa.Column('translated_text', sa.Text(), nullable=False),
        sa.Column('style', sa.String(), nullable=False),
        sa.Column('model_used', sa.String(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['cv_analysis_id'], ['cv_analyses.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # 10. Create vector similarity indexes (IVFFLAT for fast approximate search)
    # For small datasets, can use exact search. For large (>100k), use IVFFLAT.
    op.execute('CREATE INDEX ON cv_analyses USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')
    op.execute('CREATE INDEX ON jobs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')
    op.execute('CREATE INDEX ON skills USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)')


def downgrade():
    """Drop all tables and extension"""
    
    op.drop_table('experience_translations')
    op.drop_table('recommendations')
    op.drop_table('skills')
    op.drop_table('xai_explanations')
    op.drop_table('job_matches')
    op.drop_table('jobs')
    op.drop_table('cv_analyses')
    op.drop_table('users')
    
    op.execute('DROP EXTENSION IF EXISTS vector')
