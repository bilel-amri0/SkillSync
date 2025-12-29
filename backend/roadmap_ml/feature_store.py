"""Feature store builder for learning roadmap models."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "skillsync_enhanced.db"
DEFAULT_DATABASE_URL = os.getenv("SKILLSYNC_DATABASE_URL") or f"sqlite:///{DEFAULT_DATABASE_PATH}"

logger = logging.getLogger(__name__)

USER_PROFILE_SCHEMA = {
    'user_id': 'object',
    'target_roles': 'object',
    'weekly_hours_available': 'float64',
    'skill_gaps': 'object',
    'experience_years': 'float64',
    'seniority': 'object',
    'industry': 'object'
}

ROADMAP_HISTORY_SCHEMA = {
    'user_id': 'object',
    'skills_sequence': 'object',
    'planned_duration_hours': 'float64',
    'actual_duration_hours': 'float64',
    'quiz_scores': 'object',
    'satisfaction_score': 'float64'
}

RESOURCE_METADATA_SCHEMA = {
    'resource_id': 'object',
    'skill': 'object',
    'type': 'object',
    'duration_hours': 'float64',
    'cost': 'float64',
    'past_success_rate': 'float64',
    'difficulty': 'float64',
    'provider': 'object',
    'title': 'object',
    'url': 'object'
}


class RoadmapFeatureStore:
    """Utility class that loads historical datasets and produces ML-ready frames."""

    def __init__(self, data_dir: Optional[Path] = None, database_url: Optional[str] = None) -> None:
        self.data_dir = Path(data_dir or DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.database_url = database_url or DEFAULT_DATABASE_URL
        self.engine = self._init_engine(self.database_url)
        self._user_profiles: Optional[pd.DataFrame] = None
        self._roadmap_history: Optional[pd.DataFrame] = None
        self._resource_metadata: Optional[pd.DataFrame] = None
        self._resource_skill_aggregates: Optional[pd.DataFrame] = None

    def _init_engine(self, database_url: Optional[str]) -> Optional[Engine]:
        if not database_url:
            return None
        connect_args: Dict[str, Any] = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        try:
            return create_engine(database_url, connect_args=connect_args, future=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[feature-store] Failed to initialize engine (%s): %s", database_url, exc)
            return None

    def _read_query(self, query: str) -> pd.DataFrame:
        if not self.engine:
            return pd.DataFrame()
        try:
            with self.engine.connect() as connection:
                return pd.read_sql_query(text(query), connection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[feature-store] Query failed: %s", exc)
            return pd.DataFrame()

    @property
    def user_profiles(self) -> pd.DataFrame:
        if self._user_profiles is None:
            self._user_profiles = self._load_user_profiles()
        return self._user_profiles

    @property
    def roadmap_history(self) -> pd.DataFrame:
        if self._roadmap_history is None:
            self._roadmap_history = self._load_roadmap_history()
        return self._roadmap_history

    @property
    def resource_metadata(self) -> pd.DataFrame:
        if self._resource_metadata is None:
            self._resource_metadata = self._load_resource_metadata()
        return self._resource_metadata

    @property
    def resource_skill_aggregates(self) -> pd.DataFrame:
        if self._resource_skill_aggregates is None:
            metadata = self.resource_metadata.copy()
            metadata['duration_hours'] = metadata['duration_hours'].astype(float)
            metadata['cost'] = metadata['cost'].astype(float)
            metadata['past_success_rate'] = metadata['past_success_rate'].astype(float)
            metadata['difficulty'] = metadata['difficulty'].astype(float)
            grouped = metadata.groupby('skill').agg(
                avg_resource_duration=('duration_hours', 'mean'),
                avg_resource_cost=('cost', 'mean'),
                avg_resource_success=('past_success_rate', 'mean'),
                avg_resource_difficulty=('difficulty', 'mean'),
                resources_available=('resource_id', 'count')
            )
            grouped = grouped.reset_index()
            self._resource_skill_aggregates = grouped
        return self._resource_skill_aggregates

    def build_skill_examples(self) -> pd.DataFrame:
        """Explode historical roadmaps to skill-level rows for supervised training."""
        history = self.roadmap_history.copy()
        if history.empty:
            raise ValueError("Roadmap history dataset is empty")

        user_lookup = self.user_profiles.set_index('user_id').to_dict(orient='index')
        resource_lookup = self.resource_skill_aggregates.set_index('skill').to_dict(orient='index')

        records: List[Dict[str, Any]] = []
        for _, row in history.iterrows():
            skills_sequence: List[str] = row.get('skills_sequence') or []
            if not skills_sequence:
                continue
            user_ctx = user_lookup.get(row.get('user_id'), {})
            planned = float(row.get('planned_duration_hours', 0.0) or 0.0)
            actual = float(row.get('actual_duration_hours', planned) or planned)
            per_skill_planned = planned / max(len(skills_sequence), 1)
            per_skill_actual = actual / max(len(skills_sequence), 1)
            quiz_scores = self._parse_quiz_scores(row.get('quiz_scores'))
            satisfaction = float(row.get('satisfaction_score') or 3)

            for order, skill in enumerate(skills_sequence):
                skill = skill.strip()
                if not skill:
                    continue
                aggregates = resource_lookup.get(skill, {})
                quiz_score = float(quiz_scores.get(skill, satisfaction * 20))
                records.append({
                    'user_id': row.get('user_id'),
                    'skill': skill,
                    'skill_order': order,
                    'skills_in_phase': len(skills_sequence),
                    'planned_duration_hours': per_skill_planned,
                    'actual_duration_hours': per_skill_actual,
                    'weekly_hours_available': user_ctx.get('weekly_hours_available', 8),
                    'experience_years': user_ctx.get('experience_years', 4),
                    'seniority': user_ctx.get('seniority', 'Mid-level'),
                    'industry': user_ctx.get('industry', 'Tech'),
                    'target_role': user_ctx.get('target_roles', 'Generalist'),
                    'avg_resource_duration': aggregates.get('avg_resource_duration', np.nan),
                    'avg_resource_cost': aggregates.get('avg_resource_cost', np.nan),
                    'avg_resource_success': aggregates.get('avg_resource_success', np.nan),
                    'avg_resource_difficulty': aggregates.get('avg_resource_difficulty', np.nan),
                    'resource_count': aggregates.get('resources_available', 0),
                    'success_target': quiz_score / 100.0,
                    'duration_target': per_skill_actual,
                    'satisfaction_score': satisfaction,
                })

        df = pd.DataFrame.from_records(records)
        if df.empty:
            raise ValueError("No skill-level records could be constructed from roadmap history")
        df = df.fillna(df.mean(numeric_only=True))
        for column in ['seniority', 'industry', 'target_role', 'skill']:
            if column in df.columns:
                df[column] = df[column].fillna('Unknown')
        return df

    def build_resource_training_frame(self) -> pd.DataFrame:
        """Return a dataset for ranking learning resources."""
        metadata = self.resource_metadata.copy()
        if metadata.empty:
            raise ValueError("Resource metadata dataset is empty")
        metadata['effectiveness_score'] = metadata['past_success_rate'] / metadata['duration_hours'].clip(lower=0.5)
        metadata['is_budget'] = (metadata['cost'] <= metadata['cost'].median()).astype(int)
        metadata['difficulty_bucket'] = metadata['difficulty'].clip(1, 5)
        return metadata

    def default_context(self) -> Dict[str, Any]:
        profiles = self.user_profiles
        defaults = {
            'experience_years': 4.0,
            'weekly_hours_available': 8.0,
            'seniority': 'Mid-level',
            'industry': 'Tech',
            'target_role': 'Engineer'
        }

        if profiles.empty:
            return defaults

        def _safe_median(series: pd.Series, fallback: float) -> float:
            value = series.median(skipna=True)
            return float(value) if pd.notna(value) else fallback

        def _safe_mode(series: pd.Series, fallback: str) -> str:
            mode = series.mode(dropna=True)
            return mode.iloc[0] if not mode.empty else fallback

        return {
            'experience_years': _safe_median(profiles['experience_years'], defaults['experience_years']),
            'weekly_hours_available': _safe_median(profiles['weekly_hours_available'], defaults['weekly_hours_available']),
            'seniority': _safe_mode(profiles['seniority'], defaults['seniority']),
            'industry': _safe_mode(profiles['industry'], defaults['industry']),
            'target_role': _safe_mode(profiles['target_roles'], defaults['target_role'])
        }

    def _load_user_profiles(self) -> pd.DataFrame:
        if self.engine:
            query = """
                SELECT
                    id AS cv_id,
                    user_id,
                    skills,
                    gap_analysis,
                    experience_years,
                    confidence_score
                FROM cv_analyses
                ORDER BY created_at DESC
            """
            raw = self._read_query(query)
            if raw.empty:
                return self._empty_frame(USER_PROFILE_SCHEMA)
            records: List[Dict[str, Any]] = []
            for _, row in raw.iterrows():
                gap_analysis = self._ensure_dict(row.get('gap_analysis'))
                records.append({
                    'user_id': row.get('user_id') or row.get('cv_id'),
                    'target_roles': gap_analysis.get('target_role') or 'Generalist',
                    'weekly_hours_available': gap_analysis.get('weekly_hours_available') or 8.0,
                    'skill_gaps': gap_analysis.get('missing_skills') or gap_analysis.get('skill_gaps') or [],
                    'experience_years': row.get('experience_years') or 4.0,
                    'seniority': gap_analysis.get('seniority') or 'Mid-level',
                    'industry': gap_analysis.get('industry') or 'Tech'
                })
            df = pd.DataFrame.from_records(records)
            return self._ensure_schema(df, USER_PROFILE_SCHEMA)

        path = self.data_dir / 'user_profiles.json'
        if not path.exists():
            return self._empty_frame(USER_PROFILE_SCHEMA)
        raw = pd.read_json(path)
        if raw.empty:
            return self._empty_frame(USER_PROFILE_SCHEMA)
        records: List[Dict[str, Any]] = []
        for _, row in raw.iterrows():
            cv_blob = row.get('CV')
            cv_data: Dict[str, Any] = {}
            if isinstance(cv_blob, str) and cv_blob.strip():
                try:
                    cv_data = json.loads(cv_blob)
                except json.JSONDecodeError:
                    cv_data = {}
            records.append({
                'user_id': row.get('user_id'),
                'target_roles': row.get('target_roles') or cv_data.get('target_role') or 'Generalist',
                'weekly_hours_available': row.get('weekly_hours_available') or cv_data.get('weekly_hours_available') or 8,
                'skill_gaps': row.get('skill_gaps') or cv_data.get('skill_gaps') or [],
                'experience_years': cv_data.get('experience_years') or row.get('experience_years') or 4,
                'seniority': cv_data.get('seniority') or row.get('seniority') or 'Mid-level',
                'industry': cv_data.get('industry') or row.get('industry') or 'Tech'
            })
        df = pd.DataFrame.from_records(records)
        return self._ensure_schema(df, USER_PROFILE_SCHEMA)

    def _load_roadmap_history(self) -> pd.DataFrame:
        if self.engine:
            query = """
                SELECT
                    id,
                    user_id,
                    skills_sequence,
                    planned_duration_hours,
                    actual_duration_hours,
                    quiz_scores,
                    satisfaction_score
                FROM learning_roadmap_runs
                ORDER BY created_at DESC
            """
            df = self._read_query(query)
            if df.empty:
                return self._empty_frame(ROADMAP_HISTORY_SCHEMA)
            df['skills_sequence'] = df['skills_sequence'].apply(self._ensure_list)
            df['quiz_scores'] = df['quiz_scores'].apply(self._ensure_dict)
            df['planned_duration_hours'] = df['planned_duration_hours'].fillna(df['actual_duration_hours'])
            df['actual_duration_hours'] = df['actual_duration_hours'].fillna(df['planned_duration_hours'])
            df['satisfaction_score'] = df['satisfaction_score'].fillna(3.0)
            return self._ensure_schema(df, ROADMAP_HISTORY_SCHEMA)

        path = self.data_dir / 'roadmap_history.json'
        if not path.exists():
            return self._empty_frame(ROADMAP_HISTORY_SCHEMA)
        df = pd.read_json(path)
        if df.empty:
            return self._empty_frame(ROADMAP_HISTORY_SCHEMA)
        return self._ensure_schema(df, ROADMAP_HISTORY_SCHEMA)

    def _load_resource_metadata(self) -> pd.DataFrame:
        if self.engine:
            query = """
                SELECT
                    id,
                    skill,
                    tier,
                    provider,
                    resource_type,
                    duration_hours,
                    cost,
                    success_rate,
                    difficulty,
                    is_free,
                    resource_payload
                FROM learning_resource_events
                ORDER BY created_at DESC
            """
            raw = self._read_query(query)
            if raw.empty:
                return self._empty_frame(RESOURCE_METADATA_SCHEMA)
            records: List[Dict[str, Any]] = []
            for _, row in raw.iterrows():
                payload = self._ensure_dict(row.get('resource_payload'))
                records.append({
                    'resource_id': row.get('id'),
                    'skill': row.get('skill'),
                    'type': row.get('resource_type') or row.get('tier') or 'General',
                    'duration_hours': row.get('duration_hours') or payload.get('estimated_time_hours') or 6.0,
                    'cost': row.get('cost') or 0.0,
                    'past_success_rate': (row.get('success_rate') or 0.7) * 100.0,
                    'difficulty': row.get('difficulty') or (3.0 if (row.get('tier') or '').lower() == 'core' else 4.0),
                    'provider': row.get('provider') or payload.get('provider') or 'SkillSync',
                    'title': payload.get('title') or f"{row.get('skill')} {row.get('resource_type') or 'Track'}",
                    'url': payload.get('url') or payload.get('link')
                })
            df = pd.DataFrame.from_records(records)
            return self._ensure_schema(df, RESOURCE_METADATA_SCHEMA)

        path = self.data_dir / 'resource_metadata.json'
        if not path.exists():
            return self._empty_frame(RESOURCE_METADATA_SCHEMA)
        df = pd.read_json(path)
        if df.empty:
            return self._empty_frame(RESOURCE_METADATA_SCHEMA)
        return self._ensure_schema(df, RESOURCE_METADATA_SCHEMA)

    @staticmethod
    def _parse_quiz_scores(raw: Any) -> Dict[str, float]:
        if isinstance(raw, dict):
            sanitized: Dict[str, float] = {}
            for k, v in raw.items():
                if v is None:
                    continue
                try:
                    sanitized[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return sanitized
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    sanitized = {}
                    for k, v in parsed.items():
                        if v is None:
                            continue
                        try:
                            sanitized[str(k)] = float(v)
                        except (TypeError, ValueError):
                            continue
                    return sanitized
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _empty_frame(schema: Dict[str, str]) -> pd.DataFrame:
        return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})

    @staticmethod
    def _ensure_schema(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        if df.empty:
            return RoadmapFeatureStore._empty_frame(schema)
        for column in schema.keys():
            if column not in df.columns:
                df[column] = pd.Series([np.nan] * len(df), dtype=schema[column])
        return df

    @staticmethod
    def _ensure_list(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return []
        return []

    @staticmethod
    def _ensure_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {}
        return {}
