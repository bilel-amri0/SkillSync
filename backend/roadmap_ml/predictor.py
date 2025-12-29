"""Inference helpers for the learning roadmap ML stack."""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from .feature_store import RoadmapFeatureStore

logger = logging.getLogger(__name__)

_DEFAULT_PREDICTOR: Optional["RoadmapPredictor"] = None
_AI_FALLBACK_KEYWORDS = {
    'machine learning', 'deep learning', 'mlops', 'data science', 'computer vision',
    'nlp', 'llm', 'artificial intelligence', 'rag', 'tensor', 'pytorch', 'scikit'
}
_WEB_FALLBACK_KEYWORDS = {
    'react', 'node', 'javascript', 'frontend', 'php', 'django', 'rails', 'spring',
    'laravel', 'angular', 'vue'
}


class RoadmapPredictor:
    """Loads trained models and generates roadmap phase plans."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        model_dir: Optional[Path] = None,
        auto_train: bool = True
    ) -> None:
        self.feature_store = RoadmapFeatureStore(data_dir=data_dir)
        resolved = model_dir or (self.feature_store.data_dir / 'models' / 'learning_roadmap')
        self.model_dir = Path(resolved)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.duration_model = self._load_model('duration_model.joblib')
        self.success_model = self._load_model('success_model.joblib')
        self.resource_model = self._load_model('resource_ranker.joblib')

        if auto_train and (self.duration_model is None or self.success_model is None or self.resource_model is None):
            try:
                from .training import RoadmapModelTrainer

                trainer = RoadmapModelTrainer(data_dir=self.feature_store.data_dir, model_dir=self.model_dir)
                trainer.train_all()
                self.duration_model = self._load_model('duration_model.joblib')
                self.success_model = self._load_model('success_model.joblib')
                self.resource_model = self._load_model('resource_ranker.joblib')
            except Exception as exc:  # noqa: BLE001
                logger.warning("[roadmap-ml] Auto-training failed: %s", exc)

        self.resource_lookup = self.feature_store.resource_skill_aggregates.set_index('skill').to_dict(orient='index')
        self.resource_catalog = self._prepare_resource_catalog()
        self.default_context = self.feature_store.default_context()

    def build_learning_roadmap(
        self,
        primary_skills: List[str],
        gap_skills: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        ctx = {**self.default_context, **(context or {})}
        core_skills = self._sanitize_skills(primary_skills)[:4] or ['Systems Thinking', 'Delivery Excellence']
        gap_focus = self._sanitize_skills(gap_skills)[:4]
        advanced_focus = self._sanitize_skills(gap_skills)[4:8]
        if not gap_focus:
            gap_focus = core_skills[2:5] or core_skills
        if not advanced_focus:
            advanced_focus = (core_skills + gap_focus)[-3:]

        phase_specs = [
            ('Stabilize Core', core_skills, 'Medium'),
            ('Upskill for Growth', gap_focus, 'High'),
            ('Differentiate', advanced_focus, 'Medium')
        ]

        phases: List[Dict[str, Any]] = []
        total_hours = 0.0
        success_scores: List[float] = []
        for phase_index, (name, skills, effort) in enumerate(phase_specs):
            if not skills:
                continue
            per_skill_details = []
            for order, skill in enumerate(skills):
                feature_row = self._build_feature_row(skill, order, len(skills), ctx)
                hours = self._predict_duration(feature_row)
                success_prob = self._predict_success(feature_row)
                resources = self._recommend_resources(skill)
                per_skill_details.append({
                    'skill': skill,
                    'hours': hours,
                    'success_prob': success_prob,
                    'resources': resources
                })
            phase_hours = sum(item['hours'] for item in per_skill_details)
            total_hours += phase_hours
            avg_success = float(np.mean([item['success_prob'] for item in per_skill_details])) if per_skill_details else 0.72
            success_scores.append(avg_success)
            weeks = max(2, math.ceil(phase_hours / max(ctx.get('weekly_hours_available', 6), 1)))
            phases.append({
                'phase_name': name,
                'skills': skills,
                'effort_level': effort,
                'duration_weeks': weeks,
                'predicted_hours': phase_hours,
                'resources': [resource for item in per_skill_details for resource in item['resources']],
                'success_probability': avg_success,
                'skill_details': per_skill_details
            })

        if not phases:
            raise ValueError("No skills provided to build roadmap")

        personalization_grade = self._personalization_grade(core_skills, gap_focus, advanced_focus)
        overall_success = float(np.mean(success_scores)) if success_scores else 0.78
        strategy = self._learning_strategy(len(gap_focus) + len(advanced_focus))

        return {
            'phases': phases,
            'total_hours': total_hours,
            'total_weeks': sum(phase['duration_weeks'] for phase in phases),
            'personalization': personalization_grade,
            'predicted_success_rate': overall_success,
            'learning_strategy': strategy
        }

    def _build_feature_row(self, skill: str, order: int, skills_in_phase: int, ctx: Dict[str, Any]) -> pd.DataFrame:
        aggregates = self.resource_lookup.get(skill, {})
        feature_dict = {
            'skill': skill,
            'skill_order': order,
            'skills_in_phase': skills_in_phase,
            'weekly_hours_available': ctx.get('weekly_hours_available', self.default_context['weekly_hours_available']),
            'experience_years': ctx.get('experience_years', self.default_context['experience_years']),
            'seniority': ctx.get('seniority', self.default_context['seniority']),
            'industry': ctx.get('industry', self.default_context['industry']),
            'target_role': ctx.get('target_role', self.default_context['target_role']),
            'avg_resource_duration': aggregates.get('avg_resource_duration', np.nan),
            'avg_resource_cost': aggregates.get('avg_resource_cost', np.nan),
            'avg_resource_success': aggregates.get('avg_resource_success', np.nan),
            'avg_resource_difficulty': aggregates.get('avg_resource_difficulty', np.nan),
            'resource_count': aggregates.get('resources_available', 0)
        }
        df = pd.DataFrame([feature_dict])
        fill_defaults = {
            'avg_resource_duration': 6.0,
            'avg_resource_cost': float(self.feature_store.resource_metadata['cost'].median()) if not self.feature_store.resource_metadata.empty else 50.0,
            'avg_resource_success': 70.0,
            'avg_resource_difficulty': 3.0,
            'resource_count': 0
        }
        df = df.fillna(fill_defaults)
        numeric_cols = [
            'skill_order', 'skills_in_phase', 'weekly_hours_available', 'experience_years',
            'avg_resource_duration', 'avg_resource_cost', 'avg_resource_success',
            'avg_resource_difficulty', 'resource_count'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        return df

    def _predict_duration(self, features: pd.DataFrame) -> float:
        if self.duration_model is None:
            return float(features['avg_resource_duration'].fillna(6).iloc[0] * 1.2)
        prediction = float(self.duration_model.predict(features)[0])
        return max(2.0, prediction)

    def _predict_success(self, features: pd.DataFrame) -> float:
        if self.success_model is None:
            base = features['avg_resource_success'].fillna(70).iloc[0] / 100.0
            return float(min(0.95, max(0.55, base)))
        prediction = float(self.success_model.predict(features)[0])
        return min(0.97, max(0.5, prediction))

    def _recommend_resources(self, skill: str, limit: int = 4) -> List[Dict[str, Any]]:
        catalog = self.resource_catalog
        if skill not in catalog or catalog[skill].empty:
            return self._generate_fallback_resources(skill, limit)
        skill_df = catalog[skill]
        working_df = skill_df.copy()
        if self.resource_model is not None:
            feature_cols = ['duration_hours', 'cost', 'difficulty', 'effectiveness_score', 'skill', 'type', 'is_budget']
            preds = self.resource_model.predict(working_df[feature_cols])
            working_df['predicted_score'] = preds
        else:
            working_df['predicted_score'] = working_df['past_success_rate'] / working_df['duration_hours'].clip(lower=0.5)
        working_df = working_df.sort_values('predicted_score', ascending=False).head(limit)
        resources: List[Dict[str, Any]] = []
        for _, row in working_df.iterrows():
            resources.append({
                'skill': skill,
                'title': f"{skill} {row['type'].title()} #{row['resource_id']}",
                'provider': 'SkillSync Catalog',
                'duration': f"{row['duration_hours']:.1f}h",
                'rating': round(row['past_success_rate'] / 20.0, 2),
                'url': f"https://learn.skillsync.ai/{row['resource_id'].lower()}",
                'cost': f"${row['cost']:.0f}" if row['cost'] > 0 else 'Free',
                'is_free': bool(row['cost'] == 0),
                'tier': 'Core' if row['difficulty'] <= 3 else 'Advanced',
                'estimated_time_hours': int(round(row['duration_hours'])),
                'time_hours': int(round(row['duration_hours'])),
                'link': f"https://learn.skillsync.ai/{row['resource_id'].lower()}"
            })
        return resources

    def _generate_fallback_resources(self, skill: str, limit: int) -> List[Dict[str, Any]]:
        domain = self._infer_skill_domain(skill)
        tiers = self._fallback_resource_templates(domain)
        query = re.sub(r"\s+", "+", skill.strip()) or "tech"
        slug = re.sub(r"[^a-z0-9]+", "-", skill.strip().lower()) or "skill"
        resources: List[Dict[str, Any]] = []
        for index in range(min(limit, len(tiers))):
            tier_name, rating, base_hours, provider, url_template = tiers[index]
            hours = base_hours + index * 2
            url = url_template.format(query=query)
            resources.append({
                'skill': skill,
                'title': f"{skill} {tier_name} Track",
                'provider': provider,
                'duration': f"{hours}h",
                'rating': rating,
                'url': url,
                'cost': 'Free',
                'is_free': True,
                'tier': tier_name,
                'estimated_time_hours': hours,
                'time_hours': hours,
                'link': url
            })
        if len(resources) < limit:
            resources.append({
                'skill': skill,
                'title': f"{skill} Open Courseware",
                'provider': 'MIT OCW',
                'duration': '6h',
                'rating': 4.1,
                'url': f"https://ocw.mit.edu/search/?q={query}",
                'cost': 'Free',
                'is_free': True,
                'tier': 'Core',
                'estimated_time_hours': 6,
                'time_hours': 6,
                'link': f"https://ocw.mit.edu/search/?q={query}"
            })
        return resources

    def _infer_skill_domain(self, skill: str) -> str:
        lowered = skill.lower()
        if any(keyword in lowered for keyword in _AI_FALLBACK_KEYWORDS):
            return 'ai'
        if any(keyword in lowered for keyword in _WEB_FALLBACK_KEYWORDS):
            return 'web'
        return 'general'

    def _fallback_resource_templates(self, domain: str) -> List[Tuple[str, float, int, str, str]]:
        if domain == 'ai':
            return [
                ('Core', 4.5, 6, 'Coursera ML', 'https://www.coursera.org/search?query={query}+machine+learning'),
                ('Applied', 4.4, 8, 'DeepLearning.AI', 'https://www.deeplearning.ai/courses/?search={query}'),
                ('Proof', 4.6, 10, 'Kaggle Projects', 'https://www.kaggle.com/search?q={query}')
            ]
        if domain == 'web':
            return [
                ('Core', 4.3, 4, 'freeCodeCamp', 'https://www.freecodecamp.org/learn'),
                ('Applied', 4.2, 6, 'Udemy Labs', 'https://www.udemy.com/courses/search/?q={query}'),
                ('Proof', 4.1, 8, 'Frontend Mentor', 'https://www.frontendmentor.io/challenges?search={query}')
            ]
        return [
            ('Core', 4.2, 4, 'Coursera', 'https://www.coursera.org/search?query={query}'),
            ('Applied', 4.3, 6, 'Udemy Labs', 'https://www.udemy.com/courses/search/?q={query}'),
            ('Proof', 4.5, 8, 'Kaggle Projects', 'https://www.kaggle.com/search?q={query}')
        ]

    def _prepare_resource_catalog(self) -> Dict[str, pd.DataFrame]:
        df = self.feature_store.resource_metadata.copy()
        if df.empty:
            return {}
        df['effectiveness_score'] = df['past_success_rate'] / df['duration_hours'].clip(lower=0.5)
        median_cost = df['cost'].median()
        df['is_budget'] = (df['cost'] <= median_cost).astype(int)
        grouped: Dict[str, pd.DataFrame] = {}
        for skill, subset in df.groupby('skill'):
            grouped[skill] = subset.reset_index(drop=True)
        return grouped

    def _personalization_grade(self, core: List[str], gap: List[str], advanced: List[str]) -> str:
        unique_skills = len(set(core + gap + advanced))
        if unique_skills >= 9:
            return 'A+'
        if unique_skills >= 6:
            return 'A'
        if unique_skills >= 4:
            return 'A-'
        return 'B+'

    @staticmethod
    def _learning_strategy(gap_count: int) -> str:
        if gap_count <= 3:
            return 'Focused Sprint'
        if gap_count <= 6:
            return 'Balanced Growth'
        return 'Comprehensive Reinvention'

    @staticmethod
    def _sanitize_skills(skills: List[str]) -> List[str]:
        return [skill.strip().title() for skill in skills if isinstance(skill, str) and skill.strip()]

    def _load_model(self, filename: str):
        path = self.model_dir / filename
        if not path.exists():
            return None
        try:
            return joblib.load(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[roadmap-ml] Failed to load %s: %s", path, exc)
            return None


def get_default_predictor(force_reload: bool = False) -> RoadmapPredictor:
    global _DEFAULT_PREDICTOR
    if force_reload:
        _DEFAULT_PREDICTOR = None
    if _DEFAULT_PREDICTOR is None:
        _DEFAULT_PREDICTOR = RoadmapPredictor()
    return _DEFAULT_PREDICTOR


__all__ = ["RoadmapPredictor", "get_default_predictor"]
