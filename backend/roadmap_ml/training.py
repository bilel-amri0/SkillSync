"""Training utilities for learning roadmap predictive models."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in (None, ""):
    PACKAGE_ROOT = Path(__file__).resolve().parent
    sys.path.append(str(PACKAGE_ROOT.parent))
    from roadmap_ml.feature_store import RoadmapFeatureStore
else:
    from .feature_store import RoadmapFeatureStore


class RoadmapModelTrainer:
    """High-level trainer that fits and persists roadmap models from historical data."""

    def __init__(self, data_dir: Optional[Path] = None, model_dir: Optional[Path] = None) -> None:
        self.feature_store = RoadmapFeatureStore(data_dir=data_dir)
        resolved_model_dir = model_dir or (self.feature_store.data_dir / 'models' / 'learning_roadmap')
        self.model_dir = Path(resolved_model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.numeric_features = [
            'skill_order',
            'skills_in_phase',
            'weekly_hours_available',
            'experience_years',
            'avg_resource_duration',
            'avg_resource_cost',
            'avg_resource_success',
            'avg_resource_difficulty',
            'resource_count'
        ]
        self.categorical_features = ['skill', 'seniority', 'industry', 'target_role']

    def train_all(self) -> Dict[str, Dict[str, float]]:
        """Train all roadmap models and return evaluation metrics."""
        skill_df = self.feature_store.build_skill_examples()
        metrics: Dict[str, Dict[str, float]] = {}

        metrics['duration'] = self._train_duration_model(skill_df)
        metrics['success'] = self._train_success_model(skill_df)
        metrics['resource'] = self._train_resource_ranker()

        summary_path = self.model_dir / 'training_metrics.json'
        summary_path.write_text(json.dumps(metrics, indent=2))
        return metrics

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ]
        )

    def _train_duration_model(self, dataset: pd.DataFrame) -> Dict[str, float]:
        features = dataset[self.numeric_features + self.categorical_features]
        target = dataset['duration_target']
        X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=42, test_size=0.2)
        pipeline = Pipeline([
            ('pre', self._build_preprocessor()),
            ('reg', RandomForestRegressor(n_estimators=300, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        joblib.dump(pipeline, self.model_dir / 'duration_model.joblib')
        return {'mae_hours': mae, 'train_size': len(X_train), 'test_size': len(X_test)}

    def _train_success_model(self, dataset: pd.DataFrame) -> Dict[str, float]:
        features = dataset[self.numeric_features + self.categorical_features]
        target = dataset['success_target'].clip(0, 1)
        X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=24, test_size=0.2)
        pipeline = Pipeline([
            ('pre', self._build_preprocessor()),
            ('reg', GradientBoostingRegressor(random_state=24))
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        joblib.dump(pipeline, self.model_dir / 'success_model.joblib')
        return {'mae': mae, 'train_size': len(X_train), 'test_size': len(X_test)}

    def _train_resource_ranker(self) -> Dict[str, float]:
        dataset = self.feature_store.build_resource_training_frame()
        numeric = ['duration_hours', 'cost', 'difficulty', 'effectiveness_score']
        categorical = ['skill', 'type', 'is_budget']
        features = dataset[numeric + categorical]
        target = dataset['past_success_rate'] / 100.0

        numeric_transformer = Pipeline([
            ('scaler', StandardScaler())
        ])
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric),
            ('cat', categorical_transformer, categorical)
        ])

        pipeline = Pipeline([
            ('pre', preprocessor),
            ('reg', RandomForestRegressor(n_estimators=250, random_state=7))
        ])
        X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=7, test_size=0.2)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        mae = float(mean_absolute_error(y_test, preds))
        joblib.dump(pipeline, self.model_dir / 'resource_ranker.joblib')
        return {'mae': mae, 'train_size': len(X_train), 'test_size': len(X_test)}


def main() -> None:
    trainer = RoadmapModelTrainer()
    metrics = trainer.train_all()
    print("Training complete. Metrics:\n" + json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
