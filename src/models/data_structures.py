"""
Data structures for the Infant Growth Digital Twin system.
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Observation:
    age_months: float
    metric: str  # 'weight_for_age', 'length_for_age', etc.
    value: float
    z_score: float = 0.0
    percentile: float = 50.0


@dataclass
class Prediction:
    age_months: float
    metric: str
    predicted_value: float
    predicted_zscore: float
    predicted_percentile: float
    ci_lower: float
    ci_upper: float
    method: str = "ml"  # 'ml' or 'zscore_persistence'


@dataclass
class Anomaly:
    metric: str
    anomaly_type: str  # 'extreme_zscore' or 'percentile_drop'
    severity: str      # 'warning' or 'critical'
    message: str
    z_score: float = 0.0
    percentile_drop: float = 0.0


class InfantProfile:
    """Manages an individual infant's data and predictions."""

    def __init__(self, name: str, sex: str, birth_weight_kg: float = None,
                 birth_date: str = None, infant_id: str = None):
        self.name = name
        self.infant_id = infant_id or name
        self.sex = sex
        self.birth_weight_kg = birth_weight_kg
        self.birth_date = birth_date
        self.observations: List[Observation] = []

    def add_observation(self, age_months: float, metric: str, value: float,
                        who_engine=None):
        z = 0.0
        pct = 50.0
        if who_engine:
            z = who_engine.compute_zscore(metric, self.sex, age_months, value)
            pct = who_engine.zscore_to_percentile(z)
        obs = Observation(
            age_months=age_months, metric=metric, value=value,
            z_score=z, percentile=pct
        )
        self.observations.append(obs)
        return obs

    def get_observations(self, metric: str = None) -> List[Observation]:
        if metric:
            return [o for o in self.observations if o.metric == metric]
        return self.observations

    def get_latest_features(self) -> dict:
        """Extract feature dict from latest observations for ML model input."""
        features = {
            'sex_female': 1.0 if self.sex == 'female' else 0.0,
            'birth_weight_kg': self.birth_weight_kg or np.nan,
        }
        for obs in sorted(self.observations, key=lambda o: o.age_months):
            if obs.metric == 'weight_for_age':
                features['age_months'] = obs.age_months
            elif obs.metric == 'length_for_age':
                features['length_cm'] = obs.value
        return features

    def to_dict(self) -> dict:
        return {
            'infant_id': self.infant_id,
            'name': self.name,
            'sex': self.sex,
            'birth_weight_kg': self.birth_weight_kg,
            'birth_date': self.birth_date,
            'observation_count': len(self.observations),
            'observations': [
                {
                    'age_months': o.age_months,
                    'metric': o.metric,
                    'value': round(o.value, 3),
                    'z_score': round(o.z_score, 3),
                    'percentile': round(o.percentile, 1),
                }
                for o in self.observations
            ]
        }
