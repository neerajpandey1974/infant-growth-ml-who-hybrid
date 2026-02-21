"""
Hybrid Growth Predictor:
- Uses the best ML model for high-accuracy weight-for-age predictions
- Falls back to z-score persistence for other metrics or when features unavailable
- Conformal prediction for calibrated CIs
- Anomaly detection via extreme z-scores and percentile crossing
"""
import numpy as np
import pandas as pd
from typing import List, Optional

from src.models.who_engine import WHOZScoreEngine
from src.models.growth_model import GrowthModel
from src.models.data_structures import (
    InfantProfile, Observation, Prediction, Anomaly
)
from config.settings import MODEL3_FEATURES


class GrowthPredictor:
    """Hybrid Growth Predictor with ML primary and z-score persistence fallback."""

    def __init__(self, who_engine: WHOZScoreEngine,
                 weight_model: GrowthModel,
                 nhanes_population: pd.DataFrame = None):
        self.who_engine = who_engine
        self.weight_model = weight_model

        # Compute population z-score stats for fallback
        self.pop_z_stats = {}
        if nhanes_population is not None:
            for metric_col, who_metric in [
                ('weight_kg', 'weight_for_age'),
                ('length_cm', 'length_for_age'),
                ('head_circ_cm', 'head_circumference_for_age'),
            ]:
                z_col = f'{metric_col}_zscore'
                if z_col in nhanes_population.columns:
                    valid = nhanes_population[z_col].dropna()
                    if len(valid) > 0:
                        self.pop_z_stats[who_metric] = {
                            'mean': float(valid.mean()),
                            'std': float(valid.std()),
                            'median': float(valid.median()),
                        }

    def set_pop_z_stats(self, stats: dict):
        """Set population z-score statistics directly (for loading from saved model)."""
        self.pop_z_stats = stats

    def _build_feature_vector(self, infant: InfantProfile,
                               target_age: float) -> Optional[pd.DataFrame]:
        """Build a feature vector from infant observations."""
        features = {
            'age_months': target_age,
            'sex_female': 1.0 if infant.sex == 'female' else 0.0,
            'birth_weight_kg': infant.birth_weight_kg if infant.birth_weight_kg else np.nan,
        }

        # Race — default to 0 (unknown)
        for r in [2, 3, 4, 5]:
            features[f'race_{r}'] = 0.0

        # Get latest body measurements
        for obs in sorted(infant.observations, key=lambda o: o.age_months, reverse=True):
            if obs.metric == 'length_for_age' and 'length_cm' not in features:
                age_diff = target_age - obs.age_months
                if age_diff <= 3:
                    features['length_cm'] = obs.value
                else:
                    z = obs.z_score
                    projected = self.who_engine.zscore_to_value(
                        'length_for_age', infant.sex, target_age, z
                    )
                    features['length_cm'] = projected
            if obs.metric == 'head_circumference_for_age' and 'head_circ_cm' not in features:
                age_diff = target_age - obs.age_months
                if age_diff <= 3:
                    features['head_circ_cm'] = obs.value
                else:
                    z = obs.z_score
                    projected = self.who_engine.zscore_to_value(
                        'head_circumference_for_age', infant.sex, target_age, z
                    )
                    features['head_circ_cm'] = projected

        # Fill defaults
        defaults = {
            'income_poverty_ratio': 2.0, 'family_size': 4.0,
            'arm_circ_cm': np.nan, 'arm_length_cm': np.nan,
            'head_circ_cm': np.nan, 'birth_weight_kg': 3.3,
        }
        for k, v in defaults.items():
            if k not in features or (isinstance(features.get(k), float)
                                     and np.isnan(features.get(k, 0))):
                features[k] = v

        # Check minimum features
        required = set(self.weight_model.features)
        available = set(
            k for k, v in features.items()
            if not (isinstance(v, float) and np.isnan(v))
        )
        if not required.issubset(available):
            return None

        return pd.DataFrame([features])

    def predict(self, infant: InfantProfile, metric: str,
                future_ages: List[float]) -> List[Prediction]:
        """Generate growth predictions for future ages."""
        observations = infant.get_observations(metric)
        predictions = []

        for target_age in future_ages:
            # Strategy 1: ML model (primary for weight)
            if metric == 'weight_for_age':
                feat_df = self._build_feature_vector(infant, target_age)
                if feat_df is not None:
                    try:
                        pred_val, ci_lo, ci_hi = self.weight_model.predict_with_ci(
                            feat_df, coverage=0.90
                        )
                        pred_val = float(pred_val[0])
                        ci_lo = float(ci_lo[0])
                        ci_hi = float(ci_hi[0])
                        z = self.who_engine.compute_zscore(
                            metric, infant.sex, target_age, pred_val
                        )
                        pct = self.who_engine.zscore_to_percentile(z)

                        predictions.append(Prediction(
                            age_months=target_age, metric=metric,
                            predicted_value=pred_val, predicted_zscore=z,
                            predicted_percentile=pct,
                            ci_lower=max(ci_lo, 0.5), ci_upper=ci_hi,
                            method='ml'
                        ))
                        continue
                    except Exception:
                        pass

            # Strategy 2: Z-score persistence fallback
            if observations:
                last_age = max(o.age_months for o in observations)
                weights = [np.exp(0.1 * (o.age_months - last_age))
                           for o in observations]
                z_avg = np.average(
                    [o.z_score for o in observations], weights=weights
                )

                pop_stats = self.pop_z_stats.get(metric, {'mean': 0, 'std': 1})
                delta_t = max(target_age - last_age, 0.01)
                rho = 0.80
                n_obs = len(observations)
                obs_factor = 0.5 + 0.5 * min(n_obs / 5.0, 1.0)
                alpha = (rho ** (delta_t / 6.0)) * obs_factor
                z_pred = alpha * z_avg + (1 - alpha) * pop_stats['mean']

                sigma_z = pop_stats.get('std', 1.0) * (
                    0.45 + 0.07 * delta_t
                ) / (1 + 0.15 * n_obs)
                sigma_z = max(sigma_z, 0.35)

                z_lo = z_pred - 1.645 * sigma_z
                z_hi = z_pred + 1.645 * sigma_z

                pred_val = self.who_engine.zscore_to_value(
                    metric, infant.sex, target_age, z_pred
                )
                ci_lo = self.who_engine.zscore_to_value(
                    metric, infant.sex, target_age, z_lo
                )
                ci_hi = self.who_engine.zscore_to_value(
                    metric, infant.sex, target_age, z_hi
                )
                pct = self.who_engine.zscore_to_percentile(z_pred)

                predictions.append(Prediction(
                    age_months=target_age, metric=metric,
                    predicted_value=pred_val, predicted_zscore=z_pred,
                    predicted_percentile=pct,
                    ci_lower=ci_lo, ci_upper=ci_hi,
                    method='zscore_persistence'
                ))
            else:
                # No data: use WHO median
                med = self.who_engine.get_median(metric, infant.sex, target_age)
                predictions.append(Prediction(
                    age_months=target_age, metric=metric,
                    predicted_value=med, predicted_zscore=0.0,
                    predicted_percentile=50.0,
                    ci_lower=self.who_engine.zscore_to_value(
                        metric, infant.sex, target_age, -1.645
                    ),
                    ci_upper=self.who_engine.zscore_to_value(
                        metric, infant.sex, target_age, 1.645
                    ),
                    method='who_median'
                ))

        return predictions

    def detect_anomalies(self, infant: InfantProfile,
                         z_warning: float = 2.0,
                         z_critical: float = 3.0,
                         pct_drop_threshold: float = 25.0) -> List[Anomaly]:
        """Detect growth anomalies from observations."""
        anomalies = []
        for metric in set(o.metric for o in infant.observations):
            obs_sorted = sorted(
                infant.get_observations(metric),
                key=lambda o: o.age_months
            )

            # Extreme z-score detection
            for obs in obs_sorted:
                if abs(obs.z_score) >= z_critical:
                    anomalies.append(Anomaly(
                        metric=metric, anomaly_type='extreme_zscore',
                        severity='critical',
                        message=f"Z-score {obs.z_score:.2f} at {obs.age_months:.0f}mo",
                        z_score=obs.z_score
                    ))
                elif abs(obs.z_score) >= z_warning:
                    anomalies.append(Anomaly(
                        metric=metric, anomaly_type='extreme_zscore',
                        severity='warning',
                        message=f"Z-score {obs.z_score:.2f} at {obs.age_months:.0f}mo",
                        z_score=obs.z_score
                    ))

            # Percentile crossing detection
            if len(obs_sorted) >= 2:
                for i in range(1, len(obs_sorted)):
                    drop = obs_sorted[i - 1].percentile - obs_sorted[i].percentile
                    if drop > pct_drop_threshold:
                        anomalies.append(Anomaly(
                            metric=metric, anomaly_type='percentile_drop',
                            severity='warning',
                            message=(
                                f"P{obs_sorted[i-1].percentile:.0f}→"
                                f"P{obs_sorted[i].percentile:.0f} "
                                f"({obs_sorted[i-1].age_months:.0f}→"
                                f"{obs_sorted[i].age_months:.0f}mo)"
                            ),
                            percentile_drop=drop
                        ))

        return anomalies
