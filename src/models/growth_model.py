"""
Generalized Growth Model with 8-method ML comparison and conformal prediction.
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import (
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from config.settings import MODEL1_FEATURES, MODEL3_FEATURES


class GrowthModel:
    """Generalized growth model wrapping any sklearn-compatible estimator.

    Maintains interface: .predict(), .predict_with_ci(), .features, .residuals,
    .scaler, .metrics, .feature_importances, .summary()
    """

    def __init__(self, name: str, features: list, estimator=None):
        self.name = name
        self.features = features
        self.estimator_template = estimator
        self.model = None
        self.scaler = StandardScaler()
        self.residuals = None
        self.metrics = {}
        self.feature_importances = {}

    def _get_feature_importances(self, X_scaled, y):
        """Extract feature importances based on model type."""
        model = self.model
        if hasattr(model, 'coef_'):
            coefs = np.abs(np.ravel(model.coef_))
            total = coefs.sum()
            return {
                f: round(c / total * 100, 2) if total > 0 else 0
                for f, c in zip(self.features, coefs)
            }
        if hasattr(model, 'feature_importances_'):
            imps = model.feature_importances_
            total = imps.sum()
            return {
                f: round(imp / total * 100, 2) if total > 0 else 0
                for f, imp in zip(self.features, imps)
            }
        try:
            perm_result = permutation_importance(
                model, X_scaled, y, n_repeats=5, random_state=42, n_jobs=-1
            )
            imps = np.abs(perm_result.importances_mean)
            total = imps.sum()
            return {
                f: round(imp / total * 100, 2) if total > 0 else 0
                for f, imp in zip(self.features, imps)
            }
        except Exception:
            return {f: 0 for f in self.features}

    def train(self, df: pd.DataFrame, target: str = 'weight_kg'):
        X = df[self.features].values
        y = df[target].values
        X_scaled = self.scaler.fit_transform(X)

        self.model = clone(self.estimator_template)
        self.model.fit(X_scaled, y)

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_preds = cross_val_predict(
            clone(self.estimator_template), X_scaled, y, cv=cv
        )

        r2 = r2_score(y, cv_preds)
        rmse = np.sqrt(mean_squared_error(y, cv_preds))
        mae = mean_absolute_error(y, cv_preds)
        pearson_r = float(np.corrcoef(y, cv_preds)[0, 1])
        slope, intercept = np.polyfit(cv_preds, y, 1)

        self.metrics = {
            'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae),
            'pearson_r': pearson_r, 'cal_slope': float(slope),
            'n': int(len(y)), 'n_features': len(self.features),
            'method': self.name
        }

        self.residuals = np.abs(y - cv_preds)
        self.feature_importances = self._get_feature_importances(X_scaled, y)
        return self

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X_df[self.features].values)
        return self.model.predict(X_scaled)

    def predict_with_ci(self, X_df: pd.DataFrame, coverage: float = 0.90):
        """Conformal prediction: properly calibrated confidence intervals."""
        preds = self.predict(X_df)
        q = np.quantile(self.residuals, coverage)
        return preds, preds - q, preds + q

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"  {self.name}",
            f"  n = {m['n']:,}  |  Features = {m['n_features']}",
            f"  R² = {m['r2']:.4f}  |  RMSE = {m['rmse']:.3f} kg  |  MAE = {m['mae']:.3f} kg",
            f"  Pearson r = {m['pearson_r']:.4f}  |  Cal. slope = {m['cal_slope']:.4f}",
        ]
        top5 = dict(sorted(self.feature_importances.items(),
                            key=lambda x: -x[1])[:5])
        lines.append(f"  Top features: {top5}")
        return '\n'.join(lines)

    def save(self, path: Path):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'name': self.name,
            'features': self.features,
            'model': self.model,
            'scaler': self.scaler,
            'residuals': self.residuals,
            'metrics': self.metrics,
            'feature_importances': self.feature_importances,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> 'GrowthModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gm = cls(name=data['name'], features=data['features'])
        gm.model = data['model']
        gm.scaler = data['scaler']
        gm.residuals = data['residuals']
        gm.metrics = data['metrics']
        gm.feature_importances = data.get('feature_importances', {})
        return gm


def get_method_configs() -> dict:
    """Return the 8 ML method configurations."""
    configs = {
        'Elastic Net': ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            alphas=None, cv=5, random_state=42, max_iter=10000, n_jobs=-1
        ),
        'Ridge': RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5),
        'Extra Trees': ExtraTreesRegressor(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1
        ),
        'SVR': SVR(kernel='rbf', C=10.0, epsilon=0.1),
        'KNN': KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1),
    }
    if HAS_XGBOOST:
        configs['XGBoost'] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42, n_jobs=-1, verbosity=0
        )
    return configs
