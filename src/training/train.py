"""
Training Pipeline for Infant Growth ML+WHO Hybrid System.

Usage:
    python -m src.training.train

Downloads NHANES data, trains 8 ML methods, selects best, and saves models.
"""
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from config.settings import (
    MODEL1_FEATURES, MODEL3_FEATURES, MODELS_DIR, DATA_DIR
)
from src.models.who_engine import WHOZScoreEngine
from src.models.growth_model import GrowthModel, get_method_configs


def compute_nhanes_zscores(df: pd.DataFrame,
                           engine: WHOZScoreEngine) -> pd.DataFrame:
    """Compute WHO z-scores for all available metrics in the NHANES data."""
    df = df.copy()
    metric_map = {
        'weight_kg': 'weight_for_age',
        'length_cm': 'length_for_age',
        'head_circ_cm': 'head_circumference_for_age',
    }
    for col, who_metric in metric_map.items():
        if col in df.columns:
            z_col = f'{col}_zscore'
            df[z_col] = np.nan
            mask = df[col].notna() & df['age_months'].notna() & df['sex'].notna()
            for idx in df[mask].index:
                try:
                    z = engine.compute_zscore(
                        who_metric, df.loc[idx, 'sex'],
                        float(df.loc[idx, 'age_months']),
                        float(df.loc[idx, col])
                    )
                    df.loc[idx, z_col] = z
                except Exception:
                    pass
    return df


def run_training():
    """Full training pipeline."""
    print("=" * 70)
    print("  INFANT GROWTH ML+WHO HYBRID — TRAINING PIPELINE")
    print("=" * 70)

    # ── 1. Data Ingestion ──
    print("\n[1/6] Loading NHANES data...")
    from src.ingestion.pipeline import build_nhanes_dataset
    nhanes_df = build_nhanes_dataset()

    # ── 2. WHO Z-Scores ──
    print("\n[2/6] Computing WHO z-scores...")
    who_engine = WHOZScoreEngine()
    nhanes_df = compute_nhanes_zscores(nhanes_df, who_engine)

    # ── 3. Feature Engineering ──
    print("\n[3/6] Feature engineering + MICE imputation...")

    # Binary encode sex
    nhanes_df['sex_female'] = (nhanes_df['sex'] == 'female').astype(float)

    # One-hot encode race
    for r in [2, 3, 4, 5]:
        nhanes_df[f'race_{r}'] = (nhanes_df['race_eth'] == r).astype(float)

    # Model 1 imputation (demographics only)
    m1_cols = MODEL1_FEATURES + ['weight_kg']
    m1_available = [c for c in m1_cols if c in nhanes_df.columns]
    m1_data = nhanes_df[m1_available].copy()
    m1_data = m1_data[m1_data['weight_kg'].notna()].copy()

    # Drop columns that are entirely NaN (imputer can't handle them)
    all_nan_cols = [c for c in m1_data.columns if m1_data[c].isna().all()]
    if all_nan_cols:
        print(f"  Dropping all-NaN columns from Model 1: {all_nan_cols}")
        m1_data = m1_data.drop(columns=all_nan_cols)

    imputer_m1 = IterativeImputer(max_iter=10, random_state=42,
                                   sample_posterior=False)
    m1_imputed = pd.DataFrame(
        imputer_m1.fit_transform(m1_data),
        columns=m1_data.columns, index=m1_data.index
    )
    print(f"  Model 1: {len(m1_imputed)} infants ({len(m1_data.columns)-1} features)")

    # Model 3 imputation (full features)
    m3_cols = MODEL3_FEATURES + ['weight_kg']
    m3_available = [c for c in m3_cols if c in nhanes_df.columns]
    m3_data = nhanes_df[m3_available].copy()
    m3_data = m3_data[
        m3_data['weight_kg'].notna() & m3_data['length_cm'].notna()
    ].copy()

    # Drop columns that are entirely NaN
    all_nan_cols_m3 = [c for c in m3_data.columns if m3_data[c].isna().all()]
    if all_nan_cols_m3:
        print(f"  Dropping all-NaN columns from Model 3: {all_nan_cols_m3}")
        m3_data = m3_data.drop(columns=all_nan_cols_m3)

    imputer_m3 = IterativeImputer(max_iter=10, random_state=42,
                                   sample_posterior=False)
    m3_imputed = pd.DataFrame(
        imputer_m3.fit_transform(m3_data),
        columns=m3_data.columns, index=m3_data.index
    )
    print(f"  Model 3: {len(m3_imputed)} infants ({len(m3_data.columns)-1} features)")

    # ── 4. Train 8 Methods ──
    print("\n[4/6] Training 8 ML methods on Model 3 features...")
    print("=" * 70)

    # Use actual features present after NaN column removal
    m3_features_actual = [c for c in m3_data.columns if c != 'weight_kg']

    method_configs = get_method_configs()
    all_models = {}
    comparison_rows = []

    for method_name, estimator in method_configs.items():
        print(f"  Training: {method_name}...", end=" ", flush=True)
        model = GrowthModel(
            f"Model 3: {method_name}",
            m3_features_actual,
            estimator=estimator
        )
        model.train(m3_imputed, 'weight_kg')
        all_models[method_name] = model

        m = model.metrics
        comparison_rows.append({
            'Method': method_name,
            'R²': m['r2'], 'RMSE': m['rmse'], 'MAE': m['mae'],
            'Pearson r': m['pearson_r'], 'Cal. Slope': m['cal_slope'],
            '90% CI Width': round(2 * np.quantile(model.residuals, 0.90), 3),
        })
        print(f"R²={m['r2']:.4f}  RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}")

    # Comparison table
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values('R²', ascending=False).reset_index(drop=True)
    comparison_df.index += 1
    print(f"\n{'='*80}")
    print("  MODEL COMPARISON (sorted by R², 5-fold CV)")
    print(f"{'='*80}")
    print(comparison_df.to_string(float_format='%.4f'))

    # ── 5. Auto-select best ──
    best_method = comparison_df.iloc[0]['Method']
    best_model = all_models[best_method]
    print(f"\n{'*'*70}")
    print(f"  BEST METHOD: {best_method}  (R² = {best_model.metrics['r2']:.4f})")
    print(f"{'*'*70}")

    # Also train Model 1 (demographics only) with Elastic Net
    from sklearn.linear_model import ElasticNetCV
    m1_features_actual = [c for c in m1_data.columns if c != 'weight_kg']
    model1 = GrowthModel(
        "Model 1: Elastic Net (Demographics)",
        m1_features_actual,
        estimator=ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            alphas=None, cv=5, random_state=42, max_iter=10000, n_jobs=-1
        )
    )
    model1.train(m1_imputed, 'weight_kg')
    print(f"\n  Model 1 (fallback): R²={model1.metrics['r2']:.4f}")

    # ── 6. Save ──
    print("\n[5/6] Saving models...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_model.save(MODELS_DIR / 'best_model.pkl')
    model1.save(MODELS_DIR / 'model1_demographics.pkl')

    # Save imputers
    with open(MODELS_DIR / 'imputer_m3.pkl', 'wb') as f:
        pickle.dump(imputer_m3, f)

    # Save population z-score stats
    pop_z_stats = {}
    for metric_col, who_metric in [
        ('weight_kg', 'weight_for_age'),
        ('length_cm', 'length_for_age'),
        ('head_circ_cm', 'head_circumference_for_age'),
    ]:
        z_col = f'{metric_col}_zscore'
        if z_col in nhanes_df.columns:
            valid = nhanes_df[z_col].dropna()
            if len(valid) > 0:
                pop_z_stats[who_metric] = {
                    'mean': float(valid.mean()),
                    'std': float(valid.std()),
                    'median': float(valid.median()),
                }

    with open(MODELS_DIR / 'pop_z_stats.json', 'w') as f:
        json.dump(pop_z_stats, f, indent=2)

    # Save comparison table
    comparison_df.to_csv(MODELS_DIR / 'comparison.csv', index=True)

    # Save training metadata
    metadata = {
        'best_method': best_method,
        'best_r2': float(best_model.metrics['r2']),
        'best_rmse': float(best_model.metrics['rmse']),
        'n_training_samples': int(len(m3_imputed)),
        'n_nhanes_cycles': len(set(nhanes_df.get('cycle', []))),
        'n_methods_compared': len(all_models),
        'model3_features': m3_features_actual,
        'model1_features': m1_features_actual,
        'conformal_ci_width_90': float(
            2 * np.quantile(best_model.residuals, 0.90)
        ),
    }
    with open(MODELS_DIR / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save processed NHANES data for predictor
    nhanes_df.to_parquet(DATA_DIR / 'nhanes_processed.parquet', index=False)

    print(f"\n[6/6] All files saved to {MODELS_DIR}")
    print(f"  ├── best_model.pkl ({best_method})")
    print(f"  ├── model1_demographics.pkl")
    print(f"  ├── imputer_m3.pkl")
    print(f"  ├── pop_z_stats.json")
    print(f"  ├── comparison.csv")
    print(f"  ├── metadata.json")
    print(f"  └── ../nhanes_processed.parquet")

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    return best_model, who_engine, nhanes_df


if __name__ == '__main__':
    run_training()
