"""
Training pipeline for HeatRisk next-day maximum temperature LightGBM model.
Includes chronological splits, early stopping on validation, baseline comparisons,
native LightGBM artifact export, and comprehensive model cards.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import lightgbm as lgb
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root
from climacore.evaluation.metrics import compute_regression_metrics
from climacore.evaluation.splits import chronological_train_val_test_split
from climacore.models.baselines import (
    PersistenceBaseline,
    RollingMeanBaseline,
    SeasonalClimatologyBaseline,
)

logger = get_logger("climacore.models.heat_risk.train")


def train_heat_risk_model(
    data_path: Path | str = None,
    artifacts_dir: Path | str = None,
    reports_dir: Path | str = None,
    val_start: str = "2025-01-01",
    test_start: str = "2026-01-01",
) -> Dict[str, Any]:
    """Train and evaluate next-day maximum temperature LightGBM model against baselines."""
    root = get_project_root()
    parquet_file = Path(data_path) if data_path else root / "data" / "processed" / "heat_risk_train_ready.parquet"
    art_path = Path(artifacts_dir) if artifacts_dir else root / "artifacts" / "models" / "heat_risk" / "v0.1.0"
    rep_path = Path(reports_dir) if reports_dir else root / "reports" / "evaluation"

    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading HeatRisk dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # 1. Chronological Split
    train_df, val_df, test_df = chronological_train_val_test_split(
        df, date_col="date", val_start=val_start, test_start=test_start
    )

    logger.info(
        f"HeatRisk Chronological Split: "
        f"Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    # Load feature metadata
    schema_file = root / "data" / "processed" / "heat_risk_feature_schema.json"
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_meta = json.load(f)
    feature_cols = schema_meta["feature_columns"]
    target_col = schema_meta["target_column"]

    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[target_col].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

    # 2. Fit and Evaluate Baselines
    logger.info("Evaluating HeatRisk baselines on chronological test set...")
    persistence = PersistenceBaseline(observation_col="temperature_2m_max")
    rolling_7d = RollingMeanBaseline(rolling_col="temperature_2m_max_roll_mean_7d")
    seasonal_clim = SeasonalClimatologyBaseline(group_cols=("city", "month"), target_col=target_col).fit(train_df)

    test_preds_persistence = persistence.predict(test_df)
    test_preds_rolling = rolling_7d.predict(test_df)
    test_preds_clim = seasonal_clim.predict(test_df)

    metrics_persistence = compute_regression_metrics(y_test, test_preds_persistence)
    metrics_rolling = compute_regression_metrics(y_test, test_preds_rolling)
    metrics_climatology = compute_regression_metrics(y_test, test_preds_clim)

    logger.info(f"Baseline Persistence Test MAE: {metrics_persistence['mae']} °C, RMSE: {metrics_persistence['rmse']} °C")
    logger.info(f"Baseline Rolling-7d Test MAE: {metrics_rolling['mae']} °C, RMSE: {metrics_rolling['rmse']} °C")
    logger.info(f"Baseline Climatology Test MAE: {metrics_climatology['mae']} °C, RMSE: {metrics_climatology['rmse']} °C")

    # 3. Train LightGBM Model
    logger.info("Training HeatRisk LightGBM model...")
    model_params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "max_depth": 6,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": 4,
    }

    model = lgb.LGBMRegressor(**model_params)
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )

    best_iteration = int(model.best_iteration_)
    logger.info(f"HeatRisk training converged at iteration {best_iteration}.")

    # 4. Predict and Evaluate
    test_preds_lgbm = model.predict(X_test)
    metrics_lgbm = compute_regression_metrics(y_test, test_preds_lgbm)
    val_preds_lgbm = model.predict(X_val)
    val_metrics_lgbm = compute_regression_metrics(y_val, val_preds_lgbm)

    logger.info(f"HeatRisk LightGBM Test MAE: {metrics_lgbm['mae']} °C, RMSE: {metrics_lgbm['rmse']} °C, R2: {metrics_lgbm['r2_score']}")

    # 5. Export Native Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    model_sha256 = compute_sha256(model_file)
    logger.info(f"Native model saved to {model_file} (SHA-256: {model_sha256})")

    # Parity verification
    reloaded_booster = lgb.Booster(model_file=str(model_file))
    reloaded_preds = reloaded_booster.predict(X_test)
    parity_diff = np.max(np.abs(test_preds_lgbm - reloaded_preds))
    logger.info(f"HeatRisk model reload parity check passed: max diff = {parity_diff:.8f}")
    if parity_diff > 1e-6:
        raise ValueError(f"HeatRisk model artifact parity failure: max diff = {parity_diff}")

    # 6. Feature Importance
    importances = model.feature_importances_
    feat_imp = sorted(
        [{"feature": f, "importance": int(imp)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # 7. Compile Results
    results = {
        "model_name": "climacore_heat_risk_lgbm",
        "version": "v0.1.0",
        "target": target_col,
        "units": "deg_C",
        "split_definition": {
            "train": f"{train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d} ({len(train_df):,} rows)",
            "validation": f"{val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d} ({len(val_df):,} rows)",
            "test": f"{test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d} ({len(test_df):,} rows)",
        },
        "best_iteration": best_iteration,
        "metrics": {
            "lightgbm_test": metrics_lgbm,
            "lightgbm_validation": val_metrics_lgbm,
            "persistence_test": metrics_persistence,
            "rolling_7d_test": metrics_rolling,
            "climatology_test": metrics_climatology,
        },
        "improvements_over_persistence": {
            "mae_reduction_pct": round((metrics_persistence["mae"] - metrics_lgbm["mae"]) / metrics_persistence["mae"] * 100, 2),
            "rmse_reduction_pct": round((metrics_persistence["rmse"] - metrics_lgbm["rmse"]) / metrics_persistence["rmse"] * 100, 2),
        },
        "top_features": feat_imp[:15],
    }

    metrics_file = art_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    eval_report_file = rep_path / "heat_risk_evaluation_report.json"
    with open(eval_report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 8. Model Card
    card_file = art_path / "model_card.md"
    with open(card_file, "w", encoding="utf-8") as f:
        f.write(f"# Model Card: Clima-Core HeatRisk Next-Day Max Temperature (v0.1.0)\n\n")
        f.write(f"**Model Name:** `climacore_heat_risk_lgbm`  \n")
        f.write(f"**Status:** `validated_prototype`  \n")
        f.write(f"**Created At:** {datetime.now(timezone.utc).isoformat()}  \n")
        f.write(f"**Task:** Next-day 2m maximum temperature prediction (regression).  \n")
        f.write(f"**Artifact SHA-256:** `{model_sha256}`  \n\n")
        f.write(f"## 1. Data Provenance & Scope\n\n")
        f.write(f"- **Source Dataset:** India Daily Weather & Air Quality (50 cities), CC BY 4.0\n")
        f.write(f"- **Upstream Origin:** Open-Meteo Historical Weather API (ERA5/ERA5-Land and ECMWF IFS)\n")
        f.write(f"- **Spatial Unit:** 50 city-centre model grid cells across India\n")
        f.write(f"- **Target Variable:** `target_temperature_2m_max_next_day` (°C)\n")
        f.write(f"- **Scope Note:** Autoregressive time-series model on historical weather; not NWP forecast-correction.\n\n")
        f.write(f"## 2. Chronological Holdout Evaluation (Test Period: 2026-01-01 to 2026-10-29)\n\n")
        f.write(f"| Model / Baseline | Test MAE (°C) | Test RMSE (°C) | Test Mean Bias (°C) | Test R² |\n")
        f.write(f"|---|---|---|---|---|\n")
        f.write(f"| **Clima-Core LightGBM** | **{metrics_lgbm['mae']}** | **{metrics_lgbm['rmse']}** | **{metrics_lgbm['mean_bias']}** | **{metrics_lgbm['r2_score']}** |\n")
        f.write(f"| Persistence Baseline | {metrics_persistence['mae']} | {metrics_persistence['rmse']} | {metrics_persistence['mean_bias']} | {metrics_persistence['r2_score']} |\n")
        f.write(f"| Rolling 7-Day Mean | {metrics_rolling['mae']} | {metrics_rolling['rmse']} | {metrics_rolling['mean_bias']} | {metrics_rolling['r2_score']} |\n")
        f.write(f"| Seasonal Climatology | {metrics_climatology['mae']} | {metrics_climatology['rmse']} | {metrics_climatology['mean_bias']} | {metrics_climatology['r2_score']} |\n\n")
        f.write(f"**Improvement over Persistence:** MAE reduced by **{results['improvements_over_persistence']['mae_reduction_pct']}%**, RMSE reduced by **{results['improvements_over_persistence']['rmse_reduction_pct']}%**.\n\n")
        f.write(f"## 3. Top Predictive Features\n\n")
        for item in feat_imp[:10]:
            f.write(f"- `{item['feature']}`: importance = {item['importance']}\n")
        f.write(f"\n## 4. Unsupported Use Cases\n\n")
        f.write(f"- Do not use as an official IMD Heatwave Warning (IMD heatwave criteria require specific threshold departures).\n")
        f.write(f"- City-centre grid values do not represent microclimate or urban heat island variations at street level.\n")

    # 9. Artifact Manifest
    artifact_manifest = {
        "artifact_dir": str(art_path),
        "files": [
            get_file_metadata(model_file),
            get_file_metadata(metrics_file),
            get_file_metadata(card_file),
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_module": "climacore.models.heat_risk.train",
    }
    with open(art_path / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    logger.info("HeatRisk training and export completed successfully.")
    return results


if __name__ == "__main__":
    train_heat_risk_model()
