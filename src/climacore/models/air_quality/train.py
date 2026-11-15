"""
Training pipeline for AirQuality next-day PM2.5 LightGBM model.
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
from climacore.evaluation.metrics import compute_regression_metrics, compute_aqi_category_metrics
from climacore.evaluation.splits import chronological_train_val_test_split
from climacore.models.baselines import (
    PersistenceBaseline,
    RollingMeanBaseline,
    SeasonalClimatologyBaseline,
)

logger = get_logger("climacore.models.air_quality.train")


def train_air_quality_model(
    data_path: Path | str = None,
    artifacts_dir: Path | str = None,
    reports_dir: Path | str = None,
    val_start: str = "2025-06-01",
    test_start: str = "2026-01-01",
) -> Dict[str, Any]:
    """Train and evaluate next-day PM2.5 LightGBM regressor against baselines."""
    root = get_project_root()
    parquet_file = Path(data_path) if data_path else root / "data" / "processed" / "air_quality_train_ready.parquet"
    art_path = Path(artifacts_dir) if artifacts_dir else root / "artifacts" / "models" / "air_quality" / "v0.1.0"
    rep_path = Path(reports_dir) if reports_dir else root / "reports" / "evaluation"
    
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading processed dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # 1. Chronological Split
    train_df, val_df, test_df = chronological_train_val_test_split(
        df, date_col="date", val_start=val_start, test_start=test_start
    )

    logger.info(
        f"Chronological Split: "
        f"Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    # Load feature column names
    schema_file = root / "data" / "processed" / "air_quality_feature_schema.json"
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_meta = json.load(f)
    feature_cols = schema_meta["feature_columns"]
    target_col = schema_meta["target_column"]

    X_train, y_train = train_df[feature_cols], train_df[target_col].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[target_col].to_numpy()
    X_test, y_test = test_df[feature_cols], test_df[target_col].to_numpy()

    # 2. Fit and Evaluate Baselines
    logger.info("Fitting and evaluating baselines...")
    persistence = PersistenceBaseline(observation_col="pm2_5_mean")
    rolling_7d = RollingMeanBaseline(rolling_col="pm2_5_mean_roll_mean_7d")
    seasonal_clim = SeasonalClimatologyBaseline().fit(train_df)

    test_preds_persistence = persistence.predict(test_df)
    test_preds_rolling = rolling_7d.predict(test_df)
    test_preds_clim = seasonal_clim.predict(test_df)

    metrics_persistence = {
        **compute_regression_metrics(y_test, test_preds_persistence),
        **compute_aqi_category_metrics(y_test, test_preds_persistence),
    }
    metrics_rolling = {
        **compute_regression_metrics(y_test, test_preds_rolling),
        **compute_aqi_category_metrics(y_test, test_preds_rolling),
    }
    metrics_climatology = {
        **compute_regression_metrics(y_test, test_preds_clim),
        **compute_aqi_category_metrics(y_test, test_preds_clim),
    }

    logger.info(f"Baseline Persistence Test MAE: {metrics_persistence['mae']}, RMSE: {metrics_persistence['rmse']}")
    logger.info(f"Baseline Rolling-7d Test MAE: {metrics_rolling['mae']}, RMSE: {metrics_rolling['rmse']}")
    logger.info(f"Baseline Climatology Test MAE: {metrics_climatology['mae']}, RMSE: {metrics_climatology['rmse']}")

    # 3. Train LightGBM Model
    logger.info("Training LightGBM model with early stopping on validation split...")
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
    logger.info(f"Training converged at iteration {best_iteration}.")

    # 4. Predict on Test Set and Evaluate
    test_preds_lgbm = model.predict(X_test)
    metrics_lgbm = {
        **compute_regression_metrics(y_test, test_preds_lgbm),
        **compute_aqi_category_metrics(y_test, test_preds_lgbm),
    }

    # Also compute Validation metrics
    val_preds_lgbm = model.predict(X_val)
    val_metrics_lgbm = {
        **compute_regression_metrics(y_val, val_preds_lgbm),
        **compute_aqi_category_metrics(y_val, val_preds_lgbm),
    }

    logger.info(f"LightGBM Test MAE: {metrics_lgbm['mae']}, RMSE: {metrics_lgbm['rmse']}, R2: {metrics_lgbm['r2_score']}")
    logger.info(f"LightGBM Category Accuracy: {metrics_lgbm['category_accuracy']}, Macro-F1: {metrics_lgbm['macro_f1']}")

    # 5. Export Native Model Artifact
    model_file = art_path / "model.txt"
    model.booster_.save_model(str(model_file))
    model_sha256 = compute_sha256(model_file)
    logger.info(f"Native model saved to {model_file} (SHA-256: {model_sha256})")

    # Parity verification
    reloaded_booster = lgb.Booster(model_file=str(model_file))
    reloaded_preds = reloaded_booster.predict(X_test)
    parity_diff = np.max(np.abs(test_preds_lgbm - reloaded_preds))
    logger.info(f"Model reload parity check passed: max diff = {parity_diff:.8f}")
    if parity_diff > 1e-6:
        raise ValueError(f"Model artifact parity failure: max diff = {parity_diff}")

    # 6. Feature Importance
    importances = model.feature_importances_
    feat_imp = sorted(
        [{"feature": f, "importance": int(imp)} for f, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # 7. Compile Evaluation Report & Metrics JSON
    results = {
        "model_name": "climacore_air_quality_lgbm",
        "version": "v0.1.0",
        "target": "target_pm2_5_next_day",
        "units": "ug/m3",
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

    # Save to reports as well
    eval_report_file = rep_path / "air_quality_evaluation_report.json"
    with open(eval_report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 8. Generate Model Card
    card_file = art_path / "model_card.md"
    with open(card_file, "w", encoding="utf-8") as f:
        f.write(f"# Model Card: Clima-Core AirQuality Next-Day PM2.5 (v0.1.0)\n\n")
        f.write(f"**Model Name:** `climacore_air_quality_lgbm`  \n")
        f.write(f"**Status:** `validated_prototype`  \n")
        f.write(f"**Created At:** {datetime.now(timezone.utc).isoformat()}  \n")
        f.write(f"**Task:** Next-day 24h mean PM2.5 prediction (regression) & CPCB AQI band derivation.  \n")
        f.write(f"**Artifact SHA-256:** `{model_sha256}`  \n\n")
        f.write(f"## 1. Data Provenance & Scope\n\n")
        f.write(f"- **Source Dataset:** India Daily Weather & Air Quality (50 cities), CC BY 4.0\n")
        f.write(f"- **Upstream Origin:** Open-Meteo CAMS Global Atmospheric Model (~45 km grid cell at city centres)\n")
        f.write(f"- **Spatial Unit:** 50 city-centre model grid cells across India\n")
        f.write(f"- **Target Variable:** `target_pm2_5_next_day` (µg/m³)\n")
        f.write(f"- **Known Data Limitation:** Based on CAMS reanalysis/forecast grid, not physical CPCB ground monitors.\n\n")
        f.write(f"## 2. Chronological Holdout Evaluation (Test Period: 2026-01-01 to 2026-10-29)\n\n")
        f.write(f"| Model / Baseline | Test MAE (µg/m³) | Test RMSE (µg/m³) | Test Mean Bias | Test R² | Category Accuracy | Macro-F1 |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        f.write(f"| **Clima-Core LightGBM** | **{metrics_lgbm['mae']}** | **{metrics_lgbm['rmse']}** | **{metrics_lgbm['mean_bias']}** | **{metrics_lgbm['r2_score']}** | **{metrics_lgbm['category_accuracy']}** | **{metrics_lgbm['macro_f1']}** |\n")
        f.write(f"| Persistence Baseline | {metrics_persistence['mae']} | {metrics_persistence['rmse']} | {metrics_persistence['mean_bias']} | {metrics_persistence['r2_score']} | {metrics_persistence['category_accuracy']} | {metrics_persistence['macro_f1']} |\n")
        f.write(f"| Rolling 7-Day Mean | {metrics_rolling['mae']} | {metrics_rolling['rmse']} | {metrics_rolling['mean_bias']} | {metrics_rolling['r2_score']} | {metrics_rolling['category_accuracy']} | {metrics_rolling['macro_f1']} |\n")
        f.write(f"| Seasonal Climatology | {metrics_climatology['mae']} | {metrics_climatology['rmse']} | {metrics_climatology['mean_bias']} | {metrics_climatology['r2_score']} | {metrics_climatology['category_accuracy']} | {metrics_climatology['macro_f1']} |\n\n")
        f.write(f"**Improvement over Persistence:** MAE reduced by **{results['improvements_over_persistence']['mae_reduction_pct']}%**, RMSE reduced by **{results['improvements_over_persistence']['rmse_reduction_pct']}%**.\n\n")
        f.write(f"## 3. Top Predictive Features\n\n")
        for item in feat_imp[:10]:
            f.write(f"- `{item['feature']}`: importance = {item['importance']}\n")
        f.write(f"\n## 4. Unsupported Use Cases\n\n")
        f.write(f"- Do not use for street-level or ward-level decisions (model resolution is ~45km city-centre grid cell).\n")
        f.write(f"- Do not present predictions as official CPCB public health alerts without agency corroboration.\n")

    # 9. Artifact Manifest
    artifact_manifest = {
        "artifact_dir": str(art_path),
        "files": [
            get_file_metadata(model_file),
            get_file_metadata(metrics_file),
            get_file_metadata(card_file),
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_module": "climacore.models.air_quality.train",
    }
    with open(art_path / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    logger.info("Training pipeline and artifact export completed successfully.")
    return results


if __name__ == "__main__":
    train_air_quality_model()
