"""
Training pipeline for RainRisk two-part LightGBM model.
Implements:
1. Part 1: LightGBMClassifier for rain occurrence (P(Rain >= 1.0mm))
2. Part 2: LightGBMRegressor for conditional precipitation volume (mm)
3. Combined Two-Part Hurdle Model: E[Y] = P(Rain) * E[Y|Rain]
4. Direct Single Regressor Benchmark (trained on all days)
5. Comprehensive evaluation against Persistence and Seasonal Climatology baselines.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root
from climacore.evaluation.splits import chronological_train_val_test_split

logger = get_logger("climacore.models.rain_risk.train")


def train_rain_risk_models(
    data_path: Path | str = None,
    artifacts_dir: Path | str = None,
    reports_dir: Path | str = None,
    val_start: str = "2025-01-01",
    test_start: str = "2026-01-01",
) -> Dict[str, Any]:
    """Train and evaluate two-part RainRisk model against direct regressor and baselines."""
    root = get_project_root()
    parquet_file = Path(data_path) if data_path else root / "data" / "processed" / "rain_risk_train_ready.parquet"
    art_path = Path(artifacts_dir) if artifacts_dir else root / "artifacts" / "models" / "rain_risk" / "v0.1.0"
    rep_path = Path(reports_dir) if reports_dir else root / "reports" / "evaluation"

    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading RainRisk dataset from {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    # 1. Chronological Split
    train_df, val_df, test_df = chronological_train_val_test_split(
        df, date_col="date", val_start=val_start, test_start=test_start
    )

    logger.info(
        f"RainRisk Chronological Split: "
        f"Train={len(train_df):,} rows ({train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d}), "
        f"Val={len(val_df):,} rows ({val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d}), "
        f"Test={len(test_df):,} rows ({test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})"
    )

    # Load feature metadata
    schema_file = root / "data" / "processed" / "rain_risk_feature_schema.json"
    with open(schema_file, "r", encoding="utf-8") as f:
        schema_meta = json.load(f)
    feature_cols = schema_meta["feature_columns"]
    target_amount_col = schema_meta["target_amount_column"]
    target_occ_col = schema_meta["target_occurrence_column"]

    X_train = train_df[feature_cols]
    y_train_amount = train_df[target_amount_col].to_numpy()
    y_train_occ = train_df[target_occ_col].to_numpy()

    X_val = val_df[feature_cols]
    y_val_amount = val_df[target_amount_col].to_numpy()
    y_val_occ = val_df[target_occ_col].to_numpy()

    X_test = test_df[feature_cols]
    y_test_amount = test_df[target_amount_col].to_numpy()
    y_test_occ = test_df[target_occ_col].to_numpy()

    # 2. Fit and Evaluate Baselines
    logger.info("Computing Persistence and Climatology baselines on test set...")
    # Persistence
    test_pred_persist_amount = test_df["precipitation_sum"].to_numpy()
    test_pred_persist_occ = (test_df["precipitation_sum"] >= 1.0).astype(int).to_numpy()

    # Seasonal Monthly Climatology lookup fitted on training data
    clim_occ_lookup = train_df.groupby(["city", "month"])[target_occ_col].mean().to_dict()
    clim_amount_lookup = train_df.groupby(["city", "month"])[target_amount_col].mean().to_dict()

    global_occ_mean = float(y_train_occ.mean())
    global_amount_mean = float(y_train_amount.mean())

    test_pred_clim_prob = np.array([
        clim_occ_lookup.get((r["city"], r["month"]), global_occ_mean) for _, r in test_df.iterrows()
    ])
    test_pred_clim_amount = np.array([
        clim_amount_lookup.get((r["city"], r["month"]), global_amount_mean) for _, r in test_df.iterrows()
    ])

    # 3. Train Part 1: Classifier for Rain Occurrence (P(Rain >= 1.0mm))
    logger.info("Training Part 1: LightGBM Classifier for Rain Occurrence...")
    clf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
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

    classifier = lgb.LGBMClassifier(**clf_params)
    classifier.fit(
        X_train,
        y_train_occ,
        eval_set=[(X_val, y_val_occ)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    clf_best_iter = int(classifier.best_iteration_)
    logger.info(f"Part 1 Classifier converged at iteration {clf_best_iter}.")

    # Classifier predictions (probabilities)
    test_pred_prob = classifier.predict_proba(X_test)[:, 1]
    test_pred_binary = (test_pred_prob >= 0.5).astype(int)

    # 4. Train Part 2: Regressor for Conditional Amount
    # Conditional training: fit on rainy days (precipitation >= 1.0 mm)
    logger.info("Training Part 2: Conditional LightGBM Regressor (trained on rain days)...")
    train_wet_mask = y_train_amount >= 1.0
    val_wet_mask = y_val_amount >= 1.0

    X_train_wet = X_train[train_wet_mask]
    y_train_wet = y_train_amount[train_wet_mask]
    X_val_wet = X_val[val_wet_mask]
    y_val_wet = y_val_amount[val_wet_mask]

    reg_cond_params = {
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

    cond_regressor = lgb.LGBMRegressor(**reg_cond_params)
    cond_regressor.fit(
        X_train_wet,
        y_train_wet,
        eval_set=[(X_val_wet, y_val_wet)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    cond_reg_best_iter = int(cond_regressor.best_iteration_)
    logger.info(f"Part 2 Conditional Regressor converged at iteration {cond_reg_best_iter}.")

    # Conditional amount predictions
    test_pred_cond_amount = np.maximum(0.0, cond_regressor.predict(X_test))

    # 5. Combined Two-Part Model
    # E[Y] = P(Rain >= 1.0mm) * E[Y | Rain >= 1.0mm]
    test_pred_two_part = test_pred_prob * test_pred_cond_amount

    # 6. Benchmark: Direct Single Regressor (trained on all days)
    logger.info("Training Benchmark: Direct Single Regressor on all days...")
    direct_reg_params = {
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

    direct_regressor = lgb.LGBMRegressor(**direct_reg_params)
    direct_regressor.fit(
        X_train,
        y_train_amount,
        eval_set=[(X_val, y_val_amount)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    direct_best_iter = int(direct_regressor.best_iteration_)
    test_pred_direct = np.maximum(0.0, direct_regressor.predict(X_test))

    # 7. Comprehensive Evaluation Metrics
    logger.info("Computing metrics across all models...")

    # Classification Metrics (Rain Occurrence)
    clf_metrics = {
        "accuracy": round(float(accuracy_score(y_test_occ, test_pred_binary)), 4),
        "precision": round(float(precision_score(y_test_occ, test_pred_binary, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_occ, test_pred_binary, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test_occ, test_pred_binary, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test_occ, test_pred_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test_occ, test_pred_prob)), 4),
        "brier_score": round(float(brier_score_loss(y_test_occ, test_pred_prob)), 4),
    }

    clim_brier = round(float(brier_score_loss(y_test_occ, test_pred_clim_prob)), 4)
    persist_brier = round(float(brier_score_loss(y_test_occ, test_pred_persist_occ)), 4)

    # Regression Metrics (Precipitation Volume in mm)
    def calc_reg_metrics(y_true, y_pred):
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
            "mean_bias": round(float(np.mean(y_pred - y_true)), 4),
            "r2_score": round(float(r2_score(y_true, y_pred)), 4),
        }

    metrics_two_part = calc_reg_metrics(y_test_amount, test_pred_two_part)
    metrics_direct = calc_reg_metrics(y_test_amount, test_pred_direct)
    metrics_persist = calc_reg_metrics(y_test_amount, test_pred_persist_amount)
    metrics_clim = calc_reg_metrics(y_test_amount, test_pred_clim_amount)

    # Heavy Rain Event Recall (>= 15.6 mm)
    heavy_true = y_test_amount >= 15.6
    heavy_pred_two_part = test_pred_two_part >= 15.6
    heavy_pred_direct = test_pred_direct >= 15.6

    heavy_metrics = {
        "true_heavy_count": int(np.sum(heavy_true)),
        "two_part_heavy_recall": round(float(np.sum(heavy_true & heavy_pred_two_part) / max(1, np.sum(heavy_true))), 4),
        "two_part_heavy_precision": round(float(np.sum(heavy_true & heavy_pred_two_part) / max(1, np.sum(heavy_pred_two_part))), 4),
        "direct_heavy_recall": round(float(np.sum(heavy_true & heavy_pred_direct) / max(1, np.sum(heavy_true))), 4),
        "direct_heavy_precision": round(float(np.sum(heavy_true & heavy_pred_direct) / max(1, np.sum(heavy_pred_direct))), 4),
    }

    logger.info(f"Rain Occurrence Classifier PR-AUC: {clf_metrics['pr_auc']}, ROC-AUC: {clf_metrics['roc_auc']}, Brier: {clf_metrics['brier_score']}")
    logger.info(f"Two-Part Model Test MAE: {metrics_two_part['mae']} mm, RMSE: {metrics_two_part['rmse']} mm, R2: {metrics_two_part['r2_score']}")
    logger.info(f"Direct Regressor Test MAE: {metrics_direct['mae']} mm, RMSE: {metrics_direct['rmse']} mm, R2: {metrics_direct['r2_score']}")
    logger.info(f"Persistence Test MAE: {metrics_persist['mae']} mm, RMSE: {metrics_persist['rmse']} mm")
    logger.info(f"Climatology Test MAE: {metrics_clim['mae']} mm, RMSE: {metrics_clim['rmse']} mm")

    # 8. Export Native Artifacts
    clf_file = art_path / "classifier_model.txt"
    cond_reg_file = art_path / "conditional_regressor_model.txt"
    direct_reg_file = art_path / "direct_regressor_model.txt"

    classifier.booster_.save_model(str(clf_file))
    cond_regressor.booster_.save_model(str(cond_reg_file))
    direct_regressor.booster_.save_model(str(direct_reg_file))

    clf_sha256 = compute_sha256(clf_file)
    cond_reg_sha256 = compute_sha256(cond_reg_file)
    direct_reg_sha256 = compute_sha256(direct_reg_file)

    # Parity verification
    reloaded_clf = lgb.Booster(model_file=str(clf_file))
    reloaded_prob = reloaded_clf.predict(X_test)
    clf_diff = np.max(np.abs(test_pred_prob - reloaded_prob))
    logger.info(f"Part 1 Classifier reload parity: max diff = {clf_diff:.8f}")

    reloaded_reg = lgb.Booster(model_file=str(cond_reg_file))
    reloaded_cond_amount = np.maximum(0.0, reloaded_reg.predict(X_test))
    reloaded_two_part = reloaded_prob * reloaded_cond_amount
    parity_diff = np.max(np.abs(test_pred_two_part - reloaded_two_part))
    logger.info(f"Two-part combined reload parity: max diff = {parity_diff:.8f}")
    if parity_diff > 1e-6:
        raise ValueError(f"RainRisk parity failure: max diff = {parity_diff}")

    # Top Features for Classifier & Regressor
    clf_imp = sorted(
        [{"feature": f, "importance": int(imp)} for f, imp in zip(feature_cols, classifier.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # 9. Compile Results and Export
    results = {
        "model_name": "climacore_rain_risk_two_part",
        "version": "v0.1.0",
        "target": "target_precipitation_next_day",
        "units": "mm",
        "split_definition": {
            "train": f"{train_df['date'].min():%Y-%m-%d} to {train_df['date'].max():%Y-%m-%d} ({len(train_df):,} rows)",
            "validation": f"{val_df['date'].min():%Y-%m-%d} to {val_df['date'].max():%Y-%m-%d} ({len(val_df):,} rows)",
            "test": f"{test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d} ({len(test_df):,} rows)",
        },
        "models_summary": {
            "classifier_best_iteration": clf_best_iter,
            "conditional_regressor_best_iteration": cond_reg_best_iter,
            "direct_regressor_best_iteration": direct_best_iter,
        },
        "classification_metrics": {
            "lightgbm_classifier": clf_metrics,
            "persistence_brier_score": persist_brier,
            "climatology_brier_score": clim_brier,
        },
        "regression_metrics": {
            "two_part_hurdle_test": metrics_two_part,
            "direct_single_regressor_test": metrics_direct,
            "persistence_test": metrics_persist,
            "climatology_test": metrics_clim,
        },
        "heavy_rain_event_metrics": heavy_metrics,
        "top_predictive_features": clf_imp[:15],
    }

    metrics_file = art_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    eval_report_file = rep_path / "rain_risk_evaluation_report.json"
    with open(eval_report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 10. Generate Model Card
    card_file = art_path / "model_card.md"
    with open(card_file, "w", encoding="utf-8") as f:
        f.write(f"# Model Card: Clima-Core RainRisk Two-Part Model (v0.1.0)\n\n")
        f.write(f"**Model Name:** `climacore_rain_risk_two_part`  \n")
        f.write(f"**Status:** `validated_prototype`  \n")
        f.write(f"**Created At:** {datetime.now(timezone.utc).isoformat()}  \n")
        f.write(f"**Architecture:** Two-part Hurdle Model: Part 1 Classifier (P(Rain >= 1mm)) + Part 2 Conditional Regressor (Volume in mm).  \n\n")
        f.write(f"## 1. Artifact Checksums\n\n")
        f.write(f"- `classifier_model.txt`: SHA-256 `{clf_sha256}`\n")
        f.write(f"- `conditional_regressor_model.txt`: SHA-256 `{cond_reg_sha256}`\n")
        f.write(f"- `direct_regressor_model.txt`: SHA-256 `{direct_reg_sha256}`\n\n")
        f.write(f"## 2. Chronological Holdout Evaluation (Test Period: 2026-01-01 to 2026-10-29)\n\n")
        f.write(f"### A. Amount Regression Benchmark\n\n")
        f.write(f"| Model / Baseline | Test MAE (mm) | Test RMSE (mm) | Test Mean Bias (mm) | Test R² |\n")
        f.write(f"|---|---|---|---|---|\n")
        f.write(f"| **Two-Part Hurdle Model** | **{metrics_two_part['mae']}** | **{metrics_two_part['rmse']}** | **{metrics_two_part['mean_bias']}** | **{metrics_two_part['r2_score']}** |\n")
        f.write(f"| Direct Single Regressor | {metrics_direct['mae']} | {metrics_direct['rmse']} | {metrics_direct['mean_bias']} | {metrics_direct['r2_score']} |\n")
        f.write(f"| Persistence Baseline | {metrics_persist['mae']} | {metrics_persist['rmse']} | {metrics_persist['mean_bias']} | {metrics_persist['r2_score']} |\n")
        f.write(f"| Seasonal Climatology | {metrics_clim['mae']} | {metrics_clim['rmse']} | {metrics_clim['mean_bias']} | {metrics_clim['r2_score']} |\n\n")
        f.write(f"### B. Rain Occurrence Probability (P(Rain >= 1.0mm))\n\n")
        f.write(f"- **ROC-AUC:** {clf_metrics['roc_auc']}\n")
        f.write(f"- **PR-AUC:** {clf_metrics['pr_auc']}\n")
        f.write(f"- **Brier Score:** {clf_metrics['brier_score']} (vs Climatology: {clim_brier}, vs Persistence: {persist_brier})\n")
        f.write(f"- **Accuracy:** {clf_metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"- **F1-Score:** {clf_metrics['f1_score']}\n\n")
        f.write(f"## 3. Top Predictive Features\n\n")
        for item in clf_imp[:10]:
            f.write(f"- `{item['feature']}`: importance = {item['importance']}\n")
        f.write(f"\n## 4. Unsupported Use Cases\n\n")
        f.write(f"- Do not use for street-level or localized flash-flood warnings.\n")
        f.write(f"- Model estimates daily rainfall at ~9-25km city-centre grid resolution.\n")

    # 11. Artifact Manifest
    artifact_manifest = {
        "artifact_dir": str(art_path),
        "files": [
            get_file_metadata(clf_file),
            get_file_metadata(cond_reg_file),
            get_file_metadata(direct_reg_file),
            get_file_metadata(metrics_file),
            get_file_metadata(card_file),
        ],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_module": "climacore.models.rain_risk.train",
    }
    with open(art_path / "artifact_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifact_manifest, f, indent=2)

    logger.info("RainRisk training, evaluation, and export completed successfully.")
    return results


if __name__ == "__main__":
    train_rain_risk_models()
