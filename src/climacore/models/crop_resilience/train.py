"""
AgriSense Agro-Climatic Crop Suitability & Climate Resilience Model.
Trained on 34,903 historical multi-season agricultural district records across India (1997-2015).
Predicts agro-climatically optimal crop suitability given 8 soil physicochemical properties
and 11 seasonal climate variables across 10 dominant Indian staple and commercial crops.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.models.crop_resilience.train")

CROP_SOIL_FEATURES: List[str] = [
    "soil_ph_h2o_0_30cm",
    "soil_total_n_g_kg_0_30cm",
    "soil_soc_g_kg_0_30cm",
    "soil_cec_cmol_kg_0_30cm",
    "soil_clay_pct_0_30cm",
    "soil_sand_pct_0_30cm",
    "soil_silt_pct_0_30cm",
    "soil_bulk_density_kg_dm3_0_30cm",
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "humidity_pct",
    "rainfall_mm",
    "rain_days_proxy_mm_per_day",
    "solar_mj_m2_day",
    "annual_rainfall_mm",
    "annual_temp_mean_c",
    "annual_humidity_pct",
    "elevation_m",
]

TARGET_CROPS: List[str] = [
    "Rice",
    "Wheat",
    "Sugarcane",
    "Maize",
    "Potato",
    "Rapeseed &Mustard",
    "Groundnut",
    "Bajra",
    "Gram",
    "Jowar",
]


def load_crop_dataset(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and prepare agricultural training data."""
    if raw_path is None:
        root = get_project_root()
        raw_path = root / "data" / "raw" / "agriculture" / "india_district_crop_dataset_1997_2015.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"Crop dataset not found at {raw_path}")

    logger.info(f"Loading crop dataset from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)
    # Filter to top crops
    sub_df = df[df["dominant_crop"].isin(TARGET_CROPS)].copy()
    return sub_df


def train_crop_suitability_model(
    raw_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train and evaluate the LightGBM Agro-Climatic Crop Suitability Model."""
    root = get_project_root()
    if output_dir is None:
        output_dir = root / "artifacts" / "models" / "crop_resilience" / "v0.1.0"
    if report_path is None:
        report_path = root / "reports" / "evaluation" / "crop_resilience_evaluation_report.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_crop_dataset(raw_path)

    # Encode target crops
    cats = pd.CategoricalDtype(categories=TARGET_CROPS, ordered=True)
    df["crop_label"] = df["dominant_crop"].astype(cats).cat.codes

    # Temporal split: 1997-2011 Train, 2012-2013 Val, 2014-2015 Test
    train_mask = df["crop_year"] <= 2011
    val_mask = (df["crop_year"] >= 2012) & (df["crop_year"] <= 2013)
    test_mask = df["crop_year"] >= 2014

    X_train, y_train = df.loc[train_mask, CROP_SOIL_FEATURES], df.loc[train_mask, "crop_label"]
    X_val, y_val = df.loc[val_mask, CROP_SOIL_FEATURES], df.loc[val_mask, "crop_label"]
    X_test, y_test = df.loc[test_mask, CROP_SOIL_FEATURES], df.loc[test_mask, "crop_label"]

    logger.info(f"Training split: {len(X_train):,} samples across {len(TARGET_CROPS)} crop classes")
    logger.info(f"Validation split: {len(X_val):,} samples")
    logger.info(f"Test split: {len(X_test):,} samples")

    params = {
        "objective": "multiclass",
        "num_class": len(TARGET_CROPS),
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": 4,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    # Evaluate on holdout test set (2014-2015)
    test_probs = model.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)

    top1_acc = float(accuracy_score(y_test, test_preds))
    top3_correct = sum(y_test.iloc[i] in np.argsort(test_probs[i])[-3:] for i in range(len(y_test)))
    top3_acc = float(top3_correct / len(y_test))
    weighted_f1 = float(f1_score(y_test, test_preds, average="weighted", zero_division=0))
    weighted_precision = float(precision_score(y_test, test_preds, average="weighted", zero_division=0))
    weighted_recall = float(recall_score(y_test, test_preds, average="weighted", zero_division=0))

    # Baseline: Majority class (Rice)
    majority_class_code = int(y_train.mode()[0])
    majority_acc = float(accuracy_score(y_test, np.full(len(y_test), majority_class_code)))

    feature_importances = dict(zip(CROP_SOIL_FEATURES, [int(x) for x in model.feature_importances_]))
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

    # Save booster
    booster_path = output_dir / "crop_suitability_v1.booster"
    model.booster_.save_model(str(booster_path))
    logger.info(f"Saved trained booster to {booster_path}")

    # Compute hash
    hasher = hashlib.sha256()
    with open(booster_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    model_sha256 = hasher.hexdigest()

    metadata = {
        "model_name": "AgriSense Agro-Climatic Crop Suitability Model",
        "version": "v0.1.0",
        "architecture": "LightGBM Multi-class GBDT (10 Indian Crop Classes)",
        "sha256": model_sha256,
        "target_crops": TARGET_CROPS,
        "training_period": "1997-2011",
        "validation_period": "2012-2013",
        "test_period": "2014-2015",
        "total_training_samples": len(X_train),
        "total_test_samples": len(X_test),
        "features": CROP_SOIL_FEATURES,
        "feature_importances": dict(sorted_features),
        "metrics": {
            "top1_test_accuracy": round(top1_acc, 4),
            "top3_test_accuracy": round(top3_acc, 4),
            "weighted_f1": round(weighted_f1, 4),
            "weighted_precision": round(weighted_precision, 4),
            "weighted_recall": round(weighted_recall, 4),
        },
        "baseline_comparison": {
            "majority_class_name": TARGET_CROPS[majority_class_code],
            "majority_class_accuracy": round(majority_acc, 4),
            "accuracy_lift_pct": round(((top1_acc - majority_acc) / majority_acc) * 100, 2),
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved to {metadata_path}")
    logger.info(f"Evaluation report saved to {report_path}")

    return metadata


if __name__ == "__main__":
    rep = train_crop_suitability_model()
    print("Crop Resilience & Suitability Model Training Completed Successfully!")
    print(f"Top-1 Test Accuracy: {rep['metrics']['top1_test_accuracy']:.2%} (Lift: +{rep['baseline_comparison']['accuracy_lift_pct']}%)")
    print(f"Top-3 Test Accuracy: {rep['metrics']['top3_test_accuracy']:.2%}")
    print(f"Weighted F1:         {rep['metrics']['weighted_f1']}")
