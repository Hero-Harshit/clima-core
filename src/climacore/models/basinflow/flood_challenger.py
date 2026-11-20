"""BasinFlow FloodRisk Challenger Model Training & Evaluation.
Combines SCS Unit Hydrograph peak flow (qp), time-to-peak (tp), HydroRIVERS catchment geometry,
and Muskingum-Cunge routed river flow with antecedent soil moisture and infrastructure exposure.
"""

import os
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from climacore.models.basinflow.unit_hydrograph import SCSUnitHydrograph
from climacore.models.basinflow.muskingum_cunge import MuskingumCungeRouter


BASE_20_FEATURES: List[str] = [
    "rainfall_mm",
    "rain_rolling_3d",
    "rain_rolling_7d",
    "rain_rolling_14d",
    "rain_rolling_30d",
    "rain_max_3d",
    "api_14d",
    "precipitation_hours",
    "et0_fao_evapotranspiration",
    "wind_gusts_10m_max",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "topsoil_moisture_anomaly",
    "rootzone_moisture_anomaly",
    "soil_moisture_deficit_index",
    "drainage_canals_15km",
    "bridges_15km",
    "transit_stations_15km",
    "population_15km",
    "dense_cluster_pop_15km",
]

SCS_3_FEATURES: List[str] = [
    "scs_runoff_depth_mm",
    "scs_runoff_coefficient",
    "scs_potential_retention_s",
]

BASINFLOW_FEATURES: List[str] = [
    "hydrograph_peak_discharge_m3s",
    "hydrograph_time_to_peak_hours",
    "routed_river_peak_m3s",
    "upstream_catchment_sqkm",
    "river_reach_length_km",
    "river_strahler_order",
    "upstream_accum_runoff_m3",
]

ALL_BASINFLOW_FEATURES: List[str] = BASE_20_FEATURES + SCS_3_FEATURES + BASINFLOW_FEATURES


def compute_basinflow_features(df: pd.DataFrame, city_profiles_path: str) -> pd.DataFrame:
    """Vectorially enriches dataset with SCS-CN runoff, Unit Hydrograph, and HydroRIVERS metrics."""
    with open(city_profiles_path, "r") as f:
        profiles = json.load(f)

    # 1. Base SCS-CN calculations
    p = df["rainfall_mm"].values.astype(np.float32)
    # Dynamic CN approximation based on 5-day antecedent precipitation
    p5 = df["rain_rolling_7d"].values.astype(np.float32) * (5.0 / 7.0)
    
    # Standard urban composite CN_II ~ 80
    cn_ii = 80.0
    s_ii = (25400.0 / cn_ii) - 254.0  # ~63.5 mm
    
    # Adjust S for AMC condition: AMC-I (dry, P5 < 13mm), AMC-III (wet, P5 > 28mm)
    s_amc = np.where(p5 < 13.0, s_ii * 2.28, np.where(p5 > 28.0, s_ii * 0.45, s_ii)).astype(np.float32)
    ia = 0.20 * s_amc
    
    excess_p = np.maximum(p - ia, 0.0)
    q_depth = np.where(excess_p > 0.0, (excess_p ** 2) / (excess_p + s_amc), 0.0).astype(np.float32)
    c_coeff = np.where(p > 0.0, q_depth / np.maximum(p, 1e-3), 0.0).astype(np.float32)
    
    df["scs_runoff_depth_mm"] = q_depth
    df["scs_runoff_coefficient"] = c_coeff
    df["scs_potential_retention_s"] = s_amc

    # 2. Extract city HydroRIVERS metrics
    uh = SCSUnitHydrograph()
    router = MuskingumCungeRouter()

    # Precalculate per-city hydraulic constants
    city_constants = {}
    for city_name, prof in profiles.items():
        maj_riv = prof.get("nearest_major_river") or prof.get("nearest_river") or {}
        area = float(maj_riv.get("upstream_catchment_sqkm", 50.0))
        reach_len = float(maj_riv.get("reach_length_km", 10.0))
        order = float(maj_riv.get("strahler_order", 3.0))
        
        tc = uh.calculate_time_of_concentration(hydraulic_length_km=reach_len, catchment_slope_percent=1.5, cn=cn_ii)
        tp = (1.0 / 2.0) + 0.6 * tc
        
        # Test unit hydrograph to get reach attenuation factor
        test_uh = uh.generate_hydrograph(catchment_area_sqkm=area, runoff_depth_mm=10.0, hydraulic_length_km=reach_len, tc_hours=tc)
        route_test = router.route_hydrograph(test_uh["discharge_m3s"], test_uh["time_hours"], reach_length_km=reach_len)
        attenuation = float(route_test.get("peak_attenuation_ratio", 0.95))
        
        city_constants[city_name.lower()] = {
            "area": area,
            "reach_len": reach_len,
            "order": order,
            "tc": tc,
            "tp": tp,
            "attenuation": attenuation,
            "qp_per_mm": (0.208 * area) / tp,
        }

    # Map onto dataframe
    city_series = df["city"].str.lower()
    default_const = city_constants.get("delhi", {"area": 50.0, "reach_len": 10.0, "order": 3.0, "tc": 2.0, "tp": 1.7, "attenuation": 0.95, "qp_per_mm": 6.1})

    areas = np.array([city_constants.get(c, default_const)["area"] for c in city_series], dtype=np.float32)
    reach_lens = np.array([city_constants.get(c, default_const)["reach_len"] for c in city_series], dtype=np.float32)
    orders = np.array([city_constants.get(c, default_const)["order"] for c in city_series], dtype=np.float32)
    tps = np.array([city_constants.get(c, default_const)["tp"] for c in city_series], dtype=np.float32)
    attenuations = np.array([city_constants.get(c, default_const)["attenuation"] for c in city_series], dtype=np.float32)
    qp_factors = np.array([city_constants.get(c, default_const)["qp_per_mm"] for c in city_series], dtype=np.float32)

    df["hydrograph_peak_discharge_m3s"] = (qp_factors * q_depth).astype(np.float32)
    df["hydrograph_time_to_peak_hours"] = tps
    df["routed_river_peak_m3s"] = (df["hydrograph_peak_discharge_m3s"].values * attenuations).astype(np.float32)
    df["upstream_catchment_sqkm"] = areas
    df["river_reach_length_km"] = reach_lens
    df["river_strahler_order"] = orders
    df["upstream_accum_runoff_m3"] = (areas * 1e6 * (q_depth / 1000.0)).astype(np.float32)

    return df


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    """Computes 95% bootstrap confidence intervals for ROC-AUC and PR-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc_list = []
    pr_list = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        if np.sum(yt_sample) > 0 and np.sum(yt_sample) < len(yt_sample):
            roc_list.append(roc_auc_score(yt_sample, y_prob[idx]))
            pr_list.append(average_precision_score(yt_sample, y_prob[idx]))
    return {
        "roc_auc_95ci": (round(float(np.percentile(roc_list, 2.5)), 4), round(float(np.percentile(roc_list, 97.5)), 4)),
        "pr_auc_95ci": (round(float(np.percentile(pr_list, 2.5)), 4), round(float(np.percentile(pr_list, 97.5)), 4)),
    }


def train_and_evaluate_basinflow():
    data_path = "data/processed/floods/city_flood_history_aligned.parquet"
    profiles_path = "data/processed/city_hydrorivers_profiles.json"
    
    print("Loading aligned flood dataset...")
    df = pd.read_parquet(data_path)
    
    print("Computing BasinFlow unit hydrograph and HydroRIVERS reach features...")
    df = compute_basinflow_features(df, profiles_path)
    
    train_mask = df["date"] <= "2018-12-31"
    val_mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2021-12-31")
    test_mask = df["date"] >= "2022-01-01"
    
    target_col = "flood_disaster_event"
    y_test = df.loc[test_mask, target_col].values
    
    print(f"Data splits: Train={train_mask.sum():,}, Val={val_mask.sum():,}, Test={test_mask.sum():,}")
    print(f"Test positive events: {int(y_test.sum())} / {len(y_test)}")

    # 1. Evaluate Existing SCS-CN Champion (23 features)
    champion_path = "artifacts/models/flood_risk/v0.1.0/flood_risk_challenger_scs_cn.booster"
    champ_booster = lgb.Booster(model_file=champion_path)
    champ_prob = champ_booster.predict(df.loc[test_mask, BASE_20_FEATURES + SCS_3_FEATURES])
    
    champ_roc = float(roc_auc_score(y_test, champ_prob))
    champ_pr = float(average_precision_score(y_test, champ_prob))
    champ_brier = float(brier_score_loss(y_test, champ_prob))
    champ_ci = bootstrap_ci(y_test, champ_prob)
    
    print(f"\nSCS-CN Champion (23 feats): ROC-AUC = {champ_roc:.4f}, PR-AUC = {champ_pr:.4f}, Brier = {champ_brier:.4f}")

    # 2. Train BasinFlow Challenger (30 features: Base 20 + SCS 3 + BasinFlow 7)
    print("\nTraining BasinFlow FloodRisk Challenger (30 features)...")
    X_train = df.loc[train_mask, ALL_BASINFLOW_FEATURES]
    y_train = df.loc[train_mask, target_col]
    X_val = df.loc[val_mask, ALL_BASINFLOW_FEATURES]
    y_val = df.loc[val_mask, target_col]
    X_test = df.loc[test_mask, ALL_BASINFLOW_FEATURES]

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": 250,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": 2,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    challenger_prob = model.predict_proba(X_test)[:, 1]
    challenger_roc = float(roc_auc_score(y_test, challenger_prob))
    challenger_pr = float(average_precision_score(y_test, challenger_prob))
    challenger_brier = float(brier_score_loss(y_test, challenger_prob))
    challenger_ci = bootstrap_ci(y_test, challenger_prob)

    # Decision threshold 0.35
    pred_calib = (challenger_prob >= 0.35).astype(int)
    challenger_f1 = float(f1_score(y_test, pred_calib))
    challenger_prec = float(precision_score(y_test, pred_calib, zero_division=0))
    challenger_rec = float(recall_score(y_test, pred_calib))

    print(f"BasinFlow Challenger (30 feats): ROC-AUC = {challenger_roc:.4f}, PR-AUC = {challenger_pr:.4f}, Brier = {challenger_brier:.4f}")
    print(f"  F1 = {challenger_f1:.4f}, Precision = {challenger_prec:.4f}, Recall = {challenger_rec:.4f}")

    # Feature importances
    feat_imps = dict(zip(ALL_BASINFLOW_FEATURES, [int(x) for x in model.feature_importances_]))
    sorted_imps = sorted(feat_imps.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Feature Importances in BasinFlow Challenger:")
    for f_name, imp in sorted_imps[:10]:
        print(f"  {f_name:32s}: {imp}")

    # Save model artifact
    out_dir = "artifacts/models/v1_3/basinflow"
    os.makedirs(out_dir, exist_ok=True)
    booster_path = os.path.join(out_dir, "flood_risk_basinflow_challenger.booster")
    model.booster_.save_model(booster_path)

    # Compute sha256
    with open(booster_path, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()

    results = {
        "model_name": "BasinFlow FloodRisk Challenger V1",
        "features_count": len(ALL_BASINFLOW_FEATURES),
        "sha256": sha256_hash,
        "evaluation_period": "2022-2023",
        "test_samples": int(len(y_test)),
        "test_events": int(y_test.sum()),
        "champion_scs_cn": {
            "roc_auc": round(champ_roc, 4),
            "pr_auc": round(champ_pr, 4),
            "brier_score": round(champ_brier, 4),
            "ci_95": champ_ci,
        },
        "basinflow_challenger": {
            "roc_auc": round(challenger_roc, 4),
            "pr_auc": round(challenger_pr, 4),
            "brier_score": round(challenger_brier, 4),
            "f1_score": round(challenger_f1, 4),
            "precision": round(challenger_prec, 4),
            "recall": round(challenger_rec, 4),
            "ci_95": challenger_ci,
        },
        "delta": {
            "roc_auc_diff": round(challenger_roc - champ_roc, 4),
            "pr_auc_diff": round(challenger_pr - champ_pr, 4),
            "brier_diff": round(challenger_brier - champ_brier, 4),
        },
        "top_feature_importances": dict(sorted_imps[:15]),
        "gnn_blocker_declaration": {
            "status": "BLOCKED_AWAITING_CALIBRATED_DISCHARGE_SENSORS",
            "reason": (
                "Graph Neural Network (GNN) streamflow discharge modeling cannot be credibly trained "
                "or calibrated because open national telemetry feeds (India WRIS / CWC) only publish "
                "water level / gauge stage (meters), without continuous rating curves or discharge "
                "time-series (m^3/s). Synthetic or uncalibrated GNN flow predictions would violate "
                "scientific integrity standards."
            ),
        },
    }

    report_path = "reports/v1_3/basinflow_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved BasinFlow evaluation report to {report_path}")

    return results


class BasinFlowFloodRiskChallenger:
    """Inference wrapper for BasinFlow FloodRisk Challenger."""

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = "artifacts/models/v1_3/basinflow/flood_risk_basinflow_challenger.booster"
        self.booster = lgb.Booster(model_file=model_path)
        self.feature_names = ALL_BASINFLOW_FEATURES

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X)


if __name__ == "__main__":
    train_and_evaluate_basinflow()
