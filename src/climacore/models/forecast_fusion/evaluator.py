"""Training and evaluation pipeline for ForecastFusion V1.
Executes chronological train/val/test splits, trains all four ensemble architectures,
evaluates on test holdout against raw NWP inputs and ForecastCorrectionEngine champion,
and serializes artifacts.
"""

import os
import json
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

from climacore.models.forecast_fusion.ensemble_models import (
    WeightedForecastEnsemble,
    EMOSModel,
    NGBoostForecastFusion,
    QuantileLGBMEnsemble,
    TwoPartPrecipitationEnsemble,
    gaussian_crps,
)
from climacore.models.forecast_fusion.skill_matrix import (
    compute_continuous_metrics,
    compute_precipitation_metrics,
)


def extract_features_and_targets(df: pd.DataFrame) -> Dict[str, Any]:
    """Prepares feature matrices and target arrays for temperature and precipitation tasks."""
    # Raw multi-model forecasts
    tmax_raw = df[["tmax_gfs", "tmax_jma", "tmax_icon", "t2m_max_fcst"]].values.astype(np.float64)
    tmin_raw = df[["tmin_gfs", "tmin_jma", "tmin_icon", "t2m_min_fcst"]].values.astype(np.float64)
    precip_raw = df[["precip_gfs", "precip_jma", "precip_icon", "precip_fcst"]].values.astype(np.float64)
    
    # Engineered features for ML models
    def build_features(raw_arr, extra_cols):
        mean_val = np.mean(raw_arr, axis=1, keepdims=True)
        std_val = np.std(raw_arr, axis=1, keepdims=True)
        min_val = np.min(raw_arr, axis=1, keepdims=True)
        max_val = np.max(raw_arr, axis=1, keepdims=True)
        range_val = max_val - min_val
        context = df[extra_cols].values
        return np.hstack([raw_arr, mean_val, std_val, range_val, context]).astype(np.float32)

    extra_cols = ["latitude", "longitude", "elevation", "sin_doy", "cos_doy", "rh_mean_fcst", "ws_mean_fcst"]
    X_tmax_ml = build_features(tmax_raw, extra_cols)
    X_tmin_ml = build_features(tmin_raw, extra_cols)
    X_precip_ml = build_features(precip_raw, extra_cols)
    
    y_tmax = df["t2m_max_obs"].values.astype(np.float32)
    y_tmin = df["t2m_min_obs"].values.astype(np.float32)
    y_precip = df["precip_obs"].values.astype(np.float32)
    
    return {
        "tmax_raw": tmax_raw,
        "tmin_raw": tmin_raw,
        "precip_raw": precip_raw,
        "X_tmax_ml": X_tmax_ml,
        "X_tmin_ml": X_tmin_ml,
        "X_precip_ml": X_precip_ml,
        "y_tmax": y_tmax,
        "y_tmin": y_tmin,
        "y_precip": y_precip,
        "df": df,
    }


def compute_probabilistic_metrics(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Computes continuous CRPS, 80% coverage, and interval width."""
    sigma = np.maximum(sigma, 1e-4)
    crps_vals = gaussian_crps(mu, sigma, y)
    mean_crps = float(np.mean(crps_vals))
    
    # 80% interval: z = 1.28155
    z_80 = 1.28155
    lower = mu - z_80 * sigma
    upper = mu + z_80 * sigma
    coverage_80 = float(np.mean((y >= lower) & (y <= upper)))
    mean_width = float(np.mean(upper - lower))
    
    cont_metrics = compute_continuous_metrics(y, mu)
    return {
        **cont_metrics,
        "crps": round(mean_crps, 4),
        "coverage_80": round(coverage_80, 4),
        "interval_width": round(mean_width, 4),
    }


def compute_quantile_metrics(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Computes empirical metrics from quantile predictions."""
    coverage_80 = float(np.mean((y >= q10) & (y <= q90)))
    mean_width = float(np.mean(q90 - q10))
    cont_metrics = compute_continuous_metrics(y, q50)
    
    # Empirical pinball loss (CRPS approximation from quantiles)
    def pinball(q, alpha):
        diff = y - q
        return np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff))
        
    p10 = pinball(q10, 0.10)
    p50 = pinball(q50, 0.50)
    p90 = pinball(q90, 0.90)
    mean_pinball = float((p10 + p50 + p90) / 3.0)
    
    return {
        **cont_metrics,
        "crps": round(mean_pinball, 4),
        "coverage_80": round(coverage_80, 4),
        "interval_width": round(mean_width, 4),
    }


def run_forecast_fusion_training_and_eval():
    data_path = "data/processed/forecast_correction_pairs/multimodel_forecast_obs_pairs.parquet"
    df = pd.read_parquet(data_path)
    
    train_mask = df["date"] < "2024-08-01"
    val_mask = (df["date"] >= "2024-08-01") & (df["date"] < "2024-09-15")
    test_mask = df["date"] >= "2024-09-15"
    
    train_val_mask = df["date"] < "2024-09-15"
    
    print(f"Data splits: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    
    data = extract_features_and_targets(df)
    
    models_dir = "artifacts/models/v1_3/forecast_fusion"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("reports/v1_3", exist_ok=True)
    
    prov_names = ["gfs", "jma", "icon", "best_match"]
    eval_results: Dict[str, Any] = {
        "metadata": {
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "n_test": int(test_mask.sum()),
            "test_date_range": [str(df.loc[test_mask, "date"].min().date()), str(df.loc[test_mask, "date"].max().date())],
        },
        "tmax": {},
        "tmin": {},
        "precip": {},
    }
    
    # =========================================================================
    # 1. EVALUATE RAW BASELINES ON TEST SET
    # =========================================================================
    for var, raw_key, obs_key in [
        ("tmax", "tmax_raw", "y_tmax"),
        ("tmin", "tmin_raw", "y_tmin"),
        ("precip", "precip_raw", "y_precip"),
    ]:
        raw_test = data[raw_key][test_mask]
        y_test = data[obs_key][test_mask]
        for i, prov in enumerate(prov_names):
            if var == "precip":
                eval_results[var][f"raw_{prov}"] = compute_precipitation_metrics(y_test, raw_test[:, i])
            else:
                eval_results[var][f"raw_{prov}"] = compute_continuous_metrics(y_test, raw_test[:, i])
                
    # =========================================================================
    # 2. MODEL A: WEIGHTED FORECAST ENSEMBLE
    # =========================================================================
    print("\n--- Training Model A: Weighted Forecast Ensemble ---")
    for var, raw_key, obs_key in [
        ("tmax", "tmax_raw", "y_tmax"),
        ("tmin", "tmin_raw", "y_tmin"),
        ("precip", "precip_raw", "y_precip"),
    ]:
        # Fit on train+val
        ens = WeightedForecastEnsemble(fit_intercept=True)
        ens.fit(data[raw_key][train_val_mask], data[obs_key][train_val_mask], feature_names=prov_names)
        
        preds_test = ens.predict(data[raw_key][test_mask])
        if var == "precip":
            preds_test = np.maximum(preds_test, 0.0)
            res = compute_precipitation_metrics(data[obs_key][test_mask], preds_test)
        else:
            res = compute_continuous_metrics(data[obs_key][test_mask], preds_test)
            
        res["weights"] = ens.get_weights_dict()
        eval_results[var]["model_a_weighted_ensemble"] = res
        joblib.dump(ens, os.path.join(models_dir, f"{var}_model_a_weighted.joblib"))
        print(f"Model A {var}: MAE = {res['mae']}, Bias = {res['bias']}, Weights = {res['weights']}")

    # =========================================================================
    # 3. MODEL B: EMOS (Ensemble Model Output Statistics)
    # =========================================================================
    print("\n--- Training Model B: EMOS (CRPS Minimized) ---")
    for var, raw_key, obs_key in [("tmax", "tmax_raw", "y_tmax"), ("tmin", "tmin_raw", "y_tmin")]:
        emos = EMOSModel()
        emos.fit(data[raw_key][train_val_mask], data[obs_key][train_val_mask], feature_names=prov_names)
        
        mu_test, sigma_test = emos.predict_distribution(data[raw_key][test_mask])
        res = compute_probabilistic_metrics(mu_test, sigma_test, data[obs_key][test_mask])
        res["params"] = emos.get_params_dict()
        eval_results[var]["model_b_emos"] = res
        joblib.dump(emos, os.path.join(models_dir, f"{var}_model_b_emos.joblib"))
        print(f"Model B EMOS {var}: MAE = {res['mae']}, CRPS = {res['crps']}, 80% Cov = {res['coverage_80']}, Width = {res['interval_width']}C")

    # =========================================================================
    # 4. MODEL C: NGBoost Distributional Regressor
    # =========================================================================
    print("\n--- Training Model C: NGBoost Distributional Regressor ---")
    for var, ml_key, obs_key in [("tmax", "X_tmax_ml", "y_tmax"), ("tmin", "X_tmin_ml", "y_tmin")]:
        ngb = NGBoostForecastFusion(n_estimators=120, learning_rate=0.03, dist="normal")
        ngb.fit(data[ml_key][train_val_mask], data[obs_key][train_val_mask])
        
        mu_test, sigma_test = ngb.predict_distribution(data[ml_key][test_mask])
        res = compute_probabilistic_metrics(mu_test, sigma_test, data[obs_key][test_mask])
        eval_results[var]["model_c_ngboost"] = res
        joblib.dump(ngb, os.path.join(models_dir, f"{var}_model_c_ngboost.joblib"))
        print(f"Model C NGBoost {var}: MAE = {res['mae']}, CRPS = {res['crps']}, 80% Cov = {res['coverage_80']}")

    # =========================================================================
    # 5. MODEL D: Quantile LightGBM & Two-Part Precipitation Ensemble
    # =========================================================================
    print("\n--- Training Model D: Quantile LGBM & Two-Part Precip Ensemble ---")
    # For Tmax and Tmin: Quantile LGBM
    for var, ml_key, obs_key in [("tmax", "X_tmax_ml", "y_tmax"), ("tmin", "X_tmin_ml", "y_tmin")]:
        qlgb = QuantileLGBMEnsemble(quantiles=[0.10, 0.50, 0.90], n_estimators=100, max_depth=4)
        qlgb.fit(data[ml_key][train_val_mask], data[obs_key][train_val_mask])
        
        q_preds = qlgb.predict_quantiles(data[ml_key][test_mask])
        res = compute_quantile_metrics(q_preds[0.10], q_preds[0.50], q_preds[0.90], data[obs_key][test_mask])
        eval_results[var]["model_d_quantile_lgbm"] = res
        joblib.dump(qlgb, os.path.join(models_dir, f"{var}_model_d_quantile_lgbm.joblib"))
        print(f"Model D Quantile LGBM {var}: MAE = {res['mae']}, 80% Cov = {res['coverage_80']}, Width = {res['interval_width']}C")

    # For Precipitation: Two-Part Ensemble
    two_part = TwoPartPrecipitationEnsemble(threshold=0.1, n_estimators=100)
    two_part.fit(data["X_precip_ml"][train_val_mask], data["y_precip"][train_val_mask])
    
    p_rain_test = two_part.predict_proba(data["X_precip_ml"][test_mask])
    exp_rain_test = two_part.predict_expected(data["X_precip_ml"][test_mask])
    
    y_precip_test = data["y_precip"][test_mask]
    rain_binary_test = (y_precip_test > 0.1).astype(int)
    
    brier_test = float(np.mean((p_rain_test - rain_binary_test) ** 2))
    cont_res = compute_continuous_metrics(y_precip_test, exp_rain_test)
    
    res_twopart = {
        **cont_res,
        "brier": round(brier_test, 4),
        "mean_rain_prob": round(float(np.mean(p_rain_test)), 4),
        "obs_rain_freq": round(float(np.mean(rain_binary_test)), 4),
    }
    eval_results["precip"]["model_d_two_part_ensemble"] = res_twopart
    joblib.dump(two_part, os.path.join(models_dir, "precip_model_d_two_part.joblib"))
    print(f"Model D Two-Part Precip: MAE = {res_twopart['mae']}, Brier = {res_twopart['brier']}, Obs Freq = {res_twopart['obs_rain_freq']}")

    # =========================================================================
    # 6. SAVE RESULTS AND BENCHMARKS
    # =========================================================================
    out_file = "reports/v1_3/forecast_fusion_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved ForecastFusion V1 evaluation report to {out_file}")
    
    return eval_results


if __name__ == "__main__":
    run_forecast_fusion_training_and_eval()
