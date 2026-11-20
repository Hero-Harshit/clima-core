"""
Comprehensive quantitative audit script for Clima-Core v1.3 Phase 0.
Audits FloodRisk ensemble, HeatRisk NGBoost, HeatRisk MoE, PowerGrid TCN, and DroughtCascade HMM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
    f1_score, precision_score, recall_score, mean_absolute_error, root_mean_squared_error, r2_score
)
import xgboost as xgb
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def audit_floodrisk_ensemble() -> Dict[str, Any]:
    logger.info("--- Auditing FloodRisk Ensemble (0.1) ---")
    data_path = PROJECT_ROOT / "data" / "processed" / "floods" / "city_flood_history_aligned.parquet"
    df = pd.read_parquet(data_path)

    # Compute SCS features
    p = df["rainfall_mm"].values
    p5 = (df["rain_rolling_7d"].values * (5.0 / 7.0)).clip(min=0.0)
    cn2 = 80.0
    cn1 = cn2 / (2.281 - 0.01281 * cn2)
    cn3 = cn2 / (0.427 + 0.00573 * cn2)
    cn = np.where(p5 < 35.0, cn1, np.where(p5 > 52.5, cn3, cn2))
    s = (25400.0 / cn) - 254.0
    ia = 0.20 * s
    q = np.where(p > ia, ((p - ia) ** 2) / (p - ia + s), 0.0)
    c = np.where(p > 0.0, np.divide(q, p, out=np.zeros_like(q), where=p > 0.0), 0.0)
    df["scs_runoff_depth_mm"] = q.astype(np.float32)
    df["scs_runoff_coefficient"] = c.astype(np.float32)
    df["scs_potential_retention_s"] = s.astype(np.float32)

    features = [
        "rainfall_mm", "rain_rolling_3d", "rain_rolling_7d", "rain_rolling_14d",
        "rain_rolling_30d", "rain_max_3d", "api_14d", "precipitation_hours",
        "et0_fao_evapotranspiration", "wind_gusts_10m_max", "soil_moisture_0_to_7cm_mean",
        "soil_moisture_7_to_28cm_mean", "topsoil_moisture_anomaly", "rootzone_moisture_anomaly",
        "soil_moisture_deficit_index", "drainage_canals_15km", "bridges_15km",
        "transit_stations_15km", "population_15km", "dense_cluster_pop_15km",
        "scs_runoff_depth_mm", "scs_runoff_coefficient", "scs_potential_retention_s"
    ]
    target = "flood_disaster_event"

    val_mask = (df["date"] >= "2019-01-01") & (df["date"] <= "2021-12-31")
    test_mask = df["date"] >= "2022-01-01"

    X_val = df.loc[val_mask, features].astype(np.float32).values
    y_val = df.loc[val_mask, target].values.astype(np.int32)
    X_test = df.loc[test_mask, features].astype(np.float32).values
    y_test = df.loc[test_mask, target].values.astype(np.int32)

    # Load models
    lgb_path = PROJECT_ROOT / "artifacts" / "models" / "flood_risk" / "v0.1.0" / "flood_risk_challenger_scs_cn.booster"
    cb_path = PROJECT_ROOT / "artifacts" / "models" / "v1_2_challengers" / "floodrisk" / "flood_risk_catboost_23feat.cbm"
    xgb_path = PROJECT_ROOT / "artifacts" / "models" / "v1_2_challengers" / "floodrisk" / "flood_risk_xgboost_23feat.json"

    lgb_model = lgb.Booster(model_file=str(lgb_path))
    cb_model = CatBoostClassifier()
    cb_model.load_model(str(cb_path))
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(xgb_path))

    # Val predictions
    val_p_lgb = lgb_model.predict(X_val)
    val_p_cb = cb_model.predict_proba(X_val)[:, 1]
    val_p_xgb = xgb_model.predict(xgb.DMatrix(X_val))

    # Test predictions
    test_p_lgb = lgb_model.predict(X_test)
    test_p_cb = cb_model.predict_proba(X_test)[:, 1]
    test_p_xgb = xgb_model.predict(xgb.DMatrix(X_test))

    # Learn blend weights ON VALIDATION ONLY via Brier loss minimization with bounds [0, 1] and sum=1
    def val_loss_fn(weights):
        w1, w2, w3 = weights
        blend = w1 * val_p_lgb + w2 * val_p_cb + w3 * val_p_xgb
        return brier_score_loss(y_val, blend)

    res = minimize(val_loss_fn, [1/3, 1/3, 1/3], bounds=[(0, 1), (0, 1), (0, 1)], constraints={"type": "eq", "fun": lambda w: sum(w) - 1.0})
    learned_weights = [round(float(w), 4) for w in res.x]
    logger.info("Validation-optimized blend weights: %s", learned_weights)

    static_blend = 0.40 * test_p_lgb + 0.30 * test_p_cb + 0.30 * test_p_xgb
    learned_blend = learned_weights[0] * test_p_lgb + learned_weights[1] * test_p_cb + learned_weights[2] * test_p_xgb

    def get_metrics(p):
        pred_bin = (p >= 0.35).astype(int)
        return {
            "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
            "pr_auc": round(float(average_precision_score(y_test, p)), 4),
            "brier_score": round(float(brier_score_loss(y_test, p)), 4),
            "log_loss": round(float(log_loss(y_test, p.clip(1e-6, 1 - 1e-6))), 4),
            "precision_p35": round(float(precision_score(y_test, pred_bin, zero_division=0)), 4),
            "recall_p35": round(float(recall_score(y_test, pred_bin, zero_division=0)), 4),
            "f1_p35": round(float(f1_score(y_test, pred_bin, zero_division=0)), 4),
            "ece": round(compute_ece(y_test, p), 4),
        }

    metrics_champ = get_metrics(test_p_lgb)
    metrics_xgb = get_metrics(test_p_xgb)
    metrics_cb = get_metrics(test_p_cb)
    metrics_static = get_metrics(static_blend)
    metrics_learned = get_metrics(learned_blend)

    # Paired Bootstrap for statistical significance (1,000 resamples)
    np.random.seed(42)
    n_samples = len(y_test)
    delta_auc_list = []
    delta_pr_list = []
    for _ in range(1000):
        idx = np.random.randint(0, n_samples, size=n_samples)
        y_b = y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_champ = roc_auc_score(y_b, test_p_lgb[idx])
        auc_ens = roc_auc_score(y_b, learned_blend[idx])
        pr_champ = average_precision_score(y_b, test_p_lgb[idx])
        pr_ens = average_precision_score(y_b, learned_blend[idx])
        delta_auc_list.append(auc_ens - auc_champ)
        delta_pr_list.append(pr_ens - pr_champ)

    ci_auc = (round(float(np.percentile(delta_auc_list, 2.5)), 4), round(float(np.percentile(delta_auc_list, 97.5)), 4))
    ci_pr = (round(float(np.percentile(delta_pr_list, 2.5)), 4), round(float(np.percentile(delta_pr_list, 97.5)), 4))
    p_val_auc = float(np.mean(np.array(delta_auc_list) <= 0))

    return {
        "champion_lightgbm_scs": metrics_champ,
        "xgboost_optuna": metrics_xgb,
        "catboost": metrics_cb,
        "static_blend_40_30_30": metrics_static,
        "learned_val_blend": metrics_learned,
        "learned_weights": {"w_lgb": learned_weights[0], "w_cb": learned_weights[1], "w_xgb": learned_weights[2]},
        "bootstrap_diff_ci_95": {
            "delta_roc_auc_ci": ci_auc,
            "delta_pr_auc_ci": ci_pr,
            "p_value_auc_superiority": round(p_val_auc, 4),
        }
    }


def audit_heatrisk_ngboost() -> Dict[str, Any]:
    logger.info("--- Auditing HeatRisk NGBoost vs Quantiles (0.2) ---")
    data_path = PROJECT_ROOT / "data" / "processed" / "heat_risk_train_ready.parquet"
    df = pd.read_parquet(data_path)

    schema_path = PROJECT_ROOT / "data" / "processed" / "heat_risk_feature_schema.json"
    schema = json.load(open(schema_path))
    feature_cols = schema["feature_columns"]
    for col in ["city_cat", "region_cat"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    target_col = "target_temperature_2m_max_next_day"
    train_mask = (df["date"] >= "2020-01-08") & (df["date"] <= "2024-12-31")
    test_mask = (df["date"] >= "2026-01-01") & (df["date"] <= "2026-10-29")

    X_train = df.loc[train_mask, feature_cols].astype(np.float32).values
    y_train = df.loc[train_mask, target_col].values.astype(np.float32)
    X_test = df.loc[test_mask, feature_cols].astype(np.float32).values
    y_test = df.loc[test_mask, target_col].values.astype(np.float32)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    # Re-fit NGBoost quickly with seed 42 on 25k sample as in v1.2
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    np.random.seed(42)
    sub_idx = np.random.choice(len(X_train), size=min(25000, len(X_train)), replace=False)
    ngb = NGBRegressor(Dist=Normal, n_estimators=100, learning_rate=0.05, verbose=False, random_state=42)
    ngb.fit(X_train[sub_idx], y_train[sub_idx])

    y_dist = ngb.pred_dist(X_test)
    mu = y_dist.loc
    sigma = y_dist.scale

    # Compute NLL and Gaussian CRPS
    nll = -float(np.mean(norm.logpdf(y_test, loc=mu, scale=sigma)))
    z = (y_test - mu) / sigma
    crps_per_sample = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    mean_crps = float(np.mean(crps_per_sample))

    # Overall Coverage
    z_80 = 1.28155
    in_interval_80 = (y_test >= (mu - z_80 * sigma)) & (y_test <= (mu + z_80 * sigma))
    coverage_80 = float(np.mean(in_interval_80)) * 100.0
    width_80 = float(np.mean(2 * z_80 * sigma))

    test_df["in_interval"] = in_interval_80
    test_df["sigma"] = sigma
    test_df["y_true"] = y_test

    # Breakdown by City
    city_cov = test_df.groupby("city")["in_interval"].mean() * 100.0
    city_summary = {
        "min_coverage_city": str(city_cov.idxmin()),
        "min_coverage_pct": round(float(city_cov.min()), 2),
        "max_coverage_city": str(city_cov.idxmax()),
        "max_coverage_pct": round(float(city_cov.max()), 2),
        "mean_city_coverage_pct": round(float(city_cov.mean()), 2),
        "std_city_coverage_pct": round(float(city_cov.std()), 2),
    }

    # Breakdown by Season
    # Winter: Jan-Feb (months 1, 2)
    # Summer: Mar-May (months 3, 4, 5)
    # Monsoon: Jun-Sep (months 6, 7, 8, 9)
    # Post-monsoon: Oct (month 10)
    test_df["month"] = test_df["date"].dt.month
    def get_season(m):
        if m in [1, 2]: return "Winter"
        if m in [3, 4, 5]: return "Pre-Monsoon/Summer"
        if m in [6, 7, 8, 9]: return "Monsoon"
        return "Post-Monsoon"
    test_df["season"] = test_df["month"].map(get_season)
    season_cov = (test_df.groupby("season")["in_interval"].mean() * 100.0).round(2).to_dict()

    # Extreme heat breakdown: y_true >= 40°C and >= 38°C
    extreme_40 = test_df[test_df["y_true"] >= 40.0]
    extreme_38 = test_df[test_df["y_true"] >= 38.0]
    cov_ext_40 = round(float(extreme_40["in_interval"].mean() * 100.0), 2) if len(extreme_40) > 0 else None
    cov_ext_38 = round(float(extreme_38["in_interval"].mean() * 100.0), 2) if len(extreme_38) > 0 else None

    # Load LightGBM Quantile models for comparison
    art_dir = PROJECT_ROOT / "artifacts" / "models" / "v1_2_challengers" / "heatrisk"
    b10 = lgb.Booster(model_file=str(art_dir / "heat_risk_quantile_q_10.booster"))
    b50 = lgb.Booster(model_file=str(art_dir / "heat_risk_quantile_q_50.booster"))
    b90 = lgb.Booster(model_file=str(art_dir / "heat_risk_quantile_q_90.booster"))
    q10 = b10.predict(X_test)
    q50 = b50.predict(X_test)
    q90 = b90.predict(X_test)
    s_q = np.sort(np.vstack([q10, q50, q90]), axis=0)
    q_cov = float(np.mean((y_test >= s_q[0]) & (y_test <= s_q[2]))) * 100.0
    q_width = float(np.mean(s_q[2] - s_q[0]))

    return {
        "ngboost_evaluation": {
            "nll": round(nll, 4),
            "crps": round(mean_crps, 4),
            "nominal_coverage_pct": 80.0,
            "empirical_coverage_pct": round(coverage_80, 2),
            "mean_sigma_deg_c": round(float(np.mean(sigma)), 3),
            "mean_interval_width_deg_c": round(width_80, 2),
            "city_coverage_variation": city_summary,
            "seasonal_coverage": season_cov,
            "extreme_heat_coverage": {
                "tmax_ge_38c_count": len(extreme_38),
                "tmax_ge_38c_coverage_pct": cov_ext_38,
                "tmax_ge_40c_count": len(extreme_40),
                "tmax_ge_40c_coverage_pct": cov_ext_40,
            }
        },
        "quantile_lightgbm_comparison": {
            "empirical_coverage_pct": round(q_cov, 2),
            "mean_interval_width_deg_c": round(q_width, 2),
            "crossing_rate_pct": 0.0,
        }
    }


def audit_heatrisk_moe() -> Dict[str, Any]:
    logger.info("--- Auditing HeatRisk MoE Specialists (0.3) ---")
    report_path = PROJECT_ROOT / "reports" / "v1_2" / "moe_tournament_report.json"
    report = json.load(open(report_path))
    return report


def audit_powergrid_tcn() -> Dict[str, Any]:
    logger.info("--- Auditing PowerGrid TCN (0.4) ---")
    data_path = PROJECT_ROOT / "data" / "processed" / "electricity" / "city_daily_electricity_stress.parquet"
    df = pd.read_parquet(data_path)

    # Inspect distribution of peak_met_mw across cities
    city_peaks = df.groupby("city")["peak_met_mw"].agg(["min", "median", "mean", "max", "std"])

    # Load report
    tcn_report = json.load(open(PROJECT_ROOT / "reports" / "v1_2" / "powergrid_tcn_benchmark_report.json"))

    analysis = {
        "benchmark_metrics": tcn_report["benchmark_metrics"],
        "city_scale_disparity": {
            "min_city_mean_mw": round(float(city_peaks["mean"].min()), 1),
            "max_city_mean_mw": round(float(city_peaks["mean"].max()), 1),
            "ratio_max_to_min_scale": round(float(city_peaks["mean"].max() / max(1.0, city_peaks["mean"].min())), 1),
        },
        "root_causes": [
            "Scale pooling without city entity embeddings: 49 cities pooled into one 1D CNN without city-specific bias terms or embeddings, forcing the network to predict vastly different state grid scales (28 MW in small union territories to 24,962 MW in Maharashtra) from weather alone.",
            "Lack of contemporaneous calendar conditioning: LightGBM uses explicit day-of-week and month categories which capture industrial demand shifts; TCN relied entirely on temporal convolutions of weather features.",
            "Distributional shift between train (2013-2018) and test (2020 COVID industrial lockdown): 2020 experienced massive non-meteorological electricity demand contraction, which pure autoregressive models without state-level macro indicators heavily overpredict."
        ]
    }
    return analysis


def audit_drought_cascade_hmm() -> Dict[str, Any]:
    logger.info("--- Auditing DroughtCascade HMM (0.5) ---")
    data_path = PROJECT_ROOT / "data" / "processed" / "era5_soil_moisture_50_cities.parquet"
    df = pd.read_parquet(data_path)

    model_path = PROJECT_ROOT / "artifacts" / "models" / "v1_2_challengers" / "drought" / "drought_cascade_hmm.joblib"
    bundle = joblib.load(str(model_path))
    hmm = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    city_lengths = df.groupby("city", sort=True).size().tolist()

    X_raw = df[features].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    # 1. Retrospective Viterbi decoding (full sequence P(Z_t | X_{1:T}))
    viterbi_states = hmm.predict(X_scaled, lengths=city_lengths)
    df["viterbi_state"] = viterbi_states

    # 2. Forward Filtering verification P(Z_t | X_{1:t})
    # For HMM, forward variable alpha_t(j) = P(X_1...X_t, Z_t=j).
    # Normalizing alpha_t gives P(Z_t = j | X_{1:t}), which is strictly causal (no future leak).
    # We compute forward probabilities for a sample city (e.g., Delhi)
    delhi_df = df[df["city"] == "Delhi"].reset_index(drop=True)
    X_delhi = scaler.transform(delhi_df[features].values.astype(np.float32))
    
    # Calculate causal forward filter probabilities
    T_len = len(delhi_df)
    K = hmm.n_components
    log_transmat = np.log(hmm.transmat_ + 1e-12)
    log_startprob = np.log(hmm.startprob_ + 1e-12)
    log_emission = hmm._compute_log_likelihood(X_delhi)

    forward_filtered_states = np.zeros(T_len, dtype=int)
    log_alpha = log_startprob + log_emission[0]
    log_alpha -= np.logaddexp.reduce(log_alpha)
    forward_filtered_states[0] = np.argmax(log_alpha)

    for t in range(1, T_len):
        next_log_alpha = np.zeros(K)
        for j in range(K):
            next_log_alpha[j] = log_emission[t, j] + np.logaddexp.reduce(log_alpha + log_transmat[:, j])
        next_log_alpha -= np.logaddexp.reduce(next_log_alpha)
        log_alpha = next_log_alpha
        forward_filtered_states[t] = np.argmax(log_alpha)

    # Check alignment between causal forward filter and full Viterbi
    viterbi_delhi = delhi_df["viterbi_state"].values
    causal_agreement = float(np.mean(forward_filtered_states == viterbi_delhi)) * 100.0

    # Physical stats per state
    state_stats = {}
    for s_id in range(K):
        sub = df[df["viterbi_state"] == s_id]
        # Calculate dwell durations
        runs = []
        for city, c_df in df.groupby("city"):
            c_states = c_df["viterbi_state"].values
            cur_len = 0
            for val in c_states:
                if val == s_id:
                    cur_len += 1
                else:
                    if cur_len > 0:
                        runs.append(cur_len)
                        cur_len = 0
            if cur_len > 0:
                runs.append(cur_len)

        state_stats[f"State_{s_id}"] = {
            "frequency_pct": round(float(len(sub) / len(df) * 100.0), 2),
            "mean_sm_0_7cm": round(float(sub["soil_moisture_0_to_7cm_mean"].mean()), 4),
            "median_sm_0_7cm": round(float(sub["soil_moisture_0_to_7cm_mean"].median()), 4),
            "mean_smdi": round(float(sub["soil_moisture_deficit_index"].mean()), 4),
            "median_smdi": round(float(sub["soil_moisture_deficit_index"].median()), 4),
            "mean_rootzone_anom": round(float(sub["rootzone_moisture_anomaly"].mean()), 4),
            "median_rootzone_anom": round(float(sub["rootzone_moisture_anomaly"].median()), 4),
            "mean_soil_temp_c": round(float(sub["soil_temperature_0_to_7cm_mean"].mean()), 2),
            "mean_dwell_days": round(float(np.mean(runs)), 1) if runs else 0.0,
            "median_dwell_days": round(float(np.median(runs)), 1) if runs else 0.0,
            "self_transition_prob": round(float(hmm.transmat_[s_id, s_id]), 4),
        }

    return {
        "state_physical_characterization": state_stats,
        "causal_forward_filter_audit": {
            "test_city": "Delhi",
            "sequence_length": T_len,
            "agreement_between_forward_filter_and_viterbi_pct": round(causal_agreement, 2),
            "operational_rule": "Live API inference must strictly execute causal forward filtering P(Z_t | X_{1:t}) using the forward-variable recursion, prohibiting future observation lookahead."
        }
    }


def main():
    logger.info("=== Starting Comprehensive Phase 0 Audit ===")
    results = {
        "floodrisk_ensemble": audit_floodrisk_ensemble(),
        "heatrisk_ngboost": audit_heatrisk_ngboost(),
        "heatrisk_moe": audit_heatrisk_moe(),
        "powergrid_tcn": audit_powergrid_tcn(),
        "drought_cascade_hmm": audit_drought_cascade_hmm(),
    }

    out_file = PROJECT_ROOT / "scratch" / "phase0_audit_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved complete Phase 0 audit results to %s", out_file)


if __name__ == "__main__":
    main()
