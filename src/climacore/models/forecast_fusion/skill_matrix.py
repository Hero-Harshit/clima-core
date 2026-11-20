"""Skill Matrix computation for multi-model NWP verification against independent IMD observations.
Compares GFS, JMA, ICON, and Open-Meteo BestMatch across cities, seasons, and lead times.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def compute_continuous_metrics(obs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """Computes MAE, RMSE, Mean Bias, and Pearson correlation."""
    err = pred - obs
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    if len(obs) > 2 and np.std(obs) > 1e-6 and np.std(pred) > 1e-6:
        r, _ = pearsonr(obs, pred)
        corr = float(r)
    else:
        corr = 0.0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "corr": round(corr, 4),
    }


def compute_precipitation_metrics(obs: np.ndarray, pred: np.ndarray, threshold: float = 0.1) -> Dict[str, float]:
    """Computes precipitation categorical verification metrics: Brier score, POD, FAR, CSI."""
    obs_rain = (obs > threshold).astype(int)
    pred_rain = (pred > threshold).astype(int)
    
    # Hits, Misses, False Alarms, Correct Negatives
    hits = int(np.sum((obs_rain == 1) & (pred_rain == 1)))
    misses = int(np.sum((obs_rain == 1) & (pred_rain == 0)))
    fas = int(np.sum((obs_rain == 0) & (pred_rain == 1)))
    cns = int(np.sum((obs_rain == 0) & (pred_rain == 0)))
    
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = fas / (hits + fas) if (hits + fas) > 0 else 0.0
    csi = hits / (hits + misses + fas) if (hits + misses + fas) > 0 else 0.0
    brier = float(np.mean((pred_rain - obs_rain) ** 2))
    
    cont_metrics = compute_continuous_metrics(obs, pred)
    return {
        **cont_metrics,
        "pod": round(float(pod), 4),
        "far": round(float(far), 4),
        "csi": round(float(csi), 4),
        "brier": round(float(brier), 4),
    }


def compute_skill_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluates GFS, JMA, ICON, and BestMatch against IMD observations.
    
    Returns structured skill comparisons:
    - Overall metrics by variable
    - City-specific breakdown
    - Seasonal breakdown
    - Ranking summary
    """
    providers = {
        "tmax": {
            "gfs": "tmax_gfs",
            "jma": "tmax_jma",
            "icon": "tmax_icon",
            "best_match": "t2m_max_fcst",
        },
        "tmin": {
            "gfs": "tmin_gfs",
            "jma": "tmin_jma",
            "icon": "tmin_icon",
            "best_match": "t2m_min_fcst",
        },
        "precip": {
            "gfs": "precip_gfs",
            "jma": "precip_jma",
            "icon": "precip_icon",
            "best_match": "precip_fcst",
        },
    }
    
    obs_cols = {
        "tmax": "t2m_max_obs",
        "tmin": "t2m_min_obs",
        "precip": "precip_obs",
    }
    
    results: Dict[str, Any] = {
        "overall": {},
        "by_city": {},
        "by_season": {},
        "rankings": {},
    }
    
    # 1. Overall evaluation
    for var, prov_map in providers.items():
        results["overall"][var] = {}
        obs_val = df[obs_cols[var]].values
        for prov, col in prov_map.items():
            pred_val = df[col].values
            if var == "precip":
                results["overall"][var][prov] = compute_precipitation_metrics(obs_val, pred_val)
            else:
                results["overall"][var][prov] = compute_continuous_metrics(obs_val, pred_val)
                
    # 2. Breakdown by City
    for city in df["city"].unique():
        sub_df = df[df["city"] == city]
        results["by_city"][city] = {}
        for var, prov_map in providers.items():
            results["by_city"][city][var] = {}
            obs_val = sub_df[obs_cols[var]].values
            for prov, col in prov_map.items():
                pred_val = sub_df[col].values
                if var == "precip":
                    results["by_city"][city][var][prov] = compute_precipitation_metrics(obs_val, pred_val)
                else:
                    results["by_city"][city][var][prov] = compute_continuous_metrics(obs_val, pred_val)
                    
    # 3. Breakdown by Season
    for season in df["season"].unique():
        sub_df = df[df["season"] == season]
        results["by_season"][season] = {}
        for var, prov_map in providers.items():
            results["by_season"][season][var] = {}
            obs_val = sub_df[obs_cols[var]].values
            for prov, col in prov_map.items():
                pred_val = sub_df[col].values
                if var == "precip":
                    results["by_season"][season][var][prov] = compute_precipitation_metrics(obs_val, pred_val)
                else:
                    results["by_season"][season][var][prov] = compute_continuous_metrics(obs_val, pred_val)

    # 4. Determine Rankings by MAE
    for var in ["tmax", "tmin", "precip"]:
        prov_maes = {p: results["overall"][var][p]["mae"] for p in ["gfs", "jma", "icon", "best_match"]}
        sorted_provs = sorted(prov_maes.items(), key=lambda x: x[1])
        results["rankings"][var] = [{"rank": i + 1, "provider": p, "mae": mae} for i, (p, mae) in enumerate(sorted_provs)]

    return results


if __name__ == "__main__":
    import json
    df_pairs = pd.read_parquet("data/processed/forecast_correction_pairs/multimodel_forecast_obs_pairs.parquet")
    matrix = compute_skill_matrix(df_pairs)
    print("=== FORECAST SKILL MATRIX OVERALL RANKINGS ===")
    for var, ranks in matrix["rankings"].items():
        print(f"\nTarget: {var}")
        for r in ranks:
            print(f"  Rank {r['rank']}: {r['provider']} (MAE = {r['mae']})")
    
    out_path = "reports/v1_3/forecast_skill_matrix.json"
    with open(out_path, "w") as f:
        json.dump(matrix, f, indent=2)
    print(f"\nSaved full skill matrix to {out_path}")
