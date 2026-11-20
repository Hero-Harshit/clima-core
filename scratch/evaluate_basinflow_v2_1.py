"""BasinFlow v2.1 Evaluation and Tournament Script.
Evaluates Hybrid-GR4J, AR(1) Forecaster, and compiles Task A, Task B, and Task C benchmark tables.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from src.climacore.models.basinflow.camels_dataset import (
    PROCESSED_CAMELS_META,
    TRAIN_START, TRAIN_END,
    VAL_START, VAL_END,
    TEST_START, TEST_END,
    load_catchment_timeseries,
    get_static_attributes,
)
from src.climacore.models.basinflow.metrics import compute_hydrological_metrics
from src.climacore.models.basinflow.conceptual_models import GR4J
from src.climacore.models.basinflow.hybrid_gr4j import HybridGR4J
from src.climacore.models.basinflow.baselines import AR1Baseline, PersistenceBaseline
from src.climacore.models.basinflow.water_balance import compute_water_balance

def main():
    print("=" * 80)
    print("CLIMA-CORE V2.1: BASINFLOW V2.1 EVALUATION & BENCHMARK COMPILATION")
    print("=" * 80)

    with open(PROCESSED_CAMELS_META) as f:
        manifest = json.load(f)

    gauged_ids = [int(g) for g in manifest["spatial_partition"]["gauged_seen_ids"]]
    ungauged_ids = [int(g) for g in manifest["spatial_partition"]["ungauged_holdout_ids"]]
    
    gr4j_cache_file = "artifacts/models/v2_0/basinflow/gr4j_calibrated_gauges.json"
    with open(gr4j_cache_file) as f:
        cached_gr4j = json.load(f)

    v2_report_file = "reports/v2_0/basinflow_v2_tournament.json"
    with open(v2_report_file) as f:
        v2_report = json.load(f)

    os.makedirs("artifacts/models/v2_1/basinflow", exist_ok=True)
    os.makedirs("reports/v2_1", exist_ok=True)

    results_ar1 = []
    results_hybrid_gr4j = []
    hybrid_models_dict = {}
    sample_wb_reports = {}

    print(f"Fitting Hybrid-GR4J and AR(1) across {len(gauged_ids)} gauged catchments...")

    for idx, gid in enumerate(gauged_ids, 1):
        df = load_catchment_timeseries(gid)
        df_tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
        df_te = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]

        p_tr = df_tr["prcp"].values
        pet_tr = df_tr["pet"].values
        q_tr = df_tr["q_obs_mm"].values
        d_tr = df_tr["date"].values

        p_te = df_te["prcp"].values
        pet_te = df_te["pet"].values
        q_te = df_te["q_obs_mm"].values
        d_te = df_te["date"].values

        # 1. AR(1) baseline
        ar1 = AR1Baseline()
        ar1.fit(q_tr, p_tr)
        q_pred_ar1 = ar1.predict(q_te, p_te)
        results_ar1.append(compute_hydrological_metrics(q_te, q_pred_ar1))

        # 2. Hybrid-GR4J
        gr_inst = GR4J.from_dict(cached_gr4j[str(gid)])
        hybrid = HybridGR4J(gr4j_params=gr_inst.params, alpha=20.0)
        # Calibrate residual model using training data
        hybrid.calibrate(p_tr, pet_tr, q_tr, dates=d_tr, warmup_days=365)
        sim_res = hybrid.simulate(p_te, pet_te, dates=d_te)
        q_pred_hybrid = sim_res["q_sim"]
        results_hybrid_gr4j.append(compute_hydrological_metrics(q_te, q_pred_hybrid))
        hybrid_models_dict[str(gid)] = hybrid.to_dict()

        # Water balance sample on first 3 catchments
        if idx <= 3:
            area_km2 = float(df["area_km2"].iloc[0]) if "area_km2" in df.columns else None
            sample_wb_reports[str(gid)] = compute_water_balance(p_te, q_te, catchment_area_km2=area_km2)

        if idx % 20 == 0 or idx == len(gauged_ids):
            print(f"Processed {idx}/{len(gauged_ids)} catchments...")

    # Save hybrid models artifact
    out_models_path = "artifacts/models/v2_1/basinflow/hybrid_gr4j_calibrated_gauges.json"
    with open(out_models_path, "w") as f:
        json.dump(hybrid_models_dict, f, indent=2)
    print(f"Saved calibrated Hybrid-GR4J artifacts to {out_models_path}")

    # Helper function for median metrics
    def summarize(metric_list):
        summary = {}
        for k in metric_list[0].keys():
            vals = [m[k] for m in metric_list if not np.isnan(m[k])]
            summary[f"median_{k}"] = round(float(np.median(vals)), 4) if vals else 0.0
            summary[f"mean_{k}"] = round(float(np.mean(vals)), 4) if vals else 0.0
        return summary

    summary_ar1 = summarize(results_ar1)
    summary_hybrid = summarize(results_hybrid_gr4j)

    print("\n--- Summary Performance ---")
    print(f"GR4J Baseline:      Median KGE = {v2_report['summary_gauged']['GR4J']['median_kge']:.4f}, Median NSE = {v2_report['summary_gauged']['GR4J']['median_nse']:.4f}, Median RMSE = {v2_report['summary_gauged']['GR4J']['median_rmse']:.4f}")
    print(f"Hybrid-GR4J:        Median KGE = {summary_hybrid['median_kge']:.4f}, Median NSE = {summary_hybrid['median_nse']:.4f}, Median RMSE = {summary_hybrid['median_rmse']:.4f}")
    print(f"Persistence (AR0):  Median KGE = {v2_report['summary_gauged']['Persistence']['median_kge']:.4f}, Median NSE = {v2_report['summary_gauged']['Persistence']['median_nse']:.4f}")
    print(f"AR(1) Forecaster:   Median KGE = {summary_ar1['median_kge']:.4f}, Median NSE = {summary_ar1['median_nse']:.4f}")

    # Build 3-Task Tournament Report
    tournament_v2_1 = {
        "tournament_name": "BasinFlow V2.1 Hydrological Benchmark",
        "dataset": "CAMELS-IND v2.2 (Zenodo 14999580, CC-BY-4.0)",
        "spatial_splits": {
            "gauged_seen_count": len(gauged_ids),
            "ungauged_holdout_count": len(ungauged_ids)
        },
        "temporal_splits": v2_report["temporal_splits"],
        "task_a_gauged_1day_forecasting": {
            "task_description": "1-Day ahead streamflow forecasting conditioned on preceding observed discharge Q_{t-1}",
            "models": {
                "Persistence": v2_report["summary_gauged"]["Persistence"],
                "AR1_Forecaster": summary_ar1
            }
        },
        "task_b_rainfall_runoff_simulation": {
            "task_description": "Pure rainfall-runoff simulation driven solely by P_t and PET_t (zero historical streamflow conditioning)",
            "models": {
                "SeasonalClimatology": v2_report["summary_gauged"]["SeasonalClimatology"],
                "SCS-CN": v2_report["summary_gauged"]["SCS-CN"],
                "BasinFlow-V1": v2_report["summary_gauged"]["BasinFlow-V1"],
                "GR4J": v2_report["summary_gauged"]["GR4J"],
                "HBV-Light": v2_report["summary_gauged"]["HBV-Light"],
                "Standard-LSTM": v2_report["summary_gauged"]["Standard-LSTM"],
                "MC-LSTM": v2_report["summary_gauged"]["MC-LSTM"],
                "MC-LSTM-DS": v2_report["summary_gauged"]["MC-LSTM-DS"],
                "Hybrid-GR4J": summary_hybrid,
            }
        },
        "task_c_prediction_in_ungauged_basins": {
            "task_description": "Spatial transfer to catchments with zero streamflow history (PUB holdout)",
            "models": v2_report["summary_ungauged"]
        },
        "sample_water_balance_audits": sample_wb_reports,
    }

    out_report_path = "reports/v2_1/basinflow_v2_1_tournament.json"
    with open(out_report_path, "w") as f:
        json.dump(tournament_v2_1, f, indent=2)
    print(f"Exported master v2.1 tournament report to {out_report_path}")

if __name__ == "__main__":
    main()
