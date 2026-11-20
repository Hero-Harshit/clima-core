"""Benchmark verification and evaluation pipeline for DroughtCascade V2.
Evaluates causal forward filtering vs retrospective Viterbi, computes empirical dwell times,
and verifies historical multi-decade Indian drought events (2002, 2009, 2014, 2015, 2018).
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from climacore.models.drought.drought_cascade_v2 import (
    DroughtCascadeV2,
    DROUGHT_HMM_FEATURES,
    STATE_NAMES,
)


HISTORICAL_DROUGHT_WINDOWS = [
    {"name": "2002 All-India Monsoon Failure", "start": "2002-07-01", "end": "2002-10-31"},
    {"name": "2009 Severe Countrywide Drought", "start": "2009-06-01", "end": "2009-10-31"},
    {"name": "2014 El Nino Drought", "start": "2014-06-01", "end": "2014-11-30"},
    {"name": "2015 Back-to-Back El Nino Drought", "start": "2015-06-01", "end": "2015-11-30"},
    {"name": "2018-2019 Maharashtra/Central Drought", "start": "2018-09-01", "end": "2019-06-30"},
]


def run_drought_cascade_evaluation():
    # 1. Load data
    soil_path = "data/processed/era5_soil_moisture_50_cities.parquet"
    print("Loading ERA5 soil moisture records...")
    df = pd.read_parquet(soil_path)
    df["date"] = pd.to_datetime(df["date"])
    df["city_lower"] = df["city"].str.lower()
    df = df.sort_values(by=["city_lower", "date"]).reset_index(drop=True)
    print(f"Total soil records: {len(df):,} across {df['city_lower'].nunique()} cities")

    # 2. Load model
    existing_model_path = "artifacts/models/v1_2_challengers/drought/drought_cascade_hmm.joblib"
    cascade = DroughtCascadeV2()
    cascade.load_from_existing(existing_model_path)
    
    dwell_stats = cascade.get_dwell_time_statistics()
    print("\nTheoretical Expected Dwell Times:")
    for st_id, st_info in dwell_stats.items():
        print(f"  {st_info['name']:38s}: {st_info['expected_dwell_days']} days (Self-P = {st_info['self_transition_prob']}, SMDI = {st_info['mean_smdi']})")

    # 3. Test Causal Forward Filtering across 5 key cities
    eval_cities = ["delhi", "nagpur", "jaipur", "chennai", "kolkata"]
    city_eval_results = {}
    
    print("\nEvaluating Causal Forward Filtering vs Retrospective Viterbi:")
    for city in eval_cities:
        city_data = df[df["city_lower"] == city].copy()
        X_city = city_data[DROUGHT_HMM_FEATURES].values.astype(np.float64)
        
        # Causal forward probabilities
        causal_probs = cascade.forward_filter_probabilities(X_city)
        causal_states = np.argmax(causal_probs, axis=1)
        
        # Retrospective Viterbi
        viterbi_states = cascade.predict_viterbi(X_city)
        
        agreement = float(np.mean(causal_states == viterbi_states))
        
        # Severe drought state metrics
        severe_state = cascade.severe_drought_state_
        viterbi_severe = viterbi_states == severe_state
        causal_severe = causal_states == severe_state
        
        severe_recall = float(np.mean(causal_severe[viterbi_severe])) if np.sum(viterbi_severe) > 0 else 0.0
        severe_precision = float(np.mean(viterbi_severe[causal_severe])) if np.sum(causal_severe) > 0 else 0.0
        severe_f1 = float(2 * severe_precision * severe_recall / (severe_precision + severe_recall)) if (severe_precision + severe_recall) > 0 else 0.0
        
        # Multi-day forward forecast
        curr_p = causal_probs[-1]
        p_7d = cascade.predict_forward_horizon(curr_p, 7)
        p_30d = cascade.predict_forward_horizon(curr_p, 30)
        
        city_eval_results[city] = {
            "n_days": len(X_city),
            "agreement_rate": round(agreement, 4),
            "severe_drought_days_viterbi": int(np.sum(viterbi_severe)),
            "severe_drought_days_causal": int(np.sum(causal_severe)),
            "severe_drought_f1": round(severe_f1, 4),
            "severe_drought_recall": round(severe_recall, 4),
            "severe_drought_precision": round(severe_precision, 4),
            "latest_severe_drought_prob": round(float(curr_p[severe_state]), 4),
            "forward_7d_severe_prob": round(float(p_7d[severe_state]), 4),
            "forward_30d_severe_prob": round(float(p_30d[severe_state]), 4),
        }
        print(f"  {city.capitalize():10s}: Agreement = {agreement:.2%}, Severe Drought F1 = {severe_f1:.4f} (Recall = {severe_recall:.2%})")

    # 4. Historical Drought Benchmark Event Verification
    print("\nVerifying Historical Benchmark Drought Events (Nagpur - Central Dryland Cotton Belt):")
    nagpur_df = df[df["city_lower"] == "nagpur"].copy()
    X_nagpur = nagpur_df[DROUGHT_HMM_FEATURES].values.astype(np.float64)
    causal_probs_nagpur = cascade.forward_filter_probabilities(X_nagpur)
    nagpur_df["severe_drought_causal_prob"] = causal_probs_nagpur[:, cascade.severe_drought_state_]
    nagpur_df["causal_state"] = np.argmax(causal_probs_nagpur, axis=1)

    benchmark_results = []
    for bench in HISTORICAL_DROUGHT_WINDOWS:
        sub = nagpur_df[(nagpur_df["date"] >= bench["start"]) & (nagpur_df["date"] <= bench["end"])]
        if len(sub) > 0:
            mean_prob = float(sub["severe_drought_causal_prob"].mean())
            max_prob = float(sub["severe_drought_causal_prob"].max())
            severe_days = int((sub["causal_state"] == cascade.severe_drought_state_).sum())
            total_days = len(sub)
            severe_pct = float(severe_days / total_days)
            res = {
                "benchmark_name": bench["name"],
                "period": f"{bench['start']} to {bench['end']}",
                "total_days": total_days,
                "severe_drought_days": severe_days,
                "severe_drought_percentage": round(severe_pct * 100.0, 1),
                "mean_severe_prob": round(mean_prob, 4),
                "peak_severe_prob": round(max_prob, 4),
            }
            benchmark_results.append(res)
            print(f"  {bench['name']:38s}: {severe_days}/{total_days} days ({severe_pct:.1%}), Mean P = {mean_prob:.3f}, Peak P = {max_prob:.3f}")

    # 5. Save model and evaluation report
    out_dir = "artifacts/models/v1_3/drought"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "drought_cascade_v2.joblib")
    joblib.dump(cascade, model_path)
    
    full_report = {
        "model_name": "DroughtCascade V2 Causal Forward Filtering Gaussian HMM",
        "version": "v1.3.0",
        "states": STATE_NAMES,
        "features": DROUGHT_HMM_FEATURES,
        "dwell_time_statistics": dwell_stats,
        "city_evaluations": city_eval_results,
        "historical_benchmark_verifications": benchmark_results,
        "operational_mandate": (
            "Causal forward filtering P(Z_t | X_{1:t}) MUST be used for all live inference. "
            "Retrospective Viterbi decoding is restricted to offline historical regime audits."
        ),
    }
    
    out_file = "reports/v1_3/drought_cascade_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved DroughtCascade V2 evaluation report to {out_file}")
    
    return full_report


if __name__ == "__main__":
    run_drought_cascade_evaluation()
