# Clima-Core v2.0 — Phase Zero: Complete Project Audit Report

**Audit Date:** 20 November 2026  
**Auditor:** Senior ML Research Engineer & Hydrological Validation Auditor  
**System Configuration:** Intel Core i3 (11th Gen), 8 GB RAM, Windows x64, CPU-only  
**Automated Tests:** 207 / 207 passing (100% pass rate in 138s)

---

## 1. Executive Summary

A comprehensive architectural, code, artifact, dataset, and metric verification audit was conducted across the Clima-Core repository. All **17 canonical models registered in the Model Registry** have physically verified, byte-intact serialized artifacts on disk. The test suite passes completely with zero regressions.

The primary limitation of hydrological intelligence in prior versions was the reliance on administrative flood disaster declarations (IIT Delhi IFI flood events) and uncalibrated routing, because open national telemetry feeds (India WRIS / CWC) only provided gauge stage heights ($m$), lacking continuous streamflow discharge time-series ($m^3/s$). The integration of **CAMELS-IND v2.2** represents the first opportunity to train, calibrate, and benchmark conceptual and deep learning streamflow models against genuine physical discharge observations.

---

## 2. Model Implementation & Artifact Verification Matrix

| Model ID | Flagship Suite | Algorithm / Architecture | Artifact Path | Size (KB) | Operational Status | Verified On Disk |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `flood_risk_v2` | FloodRisk | LightGBM Binary GBDT | `artifacts/models/flood_risk/v0.1.0/flood_risk_v1.booster` | 398.5 | CHAMPION | **YES** |
| `flood_risk_challenger_scs_cn` | FloodRisk | LightGBM + SCS-CN (23 feats) | `artifacts/models/flood_risk/v0.1.0/flood_risk_challenger_scs_cn.booster` | 412.7 | CHAMPION (SCS) | **YES** |
| `basinflow_v1_flood` | FloodRisk | LightGBM + BasinFlow (30 feats)| `artifacts/models/v1_3/basinflow/flood_risk_basinflow_challenger.booster` | 467.4 | CHALLENGER (Diag) | **YES** |
| `flood_risk_xgboost_23feat` | FloodRisk | XGBoost (Hist, Optuna HPO) | `artifacts/models/v1_2_challengers/floodrisk/flood_risk_xgboost_23feat.json` | 241.0 | CHALLENGER | **YES** |
| `heat_risk_v3` | HeatRisk | Dual Diurnal LightGBM | `artifacts/models/heat_risk/v0.1.0/model.txt` | 2348.1 | CHAMPION (Point) | **YES** |
| `heat_risk_quantile_suite` | HeatRisk | LightGBM Tri-Quantile (Q10/50/90)| `artifacts/models/v1_2_challengers/heatrisk/heat_risk_quantile_q_50.booster` | 578.6 | CHAMPION (Quant) | **YES** |
| `heat_risk_v5_moe` | HeatRisk | Soft-Gated Regularized MoE | `artifacts/models/v1_3/heatrisk/heatrisk_moe_v5.joblib` | 3405.2 | CO_CHAMPION (Reg) | **YES** |
| `power_grid_v2_reg` | PowerGrid | LightGBM Demand Regressor | `artifacts/models/electricity_grid/v0.1.0/peak_grid_load_v1.booster` | 623.3 | CHAMPION | **YES** |
| `power_grid_v2_clf` | PowerGrid | LightGBM Stress Classifier | `artifacts/models/electricity_grid/v0.1.0/grid_deficit_hazard_v1.booster` | 515.7 | CHAMPION | **YES** |
| `power_grid_tcn_v4` | PowerGrid | PyTorch 1D Causal Dilated TCN | `artifacts/models/v1_2_challengers/powergrid/powergrid_tcn_v4.pt` | 81.8 | CHALLENGER (Rej) | **YES** |
| `agrisense_v2` | AgriSense | LightGBM Multi-Task Crop/Yield | `artifacts/models/agrisense/v0.1.0/agrisense_crop_classifier_v1.booster` | 6714.9 | CHAMPION | **YES** |
| `air_quality_v2_pm25` | AirQuality | LightGBM Next-Day PM2.5 | `artifacts/models/air_quality/v0.1.0/model.txt` | 938.2 | CHAMPION | **YES** |
| `smokepath_v2_airquality` | AirQuality | LightGBM + FRP Advection (99f)| `artifacts/models/v1_3/smoke_transport/air_quality_smokepath_challenger.booster` | 1153.1 | CHALLENGER | **YES** |
| `drought_cascade_v1` | DroughtCascade | Gaussian HMM (4 Regimes) | `artifacts/models/v1_2_challengers/drought/drought_cascade_hmm.joblib` | 2.6 | RETROSPECTIVE | **YES** |
| `drought_cascade_v2` | DroughtCascade | Causal Forward Filtering HMM | `artifacts/models/v1_3/drought/drought_cascade_v2.joblib` | 2.9 | OPERATIONAL CHAMP| **YES** |
| `forecast_fusion_v1_emos` | ForecastFusion | EMOS (CRPS Minimized Tmax) | `artifacts/models/v1_3/forecast_fusion/tmax_model_b_emos.joblib` | 0.4 | CHAMPION (Tmax) | **YES** |
| `forecast_fusion_v1_precip`| ForecastFusion | Two-Part Hurdle Ensemble | `artifacts/models/v1_3/forecast_fusion/precip_model_d_two_part.joblib` | 425.4 | CHAMPION (Precip)| **YES** |

*Verification Summary: 17 out of 17 registered models exist, load successfully, and maintain verified SHA-256 integrity.*

---

## 3. Dataset Registry & Storage Audit

- **Total Processed Files:** 41 distinct Parquet/JSON files in `data/processed/`.
- **Core Weather & Soil Reanalysis:** `weather_historical_44y_train_ready.parquet` (31.69 MB, 44 years across 50 cities) and `era5_soil_moisture_50_cities.parquet` (2.03 MB).
- **River & Catchment Infrastructure:** `city_hydrorivers_profiles.json` (50 cities mapped to nearest reach, Strahler order, distance to ocean, upstream catchment $A$ in $\text{km}^2$).
- **Disaster Records:** `city_flood_history_aligned.parquet` (13.69 MB, 803,550 daily records, 44 years of IIT Delhi IFI disaster records).
- **Electricity & Air Quality:** `city_daily_electricity_stress.parquet` (6.65 MB), `air_quality_train_ready.parquet` (17.04 MB).
- **Dataset Registry Path Correction:** Noted that the canonical dataset registry referenced `imd_temperature_1951_2024.parquet` and `imd_rainfall_1979_2024.parquet`, whereas the actual processed files on disk are named `imd_temperature_74y_50_cities.parquet` (18.64 MB) and `imd_rainfall_50_cities.parquet` (2.23 MB). The registry paths have been noted for synchronization.

---

## 4. Reconciled Tournament Decisions & Historical Rejections

1. **FloodRisk 3-Way Blend (v1.2):**
   - *Result:* ROC-AUC reached 0.8046, but PR-AUC degraded from 0.1224 to 0.1158 ($95\%\text{ CI } [-0.0159, +0.0038]$).
   - *Verdict:* **REJECTED.** LightGBM SCS-CN (23 features) retained as sole champion.
2. **PowerGrid PyTorch TCN (v1.2):**
   - *Result:* Test $R^2 = 0.5385$ vs LightGBM Champion $0.8758$.
   - *Root Causes Unmasked:* 208:1 cross-city scale disparity pooled without city entity embeddings; lack of contemporaneous calendar conditioning; COVID-19 2020 economic lockdown demand collapse.
   - *Verdict:* **REJECTED.**
3. **HeatRisk Hard-Partitioned MoE (v1.2):**
   - *Result:* Overall MAE degraded from 1.0234°C to 1.0289°C due to sample starvation in Zones 2 and 3.
   - *Resolution in v1.3:* Soft-Gated Regularized MoE V5 resolved this with a global backbone prior + residual experts + softmax gating ($\lambda = 0.60$), achieving lower RMSE (1.4160°C vs 1.4173°C) and winning 3/5 zones.
4. **BasinFlow V1 FloodRisk Challenger (v1.3):**
   - *Result:* 30-feature challenger achieved ROC-AUC 0.7880 vs SCS-CN Champion 0.8013 on administrative disaster labels.
   - *Resolution:* Retained 23-feature model as champion; preserved BasinFlow V1 as physical diagnostic routing engine; blocked GNN modeling until streamflow observations ($m^3/s$) are obtained.

---

## 5. Transition to Phase One: BasinFlow V2 & CAMELS-IND v2.2

The previous BasinFlow engine was constrained by administrative disaster labels and lack of continuous discharge sensors. **CAMELS-IND v2.2** resolves this blocker by providing:
- Catchment-aggregated hydrometeorological forcing (precipitation, temperature, PET).
- Static geological, soil, topographic, and land cover attributes.
- **Genuine observed daily streamflow discharge ($Q_{\text{obs}}$)** across peninsular Indian catchments.

Proceeding directly to **Phase One: CAMELS-IND v2.2 Integration, Catchment Selection, Conceptual Baselines (GR4J, HBV-Light), and Mass-Conserving Neural Architectures**.
