# Clima-Core v1.2 — CPU-Only Machine Learning & Model Evolution Feasibility Audit

**Document Version:** 1.2.0  
**Date:** 2026-11-20  
**Hardware Environment:** Intel Core i3 (11th Gen), 8 GB RAM, Windows x64, CPU-only execution  
**Status:** Approved & Validated (Phase 0 Complete)

---

## 1. Executive Summary & Hardware Budget

This feasibility audit formalizes the machine learning evolution initiative for Clima-Core v1.2. Clima-Core already possesses a verified multi-hazard environmental intelligence platform with five flagship ML suites, deterministic physical modeling (SCS-CN runoff, IEEE C57 thermal derating, biometeorological indices), TreeSHAP explainability, and 13 persona adapters.

All training experiments in v1.2 are executed **strictly on local CPU hardware** under disciplined resource limits:
- **Available RAM:** 8.0 GB total system RAM.
- **Process Memory Ceiling:** Max 5.0 GB peak RAM per training process, with early garbage collection (`gc.collect()`) and explicit `float32` matrix representations.
- **Compute Constraints:** Intel Core i3 (11th Gen), maximum 2 CPU threads per process (`torch.set_num_threads(2)`, `n_jobs=2`, `thread_count=2`).
- **Data Formats:** Parquet columnar files with zero-copy PyArrow deserialization; CSV files used only where upstream source format dictates.
- **No GPU / No Cloud:** 0 CUDA dependencies, 0 cloud/Kaggle instances.

---

## 2. Dataset Inventory, Dimensions & Verification

All datasets reside on disk within `data/processed/aligned/` and `data/raw/` and have been verified with checksums and schema integrity.

| Dataset Name | File Path | Rows | Cols | Disk Size | Locations | Temporal Coverage | Missing % | Primary Target(s) |
|---|---|---|---|---|---|---|---|---|
| **FloodRisk Historical Aligned** | `data/processed/aligned/city_flood_history_aligned.parquet` | 803,550 | 29 | 13.69 MB | 50 Indian Cities | 1980-01-01 to 2023-12-31 (Daily) | 0.00% | `flood_disaster_event` (Binary) |
| **HeatRisk Train-Ready** | `data/processed/aligned/heat_risk_train_ready.parquet` | 124,350 | 76 | 14.45 MB | 50 Indian Cities | 2020-01-01 to 2026-10-18 (Daily) | 0.00% | `temperature_2m_max`, `temperature_2m_min`, `apparent_temperature_max` |
| **AirQuality Train-Ready** | `data/processed/aligned/air_quality_train_ready.parquet` | 77,000 | 101 | 17.04 MB | 50 Indian Cities | 2022-01-01 to 2026-03-20 (Daily) | 0.00% | `pm2_5_mean`, `pm10_mean`, `no2_mean` |
| **PowerGrid Electricity Stress** | `data/processed/aligned/city_daily_electricity_stress.parquet` | 150,547 | 25 | 6.65 MB | 49 Indian Cities | 2013-01-01 to 2020-12-31 (Daily) | 0.00% | `peak_met_mw`, `energy_met_mu`, `peak_unmet_mw > 0` |
| **ERA5 Soil Moisture & Drought** | `data/processed/aligned/era5_soil_moisture_50_cities.parquet` | 230,118 | 10 | 2.03 MB | 14 Regional Hubs | 1980-01-01 to 2024-12-31 (Daily) | 0.00% | `soil_moisture_0_to_7cm_mean`, `rootzone_moisture_anomaly`, `soil_moisture_deficit_index` (SMDI) |
| **Weather Historical 44Y** | `data/processed/aligned/weather_historical_44y_train_ready.parquet` | 823,048 | 22 | 31.69 MB | 50 Indian Cities | 1980-01-01 to 2025-01-15 (Daily) | 0.00% | Multi-hazard historical baseline |
| **Groundwater Quality (CGWB)** | `data/raw/groundwater_quality/cgwb_groundwater_quality_cleaned.parquet` | 18,249 | 24 | 0.82 MB | 1,758 Stations | 1961–2025 (Annual Samples) | 12.52% | Chemical parameters (`F`, `As`, `NO3`, `EC`, `pH`) |
| **Crop Profiles (AgriSense)** | `data/processed/aligned/city_district_crop_profiles.parquet` | 50 | 16 | 0.02 MB | 50 Cities | 1997–2015 Historical Baseline | 0.00% | `dominant_crop_type` (Multiclass) |

---

## 3. Target Variable Distributions & Class Imbalance

1. **FloodRisk (`flood_disaster_event`):**
   - Negative (0): 793,107 (98.70%)
   - Positive (1): 10,443 (1.30%)
   - *Imbalance Ratio:* 75.9:1. Requires scale-pos-weight / class weights, PR-AUC evaluation, and Brier score calibration.

2. **HeatRisk ($T_{\max}, T_{\min}$):**
   - $T_{\max}$: Mean 32.74°C, Std 6.01°C, Min 4.20°C, Max 49.30°C.
   - $T_{\min}$: Mean 21.67°C, Std 5.86°C, Min -2.50°C, Max 35.80°C.
   - $T_{\text{apparent},\max}$: Mean 34.61°C, Std 7.85°C, Min 3.10°C, Max 54.80°C.
   - Normal distribution with seasonal bimodality in northern cities.

3. **AirQuality ($\text{PM}_{2.5}$):**
   - $\text{PM}_{2.5}$: Mean 47.92 µg/m³, Std 42.61 µg/m³, Min 1.00 µg/m³, Max 500.00 µg/m³.
   - Severe positive skewness ($\text{skew} \approx 2.4$), heavy winter fog episodes in Indo-Gangetic Plains.

4. **PowerGrid (`peak_met_mw`):**
   - Mean 4,218.4 MW, Std 3,842.1 MW, Min 28.0 MW, Max 24,962.0 MW.
   - Contiguous daily time series per state/city, driven by temperature extremes (cooling in summer, heating in northwest winters).

---

## 4. In-Memory Memory (RAM) Footprint Analysis

To ensure execution stays strictly within the 5.0 GB laptop threshold:

| Model Matrix | Dimensions | `float64` Size | `float32` Size | Chunking / Loading Strategy |
|---|---|---|---|---|
| **FloodRisk Features** | $803,550 \times 23$ | 147.8 MB | 73.9 MB | Full matrix fits in RAM; selective column projection |
| **HeatRisk Features** | $124,350 \times 76$ | 75.6 MB | 37.8 MB | In-memory load with garbage collection |
| **AirQuality Features** | $77,000 \times 101$ | 62.2 MB | 31.1 MB | In-memory load |
| **PowerGrid Sequence Windows** | $150,547 \times 30 \times 12$ | 433.6 MB | 216.8 MB | Batch-generator streaming (Batch size 32, $\sim 1.2$ MB/batch) |
| **DroughtCascade Regime** | $230,118 \times 4$ | 7.4 MB | 3.7 MB | Light vector load, fits effortlessly |

*Conclusion:* In-memory feature matrices for individual suites require **$< 250$ MB RAM** when stored in `float32`. Even with tree leaf caches and gradient buffers, peak RAM per training job will remain **$< 1.5$ GB**, well below the 5.0 GB threshold.

---

## 5. Algorithm Feasibility on 2-Thread CPU

| Algorithm | Candidate Tasks | Feasibility | Expected Runtime | Memory Footprint | Threading Config |
|---|---|---|---|---|---|
| **LightGBM** | Baseline Champion for all suites | High | 15–45 seconds | $< 350$ MB | `n_jobs=2` |
| **CatBoost** | FloodRisk, HeatRisk, PowerGrid, AirQuality | High | 60–180 seconds | $< 600$ MB | `thread_count=2` |
| **XGBoost** | FloodRisk, HeatRisk, PowerGrid | High | 45–120 seconds | $< 500$ MB | `n_jobs=2`, `tree_method='hist'` |
| **PyTorch 1D TCN** | PowerGrid Peak Load (30-day causal window) | High | 3–6 minutes (20 epochs) | $< 400$ MB | `torch.set_num_threads(2)` |
| **Quantile Regression** | HeatRisk ($T_{\max}$), PowerGrid (Peak MW) | High | 30–60 seconds per quantile | $< 300$ MB | LightGBM pinball loss (`alpha=0.1, 0.5, 0.9`) |
| **NGBoost** | HeatRisk & PowerGrid Distributional Modeling | Medium-High | 90–240 seconds | $< 800$ MB | Natural gradient boosting, 100 trees |
| **Gaussian HMM** | DroughtCascade (3–4 environmental regimes) | High | 10–25 seconds | $< 150$ MB | `hmmlearn.hmm.GaussianHMM(n_components=4)` |
| **Mixture of Experts** | Bioclimatic Zone Specialists (5 zones) | High | 45–90 seconds total | $< 400$ MB | 5 light models trained on cluster slices |

---

## 6. Scientific Guardrails & Explicit Blocker Declarations

To maintain scientific integrity and prevent hallucinated modeling:

1. **AgriYield Prediction Blocked:**
   - *Rationale:* The available crop dataset (`india_district_crop_dataset_1997_2015.csv`) contains historical district-level total crop production and area from 1997 to 2015. There is no daily/seasonal continuous yield ground truth aligned with the 50 urban centers from 2020–2026.
   - *Action:* We do **not** train a fabricated annual crop yield regression model. The valid crop suitability classifier (predicting optimal agro-climatic crop types) is preserved.

2. **WaterQuality ML Surrogate Blocked:**
   - *Rationale:* The Indian Central Pollution Control Board (CPCB) and Bureau of Indian Standards (BIS 10500:2012) Water Quality Index is an explicit, deterministic weighted arithmetic index. An ML regressor predicting WQI introduces approximation error over an authoritative deterministic formula.
   - *Action:* The verified deterministic WQI engine remains official ground truth. Chemical exceedance classification (e.g. Fluoride $>1.5$ mg/L, Nitrate $>45$ mg/L) is permissible if requested.

3. **BasinFlow Discharge GNN Blocked:**
   - *Rationale:* HydroRIVERS provides static river reach topology, upstream catchment area, and geometric attributes. Real-time streamflow gauges (CWC discharge hydrographs in $m^3/s$) are not present in the repository.
   - *Action:* GNN discharge flow routing cannot be trained without discharge sensor observations. The SCS Curve Number event runoff engine (`src/climacore/hydrology/curve_number.py`) provides the verified deterministic runoff calculation.

4. **ForecastFusion Deep Neural Architecture Blocked:**
   - *Rationale:* Paired multi-model forecast data exists for 4 cities across only 2024. A deep multi-layer neural network would overfit immediately on this small sample.
   - *Action:* Linear/Quantile bias correction and Tree residual correction are utilized instead.

---

## 7. Experimental Tournament Protocol & Champion Acceptance Criteria

For any challenger model to replace an existing champion in the Clima-Core Model Registry, it must satisfy **all 6 criteria**:
1. **Identical Chronological Holdout:** Evaluated on the exact same test holdout period (e.g., 2022–2023 for FloodRisk, 2025–2026 for HeatRisk, 2020 for PowerGrid).
2. **Primary Metric Superiority:** Must demonstrate a statistically meaningful improvement:
   - Classification: PR-AUC or ROC-AUC improvement $\ge +0.0050$.
   - Regression: $R^2$ improvement or RMSE reduction $\ge 2\%$.
3. **Calibration & Reliability:** Brier score (classification) must not degrade; regression residuals must have near-zero mean bias ($\le 0.10$).
4. **Physical Consistency:** Monotonicity constraints must be satisfied (e.g., $Q_{10} \le Q_{50} \le Q_{90}$).
5. **Inference Latency:** Single-instance CPU inference latency must remain $< 5$ ms.
6. **Reproducibility:** Training script, hyperparameters, and evaluation metrics must be saved as JSON and reproducible from seed 42.

---

## 8. Phases 1 to 5 Execution Plan

1. **Phase 1: Advanced Tree Challengers (CatBoost & XGBoost with Optuna HPO):**
   - FloodRisk: CatBoost and XGBoost on 20 features and 23 SCS-CN features.
   - HeatRisk: CatBoost and XGBoost for $T_{\max}$ and $T_{\min}$.
   - PowerGrid: CatBoost and XGBoost for `peak_met_mw`.
   - AirQuality: CatBoost for `pm2_5_mean`.
2. **Phase 2: PyTorch CPU Temporal Convolutional Network (PowerGrid V4):**
   - 1D Causal Dilated Convolutions (kernel size 3, dilations 1, 2, 4, 8), 32 channels, 30-day historical window.
3. **Phase 3: Probabilistic, Quantile & Regime MoE:**
   - Quantile LightGBM/CatBoost ($Q_{10}, Q_{50}, Q_{90}$) with pinball loss and isotonic rearrangement.
   - NGBoost normal distribution predictions ($\mu, \sigma$) for HeatRisk and PowerGrid.
   - Mixture of Experts (MoE) partitioned across 5 bioclimatic clusters (Coastal, Plains, Plateau, Arid, Hills).
4. **Phase 4: DroughtCascade V1 (HMM Environmental Regime Discovery):**
   - Unsupervised Gaussian HMM on 44-year ERA5 Soil Moisture Deficit Index (SMDI) to identify 4 latent drought states: Normal, Developing Deficit, Severe Drought, Recovery.
5. **Phase 5: Tournament Packaging, Registry & Spec Updates:**
   - Generate `reports/v1_2/champion_challenger_comparison.json` and `reports/v1_2/resource_benchmarks.json`.
   - Compile comprehensive training report `docs/v1_2_advanced_model_training_report.md`.
   - Update `src/climacore/registry/engine.py` and `spec.md` to v1.2.0.
   - Write automated unit and integration tests in `tests/test_v1_2_models.py` and ensure 100% test pass rate.
