# Clima-Core v1.3 — Model Tournament Audit & Reconciliation Report

**Document:** `docs/v1_3_tournament_reconciliation.md`  
**Date:** 2026-11-20  
**Phase:** Phase 0 (Tournament Audit and Reconciliation)  
**Hardware & Environment:** Intel Core i3 (11th Gen), 8 GB RAM, Windows x64, CPU-only execution  
**Status:** Complete, Empirically Audited & Validated  

---

## 1. Executive Summary & Purpose

This audit report reconciles the experimental results of the Clima-Core v1.2 tournament against ground-truth source datasets, model artifacts, and statistical benchmarks. In accordance with Clima-Core's core development philosophy:
- We do not promote models simply because they have higher algorithm complexity or nominal novelty.
- Every challenger must be evaluated on identical chronological holdout observations.
- All blend weights, threshold tunings, and calibration parameters must be derived exclusively from training and validation splits—never on the test holdout.
- Where a challenger fails to demonstrate statistically significant or multidimensional superiority over the existing champion, the champion is strictly preserved.

---

## 2. Section 0.1 — FloodRisk Three-Way Ensemble Audit

### 2.1 The Problem Statement
The v1.2 tournament reported an ROC-AUC of 0.8073 for an ad-hoc 3-way blend of LightGBM SCS-CN (40%), CatBoost (30%), and XGBoost (30%), but omitted PR-AUC, calibration error, and statistical significance testing.

### 2.2 Empirical Audit on 2022–2023 Holdout (36,500 Observations, 1.30% Disaster Prevalence)

We re-evaluated the individual models, derived optimal blend weights using **validation data only (2019–2021)** via constrained Brier loss optimization, and benchmarked both static and learned blends:

| Model / Ensemble | ROC-AUC | PR-AUC | Brier Score | Log Loss | Precision@0.35 | Recall@0.35 | F1@0.35 | ECE (10-bin) |
|---|---|---|---|---|---|---|---|---|
| **LightGBM SCS-CN (Champion)** | 0.8011 | **0.1224** | 0.1354 | **0.3995** | 0.0311 | 0.7324 | 0.0596 | **0.2309** |
| **XGBoost Optuna Challenger** | 0.8000 | 0.1086 | 0.1346 | 0.4049 | 0.0315 | 0.7370 | 0.0605 | 0.2395 |
| **CatBoost Challenger** | 0.8003 | 0.0654 | 0.2137 | 0.6198 | 0.0121 | 1.0000 | 0.0239 | 0.4477 |
| **Static Blend (40% LGB, 30% CB, 30% XGB)**| 0.8067 | 0.1149 | 0.1368 | 0.4327 | 0.0291 | 0.7574 | 0.0560 | 0.2985 |
| **Validation-Learned Blend (43.7% LGB, 14.5% CB, 41.8% XGB)** | **0.8046** | 0.1158 | **0.1329** | 0.4103 | 0.0304 | 0.7415 | 0.0584 | 0.2659 |

### 2.3 Statistical Significance & Bootstrap Analysis (1,000 Paired Resamples)
- **$\Delta \text{ROC-AUC}$ (Learned Blend vs Champion):** Mean $+0.0035$, 95% Confidence Interval: $[+0.0004, +0.0068]$, $p = 0.0120$.
- **$\Delta \text{PR-AUC}$ (Learned Blend vs Champion):** Mean $-0.0066$, 95% Confidence Interval: $[-0.0159, +0.0038]$, spanning zero.
- **Root-Cause Analysis:** CatBoost's balanced class weighting produced over-confident probability estimates (Brier score 0.2137, ECE 0.4477). While ensembling increases the ranking concordance (ROC-AUC), it degrades positive predictive precision (PR-AUC drops from 0.1224 to 0.1158).
- **Governance Decision:** **REJECT PROMOTION.** Retain `LightGBM SCS-CN` as the sole operational Champion. In high-imbalance disaster classification (1.3% positive), precision in the extreme upper tail is clinically more critical than small global ROC-AUC shifts. The learned blend is retained as an experimental challenger.

---

## 3. Section 0.2 — HeatRisk NGBoost Distributional Model Audit

### 3.1 The Problem Statement
The v1.2 tournament reported that NGBoost achieved 79.54% empirical coverage for an 80% nominal interval with an average predicted $\sigma = 1.25^\circ\text{C}$, but did not evaluate NLL, CRPS, city-level variation, or extreme heat performance.

### 3.2 Quantitative Evaluation on 2026 Test Holdout (15,100 Samples across 50 Cities)
- **Negative Log-Likelihood (NLL):** $1.7271$
- **Continuous Ranked Probability Score (CRPS):** $0.7625$
- **Nominal Coverage:** $80.00\%$ | **Empirical Coverage:** **$79.54\%$**
- **Average Predicted $\sigma$:** $1.250^\circ\text{C}$ | **Average Interval Width ($2 \times 1.2815\sigma$):** $3.20^\circ\text{C}$

#### Granular Slices & Stress Tests:
1. **Spatial Variation across 50 Cities:**
   - Minimum coverage city: **Srinagar** ($71.19\%$, high elevation mountain valley lapse rates).
   - Maximum coverage city: **Nashik** ($83.77\%$).
   - Mean city coverage: $79.54\%$, Standard deviation across cities: $\pm 2.84\%$.
2. **Seasonal Coverage Breakdown:**
   - Winter (Jan–Feb): $81.93\%$
   - Pre-Monsoon / Summer Heat (Mar–May): $76.72\%$ (higher forecast spread during heatwaves).
   - Monsoon (Jun–Sep): $79.90\%$
   - Post-Monsoon (Oct): $82.07\%$
3. **Extreme Temperature Performance ($T_{\max} \ge 38^\circ\text{C}$ and $T_{\max} \ge 40^\circ\text{C}$):**
   - Observations $\ge 38^\circ\text{C}$ ($N = 1,351$): Empirical coverage is **$84.68\%$**.
   - Observations $\ge 40^\circ\text{C}$ ($N = 653$): Empirical coverage is **$85.15\%$**.
   - The model dynamically expands $\sigma$ during extreme heatwaves, preventing under-coverage when health risks are highest.

### 3.3 Comparison with LightGBM Quantile Regressors ($Q_{10}, Q_{50}, Q_{90}$)
| Metric | LightGBM Quantile Suite | NGBoost Normal Model |
|---|---|---|
| Empirical 80% Coverage | 76.82% | **79.54%** |
| Mean Interval Width | **3.01°C** | 3.20°C |
| Crossing Rate | **0.00%** (Isotonic Sorted) | **0.00%** (Analytic Parametric) |
| Output Capabilities | Discrete quantiles only | Full continuous PDF $\mathcal{N}(\mu, \sigma^2)$, tail exceedance, CRPS |
| Inference Latency | **< 1 ms** | ~3 ms |

- **Governance Decision:** **CO-CHAMPION APPOINTMENT.** Retain LightGBM as the deterministic Point Champion ($R^2 = 0.9510$, MAE $1.0234^\circ\text{C}$). Promote `heat_risk_quantile_suite` as the primary operational fast quantile model, and designate `heat_risk_ngboost` as the specialized Continuous Distributional Engine for probabilistic heatwave threshold alerts.

---

## 4. Section 0.3 — HeatRisk Mixture of Experts (MoE) Audit

### 4.1 The Problem Statement
The v1.2 report claimed modest coastal improvement for MoE but omitted regional sample sufficiency and non-coastal performance.

### 4.2 Comprehensive 5-Zone Breakdown (15,100 Test Samples)

| Bioclimatic Zone | $N_{\text{train}}$ | $N_{\text{test}}$ | Specialist MAE | Global Champion MAE | MAE Delta | Specialist $R^2$ | Global $R^2$ | Superior Model |
|---|---|---|---|---|---|---|---|---|
| **Zone 0: Humid Tropical Coastal** | 18,150 | 3,020 | **0.8739°C** | 0.8816°C | **-0.0077°C** | 0.8274 | 0.8275 | **Specialist** |
| **Zone 1: Deccan Plateau Corridor** | 12,705 | 2,114 | **0.8560°C** | 0.8571°C | **-0.0011°C** | 0.9051 | 0.9060 | **Specialist** |
| **Zone 2: Indo-Gangetic Plains & East**| 23,595 | 3,926 | 1.1466°C | **1.1357°C** | +0.0109°C | 0.9182 | **0.9199** | **Global** |
| **Zone 3: Semi-Arid & Central Corridor**| 18,150 | 3,020 | 0.9680°C | **0.9446°C** | +0.0234°C | 0.9296 | **0.9332** | **Global** |
| **Zone 4: Sub-Himalayan & Montane** | 18,150 | 3,020 | **1.2129°C** | 1.2143°C | **-0.0014°C** | **0.9573** | 0.9568 | **Specialist** |
| **Overall National Holdout** | 90,750 | 15,100 | 1.0289°C | **1.0234°C** | +0.0055°C | 0.9503 | **0.9510** | **Global** |

### 4.3 Diagnostic Findings:
1. **Peripheral vs Continental Asymmetry:** Regional models outperform the global baseline only in peripheral or geographically decoupled regimes:
   - Coastal maritime climates (low diurnal range, sea-breeze moderated).
   - Montane / high elevation valleys (steep lapse rates).
2. **Sample Starvation in Continental Plains:** Hard-partitioning the training data reduces continental training records from 90,750 to 18,000–23,000, causing higher variance in the Indo-Gangetic plains and central arid corridors where cross-city synoptic weather systems are shared.
3. **Governance Decision:** **REJECT PROMOTION.** Keep the Global LightGBM model as operational Champion ($R^2 = 0.9510$, MAE $1.0234^\circ\text{C}$). Transition MoE to Phase 4 development (`HeatRisk V5`) focusing on soft gating, hierarchical global backbone conditioning, and regional residual adaptation.

---

## 5. Section 0.4 — PowerGrid Temporal Convolutional Network (TCN) Root-Cause Audit

### 5.1 The Problem Statement
The PyTorch CPU 1D Causal Dilated TCN achieved an $R^2$ of only 0.5385 (MAE 3,048.58 MW) compared to LightGBM's $R^2 = 0.8758$ (MAE 1,519.18 MW).

### 5.2 Root-Cause Deconstruction
1. **Severe Grid Scale Disparity without Entity Embeddings:**
   - In `city_daily_electricity_stress.parquet`, city-level mean peak met demand spans from **90.5 MW** (small Union Territories) to **18,810.3 MW** (Maharashtra industrial hubs)—a scale ratio of **207.9 to 1**.
   - LightGBM handles this via non-linear decision tree thresholding on population, city category codes, and historical baseline lags.
   - The TCN used a single global `StandardScaler` across all 49 cities without learnable city entity embeddings or state baseline offsets, forcing the convolutional filters to model both micro-load and macro-load cities simultaneously with the same weight tensor.
2. **Calendar vs Dynamic Autoregression Mismatch:**
   - Grid power demand in India is strongly driven by day-of-week industrial work patterns and agricultural power dispatch schedules.
   - LightGBM conditions on explicit day-of-week and month indicators. The TCN lookback sequence relied primarily on meteorology convolutions (`temp_min`, `temp_max`, `cooling_degree_days`).
3. **Macro-Economic Shock in 2020 Holdout:**
   - The test set covers 2020 (national COVID-19 lockdown months), where commercial/industrial power demand collapsed by 25–40% independently of weather. Pure autoregressive weather models heavily overpredicted demand.
4. **Governance Decision:** **RETAIN AS REJECTED / EXPERIMENTAL CHALLENGER.** The TCN validated CPU feasibility (444s, 843 MB RAM), but will remain an experimental benchmark until state-level entity embeddings and economic/holiday conditioning are introduced.

---

## 6. Section 0.5 — DroughtCascade HMM Audit & Causal Forward Filtering

### 6.1 State Characterization across 230,118 Station-Days (14 Hubs, 1980–2024)

| State ID | Physical Interpretation | Frequency | Mean SM (0–7cm) | Median SM | Mean SMDI | Median SMDI | Rootzone Anomaly | Mean Soil Temp | Mean Dwell | $P_{ii}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| **State 0** | Sub-Surface Depletion / Moderate Stress | 25.06% | 0.1746 | 0.146 | -0.0038 | 0.000 | -0.0005 | 30.26°C | 35.1 d | 0.9699 |
| **State 1** | Wet / Monsoon Moisture Surplus | 33.26% | 0.3773 | 0.389 | **+0.1726** | **+0.128** | **+0.0375** | 26.43°C | 27.0 d | 0.9608 |
| **State 2** | **Severe Rootzone Drought Cascade** | 13.36% | 0.2637 | 0.267 | **-0.2874** | **-0.263** | **-0.0649** | 29.52°C | 19.0 d | 0.9440 |
| **State 3** | Post-Monsoon Equilibrium / Transition | 28.31% | 0.1728 | 0.157 | +0.0140 | +0.003 | +0.0039 | 25.93°C | 21.1 d | 0.9501 |

- **Verification of State 2:** State 2 exhibits the lowest mean SMDI ($-0.2874$) and the most severe negative rootzone moisture anomaly ($-0.0649$), confirming that it represents **true severe agricultural rootzone drought**.

### 6.2 Causal Forward Filtering Verification: $P(Z_t \mid X_{1:t})$ vs $P(Z_t \mid X_{1:T})$
- In operational real-time forecasting, the future trajectory $X_{t+1:T}$ is unavailable. Retrospective Viterbi decoding causes lookahead data leakage.
- We implemented and audited the strict causal forward-variable recursion:
  $$\alpha_t(j) = p(x_t \mid Z_t=j) \sum_{i=1}^K \alpha_{t-1}(i) A_{ij}, \quad P(Z_t=j \mid X_{1:t}) = \frac{\alpha_t(j)}{\sum_k \alpha_t(k)}$$
- Evaluated on 16,437 daily records for Delhi (1980–2024):
  - **Causal Forward Filter vs Retrospective Viterbi Agreement:** **$93.93\%$**.
  - The $6.07\%$ disagreement occurs exclusively at regime transition inflection points, where retrospective smoothing incorporates future confirmation.
- **Operational Rule for Clima-Core v1.3:** Live inference services must strictly execute causal forward filtering $P(Z_t \mid X_{1:t})$, while Viterbi smoothing is restricted to historical archive cataloging.

---

## 7. Master Reconciliation Summary & Next Phase Directives

| Audit Area | v1.2 Reported Finding | v1.3 Reconciliation Verdict | Action Taken |
|---|---|---|---|
| **FloodRisk Ensemble** | ROC-AUC 0.8073 (Ad-hoc weights) | Validated ROC-AUC 0.8046, but PR-AUC degraded (0.1158 vs 0.1224) | Retain `LightGBM SCS-CN` as sole Champion; reject ensemble promotion. |
| **HeatRisk NGBoost** | 79.54% empirical coverage | Validated: NLL 1.7271, CRPS 0.7625, 85.15% coverage in $>40^\circ\text{C}$ heat | Promote to Continuous Distributional Co-Champion. |
| **HeatRisk MoE** | Modest coastal improvement | Validated coastal/montane lift, but degraded in central plains; overall MAE worse | Retain Global Champion; redesign MoE in Phase 4 with soft gating. |
| **PowerGrid TCN** | $R^2 = 0.5385$ | Unmasked scale disparity (208:1) and lack of city entity embeddings | Retain as experimental rejected challenger; retain LightGBM Champion. |
| **DroughtCascade HMM**| 4 latent states identified | State 2 confirmed as severe deficit; verified causal forward filter (93.93% agreement) | Mandate causal forward filtering for all live API inference. |

*Phase 0 is complete. Proceeding immediately to Phase 1: ForecastFusion V1.*
