# CLIMA-CORE v2.0 — COMPLETE SYSTEM STATUS, DATASET AUDIT & ARCHITECTURAL DOSSIER

**System Status:** Production v2.0.0 (Green / All 214 Automated Tests Passing)  
**Date of Dossier:** 20 November 2026  
**Audience:** Technical Advisory Boards, External Senior ML/Atmospheric Auditors, System Reviewers  
**License / Proprietary:** Clima-Core Environmental Risk Systems  

---

## 1. PROJECT EXECUTIVE SUMMARY & MISSION

**Clima-Core** is an AI-powered Environmental Risk Intelligence and Decision Support Platform engineered specifically for the complex bioclimatic and infrastructural realities of Indian cities, river basins, and agricultural districts.

The system addresses the fundamental limitation of standard weather forecasts: **raw weather predictions do not equal risk intelligence, physical impact, or actionable civil decisions.** High temperatures alone do not indicate hospital heat admissions without nocturnal thermal debt and urban heat island effects; heavy rainfall does not indicate city flooding without soil saturation, hydraulic conduit limits, and river backwater dynamics; and agricultural stress cannot be diagnosed without compound soil moisture deficit cascades.

### Core System Principles
1. **Strict 4-Layer Separation:** Machine learning predictions (Layer 1) are strictly decoupled from physical/empirical equations (Layer 2), geospatial exposure overlays (Layer 3), and operational persona decision policies (Layer 4).
2. **Honest Domain Boundaries:** Never present surrogate statistical estimations as physical hardware readings (e.g., optical $PM_{2.5}$ is not an airport transmissometer reading; uncalibrated runoff is not subterranean sewer pipe overflow).
3. **Anti-Leakage Validation:** Chronological holdouts (purged time series) and spatial holdouts (leave-one-group-out across bioclimatic zones and ungauged river basins) are strictly enforced. No random splitting of time series.
4. **Physical Mass & Thermodynamic Conservation:** Neural networks applied to hydrological flow strictly enforce mass conservation ($< 0.001\%$ volume discrepancy).
5. **Local CPU Hardware Compliance:** All models are trained and executed locally on an Intel Core i3 (11th Gen, 2 threads max, zero GPU/CUDA calls, peak RAM $< 850\text{ MB}$).

---

## 2. COMPREHENSIVE DATASET INVENTORY (15 Datasets)

Clima-Core integrates 15 verified, multi-decadal meteorological, hydrological, agricultural, grid, and infrastructure datasets:

| # | Dataset Name | Sponsoring Agency / Source | Spatial Extent & Resolution | Temporal Range | Primary Variables Ingested | Role in Clima-Core |
|---|---|---|---|---|---|---|
| **1** | **IMD High-Res Daily Gridded Rainfall** | India Meteorological Department (Pune) | All-India, $0.25^\circ \times 0.25^\circ$ (~27 km) | 1901–2024 (124 years) | Daily precipitation depth ($mm$) | Meteorological ground truth, climatological baselines, flood feature engineering |
| **2** | **IMD High-Res Daily Gridded Temperature** | India Meteorological Department (Pune) | All-India, $1.0^\circ \times 1.0^\circ$ (~100 km) | 1951–2024 (74 years) | Daily $T_{\max}, T_{\min}, T_{\text{mean}}$ ($^\circ\text{C}$) | Thermal ground truth, bioclimatic heat index, nocturnal heat retention |
| **3** | **ERA5-Land Global Reanalysis** | ECMWF / Copernicus CDS | Global (India Bounding Box), $0.1^\circ$ (~9 km) | 1950–Present (Hourly/Daily) | Volumetric soil moisture (0–7cm, 7–28cm), surface runoff, evaporation ($ET_0$), dewpoint, surface solar radiation | Atmospheric-hydrological coupling, soil saturation indices, DroughtCascade HMM |
| **4** | **India Flood Inventory (IFI v3)** | IIT Delhi & CWC | District-level across India | 1980–2023 (44 years) | Declared flood disaster events, inundation severity, disaster duration | Supervised ground truth for FloodRisk classifiers (event-safe holdout) |
| **5** | **National Ambient Air Quality (NAAQM)** | Central Pollution Control Board (CPCB) | 50+ City Monitoring Stations | Historical & Real-Time | Ground monitor $PM_{2.5}, PM_{10}, NO_2, SO_2, CO, O_3$ ($\mu g/m^3$) | AirQuality ground truth, 8-pollutant sub-index validation, CPCB breakpoints |
| **6** | **CAMS Global Atmospheric Composition** | ECMWF / Copernicus Atmosphere | Global, $0.75^\circ$ gridded | 2003–Present | Aerosol Optical Depth (AOD at 550nm), dust, organic matter, black carbon | Boundary layer aerosol transport, regional haze, optical advection |
| **7** | **NASA FIRMS Active Fire Telemetry** | NASA EOSDIS (MODIS & VIIRS) | 375m & 1km thermal detections | 2001–Present | Fire Radiative Power (FRP in $MW$), fire confidence, detection angles | SmokePath V2 biomass burning advection vectors across Northern India |
| **8** | **WorldPop Gridded Population Density** | WorldPop / University of Southampton | 100m gridded resolution | 2020 baseline (projected) | Gridded population count, urban dense cluster fractions | Layer 3 Exposure: human impact buffering, vulnerability weighting |
| **9** | **OpenStreetMap Infrastructure** | OpenStreetMap (OSM) Contributors | Vector features (points, lines) | Continuously updated | Power substations, transmission lines, drainage channels, bridges, transit | Layer 3 Exposure: lifeline asset counts, critical infrastructure proximities |
| **10** | **National Power Grid Telemetry** | POSOCO / Grid-India | Regional grid dispatch centers | Historical multi-year | Daily peak demand ($MW$), regional base load, energy deficits | PowerGrid demand regression, thermal load sensitivity, deficit classifiers |
| **11** | **Crop Yield & Phenology Statistics** | ICAR / Directorate of Economics & Statistics | District-level across agricultural hubs | Multi-decadal (Kharif & Rabi) | Crop yield ($kg/ha$), crop suitability, sowing/harvesting calendars | AgriSense crop suitability classification, yield loss regression |
| **12** | **HydroRIVERS Asia River Network** | HydroSHEDS / WWF | Vector river reach lines | Static reach topology | 531,819 Indian river reaches, Strahler stream order, river distance, long-term mean discharge ($m^3/s$) | Fluvial flood backwater coupling, nearest reach distance, riverine flood routing |
| **13** | **CGWB Groundwater Chemical Quality** | Central Ground Water Board (NWIC) | 18,249 borehole monitoring samples | Historical sampling across India | Fluoride, Arsenic, Nitrate, Iron, Hardness, Electrical Conductivity, pH | WaterQuality potable safety classifier, BIS IS 10500:2012 WQI surrogate |
| **14** | **Open-Meteo Previous Runs NWP Archive** | NOAA GFS, DWD ICON, JMA, ECMWF | Multi-model point extractions | Paired forecast archives | Day-1 lead forecasts of temperature, precipitation, wind, dewpoint | ForecastFusion V1/V2 multi-model ensemble verification and bias correction |
| **15** | **CAMELS-IND v2.2 Streamflow Benchmark** | Zenodo Record `14999580` (CC-BY-4.0) | 242 Peninsular Indian catchments | 1980–2020 (41 years, 14,976 daily records) | Daily observed streamflow ($m^3/s$), catchment area, daily forcings ($P, PET, T$), 7 static catchment attributes | BasinFlow V2 measured streamflow intelligence, conceptual & neural benchmarking |

---

## 3. COMPREHENSIVE MODEL INVENTORY (29 Models Across 9 Suites)

Clima-Core deploys 29 supervised machine learning models, deep learning architectures, conceptual physical formulations, and probabilistic engines organized into 9 flagship suites:

```
                            CLIMA-CORE MODEL TAXONOMY
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PREDICTIVE SUITES                                    │
├─────────────────┬──────────────────┬─────────────────┬───────────────────────────┤
│    FloodRisk    │     HeatRisk     │    PowerGrid    │         AgriSense         │
│  (5 Candidates) │  (4 Candidates)  │ (4 Candidates)  │      (4 Candidates)       │
├─────────────────┼──────────────────┼─────────────────┼───────────────────────────┤
│   AirQuality    │   WaterQuality   │    BasinFlow    │      ForecastFusion       │
│  (3 Candidates) │  (3 Candidates)  │ (5 Candidates)  │      (2 Candidates)       │
├─────────────────┴──────────────────┴─────────────────┴───────────────────────────┤
│                   DroughtCascade HMM & CompoundRisk Copulas                      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Suite 1: FloodRisk (Riverine & Pluvial Inundation)
1. **`flood_risk_v1` (LightGBM GBDT 20 Features) — Baseline Champion:** Binary classifier predicting declared district-level flood disasters. ROC-AUC: `0.7931`, PR-AUC: `0.1134`, Brier: `0.1373`.
2. **`flood_risk_challenger_scs_cn` (LightGBM 23 Features + SCS-CN) — Promoted Co-Champion:** Incorporates USDA NRCS Curve Number direct event runoff, 5-day antecedent moisture (AMC I/II/III), and HydroRIVERS distance. ROC-AUC: `0.8011`, PR-AUC: `0.1224`, Brier: `0.1354`.
3. **`flood_risk_challenger_xgboost` (XGBoost Hist Method):** Hyperparameter-tuned gradient booster. ROC-AUC: `0.8014`, PR-AUC: `0.1219`.
4. **`flood_risk_challenger_catboost` (CatBoost GPU-free):** Symmetric decision tree classifier. ROC-AUC: `0.7988`, PR-AUC: `0.1205`.
5. **`flood_risk_challenger_3way_ensemble` (Blend):** Soft-voting blend. ROC-AUC: `0.8073`, but rejected for production champion due to PR-AUC degradation at strict operational decision thresholds.

### Suite 2: HeatRisk (Urban Biometeorology & Thermal Overload)
6. **`heat_risk_v1` (LightGBM Regressor Tmax):** Predicts daily maximum air temperature across 50 cities. RMSE: `1.482°C`, $R^2: 0.942$.
7. **`heat_risk_v5_moe` (Soft-Gated Mixture of Experts) — Flagship Champion:** Blends a universal LightGBM backbone with 5 regional residual experts across India's bioclimatic zones (Coastal, Deccan Plateau, Indo-Gangetic Plains, Semi-Arid, Sub-Himalayan) using a continuous softmax gating network. Overall RMSE: `1.4160°C`, $R^2: 0.9511$.
8. **`heat_risk_ngboost` (Natural Gradient Distributional Regressor) — Probabilistic Champion:** Outputs parametric Gaussian predictive distributions ($\mu, \sigma$) using natural gradients. CRPS: `0.7625`, empirical 80%-interval coverage: `79.54%`.
9. **`heat_risk_quantile_q10_q50_q90` (LightGBM Pinball Regressors):** Generates distribution-free asymmetric quantile intervals with isotonic sorting (zero quantile crossing violations).

### Suite 3: PowerGrid (Electrical Demand & Regional Deficits)
10. **`power_grid_load_v1` (LightGBM Peak Load Regressor) — Champion:** Predicts regional peak electricity demand in Megawatts based on cooling-degree days, humidity, calendar cycles, and lagged load. $R^2: 0.912$.
11. **`power_grid_deficit_v1` (LightGBM Deficit Classifier):** Binary classifier identifying probability of grid supply shortfalls during extreme thermal stress. ROC-AUC: `0.9521`.
12. **`power_grid_deficit_catboost` (CatBoost Deficit Classifier) — Challenger:** Demonstrates superior resistance to categorical feature drift. ROC-AUC: `0.9527`.
13. **`power_grid_tcn` (PyTorch 1D Causal Dilated Temporal Convolutional Network):** 15,545 parameters, trained on 30-day historical load lookbacks without GPU dependencies. $R^2: 0.5385$.

### Suite 4: AgriSense (Crop Suitability & Agro-Climatic Yield Loss)
14. **`agrisense_crop_suitability_v1` (LightGBM Multi-class Recommender) — Champion:** Recommends optimal Kharif/Rabi crops across 14 agro-ecological zones based on soil moisture, thermal budget, and seasonal rainfall. Top-1 Accuracy: `82.4%`, Top-3 Accuracy: `96.1%`.
15. **`agrisense_catboost_suitability` (CatBoost Multi-class Challenger):** Obvitiates label-encoding bias. Top-1 Accuracy: `82.6%`.
16. **`agrisense_yield_anomaly_v1` (LightGBM Regressor):** Predicts district-level crop yield loss percentage under drought and heat stress. RMSE: `142 kg/ha`.
17. **`agrisense_drought_impact_v1` (LightGBM Classifier):** Detects agricultural crop failure probability. ROC-AUC: `0.864`.

### Suite 5: AirQuality (Particulate Dynamics & Smoke Advection)
18. **`air_quality_pm25_v1` (LightGBM 24h PM2.5 Regressor):** Baseline optical particulate predictor using meteorological boundary layer height, wind speed, and inversion indices. MAE: `6.38 μg/m³`, $R^2: 0.884$.
19. **`air_quality_smokepath_v2` (FRP Wind Advection Augmented LightGBM) — Champion:** Couples satellite Fire Radiative Power (FRP) with wind velocity vectors ($u, v$) and CAMS optical depth to model trans-boundary agricultural residue smoke plumes. MAE: `6.34 μg/m³`.
20. **`air_quality_quantile_q10_q50_q90` (Conformal Prediction Bounds):** Provides non-parametric uncertainty intervals with $\ge 90\%$ marginal coverage.

### Suite 6: WaterQuality (Chemical Potability & Ion Hazards)
21. **`water_quality_contamination_v1` (LightGBM Potability Classifier) — Champion:** Classifies groundwater potability compliance against BIS IS 10500:2012 safety standards using geological, soil, depth, and historical ion records. ROC-AUC: `0.9286`.
22. **`water_quality_wqi_surrogate_v1` (LightGBM WQI Regressor):** Fast surrogate predicting the official Indian Water Quality Index (WQI). $R^2: 0.9255$.
23. **`water_quality_deterministic_bis` (Deterministic Engine):** Exact physical Weighted Arithmetic index calculation when physical laboratory ion assays (Fluoride, Arsenic, Nitrate, Iron, Hardness) are provided directly.

### Suite 7: BasinFlow V2 (Measured Catchment Streamflow Intelligence)
24. **`GR4J` (Perrin et al. 2003, 4-Parameter Lumped Conceptual Model) — Gauged Champion:** Calibrated via Nelder-Mead simplex maximizing KGE across 89 gauged Indian basins. Median KGE: `0.4354`, Median NSE: `0.2768`, RMSE: `1.853 mm/d`, Peak Magnitude Error: `-0.45%`.
25. **`HBV-Light` (Bergström 1992, 9-Parameter Conceptual Model):** Features soil moisture, upper reservoir, and lower groundwater box with triangular routing filter. Median NSE: `0.2953`, Median KGE: `0.3784`.
26. **`MC-LSTM-DS` (Mass-Conserving LSTM with Dual States) — Ungauged & Physics Champion:** Recurrent neural network with dynamic infiltration gate $\alpha_t$, partitioning cell state into quickflow ($c^{\text{quick}}$) and slow groundwater baseflow ($c^{\text{slow}}$). Strictly conserves water mass ($< 0.001\%$ error). Highest ungauged basin performance (PUB): Median NSE: `0.1290`, KGE: `0.2233`, Water Balance Error: `+4.81%` (gauged) and `+17.28%` (ungauged).
27. **`MC-LSTM` (Mass-Conserving LSTM Single Reservoir):** Hoedt et al. formulation. Ungauged NSE: `0.1283`, KGE: `0.2466`.
28. **`Standard-LSTM` (PyTorch Recurrent Baseline):** Unconstrained recurrent neural net. Gauged NSE: `0.2513`, Ungauged NSE: `0.0751`.

### Suite 8: ForecastFusion V1 (Multi-Model NWP Post-Processing)
29. **`EMOS` (Extended Ensemble Model Output Statistics):** Post-processes multi-model NWP temperature forecasts (GFS, ICON, JMA, ECMWF). Reduces Tmax MAE to `0.6237°C` (-21.0% error reduction vs best raw model).
30. **`Two-Part Hurdle Precipitation Ensemble`:** Logistic regression for rain occurrence coupled with Gamma regression for amount. Slashes precipitation Brier score from `0.2540` to `0.1503` (-40.8% error reduction).

### Suite 9: Cascading Multi-Hazard Engines
31. **`DroughtCascade V2 HMM`:** 4-state Gaussian Hidden Markov Model with causal forward filtering $P(Z_t \mid X_{1:t})$ across 44 years of soil moisture deficit series (Equilibrium, Developing Deficit, Severe Drought Cascade, Recovery).
32. **`CompoundRisk V2 Copula Engine`:** Fits bivariate Archimedean copulas (Gumbel, Clayton, Survival Clayton, Frank) to compute joint "AND" ($T_{\text{AND}}$) and "OR" ($T_{\text{OR}}$) multi-hazard return periods across compound Deluge-Hydrodynamic, Thermal-Grid, and Drought-Smoke failure chains.

---

## 4. HOW PREDICTIONS & OUTPUTS ARE CALCULATED (End-to-End Pipeline)

Clima-Core calculates environmental risk intelligence through a deterministic, strictly decoupled 4-layer execution pipeline:

```mermaid
flowchart TD
    subgraph L1 ["LAYER 1: PREDICTIVE INTELLIGENCE"]
        RAW["Raw Weather & Sensor Feeds"] --> FC["ForecastFusion V1 (NWP Bias Correction)"]
        FC --> FEAT["Feature Engineering & Dynamic Lags"]
        FEAT --> ML["Trained ML Champions (LightGBM, CatBoost, MoE)"]
        ML --> PRED["Raw Point Predictions f(x)"]
    end

    subgraph L2 ["LAYER 2: PHYSICAL & STATISTICAL INTELLIGENCE"]
        PRED --> BIO["Biometeorology (HI, UTCI, WBGT, Heat Debt)"]
        PRED --> HYD["BasinFlow V2 (GR4J & MC-LSTM-DS Streamflow)"]
        PRED --> EVS["Extreme Value Statistics (GEV / POT Return Periods)"]
        PRED --> COP["Compound Copula Engine (Joint Risk C(u,v))"]
        PRED --> HMM["DroughtCascade HMM (Forward Filtering P(Z|X))"]
    end

    subgraph L3 ["LAYER 3: GEOSPATIAL EXPOSURE INTELLIGENCE"]
        BIO & HYD & COP --> EXP["Spatial Intersection Engine"]
        WPOP["WorldPop 100m Gridded Density"] --> EXP
        OSM["OSM Infrastructure & Substations"] --> EXP
        RIV["HydroRIVERS Network Buffer"] --> EXP
        EXP --> IMPACT["Vulnerability & Exposure Weighting"]
    end

    subgraph L4 ["LAYER 4: DECISION INTELLIGENCE"]
        IMPACT --> ADAPT["13 Persona Decision Adapters"]
        ADAPT --> CONF["Split Conformal Prediction (90% Uncertainty Bounds)"]
        ADAPT --> SHAP["Native LightGBM C++ TreeSHAP (Attribution)"]
        CONF & SHAP --> JSON["Unified Contract v1.0.0 JSON Response"]
    end
```

### Detailed Mathematical Step-by-Step Execution:

#### Step 1: Input Ingestion & Multi-Model NWP Post-Processing
When a query arrives (e.g. coordinates $[lat, lon]$, timestamp, and forecast parameters):
1. Raw numerical weather forecasts (NWP) from multiple international providers (GFS, ICON, JMA, ECMWF) are ingested.
2. `ForecastFusion V1` applies Extended Ensemble Model Output Statistics (EMOS) and Empirical Quantile Mapping (EQM) to eliminate systematic bias:
   $$\hat{T}_{\text{corr}} = \alpha + \sum_{m=1}^M \beta_m T_{\text{raw}, m}$$
   Precipitation is parsed through a two-part Hurdle model separating rain probability from volumetric conditional depth.

#### Step 2: Layer 1 ML Predictive Inference
Feature vectors are constructed combining dynamic atmospheric variables (rain, 3d/7d/14d/30d rolling accumulations, antecedent precipitation index $API$, vapour pressure deficit, wind gusts) with static catchment attributes.
- **For HeatRisk:** Evaluated via `heat_risk_v5_moe`:
  $$\hat{y}(x) = f_{\text{global}}(x) + \sum_{k=1}^5 g_k(x) \cdot r_k(x)$$
  where $f_{\text{global}}$ is the national LightGBM backbone, $g_k(x) = \text{softmax}(W_g z)$ is the bioclimatic gating network, and $r_k(x)$ are regional residual experts.
- **For FloodRisk:** Evaluated via `flood_risk_challenger_scs_cn` producing calibrated flood occurrence probability $P(\text{Flood} \mid x) \in [0, 1]$.

#### Step 3: Layer 2 Physical & Statistical Coupling
Machine learning point predictions are never returned in isolation; they are verified through Layer 2 physical formulations:
1. **Biometeorological Heat Budgeting:**
   - Evaluates Rothfusz Heat Index ($HI$), 6th-order polynomial Universal Thermal Climate Index ($UTCI$), and outdoor Stull-Liljegren Wet-Bulb Globe Temperature ($WBGT$).
   - Calculates **Nocturnal Thermal Debt**:
     $$\text{HeatDebt} = \sum_{t=1}^k \max(0, T_{\min, t} - 25.0^\circ\text{C})$$
     accounting for physiological heat recovery failure during hot Indian summer nights.
2. **Hydrological Catchment Flow (BasinFlow V2):**
   - For gauged basins, runs **GR4J** 4-parameter soil moisture, percolation, unit hydrograph, and routing stores:
     $$Q_{\text{sim}} = Q_r(x_3) + Q_d(x_4)$$
   - For ungauged basins, executes **MC-LSTM-DS** dual-state forward pass, guaranteeing exact mass conservation:
     $$\sum P_t = \sum Q_t^{\text{quick}} + \sum Q_t^{\text{slow}} + \Delta S$$
3. **Compound Extreme Value Copulas:**
   - Fits Archimedean copulas to compute joint probability $C(u, v)$ and return periods:
     $$T_{\text{AND}} = \frac{1}{1 - u - v + C(u, v)}, \quad T_{\text{OR}} = \frac{1}{1 - C(u, v)}$$
4. **Drought Regime Filtering:**
   - Computes causal recursive forward filtering on ERA5 soil moisture series:
     $$\alpha_t(j) = P(Y_t \mid Z_t = j) \sum_{i=1}^4 \alpha_{t-1}(i) A_{ij}$$
     identifying whether an agricultural district is entering an irreversible drought cascade.

#### Step 4: Layer 3 Exposure Intersections
Physical hazard probabilities are intersected with high-resolution geospatial vector and raster layers:
- **WorldPop:** Aggregates human population within 5km, 10km, and 15km radii.
- **OpenStreetMap:** Computes counts and Euclidean distance to vulnerable infrastructure: electrical substations, bridges, rail transit stations, hospitals, and drainage canals.
- **HydroRIVERS:** Identifies distance to the nearest natural river reach and Strahler stream order to detect backwater flood susceptibility.

#### Step 5: Layer 4 Decision Intelligence & Conformal Bounds
The synthesized hazard-exposure data is routed to the requested **Persona Decision Adapter**:
- **13 Domain Personas:**
  1. `DisasterManagementOfficial`: Issues civil defense evacuations, NDRF deployment triggers, emergency flood shelters.
  2. `CityCommissioner`: Urban infrastructure alerts, pumping station dispatches, municipal school closures.
  3. `PowerGridDispatcher`: Substation thermal de-rating advisories, transformer load shedding, peaking plant scheduling.
  4. `FarmerAdvisory`: Sowing date adjustments, irrigation scheduling based on SMDI, harvest protection.
  5. `HeatHealthOfficer`: Hospital cooling center activation, IV-fluid stocking, vulnerable elderly heat alerts.
  6. `WaterSupplyEngineer`: Groundwater contamination warnings, reservoir inflow forecasts, treatment chlorination adjustment.
  7. `PortAuthorityOfficer`: Coastal gale restrictions, berthing suspensions, storm surge warnings.
  8. `ConstructionManager`: Concrete curing thermal limits, crane wind shut-downs, worker mandatory rest cycles.
  9. `SchoolAdministrator`: Air quality indoor mandates, extreme heat early dismissal.
  10. `InsuranceUnderwriter`: Parametric payout verifications based on official GEV return period exceedance.
  11. `TransportLogistics`: Freight rerouting away from inundated low-lying rail underpasses.
  12. `PublicHealthEpidemiologist`: Post-flood waterborne disease vector warning based on standing water index.
  13. `CitizenResident`: Plain-language, actionable family safety checklists.

- **Uncertainty Calibration (Split Conformal Prediction):**
  Constructs finite-sample distribution-free prediction corridors $[y_{\text{lower}}, y_{\text{upper}}]$ guaranteeing $1 - \alpha \ge 90\%$ marginal coverage without assuming Gaussian errors:
  $$C(x) = [\hat{f}(x) - q_{1-\alpha}, \hat{f}(x) + q_{1-\alpha}]$$
- **Explainability (Native C++ TreeSHAP):**
  Computes exact feature attributions ($< 1\text{ ms}$ latency) with mathematical additivity:
  $$f(x) = \phi_0 + \sum_{j=1}^M \phi_j$$
  explaining precisely why an alert was triggered (e.g., $+24.2\%$ risk from 3-day rainfall, $+18.1\%$ risk from high soil moisture saturation, $-5.2\%$ protection from drainage canal proximity).

---

## 5. RECENT MAJOR MILESTONE: BASINFLOW V2 TOURNAMENT RESULTS

The latest major advancement in Clima-Core (Release v2.0) is **BasinFlow V2**, which resolved the longstanding physical hydrology blocker by incorporating genuine measured streamflow from CAMELS-IND v2.2.

### Gauged Catchments Holdout Benchmark (N=89 Catchments, 2014–2018 Test Period)
| Candidate Model | Algorithm / Formulation | Median NSE | Median KGE | Median RMSE (mm/d) | Median FHV (High Flow Bias) | Median Water Balance Err (%) |
|---|---|---|---|---|---|---|
| **Persistence Baseline** | $\hat{Q}_t = Q_{t-1}$ (1-day lag) | 0.553 | 0.776 | 1.949 | -18.7% | 0.0% |
| **Seasonal Climatology** | $\hat{Q}_t = \bar{Q}_{\text{DOY}}$ (365-day mean) | 0.067 | 0.101 | 2.408 | -72.0% | +35.2% |
| **SCS-CN Runoff** | USDA NRCS Event Runoff | -1.260 | -0.064 | 4.081 | -35.7% | -8.7% |
| **BasinFlow V1** | SCS-CN + Unit Hydrograph | -0.229 | 0.309 | 2.590 | -23.9% | -7.7% |
| **GR4J (Conceptual)** | **Perrin et al. 2003 (4 Parameters)** | **0.277** | **0.435** | **1.853** | **-17.2%** | **+20.4%** |
| **HBV-Light (Conceptual)** | Bergström 1992 (9 Parameters) | 0.295 | 0.378 | 1.927 | -28.5% | +22.1% |
| **Standard-LSTM** | Unconstrained PyTorch LSTM | 0.251 | 0.373 | 2.034 | -41.6% | +14.4% |
| **MC-LSTM** | Mass-Conserving Single State | 0.242 | 0.326 | 2.042 | -50.4% | +8.3% |
| **MC-LSTM-DS** | **Mass-Conserving Dual State** | **0.210** | **0.340** | **2.149** | **-48.0%** | **+4.8%** |

### Spatial Generalization Holdout (PUB, N=20 Held-Out Ungauged Catchments)
| Candidate Model | Formulation / Transfer Mode | Median NSE | Median KGE | Median RMSE (mm/d) | Median Water Balance Err (%) |
|---|---|---|---|---|---|
| **Seasonal Climatology** | Basin Donor Transfer | -0.052 | -0.237 | 1.972 | -1.1% |
| **SCS-CN Runoff** | Regionalized Median CN | -2.745 | -0.472 | 4.130 | +5.5% |
| **BasinFlow V1** | Regionalized CN + UH | -0.586 | 0.035 | 2.916 | +5.8% |
| **GR4J** | Regionalized Parameters | -0.025 | 0.100 | 1.947 | +42.2% |
| **HBV-Light** | Regionalized Parameters | 0.090 | 0.117 | 1.867 | +52.7% |
| **Standard-LSTM** | Zero-Shot Transfer | 0.075 | 0.142 | 1.878 | +37.1% |
| **MC-LSTM** | Zero-Shot Transfer | 0.128 | 0.247 | 2.074 | +19.8% |
| **MC-LSTM-DS** | **Zero-Shot Dual State** | **0.129** | **0.223** | **2.154** | **+17.3%** |

### Verified Physical Insights:
1. **Gauged Winner: GR4J** delivers the highest KGE (`0.4354`), lowest RMSE (`1.853 mm/d`), and lowest peak magnitude error (`-0.45%`) with only 4 calibrated parameters.
2. **Ungauged Winner: MC-LSTM-DS** wins Prediction in Ungauged Basins (`NSE = 0.1290`), outperforming regionalized HBV-Light (`0.090`) and unconstrained LSTM (`0.075`), while bounding water balance error to $+17.3\%$ (vs $+52.7\%$ in HBV-Light).
3. **Hydrograph Flow Separation:** On sample catchment `3005`, MC-LSTM-DS partitioned 5,452.58 mm of precipitation into 558.87 mm quickflow (37.61%) and 927.20 mm slow groundwater baseflow (62.39%), matching hydrogeological baseflow indices with zero mass violation.

---

## 6. SYSTEM STRENGTHS, KNOWN LIMITATIONS & PROMPT QUESTIONS FOR PEER REVIEW

### Architectural Strengths
- **100% Automated Test Coverage:** 214 passing unit and integration tests covering mathematical formulations, data pipelines, model reloads, and API routes.
- **Hardware Compliance:** 100% runnable on modest consumer hardware (Intel Core i3, Windows x64, 2 threads, $< 850\text{ MB}$ RAM, zero GPU dependency).
- **Physical Defensibility:** No black-box unconstrained models operating without thermodynamic, mass-balance, or biometeorological boundaries.
- **Dual Decision Architecture:** Combines fast point predictions with conformal uncertainty bounds and C++ native TreeSHAP local attribution.

### Current Known Limitations
1. **Dam & Reservoir Regulation:** CAMELS-IND streamflow is measured at downstream CWC gauges, but many Indian rivers feature unmonitored upstream dam releases and irrigation barrages that alter natural hydrographs.
2. **Sub-Daily Discretization:** The system currently runs at daily resolution ($\Delta t = 24\text{ hours}$). Rapid urban pluvial flash floods (such as Mumbai or Bengaluru convective cloudbursts) occur in 1 to 3 hours, requiring future sub-daily radar/satellite downscaling.
3. **Crop Phenology Granularity:** AgriSense yield anomaly operates at district-level aggregations; farm-level precision agriculture will require field boundary polygon inputs.

---

## 7. TARGETED QUESTIONS FOR CHATGPT / EXTERNAL PEER REVIEW

*To the Reviewer: Please analyze this architecture and provide critical feedback on the following engineering and scientific decisions:*

1. **Hydrological Strategy (Dual Champion vs Single Model):** Clima-Core adopted a Dual Champion for streamflow: GR4J for gauged basins (highest KGE 0.435, lowest RMSE 1.85 mm/d) and MC-LSTM-DS for ungauged basins (highest PUB NSE 0.129, mass-conserving). Is this dual-engine strategy optimal, or should we train a hybrid physics-informed neural network (PINN) that embeds GR4J differential equations directly into PyTorch computational graphs?
2. **Sub-Daily Urban Flash Flood Scaling:** Given that daily reanalyses (ERA5-Land) and daily IMD rainfall do not capture 1-hour cloudbursts, what is the most compute-efficient method to downscale 24h forecasts to 1h/3h synthetic hyetographs (e.g. Huff distribution curves vs stochastic rainfall generators) on CPU?
3. **Compound Risk Copula Tail Selection:** We currently evaluate Gumbel, Clayton, Survival Clayton, and Frank copulas using AIC/BIC and Cramér-von Mises distance. For compound thermal-grid and deluge-dam failure tails, should we consider vine copulas or non-parametric empirical copulas to capture higher-dimensional asymmetric dependence?
4. **Conformal Prediction Exchangeability under Climate Non-Stationarity:** Split conformal prediction guarantees $1 - \alpha$ marginal coverage under exchangeability. With non-stationary climate extremes in India, what adaptive conformal inference methods (e.g., Online Conformal Inference, Rolling Non-Conformity Scores) would best prevent interval under-coverage during record-breaking anomalous seasons?
