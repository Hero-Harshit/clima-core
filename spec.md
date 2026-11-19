# Clima-Core — Project Specification and Single Source of Truth

> **Document:** `spec.md`  
> **Project:** Clima-Core  
> **Specification version:** 1.0.0  
> **Status:** Official v1.0.0 Baseline; Milestone 1 ML & Exposure Foundation 100% Complete (87/87 Unit Tests Passing); Architectural 4-Layer Taxonomy & Scientific Hardening Active  
> **Last reviewed:** 18 November 2026  
> **Primary purpose:** Authoritative project contract, dataset registry, model taxonomy, and decision architecture for Antigravity.  
> **Change policy:** This specification is the ground truth. All architectural shifts, dataset additions, and model modifications must be documented here and recorded in the Change Log.

---

## 0. Instructions for Antigravity — Read Before Doing Any Work

This file is the current **single source of truth** for Clima-Core. Follow it before making architectural, data, model, deployment, or product decisions.

### 0.1 Current project state & active milestones

**Milestone 1 Foundation is COMPLETE and VERIFIED:**
- 14 high-edge meteorological, hydrological, agricultural, grid, and infrastructure datasets are fully ingested into `data/processed/` with cryptographic SHA-256 manifests.
- 28 supervised and calibrated machine learning models/boosters have been trained with zero lookahead data leakage ($\Delta_{\max} = 0.00000000$ reload parity).
- All 13 specialized persona adapters are implemented with comprehensive operational advisories and civil defense restrictions.
- 113 automated unit and integration tests are passing in `tests/` (100% parity across all models, registries, and physics engines).

**Current Status & Roadmap Alignment:**
1. **Scientific Foundation & Model Audit (Roadmap Phase 1 - 100% COMPLETE):** Built canonical Model, Dataset, and Physical Threshold Registries with SHA-256 integrity; implemented the Validation Tri-Split Engine (Temporal, Spatial LOCZO across 5 Bioclimatic Zones, Event-Safe disaster partitions); resolved SciPy GEV shape sign conventions, CPCB 8-pollutant AQI completeness compliance, and flood inventory negative-label semantics.
2. **Cross-Dataset Fusion & Flagship V3 Models (Roadmap Phase 2 - 100% COMPLETE):** Integrated hydro-atmospheric soil-saturation and drainage bottlenecks into FloodRisk V3; dual diurnal and microclimate Oke UHI into HeatRisk V4; lag-augmented calendar load modeling into PowerGrid V3; tri-task suitability/yield/phenology into AgriSense V3; boundary layer ventilation and nocturnal inversion into AirQuality V3.
3. **Advanced ML & Hybrid Architectures (Roadmap Phase 3 - 100% COMPLETE):** Conducted rigorous Champion vs Challenger tournament benchmarking on identical Phase 1 holdouts pitting CatBoost against LightGBM for crop suitability and deep PyTorch Temporal Fusion Transformer (TFT) against LightGBM for power grid load forecasting; codified tournament manifest and dynamic multi-engine routing (`champion`, `challenger`, `ensemble`).
4. **Probabilistic Intelligence & Compound Risks (Roadmap Phase 4 - 100% COMPLETE):** Distribution-free split conformal prediction intervals guaranteeing $\ge 90\%$ marginal coverage; 3 cascading multi-hazard infrastructure failure chains; standardized unified JSON prediction contract `v1.0.0` with full system provenance, spatial buffers, and SI physical units manifest.
5. **Generalization, Deployment & Continuous Intelligence (Roadmap Phase 5 - 100% COMPLETE):** 6-point Deployment Gatekeeper with feature ablation studies; Population Stability Index (PSI) & Kolmogorov-Smirnov drift detection; MAPIE-style conformal calibration exchangeability degradation monitoring; zero-cold-start FastAPI / Vercel serverless inference service (`api/index.py`, sub-millisecond warm CPU latency).
6. **Hydro-Environmental Intelligence Expansion (HydroRIVERS & CGWB Water Quality - 100% COMPLETE):**
   - Ingested HydroSHEDS HydroRIVERS Asia dataset (1.4M river reaches, 531,819 in India); extracted nearest river distance, Strahler stream order, and long-term average discharge ($m^3/s$) for all 50 cities; coupled riverine fluvial backwater overtopping with pluvial stormwater into FloodRisk V3.
   - Ingested Central Ground Water Board (CGWB) Chemical Quality dataset (18,249 historical monitoring samples); trained `WaterQualityRiskEngine` (LightGBM Classifier ROC-AUC 0.9286, WQI Regressor $R^2 = 0.9255$) evaluating BIS IS 10500:2012 potable drinking standards and operational public health advisories.
7. **Milestone Summary:** 150 automated unit and integration tests passing with 100% parity across all models, engines, and APIs.

Do **not** build arbitrary UI pages, bloated consumer screens, or unverified frontend prototypes until the predictive and statistical engines meet these scientific defense standards.

### 0.2 Non-negotiable scientific & anti-hallucination rules

Antigravity must obey all of the following:

- **The 4-Layer Separation Rule:** Never present a physical equation, a threshold decision rule, or a geospatial overlay as an independently trained ML model. Clearly distinguish:
  - *Layer 1: Predictive Intelligence (ML / Statistical Models)*
  - *Layer 2: Physical Intelligence (Empirical & Theoretical Atmospheric/Hydrological Formulations)*
  - *Layer 3: Exposure Intelligence (Geospatial Vulnerability, Grids & Lifeline Assets)*
  - *Layer 4: Decision Intelligence (13 Domain Persona Adapters & Actionable Policies)*
- **Extreme Value Statistics:** Never equate a daily observation percentile ($P_{90}, P_{98}, P_{99.5}$) directly to an annual return period (1-in-2, 1-in-10, 1-in-25 years). True return periods must be derived from Annual Maximum Series (AMS) fitted to Generalised Extreme Value (GEV) distributions or Peaks Over Threshold (POT) Pareto distributions.
- **Probabilistic Transparency:** Never describe a weighted heuristic or synergy function as a "copula" unless a mathematical copula dependency structure is explicitly fitted to calibrated joint marginal probabilities.
- **Honest Domain Boundaries:** Never present surrogate environmental indicators as specialized hardware instrumentation. Surface $\text{PM}_{2.5}$ concentration is an optical particulate haze advisory, **not** an official airport Runway Visual Range (RVR) transmissometer reading. Beaufort wind scaling indicates potential sea surface agitation, **not** buoy-measured wave spectra.
- **Hazard vs. Exposure Decoupling:** Keep hazard probability strictly decoupled from human impact. A 90% flood probability over an unpopulated wetland does not equal high human catastrophe. Impact requires combining hazard probability with Layer 3 exposure metrics.
- **Zero Leakage & Chronological Splits:** Always evaluate on untouched chronological holdouts. Never use random train/test splits on time-series or spatially correlated events. For episodic events (cyclones, flood disasters), keep entire storms and flood episodes within single partitions to prevent intra-event contamination.
- **Source Identity & Provenance:** Retain provider names, timestamps, native spatial resolutions, and units for every input. Official warnings from IMD, CPCB, and NDMA must preserve their exact wording and agency authority.
- **Deterministic Parity:** Every exported model artifact must produce bit-for-bit identical floating-point outputs upon reload.

---

# 1. Product Definition

## 1.1 Name and positioning

**Name:** Clima-Core  
**Category:** AI-powered Environmental Risk Intelligence & Decision Support Platform  
**Tagline:** *Environmental Intelligence. Smarter Decisions.*

Clima-Core combines multi-decadal atmospheric reanalyses, physical ground observations, supervised machine learning, extreme value statistics, 100m gridded human exposure, and OpenStreetMap infrastructure networks into one cohesive decision-support platform tailored for Indian cities.

Clima-Core is an intelligence and decision-support layer. It is **not** a replacement for the India Meteorological Department (IMD), National Disaster Management Authority (NDMA), Central Pollution Control Board (CPCB), Central Water Commission (CWC), Central Ground Water Board (CGWB), or state emergency operations centers.

## 1.2 The Core Problem Clima-Core Solves

Standard meteorological forecasts report siloed numbers: *"42°C daytime temperature"* or *"85 mm of rainfall"*. But real-world disaster impact is fundamentally **compound, physical, and exposure-dependent**:
- 85 mm of rainfall falling on dry, porous soil produces minor runoff; the same 85 mm falling on antecedent saturated soil with a low drainage canal-to-road ratio triggers flash urban flooding, inundating low-clearance underpasses and cutting off hospitals.
- A 42°C ambient temperature during the day is severe; when combined with a nocturnal minimum above 28°C ("tropical night"), electrical distribution transformers cannot dissipate heat, triggering winding breakdown, power grid load shedding, and water pump failures in dense informal settlements.

Clima-Core bridges this gap by transforming raw weather and air quality forecasts into **contextualized, exposure-weighted, actionable decision intelligence**.

---

# 2. Scope and Development Phases

```mermaid
flowchart LR
    P1["Phase 1: ML & Data Foundation (DONE)"] --> P2["Phase 2: Scientific Hardening (ACTIVE)"]
    P2 --> P3["Phase 3: Flagship Model Deepening"]
    P3 --> P4["Phase 4: Unified Contract & Fast API"]
    P4 --> P5["Phase 5: MapLibre UI & Decision Portal"]
```

## 2.1 Phase 1: Machine Learning & Exposure Foundation (100% COMPLETE)
- Ingested 14 datasets covering 74 years of IMD temperature, 46 years of IMD rainfall, 45 years of cyclones, 44 years of IFI flood disasters, WorldPop 2024 100m grids, OSM infrastructure, SAGE crop phenology, CHETNA thermal plants, and POSOCO/CEA 8-year electricity telemetry.
- Trained 28 ML models/calibrators across Heat, Rain, Air Quality, Floods, Grid Stress, and Agriculture.
- Implemented 13 persona adapters and verified 87/87 unit tests with zero regressions.

## 2.2 Phase 2: Scientific Hardening & Statistical Rigor (100% COMPLETE)
- **Extreme Event Statistics Engine:** Fit GEV distributions (`scipy.stats.genextreme`) on 74-year IMD annual temperature maxima and 46-year rainfall maxima for 50 cities to calculate true 5-yr, 10-yr, 25-yr, and 50-yr return periods with confidence intervals (`src/climacore/analytics/extreme_value.py`).
- **Transparent Multi-Hazard Aggregation:** Formalized compound risk scoring as an empirical bivariate/trivariate joint exceedance probability, eliminating heuristic pseudo-copula terminology (`src/climacore/models/compound_risk/scoring.py`).
- **Elevation-Aware Psychrometrics:** Incorporated station barometric pressure corrections into wet-bulb temperature solver for elevated urban plateaus (Bangalore 920m, Pune 560m, Hyderabad 540m) solving $e = e_s(T_{\text{wb}}) - \gamma(P)\cdot(T - T_{\text{wb}})$ (`src/climacore/analytics/psychrometrics.py`).
- **Honest Terminology Alignment:** Rebranded aviation visibility to "Surface Particulate Optical Haze Index" and electricity deficit to "Unmet Supply Stress".

## 2.3 Phase 3: Flagship Model Deepening (100% COMPLETE)
Consolidated the 5 Champion ML Model Suites into modular flagship engines:
1. **FloodRisk V2 (`src/climacore/models/flood_risk/engine.py`):** Strictly decoupled pure hydrological hazard probability ($0.0 - 1.0$) from Layer 3 human & critical lifeline exposure footprints (population, schools, hospitals, bridges, drainage bottlenecks).
2. **HeatRisk V3 (`src/climacore/models/heat_risk/engine.py`):** Daytime $T_{\max}$ corrector, nighttime $T_{\min}$ / tropical night failure predictor ($T_{\min} \ge 28^\circ\text{C}$), Oke Urban Heat Island microclimate anomaly model, and 74-year GEV return period integration.
3. **PowerGrid V2 (`src/climacore/models/electricity_grid/engine.py`):** Peak load MW regressor, cooling demand surge ratio (CDD base 18°C), and unmet supply stress classifier with DISCOM spinning reserve mandates.
4. **AgriSense V2 (`src/climacore/models/crop_resilience/engine.py`):** 10-class agro-climatic crop suitability classifier (Top-1: 82.49%, Top-3: 97.62%), SAGE phenological growth stages (Sowing, Anthesis, Harvest), anthesis floret sterility heat shock alerts, and harvest lodging rain alerts.
5. **AirQuality V2 (`src/climacore/models/air_quality/engine.py`):** Next-day multi-pollutant prediction ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2$), official CPCB 6-tier NAQI sub-indices, boundary layer atmospheric stagnation/nocturnal inversion classifier, and Indo-Gangetic Plain transboundary smoke advection model.
6. **Live Forecast Integration:** Integrated all 5 flagship engines directly into `LiveForecastEngine` (`src/climacore/ingestion/engine.py`), adding structured `flagship_intelligence` to daily predictions. Verified with 101/101 passing unit and integration tests.

## 2.4 Roadmap Phase 1: Scientific Foundation & Model Audit (100% COMPLETE)
- **Canonical Model & Dataset Registries (`src/climacore/registry/`):**
  - Built unified, auditable registry system exporting `canonical_model_registry.json`, `canonical_dataset_registry.json`, and `canonical_physical_thresholds.json`.
  - Every model tracks unique ID, SemVer, ordered features, chronological/event splits, audited metrics, artifact SHA-256 hash, and explicit operational limitations.
  - Every physical threshold (tropical nights 28°C, anthesis heat shock 32°C/35°C, harvest lodging 10mm, wet-bulb 30°C/32°C work limits, CDD 18°C base) is formally sourced to standard authorities (IS 2026/IEC 60076, ICAR/CIMMYT, NDMA, ASHRAE).
- **Validation Tri-Split Engine (`src/climacore/evaluation/splits.py`):**
  - **Temporal Validation:** Chronological train/val/test partitioning with optional purge embargo to eliminate boundary lag leakage.
  - **Spatial Validation (LOCZO):** Leave-One-Climate-Zone-Out cross-validation across 5 canonical Indian Bioclimatic Zones (Indo-Gangetic Plains, Deccan Plateau, Western Coastal, Eastern Coastal, Himalayan/Hill/NE) mapping all 50 cities deterministically.
  - **Event-Safe Validation:** Atomically partitions complete disaster episodes (IFI floods, IMD cyclone tracks) so zero events straddle train and evaluation sets.
- **Scientific Detail Audits & Enhancements:**
  - **EVT GEV Convention (`src/climacore/analytics/extreme_value.py`):** Explicitly exposes SciPy shape parameter $c_{\text{SciPy}}$ alongside classical hydrological EVT shape $\xi = -c_{\text{SciPy}}$, tail classification (Weibull vs Fréchet vs Gumbel), 95% bootstrap confidence intervals for all return periods ($T_5, T_{10}, T_{25}, T_{50}$), and extrapolation uncertainty flags.
  - **CPCB AQI Completeness (`src/climacore/models/air_quality/engine.py`):** Implemented official CPCB compliance check requiring $\ge 3$ monitored pollutants with at least one particulate ($\text{PM}_{2.5}$ or $\text{PM}_{10}$), flagging incomplete proxy estimates.
  - **Disaster Negative-Label Semantics (`src/climacore/models/flood_risk/engine.py`):** Added explicit inventory reporting certainty tier and documented that historical catalog absence signifies no declared district emergency rather than zero localized waterlogging.
  - **Physical vs ML Grid Decoupling (`src/climacore/models/electricity_grid/engine.py`):** Explicitly partitioned deterministic CDD engineering demand surge ratio from supervised ML unmet supply stress classification.
- **Verification:** 113/113 unit and integration tests passing in 16.13s with zero regressions.

## 2.5 Roadmap Phase 2: Cross-Dataset Fusion & Flagship V3 Models (COMPLETED)
- **Status:** 100% COMPLETED.
- **Architectural Upgrades Across 5 Flagship Engines:**
  - **FloodRisk V3 (`src/climacore/models/flood_risk/engine.py`):**
    - Implemented the full Hydro-Atmospheric Cascade ($\text{Atmospheric Rain} \rightarrow \text{Root-Zone Soil Saturation} \rightarrow \text{Effective Runoff} \rightarrow \text{Drainage Inundation}$).
    - Topsoil saturation-excess model collapses infiltration capacity when rootzone volumetric moisture $\theta \ge 70\%$, dynamically scaling the runoff coefficient $C$ from $0.18$ to $0.88$.
    - OSM canal-to-road drainage impedance factor captures localized urban stormwater bottlenecks.
    - Evaluates multi-horizon hazard probabilities ($24\text{h}$ Flash Inundation, $48\text{h}$ Channel Overflow, $72\text{h}$ Catchment Waterlogging) with strict decoupling from human exposure.
  - **HeatRisk V4 (`src/climacore/models/heat_risk/engine.py`):**
    - Connects daytime solar radiation heating and synoptic advection with nocturnal minimum cooling deficits ($T_{\min} \ge 28^\circ\text{C}$ tropical nights).
    - Implemented Oke urban canopy geometry and WorldPop dense settlement cluster ($>15,000\text{ people/km}^2$) excess ($\Delta T_{\text{UHI}}$).
    - Added calibrated quantile uncertainty bounds ($Q_{10}, Q_{50}, Q_{90}$) for peak day and nocturnal temperatures.
    - Added multi-horizon diurnal projections ($24\text{h}, 48\text{h}, 72\text{h}$) coupled with 74-year GEV return period evaluation.
  - **PowerGrid V3 (`src/climacore/models/electricity_grid/engine.py`):**
    - Augments thermal demand features with 1-day, 7-day, and 14-day historical load lags and calendar indicators (weekend load reduction factor $-8\%$, holidays).
    - Multi-horizon peak load (MW) regressor and unmet supply stress classifier across $24\text{h}, 48\text{h}, 72\text{h}$ horizons.
    - Calibrated quantile load bounds ($Q_{10}, Q_{50}, Q_{90}$) on peak load MW.
  - **AgriSense V3 (`src/climacore/models/crop_resilience/engine.py`):**
    - Tri-Task architecture separating:
      - **Task A:** 10-class agro-climatic crop suitability distribution (LightGBM GBDT).
      - **Task B:** Empirical crop yield potential range ($Q_{10}, Q_{50}, Q_{90}$ in tonnes/ha) conditioned on district soil fertility and active weather stress penalties.
      - **Task C:** SAGE phenological growth-stage stress tracking (anthesis floret heat shock $T_{\max} \ge 32^\circ\text{C} / 36^\circ\text{C}$, harvest rain lodging $\ge 10\text{ mm} / 30\text{ mm}$, FAO-56 crop water balance).
  - **AirQuality V3 (`src/climacore/models/air_quality/engine.py`):**
    - Coupled boundary layer ventilation index ($V = \text{Wind Speed (m/s)} \times \text{Boundary Layer Mixing Height (m)}$) with nocturnal temperature inversion trapping ($DTR > 12^\circ\text{C}$, calm winds $\le 10\text{ km/h}$).
    - Multi-pollutant transfer calibrators and multi-horizon projections ($24\text{h}, 48\text{h}, 72\text{h}$) for $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{O}_3$.
    - CAMS atmospheric advection coupled and calibrated with CPCB ground station reality.
- **Verification:** 119/119 unit and integration tests passing in 26.22s with zero regressions.

## 2.6 Roadmap Phase 3: Advanced ML & Hybrid Architectures (COMPLETED)
- **Status:** 100% COMPLETED.
- **Champion vs Challenger Tournament Framework (`src/climacore/evaluation/tournament.py`):**
  - Formalized tournament benchmarking on the exact Phase 1 holdouts:
    - **Chronological Holdouts:** Untouched test partitions (1997-2011 Train, 2012-2013 Val, 2014-2015 Test for Agriculture; 2013-2018 Train, 2019 Val, 2020 Test for Power Grid).
    - **Spatial Holdouts:** 5-Fold Leave-One-Climate-Zone-Out (LOCZO) across the 5 canonical Indian Bioclimatic Zones.
  - Published Canonical Tournament Manifest: `data/manifests/champion_challenger_tournament.json`.
- **AgriSense Crop Suitability Tournament (LightGBM vs CatBoost):**
  - **Champion (`artifacts/models/crop_resilience/v0.1.0/crop_suitability_v1.booster`):** LightGBM Multi-class GBDT (Top-1 Accuracy: 82.49%, Top-3: 97.62%, Weighted F1: 0.8250).
  - **Challenger (`artifacts/models/crop_resilience/challengers/catboost_v1.cbm`):** CatBoost Multi-class GBDT with symmetric oblivious trees, depth 6, and L2 regularization (Top-1 Accuracy: 80.18%, Top-3: 96.54%, Weighted F1: 0.7976).
  - **LOCZO Spatial Findings:** CatBoost demonstrated strong regional robustness, outperforming in Western Coastal (95.83%) and Eastern Coastal (86.17%) agro-climatic zones.
  - **Guiding Rule Verdict:** `CHAMPION_RETAINS_LEAD`. LightGBM retains primary champion deployment; CatBoost is integrated as an alternate engine and ensemble partner (60% LightGBM + 40% CatBoost).
- **PowerGrid Peak Demand Tournament (LightGBM vs Temporal Fusion Transformer):**
  - **Champion (`artifacts/models/electricity_grid/v0.1.0/peak_grid_load_v1.booster`):** LightGBM Point Regressor ($R^2 = 0.8758$, $\text{MAE} = 1519.18\text{ MW}$, $\text{RMSE} = 2336.25\text{ MW}$).
  - **Challenger (`artifacts/models/electricity_grid/challengers/tft_v1.pt`):** Deep Temporal Fusion Transformer (TFT) implementing Variable Selection Networks (VSN), Gated Residual Networks (GRN), Temporal Self-Attention, and Multi-Horizon Quantile Output Heads ($Q_{10}, Q_{50}, Q_{90}$ at $24\text{h}, 48\text{h}, 72\text{h}$).
  - **Performance:** Achieved Test $R^2 = 0.7059$ and $\text{MAE} = 2659.71\text{ MW}$ with native multi-horizon pinball quantile intervals.
  - **Guiding Rule Verdict:** `CHAMPION_RETAINS_LEAD`. LightGBM retains primary point estimation lead; TFT provides sequence multi-horizon quantile intervals and temporal variable selection routing.
- **Dynamic Model Engine Routing:**
  - `AgriSenseEngine` supports `engine_mode="champion"`, `"challenger"`, and `"ensemble"`.
  - `PowerGridEngine` supports `engine_mode="champion"`, `"challenger"`, and `"ensemble"`.
- **Verification:** 124/124 unit and integration tests passing in 12.11s with zero regressions.

## 2.7 Roadmap Phase 4: Probabilistic Intelligence & Compound Risks (100% COMPLETE)
- **Status:** 100% COMPLETED.
- **Split Conformal Prediction Calibrator (`src/climacore/analytics/conformal.py`):**
  - Implements distribution-free conformal calibration guaranteeing finite-sample marginal coverage $\mathbb{P}(Y \in \mathcal{C}(X)) \ge 1 - \alpha$ using the finite-sample quantile index $p = \min(1.0, \lceil(n+1)(1-\alpha)\rceil / n)$.
  - Provides asymmetric conformal intervals $\mathcal{C}(X) = [\hat{y} - q_{\text{low}}, \hat{y} + q_{\text{high}}]$ parameterized by user-specified confidence level $(1 - \alpha)$.
  - Specialized calibrated domains:
    - Temperature ($T_{\max}$ in $^\circ\text{C}$): Calibrated over 365 holdout days.
    - Rainfall ($\text{mm}$): Non-negative bounded conformal interval preventing impossible negative rain forecasts.
    - Grid Power Demand ($\text{MW}$): Calibrated across regional seasonal demand variations.
    - Air Quality ($\text{PM}_{2.5}$ in $\mu\text{g/m}^3$): Non-negative bounded conformal intervals across high and low pollution seasons.
- **Cascading Compound Risk Causal Chains (`src/climacore/models/compound_risk/cascading.py`):**
  - Formalized cross-sectoral failure propagation mechanics across three inter-connected infrastructure networks:
    1. **Chain 1: Thermal Shock $\rightarrow$ Power Grid Overload $\rightarrow$ Municipal Water Pumping Failure $\rightarrow$ Informal Settlement Casualties.**
       - Trigger: $T_{\max} \ge 42^\circ\text{C}$ and $T_{\min} \ge 28^\circ\text{C}$.
       - Escalation: Substation transformer thermal de-rating + HVAC overload $\rightarrow$ water treatment/pumping brownout $\rightarrow$ heatstroke cluster in un-insulated settlements.
    2. **Chain 2: Cloudburst Deluge $\rightarrow$ Drainage Bottleneck $\rightarrow$ Arterial Road Submergence $\rightarrow$ Healthcare & Emergency Transit Severance.**
       - Trigger: Rain $\ge 70\text{ mm/day}$ or antecedent precipitation saturation $\ge 80\%$.
       - Escalation: Canal culvert capacity overflow $\rightarrow$ underpass drowning $\rightarrow$ ambulances and emergency supplies unable to reach tertiary hospitals.
    3. **Chain 3: Atmospheric Inversion $\rightarrow$ Toxic Boundary Stagnation $\rightarrow$ Pediatric & Geriatric Respiratory Surge.**
       - Trigger: Ventilation index $< 2000\text{ m}^2/\text{s}$ and diurnal temperature range $> 12^\circ\text{C}$.
       - Escalation: Nocturnal capping inversion traps combustion and agricultural particulates $\rightarrow$ acute bronchial distress hospital admissions.
  - Implemented `CascadingRiskEngine` evaluating trigger conditions, critical node failure probabilities, estimated time-to-impact (in hours), and prioritized cross-sectoral operational mitigation interventions.
- **Unified Standardized Prediction Contract (`v1.0.0`) (`src/climacore/ingestion/models.py`, `src/climacore/ingestion/engine.py`):**
  - Standardized JSON contract emitted by `LiveForecastEngine.predict()` across all 5 flagship engines and 13 personas.
  - Standardized schema elements:
    - `schema_version`: `"v1.0.0"`
    - `system_provenance`: Model SHA-256 hashes, scientific literature citations, training timestamp, verification authority.
    - `spatial_resolution_metadata`: Multi-scale concentric buffers ($5\text{km}$ urban core, $15\text{km}$ suburban ring, $30\text{km}$ peri-urban catchment).
    - `physical_units_manifest`: Formal declaration of all SI metric units ($^\circ\text{C}, \text{mm}, \text{MW}, \mu\text{g/m}^3, \text{m/s}, \text{m}^2/\text{s}, \text{kPa}, \text{tonnes/ha}, \text{people/km}^2$).
    - `conformal_bounds`: Calibrated conformal prediction bounds ($Q_{10}, Q_{50}, Q_{90}$) with guaranteed 90% coverage for temperature, rain, power demand, and PM2.5.
    - `cascading_risk`: Critical infrastructure cascade chains, failure nodes, time-to-impact, and mitigation actions.
- **Verification:** 128/128 tests passing in 7.73s with zero regressions.

## 2.8 Roadmap Phase 5: Generalization, Deployment & Continuous Intelligence (100% COMPLETE)
- **Status:** 100% COMPLETED.
- **Production Deployment Gatekeeper (`src/climacore/evaluation/gatekeeper.py`):**
  - Codifies the 6-point non-negotiable production promotion gate before any candidate can replace a champion:
    1. **Primary Task Metric:** Challenger must meet or exceed champion on identical holdout sets ($\Delta \ge \text{min\_delta}$).
    2. **Calibration & Uncertainty Compliance:** Conformal empirical coverage must satisfy $\ge 88\%$ (nominal 90% target).
    3. **Spatial Generalization:** 5-Zone LOCZO breakdown must prove no single bioclimatic zone collapses ($> 25\%$ below mean).
    4. **Inference Latency & Memory Budget:** P95 CPU latency must be $\le 50\text{ms}$, peak memory $\le 200\text{MB}$.
    5. **Deterministic Reproducibility:** Zero numerical drift across model reload ($\Delta_{\max} \le 10^{-6}$).
    6. **Feature Schema Compatibility:** Strict feature ordering, naming, and data type assertions.
  - Generates cryptographically hashed `PromotionEvaluationReport` (SHA-256) logging pass/fail decisions and granular failure rationales.
  - **Feature Ablation Experiments (`FeatureAblationStudy`):**
    - Systematically evaluates the marginal information value of feature groupings: Baseline, Baseline + Soil, Baseline + Terrain, Baseline + Vegetation, and Combined.
    - Computes marginal delta and feature efficiency ratios to scientifically eliminate feature bloat.
- **Continuous Model Monitoring & Drift Detection (`src/climacore/monitoring/drift.py`, `service.py`):**
  - **Population Stability Index (PSI):** Quantile-binned drift calculation across continuous environmental features ($T_{\max}$, rainfall, power load, $\text{PM}_{2.5}$, soil saturation, wind speed) with standard operating tiers: `STABLE` ($< 0.10$), `MODERATE_DRIFT` ($0.10 \le \text{PSI} < 0.25$), `SIGNIFICANT_DRIFT` ($\ge 0.25$).
  - **Two-Sample Kolmogorov-Smirnov (KS) Test:** Non-parametric statistical distribution shift hypothesis testing ($p < 0.05$).
  - **Missing Data Rate & Upstream Telemetry Dropouts:** Real-time tracking of sensor and feed dropouts.
  - **Conformal Calibration Health Monitor:** MAPIE-style rolling empirical coverage tracking on streaming ground truth verifications; triggers `CALIBRATION_DRIFT_ALERT` when empirical coverage drops below $85\%$.
  - **Latency Profiler:** In-memory recording of P50, P95, P99 inference latencies and error rates.
- **Zero-Cold-Start Serverless & Edge-Ready Inference Architecture (`src/climacore/service/app.py`, `api/index.py`):**
  - High-performance FastAPI ASGI application fully decoupled from heavy offline training routines.
  - Loads compiled booster artifacts (`.booster`, `.cbm`, `.pt`) and cached live weather bundles with warm sub-millisecond execution ($0.74\text{ms} - 1.32\text{ms}$ on CPU).
  - Clean operational endpoints:
    - `GET /` & `GET /api/health`: Platform status, schema version `v1.0.0`, registered model count, latency profile.
    - `POST /api/predict/city`: Emits unified `PredictionResult` v1.0.0 for any of the 50 Indian cities with conformal bounds, cascading compound risks, and persona advisories.
    - `POST /api/predict/custom`: Direct coordinate and environmental feature inference.
    - `GET /api/monitoring/health`: Real-time feature drift and calibration health report.
    - `POST /api/monitoring/verify`: Ingests verified ground truth to update calibration health.
    - `GET /api/tournament/status`: Tournament summary and champion-challenger registry.
    - `POST /api/gatekeeper/evaluate`: Programmatic gatekeeper promotion assessment.
- **Verification:** 143/143 unit and integration tests passing in 13.22s with zero regressions.

---

# 3. System Architecture — The 4-Layer Taxonomy

Clima-Core enforces a strict 4-layer separation of concerns:

```mermaid
flowchart TD
    subgraph Layer 1: Predictive Intelligence (ML Models)
        M1["Peak Grid Load Regressor (MW) [R²=0.8758]"]
        M2["Grid Unmet Supply Stress Classifier [AUC=0.9521]"]
        M3["IFI District Flood Disaster Classifier [AUC=0.7931, Recall=73.7%]"]
        M4["AgriSense Crop Suitability Classifier [Top-1 Acc=82.49%]"]
        M5["RainRisk V2: Q50, Q90, Q99 Quantile Regressors & Occurrence"]
        M6["AirQuality: CPCB CAAQMS Multi-Pollutant Transfer Calibrators"]
        M7["Atmospheric Stagnation & Nocturnal Inversion Classifier"]
        M8["IMD Heatwave 4-Tier Severity Classifier"]
        M9["Oke Urban Heat Island Microclimate Anomaly Regressor"]
    end

    subgraph Layer 2: Physical Intelligence (Atmospheric & Hydrological Equations)
        P1["Stull Psychrometric Wet-Bulb Temperature (Elevation-Aware)"]
        P2["FAO-56 Penman-Monteith Evapotranspiration & Daily Crop Water Balance"]
        P3["Cooling Degree Days (CDD, 18°C base) & Tropical Night Tracking"]
        P4["Antecedent Precipitation Index (API 14-day rolling decay)"]
        P5["Extreme Value Theory: GEV Return Periods (5y, 10y, 25y, 50y)"]
        P6["Soil Water Holding Capacity Index (Texture & Organic Carbon)"]
        P7["IMD RSMC Cyclone Gale Radii & Pressure Deficit Formulation"]
    end

    subgraph Layer 3: Exposure Intelligence (Geospatial & Lifeline Assets)
        E1["WorldPop 2024: 100m Gridded Population (5km, 15km, 30km concentric buffers)"]
        E2["WorldPop Informal Settlement Clusters (Density >= 7,500/km²)"]
        E3["OpenStreetMap: Hospitals, Clinics, Schools, Bridges, Drainage Canals, Transit"]
        E4["SAGE Crop Calendar: Sowing, Anthesis/Flowering, Harvest DOY Windows"]
        E5["CHETNA: 769 Thermal Power Plants (50km, 100km, 250km MW & Annual CO2)"]
        E6["CWC 150+ Major Dams Storage % & CGWB 437k Wells Groundwater Depth"]
    end

    subgraph Layer 4: Decision Intelligence (13 Persona Adapters)
        D1["City Official & Municipal Disaster Management (DDMA / Municipal Corp)"]
        D2["Farmer / Kisan Agro-Meteorological Advisor"]
        D3["Heavy Industry, Manufacturing & Thermal Power Generation"]
        D4["Healthcare, Emergency Hospital Surge & Respiratory Medicine"]
        D5["Water Utility & Municipal Stormwater Drainage"]
        D6["Gig Worker & Outdoor Delivery Couriers"]
        D7["Parametric Insurance, Actuarial & CAT Underwriting"]
        D8["Urban Commuter & Public Transit Networks"]
        D9["Student, School Administrator & Youth Sports"]
        D10["Cold Chain & Temperature-Sensitive Logistics"]
        D11["Maritime, Coastal Fishery & Port Authorities"]
        D12["Aviation & Flight Operations"]
        D13["Tourism, Pilgrimage & Outdoor Event Organizers"]
    end

    Layer 1 --> Layer 4
    Layer 2 --> Layer 4
    Layer 3 --> Layer 4
```

## 3.2 Core architectural rule

A hazard module must not depend directly on another hazard module's internal implementation. Every module consumes validated shared input contracts and emits a common prediction contract. The CompoundRisk engine consumes those outputs through the contract, not by importing private model internals.

## 3.3 Keep these concerns separate

- **Source data:** what an external provider supplied.
- **Observation/analysis:** measured or analysed historical condition.
- **Forecast:** what a forecast system predicted for a future valid time.
- **ML estimate:** what Clima-Core's trained model estimates.
- **Risk score:** a derived interpretation of a hazard and, later, exposure/vulnerability.
- **Official warning:** a warning issued by an authorised agency.

Do not collapse these into a single generic `prediction` field without a `product_type` or equivalent provenance label.

---

# 4. Data Sources and Canonical Dataset Registry

The following registry is the **single source of truth** for all data ingested and processed in Clima-Core. Every dataset has a documented provenance, spatial resolution, temporal range, row count, and cryptographic SHA-256 manifest.

### Canonical Ingested Dataset Registry

| Source ID | Processed Path | Records / Features | Temporal Range | Provider / Citation | Purpose & Spatial Resolution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `india_districts` | `data/processed/boundaries/india_districts.geojson` | 594 Districts (`EPSG:4326`) | Static 2024 | Survey of India / NWIC | Administrative boundary polygons & spatial clipping |
| `india_states` | `data/processed/boundaries/india_states.geojson` | 36 States/UTs (`EPSG:4326`) | Static 2024 | Survey of India / NWIC | State macro-regions & disaster management boundaries |
| `worldpop_2024` | `data/processed/population/city_population_exposure.parquet` | 50 cities (210.4M people evaluated) | 2024 Estimated | WorldPop / Univ. of Southampton | 100m constrained population; 5km, 15km, 30km concentric buffers; dense clusters ($\ge 7,500/\text{km}^2$) |
| `osm_infrastructure` | `data/processed/infrastructure/city_infrastructure_exposure.parquet` | 50 cities (107k+ assets mapped) | Static 2024 | OpenStreetMap / Geofabrik | Hospitals, clinics, schools, bridges, drainage canals/waterways, arterial roads, transit hubs |
| `ifi_flood_inventory` | `data/processed/floods/city_flood_history_aligned.parquet` | 803,550 daily rows (10,443 disaster-days) | 1980–2023 | IIT Delhi / Zenodo | 44-year ground truth disaster event records (fatalities, displacement, severity) aligned across 50 cities |
| `agrisense_crops_soil` | `data/processed/agriculture/city_district_crop_profiles.parquet` | 34,903 district-season rows | 1997–2015 | ICRISAT / SoilGrids 250m / ICAR | 20 soil physicochemical properties (pH, SOC, CEC, texture, WHC) & dominant multi-season crops |
| `electricity_demand` | `data/processed/electricity/city_daily_electricity_stress.parquet` | 150,547 daily rows | 2013–2020 | Univ. of Reading / Newcastle / POSOCO / CEA | Daily energy met (MU), peak load (MW), unmet load shedding (MW), UTCI, CDD across 50 cities |
| `crop_calendar_sage` | `data/processed/agriculture/city_crop_calendars.parquet` | 249 crop cycles across 50 cities | Climatological | Univ. of Wisconsin SAGE (Sacks et al.) | Sowing, Anthesis/Flowering, and Harvesting DOY windows for Rice, Wheat, Cotton, Maize, Pulses |
| `chetna_power_plants` | `data/processed/power_plants/city_power_plant_exposure.parquet` | 769 thermal facilities (50 cities evaluated) | 2018–2025 | IIT Bombay / LSCE / Zenodo | 50km, 100km, 250km regional thermal capacity (MW), annual CO2 (Mt), and closest plant bearing/distance |
| `imd_temperature_74y` | `data/processed/imd_temperature_74y_50_cities.parquet` | 1,351,450 daily rows | 1951–2024 | IMD Pune (Srivastava et al., 2009) | 1.0° daily temperature grids (MaxT, MinT, DTR, Return Periods) across 50 cities |
| `imd_rainfall_46y` | `data/processed/imd_rainfall_50_cities.parquet` | 840,100 daily rows | 1980–2025 | IMD Pune (Pai et al., 2014) | 0.25° physical ground rain gauge reanalysis across 50 cities |
| `imd_cyclones_45y` | `data/processed/cyclones/north_indian_ocean_cyclone_tracks_1982_2026.parquet` | 7,456 track points (421 storms) | 1982–2026 | IMD RSMC New Delhi | 3h/6h cyclone trajectories, central pressure, gale radii, 100km/250km/500km city proximity buffers |
| `ndma_hap_triggers` | `data/processed/ndma_hap_triggers.parquet` | 50 cities categorized | Current Policy | NDMA & Municipal Corporations | Regional Heat Action Plan triggers (Plains, Coastal, Hills, and Pioneer Municipal HAPs) |
| `era5_soil_moisture` | `data/processed/era5_soil_moisture_50_cities.parquet` | 230,118 daily rows | 1980–2024 | ECMWF / Copernicus CDS | Topsoil (0–7 cm), root-zone (7–28 cm), & SMDI across 50 cities |
| `cgwb_groundwater` | `data/processed/cgwb_groundwater_levels.parquet` | 437,561 well records | 2014–2024 | Central Ground Water Board / NWIC | Aquifer water table depth (mbgl) and groundwater drought depletion rates |
| `cwc_reservoirs` | `data/processed/cwc_reservoir_storage.parquet` | 1,417,209 telemetry rows | 2010–2024 | Central Water Commission / NWIC | 150+ major national dam storage percentages and water reserve buffers |
| `city_weather_aqi_50` | `data/processed/weather_daily.parquet` | 125,500 daily rows | 2020–2026 | Open-Meteo / CAMS / ERA5 | Starter 50-city weather and air quality baseline |

### Ingestion & Processing Relationships Explained

- **IFI Flood Inventory:** The raw database contains 10,443 confirmed disaster event records with recorded human fatalities and displacements. When projected across the 50 urban centers over the 44-year observation span (1980–2023), it yields 803,550 city-day records, with 10,443 positive flood-disaster days.
- **AgriSense Soil & Crops:** Combines 34,903 district-season records from 646 districts with SoilGrids 250m raster points mapped to the 50 city agricultural hinterlands.
- **CHETNA Power Plants:** Evaluates 769 thermal power generation facilities geo-located across India. For each of the 50 cities, concentric Haversine distance scans compute installed generation capacity (MW) and emissions within 50 km, 100 km, and 250 km.

---

## 4.1 Priority starter dataset: India Daily Weather & Air Quality — 50 cities

**Dataset landing page:**  
https://huggingface.co/datasets/AkshayKarthick/india-weather-air-quality  
**Kaggle mirror:**  
https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality

**Current documented coverage at review time (15 Nov 2026):**

- `weather_daily.csv`: approximately 125,500 rows, 1 January 2020 to 14 November 2026.
- `air_quality_daily.csv`: approximately 78,150 rows, 5 August 2022 to 14 November 2026.
- `cities.csv`: 50 city records with state, region, coordinates, and elevation.
- Declared dataset license: CC BY 4.0. Retain the creator's attribution and the upstream provider attributions.

### Weather columns documented by the dataset

- `date` — local date, Asia/Kolkata.
- `city` — city name.
- `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean` — °C.
- `apparent_temperature_max` — °C.
- `precipitation_sum` — mm.
- `precipitation_hours` — hours.
- `relative_humidity_2m_mean` — percent.
- `wind_speed_10m_max` — km/h.
- `wind_direction_10m_dominant` — degrees from north.
- `shortwave_radiation_sum` — MJ/m².
- `et0_fao_evapotranspiration` — mm.
- `weather_code` — WMO weather code.

### Air-quality columns documented by the dataset

- `date`, `city`.
- `pm2_5_mean`, `pm2_5_max` — µg/m³.
- `pm10_mean`, `pm10_max` — µg/m³.
- `no2_mean`, `so2_mean` — µg/m³.
- `o3_8h_max`, `co_8h_max` — µg/m³.
- `dust_mean` — µg/m³.
- `aod_mean` — aerosol optical depth.
- `us_aqi_max`.
- `hours_available`.
- `india_aqi`, `india_aqi_category`, `dominant_pollutant`.
- `india_aqi_pm`, `india_aqi_pm_category`.

### Important dataset limitations

- The city values are for model grid cells at city centres; they are **not automatically ground-station observations**.
- The weather history is based on Open-Meteo historical weather sources, including ERA5/ERA5-Land and ECMWF IFS products; approximate native resolution varies by product (roughly 9–25 km according to the dataset description).
- The air-quality history is based on the CAMS global model through Open-Meteo, around 45 km resolution, not CPCB ground monitors.
- The dataset notes that CAMS surface ozone may be overestimated over India. Its `india_aqi_pm` field, based on PM2.5 and PM10, is recommended over its all-pollutant `india_aqi` field for many analyses.
- Air-quality history starts later than weather history.
- City names and local-day aggregation must be checked before any join. Use `city` plus the documented local date, not city name alone.

### Intended use

This is the **first dataset to inspect and use for the starter data pipeline** because it provides manageable, structured daily weather and air-quality histories for 50 Indian cities. It can support EDA, feature-pipeline tests, prototype time-series baselines, and a city-level AirQuality experiment.

It is not, by itself, a historical forecast archive. Do not claim that a model trained on its historical weather rows has learned to correct the forecast system. For genuine forecast correction, pair archived forecast runs with verifying data.

## 4.2 High-priority forecast-training source: Open-Meteo Previous Runs API

**Documentation:** https://open-meteo.com/en/docs/previous-runs-api

Purpose: obtain forecasts from previous model runs at fixed lead-time offsets and compare them with later verifying data. This is a key source for training and evaluating forecast-bias-correction models.

The API documentation describes lead offsets from day 0 through day 7. For example, a `temperature_2m_previous_day1` value represents a temperature forecast issued 24 hours before its valid time. The archive begins around January 2024 for most models; GFS 2 m temperature extends back to March 2021, while some JMA products have older availability. Model and variable coverage varies.

Variables listed in the current documentation include:

- 2 m temperature.
- Relative humidity and dew point.
- Apparent temperature.
- Precipitation, rain, and showers.
- Weather code.
- Sea-level and surface pressure.
- Cloud cover.
- 10 m wind speed and direction.

### Use requirements

- Inspect exact model, variable, lead-time, date, and location availability before choosing a training sample.
- Do not assume every model has every lead time or every location/time populated.
- Preserve the selected model/provider identifier and lead time as metadata/features where appropriate.
- Align forecasts with observations by **valid time**, location/grid cell, model/provider, and lead time. Preserve issue/run time where the selected API product exposes it.
- Do not mix forecasts from different model providers as if they were identical. If blending them, make the blend explicit and evaluate it against each source baseline.
- The live Forecast API and historical Previous Runs API must use compatible variables, units, model selection, and feature transformations at inference time.
- The Previous Runs API provides fixed lead-time offsets. If a research question needs the complete forecast sequence from one initialisation, investigate the Single Runs API instead.

### Forecast data pairing rule

For a given location and valid date `t`:

- Feature: forecast for `t` that was available at a stated earlier lead time (for example, day-1 forecast).
- Target: verifying observation or analysis for `t` from a declared source.
- Optional features: additional forecast variables from the same run/lead time and past observations available before issue time.

**Never use the realised weather for date `t` as an input when predicting the realised weather for date `t`.** It is the target/verification data.

### Terms

Open-Meteo's free API is described as non-commercial/fair-use with request limits and CC BY 4.0 data attribution. The current terms specify thresholds including fewer than 10,000 API calls/day for the free API and require appropriate credit. Recheck the provider's current terms before public release or any commercial use. Do not assume a free research/hackathon allowance automatically covers future commercial deployment.

## 4.3 Open-Meteo live Forecast API

**Documentation:** https://open-meteo.com/en/docs

Purpose: future inference-time source for current forecasts. It is not a substitute for historical training data.

Use it only after the training and inference contract is established. Cache responses; request multiple locations efficiently; avoid one external call per map feature. Store retrieval time, model/provider, valid time, units, and raw-response provenance.

Free API terms are non-commercial/fair-use and require attribution under CC BY 4.0. Check terms before production/commercial usage.

## 4.4 IMD daily gridded rainfall — long historical precipitation record

**Official IMD page:**  
https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html

Current documentation describes:

- Daily rainfall over India at 0.25° × 0.25°.
- 1901–2024 coverage.
- 135 × 129 grid points.
- Rainfall in millimetres.
- NetCDF yearly files with 365/366 daily records.

Purpose:

- Long-term precipitation analysis.
- Rainfall climatology and anomaly features.
- Candidate verifying/target dataset for rainfall studies, subject to spatial/temporal alignment and source limitations.

Limitations:

- This is a historical gridded rainfall dataset, not a live forecast feed.
- It does not by itself train a future forecast model; forecast features must be supplied from a forecast archive.
- Check the actual download format, coordinate order, missing-value convention, calendar handling, and file integrity before processing.
- The page provides a recommended scientific citation and disclaimer. Do not assume redistribution rights that are not explicitly stated. Preserve attribution and verify terms before publishing the raw data or derived redistribution.

Recommended paper citation shown by IMD: Pai et al. (2014), development of a 0.25° daily gridded rainfall dataset over India, *MAUSAM*, 65(1), pp. 1–18.

## 4.5 ERA5-Land reanalysis

**Copernicus Climate Data Store:**  
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land

Documented characteristics:

- Global land reanalysis.
- Approximately 0.1° / 9 km native grid.
- Hourly data from 1950 to present.
- Includes near-surface meteorological and land variables, including soil moisture and soil temperature.
- Dataset page lists a CC BY licence.

Purpose:

- Historical weather/context features.
- Soil-moisture and land-condition research for a future WaterRisk module.
- A consistent analysis/reanalysis source for model development where appropriate.

Important: ERA5-Land is a reanalysis product, not an independent physical station measurement. If both forecast input and verification target come from related model/reanalysis systems, describe the task honestly as forecast-to-reanalysis correction/estimation, and seek independent station validation where feasible.

Download only the required India bounding box, variables, temporal interval, and date range. Do not download the full global archive for this MVP.

## 4.6 CHIRPS rainfall dataset

**Dataset information:**  
https://www.chc.ucsb.edu/data  
**Earth Engine catalogue:**  
https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY

Documented characteristics:

- Daily gridded rainfall.
- Approximately 0.05° spatial resolution.
- Starts in 1981.
- Combines satellite imagery and station data.
- Public-domain statement is shown in the Earth Engine catalogue.

Purpose: an alternative/complementary rainfall history for spatial rainfall analysis, drought/monsoon context, or cross-checking. Confirm exact product version, end date, provisional/final status, India coverage, download mechanism, and units before use.

CHIRPS is a precipitation estimate product, not a direct station observation. Keep it distinct from IMD gridded rain and model forecasts.

## 4.7 NASA GPM IMERG

**Official data page:** https://gpm.nasa.gov/data/imerg

Purpose: satellite-based precipitation estimates, including near-real-time and historical products. The programme describes half-hourly products and a record spanning more than two decades through TRMM/GPM continuity.

Use for event comparison or later higher-frequency rainfall research only if the required product, version, access method, latency, unit scaling, and license/data policy have been checked. Do not begin by downloading the global archive.

## 4.8 CPCB / Government of India real-time AQI data

**OGD catalogue:**  
https://www.data.gov.in/catalog/real-time-air-quality-index

The catalogue describes hourly real-time National AQI and pollutant values from monitoring stations, with CPCB identified as the source agency. It also warns that live field-instrument feeds may contain abnormal values.

Purpose:

- Ground-monitor observations for AirQuality validation where station coverage and access permit.
- Later live station context.

Limitations:

- The real-time feed is not automatically a convenient, complete historical training archive.
- Inspect station metadata, pollutant availability, API access requirements, update frequency, units, data gaps, calibration/quality flags, and historical download access.
- Do not merge CPCB station data with CAMS grid-cell data as if they were interchangeable. A station-to-grid comparison requires explicit spatial matching and a documented methodology.

## 4.9 NOAA Global Summary of the Day / station datasets — optional independent validation

**NCEI overview:**  
https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc%3AC00516/html  
**NCEI land-based station data:**  
https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database

The NOAA GSOD description covers thousands of global stations and daily temperature, pressure, wind, precipitation, and weather indicators. Station coverage and field availability vary.

Use only after filtering Indian stations, checking the station inventory, units, quality/flag columns, time conventions, and missingness. NOAA's daily summaries use UTC day conventions and may not line up with India local-day definitions. A different station daily-summary product may be more suitable for some variables.

This is an optional validation source, not a guaranteed dense India-wide dataset.

## 4.10 INDmet — candidate high-resolution India dataset, licensing unresolved

**Zenodo record:** https://zenodo.org/records/15430548

Documented characteristics:

- Daily precipitation, maximum temperature, and minimum temperature.
- 1981–2024.
- 0.05° grid.
- Derived using IMD observations, CHIRPS, and ERA5-Land inputs.
- District, state, taluka, gridded, and NetCDF collections are listed.

The Zenodo record currently does not display a clear license in its rights field. **Do not use it as a production dependency or redistribute its files until license/permission is clarified with the authors.** It may be investigated as a candidate dataset and used only when its terms are resolved.

The dataset is a blended/regridded product. Do not describe it as independent station observations.

## 4.11 Kaggle Rainfall in India — climate context only

**Dataset:** https://www.kaggle.com/datasets/rajanand/rainfall-in-india

The dataset describes monthly rainfall for 36 meteorological subdivisions from 1901–2015, plus district rainfall normals for 1951–2000. Its listed license is CC BY-SA 4.0.

Use for seasonal/climate context and basic sanity checks. It is **not** the primary source for daily or hourly risk prediction because its main historical rainfall data are monthly and subdivision-level.

## 4.12 IndiaWeatherBench — research-only, too large for initial MVP

**Dataset:** https://huggingface.co/datasets/tungnd/IndiaWeatherBench

This is a large gridded weather dataset with many channels and a multi-decade time series. It is cited as a research candidate, not an MVP dependency. The approximate size previously identified is hundreds of gigabytes; inspect the current dataset card and exact license before any use. Its listed non-commercial/share-alike terms should be checked. Do not download it for the first milestone.

## 4.13 Geographic boundaries — future map support

**National Water Data Portal / NWIC district boundary dataset:**  
https://nwdp.nwic.gov.in/dataset/district-boundary

The portal lists district-boundary KML, GeoJSON, and shapefile resources associated with Survey of India and describes 733 districts in that dataset's metadata.

Before use:

- Confirm the current boundary vintage, state/district codes, names, and geometry validity.
- Verify usage/redistribution terms and attribution.
- Check whether the district count and boundaries align with the date of any environmental data.
- Avoid joining by district name alone; use stable codes or a maintained mapping table.

Do not assume a boundary dataset is automatically open for every use because it can be downloaded.

## 4.14 Additional India city-weather dataset — 1980 onward (candidate)

**Kaggle discussion and dataset links:**  
https://www.kaggle.com/discussions/accomplishments/543282

The author's dataset collection describes hourly and daily weather for 50 populous Indian cities from 1980 to the present, with decade-wise files. It states that the underlying weather source is Open-Meteo.

Use as an optional longer-history candidate, especially for seasonal climatology and city-level weather research. Before use, open the actual linked decade datasets, inspect their schemas, file availability, source attribution, date coverage, missing values, and licenses. Do not treat the Kaggle discussion itself as a complete data card or proof that every decade file is available. This may overlap with the 50-city starter dataset, so compare data quality rather than automatically downloading both.

## 4.15 Daily Delhi Climate — narrow local candidate

**Kaggle dataset:** https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data

Previously identified as a Delhi daily climate time series covering approximately 2013–2017, with a CC0 listing. Verify the current data card, exact schema, source, and license before use.

Use only for a Delhi-specific exploratory baseline or code test. It is too geographically narrow and short to support a nationwide Clima-Core claim.

## 4.16 World Weather Repository — secondary candidate, license to verify

**Kaggle dataset:** https://www.kaggle.com/datasets/nidulaelgiriyewithana/world-weather-repository

Previously identified as a global city weather and AQI dataset beginning around August 2023, with many variables. The license and upstream provenance require verification.

Do not make this a core dependency until its license, data source, time coverage, locations, and variable definitions are checked. It may be useful for a secondary comparison, but its shorter history makes it unsuitable as the only long-term training source.

## 4.17 Weather Forecasting — Datavidia competition dataset

**Kaggle competition:** https://www.kaggle.com/competitions/weather-forecasting-datavidia/data

The competition page describes daily and hourly weather files and a dataset size around 77.85 MB. Its license is listed as **Subject to Competition Rules**, and access is invitation/rules restricted.

Treat as a competition-only research candidate. Do not reuse or redistribute its data outside the competition's allowed scope without checking and complying with the rules. Do not make this a default Clima-Core dependency.

## 4.18 IMD gridded maximum and minimum temperature — additional historical targets/context

**Maximum temperature (1° daily):**  
https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html  
**Minimum temperature (1° daily):**  
https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html

Current IMD pages describe:

- 1.0° × 1.0° daily gridded maximum and minimum temperature data.
- 1951–2024 coverage.
- 31 × 31 grid points.
- Temperature in °C.
- The pages note that data from 2008 onward are based on a relatively smaller number of stations (around 180) reporting operationally in real time.

Purpose: potential historical temperature target/context for HeatRisk research, with a coarser spatial resolution than the city-centre dataset and a caveat about station counts after 2008.

These files are binary and require correct grid/time parsing. Inspect the sample reader and missing-value conventions. The page lists a recommended citation to Srivastava et al. (2009) and an IMD disclaimer. Do not assume that downloadable means unrestricted redistribution; check the applicable terms and cite the provider.

## 4.19 CHIRPS-GEFS — candidate external rainfall forecast source

**Climate Hazards Center page:** https://chc.ucsb.edu/data/chirps-gefs  
**Related dataset information:** https://www.chc.ucsb.edu/data

Important current product update:

- The Climate Hazards Center states that, as of July 2026, its operational forecast stream uses **CHIRPS3-GEFS**, with CHIRPS version 3 used for bias correction.
- CHIRPS2-GEFS operational production ended on 1 July 2026.
- The CHIRPS-GEFS family combines NCEP GEFS precipitation forecasts with CHIRPS-based bias correction/downscaling.
- Current CHIRPS3-GEFS products include daily 5-day, 10-day, and 15-day forecasts and other accumulation products; the dataset page describes next 1-to-16-day forecast capability at 0.05°.
- The current CHIRPS3-GEFS archive is described as beginning in January 2001, with product details and coverage depending on forecast type.

This is a potentially useful forecast input for RainRisk. It is not an independent observation target because its bias correction uses CHIRPS. Before use, inspect the exact forecast archive, issue/valid-time structure, version, file availability, units, missing dates, and license/terms. Preserve the fact that CHIRPS3-GEFS is an external forecast product. Compare it against IMD/other verifying data and do not imply Clima-Core generated the raw forecast.

## 4.20 Starter data acquisition order

For the initial audit:

1. Inspect the 50-city India Daily Weather & Air Quality dataset. Download only the three documented CSV files needed for the first data pipeline.
2. Query a very small India sample from Open-Meteo Previous Runs API to test whether historical forecasts, variables, locations, and lead times are available as required.
3. Inspect a small IMD rainfall NetCDF sample or one year before attempting a large date range.
4. Only then decide the first model's final training table.
5. Add CPCB/NOAA station validation if accessible and useful.
6. Defer INDmet until license clarification; defer IndiaWeatherBench and large global downloads.

---

# 5. Data Engineering Rules

## 5.1 Raw, interim, and processed data separation

Use this structure (adjust names only if an existing repository already has a sound equivalent):

```text
data/
  raw/          # Original downloaded files; never edit in place; gitignored
  interim/      # Parsed, normalised, joined but not final training tables
  processed/    # Versioned features/labels for training; usually Parquet
  external/     # Small third-party reference files if license permits
  manifests/    # Source manifests, checksums, retrieval details
```

Do not commit large raw or processed data files to Git. Add `.gitkeep` files if needed. Keep a reproducible manifest for every dataset version.

## 5.2 Preferred file formats

- CSV: source data when that is what the provider supplies.
- Parquet: processed tabular datasets and feature tables.
- NetCDF/GRIB: gridded meteorological source formats where supplied.
- GeoJSON: small geographic boundary/reference files only when permitted and practical.
- JSON/YAML: metadata, configuration, model cards, and manifests.

Use `float32` where precision permits. Preserve original integer/time/string fields as needed. Do not downcast blindly.

## 5.3 Time conventions

- Use explicit UTC timestamps for forecast issue/run time and valid time when available.
- Use `Asia/Kolkata` for Indian local-day features and daily aggregations where the source defines local dates.
- Never assume midnight UTC equals midnight India local time.
- Store `issue_time`, `valid_time`, `lead_hours` or `lead_days`, and `timezone` explicitly for forecasts.
- Store data retrieval time separately from the time the weather applies to.
- Check leap years, daylight/time-zone metadata, and the source's daily aggregation definition.

## 5.4 Spatial conventions

- Use WGS84 latitude/longitude for geographic coordinates unless a source specifies otherwise.
- Preserve original grid coordinates and resolution.
- Keep a stable `location_id`/`grid_id`/`station_id`; do not use display names as the only key.
- Record whether values refer to a station, grid cell, city centre, district average, or polygon aggregate.
- Use spatial joins only after checking coordinate reference systems, geometry validity, and edge cases.
- Do not resample or interpolate without recording the method and resulting resolution.

## 5.5 Data quality checks required before training

At minimum, generate a report covering:

- File existence, file size, checksum, encoding, and schema.
- Required-column presence and unexpected-column changes.
- Date parsing and date range.
- Duplicate primary keys.
- Missing values by column, city/station/grid, month, and year.
- Missing date sequences and irregular sampling.
- Units and unit conversions.
- Physically implausible values and documented source flags.
- Outliers that may be real extremes versus data errors.
- City, state, station, grid, and coordinate consistency.
- Forecast/observation join coverage by lead time and model.
- Target availability and label balance.
- Distribution shift between train, validation, and test periods.

Do not automatically drop extreme weather observations as outliers. They may be the most important examples. Flag them, investigate provenance, and document any exclusion rule.

## 5.6 Missing-value policy

- Never fill all missing values with zero.
- Distinguish “no rainfall” from “rainfall missing”.
- Distinguish zero pollutant concentration from unavailable pollutant data.
- Use only causal/time-safe imputation based on information available at prediction time.
- Fit imputation/preprocessing parameters on training data only.
- Add missingness indicators when they carry information and are justified.
- Record every excluded row and why it was excluded.

## 5.7 Provenance and manifests

Each dataset manifest should include fields equivalent to:

```yaml
source_id: india_weather_air_quality_50_cities
source_url: https://huggingface.co/datasets/AkshayKarthick/india-weather-air-quality
retrieved_at_utc: "<filled by ingestion script>"
license: CC-BY-4.0
upstream_sources:
  - Open-Meteo Historical Weather API
  - Open-Meteo Air Quality API / CAMS
files:
  - name: weather_daily.csv
    sha256: "<computed checksum>"
  - name: air_quality_daily.csv
    sha256: "<computed checksum>"
  - name: cities.csv
    sha256: "<computed checksum>"
coverage:
  weather_start: "2020-01-01"
  weather_end: "<read from file>"
  air_quality_start: "<read from file>"
  air_quality_end: "<read from file>"
spatial_unit: city-centre model grid cell
value_type: model-derived historical estimate
notes: "Do not present as CPCB station observations."
```

Do not leave placeholders in a final generated manifest. The ingestion code must fill actual retrieval times, checksums, and observed coverage from the files.

---

# 6. Machine-Learning & Scientific Intelligence Strategy

Clima-Core strictly structures all computational models across the **4-Layer Intelligence Architecture**. Deterministic equations and GIS spatial overlays are never presented as trained ML models.

## 6.1 The 4-Layer Architectural Separation

1. **Layer 1: Predictive Intelligence (ML / Statistical Models)**
   - Supervised GBDT boosters (LightGBM, CatBoost), Quantile Regressors, and transfer calibrators.
   - Every model has an explicit target, feature set, loss function, chronological holdout split, baseline comparison, and saved `.booster` artifact.
2. **Layer 2: Physical Intelligence (Atmospheric & Hydrological Formulations)**
   - Established deterministic thermodynamic and hydrological formulas: elevation-aware Stull wet-bulb temperature, FAO-56 Penman-Monteith evapotranspiration, Cooling Degree Days, Antecedent Precipitation Index, and Extreme Value Theory (GEV) return periods.
3. **Layer 3: Exposure Intelligence (Geospatial & Lifeline Assets)**
   - Static and dynamic spatial datasets: WorldPop 2024 100m population rasters, OSM critical infrastructure nodes (hospitals, schools, bridges, drainage canals), SAGE crop calendar phenological windows, CHETNA thermal power plant locations, and CWC/CGWB water storage levels.
4. **Layer 4: Decision Intelligence (13 Persona Adapters)**
   - Tailored operational advisory and civil protection logic translating Layer 1, 2, and 3 signals into sector-specific action plans, threshold breaches, and emergency restrictions.

---

## 6.2 Flagship 1: FloodRisk V2 (Hazard Probability vs. Human Exposure)

- **Source:** IIT Delhi India Flood Inventory (IFI) v3 (1980–2023 disaster event records).
- **Core Principle:** Strictly decouple flood occurrence probability from human/infrastructure impact.
- **Component A: Flood Occurrence Classifier (`flood_risk_v1.booster`)**
  - **Target:** Occurrence of a declared district flood disaster event.
  - **Features:** 1d, 3d, 7d, 14d, 30d antecedent rainfall; 14-day Antecedent Precipitation Index ($\text{API}_{14d}$); ERA5-Land topsoil ($0-7\text{ cm}$) and root-zone ($7-28\text{ cm}$) volumetric moisture; OSM drainage canal count and drainage-to-arterial road ratio.
  - **Holdout Test Metrics:** ROC-AUC $= \mathbf{0.7931}$, Recall $= \mathbf{73.70\%}$ (at calibrated threshold 0.35, vs. 13.83% naive rainfall threshold).
- **Component B: Human & Infrastructure Exposure Overlay**
  - Evaluates WorldPop 15km population, dense informal cluster population ($\ge 7,500/\text{km}^2$), and counts of hospitals, schools, and bridges exposed within the predicted hydrological basin.

---

## 6.3 Flagship 2: HeatRisk V3 (Urban Microclimate & Dual Diurnal Regressors)

- **Source:** 44-year ERA5-Land reanalysis, 74-year IMD 1.0° daily temperature grids, and NDMA Heat Action Plans.
- **Component A: Dual Diurnal Models**
  - **Daytime Model (`temp_max`):** Corrects peak daytime solar heating and advection.
  - **Nighttime Model (`temp_min`):** Predicts nocturnal minimum temperature and tropical night conditions ($T_{\min} \ge 28^\circ\text{C}$), critical for transformer thermal failure.
- **Component B: IMD Heatwave 4-Tier Severity Classifier (`heatwave_alert`)**
  - Classifies Normal, Yellow, Orange, and Red Alert heatwaves based on consecutive threshold departures over climatological normal (Test Weighted F1: 0.88).
- **Component C: Oke Urban Heat Island Microclimate Anomaly (`uhi_temp_delta`)**
  - Models urban canopy layer excess temperature ($\Delta T_{\text{UHI}}$) from building geometry, impervious density, and nocturnal wind stagnation.

---

## 6.4 Flagship 3: PowerGrid V2 (Heat-to-Electricity Stress Intelligence)

- **Source:** POSOCO / CEA state-level electricity telemetry (2013–2020) + ERA5-Land / UTCI.
- **Component A: Peak Grid Load Regressor (`peak_grid_load_v1.booster`)**
  - **Target:** Daily state peak electricity load in Megawatts (MW).
  - **Features:** $T_{\min}, T_{\text{mean}}, T_{\max}$, dewpoint, mean/peak UTCI, Cooling Degree Days (CDD), tropical night flag, diurnal range, solar radiation, day-of-week, metro population exposure.
  - **Holdout Test Metrics:** $R^2 = \mathbf{0.8758}$, MAE $= \mathbf{1,519.18\text{ MW}}$ on unseen 2020 holdout data.
- **Component B: Grid Unmet Supply Stress Classifier (`grid_deficit_hazard_v1.booster`)**
  - **Target:** Occurrence of unmet electricity supply / load shedding under thermal grid overload ($\text{peak\_unmet\_mw} > 0$).
  - **Holdout Test Metrics:** ROC-AUC $= \mathbf{0.9521}$, PR-AUC $= \mathbf{0.8065}$, Deficit Recall $= \mathbf{96.41\%}$ (at calibrated 0.40 threshold).
  - *Domain Note:* Labeled as **Unmet Supply Stress**, acknowledging that load shedding reflects supply deficits rather than unconstrained blackouts.

---

## 6.5 Flagship 4: AgriSense V2 (Agro-Climatic Suitability & Phenological Stress)

- **Source:** ICRISAT / ICAR District Crop Statistics (1997–2015), SoilGrids 250m, and Univ. of Wisconsin SAGE Crop Calendar.
- **Component A: Soil-Agro Crop Suitability Classifier (`crop_suitability_v1.booster`)**
  - **Target:** Optimal crop class among 10 dominant Indian crops (Rice, Wheat, Sugarcane, Maize, Potato, Mustard, Groundnut, Bajra, Gram, Jowar).
  - **Features:** 8 soil physicochemical properties (pH, organic carbon, CEC, nitrogen, sand/clay/silt %, water holding capacity) + 11 seasonal climate variables.
  - **Holdout Test Metrics:** Top-1 Accuracy $= \mathbf{82.49\%}$, Top-3 Accuracy $= \mathbf{97.62\%}$, Weighted F1 $= \mathbf{0.8250}$.
- **Component B: SAGE Phenological Growth-Stage Engine**
  - Tracks 249 state-level Indian crop growth cycles. Identifies exact calendar windows for Sowing, Anthesis/Flowering, and Harvesting.
  - Triggers **Phenological Anthesis Heat Alerts** ($T_{\max} \ge 32^\circ\text{C}$ during flowering causing floret abortion) and **Harvest Lodging Rain Warnings** ($\ge 10\text{ mm}$ rain during maturity).

---

## 6.6 Flagship 5: AirQuality V2 (Multi-Pollutant Transfer & Atmospheric Stagnation)

- **Source:** CPCB CAAQMS ground monitors (2015–2020) and CAMS reanalysis.
- **Component A: Multi-Pollutant Transfer Calibrators (`cpcb_multipollutant`)**
  - Calibrates physical station ground realities for $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$ from regional meteorological and atmospheric advection vectors.
- **Component B: Atmospheric Stagnation & Inversion Classifier (`atmospheric_stagnation`)**
  - Predicts severe ground-level pollutant trapping from boundary layer height collapses and nocturnal surface temperature inversions.
- **Component C: Indo-Gangetic Transboundary Smoke Regressor (`transboundary_smoke`)**
  - Models agricultural stubble fire advection plumes using upwind trajectory projections.

---

## 6.7 RainRisk V2 Suite: Extreme Precipitation & Quantile Estimation

- **Source:** IMD Pune 46-year 0.25° gridded daily rainfall (1980–2025) and ERA5-Land.
- **Architecture:** Two-stage multi-tier design:
  1. `LightGBMClassifier` for rain occurrence ($\ge 0.1\text{ mm}$, ROC-AUC: 0.892).
  2. Quantile Regressors for median ($\text{Q}_{50}$), heavy ($\text{Q}_{90}$), and extreme cloudburst ($\text{Q}_{99}$) tail precipitation.
  3. `heavy_rain_hazard` binary classifier for IMD Heavy Rain exceedance ($\ge 64.5\text{ mm/day}$, ROC-AUC: 0.924).

---

## 6.8 Extreme Value Statistics Engine (Rigorous EVT vs. Daily Percentiles)

- **Correction of Daily Percentile Substitutions:** Clima-Core rejects the naive mapping of daily observation percentiles ($P_{90}, P_{98}, P_{99.5}$) to annual return periods (1-in-2, 1-in-10, 1-in-25 years).
- **Formal EVT Framework:**
  - Extracts the **Annual Maximum Series (AMS)** across the 74-year IMD temperature grids (1951–2024) and 46-year IMD rainfall grids (1980–2025) for each of the 50 cities.
  - Fits a **Generalised Extreme Value (GEV)** distribution using `scipy.stats.genextreme`:
    $$G(x) = \exp\left\{ -\left[ 1 + \xi \left( \frac{x - \mu}{\sigma} \right) \right]^{-1/\xi} \right\}$$
  - Derives true $T$-year return levels ($x_T$) for $T \in \{5, 10, 25, 50, 100\text{ years}\}$ with parametric bootstrap confidence intervals.

---

## 6.9 Transparent Multi-Hazard Compound Risk Index

- **Correction of Pseudo-Copula Terminology:** Clima-Core explicitly clarifies that its multi-hazard aggregation is an **empirical, probabilistic joint exceedance index**, not a fitted parametric copula.
- **Formulation:** Evaluates joint marginal exceedances across Heat, Air Quality, and Severe Rain, incorporating physical synergy amplification penalties when multiple tail hazards co-occur.

---

## 6.10 Sourced Physical Formulations & Elevation Awareness

1. **Elevation-Aware Stull Psychrometric Wet-Bulb Temperature:**
   - Standard Stull (2011) formula is adjusted for barometric pressure variation at elevated cities (e.g., Bangalore at 920m, Pune at 560m, Hyderabad at 540m).
2. **Cooling Degree Days (CDD):**
   - Computed as $\max(0, T_{\text{mean}} - 18.0^\circ\text{C})$ based on standard Indian HVAC engineering reference standards.
3. **Soil Water Holding Capacity Index (WHC):**
   - Documented empirical proxy: $\text{WHC} = 0.4 \cdot (\% \text{Clay}) + 0.2 \cdot (\% \text{Silt}) + 0.05 \cdot (\% \text{Sand}) + 0.1 \cdot (\text{SOC g/kg})$.

---

## 6.11 The 13 Domain Persona Adapters

Each day of the 7-day forecast evaluates all 13 specialized persona decision models:

| Persona ID | Title & Primary Beneficiary | Critical Inputs Consumed | Key Outputs & Decision Thresholds |
| :--- | :--- | :--- | :--- |
| `city_official` | City Official & Municipal Disaster Management | CDD, $T_{\min}$, PM2.5, rain, IFI flood risk, peak MW, blackout prob, CHETNA MW | Municipal Action Tier (Stage 1-3), DISCOM spinning reserve mandates, hospital generator pre-positioning |
| `farmer` | Farmer / Kisan Agro-Meteorological Advisor | FAO-56 $\text{ET}_0$, rain, wind, RH, SMDI, SAGE growth stages, soil WHC/pH | Water balance, spraying window, anthesis thermal alert ($T_{\max} \ge 32^\circ\text{C}$), harvest lodging warning |
| `industry` | Heavy Industry & Power Generation | $T_{\max}$, RH, wet-bulb, wind, stagnation, grid deficit prob, CHETNA 100km MW | Condenser efficiency loss %, turbine derating (wet-bulb $\ge 30^\circ\text{C}$), tank farm loading ban, stack throttling |
| `gig_worker` | Gig Worker & Outdoor Delivery Couriers | Apparent temp, wet-bulb, PM2.5, rain, gale gusts | Outdoor exertion stress, delivery shift restrictions, mandatory 15-min paid hydration breaks |
| `healthcare` | Healthcare & Hospital Surge | $T_{\max}$, apparent temp, PM2.5, PM10, NO2, OSM hospital count, WorldPop pop | ED presentation surge multiplier, acute respiratory distress index, additional presentations per hospital |
| `water` | Water Utility & Municipal Drainage | Rain, antecedent rain (3d, 7d), IFI flood risk, CWC dam %, CGWB water table, canals | Drainage capacity exceedance index, outfall desilting dispatch, sluice gate opening advisory, drought rationing |
| `insurance` | Parametric Insurance & CAT Underwriting | $T_{\max}$, rain, gale gust, IFI flood risk, GEV return periods, WorldPop exposure | Parametric trigger payout breach, catastrophe excess-of-loss index, claims liquidity reserving |
| `commuter` | Urban Commuter & Public Transit | Rain, apparent temp, PM2.5, gale gusts, OSM transit nodes & bridges | Road delay multiplier (1.0x-2.5x), metro track bucking risk ($T_{\max} \ge 43^\circ\text{C}$), underpass flooding alert |
| `student` | Student, School Admin & Youth Sports | $T_{\max}$, apparent temp, PM2.5, PM10, rain, gale gusts, OSM schools | Child heat illness index, outdoor sports suspension (PM2.5 $\ge 150$), mandatory school closure (PM2.5 $\ge 350$) |
| `cold_chain` | Cold Chain & Logistics | Ambient $T_{\max}$, RH, solar radiation, diurnal range, grid deficit prob | Reefer thermal ingress multiplier, auxiliary fuel burn surge, nighttime transit dispatch recommendation |
| `maritime` | Maritime, Fishery & Port Authority | Gale gusts, sustained wind, cyclone distance/grade, rain prob | Coastal wave agitation tier, mechanized trawler sea ban (wind $>45\text{ km/h}$), container crane shutdown |
| `aviation` | Aviation & Flight Operations | $T_{\max}$, surface pressure, wind, gale gusts, PM2.5 (optical haze proxy) | High density altitude takeoff payload penalty, low-level wind shear hazard, optical haze advisory |
| `tourism` | Tourism, Pilgrimage & Events | Apparent temp, wet-bulb, PM2.5, rain, gale gusts, IMD heat alert | Tourism outdoor comfort index (TOCI), mass pilgrimage queue thermal restriction (11:30 AM - 4:30 PM) |

---

---

# 7. Feature Engineering Contract

Feature engineering must be shared and reusable, but each model must declare its exact feature list in a configuration/model card.

## 7.1 Permitted feature groups

- Time: day of year, month, monsoon/season label if defined, day of week where relevant.
- Location: stable location/grid/station ID, latitude, longitude, elevation, region.
- Forecast: selected variables from the historical/live model forecast for the target valid time.
- Lagged observations: values available before the prediction issue time.
- Rolling features: mean, sum, max, min, standard deviation over past-only windows.
- Derived features: temperature-humidity indicators, rainfall accumulations, anomaly vs seasonal climatology, where formulas are documented.
- Forecast metadata: model provider, run/issue time, lead time, resolution, and source version when useful.

## 7.2 Lag and rolling-window rules

- Use `shift(1)` or equivalent before rolling over historical values when predicting the next valid day.
- Confirm feature timestamps are earlier than the model's issue time.
- Group by location before computing lags and rolling statistics.
- Sort chronologically within each group.
- Test the first rows of every location for correct null behaviour.
- Do not use backward fill from future dates.
- Do not calculate global statistics using validation/test periods.

## 7.3 Cyclical time features

Cyclical encoding (sine/cosine) may be used for day of year or wind direction, but retain a documented transformation. Use one consistent implementation across training and inference.

## 7.4 Feature parity

Training and inference must call the same feature-building implementation. Do not write one feature formula in a notebook and a different formula in the future API. Save ordered feature names and schema/version with each artifact.

---

# 8. Model Validation and Experiment Protocol

## 8.1 Chronological data splits

Use a chronological split, not random shuffling of rows. The exact cutoff dates must be selected after inspecting actual coverage.

A reasonable starting plan for sufficiently long records is:

- Earliest time period: training.
- Later contiguous period: validation and model selection.
- Most recent untouched period: final test.

For short forecast-archive records, use expanding-window or rolling-origin validation where possible. Ensure that the test period is not used for feature selection, threshold selection, hyperparameter tuning, or early stopping.

## 8.2 Group/spatial generalisation checks

Because multiple cities or grid cells share weather patterns, report both:

- Performance on future dates for locations seen during training.
- Where sample size permits, performance on held-out cities/regions or stations.

Do not claim nationwide generalisation from a small number of city points without a limitation statement.

## 8.3 Baselines required

At least one strong baseline per task:

- HeatRisk: raw official forecast; optionally a seasonal/climatology baseline.
- RainRisk: raw precipitation forecast plus a simple event/climatology baseline.
- AirQuality: persistence (tomorrow equals recent value) and seasonal baseline.

The ML model should not be considered successful merely because its score is non-zero or because it beats a deliberately weak baseline.

## 8.4 Metrics

Regression:

- MAE.
- RMSE.
- Mean bias/error.
- R² only as a supplementary metric, not the sole result.

Classification:

- Precision, recall, F1.
- PR-AUC when events are imbalanced.
- ROC-AUC only as an additional metric where meaningful.
- Confusion matrix.
- Brier score and calibration plot for probabilities.

Risk/rare events:

- Event recall and false-alarm rate.
- Performance by severity band.
- Performance across season/region where the sample permits.

## 8.5 Calibration and uncertainty

- A classifier's raw probability is not automatically calibrated.
- Use calibration methods only with a separate validation procedure.
- Prediction intervals require a defined method and empirical coverage test.
- If uncertainty has not been computed, do not display a fabricated confidence percentage.
- Store `uncertainty_method` and `uncertainty_value` as null/unknown where unavailable.

## 8.6 Acceptance criteria for an ML model

A model is not ready for deployment until:

1. Dataset provenance, terms, schema, units, and data quality are recorded.
2. Leakage checks pass.
3. A chronological holdout evaluation is complete.
4. It is compared against the correct baseline/source forecast.
5. Error is analysed by lead time and geography.
6. Model artifact can be reloaded and reproduces predictions.
7. Feature schema and preprocessing are versioned.
8. Model card records limitations and unsupported use cases.

Do not invent a universal required MAE value. Define model-specific go/no-go thresholds after observing the baseline and the intended operational tolerance.

---

# 9. Recommended Technology and Libraries

## 9.1 Training environment — current phase

Recommended language: **Python 3.12** unless a verified dependency requires another supported version.

Core libraries:

| Purpose | Library | Use |
|---|---|---|
| Data manipulation | `pandas` | Tabular data, joins, time series |
| Numerical computing | `numpy` | Arrays and transformations |
| Efficient storage | `pyarrow` | Parquet read/write |
| Classical ML and metrics | `scikit-learn` | Splits, preprocessing, baselines, metrics |
| Primary ML | `lightgbm` | Main tabular model family |
| Secondary ML | `xgboost` | Optional benchmark |
| API/data retrieval | `httpx` or `requests` | HTTP access to verified sources |
| Config | `PyYAML` and/or `pydantic-settings` | Config validation |
| Model persistence | Native LightGBM text format; XGBoost JSON/UBJSON | Versioned trained model files |
| Visual EDA/reporting | `matplotlib` | Basic plots in reports/notebooks |
| Explainability | `shap` | Optional after a model is trained; not mandatory for baseline |

Optional geospatial/scientific packages only when a task requires them:

- `xarray` for labelled multidimensional arrays.
- `netCDF4` for NetCDF.
- `cfgrib`/`eccodes` only if a selected GRIB source requires them and installation is tested.
- `geopandas`, `shapely`, `pyproj`, `rasterio` for later GIS/data preparation.

Do not install TensorFlow or PyTorch for the initial tree-based training pipeline. Do not install all geospatial libraries in the runtime environment by default.

## 9.2 Dependency management

- Use `pyproject.toml` and a lock file (`uv.lock` if using `uv`) or another reproducible pinned dependency workflow.
- Pin tested versions after the actual environment is built; do not blindly copy a version number from an example.
- Separate training-only dependencies from future inference dependencies, for example `requirements-train.txt` and `requirements-api.txt`, or optional dependency groups in `pyproject.toml`.
- Keep notebooks optional; production feature/model code must live in importable Python modules.
- Do not add `jupyter`, plotting, large GIS, or training-only dependencies to the Vercel inference bundle unless required.

## 9.3 Compute strategy

User development machine:

- Intel Core i3 11th Gen.
- 8 GB RAM.
- No dedicated GPU specified.

This is suitable for a modest tabular ML prototype with a manageable dataset, especially LightGBM/XGBoost on CPU. Use small batches and Parquet. Avoid loading large 22 GB grids or many expanded copies of the same data into memory. Antigravity Pro can help generate/debug the code; actual local Python training is executed by the local Python process, not by Vercel.

Kaggle notebooks may be used for larger data preparation or experiments. A GPU is not required for the planned tree-based models. Respect the current platform's session, storage, and accelerator quotas; do not assume unlimited execution.

Do not train models at Vercel request time. Vercel is a future inference host, not the training machine.

---

# 10. Suggested Repository Structure

First inspect the existing repository. If it already contains a consistent layout, adapt rather than destroying it. The target modular shape is:

```text
clima-core/
│
├── spec.md
├── README.md
├── pyproject.toml
├── uv.lock                         # if uv is selected
├── .python-version                 # 3.12 if compatible
├── .gitignore
├── .env.example                    # names only; no secrets
│
├── configs/
│   ├── project.yaml
│   ├── data_sources.yaml
│   ├── features/
│   │   ├── common.yaml
│   │   ├── heat_risk.yaml
│   │   ├── rain_risk.yaml
│   │   └── air_quality.yaml
│   └── models/
│       ├── heat_risk.yaml
│       ├── rain_risk.yaml
│       └── air_quality.yaml
│
├── data/
│   ├── raw/                         # gitignored
│   ├── interim/                     # gitignored
│   ├── processed/                   # gitignored or DVC-managed
│   ├── external/                    # only permitted small external files
│   └── manifests/
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_heat_risk_eda.ipynb
│   ├── 03_rain_risk_eda.ipynb
│   └── 04_air_quality_eda.ipynb
│
├── src/
│   └── climacore/
│       ├── __init__.py
│       ├── config/
│       │   ├── loader.py
│       │   └── schemas.py
│       ├── common/
│       │   ├── contracts.py
│       │   ├── exceptions.py
│       │   ├── logging.py
│       │   └── provenance.py
│       ├── data/
│       │   ├── sources/
│       │   │   ├── base.py
│       │   │   ├── india_weather_air_quality.py
│       │   │   ├── open_meteo_previous_runs.py
│       │   │   ├── open_meteo_forecast.py
│       │   │   ├── imd_rainfall.py
│       │   │   ├── cpcb_aqi.py
│       │   │   └── era5_land.py
│       │   ├── download.py
│       │   ├── validation.py
│       │   ├── schemas.py
│       │   ├── normalisation.py
│       │   ├── joins.py
│       │   └── storage.py
│       ├── features/
│       │   ├── common.py
│       │   ├── temporal.py
│       │   ├── lags.py
│       │   ├── rolling.py
│       │   ├── weather.py
│       │   ├── air_quality.py
│       │   └── spatial.py
│       ├── models/
│       │   ├── base.py
│       │   ├── baselines.py
│       │   ├── heat_risk/
│       │   │   ├── dataset.py
│       │   │   ├── features.py
│       │   │   ├── train.py
│       │   │   ├── predict.py
│       │   │   └── risk_rules.py
│       │   ├── rain_risk/
│       │   │   ├── dataset.py
│       │   │   ├── features.py
│       │   │   ├── train.py
│       │   │   └── predict.py
│       │   ├── air_quality/
│       │   │   ├── dataset.py
│       │   │   ├── features.py
│       │   │   ├── train.py
│       │   │   └── predict.py
│       │   ├── storm_risk/
│       │   │   └── official_alerts.py
│       │   ├── water_risk/           # future placeholder only if needed
│       │   ├── carbon_watch/         # future placeholder only if needed
│       │   └── compound_risk/
│       │       ├── scoring.py
│       │       └── config.py
│       ├── evaluation/
│       │   ├── splits.py
│       │   ├── metrics.py
│       │   ├── calibration.py
│       │   ├── compare_baselines.py
│       │   └── reports.py
│       ├── artifacts/
│       │   ├── registry.py
│       │   └── metadata.py
│       └── cli/
│           └── main.py
│
├── artifacts/                       # local model outputs; usually gitignored
│   ├── models/
│   ├── model_cards/
│   └── metadata/
│
├── reports/
│   ├── data_audit/
│   ├── eda/
│   ├── experiments/
│   └── evaluation/
│
└── tests/
    ├── unit/
    ├── data/
    ├── features/
    ├── models/
    └── integration/
```

Avoid creating empty placeholder modules for every future idea unless they are needed. The structure is a modular target, not a requirement to generate hundreds of empty files at once.

## 10.1 Module responsibilities

- `data/sources/`: one adapter per provider; provider-specific parsing stays here.
- `data/validation.py`: schema, ranges, duplicates, missingness, and source quality.
- `data/normalisation.py`: units, timezone, and naming normalisation.
- `features/`: reusable feature transformations with explicit time-safety.
- `models/<hazard>/`: model-specific dataset construction, training, and inference.
- `evaluation/`: shared chronological splits and metrics.
- `common/contracts.py`: common data and prediction structures.
- `artifacts/`: model loading, version metadata, and checksums.
- `cli/`: command-line entry points; notebooks call these modules rather than duplicating production logic.

---

# 11. Common Data and Prediction Contracts

Exact schemas may evolve, but all modules must follow common concepts.

## 11.1 Canonical forecast-input record

Expected logical fields (not every provider supplies all fields):

```text
source_id
provider
model_id
model_version                  # if supplied by provider
location_id
location_type                  # station | grid_cell | city_centre | district
latitude
longitude
issue_time_utc                 # when forecast was issued, if available
valid_time_utc                 # time the forecast applies to
lead_hours
lead_days
variable
value
unit
spatial_resolution
retrieved_at_utc
source_quality_flag
```

Do not make missing provider metadata up. Allow nulls with a reason.

## 11.2 Canonical observation/analysis record

```text
source_id
provider
product_type                   # station_observation | gridded_analysis | reanalysis | satellite_estimate | model_derived
location_id
location_type
latitude
longitude
observation_time_utc
local_date                     # when daily/local aggregation applies
variable
value
unit
spatial_resolution
retrieved_at_utc
quality_flag
```

## 11.3 Canonical prediction record & Unified Persona Contract

Every prediction emitted by the `LiveForecastEngine` adheres to a single unified contract consumed by all 13 persona adapters and future conversational LLM/RAG interfaces:

```json
{
  "prediction_id": "pred_nagpur_20261118_d1",
  "city": "Nagpur",
  "state": "Maharashtra",
  "coordinates": {"latitude": 21.1458, "longitude": 79.0882},
  "issued_at_utc": "2026-11-18T06:00:00Z",
  "valid_date": "2026-11-19",
  "forecast_horizon_days": 1,
  "layer_1_predictive": {
    "temp_max_corrected_c": 38.4,
    "temp_min_c": 22.1,
    "apparent_temp_max_c": 41.2,
    "precipitation_mm_q50": 18.2,
    "precipitation_mm_q90": 42.5,
    "flood_disaster_probability": 0.38,
    "pm2_5_calibrated_ug_m3": 112.4,
    "atmospheric_stagnation_flag": 0,
    "peak_grid_load_mw": 5420.0,
    "grid_unmet_supply_stress_prob": 0.18
  },
  "layer_2_physical": {
    "wet_bulb_temp_c": 27.2,
    "cooling_degree_days_cdd": 12.2,
    "tropical_night_active": false,
    "daily_crop_water_balance_mm": -2.8,
    "extreme_value_gev_return_period": "1-in-5 Year Event (GEV CI: [3.8, 6.2])"
  },
  "layer_3_exposure": {
    "worldpop_population_15km": 2840000,
    "dense_cluster_residents": 312000,
    "hospitals_15km": 42,
    "schools_15km": 118,
    "bridges_15km": 64,
    "drainage_canals_15km": 38,
    "active_crop_stage": "Rabi Wheat: Sowing Window",
    "regional_thermal_capacity_100km_mw": 29092.0,
    "closest_thermal_plant": "KORADI TPP (2,400 MW, 12.4km N)"
  },
  "layer_4_decision": {
    "compound_risk_score": 64.2,
    "compound_risk_category": "High Alert",
    "evaluated_personas": [
      "city_official", "farmer", "industry", "healthcare", 
      "water", "gig_worker", "insurance", "commuter", 
      "student", "cold_chain", "maritime", "aviation", "tourism"
    ]
  },
  "uncertainty_and_limitations": {
    "pm25_source": "CPCB CAAQMS transfer calibrator",
    "aviation_advisory": "Optical particulate haze advisory only (not hardware RVR)",
    "grid_status": "Unmet supply stress likelihood based on POSOCO 2013-2020 history"
  }
}
```

## 11.4 Standard result provenance

Every output ultimately shown to a user must make it possible to recover:

- What source(s) were used.
- When the input data were retrieved.
- What time period the result applies to.
- Which model and model version created it.
- What geographic resolution it represents.
- Whether it is a raw external forecast, ML-corrected forecast, model-derived estimate, rule-based risk score, or official warning.
- What limitations apply.

---

# 12. CLI and Training Workflow

Implement a small CLI so experiments are reproducible and do not depend on running notebook cells in a particular order.

Suggested commands (Antigravity may refine syntax but must preserve these capabilities):

```bash
# Inspect available sources and required configuration
python -m climacore.cli.main sources

# Download or ingest a selected source into data/raw/
python -m climacore.cli.main ingest --source india_weather_air_quality_50_cities

# Audit schema, quality, coverage and source manifest
python -m climacore.cli.main audit --source india_weather_air_quality_50_cities

# Build a leakage-safe training table
python -m climacore.cli.main prepare --model heat_risk

# Train a configured model
python -m climacore.cli.main train --model heat_risk

# Evaluate against chronological holdout and baselines
python -m climacore.cli.main evaluate --model heat_risk

# Export a versioned artifact and model card
python -m climacore.cli.main export --model heat_risk
```

If commands are not yet implemented, do not pretend they work. Implement them incrementally and document actual usage in `README.md`.

## 12.1 Required model-training pipeline stages

1. Load config and validate it.
2. Verify source files/manifests.
3. Validate schema and source coverage.
4. Build a time-aligned dataset.
5. Perform temporal leakage checks.
6. Split chronologically.
7. Fit preprocessing only on training data.
8. Train simple baseline.
9. Train LightGBM model.
10. Tune only using validation data.
11. Evaluate once on untouched test period.
12. Compare to raw forecast and simple baselines.
13. Save model, ordered feature list, configs, metrics, and model card.
14. Reload saved model and reproduce sample predictions.

## 12.2 Reproducibility

Record:

- Git commit or code revision where available.
- Dataset source IDs, versions, checksums, and date ranges.
- Feature schema/version.
- Train/validation/test cutoffs.
- Model hyperparameters.
- Random seeds.
- Python and library versions.
- Training timestamp.
- Metrics and baseline comparison.
- Artifact checksum.

---

# 13. Model Artifact and Model Card Requirements

Every exported model should have a folder similar to:

```text
artifacts/models/heat_risk/v0.1.0/
  model.txt                      # LightGBM native model, if selected
  feature_schema.json
  preprocessing.json             # only if preprocessing is required
  model_config.yaml
  metrics.json
  model_card.md
  artifact_manifest.json
```

Do not store a fitted preprocessing object separately if it is not needed. If one is needed, version and load it together with the model.

Each `model_card.md` must contain:

- Model name, module, version, status.
- Intended task and intended users.
- Training data sources, license notes, and coverage.
- Target definition and unit.
- Prediction horizon.
- Exact feature list and feature availability assumptions.
- Model algorithm and configuration.
- Chronological split dates.
- Baseline metrics and ML metrics.
- Performance by lead time/location/season where possible.
- Calibration/uncertainty method, or explicit statement that none is available.
- Known limitations, likely failure modes, and unsupported use cases.
- Geographic and temporal resolution.
- Model artifact checksum and reproducibility details.
- Date of last evaluation.

Model status should be one of:

- `research` — exploratory, not validated.
- `prototype` — passes basic tests, but not operationally validated.
- `validated_prototype` — passes defined holdout tests and baseline comparison for a limited scope.
- `production_candidate` — requires additional operational, monitoring, and governance review.

Never label a model `production` solely because it can be loaded by an API.

---

# 14. Future Backend and Deployment Plan — Not Active Yet

## 14.1 Frontend

Planned stack:

- Next.js.
- React.
- TypeScript.
- Vercel deployment.

No frontend code should be created in the current ML-only milestone.

## 14.2 Backend

Planned stack:

- Python.
- FastAPI.
- Pydantic request/response validation.
- Vercel Python runtime for lightweight request-driven inference, subject to dependency and latency tests.

Vercel announced Python as a first-class runtime on 15 October 2026, with automatic framework support including FastAPI. Current docs show Python 3.12 as the default, with 3.13/3.14 also listed. Use Python 3.12 for local/runtime parity unless a verified compatibility reason requires otherwise.

## 14.3 Current Vercel constraints to re-check at deployment

The Vercel documentation reviewed on 15 November 2026 lists these Hobby-function limits:

- 2 GB memory and 1 vCPU.
- Python function bundle limit of 500 MB uncompressed under the standard bundle limit.
- Hobby function maximum duration listed as 300 seconds with Fluid Compute.

These are platform limits, not an estimate of our model's actual consumption. An earlier rough planning range for a small tree-model inference function was approximately 210–650 MB including runtime, libraries, model, and request data; this is **not a benchmark or guarantee**. Measure cold-start time, peak resident memory, bundle size, and prediction latency with the real artifact before deployment. Limits and plan terms can change, so recheck official Vercel documentation immediately before deploying.

**Architecture decision:** Vercel functions will load an already-trained model and perform inference. They will not download and process giant historical datasets or train the model during a request.

## 14.4 Inference optimisation

- Use small, versioned model artifacts.
- Split training-only and API/runtime dependencies.
- Do not bundle notebooks, raw datasets, training reports, or large GIS libraries into the Python function.
- Use batch inference where practical.
- Cache external weather/air-quality responses and derived predictions.
- Avoid one provider request per map cell.
- Precompute heavy spatial aggregations outside the request path.
- Return concise prediction records, not large raw forecast grids.
- Add timeouts, retries with backoff, provider error handling, and stale-data indicators.
- Benchmark actual memory and CPU; do not rely on estimated RAM figures as measured facts.

## 14.5 Supabase

Planned use:

- Supabase Postgres: locations, source metadata, model registry metadata, cached forecast snapshots, prediction records, warning metadata, and audit information.
- Supabase Storage: selected model artifacts or processed assets only if the storage limits, access controls, and latency are acceptable.
- PostGIS extension: consider later for spatial queries if needed and available on the selected plan.

Security rules:

- Keep service-role credentials server-side only.
- Never put service-role keys in `NEXT_PUBLIC_*` variables or frontend code.
- Use least-privilege policies and Row Level Security where relevant.
- Do not store raw data without checking source license/retention rules.
- Do not treat Supabase as an unlimited storage or compute engine; recheck plan quotas.

## 14.6 Future API shape

Possible API contracts, to be finalised after model interfaces are stable:

```text
GET /api/health
GET /api/v1/risks/heat
GET /api/v1/risks/rain
GET /api/v1/risks/air-quality
GET /api/v1/risks/compound
GET /api/v1/alerts
GET /api/v1/locations
```

These are proposed route names, not existing endpoints. Do not claim they exist until implemented and tested.

---

# 15. Future Interactive Map Architecture — Not Active Yet

## 15.1 Map technology

- Map renderer: **MapLibre GL JS**.
- Prototype basemap: **OpenFreeMap** public vector-tile/style service.
- Suggested starter style from OpenFreeMap Quick Start: `https://tiles.openfreemap.org/styles/liberty`.
- MapLibre is the renderer; OpenFreeMap supplies basemap tiles/styles. They are separate components.

OpenFreeMap states that its public instance is free to use without registration/API key and that its map data comes from OpenStreetMap/OpenMapTiles. Its public service is donation-supported, so do not treat uptime or service policy as a contractual production guarantee. Recheck its terms and attribution before launch. Include visible map attribution.

## 15.2 Map layers — intended modular design

Each layer is independently switchable and must be added only when the corresponding source/model output is available:

- HeatRisk: heat/temperature risk polygons or points.
- RainRisk: rainfall amount/probability or risk surface.
- StormRisk: official warning polygons/markers with agency and validity time.
- AirQuality: CPCB station points and/or model-grid estimates, with the distinction visible.
- WaterRisk: soil-moisture/drought layers when available.
- CarbonWatch: emissions estimates only after an inventory method exists.
- CompoundRisk: combined score, with constituent hazard scores discoverable.

## 15.3 Geospatial rendering rules

- Use GeoJSON sources for manageable point/polygon data.
- Use raster/vector tiles or pre-generated spatial assets for large gridded layers rather than sending huge raw arrays to the browser.
- Never convert city-centre points into district-wide certainty without a valid aggregation/interpolation method.
- The map should communicate the native source resolution and aggregation method.
- District boundaries are only a geographic reference. They do not increase the resolution of the underlying weather data.
- For every layer, show its source, issue/valid time, update time, data type, spatial resolution, model version, and uncertainty/limitations.
- Use a stable layer registry/configuration so adding a new hazard does not require rewriting the map core.

## 15.4 Future map feature properties

A geographic feature should expose fields equivalent to:

```text
id
location_id
location_name
location_type
risk_value
risk_category
measured_or_forecast_value
unit
source_id
product_type
issue_time
valid_time
retrieved_at
spatial_resolution
model_name
model_version
uncertainty
methodology_version
limitations
```

Fields that are unknown remain null/omitted with a clear state; they must not be fabricated.

---

# 16. Logging, Testing, and Code Quality

## 16.1 Logging

- Use Python's standard `logging` or a minimal structured logging solution.
- Log source IDs, row counts, date coverage, elapsed time, and clear error context.
- Never log secrets or full sensitive environment variables.
- Do not log an API response as valid if parsing or validation failed.

## 16.2 Tests required

- Unit tests for each source parser using small fixtures.
- Tests for timezone conversion and local-day aggregation.
- Tests for date alignment between forecast and verification values.
- Tests that lags and rolling features never use future rows.
- Tests for duplicate and missing-key handling.
- Tests for unit conversions.
- Tests for model artifact save/load prediction parity.
- Tests that predictions include the required provenance metadata.
- Integration tests against saved fixtures; live API tests should be optional and not required for every test run.

## 16.3 Code quality

- Type hints for public functions.
- Docstrings for data-source adapters and model interfaces.
- Small functions with one responsibility.
- No duplicated feature logic between notebook and production code.
- No one-file monolith for all sources/models.
- No swallowed exceptions or silent fallback to invented values.
- Explicit errors when required source fields are missing.

---

# 17. Security, Reliability, and Responsible Use

- Use `.env` locally and Vercel/Supabase environment settings later. Commit only `.env.example` with variable names and placeholder text.
- Keep credentials out of notebooks and logs.
- Validate and limit external inputs, location requests, and batch sizes.
- Respect API terms and rate limits.
- Cache provider responses with a defined freshness policy.
- Include stale-data status when upstream sources fail.
- Do not make autonomous emergency decisions. Clima-Core supports authorised staff; it does not replace them.
- Avoid presenting an ML estimate as an official warning.
- Clearly state that a prototype is not an emergency service and users should follow official agency guidance.
- Where results affect public safety, preserve traceability and conservative wording.

---

# 18. First Work Session — Exact Implementation Order

Antigravity should execute the following sequence and stop at the phase gates.

### 24-hour feasibility checkpoint

Within the first working day, establish whether the data path is real:

- Download/inspect the manageable starter dataset.
- Obtain a small sample of historical forecast runs for Indian locations.
- Confirm that at least one forecast variable/lead time can be paired with a later verifying target using the same valid time and a defensible spatial mapping.
- Build a small training table and compute the unmodified forecast baseline metric.

If a usable forecast/verification pair cannot be obtained or aligned, do not pretend the correction model is viable. Switch to the AirQuality time-series prototype or narrow the first task, and record why. The main project risk is data alignment and validation, not whether the laptop or Vercel has enough compute.

### Suggested one-week training-first schedule

This is a prioritisation guide, not a promise that all modules will be complete in one week:

| Timebox | Priority |
|---|---|
| Day 1 | Data audit, source/license checks, Previous Runs availability, baseline feasibility checkpoint |
| Day 2 | Clean pipeline, canonical schemas, leakage-safe features, chronological splits |
| Day 3 | Train first LightGBM model and baseline comparison |
| Day 4 | Validation, error analysis by lead time/location/season, fix feature/data issues |
| Day 5 | Start the next highest-value hazard model only if first model is sound; otherwise improve first model |
| Day 6 | Model artifact export, model card, tests, reproducibility, optional second prototype |
| Day 7 | Freeze ML results, document limitations, prepare hackathon technical explanation; do not start UI before the ML milestone is accepted |

Do not spread the week across every planned module. One credible, evaluated model is more valuable than several unvalidated model names.

### Step 1 — Inspect repository

- Identify existing framework, files, Python environment, Git state, and current code.
- Do not delete or overwrite existing work.
- Report the current structure and any conflicts with this spec.

### Step 2 — Create the ML-only foundation

- Establish the Python package and configuration system.
- Add `.gitignore` for data, model binaries, environments, cache, and secrets.
- Add dependency lock strategy.
- Add logging, shared exceptions, and common contracts.
- Do not create frontend files or map components.

### Step 3 — Audit the 50-city starter dataset

- Retrieve or ask the user to place the source files locally if direct download is unavailable.
- Inspect actual CSV columns and row counts.
- Verify actual start/end dates and unique city count.
- Verify city coordinate metadata.
- Profile missingness, duplicates, units, and time coverage.
- Write a data-audit report and a populated manifest.
- Confirm the license and attribution note from the dataset page.

### Step 4 — Audit historical forecast availability

- Query small sample locations in India from Open-Meteo Previous Runs.
- Check available weather models, forecast lead times, variables, and dates.
- Save sample raw responses and a coverage report.
- Confirm exact matching capability with historical targets.
- Check live API terms and the rate/attribution requirements.

### Step 5 — Choose the first training target based on the audit

Default priority is:

1. HeatRisk forecast correction if the historical forecast/verification pair is available and aligned.
2. RainRisk if rainfall forecast and verification coverage is more complete.
3. AirQuality next-day PM2.5 model as a city-level prototype if the above historical forecast pairs are not sufficiently available.

This order is a decision rule, not a claim that the source data will necessarily support it. Record why the selected task is data-ready. Do not pretend the preferred model can be trained if its target/features do not exist.

### Step 6 — Build baselines before ML

- For HeatRisk, compute raw forecast MAE/RMSE and bias against the declared verification source.
- For RainRisk, compute raw forecast amount error and event skill at the documented threshold.
- For AirQuality, compute persistence and seasonal baseline error.

### Step 7 — Train first LightGBM model

- Use chronological splits.
- Use time-safe, group-aware features.
- Train a modest model first; no broad hyperparameter search.
- Compare to baseline on the untouched test set.
- Save the artifact, metrics, and model card.

### Step 8 — Validate reproducibility

- Reload saved artifact.
- Generate predictions for a fixed sample.
- Confirm ordered features and output values match the training-time export.
- Run the tests.

### Step 9 — Stop and report

At the end of the first milestone, provide:

- Repository structure created/modified.
- Sources actually downloaded and their licenses/attribution status.
- Data audit findings.
- Final training sample size and time range.
- Model selected and why.
- Baseline vs ML results on test data.
- Model artifact path and checksum.
- Model card path.
- Known limitations and next recommended task.

**Do not begin UI development at the end of this milestone unless the user explicitly changes the phase priority.**

---

# 19. Project Success Criteria

The first milestone is successful when Clima-Core has at least one reproducible, leakage-checked, chronologically evaluated ML model whose output is honestly scoped and whose performance is compared to a real baseline.

The first milestone is **not** successful merely because:

- A notebook runs once.
- A model object is saved without evaluation.
- Training accuracy is high.
- A random train/test split produces a strong score.
- A dashboard displays colourful risk categories.
- The project uses many libraries or algorithms.
- An AI assistant generated a large amount of code.

The wider project is successful when it can demonstrate how environmental source data become traceable, validated, useful decision-support outputs for a clearly identified audience.

---

# 20. Reference Links

These links are the primary documentation references reviewed for this initial specification. Recheck them before implementation because APIs, terms, versions, and plan limits can change.

## Data and APIs

- India Daily Weather & Air Quality dataset (Hugging Face): https://huggingface.co/datasets/AkshayKarthick/india-weather-air-quality
- Kaggle mirror: https://www.kaggle.com/datasets/akshaykarthickms007/india-daily-weather-and-air-quality
- Open-Meteo Previous Runs API: https://open-meteo.com/en/docs/previous-runs-api
- Open-Meteo Historical Forecast API: https://open-meteo.com/en/docs/historical-forecast-api
- Open-Meteo Historical Weather API: https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Forecast API: https://open-meteo.com/en/docs
- Open-Meteo Air Quality API: https://open-meteo.com/en/docs/air-quality-api
- Open-Meteo terms: https://open-meteo.com/en/terms
- IMD daily gridded rainfall: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html
- ERA5-Land: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- CHIRPS: https://www.chc.ucsb.edu/data
- CHIRPS Earth Engine catalogue: https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY
- NASA GPM IMERG: https://gpm.nasa.gov/data/imerg
- CPCB real-time AQI OGD catalogue: https://www.data.gov.in/catalog/real-time-air-quality-index
- NOAA GSOD metadata: https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc%3AC00516/html
- NOAA station data: https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database
- INDmet Zenodo record: https://zenodo.org/records/15430548
- Kaggle Rainfall in India: https://www.kaggle.com/datasets/rajanand/rainfall-in-india
- India's Top 50 Most Populous Cities Weather Data: https://www.kaggle.com/discussions/accomplishments/543282
- Daily Delhi Climate: https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data
- World Weather Repository: https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository
- Weather Forecasting - Datavidia competition: https://www.kaggle.com/competitions/weather-forecasting-datavidia/data
- IMD daily maximum temperature: https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html
- IMD daily minimum temperature: https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html
- CHIRPS-GEFS / CHIRPS3-GEFS: https://chc.ucsb.edu/data/chirps-gefs
- IndiaWeatherBench: https://huggingface.co/datasets/tungnd/IndiaWeatherBench
- NWIC district boundary data: https://nwdp.nwic.gov.in/dataset/district-boundary

## Platform and mapping documentation

- Vercel Python runtime: https://vercel.com/docs/functions/runtimes/python
- Vercel function limits: https://vercel.com/docs/functions/limitations
- Vercel FastAPI guide: https://vercel.com/kb/guide/ship-a-fastapi-app-on-vercel
- MapLibre GL JS: https://maplibre.org/maplibre-gl-js/docs/
- MapLibre GeoJSON source: https://maplibre.org/maplibre-gl-js/docs/API/classes/GeoJSONSource/
- OpenFreeMap: https://openfreemap.org/
- OpenFreeMap Quick Start: https://openfreemap.org/quick_start/

## ML documentation

- LightGBM Python introduction: https://lightgbm.readthedocs.io/en/stable/Python-Intro.html
- XGBoost Python API: https://xgboost.readthedocs.io/en/stable/python/python_api.html
- Scikit-learn: https://scikit-learn.org/stable/

---

# 21. Change Log

| Version | Date | Change |
|---|---|---|
| **1.0.0** | 18 Nov 2026 | **Major Constitutional & Architectural Reset.** Ingests 14 verified multi-decadal datasets (WorldPop 2024, OSM infrastructure, IIT Delhi IFI v3 floods, AgriSense crop soils, SAGE crop phenology, CHETNA thermal plants, POSOCO/CEA electricity telemetry, IMD 74y temperature & 46y rainfall, 45y cyclones). Formally establishes the **4-Layer Architecture** (Predictive, Physical, Exposure, Decision). Implements the 13 Persona Adapters. Replaces daily percentile shortcuts with formal Extreme Value Theory (EVT) GEV distributions; formalizes the Multi-Hazard Compound Index as an empirical joint exceedance framework; adds elevation-aware psychrometrics and decouples pure hazard probability from human exposure impact. Verified by 87/87 unit tests passing. |
| 0.1.0 | 15 Nov 2026 | Initial single-source-of-truth specification. Establishes hybrid intelligence, modular hazard architecture, ML-only first phase, source registry, data/ML validation rules, future deployment and map direction. |

## Change process

For every material change, add a row above with:

- Version and date.
- What changed.
- Why it changed.
- Which modules, datasets, algorithms, APIs, or deployment assumptions are affected.
- Any migration or backward-compatibility notes.

**Do not silently rewrite the product direction, replace the chosen primary algorithm, change data sources, or start the UI phase. Update this document first.**
