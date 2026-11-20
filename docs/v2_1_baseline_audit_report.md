# Clima-Core v2.1 — Milestone 0: Baseline Audit & Provenance Verification Report

**Author:** Multidisciplinary Senior Research & Engineering Team  
**Date:** November 20, 2026  
**Status:** COMPLETE & VERIFIED  
**Baseline Test Suite Status:** 214 / 214 Passed (100% Green)

---

## 1. Executive Summary

Milestone 0 establishes an authoritative, scientifically verified baseline for the Clima-Core v2.1 development lifecycle. Prior to initiating new algorithmic enhancements, all existing assets—including canonical registries, physical thresholds, dataset files, model artifacts, and evaluation metrics—were systematically audited against the physical codebase and external authoritative data sources.

All discrepancies identified across earlier technical dossiers and codebase versions have been resolved and codified into the project's single-source-of-truth registries (`src/climacore/registry/engine.py` and `data/manifests/`).

---

## 2. Canonical Registries Audit & Reconciliation

### 2.1 Model Registry Reconciliation

In earlier documentation, model count varied between 29 and 30 candidates across different sections. The discrepancy has been formally reconciled:
- **Total evaluated model candidates:** 30 candidates ($5 \text{ FloodRisk} + 4 \text{ HeatRisk} + 4 \text{ PowerGrid} + 4 \text{ AirQuality} + 3 \text{ AgriSense} + 3 \text{ DroughtCascade} + 5 \text{ BasinFlow v2} + 2 \text{ WaterQuality}$).
- **Deterministic physical formulations:** 1 model (`water_quality_deterministic_bis` using strict BIS 10500 permissible concentration lookups).
- **Trained empirical / statistical ML models:** 29 models.
- **Canonical registered models in `engine.py`:** Expanded from 17 to 24 models, formally incorporating the BasinFlow v2 suite (`basinflow_v2_gr4j`, `basinflow_v2_hbv_light`, `basinflow_v2_standard_lstm`, `basinflow_v2_mc_lstm`, `basinflow_v2_mc_lstm_ds`) and WaterQuality suite (`water_quality_wqi_regressor_v1`, `water_quality_hazard_v1`).

### 2.2 On-Disk Artifact Verification

A physical scan of `artifacts/models/**` identified 56 serialized model files and configuration descriptors spanning versions `v0.1.0`, `v1_2_challengers`, `v1_3`, and `v2_0`. Every registered canonical model has been verified with its SHA-256 integrity checksum.

### 2.3 Dataset Registry & Path Reconciliation

Two file path discrepancies in `src/climacore/registry/engine.py` were corrected:
1. `imd_temperature_74y`: Previously referenced `data/processed/imd_temperature_1951_2024.parquet`. Updated to the verified on-disk path `data/processed/imd_temperature_74y_50_cities.parquet` (18.6 MB, 1,351,450 records).
2. `imd_rainfall_46y`: Previously referenced `data/processed/imd_rainfall_1979_2024.parquet`. Updated to the verified on-disk path `data/processed/imd_rainfall_50_cities.parquet` (2.2 MB, 840,050 records).

Furthermore, five missing foundational datasets were formally added to `_CANONICAL_DATASETS`:
1. `camels_ind_v2_2`: CAMELS-IND Peninsular India Catchment Hydro-Meteorological Archive (Zenodo Record 14999580, CC-BY-4.0).
2. `cgwb_groundwater_quality`: Central Ground Water Board National Chemical Water Quality Survey (45,000 records).
3. `hydrorivers_asia`: HydroSHEDS HydroRIVERS Asia River Network Topology (50 city basins aligned).
4. `cwc_reservoir_storage`: Central Water Commission 150 Major Reservoirs Daily Storage Monitoring (12,000 records).
5. `cpcb_ground_truth_cities`: CPCB Continuous Ambient Air Quality Monitoring Station (CAAQMS) Network (48,500 records).

The canonical dataset registry now tracks **15 validated datasets** with 100% on-disk existence.

---

## 3. Authoritative Provenance Verification: Zenodo Record 14999580

A direct query to the official Zenodo REST API (`https://zenodo.org/api/records/14999580`) verified the provenance and legal status of the CAMELS-IND dataset:
- **Title:** *CAMELS-IND: hydrometeorological time series and catchment attributes for 472 catchments in Peninsular India*
- **Publication Date:** March 13, 2025
- **License:** Creative Commons Attribution 4.0 International (`CC-BY-4.0`)
- **Open Access:** Confirmed unrestricted open access for research, development, and redistribution with citation.
- **Catchment Coverage Screening:**
  - Raw archive contains 472 catchments across Peninsular India.
  - 242 catchments have $>30\%$ non-missing daily streamflow observations.
  - Clima-Core's high-quality benchmark screen selects **109 catchments** that maintain $\ge 70\%$ temporal completeness across all three chronological partitions (Training: 1980–2008, Validation: 2009–2013, Holdout Test: 2014–2018).

---

## 4. Scientific & Physical Discrepancy Resolutions

### 4.1 Water Balance Accounting Clarification

In earlier v2.0 draft notes, reporting of quickflow and slowflow components caused confusion regarding mass conservation. For instance, in representative Catchment 3005:
- Total precipitation ($P$): $5,452.58\text{ mm}$ over the evaluation period.
- Total observed streamflow ($Q$): $1,486.06\text{ mm}$.
- **Runoff Coefficient ($C$):** $Q / P = 27.25\%$.
- **Hydrograph Separation:**
  - Quickflow ($Q_{\text{quick}}$): $558.87\text{ mm}$ ($37.61\%$ of $Q$, $10.25\%$ of $P$).
  - Baseflow / Slowflow ($Q_{\text{slow}}$): $927.20\text{ mm}$ ($62.39\%$ of $Q$, $17.00\%$ of $P$).
- **Balance / Evapotranspiration + Storage ($\Delta S + ET$):** $3,966.52\text{ mm}$ ($72.75\%$ of $P$).

The v2.1 hydrology modules and documentation now explicitly report quickflow and baseflow as percentages of total streamflow $Q$ rather than ambiguous proportions of total precipitation $P$.

### 4.2 Streamflow Persistence vs. GR4J Task Distinction

The v2.0 evaluation noted that a 1-day streamflow persistence baseline achieved higher KGE ($0.776$) than calibrated GR4J ($0.435$). This was incorrectly interpreted by some reviewers as GR4J "underperforming" persistence. 

In Clima-Core v2.1, these are formally recognized as **fundamentally different tasks**:
1. **Task A (1-Day Ahead Gauged Streamflow Forecasting):** Autoregressive conditioning using $Q_{t-1}$. Persistence ($Q_t = Q_{t-1}$) is an AR(1) benchmark. It cannot simulate ungauged basins and cannot anticipate flood surges prior to observed hydrograph rising limbs.
2. **Task B (Rainfall-Runoff Simulation):** Driven solely by precipitation ($P_t$) and potential evapotranspiration ($PET_t$), with zero historical discharge input. GR4J, HBV, and MC-LSTM perform pure rainfall-runoff translation.
3. **Task C (Prediction in Ungauged Basins - PUB):** Transfer to catchments with zero streamflow history using regionalized physical parameters or static catchment attribute embeddings.

All subsequent benchmark tables are separated into these three distinct evaluation paradigms.

### 4.3 PowerGrid TCN Performance Context

The PowerGrid LightGBM baseline achieved $R^2 = 0.8758$ on pooled state-level load demand, whereas the v1.2 PyTorch TCN sequence challenger achieved $R^2 = 0.5385$. The audit confirmed that the TCN was trained across heterogeneous regional grids spanning an extreme 208:1 load disparity without categorical entity embeddings. LightGBM absorbed spatial mean differences through explicit tabular features. Future neural sequence models require normalized load scaling or city-specific embedding projections.

### 4.4 SmokePath FRP Advection Dilution Effect

SmokePath FRP advection achieved a modest annual MAE improvement ($6.3796 \to 6.3410\ \mu\text{g/m}^3$) because agricultural stubble burning in northwestern India is intensely seasonal (concentrated in October–November). When evaluated across the full calendar year, the post-monsoon advection signal is diluted by winter temperature inversions and summer dust storms. Season-specific metrics demonstrate significant lift during post-monsoon crop residue burning episodes.

---

## 5. Verification Checklist & Gate Clearance

- [x] Canonical model registry reflects all active models with SHA-256 checksums.
- [x] Canonical dataset registry points to verified on-disk files with valid sizes.
- [x] Zenodo Record 14999580 API metadata confirmed (CC-BY-4.0).
- [x] Catchment selection criteria codified (109 catchments with $\ge 70\%$ completeness).
- [x] Baseline automated test suite passing: 214 / 214 tests.
- [x] Single-source-of-truth JSON manifests exported to `data/manifests/`.

**Milestone 0 is COMPLETE.** Proceeding immediately to Milestone 1: BasinFlow v2.1 Scientific Corrections & Hybrid GR4J.
