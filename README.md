# Clima-Core — ML Foundation (Phase 1)

AI-powered Environmental Risk Intelligence and Decision Support Platform.

> **Status:** Milestone 1 complete. Core data ingestion, data audit, time-safe feature engineering, LightGBM training, chronological evaluation, artifact export, and model cards implemented.

---

## 1. Project Overview

Clima-Core combines meteorological reanalysis, CAMS atmospheric chemistry estimates, and machine-learning intelligence to provide short-range environmental risk foresight for decision-makers across Indian cities.

In accordance with `spec.md`, Phase 1 prioritizes sound scientific methodology:
- Strict **chronological splits** (no random train/test shuffling).
- Strict **time-safe feature engineering** (no future data leakage).
- Comparison against strong physical baselines (Persistence, 7-Day Rolling Mean, Seasonal Climatology).
- Native artifact export with SHA-256 checksums and model cards.

---

## 2. Environment Setup

The project uses Python 3.12 and `uv` for reproducible virtual environment and dependency management.

```bash
# Create virtual environment
uv venv --python 3.12 .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Install package and dependencies in editable mode
uv pip install -e ".[dev]"
```

---

## 3. CLI Usage

The project provides a unified CLI for all machine learning workflows:

```bash
# 1. Inspect registered data sources
python -m climacore.cli.main sources

# 2. Run data audit and generate checksum manifest
python -m climacore.cli.main audit --source india_weather_air_quality_50_cities

# 3. Build time-safe processed features
python -m climacore.cli.main prepare --model air_quality
python -m climacore.cli.main prepare --model heat_risk
python -m climacore.cli.main prepare --model rain_risk

# 4. Train LightGBM model and export artifacts
python -m climacore.cli.main train --model air_quality
python -m climacore.cli.main train --model heat_risk
python -m climacore.cli.main train --model rain_risk

# 5. Display chronological evaluation report
python -m climacore.cli.main evaluate --model air_quality
python -m climacore.cli.main evaluate --model heat_risk
python -m climacore.cli.main evaluate --model rain_risk

# 6. Run unit tests
pytest -v tests
```

---

## 4. Trained Models Summary

### A. AirQuality Model (`v0.1.0`)
- **Target:** Next-day 24h mean $\text{PM}_{2.5}$ concentration ($\mu\text{g/m}^3$)
- **Algorithm:** LightGBM Regressor (334 iterations, early stopping on validation set)
- **Chronological Split:**
  - **Train:** 2022-08-12 to 2025-05-31 (51,200 rows)
  - **Validation:** 2025-06-01 to 2025-12-31 (10,700 rows)
  - **Test (Holdout):** 2026-01-01 to 2026-10-29 (15,100 rows untouched)
- **Holdout Test Results:**
  - **MAE:** **6.38 $\mu\text{g/m}^3$** (8.2% improvement over persistence baseline of 6.95)
  - **RMSE:** **9.63 $\mu\text{g/m}^3$** (10.1% improvement over persistence baseline of 10.71)
  - **$R^2$:** **0.8725**
  - **CPCB AQI Category Accuracy:** **80.74%**
  - **Artifact:** `artifacts/models/air_quality/v0.1.0/model.txt` (SHA-256: `c1b53c69f26c8ecaa07994e5899dc7bda19a9ca8b3914b91e65c7328371bb2fb`)

### B. HeatRisk Model (`v0.1.0`)
- **Target:** Next-day 2m maximum temperature ($^\circ\text{C}$)
- **Algorithm:** LightGBM Regressor (830 iterations, early stopping on validation set)
- **Chronological Split:**
  - **Train:** 2020-01-08 to 2024-12-31 (91,000 rows)
  - **Validation:** 2025-01-01 to 2025-12-31 (18,250 rows)
  - **Test (Holdout):** 2026-01-01 to 2026-10-29 (15,100 rows untouched)
- **Holdout Test Results:**
  - **MAE:** **1.02 $^\circ\text{C}$** (vs 1.11 $^\circ\text{C}$ persistence, vs 1.95 $^\circ\text{C}$ climatology)
  - **RMSE:** **1.42 $^\circ\text{C}$** (vs 1.56 $^\circ\text{C}$ persistence, vs 2.54 $^\circ\text{C}$ climatology)
  - **$R^2$:** **0.9510**
  - **Artifact:** `artifacts/models/heat_risk/v0.1.0/model.txt` (SHA-256: `92843a9d316b1a3a6420cdfd46fca44629c22c753fdb180373b99478d94ada1a`)

### C. RainRisk Model (`v0.1.0` — Two-Part Hurdle Model)
- **Target:** Next-day precipitation occurrence ($\ge 1.0\text{ mm}$) and volume ($\text{mm}$)
- **Architecture:**
  - **Part 1:** LightGBM Classifier for Rain Occurrence ($P(\text{Rain} \ge 1.0\text{ mm})$)
  - **Part 2:** Conditional LightGBM Regressor for Volume ($E[\text{Rain} \mid \text{Rain} \ge 1.0\text{ mm}]$)
  - **Combined Expectation:** $\hat{y} = P(\text{Rain} \ge 1.0\text{ mm}) \times E[\text{Rain} \mid \text{Rain} \ge 1.0\text{ mm}]$
- **Chronological Split:**
  - **Train:** 2020-01-08 to 2024-12-31 (91,000 rows)
  - **Validation:** 2025-01-01 to 2025-12-31 (18,250 rows)
  - **Test (Holdout):** 2026-01-01 to 2026-10-29 (15,100 rows untouched)
- **Holdout Test Results:**
  - **Classifier PR-AUC:** **0.8840** | **ROC-AUC:** **0.9134** | **Brier Score:** **0.1157** (vs 0.2319 persistence)
  - **Two-Part MAE:** **3.18 mm** (outperformed Direct Single Regressor at 3.25 mm and Persistence at 3.38 mm)
  - **Two-Part RMSE:** **7.40 mm** (outperformed Persistence at 8.93 mm and Climatology at 8.44 mm)
  - **Artifacts:** `artifacts/models/rain_risk/v0.1.0/` (contains `classifier_model.txt`, `conditional_regressor_model.txt`, `direct_regressor_model.txt`)

---

## 5. Directory Structure

```text
clima-core/
├── spec.md                              # Single source of truth specification
├── README.md                            # Project documentation & execution guide
├── pyproject.toml                       # Python package configuration
├── configs/
│   ├── project.yaml                     # Project-level settings & conventions
│   └── data_sources.yaml                # Provenance, licenses, and data source registry
├── data/
│   ├── raw/                             # Downloaded raw CSV files (gitignored)
│   ├── interim/                         # Staging tables
│   ├── processed/                       # Train-ready Parquet datasets
│   └── manifests/                       # SHA-256 dataset provenance manifests
├── artifacts/
│   └── models/                          # Exported native models, metadata & model cards
├── reports/
│   ├── data_audit/                      # JSON & Markdown data quality reports
│   └── evaluation/                      # Holdout test metrics & comparisons
├── src/climacore/
│   ├── common/                          # Pydantic contracts, logging, provenance, exceptions
│   ├── config/                          # YAML loaders
│   ├── data/                            # Auditing, schema checks, API adapters
│   ├── features/                        # Time-safe lag, rolling, and temporal transforms
│   ├── models/                          # Baseline models, LightGBM pipelines
│   ├── evaluation/                      # Chronological splits, regression & AQI metrics
│   └── cli/                             # Main CLI entrypoint
└── tests/                               # Comprehensive unit tests
```
