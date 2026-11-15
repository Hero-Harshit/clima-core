# Clima-Core — Project Specification and Single Source of Truth

> **Document:** `spec.md`  
> **Project:** Clima-Core  
> **Specification version:** 0.1.0  
> **Status:** Initial technical baseline; data audit and model validation are still required  
> **Last reviewed:** 15 November 2026  
> **Primary purpose:** Give Antigravity a reliable, explicit project contract.  
> **Change policy:** This specification is designed to evolve. Do not silently change its decisions; propose and document changes in the Change Log.

---

## 0. Instructions for Antigravity — Read Before Doing Any Work

This file is the current **single source of truth** for Clima-Core. Follow it before making architectural, data, model, deployment, or product decisions.

### 0.1 First milestone: model training only

**The immediate task is to build the machine-learning data and training pipeline. Do not build the UI now.**

For the first milestone, work only on:

1. Repository inspection and a clean, modular Python project structure.
2. Dataset acquisition or documented local-data instructions.
3. Dataset inspection, source/column verification, data quality checks, and exploratory analysis.
4. Time-safe feature engineering and baseline creation.
5. Training the first selected model or models.
6. Chronological evaluation against appropriate baselines.
7. Model artifact export, model cards, metrics, and reproducibility metadata.

Do **not** build Next.js pages, React components, dashboards, maps, charts, login screens, or frontend styling during this milestone. Do not spend time on visual design. Do not deploy an unvalidated model merely to demonstrate deployment.

Future deployment and UI architecture are described later in this document so that today's ML code has clean boundaries. They are not permission to start those phases now.

### 0.2 Non-negotiable anti-hallucination rules

Antigravity must obey all of the following:

- Never invent a dataset, file, column, variable, endpoint, API response, model result, metric, benchmark, license, attribution requirement, or deployment capability.
- Before using a dataset, inspect its actual files, schema, date range, units, missingness, geographic coverage, source method, and license/terms. A dataset name or Kaggle description is not proof that every claimed field exists.
- If a source cannot be reached, report that clearly and continue only with an explicitly documented fallback. Do not fabricate a local copy or pretend a download succeeded.
- Keep source observations, reanalysis, satellite estimates, numerical weather forecasts, official warnings, and model predictions as distinct data types. Do not label one as another.
- Historical weather values alone do not prove that a model can forecast the future. A genuine forecast-correction model needs historical forecasts as they were issued, paired with the later verifying observation or analysis for the same valid time and location.
- Never use information that would not have been available at prediction time. Prevent target leakage, future-data leakage, and train/test contamination.
- Never claim that a machine-learning model improves on an official forecast unless a chronological holdout evaluation demonstrates that improvement.
- Never invent accuracy, confidence, probability, uncertainty, resolution, or coverage. If it has not been measured, label it as unknown or not yet evaluated.
- Never describe a city-centre model-grid value as a ward-, street-, household-, or building-level measurement.
- Do not claim that Clima-Core predicts cyclones, thunderstorms, floods, heatwaves, or other severe events independently unless the relevant model, source data, labels, and validation actually exist.
- Do not treat official warning systems as an ML model. Preserve the identity and wording of the official issuing agency.
- Never hard-code a policy threshold, hazard threshold, compound-risk weight, or vulnerability score without a documented source or an explicitly labelled prototype assumption.
- Do not silently substitute a dataset because a preferred dataset is difficult to obtain. Record the substitution in `data_sources.yaml`, this spec's Change Log, and the model card.
- Do not add heavy or unnecessary libraries just because they are popular. Explain the need before adding them.
- Do not rewrite or delete existing user files before inspecting the repository. Preserve existing work and adapt this specification to the repository's actual state where appropriate.
- Do not expose credentials, API keys, Supabase service-role keys, or private environment variables in source code, logs, notebooks, Git, or frontend bundles.

### 0.3 How to handle uncertainty or a specification conflict

If something is not defined here:

1. Inspect the actual project and source documentation.
2. Choose the smallest reversible implementation that preserves modularity.
3. Put uncertain values in configuration, not hard-coded logic.
4. Add a `TODO` or `NEEDS-VERIFICATION` note with the reason and the source that must be checked.
5. Explain the decision in the implementation summary.

If code requirements conflict with this file, do not silently choose. Explain the conflict and propose a spec change. Once the change is approved, update this file and the Change Log before implementing the new direction.

---

# 1. Product Definition

## 1.1 Name and positioning

**Name:** Clima-Core  
**Category:** AI-powered Environmental Risk Intelligence and Decision Support Platform  
**Suggested tagline:** *Environmental Intelligence. Smarter Decisions.*

Clima-Core is intended to combine meteorological information, environmental measurements, machine-learning estimates, official alerts, and geographic context to help people understand where environmental risks are emerging and which locations may need attention.

Clima-Core is an intelligence and decision-support layer. It is **not** a replacement for the India Meteorological Department (IMD), National Disaster Management Authority (NDMA), Central Pollution Control Board (CPCB), state agencies, local administrations, or emergency services.

## 1.2 Problem statement

Extreme heat, intense rainfall, air pollution, drought conditions, and severe weather can affect Indian cities and communities. Useful information exists across weather forecasts, environmental monitoring systems, official warnings, and historical datasets, but decision-makers still need to interpret information across multiple environmental domains and understand which places and populations may be exposed.

Clima-Core aims to bring those signals into one consistent intelligence workflow. It will estimate selected environmental risks, retain the source and timestamp of every input, represent geographic coverage honestly, and eventually help administrators prioritise attention.

## 1.3 Primary target audience

The primary users are institutional decision-makers and public-sector teams, including:

- State and District Disaster Management Authorities.
- District Collectors and district administrations.
- Municipal corporations and city administrations.
- State Pollution Control Boards and environmental departments.
- Urban planners and infrastructure-planning teams.
- Public-health departments and emergency-planning teams.

Secondary beneficiaries may include citizens, schools, hospitals, universities, researchers, and organisations that need to understand environmental conditions.

**Do not position the first release as a generic consumer weather app.** Its differentiator is the interpretation and combination of environmental signals for decision support.

## 1.4 What Clima-Core should help answer

Examples of intended questions:

- Which areas are forecast to experience unusually high temperatures?
- Where is heavy rainfall forecast, and what is the associated rainfall risk?
- Which locations have deteriorating particulate-matter conditions?
- Which official weather warnings currently apply to a location?
- Where do multiple environmental hazards overlap?
- Which exposed populations or critical facilities may need attention, if reliable vulnerability data are available?
- What data source, timestamp, forecast horizon, geographic resolution, and uncertainty support each displayed result?

## 1.5 Product principles

1. **Evidence first:** source, time, units, resolution, and methodology accompany every result.
2. **AI with a purpose:** use ML where it offers measurable value, not as decoration.
3. **Hybrid intelligence:** use authoritative forecasts and monitoring sources as inputs; use ML for correction, estimation, classification, and impact-oriented analysis.
4. **Explainability:** show the factors behind a risk result where possible.
5. **Geographic honesty:** never imply more local precision than the input data supports.
6. **Modularity:** each hazard is an independent module that follows shared contracts.
7. **Graceful degradation:** a missing data provider must not crash all other hazard modules.
8. **Reproducibility:** every trained model must be tied to its data version, configuration, code revision, and evaluation report.

---

# 2. Scope and Development Phases

## 2.1 Immediate scope — Phase 1: ML foundation and model training

This is the current active phase.

Deliverables:

- Python project scaffold.
- Data-source registry and source adapters.
- Download/ingestion utilities for selected sources.
- Dataset manifests, schema checks, data-quality reports, and EDA.
- A common training dataset contract.
- Leakage-safe feature engineering.
- Baseline and LightGBM experiments.
- Chronological train/validation/test evaluation.
- Saved model artifacts and preprocessing metadata.
- Model cards and experiment reports.
- Unit tests for data alignment, feature generation, and inference parity.

**No UI or public-facing API is part of this phase.**

## 2.2 Phase 2: model improvement and risk-engine foundation

After Phase 1 has reliable results:

- Improve and validate HeatRisk and RainRisk.
- Build AirQuality next-day estimation as the next candidate model.
- Add calibrated probabilities or prediction intervals only where evaluation supports them.
- Implement a transparent, configurable CompoundRisk engine after individual hazard outputs are available.
- Add official-alert ingestion only after a real, supportable source/API is verified.

## 2.3 Phase 3: inference service and data storage

Planned, not active now:

- Python FastAPI inference service.
- Model loading and batch prediction.
- Supabase Postgres for metadata, cached forecast inputs, and prediction records.
- Supabase Storage or another controlled artifact store for model files if appropriate.
- Input/output schemas, versioning, health checks, rate limits, and monitoring.

## 2.4 Phase 4: UI and map application

Planned, not active now:

- Next.js + React frontend.
- MapLibre GL JS interactive map.
- OpenFreeMap basemap for the prototype.
- Independent toggleable environmental risk layers.
- Source, timestamp, horizon, resolution, and uncertainty information in the interface.

## 2.5 Phase 5: vulnerability and impact intelligence

Later, subject to available datasets and licensing:

- Population exposure.
- Schools, hospitals, shelters, and other critical facilities.
- Urban density, green cover, and land-surface information.
- Low-lying terrain or flood susceptibility.
- District/ward-level vulnerability overlays where data resolution permits.

## 2.6 Non-goals for the first hackathon MVP

- Building a numerical weather prediction model from scratch.
- Training a deep neural atmospheric model on a laptop.
- Claiming independent cyclone or severe-storm prediction without radar/lightning training data and validation.
- Creating a public emergency-warning authority.
- Claiming street-level prediction from city-centre or coarse grid data.
- Building carbon-emissions inventories without a traceable methodology and activity data.
- Building a full GIS data platform or storing massive raw global datasets.
- Building frontend screens before the initial model-training milestone is complete.

---

# 3. System Architecture — High-Level Direction

## 3.1 Hybrid intelligence architecture

Clima-Core uses external forecasts and environmental data as inputs. Its own models add a carefully evaluated intelligence layer.

Expected future flow:

```text
Official / external data sources
  ├── Numerical weather forecasts
  ├── Historical forecast runs
  ├── Historical weather observations / reanalysis
  ├── Air-quality model data / monitoring stations
  ├── Official weather and disaster warnings
  └── Geographic and exposure datasets
             |
             v
      Source adapters
             |
             v
   Validation + normalisation
             |
             v
      Feature engineering
             |
             v
   Independent hazard models
      ├── HeatRisk
      ├── RainRisk
      ├── AirQuality
      └── (future) WaterRisk
             |
             v
   Transparent risk/impact engine
      ├── Official warning context
      ├── Hazard probabilities/severity
      └── Vulnerability/exposure (when supported)
             |
             v
  Versioned predictions + provenance
             |
             v
  Future FastAPI service / Supabase
             |
             v
  Future Next.js + MapLibre application
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

# 4. Data Sources and Dataset Registry

The source registry must live in configuration, for example `configs/data_sources.yaml`. Do not scatter source URLs, license assumptions, column names, or provider-specific logic across model code.

Every source entry must record:

- Stable source ID and display name.
- Official/source URL and download/API method.
- Date downloaded and retrieval timestamp.
- Data provider and upstream data provider, if different.
- License/terms and attribution requirements.
- Geographic coverage and spatial resolution.
- Temporal coverage and temporal resolution.
- Variables and units.
- Time zone and daily aggregation convention.
- Whether values are observations, reanalysis, satellite estimates, forecast values, or modelled estimates.
- File names, checksums, and schema version.
- Known limitations and verification state.

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

# 6. Machine-Learning Strategy

## 6.1 Primary model family

**Primary algorithm: LightGBM.**

Why:

- Strong fit for structured/tabular weather and environmental data.
- Handles non-linear relationships and feature interactions.
- CPU training and inference are practical for this project's scale.
- Supports regression and classification.
- Model files can be saved compactly and loaded for inference.

**Secondary benchmark: XGBoost.** Use it only as a comparison when the data and environment support it. It is not mandatory to train both for every module.

**Simple baselines:** persistence/last-value, seasonal climatology, raw official forecast, and a simple scikit-learn model such as Random Forest or HistGradientBoosting where useful.

**Not part of the initial MVP:** LSTM, ConvLSTM, CNN weather models, Transformers, or training a full NWP/weather foundation model. Do not add deep learning merely to make the project sound more advanced.

## 6.2 Model selection principles

- Start with the simplest baseline.
- Train LightGBM only after the dataset passes quality and leakage checks.
- Compare against the relevant source forecast, not just against a weak baseline.
- Use a time-based split, never a random row split as the only evaluation.
- Keep model parameters in configuration files.
- Use deterministic seeds where supported.
- Track package versions and selected feature columns.
- Use early stopping based on validation data.
- Test whether an apparent improvement persists across dates, cities/regions, and hazard seasons.

## 6.3 HeatRisk — priority hazard module

### Objective

Estimate future heat-related environmental conditions and derive a transparent heat-risk level from validated forecast information.

### Recommended first ML task

**Forecast bias correction for next-day or selected lead-time maximum temperature/apparent temperature.**

Inputs should include, where actually available at issue time:

- Official/model forecast temperature for the target valid time.
- Forecast apparent temperature, relative humidity/dew point, wind, cloud, pressure, and radiation variables where available.
- Recent observed/reanalysis conditions only if they were available before the forecast issue time.
- Calendar/season features.
- Location identity or geographic features appropriate to the training population.
- Forecast model/provider and lead-time metadata where relevant.

Recommended target:

- `temperature_error = verifying_temperature - raw_forecast_temperature`, or directly the verifying max temperature.
- Prefer residual correction if it improves comparison with the raw official forecast and is stable.

Primary algorithm: `LightGBMRegressor`. Benchmark against raw forecast and a simple bias-correction baseline. Use XGBoost only as a secondary model if useful.

### Risk derivation

Do not invent an Indian heatwave definition. Derive official heatwave categories only after locating and implementing the appropriate current IMD criteria, including station/climate-zone and minimum-temperature/departure requirements where applicable. Put thresholds and rules in versioned configuration, with source URL and effective date.

If official criteria are not implemented, output a measured temperature/apparent-temperature forecast and a clearly labelled **prototype heat-stress indicator**, not an official heatwave warning.

### Evaluation

- MAE and RMSE for temperature.
- Bias (mean error).
- Compare raw forecast vs corrected forecast on exactly the same chronological test rows.
- Report by lead time, city/region, season, and event subset where sample size permits.
- Optional prediction intervals only after implementation and coverage evaluation.

## 6.4 RainRisk — priority hazard module

### Objective

Estimate future precipitation and heavy-rainfall risk using forecast inputs plus historical precipitation context.

### Recommended modelling design

Rainfall has many zero values and a skewed distribution. Evaluate a two-part design:

1. `LightGBMClassifier` for probability of a defined rainfall event.
2. `LightGBMRegressor` for rainfall amount, either conditional on rain or with a separately justified target strategy.

Compare against a single regressor and raw forecast baseline. Do not assume the two-part model is automatically better.

### Candidate inputs

- Forecast precipitation/rain/showers for the target valid period.
- Forecast humidity, wind, pressure, cloud, CAPE or other instability features only if the selected model and geography actually supply them.
- Recent precipitation lags and rolling totals computed without future data.
- Season/monsoon/calendar features.
- Location or grid features.
- Forecast model and lead-time metadata.

### Candidate targets

- Precipitation amount in mm.
- Probability of measurable rain.
- Probability of heavy-rainfall threshold exceedance.

Thresholds must be taken from a documented, appropriate IMD classification or another explicitly selected standard. Do not choose a heavy-rain threshold merely because it makes the classes balanced. Keep thresholds in config and cite the source.

### Data candidates

- Open-Meteo Previous Runs precipitation/rain variables as historical forecast inputs.
- IMD 0.25° daily gridded rainfall as a long historical target/context dataset, subject to alignment and usage review.
- CHIRPS as an alternative/comparison source.
- GPM IMERG for later sub-daily satellite precipitation comparison.

### Evaluation

For amount regression: MAE, RMSE, bias, and event-focused error analysis.  
For event probability: precision, recall, F1, PR-AUC, Brier score, and calibration.  
For threshold events: confusion matrix and false-negative/false-alarm analysis.  
Compare to raw forecast and a simple climatology/persistence baseline.

## 6.5 AirQuality — priority environmental module

### Objective

Estimate near-term particulate-matter conditions and derive an India-relevant AQI indicator when the required pollutant data are available.

### Recommended first task

Prototype a next-day PM2.5 regression model for the 50-city dataset, using time-series features. The dataset's air-quality labels are CAMS model-derived values, not CPCB station observations. Therefore the first model must be described as predicting a **model-derived PM2.5 estimate**, unless independent station data are used as the target.

### Candidate inputs

- Previous PM2.5 values (lags 1, 2, 3, 7, and other validated windows).
- Rolling PM2.5 means/maxima computed using past-only windows.
- Past PM10 and other pollutants where available and reliable.
- Weather history and forecast variables available at prediction time.
- Wind speed/direction, humidity, temperature, rainfall, and seasonal/calendar features.
- City coordinates/elevation/region where appropriate.

### Target and outputs

- Predict next-day PM2.5 (µg/m³).
- Derive India AQI or PM-based category using the official CPCB breakpoint method only after implementing the correct pollutant averaging windows, minimum pollutant availability rules, and current official thresholds.
- Prefer PM2.5/PM10-based index if using the dataset's documented caveat about CAMS ozone. Do not silently output the dataset's all-pollutant `india_aqi` as if it were station-verified CPCB AQI.

### Evaluation

- MAE, RMSE, bias for PM2.5.
- Category accuracy, macro-F1, confusion matrix, and severe-category recall if categories are produced.
- Compare against persistence and seasonal baselines.
- For external validity, compare to CPCB station data where historical station coverage and spatial/time alignment allow.

## 6.6 StormRisk — official warning integration first

Do **not** train an independent severe-storm predictor in the initial MVP.

Initial direction:

- Integrate official IMD warnings/advisories only when a reliable source/API/feed is verified.
- Preserve issuing agency, warning text/category, issue time, expiry/valid period, geographic applicability, and source link.
- Add forecast-derived context only as clearly labelled supporting information.

A future ML storm/convective module would require suitable historical radar, satellite, lightning, CAPE, wind, rainfall, and event-label data. Do not claim that ordinary daily city weather rows are sufficient for severe storm nowcasting.

## 6.7 WaterRisk — future module

Potential future tasks:

- Soil-moisture anomaly estimation.
- Rainfall deficit/drought indicators.
- Evapotranspiration and land-condition analysis.

Candidate data: ERA5-Land soil moisture/temperature, CHIRPS/IMD rainfall, and any verified official drought indicators. This module must wait until data, target definitions, and validation methodology are chosen.

## 6.8 CarbonWatch — future module

CarbonWatch must distinguish:

- Ambient CO₂ concentration measurements.
- Satellite/model atmospheric concentration products.
- Estimated emissions derived from human activity and emission factors.

These are different quantities. Do not call an emissions estimate a direct CO₂ measurement. A credible emissions module requires a traceable inventory methodology, activity data, emission factors, spatial allocation method, and uncertainty documentation. It is not a first-milestone model.

## 6.9 CompoundRisk — transparent aggregation before ML

CompoundRisk is initially a **deterministic, explainable risk aggregation engine**, not a separately trained model.

It may combine validated hazard outputs with a vulnerability/exposure index once suitable data are available.

A configurable prototype form may be:

```text
compound_score = sum(weight_i * hazard_score_i * vulnerability_factor_i)
```

Where:

- Each hazard score is normalised to a documented scale, normally 0–1.
- Weights are configured, documented, and versioned; weights must be non-negative and sum to 1 if using this weighted-sum interpretation.
- Vulnerability factor is a documented value/index, not a made-up number.
- Missing hazards or missing vulnerability data must be handled explicitly; do not silently treat missing data as zero risk.

This formula is only a starting architecture, not an approved scientific risk model. It must be reviewed and tested before being presented as operational. Show the component hazards and methodology so the user can understand the compound result.

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

## 11.3 Canonical prediction record

```text
prediction_id
module                         # heat_risk | rain_risk | air_quality | ...
model_name
model_version
model_artifact_checksum
location_id
location_type
latitude
longitude
issue_time_utc
valid_time_utc
lead_hours
lead_days
prediction_type                # forecast_correction | regression | probability | rule_based_score | official_alert
variable
value
unit
risk_category                  # nullable unless validly derived
probability                    # nullable; only if calibrated/meaningful
uncertainty_value              # nullable; only if computed
uncertainty_method             # nullable
input_sources
source_timestamps
spatial_resolution
methodology_version
created_at_utc
limitations
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
| 0.1.0 | 15 Nov 2026 | Initial single-source-of-truth specification. Establishes hybrid intelligence, modular hazard architecture, ML-only first phase, source registry, data/ML validation rules, future deployment and map direction. |

## Change process

For every material change, add a row above with:

- Version and date.
- What changed.
- Why it changed.
- Which modules, datasets, algorithms, APIs, or deployment assumptions are affected.
- Any migration or backward-compatibility notes.

**Do not silently rewrite the product direction, replace the chosen primary algorithm, change data sources, or start the UI phase. Update this document first.**
