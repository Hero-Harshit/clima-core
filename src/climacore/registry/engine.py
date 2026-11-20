"""
Master Canonical Registry Engine for Clima-Core.
Provides single-source-of-truth registries for:
  1. ML Models (Flagship Champions, features, checksums, metrics, and limitations)
  2. Canonical Datasets (ingested records, source agencies, temporal ranges, and hashes)
  3. Sourced Physical Thresholds (biophysical and engineering standards with citations)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root
from climacore.registry.models import ModelRecord, DatasetRecord, PhysicalThresholdRecord

logger = get_logger("climacore.registry.engine")

_CANONICAL_MODELS: Optional[Dict[str, ModelRecord]] = None
_CANONICAL_DATASETS: Optional[Dict[str, DatasetRecord]] = None
_CANONICAL_THRESHOLDS: Optional[Dict[str, PhysicalThresholdRecord]] = None


def get_model_registry() -> Dict[str, ModelRecord]:
    """Load or build the canonical ML model registry."""
    global _CANONICAL_MODELS
    if _CANONICAL_MODELS is not None:
        return _CANONICAL_MODELS

    root = get_project_root()
    art_dir = root / "artifacts" / "models"

    def get_hash(rel_path: str) -> str:
        p = root / rel_path
        if p.exists():
            return compute_sha256(p)
        return "PENDING_SERIALIZATION"

    models: List[ModelRecord] = [
        ModelRecord(
            model_id="flood_risk_v2",
            name="FloodRisk V2 Pure Hydrological Hazard Classifier",
            version="2.0.0",
            flagship_suite="FloodRisk",
            task_type="binary_classification",
            algorithm="LightGBM GBDT",
            target="flood_disaster_event",
            target_description="Declared district-level flood disaster occurrence (IFI v3)",
            training_period="1980-2018",
            validation_period="2019-2021",
            test_period="2022-2023",
            validation_strategy="event_safe_and_chronological",
            features=[
                "rainfall_mm", "rain_rolling_3d", "rain_rolling_7d", "rain_rolling_14d",
                "rain_rolling_30d", "rain_max_3d", "api_14d", "precipitation_hours",
                "et0_fao_evapotranspiration", "wind_gusts_10m_max",
                "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
                "topsoil_moisture_anomaly", "rootzone_moisture_anomaly",
                "soil_moisture_deficit_index", "drainage_canals_15km",
                "bridges_15km", "transit_stations_15km", "population_15km",
                "dense_cluster_pop_15km"
            ],
            features_count=20,
            source_datasets=["IIT Delhi IFI v3", "IMD Gridded Rainfall", "ERA5-Land", "OSM", "WorldPop"],
            metrics={
                "roc_auc": 0.7931,
                "pr_auc": 0.1134,
                "brier_score": 0.1373,
                "precision_at_p35": 0.0314,
                "recall_at_p35": 0.7370,
            },
            baseline_metrics={"persistence_pr_auc": 0.0121, "seasonal_climatology_auc": 0.5840},
            artifact_path="artifacts/models/flood_risk/v0.1.0/flood_risk_v1.booster",
            artifact_sha256=get_hash("artifacts/models/flood_risk/v0.1.0/flood_risk_v1.booster"),
            known_limitations=[
                "Disaster inventories record district-level severe inundation; localized micro-urban puddling is unrecorded.",
                "Pure surface runoff volume does not predict subterranean pipe overflow without hydraulic conduit dimensions."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="flood_risk_challenger_scs_cn",
            name="FloodRisk SCS-CN Hydrological Runoff Challenger",
            version="3.1.0",
            flagship_suite="FloodRisk",
            task_type="classification",
            algorithm="LightGBM Binary GBDT + USDA NRCS SCS-CN Runoff",
            target="flood_disaster_event",
            target_description="Declared district-level flood disaster occurrence (IFI v3)",
            training_period="1980-2018",
            validation_period="2019-2021",
            test_period="2022-2023",
            validation_strategy="event_safe_and_chronological",
            features=[
                "rainfall_mm", "rain_rolling_3d", "rain_rolling_7d", "rain_rolling_14d",
                "rain_rolling_30d", "rain_max_3d", "api_14d", "precipitation_hours",
                "et0_fao_evapotranspiration", "wind_gusts_10m_max",
                "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
                "topsoil_moisture_anomaly", "rootzone_moisture_anomaly",
                "soil_moisture_deficit_index", "drainage_canals_15km",
                "bridges_15km", "transit_stations_15km", "population_15km",
                "dense_cluster_pop_15km", "scs_runoff_depth_mm",
                "scs_runoff_coefficient", "scs_potential_retention_s"
            ],
            features_count=23,
            source_datasets=["IIT Delhi IFI v3", "IMD Gridded Rainfall", "ERA5-Land", "OSM", "WorldPop", "USDA NRCS NEH-4"],
            metrics={
                "roc_auc": 0.8011,
                "pr_auc": 0.1224,
                "brier_score": 0.1354,
            },
            baseline_metrics={"champion_roc_auc": 0.7931, "champion_pr_auc": 0.1134},
            artifact_path="artifacts/models/flood_risk/v0.1.0/flood_risk_challenger_scs_cn.booster",
            artifact_sha256=get_hash("artifacts/models/flood_risk/v0.1.0/flood_risk_challenger_scs_cn.booster"),
            known_limitations=[
                "SCS-CN estimates surface runoff depth and retention; does not model conduit flow or backwater."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="heat_risk_v3",
            name="HeatRisk V3 Dual Diurnal Temperature Model",
            version="3.0.0",
            flagship_suite="HeatRisk",
            task_type="regression",
            algorithm="LightGBM GBDT",
            target="temperature_2m_max",
            target_description="Daily maximum 2m air temperature in urban core (°C)",
            training_period="1980-2015",
            validation_period="2016-2019",
            test_period="2020-2023",
            validation_strategy="chronological",
            features=[
                "temp_max_lag_1", "temp_max_lag_2", "temp_max_lag_3", "temp_max_lag_7",
                "temp_min_lag_1", "relative_humidity_lag_1", "shortwave_radiation_lag_1",
                "day_of_year", "sin_doy", "cos_doy", "elevation_m", "latitude", "longitude"
            ],
            features_count=13,
            source_datasets=["ERA5-Land Climate Master", "IMD 74-Year Gridded Temperature"],
            metrics={
                "test_mae_c": 1.4120,
                "test_rmse_c": 1.9840,
                "test_r2": 0.9120,
                "mean_bias_c": -0.0420,
            },
            baseline_metrics={"persistence_mae_c": 2.1400, "seasonal_climatology_mae_c": 3.6500},
            artifact_path="artifacts/models/heat_risk/v0.1.0/model.txt",
            artifact_sha256=get_hash("artifacts/models/heat_risk/v0.1.0/model.txt"),
            known_limitations=[
                "Model predicts ~9km grid-mean temperature; localized street canyon microclimates require Oke UHI delta.",
                "Surface stations with poor vegetation shading can measure 1-2°C higher than model grid cell."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="power_grid_v2_reg",
            name="PowerGrid V2 Peak Load Demand Regressor",
            version="2.0.0",
            flagship_suite="PowerGrid",
            task_type="regression",
            algorithm="LightGBM GBDT",
            target="peak_met_mw",
            target_description="State-level peak electricity load met in Megawatts (MW)",
            training_period="2013-2017",
            validation_period="2018",
            test_period="2019-2020",
            validation_strategy="chronological",
            features=[
                "temp_min_c", "temp_mean_c", "temp_max_c", "dewpoint_mean_c",
                "cooling_degree_days", "is_tropical_night", "diurnal_temp_range",
                "solar_radiation_mj", "day_of_week", "is_weekend", "month", "population_15km"
            ],
            features_count=12,
            source_datasets=["POSOCO / CEA Telemetry", "ERA5-Land"],
            metrics={
                "test_r2": 0.8758,
                "test_mae_mw": 512.4,
                "test_rmse_mw": 784.6,
                "test_mape_pct": 6.84,
            },
            baseline_metrics={"persistence_r2": 0.7420, "rolling_7d_r2": 0.7810},
            artifact_path="artifacts/models/electricity_grid/v0.1.0/peak_grid_load_v1.booster",
            artifact_sha256=get_hash("artifacts/models/electricity_grid/v0.1.0/peak_grid_load_v1.booster"),
            known_limitations=[
                "Aggregated at state level; intra-city feeder loading cannot be observed directly without DISCOM SCADA.",
                "Unscheduled industrial power outages can artificially depress peak load met."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="power_grid_v2_clf",
            name="PowerGrid V2 Unmet Supply Stress Classifier",
            version="2.0.0",
            flagship_suite="PowerGrid",
            task_type="binary_classification",
            algorithm="LightGBM Binary Classifier",
            target="is_grid_deficit_day",
            target_description="Occurrence of unmet power demand / generation deficit (>50 MW unserved load)",
            training_period="2013-2017",
            validation_period="2018",
            test_period="2019-2020",
            validation_strategy="chronological",
            features=[
                "temp_max_c", "cooling_degree_days", "is_tropical_night",
                "diurnal_temp_range", "solar_radiation_mj", "day_of_week",
                "month", "population_15km"
            ],
            features_count=8,
            source_datasets=["POSOCO / CEA Telemetry", "ERA5-Land"],
            metrics={
                "test_roc_auc": 0.9521,
                "test_recall": 0.9641,
                "test_precision": 0.8842,
                "brier_score": 0.0412,
            },
            baseline_metrics={"persistence_auc": 0.6120},
            artifact_path="artifacts/models/electricity_grid/v0.1.0/grid_deficit_hazard_v1.booster",
            artifact_sha256=get_hash("artifacts/models/electricity_grid/v0.1.0/grid_deficit_hazard_v1.booster"),
            known_limitations=[
                "Excludes forced outage contingencies from sudden transmission corridor trips outside the urban zone."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="agrisense_v2",
            name="AgriSense V2 10-Class Crop Suitability Classifier",
            version="2.0.0",
            flagship_suite="AgriSense",
            task_type="multiclass_classification",
            algorithm="LightGBM Multi-class GBDT",
            target="dominant_crop",
            target_description="Optimal staple/commercial crop selection across 10 classes",
            training_period="1997-2011",
            validation_period="2012-2013",
            test_period="2014-2015",
            validation_strategy="chronological",
            features=[
                "soil_ph_h2o_0_30cm", "soil_total_n_g_kg_0_30cm", "soil_soc_g_kg_0_30cm",
                "soil_cec_cmol_kg_0_30cm", "soil_clay_pct_0_30cm", "soil_sand_pct_0_30cm",
                "soil_silt_pct_0_30cm", "soil_bulk_density_kg_dm3_0_30cm", "temp_mean_c",
                "temp_max_c", "temp_min_c", "humidity_pct", "rainfall_mm",
                "rain_days_proxy_mm_per_day", "solar_mj_m2_day", "annual_rainfall_mm",
                "annual_temp_mean_c", "annual_humidity_pct", "elevation_m"
            ],
            features_count=19,
            source_datasets=["ICRISAT / ICAR Crop Database", "SoilGrids 250m"],
            metrics={
                "top1_test_accuracy": 0.8249,
                "top3_test_accuracy": 0.9762,
                "weighted_f1": 0.8250,
                "weighted_precision": 0.8403,
                "weighted_recall": 0.8249,
            },
            baseline_metrics={"majority_class_accuracy": 0.4954, "accuracy_lift_pct": 66.51},
            artifact_path="artifacts/models/crop_resilience/v0.1.0/crop_suitability_v1.booster",
            artifact_sha256=get_hash("artifacts/models/crop_resilience/v0.1.0/crop_suitability_v1.booster"),
            known_limitations=[
                "Predicts historical agronomic suitability, not real-time market farmgate procurement prices.",
                "Assumes rainfed or standard canal irrigation; does not observe private tubewell depth."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="air_quality_v2_pm25",
            name="AirQuality V2 Next-Day PM2.5 Regressor",
            version="2.0.0",
            flagship_suite="AirQuality",
            task_type="regression",
            algorithm="LightGBM GBDT",
            target="target_pm2_5_next_day",
            target_description="Next-day 24h mean PM2.5 particulate concentration (µg/m³)",
            training_period="2022-08-12 to 2025-05-31",
            validation_period="2025-06-01 to 2025-12-31",
            test_period="2026-01-01 to 2026-10-29",
            validation_strategy="chronological",
            features=[
                "pm2_5_mean", "pm2_5_max", "pm10_mean", "no2_mean", "aod_mean",
                "temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_max",
                "wind_direction_10m_dominant", "precipitation_sum", "day_of_year", "month"
            ],
            features_count=12,
            source_datasets=["Open-Meteo CAMS Air Quality", "ERA5-Land"],
            metrics={
                "test_mae_ug_m3": 6.3796,
                "test_rmse_ug_m3": 9.6255,
                "test_r2": 0.8725,
                "cpcb_category_accuracy": 0.8074,
                "mean_bias_ug_m3": -0.2298,
            },
            baseline_metrics={"persistence_mae": 6.9495, "rolling_7d_mae": 9.3891},
            artifact_path="artifacts/models/air_quality/v0.1.0/model.txt",
            artifact_sha256=get_hash("artifacts/models/air_quality/v0.1.0/model.txt"),
            known_limitations=[
                "Based on ~45km CAMS atmospheric reanalysis grid cells; hyper-local street-level traffic microclimates differ.",
                "Requires CPCB ground monitor calibration transfer before serving legal environmental compliance notices."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="flood_risk_xgboost_23feat",
            name="FloodRisk XGBoost SCS-CN Runoff Challenger",
            version="1.2.0",
            flagship_suite="FloodRisk",
            task_type="binary_classification",
            algorithm="XGBoost Classifier (Hist, Optuna HPO)",
            target="flood_disaster_event",
            target_description="Declared district-level flood disaster occurrence (IFI v3)",
            training_period="1980-2018",
            validation_period="2019-2021",
            test_period="2022-2023",
            validation_strategy="event_safe_and_chronological",
            features=[
                "rainfall_mm", "rain_rolling_3d", "rain_rolling_7d", "rain_rolling_14d",
                "rain_rolling_30d", "rain_max_3d", "api_14d", "precipitation_hours",
                "et0_fao_evapotranspiration", "wind_gusts_10m_max",
                "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
                "topsoil_moisture_anomaly", "rootzone_moisture_anomaly",
                "soil_moisture_deficit_index", "drainage_canals_15km",
                "bridges_15km", "transit_stations_15km", "population_15km",
                "dense_cluster_pop_15km", "scs_runoff_depth_mm",
                "scs_runoff_coefficient", "scs_potential_retention_s"
            ],
            features_count=23,
            source_datasets=["IIT Delhi IFI v3", "IMD Gridded Rainfall", "ERA5-Land", "OSM", "WorldPop", "USDA NRCS NEH-4"],
            metrics={
                "roc_auc": 0.8014,
                "pr_auc": 0.1077,
                "brier_score": 0.1374,
                "recall_at_p35": 0.7370,
            },
            baseline_metrics={"champion_roc_auc": 0.8011, "champion_pr_auc": 0.1224},
            artifact_path="artifacts/models/v1_2_challengers/floodrisk/flood_risk_xgboost_23feat.json",
            artifact_sha256=get_hash("artifacts/models/v1_2_challengers/floodrisk/flood_risk_xgboost_23feat.json"),
            known_limitations=[
                "SCS-CN excess runoff captures hydrological depth; does not observe internal stormwater drainage culvert blockages."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="heat_risk_quantile_suite",
            name="HeatRisk V1.2 Tri-Quantile Temperature Uncertainty Suite",
            version="1.2.0",
            flagship_suite="HeatRisk",
            task_type="quantile_regression",
            algorithm="LightGBM Quantile Regressors (Pinball Loss Q10, Q50, Q90)",
            target="target_temperature_2m_max_next_day",
            target_description="Next-day maximum temperature 80% confidence corridor [Q10, Q90] (°C)",
            training_period="2020-01-08 to 2024-12-31",
            validation_period="2025-01-01 to 2025-12-31",
            test_period="2026-01-01 to 2026-10-29",
            validation_strategy="chronological",
            features=[
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_max", "precipitation_sum", "relative_humidity_2m_mean"
            ],
            features_count=71,
            source_datasets=["ERA5-Land Climate Master", "IMD Gridded Temperature"],
            metrics={
                "pinball_loss_q10": 0.2762,
                "pinball_loss_q50": 0.5116,
                "pinball_loss_q90": 0.2103,
                "empirical_coverage_80pct": 76.82,
                "avg_interval_width_c": 3.01,
                "monotonicity_crossing_pct": 0.0,
            },
            baseline_metrics={"point_champion_r2": 0.9510, "point_champion_mae_c": 1.0234},
            artifact_path="artifacts/models/v1_2_challengers/heatrisk/heat_risk_quantile_q_50.booster",
            artifact_sha256=get_hash("artifacts/models/v1_2_challengers/heatrisk/heat_risk_quantile_q_50.booster"),
            known_limitations=[
                "Captures synoptic forecast spread; does not account for micro-scale localized building shadow variations."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="power_grid_tcn_v4",
            name="PowerGrid V4 Causal Dilated 1D TCN Sequence Challenger",
            version="4.0.0",
            flagship_suite="PowerGrid",
            task_type="regression",
            algorithm="PyTorch 1D Causal Dilated Convolutional Network with Residual Blocks",
            target="peak_met_mw",
            target_description="State/City Peak electricity load met in Megawatts (MW)",
            training_period="2013-2018",
            validation_period="2019",
            test_period="2020",
            validation_strategy="chronological_sliding_window",
            features=[
                "temp_min_c", "temp_mean_c", "temp_max_c", "dewpoint_mean_c",
                "utci_mean_c", "utci_max_c", "cooling_degree_days", "heating_degree_days",
                "is_tropical_night", "diurnal_temp_range", "solar_radiation_mj", "cloud_cover_pct"
            ],
            features_count=12,
            source_datasets=["POSOCO / CEA Telemetry", "ERA5-Land"],
            metrics={
                "test_r2": 0.5385,
                "test_mae_mw": 3048.58,
                "test_rmse_mw": 4502.47,
                "trainable_parameters": 15545,
            },
            baseline_metrics={"champion_r2": 0.8758, "champion_mae_mw": 1519.18},
            artifact_path="artifacts/models/v1_2_challengers/powergrid/powergrid_tcn_v4.pt",
            artifact_sha256=get_hash("artifacts/models/v1_2_challengers/powergrid/powergrid_tcn_v4.pt"),
            known_limitations=[
                "Requires contiguous 30-day lookback series; missing dates must be linearly interpolated."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="drought_cascade_v1",
            name="DroughtCascade V1 Multi-Decadal Latent Regime Engine",
            version="1.0.0",
            flagship_suite="DroughtCascade",
            task_type="clustering",
            algorithm="Gaussian Hidden Markov Model (Diagonal Covariance, 4 Regimes)",
            target="latent_drought_regime",
            target_description="Latent agricultural & hydrological drought regime (Equilibrium, Deficit, Severe, Recovery)",
            training_period="1980-2024",
            validation_period="1980-2024",
            test_period="1980-2024",
            validation_strategy="continuous_historical_decoding",
            features=[
                "soil_moisture_deficit_index", "rootzone_moisture_anomaly",
                "soil_moisture_0_to_7cm_mean", "soil_temperature_0_to_7cm_mean"
            ],
            features_count=4,
            source_datasets=["ERA5-Land 44-Year Soil Moisture Archive"],
            metrics={
                "regime_0_persistence": 0.9610,
                "regime_1_persistence": 0.9500,
                "regime_2_severe_drought_persistence": 0.9700,
                "severe_drought_expected_duration_days": 33.2,
            },
            baseline_metrics={"naive_threshold_persistence": 0.7200},
            artifact_path="artifacts/models/v1_2_challengers/drought/drought_cascade_hmm.joblib",
            artifact_sha256=get_hash("artifacts/models/v1_2_challengers/drought/drought_cascade_hmm.joblib"),
            known_limitations=[
                "Identifies rootzone and topsoil moisture stress; groundwater table depletion requires CGWB well integration."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="forecast_fusion_v1_emos",
            name="ForecastFusion V1 Multi-Model EMOS Distributional Engine",
            version="1.3.0",
            flagship_suite="ForecastFusion",
            task_type="probabilistic_regression",
            algorithm="Ensemble Model Output Statistics (CRPS Loss Minimization)",
            target="temperature_2m_max",
            target_description="Maximum daily 2m air temperature continuous Gaussian distribution N(mu, sigma^2)",
            training_period="2024-01-20 to 2024-09-14",
            validation_period="2024-08-01 to 2024-09-14",
            test_period="2024-09-15 to 2024-11-15",
            validation_strategy="chronological",
            features=["tmax_gfs", "tmax_jma", "tmax_icon", "t2m_max_fcst"],
            features_count=4,
            source_datasets=["NOAA GFS", "Japan JMA", "DWD ICON", "Open-Meteo BestMatch", "IMD Gridded Temperature"],
            metrics={
                "test_mae_c": 0.6237,
                "test_rmse_c": 0.7927,
                "crps": 0.4764,
                "coverage_80pct": 95.56,
                "mean_interval_width_c": 3.24,
            },
            baseline_metrics={"raw_dwd_icon_mae_c": 0.7900, "raw_bestmatch_mae_c": 0.9296},
            artifact_path="artifacts/models/v1_3/forecast_fusion/tmax_model_b_emos.joblib",
            artifact_sha256=get_hash("artifacts/models/v1_3/forecast_fusion/tmax_model_b_emos.joblib"),
            known_limitations=[
                "Trained on 4 representative Indian bioclimatic zones; mountain valleys require local station downscaling."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="forecast_fusion_v1_precip",
            name="ForecastFusion V1 Two-Part Hurdle Precipitation Ensemble",
            version="1.3.0",
            flagship_suite="ForecastFusion",
            task_type="zero_inflated_regression",
            algorithm="Two-Part Hurdle Ensemble (LGBM Classifier + Quantile LGBM Regressor)",
            target="precipitation_sum",
            target_description="Daily precipitation expected amount and occurrence probability P(Rain > 0.1 mm)",
            training_period="2024-01-20 to 2024-09-14",
            validation_period="2024-08-01 to 2024-09-14",
            test_period="2024-09-15 to 2024-11-15",
            validation_strategy="chronological",
            features=["precip_gfs", "precip_jma", "precip_icon", "precip_fcst", "latitude", "longitude", "elevation"],
            features_count=11,
            source_datasets=["NOAA GFS", "Japan JMA", "DWD ICON", "Open-Meteo BestMatch", "IMD Gridded Rainfall"],
            metrics={
                "test_mae_mm": 4.2685,
                "test_rmse_mm": 10.2302,
                "brier_score": 0.1503,
                "rain_prob_frequency": 0.3185,
            },
            baseline_metrics={"raw_bestmatch_brier": 0.2540, "raw_gfs_brier": 0.2137},
            artifact_path="artifacts/models/v1_3/forecast_fusion/precip_model_d_two_part.joblib",
            artifact_sha256=get_hash("artifacts/models/v1_3/forecast_fusion/precip_model_d_two_part.joblib"),
            known_limitations=[
                "Resolves convective initiation; localized cloudbursts (<5km) require Doppler weather radar extrapolation."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="basinflow_v1_flood",
            name="BasinFlow V1 Hydrological Routing & FloodRisk Challenger",
            version="1.3.0",
            flagship_suite="FloodRisk",
            task_type="binary_classification",
            algorithm="LightGBM Classifier + SCS Unit Hydrograph + Muskingum-Cunge Reach Routing",
            target="flood_disaster_event",
            target_description="Declared district-level flood disaster occurrence (IFI v3)",
            training_period="1980-2018",
            validation_period="2019-2021",
            test_period="2022-2023",
            validation_strategy="event_safe_and_chronological",
            features=[
                "rainfall_mm", "rain_rolling_30d", "api_14d", "scs_runoff_depth_mm",
                "hydrograph_peak_discharge_m3s", "routed_river_peak_m3s", "upstream_catchment_sqkm",
                "river_reach_length_km", "river_strahler_order"
            ],
            features_count=30,
            source_datasets=["IIT Delhi IFI v3", "HydroSHEDS HydroRIVERS", "IMD Gridded Rainfall", "USDA NRCS NEH-630"],
            metrics={
                "roc_auc": 0.7880,
                "pr_auc": 0.1223,
                "brier_score": 0.1370,
                "recall_at_035": 0.7143,
            },
            baseline_metrics={"champion_roc_auc": 0.8013, "champion_pr_auc": 0.1226},
            artifact_path="artifacts/models/v1_3/basinflow/flood_risk_basinflow_challenger.booster",
            artifact_sha256=get_hash("artifacts/models/v1_3/basinflow/flood_risk_basinflow_challenger.booster"),
            known_limitations=[
                "GNN streamflow discharge modeling is blocked awaiting open continuous gauge rating curves (m3/s)."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="smokepath_v2_airquality",
            name="SmokePath V2 Biomass Burning Smoke Transport Challenger",
            version="1.3.0",
            flagship_suite="AirQuality",
            task_type="regression",
            algorithm="LightGBM Regressor + Satellite FRP Advection + Wind Vectors + CAMS AOD",
            target="target_pm2_5_next_day",
            target_description="Next-day daily mean PM2.5 air pollutant concentration (ug/m3)",
            training_period="2022-08-12 to 2025-05-31",
            validation_period="2025-06-01 to 2025-12-31",
            test_period="2026-01-01 to 2026-10-29",
            validation_strategy="chronological",
            features=[
                "pm2_5_mean", "wind_u_component", "wind_v_component", "smoke_advection_index",
                "aod_smoke_coupling", "ventilation_index_proxy", "stagnation_inversion_index"
            ],
            features_count=99,
            source_datasets=["CPCB Station Network", "Open-Meteo CAMS", "NASA FIRMS Proxy", "ERA5-Land"],
            metrics={
                "test_mae_ug_m3": 6.3410,
                "test_rmse_ug_m3": 9.5755,
                "test_r2": 0.8738,
            },
            baseline_metrics={"champion_mae": 6.3796, "champion_rmse": 9.6255},
            artifact_path="artifacts/models/v1_3/smoke_transport/air_quality_smokepath_challenger.booster",
            artifact_sha256=get_hash("artifacts/models/v1_3/smoke_transport/air_quality_smokepath_challenger.booster"),
            known_limitations=[
                "Observational transport indicator; does not claim legal source attribution without chemical speciation."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="drought_cascade_v2",
            name="DroughtCascade V2 Causal Forward Filtering Gaussian HMM",
            version="2.0.0",
            flagship_suite="DroughtCascade",
            task_type="clustering",
            algorithm="Gaussian HMM with Sequential Causal Forward Filtering P(Z_t | X_{1:t})",
            target="causal_drought_regime",
            target_description="Online agricultural & hydrological drought state without retrospective lookahead bias",
            training_period="1980-2024",
            validation_period="1980-2024",
            test_period="1980-2024",
            validation_strategy="causal_online_filtering",
            features=[
                "soil_moisture_deficit_index", "rootzone_moisture_anomaly",
                "soil_moisture_0_to_7cm_mean", "soil_temperature_0_to_7cm_mean"
            ],
            features_count=4,
            source_datasets=["ERA5-Land 44-Year Soil Moisture Archive"],
            metrics={
                "causal_viterbi_agreement_pct": 93.93,
                "severe_drought_f1": 0.9129,
                "severe_drought_recall_pct": 90.18,
                "severe_drought_expected_dwell_days": 17.9,
            },
            baseline_metrics={"retrospective_viterbi_agreement": 100.0},
            artifact_path="artifacts/models/v1_3/drought/drought_cascade_v2.joblib",
            artifact_sha256=get_hash("artifacts/models/v1_3/drought/drought_cascade_v2.joblib"),
            known_limitations=[
                "Causal filtering mandated for all live API calls; retrospective Viterbi restricted to offline audits."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="heat_risk_v5_moe",
            name="HeatRisk V5 Soft-Gated Regularized Mixture of Experts",
            version="5.0.0",
            flagship_suite="HeatRisk",
            task_type="regression",
            algorithm="Soft-Gated Regularized MoE (Global Backbone + 5 Regional Residual Experts + Softmax Gating)",
            target="target_temperature_2m_max_next_day",
            target_description="Next-day maximum temperature across 5 specialized bioclimatic regimes (°C)",
            training_period="2020-01-08 to 2024-12-31",
            validation_period="2025-01-01 to 2025-12-31",
            test_period="2026-01-01 to 2026-10-29",
            validation_strategy="chronological",
            features=[
                "temperature_2m_max", "temperature_2m_min", "latitude", "longitude",
                "elevation_m", "sin_day_of_year", "cos_day_of_year"
            ],
            features_count=71,
            source_datasets=["ERA5-Land Climate Master", "IMD Gridded Temperature"],
            metrics={
                "test_mae_c": 1.0237,
                "test_rmse_c": 1.4160,
                "test_r2": 0.9511,
                "optimal_shrinkage_lambda": 0.60,
            },
            baseline_metrics={"global_champion_mae_c": 1.0234, "global_champion_rmse_c": 1.4173},
            artifact_path="artifacts/models/v1_3/heatrisk/heatrisk_moe_v5.joblib",
            artifact_sha256=get_hash("artifacts/models/v1_3/heatrisk/heatrisk_moe_v5.joblib"),
            known_limitations=[
                "Outperforms global champion across Deccan Plateau, Montane, and Semi-Arid zones; deployed as Co-Champion."
            ],
            operational_status="CO_CHAMPION",
        ),
        ModelRecord(
            model_id="basinflow_v2_gr4j",
            name="BasinFlow V2 GR4J Lumped Conceptual Hydrological Model",
            version="2.0.0",
            flagship_suite="FloodRisk",
            task_type="regression",
            algorithm="Génie Rural à 4 paramètres Journalier (GR4J)",
            target="streamflow_mm_day",
            target_description="Daily basin catchment streamflow runoff discharge (mm/day)",
            training_period="1980-2008",
            validation_period="2009-2013",
            test_period="2014-2018",
            validation_strategy="chronological_and_ungauged_split",
            features=["precipitation_mm_day", "potential_evapotranspiration_mm_day"],
            features_count=2,
            source_datasets=["CAMELS-IND Peninsular Catchments (Zenodo 14999580)"],
            metrics={
                "median_kge": 0.4350,
                "median_nse": 0.3870,
                "median_rmse_mm": 1.8530,
            },
            baseline_metrics={"persistence_kge": 0.7760, "persistence_nse": 0.5530},
            artifact_path="artifacts/models/v2_0/basinflow/gr4j_calibrated_gauges.json",
            artifact_sha256=get_hash("artifacts/models/v2_0/basinflow/gr4j_calibrated_gauges.json"),
            known_limitations=[
                "Lumped reservoir structure assumes uniform basin precipitation; flash convective storms in large basins exhibit lag.",
                "Persistence baseline represents 1-day ahead autoregression, whereas GR4J operates as pure rainfall-runoff simulation."
            ],
            operational_status="BENCHMARK",
        ),
        ModelRecord(
            model_id="basinflow_v2_hbv_light",
            name="BasinFlow V2 HBV-Light Conceptual Catchment Model",
            version="2.0.0",
            flagship_suite="FloodRisk",
            task_type="regression",
            algorithm="HBV-Light (Hydrologiska Byråns Vattenbalansavdelning)",
            target="streamflow_mm_day",
            target_description="Daily basin catchment streamflow runoff discharge (mm/day)",
            training_period="1980-2008",
            validation_period="2009-2013",
            test_period="2014-2018",
            validation_strategy="chronological_and_ungauged_split",
            features=["precipitation_mm_day", "temperature_c_mean", "potential_evapotranspiration_mm_day"],
            features_count=3,
            source_datasets=["CAMELS-IND Peninsular Catchments (Zenodo 14999580)"],
            metrics={
                "median_kge": 0.4520,
                "median_nse": 0.4010,
                "median_rmse_mm": 1.7920,
            },
            baseline_metrics={"gr4j_kge": 0.4350},
            artifact_path="artifacts/models/v2_0/basinflow/hbv_calibrated_gauges.json",
            artifact_sha256=get_hash("artifacts/models/v2_0/basinflow/hbv_calibrated_gauges.json"),
            known_limitations=[
                "Nonlinear soil moisture routine requires careful parameter bounding under extreme Indian monsoon flash events."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="basinflow_v2_standard_lstm",
            name="BasinFlow V2 Standard Entity-Aware LSTM",
            version="2.0.0",
            flagship_suite="FloodRisk",
            task_type="regression",
            algorithm="PyTorch Entity-Aware 2-Layer LSTM with Catchment Static Attributes",
            target="streamflow_mm_day",
            target_description="Daily basin catchment streamflow runoff discharge (mm/day)",
            training_period="1980-2008",
            validation_period="2009-2013",
            test_period="2014-2018",
            validation_strategy="chronological_and_ungauged_split",
            features=["precipitation_mm_day", "temperature_c_mean", "potential_evapotranspiration_mm_day"],
            features_count=3,
            source_datasets=["CAMELS-IND Peninsular Catchments (Zenodo 14999580)"],
            metrics={
                "median_kge": 0.6120,
                "median_nse": 0.5840,
                "median_rmse_mm": 1.3420,
            },
            baseline_metrics={"gr4j_kge": 0.4350, "hbv_kge": 0.4520},
            artifact_path="artifacts/models/v2_0/basinflow/standard_lstm_weights.pt",
            artifact_sha256=get_hash("artifacts/models/v2_0/basinflow/standard_lstm_weights.pt"),
            known_limitations=[
                "Unconstrained neural architecture does not guarantee mass conservation; can hallucinate streamflow in multi-year droughts."
            ],
            operational_status="CHALLENGER",
        ),
        ModelRecord(
            model_id="basinflow_v2_mc_lstm",
            name="BasinFlow V2 Mass-Conserving LSTM (MC-LSTM)",
            version="2.0.0",
            flagship_suite="FloodRisk",
            task_type="regression",
            algorithm="Mass-Conserving LSTM (Hoedt et al. 2021) with Strictly Non-Negative Mass Allocation",
            target="streamflow_mm_day",
            target_description="Daily basin catchment streamflow runoff discharge (mm/day)",
            training_period="1980-2008",
            validation_period="2009-2013",
            test_period="2014-2018",
            validation_strategy="chronological_and_ungauged_split",
            features=["precipitation_mm_day", "temperature_c_mean", "potential_evapotranspiration_mm_day"],
            features_count=3,
            source_datasets=["CAMELS-IND Peninsular Catchments (Zenodo 14999580)"],
            metrics={
                "median_kge": 0.5890,
                "median_nse": 0.5630,
                "median_rmse_mm": 1.3950,
                "mass_conservation_guarantee": "Exact",
            },
            baseline_metrics={"standard_lstm_kge": 0.6120, "gr4j_kge": 0.4350},
            artifact_path="artifacts/models/v2_0/basinflow/mc_lstm_weights.pt",
            artifact_sha256=get_hash("artifacts/models/v2_0/basinflow/mc_lstm_weights.pt"),
            known_limitations=[
                "Reduces peak overestimation risk by enforcing mass conservation, with slight loss in unconstrained peak fitting."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="basinflow_v2_mc_lstm_ds",
            name="BasinFlow V2 Mass-Conserving LSTM with Dynamic States",
            version="2.0.0",
            flagship_suite="FloodRisk",
            task_type="regression",
            algorithm="MC-LSTM with Explicit Dynamic Storage State Tracking & Deep Percolation Outflow",
            target="streamflow_mm_day",
            target_description="Daily basin catchment streamflow runoff discharge (mm/day)",
            training_period="1980-2008",
            validation_period="2009-2013",
            test_period="2014-2018",
            validation_strategy="chronological_and_ungauged_split",
            features=["precipitation_mm_day", "temperature_c_mean", "potential_evapotranspiration_mm_day"],
            features_count=3,
            source_datasets=["CAMELS-IND Peninsular Catchments (Zenodo 14999580)"],
            metrics={
                "median_kge": 0.6010,
                "median_nse": 0.5750,
                "median_rmse_mm": 1.3680,
            },
            baseline_metrics={"mc_lstm_kge": 0.5890},
            artifact_path="artifacts/models/v2_0/basinflow/mc_lstm_ds_weights.pt",
            artifact_sha256=get_hash("artifacts/models/v2_0/basinflow/mc_lstm_ds_weights.pt"),
            known_limitations=[
                "Physically constrained dynamic reservoir partitioning; recommended for hydro-meteorological drought-flood transitions."
            ],
            operational_status="CO_CHAMPION",
        ),
        ModelRecord(
            model_id="water_quality_wqi_regressor_v1",
            name="WaterQuality V1 Continuous Water Quality Index (WQI) Regressor",
            version="1.0.0",
            flagship_suite="WaterQuality",
            task_type="regression",
            algorithm="LightGBM GBDT Regressor",
            target="water_quality_index_cpcb",
            target_description="Comprehensive CPCB 100-point Water Quality Index from chemical & physical indicators",
            training_period="2018-2022",
            validation_period="2023",
            test_period="2024",
            validation_strategy="chronological",
            features=["electrical_conductivity_us_cm", "total_dissolved_solids_mg_l", "fluoride_mg_l", "nitrate_mg_l", "ph"],
            features_count=5,
            source_datasets=["Central Ground Water Board (CGWB) Chemical Survey"],
            metrics={
                "test_r2": 0.9410,
                "test_mae": 3.8200,
                "test_rmse": 5.1400,
            },
            baseline_metrics={"deterministic_wqi_r2": 1.0000},
            artifact_path="artifacts/models/water_quality/v0.1.0/wqi_regressor_v1.booster",
            artifact_sha256=get_hash("artifacts/models/water_quality/v0.1.0/wqi_regressor_v1.booster"),
            known_limitations=[
                "Requires laboratory chemical titration or calibrated multiparameter sonde; optical proxies alone cannot replace fluoride test."
            ],
            operational_status="CHAMPION",
        ),
        ModelRecord(
            model_id="water_quality_hazard_v1",
            name="WaterQuality V1 Groundwater Contamination Hazard Classifier",
            version="1.0.0",
            flagship_suite="WaterQuality",
            task_type="binary_classification",
            algorithm="LightGBM Binary Classifier",
            target="is_potable_bis_10500",
            target_description="Non-potable water hazard classification (BIS 10500 permissible limit exceedance)",
            training_period="2018-2022",
            validation_period="2023",
            test_period="2024",
            validation_strategy="chronological",
            features=["electrical_conductivity_us_cm", "total_dissolved_solids_mg_l", "fluoride_mg_l", "nitrate_mg_l", "ph"],
            features_count=5,
            source_datasets=["Central Ground Water Board (CGWB) Chemical Survey"],
            metrics={
                "roc_auc": 0.9620,
                "pr_auc": 0.9480,
                "f1_score": 0.8950,
            },
            baseline_metrics={"prevalence_majority_f1": 0.6200},
            artifact_path="artifacts/models/water_quality/v0.1.0/water_hazard_v1.booster",
            artifact_sha256=get_hash("artifacts/models/water_quality/v0.1.0/water_hazard_v1.booster"),
            known_limitations=[
                "Predicts inorganic and physical chemical potability; biological pathogens (E. coli, coliform) require bacteriological incubation."
            ],
            operational_status="CHAMPION",
        ),
    ]

    _CANONICAL_MODELS = {m.model_id: m for m in models}
    return _CANONICAL_MODELS


def get_dataset_registry() -> Dict[str, DatasetRecord]:
    """Load or compile the canonical dataset registry covering all 14 ingested sources."""
    global _CANONICAL_DATASETS
    if _CANONICAL_DATASETS is not None:
        return _CANONICAL_DATASETS

    root = get_project_root()

    def get_hash(rel_path: str) -> str:
        p = root / rel_path
        if p.exists():
            return compute_sha256(p)
        return "PENDING_VERIFICATION"

    datasets: List[DatasetRecord] = [
        DatasetRecord(
            dataset_id="imd_temperature_74y",
            name="IMD 74-Year Gridded Daily Maximum & Minimum Temperature",
            source_agency="India Meteorological Department (IMD)",
            citation="IMD National Climate Centre (1951-2024), 1.0 degree high-resolution daily gridded temperature dataset.",
            temporal_range="1951-2024",
            spatial_coverage="All India (50 Urban Centers Aligned)",
            spatial_resolution="1.0 degree gridded",
            temporal_resolution="Daily",
            total_records=1351450,
            processed_file_path="data/processed/imd_temperature_74y_50_cities.parquet",
            sha256=get_hash("data/processed/imd_temperature_74y_50_cities.parquet"),
            feature_columns=["temp_max", "temp_min", "temp_mean", "annual_max_temp"]
        ),
        DatasetRecord(
            dataset_id="imd_rainfall_46y",
            name="IMD 46-Year Gridded Daily Precipitation",
            source_agency="India Meteorological Department (IMD)",
            citation="Pai et al. (2014), Development of a new high spatial resolution (0.25 x 0.25) long period (1901-2024) daily gridded rainfall data set over India, Mausam.",
            temporal_range="1979-2024",
            spatial_coverage="All India (50 Urban Centers Aligned)",
            spatial_resolution="0.25 degree gridded",
            temporal_resolution="Daily",
            total_records=840050,
            processed_file_path="data/processed/imd_rainfall_50_cities.parquet",
            sha256=get_hash("data/processed/imd_rainfall_50_cities.parquet"),
            feature_columns=["precipitation_mm", "annual_max_rainfall"]
        ),
        DatasetRecord(
            dataset_id="era5_land_climate_master",
            name="ERA5-Land 44-Year Historical Meteorological Archive",
            source_agency="ECMWF / Copernicus Climate Change Service",
            citation="Muñoz-Sabater et al. (2021), ERA5-Land: a state-of-the-art global reanalysis dataset for land applications, Earth System Science Data.",
            temporal_range="1980-2023",
            spatial_coverage="50 Indian Cities",
            spatial_resolution="~9 km (0.1 degree)",
            temporal_resolution="Daily",
            total_records=823048,
            processed_file_path="data/processed/weather_historical_44y_train_ready.parquet",
            sha256=get_hash("data/processed/weather_historical_44y_train_ready.parquet"),
            feature_columns=[
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_max", "precipitation_sum", "wind_speed_10m_max",
                "relative_humidity_2m_mean", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
            ]
        ),
        DatasetRecord(
            dataset_id="ifi_flood_inventory",
            name="IIT Delhi India Flood Inventory (IFI) v3",
            source_agency="IIT Delhi Department of Civil Engineering",
            citation="India Flood Inventory v3 (1980-2023), Hydro-meteorological disaster event catalog.",
            temporal_range="1980-2023",
            spatial_coverage="50 Indian Cities (District-Aligned)",
            spatial_resolution="District Administrative Unit",
            temporal_resolution="Daily (Disaster Event Tagged)",
            total_records=803550,
            processed_file_path="data/processed/floods/city_flood_history_aligned.parquet",
            sha256=get_hash("data/processed/floods/city_flood_history_aligned.parquet"),
            feature_columns=["flood_disaster_event", "rainfall_mm", "api_14d", "drainage_canals_15km"]
        ),
        DatasetRecord(
            dataset_id="posoco_electricity_demand",
            name="India Daily Electricity Demand & Grid Heat Stress",
            source_agency="POSOCO, Central Electricity Authority (CEA)",
            citation="University of Reading / Newcastle University (Zenodo), National Grid Telemetry 2013-2020.",
            temporal_range="2013-2020",
            spatial_coverage="All India (State-level for 50 cities)",
            spatial_resolution="State Grids",
            temporal_resolution="Daily",
            total_records=150547,
            processed_file_path="data/processed/electricity/city_daily_electricity_stress.parquet",
            sha256=get_hash("data/processed/electricity/city_daily_electricity_stress.parquet"),
            feature_columns=["peak_met_mw", "cooling_degree_days", "is_tropical_night", "is_grid_deficit_day"]
        ),
        DatasetRecord(
            dataset_id="agrisense_crop_soil",
            name="ICRISAT / ICAR District Agricultural Soil & Crop Statistics",
            source_agency="ICAR, ICRISAT, ISRIC SoilGrids 250m",
            citation="District-level Crop Production Statistics of India (1997-2015), Directorate of Economics & Statistics.",
            temporal_range="1997-2015",
            spatial_coverage="50 Urban Hinterland Districts",
            spatial_resolution="District / 250m Soil Profile",
            temporal_resolution="Seasonal / Annual",
            total_records=34903,
            processed_file_path="data/processed/agriculture/city_district_crop_profiles.parquet",
            sha256=get_hash("data/processed/agriculture/city_district_crop_profiles.parquet"),
            feature_columns=["soil_ph", "soil_organic_carbon_g_kg", "soil_water_holding_capacity", "dominant_crop"]
        ),
        DatasetRecord(
            dataset_id="sage_crop_calendar",
            name="SAGE Global Crop Calendar (India Phenology)",
            source_agency="University of Wisconsin-Madison (SAGE)",
            citation="Sacks et al. (2010), Crop planting and harvesting dates for the GAEZ study.",
            temporal_range="Canonical / Multi-decade",
            spatial_coverage="50 Indian Cities",
            spatial_resolution="District / Agro-Ecological Subregion",
            temporal_resolution="Calendar DOY Ranges",
            total_records=50,
            processed_file_path="data/processed/agriculture/city_crop_calendars.json",
            sha256=get_hash("data/processed/agriculture/city_crop_calendars.json"),
            feature_columns=["kharif_sowing_doy_range", "kharif_anthesis_doy_range", "kharif_harvest_doy_range"]
        ),
        DatasetRecord(
            dataset_id="worldpop_2024",
            name="WorldPop High-Resolution 100m Population Exposure",
            source_agency="WorldPop, University of Southampton",
            citation="WorldPop (2024), Global high-resolution population distribution datasets for 2024.",
            temporal_range="2024",
            spatial_coverage="50 Indian Cities (Concentric 5km, 15km, 30km rings)",
            spatial_resolution="100m raster grid",
            temporal_resolution="Static (2024 projection)",
            total_records=50,
            processed_file_path="data/processed/population/city_population_exposure.json",
            sha256=get_hash("data/processed/population/city_population_exposure.json"),
            feature_columns=["population_5km", "population_15km", "population_30km", "dense_cluster_pop_15km"]
        ),
        DatasetRecord(
            dataset_id="osm_critical_infrastructure",
            name="OpenStreetMap Urban Lifeline Infrastructure Snapshot",
            source_agency="OpenStreetMap Contributors via Overpass API",
            citation="OpenStreetMap Foundation (2024), Critical civil lifeline infrastructure tags.",
            temporal_range="2024",
            spatial_coverage="50 Indian Cities (15 km radial buffer)",
            spatial_resolution="Vector features / 15km buffer",
            temporal_resolution="Static Snapshot",
            total_records=50,
            processed_file_path="data/processed/infrastructure/city_infrastructure_exposure.json",
            sha256=get_hash("data/processed/infrastructure/city_infrastructure_exposure.json"),
            feature_columns=["hospitals_15km", "schools_15km", "bridges_15km", "drainage_canals_15km"]
        ),
        DatasetRecord(
            dataset_id="chetna_power_plants",
            name="CHETNA Coal Thermal Power Plant Point-Source Inventory",
            source_agency="Center for High Energy Transitions & Air (CHETNA)",
            citation="CHETNA National Thermal Power Plant Geodatabase (2024).",
            temporal_range="2024",
            spatial_coverage="276 Operational Units across India",
            spatial_resolution="Point Coordinates / 100km buffer",
            temporal_resolution="Operational Snapshot",
            total_records=276,
            processed_file_path="data/processed/power_plants/city_power_plant_exposure.json",
            sha256=get_hash("data/processed/power_plants/city_power_plant_exposure.json"),
            feature_columns=["plants_within_100km", "thermal_capacity_100km_mw", "closest_plant_distance_km"]
        ),
        DatasetRecord(
            dataset_id="camels_ind_v2_2",
            name="CAMELS-IND Peninsular India Catchment Hydro-Meteorological Archive",
            source_agency="Indian Institute of Technology (IIT) / Zenodo",
            citation="CAMELS-IND: hydrometeorological time series and catchment attributes for 472 catchments in Peninsular India (Zenodo 14999580, CC-BY-4.0).",
            temporal_range="1980-2018",
            spatial_coverage="Peninsular India (109 Quality-Screened Gauged Catchments)",
            spatial_resolution="Catchment Aggregate Polygons",
            temporal_resolution="Daily",
            total_records=1551060,
            processed_file_path="data/processed/camels_ind_selected_catchments.json",
            sha256=get_hash("data/processed/camels_ind_selected_catchments.json"),
            feature_columns=["streamflow_mm_day", "precipitation_mm_day", "potential_evapotranspiration_mm_day", "temperature_c_mean"]
        ),
        DatasetRecord(
            dataset_id="cgwb_groundwater_quality",
            name="Central Ground Water Board National Chemical Water Quality Survey",
            source_agency="Central Ground Water Board (CGWB), Ministry of Jal Shakti",
            citation="CGWB Ground Water Quality in Shallow Aquifers of India (2020-2024), BIS 10500 compliance monitoring.",
            temporal_range="2018-2024",
            spatial_coverage="All India (District Monitoring Wells)",
            spatial_resolution="Point Aquifer Borewell",
            temporal_resolution="Pre/Post-Monsoon Biennial",
            total_records=45000,
            processed_file_path="data/processed/cgwb_groundwater_quality_cleaned.parquet",
            sha256=get_hash("data/processed/cgwb_groundwater_quality_cleaned.parquet"),
            feature_columns=["electrical_conductivity_us_cm", "total_dissolved_solids_mg_l", "fluoride_mg_l", "nitrate_mg_l", "arsenic_mg_l", "bis_potable"]
        ),
        DatasetRecord(
            dataset_id="hydrorivers_asia",
            name="HydroSHEDS HydroRIVERS Asia River Network Topology",
            source_agency="World Wildlife Fund (WWF) / HydroSHEDS",
            citation="Lehner & Grill (2013), Global river hydrography and network routing database, HydroSHEDS.",
            temporal_range="Static Reference",
            spatial_coverage="Pan-Asia (50 Indian City Basins Aligned)",
            spatial_resolution="Vector River Reach Polylines (15-arcsec DEM)",
            temporal_resolution="Static Topology",
            total_records=50,
            processed_file_path="data/processed/city_hydrorivers_profiles.json",
            sha256=get_hash("data/processed/city_hydrorivers_profiles.json"),
            feature_columns=["river_reaches_count", "primary_river_name", "max_strahler_order", "total_upstream_area_sqkm"]
        ),
        DatasetRecord(
            dataset_id="cwc_reservoir_storage",
            name="Central Water Commission 150 Major Reservoirs Daily Storage Monitoring",
            source_agency="Central Water Commission (CWC)",
            citation="CWC Live Storage Status of 150 Major Reservoirs in the Country.",
            temporal_range="2020-2024",
            spatial_coverage="150 Major Indian Reservoirs",
            spatial_resolution="Dam Reservoir Site",
            temporal_resolution="Daily / Weekly",
            total_records=12000,
            processed_file_path="data/processed/cwc_reservoir_storage.parquet",
            sha256=get_hash("data/processed/cwc_reservoir_storage.parquet"),
            feature_columns=["storage_bcm", "full_reservoir_level_m", "percentage_of_foc"]
        ),
        DatasetRecord(
            dataset_id="cpcb_ground_truth_cities",
            name="CPCB Continuous Ambient Air Quality Monitoring Station (CAAQMS) Network",
            source_agency="Central Pollution Control Board (CPCB)",
            citation="CPCB National Air Quality Monitoring Programme (NAMP) Real-time Continuous Telemetry.",
            temporal_range="2022-2024",
            spatial_coverage="50 Indian Cities Ground Stations",
            spatial_resolution="Ground Station Point Sensor",
            temporal_resolution="Daily Station-Aggregated",
            total_records=48500,
            processed_file_path="data/processed/cpcb_ground_truth_cities.parquet",
            sha256=get_hash("data/processed/cpcb_ground_truth_cities.parquet"),
            feature_columns=["pm2_5_cpcb", "pm10_cpcb", "no2_cpcb", "so2_cpcb", "cpcb_aqi"]
        ),
    ]

    _CANONICAL_DATASETS = {d.dataset_id: d for d in datasets}
    return _CANONICAL_DATASETS


def get_physical_thresholds_registry() -> Dict[str, PhysicalThresholdRecord]:
    """Load or compile the canonical physical and engineering threshold registry."""
    global _CANONICAL_THRESHOLDS
    if _CANONICAL_THRESHOLDS is not None:
        return _CANONICAL_THRESHOLDS

    thresholds: List[PhysicalThresholdRecord] = [
        PhysicalThresholdRecord(
            parameter_id="tropical_night_tmin",
            name="Tropical Night Transformer Cooling Failure Threshold",
            threshold_value=28.0,
            unit="degC",
            source_authority="Bureau of Indian Standards (IS 2026) / International Electrotechnical Commission (IEC 60076)",
            citation="IEC 60076-7: Loading guide for oil-immersed power transformers.",
            physical_mechanism="Ambient night temperature >=28°C prevents nocturnal convective cooling of distribution transformer radiator fins, causing dielectric oil winding breakdown under heavy AC peak load.",
            geographical_scope="All India Urban Areas",
            operational_action="Pre-position mobile diesel generators at hospitals; trigger DISCOM thermal inspection of distribution feeders."
        ),
        PhysicalThresholdRecord(
            parameter_id="anthesis_heat_shock_wheat",
            name="Wheat Anthesis Pollen Sterility Thermal Threshold",
            threshold_value=32.0,
            unit="degC",
            source_authority="Indian Council of Agricultural Research (ICAR) / CIMMYT",
            citation="Porter & Gawith (1999), Temperatures and the growth and development of wheat: a review, Eur. J. Agron.",
            physical_mechanism="Daytime maximum temperature >=32°C during flowering (anthesis) causes tapetal cell degradation in anthers, resulting in non-viable pollen and irreversible floret sterility.",
            geographical_scope="Indo-Gangetic Plains & Central India Wheat Belt",
            operational_action="Apply light sprinkler irrigation in evening to induce canopy evaporative cooling; monitor grain weight."
        ),
        PhysicalThresholdRecord(
            parameter_id="harvest_lodging_rainfall",
            name="Crop Maturity Field Lodging & Sprouting Rainfall Threshold",
            threshold_value=10.0,
            unit="mm/day",
            source_authority="ICAR Indian Agricultural Research Institute (IARI)",
            citation="IARI Agro-meteorological Advisory Services Protocol (2022).",
            physical_mechanism="Precipitation >=10 mm during grain physiological maturity softens stem culms, inducing mechanical plant lodging and pre-harvest seed sprouting in standing panicles.",
            geographical_scope="All Indian Agricultural Hinterlands",
            operational_action="Accelerate combine harvesting; protect harvested field heaps with waterproof tarpaulins."
        ),
        PhysicalThresholdRecord(
            parameter_id="wet_bulb_moderate_work_cap",
            name="Wet-Bulb Temperature Occupational Exertion Warning",
            threshold_value=30.0,
            unit="degC",
            source_authority="National Disaster Management Authority (NDMA) / ISO 7243",
            citation="NDMA National Guidelines for Preparation of Action Plan - Prevention and Management of Heat Wave (2019).",
            physical_mechanism="At wet-bulb >=30°C, human evaporative sweat efficiency drops significantly; continuous unshaded moderate labor rapidly leads to heat exhaustion and core hyperthermia.",
            geographical_scope="All India",
            operational_action="Implement mandatory 15-minute rest breaks per hour in shaded cooling zones with ORS hydration."
        ),
        PhysicalThresholdRecord(
            parameter_id="cooling_degree_day_base",
            name="Cooling Degree Day Engineering Reference Base",
            threshold_value=18.0,
            unit="degC",
            source_authority="ASHRAE Fundamentals / Central Electricity Authority (CEA)",
            citation="ASHRAE Standard 90.1: Energy Standard for Buildings Except Low-Rise Residential Buildings.",
            physical_mechanism="Mean daily ambient temperature above 18°C causes net heat gain across building envelopes, necessitating mechanical chiller and compressor refrigeration.",
            geographical_scope="All India Urban Areas",
            operational_action="Scale predicted municipal electricity load by cooling demand surge ratio above baseline."
        ),
    ]

    _CANONICAL_THRESHOLDS = {t.parameter_id: t for t in thresholds}
    return _CANONICAL_THRESHOLDS


def export_canonical_registries(output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Export all canonical registries to standardized JSON manifests on disk."""
    root = get_project_root()
    dest = output_dir or (root / "data" / "manifests")
    dest.mkdir(parents=True, exist_ok=True)

    models = get_model_registry()
    datasets = get_dataset_registry()
    thresholds = get_physical_thresholds_registry()

    p_models = dest / "canonical_model_registry.json"
    p_datasets = dest / "canonical_dataset_registry.json"
    p_thresholds = dest / "canonical_physical_thresholds.json"

    with open(p_models, "w", encoding="utf-8") as f:
        json.dump([m.model_dump() for m in models.values()], f, indent=2)

    with open(p_datasets, "w", encoding="utf-8") as f:
        json.dump([d.model_dump() for d in datasets.values()], f, indent=2)

    with open(p_thresholds, "w", encoding="utf-8") as f:
        json.dump([t.model_dump() for t in thresholds.values()], f, indent=2)

    logger.info(f"Exported canonical registries to {dest}: {len(models)} models, {len(datasets)} datasets, {len(thresholds)} thresholds.")
    return {
        "models": p_models,
        "datasets": p_datasets,
        "thresholds": p_thresholds,
    }
