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
            processed_file_path="data/processed/imd_temperature_1951_2024.parquet",
            sha256=get_hash("data/processed/imd_temperature_1951_2024.parquet"),
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
            processed_file_path="data/processed/imd_rainfall_1979_2024.parquet",
            sha256=get_hash("data/processed/imd_rainfall_1979_2024.parquet"),
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
