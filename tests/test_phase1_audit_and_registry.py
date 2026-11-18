"""
Unit test suite for Phase 1: Scientific Foundation & Model Audit.
Covers:
  1. Canonical Model, Dataset, and Physical Threshold Registries
  2. The Validation Tri-Split Engine (Temporal, Spatial LOCZO, and Event-Safe)
  3. SciPy GEV sign convention (c = -xi) and extrapolation uncertainty flags
  4. CPCB 8-pollutant AQI completeness and missing data quality flags
  5. Flood inventory reporting uncertainty and negative-label semantics
  6. Grid stress physical heuristic vs. supervised ML probability separation
"""

import pytest
import pandas as pd
import numpy as np

from climacore.registry import (
    get_model_registry,
    get_dataset_registry,
    get_physical_thresholds_registry,
    export_canonical_registries,
)
from climacore.evaluation.splits import (
    chronological_train_val_test_split,
    leave_one_climate_zone_out_split,
    event_safe_train_val_test_split,
    assign_city_climate_zone,
    BIOCLIMATIC_ZONES,
)
from climacore.analytics.extreme_value import evaluate_return_period
from climacore.models.air_quality.engine import AirQualityEngine
from climacore.models.flood_risk.engine import FloodRiskEngine
from climacore.models.electricity_grid.engine import PowerGridEngine


def test_canonical_model_registry():
    """Verify Canonical Model Registry integrity, feature schemas, and artifact checksums."""
    registry = get_model_registry()
    assert len(registry) >= 6

    # Verify champions exist
    assert "flood_risk_v2" in registry
    assert "heat_risk_v3" in registry
    assert "power_grid_v2_reg" in registry
    assert "power_grid_v2_clf" in registry
    assert "agrisense_v2" in registry
    assert "air_quality_v2_pm25" in registry

    flood = registry["flood_risk_v2"]
    assert flood.operational_status == "CHAMPION"
    assert flood.task_type == "binary_classification"
    assert len(flood.features) == flood.features_count
    assert len(flood.known_limitations) >= 1
    assert len(flood.artifact_sha256) == 64

    agri = registry["agrisense_v2"]
    assert agri.metrics["top1_test_accuracy"] == 0.8249
    assert agri.metrics["top3_test_accuracy"] == 0.9762


def test_canonical_dataset_registry():
    """Verify Canonical Dataset Registry covers all foundational sources with record counts."""
    datasets = get_dataset_registry()
    assert len(datasets) >= 10

    assert "imd_temperature_74y" in datasets
    assert "imd_rainfall_46y" in datasets
    assert "era5_land_climate_master" in datasets
    assert "ifi_flood_inventory" in datasets
    assert "posoco_electricity_demand" in datasets

    imd_t = datasets["imd_temperature_74y"]
    assert imd_t.total_records >= 1_000_000
    assert "IMD" in imd_t.source_agency
    assert len(imd_t.feature_columns) >= 3


def test_canonical_physical_thresholds_registry():
    """Verify physical threshold registry contains formal standards and citations."""
    thresh = get_physical_thresholds_registry()
    assert len(thresh) >= 5

    assert "tropical_night_tmin" in thresh
    assert "anthesis_heat_shock_wheat" in thresh
    assert "wet_bulb_moderate_work_cap" in thresh

    tn = thresh["tropical_night_tmin"]
    assert tn.threshold_value == 28.0
    assert tn.unit == "degC"
    assert "IEC 60076" in tn.citation or "IS 2026" in tn.source_authority
    assert "transformer" in tn.physical_mechanism.lower()


def test_export_canonical_registries(tmp_path):
    """Verify export_canonical_registries dumps valid manifests to disk."""
    out_paths = export_canonical_registries(output_dir=tmp_path)
    assert out_paths["models"].exists()
    assert out_paths["datasets"].exists()
    assert out_paths["thresholds"].exists()


def test_validation_tri_split_temporal():
    """Verify chronological split with optional purge embargo."""
    dates = pd.date_range("2024-01-01", "2026-06-01", freq="D")
    df = pd.DataFrame({"date": dates, "value": np.arange(len(dates))})

    train, val, test = chronological_train_val_test_split(
        df,
        date_col="date",
        val_start="2025-06-01",
        test_start="2026-01-01",
        purge_days=14,  # 14-day embargo before splits
    )

    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    assert train["date"].max() <= pd.Timestamp("2025-05-18")
    assert val["date"].min() >= pd.Timestamp("2025-06-01")
    assert test["date"].min() >= pd.Timestamp("2026-01-01")


def test_validation_tri_split_spatial_loczo():
    """Verify Leave-One-Climate-Zone-Out (LOCZO) spatial split generator across 5 Indian zones."""
    records = []
    for zone, cities in BIOCLIMATIC_ZONES.items():
        for c in cities:
            records.append({"city": c, "obs": np.random.randn()})
    df = pd.DataFrame(records)

    folds = list(leave_one_climate_zone_out_split(df, city_col="city"))
    assert len(folds) == 5

    for held_out_zone, train_df, test_df in folds:
        assert held_out_zone in BIOCLIMATIC_ZONES
        assert len(test_df) > 0
        assert len(train_df) > 0
        # Zero spatial leakage between held-out zone and training set
        assert set(test_df["climate_zone"]) == {held_out_zone}
        assert held_out_zone not in set(train_df["climate_zone"])


def test_validation_tri_split_event_safe():
    """Verify event-safe split keeps complete disaster episodes atomically together."""
    events = [f"event_{i}" for i in range(20)]
    records = []
    for ev in events:
        for _ in range(5):  # 5 observations per event
            records.append({"event_id": ev, "metric": np.random.randn()})
    df = pd.DataFrame(records)

    train_df, val_df, test_df = event_safe_train_val_test_split(
        df,
        event_id_col="event_id",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_events = set(train_df["event_id"])
    val_events = set(val_df["event_id"])
    test_events = set(test_df["event_id"])

    # Disjoint event sets (strictly zero leakage)
    assert len(train_events.intersection(val_events)) == 0
    assert len(train_events.intersection(test_events)) == 0
    assert len(val_events.intersection(test_events)) == 0
    assert len(train_events) + len(val_events) + len(test_events) == 20


def test_city_climate_zone_assignment():
    """Verify deterministic assignment of all representative cities to their climate zone."""
    assert assign_city_climate_zone("Delhi") == "INDO_GANGETIC_PLAINS"
    assert assign_city_climate_zone("Lucknow") == "INDO_GANGETIC_PLAINS"
    assert assign_city_climate_zone("Mumbai") == "WESTERN_COASTAL"
    assert assign_city_climate_zone("Kochi") == "WESTERN_COASTAL"
    assert assign_city_climate_zone("Chennai") == "EASTERN_COASTAL"
    assert assign_city_climate_zone("Kolkata") == "EASTERN_COASTAL"
    assert assign_city_climate_zone("Bengaluru") == "DECCAN_PLATEAU_SEMI_ARID"
    assert assign_city_climate_zone("Hyderabad") == "DECCAN_PLATEAU_SEMI_ARID"
    assert assign_city_climate_zone("Srinagar") == "HIMALAYAN_HILL_NE"
    assert assign_city_climate_zone("Shimla") == "HIMALAYAN_HILL_NE"


def test_gev_scipy_sign_convention_and_extrapolation():
    """Verify GEV SciPy sign convention (c = -xi) and extrapolation uncertainty flags."""
    res = evaluate_return_period("Delhi", 47.5, hazard_type="temperature")
    assert "c_scipy" in res
    assert "xi_evt" in res
    assert res["c_scipy"] == pytest.approx(-res["xi_evt"], abs=1e-4)
    assert "tail_behavior" in res
    assert "extrapolation_uncertainty_flag" in res
    assert "ci_95_t25yr" in res
    assert len(res["ci_95_t25yr"]) == 2


def test_air_quality_cpcb_completeness_audit():
    """Verify AirQualityEngine flags non-compliant incomplete pollutant inputs."""
    engine = AirQualityEngine()

    # Incomplete: Only PM2.5 provided (1 pollutant, CPCB requires min 3)
    res_incomplete = engine.evaluate_air_quality(city="Delhi", pm2_5_ug_m3=85.0)
    assert res_incomplete["cpcb_compliance"]["is_cpcb_compliant"] is False
    assert len(res_incomplete["cpcb_compliance"]["data_quality_flags"]) >= 1

    # Compliant: PM2.5, PM10, and NO2 provided
    res_compliant = engine.evaluate_air_quality(
        city="Delhi",
        pm2_5_ug_m3=85.0,
        pm10_ug_m3=140.0,
        no2_ug_m3=45.0,
    )
    assert res_compliant["cpcb_compliance"]["is_cpcb_compliant"] is True
    assert res_compliant["cpcb_compliance"]["has_mandatory_pm"] is True
    assert res_compliant["cpcb_compliance"]["monitored_count"] >= 3


def test_flood_negative_label_uncertainty():
    """Verify FloodRiskEngine includes inventory reporting certainty context."""
    engine = FloodRiskEngine()
    res = engine.evaluate_flood_risk(city="Delhi", rainfall_mm=5.0)
    assert "inventory_reporting_context" in res
    assert "negative_label_uncertainty_note" in res["inventory_reporting_context"]
    assert "IFI" in res["inventory_reporting_context"]["source_catalog"]


def test_power_grid_physical_vs_ml_separation():
    """Verify PowerGridEngine distinctly separates engineering heuristic from supervised ML."""
    engine = PowerGridEngine()
    res = engine.evaluate_power_grid_risk(temp_max_c=42.0, temp_min_c=29.0)
    assert "physical_engineering_indicators" in res
    assert res["physical_engineering_indicators"]["derivation_type"] == "DETERMINISTIC_ENGINEERING_HEURISTIC"
    assert "calibrated_ml_predictions" in res
    assert res["calibrated_ml_predictions"]["derivation_type"] == "SUPERVISED_ML_INFERENCE"
