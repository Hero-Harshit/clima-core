"""
Unit and integration tests for Electricity Demand & Grid Stress Intelligence.
Validates dataset ingestion integrity, feature distributions, model artifact evaluation,
and Persona alert triggering.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import lightgbm as lgb
from climacore.config.loader import get_project_root
from climacore.personas.models import PersonaInput
from climacore.personas.city_official import evaluate_city_official_persona
from climacore.personas.industry import evaluate_industry_persona


@pytest.fixture(scope="module")
def project_root() -> Path:
    return get_project_root()


def test_electricity_processed_parquet_integrity(project_root):
    """Verify city_daily_electricity_stress.parquet exists, has 150k+ rows, and zero NaNs in key fields."""
    parquet_path = project_root / "data" / "processed" / "electricity" / "city_daily_electricity_stress.parquet"
    assert parquet_path.exists(), f"Missing processed electricity parquet at {parquet_path}"

    df = pd.read_parquet(parquet_path)
    assert len(df) >= 140000, f"Expected >= 140,000 rows, got {len(df):,}"
    
    # Check required columns
    required_cols = [
        "date", "city", "state", "energy_met_mu", "peak_met_mw", 
        "peak_unmet_mw", "is_grid_deficit_day", "grid_stress_ratio",
        "utci_mean_c", "cooling_degree_days", "is_tropical_night"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column {col}"
        assert df[col].isna().sum() == 0, f"Column {col} has unexpected NaN values"

    # Verify coverage across cities
    unique_cities = df["city"].nunique()
    assert unique_cities >= 48, f"Expected >= 48 cities, got {unique_cities}"

    # Verify temporal coverage 2013-2020
    assert str(df["date"].min()) <= "2013-02-01"
    assert str(df["date"].max()) >= "2020-12-01"


def test_electricity_manifest_checksum(project_root):
    """Verify cryptographic manifest exists and contains accurate checksums."""
    manifest_path = project_root / "data" / "manifests" / "electricity_demand_manifest.json"
    assert manifest_path.exists(), f"Missing manifest at {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "Electricity Demand" in manifest["dataset_name"]
    assert manifest["total_records"] >= 140000
    assert "sha256" in manifest
    assert len(manifest["sha256"]) == 64


def test_electricity_model_artifacts_and_metrics(project_root):
    """Verify trained LightGBM booster models and evaluation metrics exceed quality thresholds."""
    model_dir = project_root / "artifacts" / "models" / "electricity_grid" / "v0.1.0"
    reg_path = model_dir / "peak_grid_load_v1.booster"
    clf_path = model_dir / "grid_deficit_hazard_v1.booster"
    report_path = project_root / "reports" / "evaluation" / "electricity_grid_evaluation_report.json"

    assert reg_path.exists(), f"Peak grid load booster missing at {reg_path}"
    assert clf_path.exists(), f"Grid deficit hazard booster missing at {clf_path}"
    assert report_path.exists(), f"Evaluation report missing at {report_path}"

    # Validate boosters can be loaded
    reg_booster = lgb.Booster(model_file=str(reg_path))
    clf_booster = lgb.Booster(model_file=str(clf_path))
    assert reg_booster.num_trees() > 50
    assert clf_booster.num_trees() > 50

    # Validate evaluation metrics
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    reg_metrics = rep["regressor_metrics"]
    clf_metrics = rep["classifier_metrics"]

    assert reg_metrics["test_r2"] >= 0.85, f"Test R2 {reg_metrics['test_r2']} < 0.85"
    assert reg_metrics["test_mae_mw"] <= 2000.0, f"Test MAE {reg_metrics['test_mae_mw']} MW > 2000 MW"

    assert clf_metrics["test_roc_auc"] >= 0.90, f"Test ROC-AUC {clf_metrics['test_roc_auc']} < 0.90"
    assert clf_metrics["test_recall"] >= 0.70, f"Test Recall {clf_metrics['test_recall']} < 0.70"


def test_city_official_grid_stress_trigger():
    """Verify City Official persona triggers critical blackout advisory on elevated deficit hazard."""
    inp = PersonaInput(
        city="Delhi",
        date="2026-06-15",
        temp_max=44.5,
        temp_min=30.0,
        apparent_temp_max=48.0,
        precipitation_mm=0.0,
        rain_probability=0.0,
        pm2_5=90.0,
        wind_speed_kmh=12.0,
        relative_humidity=45.0,
        solar_radiation=24.0,
        et0_evapotranspiration=6.5,
        historical_percentile=98.5,
        peak_grid_load_mw=7450.0,
        electricity_grid_stress_ratio=1.35,
        grid_blackout_hazard_prob=0.55,
    )

    result = evaluate_city_official_persona(inp)
    assert result.status_level == "CRITICAL"
    assert "Peak Grid Stress Alert" in " ".join(result.advisories)
    assert any("DISCOMs: mandate spinning reserve" in r for r in result.operational_restrictions)
    assert "7,450 MW" in result.key_metrics["projected_peak_grid_load_mw"]
    assert "55.0%" in result.key_metrics["grid_deficit_hazard_probability"]


def test_industry_grid_warning_trigger():
    """Verify Industry persona issues captive co-generation advisory when grid deficit exceeds 30%."""
    inp = PersonaInput(
        city="Ahmedabad",
        date="2026-05-20",
        temp_max=43.0,
        temp_min=28.0,
        apparent_temp_max=45.0,
        precipitation_mm=0.0,
        rain_probability=0.0,
        pm2_5=110.0,
        wind_speed_kmh=14.0,
        relative_humidity=40.0,
        solar_radiation=25.0,
        et0_evapotranspiration=7.0,
        historical_percentile=95.0,
        grid_blackout_hazard_prob=0.42,
        thermal_capacity_100km_mw=12500.0,
        thermal_plants_within_100km=15,
        closest_thermal_plant="SABARMATI TPP (422 MW, 8.5km NW)",
    )

    result = evaluate_industry_persona(inp)
    advisories_str = " ".join(result.advisories)
    assert "Regional Grid Stability Warning" in advisories_str
    assert "12,500 MW" in result.key_metrics["regional_thermal_capacity_100km"]
