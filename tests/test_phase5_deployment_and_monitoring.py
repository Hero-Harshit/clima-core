"""
Unit & Integration Tests for Phase 5: Generalization, Deployment & Continuous Intelligence.
Validates:
  1. Deployment Gatekeeper 6-point promotion rules and failure reasons.
  2. Feature Ablation Study and marginal utility calculation.
  3. Population Stability Index (PSI) and Kolmogorov-Smirnov distribution drift detection.
  4. Conformal Calibration Health Monitor and exchangeability alerts.
  5. FastAPI Serverless Inference Service (/api/health, /api/predict/city, /api/predict/custom, /api/monitoring/*).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from climacore.evaluation.gatekeeper import (
    DeploymentGatekeeper,
    GatekeeperCriteria,
    FeatureAblationStudy,
)
from climacore.monitoring.drift import (
    calculate_psi,
    calculate_ks_drift,
    calculate_missing_rate,
    evaluate_feature_drift,
)
from climacore.monitoring.service import (
    CalibrationHealthMonitor,
    LatencyProfiler,
    ModelMonitoringService,
)
from climacore.service.app import app


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


# ----------------------------------------------------------------------
# 1. Deployment Gatekeeper & Promotion Gate Tests
# ----------------------------------------------------------------------
def test_gatekeeper_promotion_success():
    """Verify that a candidate meeting all 6 criteria is approved for promotion."""
    gatekeeper = DeploymentGatekeeper(
        criteria=GatekeeperCriteria(
            min_metric_delta=0.01,
            higher_is_better=True,
            min_conformal_coverage=0.88,
            max_spatial_drop_ratio=0.20,
        )
    )

    sample_input = np.array([[1.0, 2.0, 3.0]])
    dummy_predict = lambda x: np.sum(x, axis=1) * 2.0

    report = gatekeeper.evaluate_promotion(
        model_name="test_crop_model",
        champion_metric=0.80,
        challenger_metric=0.84,
        champion_version="v2.0.0",
        challenger_version="v3.0.0",
        conformal_coverage=0.91,
        loczo_zone_scores={
            "Indo-Gangetic": 0.85,
            "Deccan Plateau": 0.83,
            "Western Coastal": 0.86,
            "Eastern Coastal": 0.82,
            "Semi-Arid NW": 0.84,
        },
        predict_fn=dummy_predict,
        sample_input=sample_input,
        expected_schema_features=["f1", "f2", "f3"],
        candidate_features=["f1", "f2", "f3"],
    )

    assert report.overall_verdict == "PASS_PROMOTE"
    assert report.passed_all_gates is True
    assert len(report.failure_reasons) == 0
    assert len(report.provenance_hash) == 64
    assert report.gate_results["1_primary_task_metric"]["passed"] is True
    assert report.gate_results["2_calibration_uncertainty"]["passed"] is True
    assert report.gate_results["3_spatial_generalization"]["passed"] is True
    assert report.gate_results["4_latency_and_memory"]["passed"] is True
    assert report.gate_results["5_reproducibility"]["passed"] is True
    assert report.gate_results["6_schema_compatibility"]["passed"] is True


def test_gatekeeper_rejection_on_inferior_metric():
    """Verify that a candidate with lower primary performance is rejected."""
    gatekeeper = DeploymentGatekeeper(criteria=GatekeeperCriteria(higher_is_better=True))

    report = gatekeeper.evaluate_promotion(
        model_name="test_power_grid",
        champion_metric=0.8758,
        challenger_metric=0.7059,  # Challenger lower than champion
        champion_version="v2.0.0",
        challenger_version="v3.0.0",
        conformal_coverage=0.92,
    )

    assert report.overall_verdict == "FAIL_RETAIN_CHAMPION"
    assert report.passed_all_gates is False
    assert any("Primary metric failure" in r for r in report.failure_reasons)


def test_gatekeeper_rejection_on_spatial_zone_collapse():
    """Verify rejection when a model collapses in a specific bioclimatic zone."""
    gatekeeper = DeploymentGatekeeper(criteria=GatekeeperCriteria(max_spatial_drop_ratio=0.25))

    report = gatekeeper.evaluate_promotion(
        model_name="test_flood_model",
        champion_metric=0.75,
        challenger_metric=0.78,
        loczo_zone_scores={
            "Indo-Gangetic": 0.82,
            "Deccan Plateau": 0.80,
            "Western Coastal": 0.85,
            "Eastern Coastal": 0.83,
            "Semi-Arid NW": 0.40,  # Severe zone collapse (> 50% drop)
        },
    )

    assert report.overall_verdict == "FAIL_RETAIN_CHAMPION"
    assert any("Spatial generalization failure" in r for r in report.failure_reasons)


def test_gatekeeper_rejection_on_schema_mismatch():
    """Verify rejection when candidate feature schema misses expected production features."""
    gatekeeper = DeploymentGatekeeper(criteria=GatekeeperCriteria(require_exact_schema_match=True))

    report = gatekeeper.evaluate_promotion(
        model_name="test_air_quality",
        champion_metric=6.5,
        challenger_metric=6.0,
        expected_schema_features=["pm25", "t_max", "wind_speed", "boundary_layer_height"],
        candidate_features=["pm25", "t_max", "wind_speed"],  # missing boundary_layer_height
    )

    assert report.overall_verdict == "FAIL_RETAIN_CHAMPION"
    assert any("Schema compatibility failure" in r for r in report.failure_reasons)


# ----------------------------------------------------------------------
# 2. Feature Ablation Study Tests
# ----------------------------------------------------------------------
def test_feature_ablation_study():
    """Verify systematic quantification of marginal feature utility."""
    # Synthetic metric function: terrain adds high value, vegetation adds moderate, soil adds low
    def dummy_evaluator(feats: list[str]) -> float:
        score = 0.70
        if "soil_moisture_surface" in feats:
            score += 0.002  # marginal (<0.5%)
        if "drainage_canal_bottleneck_ratio" in feats:
            score += 0.060  # high (>5%)
        if "ndvi_anomaly" in feats:
            score += 0.020  # moderate (>2%)
        return score

    study = FeatureAblationStudy(
        evaluator_fn=dummy_evaluator,
        baseline_features=["rain_sum", "t_max"],
    )

    report = study.run_study(higher_is_better=True)

    experiments = report["experiments"]
    assert "baseline" in experiments
    assert "exp_a_soil" in experiments
    assert "exp_b_terrain" in experiments
    assert "exp_c_vegetation" in experiments
    assert "exp_d_all_combined" in experiments

    # Terrain should earn its place; soil alone should not (delta < 0.005)
    assert experiments["exp_b_terrain"]["earns_place"] is True
    assert experiments["exp_a_soil"]["earns_place"] is False
    assert report["best_feature_group"] in ["exp_d_all_combined", "exp_b_terrain"]


# ----------------------------------------------------------------------
# 3. Drift Detection (PSI, KS, Missingness) Tests
# ----------------------------------------------------------------------
def test_psi_and_ks_stable_distribution():
    """Verify that identical distributions yield low PSI and no KS drift."""
    np.random.seed(42)
    ref = np.random.normal(loc=30.0, scale=5.0, size=1000)
    prod = np.random.normal(loc=30.05, scale=4.95, size=1000)

    psi_val, psi_status = calculate_psi(ref, prod)
    assert psi_val < 0.10
    assert psi_status == "STABLE"

    ks_stat, ks_pval, ks_drift = calculate_ks_drift(ref, prod, alpha=0.01)
    assert ks_drift is False
    assert ks_pval > 0.01


def test_psi_and_ks_shifted_distribution():
    """Verify that a significantly shifted distribution triggers PSI and KS alerts."""
    np.random.seed(42)
    ref = np.random.normal(loc=25.0, scale=4.0, size=1000)
    prod = np.random.normal(loc=38.0, scale=6.0, size=1000)  # Severe thermal heatwave shift

    psi_val, psi_status = calculate_psi(ref, prod)
    assert psi_val >= 0.25
    assert psi_status == "SIGNIFICANT_DRIFT"

    ks_stat, ks_pval, ks_drift = calculate_ks_drift(ref, prod, alpha=0.05)
    assert ks_drift is True
    assert ks_pval < 1e-5


def test_missing_rate_calculation():
    """Verify accurate detection of telemetry dropouts."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    rate = calculate_missing_rate(arr)
    assert abs(rate - 0.40) < 1e-5


# ----------------------------------------------------------------------
# 4. Calibration Health Monitoring Tests
# ----------------------------------------------------------------------
def test_calibration_health_nominal():
    """Verify healthy status when empirical coverage matches target 90%."""
    monitor = CalibrationHealthMonitor(target_coverage=0.90, alert_threshold=0.85)

    # Record 50 observations where 46 are covered (92% coverage)
    for i in range(50):
        covered = i < 46
        y = 35.0 if covered else 50.0
        monitor.record_verification("temperature", y_true=y, interval_lower=30.0, interval_upper=40.0)

    health = monitor.get_health_status()
    assert health["status"] == "HEALTHY"
    assert health["alert_active"] is False
    assert health["empirical_coverage"] >= 0.90


def test_calibration_health_degradation_alert():
    """Verify calibration drift alert when empirical coverage falls below threshold."""
    monitor = CalibrationHealthMonitor(target_coverage=0.90, alert_threshold=0.85)

    # Record 50 observations where only 35 are covered (70% coverage < 85% alert threshold)
    for i in range(50):
        covered = i < 35
        y = 35.0 if covered else 55.0
        monitor.record_verification("temperature", y_true=y, interval_lower=30.0, interval_upper=40.0)

    health = monitor.get_health_status()
    assert health["status"] == "CALIBRATION_DRIFT_ALERT"
    assert health["alert_active"] is True
    assert "TRIGGER_CONFORMAL_RECALIBRATION" in health["recommendation"]


# ----------------------------------------------------------------------
# 5. FastAPI Serverless Inference Service Integration Tests
# ----------------------------------------------------------------------
def test_api_root_and_health(api_client: TestClient):
    """Test / and /api/health endpoints."""
    r_root = api_client.get("/")
    assert r_root.status_code == 200
    assert r_root.json()["status"] == "OPERATIONAL"

    r_health = api_client.get("/api/health")
    assert r_health.status_code == 200
    data = r_health.json()
    assert data["status"] == "HEALTHY"
    assert data["schema_version"] == "v1.0.0"
    assert data["registered_models_count"] >= 5
    assert len(data["flagship_champions"]) >= 5


def test_api_personas_and_cities_endpoints(api_client: TestClient):
    """Test /api/personas and /api/cities listing and filtering."""
    r_personas = api_client.get("/api/personas")
    assert r_personas.status_code == 200
    p_data = r_personas.json()
    assert p_data["total"] == 13
    assert len(p_data["personas"]) == 13
    assert any(p["id"] == "farmer" for p in p_data["personas"])
    assert any(p["id"] == "maritime" for p in p_data["personas"])

    r_cities = api_client.get("/api/cities")
    assert r_cities.status_code == 200
    c_data = r_cities.json()
    assert c_data["total"] >= 50
    assert any(c["city"] == "Mumbai" for c in c_data["cities"])

    # Query filtering
    r_search = api_client.get("/api/cities?q=chennai")
    assert r_search.status_code == 200
    s_data = r_search.json()
    assert s_data["total"] >= 1
    assert s_data["cities"][0]["city"] == "Chennai"


def test_api_predict_city(api_client: TestClient):
    """Test /api/predict/city returns valid unified prediction contract v1.0.0."""
    r = api_client.post("/api/predict/city", json={"city": "delhi"})
    assert r.status_code == 200
    res = r.json()
    assert res["schema_version"] == "v1.0.0"
    assert res["city"] == "Delhi"
    assert "daily_predictions" in res
    assert len(res["daily_predictions"]) > 0

    first_day = res["daily_predictions"][0]
    assert "conformal_bounds" in first_day
    assert "cascading_risk" in first_day
    assert "physical_units_manifest" in res


def test_api_predict_custom(api_client: TestClient):
    """Test /api/predict/custom evaluates on-demand environmental coordinates with fast latency."""
    payload = {
        "city_name": "Jaipur",
        "latitude": 26.9124,
        "longitude": 75.7873,
        "temperature_max": 43.5,
        "temperature_min": 29.5,
        "rainfall_mm": 0.0,
        "relative_humidity": 25.0,
        "wind_speed_kmh": 14.0,
        "pm25": 85.0,
        "soil_saturation_ratio": 0.20,
    }
    r = api_client.post("/api/predict/custom", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert res["schema_version"] == "v1.0.0"
    assert res["location"]["city_name"] == "Jaipur"
    assert "predictions" in res
    assert "heat_risk" in res["predictions"]
    assert "conformal_bounds" in res
    assert "cascading_risk" in res
    assert res["latency_ms"] < 1000.0  # Fast inference execution


def test_api_monitoring_and_verification(api_client: TestClient):
    """Test monitoring endpoints and ground-truth verification logging."""
    r_mon = api_client.get("/api/monitoring/health")
    assert r_mon.status_code == 200
    mon_data = r_mon.json()
    assert "overall_status" in mon_data
    assert "feature_drift_metrics" in mon_data

    # Log verification observation
    r_ver = api_client.post(
        "/api/monitoring/verify",
        json={
            "target_name": "temperature_max",
            "y_true": 38.5,
            "interval_lower": 35.0,
            "interval_upper": 42.0,
        },
    )
    assert r_ver.status_code == 200
    ver_data = r_ver.json()
    assert ver_data["recorded"] is True
    assert ver_data["covered"] is True


def test_api_gatekeeper_evaluate_endpoint(api_client: TestClient):
    """Test programmatic gatekeeper evaluation endpoint."""
    r = api_client.post(
        "/api/gatekeeper/evaluate",
        json={
            "model_name": "agrisense_v3",
            "champion_metric": 0.8249,
            "challenger_metric": 0.8500,
            "conformal_coverage": 0.91,
            "expected_features": ["n", "p", "k", "temp", "humidity", "ph", "rainfall"],
            "candidate_features": ["n", "p", "k", "temp", "humidity", "ph", "rainfall"],
        },
    )
    assert r.status_code == 200
    res = r.json()
    assert res["overall_verdict"] == "PASS_PROMOTE"
    assert res["passed_all_gates"] is True
