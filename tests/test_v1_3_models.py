"""Automated unit and integration test suite for Clima-Core v1.3 advanced models and engines.
Tests ForecastFusion V1, BasinFlow V1, SmokePath V2, DroughtCascade V2,
HeatRisk V5 Soft-Gated MoE, and CompoundRisk V2.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
import os

from climacore.models.forecast_fusion.skill_matrix import (
    compute_continuous_metrics,
    compute_precipitation_metrics,
)
from climacore.models.forecast_fusion.ensemble_models import (
    WeightedForecastEnsemble,
    EMOSModel,
    QuantileLGBMEnsemble,
    TwoPartPrecipitationEnsemble,
    gaussian_crps,
)
from climacore.models.basinflow.unit_hydrograph import SCSUnitHydrograph
from climacore.models.basinflow.muskingum_cunge import MuskingumCungeRouter
from climacore.models.basinflow.flood_challenger import BasinFlowFloodRiskChallenger, ALL_BASINFLOW_FEATURES
from climacore.models.smoke_transport.engine import (
    SmokePathTransportEngine,
    compute_haversine_distance_km,
    compute_bearing_degrees,
    compute_seasonal_fire_radiative_power,
)
from climacore.models.drought.drought_cascade_v2 import DroughtCascadeV2, DROUGHT_HMM_FEATURES
from climacore.models.heat_risk.moe_v2 import SoftGatedMoEV5, CITY_TO_ZONE
from climacore.analytics.compound_risk_v2 import CompoundRiskV2, CompoundCopulaEngine


# =============================================================================
# 1. FORECASTFUSION V1 TESTS
# =============================================================================

def test_forecast_fusion_skill_metrics():
    obs = np.array([25.0, 28.0, 32.0, 35.0, 30.0], dtype=np.float64)
    pred = np.array([26.0, 27.5, 33.0, 34.0, 30.5], dtype=np.float64)
    
    m = compute_continuous_metrics(obs, pred)
    assert m["mae"] > 0.0
    assert m["rmse"] >= m["mae"]
    assert -5.0 < m["bias"] < 5.0
    assert 0.8 < m["corr"] <= 1.0

    # Precipitation metrics
    obs_rain = np.array([0.0, 0.0, 5.0, 15.0, 0.0], dtype=np.float64)
    pred_rain = np.array([0.0, 1.0, 4.5, 12.0, 0.0], dtype=np.float64)
    pm = compute_precipitation_metrics(obs_rain, pred_rain, threshold=0.1)
    assert 0.0 <= pm["pod"] <= 1.0
    assert 0.0 <= pm["far"] <= 1.0
    assert 0.0 <= pm["csi"] <= 1.0
    assert 0.0 <= pm["brier"] <= 1.0


def test_forecast_fusion_weighted_ensemble():
    X = np.array([[20.0, 21.0, 20.5], [30.0, 31.0, 29.5], [25.0, 24.5, 25.5]])
    y = np.array([20.2, 30.1, 25.2])
    
    ens = WeightedForecastEnsemble(fit_intercept=True)
    ens.fit(X, y, feature_names=["m1", "m2", "m3"])
    preds = ens.predict(X)
    
    assert len(preds) == 3
    assert np.allclose(np.sum(ens.weights_), 1.0, atol=1e-3)
    assert np.all(ens.weights_ >= -1e-5)


def test_forecast_fusion_emos_distribution():
    X = np.array([[20.0, 21.0], [30.0, 31.0], [25.0, 24.5], [40.0, 42.0]])
    y = np.array([20.5, 30.2, 25.0, 41.0])
    
    emos = EMOSModel()
    emos.fit(X, y)
    mu, sigma = emos.predict_distribution(X)
    
    assert len(mu) == 4
    assert len(sigma) == 4
    assert np.all(sigma > 0.0)
    crps = gaussian_crps(mu, sigma, y)
    assert np.all(crps >= 0.0)


def test_forecast_fusion_artifacts_exist():
    art_dir = "artifacts/models/v1_3/forecast_fusion"
    assert os.path.exists(os.path.join(art_dir, "tmax_model_b_emos.joblib"))
    assert os.path.exists(os.path.join(art_dir, "tmin_model_b_emos.joblib"))
    assert os.path.exists(os.path.join(art_dir, "precip_model_d_two_part.joblib"))


# =============================================================================
# 2. BASINFLOW V1 TESTS
# =============================================================================

def test_basinflow_scs_unit_hydrograph():
    uh = SCSUnitHydrograph(delta_t_hours=1.0)
    tc = uh.calculate_time_of_concentration(hydraulic_length_km=10.0, catchment_slope_percent=1.5, cn=75.0)
    assert 0.5 <= tc <= 24.0

    res = uh.generate_hydrograph(catchment_area_sqkm=100.0, runoff_depth_mm=30.0, hydraulic_length_km=10.0)
    assert res["peak_discharge_m3s"] > 0.0
    assert res["time_to_peak_hours"] > 0.0
    # Volume conservation within 0.1%
    assert abs(res["volume_conservation_ratio"] - 1.0) < 0.005


def test_basinflow_muskingum_cunge_routing():
    uh = SCSUnitHydrograph(delta_t_hours=1.0)
    uh_res = uh.generate_hydrograph(catchment_area_sqkm=100.0, runoff_depth_mm=25.0, hydraulic_length_km=10.0)
    
    router = MuskingumCungeRouter(manning_n=0.035)
    routed = router.route_hydrograph(
        inflow_m3s=uh_res["discharge_m3s"],
        time_hours=uh_res["time_hours"],
        reach_length_km=15.0,
        bed_slope_m_per_m=0.001,
        channel_width_meters=20.0,
    )
    assert routed["peak_outflow_m3s"] <= routed["peak_inflow_m3s"] * 1.01
    assert routed["peak_attenuation_ratio"] <= 1.01
    assert abs(routed["volume_conservation_ratio"] - 1.0) < 0.01
    assert abs(sum(routed["coefficients"].values()) - 1.0) < 1e-4


def test_basinflow_challenger_artifact_exists():
    path = "artifacts/models/v1_3/basinflow/flood_risk_basinflow_challenger.booster"
    assert os.path.exists(path)
    challenger = BasinFlowFloodRiskChallenger(path)
    assert len(challenger.feature_names) == 30
    assert "hydrograph_peak_discharge_m3s" in challenger.feature_names
    assert "routed_river_peak_m3s" in challenger.feature_names


# =============================================================================
# 3. SMOKEPATH V2 TESTS
# =============================================================================

def test_smokepath_physics_and_vectors():
    lats = np.array([28.61, 19.07])
    lons = np.array([77.20, 72.87])
    
    # Distance from NW burning centroid (30.5N, 75.5E)
    dists = compute_haversine_distance_km(lats, lons, 30.5, 75.5)
    assert dists[0] < dists[1]  # Delhi is closer than Mumbai
    
    # Bearing in [0, 360)
    bearings = compute_bearing_degrees(30.5, 75.5, lats, lons)
    assert np.all((bearings >= 0.0) & (bearings < 360.0))
    
    # Seasonal FRP peaks in autumn (DOY 305)
    frp_autumn = compute_seasonal_fire_radiative_power(np.array([305.0]))
    frp_summer = compute_seasonal_fire_radiative_power(np.array([200.0]))
    assert frp_autumn[0] > frp_summer[0] * 5.0


def test_smokepath_enrichment_and_artifact():
    df = pd.DataFrame({
        "wind_speed_10m_max": [4.0, 6.0],
        "wind_direction_10m_dominant": [315.0, 90.0],
        "latitude": [28.61, 26.85],
        "longitude": [77.20, 80.94],
        "day_of_year": [305, 305],
        "aod_mean": [0.65, 0.40],
        "temperature_2m_max": [30.0, 28.0],
        "temperature_2m_min": [18.0, 16.0],
        "relative_humidity_2m_mean": [60.0, 75.0],
    })
    enriched = SmokePathTransportEngine.enrich_smoke_features(df)
    assert "wind_u_component" in enriched.columns
    assert "wind_v_component" in enriched.columns
    assert "smoke_advection_index" in enriched.columns
    assert "aod_smoke_coupling" in enriched.columns
    assert "ventilation_index_proxy" in enriched.columns
    assert "stagnation_inversion_index" in enriched.columns

    art_path = "artifacts/models/v1_3/smoke_transport/air_quality_smokepath_challenger.booster"
    assert os.path.exists(art_path)


# =============================================================================
# 4. DROUGHTCASCADE V2 TESTS
# =============================================================================

def test_drought_cascade_v2_causal_filtering():
    model_path = "artifacts/models/v1_3/drought/drought_cascade_v2.joblib"
    assert os.path.exists(model_path)
    
    cascade = joblib.load(model_path)
    assert isinstance(cascade, DroughtCascadeV2)
    assert cascade.hmm is not None
    assert cascade.scaler is not None

    # Synthetic test sequence
    np.random.seed(42)
    X_test = np.random.randn(20, 4)
    probs = cascade.forward_filter_probabilities(X_test)
    
    assert probs.shape == (20, 4)
    # Check probability simplex: sum across states equals 1.0
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0)

    # Multi-step forward horizon
    p_forward_7d = cascade.predict_forward_horizon(probs[-1], horizon_days=7)
    assert np.isclose(np.sum(p_forward_7d), 1.0)

    # Dwell times
    dwells = cascade.get_dwell_time_statistics()
    assert len(dwells) == 4
    assert dwells["state_2"]["expected_dwell_days"] > 5.0


# =============================================================================
# 5. HEATRISK V5 SOFT-GATED MOE TESTS
# =============================================================================

def test_heatrisk_moe_v5_model():
    moe_path = "artifacts/models/v1_3/heatrisk/heatrisk_moe_v5.joblib"
    assert os.path.exists(moe_path)
    
    moe = joblib.load(moe_path)
    assert isinstance(moe, SoftGatedMoEV5)
    assert moe.global_backbone is not None
    assert len(moe.experts_) == 5
    assert moe.gating_network is not None
    assert moe.lambda_shrinkage == 0.60


# =============================================================================
# 6. COMPOUNDRISK V2 TESTS
# =============================================================================

def test_compound_risk_v2_cascading_network():
    cr = CompoundRiskV2()
    assert isinstance(cr.copula_engine, CompoundCopulaEngine)

    res = cr.evaluate_cascading_network(
        city="Mumbai",
        temp_max_c=36.0,
        temp_min_c=28.0,
        rainfall_mm=120.0,
        pm2_5_ug_m3=45.0,
        is_tropical_night=True,
        grid_deficit_prob=0.15,
        flood_hazard_prob=0.75,
        drought_regime_prob=0.05,
        smoke_advection_index=10.0,
        stagnation_index=15.0,
        hydrograph_peak_m3s=450.0,
        routed_river_peak_m3s=420.0,
        copula_p_and=0.08,
    )

    assert "compound_cascade_score" in res
    assert 0.0 <= res["compound_cascade_score"] <= 100.0
    assert len(res["uncertainty_corridor_95ci"]) == 2
    assert res["uncertainty_corridor_95ci"][0] <= res["compound_cascade_score"] <= res["uncertainty_corridor_95ci"][1]
    assert "pathway_c_deluge_runoff_lifelines" in res["pathways"]
    assert res["dominant_pathway"] == "pathway_c_deluge_runoff_lifelines"
    assert res["governing_policy_action"] in [
        "ACTIVATE_FULL_MULTI_AGENCY_CASCADE_PROTOCOL",
        "STANDBY_CROSS_SECTORAL_RESPONSE",
        "ROUTINE_MONITORING",
    ]
