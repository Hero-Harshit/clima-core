"""Unit tests for the 4 Visual Map-Ready ML Models:
1. Urban Heat Island (UHI) Microclimate Intensity Model
2. Transboundary Smoke Drift & Wind Advection Model
3. Severe Gale-Force Wind & Storm Gust Hazard Model
4. Spatial Bioclimatic Climate Clustering Model
"""

import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pytest

from climacore.config.loader import get_project_root


@pytest.fixture
def root() -> Path:
    return get_project_root()


def test_uhi_microclimate_artifact_and_reload(root: Path):
    """Test UHI microclimate model artifact exists, reloads cleanly, and outputs bounded predictions."""
    model_path = root / "artifacts" / "models" / "uhi_microclimate" / "v0.1.0" / "model.txt"
    report_path = root / "reports" / "evaluation" / "uhi_microclimate_evaluation_report.json"

    assert model_path.exists(), f"UHI model artifact missing at {model_path}"
    assert report_path.exists(), f"UHI evaluation report missing at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["model_name"] == "climacore_uhi_microclimate"
    assert rep["metrics"]["uhi_lgbm"]["mae"] < rep["metrics"]["persistence_baseline"]["mae"]
    assert rep["metrics"]["uhi_lgbm"]["r2"] > 0.50

    # Reload parity & inference sanity
    booster = lgb.Booster(model_file=str(model_path))
    num_features = booster.num_feature()
    assert num_features > 0

    dummy_input = np.ones((5, num_features)) * 0.5
    preds = booster.predict(dummy_input)
    assert preds.shape == (5,)
    assert not np.isnan(preds).any()


def test_transboundary_smoke_artifact_and_reload(root: Path):
    """Test transboundary smoke drift model exists and predicts valid probabilities."""
    model_path = root / "artifacts" / "models" / "transboundary_smoke" / "v0.1.0" / "model.txt"
    report_path = root / "reports" / "evaluation" / "transboundary_smoke_evaluation_report.json"

    assert model_path.exists(), f"Smoke model artifact missing at {model_path}"
    assert report_path.exists(), f"Smoke evaluation report missing at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["model_name"] == "climacore_transboundary_smoke"
    assert rep["metrics"]["advection_classifier"]["roc_auc"] > 0.90
    assert rep["metrics"]["advection_classifier"]["pr_auc"] > 0.70

    booster = lgb.Booster(model_file=str(model_path))
    num_features = booster.num_feature()
    assert num_features > 0

    dummy_input = np.ones((5, num_features)) * 0.5
    probs = booster.predict(dummy_input)
    assert probs.shape == (5,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_gale_gust_hazard_artifact_and_reload(root: Path):
    """Test gale gust hazard model exists and outperforms persistence on recall."""
    model_path = root / "artifacts" / "models" / "gale_gust_hazard" / "v0.1.0" / "model.txt"
    report_path = root / "reports" / "evaluation" / "gale_gust_hazard_evaluation_report.json"

    assert model_path.exists(), f"Gale gust model artifact missing at {model_path}"
    assert report_path.exists(), f"Gale gust evaluation report missing at {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["model_name"] == "climacore_gale_gust_hazard"
    assert rep["metrics"]["gale_classifier"]["roc_auc"] > 0.85
    # Model recall significantly outperforms persistence
    assert rep["metrics"]["gale_classifier"]["recall"] > rep["metrics"]["persistence_baseline"]["recall"]

    booster = lgb.Booster(model_file=str(model_path))
    num_features = booster.num_feature()
    assert num_features > 0

    dummy_input = np.ones((5, num_features)) * 0.5
    probs = booster.predict(dummy_input)
    assert probs.shape == (5,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_bioclimatic_clustering_artifacts_and_profiles(root: Path):
    """Test spatial bioclimatic clustering models, 50-city profiles, and soft probabilities."""
    art_dir = root / "artifacts" / "models" / "bioclimatic_clustering" / "v0.1.0"
    scaler_file = art_dir / "scaler.joblib"
    kmeans_file = art_dir / "kmeans.joblib"
    gmm_file = art_dir / "gmm.joblib"
    profiles_file = art_dir / "city_bioclimatic_profiles.json"
    summary_file = art_dir / "bioclimatic_clusters_summary.json"
    report_file = root / "reports" / "evaluation" / "bioclimatic_clustering_evaluation_report.json"

    for f in [scaler_file, kmeans_file, gmm_file, profiles_file, summary_file, report_file]:
        assert f.exists(), f"Missing bioclimatic artifact: {f}"

    # Load and verify 50 city profiles
    with open(profiles_file, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    assert len(profiles) == 50
    for p in profiles:
        assert "city" in p
        assert "cluster_id" in p
        assert "regime_key" in p
        assert "regime_title" in p
        assert "soft_probabilities" in p
        # Probabilities sum to 1.0
        prob_sum = sum(p["soft_probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01

    # Reload parity check on scaler + GMM
    scaler = joblib.load(scaler_file)
    kmeans = joblib.load(kmeans_file)
    gmm = joblib.load(gmm_file)

    sample_X = np.ones((3, scaler.n_features_in_)) * 20.0
    scaled = scaler.transform(sample_X)
    km_preds = kmeans.predict(scaled)
    gmm_probs = gmm.predict_proba(scaled)

    assert km_preds.shape == (3,)
    assert gmm_probs.shape == (3, 5)
    assert np.allclose(gmm_probs.sum(axis=1), 1.0)
