"""Spatial Bioclimatic Climate Clustering Model.

Performs unsupervised clustering and probabilistic Gaussian Mixture Modeling (GMM)
on 44-year historical climate normals (1980-2024, 820k+ records) across all 50 Indian cities.

Segments cities into distinct bioclimatic hazard regimes with soft probabilistic memberships,
enabling spatial hazard choropleths, multi-hazard vulnerability mapping, and regional adaptation profiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root

logger = get_logger("climacore.models.bioclimatic_clustering.train")

# Domain knowledge: Archetypal hazard descriptions for bioclimatic regimes
BIOCLIMATIC_REGIMES = {
    "HUMID_TROPICAL_COASTAL": {
        "title": "Humid Tropical Coastal & Storm Surge",
        "description": "Low diurnal temperature variation, persistent high apparent temperature (WBGT stress), tropical cyclone and gale-force wind exposure.",
        "primary_risks": ["Tropical Cyclones", "Gale Gusts", "Extreme Apparent Heat / Humidity", "Coastal Flooding"],
    },
    "DECCAN_PLATEAU_HIGH_WIND": {
        "title": "Deccan Plateau & Semi-Arid High-Ventilation Corridor",
        "description": "Moderate elevation, balanced thermal regimes, strong seasonal wind shears with high annual gale frequencies, lower winter stagnation.",
        "primary_risks": ["Gale Wind Hazards", "Seasonal Water Scarcity", "Urban Flash Flooding", "Pesticide Spray Drift"],
    },
    "INDO_GANGETIC_THERMAL_INVERSION": {
        "title": "Indo-Gangetic Thermal Extremes & Smog Inversion Corridor",
        "description": "High continentality, severe peak summer heatwaves (>41°C), extreme winter atmospheric stagnation trapping hazardous aerosols and transboundary smoke.",
        "primary_risks": ["Severe Winter Smog", "Atmospheric Inversion", "Extreme Heatwaves", "Post-Harvest Smoke Drift"],
    },
    "SUB_HIMALAYAN_MONTANE": {
        "title": "Sub-Himalayan Montane / Valley Regime",
        "description": "Elevated altitude, low annual thermal maximums, cold waves, distinct orographic wind and persistent valley inversion dynamics.",
        "primary_risks": ["Cold Waves", "Valley Smog Trapping", "Frost Damage", "Orographic Squalls"],
    },
    "HUMID_MONSOON_DELUGE_COASTAL": {
        "title": "Humid Monsoon Deluge & Subtropical Delta Belt",
        "description": "Immense seasonal monsoon precipitation (>1500 mm), frequent extreme downpours, elevated apparent temperature, high flash flood vulnerability.",
        "primary_risks": ["Extreme Downpours", "Urban / Riverine Flooding", "Sticky Thermal Stress", "Infrastructure Disruption"],
    },
}


def compute_44y_bioclimatic_normals() -> pd.DataFrame:
    """Compute 44-year aggregated climatic normals for each city."""
    root = get_project_root()
    hist_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"

    logger.info(f"Loading 44-year historical dataset from {hist_path}...")
    df = pd.read_parquet(hist_path)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    total_years = df["year"].nunique()

    city_records = []
    for city, grp in df.groupby("city"):
        # Thermal metrics
        mean_tmax = float(grp["temperature_2m_max"].mean())
        p95_tmax = float(grp["temperature_2m_max"].quantile(0.95))
        mean_tmin = float(grp["temperature_2m_min"].mean())
        mean_dtr = float((grp["temperature_2m_max"] - grp["temperature_2m_min"]).mean())
        tropical_nights_rate = float((grp["temperature_2m_min"] >= 25.0).mean())
        heatwave_day_rate = float((grp["temperature_2m_max"] >= 40.0).mean())

        # Apparent heat & humidity stress
        mean_apparent_tmax = float(grp["apparent_temperature_max"].mean())
        thermal_stress_delta = float(mean_apparent_tmax - mean_tmax)

        # Precipitation metrics
        total_precip = float(grp["precipitation_sum"].sum())
        mean_annual_rainfall_mm = float(total_precip / total_years)
        monsoon_precip = float(grp[grp["month"].isin([6, 7, 8, 9])]["precipitation_sum"].sum())
        monsoon_concentration = float(monsoon_precip / max(1.0, total_precip))
        heavy_rain_days_per_year = float((grp["precipitation_sum"] >= 35.5).sum() / total_years)
        dry_days_rate = float((grp["precipitation_sum"] == 0.0).mean())

        # Wind & ventilation metrics
        mean_wind_speed = float(grp["wind_speed_10m_max"].mean())
        gale_gust_days_per_year = float((grp["wind_gusts_10m_max"] >= 50.0).sum() / total_years)
        stagnation_day_rate = float((grp["wind_speed_10m_max"] <= 8.0).mean())

        # Solar & evapotranspiration
        mean_solar = float(grp["shortwave_radiation_sum"].mean())
        total_et0 = float(grp["et0_fao_evapotranspiration"].sum())
        mean_annual_et0 = float(total_et0 / total_years)
        aridity_index = float(mean_annual_rainfall_mm / (mean_annual_et0 + 1.0))

        city_records.append({
            "city": city,
            "mean_tmax": round(mean_tmax, 2),
            "p95_tmax": round(p95_tmax, 2),
            "mean_tmin": round(mean_tmin, 2),
            "mean_dtr": round(mean_dtr, 2),
            "tropical_nights_rate": round(tropical_nights_rate, 4),
            "heatwave_day_rate": round(heatwave_day_rate, 4),
            "mean_apparent_tmax": round(mean_apparent_tmax, 2),
            "thermal_stress_delta": round(thermal_stress_delta, 2),
            "mean_annual_rainfall_mm": round(mean_annual_rainfall_mm, 1),
            "monsoon_concentration": round(monsoon_concentration, 4),
            "heavy_rain_days_per_year": round(heavy_rain_days_per_year, 2),
            "dry_days_rate": round(dry_days_rate, 4),
            "mean_wind_speed": round(mean_wind_speed, 2),
            "gale_gust_days_per_year": round(gale_gust_days_per_year, 2),
            "stagnation_day_rate": round(stagnation_day_rate, 4),
            "mean_solar": round(mean_solar, 2),
            "mean_annual_et0": round(mean_annual_et0, 1),
            "aridity_index": round(aridity_index, 3),
        })

    normals_df = pd.DataFrame(city_records).sort_values("city").reset_index(drop=True)
    logger.info(f"Computed 44-year climate normals for {len(normals_df)} cities.")
    return normals_df


def assign_regime_labels(cluster_centers: np.ndarray, feature_names: List[str], k: int) -> Dict[int, str]:
    """Map each numerical cluster ID to its closest physical bioclimatic regime."""
    centers_df = pd.DataFrame(cluster_centers, columns=feature_names)
    regime_map: Dict[int, str] = {}

    for idx, row in centers_df.iterrows():
        annual_rain = row.get("mean_annual_rainfall_mm", 0)
        thermal_delta = row.get("thermal_stress_delta", 0)
        heat_rate = row.get("heatwave_day_rate", 0)
        stagnation = row.get("stagnation_day_rate", 0)
        mean_tmax = row.get("mean_tmax", 0)
        gale_days = row.get("gale_gust_days_per_year", 0)

        if mean_tmax < 22.0:
            candidate = "SUB_HIMALAYAN_MONTANE"
        elif annual_rain > 1400.0:
            candidate = "HUMID_MONSOON_DELUGE_COASTAL"
        elif gale_days > 25.0:
            candidate = "DECCAN_PLATEAU_HIGH_WIND"
        elif thermal_delta > 3.0:
            candidate = "HUMID_TROPICAL_COASTAL"
        else:
            candidate = "INDO_GANGETIC_THERMAL_INVERSION"

        regime_map[int(idx)] = candidate

    return regime_map


def train_bioclimatic_clustering_model() -> Dict[str, Any]:
    """Train and evaluate the Spatial Bioclimatic Clustering Model."""
    root = get_project_root()
    art_path = root / "artifacts" / "models" / "bioclimatic_clustering" / "v0.1.0"
    rep_path = root / "reports" / "evaluation"
    art_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    normals_df = compute_44y_bioclimatic_normals()

    feature_cols = [
        "mean_tmax",
        "p95_tmax",
        "mean_dtr",
        "tropical_nights_rate",
        "heatwave_day_rate",
        "mean_apparent_tmax",
        "thermal_stress_delta",
        "mean_annual_rainfall_mm",
        "monsoon_concentration",
        "heavy_rain_days_per_year",
        "dry_days_rate",
        "mean_wind_speed",
        "gale_gust_days_per_year",
        "stagnation_day_rate",
        "mean_solar",
        "mean_annual_et0",
        "aridity_index",
    ]

    X = normals_df[feature_cols].to_numpy()

    # Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Search for optimal k in range [4, 7]
    best_k = 5
    best_silhouette = -1.0
    k_metrics = {}

    for k in [4, 5, 6, 7]:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))
        k_metrics[f"k_{k}"] = {"silhouette": round(sil, 4), "davies_bouldin": round(db, 4)}
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    logger.info(f"Evaluated cluster counts: {k_metrics}. Optimal k = {best_k} (Silhouette = {best_silhouette:.4f})")

    # Fit final KMeans and GMM
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=best_k, random_state=42, max_iter=200, n_init=10)
    gmm.fit(X_scaled)
    gmm_probs = gmm.predict_proba(X_scaled)
    gmm_labels = gmm.predict(X_scaled)

    # Compute unscaled cluster centers for interpretable regime assignment
    unscaled_centers = np.array([
        normals_df.loc[kmeans_labels == i, feature_cols].mean().to_numpy()
        for i in range(best_k)
    ])
    regime_labels = assign_regime_labels(unscaled_centers, feature_cols, best_k)

    # Generate Per-City Bioclimatic Profiles
    city_profiles = []
    for i, row in normals_df.iterrows():
        cid = int(kmeans_labels[i])
        regime_key = regime_labels[cid]
        regime_meta = BIOCLIMATIC_REGIMES.get(regime_key, BIOCLIMATIC_REGIMES["DECCAN_PLATEAU_HIGH_WIND"])

        profile = {
            "city": row["city"],
            "cluster_id": cid,
            "regime_key": regime_key,
            "regime_title": regime_meta["title"],
            "regime_description": regime_meta["description"],
            "primary_risks": regime_meta["primary_risks"],
            "soft_probabilities": {
                f"cluster_{c}": round(float(gmm_probs[i, c]), 4)
                for c in range(best_k)
            },
            "climate_normals": {col: row[col] for col in feature_cols},
        }
        city_profiles.append(profile)

    # Export Trained Models
    scaler_file = art_path / "scaler.joblib"
    kmeans_file = art_path / "kmeans.joblib"
    gmm_file = art_path / "gmm.joblib"

    joblib.dump(scaler, scaler_file)
    joblib.dump(kmeans, kmeans_file)
    joblib.dump(gmm, gmm_file)

    # Reload parity check
    reloaded_scaler = joblib.load(scaler_file)
    reloaded_kmeans = joblib.load(kmeans_file)
    reloaded_gmm = joblib.load(gmm_file)

    reloaded_X_scaled = reloaded_scaler.transform(X)
    reloaded_km_labels = reloaded_kmeans.predict(reloaded_X_scaled)
    reloaded_gmm_probs = reloaded_gmm.predict_proba(reloaded_X_scaled)

    km_parity_diff = float(np.max(np.abs(kmeans_labels - reloaded_km_labels)))
    gmm_parity_diff = float(np.max(np.abs(gmm_probs - reloaded_gmm_probs)))
    logger.info(f"Bioclimatic Reload Parity: KMeans diff = {km_parity_diff:.8f}, GMM diff = {gmm_parity_diff:.8f}")

    # Export City Profiles JSON
    profiles_json = art_path / "city_bioclimatic_profiles.json"
    with open(profiles_json, "w", encoding="utf-8") as f:
        json.dump(city_profiles, f, indent=2)

    # Cluster summary
    cluster_summary = {}
    for c in range(best_k):
        cities_in_c = [p["city"] for p in city_profiles if p["cluster_id"] == c]
        regime_k = regime_labels[c]
        cluster_summary[f"cluster_{c}"] = {
            "cluster_id": c,
            "regime_key": regime_k,
            "regime_title": BIOCLIMATIC_REGIMES[regime_k]["title"],
            "city_count": len(cities_in_c),
            "cities": cities_in_c,
            "centroid_normals": {
                col: round(float(unscaled_centers[c][idx]), 2)
                for idx, col in enumerate(feature_cols)
            },
        }

    clusters_json = art_path / "bioclimatic_clusters_summary.json"
    with open(clusters_json, "w", encoding="utf-8") as f:
        json.dump(cluster_summary, f, indent=2)

    # Provenance
    scaler_sha256 = compute_sha256(scaler_file)
    kmeans_sha256 = compute_sha256(kmeans_file)
    gmm_sha256 = compute_sha256(gmm_file)
    profiles_sha256 = compute_sha256(profiles_json)

    results = {
        "model_name": "climacore_bioclimatic_clustering",
        "cities_count": len(normals_df),
        "climate_normals_span": "1980 to 2024 (44 years)",
        "features_count": len(feature_cols),
        "optimal_k": best_k,
        "k_selection_metrics": k_metrics,
        "metrics": {
            "silhouette_score": round(best_silhouette, 4),
            "davies_bouldin_score": k_metrics[f"k_{best_k}"]["davies_bouldin"],
            "kmeans_reload_parity_diff": km_parity_diff,
            "gmm_reload_parity_diff": gmm_parity_diff,
        },
        "regime_assignments": {c: regime_labels[c] for c in range(best_k)},
        "artifacts": {
            "scaler": {"file": str(scaler_file.name), "sha256": scaler_sha256},
            "kmeans": {"file": str(kmeans_file.name), "sha256": kmeans_sha256},
            "gmm": {"file": str(gmm_file.name), "sha256": gmm_sha256},
            "city_profiles": {"file": str(profiles_json.name), "sha256": profiles_sha256},
        },
    }

    report_json = rep_path / "bioclimatic_clustering_evaluation_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Bioclimatic Clustering complete. Optimal k = {best_k}, Silhouette = {best_silhouette:.4f}")
    return results


if __name__ == "__main__":
    train_bioclimatic_clustering_model()
