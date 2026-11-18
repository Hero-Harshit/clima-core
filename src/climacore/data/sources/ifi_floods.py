"""
India Flood Inventory-Impacts (IFI) Ingestion & Hydrological Risk Feature Pipeline.
Aligns 1980-2023 disaster event records from IIT Delhi's India Flood Inventory (v3)
with 44-year daily IMD gridded precipitation, ERA5-Land multi-layer soil moisture,
and OpenStreetMap drainage/bridge infrastructure exposure across 50 Indian cities.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import unicodedata
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.ifi_floods")

# 50 target cities mapped to district names and aliases present in IFI v3
CITY_IFI_DISTRICTS_MAP: Dict[str, List[str]] = {
    "Agartala": ["west tripura", "tripura"],
    "Agra": ["agra"],
    "Ahmedabad": ["ahmedabad", "ahmadabad"],
    "Amritsar": ["amritsar"],
    "Bengaluru": ["bangalore", "bengaluru", "bangalore urban", "bangalore rural"],
    "Bhopal": ["bhopal"],
    "Bhubaneswar": ["khordha", "khurda", "bhubaneswar"],
    "Chandigarh": ["chandigarh"],
    "Chennai": ["chennai", "madras"],
    "Coimbatore": ["coimbatore"],
    "Dehradun": ["dehradun"],
    "Delhi": ["delhi", "new delhi"],
    "Gangtok": ["east sikkim", "sikkim", "gangtok"],
    "Guwahati": ["kamrup", "guwahati"],
    "Gwalior": ["gwalior"],
    "Hyderabad": ["hyderabad"],
    "Imphal": ["imphal", "manipur"],
    "Indore": ["indore"],
    "Jaipur": ["jaipur"],
    "Jammu": ["jammu"],
    "Jodhpur": ["jodhpur"],
    "Kanpur": ["kanpur"],
    "Kochi": ["ernakulam", "kochi", "cochin"],
    "Kolkata": ["kolkata", "calcutta", "howrah"],
    "Leh": ["leh", "ladakh"],
    "Lucknow": ["lucknow"],
    "Ludhiana": ["ludhiana"],
    "Madurai": ["madurai"],
    "Mangaluru": ["dakshina kannada", "mangalore"],
    "Mumbai": ["mumbai", "bombay", "thane"],
    "Mysuru": ["mysore", "mysuru"],
    "Nagpur": ["nagpur"],
    "Nashik": ["nashik", "nasik"],
    "Panaji": ["north goa", "goa", "panaji"],
    "Patna": ["patna"],
    "Port Blair": ["south andaman", "andaman", "port blair"],
    "Prayagraj": ["allahabad", "prayagraj"],
    "Puducherry": ["pondicherry", "puducherry"],
    "Pune": ["pune", "poona"],
    "Raipur": ["raipur"],
    "Ranchi": ["ranchi"],
    "Shillong": ["east khasi hills", "khasi hills", "shillong", "meghalaya"],
    "Shimla": ["shimla", "simla"],
    "Srinagar": ["srinagar"],
    "Surat": ["surat"],
    "Thiruvananthapuram": ["thiruvananthapuram", "trivandrum"],
    "Vadodara": ["vadodara", "baroda"],
    "Varanasi": ["varanasi", "banaras", "benares"],
    "Vijayawada": ["krishna", "vijayawada"],
    "Visakhapatnam": ["visakhapatnam", "vizag"],
}


def _normalize_city_name(c: Any) -> str:
    """Normalize city name removing diacritics and aliasing satellite towns."""
    norm = unicodedata.normalize("NFKD", str(c)).encode("ASCII", "ignore").decode("utf-8")
    mapping = {
        "Bangalore": "Bengaluru",
        "Vijayavada": "Vijayawada",
        "Vishakhapatnam": "Visakhapatnam",
        "Haora": "Kolkata",
        "Thane": "Mumbai",
        "Kalyan": "Mumbai",
        "Vasai-Virar": "Mumbai",
        "Pimpri-Chinchwad": "Pune",
    }
    return mapping.get(norm, norm)


def load_ifi_raw(raw_path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw India Flood Inventory CSV."""
    if raw_path is None:
        root = get_project_root()
        raw_path = root / "data" / "raw" / "floods_ifi" / "India_Flood_Inventory_v3.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"IFI raw dataset not found at {raw_path}")

    logger.info(f"Loading raw IFI dataset from {raw_path}...")
    df = pd.read_csv(raw_path)
    df["start"] = pd.to_datetime(df["Start Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    df["end"] = pd.to_datetime(df["End Date"], format="%d-%m-%Y %H:%M", errors="coerce")
    return df


def extract_city_flood_intervals(df_ifi: pd.DataFrame) -> pd.DataFrame:
    """Tag dates with flood occurrence across 50 cities."""
    df_valid = df_ifi.dropna(subset=["start", "end"]).copy()
    city_records: List[Dict[str, Any]] = []

    for city, aliases in CITY_IFI_DISTRICTS_MAP.items():
        pattern = "|".join([rf"\b{a}\b" for a in aliases])
        matches = df_valid[df_valid["Districts"].str.lower().str.contains(pattern, na=False, regex=True)]

        for _, row in matches.iterrows():
            s = max(row["start"].floor("D"), pd.Timestamp("1980-01-01"))
            e = min(row["end"].floor("D"), pd.Timestamp("2023-12-31"))
            if s <= e:
                # Cap prolonged river basin flooding to 14 days per burst
                if (e - s).days > 14:
                    e = s + pd.Timedelta(days=14)

                fatalities = float(row["Human fatality"]) if pd.notna(row["Human fatality"]) else 0.0
                displaced = float(row["Human Displaced"]) if pd.notna(row["Human Displaced"]) else 0.0
                cause = str(row["Main Cause"]) if pd.notna(row["Main Cause"]) else "flood"

                dates = pd.date_range(s, e, freq="D")
                for d in dates:
                    city_records.append({
                        "city": city,
                        "date": d,
                        "flood_disaster_event": 1,
                        "fatalities": fatalities,
                        "displaced": displaced,
                        "flood_cause": cause,
                    })

    if not city_records:
        return pd.DataFrame(columns=["city", "date", "flood_disaster_event", "fatalities", "displaced", "flood_cause"])

    res = pd.DataFrame(city_records)
    # Deduplicate in case multiple reports cover the same city on the same day
    res = res.groupby(["city", "date"]).agg({
        "flood_disaster_event": "max",
        "fatalities": "max",
        "displaced": "max",
        "flood_cause": "first",
    }).reset_index()

    return res


def compute_api_series(rain_series: np.ndarray, decay: float = 0.85) -> np.ndarray:
    """Compute 14-day Antecedent Precipitation Index (API)."""
    api = np.zeros_like(rain_series, dtype=np.float32)
    for i in range(1, len(rain_series)):
        api[i] = rain_series[i] + decay * api[i - 1]
    return api


def build_aligned_flood_dataset(
    output_parquet: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the consolidated 44-year flood disaster training dataset across 50 cities."""
    root = get_project_root()
    if output_parquet is None:
        output_parquet = root / "data" / "processed" / "floods" / "city_flood_history_aligned.parquet"
    if manifest_path is None:
        manifest_path = root / "data" / "manifests" / "ifi_flood_manifest.json"

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Ingest IFI Ground Truth
    df_ifi = load_ifi_raw()
    city_floods = extract_city_flood_intervals(df_ifi)
    city_floods["date"] = pd.to_datetime(city_floods["date"])
    logger.info(f"Tagged {len(city_floods):,} positive flood-days across 50 cities.")

    # 2. Canonical spine: IMD Rainfall across 50 cities (1980-2023)
    imd_path = root / "data" / "processed" / "imd_rainfall_50_cities.parquet"
    if not imd_path.exists():
        raise FileNotFoundError(f"IMD rainfall dataset not found at {imd_path}")

    logger.info(f"Loading IMD rainfall spine from {imd_path}...")
    df_spine = pd.read_parquet(imd_path)
    df_spine["date"] = pd.to_datetime(df_spine["date"])
    df_spine = df_spine[(df_spine["date"] >= "1980-01-01") & (df_spine["date"] <= "2023-12-31")].copy()
    df_spine["rainfall_mm"] = df_spine["imd_rainfall_mm"].fillna(0.0).astype(np.float32)

    # 3. Weather Historical (atmospheric variables)
    weather_path = root / "data" / "processed" / "weather_historical_44y_train_ready.parquet"
    if weather_path.exists():
        logger.info(f"Merging atmospheric variables from {weather_path}...")
        df_weather = pd.read_parquet(weather_path)
        df_weather["date"] = pd.to_datetime(df_weather["date"])
        df_weather["city"] = df_weather["city"].apply(_normalize_city_name)
        keep_w = ["city", "date", "precipitation_hours", "et0_fao_evapotranspiration", "wind_gusts_10m_max"]
        df_weather_sub = df_weather[keep_w].drop_duplicates(subset=["city", "date"])
        df_spine = df_spine.merge(df_weather_sub, on=["city", "date"], how="left")

    if "precipitation_hours" not in df_spine.columns:
        df_spine["precipitation_hours"] = 0.0
    else:
        df_spine["precipitation_hours"] = df_spine["precipitation_hours"].fillna(0.0).astype(np.float32)

    if "et0_fao_evapotranspiration" not in df_spine.columns:
        df_spine["et0_fao_evapotranspiration"] = 3.5
    else:
        df_spine["et0_fao_evapotranspiration"] = df_spine["et0_fao_evapotranspiration"].fillna(3.5).astype(np.float32)

    if "wind_gusts_10m_max" not in df_spine.columns:
        df_spine["wind_gusts_10m_max"] = 20.0
    else:
        df_spine["wind_gusts_10m_max"] = df_spine["wind_gusts_10m_max"].fillna(20.0).astype(np.float32)

    # 4. ERA5 Soil Moisture
    soil_path = root / "data" / "processed" / "era5_soil_moisture_50_cities.parquet"
    if soil_path.exists():
        logger.info(f"Merging soil moisture covariates from {soil_path}...")
        df_soil = pd.read_parquet(soil_path)
        df_soil["date"] = pd.to_datetime(df_soil["date"])
        df_soil["city"] = df_soil["city"].apply(_normalize_city_name)
        soil_cols = [
            "city", "date", "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
            "topsoil_moisture_anomaly", "rootzone_moisture_anomaly", "soil_moisture_deficit_index"
        ]
        available_cols = [c for c in soil_cols if c in df_soil.columns]
        df_soil_sub = df_soil[available_cols].drop_duplicates(subset=["city", "date"])
        df_spine = df_spine.merge(df_soil_sub, on=["city", "date"], how="left")

    soil_feat_names = [
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "topsoil_moisture_anomaly", "rootzone_moisture_anomaly", "soil_moisture_deficit_index"
    ]
    for s_col in soil_feat_names:
        if s_col in df_spine.columns:
            mean_val = float(df_spine[s_col].dropna().mean()) if len(df_spine[s_col].dropna()) > 0 else 0.0
            df_spine[s_col] = df_spine[s_col].fillna(mean_val).astype(np.float32)
        else:
            df_spine[s_col] = 0.0

    # 5. Infrastructure Exposure (OSM)
    infra_path = root / "data" / "processed" / "infrastructure" / "city_infrastructure_exposure.parquet"
    if infra_path.exists():
        logger.info(f"Merging critical infrastructure metrics from {infra_path}...")
        df_infra = pd.read_parquet(infra_path)
        infra_keep = [
            "city", "drainage_canals_15km", "bridges_15km", "transit_stations_15km",
            "arterial_roads_15km", "industrial_zones_15km"
        ]
        available_infra = [c for c in infra_keep if c in df_infra.columns]
        df_spine = df_spine.merge(df_infra[available_infra], on="city", how="left")

    for i_col in ["drainage_canals_15km", "bridges_15km", "transit_stations_15km", "arterial_roads_15km", "industrial_zones_15km"]:
        if i_col in df_spine.columns:
            df_spine[i_col] = df_spine[i_col].fillna(0).astype(np.float32)

    # 6. Population Exposure (WorldPop)
    pop_path = root / "data" / "processed" / "population" / "city_population_exposure.parquet"
    if pop_path.exists():
        logger.info(f"Merging population exposure metrics from {pop_path}...")
        df_pop = pd.read_parquet(pop_path)
        pop_keep = ["city", "population_15km", "dense_cluster_pop_15km"]
        available_pop = [c for c in pop_keep if c in df_pop.columns]
        df_spine = df_spine.merge(df_pop[available_pop], on="city", how="left")

    for p_col in ["population_15km", "dense_cluster_pop_15km"]:
        if p_col in df_spine.columns:
            df_spine[p_col] = df_spine[p_col].fillna(0).astype(np.float32)

    # 7. Hydrological Rolling Features per city
    logger.info("Computing antecedent rainfall and hydrological rolling features...")
    df_spine = df_spine.sort_values(["city", "date"]).reset_index(drop=True)

    city_dfs = []
    for city, grp in df_spine.groupby("city", sort=False):
        g = grp.copy()
        rain = g["rainfall_mm"].values
        g["rain_rolling_3d"] = g["rainfall_mm"].rolling(3, min_periods=1).sum().astype(np.float32)
        g["rain_rolling_7d"] = g["rainfall_mm"].rolling(7, min_periods=1).sum().astype(np.float32)
        g["rain_rolling_14d"] = g["rainfall_mm"].rolling(14, min_periods=1).sum().astype(np.float32)
        g["rain_rolling_30d"] = g["rainfall_mm"].rolling(30, min_periods=1).sum().astype(np.float32)
        g["rain_max_3d"] = g["rainfall_mm"].rolling(3, min_periods=1).max().astype(np.float32)
        g["api_14d"] = compute_api_series(rain, decay=0.85).astype(np.float32)
        city_dfs.append(g)

    df_features = pd.concat(city_dfs, ignore_index=True)

    # 8. Merge IFI ground truth flood labels
    logger.info("Merging IFI ground truth flood labels...")
    df_aligned = df_features.merge(city_floods, on=["city", "date"], how="left")
    df_aligned["flood_disaster_event"] = df_aligned["flood_disaster_event"].fillna(0).astype(int)
    df_aligned["fatalities"] = df_aligned["fatalities"].fillna(0.0).astype(np.float32)
    df_aligned["displaced"] = df_aligned["displaced"].fillna(0.0).astype(np.float32)
    df_aligned["flood_cause"] = df_aligned["flood_cause"].fillna("none")

    # Save aligned dataset
    logger.info(f"Saving aligned flood dataset to {output_parquet}...")
    df_aligned.to_parquet(output_parquet, index=False)

    # Write manifest
    hasher = hashlib.sha256()
    with open(output_parquet, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    sha256_hash = hasher.hexdigest()

    manifest = {
        "dataset_name": "IFI Flood Disaster & Hydrological Exposure Aligned (1980-2023)",
        "source": "IIT Delhi India Flood Inventory (IFI) v3, IMD Gridded Rainfall, ERA5-Land, OSM Infrastructure",
        "output_file": str(output_parquet),
        "sha256": sha256_hash,
        "total_records": len(df_aligned),
        "total_cities": int(df_aligned["city"].nunique()),
        "positive_flood_days": int(df_aligned["flood_disaster_event"].sum()),
        "positive_rate_pct": float(np.round((df_aligned["flood_disaster_event"].sum() / len(df_aligned)) * 100, 3)),
        "date_range": [str(df_aligned["date"].min()), str(df_aligned["date"].max())],
        "feature_columns": [
            "rainfall_mm", "rain_rolling_3d", "rain_rolling_7d", "rain_rolling_14d",
            "rain_rolling_30d", "rain_max_3d", "api_14d", "precipitation_hours",
            "et0_fao_evapotranspiration", "wind_gusts_10m_max",
            "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
            "topsoil_moisture_anomaly", "rootzone_moisture_anomaly", "soil_moisture_deficit_index",
            "drainage_canals_15km", "bridges_15km", "transit_stations_15km",
            "population_15km", "dense_cluster_pop_15km"
        ],
        "target_column": "flood_disaster_event",
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"IFI manifest written to {manifest_path}")
    return df_aligned


if __name__ == "__main__":
    df = build_aligned_flood_dataset()
    print(f"Built aligned flood dataset with {len(df):,} rows and {df['flood_disaster_event'].sum():,} flood-days across {df['city'].nunique()} cities.")
