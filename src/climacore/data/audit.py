"""
Data auditing module for Clima-Core.
Performs thorough schema, missingness, duplicate, temporal, and physical checks
on raw datasets and produces structured audit reports and provenance manifests.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from climacore.common.contracts import DatasetManifest, FileChecksum
from climacore.common.exceptions import DataValidationError
from climacore.common.logging import get_logger
from climacore.common.provenance import get_file_metadata
from climacore.config.loader import get_project_root, get_data_sources_config

logger = get_logger("climacore.data.audit")


def audit_india_50_cities_dataset(
    raw_dir: Path | str,
    output_dir: Path | str,
    manifest_dir: Path | str,
) -> Dict[str, Any]:
    """Audit weather_daily.csv, air_quality_daily.csv, and cities.csv."""
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    man_path = Path(manifest_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    man_path.mkdir(parents=True, exist_ok=True)

    weather_file = raw_path / "weather_daily.csv"
    aq_file = raw_path / "air_quality_daily.csv"
    cities_file = raw_path / "cities.csv"

    for file_p in [weather_file, aq_file, cities_file]:
        if not file_p.exists():
            raise FileNotFoundError(f"Required raw dataset file missing: {file_p}")

    logger.info("Starting audit of India 50 cities dataset...")

    # 1. Inspect Files and Checksums
    weather_meta = get_file_metadata(weather_file)
    aq_meta = get_file_metadata(aq_file)
    cities_meta = get_file_metadata(cities_file)

    files_checksums = [
        FileChecksum(**weather_meta),
        FileChecksum(**aq_meta),
        FileChecksum(**cities_meta),
    ]

    # 2. Inspect Cities Metadata
    logger.info("Auditing cities.csv...")
    df_cities = pd.read_csv(cities_file)
    cities_cols = list(df_cities.columns)
    cities_count = len(df_cities)
    cities_unique = df_cities["city"].nunique() if "city" in df_cities.columns else 0

    cities_profile = {
        "rows": cities_count,
        "unique_cities": cities_unique,
        "columns": cities_cols,
        "has_nulls": bool(df_cities.isnull().any().any()),
        "null_counts": df_cities.isnull().sum().to_dict(),
        "lat_range": [float(df_cities["latitude"].min()), float(df_cities["latitude"].max())] if "latitude" in df_cities else None,
        "lon_range": [float(df_cities["longitude"].min()), float(df_cities["longitude"].max())] if "longitude" in df_cities else None,
    }

    # 3. Inspect Weather Daily
    logger.info("Auditing weather_daily.csv...")
    df_weather = pd.read_csv(weather_file)
    weather_rows, weather_cols_count = df_weather.shape
    df_weather["date"] = pd.to_datetime(df_weather["date"])
    weather_start = df_weather["date"].min().strftime("%Y-%m-%d")
    weather_end = df_weather["date"].max().strftime("%Y-%m-%d")
    weather_cities = df_weather["city"].nunique()

    # Duplicate check on (city, date)
    weather_dups = int(df_weather.duplicated(subset=["city", "date"]).sum())

    # Missingness profiling
    weather_missing = df_weather.isnull().sum().to_dict()
    weather_missing_pct = {k: float(v / weather_rows * 100) for k, v in weather_missing.items()}

    # Physical plausibility checks
    temp_anomalies = int(((df_weather["temperature_2m_max"] > 55) | (df_weather["temperature_2m_min"] < -15)).sum())
    rain_negative = int((df_weather["precipitation_sum"] < 0).sum()) if "precipitation_sum" in df_weather else 0
    rh_out_of_bounds = int(((df_weather["relative_humidity_2m_mean"] < 0) | (df_weather["relative_humidity_2m_mean"] > 100)).sum()) if "relative_humidity_2m_mean" in df_weather else 0

    weather_profile = {
        "rows": weather_rows,
        "columns": list(df_weather.columns),
        "start_date": weather_start,
        "end_date": weather_end,
        "unique_cities": weather_cities,
        "duplicates_city_date": weather_dups,
        "missing_counts": weather_missing,
        "missing_pct": weather_missing_pct,
        "physical_violations": {
            "temperature_out_of_range": temp_anomalies,
            "negative_precipitation": rain_negative,
            "relative_humidity_out_of_0_100": rh_out_of_bounds,
        },
        "temperature_2m_max_summary": {
            "min": float(df_weather["temperature_2m_max"].min()),
            "mean": float(df_weather["temperature_2m_max"].mean()),
            "max": float(df_weather["temperature_2m_max"].max()),
        },
        "precipitation_sum_summary": {
            "min": float(df_weather["precipitation_sum"].min()),
            "mean": float(df_weather["precipitation_sum"].mean()),
            "max": float(df_weather["precipitation_sum"].max()),
        }
    }

    # 4. Inspect Air Quality Daily
    logger.info("Auditing air_quality_daily.csv...")
    df_aq = pd.read_csv(aq_file)
    aq_rows, aq_cols_count = df_aq.shape
    df_aq["date"] = pd.to_datetime(df_aq["date"])
    aq_start = df_aq["date"].min().strftime("%Y-%m-%d")
    aq_end = df_aq["date"].max().strftime("%Y-%m-%d")
    aq_cities = df_aq["city"].nunique()

    aq_dups = int(df_aq.duplicated(subset=["city", "date"]).sum())
    aq_missing = df_aq.isnull().sum().to_dict()
    aq_missing_pct = {k: float(v / aq_rows * 100) for k, v in aq_missing.items()}

    # Check negative pollutant values
    pm25_neg = int((df_aq["pm2_5_mean"] < 0).sum()) if "pm2_5_mean" in df_aq else 0
    pm10_neg = int((df_aq["pm10_mean"] < 0).sum()) if "pm10_mean" in df_aq else 0

    aq_profile = {
        "rows": aq_rows,
        "columns": list(df_aq.columns),
        "start_date": aq_start,
        "end_date": aq_end,
        "unique_cities": aq_cities,
        "duplicates_city_date": aq_dups,
        "missing_counts": aq_missing,
        "missing_pct": aq_missing_pct,
        "physical_violations": {
            "negative_pm2_5_mean": pm25_neg,
            "negative_pm10_mean": pm10_neg,
        },
        "pm2_5_mean_summary": {
            "min": float(df_aq["pm2_5_mean"].dropna().min()),
            "mean": float(df_aq["pm2_5_mean"].dropna().mean()),
            "max": float(df_aq["pm2_5_mean"].dropna().max()),
        } if "pm2_5_mean" in df_aq else {},
        "pm10_mean_summary": {
            "min": float(df_aq["pm10_mean"].dropna().min()),
            "mean": float(df_aq["pm10_mean"].dropna().mean()),
            "max": float(df_aq["pm10_mean"].dropna().max()),
        } if "pm10_mean" in df_aq else {},
    }

    # 5. City Name Cross-Matching
    weather_city_set = set(df_weather["city"].unique())
    aq_city_set = set(df_aq["city"].unique())
    cities_meta_set = set(df_cities["city"].unique())

    city_alignment = {
        "cities_in_metadata_not_in_weather": list(cities_meta_set - weather_city_set),
        "cities_in_weather_not_in_metadata": list(weather_city_set - cities_meta_set),
        "cities_in_weather_not_in_aq": list(weather_city_set - aq_city_set),
        "all_match": (cities_meta_set == weather_city_set == aq_city_set),
    }

    # Compile Overall Audit Report
    audit_report = {
        "dataset": "india_weather_air_quality_50_cities",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [f.model_dump() for f in files_checksums],
        "cities_profile": cities_profile,
        "weather_profile": weather_profile,
        "air_quality_profile": aq_profile,
        "city_alignment": city_alignment,
        "validation_passed": (
            weather_dups == 0
            and aq_dups == 0
            and city_alignment["all_match"]
            and rain_negative == 0
            and pm25_neg == 0
        ),
    }

    # Save JSON report
    report_json_path = out_path / "india_weather_air_quality_audit.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    # Save Human-readable Markdown summary
    report_md_path = out_path / "india_weather_air_quality_audit.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Data Audit Report: India Daily Weather & Air Quality (50 Cities)\n\n")
        f.write(f"**Audited At:** {audit_report['audited_at_utc']}\n")
        f.write(f"**Validation Status:** {'PASSED' if audit_report['validation_passed'] else 'FLAGGED'}\n\n")
        f.write(f"## 1. File Integrity & Provenance\n\n")
        for f_meta in files_checksums:
            f.write(f"- **{f_meta.name}**: {f_meta.size_bytes:,} bytes | SHA-256: `{f_meta.sha256}`\n")
        f.write(f"\n## 2. Coverage & Shape Summary\n\n")
        f.write(f"- **Cities Count:** {cities_unique} cities across India\n")
        f.write(f"- **Weather Records:** {weather_rows:,} rows ({weather_start} to {weather_end})\n")
        f.write(f"- **Air Quality Records:** {aq_rows:,} rows ({aq_start} to {aq_end})\n")
        f.write(f"- **Primary Key Duplicates:** Weather: {weather_dups}, AQ: {aq_dups}\n")
        f.write(f"- **City Alignment:** {'All 50 cities match perfectly across all 3 tables' if city_alignment['all_match'] else 'City mismatch detected!'}\n")
        f.write(f"\n## 3. Physical Sanity Checks\n\n")
        f.write(f"- Negative Rain: {rain_negative}\n")
        f.write(f"- Negative PM2.5: {pm25_neg}\n")
        f.write(f"- Extreme Temperature Flag (>55°C or <-15°C): {temp_anomalies}\n")

    # Generate Populated Manifest
    manifest = DatasetManifest(
        source_id="india_weather_air_quality_50_cities",
        display_name="India Daily Weather & Air Quality — 50 cities",
        source_url="https://huggingface.co/datasets/AkshayKarthick/india-weather-air-quality",
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        license="CC-BY-4.0",
        attribution="Akshay Karthick; Open-Meteo Historical Weather API; CAMS",
        upstream_sources=[
            "Open-Meteo Historical Weather API (ERA5/ERA5-Land and ECMWF IFS)",
            "Open-Meteo Air Quality API (Copernicus Atmosphere Monitoring Service - CAMS)",
        ],
        files=files_checksums,
        coverage={
            "weather_start": weather_start,
            "weather_end": weather_end,
            "weather_rows": weather_rows,
            "air_quality_start": aq_start,
            "air_quality_end": aq_end,
            "air_quality_rows": aq_rows,
            "unique_cities": cities_unique,
        },
        spatial_unit="city-centre model grid cell",
        value_type="model_derived",
        notes="Grid cell estimates from ERA5/CAMS; not physical CPCB ground station monitors. CAMS ozone may be elevated.",
    )

    manifest_path = man_path / "india_weather_air_quality_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.info("Data audit completed successfully. Manifest and reports generated.")
    return audit_report


if __name__ == "__main__":
    root = get_project_root()
    audit_india_50_cities_dataset(
        raw_dir=root / "data" / "raw" / "india_weather_air_quality",
        output_dir=root / "reports" / "data_audit",
        manifest_dir=root / "data" / "manifests",
    )
