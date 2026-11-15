"""
Source adapter for 44-Year Historical Weather Archive (1980-2024).
Combines decade-wise daily weather records across 50 Indian cities,
computes standard 30-year WMO climate normals (1991-2020), and saves processed Parquet.
"""

from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime, timezone
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256, get_file_metadata
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.historical_1980")


def compile_historical_44y_weather(
    raw_dir: Path | str = None,
    output_dir: Path | str = None,
) -> pd.DataFrame:
    """Load, standardize, and concatenate 1980-2024 daily weather data."""
    root = get_project_root()
    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / "india_weather_historical_1980"
    out_path = Path(output_dir) if output_dir else root / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    decade_files = [
        raw_path / "1980-1989" / "daily_data_combined_1980_to_1989.csv",
        raw_path / "1990-1999" / "daily_data_combined_1990_to_1999.csv",
        raw_path / "2000-2009" / "daily_data_combined_2000_to_2009.csv",
        raw_path / "2010-2019" / "daily_data_combined_2010_to_2019.csv",
        raw_path / "2020-ytd" / "daily_data.csv",
    ]

    dfs = []
    file_metadata = []

    for f in decade_files:
        if not f.exists():
            raise FileNotFoundError(f"Missing historical weather file: {f}")
        logger.info(f"Loading {f.name} ({f.parent.name})...")
        file_metadata.append(get_file_metadata(f))
        
        df = pd.read_csv(f)
        rename_map = {
            "city_name": "city",
            "datetime": "date",
        }
        df = df.rename(columns=rename_map)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        dfs.append(df)

    logger.info("Concatenating all decades...")
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["city", "date"]).sort_values(by=["city", "date"]).reset_index(drop=True)

    out_file = out_path / "weather_historical_44y_train_ready.parquet"
    combined.to_parquet(out_file, index=False)
    logger.info(f"Saved 44-year historical weather dataset to {out_file}: {len(combined):,} rows.")

    # Compute 30-Year WMO Climate Normals (1991 - 2020 standard baseline period)
    logger.info("Computing 30-year WMO climate normals (1991-2020)...")
    combined["year"] = pd.to_datetime(combined["date"]).dt.year
    combined["month"] = pd.to_datetime(combined["date"]).dt.month
    wmo_subset = combined[(combined["year"] >= 1991) & (combined["year"] <= 2020)]

    normals = wmo_subset.groupby(["city", "month"]).agg({
        "temperature_2m_max": ["mean", "std"],
        "temperature_2m_min": ["mean", "std"],
        "precipitation_sum": ["mean", "sum"],
    }).reset_index()

    normals_dict = {}
    for _, row in normals.iterrows():
        c = row["city"].values[0] if hasattr(row["city"], "values") else row["city"]
        m = int(row["month"].values[0] if hasattr(row["month"], "values") else row["month"])
        key = f"{c}_{m}"
        normals_dict[key] = {
            "city": c,
            "month": m,
            "tmax_norm_mean": round(float(row[("temperature_2m_max", "mean")]), 2),
            "tmax_norm_std": round(float(row[("temperature_2m_max", "std")]), 2),
            "tmin_norm_mean": round(float(row[("temperature_2m_min", "mean")]), 2),
            "precip_norm_mean": round(float(row[("precipitation_sum", "mean")]), 2),
        }

    normals_file = out_path / "wmo_climate_normals_1991_2020.json"
    with open(normals_file, "w", encoding="utf-8") as f:
        json.dump(normals_dict, f, indent=2)

    # Manifest
    man_path = root / "data" / "manifests" / "historical_1980_manifest.json"
    manifest_data = {
        "source_id": "india_weather_historical_1980",
        "display_name": "India 50 Cities 44-Year Historical Weather Archive (1980-2024)",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo ERA5 / ECMWF historical archive via curated Kaggle decades",
        "spatial_resolution": "50 city-centre model grid cells",
        "files": file_metadata,
        "total_records": len(combined),
        "cities_count": int(combined["city"].nunique()),
        "start_date": combined["date"].min(),
        "end_date": combined["date"].max(),
        "wmo_climate_normals_period": "1991-2020",
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return combined


if __name__ == "__main__":
    compile_historical_44y_weather()
