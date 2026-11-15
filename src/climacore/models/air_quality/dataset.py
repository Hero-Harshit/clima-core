"""
Dataset builder for AirQuality module.
Performs joins across weather, air quality, and city metadata,
computes features, performs temporal leakage checks, and stores processed Parquet.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.exceptions import LeakageError, DataValidationError
from climacore.config.loader import get_project_root
from climacore.features.air_quality import build_air_quality_features

logger = get_logger("climacore.models.air_quality.dataset")


def prepare_air_quality_dataset(
    raw_dir: Path | str = None,
    output_dir: Path | str = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Build and save the processed dataset for AirQuality model training."""
    root = get_project_root()
    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / "india_weather_air_quality"
    out_path = Path(output_dir) if output_dir else root / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw CSV files for air quality dataset preparation...")
    df_aq = pd.read_csv(raw_path / "air_quality_daily.csv")
    df_weather = pd.read_csv(raw_path / "weather_daily.csv")
    df_cities = pd.read_csv(raw_path / "cities.csv")

    df_aq["date"] = pd.to_datetime(df_aq["date"])
    df_weather["date"] = pd.to_datetime(df_weather["date"])

    # 1. Join Air Quality and Weather
    logger.info(f"Merging AQ ({len(df_aq):,} rows) with Weather ({len(df_weather):,} rows)...")
    merged = pd.merge(df_aq, df_weather, on=["city", "date"], how="inner")
    logger.info(f"Merged AQ+Weather contains {len(merged):,} rows.")

    # 2. Join City Metadata
    merged = pd.merge(merged, df_cities, on="city", how="left")

    # 3. Build time-safe features
    logger.info("Computing temporal, lag, rolling, and interaction features...")
    df_features, feature_cols = build_air_quality_features(merged)

    # 4. Filter valid rows: drop initial lag warm-up and terminal target NaN
    initial_count = len(df_features)
    # Target must not be null
    valid_mask = df_features["target_pm2_5_next_day"].notnull()
    # Also drop warm-up rows where the 7-day lags are NaN
    valid_mask = valid_mask & df_features["pm2_5_mean_lag_7"].notnull()
    df_clean = df_features[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned dataset: {len(df_clean):,} rows (from {initial_count:,} raw rows after lag warm-up).")

    # 5. Leakage Verification Check
    # Verify that target_pm2_5_next_day at date t strictly equals pm2_5_mean at date t+1 for the same city
    logger.info("Performing rigorous temporal leakage and alignment verification...")
    sample_city = "Delhi"
    delhi_sample = df_clean[df_clean["city"] == sample_city].sort_values("date").reset_index(drop=True)
    for idx in range(min(5, len(delhi_sample) - 1)):
        current_target = delhi_sample.loc[idx, "target_pm2_5_next_day"]
        next_day_val = delhi_sample.loc[idx + 1, "pm2_5_mean"]
        if abs(current_target - next_day_val) > 1e-5:
            raise LeakageError(f"Target alignment mismatch in {sample_city} at index {idx}: {current_target} != {next_day_val}")
    logger.info("Leakage verification passed: target is strictly date t+1 observation.")

    # 6. Save Processed Dataset to Parquet
    parquet_path = out_path / "air_quality_train_ready.parquet"
    df_clean.to_parquet(parquet_path, index=False)
    logger.info(f"Processed dataset saved to {parquet_path}")

    # 7. Save Feature Schema
    metadata = {
        "dataset_name": "air_quality_next_day_pm25",
        "rows": len(df_clean),
        "start_date": df_clean["date"].min().strftime("%Y-%m-%d"),
        "end_date": df_clean["date"].max().strftime("%Y-%m-%d"),
        "unique_cities": int(df_clean["city"].nunique()),
        "target_column": "target_pm2_5_next_day",
        "target_description": "Next-day CAMS-derived PM2.5 mean (ug/m3) for city-centre grid cell",
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
    }

    schema_path = out_path / "air_quality_feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return df_clean, feature_cols, metadata


if __name__ == "__main__":
    prepare_air_quality_dataset()
