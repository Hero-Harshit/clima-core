"""
Dataset builder for RainRisk module.
Builds and validates processed tabular datasets for next-day precipitation occurrence
and volume prediction across 50 Indian cities.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.exceptions import LeakageError
from climacore.config.loader import get_project_root
from climacore.features.rain_risk import build_rain_risk_features

logger = get_logger("climacore.models.rain_risk.dataset")


def prepare_rain_risk_dataset(
    raw_dir: Path | str = None,
    output_dir: Path | str = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Build and save processed dataset for RainRisk two-part model training."""
    root = get_project_root()
    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / "india_weather_air_quality"
    out_path = Path(output_dir) if output_dir else root / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw weather and city data for RainRisk...")
    df_weather = pd.read_csv(raw_path / "weather_daily.csv")
    df_cities = pd.read_csv(raw_path / "cities.csv")

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    merged = pd.merge(df_weather, df_cities, on="city", how="left")
    merged = merged.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Build features
    logger.info("Computing precipitation lags, accumulation windows, and moisture features...")
    df_features, feature_cols = build_rain_risk_features(merged)

    # 2. Filter valid rows: drop initial 7-day warm-up and terminal target NaN
    valid_mask = df_features["target_precipitation_next_day"].notnull()
    valid_mask = valid_mask & df_features["precipitation_sum_lag_7"].notnull()
    df_clean = df_features[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned RainRisk dataset contains {len(df_clean):,} rows.")

    # 3. Rigorous Leakage Check
    logger.info("Verifying temporal target alignment for RainRisk...")
    sample_city = "Mumbai"
    mumbai_sample = df_clean[df_clean["city"] == sample_city].sort_values("date").reset_index(drop=True)
    for idx in range(min(5, len(mumbai_sample) - 1)):
        current_target = mumbai_sample.loc[idx, "target_precipitation_next_day"]
        next_day_val = mumbai_sample.loc[idx + 1, "precipitation_sum"]
        if abs(current_target - next_day_val) > 1e-5:
            raise LeakageError(f"Target alignment mismatch in {sample_city} at index {idx}: {current_target} != {next_day_val}")
    logger.info("RainRisk temporal leakage verification passed.")

    # 4. Save to Parquet
    parquet_path = out_path / "rain_risk_train_ready.parquet"
    df_clean.to_parquet(parquet_path, index=False)
    logger.info(f"RainRisk processed dataset saved to {parquet_path}")

    # 5. Metadata Schema
    metadata = {
        "dataset_name": "rain_risk_next_day_precipitation",
        "rows": len(df_clean),
        "start_date": df_clean["date"].min().strftime("%Y-%m-%d"),
        "end_date": df_clean["date"].max().strftime("%Y-%m-%d"),
        "unique_cities": int(df_clean["city"].nunique()),
        "target_amount_column": "target_precipitation_next_day",
        "target_occurrence_column": "target_rain_occurrence_next_day",
        "target_imd_rainy_day_column": "target_imd_rainy_day_next_day",
        "target_heavy_rain_column": "target_heavy_rain_next_day",
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
    }

    schema_path = out_path / "rain_risk_feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return df_clean, feature_cols, metadata


if __name__ == "__main__":
    prepare_rain_risk_dataset()
