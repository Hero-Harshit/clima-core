"""
Dataset builder for HeatRisk temperature module.
Processes historical weather data (2020-2026), adds time-safe lag and rolling features,
and prepares chronological train/val/test datasets for next-day maximum temperature prediction.
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import json
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.exceptions import LeakageError
from climacore.config.loader import get_project_root
from climacore.features.temporal import add_temporal_features
from climacore.features.lags import add_lag_features
from climacore.features.rolling import add_rolling_features

logger = get_logger("climacore.models.heat_risk.dataset")


def prepare_heat_risk_dataset(
    raw_dir: Path | str = None,
    output_dir: Path | str = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Build and save processed dataset for HeatRisk maximum temperature prediction."""
    root = get_project_root()
    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / "india_weather_air_quality"
    out_path = Path(output_dir) if output_dir else root / "data" / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading raw weather and city data for HeatRisk...")
    df_weather = pd.read_csv(raw_path / "weather_daily.csv")
    df_cities = pd.read_csv(raw_path / "cities.csv")

    df_weather["date"] = pd.to_datetime(df_weather["date"])
    merged = pd.merge(df_weather, df_cities, on="city", how="left")
    merged = merged.sort_values(by=["city", "date"]).reset_index(drop=True)

    # 1. Temporal features
    merged = add_temporal_features(merged, date_col="date")

    # 2. Lag features
    temp_lags = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "relative_humidity_2m_mean",
        "shortwave_radiation_sum",
        "wind_speed_10m_max",
    ]
    merged = add_lag_features(
        merged,
        group_col="city",
        date_col="date",
        target_cols=temp_lags,
        lags=[1, 2, 3, 7],
    )

    # 3. Rolling window features
    merged = add_rolling_features(
        merged,
        group_col="city",
        date_col="date",
        target_cols=["temperature_2m_max", "apparent_temperature_max"],
        windows=[3, 7, 14],
        shift_before_rolling=0,
    )

    # 4. Target: Next-day maximum temperature (t+1)
    merged["target_temperature_2m_max_next_day"] = merged.groupby("city")["temperature_2m_max"].shift(-1)

    # Convert categoricals
    merged["city_cat"] = merged["city"].astype("category")
    if "region" in merged.columns:
        merged["region_cat"] = merged["region"].astype("category")

    # Filter out initial lag warm-up rows and final terminal target NaN
    valid_mask = merged["target_temperature_2m_max_next_day"].notnull()
    valid_mask = valid_mask & merged["temperature_2m_max_lag_7"].notnull()
    df_clean = merged[valid_mask].copy().reset_index(drop=True)
    logger.info(f"Cleaned HeatRisk dataset contains {len(df_clean):,} rows.")

    # Leakage check
    sample_city = "Jaipur"
    jaipur_sample = df_clean[df_clean["city"] == sample_city].sort_values("date").reset_index(drop=True)
    for idx in range(min(5, len(jaipur_sample) - 1)):
        current_target = jaipur_sample.loc[idx, "target_temperature_2m_max_next_day"]
        next_day_val = jaipur_sample.loc[idx + 1, "temperature_2m_max"]
        if abs(current_target - next_day_val) > 1e-5:
            raise LeakageError(f"Target alignment mismatch in {sample_city} at index {idx}: {current_target} != {next_day_val}")
    logger.info("HeatRisk leakage verification passed.")

    excluded = {
        "date",
        "city",
        "state",
        "region",
        "target_temperature_2m_max_next_day",
    }
    feature_cols = [c for c in df_clean.columns if c not in excluded]

    parquet_path = out_path / "heat_risk_train_ready.parquet"
    df_clean.to_parquet(parquet_path, index=False)
    logger.info(f"HeatRisk processed dataset saved to {parquet_path}")

    metadata = {
        "dataset_name": "heat_risk_next_day_temp_max",
        "rows": len(df_clean),
        "start_date": df_clean["date"].min().strftime("%Y-%m-%d"),
        "end_date": df_clean["date"].max().strftime("%Y-%m-%d"),
        "unique_cities": int(df_clean["city"].nunique()),
        "target_column": "target_temperature_2m_max_next_day",
        "target_description": "Next-day maximum temperature at 2m (deg C) for city-centre grid cell",
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
    }

    schema_path = out_path / "heat_risk_feature_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return df_clean, feature_cols, metadata


if __name__ == "__main__":
    prepare_heat_risk_dataset()
