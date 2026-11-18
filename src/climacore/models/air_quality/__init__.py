"""Air Quality Model Package."""
from climacore.models.air_quality.train import train_air_quality_model
from climacore.models.air_quality.engine import (
    AirQualityEngine,
    load_air_quality_boosters,
    compute_sub_index,
    classify_naqi,
    PM25_BREAKPOINTS,
    PM10_BREAKPOINTS,
    NO2_BREAKPOINTS,
)

__all__ = [
    "train_air_quality_model",
    "AirQualityEngine",
    "load_air_quality_boosters",
    "compute_sub_index",
    "classify_naqi",
    "PM25_BREAKPOINTS",
    "PM10_BREAKPOINTS",
    "NO2_BREAKPOINTS",
]
