"""Crop Resilience & Suitability Model Package."""
from climacore.models.crop_resilience.train import (
    train_crop_suitability_model,
    CROP_SOIL_FEATURES,
    TARGET_CROPS,
)

__all__ = ["train_crop_suitability_model", "CROP_SOIL_FEATURES", "TARGET_CROPS"]
