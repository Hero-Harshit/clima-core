"""Clima-Core Canonical Registry Package."""
from climacore.registry.models import ModelRecord, DatasetRecord, PhysicalThresholdRecord
from climacore.registry.engine import (
    get_model_registry,
    get_dataset_registry,
    get_physical_thresholds_registry,
    export_canonical_registries,
)

__all__ = [
    "ModelRecord",
    "DatasetRecord",
    "PhysicalThresholdRecord",
    "get_model_registry",
    "get_dataset_registry",
    "get_physical_thresholds_registry",
    "export_canonical_registries",
]
