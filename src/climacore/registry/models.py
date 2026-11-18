"""
Pydantic data models for Clima-Core Canonical Registry.
Covers:
  1. ModelRecord (auditable ML model registry entry)
  2. DatasetRecord (canonical dataset registry entry)
  3. PhysicalThresholdRecord (sourced physical equation / threshold entry)
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class ModelRecord(BaseModel):
    """Canonical registry entry for an operational or champion ML model."""
    model_id: str = Field(..., description="Unique model identifier, e.g., 'flood_risk_v2'")
    name: str = Field(..., description="Human-readable model name")
    version: str = Field(..., description="SemVer string, e.g., '2.0.0'")
    flagship_suite: str = Field(..., description="Flagship suite name (FloodRisk, HeatRisk, PowerGrid, AgriSense, AirQuality)")
    task_type: str = Field(..., description="binary_classification | multiclass_classification | regression | quantile_regression")
    algorithm: str = Field(..., description="LightGBM GBDT | CatBoost | Scipy GEV | etc.")
    target: str = Field(..., description="Target variable name")
    target_description: str = Field(..., description="Detailed description of predicted target")
    training_period: str = Field(..., description="Date range of training split")
    validation_period: Optional[str] = Field(None, description="Date range of validation split")
    test_period: Optional[str] = Field(None, description="Date range of test split")
    validation_strategy: str = Field(..., description="chronological | event_safe | spatial_loczo")
    features: List[str] = Field(..., description="Exact ordered feature columns")
    features_count: int = Field(..., description="Total feature count")
    source_datasets: List[str] = Field(..., description="Names of source datasets contributing features")
    metrics: Dict[str, Any] = Field(..., description="Audited test set metrics (e.g., R2, MAE, AUC, Top-1)")
    baseline_metrics: Optional[Dict[str, Any]] = Field(None, description="Comparison vs naive persistence / climatology")
    artifact_path: str = Field(..., description="Path to serialized model artifact on disk")
    artifact_sha256: str = Field(..., description="SHA-256 checksum of model artifact")
    known_limitations: List[str] = Field(default_factory=list, description="Documented limitations and unmodeled boundary conditions")
    operational_status: str = Field(default="CHAMPION", description="CHAMPION | CHALLENGER | RETIRED | PROVISIONAL")


class DatasetRecord(BaseModel):
    """Canonical registry entry for an ingested historical or spatial dataset."""
    dataset_id: str = Field(..., description="Unique dataset key, e.g., 'imd_temperature_74y'")
    name: str = Field(..., description="Canonical dataset title")
    source_agency: str = Field(..., description="IMD | ECMWF | CPCB | IIT Delhi | CEA | ICAR | WorldPop | OSM")
    citation: str = Field(..., description="Formal scientific citation or data license")
    temporal_range: str = Field(..., description="Years covered, e.g., '1951-2024'")
    spatial_coverage: str = Field(..., description="50 Indian Cities | All India | State Level")
    spatial_resolution: str = Field(..., description="1.0 deg | 0.25 deg | 100m | Point | District")
    temporal_resolution: str = Field(..., description="Daily | Hourly | Seasonal | Static")
    total_records: int = Field(..., description="Total row count in processed table")
    processed_file_path: str = Field(..., description="Path to processed parquet/json file")
    sha256: str = Field(..., description="SHA-256 checksum of processed file")
    feature_columns: List[str] = Field(..., description="Column names ingested into Clima-Core")


class PhysicalThresholdRecord(BaseModel):
    """Sourced physical, physiological, or engineering threshold."""
    parameter_id: str = Field(..., description="e.g. 'tropical_night_tmin'")
    name: str = Field(..., description="Human-readable title")
    threshold_value: float = Field(..., description="Numeric value triggering alert")
    unit: str = Field(..., description="degC | mm | hPa | etc.")
    source_authority: str = Field(..., description="Official standard or scientific literature source")
    citation: str = Field(..., description="Standard reference, e.g., 'IEC 60076 / IS 2026'")
    physical_mechanism: str = Field(..., description="Explanation of why threshold causes physical failure or stress")
    geographical_scope: str = Field(..., description="All India | Coastal Cities | Plains | Hill Stations")
    operational_action: str = Field(..., description="Civil or operational response triggered by threshold")
