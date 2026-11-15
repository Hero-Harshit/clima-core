"""
Canonical data and prediction contracts for Clima-Core.
Defines Pydantic schemas enforcing Section 11 of the specification.
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class CanonicalForecastRecord(BaseModel):
    """Logical contract for raw or standardized forecast inputs."""
    source_id: str
    provider: str
    model_id: str
    model_version: Optional[str] = None
    location_id: str
    location_type: Literal["station", "grid_cell", "city_centre", "district"]
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    issue_time_utc: Optional[datetime] = None
    valid_time_utc: datetime
    lead_hours: Optional[float] = None
    lead_days: Optional[int] = None
    variable: str
    value: float
    unit: str
    spatial_resolution: Optional[str] = None
    retrieved_at_utc: datetime
    source_quality_flag: Optional[str] = None


class CanonicalObservationRecord(BaseModel):
    """Logical contract for observations, reanalysis, or ground measurements."""
    source_id: str
    provider: str
    product_type: Literal[
        "station_observation",
        "gridded_analysis",
        "reanalysis",
        "satellite_estimate",
        "model_derived"
    ]
    location_id: str
    location_type: Literal["station", "grid_cell", "city_centre", "district"]
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    observation_time_utc: Optional[datetime] = None
    local_date: Optional[date] = None
    variable: str
    value: float
    unit: str
    spatial_resolution: Optional[str] = None
    retrieved_at_utc: datetime
    quality_flag: Optional[str] = None


class CanonicalPredictionRecord(BaseModel):
    """Standardized output emitted by all Clima-Core hazard modules."""
    prediction_id: str
    module: Literal["heat_risk", "rain_risk", "air_quality", "storm_risk", "water_risk", "compound_risk"]
    model_name: str
    model_version: str
    model_artifact_checksum: str
    location_id: str
    location_type: Literal["station", "grid_cell", "city_centre", "district"]
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    issue_time_utc: datetime
    valid_time_utc: datetime
    lead_hours: Optional[float] = None
    lead_days: Optional[int] = None
    prediction_type: Literal["forecast_correction", "regression", "probability", "rule_based_score", "official_alert"]
    variable: str
    value: float
    unit: str
    risk_category: Optional[str] = None
    probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainty_value: Optional[float] = None
    uncertainty_method: Optional[str] = None
    input_sources: List[str]
    source_timestamps: Dict[str, str]
    spatial_resolution: str
    methodology_version: str
    created_at_utc: datetime
    limitations: str


class FileChecksum(BaseModel):
    name: str
    sha256: str
    size_bytes: int


class DatasetManifest(BaseModel):
    """Manifest tracking dataset provenance, checksums, and coverage."""
    source_id: str
    display_name: str
    source_url: str
    retrieved_at_utc: str
    license: str
    attribution: str
    upstream_sources: List[str]
    files: List[FileChecksum]
    coverage: Dict[str, Any]
    spatial_unit: str
    value_type: str
    notes: str
