"""
Data contracts and Pydantic schemas for Clima-Core Live Ingestion Pipeline.
Defines standardized, immutable representations for real-time 7-day weather, air quality,
probabilistic conformal intervals, cascading failure chains, and persona-tailored multi-hazard predictions.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Physical Units Manifest mapping every environmental and engineering variable to its SI/engineering unit
PHYSICAL_UNITS_MANIFEST: Dict[str, str] = {
    "temp_max": "deg_C",
    "temp_min": "deg_C",
    "apparent_temp_max": "deg_C",
    "wet_bulb_temp_c": "deg_C",
    "cooling_deficit_c": "deg_C",
    "uhi_temperature_delta_c": "deg_C",
    "precipitation_mm": "mm",
    "rain_rolling_3d_mm": "mm",
    "rain_rolling_7d_mm": "mm",
    "antecedent_precipitation_index_14d": "mm",
    "effective_surface_runoff_mm": "mm",
    "et0_fao56_mm_day": "mm/day",
    "water_balance_mm_day": "mm/day",
    "wind_speed_kmh": "km/h",
    "gale_wind_gust_kmh": "km/h",
    "relative_humidity": "percent",
    "soil_moisture_rootzone": "m3/m3",
    "rootzone_saturation_pct": "percent",
    "solar_radiation": "MJ/m2",
    "pm2_5": "ug/m3",
    "pm10": "ug/m3",
    "no2": "ug/m3",
    "o3": "ug/m3",
    "so2": "ug/m3",
    "co": "ug/m3",
    "peak_load_mw": "MW",
    "ventilation_index_m2_s": "m2/s",
    "compound_risk_score": "score_0_to_100",
    "cascading_risk_score": "score_0_to_100",
    "impact_severity_index": "score_0_to_100",
    "agro_stress_score": "score_0_to_100",
    "hydrological_hazard_probability": "probability_0_to_1",
    "unmet_supply_stress_probability": "probability_0_to_1",
    "cooling_demand_surge_ratio": "ratio_baseline_1_0",
}


class LiveDayForecast(BaseModel):
    """Normalized environmental and atmospheric forecast for a single calendar day."""
    date: str = Field(..., description="Date string YYYY-MM-DD")
    day_offset: int = Field(..., ge=0, le=14, description="Forecast day index (0=Today, 1=Tomorrow, etc.)")
    temp_max: float = Field(..., description="Daily maximum 2m air temperature in °C")
    temp_min: float = Field(..., description="Daily minimum 2m air temperature in °C")
    apparent_temp_max: float = Field(..., description="Daily maximum apparent temperature in °C")
    precipitation_mm: float = Field(default=0.0, description="Forecasted 24h total rainfall in mm")
    rain_probability: float = Field(default=0.0, description="Probability of precipitation (0.0 to 1.0)")
    wind_speed_kmh: float = Field(default=12.0, description="Daily maximum 10m wind speed in km/h")
    gale_wind_gust_kmh: float = Field(default=16.0, description="Daily peak wind gust speed in km/h")
    relative_humidity: float = Field(default=50.0, description="Daily mean relative humidity in %")
    solar_radiation: float = Field(default=18.0, description="Shortwave solar radiation in MJ/m²")
    et0_evapotranspiration: float = Field(default=4.5, description="Reference FAO-56 evapotranspiration in mm/day")
    pm2_5: float = Field(default=55.0, description="Mean daily PM2.5 concentration in µg/m³")
    pm10: float = Field(default=95.0, description="Mean daily PM10 concentration in µg/m³")
    no2: float = Field(default=25.0, description="Mean daily NO2 concentration in µg/m³")
    so2: float = Field(default=8.0, description="Mean daily SO2 concentration in µg/m³")
    co: float = Field(default=500.0, description="Mean daily CO concentration in µg/m³")
    ozone: float = Field(default=40.0, description="Mean daily Ozone concentration in µg/m³")
    soil_moisture_0_to_7cm: Optional[float] = Field(default=None, description="Volumetric topsoil water content m³/m³")
    soil_moisture_7_to_28cm: Optional[float] = Field(default=None, description="Volumetric root-zone water content m³/m³")
    wet_bulb_temp_c: Optional[float] = Field(default=None, description="Psychrometric wet-bulb temperature in °C")
    imd_heatwave_alert: int = Field(default=0, description="IMD Heatwave Alert: 0=None, 1=Yellow, 2=Orange, 3=Red")
    atmospheric_stagnation: int = Field(default=0, description="Inversion / boundary layer stagnation flag (0 or 1)")
    historical_percentile: float = Field(default=50.0, description="Historical 74-year temperature percentile (0-100)")
    soil_moisture_deficit_index: Optional[float] = Field(default=None, description="Root-Zone Soil Moisture Deficit Index (SMDI)")
    flash_drought_alert: Optional[int] = Field(default=0, description="Agricultural flash drought alert flag (0 or 1)")
    cyclone_distance_km: Optional[float] = Field(default=None, description="Distance to nearest active cyclonic system in km")
    cyclone_grade: Optional[str] = Field(default=None, description="IMD storm intensity scale grade")


class LiveCityForecast(BaseModel):
    """Complete multi-day live forecast bundle for a target city or coordinate."""
    city: str
    state: Optional[str] = None
    latitude: float
    longitude: float
    elevation_m: Optional[float] = None
    timezone: str = "Asia/Kolkata"
    retrieved_at_utc: str
    is_cached: bool = False
    cached_at_utc: Optional[str] = None
    days: List[LiveDayForecast]


class LiveDayPrediction(BaseModel):
    """Evaluated risk score, conformal intervals, cascading chains, and persona intelligence for one forecast day."""
    date: str
    day_offset: int
    day_name: str
    forecast_metrics: Dict[str, Any]
    compound_risk: Dict[str, Any]
    personas: Dict[str, Any]
    flagship_intelligence: Optional[Dict[str, Any]] = Field(
        default=None, description="Detailed evaluations from the 5 Flagship Engines (Flood, Heat, Power, Agri, Air)"
    )
    conformal_bounds: Optional[Dict[str, Any]] = Field(
        default=None, description="Split conformal prediction intervals guaranteeing >= 90% finite-sample marginal coverage"
    )
    cascading_risk: Optional[Dict[str, Any]] = Field(
        default=None, description="Cascading multi-hazard infrastructure dependency failure chains"
    )


class LivePredictionResult(BaseModel):
    """Standardized Canonical PredictionResult contract for Clima-Core (v1.0.0)."""
    schema_version: str = Field(default="v1.0.0", description="Contract schema version")
    city: str
    state: Optional[str]
    latitude: float
    longitude: float
    retrieved_at_utc: str
    is_cached: bool
    selected_persona: str
    days_count: int
    peak_compound_risk: float
    peak_risk_date: str
    peak_risk_category: str
    daily_predictions: List[LiveDayPrediction]
    system_provenance: Optional[Dict[str, Any]] = Field(
        default=None, description="Model versions, SHA-256 hashes, and canonical dataset provenance"
    )
    spatial_resolution_metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Concentric buffer radii, WorldPop and atmospheric grid resolutions"
    )
    physical_units_manifest: Optional[Dict[str, str]] = Field(
        default=None, description="Explicit mapping of every environmental and engineering key to its SI/engineering unit"
    )
