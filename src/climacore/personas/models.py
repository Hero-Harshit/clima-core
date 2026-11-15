"""
Data structures and schemas for Clima-Core Persona Intelligence Engine.
Defines input parameters from physical ML models and structured persona advisory outputs.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class PersonaInput(BaseModel):
    """Input payload containing physical hazard forecasts and atmospheric metrics."""
    city: str
    date: str
    temp_max: float = Field(..., description="Forecasted maximum 2m dry-bulb temperature in °C")
    temp_min: float = Field(default=22.0, description="Forecasted minimum 2m temperature in °C")
    apparent_temp_max: Optional[float] = Field(default=None, description="Apparent temperature / Heat Index in °C")
    precipitation_mm: float = Field(default=0.0, description="Forecasted 24h precipitation in mm")
    rain_probability: float = Field(default=0.0, description="Probability of rain >= 1.0 mm (0.0 to 1.0)")
    pm2_5: float = Field(..., description="Forecasted PM2.5 concentration in µg/m³")
    pm10: Optional[float] = Field(default=None, description="Forecasted PM10 concentration in µg/m³")
    no2: Optional[float] = Field(default=None, description="Forecasted NO2 concentration in µg/m³")
    wind_speed_kmh: float = Field(default=12.0, description="Forecasted maximum 10m wind speed in km/h")
    relative_humidity: float = Field(default=50.0, description="Forecasted mean relative humidity in %")
    solar_radiation: float = Field(default=20.0, description="Forecasted shortwave radiation in MJ/m²")
    et0_evapotranspiration: float = Field(default=5.0, description="Reference evapotranspiration in mm/day")
    imd_heatwave_alert: int = Field(default=0, description="IMD Heatwave level: 0=None, 1=Yellow, 2=Orange, 3=Red")
    atmospheric_stagnation: int = Field(default=0, description="Atmospheric stagnation trapping flag (0 or 1)")
    historical_percentile: float = Field(default=50.0, description="Empirical 44-year historical severity percentile (0-100)")
    soil_moisture_deficit_index: Optional[float] = Field(default=None, description="Root-Zone / Topsoil Moisture Deficit Index (-3.0 extreme dry to +3.0 wet)")
    flash_drought_alert: Optional[int] = Field(default=0, description="Agricultural flash drought hazard flag (0 or 1)")
    gale_wind_gust_kmh: Optional[float] = Field(default=None, description="Forecasted peak wind gust speed in km/h")
    cyclone_distance_km: Optional[float] = Field(default=None, description="Distance to nearest active cyclonic system in km")
    cyclone_grade: Optional[str] = Field(default=None, description="IMD intensity grade of nearest cyclonic system (e.g. CS, SCS, VSCS, SuCS)")
    groundwater_depletion_level: Optional[float] = Field(default=None, description="Aquifer water table depth in meters below ground level (mbgl)")
    reservoir_storage_percent: Optional[float] = Field(default=None, description="Current reservoir storage as % of live capacity")
    wet_bulb_temp_c: Optional[float] = Field(default=None, description="Estimated wet-bulb temperature in °C")
    surface_solar_radiation_w_m2: Optional[float] = Field(default=None, description="Solar surface irradiance in W/m²")
    uhi_temperature_delta_c: Optional[float] = Field(default=None, description="Urban Heat Island temperature anomaly in °C")


class PersonaResult(BaseModel):
    """Structured decision output tailored for a specific occupation/persona."""
    persona_id: str
    persona_title: str
    status_level: str  # e.g., "NORMAL", "CAUTION", "HIGH_ALERT", "CRITICAL"
    impact_score: float = Field(..., ge=0.0, le=100.0, description="Normalized impact severity score from 0 to 100")
    key_metrics: Dict[str, Any]
    advisories: List[str]
    operational_restrictions: List[str]
