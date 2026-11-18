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
    population_5km: Optional[int] = Field(default=None, description="WorldPop 2024 population within 5km core radius")
    population_15km: Optional[int] = Field(default=None, description="WorldPop 2024 population within 15km metro radius")
    population_30km: Optional[int] = Field(default=None, description="WorldPop 2024 population within 30km urban region")
    dense_cluster_pop_15km: Optional[int] = Field(default=None, description="WorldPop 2024 population in high-density informal/urban clusters (>=75/cell) within 15km")
    urban_exposure_tier: Optional[str] = Field(default=None, description="Urban Exposure Tier from WorldPop 2024")
    # OpenStreetMap Critical Infrastructure
    hospitals_15km: Optional[int] = Field(default=None, description="Count of hospitals within 15km metro radius")
    clinics_15km: Optional[int] = Field(default=None, description="Count of clinics and primary care within 15km metro radius")
    total_healthcare_15km: Optional[int] = Field(default=None, description="Total healthcare facilities within 15km metro radius")
    schools_15km: Optional[int] = Field(default=None, description="Count of educational institutions within 15km metro radius")
    bridges_15km: Optional[int] = Field(default=None, description="Count of bridges and flyovers within 15km metro radius")
    drainage_canals_15km: Optional[int] = Field(default=None, description="Count of drainage canals and waterways within 15km metro radius")
    arterial_roads_15km: Optional[int] = Field(default=None, description="Count of primary/trunk arterial road links within 15km metro radius")
    industrial_zones_15km: Optional[int] = Field(default=None, description="Count of industrial zones within 15km metro radius")
    transit_stations_15km: Optional[int] = Field(default=None, description="Count of rail transit stations and airports within 15km metro radius")
    people_per_hospital_15km: Optional[int] = Field(default=None, description="Ratio of 15km population per hospital facility")
    people_per_school_15km: Optional[int] = Field(default=None, description="Ratio of 15km population per school facility")
    critical_infrastructure_density_km2: Optional[float] = Field(default=None, description="Density of critical facilities per km2")
    drainage_arterial_ratio: Optional[float] = Field(default=None, description="Drainage canals to arterial roads ratio")
    # India Flood Inventory (IFI) Supervised Risk
    flood_disaster_risk_prob: Optional[float] = Field(default=None, description="Calibrated probability of district flood disaster from IFI model (0.0 to 1.0)")
    antecedent_rain_3d_mm: Optional[float] = Field(default=None, description="Rolling 3-day accumulated precipitation in mm")
    antecedent_rain_7d_mm: Optional[float] = Field(default=None, description="Rolling 7-day accumulated precipitation in mm")
    # AgriSense District Crop & Soil Resilience
    agricultural_district: Optional[str] = Field(default=None, description="Matched agricultural hinterland district")
    dominant_kharif_crop: Optional[str] = Field(default=None, description="Dominant Kharif season crop")
    dominant_rabi_crop: Optional[str] = Field(default=None, description="Dominant Rabi season crop")
    dominant_crop_overall: Optional[str] = Field(default=None, description="Overall dominant crop in the agricultural hinterland")
    soil_ph: Optional[float] = Field(default=None, description="Soil pH in 0-30cm root zone")
    soil_organic_carbon_g_kg: Optional[float] = Field(default=None, description="Soil organic carbon in g/kg")
    soil_fertility_index: Optional[float] = Field(default=None, description="Composite soil fertility score (0-100)")
    soil_water_holding_capacity: Optional[float] = Field(default=None, description="Estimated soil water holding capacity index")
    # Heat-to-Electricity Grid Stress Intelligence
    electricity_grid_stress_ratio: Optional[float] = Field(default=None, description="Forecasted electricity demand surge ratio relative to baseline (e.g. 1.25 = +25%)")
    peak_grid_load_mw: Optional[float] = Field(default=None, description="Forecasted peak state electricity load in Megawatts (MW)")
    grid_blackout_hazard_prob: Optional[float] = Field(default=None, description="Probability of peak unmet supply / load shedding under thermal stress (0.0 to 1.0)")
    # SAGE Crop Calendar & Phenological Growth-Stages
    crop_growth_stage: Optional[str] = Field(default=None, description="Current estimated crop phenological stage (e.g. Anthesis/Flowering, Vegetative, Harvesting)")
    crop_sowing_dates: Optional[str] = Field(default=None, description="Normal sowing window dates from SAGE calendar")
    crop_anthesis_dates: Optional[str] = Field(default=None, description="Critical flowering/anthesis window dates from SAGE calendar")
    crop_harvest_dates: Optional[str] = Field(default=None, description="Normal harvesting window dates from SAGE calendar")
    # CHETNA Industrial Power Plants Point-Source Exposure
    thermal_plants_within_100km: Optional[int] = Field(default=None, description="Number of thermal power plants within 100km regional airshed")
    thermal_capacity_100km_mw: Optional[float] = Field(default=None, description="Total generating capacity of thermal power plants within 100km in MW")
    closest_thermal_plant: Optional[str] = Field(default=None, description="Name, distance, and bearing of closest major thermal power plant")


class PersonaResult(BaseModel):
    """Structured decision output tailored for a specific occupation/persona."""
    persona_id: str
    persona_title: str
    status_level: str  # e.g., "NORMAL", "CAUTION", "HIGH_ALERT", "CRITICAL"
    impact_score: float = Field(..., ge=0.0, le=100.0, description="Normalized impact severity score from 0 to 100")
    key_metrics: Dict[str, Any]
    advisories: List[str]
    operational_restrictions: List[str]
