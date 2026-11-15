"""
Configuration and normalization logic for the CompoundRisk engine.
Maps physical hazard quantities to standardized 0-100 risk scores using
documented scientific thresholds (IMD, CPCB, WMO).
"""

from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field


class HazardWeights(BaseModel):
    """Configurable weights for hazard aggregation. Must sum to 1.0."""
    air_quality: float = Field(0.35, ge=0.0, le=1.0)
    heat_risk: float = Field(0.35, ge=0.0, le=1.0)
    rain_risk: float = Field(0.30, ge=0.0, le=1.0)

    def validate_weights(self):
        total = self.air_quality + self.heat_risk + self.rain_risk
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Hazard weights must sum to 1.0, got {total:.4f}")


def normalize_air_quality(pm25_val: float) -> Tuple[float, str]:
    """
    Map 24h mean PM2.5 (ug/m3) to a 0-100 risk score based on CPCB categories:
    - 0-30 ug/m3 (Good): 0 - 20
    - 30-60 ug/m3 (Satisfactory): 20 - 40
    - 60-90 ug/m3 (Moderate): 40 - 60
    - 90-120 ug/m3 (Poor): 60 - 75
    - 120-250 ug/m3 (Very Poor): 75 - 90
    - >250 ug/m3 (Severe): 90 - 100
    """
    if pm25_val is None or pm25_val < 0:
        return 0.0, "Unknown"

    if pm25_val <= 30:
        score = (pm25_val / 30.0) * 20.0
        cat = "Good"
    elif pm25_val <= 60:
        score = 20.0 + ((pm25_val - 30.0) / 30.0) * 20.0
        cat = "Satisfactory"
    elif pm25_val <= 90:
        score = 40.0 + ((pm25_val - 60.0) / 30.0) * 20.0
        cat = "Moderate"
    elif pm25_val <= 120:
        score = 60.0 + ((pm25_val - 90.0) / 30.0) * 15.0
        cat = "Poor"
    elif pm25_val <= 250:
        score = 75.0 + ((pm25_val - 120.0) / 130.0) * 15.0
        cat = "Very Poor"
    else:
        score = 90.0 + min(10.0, ((pm25_val - 250.0) / 150.0) * 10.0)
        cat = "Severe"

    return min(100.0, max(0.0, score)), cat


def normalize_heat_risk(temp_max: float) -> Tuple[float, str]:
    """
    Map 2m max temperature (deg C) to a 0-100 risk score:
    - < 30 C (Comfortable): 0 - 20
    - 30 - 35 C (Warm): 20 - 40
    - 35 - 40 C (Hot / Caution): 40 - 65
    - 40 - 43 C (Extreme Caution): 65 - 80
    - 43 - 46 C (Danger / Heatwave): 80 - 95
    - > 46 C (Extreme Danger / Severe Heatwave): 95 - 100
    """
    if temp_max is None:
        return 0.0, "Unknown"

    if temp_max <= 30.0:
        score = max(0.0, (temp_max / 30.0) * 20.0)
        cat = "Low"
    elif temp_max <= 35.0:
        score = 20.0 + ((temp_max - 30.0) / 5.0) * 20.0
        cat = "Moderate"
    elif temp_max <= 40.0:
        score = 40.0 + ((temp_max - 35.0) / 5.0) * 25.0
        cat = "Elevated"
    elif temp_max <= 43.0:
        score = 65.0 + ((temp_max - 40.0) / 3.0) * 15.0
        cat = "High"
    elif temp_max <= 46.0:
        score = 80.0 + ((temp_max - 43.0) / 3.0) * 15.0
        cat = "Heatwave"
    else:
        score = 95.0 + min(5.0, ((temp_max - 46.0) / 4.0) * 5.0)
        cat = "Severe Heatwave"

    return min(100.0, max(0.0, score)), cat


def normalize_rain_risk(precip_mm: float, rain_prob: float = None) -> Tuple[float, str]:
    """
    Map daily precipitation volume (mm) to a 0-100 risk score based on IMD categories:
    - < 1.0 mm (No / Trace rain): 0 - 5
    - 1.0 - 15.5 mm (Light rain): 5 - 30
    - 15.6 - 64.4 mm (Moderate rain / Urban drainage stress): 30 - 65
    - 64.5 - 115.5 mm (Heavy rain / Waterlogging risk): 65 - 85
    - 115.6 - 204.4 mm (Very Heavy rain / Flash flood danger): 85 - 95
    - > 204.5 mm (Extremely Heavy rain): 95 - 100
    """
    if precip_mm is None or precip_mm < 0:
        return 0.0, "No Rain"

    if precip_mm < 1.0:
        score = (precip_mm / 1.0) * 5.0
        cat = "No Rain"
    elif precip_mm <= 15.5:
        score = 5.0 + ((precip_mm - 1.0) / 14.5) * 25.0
        cat = "Light Rain"
    elif precip_mm <= 64.4:
        score = 30.0 + ((precip_mm - 15.5) / 48.9) * 35.0
        cat = "Moderate Rain"
    elif precip_mm <= 115.5:
        score = 65.0 + ((precip_mm - 64.4) / 51.1) * 20.0
        cat = "Heavy Rain"
    elif precip_mm <= 204.4:
        score = 85.0 + ((precip_mm - 115.5) / 88.9) * 10.0
        cat = "Very Heavy Rain"
    else:
        score = 95.0 + min(5.0, ((precip_mm - 204.4) / 100.0) * 5.0)
        cat = "Extremely Heavy Rain"

    return min(100.0, max(0.0, score)), cat


def classify_compound_risk_category(score: float) -> str:
    """Map 0-100 compound score to operational severity level."""
    if score < 25.0:
        return "Low"
    elif score < 50.0:
        return "Moderate"
    elif score < 70.0:
        return "High"
    elif score < 85.0:
        return "Severe"
    else:
        return "Critical"
