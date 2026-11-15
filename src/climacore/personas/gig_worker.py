"""
Gig Worker & Outdoor Labor Persona Adapter.
Evaluates safe working hours, heat exhaustion risk (ILO/OSHA WBGT standards),
hydration requirements, and two-wheeler delivery flood transit risks.
"""

import math
from typing import Dict, Any, List
from climacore.personas.models import PersonaInput, PersonaResult


def estimate_wbgt(temp_c: float, rh_pct: float, solar_rad: float = 20.0, wind_kmh: float = 12.0) -> float:
    """
    Estimate Wet-Bulb Globe Temperature (WBGT in °C) for outdoor heavy physical work.
    Uses simplified natural wet bulb and dry bulb formulation with solar radiation load.
    """
    # Saturated vapor pressure (hPa)
    vp_sat = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
    vp_act = (rh_pct / 100.0) * vp_sat
    # Outdoor WBGT approximation
    wbgt_approx = 0.567 * temp_c + 0.393 * vp_act + 3.94 + min(4.0, (solar_rad / 25.0) * 3.0) - min(2.0, (wind_kmh / 20.0) * 1.5)
    return round(float(wbgt_approx), 1)


def evaluate_gig_worker_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate occupational thermal stress and transit risks for outdoor workers."""
    advisories = []
    restrictions = []

    t_max = inp.temp_max
    rh = inp.relative_humidity
    precip = inp.precipitation_mm

    # 1. Compute Outdoor WBGT
    wbgt = estimate_wbgt(t_max, rh, inp.solar_radiation, inp.wind_speed_kmh)

    # 2. ILO / OSHA Work-Rest Cycle (Minutes of Work per Hour during peak afternoon)
    if wbgt >= 33.0 or inp.imd_heatwave_alert >= 2:
        work_mins = 15
        rest_mins = 45
        work_tier = "EXTREME_RESTRICTION"
        lost_hours = 4.5
        restrictions.append("Mandate 45 minutes of shaded rest per hour between 12:00 PM and 4:00 PM.")
        restrictions.append("Employers/platforms must suspend delivery SLA penalties during peak heat hours.")
    elif wbgt >= 31.0 or inp.imd_heatwave_alert == 1:
        work_mins = 30
        rest_mins = 30
        work_tier = "HIGH_RESTRICTION"
        lost_hours = 3.0
        restrictions.append("Mandate 30 minutes of shaded rest per hour between 12:30 PM and 3:30 PM.")
    elif wbgt >= 28.0:
        work_mins = 45
        rest_mins = 15
        work_tier = "MODERATE_RESTRICTION"
        lost_hours = 1.5
        advisories.append("Take 15 minutes rest in shade every hour.")
    else:
        work_mins = 60
        rest_mins = 0
        work_tier = "UNRESTRICTED"
        lost_hours = 0.0

    # 3. Hydration Requirement
    if wbgt >= 30.0:
        water_rate = "1.0 Litre/hour with oral rehydration salts (ORS)"
        advisories.append(f"Drink at least {water_rate}. Do not wait until thirsty.")
    elif wbgt >= 26.0:
        water_rate = "750 ml/hour"
        advisories.append(f"Drink at least {water_rate} to replace sweat losses.")
    else:
        water_rate = "500 ml/hour"

    # 4. Two-Wheeler / Delivery Road Flash Flood Warning
    road_hazard = "LOW"
    if precip >= 35.0 and inp.rain_probability >= 0.5:
        road_hazard = "CRITICAL_FLOODING"
        restrictions.append("Avoid low-lying underpasses and arterial roads with high waterlogging risk.")
        advisories.append("Delivery riders: waterproof electronics and seek safe elevated shelter.")
    elif precip >= 15.0:
        road_hazard = "MODERATE_WATERLOGGING"
        advisories.append("Slick road conditions and reduced braking efficiency.")

    # 5. Air Pollution Respiratory Burden
    if inp.pm2_5 >= 150.0:
        advisories.append("Severe particulate exposure for outdoor delivery riders. Wear well-fitted N95 masks.")

    # Overall Labor Vulnerability Score (0 to 100)
    heat_labor_score = min(100.0, max(0.0, (wbgt - 24.0) / (34.0 - 24.0) * 100.0))
    rain_labor_score = min(100.0, (precip / 45.0) * 100.0)
    labor_score = round(float(0.65 * heat_labor_score + 0.35 * rain_labor_score), 1)

    if labor_score >= 75.0 or work_tier == "EXTREME_RESTRICTION":
        status = "CRITICAL"
    elif labor_score >= 50.0 or work_tier == "HIGH_RESTRICTION":
        status = "HIGH_ALERT"
    elif labor_score >= 30.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="gig_worker_labor",
        persona_title="Gig Worker & Outdoor Labor",
        status_level=status,
        impact_score=labor_score,
        key_metrics={
            "estimated_wbgt": f"{wbgt:.1f} °C",
            "hourly_work_schedule": f"{work_mins} min work / {rest_mins} min rest",
            "lost_productive_work_hours": f"{lost_hours:.1f} hours",
            "recommended_hydration": water_rate,
            "delivery_road_flood_risk": road_hazard,
            "respiratory_inhalation_risk": "Severe" if inp.pm2_5 >= 200 else ("High" if inp.pm2_5 >= 100 else "Moderate"),
        },
        advisories=advisories or ["Normal outdoor working conditions. Maintain routine hydration."],
        operational_restrictions=restrictions or ["No outdoor labor restrictions active."],
    )
