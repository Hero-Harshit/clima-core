"""
Heavy Industry, Manufacturing & Thermal Power Generation Persona Adapter.
Evaluates psychrometric wet-bulb cooling tower heat rejection limits, atmospheric inversion stack flue trapping,
volatile chemical tank farm vapor pressure buildup, and boiler/transformer thermal load limits.
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def estimate_stull_wet_bulb(temp_c: float, rh_pct: float) -> float:
    """Calculate psychrometric wet-bulb temperature using Stull's empirical formula."""
    t = float(temp_c)
    rh = float(np.clip(rh_pct, 1.0, 99.0))
    tw = (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )
    return round(float(tw), 2)


def evaluate_industry_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate industrial processing, thermal power generation, and environmental emission hazards."""
    advisories: List[str] = []
    restrictions: List[str] = []

    t_max = inp.temp_max
    rh = inp.relative_humidity
    wind_kmh = inp.wind_speed_kmh
    stagnation = inp.atmospheric_stagnation
    pm25 = inp.pm2_5

    # 1. Psychrometric Wet-Bulb Cooling Tower Performance
    twb = inp.wet_bulb_temp_c if inp.wet_bulb_temp_c is not None else estimate_stull_wet_bulb(t_max, rh)
    
    # Cooling Tower Approach: Thermal rejection limit is severely degraded above 28°C
    if twb >= 30.0:
        cooling_status = "CRITICAL_COOLING_DEFICIT (Wet Bulb >= 30°C)"
        cooling_loss_pct = min(40.0, 15.0 + (twb - 28.0) * 12.0)
        restrictions.append("Thermal Power & Captive Plants: Enforce planned 15-25% turbine generation derating to prevent condenser vacuum collapse.")
        restrictions.append("Continuous chemical processes: reduce exothermic reactor feed rates.")
        advisories.append("Activate secondary auxiliary cooling water pumps and chiller booster loops.")
    elif twb >= 28.0:
        cooling_status = "ELEVATED_APPROACH_TEMPERATURE (Wet Bulb 28-30°C)"
        cooling_loss_pct = round((twb - 26.0) * 5.0, 1)
        advisories.append("Evaporative cooling tower approach compressed; monitor condenser discharge water temperature.")
    else:
        cooling_status = "OPTIMAL_HEAT_REJECTION"
        cooling_loss_pct = 0.0

    # 2. Atmospheric Inversion Stack Flue Trapping & CPCB Compliance
    if stagnation == 1 and wind_kmh <= 8.0:
        dispersion_status = "SEVERE_INVERSION_STACK_TRAPPING"
        dispersion_risk = 90.0
        restrictions.append("State Pollution Control Board (SPCB) emergency clause: Industrial diesel generator (DG) sets banned except for emergency life-safety.")
        restrictions.append("Soot blowing and scheduled baghouse filter purging strictly prohibited during nighttime inversion hours.")
        advisories.append("Continuous Emission Monitoring Systems (CEMS): Switch boiler combustion to low-excess-air mode to minimize NOx/SO2.")
    elif stagnation == 1 or wind_kmh <= 10.0:
        dispersion_status = "POOR_DISPERSION"
        dispersion_risk = 55.0
        advisories.append("Plume rise constrained; monitor boundary fence line toxic gas monitors for ground-level accumulation.")
    else:
        dispersion_status = "NORMAL_ATMOSPHERIC_DISPERSION"
        dispersion_risk = 15.0

    # 3. Volatile Chemical Storage & Tank Farm Thermal Hazard
    if t_max >= 43.0:
        tank_status = "HIGH_THERMAL_EXPANSION_RISK"
        tank_risk = 85.0
        restrictions.append("Solvent, LPG, and volatile organic compound (VOC) bulk road tanker loading prohibited between 11:30 AM and 4:30 PM.")
        advisories.append("Engage automated water deluge sprinkler cooling on white oil and naphtha storage tanks to control vapor pressure.")
    elif t_max >= 40.0:
        tank_status = "MODERATE_VAPOR_PRESSURE_SURGE"
        tank_risk = 50.0
        advisories.append("Inspect pressure-vacuum (PV) breather valves on chemical bulk storage tanks.")
    else:
        tank_status = "SAFE_THERMAL_RANGE"
        tank_risk = 10.0

    # 4. Composite Industrial Operational Risk Score (0 to 100)
    score = round(float(
        0.40 * (cooling_loss_pct * 2.5) +
        0.35 * dispersion_risk +
        0.25 * tank_risk
    ), 1)
    score = min(100.0, score)

    if score >= 70.0 or twb >= 30.5:
        status = "CRITICAL"
    elif score >= 45.0 or dispersion_risk >= 80.0:
        status = "HIGH_ALERT"
    elif score >= 25.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="industry_manufacturing",
        persona_title="Heavy Industry, Manufacturing & Thermal Power Generation",
        status_level=status,
        impact_score=score,
        key_metrics={
            "wet_bulb_temperature": f"{twb:.1f} °C",
            "cooling_tower_status": cooling_status,
            "estimated_cooling_efficiency_loss": f"{cooling_loss_pct:.1f} %",
            "stack_plume_dispersion": dispersion_status,
            "chemical_tank_farm_hazard": tank_status,
            "cpcb_cems_regulatory_risk": "HIGH_VIOLATION_PROBABILITY" if dispersion_risk >= 80.0 else "STANDARD_MONITORING",
            "ambient_tmax": f"{t_max:.1f} °C",
            "surface_wind_speed": f"{wind_kmh:.1f} km/h"
        },
        advisories=advisories or ["Industrial plant thermodynamic and environmental systems operating within baseline parameters."],
        operational_restrictions=restrictions or ["No environmental or industrial production curtailment mandates active."]
    )
