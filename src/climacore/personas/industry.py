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

    # 1. Elevation and Pressure-Aware Psychrometric Wet-Bulb Cooling Tower Performance
    from climacore.analytics.psychrometrics import calculate_elevation_aware_wet_bulb
    tw_res = calculate_elevation_aware_wet_bulb(t_max, rh, city=inp.city)
    twb = inp.wet_bulb_temp_c if inp.wet_bulb_temp_c is not None else tw_res["wet_bulb_temp_c"]
    
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

    # 4. Regional Power Plant Airshed & Grid Stability Hazards
    if inp.grid_blackout_hazard_prob is not None and inp.grid_blackout_hazard_prob >= 0.30:
        advisories.append(
            f"Regional Grid Stability Warning: DISCOM deficit hazard {inp.grid_blackout_hazard_prob:.1%}. Ensure captive co-generation synchronized and prepare for emergency frequency/voltage drops."
        )
    if inp.thermal_capacity_100km_mw is not None and inp.thermal_capacity_100km_mw >= 10000.0 and dispersion_risk >= 80.0:
        advisories.append(
            f"Point-Source Emission Confluence: Regional thermal capacity ({inp.thermal_capacity_100km_mw:,.0f} MW within 100km) combined with severe inversion mandates proactive flue-gas throttling."
        )

    # 5. Composite Industrial Operational Risk Score (0 to 100)
    score = round(float(
        0.40 * (cooling_loss_pct * 2.5) +
        0.35 * dispersion_risk +
        0.25 * tank_risk
    ), 1)
    if inp.grid_blackout_hazard_prob is not None:
        score = max(score, round(float(inp.grid_blackout_hazard_prob * 100.0 * 0.8), 1))
    score = min(100.0, score)

    if score >= 70.0 or twb >= 30.5:
        status = "CRITICAL"
    elif score >= 45.0 or dispersion_risk >= 80.0:
        status = "HIGH_ALERT"
    elif score >= 25.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    metrics: Dict[str, Any] = {
        "wet_bulb_temperature": f"{twb:.1f} °C",
        "station_pressure": f"{tw_res['station_pressure_hpa']} hPa (Elev: {tw_res['elevation_m']:.0f}m)",
        "cooling_tower_status": cooling_status,
        "estimated_cooling_efficiency_loss": f"{cooling_loss_pct:.1f} %",
        "stack_plume_dispersion": dispersion_status,
        "chemical_tank_farm_hazard": tank_status,
        "cpcb_cems_regulatory_risk": "HIGH_VIOLATION_PROBABILITY" if dispersion_risk >= 80.0 else "STANDARD_MONITORING",
        "ambient_tmax": f"{t_max:.1f} °C",
        "surface_wind_speed": f"{wind_kmh:.1f} km/h"
    }

    if inp.thermal_capacity_100km_mw is not None:
        metrics["regional_thermal_capacity_100km"] = f"{inp.thermal_capacity_100km_mw:,.0f} MW ({inp.thermal_plants_within_100km or 0} plants)"
        if inp.closest_thermal_plant:
            metrics["closest_thermal_plant"] = inp.closest_thermal_plant

    return PersonaResult(
        persona_id="industry_manufacturing",
        persona_title="Heavy Industry, Manufacturing & Thermal Power Generation",
        status_level=status,
        impact_score=score,
        key_metrics=metrics,
        advisories=advisories or ["Industrial plant thermodynamic and environmental systems operating within baseline parameters."],
        operational_restrictions=restrictions or ["No environmental or industrial production curtailment mandates active."]
    )
