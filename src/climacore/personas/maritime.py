"""
Maritime, Commercial Port & Marine Fisheries Persona Adapter.
Evaluates Beaufort sea state wind limits, IMD cyclone track proximity buffers (100km, 250km, 500km),
harbor gantry crane cutoff thresholds, and coastal vessel sailing prohibitions.
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_maritime_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate marine navigation, harbor operations, and artisanal fishery hazards."""
    advisories: List[str] = []
    restrictions: List[str] = []

    wind_speed = inp.wind_speed_kmh
    gust_speed = inp.gale_wind_gust_kmh if inp.gale_wind_gust_kmh is not None else wind_speed * 1.35
    rain_mm = inp.precipitation_mm
    cyc_dist = inp.cyclone_distance_km
    cyc_grade = inp.cyclone_grade or "UNSPECIFIED"

    # 1. Beaufort Scale & Marine Sea State Classification
    if gust_speed >= 88.0 or wind_speed >= 65.0:
        sea_state = "VERY_ROUGH_TO_HIGH (Beaufort Force 8-10, Gale to Storm)"
        sea_risk = 95.0
        restrictions.append("Complete prohibition on all fishing trawlers, coastal ferries, and unmoored barges entering the sea.")
        restrictions.append("Commercial Port Trusts: suspend all ship berthing, pilotage, and anchorages immediately.")
        advisories.append("Secure double-spring mooring lines on all docked commercial vessels.")
    elif gust_speed >= 60.0 or wind_speed >= 45.0:
        sea_state = "ROUGH (Beaufort Force 6-7, Strong Breeze to Near Gale)"
        sea_risk = 70.0
        restrictions.append("Artisanal non-motorized and small motorized fishing boats prohibited from venturing beyond 5 nautical miles.")
        advisories.append("Coast Guard and State Fisheries Department: initiate radio recall broadcasts to high-seas vessels.")
    elif wind_speed >= 30.0:
        sea_state = "MODERATE (Beaufort Force 4-5)"
        sea_risk = 40.0
        advisories.append("Exercise caution in outer harbor channels; choppy waves and reduced small-craft stability anticipated.")
    else:
        sea_state = "SLIGHT_TO_CALM (Beaufort Force 1-3)"
        sea_risk = 15.0

    # 2. IMD Cyclone Proximity Buffer Engine
    cyc_impact = 0.0
    cyc_desc = "NO_CYCLONE_HAZARD"
    if cyc_dist is not None:
        if cyc_dist <= 100.0:
            cyc_desc = f"DIRECT_STRIKE_ZONE ({cyc_dist:.1f} km, Grade: {cyc_grade})"
            cyc_impact = 100.0
            restrictions.append(f"Port Trust: Issue Local Cautionary Signal No. 10/11 (Great Danger). Evacuate low-lying docks.")
            advisories.append(f"Storm surge inundation anticipated along coastal harbors within {cyc_dist:.1f} km.")
        elif cyc_dist <= 250.0:
            cyc_desc = f"DESTRUCTIVE_GALE_RADIUS ({cyc_dist:.1f} km, Grade: {cyc_grade})"
            cyc_impact = 80.0
            restrictions.append(f"Port Trust: Hoist Danger Signal No. 8/9. Cease oil bunkering and hazardous cargo handling.")
            advisories.append(f"High-frequency squally weather and heavy coastal outer squalls expected.")
        elif cyc_dist <= 500.0:
            cyc_desc = f"OUTER_CONVECTIVE_RAINBAND ({cyc_dist:.1f} km, Grade: {cyc_grade})"
            cyc_impact = 45.0
            advisories.append(f"Cyclonic system detected {cyc_dist:.1f} km offshore. Monitor 3-hourly RSMC bulletin updates.")
        else:
            cyc_desc = f"REMOTE_SYSTEM ({cyc_dist:.1f} km)"
            cyc_impact = 10.0

    # 3. Port Crane & Cargo Operation Threshold
    # Standard engineering cutoff for container quay cranes is 60-65 km/h (17-18 m/s)
    crane_cutoff = gust_speed >= 60.0 or wind_speed >= 50.0
    if crane_cutoff:
        restrictions.append("Container Quay Cranes (STS) & Rubber-Tired Gantry (RTG) operations MUST HALT due to wind load limits.")
        advisories.append("Lock storm pins and rail clamps on all harbor cranes to prevent wind runaway.")

    # 4. Overall Maritime Composite Severity (0 to 100)
    composite_score = round(float(
        0.50 * max(sea_risk, cyc_impact) +
        0.30 * min(100.0, (gust_speed / 90.0) * 100.0) +
        0.20 * min(100.0, (rain_mm / 60.0) * 100.0)
    ), 1)

    if composite_score >= 75.0 or (cyc_dist is not None and cyc_dist <= 150.0):
        status = "CRITICAL"
    elif composite_score >= 50.0 or crane_cutoff:
        status = "HIGH_ALERT"
    elif composite_score >= 30.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="maritime_coastal",
        persona_title="Maritime, Commercial Port & Marine Fisheries",
        status_level=status,
        impact_score=composite_score,
        key_metrics={
            "beaufort_sea_state": sea_state,
            "surface_wind_speed": f"{wind_speed:.1f} km/h",
            "peak_gust_speed": f"{gust_speed:.1f} km/h",
            "cyclone_proximity_status": cyc_desc,
            "container_crane_operations": "SHUTDOWN_MANDATED" if crane_cutoff else "OPERATIONAL",
            "coastal_fishing_clearance": "SAILING_PROHIBITED" if (sea_risk >= 70.0 or (cyc_dist and cyc_dist <= 250.0)) else "SAIL_WITH_CAUTION" if sea_risk >= 40.0 else "CLEAR_TO_SAIL",
            "forecasted_coastal_rain": f"{rain_mm:.1f} mm"
        },
        advisories=advisories or ["Marine coastal conditions normal; standard navigation watch required."],
        operational_restrictions=restrictions or ["No maritime or port operational restrictions active."]
    )
