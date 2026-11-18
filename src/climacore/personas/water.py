"""
Water Utility, Municipal Stormwater & Aquifer Management Persona Adapter.
Evaluates urban runoff coefficients, stormwater drain pre-desilting alerts, sewage treatment plant (STP)
hydraulic surges, CWC reservoir storage buffers, and CGWB groundwater depletion levels.
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_water_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate municipal water supply, stormwater drainage, and sewage treatment plant operational risks."""
    advisories: List[str] = []
    restrictions: List[str] = []

    rain_mm = inp.precipitation_mm
    rain_prob = inp.rain_probability
    smdi = inp.soil_moisture_deficit_index if inp.soil_moisture_deficit_index is not None else 0.0
    res_pct = inp.reservoir_storage_percent if inp.reservoir_storage_percent is not None else 62.0
    gw_depth = inp.groundwater_depletion_level if inp.groundwater_depletion_level is not None else 14.5

    # 1. Urban Soil Saturation & Runoff Coefficient (Rational Method)
    # SMDI > 1.0 indicates wet/saturated soil; SMDI < -1.0 indicates dry deficit
    if smdi >= 1.5:
        runoff_coeff = 0.90
        sat_status = "FULLY_SATURATED_SOIL (Extreme Runoff Infiltration Barrier)"
    elif smdi >= 0.5:
        runoff_coeff = 0.72
        sat_status = "MODERATELY_SATURATED_SOIL"
    elif smdi <= -1.5:
        runoff_coeff = 0.35
        sat_status = "PARCHED_SOIL (High Initial Infiltration Capacity)"
    else:
        runoff_coeff = 0.55
        sat_status = "NORMAL_ABSORPTION_CAPACITY"

    # 2. Stormwater Drain Desilting & Underpass Flash Flooding
    effective_runoff_volume = rain_mm * runoff_coeff
    if effective_runoff_volume >= 35.0 or (rain_mm >= 50.0 and rain_prob >= 0.60):
        drain_status = "CRITICAL_DRAIN_SURCHARGE_RISK"
        drain_risk = 95.0
        restrictions.append("Municipal Drain Division: Deploy mobile diesel de-watering pumps to all known low-lying underpasses.")
        advisories.append(f"High effective runoff ({effective_runoff_volume:.1f} mm) will overwhelm municipal gravity drainage channels.")
    elif effective_runoff_volume >= 18.0 or rain_mm >= 30.0:
        drain_status = "ELEVATED_RUNOFF_CHOKE_RISK"
        drain_risk = 60.0
        advisories.append("Pre-emptive trash-screen desilting mandated at primary nallah outfalls within 24 hours.")
    else:
        drain_status = "NORMAL_DRAINAGE_CAPACITY"
        drain_risk = 15.0

    # 3. Municipal Raw Water Supply & Dam Reservoir Buffer
    if res_pct < 20.0 or gw_depth >= 35.0:
        supply_status = "SEVERE_WATER_DEFICIT (Rationing Mandated)"
        supply_risk = 90.0
        restrictions.append("State Water Board: Mandate 20% municipal piped water rationing; prohibit potable water for vehicle washing.")
        advisories.append("Mobilize deep-bore emergency groundwater extraction tankers for vulnerable urban wards.")
    elif res_pct < 35.0 or gw_depth >= 25.0:
        supply_status = "VULNERABLE_STORAGE_RESERVE"
        supply_risk = 55.0
        advisories.append("Reservoir live storage below seasonal baseline. Prioritize drinking water over industrial allocations.")
    else:
        supply_status = "ADEQUATE_STORAGE_BUFFER"
        supply_risk = 15.0

    # 4. Sewage Treatment Plant (STP) Hydraulic Inflow Surge
    if rain_mm >= 60.0:
        restrictions.append("STP Operators: Prepare storm diversion bypass gates to protect biological secondary aeration basins.")

    # 5. India Flood Inventory (IFI) Calibrated Flood Disaster Model
    flood_prob = inp.flood_disaster_risk_prob
    if flood_prob is not None and flood_prob >= 0.35:
        drain_risk = max(drain_risk, 92.0)
        restrictions.append(f"IFI DISASTER ALERT: District flood disaster probability elevated to {flood_prob:.1%}. Pre-deploy emergency de-watering pump sets to low-lying drainage basins.")
        advisories.append("Hydrological saturation warning: Multi-day antecedent precipitation and drainage canal constraints indicate severe inundation vulnerability.")

    # 6. Composite Water Utility Risk Score (0 to 100)
    score = round(float(
        0.50 * drain_risk +
        0.30 * supply_risk +
        0.20 * min(100.0, (rain_mm / 60.0) * 100.0)
    ), 1)

    if score >= 70.0 or drain_risk >= 90.0:
        status = "CRITICAL"
    elif score >= 45.0 or drain_risk >= 60.0:
        status = "HIGH_ALERT"
    elif score >= 25.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    metrics: Dict[str, Any] = {
        "urban_runoff_coefficient": f"{runoff_coeff:.2f}",
        "soil_saturation_state": sat_status,
        "stormwater_drain_status": drain_status,
        "forecasted_rain_volume": f"{rain_mm:.1f} mm",
        "effective_surface_runoff": f"{effective_runoff_volume:.1f} mm",
        "reservoir_live_storage": f"{res_pct:.1f} % ({supply_status})",
        "groundwater_depth_mbgl": f"{gw_depth:.1f} mbgl",
        "drain_desilting_alert": "MANDATORY_IMMEDIATE_ACTION" if drain_risk >= 60.0 else "ROUTINE_SCHEDULE",
        "ifi_flood_disaster_risk": f"{flood_prob:.1%}" if flood_prob is not None else "N/A",
    }

    if inp.drainage_canals_15km is not None and inp.drainage_canals_15km > 0:
        metrics["mapped_drainage_channels"] = (
            f"{inp.drainage_canals_15km:,} canals (ratio {inp.drainage_arterial_ratio or 0.0:.2f} to arterial roads)"
        )
        if rain_mm >= 30.0:
            advisories.append(
                f"Stormwater operational dispatch: Clear outfall bottlenecks along {inp.drainage_canals_15km:,} mapped natural/engineered canals."
            )

    return PersonaResult(
        persona_id="water_utilities",
        persona_title="Water Utility, Municipal Stormwater & Aquifer Management",
        status_level=status,
        impact_score=score,
        key_metrics=metrics,
        advisories=advisories or ["Municipal water reservoirs and drainage systems operating within stable parameters."],
        operational_restrictions=restrictions or ["No water rationing or stormwater emergency diversions active."]
    )
