"""
Cold Chain Logistics, Perishable Food & Pharmaceutical Transit Persona Adapter.
Evaluates refrigeration unit thermodynamic compressor pull-down loads, extra fuel burn surcharges,
perishable dairy/produce spoilage vulnerability, and nocturnal transit schedule optimization.
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_cold_chain_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate refrigerated transit energy demand, cargo temperature breach risks, and highway dispatch timing."""
    advisories: List[str] = []
    restrictions: List[str] = []

    t_max = inp.temp_max
    t_min = inp.temp_min
    apparent_t = inp.apparent_temp_max if inp.apparent_temp_max is not None else t_max + 3.0
    rain_mm = inp.precipitation_mm
    pm25 = inp.pm2_5

    # 1. Refrigeration Compressor Thermal Lift & Surcharge Fuel Burn
    # Base reference ambient: 25.0°C. For ambient > 30°C, compressor fuel burn increases significantly
    thermal_lift = max(0.0, t_max - 25.0)
    extra_fuel_burn_pct = round(min(60.0, thermal_lift * 2.2), 1)

    if t_max >= 43.0 or apparent_t >= 47.0:
        compressor_status = "EXTREME_CONDENSER_STRAIN (Compressor 100% Continuous Duty)"
        compressor_risk = 95.0
        restrictions.append("Pharma & Vaccine Shipments (+2°C to +8°C): Mandatory secondary battery/dry-ice backup redundancy.")
        restrictions.append("Reefer Drivers: Prohibit opening cargo bay doors between 11:30 AM and 5:00 PM except in sealed temperature-dock bays.")
        advisories.append(f"Anticipate {extra_fuel_burn_pct}% higher auxiliary diesel fuel burn for transport refrigeration units (TRUs).")
    elif t_max >= 38.0 or apparent_t >= 42.0:
        compressor_status = "ELEVATED_THERMAL_LIFT (Duty Cycle > 85%)"
        compressor_risk = 65.0
        advisories.append("Pre-cool reefer cargo boxes to -2°C below target setpoint for 60 minutes prior to loading.")
    else:
        compressor_status = "STANDARD_REFRIGERATION_LOAD"
        compressor_risk = 20.0

    # 2. Perishable Spoilage Hazard Index (Heat + Waterlogging Delay)
    # Monsoon waterlogging or heavy smog adds highway delays, compounding heat spoilage
    delay_factor = (1.5 if rain_mm >= 35.0 else 1.0) * (1.2 if pm25 >= 200.0 else 1.0)
    spoilage_index = min(100.0, round(float((compressor_risk * 0.70 + min(30.0, rain_mm * 0.5)) * delay_factor), 1))

    if spoilage_index >= 75.0:
        spoilage_tier = "CRITICAL_SPOILAGE_RISK"
        restrictions.append("Dairy & Ice Cream Transporters: Restrict transit distance to < 150 km or postpone daytime dispatch.")
        advisories.append("Route delays from waterlogged arterial highways will cause internal temperature spikes above critical +6°C.")
    elif spoilage_index >= 45.0:
        spoilage_tier = "MODERATE_HEAT_DEGRADATION"
        advisories.append("Verify automated data-logger calibration; inspect door gasket seals for thermal leakage.")
    else:
        spoilage_tier = "LOW_TRANSIT_SPOILAGE_RISK"

    # 3. Nocturnal Dispatch Optimization Window
    diurnal_delta = t_max - t_min
    if diurnal_delta >= 10.0 and t_max >= 38.0:
        recommended_dispatch = f"NOCTURNAL_DISPATCH_MANDATED (Depart 21:30 - 05:30 to save {diurnal_delta:.1f}°C thermal lift)"
        advisories.append(f"Rescheduling dispatch to nocturnal hours reduces compressor peak load by ~{min(40.0, diurnal_delta * 2.0):.0f}%.")
    else:
        recommended_dispatch = "STANDARD_DAYTIME_DISPATCH_ACCEPTABLE"

    # 4. Composite Cold Chain Score (0 to 100)
    score = round(float(0.55 * compressor_risk + 0.30 * spoilage_index + 0.15 * min(100.0, (rain_mm / 50.0) * 100.0)), 1)
    score = min(100.0, score)

    if score >= 70.0:
        status = "CRITICAL"
    elif score >= 45.0:
        status = "HIGH_ALERT"
    elif score >= 25.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="cold_chain_logistics",
        persona_title="Cold Chain Logistics, Perishable Food & Pharmaceutical Transit",
        status_level=status,
        impact_score=score,
        key_metrics={
            "compressor_operating_status": compressor_status,
            "auxiliary_fuel_surcharge": f"+{extra_fuel_burn_pct:.1f} %",
            "perishable_spoilage_tier": spoilage_tier,
            "spoilage_vulnerability_score": f"{spoilage_index:.1f} / 100",
            "ambient_tmax": f"{t_max:.1f} °C",
            "apparent_heat_index": f"{apparent_t:.1f} °C",
            "recommended_dispatch_window": recommended_dispatch,
            "pharma_vaccine_protocol": "BACKUP_ACTIVE_REDUNDANCY" if t_max >= 42.0 else "STANDARD_MONITORING"
        },
        advisories=advisories or ["Cold chain transit conditions stable; standard reefer operating parameters sufficient."],
        operational_restrictions=restrictions or ["No freight dispatch prohibitions or mandatory nocturnal rescheduling active."]
    )
