"""
Aviation, Airport Operations & Airside Ground Safety Persona Adapter.
Evaluates Runway Visual Range (RVR / CAT-III ILS low-visibility procedures), high-density altitude aircraft
takeoff weight penalties (hot & high operations), low-level wind shear gusts, and apron tarmac thermal limits.
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def estimate_optical_visibility_m(pm25: float, pm10: float, rh: float) -> float:
    """Estimate optical surface visibility in meters using Koschmieder's aerosol formulation."""
    # Hygroscopic aerosol swelling multiplier
    f_rh = 1.0 + 0.25 * ((rh / 100.0) ** 3) if rh > 60.0 else 1.0
    beta_ext = (0.0035 * pm25 + 0.0008 * (pm10 - pm25 if pm10 > pm25 else 0.0)) * f_rh + 0.015
    vis_km = 3.912 / beta_ext
    return float(np.clip(vis_km * 1000.0, 100.0, 10000.0))


def evaluate_aviation_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate flight operations, runway instrument procedures, and airside ground safety."""
    advisories: List[str] = []
    restrictions: List[str] = []

    t_max = inp.temp_max
    rh = inp.relative_humidity
    wind_kmh = inp.wind_speed_kmh
    gust_kmh = inp.gale_wind_gust_kmh if inp.gale_wind_gust_kmh is not None else wind_kmh * 1.35
    pm25 = inp.pm2_5
    pm10 = inp.pm10 if inp.pm10 is not None else pm25 * 1.7
    stagnation = inp.atmospheric_stagnation

    # 1. Optical Visibility & Instrument Landing System (ILS) Category
    vis_m = estimate_optical_visibility_m(pm25, pm10, rh)
    # Severe winter stagnation inversion clamps visibility
    if stagnation == 1 and rh >= 75.0 and pm25 >= 180.0:
        vis_m = min(vis_m, 250.0)

    if vis_m < 200.0:
        ils_cat = "CAT-IIIb (Runway Visual Range < 200m)"
        vis_risk = 95.0
        restrictions.append("Airport Operator: Initiate Low Visibility Procedures (LVP Phase 2). Ground vehicle movements restricted on taxiways.")
        restrictions.append("Airlines: Mandatory diversions for non-CAT-III certified airframes and flight crews.")
        advisories.append("Severe departure flight slot congestion anticipated; tarmac hold delays exceeding 90 minutes.")
    elif vis_m < 550.0:
        ils_cat = "CAT-II / CAT-IIIa (RVR 200m - 550m)"
        vis_risk = 70.0
        advisories.append("LVP Phase 1 active. Air Traffic Control (ATC) spacing between landing aircraft increased by 5-8 nautical miles.")
    elif vis_m < 1500.0:
        ils_cat = "CAT-I Instrument Approach (RVR 550m - 1500m)"
        vis_risk = 35.0
    else:
        ils_cat = "VFR / Normal Visual Operations (> 1500m)"
        vis_risk = 10.0

    # 2. High-Density Altitude & Takeoff Weight Penalty (Hot & High operations)
    # Standard sea level ISA temperature is 15.0°C. Density altitude expands significantly above 38°C
    if t_max >= 44.0:
        da_status = "EXTREME_DENSITY_ALTITUDE_PENALTY"
        mtow_penalty_pct = round(float((t_max - 35.0) * 0.85), 1)
        restrictions.append(f"Flight Dispatchers: Enforce {mtow_penalty_pct}% Maximum Takeoff Weight (MTOW) reduction. Restrict belly cargo.")
        advisories.append("Reduced climb gradient performance post-takeoff; runway length requirement extended by up to 25%.")
    elif t_max >= 40.0:
        da_status = "MODERATE_PAYLOAD_RESTRICTION"
        mtow_penalty_pct = round(float((t_max - 35.0) * 0.50), 1)
        advisories.append("High surface temperatures reducing engine takeoff thrust margins. Check performance charts before flight release.")
    else:
        da_status = "NORMAL_AIRCRAFT_PERFORMANCE"
        mtow_penalty_pct = 0.0

    # 3. Airside Apron Tarmac Thermal Stress
    # Dark asphalt apron surface absorbs solar radiation: T_tarmac ≈ T_max + 14°C
    tarmac_temp = round(float(t_max + 14.0 if inp.solar_radiation >= 18.0 else t_max + 8.0), 1)
    if tarmac_temp >= 54.0 or t_max >= 42.0:
        tarmac_status = f"HAZARDOUS_HEAT_STRESS ({tarmac_temp:.1f} °C)"
        restrictions.append("Ground Handlers & Ramp Agents: Mandate 15-minute rest breaks inside air-conditioned break rooms per 45 minutes of ramp duty.")
        advisories.append("Equip refueling and baggage handling personnel with high-insulation safety boots and electrolyte packs.")
    elif tarmac_temp >= 46.0:
        tarmac_status = f"ELEVATED_TARMAC_HEAT ({tarmac_temp:.1f} °C)"
    else:
        tarmac_status = f"NORMAL ({tarmac_temp:.1f} °C)"

    # 4. Wind Shear & Crosswind Advisory
    if gust_kmh >= 55.0 or wind_kmh >= 40.0:
        restrictions.append("Airport Duty Manager: Issue Low-Level Wind Shear (LLWS) and convective crosswind advisory on ATIS frequency.")
        advisories.append("Light aircraft operations suspended; baggage containers and equipment dollies must be secured.")

    # 5. Composite Aviation Risk Score (0 to 100)
    score = round(float(
        0.45 * vis_risk +
        0.30 * min(100.0, (mtow_penalty_pct / 10.0) * 100.0) +
        0.25 * min(100.0, (gust_kmh / 65.0) * 100.0)
    ), 1)

    if score >= 70.0 or vis_m < 200.0:
        status = "CRITICAL"
    elif score >= 45.0 or vis_m < 550.0:
        status = "HIGH_ALERT"
    elif score >= 25.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="aviation_airport",
        persona_title="Aviation, Airport Operations & Airside Ground Safety",
        status_level=status,
        impact_score=score,
        key_metrics={
            "runway_visual_range_m": f"{int(vis_m)} m",
            "ils_operating_category": ils_cat,
            "density_altitude_status": da_status,
            "mtow_payload_penalty": f"-{mtow_penalty_pct:.1f} %" if mtow_penalty_pct > 0 else "None",
            "apron_tarmac_surface_temp": f"{tarmac_temp:.1f} °C ({tarmac_status})",
            "surface_wind_speed": f"{wind_kmh:.1f} km/h",
            "peak_gust_wind": f"{gust_kmh:.1f} km/h",
            "low_visibility_procedures": "LVP_ACTIVE" if vis_m < 550.0 else "INACTIVE"
        },
        advisories=advisories or ["Airfield meteorological conditions clear; normal visual flight operations permitted."],
        operational_restrictions=restrictions or ["No airport operational constraints or payload penalties active."]
    )
