"""
City Official, Disaster Management & Infrastructure Persona Adapter.
Evaluates municipal power grid cooling demand surge, distribution transformer thermal failure risk,
44-year historical rarity / return period declarations, and municipal emergency mitigation deployments.
"""

from typing import Dict, Any, List
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_city_official_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate civil protection, utility grid, and municipal infrastructure risks."""
    advisories = []
    restrictions = []

    t_max = inp.temp_max
    t_min = inp.temp_min
    t_mean = (t_max + t_min) / 2.0
    precip = inp.precipitation_mm
    pm25 = inp.pm2_5

    # 1. Cooling Degree Days (CDD) & Power Grid Peak AC Demand
    # Standard engineering reference base: 18.0°C
    cdd = max(0.0, t_mean - 18.0)
    # Tropical Night Indicator: If night temp stays above 28°C, transformers cannot dissipate heat
    is_tropical_night = t_min >= 28.0

    # Grid Stress Score (0 to 100)
    grid_stress = min(100.0, (cdd / 16.0) * 70.0 + (30.0 if is_tropical_night else 0.0))

    if grid_stress >= 75.0 or (is_tropical_night and t_max >= 43.0):
        grid_status = "CRITICAL_BLACKOUT_RISK"
        restrictions.append("State DISCOMs: mandate spinning reserve power buffers; schedule non-emergency industrial load shedding.")
        advisories.append("Pre-position mobile diesel generator units at major government hospitals and water pump stations.")
    elif grid_stress >= 50.0:
        grid_status = "ELEVATED_COOLING_SURGE"
        advisories.append("Peak residential air conditioning surge anticipated between 1:30 PM - 5:00 PM and 10:00 PM - 2:00 AM.")
    else:
        grid_status = "STABLE"

    # 2. 44-Year Statistical Rarity & Return Period Declaration
    # Translates empirical historical percentile into civil defense language
    pct = inp.historical_percentile
    if pct >= 99.5:
        rarity_desc = "UNPRECEDENTED_HISTORICAL_EVENT (>= 1-in-25 Year Extreme)"
        restrictions.append("Activate District Disaster Management Authority (DDMA) Emergency Operations Center.")
    elif pct >= 98.0:
        rarity_desc = "SEVERE_CLIMATE_ANOMALY (>= 1-in-10 Year Extreme)"
        advisories.append("Mobilize quick-response civil defense and municipal pumping teams.")
    elif pct >= 90.0:
        rarity_desc = "TOP_DECIMAL_ANOMALY (1-in-2 to 1-in-5 Year)"
    else:
        rarity_desc = "TYPICAL_SEASONAL_VARIABILITY"

    # 3. Municipal Emergency Response Deployment Tier
    if inp.imd_heatwave_alert >= 2 or pm25 >= 250.0 or precip >= 65.0:
        tier = "STAGE_3_FULL_EMERGENCY"
        restrictions.append("Enforce ban on non-essential construction and demolition (C&D) activities.")
        restrictions.append("Deploy anti-smog guns and mechanical water sweepers on the top 10 most congested corridors.")
        advisories.append("Open air-conditioned municipal community cooling centers in high-density informal settlement zones.")
    elif inp.imd_heatwave_alert == 1 or pm25 >= 150.0 or precip >= 35.0:
        tier = "STAGE_2_TARGETED_DEPLOYMENT"
        advisories.append("Enforce strict dust suppression at construction sites and intensify traffic light synchronization.")
    else:
        tier = "STAGE_1_ROUTINE_MONITORING"

    # Overall Municipal Risk Score (0 to 100)
    muni_score = round(float(0.40 * grid_stress + 0.35 * min(100.0, (pm25 / 200.0) * 100.0) + 0.25 * min(100.0, (precip / 50.0) * 100.0)), 1)

    if muni_score >= 75.0 or tier == "STAGE_3_FULL_EMERGENCY":
        status = "CRITICAL"
    elif muni_score >= 50.0 or tier == "STAGE_2_TARGETED_DEPLOYMENT":
        status = "HIGH_ALERT"
    elif muni_score >= 30.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="city_official_infrastructure",
        persona_title="City Official, Municipal Planner & Disaster Management",
        status_level=status,
        impact_score=muni_score,
        key_metrics={
            "power_grid_cooling_stress": f"{grid_stress:.1f} / 100 ({grid_status})",
            "cooling_degree_days_cdd": f"{cdd:.1f} °C-days",
            "tropical_night_failure_risk": "ACTIVE (Min Temp >= 28°C)" if is_tropical_night else "INACTIVE",
            "historical_climate_rarity": rarity_desc,
            "municipal_action_tier": tier,
            "forecasted_pm25": f"{pm25:.1f} µg/m³",
            "forecasted_rain": f"{precip:.1f} mm",
            "imd_heat_action_plan_tier": ["None", "Yellow Alert", "Orange Alert", "Red Alert"][min(3, inp.imd_heatwave_alert)],
        },
        advisories=advisories or ["Municipal infrastructure systems operating within standard capacity limits."],
        operational_restrictions=restrictions or ["No civil protection restrictions active."],
    )
