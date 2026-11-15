"""
Daily Commuter & Office Worker Persona Adapter.
Evaluates urban transit disruption, smog-induced low visibility,
evening flash waterlogging on arterial roads, and vehicle thermal overheating stress.
"""

from typing import Dict, Any, List
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_commuter_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate urban transit and commute disruption risks for daily office travelers."""
    advisories = []
    restrictions = []

    # 1. Smog / Fog Visibility Reduction (Winter morning commute: 7:30 - 9:30 AM)
    pm25 = inp.pm2_5
    rh = inp.relative_humidity
    is_stagnant = bool(inp.atmospheric_stagnation)

    # Visibility index: high aerosol + moisture + stagnant boundary layer
    if (pm25 >= 200.0 or (pm25 >= 120.0 and rh >= 80.0)) and is_stagnant:
        visibility_status = "CRITICALLY_LOW (<200m)"
        restrictions.append("Dense smog/fog hazard on highways and expressways. Reduce vehicular speed.")
        advisories.append("Use low-beam headlights and maintain doubled following distance.")
    elif pm25 >= 120.0:
        visibility_status = "REDUCED (200m - 500m)"
        advisories.append("Moderate haze and reduced visibility during morning rush hour.")
    else:
        visibility_status = "CLEAR (>1000m)"

    # 2. Arterial Road Waterlogging & Transit Delay (Evening commute: 5:00 - 8:30 PM)
    precip = inp.precipitation_mm
    rain_prob = inp.rain_probability

    if precip >= 40.0 and rain_prob >= 0.5:
        transit_delay = "SEVERE_GRIDLOCK (+60 to 90 mins)"
        restrictions.append("Major city underpasses and key intersections at risk of severe inundation.")
        advisories.append("Consider remote work (WFH) or depart before 4:00 PM to avoid commute gridlock.")
    elif precip >= 15.0 and rain_prob >= 0.4:
        transit_delay = "MODERATE_DELAY (+20 to 40 mins)"
        advisories.append("Waterlogging likely on secondary routes. Expect slowed traffic.")
    else:
        transit_delay = "NORMAL_FLOW"

    # 3. Vehicle Engine & Cabin Heat Stress
    t_max = inp.temp_max
    if t_max >= 43.0:
        cabin_heat = "EXTREME"
        advisories.append("Avoid leaving children, pets, or electronics in parked vehicles. Cabin interior can reach 60°C within 15 minutes.")
        advisories.append("Engine cooling systems under stress in stop-and-go traffic; check coolant and tire pressure.")
    elif t_max >= 39.0:
        cabin_heat = "HIGH"
    else:
        cabin_heat = "NORMAL"

    # Compute Commuter Disruption Score (0 to 100)
    vis_score = 100.0 if "CRITICALLY" in visibility_status else (50.0 if "REDUCED" in visibility_status else 10.0)
    rain_score = min(100.0, (precip / 40.0) * 100.0)
    heat_score = min(100.0, max(0.0, (t_max - 35.0) / (45.0 - 35.0) * 100.0))
    commute_score = round(float(0.40 * rain_score + 0.35 * vis_score + 0.25 * heat_score), 1)

    if commute_score >= 75.0:
        status = "CRITICAL"
    elif commute_score >= 50.0:
        status = "HIGH_ALERT"
    elif commute_score >= 30.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="daily_commuter",
        persona_title="Daily Commuter & Office Worker",
        status_level=status,
        impact_score=commute_score,
        key_metrics={
            "commute_visibility": visibility_status,
            "estimated_transit_delay": transit_delay,
            "vehicle_cabin_heat_stress": cabin_heat,
            "forecasted_pm25": f"{pm25:.1f} µg/m³",
            "forecasted_rain": f"{precip:.1f} mm",
            "peak_afternoon_temp": f"{t_max:.1f} °C",
        },
        advisories=advisories or ["Normal traffic and transit conditions. No significant weather delays anticipated."],
        operational_restrictions=restrictions or ["No transit restrictions active."],
    )
