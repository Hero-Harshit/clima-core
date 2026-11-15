"""
Tourism, Hospitality & Cultural Heritage Custodian Persona Adapter.
Evaluates exposed heritage monument stone surface thermal burns, pilgrimage queue heatstroke risks,
outdoor event rain cancellation probabilities, and panoramic sightseeing visual clarity.
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_tourism_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate tourist safety, heritage monument protection, pilgrimage crowd thermal stress, and hospitality event risks."""
    advisories: List[str] = []
    restrictions: List[str] = []

    t_max = inp.temp_max
    apparent_t = inp.apparent_temp_max if inp.apparent_temp_max is not None else t_max + 2.5
    solar_rad = inp.solar_radiation
    rain_mm = inp.precipitation_mm
    rain_prob = inp.rain_probability
    pm25 = inp.pm2_5

    # 1. Exposed Stone Monument Surface Heat Burn Hazard
    # Red sandstone, basalt, and marble plinths absorb high solar irradiance
    stone_surface_temp = round(float(t_max + (solar_rad / 25.0) * 16.0), 1)
    if stone_surface_temp >= 52.0 or t_max >= 42.0:
        stone_status = f"HAZARDOUS_SURFACE_BURN_RISK ({stone_surface_temp:.1f} °C)"
        stone_risk = 90.0
        restrictions.append("Archaeological Survey of India (ASI) / Monument Trust: Lay jute runners/coir mats on exposed courtyards; ban barefoot walking on stone plinths.")
        restrictions.append("Suspend outdoor guided heritage walks between 11:30 AM and 4:00 PM.")
        advisories.append("Monument surface stone reaching thermal burn temperatures; advise tourists to wear thick socks or footwear coverings.")
    elif stone_surface_temp >= 44.0:
        stone_status = f"ELEVATED_SURFACE_HEAT ({stone_surface_temp:.1f} °C)"
        stone_risk = 55.0
        advisories.append("Exposed stone walkways hot under direct sunlight; deploy water sprinkling on open monument walkways.")
    else:
        stone_status = f"SAFE_WALKING_TEMPERATURE ({stone_surface_temp:.1f} °C)"
        stone_risk = 15.0

    # 2. Pilgrim Queue Thermal Heatstroke Hazard
    if apparent_t >= 44.0 or inp.imd_heatwave_alert >= 2:
        pilgrim_status = "CRITICAL_QUEUE_HEATSTROKE_HAZARD"
        queue_risk = 95.0
        restrictions.append("Pilgrimage Temple Trusts: Prohibit unshaded queue lines exceeding 30 minutes in open courtyards.")
        advisories.append("Activate high-pressure misting fans and deploy volunteer free ORS hydration stations every 50 meters along queue barricades.")
    elif apparent_t >= 39.0 or inp.imd_heatwave_alert == 1:
        pilgrim_status = "MODERATE_HEAT_EXHAUSTION_RISK"
        queue_risk = 60.0
        advisories.append("Erect temporary tensile shade canopies over open pilgrim pathways.")
    else:
        pilgrim_status = "NORMAL_QUEUE_CONDITIONS"
        queue_risk = 20.0

    # 3. Outdoor Event & Destination Wedding Rain Cancellation Probability
    if rain_prob >= 0.70 or rain_mm >= 25.0:
        event_status = "HIGH_CANCELLATION_MANDATED"
        event_cancel_prob = min(98.0, max(80.0, rain_prob * 100.0))
        restrictions.append("Hospitality & Event Planners: Shift outdoor banquets, wedding lawns, and cultural amphitheaters to indoor halls.")
        advisories.append(f"Imminent monsoonal rain disruption ({rain_mm:.1f} mm, {rain_prob*100:.0f}% probability). Protect electronic sound and lighting gear.")
    elif rain_prob >= 0.40 or rain_mm >= 10.0:
        event_status = "MODERATE_RAIN_CONTINGENCY_NEEDED"
        event_cancel_prob = 50.0
        advisories.append("Ensure waterproof marquee covers and standby indoor backup banquet halls are on immediate standby.")
    else:
        event_status = "CLEAR_OUTDOOR_CONDITIONS"
        event_cancel_prob = 10.0

    # 4. Sightseeing Visual Clarity & Air Quality Score
    if pm25 >= 200.0:
        clarity_status = "SEVERE_SMOG_HAZE (Viewpoint Clarity Obscured)"
    elif pm25 >= 100.0:
        clarity_status = "MODERATE_HAZE"
    else:
        clarity_status = "EXCELLENT_PANORAMIC_CLARITY"

    # 5. Composite Tourism Risk Score (0 to 100)
    score = round(float(
        0.35 * queue_risk +
        0.30 * stone_risk +
        0.25 * event_cancel_prob +
        0.10 * min(100.0, (pm25 / 250.0) * 100.0)
    ), 1)

    if score >= 70.0:
        status = "CRITICAL"
    elif score >= 45.0:
        status = "HIGH_ALERT"
    elif score >= 25.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    return PersonaResult(
        persona_id="tourism_hospitality",
        persona_title="Tourism, Hospitality & Cultural Heritage Custodian",
        status_level=status,
        impact_score=score,
        key_metrics={
            "heritage_monument_surface_temp": f"{stone_surface_temp:.1f} °C ({stone_status})",
            "pilgrim_queue_heat_status": pilgrim_status,
            "apparent_heat_index": f"{apparent_t:.1f} °C",
            "outdoor_event_rain_risk": f"{event_cancel_prob:.0f} % ({event_status})",
            "forecasted_rain": f"{rain_mm:.1f} mm",
            "sightseeing_visual_clarity": clarity_status,
            "air_quality_pm25": f"{pm25:.1f} µg/m³"
        },
        advisories=advisories or ["Tourism and heritage site conditions comfortable; standard visitor protocols apply."],
        operational_restrictions=restrictions or ["No monument visitor restrictions or outdoor event cancellations active."]
    )
