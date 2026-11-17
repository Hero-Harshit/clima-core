"""
Student & Parent Persona Adapter.
Evaluates morning school assembly safety, afternoon heatstroke risk during dismissal,
and school commute transit flood waterlogging.
"""

from typing import Dict, Any, List
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_student_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate risks specifically impacting school/college students and parents."""
    advisories = []
    restrictions = []

    # 1. Morning School Assembly Safety (7:30 AM - 9:00 AM)
    # PM2.5 threshold per Indian National Clean Air Action recommendations
    pm25 = inp.pm2_5
    apparent_temp = inp.apparent_temp_max or inp.temp_max
    precip = inp.precipitation_mm

    # Air Quality Morning Assembly Score (0-100)
    aq_score = min(100.0, (pm25 / 250.0) * 100.0)

    # Afternoon Heat Stress Score (0-100)
    heat_score = min(100.0, max(0.0, (apparent_temp - 32.0) / (45.0 - 32.0) * 100.0))

    # Transit Waterlogging Score (0-100)
    flood_score = min(100.0, (precip / 50.0) * 100.0 * max(0.5, inp.rain_probability))

    # Overall Student Impact Score (weighted heavily towards morning air & afternoon heat)
    impact_score = round(float(0.45 * aq_score + 0.35 * heat_score + 0.20 * flood_score), 1)

    # Status level
    if impact_score >= 75.0 or inp.imd_heatwave_alert >= 2 or pm25 >= 250.0:
        status = "CRITICAL"
    elif impact_score >= 50.0 or inp.imd_heatwave_alert == 1 or pm25 >= 120.0:
        status = "HIGH_ALERT"
    elif impact_score >= 30.0 or pm25 >= 60.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    # Specific operational guidelines per Ministry of Education heatwave & pollution circulars
    assembly_status = "OUTDOOR_SAFE"
    if pm25 >= 200.0 or (inp.atmospheric_stagnation and pm25 >= 100.0):
        assembly_status = "MANDATORY_INDOORS"
        restrictions.append("Morning assembly must NOT be conducted outdoors due to high particulate toxicity.")
        advisories.append("Ensure classroom windows remain closed during morning rush hours.")
    elif pm25 >= 100.0:
        assembly_status = "RESTRICTED"
        advisories.append("Shorten outdoor morning assembly to under 10 minutes.")

    # Outdoor Sports & Physical Education (PE)
    outdoor_sports_safe = True
    if apparent_temp >= 42.0 or inp.imd_heatwave_alert >= 1:
        outdoor_sports_safe = False
        restrictions.append("Cancel all outdoor sports, PT periods, and playground sessions between 11:00 AM and 4:00 PM.")
        advisories.append("Equip school buses with oral rehydration solution (ORS) and shaded boarding points.")
    elif apparent_temp >= 38.0:
        advisories.append("Limit continuous physical education to 20 minutes with mandatory hydration breaks.")

    # Transit / School Bus Flood Warning
    bus_flood_risk = "LOW"
    if precip >= 30.0 and inp.rain_probability >= 0.5:
        bus_flood_risk = "HIGH"
        restrictions.append("High probability of underpass waterlogging on school bus transit corridors.")
        advisories.append("Advise parents of potential 30-45 minute school bus transit delays.")
    elif precip >= 10.0:
        bus_flood_risk = "MODERATE"
        advisories.append("Light to moderate rainfall expected during school commute.")

    metrics: Dict[str, Any] = {
        "morning_assembly_status": assembly_status,
        "outdoor_sports_allowed": outdoor_sports_safe,
        "school_bus_flood_risk": bus_flood_risk,
        "expected_pm25_exposure": f"{pm25:.1f} µg/m³",
        "afternoon_perceived_heat": f"{apparent_temp:.1f} °C",
        "heatwave_warning_tier": ["None", "Yellow", "Orange", "Red"][min(3, inp.imd_heatwave_alert)],
    }

    if inp.schools_15km is not None and inp.schools_15km > 0:
        metrics["schools_in_district_zone"] = (
            f"{inp.schools_15km:,} schools (1 facility per {(inp.people_per_school_15km or 0):,} citizens)"
        )
        if status in ["HIGH_ALERT", "CRITICAL"]:
            restrictions.append(
                f"District Education Office protocol: Enforce indoor ventilation mandates or class timetable adjustments across {inp.schools_15km:,} schools."
            )

    return PersonaResult(
        persona_id="student_parent",
        persona_title="Student, Parent & School Administration",
        status_level=status,
        impact_score=impact_score,
        key_metrics=metrics,
        advisories=advisories or ["Normal school schedule and outdoor activities safe to proceed."],
        operational_restrictions=restrictions or ["No administrative restrictions in effect."],
    )
