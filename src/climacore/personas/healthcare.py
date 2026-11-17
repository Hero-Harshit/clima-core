"""
Healthcare & High-Risk Vulnerable Persona Adapter.
Evaluates respiratory toxicity burden (ICMR clinical standards for asthma/COPD),
cardiovascular thermal shock, and safe outdoor breathing windows for elderly citizens and children.
"""

from typing import Dict, Any, List
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_healthcare_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate medical and physiological health vulnerability for sensitive populations."""
    advisories = []
    restrictions = []

    pm25 = inp.pm2_5
    pm10 = inp.pm10 or (pm25 * 1.6)
    no2 = inp.no2 or 25.0
    t_max = inp.temp_max
    t_min = inp.temp_min
    apparent_temp = inp.apparent_temp_max or t_max

    # 1. Multi-Pollutant Airway Toxicity Index (ICMR / AIIMS guidelines)
    # PM2.5 penetrates deep alveolar tissue; NO2 triggers immediate bronchial inflammation
    toxicity_score = min(100.0, (pm25 / 200.0) * 70.0 + (no2 / 80.0) * 30.0)

    # 2. Cardiovascular Thermal Strain
    # Extreme heat dilates vessels and strains cardiac output; extreme cold elevates blood pressure
    diurnal_range = t_max - t_min
    thermal_strain = 0.0
    if apparent_temp >= 42.0 or inp.imd_heatwave_alert >= 1:
        thermal_strain = min(100.0, (apparent_temp - 38.0) / 7.0 * 100.0)
    elif t_min <= 8.0:
        thermal_strain = min(100.0, (10.0 - t_min) / 6.0 * 80.0)

    # 3. Safe Outdoor Exercise / Walking Window
    # In winter/stagnant air, mornings (6-9 AM) are the most toxic due to surface trapping
    if pm25 >= 150.0:
        walking_window = "NOT_RECOMMENDED_REMAIN_INDOORS"
        restrictions.append("Elderly individuals and asthmatic patients should avoid all outdoor walks.")
        advisories.append("Keep rescue inhalers accessible. Run indoor air purifiers on continuous mode.")
    elif pm25 >= 90.0:
        walking_window = "RESTRICTED_AFTERNOON_ONLY (1:00 PM - 3:30 PM)"
        advisories.append("Avoid early morning or late evening walks when particulate concentrations peak near ground level.")
    else:
        walking_window = "FAVORABLE (Morning / Evening)"

    # Total Clinical Vulnerability Score (0 to 100)
    health_score = round(float(0.60 * toxicity_score + 0.40 * thermal_strain), 1)

    if health_score >= 75.0 or pm25 >= 250.0 or apparent_temp >= 44.0:
        status = "CRITICAL"
        restrictions.append("High hospital triage surge anticipated for acute respiratory and cardiac admissions.")
    elif health_score >= 50.0 or pm25 >= 120.0 or apparent_temp >= 40.0:
        status = "HIGH_ALERT"
    elif health_score >= 30.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    if apparent_temp >= 41.0:
        advisories.append("Severe cardiovascular strain: increase oral fluid intake, monitor elderly relatives for confusion or lethargy.")
    if diurnal_range >= 16.0:
        advisories.append(f"Wide diurnal temperature swing ({diurnal_range:.1f}°C) may trigger seasonal bronchial hypersensitivity.")

    metrics: Dict[str, Any] = {
        "respiratory_toxicity_burden": f"{toxicity_score:.1f} / 100",
        "cardiovascular_thermal_strain": f"{thermal_strain:.1f} / 100",
        "safe_outdoor_walk_window": walking_window,
        "forecasted_pm25": f"{pm25:.1f} µg/m³",
        "forecasted_no2": f"{no2:.1f} µg/m³",
        "perceived_heat_index": f"{apparent_temp:.1f} °C",
        "hospital_surge_alert": "HIGH" if health_score >= 70.0 else ("MODERATE" if health_score >= 45.0 else "LOW"),
    }

    if inp.population_15km is not None:
        vulnerable_pool = int(round(inp.population_15km * 0.20))
        surge_factor = ((health_score / 100.0) ** 1.5) * 0.003
        projected_triage = max(10, int(round(vulnerable_pool * surge_factor)))
        metrics["exposed_population_15km"] = f"{inp.population_15km:,} citizens"
        metrics["vulnerable_demographic_pool"] = f"{vulnerable_pool:,} (pediatric <5 & elderly 65+)"
        metrics["projected_emergency_triage_demand"] = f"~{projected_triage:,} acute clinical admissions"

        if health_score >= 60.0 and projected_triage >= 500:
            advisories.append(
                f"Anticipate acute healthcare surge: ~{projected_triage:,} elderly and asthmatic patients projected to require emergency triage/nebulizer beds within 15km metro radius."
            )

    if inp.hospitals_15km is not None and inp.hospitals_15km > 0:
        metrics["hospitals_in_exposure_zone"] = (
            f"{inp.hospitals_15km:,} hospitals (1 facility per {(inp.people_per_hospital_15km or 0):,} citizens)"
        )
        if inp.population_15km is not None:
            triage_per_hosp = round(projected_triage / max(1, inp.hospitals_15km), 1)
            metrics["average_surge_load_per_hospital"] = f"~{triage_per_hosp} admissions / hospital"
            if health_score >= 60.0 and triage_per_hosp >= 2.0:
                advisories.append(
                    f"Facility triage notice: Average surge load of ~{triage_per_hosp} incremental acute presentations across {inp.hospitals_15km:,} hospitals."
                )

    return PersonaResult(
        persona_id="healthcare_vulnerable",
        persona_title="Healthcare & High-Risk Vulnerable Population",
        status_level=status,
        impact_score=health_score,
        key_metrics=metrics,
        advisories=advisories or ["Environmental conditions within acceptable clinical bounds for sensitive groups."],
        operational_restrictions=restrictions or ["No clinical activity restrictions active."],
    )
