"""
Persona Intelligence Adapter Engine for Clima-Core.
Orchestrates persona routing and converts physical environmental predictions into
actionable, domain-specific decision and restriction matrices across 13 distinct economic sectors.
"""

from typing import Dict, Any, List, Optional
from climacore.personas.models import PersonaInput, PersonaResult
from climacore.personas.student import evaluate_student_persona
from climacore.personas.farmer import evaluate_farmer_persona
from climacore.personas.gig_worker import evaluate_gig_worker_persona
from climacore.personas.commuter import evaluate_commuter_persona
from climacore.personas.healthcare import evaluate_healthcare_persona
from climacore.personas.city_official import evaluate_city_official_persona
from climacore.personas.maritime import evaluate_maritime_persona
from climacore.personas.industry import evaluate_industry_persona
from climacore.personas.aviation import evaluate_aviation_persona
from climacore.personas.water import evaluate_water_persona
from climacore.personas.insurance import evaluate_insurance_persona
from climacore.personas.cold_chain import evaluate_cold_chain_persona
from climacore.personas.tourism import evaluate_tourism_persona

PERSONA_REGISTRY = {
    # 1. Student & Education
    "student": evaluate_student_persona,
    "parent": evaluate_student_persona,
    "student_parent": evaluate_student_persona,
    "education": evaluate_student_persona,

    # 2. Farmer & Agriculture
    "farmer": evaluate_farmer_persona,
    "kisan": evaluate_farmer_persona,
    "farmer_kisan": evaluate_farmer_persona,
    "agriculture": evaluate_farmer_persona,

    # 3. Gig Worker & Labor
    "gig_worker": evaluate_gig_worker_persona,
    "labor": evaluate_gig_worker_persona,
    "gig_worker_labor": evaluate_gig_worker_persona,
    "delivery": evaluate_gig_worker_persona,

    # 4. Commuter & Transport
    "commuter": evaluate_commuter_persona,
    "office_worker": evaluate_commuter_persona,
    "daily_commuter": evaluate_commuter_persona,
    "transport": evaluate_commuter_persona,

    # 5. Healthcare & Vulnerable
    "healthcare": evaluate_healthcare_persona,
    "vulnerable": evaluate_healthcare_persona,
    "healthcare_vulnerable": evaluate_healthcare_persona,
    "health": evaluate_healthcare_persona,

    # 6. City Official & Disaster Management
    "city_official": evaluate_city_official_persona,
    "planner": evaluate_city_official_persona,
    "infrastructure": evaluate_city_official_persona,
    "city_official_infrastructure": evaluate_city_official_persona,
    "municipal": evaluate_city_official_persona,

    # 7. Maritime & Ports
    "maritime": evaluate_maritime_persona,
    "marine": evaluate_maritime_persona,
    "fishermen": evaluate_maritime_persona,
    "port": evaluate_maritime_persona,
    "harbor": evaluate_maritime_persona,
    "maritime_coastal": evaluate_maritime_persona,

    # 8. Industry & Manufacturing
    "industry": evaluate_industry_persona,
    "industrial": evaluate_industry_persona,
    "manufacturing": evaluate_industry_persona,
    "thermal_plant": evaluate_industry_persona,
    "factory": evaluate_industry_persona,
    "industry_manufacturing": evaluate_industry_persona,

    # 9. Aviation & Airports
    "aviation": evaluate_aviation_persona,
    "airport": evaluate_aviation_persona,
    "airline": evaluate_aviation_persona,
    "flight": evaluate_aviation_persona,
    "airside": evaluate_aviation_persona,
    "aviation_airport": evaluate_aviation_persona,

    # 10. Water Utilities & Stormwater
    "water": evaluate_water_persona,
    "utilities": evaluate_water_persona,
    "stormwater": evaluate_water_persona,
    "drainage": evaluate_water_persona,
    "groundwater": evaluate_water_persona,
    "water_utilities": evaluate_water_persona,

    # 11. Insurance & Actuarial
    "insurance": evaluate_insurance_persona,
    "underwriter": evaluate_insurance_persona,
    "actuary": evaluate_insurance_persona,
    "catastrophe_risk": evaluate_insurance_persona,
    "parametric": evaluate_insurance_persona,
    "insurance_actuarial": evaluate_insurance_persona,

    # 12. Cold Chain Logistics
    "cold_chain": evaluate_cold_chain_persona,
    "reefer": evaluate_cold_chain_persona,
    "perishables": evaluate_cold_chain_persona,
    "pharma_logistics": evaluate_cold_chain_persona,
    "dairy": evaluate_cold_chain_persona,
    "cold_chain_logistics": evaluate_cold_chain_persona,

    # 13. Tourism & Hospitality
    "tourism": evaluate_tourism_persona,
    "hospitality": evaluate_tourism_persona,
    "heritage": evaluate_tourism_persona,
    "pilgrimage": evaluate_tourism_persona,
    "monuments": evaluate_tourism_persona,
    "tourism_hospitality": evaluate_tourism_persona,
}


def evaluate_persona(persona_name: str, inp: PersonaInput) -> PersonaResult:
    """Evaluate risks for a specific persona/occupation."""
    norm_key = str(persona_name).strip().lower().replace("-", "_")
    if norm_key not in PERSONA_REGISTRY:
        valid_keys = sorted(list(set(PERSONA_REGISTRY.keys())))
        raise ValueError(f"Unknown persona: '{persona_name}'. Valid choices are: {valid_keys}")
    handler = PERSONA_REGISTRY[norm_key]
    return handler(inp)


def evaluate_all_personas(inp: PersonaInput) -> Dict[str, Dict[str, Any]]:
    """Evaluate all 13 distinct personas simultaneously for comprehensive city reporting."""
    canonical_personas = [
        "student_parent",
        "farmer_kisan",
        "gig_worker_labor",
        "daily_commuter",
        "healthcare_vulnerable",
        "city_official_infrastructure",
        "maritime_coastal",
        "industry_manufacturing",
        "aviation_airport",
        "water_utilities",
        "insurance_actuarial",
        "cold_chain_logistics",
        "tourism_hospitality",
    ]
    results = {}
    for p_id in canonical_personas:
        res = evaluate_persona(p_id, inp)
        results[p_id] = res.model_dump()
    return results
