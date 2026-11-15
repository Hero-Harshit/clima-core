"""
Persona Intelligence Adapter Engine for Clima-Core.
Orchestrates persona routing and converts physical environmental predictions into
actionable, domain-specific decision and restriction matrices.
"""

from typing import Dict, Any, List, Optional
from climacore.personas.models import PersonaInput, PersonaResult
from climacore.personas.student import evaluate_student_persona
from climacore.personas.farmer import evaluate_farmer_persona
from climacore.personas.gig_worker import evaluate_gig_worker_persona
from climacore.personas.commuter import evaluate_commuter_persona
from climacore.personas.healthcare import evaluate_healthcare_persona
from climacore.personas.city_official import evaluate_city_official_persona

PERSONA_REGISTRY = {
    "student": evaluate_student_persona,
    "parent": evaluate_student_persona,
    "student_parent": evaluate_student_persona,
    "farmer": evaluate_farmer_persona,
    "kisan": evaluate_farmer_persona,
    "farmer_kisan": evaluate_farmer_persona,
    "gig_worker": evaluate_gig_worker_persona,
    "labor": evaluate_gig_worker_persona,
    "gig_worker_labor": evaluate_gig_worker_persona,
    "commuter": evaluate_commuter_persona,
    "office_worker": evaluate_commuter_persona,
    "daily_commuter": evaluate_commuter_persona,
    "healthcare": evaluate_healthcare_persona,
    "vulnerable": evaluate_healthcare_persona,
    "healthcare_vulnerable": evaluate_healthcare_persona,
    "city_official": evaluate_city_official_persona,
    "planner": evaluate_city_official_persona,
    "infrastructure": evaluate_city_official_persona,
    "city_official_infrastructure": evaluate_city_official_persona,
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
    """Evaluate all 6 distinct personas simultaneously for comprehensive city reporting."""
    canonical_personas = [
        "student_parent",
        "farmer_kisan",
        "gig_worker_labor",
        "daily_commuter",
        "healthcare_vulnerable",
        "city_official_infrastructure",
    ]
    results = {}
    for p_id in canonical_personas:
        res = evaluate_persona(p_id, inp)
        results[p_id] = res.model_dump()
    return results
