"""
Persona Intelligence Module for Clima-Core.
Translates physical ML predictions into tailored domain recommendations for 13 key personas across Indian economic sectors.
"""

from climacore.personas.models import PersonaInput, PersonaResult
from climacore.personas.engine import evaluate_persona, evaluate_all_personas, PERSONA_REGISTRY

__all__ = [
    "PersonaInput",
    "PersonaResult",
    "evaluate_persona",
    "evaluate_all_personas",
    "PERSONA_REGISTRY",
]
