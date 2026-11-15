"""
Persona Intelligence Module for Clima-Core.
Translates physical ML predictions into tailored domain recommendations for 6 key personas.
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
