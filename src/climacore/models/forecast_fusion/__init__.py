"""ForecastFusion V1: Multi-source weather forecast intelligence and probabilistic ensemble engine.
Combines multiple numerical weather prediction models (GFS, JMA, ICON, BestMatch)
with verified IMD gridded ground truth observations.
"""

from climacore.models.forecast_fusion.skill_matrix import compute_skill_matrix
from climacore.models.forecast_fusion.ensemble_models import (
    WeightedForecastEnsemble,
    EMOSModel,
    NGBoostForecastFusion,
    QuantileLGBMEnsemble,
)

__all__ = [
    "compute_skill_matrix",
    "WeightedForecastEnsemble",
    "EMOSModel",
    "NGBoostForecastFusion",
    "QuantileLGBMEnsemble",
]
