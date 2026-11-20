"""BasinFlow V1: Physical and Spatial Hydrological Modelling Engine.
Includes:
- SCS Dimensionless Unit Hydrograph (rainfall-excess to time-distributed runoff hydrograph)
- Muskingum-Cunge Hydraulic Channel Routing (HydroRIVERS reach routing)
- BasinFlow FloodRisk Challenger Model
"""

from climacore.models.basinflow.unit_hydrograph import SCSUnitHydrograph
from climacore.models.basinflow.muskingum_cunge import MuskingumCungeRouter
from climacore.models.basinflow.flood_challenger import BasinFlowFloodRiskChallenger
from climacore.models.basinflow.conceptual_models import GR4J, HBVLight
from climacore.models.basinflow.hybrid_gr4j import HybridGR4J
from climacore.models.basinflow.water_balance import compute_water_balance, eckhardt_filter
from climacore.models.basinflow.baselines import (
    PersistenceBaseline,
    SeasonalClimatologyBaseline,
    SCSCNBaseline,
    AR1Baseline,
)

__all__ = [
    "SCSUnitHydrograph",
    "MuskingumCungeRouter",
    "BasinFlowFloodRiskChallenger",
    "GR4J",
    "HBVLight",
    "HybridGR4J",
    "PersistenceBaseline",
    "SeasonalClimatologyBaseline",
    "SCSCNBaseline",
    "AR1Baseline",
    "compute_water_balance",
    "eckhardt_filter",
]
