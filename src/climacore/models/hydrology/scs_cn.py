"""SCS Curve Number Hydrological Runoff Engine V1.

Implements the deterministic USDA Natural Resources Conservation Service (NRCS)
National Engineering Handbook (NEH) Part 630 / TR-55 event-runoff formulation.

Key Capabilities:
1. Hydrologic Soil Group (HSG A, B, C, D) and Land Cover CN derivation.
2. Dynamic Antecedent Moisture Condition (AMC-I Dry, AMC-II Average, AMC-III Saturated)
   derived from 5-day antecedent precipitation (P5).
3. Connected Impervious Area (CIA) composite CN calculation without double-counting.
4. Configurable initial abstraction ratio (lambda = 0.20 standard, lambda = 0.05 Hawkins et al.).
5. Event Runoff Depth Q (mm) and Catchment Volume V (m^3) with explicit SI units.
6. Sensitivity analysis of initial abstraction and retention.

CRITICAL ARCHITECTURAL BOUNDARY & NON-CLAIM DISCLAIMER:
    Runoff depth Q (mm) and runoff volume V (m^3) represent the total volume of
    excess rainfall generated across a catchment surface that does not infiltrate
    or become trapped in surface depressions.

    This engine does NOT claim to predict subterranean urban drain pipe backwater,
    manhole surcharge, or street-level conduit overflow without explicit hydraulic
    data, specifically:
    (1) Subterranean pipe diameters, materials, Manning's n roughness, and invert slopes;
    (2) Dynamic 1D/2D hydrodynamic routing (e.g. SWMM / HEC-RAS);
    (3) Sub-hourly hyetograph rainfall intensity (e.g. mm/15min peak burst);
    (4) Downstream boundary conditions (river stage, tidal head, outfall flap gate submergence).
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HydrologicSoilGroup(str, Enum):
    """USDA NRCS Hydrologic Soil Groups."""
    A = "A"  # High infiltration rate (deep sand, aggregated silts, >0.30 in/hr)
    B = "B"  # Moderate infiltration rate (shallow loess, sandy loam, 0.15-0.30 in/hr)
    C = "C"  # Slow infiltration rate (clay loams, shallow sandy soils, 0.05-0.15 in/hr)
    D = "D"  # Very slow infiltration (heavy clays, high water table, <0.05 in/hr)


class AntecedentMoistureCondition(str, Enum):
    """Antecedent Moisture Conditions."""
    AMC_I = "AMC_I"      # Dry soils (lower runoff potential)
    AMC_II = "AMC_II"    # Average soil moisture (standard reference)
    AMC_III = "AMC_III"  # Saturated soils (near maximum runoff potential)


class LandCoverType(str, Enum):
    """Representative Land Cover / Land Use classifications."""
    FOREST_GOOD_COVER = "forest_good"
    PASTURE_RANGE_GOOD = "pasture_good"
    CULTIVATED_ROW_CROPS = "cultivated_crops"
    OPEN_SPACES_PARKS = "open_spaces_parks"
    RESIDENTIAL_LOW_DENSITY = "residential_low"      # ~25% impervious
    RESIDENTIAL_MEDIUM_DENSITY = "residential_med"   # ~38% impervious
    RESIDENTIAL_HIGH_DENSITY = "residential_high"    # ~65% impervious
    COMMERCIAL_BUSINESS = "commercial_business"      # ~85% impervious
    INDUSTRIAL = "industrial"                        # ~72% impervious
    PAVED_IMPERVIOUS = "paved_impervious"            # 100% impervious (roads, roofs)


# TR-55 Table 2-2 standard CN_II values by Land Cover and Hydrologic Soil Group
TR55_CN_TABLE: Dict[LandCoverType, Dict[HydrologicSoilGroup, float]] = {
    LandCoverType.FOREST_GOOD_COVER: {
        HydrologicSoilGroup.A: 30.0,
        HydrologicSoilGroup.B: 55.0,
        HydrologicSoilGroup.C: 70.0,
        HydrologicSoilGroup.D: 77.0,
    },
    LandCoverType.PASTURE_RANGE_GOOD: {
        HydrologicSoilGroup.A: 39.0,
        HydrologicSoilGroup.B: 61.0,
        HydrologicSoilGroup.C: 74.0,
        HydrologicSoilGroup.D: 80.0,
    },
    LandCoverType.CULTIVATED_ROW_CROPS: {
        HydrologicSoilGroup.A: 72.0,
        HydrologicSoilGroup.B: 81.0,
        HydrologicSoilGroup.C: 88.0,
        HydrologicSoilGroup.D: 91.0,
    },
    LandCoverType.OPEN_SPACES_PARKS: {
        HydrologicSoilGroup.A: 49.0,
        HydrologicSoilGroup.B: 69.0,
        HydrologicSoilGroup.C: 79.0,
        HydrologicSoilGroup.D: 84.0,
    },
    LandCoverType.RESIDENTIAL_LOW_DENSITY: {
        HydrologicSoilGroup.A: 54.0,
        HydrologicSoilGroup.B: 70.0,
        HydrologicSoilGroup.C: 80.0,
        HydrologicSoilGroup.D: 85.0,
    },
    LandCoverType.RESIDENTIAL_MEDIUM_DENSITY: {
        HydrologicSoilGroup.A: 61.0,
        HydrologicSoilGroup.B: 75.0,
        HydrologicSoilGroup.C: 83.0,
        HydrologicSoilGroup.D: 87.0,
    },
    LandCoverType.RESIDENTIAL_HIGH_DENSITY: {
        HydrologicSoilGroup.A: 77.0,
        HydrologicSoilGroup.B: 85.0,
        HydrologicSoilGroup.C: 90.0,
        HydrologicSoilGroup.D: 92.0,
    },
    LandCoverType.COMMERCIAL_BUSINESS: {
        HydrologicSoilGroup.A: 89.0,
        HydrologicSoilGroup.B: 92.0,
        HydrologicSoilGroup.C: 94.0,
        HydrologicSoilGroup.D: 95.0,
    },
    LandCoverType.INDUSTRIAL: {
        HydrologicSoilGroup.A: 81.0,
        HydrologicSoilGroup.B: 88.0,
        HydrologicSoilGroup.C: 91.0,
        HydrologicSoilGroup.D: 93.0,
    },
    LandCoverType.PAVED_IMPERVIOUS: {
        HydrologicSoilGroup.A: 98.0,
        HydrologicSoilGroup.B: 98.0,
        HydrologicSoilGroup.C: 98.0,
        HydrologicSoilGroup.D: 98.0,
    },
}


class RunoffResult(BaseModel):
    """Deterministic SCS Runoff Computation Result with Explicit Physical Units."""
    rainfall_depth_mm: float = Field(..., description="Total event precipitation in millimeters [mm]")
    antecedent_precipitation_5day_mm: float = Field(..., description="5-day cumulative antecedent rainfall [mm]")
    antecedent_moisture_condition: AntecedentMoistureCondition
    base_curve_number_cn2: float = Field(..., description="Standard reference CN for AMC-II")
    adjusted_curve_number: float = Field(..., description="CN adjusted for dynamic AMC-I, II, or III")
    initial_abstraction_ratio: float = Field(..., description="Configurable ratio lambda (e.g. 0.20 or 0.05)")
    potential_retention_s_mm: float = Field(..., description="Potential maximum retention S in millimeters [mm]")
    initial_abstraction_ia_mm: float = Field(..., description="Initial abstraction Ia = lambda * S [mm]")
    runoff_depth_mm: float = Field(..., description="Direct surface runoff depth Q in millimeters [mm]")
    runoff_coefficient: float = Field(..., description="Ratio of direct runoff to event rainfall (Q / P)")
    catchment_area_km2: float = Field(..., description="Catchment contributing area in square kilometers [km²]")
    runoff_volume_m3: float = Field(..., description="Total volumetric surface runoff in cubic meters [m³]")
    connected_impervious_fraction: float = Field(..., description="Fraction of catchment area with directly connected impervious surface")
    drainage_overflow_claim: bool = Field(
        default=False,
        description="Always False. Surface runoff volume alone does not determine urban pipe overflow without conduit network data."
    )
    disclaimer: str = Field(
        default=(
            "Runoff depth and volume represent deterministic hydrological excess rainfall. "
            "No urban stormwater pipe surcharge or drain overflow is claimed without subterranean "
            "pipe geometry, routing capacity, and outfall stage boundary data."
        )
    )

    model_config = {"frozen": True}


class SCSCurveNumberEngine:
    """USDA NRCS SCS Curve Number event-runoff calculation engine."""

    def __init__(self, default_lambda: float = 0.20):
        if not (0.01 <= default_lambda <= 0.30):
            raise ValueError(f"Initial abstraction ratio lambda must be between 0.01 and 0.30, got {default_lambda}")
        self.default_lambda = default_lambda

    @staticmethod
    def classify_amc(p5_mm: float, is_growing_season: bool = True) -> AntecedentMoistureCondition:
        """Determine Antecedent Moisture Condition from 5-day antecedent precipitation (P5).

        Growing Season (default for Indian monsoon & warm tropics):
            AMC-I:   P5 < 35.0 mm
            AMC-II:  35.0 <= P5 <= 52.5 mm
            AMC_III: P5 > 52.5 mm

        Dormant Season (winter/cool dry months):
            AMC-I:   P5 < 12.5 mm
            AMC-II:  12.5 <= P5 <= 27.5 mm
            AMC_III: P5 > 27.5 mm
        """
        if is_growing_season:
            if p5_mm < 35.0:
                return AntecedentMoistureCondition.AMC_I
            elif p5_mm <= 52.5:
                return AntecedentMoistureCondition.AMC_II
            else:
                return AntecedentMoistureCondition.AMC_III
        else:
            if p5_mm < 12.5:
                return AntecedentMoistureCondition.AMC_I
            elif p5_mm <= 27.5:
                return AntecedentMoistureCondition.AMC_II
            else:
                return AntecedentMoistureCondition.AMC_III

    @staticmethod
    def adjust_cn_for_amc(cn2: float, amc: AntecedentMoistureCondition) -> float:
        """Adjust base Curve Number (CN_II) to AMC-I (dry) or AMC-III (wet).

        Uses standard NRCS NEH-4 empirical transformations (Sobhani, 1975; Hawkins et al., 1985):
            CN_I   = CN_II / (2.281 - 0.01281 * CN_II)
            CN_III = CN_II / (0.427 + 0.00573 * CN_II)
        """
        cn2 = float(np.clip(cn2, 30.0, 98.0))
        if amc == AntecedentMoistureCondition.AMC_II:
            return cn2
        elif amc == AntecedentMoistureCondition.AMC_I:
            cn1 = cn2 / (2.281 - 0.01281 * cn2)
            return float(np.clip(cn1, 15.0, 98.0))
        elif amc == AntecedentMoistureCondition.AMC_III:
            cn3 = cn2 / (0.427 + 0.00573 * cn2)
            return float(np.clip(cn3, cn2, 99.0))
        return cn2

    @staticmethod
    def compute_composite_cn(
        pervious_cn: float,
        connected_impervious_fraction: float,
        impervious_cn: float = 98.0,
    ) -> float:
        """Compute composite Curve Number handling connected impervious area without double-counting.

        Formula:
            CN_composite = (1 - f_imp) * CN_pervious + f_imp * CN_impervious
        """
        f_imp = float(np.clip(connected_impervious_fraction, 0.0, 1.0))
        return float((1.0 - f_imp) * pervious_cn + f_imp * impervious_cn)

    def calculate_event_runoff(
        self,
        rainfall_mm: float,
        p5_antecedent_mm: float = 0.0,
        cn2: Optional[float] = None,
        land_cover: Optional[LandCoverType] = None,
        soil_group: Optional[HydrologicSoilGroup] = None,
        connected_impervious_fraction: float = 0.0,
        catchment_area_km2: float = 1.0,
        initial_abstraction_ratio: Optional[float] = None,
        is_growing_season: bool = True,
    ) -> RunoffResult:
        """Calculate event runoff depth Q [mm] and volume V [m³].

        Args:
            rainfall_mm: Total event precipitation in millimeters.
            p5_antecedent_mm: 5-day antecedent precipitation sum in millimeters.
            cn2: Direct base CN_II, or derived from (land_cover, soil_group).
            land_cover: LandCoverType enum if deriving base CN.
            soil_group: HydrologicSoilGroup enum if deriving base CN.
            connected_impervious_fraction: Fraction (0.0 to 1.0) of connected impervious area.
            catchment_area_km2: Catchment surface area in square kilometers.
            initial_abstraction_ratio: Lambda ratio (defaults to self.default_lambda, e.g. 0.20 or 0.05).
            is_growing_season: Season flag for AMC thresholds.

        Returns:
            RunoffResult with explicit SI units and anti-hallucination disclaimer.
        """
        # 1. Derive base CN_II
        if cn2 is not None:
            base_pervious_cn = float(np.clip(cn2, 30.0, 98.0))
        elif land_cover is not None and soil_group is not None:
            base_pervious_cn = TR55_CN_TABLE[land_cover][soil_group]
        else:
            raise ValueError("Must provide either 'cn2' directly or both 'land_cover' and 'soil_group'.")

        # 2. Handle Connected Impervious Area (CIA) composite without double-counting
        if connected_impervious_fraction > 0.0:
            base_cn2 = self.compute_composite_cn(base_pervious_cn, connected_impervious_fraction)
        else:
            base_cn2 = base_pervious_cn

        # 3. Classify AMC from P5
        amc = self.classify_amc(p5_antecedent_mm, is_growing_season=is_growing_season)

        # 4. Adjust CN for AMC
        adjusted_cn = self.adjust_cn_for_amc(base_cn2, amc)

        # 5. Potential maximum retention S [mm]
        # S = (25400 / CN) - 254
        s_retention = (25400.0 / adjusted_cn) - 254.0

        # 6. Initial Abstraction Ia [mm]
        lam = initial_abstraction_ratio if initial_abstraction_ratio is not None else self.default_lambda
        if lam == 0.05:
            # Hawkins et al. (2002) modified retention relation for lambda = 0.05
            # S_0.05 = 1.33 * S_0.20 ^ 1.15
            s_retention = 1.33 * (s_retention ** 1.15)
            ia = 0.05 * s_retention
        else:
            ia = lam * s_retention

        # 7. Direct Event Runoff Depth Q [mm]
        p = max(0.0, float(rainfall_mm))
        if p <= ia:
            q_runoff = 0.0
        else:
            q_runoff = ((p - ia) ** 2) / (p - ia + s_retention)
        q_runoff_rounded = round(q_runoff, 2)

        # 8. Runoff Coefficient (Q / P)
        c_runoff = (q_runoff / p) if p > 0.0 else 0.0

        # 9. Direct Event Runoff Volume V [m³]
        # V = Q (mm) * 1e-3 (m/mm) * Area (km²) * 1e6 (m²/km²) = Q * Area * 1000
        area_km2 = max(1e-4, float(catchment_area_km2))
        v_runoff = q_runoff_rounded * area_km2 * 1000.0

        return RunoffResult(
            rainfall_depth_mm=round(p, 2),
            antecedent_precipitation_5day_mm=round(p5_antecedent_mm, 2),
            antecedent_moisture_condition=amc,
            base_curve_number_cn2=round(base_cn2, 2),
            adjusted_curve_number=round(adjusted_cn, 2),
            initial_abstraction_ratio=round(lam, 3),
            potential_retention_s_mm=round(s_retention, 2),
            initial_abstraction_ia_mm=round(ia, 2),
            runoff_depth_mm=q_runoff_rounded,
            runoff_coefficient=round(c_runoff, 4),
            catchment_area_km2=round(area_km2, 4),
            runoff_volume_m3=round(v_runoff, 2),
            connected_impervious_fraction=round(connected_impervious_fraction, 4),
            drainage_overflow_claim=False,
        )

    def test_lambda_sensitivity(
        self,
        rainfall_mm: float,
        cn2: float,
        p5_antecedent_mm: float = 0.0,
    ) -> Dict[str, RunoffResult]:
        """Compute runoff under both lambda = 0.20 (NRCS standard) and lambda = 0.05 (Hawkins et al.)."""
        res_020 = self.calculate_event_runoff(
            rainfall_mm=rainfall_mm,
            cn2=cn2,
            p5_antecedent_mm=p5_antecedent_mm,
            initial_abstraction_ratio=0.20,
        )
        res_005 = self.calculate_event_runoff(
            rainfall_mm=rainfall_mm,
            cn2=cn2,
            p5_antecedent_mm=p5_antecedent_mm,
            initial_abstraction_ratio=0.05,
        )
        return {"lambda_0.20": res_020, "lambda_0.05": res_005}


def calculate_scs_runoff(
    rainfall_mm: float,
    cn: float,
    area_km2: float = 1.0,
    p5_mm: float = 0.0,
    connected_impervious: float = 0.0,
    lambda_ratio: float = 0.20,
) -> RunoffResult:
    """Convenience helper for instant SCS runoff calculation."""
    engine = SCSCurveNumberEngine(default_lambda=lambda_ratio)
    return engine.calculate_event_runoff(
        rainfall_mm=rainfall_mm,
        cn2=cn,
        catchment_area_km2=area_km2,
        p5_antecedent_mm=p5_mm,
        connected_impervious_fraction=connected_impervious,
        initial_abstraction_ratio=lambda_ratio,
    )
