"""
Water Quality Risk Engine for Clima-Core.
Evaluates groundwater potability, chemical contamination hazard, Water Quality Index (WQI),
and BIS IS 10500:2012 drinking water health advisories across Indian urban aquifers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import lightgbm as lgb
import numpy as np

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root
from climacore.features.groundwater_quality import BIS_STANDARDS, compute_water_quality_index

logger = get_logger("climacore.models.water_quality.engine")

# Regional aquifer baseline medians (derived from CGWB monitoring network)
DEFAULT_AQUIFER_BASELINES: Dict[str, float] = {
    "ph": 7.8,
    "electric_conductivity": 980.0,
    "total_hardness": 340.0,
    "chloride": 110.0,
    "sulphate": 95.0,
    "calcium": 65.0,
    "magnesium": 42.0,
    "sodium": 40.0,
}


class WaterQualityRiskEngine:
    """
    Evaluates groundwater contamination risk and drinking water potability.
    Combines supervised GBDT hazard modeling with BIS IS 10500:2012 chemical standards.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        root = get_project_root()
        if models_dir is None:
            models_dir = root / "artifacts" / "models" / "water_quality" / "v0.1.0"
        self.models_dir = models_dir
        self.clf_booster: Optional[lgb.Booster] = None
        self.reg_booster: Optional[lgb.Booster] = None
        self._load_models()

    def _load_models(self):
        clf_file = self.models_dir / "water_hazard_v1.booster"
        reg_file = self.models_dir / "wqi_regressor_v1.booster"

        if clf_file.exists():
            try:
                self.clf_booster = lgb.Booster(model_file=str(clf_file))
                logger.info(f"Loaded Water Hazard classifier from {clf_file}")
            except Exception as e:
                logger.warning(f"Could not load water hazard classifier: {e}")

        if reg_file.exists():
            try:
                self.reg_booster = lgb.Booster(model_file=str(reg_file))
                logger.info(f"Loaded WQI regressor from {reg_file}")
            except Exception as e:
                logger.warning(f"Could not load WQI regressor: {e}")

    def calculate_exact_wqi_and_exceedances(
        self,
        chemical_params: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compute exact BIS IS 10500:2012 Weighted Arithmetic WQI and exceedances deterministically."""
        wqi, category = compute_water_quality_index(chemical_params)
        exceedances = []
        for param, val in chemical_params.items():
            if val is not None and param in BIS_STANDARDS:
                std = BIS_STANDARDS[param]
                if "permissible" in std and val > std["permissible"]:
                    exceedances.append(f"{param} ({val}) exceeds permissible limit ({std['permissible']})")
                elif "acceptable" in std and val > std["acceptable"]:
                    exceedances.append(f"{param} ({val}) exceeds acceptable limit ({std['acceptable']})")
        return {
            "exact_wqi": wqi,
            "wqi_category": category,
            "bis_standard_exceedances": exceedances,
        }

    def evaluate_water_quality(
        self,
        latitude: float,
        longitude: float,
        city: str = "",
        ph: Optional[float] = None,
        electric_conductivity: Optional[float] = None,
        total_hardness: Optional[float] = None,
        chloride: Optional[float] = None,
        sulphate: Optional[float] = None,
        calcium: Optional[float] = None,
        magnesium: Optional[float] = None,
        sodium: Optional[float] = None,
        is_drought_stressed: bool = False,
        is_flood_inundated: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate groundwater chemical quality, WQI, and potability hazard.
        Combines exact deterministic BIS IS 10500:2012 calculations with LightGBM surrogate predictions.
        """
        has_user_measurements = any(
            p is not None for p in [ph, electric_conductivity, total_hardness, chloride, sulphate, calcium, magnesium, sodium]
        )

        # Populate missing values with regional aquifer baselines
        v_ph = ph if ph is not None else DEFAULT_AQUIFER_BASELINES["ph"]
        v_ec = electric_conductivity if electric_conductivity is not None else DEFAULT_AQUIFER_BASELINES["electric_conductivity"]
        v_th = total_hardness if total_hardness is not None else DEFAULT_AQUIFER_BASELINES["total_hardness"]
        v_cl = chloride if chloride is not None else DEFAULT_AQUIFER_BASELINES["chloride"]
        v_so4 = sulphate if sulphate is not None else DEFAULT_AQUIFER_BASELINES["sulphate"]
        v_ca = calcium if calcium is not None else DEFAULT_AQUIFER_BASELINES["calcium"]
        v_mg = magnesium if magnesium is not None else DEFAULT_AQUIFER_BASELINES["magnesium"]
        v_na = sodium if sodium is not None else DEFAULT_AQUIFER_BASELINES["sodium"]

        # Drought stress concentration effect: low recharge concentrates dissolved minerals (+15% to +25%)
        if is_drought_stressed:
            v_ec *= 1.20
            v_th *= 1.15
            v_cl *= 1.25

        # Feature vector for ML booster
        # Order: latitude, longitude, ph, electric_conductivity, total_hardness, chloride, sulphate, calcium, magnesium, sodium
        features = np.array([[
            latitude,
            longitude,
            v_ph,
            v_ec,
            v_th,
            v_cl,
            v_so4,
            v_ca,
            v_mg,
            v_na,
        ]], dtype=float)

        # 1. Predict Hazard Probability via ML Surrogate
        if self.clf_booster is not None:
            hazard_prob = float(self.clf_booster.predict(features)[0])
        else:
            # Deterministic heuristic fallback
            tds_est = v_ec * 0.64
            hazard_prob = 0.85 if (tds_est > 1500.0 or v_th > 500.0) else 0.40 if tds_est > 500.0 else 0.15

        # 2. Predict WQI Score via ML Surrogate
        if self.reg_booster is not None:
            surrogate_pred_wqi = float(self.reg_booster.predict(features)[0])
        else:
            surrogate_pred_wqi = float((v_ec * 0.64 / 500.0) * 45.0 + (v_th / 200.0) * 35.0)

        surrogate_pred_wqi = max(10.0, round(surrogate_pred_wqi, 1))

        # 3. Direct Deterministic BIS Calculation
        active_sample = {
            "ph": v_ph,
            "total_dissolved_solids": v_ec * 0.64,
            "total_hardness": v_th,
            "chloride": v_cl,
            "sulphate": v_so4,
            "calcium": v_ca,
            "magnesium": v_mg,
        }
        exact_wqi, exact_category = compute_water_quality_index(active_sample)

        # If user provided chemical measurements, use exact BIS formula; otherwise use surrogate with regional defaults
        if has_user_measurements and not np.isnan(exact_wqi):
            pred_wqi = exact_wqi
            wqi_calc_mode = "exact_deterministic_formula"
        else:
            pred_wqi = surrogate_pred_wqi
            wqi_calc_mode = "regional_baseline_surrogate"

        # Categorize WQI
        if pred_wqi < 25.0:
            wqi_grade = "EXCELLENT"
            potability_status = "Potable (Direct Drinking Safe)"
        elif pred_wqi < 50.0:
            wqi_grade = "GOOD"
            potability_status = "Potable (Standard Conventional Treatment)"
        elif pred_wqi < 75.0:
            wqi_grade = "POOR"
            potability_status = "Marginal (Filtration & Boiling Required)"
        elif pred_wqi <= 100.0:
            wqi_grade = "VERY_POOR"
            potability_status = "Non-Potable (Advanced RO / De-mineralization Required)"
        else:
            wqi_grade = "UNSUITABLE_FOR_DRINKING"
            potability_status = "Hazardous (Unfit for Human Consumption)"

        # 4. Chemical Exceedance Analysis against BIS IS 10500:2012
        tds_est = round(v_ec * 0.64, 1)
        exceedances: List[str] = []
        if tds_est > BIS_STANDARDS["total_dissolved_solids"]["permissible"]:
            exceedances.append(f"TDS ({tds_est:.0f} mg/L) exceeds permissible limit (2000 mg/L)")
        elif tds_est > BIS_STANDARDS["total_dissolved_solids"]["acceptable"]:
            exceedances.append(f"TDS ({tds_est:.0f} mg/L) exceeds acceptable limit (500 mg/L)")

        if v_th > BIS_STANDARDS["total_hardness"]["permissible"]:
            exceedances.append(f"Total Hardness ({v_th:.0f} mg/L) exceeds permissible limit (600 mg/L)")
        elif v_th > BIS_STANDARDS["total_hardness"]["acceptable"]:
            exceedances.append(f"Total Hardness ({v_th:.0f} mg/L) exceeds acceptable limit (200 mg/L)")

        if v_cl > BIS_STANDARDS["chloride"]["acceptable"]:
            exceedances.append(f"Chloride ({v_cl:.0f} mg/L) exceeds acceptable limit (250 mg/L)")

        if not (6.5 <= v_ph <= 8.5):
            exceedances.append(f"pH ({v_ph:.1f}) outside safe range (6.5 - 8.5)")

        # 5. Actionable Health & Public Works Advisories
        advisories: List[str] = []
        if wqi_grade in ["VERY_POOR", "UNSUITABLE_FOR_DRINKING"]:
            advisories.append("MUNICIPAL_ADVISORY: Prohibit direct extraction from shallow borewells for drinking water supply.")
            advisories.append("HOUSEHOLD_ADVISORY: Mandate Reverse Osmosis (RO) filtration or municipal piped water tankers.")
        elif wqi_grade == "POOR":
            advisories.append("HOUSEHOLD_ADVISORY: Domestic boiling and mechanical carbon filtration advised before consumption.")

        if is_flood_inundated:
            advisories.append("POST_FLOOD_ALERT: Borewells at acute risk of septic/sewage back-siphonage; chlorination required.")

        return {
            "city": city,
            "coordinates": {"latitude": latitude, "longitude": longitude},
            "contamination_hazard_probability": round(hazard_prob, 4),
            "predicted_wqi": pred_wqi,
            "exact_formula_wqi": exact_wqi if not np.isnan(exact_wqi) else None,
            "surrogate_predicted_wqi": surrogate_pred_wqi,
            "wqi_grade": wqi_grade,
            "potability_status": potability_status,
            "wqi_calculation_mode": wqi_calc_mode,
            "is_deterministic_surrogate": True,
            "surrogate_fidelity_note": (
                "ML boosters wqi_regressor_v1 and water_hazard_v1 are deterministic surrogates approximating "
                "the BIS IS 10500:2012 Weighted Arithmetic index from chemical ion concentrations. "
                "They do not predict future groundwater quality changes or hydrological contaminant transport."
            ),
            "estimated_parameters": {
                "ph": round(v_ph, 2),
                "electric_conductivity_us_cm": round(v_ec, 1),
                "total_dissolved_solids_mg_l": tds_est,
                "total_hardness_mg_l": round(v_th, 1),
                "chloride_mg_l": round(v_cl, 1),
                "sulphate_mg_l": round(v_so4, 1),
                "calcium_mg_l": round(v_ca, 1),
                "magnesium_mg_l": round(v_mg, 1),
                "sodium_mg_l": round(v_na, 1),
            },
            "bis_standard_exceedances": exceedances,
            "operational_advisories": advisories,
            "environmental_coupling": {
                "drought_concentration_active": is_drought_stressed,
                "flood_inundation_leaching_active": is_flood_inundated,
            },
        }
