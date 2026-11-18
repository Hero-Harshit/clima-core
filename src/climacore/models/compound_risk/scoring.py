"""
CompoundRisk Engine for Clima-Core.
Aggregates independent hazard estimates (HeatRisk, RainRisk, AirQuality) into
an explainable 0-100 Compound Risk Index with factor attribution and non-linear
multi-hazard amplification.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from climacore.models.compound_risk.config import (
    HazardWeights,
    normalize_air_quality,
    normalize_heat_risk,
    normalize_rain_risk,
    classify_compound_risk_category,
)


class CompoundRiskEngine:
    """
    Deterministic, transparent multi-hazard risk engine.
    Computes normalized sub-hazard scores, applies configurable weighting,
    detects multi-hazard co-occurrence amplification, and outputs driver attribution.
    """

    def __init__(
        self,
        weights: Optional[HazardWeights] = None,
        compound_amplification_factor: float = 0.25,
        elevated_threshold: float = 50.0,
    ):
        self.weights = weights or HazardWeights()
        self.weights.validate_weights()
        self.gamma = compound_amplification_factor
        self.elevated_thresh = elevated_threshold

    def calculate_single(
        self,
        pm2_5_pred: float,
        temp_max_pred: float,
        precip_pred: float,
        rain_prob: Optional[float] = None,
        city: str = "Unknown",
        date: str = "Unknown",
    ) -> Dict[str, Any]:
        """Compute compound risk score and explainability breakdown for a single location/date."""
        # 1. Normalize individual hazards (0 - 100)
        aq_score, aq_cat = normalize_air_quality(pm2_5_pred)
        heat_score, heat_cat = normalize_heat_risk(temp_max_pred)
        rain_score, rain_cat = normalize_rain_risk(precip_pred, rain_prob)

        # 2. Weighted Base Score
        w_aq = self.weights.air_quality
        w_heat = self.weights.heat_risk
        w_rain = self.weights.rain_risk

        base_score = (w_aq * aq_score) + (w_heat * heat_score) + (w_rain * rain_score)

        # 3. Multi-Hazard Compounding Amplification
        # If two or more hazards are simultaneously elevated (>= threshold), apply co-occurrence multiplier
        elevated_hazards = [s for s in [aq_score, heat_score, rain_score] if s >= self.elevated_thresh]
        multiplier = 1.0
        is_compounded = len(elevated_hazards) >= 2

        if is_compounded:
            # Multiplier proportional to the interaction product of top 2 elevated hazards
            sorted_elevated = sorted(elevated_hazards, reverse=True)
            interaction = (sorted_elevated[0] / 100.0) * (sorted_elevated[1] / 100.0)
            multiplier = 1.0 + (self.gamma * interaction)

        compound_score = min(100.0, base_score * multiplier)
        severity_category = classify_compound_risk_category(compound_score)

        # 4. Empirical Multi-Hazard Joint Tail Exceedance Probability
        # Decouples pure marginal probabilities and quantifies bivariate / trivariate co-occurrence
        p_aq = float(np.clip(aq_score / 100.0, 0.0, 1.0))
        p_heat = float(np.clip(heat_score / 100.0, 0.0, 1.0))
        p_rain = float(np.clip(rain_score / 100.0, 0.0, 1.0))

        bivariate_joint_probs = {
            "heat_and_air_pollution": round(p_heat * p_aq, 4),
            "heat_and_severe_rain": round(p_heat * p_rain, 4),
            "air_pollution_and_rain": round(p_aq * p_rain, 4),
        }
        trivariate_joint_prob = round(p_heat * p_aq * p_rain, 5)

        # 5. Factor Attribution (% contribution to base score)
        total_unweighted = aq_score + heat_score + rain_score
        if total_unweighted > 0:
            aq_contrib = round(float((w_aq * aq_score) / max(1e-5, base_score) * 100.0), 1)
            heat_contrib = round(float((w_heat * heat_score) / max(1e-5, base_score) * 100.0), 1)
            rain_contrib = round(float((w_rain * rain_score) / max(1e-5, base_score) * 100.0), 1)
        else:
            aq_contrib = heat_contrib = rain_contrib = 33.3

        # Dominant Driver
        hazard_dict = {
            "Air Quality (PM2.5)": aq_score,
            "Extreme Heat": heat_score,
            "Rainfall Hazard": rain_score,
        }
        dominant_driver = max(hazard_dict, key=hazard_dict.get)

        # Actionable Advisories
        advisories = []
        if aq_score >= 75.0:
            advisories.append("Air Quality Alert: High particulate matter. Limit outdoor exertion.")
        if heat_score >= 80.0:
            advisories.append("Heat Alert: High temperature stress. Maintain hydration and shade.")
        if rain_score >= 65.0:
            advisories.append("Precipitation Alert: Significant rainfall expected. Monitor localized waterlogging.")
        if is_compounded:
            advisories.append("Compound Hazard Warning: Multiple environmental risks co-occurring simultaneously.")

        return {
            "city": city,
            "date": date,
            "compound_risk_score": round(float(compound_score), 1),
            "severity_category": severity_category,
            "is_compounded_event": is_compounded,
            "compound_multiplier": round(float(multiplier), 3),
            "dominant_driver": dominant_driver,
            "joint_tail_exceedance": {
                "bivariate_probabilities": bivariate_joint_probs,
                "trivariate_probability": trivariate_joint_prob,
                "compounding_active": is_compounded,
            },
            "sub_hazards": {
                "air_quality": {
                    "raw_value": round(float(pm2_5_pred), 1),
                    "unit": "ug/m3",
                    "score_0_100": round(float(aq_score), 1),
                    "category": aq_cat,
                    "contribution_pct": aq_contrib,
                },
                "heat_risk": {
                    "raw_value": round(float(temp_max_pred), 1),
                    "unit": "deg_C",
                    "score_0_100": round(float(heat_score), 1),
                    "category": heat_cat,
                    "contribution_pct": heat_contrib,
                },
                "rain_risk": {
                    "raw_value": round(float(precip_pred), 1),
                    "unit": "mm",
                    "score_0_100": round(float(rain_score), 1),
                    "category": rain_cat,
                    "contribution_pct": rain_contrib,
                },
            },
            "advisories": advisories,
        }

    def evaluate_dataframe(
        self,
        df: pd.DataFrame,
        pm2_5_col: str,
        temp_max_col: str,
        precip_col: str,
        city_col: str = "city",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Process a tabular DataFrame and return compound scores with explanations."""
        records = []
        for _, row in df.iterrows():
            rec = self.calculate_single(
                pm2_5_pred=row[pm2_5_col],
                temp_max_pred=row[temp_max_col],
                precip_pred=row[precip_col],
                city=str(row[city_col]),
                date=str(row[date_col]),
            )
            records.append({
                "city": rec["city"],
                "date": rec["date"],
                "compound_risk_score": rec["compound_risk_score"],
                "severity_category": rec["severity_category"],
                "is_compounded_event": rec["is_compounded_event"],
                "dominant_driver": rec["dominant_driver"],
                "aq_score": rec["sub_hazards"]["air_quality"]["score_0_100"],
                "heat_score": rec["sub_hazards"]["heat_risk"]["score_0_100"],
                "rain_score": rec["sub_hazards"]["rain_risk"]["score_0_100"],
                "advisories_count": len(rec["advisories"]),
            })
        return pd.DataFrame(records)
