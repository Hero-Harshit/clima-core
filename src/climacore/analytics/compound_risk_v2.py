"""CompoundRisk V2: Cascading Multi-Hazard Dependency & Directed Graphical Engine.
Preserves bivariate Archimedean copula tail modeling (Gumbel, Clayton, Frank)
while introducing multi-stage directed cascading failure pathways across
energy lifelines, atmospheric inversions, and hydrometeorological transport networks.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import scipy.stats as stats

# Preserve and expose full Copula Engine functionality
from climacore.analytics.copula import (
    CompoundCopulaEngine,
    CopulaFamily,
    CopulaFitResult,
    JointReturnPeriodResult,
)


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0))))


class CompoundRiskV2:
    """Evaluates multi-stage cascading risk networks and joint copula tail dependence."""

    def __init__(self, copula_engine: Optional[CompoundCopulaEngine] = None):
        self.copula_engine = copula_engine or CompoundCopulaEngine()

    def evaluate_cascading_network(
        self,
        city: str,
        temp_max_c: float,
        temp_min_c: float,
        rainfall_mm: float,
        pm2_5_ug_m3: float,
        is_tropical_night: bool,
        grid_deficit_prob: float,
        flood_hazard_prob: float,
        drought_regime_prob: float = 0.0,
        smoke_advection_index: float = 0.0,
        stagnation_index: float = 1.0,
        hydrograph_peak_m3s: float = 0.0,
        routed_river_peak_m3s: float = 0.0,
        population_15km: Optional[int] = None,
        dense_cluster_pop_15km: Optional[int] = None,
        bridges_15km: Optional[int] = None,
        drainage_canals_15km: Optional[int] = None,
        arterial_roads_15km: Optional[int] = None,
        copula_p_and: float = 0.0,
    ) -> Dict[str, Any]:
        """Evaluates all 3 cascading failure networks with uncertainty corridors and composite score."""
        pop = population_15km or 2_500_000
        dense_pop = dense_cluster_pop_15km or int(pop * 0.15)
        brdg = bridges_15km or 40
        canals = drainage_canals_15km or 50
        roads = arterial_roads_15km or 100
        drainage_ratio = canals / max(1, roads)

        pathways = {}
        pathway_scores = []

        # =========================================================================
        # PATHWAY A: Heat -> Grid Deficit -> Municipal Water -> Health Casualty Cascade
        # =========================================================================
        # Node A1: Thermal Overheat Trigger
        p_heat = sigmoid(0.4 * (temp_max_c - 40.0) + (1.2 if is_tropical_night else -0.5))
        # Node A2: Substation Transformer Thermal Derating
        p_derate = float(np.clip(0.40 * p_heat + 0.35 * (1.0 if is_tropical_night else 0.0) + 0.25 * grid_deficit_prob, 0.0, 0.98))
        # Node A3: Feeder Outage & Substation Trip
        p_trip = float(1.0 - (1.0 - p_derate) * (1.0 - np.clip(grid_deficit_prob, 0.0, 0.95)))
        # Node A4: Municipal Water Pumping & Cold Chain Disruption
        p_pumping_outage = float(0.85 * p_trip)
        # Node A5: Dense Informal Cluster Heat Casualty Vulnerability
        p_casualty_spike = float(np.clip(0.35 * p_pumping_outage + 0.45 * (dense_pop / 1_000_000.0) * p_heat + 0.20 * (1.0 if is_tropical_night else 0.0), 0.0, 0.95))

        pathway_a_prob = float(p_heat * p_trip * p_casualty_spike) ** (1.0 / 3.0)
        pathway_scores.append(pathway_a_prob)

        pathways["pathway_a_thermal_grid_health"] = {
            "name": "Extreme Heat -> Power Grid -> Pumping & Casualty Cascade",
            "trigger_ambient_tmax_c": temp_max_c,
            "tropical_night": is_tropical_night,
            "nodes": [
                {"node": "Thermal Overheat", "prob": round(p_heat, 3)},
                {"node": "Transformer Derating", "prob": round(p_derate, 3)},
                {"node": "Substation Trip", "prob": round(p_trip, 3)},
                {"node": "Municipal Pumping Outage", "prob": round(p_pumping_outage, 3)},
                {"node": "Informal Casualty Surge", "prob": round(p_casualty_spike, 3)},
            ],
            "end_to_end_propagation_prob": round(pathway_a_prob, 4),
            "severity_tier": "CRITICAL" if pathway_a_prob >= 0.50 else ("HIGH" if pathway_a_prob >= 0.25 else "MODERATE"),
        }

        # =========================================================================
        # PATHWAY B: Drought -> Fire Advection -> Inversion Stagnation -> Smog Crisis
        # =========================================================================
        # Node B1: Agricultural Soil Drought Trigger
        p_drought = float(np.clip(drought_regime_prob, 0.0, 1.0))
        # Node B2: Upstream Biomass Burning Fire Activity
        p_fire = float(np.clip(p_drought * 0.6 + 0.4 * min(smoke_advection_index / 100.0, 1.0), 0.0, 0.95))
        # Node B3: Boundary Layer Inversion Trap
        p_inversion = float(sigmoid(0.08 * stagnation_index - 1.5))
        # Node B4: Severe PM2.5 Accumulation
        p_pm25 = float(np.clip(pm2_5_ug_m3 / 250.0, 0.0, 1.0))
        # Node B5: Respiratory Emergency Spike
        p_resp_spike = float(np.clip(0.50 * p_inversion * p_pm25 + 0.35 * p_fire + 0.15 * (pop / 2_500_000.0), 0.0, 0.96))

        pathway_b_prob = float(p_fire * p_inversion * p_resp_spike) ** (1.0 / 3.0)
        pathway_scores.append(pathway_b_prob)

        pathways["pathway_b_drought_smoke_health"] = {
            "name": "Drought -> Fire Advection -> Inversion Stagnation -> Toxic Smog Cascade",
            "trigger_drought_prob": round(p_drought, 3),
            "pm25_ug_m3": pm2_5_ug_m3,
            "nodes": [
                {"node": "Agricultural Drought", "prob": round(p_drought, 3)},
                {"node": "Fire Advection Influx", "prob": round(p_fire, 3)},
                {"node": "Inversion Trap", "prob": round(p_inversion, 3)},
                {"node": "Severe PM2.5 Exceedance", "prob": round(p_pm25, 3)},
                {"node": "Respiratory Surge", "prob": round(p_resp_spike, 3)},
            ],
            "end_to_end_propagation_prob": round(pathway_b_prob, 4),
            "severity_tier": "CRITICAL" if pathway_b_prob >= 0.50 else ("HIGH" if pathway_b_prob >= 0.25 else "MODERATE"),
        }

        # =========================================================================
        # PATHWAY C: Deluge -> Runoff Hydrograph -> Fluvial Choke -> Lifeline Severance
        # =========================================================================
        # Node C1: Extreme Rainfall Trigger
        p_deluge = float(sigmoid(0.08 * (rainfall_mm - 50.0)))
        # Node C2: Unit Hydrograph Peak Runoff Surge
        p_runoff = float(np.clip(flood_hazard_prob * 0.7 + 0.3 * min(hydrograph_peak_m3s / 500.0, 1.0), 0.0, 0.98))
        # Node C3: River Reach Overbank Surcharge
        p_fluvial_choke = float(np.clip(0.5 * p_runoff + 0.5 * min(routed_river_peak_m3s / 400.0, 1.0), 0.0, 0.95))
        # Node C4: Arterial Roadway / Subway Inundation
        p_road_submergence = float(np.clip(p_fluvial_choke * (1.2 - min(drainage_ratio, 0.8)), 0.0, 0.96))
        # Node C5: Critical Lifeline Severance (Bridge & Hospital Access Cut)
        p_lifeline_severance = float(np.clip(0.60 * p_road_submergence + 0.25 * (brdg / 50.0) * p_deluge + 0.15, 0.0, 0.95))

        pathway_c_prob = float(p_deluge * p_fluvial_choke * p_lifeline_severance) ** (1.0 / 3.0)
        pathway_scores.append(pathway_c_prob)

        pathways["pathway_c_deluge_runoff_lifelines"] = {
            "name": "Deluge -> Runoff Peak -> Fluvial Surcharge -> Lifeline Severance",
            "trigger_rainfall_mm": rainfall_mm,
            "hydrograph_peak_m3s": round(hydrograph_peak_m3s, 2),
            "nodes": [
                {"node": "Deluge Burst", "prob": round(p_deluge, 3)},
                {"node": "Peak Runoff Hydrograph", "prob": round(p_runoff, 3)},
                {"node": "Fluvial Reach Choke", "prob": round(p_fluvial_choke, 3)},
                {"node": "Arterial Road Submergence", "prob": round(p_road_submergence, 3)},
                {"node": "Emergency Lifeline Severance", "prob": round(p_lifeline_severance, 3)},
            ],
            "end_to_end_propagation_prob": round(pathway_c_prob, 4),
            "severity_tier": "CRITICAL" if pathway_c_prob >= 0.50 else ("HIGH" if pathway_c_prob >= 0.25 else "MODERATE"),
        }

        # =========================================================================
        # COMPOSITE CASCADING COMPOUND SCORE (0 - 100) & UNCERTAINTY CORRIDOR
        # =========================================================================
        max_pathway_prob = float(np.max(pathway_scores))
        
        # Integration of Bivariate Copula Tail Probability and Directed Cascade Probability
        # w_copula = 0.35, w_cascade = 0.65
        copula_contribution = float(np.clip(copula_p_and * 10.0, 0.0, 1.0))
        composite_prob = 0.35 * copula_contribution + 0.65 * max_pathway_prob
        composite_score = round(float(np.clip(composite_prob * 100.0, 0.0, 100.0)), 1)

        # Bayesian Uncertainty Corridor (95% credible interval using Beta prior/posterior approximation)
        alpha_post = 2.0 + composite_prob * 30.0
        beta_post = 2.0 + (1.0 - composite_prob) * 30.0
        lower_score = round(float(stats.beta.ppf(0.025, alpha_post, beta_post) * 100.0), 1)
        upper_score = round(float(stats.beta.ppf(0.975, alpha_post, beta_post) * 100.0), 1)

        return {
            "city": city,
            "compound_cascade_score": composite_score,
            "uncertainty_corridor_95ci": [lower_score, upper_score],
            "dominant_pathway": max(pathways.items(), key=lambda x: x[1]["end_to_end_propagation_prob"])[0],
            "pathways": pathways,
            "copula_and_probability": round(copula_p_and, 5),
            "governing_policy_action": (
                "ACTIVATE_FULL_MULTI_AGENCY_CASCADE_PROTOCOL" if composite_score >= 60.0
                else ("STANDBY_CROSS_SECTORAL_RESPONSE" if composite_score >= 35.0 else "ROUTINE_MONITORING")
            ),
        }
