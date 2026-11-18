"""
Cascading Multi-Hazard Compound Risk Engine for Clima-Core (Phase 4).
Models non-linear inter-sectoral failure chains and infrastructure dependency collapses:
  Chain 1: Extreme Thermal Shock -> Power Grid Overload -> Municipal Pumping Outage -> Vulnerable Cluster Heat Casualties.
  Chain 2: Cloudburst Deluge -> Drainage Canal Surcharge -> Transportation Lifeline Severance -> Emergency Medical Disruption.
  Chain 3: Atmospheric Inversion -> Toxic Boundary Stagnation -> School & Sensitive Population Health Crisis.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import numpy as np


class CascadingRiskEngine:
    """
    Evaluates cascading multi-hazard dependency failure chains across
    meteorological triggers, engineering lifelines, and demographic exposure nodes.
    """

    def evaluate_cascading_chains(
        self,
        city: str,
        temp_max_c: float,
        temp_min_c: float,
        rainfall_mm: float,
        pm2_5_ug_m3: float,
        is_tropical_night: bool,
        grid_deficit_prob: float,
        flood_hazard_prob: float,
        stagnation_flag: int,
        population_15km: Optional[int] = None,
        dense_cluster_pop_15km: Optional[int] = None,
        hospitals_15km: Optional[int] = None,
        bridges_15km: Optional[int] = None,
        drainage_canals_15km: Optional[int] = None,
        arterial_roads_15km: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate all 3 canonical cascading failure chains and compute composite cascading vulnerability.
        """
        pop = population_15km or 2_500_000
        dense_pop = dense_cluster_pop_15km or int(pop * 0.15)
        hosp = hospitals_15km or 30
        brdg = bridges_15km or 40
        canals = drainage_canals_15km or 50
        roads = arterial_roads_15km or 100
        drainage_ratio = canals / max(1, roads)

        active_chains = []
        node_failures = []
        mitigation_actions = []

        # =========================================================================
        # CHAIN 1: Heat -> Grid Deficit -> Water Pumping -> Informal Health Crisis
        # =========================================================================
        heat_trigger = bool(temp_max_c >= 42.0 or (temp_max_c >= 40.0 and is_tropical_night))
        if heat_trigger:
            chain_1_nodes = [
                {
                    "node_id": "TRANSFORMER_COOLING_IMPAIRMENT",
                    "system": "Electrical Distribution Infrastructure",
                    "failure_probability": round(min(0.95, 0.40 + (0.35 if is_tropical_night else 0.0) + (temp_max_c - 40.0) * 0.05), 2),
                    "estimated_time_to_failure": "0-4 hours of nocturnal peak strain",
                    "status": "ELEVATED_FAILURE_RISK",
                },
                {
                    "node_id": "MUNICIPAL_WATER_PUMPING_DISRUPTION",
                    "system": "Municipal Water Supply Network",
                    "failure_probability": round(min(0.90, grid_deficit_prob * 0.85), 2),
                    "estimated_time_to_failure": "2-6 hours post feeder trip",
                    "status": "SECONDARY_CASCADE_VULNERABLE",
                },
                {
                    "node_id": "INFORMAL_SETTLEMENT_HEAT_CASUALTY_SPIKE",
                    "system": "Public Health & Emergency Response",
                    "failure_probability": round(min(0.92, 0.35 + (dense_pop / 1_000_000.0) * 0.25 + (0.25 if is_tropical_night else 0.0)), 2),
                    "estimated_time_to_failure": "12-24 hours cumulative thermal exposure",
                    "impacted_population": dense_pop,
                    "status": "CRITICAL_CASUALTY_ZONE",
                },
            ]
            active_chains.append({
                "chain_id": "CASCADE_THERMAL_GRID_HEALTH",
                "name": "Thermal Shock -> Power Deficit -> Municipal Water & Health Crisis",
                "trigger_condition": f"Tmax {temp_max_c:.1f}°C with Tropical Night ({temp_min_c:.1f}°C min)",
                "governing_nodes": chain_1_nodes,
                "severity": "CRITICAL" if (grid_deficit_prob >= 0.30 and is_tropical_night) else "HIGH",
            })
            node_failures.extend(chain_1_nodes)
            mitigation_actions.append("DISCOM Mandate: Ring-fence dedicated electrical feeders for municipal water treatment and major civil hospitals.")
            mitigation_actions.append(f"Municipal Water Directorate: Deploy tanker reserves and hydration kiosks to {dense_pop:,} informal settlement residents.")

        # =========================================================================
        # CHAIN 2: Deluge -> Drainage Surcharge -> Transportation Lifeline Severance
        # =========================================================================
        flood_trigger = bool(rainfall_mm >= 50.0 or flood_hazard_prob >= 0.30)
        if flood_trigger:
            chain_2_nodes = [
                {
                    "node_id": "SURFACE_DRAINAGE_SURCHARGE",
                    "system": "Urban Stormwater Network",
                    "failure_probability": round(min(0.98, flood_hazard_prob * 1.15), 2),
                    "estimated_time_to_failure": "1-3 hours of intense precipitation",
                    "status": "IMMEDIATE_OVERLAND_FLOW",
                },
                {
                    "node_id": "CRITICAL_UNDERPASS_AND_BRIDGE_SUBMERGENCE",
                    "system": "Arterial Road Network",
                    "failure_probability": round(min(0.95, (rainfall_mm / 90.0) * 0.70 + (0.25 if drainage_ratio < 0.25 else 0.0)), 2),
                    "estimated_time_to_failure": "2-5 hours post rainfall onset",
                    "bridges_at_risk": brdg,
                    "status": "ROADWAY_PARALYSIS",
                },
                {
                    "node_id": "EMERGENCY_AMBULANCE_ISOLATION",
                    "system": "Healthcare Lifeline Transit",
                    "failure_probability": round(min(0.90, (rainfall_mm / 100.0) * 0.65 + (0.20 if brdg > 30 else 0.0)), 2),
                    "estimated_time_to_failure": "3-8 hours duration",
                    "hospitals_impacted": hosp,
                    "status": "LOGISTIC_SEVERANCE",
                },
            ]
            active_chains.append({
                "chain_id": "CASCADE_DELUGE_DRAINAGE_TRANSIT",
                "name": "Cloudburst Deluge -> Drainage Bottleneck -> Emergency Transit Severance",
                "trigger_condition": f"Rainfall {rainfall_mm:.1f} mm with canal ratio {drainage_ratio:.2f}",
                "governing_nodes": chain_2_nodes,
                "severity": "CRITICAL" if (rainfall_mm >= 80.0 or drainage_ratio < 0.20) else "HIGH",
            })
            node_failures.extend(chain_2_nodes)
            mitigation_actions.append(f"Traffic Police & Municipal DDMA: Pre-emptively barricade {brdg} mapped low-clearance underpasses.")
            mitigation_actions.append(f"Emergency Medical Services: Re-route ambulance corridors to maintain access to {hosp} healthcare facilities.")

        # =========================================================================
        # CHAIN 3: Inversion -> Toxic Trapping -> Sensitive Population Health Crisis
        # =========================================================================
        aq_trigger = bool(pm2_5_ug_m3 >= 150.0 or (stagnation_flag and pm2_5_ug_m3 >= 100.0))
        if aq_trigger:
            chain_3_nodes = [
                {
                    "node_id": "BOUNDARY_LAYER_NOCTURNAL_INVERSION_LID",
                    "system": "Atmospheric Boundary Layer",
                    "failure_probability": 0.88 if stagnation_flag else 0.60,
                    "estimated_time_to_failure": "Immediate (nocturnal cooling onset)",
                    "status": "DISPERSION_SHUTDOWN",
                },
                {
                    "node_id": "GROUND_LEVEL_TOXIC_CONCENTRATION_TRAP",
                    "system": "Urban Airshed Dispersion",
                    "failure_probability": round(min(0.98, pm2_5_ug_m3 / 250.0), 2),
                    "estimated_time_to_failure": "6-18 hours persistence",
                    "status": "SEVERE_ACCUMULATION",
                },
                {
                    "node_id": "PEDIATRIC_AND_GERIATRIC_PULMONARY_SURGE",
                    "system": "Healthcare Inpatient Capacity",
                    "failure_probability": round(min(0.92, 0.45 + (pm2_5_ug_m3 / 400.0) * 0.45), 2),
                    "estimated_time_to_failure": "24-48 hours lag post exposure",
                    "status": "ACUTE_INPATIENT_STRAIN",
                },
            ]
            active_chains.append({
                "chain_id": "CASCADE_INVERSION_AIR_POLLUTION",
                "name": "Atmospheric Inversion -> Toxic Trapping -> Pediatric/Geriatric Emergency Surge",
                "trigger_condition": f"PM2.5 {pm2_5_ug_m3:.1f} µg/m³ with atmospheric stagnation flag {stagnation_flag}",
                "governing_nodes": chain_3_nodes,
                "severity": "CRITICAL" if pm2_5_ug_m3 >= 250.0 else "HIGH",
            })
            node_failures.extend(chain_3_nodes)
            mitigation_actions.append("Education Dept: Suspend all morning outdoor physical activities and convert primary schooling to remote instruction.")
            mitigation_actions.append("Hospital Administration: Stock supplementary bronchodilators, nebulizers, and oxygen beds across municipal emergency wards.")

        # Composite Cascading Risk Score (0 to 100)
        if not active_chains:
            cascade_score = 12.0
            overall_severity = "NORMAL_OPERATIONAL"
        else:
            base_cascade = len(active_chains) * 28.0
            max_failure_prob = max(node["failure_probability"] for node in node_failures) if node_failures else 0.2
            cascade_score = float(np.clip(base_cascade + max_failure_prob * 25.0, 20.0, 98.0))
            if cascade_score >= 75.0 or len(active_chains) >= 2:
                overall_severity = "CATASTROPHIC_MULTI_SECTOR_CASCADE"
            elif cascade_score >= 50.0:
                overall_severity = "ELEVATED_INTER_SECTOR_STRAIN"
            else:
                overall_severity = "MODERATE_LOCALIZED_CASCADE"

        return {
            "city": city,
            "cascading_risk_score": round(cascade_score, 1),
            "cascading_severity_tier": overall_severity,
            "active_cascading_chains_count": len(active_chains),
            "active_chains": active_chains,
            "critical_node_failures": node_failures,
            "cross_sectoral_interventions": mitigation_actions or ["Infrastructure networks operating within coupled safety margins."],
        }
