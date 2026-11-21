"""
Compound Domino Cascade Simulator.
Models multi-tier domino failure propagation across coupled urban, energy,
cold-chain, and civil infrastructure systems under extreme climate shocks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CascadeNode(BaseModel):
    """Individual domino node in the failure chain."""
    step: int
    sector: str  # 'Meteorology', 'Power Grid', 'Cold Chain', 'Urban Transit', 'Public Health'
    event: str
    impact_level: str  # 'LOW', 'MODERATE', 'SEVERE', 'CATASTROPHIC'
    conditional_probability: float
    physical_mechanism: str


class DominoCascadeResult(BaseModel):
    """Complete simulated domino failure cascade."""
    city: str
    initiating_shock: str
    severity_tier: str
    nodes: List[CascadeNode]
    critical_path_summary: str
    total_systemic_vulnerability_score: float  # 0 to 100
    emergency_interventions: List[str]
    mermaid_diagram: str


def simulate_compound_cascade(
    city: str,
    temp_max_c: float,
    temp_min_c: float,
    rainfall_mm: float,
    wet_bulb_c: Optional[float] = None,
    pm25: Optional[float] = None,
    paved_fraction: float = 0.70,
) -> DominoCascadeResult:
    """
    Simulate quantitative domino chain reaction across coupled municipal and physical networks.
    """
    nodes: List[CascadeNode] = []
    interventions: List[str] = []
    step_counter = 1

    # Inferred wet-bulb if missing
    effective_tw = wet_bulb_c if wet_bulb_c is not None else round(temp_max_c * 0.65 + 5.0, 1)

    # -------------------------------------------------------------------------
    # 1. Thermal Shock Domino Chain
    # -------------------------------------------------------------------------
    if temp_max_c >= 40.0:
        is_severe_heat = temp_max_c >= 44.0
        shock_title = f"Extreme Thermal Influx ({temp_max_c:.1f}°C High)"

        # Node 1: Primary Thermal Shock
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Meteorology",
            event=f"Ambient Heatwave Surge ({temp_max_c:.1f}°C)",
            impact_level="SEVERE" if is_severe_heat else "MODERATE",
            conditional_probability=1.0,
            physical_mechanism="Synoptic boundary layer advection exceeding human and mechanical cooling thresholds."
        ))
        step_counter += 1

        # Node 2: Power Grid AC Cooling Surge
        grid_prob = 0.88 if is_severe_heat else 0.65
        surge_mw_pct = int(15 + (temp_max_c - 40.0) * 4.5)
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Power Grid",
            event=f"Peak Cooling Demand Spike (+{surge_mw_pct}% MW Load)",
            impact_level="SEVERE" if is_severe_heat else "MODERATE",
            conditional_probability=grid_prob,
            physical_mechanism="Domestic and commercial HVAC compressors operating at 100% duty cycle."
        ))
        step_counter += 1

        # Node 3: Transformer Nocturnal Dissipation Collapse (CEA Rule)
        if temp_min_c >= 28.0:
            nodes.append(CascadeNode(
                step=step_counter,
                sector="Power Grid Substation",
                event=f"Tropical Night Thermal Collapse (Tmin {temp_min_c:.1f}°C)",
                impact_level="CATASTROPHIC" if is_severe_heat else "SEVERE",
                conditional_probability=0.82,
                physical_mechanism="Substation transformers fail to cool overnight; insulating oil temperature stays > 95°C."
            ))
            interventions.append("Activate auxiliary water misting and mobile cooling fans at primary distribution substations.")
            step_counter += 1

        # Node 4: Cold-Chain Storage & Reefer Logistics Degradation
        cold_prob = 0.74 if is_severe_heat else 0.45
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Cold Chain & Food Security",
            event="Reefer Transport Cooling Breach & Perishable Spoilage",
            impact_level="SEVERE" if is_severe_heat else "MODERATE",
            conditional_probability=cold_prob,
            physical_mechanism="Refrigerated transport units exceed ambient heat rejection delta; vaccine/dairy spoilage accelerates 3x."
        ))
        interventions.append("Switch cold storage warehouses to dedicated diesel generator peakers with dual-circuit redundancy.")
        step_counter += 1

        # Node 5: Public Health & Labor Halt (NDMA Rule)
        if effective_tw >= 29.0:
            nodes.append(CascadeNode(
                step=step_counter,
                sector="Public Health & Civil Labor",
                event=f"Wet-Bulb Labor Shutdown (Tw {effective_tw:.1f}°C)",
                impact_level="CATASTROPHIC" if effective_tw >= 31.0 else "SEVERE",
                conditional_probability=0.92,
                physical_mechanism="Perspiration cooling ceases; severe risk of fatal exertional heatstroke for gig/construction workers."
            ))
            interventions.append("Enforce statutory NDMA prohibition on outdoor physical labor between 12:00 PM and 3:30 PM.")
            step_counter += 1

    # -------------------------------------------------------------------------
    # 2. Hydrological Cloudburst Domino Chain
    # -------------------------------------------------------------------------
    elif rainfall_mm >= 45.0:
        is_cloudburst = rainfall_mm >= 100.0
        shock_title = f"Precipitation Cloudburst Influx ({rainfall_mm:.1f} mm)"

        # Node 1: Rainfall Influx
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Hydrology",
            event=f"Deluge Rainfall Influx ({rainfall_mm:.1f} mm)",
            impact_level="CATASTROPHIC" if is_cloudburst else "SEVERE",
            conditional_probability=1.0,
            physical_mechanism="Convective storm cluster depositing intense precipitation exceeding infiltration capacity."
        ))
        step_counter += 1

        # Node 2: SCS Surface Runoff Surcharge
        runoff_mm = max(0.0, (rainfall_mm - 0.2 * 50.0) ** 2 / (rainfall_mm + 0.8 * 50.0))
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Urban Drainage (CPHEEO)",
            event=f"Gravity Storm Sewer Surcharge ({runoff_mm:.1f} mm Runoff)",
            impact_level="SEVERE",
            conditional_probability=0.89,
            physical_mechanism=f"Paved impervious surface ({int(paved_fraction*100)}%) prevents soil percolation, overwhelming 35mm/h storm canals."
        ))
        interventions.append("Deploy high-capacity diesel de-watering pumps to primary subway underpasses and hospital basements.")
        step_counter += 1

        # Node 3: Urban Road Network & Transit Paralysis
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Urban Transit & Logistics",
            event="Arterial Road & Subway Underpass Submergence",
            impact_level="SEVERE" if is_cloudburst else "MODERATE",
            conditional_probability=0.81,
            physical_mechanism="Low-lying transit junctions submerge > 0.6m, paralyzing bus transit and halting last-mile delivery fleets."
        ))
        interventions.append("Activate municipal traffic police diversions around designated red-zone waterlogged junctions.")
        step_counter += 1

        # Node 4: Post-Deluge Vector Disease Surge (NCDC Rule)
        nodes.append(CascadeNode(
            step=step_counter,
            sector="Public Health",
            event="Stagnant Ponding & Mosquito Vector Breeding Window",
            impact_level="MODERATE",
            conditional_probability=0.68,
            physical_mechanism="Un-drained standing pools trigger rapid Aedes/Anopheles larval growth with Dengue/Malaria outbreak spike in 10-14 days."
        ))
        interventions.append("Schedule municipal anti-larval chemical fogging and Temephos application within 5 days of rainfall cessation.")
        step_counter += 1

    # -------------------------------------------------------------------------
    # 3. Default Moderate Cascade
    # -------------------------------------------------------------------------
    else:
        shock_title = "Standard Climatic Baseline Conditions"
        nodes.append(CascadeNode(
            step=1,
            sector="Meteorology",
            event=f"Stable Atmospheric Conditions (Max {temp_max_c:.1f}°C, Rain {rainfall_mm:.1f}mm)",
            impact_level="LOW",
            conditional_probability=1.0,
            physical_mechanism="Parameters remain within design capacity of municipal infrastructure."
        ))
        interventions.append("Maintain baseline routine monitoring across all public utility sectors.")

    # Compute systemic vulnerability score
    vuln_weights = {"LOW": 10.0, "MODERATE": 35.0, "SEVERE": 70.0, "CATASTROPHIC": 95.0}
    scores = [vuln_weights.get(n.impact_level, 20.0) * n.conditional_probability for n in nodes]
    vuln_score = round(sum(scores) / max(1, len(scores)), 1)
    severity_tier = "CRITICAL" if vuln_score >= 70.0 else "ELEVATED" if vuln_score >= 40.0 else "NOMINAL"

    # Generate Mermaid Diagram String
    mermaid_lines = ["graph TD"]
    for i in range(len(nodes) - 1):
        curr_n = nodes[i]
        next_n = nodes[i + 1]
        cid = f"N{curr_n.step}"
        nid = f"N{next_n.step}"
        label = f"{int(next_n.conditional_probability * 100)}% prob"
        mermaid_lines.append(f"    {cid}[\"{curr_n.sector}: {curr_n.event}\"] -->|{label}| {nid}[\"{next_n.sector}: {next_n.event}\"]")
    mermaid_str = "\n".join(mermaid_lines)

    summary = (
        f"Compound domino sequence for {city} initiated by {shock_title}. "
        f"Propagation encompasses {len(nodes)} coupled subsystems resulting in a Systemic Vulnerability Index of {vuln_score}/100 ({severity_tier})."
    )

    return DominoCascadeResult(
        city=city,
        initiating_shock=shock_title,
        severity_tier=severity_tier,
        nodes=nodes,
        critical_path_summary=summary,
        total_systemic_vulnerability_score=vuln_score,
        emergency_interventions=interventions,
        mermaid_diagram=mermaid_str,
    )
