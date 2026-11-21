"""
Autonomous Multi-Agent Crisis Council (Synthetic Incident Command System - ICS War Room).
Simulates multi-sector authority deliberation across Public Health, Power Grid, and
Supply Chain Logistics under severe climate shocks, synthesizing an optimized compromise directive.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CouncilMemberProposal(BaseModel):
    """Sector authority position paper presented to the Crisis Council."""
    member_id: str
    title: str
    core_priority: str
    demands: List[str]
    objections_to_others: Dict[str, str]
    statutory_standard: str
    resource_request: str


class DebateTurn(BaseModel):
    """Chronological turn in the War Room debate transcript."""
    turn_number: int
    speaker_id: str
    speaker_title: str
    stance: str  # 'OPENING_POSITION', 'CROSS_SECTOR_OBJECTION', 'COUNTERPROPOSAL', 'EXECUTIVE_SYNTHESIS'
    argument: str


class ExecutiveDirective(BaseModel):
    """Enforceable operational policy directive formulated by the Incident Commander."""
    directive_id: str
    title: str
    mandate: str
    enforcement_window: str
    affected_sectors: List[str]
    mitigated_risk: str


class CrisisCouncilDebate(BaseModel):
    """Complete multi-agent war-room deliberation and executive incident consensus."""
    incident_id: str
    city: str
    incident_type: str
    severity_tier: str  # 'CRITICAL', 'HIGH', 'MODERATE'
    initiating_metrics: Dict[str, Any]
    member_proposals: List[CouncilMemberProposal]
    debate_transcript: List[DebateTurn]
    friction_matrix: Dict[str, List[str]]
    executive_directives: List[ExecutiveDirective]
    accepted_tradeoffs: List[str]
    life_preservation_index: float  # 0.0 to 100.0
    economic_continuity_index: float  # 0.0 to 100.0
    verification_hash: str
    ics_form_reference: str = "ICS-201 Incident Briefing Protocol"
    timestamp_ist: str
    mermaid_sequence_diagram: str


def simulate_crisis_council_deliberation(
    city: str,
    temp_max_c: float,
    temp_min_c: float,
    rainfall_mm: float,
    wet_bulb_c: Optional[float] = None,
    pm25: Optional[float] = None,
    peak_grid_load_mw: Optional[float] = None,
) -> CrisisCouncilDebate:
    """
    Run deterministic multi-agent deliberation resolving real-world conflicts between
    Public Health, Power Grid, and Cold-Chain Logistics under extreme atmospheric stress.
    """
    tw = wet_bulb_c if wet_bulb_c is not None else round(temp_max_c * 0.65 + 5.0, 1)
    grid_mw = peak_grid_load_mw if peak_grid_load_mw is not None else 8500.0

    # Determine primary incident archetype
    if rainfall_mm >= 70.0:
        incident_type = "URBAN_CLOUDBURST_DELUGE"
        severity = "CRITICAL" if rainfall_mm >= 120.0 else "HIGH"
    elif temp_max_c >= 40.0 or tw >= 28.5:
        incident_type = "SEVERE_HEATWAVE_GRID_COLLAPSE"
        severity = "CRITICAL" if (temp_max_c >= 44.0 or tw >= 30.0) else "HIGH"
    elif pm25 is not None and pm25 >= 250.0:
        incident_type = "SEVERE_HAZE_STAGNATION"
        severity = "HIGH"
    else:
        incident_type = "COMPOUND_ATMOSPHERIC_VOLATILITY"
        severity = "MODERATE"

    now_ist = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S IST")
    hash_seed = f"{city}-{incident_type}-{temp_max_c}-{rainfall_mm}-{int(time.time())}"
    raw_hash = hashlib.sha256(hash_seed.encode("utf-8")).hexdigest()[:16].upper()
    incident_id = f"ICS-201-{city[:3].upper()}-{raw_hash[:6]}"

    member_proposals: List[CouncilMemberProposal] = []
    debate_transcript: List[DebateTurn] = []
    friction_matrix: Dict[str, List[str]] = {}
    directives: List[ExecutiveDirective] = []
    accepted_tradeoffs: List[str] = []

    # =========================================================================
    # ARCHETYPE 1: SEVERE HEATWAVE & POWER GRID OVERLOAD
    # =========================================================================
    if incident_type == "SEVERE_HEATWAVE_GRID_COLLAPSE":
        # 1. Health Proposal
        member_proposals.append(CouncilMemberProposal(
            member_id="public_health_officer",
            title="NDMA Public Health & Civil Defense Director",
            core_priority="Wet-Bulb Hyperthermia Prevention & Zero Heat Casualty Mandate",
            demands=[
                "Issue mandatory Section 144 curfew on outdoor manual labor between 10:30 and 16:30 IST.",
                "Requisition 40 municipal schools and community halls as emergency air-conditioned cooling centers.",
                "Erect 80 hydration and oral rehydration salt (ORS) misting kiosks across bus depots and metro interchanges.",
                "Order emergency departments to pre-position rapid-immersion ice tubs for exertional heat stroke patients.",
            ],
            objections_to_others={
                "grid_operations_chief": "Grid chief's proposal to institute rolling residential power cuts will turn uninsulated urban homes into hyperthermic death-traps for vulnerable elderly citizens.",
                "coldchain_logistics_director": "Allowing midday diesel reefer truck movements through congested city centers adds significant localized thermal exhaust and toxic particulate emissions."
            },
            statutory_standard="NDMA National Heat Action Plan (HAP) Guidelines 2024 / ISO 7243 WBGT Criteria",
            resource_request="₹4.5 Crores from State Disaster Response Fund (SDRF) for mobile misting units and ORS packets.",
        ))

        # 2. Grid Proposal
        member_proposals.append(CouncilMemberProposal(
            member_id="grid_operations_chief",
            title="State Load Dispatch Centre (SLDC) Grid Operations Chief",
            core_priority="Bulk Transmission Grid Stability & Transformer Thermal Derating Prevention",
            demands=[
                f"Peak cooling demand is surging by +{int(20 + (temp_max_c - 40) * 3.5)}% ({grid_mw * 1.25:.0f} MW projected). Non-essential commercial HVAC must be capped at 25°C thermostat floor.",
                "Prohibit daytime agricultural feeder energization; restrict tubewell irrigation exclusively to 09:00 - 13:00 solar hours.",
                "Deny uncoordinated new cooling center connections to already stressed 33/11 kV city substations running at 94% thermal capacity.",
                "Retain 850 MW of fast-acting spinning hydro reserves to prevent frequency dipping below the 49.90 Hz statutory boundary.",
            ],
            objections_to_others={
                "public_health_officer": "Health department cannot indiscriminately fire up 40 massive AC cooling centers on weak distribution transformers without triggering explosive transformer oil flashovers (IEEE Std C57.91-2011).",
                "coldchain_logistics_director": "Cold storage warehouses cannot run uninterrupted deep-chilling compressors during the 18:00 - 22:00 evening peak without paying punitive peak-tariff surcharges or facing scheduled load-shedding."
            },
            statutory_standard="Central Electricity Authority (CEA) Technical Standards for Grid Connectivity / IEEE Std C57.91-2011",
            resource_request="Immediate authorization to schedule 600 MW emergency inter-regional gas-fired spinning peaker capacity.",
        ))

        # 3. Logistics Proposal
        member_proposals.append(CouncilMemberProposal(
            member_id="coldchain_logistics_director",
            title="State Cold-Chain & Perishable Food Logistics Director",
            core_priority="Cold-Chain Integrity & Perishable Food/Medicine Spoilage Mitigation",
            demands=[
                "Establish dedicated 'Reefer Green Corridors' with dedicated traffic police pilot escorts along the outer expressway bypass.",
                "Permit APMC wholesale agricultural markets to execute auctions exclusively in the nocturnal cool window (04:00 - 08:30 IST).",
                "Designate cold-chain logistics warehouses storing dairy, poultry, and biological vaccines as Tier-1 Non-Curtailable Power Assets.",
                "Pre-cool refrigerated container fleet and static cold storage hubs down to -4°C during the overnight off-peak trough (00:00 - 06:00 IST).",
            ],
            objections_to_others={
                "public_health_officer": "Total daytime vehicle curfews will trap 2,800 perishable milk and vegetable reefers in 45°C ambient heat on arterial highways, causing diesel compressor failure and ₹42 Crores in food spoilage.",
                "grid_operations_chief": "Cutting power to peripheral cold-storage warehouses will destroy 180,000 doses of temperature-sensitive insulin and biological vaccines within 4 hours."
            },
            statutory_standard="National Centre for Cold-chain Development (NCCD) Guidelines / BIS IS 10500 Potable Integrity",
            resource_request="Priority diesel refueling allocation for 1,200 emergency reefer generator units.",
        ))

        # Debate Turns
        debate_transcript = [
            DebateTurn(
                turn_number=1,
                speaker_id="public_health_officer",
                speaker_title="NDMA Public Health Director",
                stance="OPENING_POSITION",
                argument=f"Wet-bulb temperature in {city} is breaching {tw:.1f}°C alongside ambient highs of {temp_max_c:.1f}°C. Human sweat evaporation is thermodynamically restricted. We must immediately declare an occupational shutdown between 10:30 and 16:30 and energize 40 public cooling shelters.",
            ),
            DebateTurn(
                turn_number=2,
                speaker_id="grid_operations_chief",
                speaker_title="SLDC Grid Chief",
                stance="CROSS_SECTOR_OBJECTION",
                argument=f"An uncoordinated cooling shelter roll-out will collapse our 33 kV urban substations. Step-up transformers are already operating with top-oil temperatures at 89°C. If air conditioning ramps up unchecked, we risk a multi-district blackout that would take down hospitals and municipal water pumping alike.",
            ),
            DebateTurn(
                turn_number=3,
                speaker_id="coldchain_logistics_director",
                speaker_title="Logistics Director",
                stance="CROSS_SECTOR_OBJECTION",
                argument="If Health blocks arterial freight movement and Grid sheds power to cold storage parks, the entire city's perishable milk, poultry, and life-saving insulin supply will spoil by midnight. We demand non-negotiable Tier-1 feeder priority and bypass green corridors.",
            ),
            DebateTurn(
                turn_number=4,
                speaker_id="public_health_officer",
                speaker_title="NDMA Public Health Director",
                stance="COUNTERPROPOSAL",
                argument="We concede that hospital and cold-chain power must be preserved. In exchange, Cold Chain must complete deliveries before 09:30 IST, and Grid must guarantee that primary municipal cooling hubs receive uninterrupted power with a mandatory 25°C thermostat floor.",
            ),
            DebateTurn(
                turn_number=5,
                speaker_id="incident_commander",
                speaker_title="Incident Commander (Municipal Commissioner)",
                stance="EXECUTIVE_SYNTHESIS",
                argument=f"Council arguments heard. We will not allow a catastrophic choice between heat stroke and grid collapse. We hereby enforce an optimized ICS-201 compromise: Bimodal shift schedules, Tier-1 Express Feeder designation, and Reefer Green Corridors.",
            ),
        ]

        # Friction Matrix
        friction_matrix = {
            "Health_vs_Grid": [
                "Health's cooling shelters demand +180 MW localized power vs Grid's transformer thermal overload limits.",
                "Grid's proposed residential load-shedding conflicts with Health's elderly hyperthermia prevention.",
            ],
            "Health_vs_Logistics": [
                "Health's daytime transit curb traps 2,800 perishable reefer trucks in 45°C ambient solar radiation.",
                "Logistics' truck exhaust worsens local urban street canyon thermal index.",
            ],
            "Grid_vs_Logistics": [
                "Logistics demands continuous cold-storage compressor energization vs Grid's 18:00-22:00 peak shaving requirement.",
            ]
        }

        # Executive Directives
        directives = [
            ExecutiveDirective(
                directive_id="DIR-HEAT-01",
                title="Bimodal Shift Work & Mandatory Midday Labor Cessation",
                mandate="All outdoor construction and physical labor is legally shifted to dawn (05:00 - 10:00 IST) and evening (18:00 - 21:00 IST) shifts. Complete work halt enforced between 10:30 and 17:00 IST with zero wage deduction under Disaster Management Act 2005.",
                enforcement_window="10:30 - 17:00 IST Daily",
                affected_sectors=["Public Health", "Construction", "Municipal Administration"],
                mitigated_risk="Eliminates peak-hour exertional heat strokes and hyperthermic casualties.",
            ),
            ExecutiveDirective(
                directive_id="DIR-GRID-02",
                title="Tier-1 Express Feeder Ringfencing & Mandatory 25°C AC Floor",
                mandate="Designate all Government Hospitals, Trauma ICUs, Municipal Pumping Stations, and Registered Cold Storage Warehouses as Tier-1 Non-Curtailable Express Feeders. Mandate a strict 25°C thermostat floor on commercial air conditioners to curb peak demand spike to under +16%.",
                enforcement_window="Immediate (24/7 Duration of Heatwave)",
                affected_sectors=["Power Grid", "Commercial Retail", "Hospitals"],
                mitigated_risk="Prevents transformer thermal flashover and regional grid collapse.",
            ),
            ExecutiveDirective(
                directive_id="DIR-LOG-03",
                title="Reefer Freight Green Corridor & Mandi Nocturnal Trading",
                mandate="Establish dedicated toll-free Green Corridors on elevated outer bypass expressways with traffic police priority for refrigerated dairy and pharmaceutical trucks. Shift APMC wholesale mandi auctions exclusively to 03:30 - 08:30 IST.",
                enforcement_window="03:30 - 08:30 IST & 21:00 - 02:00 IST",
                affected_sectors=["Cold Chain", "Traffic Police", "Agriculture Mandis"],
                mitigated_risk="Prevents ₹42+ Crores in cold-chain food and vaccine spoilage.",
            ),
            ExecutiveDirective(
                directive_id="DIR-CIVIL-04",
                title="Municipal Pre-Cooling Protocol & Solar-Powered Misting Kiosks",
                mandate="Enforce mandatory thermal pre-cooling of cold-storage facilities to -4°C during the 00:00 - 06:00 off-peak trough. Deploy 80 decentralized solar-battery misting and ORS kiosks at major transit interchanges.",
                enforcement_window="00:00 - 06:00 IST (Pre-cooling) & 09:00 - 18:00 IST (Misting)",
                affected_sectors=["Public Health", "Municipal Transit", "Power Grid"],
                mitigated_risk="Flattens evening grid load spikes while protecting transit commuters.",
            ),
        ]

        accepted_tradeoffs = [
            "Commercial shopping malls and office towers will operate with reduced cooling (25°C floor) causing mild indoor thermal discomfort.",
            "Midday retail and non-essential commercial activities will experience reduced economic footfall between 11:00 and 16:30.",
            "Agricultural tubewell pumping is curtailed during the evening peak, requiring farmers to irrigate during morning solar hours.",
        ]
        life_index = 94.5
        econ_index = 82.0

        seq_diagram = """sequenceDiagram
    autonumber
    actor H as NDMA Public Health Officer
    actor G as SLDC Grid Operations Chief
    actor L as Cold-Chain Logistics Director
    actor C as Incident Commander (DM)
    H->>C: Demand mandatory Section 144 midday labor halt & 40 AC cooling hubs
    G->>C: Object: Residential + AC load (+32%) threatens transformer thermal blowout
    L->>C: Object: Midday transit blockade strands 2,800 perishable reefer trucks
    H->>L: Propose nocturnal shift (04:00-08:30) for APMC mandi & refrigerated delivery
    G->>H: Agree to supply 40 cooling centers IF commercial AC is restricted to 25C floor
    C->>H: Enact DIR-01: Bimodal Shift Work (05:00-10:00 & 18:00-21:00)
    C->>G: Enact DIR-02: Ringfence Hospitals & Cold Storage onto Tier-1 Express Feeders
    C->>L: Enact DIR-03: Open Dedicated Reefer Green Corridors on outer bypass
    C->>G: Enact DIR-04: Mandate overnight pre-cooling (00:00-06:00) during grid trough"""

    # =========================================================================
    # ARCHETYPE 2: URBAN CLOUDBURST & PLUVIAL INUNDATION
    # =========================================================================
    elif incident_type == "URBAN_CLOUDBURST_DELUGE":
        member_proposals.append(CouncilMemberProposal(
            member_id="public_health_officer",
            title="NDMA Public Health & Civil Defense Director",
            core_priority="Drowning Prevention, Slum Evacuation & Waterborne Disease Control",
            demands=[
                f"Emergency evacuation of 18 low-lying informal settlements within 400m of swollen drainage channels ({rainfall_mm:.1f} mm deluge).",
                "Immediate closure of submerged pedestrian subways and road underpasses.",
                "Preposition 50,000 chlorine purification tablets and oral doxycycline packets for leptospirosis prophylaxis.",
            ],
            objections_to_others={
                "grid_operations_chief": "Grid's delay in de-energizing submerged roadside distribution boxes poses severe electrocution hazard to wading pedestrians.",
                "coldchain_logistics_director": "Heavy transport trucks navigating flooded streets create wake waves that flood ground-floor residential homes.",
            },
            statutory_standard="NDMA Urban Flooding Guidelines / National Disaster Management Act 2005",
            resource_request="Deployment of 4 National Disaster Response Force (NDRF) inflatable boat teams.",
        ))

        member_proposals.append(CouncilMemberProposal(
            member_id="grid_operations_chief",
            title="State Load Dispatch Centre (SLDC) Grid Operations Chief",
            core_priority="Electrocution Prevention & Dewatering Pump Station Feeder Reliability",
            demands=[
                "Selective smart de-energization of 11 kV feeders in inundated wards with water levels > 0.5m.",
                "Absolute priority power supply to municipal stormwater pumping stations at outfall gates.",
                "Mandatory deployment of auxiliary mobile diesel generators to all tertiary drainage sumps.",
            ],
            objections_to_others={
                "public_health_officer": "Blanket city-wide power cuts requested by local police will disable stormwater outfall pumps, exacerbating the flood depth.",
                "coldchain_logistics_director": "Trucks attempting to navigate flooded underpasses are knocking down low-hanging overhead cables and submerging pillar boxes."
            },
            statutory_standard="Central Electricity Authority Safety Regulations 2023",
            resource_request="Emergency allocation of 12 mobile high-capacity diesel generators.",
        ))

        member_proposals.append(CouncilMemberProposal(
            member_id="coldchain_logistics_director",
            title="State Cold-Chain & Perishable Food Logistics Director",
            core_priority="Essential Cargo Mobility & Food Pipeline Protection",
            demands=[
                "Barricade flooded low-lying underpasses while preserving elevated flyover arterial highways exclusively for essential supply vehicles.",
                "Designate high-ground logistics staging yards for milk tankers and medical oxygen carriers.",
                "Ensure emergency fuel distribution stations retain priority power for reefer transport and ambulances.",
            ],
            objections_to_others={
                "public_health_officer": "Complete transit shutdown isolates city hospitals from oxygen tanker replenishment.",
                "grid_operations_chief": "De-energizing logistics hubs halts warehouse cold-chain refrigeration.",
            },
            statutory_standard="Motor Vehicles (Emergency Transit) Regulations",
            resource_request="Dedicated traffic police pilot vehicles for 15 medical oxygen tankers.",
        ))

        debate_transcript = [
            DebateTurn(
                turn_number=1,
                speaker_id="public_health_officer",
                speaker_title="NDMA Public Health Director",
                stance="OPENING_POSITION",
                argument=f"Cloudburst rainfall of {rainfall_mm:.1f} mm has inundated low-lying settlements. We must immediately cut power to flooded wards to prevent electrocution and evacuate 8,000 citizens to school shelters.",
            ),
            DebateTurn(
                turn_number=2,
                speaker_id="grid_operations_chief",
                speaker_title="SLDC Grid Chief",
                stance="CROSS_SECTOR_OBJECTION",
                argument="A blanket blackout will shut down the municipal dewatering pumps at the river sluice gates, causing floodwaters to rise another 1.2 meters. We must selectively isolate street-level distribution boxes while keeping pump feeders energized.",
            ),
            DebateTurn(
                turn_number=3,
                speaker_id="coldchain_logistics_director",
                speaker_title="Logistics Director",
                stance="CROSS_SECTOR_OBJECTION",
                argument="Key arterial underpasses are drowned. If trucks are halted indiscriminately, medical oxygen tankers cannot reach city trauma wards. We must enforce elevated flyover routing.",
            ),
            DebateTurn(
                turn_number=4,
                speaker_id="incident_commander",
                speaker_title="Incident Commander (Municipal Commissioner)",
                stance="EXECUTIVE_SYNTHESIS",
                argument="Directives formulated: Selective smart feeder de-energization, continuous dewatering pump power ringfencing, elevated arterial green corridors for oxygen and perishables, and NDRF boat evacuations.",
            ),
        ]

        friction_matrix = {
            "Health_vs_Grid": [
                "Health demands broad feeder shutdowns to stop electrocution vs Grid needing power to run high-capacity dewatering pumps.",
            ],
            "Health_vs_Logistics": [
                "Health's full traffic shutdown blocks medical oxygen and emergency food resupply.",
            ],
            "Grid_vs_Logistics": [
                "Flooded trucks risking collisions with electrical substations and submerged pillar boxes.",
            ]
        }

        directives = [
            ExecutiveDirective(
                directive_id="DIR-FLOOD-01",
                title="Selective Feeder Isolation & Dewatering Pump Protection",
                mandate="Enforce automated SCADA trip of ground-mounted 11 kV distribution boxes in wards exceeding 0.4m flood depth. Stormwater outfall pumping stations ringfenced with dual-circuit express feeders and on-site diesel backup.",
                enforcement_window="Immediate (Active Deluge Window)",
                affected_sectors=["Power Grid", "Municipal Drainage", "Disaster Management"],
                mitigated_risk="Eliminates public electrocutions while maintaining 100% dewatering pumping capacity.",
            ),
            ExecutiveDirective(
                directive_id="DIR-FLOOD-02",
                title="Elevated Flyover Green Corridor & Underpass Hard Closure",
                mandate="Physical deployment of automated flood barriers at all 14 urban subway underpasses. Elevated flyovers and arterial bypasses reserved exclusively for emergency responders, ambulances, oxygen tankers, and essential food reefers.",
                enforcement_window="Continuous until floodwaters recede below 0.15m",
                affected_sectors=["Traffic Police", "Logistics", "Emergency Medical Services"],
                mitigated_risk="Prevents vehicle submersion drownings and maintains life-saving hospital supply chains.",
            ),
            ExecutiveDirective(
                directive_id="DIR-FLOOD-03",
                title="Preemptive Riverside Slum Evacuation & Medical Prophylaxis",
                mandate="Mobilize 4 NDRF boat companies to evacuate 18 low-lying informal settlements to designated municipal high-ground relief centers. Distribute 50,000 chlorine water purification packs and doxycycline to prevent leptospirosis.",
                enforcement_window="Next 4 Hours",
                affected_sectors=["Public Health", "NDRF", "Civil Defense"],
                mitigated_risk="Prevents flood drownings and waterborne epidemic outbreaks.",
            ),
        ]

        accepted_tradeoffs = [
            "Certain flooded residential pockets will face temporary selective power cuts until municipal dewatering lowers water levels.",
            "Private vehicle transit on arterial roads is restricted, requiring commuters to use elevated mass transit or remain sheltered.",
            "Retail distribution of non-essential goods is suspended for 24 hours.",
        ]
        life_index = 96.0
        econ_index = 74.5

        seq_diagram = """sequenceDiagram
    autonumber
    actor H as NDMA Public Health Officer
    actor G as SLDC Grid Operations Chief
    actor L as Cold-Chain Logistics Director
    actor C as Incident Commander (DM)
    H->>C: Demand immediate power cuts to flooded wards & slum evacuation
    G->>C: Warn: Blanket blackout will disable municipal dewatering pumps
    L->>C: Warn: Underpass inundation blocks medical oxygen tankers to trauma ICUs
    G->>H: Propose smart selective SCADA trips: cut street pillars, keep pumps running
    L->>H: Request elevated flyover corridor reserved for oxygen & perishable reefers
    C->>G: Enact DIR-01: Selective Feeder Isolation & Pump Station Ringfencing
    C->>L: Enact DIR-02: Hard Underpass Closure & Elevated Flyover Green Corridor
    C->>H: Enact DIR-03: NDRF Riverside Slum Evacuation & Water Purification Protocol"""

    # =========================================================================
    # ARCHETYPE 3: BALANCED ATMOSPHERIC VOLATILITY / HAZE
    # =========================================================================
    else:
        member_proposals.append(CouncilMemberProposal(
            member_id="public_health_officer",
            title="NDMA Public Health & Civil Defense Director",
            core_priority="Respiratory Vulnerability Protection & Air Quality Triage",
            demands=[
                "Issue public advisory for vulnerable cardiac/asthmatic citizens to remain indoors.",
                "Deploy anti-smog misting cannons along arterial boulevards and high-density bus stands.",
                "Enforce construction site dust suppression protocols under CPCB guidelines.",
            ],
            objections_to_others={
                "grid_operations_chief": "Thermal coal plants upwind continue contributing secondary aerosol sulfates.",
                "coldchain_logistics_director": "Older diesel trucks idling at toll gates intensify ground-level PM2.5 spikes.",
            },
            statutory_standard="CPCB Graded Response Action Plan (GRAP) Stage-III / National Ambient Air Quality Standards",
            resource_request="₹1.2 Crores for mobile water sprinkler deployment.",
        ))

        member_proposals.append(CouncilMemberProposal(
            member_id="grid_operations_chief",
            title="State Load Dispatch Centre (SLDC) Grid Operations Chief",
            core_priority="Renewable Power Dispatch & Clean Merit-Order Balancing",
            demands=[
                "Maximize green energy absorption from regional solar/wind corridors.",
                "Throttle oldest sub-critical thermal units located within a 50 km radius of the urban center.",
                "Maintain baseline voltage stability across transmission ring.",
            ],
            objections_to_others={
                "public_health_officer": "Sudden closure of thermal peakers without battery storage creates voltage instability.",
                "coldchain_logistics_director": "Charging electric commercial delivery fleets during peak hours strains substation transformers.",
            },
            statutory_standard="CEA Merit-Order Dispatch Regulations",
            resource_request="Authorization to dispatch gas peaker reserves.",
        ))

        member_proposals.append(CouncilMemberProposal(
            member_id="coldchain_logistics_director",
            title="State Cold-Chain & Perishable Food Logistics Director",
            core_priority="Fleet Emission Minimization & Night-Time Logistics Flow",
            demands=[
                "Mandate night-time entry (22:00 - 06:00 IST) for heavy commercial trucks to eliminate daytime idling in traffic.",
                "Fast-track dedicated FASTag lanes for non-polluting CNG/Electric delivery vehicles.",
                "Maintain continuous power for warehouse refrigeration cold rooms.",
            ],
            objections_to_others={
                "public_health_officer": "Complete bans on diesel trucks jeopardize dairy and grocery supply chains.",
                "grid_operations_chief": "Demanding smart charging curtails fleet operational flexibility.",
            },
            statutory_standard="National Logistics Policy (NLP) 2022",
            resource_request="Priority EV fast-charging access at municipal depots.",
        ))

        debate_transcript = [
            DebateTurn(
                turn_number=1,
                speaker_id="public_health_officer",
                speaker_title="NDMA Public Health Director",
                stance="OPENING_POSITION",
                argument=f"Atmospheric conditions in {city} indicate severe boundary layer stagnation. We must enforce GRAP measures, halt construction dust, and restrict heavy diesel traffic.",
            ),
            DebateTurn(
                turn_number=2,
                speaker_id="grid_operations_chief",
                speaker_title="SLDC Grid Chief",
                stance="CROSS_SECTOR_OBJECTION",
                argument="We can ramp down older coal units near the city by 30%, provided we have clearance to dispatch hydro and gas reserves without penal merit-order charges.",
            ),
            DebateTurn(
                turn_number=3,
                speaker_id="coldchain_logistics_director",
                speaker_title="Logistics Director",
                stance="COUNTERPROPOSAL",
                argument="Shift heavy commercial vehicle movements strictly to the nocturnal window (22:00 - 06:00 IST) to eliminate idling emissions during daytime thermal inversion.",
            ),
            DebateTurn(
                turn_number=4,
                speaker_id="incident_commander",
                speaker_title="Incident Commander (Municipal Commissioner)",
                stance="EXECUTIVE_SYNTHESIS",
                argument="Directives approved: GRAP Stage-III construction restrictions, nocturnal freight transit windows, and clean merit-order grid dispatch.",
            ),
        ]

        friction_matrix = {
            "Health_vs_Logistics": ["Daytime diesel truck exhaust exacerbates boundary layer particulate traps."],
            "Grid_vs_Health": ["Curtailing local thermal peakers requires expensive imported gas power."],
        }

        directives = [
            ExecutiveDirective(
                directive_id="DIR-HAZE-01",
                title="Nocturnal Freight Window & Mechanical Dust Suppression",
                mandate="Heavy non-essential diesel trucks restricted to 22:00 - 06:00 IST transit. Municipal anti-smog water cannons deployed along top 10 pollution hotspots twice daily.",
                enforcement_window="22:00 - 06:00 IST (Transit) & 06:00 - 10:00 IST (Sprinklers)",
                affected_sectors=["Logistics", "Municipal Administration", "Public Health"],
                mitigated_risk="Reduces street-level respiratory exposure by an estimated 28%.",
            ),
            ExecutiveDirective(
                directive_id="DIR-HAZE-02",
                title="Environmental Merit-Order Generation Dispatch",
                mandate="SLDC shall curtail generation at sub-critical thermal plants within 50 km by 35%, substituting power from regional solar and hydro reserves.",
                enforcement_window="Immediate (Continuous for 48 Hours)",
                affected_sectors=["Power Grid", "Renewable Energy"],
                mitigated_risk="Cuts regional sulfur dioxide and particulate plume advection.",
            ),
        ]

        accepted_tradeoffs = [
            "Higher wholesale electricity tariff incurred during gas-peaker operation.",
            "Delivery schedules for non-perishable freight delayed by 12 hours.",
        ]
        life_index = 88.0
        econ_index = 91.0

        seq_diagram = """sequenceDiagram
    autonumber
    actor H as NDMA Public Health Officer
    actor G as SLDC Grid Operations Chief
    actor L as Cold-Chain Logistics Director
    actor C as Incident Commander (DM)
    H->>C: Demand GRAP Stage-III enforcement & daytime diesel truck bans
    G->>C: Agree to throttle local coal plants by 35% via clean hydro dispatch
    L->>C: Propose nocturnal freight transit window (22:00-06:00) to stop daytime idling
    C->>H: Enact DIR-01: Nocturnal Freight Window & Water Cannons
    C->>G: Enact DIR-02: Environmental Merit-Order Grid Dispatch"""

    return CrisisCouncilDebate(
        incident_id=incident_id,
        city=city,
        incident_type=incident_type,
        severity_tier=severity,
        initiating_metrics={
            "temperature_max_c": temp_max_c,
            "temperature_min_c": temp_min_c,
            "rainfall_mm": rainfall_mm,
            "wet_bulb_c": tw,
            "pm25": pm25,
            "peak_grid_load_mw": grid_mw,
        },
        member_proposals=member_proposals,
        debate_transcript=debate_transcript,
        friction_matrix=friction_matrix,
        executive_directives=directives,
        accepted_tradeoffs=accepted_tradeoffs,
        life_preservation_index=life_index,
        economic_continuity_index=econ_index,
        verification_hash=raw_hash,
        timestamp_ist=now_ist,
        mermaid_sequence_diagram=seq_diagram,
    )
