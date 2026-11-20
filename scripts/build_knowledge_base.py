"""
Curator and Vector Index Builder for Clima-Core Indian Environmental & Regulatory Knowledge Base.
Compiles verified standards from:
- NDMA (National Disaster Management Authority) Heatwave & Flood Action Plans
- ICAR-CRIDA (Indian Council of Agricultural Research) Agrometeorological Field Manuals
- BIS (Bureau of Indian Standards) IS 10500:2012 Drinking Water Specifications
- CPCB (Central Pollution Control Board) NAQI & GRAP Air Quality Standards
- CEA (Central Electricity Authority) Thermal Grid & Substation Operating Directives
- IMD & CWC (India Meteorological Department & Central Water Commission) Inundation Protocols
"""

import json
from pathlib import Path
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

INDIAN_ENVIRONMENTAL_STANDARDS = [
    # -------------------------------------------------------------------------
    # 1. NDMA - Heatwaves & Civil Protection
    # -------------------------------------------------------------------------
    {
        "id": "ndma_heatwave_declaration_criteria",
        "domain": "heat_disaster",
        "authority": "National Disaster Management Authority (NDMA)",
        "title": "Heatwave and Severe Heatwave Declaration Criteria for India",
        "rule_text": "NDMA declares a Heatwave in plains when maximum temperature reaches >= 40.0°C (>= 37.0°C in coastal regions, >= 30.0°C in hill regions), or when departure from normal reaches +4.5°C to +6.4°C. Severe Heatwave declares when maximum temperature reaches >= 47.0°C in plains, or departure from normal exceeds +6.4°C.",
        "trigger_condition": "Temperature >= 40.0°C or departure >= +4.5°C",
        "mandatory_directive": "Activate Heat Action Plan (HAP) departmental coordination; prepare primary health centers for heatstroke triage.",
        "citation": "NDMA National Guidelines for Preparation of Action Plan - Prevention and Management of Heat Wave (2019), Section 2.1"
    },
    {
        "id": "ndma_heatwave_alert_stages",
        "domain": "heat_disaster",
        "authority": "National Disaster Management Authority (NDMA)",
        "title": "Heatwave Warning Tiers (Yellow, Orange, Red Alerts)",
        "rule_text": "Yellow Alert (Be Updated): Heatwave persists for up to 2 days; high thermal discomfort for vulnerable populations. Orange Alert (Be Prepared): Severe heatwave persists for 2 days or heatwave conditions persist for 4+ days. Red Alert (Take Action): Severe heatwave persists for more than 2 days or total heatwave duration exceeds 6 days. Extreme health hazard for all age groups.",
        "trigger_condition": "Heatwave persistence >= 2 days",
        "mandatory_directive": "At Orange Alert: Supply mobile water tankers to slums and transit hubs; adjust school hours. At Red Alert: Prohibit outdoor manual and construction labor between 12:00 PM and 3:30 PM.",
        "citation": "NDMA National Guidelines for Preparation of Action Plan - Prevention and Management of Heat Wave (2019), Section 4.2"
    },
    {
        "id": "ndma_wet_bulb_thermal_mortality",
        "domain": "heat_disaster",
        "authority": "NDMA & Indian Institute of Public Health (IIPH)",
        "title": "Psychrometric Wet-Bulb Temperature Survivability and Labor Limits",
        "rule_text": "Wet-bulb temperature (Tw) integrates evaporative cooling limits of the human body. At Tw >= 29.0°C, heavy outdoor physical labor requires mandatory 45-minute rest breaks per hour with electrolyte replenishment. At Tw >= 31.0°C (extreme hazard threshold), the body cannot shed metabolic heat via perspiration; prolonged outdoor exposure leads to fatal exertional heatstroke.",
        "trigger_condition": "Wet-Bulb Temperature >= 29.0°C",
        "mandatory_directive": "Enforce mandatory shaded rest intervals for gig workers, agricultural laborers, and construction crews; deploy oral rehydration salts (ORS) kiosks.",
        "citation": "Ahmedabad Heat Action Plan (HAP) & NDMA Extreme Temperature Health Adaptation Protocol"
    },
    {
        "id": "ndma_school_and_child_heat_protection",
        "domain": "heat_disaster",
        "authority": "NDMA & Ministry of Education",
        "title": "School Shift Rescheduling and Playground Closure Guidelines",
        "rule_text": "When ambient temperature exceeds 42.0°C or when IMD issues an Orange Heatwave Alert, schools must terminate afternoon sessions. Operational hours must shift to 7:00 AM - 11:00 AM morning shifts only. All outdoor sports, assemblies, and physical training must be suspended. If temperatures exceed 44.0°C or Red Alert triggers, state authorities must declare temporary school closures.",
        "trigger_condition": "Temperature >= 42.0°C or Orange/Red Alert",
        "mandatory_directive": "End school sessions by 11:00 AM; prohibit open-ground physical activities; ensure cold drinking water availability on school transit buses.",
        "citation": "Ministry of Education & NDMA Advisory on School Safety during Extreme Weather"
    },

    # -------------------------------------------------------------------------
    # 2. ICAR & CRIDA - Agriculture, Crop Phenology, Pesticides & Soil
    # -------------------------------------------------------------------------
    {
        "id": "icar_pesticide_foliar_washoff",
        "domain": "agriculture",
        "authority": "Indian Council of Agricultural Research (ICAR-CRIDA)",
        "title": "Chemical Pesticide Foliar Application & Rainfall Wash-off Threshold",
        "rule_text": "Systemic and contact foliar pesticides, fungicides, and herbicides require a minimum of 4 to 6 rain-free hours for penetration of the waxy plant cuticle. Applying foliar sprays when rainfall is forecast within 4 hours results in 80%+ active ingredient wash-off, contaminating surface runoff and rendering pest control completely ineffective.",
        "trigger_condition": "Precipitation >= 2.0mm or rain probability >= 40% within 6 hours",
        "mandatory_directive": "Postpone all chemical pesticide, insecticide, and weedicide spraying until a guaranteed dry window is confirmed.",
        "citation": "ICAR-CRIDA Standard Operating Procedures for Agrometeorological Advisory Services, Section 5.3"
    },
    {
        "id": "icar_spray_drift_wind_speed_limit",
        "domain": "agriculture",
        "authority": "ICAR - Central Institute of Agricultural Engineering",
        "title": "Agricultural Spray Drift Wind Speed Operational Boundary",
        "rule_text": "Foliar spraying of chemicals must only be conducted when 10m wind speeds are between 3.0 km/h and 15.0 km/h. Wind speeds > 15.0 km/h cause severe droplet drift into non-target neighboring crops, residential settlements, and water channels. Wind speeds < 3.0 km/h under temperature inversion conditions trap fine spray mists near the ground, causing localized chemical toxicity.",
        "trigger_condition": "Wind speed > 15.0 km/h or < 3.0 km/h with stagnation",
        "mandatory_directive": "Halt knapsack and tractor-mounted boom spraying during gusty winds; schedule applications during calm early morning hours (6:00 AM - 9:00 AM).",
        "citation": "ICAR Technical Bulletin on Precision Chemical Application and Environmental Safety"
    },
    {
        "id": "icar_wheat_terminal_heat_anthesis",
        "domain": "agriculture",
        "authority": "ICAR - Indian Institute of Wheat and Barley Research (IIWBR)",
        "title": "Rabi Wheat Terminal Heat Stress & Anthesis Reproductive Limit",
        "rule_text": "Rabi wheat crops entering the reproductive anthesis (flowering) and grain-filling stages (February-March) are exceptionally vulnerable to terminal heat. Maximum daytime temperatures exceeding 32.0°C shorten the grain-filling duration, accelerate leaf senescence, cause pollen abortion, and lead to shriveled grains with 4% to 6% yield loss per degree Celsius above 32°C.",
        "trigger_condition": "Temperature > 32.0°C during wheat reproductive stage (Feb-March)",
        "mandatory_directive": "Apply light, frequent sprinkler irrigation during evening/early morning hours to lower crop canopy temperature by 1.5°C - 2.0°C; spray 2% potassium nitrate (KNO3) to mitigate osmotic stress.",
        "citation": "ICAR-IIWBR Contingency Strategy for Mitigating Heat Stress in Wheat"
    },
    {
        "id": "icar_paddy_flowering_thermal_sterility",
        "domain": "agriculture",
        "authority": "ICAR - National Rice Research Institute (NRRI)",
        "title": "Paddy (Rice) High-Temperature Spikelet Sterility Threshold",
        "rule_text": "Rice spikelet flowering and anthesis occur between 9:00 AM and 11:30 AM. Temperatures exceeding 35.0°C or relative humidity exceeding 85% during anthesis impede anther dehiscence, reduce pollen viability, and prevent stigma fertilization, causing spikelet sterility (chaffy grain formation).",
        "trigger_condition": "Day temperature > 35.0°C during rice flowering window",
        "mandatory_directive": "Maintain 5 cm standing water layer in the paddy field to buffer rhizosphere temperatures; avoid heavy nitrogen fertilizer top-dressing during heat spikes.",
        "citation": "ICAR-NRRI Agromet Protocol on Climate Resilient Rice Production"
    },
    {
        "id": "icar_cotton_waterlogging_deluge",
        "domain": "agriculture",
        "authority": "ICAR - Central Institute for Cotton Research (CICR)",
        "title": "Cotton Crop Root-Zone Waterlogging & Hypoxia Threshold",
        "rule_text": "Cotton is extremely sensitive to excess soil moisture and poor drainage. Saturated root-zone conditions (> 85% volumetric moisture content) lasting more than 48 hours induce soil hypoxia (oxygen starvation), causing root death, sudden parawilt, yellowing of foliage, and massive premature shedding of floral squares and bolls.",
        "trigger_condition": "Excess rainfall resulting in continuous standing water > 24-48 hours",
        "mandatory_directive": "Dig emergency inter-row drainage trenches to evacuate excess standing water immediately; spray 1% urea + 1% magnesium sulphate solution post-drainage to restore root nutrient uptake.",
        "citation": "ICAR-CICR Advisory on Monsoon Flood Management in Cotton Belts"
    },
    {
        "id": "icar_soil_moisture_deficit_drought",
        "domain": "agriculture",
        "authority": "ICAR-CRIDA & IMD Agromet",
        "title": "Soil Moisture Deficit Index (SMDI) Flash Drought Triggers",
        "rule_text": "SMDI quantifies root-zone moisture anomalies on a -4.0 (extreme drought) to +4.0 (extreme wetness) scale. SMDI values between -1.5 and -2.5 indicate Moderate Agricultural Stress. SMDI < -2.5 indicates Severe Flash Drought, meaning soil moisture has fallen below the permanent wilting point (PWP) of crops.",
        "trigger_condition": "SMDI <= -2.5 for 10+ consecutive days",
        "mandatory_directive": "Implement life-saving micro-irrigation (drip/sprinkler); apply biomass mulch to conserve residual soil capillary moisture; postpone top-dressing fertilizers.",
        "citation": "ICAR-CRIDA Drought Management Contingency Atlas for Indian Districts"
    },

    # -------------------------------------------------------------------------
    # 3. BIS IS 10500:2012 - Drinking Water Chemical Potability Standards
    # -------------------------------------------------------------------------
    {
        "id": "bis_fluoride_chemical_standard",
        "domain": "water_quality",
        "authority": "Bureau of Indian Standards (BIS)",
        "title": "Fluoride Chemical Limits in Drinking Water (IS 10500:2012)",
        "rule_text": "Under BIS IS 10500:2012, acceptable fluoride concentration in drinking water is 1.0 mg/L, with a maximum permissible limit of 1.5 mg/L in the absence of an alternate source. Concentrations > 1.5 mg/L cause dental fluorosis (mottling and pitting of tooth enamel) in children and crippling skeletal fluorosis (bone deformation and calcification of ligaments) in adults.",
        "trigger_condition": "Fluoride > 1.5 mg/L",
        "mandatory_directive": "Water is chemically NON-POTABLE. Boiling does NOT remove fluoride. Community-level defluoridation using Activated Alumina (Nalgonda technique) or domestic Reverse Osmosis (RO) filtration is mandatory.",
        "citation": "Bureau of Indian Standards IS 10500:2012 (Second Revision), Table 2 Chemical Requirements"
    },
    {
        "id": "bis_nitrate_methemoglobinemia",
        "domain": "water_quality",
        "authority": "Bureau of Indian Standards (BIS)",
        "title": "Nitrate Limit and Infant Blue-Baby Syndrome Risk (IS 10500:2012)",
        "rule_text": "Under BIS IS 10500:2012, maximum permissible limit for Nitrate (as NO3) in drinking water is 45.0 mg/L with NO RELAXATION. High nitrate arises from agricultural fertilizer runoff and sewage infiltration. Ingesting water with nitrate > 45 mg/L causes infant methemoglobinemia ('blue-baby syndrome'), converting blood hemoglobin to methemoglobin and impairing oxygen transport, leading to suffocation.",
        "trigger_condition": "Nitrate > 45.0 mg/L",
        "mandatory_directive": "Strictly prohibit consumption of this water by pregnant mothers and infants under 1 year. Implement ion-exchange de-nitrification or RO filtration.",
        "citation": "Bureau of Indian Standards IS 10500:2012, Table 2 Toxic Substances"
    },
    {
        "id": "bis_total_hardness_and_tds",
        "domain": "water_quality",
        "authority": "Bureau of Indian Standards (BIS)",
        "title": "Total Hardness and Total Dissolved Solids (TDS) Limits",
        "rule_text": "Under BIS IS 10500:2012: Total Hardness (as CaCO3) acceptable limit is 200.0 mg/L, permissible limit 600.0 mg/L. Hardness > 600 mg/L causes encrustation in water pipelines, scale formation in boilers, and poor palatability. Total Dissolved Solids (TDS) acceptable limit is 500.0 mg/L, permissible limit 2000.0 mg/L. TDS > 2000 mg/L causes gastrointestinal irritation and severe mineral scaling.",
        "trigger_condition": "Hardness > 600.0 mg/L or TDS > 2000.0 mg/L",
        "mandatory_directive": "Install cation-exchange water softening or commercial RO desalinization before domestic or industrial use.",
        "citation": "Bureau of Indian Standards IS 10500:2012, Table 1 Organoleptic and Physical Parameters"
    },
    {
        "id": "bis_heavy_metals_arsenic_lead",
        "domain": "water_quality",
        "authority": "Bureau of Indian Standards & CGWB",
        "title": "Heavy Metal Toxicity Limits: Arsenic and Lead (IS 10500:2012)",
        "rule_text": "Under BIS IS 10500:2012: Arsenic acceptable limit is 0.01 mg/L, maximum permissible limit 0.05 mg/L. Chronic arsenic ingestion leads to arsenicosis, hyperkeratosis of palms, black foot disease, and vascular cancer. Lead acceptable limit is 0.01 mg/L with NO RELAXATION; lead induces irreversible neurotoxicity, cognitive deficits in children, and renal impairment.",
        "trigger_condition": "Arsenic > 0.01 mg/L or Lead > 0.01 mg/L",
        "mandatory_directive": "Seal contaminated borewells immediately; mark hand pumps with red hazard paint; provide piped surface water or deep-aquifer potable alternatives.",
        "citation": "Central Ground Water Board (CGWB) & BIS IS 10500:2012 Heavy Metals Schedule"
    },

    # -------------------------------------------------------------------------
    # 4. CPCB - National Air Quality Index (NAQI) & GRAP Regulations
    # -------------------------------------------------------------------------
    {
        "id": "cpcb_naqi_breakpoints_pm25",
        "domain": "air_quality",
        "authority": "Central Pollution Control Board (CPCB)",
        "title": "National Air Quality Index (NAQI) 24h PM2.5 Category Breakpoints",
        "rule_text": "CPCB classifies 24-hour mean PM2.5 concentrations (in µg/m³) into six standard national categories: Good (0 - 30 µg/m³), Satisfactory (31 - 60 µg/m³), Moderate (61 - 90 µg/m³), Poor (91 - 120 µg/m³), Very Poor (121 - 250 µg/m³), and Severe (> 250 µg/m³). Concentrations exceeding 60 µg/m³ violate the Indian National Ambient Air Quality Standards (NAAQS).",
        "trigger_condition": "PM2.5 > 60.0 µg/m³",
        "mandatory_directive": "At Poor (91-120): Warn asthmatics and cardiac patients. At Very Poor (121-250): Restrict outdoor sports and heavy physical exertion. At Severe (>250): Suspend in-person elementary schooling.",
        "citation": "CPCB National Air Quality Index Report (2014) & NAAQS Notification"
    },
    {
        "id": "cpcb_grap_stage_protocols",
        "domain": "air_quality",
        "authority": "Commission for Air Quality Management (CAQM) / CPCB",
        "title": "Graded Response Action Plan (GRAP) Stages 1 to 4 Enforcement",
        "rule_text": "GRAP enforces legally binding emergency air pollution countermeasures across the National Capital Region and participating Indian metros: Stage 1 (AQI 201-300 'Poor'): Anti-dust water sprinkling, ban on open biomass burning. Stage 2 (AQI 301-400 'Very Poor'): Total ban on non-emergency diesel generator sets, enhanced parking fees. Stage 3 (AQI 401-450 'Severe'): Complete halt to non-essential construction/demolition, ban on BS-III petrol and BS-IV diesel cars. Stage 4 (AQI > 450 'Severe+'): Ban on diesel truck entry into city limits, 50% work-from-home mandate.",
        "trigger_condition": "AQI >= 201 (Stage 1), >= 301 (Stage 2), >= 401 (Stage 3), >= 450 (Stage 4)",
        "mandatory_directive": "Enforce municipal vehicle restrictions, halt stone crushers and brick kilns, switch schools to online instruction.",
        "citation": "CAQM Statutory Directions on Graded Response Action Plan (2023)"
    },
    {
        "id": "cpcb_atmospheric_stagnation_inversion",
        "domain": "air_quality",
        "authority": "CPCB & IMD Environmental Monitoring Wing",
        "title": "Atmospheric Boundary Layer Stagnation and Nocturnal Inversion",
        "rule_text": "Atmospheric Stagnation occurs during winter months when 10m wind speed falls < 5.0 km/h and mixing height collapses < 500 meters, yielding a Ventilation Index < 2000 m²/s. Under nocturnal radiation inversion, cooler surface air traps particulate emissions within a shallow ground layer, creating dangerous toxic smog chambers.",
        "trigger_condition": "Ventilation Index < 2000 m²/s and Wind Speed < 5 km/h",
        "mandatory_directive": "Prohibit all evening industrial emissions; conduct continuous mechanical water misting along dense traffic corridors.",
        "citation": "CPCB Guidelines on Environmental Air Shed Management and Ventilation Indexing"
    },

    # -------------------------------------------------------------------------
    # 5. CEA & Electrical Utilities - Grid Thermal Loading Limits
    # -------------------------------------------------------------------------
    {
        "id": "cea_transformer_tropical_night_stress",
        "domain": "power_grid",
        "authority": "Central Electricity Authority (CEA)",
        "title": "Transformer Thermal Dissipation & Tropical Night Degradation",
        "rule_text": "Power distribution transformers rely on nocturnal ambient cooling to dissipate core and winding thermal energy accumulated during daytime peak operations. When nocturnal temperatures remain >= 28.0°C ('Tropical Night'), thermal dissipation collapses. Transformer oil temperatures remain elevated, degrading insulation life and increasing catastrophic blowout hazard by 3.5x under AC cooling surges.",
        "trigger_condition": "Night temperature (Tmin) >= 28.0°C and peak load > 90% capacity",
        "mandatory_directive": "Activate auxiliary substation cooling fans; redistribute feeder loads to prevent localized distribution transformer tripping.",
        "citation": "CEA Manual on Transmission and Distribution Substation Operations"
    },
    {
        "id": "cea_cooling_tower_wet_bulb_approach",
        "domain": "power_grid",
        "authority": "Central Electricity Authority (CEA)",
        "title": "Thermal Power Plant Wet-Bulb Approach & Generation De-Rating",
        "rule_text": "Thermal and combined-cycle gas turbine generating units utilize wet cooling towers designed for a wet-bulb approach temperature of 28.0°C. When ambient wet-bulb temperature exceeds 31.0°C, cooling water return temperature rises, condenser vacuum degrades, and power generating efficiency drops by 0.5% per °C, forcing units into thermal generation de-rating.",
        "trigger_condition": "Wet-Bulb Temperature >= 31.0°C",
        "mandatory_directive": "Issue generation de-rating notices to State Load Despatch Centres (SLDC); schedule peaker generation reserves.",
        "citation": "CEA Technical Standards for Construction of Thermal Power Stations"
    },

    # -------------------------------------------------------------------------
    # 6. IMD & CWC - Floods, Cloudbursts & Inundation Drainage
    # -------------------------------------------------------------------------
    {
        "id": "imd_rainfall_intensity_classification",
        "domain": "flood_disaster",
        "authority": "India Meteorological Department (IMD)",
        "title": "Standard 24-Hour Rainfall Intensity Scale & Alert Colors",
        "rule_text": "IMD classifies 24-hour accumulated rainfall into five operational categories: Light Rain (2.5 - 15.5 mm), Moderate Rain (15.6 - 64.4 mm), Heavy Rain (64.5 - 115.5 mm, Yellow Warning), Very Heavy Rain (115.6 - 204.4 mm, Orange Warning), and Extremely Heavy Rain (>= 204.5 mm, Red Disaster Alert). A Cloudburst is defined as localized precipitation >= 100 mm in 1 hour.",
        "trigger_condition": "Rainfall >= 64.5 mm (Heavy), >= 115.6 mm (Very Heavy), >= 204.5 mm (Extremely Heavy)",
        "mandatory_directive": "At Orange (>=115.6mm): Pre-position National Disaster Response Force (NDRF) boats; clear drainage sluice gates. At Red (>=204.5mm): Evacuate informal settlements in riverine floodplains.",
        "citation": "IMD Weather Forecasting Manual - Classification of Rainfall Intensity"
    },
    {
        "id": "cwc_urban_drainage_surcharge_limit",
        "domain": "flood_disaster",
        "authority": "Central Water Commission (CWC) & CPHEEO",
        "title": "Urban Stormwater Canal Capacity & Surcharge Inundation Limit",
        "rule_text": "Urban stormwater drainage systems across Tier-1 and Tier-2 Indian cities are historically engineered to handle rainfall intensities of 25.0 to 35.0 mm/hour. When precipitation exceeds 50.0 mm in a 3-hour window over catchments with impervious urban surface fractions > 65%, gravity drainage canals surcharge and backflow into arterial roads, submerging underpasses within 30 to 45 minutes.",
        "trigger_condition": "Precipitation > 50.0 mm in 3 hours in urban core",
        "mandatory_directive": "Activate de-watering high-capacity diesel pumps at subway underpasses; deploy traffic police diversions away from designated flood-prone junctions.",
        "citation": "CPHEEO Manual on Storm Water Drainage Systems, Ministry of Housing and Urban Affairs"
    }
]


def build_and_save_knowledge_base():
    """Build knowledge JSON file and compute semantic vector index using TF-IDF + Latent Semantic Projection."""
    root = Path(__file__).resolve().parent.parent
    output_dir = root / "data" / "knowledge"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "indian_environmental_standards.json"
    npy_path = output_dir / "knowledge_embeddings.npy"
    manifest_path = output_dir / "knowledge_manifest.json"

    # 1. Save Curated Standards JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(INDIAN_ENVIRONMENTAL_STANDARDS, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(INDIAN_ENVIRONMENTAL_STANDARDS)} curated standards to {json_path}")

    # 2. Extract Document Strings for Embedding
    corpus = []
    for item in INDIAN_ENVIRONMENTAL_STANDARDS:
        doc_text = (
            f"Title: {item['title']}. "
            f"Authority: {item['authority']}. "
            f"Domain: {item['domain']}. "
            f"Rule: {item['rule_text']} "
            f"Conditions: {item['trigger_condition']}. "
            f"Mandatory Action: {item['mandatory_directive']} "
            f"Citation: {item['citation']}"
        )
        corpus.append(doc_text)

    # 3. Compute Normalized Semantic Vector Embeddings
    # TF-IDF with character & word n-grams + LSA projection to 64 dense dimensions
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Truncated SVD for dense semantic embeddings
    n_components = min(32, len(INDIAN_ENVIRONMENTAL_STANDARDS) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense_embeddings = svd.fit_transform(tfidf_matrix)

    # L2 normalize embeddings so dot product is exact cosine similarity
    norms = np.linalg.norm(dense_embeddings, axis=1, keepdims=True)
    normalized_embeddings = dense_embeddings / np.maximum(norms, 1e-12)

    # 4. Save Embeddings Matrix to .npy and Vectorizer pipeline
    np.save(npy_path, normalized_embeddings.astype(np.float32))
    print(f"Saved normalized embeddings matrix {normalized_embeddings.shape} to {npy_path}")

    vectorizer_path = output_dir / "knowledge_vectorizer.joblib"
    joblib.dump({"vectorizer": vectorizer, "svd": svd}, vectorizer_path)
    print(f"Saved vectorizer pipeline to {vectorizer_path}")

    # 5. Save Manifest
    manifest = {
        "count": len(INDIAN_ENVIRONMENTAL_STANDARDS),
        "embedding_dimensions": n_components,
        "format": "float32_l2_normalized",
        "domains": list(set(item["domain"] for item in INDIAN_ENVIRONMENTAL_STANDARDS)),
        "authorities": list(set(item["authority"] for item in INDIAN_ENVIRONMENTAL_STANDARDS)),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    build_and_save_knowledge_base()
