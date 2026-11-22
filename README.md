# ClimaCore Intelligence

> **Institutional Climate Risk Intelligence & Operational Decision Support Platform**  
> Moving from reactive weather alerts to proactive, compound climate risk intelligence for India's vital infrastructure and diverse communities.

---

## Key Highlights & Core Capabilities

| Feature Category | Capacity & Specifications |
| :--- | :--- |
| **Occupations & Personas** | **13 Targeted Stakeholder Personas** calibrated with domain-specific hazard thresholds |
| **Scientific ML Models** | **25 Production Machine Learning & Statistical Physics Models** |
| **Dedicated Utility Tools** | **16 Tactical Decision Simulators** + 2 Strategic Incident Command Consoles |
| **Supported Languages** | **7 Languages** (English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati) |
| **Language Coverage Scope** | Multi-language translation is focused on grassroots **Farmer & Agriculture utilities** due to token optimization |
| **Command Intelligence** | **3 Interactive Assistant Command Modes** (`/plan`, `/whatif`, `/compare`) |
| **Calendar Scheduling** | **1-Click Google Calendar Intent Planner** for operational weather safety windows |
| **Ground-Truth Datasets** | **15 Institutional Datasets** (IMD, CPCB, ICAR, CWC, POSOCO, ERA5, Agmarknet) |
| **Statistical Calibration** | **Conformal Prediction Intervals (>= 90% finite-sample coverage guarantees)** |

> [!NOTE]
> **Multilingual Scope Notice**: Full 7-language translation and dynamic Indic script localization are actively enabled for the Farmer and Agriculture decision utilities (Crop Harvest Spoilage Radar and Mandi Commodity Prices). To optimize LLM context token usage while delivering the highest grassroots impact, other specialized institutional modules operate in standard English.

---

## Executive Overview

ClimaCore Intelligence is an AI-powered Environmental Risk Intelligence and Decision Support Platform that unifies machine learning, physical process modelling, geospatial analysis, and multi-agent reasoning into a single operational interface. 

Conventional weather applications present raw numerical data like rainfall volume or maximum temperature that force users to guess their practical risk. ClimaCore transforms raw, complex weather observations into actionable, localized operational decisions across four key pillars:

1. **Compound Multi-Hazard Modeling**: Simultaneously assesses cascading hazards including extreme wet-bulb heat stress, urban stormwater surcharge, electrical substation overload, and crop disease vectors.
2. **Mathematically Calibrated Uncertainty**: Replaces single-point guesses with distribution-free conformal prediction intervals, providing guaranteed 90% confidence bands for risk-sensitive planning.
3. **Sector-Specific Decision Intelligence**: Automatically maps meteorological conditions directly into the operational workflows of 13 specific Indian professions.
4. **Verified Institutional Grounding**: Grounded strictly in empirical data from official Indian scientific agencies without unverified hallucinations.

---

## Platform Architecture Layers

ClimaCore Intelligence is structured into four cohesive engineering and scientific layers:


### 1. Predictive Intelligence Layer
* **Ensemble Machine Learning**: Fast, highly calibrated gradient-boosted decision trees (LightGBM, XGBoost) and deep recurrent architectures.
* **Two-Part Hurdle Models**: Deconstructs zero-inflated meteorological events (such as sporadic rainfall) into occurrence classification followed by conditional volume regression.
* **Multi-Horizon Lead Times**: Generates concurrent predictions across 24-hour, 48-hour, and 72-hour operational horizons.

### 2. Physical & Statistical Process Layer
* **Hydrological Engineering**: Computes soil retention, infiltration rates, and surface runoff using the USDA SCS-CN (Soil Conservation Service Curve Number) methodology.
* **Biometeorology & Human Heat Strain**: Dynamically computes Wet-Bulb Globe Temperature (WBGT) and apparent heat indices to protect outdoor labor.
* **Conformal Prediction Engine**: Calibrates non-parametric prediction intervals using historical holdout residuals, guaranteeing a minimum of 90% empirical coverage under arbitrary distributions.

### 3. Geospatial & Exposure Layer
* **Critical Lifeline Mapping**: Overlays highway corridors, passenger rail lines, electricity substations, and medical centers against hazard footprints.
* **Topographic Hydrology**: Traces upstream catchment routing using HydroSHEDS and Central Water Commission drainage basins.
* **Vulnerability Weighting**: Integrates WorldPop 100-meter population density grids to measure human and asset exposure.

### 4. Decision Support & Action Layer
* **Occupational Calibrations**: Translates physical metrics into operational dispatches customized for 13 distinct professions.
* **Multi-Agent Reasoning**: Features the ClimaCore Assistant and Crisis Council incident commander for automated deliberation during disasters.
* **Grassroots Multilingual Pipeline**: Localizes agritech utilities into 7 Indian languages with zero translation latency.

---

## Solving the Core Challenges of Climate Risk

### 1. Multilingual Agritech & Grassroots Advisory
* **Simple, Contextualized Farm Guidance**: Delivers agronomic and weather recommendations in 7 regional Indian languages (English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati).
* **Integrated Market Telemetry**: Fuses live atmospheric observations with official APMC Mandi commodity trade records across major market yards, identifying inter-mandi price spreads and optimal dispatch timing.
* **Biological Early Warnings**: Embeds ICAR and state horticulture threshold models (such as potato late blight and rice blast moisture windows) to warn farmers before crop pathogen incubation occurs.

### 2. Predictive Micro-Location Climate & Yield Engine
* **Hyper-Local Forecasting**: Replaces coarse regional averages with micro-location predictions accounting for urban heat islands and coastal squall fronts.
* **Bounded Confidence Intervals**: Employs Conformal Prediction to produce mathematically validated 90% confidence intervals, giving operational planners reliable risk bounds rather than misleading point estimates.
* **Physical Soil-Water Coupling**: Bridges physical soil moisture balances with ML regressors to predict agricultural drought, root-zone saturation, and post-harvest drying risks.

### 3. Smart Grid & Urban Infrastructure Resource Controller
* **Grid Peak Demand Foresight**: Models electrical transmission loads and transformer heat stress driven by surging air conditioning and cooling demand during heatwaves.
* **Critical Facility Protection**: Models hospital generator fuel reserves during grid outages, thermal fatigue on industrial metal roofs, and cold storage refrigeration breaches.
* **Water Treatment Assurance**: Evaluates storm runoff silt volumes and turbidity spikes to safeguard municipal drinking water filtration intake works.

### 4. Emergency Response & Supply Chain Logistics Swarm
* **Corridor Flood Severance**: Assesses logistics arteries, rail links, and relief routes for flash-flood submergence risks.
* **Synchronized Crisis Council**: Orchestrates multi-stakeholder disaster deliberation matrixes coordinating civil defense, hospitals, energy dispatchers, and public transport operators.
* **Parametric Micro-Insurance**: Calculates instant, objective index thresholds for catastrophic weather events, enabling immediate automated relief funding without months of physical damage surveys.

---

## Institutional Ground-Truth Datasets (15 Canonical Registries)

ClimaCore Intelligence is built upon **15 canonical, verified institutional datasets** encompassing over 1,000,000 empirical ground observations across India:

| # | Dataset Registry | Institutional Provider | Observation Scope & Temporal Span |
| :---: | :--- | :--- | :--- |
| **1** | **IMD Gridded Daily Maximum & Minimum Temperature** | India Meteorological Department (Pune) | 74 years of daily temperature records (0.50 degree grid) across India |
| **2** | **IMD Gridded Daily Precipitation** | India Meteorological Department (Pai et al.) | 46 years of gridded daily rain-gauge measurements (0.25 degree grid) |
| **3** | **ERA5-Land Historical Meteorological Archive** | ECMWF / Copernicus Climate Change Service | 44 years of hourly reanalysis: soil moisture, boundary layer, radiation |
| **4** | **India Flood Inventory (IFI v3)** | Indian Institute of Technology (IIT Delhi) | Spatial inventory of historical flood events, dates, and inundation extent |
| **5** | **National Daily Electricity Demand & Grid Heat Stress** | POSOCO / Central Electricity Authority | Daily regional power consumption, peak load demand, and frequency telemetry |
| **6** | **District Agricultural Soil & Crop Statistics** | ICRISAT / ICAR Directorate of Economics | 50+ years of district-level crop acreage, yield history, and soil characteristics |
| **7** | **Global Crop Phenology Calendar** | SAGE / University of Wisconsin | Sowing, anthesis, and harvesting dates for 19 staple Indian field crops |
| **8** | **High-Resolution Population Exposure (100m)** | WorldPop / University of Southampton | Gridded human population density and settlement distribution surfaces |
| **9** | **Urban Lifeline Infrastructure Snapshot** | OpenStreetMap Foundation | Geographic vectors for hospitals, power substations, bridges, and highways |
| **10** | **Coal Thermal Power Plant Inventory** | CHETNA Clean Energy Database | Geographic coordinates, cooling water sources, and capacity ratings |
| **11** | **Peninsular India Catchment Hydro-Meteorology** | CAMELS-IND Hydrological Archive | Basin-averaged rainfall, gauge streamflow, and evapotranspiration data |
| **12** | **National Groundwater Chemical Quality Survey** | Central Ground Water Board (CGWB) | Heavy metal, fluoride, arsenic, nitrate, and electrical conductivity metrics |
| **13** | **HydroRIVERS Asia Stream Network Topology** | HydroSHEDS / World Wildlife Fund | River flow direction, upstream catchment area, and hydrologic connectivity |
| **14** | **Major Reservoirs Daily Storage Monitoring** | Central Water Commission (CWC) | Live storage levels, Full Reservoir Level (FRL), and flood cushion margins |
| **15** | **Continuous Ambient Air Quality Stations (CAAQMS)** | Central Pollution Control Board (CPCB) | Physical surface station monitoring for PM2.5, PM10, NO2, SO2, CO, and Ozone |

---

## 25 Scientific Machine Learning & AI Models

ClimaCore maintains a model registry spanning 25 distinct scientific pipelines evaluated under strict chronological holdouts:

### Atmospheric & Heat Risk Suite
1. **HeatRisk V3 Dual Diurnal Temperature Regressor**: Next-day 2m maximum and minimum temperature forecasting.
2. **HeatRisk V1.2 Tri-Quantile Uncertainty Suite**: Quantile regressors at 10th, 50th, and 90th percentiles for risk boundaries.
3. **HeatRisk V5 Soft-Gated Regularized Mixture of Experts**: Regional specialized models routed via gating networks.
4. **Wet-Bulb Globe Temperature (WBGT) Sensor**: Computes biometeorological heat strain via the Stull equation.
5. **Urban Heat Island Differential Calculator**: Estimates urban-to-rural temperature gradients based on satellite imperviousness.

### Hydrology & Inundation Suite
6. **FloodRisk V2 Pure Hydrological Hazard Classifier**: Distinguishes flash flood occurrence from routine rainfall.
7. **FloodRisk SCS-CN Hydrological Runoff Challenger**: Physically constrained rainfall-runoff depth estimation.
8. **FloodRisk XGBoost Runoff Estimator**: Non-linear boosting model for soil saturation interaction effects.
9. **BasinFlow V1 Hydrological Routing Challenger**: Kinematic river reach discharge and travel-time modeling.
10. **BasinFlow V2 GR4J Lumped Conceptual Model**: 4-parameter continuous catchment water balance simulator.
11. **BasinFlow V2 HBV-Light Catchment Model**: Semi-distributed snowpack, soil moisture, and reservoir routing model.
12. **BasinFlow V2 Standard Entity-Aware LSTM**: Deep recurrent network encoding catchment physical static attributes.
13. **BasinFlow V2 Mass-Conserving LSTM (MC-LSTM)**: Neural network enforcing strict physical conservation of mass.
14. **BasinFlow V2 Dynamic MC-LSTM**: Mass-conserving LSTM with time-varying state transitions.
15. **Extreme Heavy Downpour Hurdle Classifier**: Flags torrential downpours exceeding the 95th historical percentile.

### Power Grid & Energy Infrastructure Suite
16. **PowerGrid V2 Peak Load Demand Regressor**: Predicts MW electricity demand driven by heat and humidity surges.
17. **PowerGrid V2 Unmet Supply Stress Classifier**: Detects probability of forced load-shedding and voltage drops.
18. **PowerGrid V4 Causal Dilated 1D TCN**: Temporal convolutional network capturing multi-day heat accumulation.

### Agriculture, Air Quality & Environmental Quality Suite
19. **AgriSense V2 10-Class Crop Suitability Classifier**: Evaluates seasonal growing suitability for 10 staple crops.
20. **Crop Anthesis Heat Stress Index**: Detects reproductive stage thermal disruption in cereal grains.
21. **AirQuality V2 Next-Day PM2.5 Regressor**: Forecasts 24-hour mean fine particulate matter concentrations.
22. **SmokePath V2 Biomass Burning Smoke Transport Model**: Tracks seasonal crop-residue smoke dispersion vectors.
23. **DroughtCascade V1 Multi-Decadal Latent Regime Engine**: Identifies persistent hydrological drought transitions.
24. **DroughtCascade V2 Causal Gaussian HMM**: Uncovers hidden state transitions between wet, normal, and dry regimes.
25. **WaterQuality V1 WQI Regressor & Contamination Classifier**: Predicts river turbidity and chemical water quality indices.

---

## 13 Calibrated Stakeholder Personas

ClimaCore dynamically recalibrates its entire threat assessment, visual presentation, and advisory logic based on the user's operational role:

1. **Farmer & Agriculture**: Crop heat stress, soil moisture deficit, irrigation scheduling, and rainfall deluge alerts.
2. **Student & School**: School closure advisories, heatwave commute exposure, bus safety, and indoor AQI warnings.
3. **Gig Worker & Outdoor Labor**: Outdoor wet-bulb heat stress, delivery hazard indexes, and hydration rest stops.
4. **Daily Commuter & Transit**: Road waterlogging, suburban rail disruption, localized urban flooding, and storm wind gusts.
5. **Healthcare & Hospital Operations**: Hospital heatstroke influx surge preparedness, respiratory admissions, and ICU backup power.
6. **City Official & Disaster Management**: NDMA Heat Action Plan triggers, municipal flood alert levels, and shelter deployment.
7. **Maritime & Fishermen**: Coastal sea state, cyclone proximity, Beaufort gale force alerts, and rough sea fishing bans.
8. **Industry & Manufacturing**: Cooling tower heat dissipation, peak electrical demand surcharges, and factory floor thermal limits.
9. **Aviation & Airports**: Crosswind runway limits, convective thunderstorms, air density altitude, and ramp ground stops.
10. **Water Utility & Stormwater**: Stormwater pumping station capacity, reservoir buffer management, and sewer overflow hazards.
11. **Insurance & Catastrophe Actuary**: Parametric weather derivative triggers, crop insurance claims, and catastrophe portfolio exposure.
12. **Cold Chain Logistics**: Reefer transport thermal breach probability, ambient cooling deficits, and dairy preservation.
13. **Tourism & Outdoor Events**: Outdoor comfort index, lightning strike exposure, and optimal event staging windows.

---

## 16 Dedicated Decision Utility Simulators

In addition to the primary city risk assessment, ClimaCore provides **16 specialized utility tools** designed to answer tactical operational questions:

1. **Crop Harvest Spoilage Radar**: Live grain moisture, dew point condensation, and post-harvest drying risk calculator. *(Full 7-language localization)*
2. **APMC Mandi Commodity Prices**: Live market arrivals, modal rates, inter-mandi price spreads, and 1-click harvest and sell verdicts. *(Full 7-language localization)*
3. **Monsoon Exam Commute Planner**: Evaluates safe departure buffers, low-lying waterlogged transit bottlenecks, and public transit reliability on examination days.
4. **Metro vs Road Flood Intuition Engine**: Side-by-side comparison of surface road disruption versus underground/elevated metro reliability during storm surges.
5. **Hospital Emergency Fuel Buffer Calculator**: Models hospital ICU generator diesel burn rates under extended electrical grid outages caused by heat or storms.
6. **Heat Stroke Emergency Ward Surge Predictor**: Forecasts admissions for heat exhaustion and dehydration based on consecutive high wet-bulb hours.
7. **Cold Storage Blackout Defense**: Calculates thermal inertia and perishables spoilage hours inside rural cold storage facilities during power failures.
8. **Delivery Corridor Submergence Radar**: Identifies flood-prone highway choke points, routing medical supply and emergency relief convoys safely.
9. **Factory Tin-Roof Thermal Fatigue Model**: Simulates indoor ambient heat under uninsulated industrial tin sheds to plan worker shift rotations.
10. **Clean Air Workout & Activity Optimizer**: Identifies diurnal clean air windows where particulate pollution (PM2.5) drops below danger thresholds.
11. **Runway Crosswind & Gust Safety Monitor**: Evaluates crosswind vectors, convective wind shear, and air density altitude for regional airports.
12. **Water Treatment Turbidity & Runoff Silt Sentinel**: Predicts post-storm river turbidity surges to alert water treatment plants before filtration filters clog.
13. **Parametric Drought & Flood Insurance Trigger**: Provides transparent, automated index triggers for instant crop micro-insurance payouts.
14. **Coastal Squall & Fishermen Sea-State Monitor**: Computes wave heights, offshore gale wind risks, and cyclone approach bearings for coastal communities.
15. **Ice-to-Fish Ratio Preservation Calculator**: Calculates the extra ice tonnage required by artisanal fishing boats during high marine heatwaves.
16. **Tourist Outdoor Comfort & Attraction Optimizer**: Ranks sightseeing hours based on solar UV intensity, heat index, and sudden rain probability.

### Strategic Command Consoles
* **Crisis Council Multi-Stakeholder Matrix**: AI-driven emergency council that generates synchronized action plans across municipal authorities, hospitals, grid operators, and citizens.
* **Historical Disaster Replay Sandbox**: Re-simulates major historical Indian climate events (e.g., 2005 Mumbai Cloudburst, 1999 Odisha Super Cyclone) to evaluate preparedness and response strategies.

---

## Interactive Assistant Command Tools

The ClimaCore Intelligence Assistant features three interactive, slash-command modes:

* **`/plan` (Operational Risk Action Plan)**: Generates a prioritized, time-stamped checklist customized for the selected city, occupation, and upcoming 72-hour meteorological threats.
* **`/whatif` (Hypothetical Shock Simulation)**: Evaluates what would happen if a specific meteorological parameter experiences a shock (e.g., rainfall surges by +50 mm, or ambient temperature rises by +5 degrees C).
* **`/compare` (Multi-City Risk Benchmarking)**: Computes side-by-side compound risk ratings between two or more Indian cities to optimize logistics, travel, or supply chain distribution.

---

## Google Calendar Smart Planner Integration

ClimaCore bridges predictive climate foresight directly into real-world personal scheduling:
* **1-Click Safety Window Export**: When favorable weather windows or severe hazard periods are identified, users can export calibrated operational reminders directly into Google Calendar.
* **Automated Context Injection**: Calendar events are automatically populated with location coordinates, safety protocols, recommended hydration schedules, and contingency contacts.

---

## Verified Institutional Grounding

ClimaCore strictly adheres to authoritative national and international scientific benchmarks:
* **India Meteorological Department (IMD)**: Heat Action Plan stage triggers and rainfall classification standards.
* **Central Pollution Control Board (CPCB)**: National Ambient Air Quality Standards (NAAQS) and AQI sub-index break points.
* **Indian Council of Agricultural Research (ICAR)**: Crop phenology thresholds, thermal heat tolerances, and fungal pathogen infection curves.
* **Agmarknet (Ministry of Agriculture & Farmers Welfare)**: Certified daily agricultural produce market arrivals and modal price books.
* **Central Water Commission (CWC)**: Basin flood alert levels and reservoir storage thresholds.

---

## Technical Specifications Summary

* **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons, MapLibre GL.
* **Backend**: Python 3.12, FastAPI, Uvicorn, Pydantic v2, LightGBM, NumPy, Pandas, Scipy.
* **Generative Intelligence**: Google Gemini API (`gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`) with structured fallback heuristics.
* **Deployment Architecture**: Vercel Serverless Functions and Edge-compatible ASGI pipeline.
