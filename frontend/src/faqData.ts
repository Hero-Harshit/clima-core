export type FAQItem = {
  id: number;
  category: string;
  question: string;
  answer: string;
};

export const FAQ_DATA: FAQItem[] = [
  {
    "id": 1,
    "category": "Platform & Architecture",
    "question": "What is ClimaCore and what problem does it solve?",
    "answer": "ClimaCore is an open science environmental intelligence and decision support platform engineered specifically for the Indian subcontinent. Traditional forecasting systems operate in silos, reporting temperatures, rainfall, air quality, and river levels independently. ClimaCore unifies these disparate physical domains into a cohesive four-layer architecture, coupling machine learning with physical process models to simulate how environmental hazards compound, ripple through critical infrastructure, and affect distinct socio-economic personas."
  },
  {
    "id": 2,
    "category": "Platform & Architecture",
    "question": "How is ClimaCore structured across its four intelligence layers?",
    "answer": "The platform operates on four hierarchical layers:\n\nLayer 1 (Predictive Intelligence) executes calibrated machine learning models across 7-day horizons.\n\nLayer 2 (Physical and Statistical Intelligence) enforces thermodynamic, hydrological, and conservation laws through models such as USDA SCS-CN, GR4J rainfall-runoff, and boundary layer thermodynamics.\n\nLayer 3 (Exposure Intelligence) overlays real-time predictions onto geospatial infrastructure graphs, river reaches (HydroRIVERS), power grids, and population densities (WorldPop).\n\nLayer 4 (Decision Intelligence) translates technical indicators into actionable, persona-specific guidance, emergency operational protocols, and cross-sector impact cascades."
  },
  {
    "id": 3,
    "category": "Platform & Architecture",
    "question": "Who is ClimaCore built for and how is it used in practice?",
    "answer": "ClimaCore is designed for a diverse cross-section of society. Disaster management authorities (NDMA, SDMA, DDMA) utilize the War Room and Crisis Council for real-time stress testing and evacuation planning. Municipal authorities and urban planners monitor localized urban heat island anomalies, drainage bottlenecks, and water treatment requirements. Grid operators (POSOCO / Grid-India) forecast peak thermal electricity demand and transmission line derating risks. Agricultural extension workers track crop calendar phenology, root-zone moisture deficits, and heat stress windows. Vulnerable citizens, gig workers, traffic police, and healthcare providers receive tailored health advisories and operational guidance grounded in local conditions."
  },
  {
    "id": 4,
    "category": "Platform & Architecture",
    "question": "Is ClimaCore an official warning service or emergency dispatch tool?",
    "answer": "No. ClimaCore is an independent environmental intelligence and research platform designed to complement, not replace, official warning bulletins issued by the India Meteorological Department (IMD), the Central Water Commission (CWC), or the National Disaster Management Authority (NDMA). In emergency scenarios, users must follow directives issued by local civil defense and statutory emergency responders. ClimaCore provides scientific context, spatial simulations, and predictive risk estimates to support proactive planning."
  },
  {
    "id": 5,
    "category": "Platform & Architecture",
    "question": "How does ClimaCore handle offline, edge, or low-connectivity deployments?",
    "answer": "ClimaCore is built with an offline-first resilient architecture. The frontend bundles local vector tile renderers, cached baseline risk maps, and offline fallback heuristic engines. When network connectivity drops, the client continues to provide localized rule-based advisories and cached hazard matrices until synchronization with the backend inference engine is restored."
  },
  {
    "id": 6,
    "category": "Machine Learning & Uncertainty",
    "question": "Why does ClimaCore prioritize conformal prediction over raw model probabilities?",
    "answer": "Standard neural networks and gradient boosting models tend to be overconfident, especially during extreme, unprecedented weather anomalies. ClimaCore integrates split conformal prediction and non-exchangeable conformal inference to wrap point predictions in mathematically guaranteed prediction intervals at a specified confidence level (such as 90% coverage). Rather than stating tomorrow will reach 42.1 degrees Celsius, ClimaCore provides an interval of 40.8 to 43.4 degrees Celsius with rigorous marginal coverage guarantees."
  },
  {
    "id": 7,
    "category": "Machine Learning & Uncertainty",
    "question": "What machine learning algorithms are utilized across the platform?",
    "answer": "ClimaCore deploys an ensemble of state-of-the-art machine learning models tailored to specific physical domains. This includes LightGBM Gradient Boosted Decision Trees (GBDT) for extreme temperature and flood classification, XGBoost with Optuna hyperparameter optimization, CatBoost for categorical soil-climate interactions, soft-gated Mixture of Experts (MoE) architectures, and Ensemble Model Output Statistics (EMOS) for continuous ranked probability score (CRPS) minimization."
  },
  {
    "id": 8,
    "category": "Machine Learning & Uncertainty",
    "question": "What is the difference between Champion, Challenger, and Benchmark models?",
    "answer": "In ClimaCore, every production prediction is served by a validated Champion model that has demonstrated superior empirical performance across out-of-time evaluation splits. Alongside each champion, one or more Challenger models run in parallel shadow mode to benchmark emerging algorithms (such as XGBoost or neural MoE architectures). Benchmark models represent classical baseline methods (such as persistence, climatology, or pure physics equations) against which all machine learning improvements are rigorously quantified."
  },
  {
    "id": 9,
    "category": "Machine Learning & Uncertainty",
    "question": "How does ClimaCore validate models against historical extreme weather events?",
    "answer": "All models undergo temporal out-of-sample cross-validation and historical disaster replay validation. The suite is evaluated against landmark historical Indian disasters, including the 2015 Chennai floods, the 2018 Kerala floods, the 2022 South Asian heatwave, and the 2023 North India monsoon surges. Performance metrics reported include ROC-AUC, Precision-Recall AUC (PR-AUC), Brier Score, Continuous Ranked Probability Score (CRPS), and Kling-Gupta Efficiency (KGE)."
  },
  {
    "id": 10,
    "category": "Machine Learning & Uncertainty",
    "question": "How does ClimaCore prevent data leakage during time-series model training?",
    "answer": "Strict chronological splitting is enforced across all training, calibration, and test folds. Feature engineering pipelines strictly avoid using future indicators or unlagged observations. Calibration sets for conformal prediction are held completely separate from model training sets to ensure exchangeability assumptions remain valid. Spatial cross-validation groups geographically clustered stations to prevent spatial autocorrelation leakage."
  },
  {
    "id": 11,
    "category": "Flood & Hydrology",
    "question": "How does the FloodRisk V3 model calculate urban and district flood risk?",
    "answer": "FloodRisk V3 integrates a hybrid machine learning and physical hydrology architecture. It combines a 23-feature LightGBM classification engine with the USDA Natural Resources Conservation Service Soil Conservation Service Curve Number (SCS-CN) runoff formulation. By incorporating 3-day antecedent precipitation, 14-day antecedent precipitation index (API-14), catchment terrain slope from SRTM, river distance from HydroRIVERS, and soil hydrologic group classifications from SoilGrids, the model predicts both runoff depth (mm) and binary disaster occurrence probability."
  },
  {
    "id": 12,
    "category": "Flood & Hydrology",
    "question": "What is the Antecedent Precipitation Index (API-14) and why is it crucial?",
    "answer": "The Antecedent Precipitation Index (API-14) measures accumulated soil wetness over the preceding 14 days, with daily rainfall exponentially decayed by an evapotranspiration recession constant (typically k = 0.88 to 0.92). Dry soil can absorb significant rainfall without generating destructive surface runoff. Conversely, when API-14 is elevated, catchment retention capacity is exhausted, meaning even moderate precipitation can trigger rapid flash flooding and widespread inundation."
  },
  {
    "id": 13,
    "category": "Flood & Hydrology",
    "question": "How does the hybrid GR4J model function within the hydrological pipeline?",
    "answer": "GR4J (Genie Rural a 4 parametres Journalier) is a lumped conceptual rainfall-runoff model parameterized by four physical parameters: production store capacity (x1), groundwater exchange coefficient (x2), routing store capacity (x3), and unit hydrograph time base (x4). ClimaCore pairs GR4J with gradient boosting error correction to simulate daily streamflow, baseflow recession, and flood hydrographs across major river basins tracked by Central Water Commission gauge stations."
  },
  {
    "id": 14,
    "category": "Flood & Hydrology",
    "question": "How does ClimaCore simulate urban drainage bottlenecks during intense downpours?",
    "answer": "Urban environments replace permeable topsoil with asphalt and concrete, dramatically increasing the SCS Runoff Curve Number to 90 or higher. ClimaCore evaluates rainfall intensity against estimated municipal stormwater drainage thresholds (typically 20 to 50 mm per hour in Indian cities). When projected hourly precipitation exceeds municipal conveyance capacity, the model flags localized flash flood and waterlogging alerts regardless of regional river levels."
  },
  {
    "id": 15,
    "category": "Flood & Hydrology",
    "question": "What spatial datasets are used to evaluate flood exposure?",
    "answer": "Flood inundation footprints are intersected with high-resolution geospatial datasets:\n\nHydroRIVERS for stream reach flow lines, Strahler stream orders, and upstream catchment areas.\n\nSRTM 30m digital elevation models for slope, aspect, and Topographic Wetness Index (TWI).\n\nWorldPop 100m gridded population layers for localized headcount exposure.\n\nOpenStreetMap infrastructure layers for hospitals, schools, electrical substations, and critical transport corridors."
  },
  {
    "id": 16,
    "category": "Heat Stress & Biometeorology",
    "question": "How does HeatRisk V4 differ from standard temperature forecasts?",
    "answer": "Traditional forecasts report only dry-bulb air temperature. HeatRisk V4 models the human physiological heat balance. It calculates Apparent Temperature (\"Feels Like\"), Wet-Bulb Globe Temperature (WBGT), and the Simplified Wet-Bulb Globe Temperature (sWBGT) by synthesizing 2-meter air temperature, relative humidity, 10-meter wind speed, solar irradiance, and urban thermal mass. This accurately reflects the body diminished ability to cool itself through sweat evaporation in hot and humid climates."
  },
  {
    "id": 17,
    "category": "Heat Stress & Biometeorology",
    "question": "What is Wet-Bulb Temperature and why is it a critical survival threshold?",
    "answer": "Wet-bulb temperature represents the lowest temperature achievable solely by evaporative cooling of a wetted surface exposed to air. When wet-bulb temperature reaches 35 degrees Celsius (or around 31 to 32 degrees Celsius for individuals performing physical labor), the thermodynamic gradient between the human core and the ambient air vanishes. Under these conditions, the body cannot shed metabolic heat through perspiration, leading to dangerous hyperthermia and fatal heatstroke within hours without active mechanical cooling."
  },
  {
    "id": 18,
    "category": "Heat Stress & Biometeorology",
    "question": "What is the Urban Heat Island (UHI) delta and how is it quantified?",
    "answer": "The Urban Heat Island (UHI) delta is the temperature differential between dense built-up city centers and surrounding rural or vegetated areas. ClimaCore estimates UHI delta by analyzing impervious surface fractions, Moderate Resolution Imaging Spectroradiometer (MODIS) land surface temperatures, anthropogenic heat emissions, and building density. During heatwave episodes, urban cores can remain 3 to 7 degrees Celsius warmer overnight than adjacent rural plains, denying vulnerable residents nocturnal cooling relief."
  },
  {
    "id": 19,
    "category": "Heat Stress & Biometeorology",
    "question": "How does ClimaCore predict nocturnal heat stress and tropical nights?",
    "answer": "A \"tropical night\" is defined as a 24-hour period where the minimum overnight temperature does not drop below 25 degrees Celsius (or 27 degrees Celsius during severe heatwaves). ClimaCore utilizes a specialized diurnal temperature cycle decay model to predict minimum nocturnal temperatures. When nocturnal minimums remain elevated, cumulative cardiovascular stress surges because the human body cannot recover from daytime heat exposure, drastically elevating hospital admission rates."
  },
  {
    "id": 20,
    "category": "Heat Stress & Biometeorology",
    "question": "What is the Soft Mixture of Experts (MoE) architecture used in HeatRisk?",
    "answer": "The HeatRisk Soft MoE architecture employs five specialized regression experts, each trained on distinct climatic regimes: arid desert, humid coastal, sub-tropical continental, montane, and tropical savannah. A softmax gating network dynamically weights the contribution of each expert based on current atmospheric state vectors (such as boundary layer humidity, surface pressure, and potential temperature), achieving superior generalizability across the vast climatic diversity of India."
  },
  {
    "id": 21,
    "category": "Air Quality & Atmosphere",
    "question": "What pollutants does the AirQuality V3 engine track and forecast?",
    "answer": "AirQuality V3 predicts concentrations and National Air Quality Index (NAQI) sub-indices for key criteria pollutants: Fine Particulate Matter (PM2.5), Coarse Particulate Matter (PM10), Nitrogen Dioxide (NO2), Sulfur Dioxide (SO2), Carbon Monoxide (CO), and Surface Ozone (O3). Models are calibrated using ground-truth observational telemetry from the Central Pollution Control Board (CPCB) continuous ambient air quality monitoring network."
  },
  {
    "id": 22,
    "category": "Air Quality & Atmosphere",
    "question": "How does ClimaCore detect and model nocturnal boundary layer temperature inversions?",
    "answer": "During winter months across the Indo-Gangetic Plains, rapid radiative cooling of the ground produces thermal inversions where warm air aloft caps colder, denser air near the surface. ClimaCore simulates the planetary boundary layer height (PBLH) and calculates an inversion strength index from European Centre for Medium-Range Weather Forecasts (ECMWF) ERA5 vertical temperature profiles. When boundary layer height compresses below 200 meters with wind speeds under 1.5 m/s, the system flags boundary-layer stagnation, signaling rapid accumulation of toxic surface pollutants."
  },
  {
    "id": 23,
    "category": "Air Quality & Atmosphere",
    "question": "How does ClimaCore incorporate agricultural stubble burning and biomass plumes?",
    "answer": "ClimaCore ingests thermal anomaly and Fire Radiative Power (FRP) detections from NASA VIIRS and MODIS satellites. Downwind transport and smoke dispersion are modeled using trajectory simulations coupled with boundary-layer wind vectors, allowing the platform to estimate the transboundary contribution of agricultural crop residue burning to urban airsheds in northern India."
  },
  {
    "id": 24,
    "category": "Air Quality & Atmosphere",
    "question": "What is the difference between primary pollutant accumulation and secondary ozone formation?",
    "answer": "Primary pollutants (such as PM2.5, NO2, and CO) reach peak ground concentrations under cold, stagnant nocturnal inversion conditions due to direct vehicular, industrial, and biomass emissions. In contrast, ground-level ozone (O3) is a secondary photochemical oxidant synthesized when nitrogen oxides and volatile organic compounds react in the presence of intense solar ultraviolet radiation and high temperatures. ClimaCore models both mechanisms independently to alert users to daytime ozone surges as well as nocturnal particulate spikes."
  },
  {
    "id": 25,
    "category": "Air Quality & Atmosphere",
    "question": "How are individual pollutant sub-indices converted into the National Air Quality Index (NAQI)?",
    "answer": "Following Central Pollution Control Board (CPCB) standards, individual pollutant concentrations are converted into non-linear sub-indices through segmented linear interpolation against defined health thresholds. The overall reported AQI corresponds to the maximum sub-index across all measured pollutants (the dominant pollutant principle), with descriptive health categories ranging from Good (0 to 50), Satisfactory (51 to 100), Moderate (101 to 200), Poor (201 to 300), Very Poor (301 to 400), to Severe (401 to 500)."
  },
  {
    "id": 26,
    "category": "Power Grid & Energy",
    "question": "How does the PowerGrid V3 model predict electricity demand and blackout probability?",
    "answer": "PowerGrid V3 correlates meteorological variables with regional power consumption data published by the Central Electricity Authority (CEA) and Power System Operation Corporation (POSOCO / Grid-India). By analyzing cooling degree days (CDD), wet-bulb humidity penalties, agricultural irrigation pumping cycles, and industrial schedules, the model forecasts next-day peak demand (in Megawatts) and computes a probabilistic blackout vulnerability score based on regional reserve margins and transmission limits."
  },
  {
    "id": 27,
    "category": "Power Grid & Energy",
    "question": "What is the relationship between extreme heat and power transmission efficiency?",
    "answer": "Elevated ambient temperatures impose a double penalty on power systems. First, consumer demand surges due to air conditioning and agricultural pumping. Second, physical power infrastructure suffers thermal derating: overhead transmission lines experience thermal sag and increased electrical resistance (I-squared-R losses), while power transformers overheat, reducing their safe operating throughput. ClimaCore explicitly models this infrastructure stress factor."
  },
  {
    "id": 28,
    "category": "Power Grid & Energy",
    "question": "How does rainfall or storm activity affect grid stability in ClimaCore?",
    "answer": "While moderate rainfall often reduces air conditioning demand, convective storm cells, high wind gusts, and severe lightning present physical hazards to electrical distribution networks. ClimaCore evaluates storm wind thresholds alongside soil saturation around transmission towers to estimate line tripping, tree-fall outages, and localized distribution failures."
  },
  {
    "id": 29,
    "category": "Power Grid & Energy",
    "question": "How does agricultural demand interact with municipal electricity consumption?",
    "answer": "In agricultural states, rural irrigation feeders supply deep groundwater tubewells during dry spells. When monsoon rains fail or delay, agricultural power demand peaks simultaneously with municipal cooling demand, creating compound stress on regional substations. ClimaCore models this multi-sector load synchrony to alert grid dispatchers before supply deficits emerge."
  },
  {
    "id": 30,
    "category": "Agriculture & AgriSense",
    "question": "What is AgriSense V3 and what indicators does it evaluate for Indian farming?",
    "answer": "AgriSense V3 is an agro-climatic intelligence engine calibrated to Indian agricultural seasons (Kharif, Rabi, and Zaid). It tracks root-zone soil moisture deficits, accumulated thermal time (Growing Degree Days), evapotranspiration rates (Penman-Monteith formulation), and crop-specific phenological susceptibility windows for major staple crops, including rice, wheat, cotton, sugarcane, pulses, and jute."
  },
  {
    "id": 31,
    "category": "Agriculture & AgriSense",
    "question": "How does ClimaCore identify and signal a flash drought?",
    "answer": "Unlike conventional droughts that develop over months, flash droughts emerge rapidly within two to three weeks due to an acute combination of abnormally high temperatures, strong winds, intense solar radiation, and persistent rainfall deficits. This triggers rapid atmospheric evaporative demand (an evaporative stress spike) that quickly depletes root-zone soil moisture. AgriSense V3 detects the velocity of soil moisture depletion to issue early warnings before irreversible crop wilting occurs."
  },
  {
    "id": 32,
    "category": "Agriculture & AgriSense",
    "question": "What is the role of the Normalized Difference Vegetation Index (NDVI) in AgriSense?",
    "answer": "AgriSense integrates satellite-derived vegetation indices, including 16-day MODIS and Sentinel-2 NDVI anomalies, to evaluate real-time crop vigor against historical 10-year baseline medians. Significant negative NDVI anomalies combined with elevated soil moisture deficits validate ground-level vegetative stress and guide localized irrigation advisory delivery."
  },
  {
    "id": 33,
    "category": "Agriculture & AgriSense",
    "question": "How does ClimaCore provide crop-specific harvesting and sowing windows?",
    "answer": "For every supported district and crop, AgriSense maps agricultural calendars against incoming 7-day weather predictions. For example, during the Kharif rice or jute harvesting window, the system alerts farmers to incoming rain bands that could ruin harvested grain drying in open fields, while during sowing windows, it highlights optimum soil moisture and temperature thresholds for successful seed germination."
  },
  {
    "id": 34,
    "category": "Water Quality & Potability",
    "question": "How does the WaterQuality V1 engine assess groundwater potability?",
    "answer": "WaterQuality V1 evaluates chemical and physical parameters from over 15,000 monitoring wells maintained by the Central Ground Water Board (CGWB) and state water departments. The engine calculates a composite Water Quality Index (WQI) based on Total Dissolved Solids (TDS), electrical conductivity, pH, fluoride, nitrate, arsenic, and total hardness relative to Bureau of Indian Standards (BIS IS 10500:2012) drinking water specifications."
  },
  {
    "id": 35,
    "category": "Water Quality & Potability",
    "question": "What causes groundwater contamination to worsen during dry or monsoon seasons?",
    "answer": "Groundwater chemistry exhibits strong seasonal dynamics. In prolonged dry seasons, intense water extraction causes the water table to drop, concentrating dissolved salts, fluorides, and heavy metals. Conversely, the arrival of heavy monsoon rains can cause rapid recharge through porous topsoil, flushing agricultural nitrates, industrial effluents, and surface pathogens into shallow aquifers. ClimaCore models these seasonal hydrologic dynamics to adjust potability risk indicators."
  },
  {
    "id": 36,
    "category": "Water Quality & Potability",
    "question": "What is the Water Quality Index (WQI) scale used on the platform?",
    "answer": "ClimaCore employs a weighted arithmetic Water Quality Index (WQI) normalized on a 0 to 100 scale:\n\nExcellent (WQI 0 to 25): Fully compliant with BIS standards, suitable for direct consumption.\n\nGood (WQI 26 to 50): Compliant with minor aesthetic variance, suitable for consumption.\n\nPoor (WQI 51 to 75): Exceeds recommended limits for one or more parameters, treatment recommended.\n\nVery Poor (WQI 76 to 100): Significantly exceeds permissible limits, unsafe for drinking without advanced purification (such as reverse osmosis or ion exchange)."
  },
  {
    "id": 37,
    "category": "Water Quality & Potability",
    "question": "How can local communities and municipalities use water quality predictions?",
    "answer": "Local authorities use water quality outputs to plan municipal filtration protocols, schedule water tanker deliveries to contaminated wards, and prioritize borehole testing. Healthcare professionals cross-reference water quality alerts with gastrointestinal disease surveillance to prepare localized medical interventions."
  },
  {
    "id": 38,
    "category": "Multi-Hazard Cascades & War Room",
    "question": "What is a multi-hazard cascade and how does ClimaCore model it?",
    "answer": "Environmental disasters rarely occur in isolation; one hazard triggers failure in another domain. For instance, an extreme heatwave surges air conditioner electricity demand; overloaded power transformers fail; municipal water pumping stations shut down; residents face simultaneous heat stroke and acute water shortages. ClimaCore uses directed acyclic dependency graphs and multivariate copula models to simulate these cascading chain reactions and quantify cross-sector systemic vulnerability."
  },
  {
    "id": 39,
    "category": "Multi-Hazard Cascades & War Room",
    "question": "What is the War Room and how do emergency managers use it?",
    "answer": "The War Room is ClimaCore situational command center. It provides incident commanders with live synoptic views of all active environmental hazards, automated cascade dependency trees, geographical critical infrastructure overlays, and real-time population vulnerability metrics across an entire metropolitan region or state."
  },
  {
    "id": 40,
    "category": "Multi-Hazard Cascades & War Room",
    "question": "What is the Crisis Council feature within ClimaCore?",
    "answer": "The Crisis Council is an interactive tabletop simulation environment where emergency planners can test \"what-if\" scenarios before an event strikes. Planners can manually adjust peak temperatures, simulate an additional 100 mm of unpredicted deluge, or simulate a major substation blackout. The system recalculates all downstream physical models in real time, projecting changes in hospital surge capacity, road passability, and emergency shelter requirements."
  },
  {
    "id": 41,
    "category": "Multi-Hazard Cascades & War Room",
    "question": "What is Disaster Replay and how is it used for training and validation?",
    "answer": "Disaster Replay allows planners, researchers, and students to step through historical Indian extreme weather events hour by hour. By loading benchmark events (such as the 2015 Chennai floods or the 2022 heatwaves), users observe how the models reconstruct historical conditions and evaluate whether current predictive thresholds would have provided adequate early warning lead times."
  },
  {
    "id": 42,
    "category": "Multi-Hazard Cascades & War Room",
    "question": "How does ClimaCore calculate the Compound Risk Score?",
    "answer": "The Compound Risk Score is a normalized index between 0 and 100 derived from multivariate Vine Copula joint probability distributions. Rather than simply adding individual risk scores together, the copula engine accounts for non-linear tail dependence between simultaneous hazards (such as high heat occurring concurrently with severe particulate pollution, or high rainfall coinciding with high coastal storm surge)."
  },
  {
    "id": 43,
    "category": "Personas & ClimaCore AI",
    "question": "Why does ClimaCore customize briefings for different personas and occupations?",
    "answer": "The impact of a 42-degree Celsius day is drastically different for an air-conditioned office worker compared to an outdoor construction laborer, a delivery gig worker on a motorcycle, or a traffic police officer standing on hot asphalt for eight hours. ClimaCore provides tailored advisories calibrated to the specific physiological exposure, physical exertion, and operational environment of eight distinct personas."
  },
  {
    "id": 44,
    "category": "Personas & ClimaCore AI",
    "question": "Which personas are supported in ClimaCore today?",
    "answer": "ClimaCore currently provides tailored briefings for eight distinct operational groups:\n\n1. Gig Workers and Outdoor Laborers (delivery drivers, construction workers, street vendors).\n2. Traffic Police and Transport Workers (roadside exposure, particulate inhalation, heat radiance).\n3. Agricultural Producers and Smallholders (irrigation management, crop protection, livestock welfare).\n4. Industrial and Warehouse Managers (thermal stress on factory floors, inventory humidity control).\n5. Municipal and Civil Defense Officers (drainage management, water distribution, heat shelter logistics).\n6. Healthcare and Hospital Administrators (heatstroke bed availability, respiratory disease surge prep).\n7. Vulnerable Citizens and Elderly Caregivers (indoor thermal safety, hydration protocols, air filtration).\n8. School Administrators and Educators (outdoor sports restrictions, classroom heat safety, school bus ventilation)."
  },
  {
    "id": 45,
    "category": "Personas & ClimaCore AI",
    "question": "How does the ClimaCore AI integration generate persona briefings?",
    "answer": "ClimaCore utilizes a strictly grounded, schema-driven ClimaCore Neural Intelligence engine. The backend feeds deterministic physical model outputs (such as exact wet-bulb temperature, AQI sub-indices, flood runoff depth, and workability scores) into the synthesis pipeline. ClimaCore translates these quantitative metrics into natural, highly empathetic, occupational operational guidance and sector advisories without hallucinating facts or overriding physical model outputs."
  },
  {
    "id": 46,
    "category": "Personas & ClimaCore AI",
    "question": "What happens if an external inference endpoint is unavailable or rate-limited?",
    "answer": "Reliability is paramount in environmental intelligence. If any inference endpoint experiences a network interruption or latency, the ClimaCore backend immediately falls back to a deterministic, rule-based expert recommendation matrix. Users will always see validated operational guidance and actionable safety advisories with guaranteed high availability."
  },
  {
    "id": 47,
    "category": "Data Governance & Science",
    "question": "Where does ClimaCore source its raw environmental and geospatial data?",
    "answer": "All input data originates from authoritative, open-access scientific repositories and statutory government agencies. Major sources include the India Meteorological Department (IMD), Central Water Commission (CWC), Central Ground Water Board (CGWB), Central Pollution Control Board (CPCB), European Centre for Medium-Range Weather Forecasts (ECMWF ERA5), NASA Earthdata, HydroSHEDS / HydroRIVERS, SRTM topographic data, and WorldPop demographic rasters."
  },
  {
    "id": 48,
    "category": "Data Governance & Science",
    "question": "How does ClimaCore protect user privacy and handle location data?",
    "answer": "ClimaCore adheres to strict privacy-by-design principles. User geolocation data is processed ephemerally on the client or resolved to regional district centroids. The platform does not track individual user identities, does not sell telemetry to third-party data brokers, and does not retain personal location search histories. All spatial analytical processing focuses on environmental geography, not personal identity."
  },
  {
    "id": 49,
    "category": "Data Governance & Science",
    "question": "Is ClimaCore open source and how can researchers contribute?",
    "answer": "Yes. ClimaCore is developed as an open science initiative. Codebases, model architecture specifications, training benchmarks, and evaluation notebooks are published under open licenses. Academic researchers, hydrologists, and data scientists are encouraged to benchmark their own challenger models, submit dataset connectors, and audit algorithmic assumptions via our public repositories."
  },
  {
    "id": 50,
    "category": "Data Governance & Science",
    "question": "How does ClimaCore ensure transparency and explainability in its predictions?",
    "answer": "ClimaCore features a dedicated Transparency and Trust engine that exposes the exact model names, versions, input variables, physical units, and data providers behind every briefing. Feature attribution displays show the relative contribution of each meteorological variable to the final risk output, ensuring that users, scientists, and officials can always trace and verify the reasoning behind every prediction."
  }
];

export const FAQ_CATEGORIES = [
  "All",
  "Platform & Architecture",
  "Machine Learning & Uncertainty",
  "Flood & Hydrology",
  "Heat Stress & Biometeorology",
  "Air Quality & Atmosphere",
  "Power Grid & Energy",
  "Agriculture & AgriSense",
  "Water Quality & Potability",
  "Multi-Hazard Cascades & War Room",
  "Personas & ClimaCore AI",
  "Data Governance & Science",
] as const;
