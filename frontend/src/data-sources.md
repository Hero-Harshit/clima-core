# Data & Data Sources

## The Data Behind Environmental Intelligence
Reliable environmental intelligence begins with reliable data.

Clima-Core brings together information from multiple meteorological, hydrological, atmospheric, agricultural, geographical, and infrastructure-related sources to develop a broader understanding of environmental conditions and their potential impacts.

Our platform uses a combination of historical observations, scientific datasets, numerical weather prediction products, satellite-derived information, monitoring networks, and geospatial resources.

These datasets provide the foundation for our machine learning models, physical calculations, statistical analysis, and geographical exposure assessments.

Rather than depending on a single data provider, Clima-Core integrates information from different sources according to the environmental problem being investigated.

Our objective is to transform diverse scientific information into structured, traceable, and meaningful environmental intelligence.

---

# 1. Our Data Philosophy
Clima-Core follows a data-driven approach supported by five important principles.

### 1.1 Scientific Credibility
We prioritise environmental datasets from recognised government agencies, research institutions, scientific organisations, and established data providers.

### 1.2 Source Transparency
We aim to identify the original providers of our datasets and acknowledge the organisations, researchers, and communities responsible for making them available.

### 1.3 Data Quality
Environmental information is inspected and processed according to its intended analytical purpose. Relevant checks may include missing-value detection, duplicate identification, unit verification, temporal alignment, and geographical consistency.

### 1.4 Responsible Integration
Different datasets may have different spatial resolutions, temporal coverage, units, measurement methods, and levels of uncertainty.

We account for these differences when combining information within analytical pipelines.

### 1.5 Traceability
Where supported by the underlying system, datasets and model outputs are associated with source information, versions, processing records, and other relevant provenance metadata.

---

# 2. Categories of Environmental Data
Clima-Core works with several broad categories of environmental information.

## Meteorological Data
Information relating to rainfall, temperature, humidity, wind, atmospheric conditions, and weather forecasts.

This information supports weather analysis, heat-risk assessment, rainfall modelling, drought analysis, and flood-related predictions.

## Hydrological Data
Information relating to rivers, streamflow, catchments, groundwater, reservoir storage, soil moisture, and rainfall-runoff processes.

It supports hydrological modelling, flood intelligence, drought assessment, and water-resource analysis.

## Atmospheric and Air Quality Data
Information relating to particulate matter, atmospheric composition, aerosol properties, and active fire detections.

It supports pollutant estimation, smoke-related analysis, and environmental air-quality assessment.

## Agricultural and Soil Data
Information relating to agricultural production, crop suitability, soil characteristics, crop calendars, and environmental conditions affecting agriculture.

It supports agricultural intelligence and the analysis of climate-related crop stress.

## Geospatial and Population Data
Information describing geographical features, river networks, elevation, population distribution, roads, and critical infrastructure.

It provides the spatial context needed to understand where environmental hazards may be relevant.

## Energy and Infrastructure Data
Information relating to historical electricity demand, regional grid conditions, and environmental factors associated with infrastructure stress.

It supports the analysis of environmental influences on electricity systems and essential services.

---

# 3. Our Primary Data Sources
The following sections describe the major source families represented in the Clima-Core data registry.

The exact dataset version, downloaded period, geographical coverage, and processing configuration may vary between individual model pipelines.

## 3.1 India Meteorological Department (IMD)
**Data category:** Meteorological observations

**Official source:** [https://mausam.imd.gov.in/](https://mausam.imd.gov.in/)

The India Meteorological Department is India's national meteorological agency and an important source of meteorological information used within Clima-Core.

Our meteorological data foundation includes daily gridded rainfall and temperature products.

### A. IMD Gridded Rainfall
**Dataset characteristics**

- Provider: India Meteorological Department.
- Spatial resolution: 0.25° × 0.25°.
- Temporal coverage of the referenced product: 1901 to 2024.
- Temporal resolution: Daily.
- Primary variable: Rainfall depth in millimetres.
The IMD gridded rainfall dataset provides long-term daily rainfall information across India.

**How Clima-Core uses it**

The dataset supports:

- Historical rainfall analysis.
- Long-term precipitation patterns.
- Rainfall accumulation features.
- Antecedent precipitation calculations.
- Flood-risk modelling.
- Drought-related analysis.
- Extreme rainfall assessment.
- Historical environmental comparisons.
Long historical coverage is particularly valuable for studying seasonal behaviour, rainfall variability, and unusual precipitation events.

**Important limitation**

A daily gridded rainfall value does not identify the exact hourly distribution of rainfall.

The grid represents an interpolated spatial dataset and should not be interpreted as a direct measurement at every location or as a precise urban rain-gauge observation.

### B. IMD Gridded Temperature
**Dataset characteristics**

- Provider: India Meteorological Department.
- Spatial resolution: 1.0° × 1.0°.
- Temporal coverage of the referenced product: 1951 to 2024.
- Temporal resolution: Daily.
- Variables: Maximum and minimum temperature products.
- Unit: Degrees Celsius.
**How Clima-Core uses it**

The temperature datasets support:

- Historical temperature analysis.
- Heat-risk model development.
- Temperature anomaly calculations.
- Seasonal and regional comparisons.
- Thermal stress analysis.
- Climate variability studies.
**Important limitation**

The 1° grid is relatively coarse for detailed urban analysis.

The IMD documentation also notes that the minimum-temperature grid from 2008 onwards uses a smaller number of reporting stations than the earlier historical product.

This is important when interpreting long-term comparisons and evaluating city-level predictions.

Source: [IMD Gridded Rainfall](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_Bin.html) | [IMD Gridded Temperature](https://www.imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html)

---

## 3.2 ERA5-Land Reanalysis
**Data category:** Meteorological and land-surface information

**Provider:** European Centre for Medium-Range Weather Forecasts (ECMWF), through the Copernicus Climate Change Service.

**Official source:** [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

ERA5-Land is a global land-surface reanalysis dataset that provides historical estimates of meteorological and land-surface conditions.

It combines a land-surface modelling system with meteorological forcing from ERA5.

**Dataset characteristics**

- Spatial resolution: Approximately 0.1° or 9 km.
- Official consolidated product coverage: From January 1950 to approximately two to three months before the present, subject to product updates.
- Multiple hourly and derived variables.
- Global geographical coverage.
Variables relevant to Clima-Core include:

- 2-metre air temperature.
- Dew-point temperature.
- Precipitation.
- Volumetric soil moisture.
- Surface runoff.
- Evapotranspiration-related variables.
- Wind components.
- Surface radiation.
- Other land-surface indicators.
**How Clima-Core uses it**

ERA5-Land contributes to:

- Hydro-atmospheric analysis.
- Soil moisture estimation.
- Antecedent wetness assessment.
- Drought monitoring.
- Environmental feature engineering.
- Evapotranspiration analysis.
- Heat and moisture-related modelling.
It is particularly useful because it provides temporally consistent gridded information across long historical periods.

**Important limitation**

Reanalysis is not identical to a direct observation.

Its values are generated through a numerical modelling and data-assimilation system. Spatial resolution, forcing characteristics, temporal aggregation, and variable definitions must be considered when interpreting the results.

Source: [ERA5-Land Documentation](https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation)

---

## 3.3 India Flood Inventory (IFI v3)
**Data category:** Hydrological and historical flood-event information

**Source identified in project records:** IIT Delhi and associated Indian water-sector data sources.

The India Flood Inventory is used within Clima-Core as a source of historical flood-event information.

Unlike rainfall datasets, an event inventory provides information about recorded flood occurrences and their geographical and temporal context.

**Project dataset characteristics**

- Spatial coverage: District-level Indian flood records.
- Temporal coverage of the ingested version: 1980 to 2023, as recorded in the project dossier.
- Information: Historical declared flood events and associated event attributes, subject to the available inventory fields.
**How Clima-Core uses it**

The inventory supports:

- Supervised FloodRisk model development.
- Historical flood-event identification.
- Flood occurrence classification.
- Model benchmarking.
- Event-based validation.
- Historical geographical risk analysis.
It helps connect meteorological and hydrological conditions with recorded flood-event outcomes.

**Important limitation**

A district-level event record does not necessarily provide precise flood boundaries, inundation depths, hourly flood timing, or building-level impact information.

The coverage and reliability of the event labels depend on the underlying reporting and documentation process.

The exact dataset citation, release version, access conditions, and licensing information should be obtained from the corresponding local data registry and original source documentation.

---

## 3.4 Central Pollution Control Board (CPCB)
**Data category:** Air-quality monitoring observations

**Official source:** [https://cpcb.nic.in/](https://cpcb.nic.in/)

The Central Pollution Control Board is an important national source of environmental monitoring information, including ambient air quality.

Clima-Core uses CPCB monitoring information as an important reference for air-quality analysis and model evaluation.

**Relevant variables may include**

- PM2.5.
- PM10.
- Nitrogen dioxide (NO₂).
- Sulphur dioxide (SO₂).
- Carbon monoxide (CO).
- Ozone (O₃).
- Other monitored atmospheric pollutants, where available.
**How Clima-Core uses it**

The data support:

- Historical pollutant concentration analysis.
- AirQuality model training and evaluation.
- Ground-monitoring comparisons.
- Pollution category analysis.
- Model calibration and validation.
- Geographical monitoring-station analysis.
CPCB information also provides the reference framework for India's National Air Quality Index and its pollutant sub-index calculations.

**Important limitation**

Monitoring-station coverage is geographically uneven.

Different stations may have different observation periods, equipment characteristics, missing records, and pollutant availability.

A station measurement should not automatically be interpreted as the pollutant concentration at every location within a city.

Source: [CPCB National AQI](https://airquality.cpcb.gov.in/AQI_India/) | [CPCB Air Quality Resources](https://app.cpcbccr.com/ccr_docs)

---

## 3.5 Copernicus Atmosphere Monitoring Service (CAMS)
**Data category:** Atmospheric composition and aerosol information

**Provider:** Copernicus Atmosphere Monitoring Service, implemented by ECMWF.

**Official source:** [https://atmosphere.copernicus.eu/](https://atmosphere.copernicus.eu/)

CAMS provides atmospheric composition information generated through satellite observations, atmospheric modelling, and data-assimilation methods.

Clima-Core uses CAMS-related information to complement ground-level air-quality observations and meteorological variables.

**Relevant information includes**

- Aerosol-related variables.
- Aerosol optical depth.
- Atmospheric particulate components.
- Dust.
- Organic matter.
- Black carbon.
- Other available atmospheric composition fields.
**How Clima-Core uses it**

CAMS information supports:

- Atmospheric haze analysis.
- AirQuality modelling.
- Regional aerosol assessment.
- Smoke-related environmental analysis.
- Comparisons with ground-based particulate measurements.
It provides broader atmospheric context beyond individual monitoring stations.

**Important limitation**

Aerosol Optical Depth measures the integrated aerosol extinction through an atmospheric column.

It is not the same as ground-level PM2.5 concentration.

Atmospheric composition products also have their own spatial and temporal resolutions, which must be respected when integrating them with local monitoring data.

Source: [CAMS Reanalysis Documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=212442031)

---

## 3.6 NASA Fire Information for Resource Management System (FIRMS)
**Data category:** Satellite-based active fire observations

**Provider:** NASA Earth Science Data and Information System.

**Official source:** [https://firms.modaps.eosdis.nasa.gov/](https://firms.modaps.eosdis.nasa.gov/)

NASA FIRMS distributes active fire detection information from satellite instruments, including MODIS and VIIRS.

These products identify satellite-observed thermal anomalies associated with possible active fire activity.

**Relevant information includes**

- Fire detection coordinates.
- Observation time.
- Fire Radiative Power (FRP), where available.
- Detection confidence.
- Satellite and sensor identification.
**How Clima-Core uses it**

FIRMS information supports:

- Active fire analysis.
- Fire Radiative Power features.
- SmokePath modelling.
- Investigation of potential biomass-burning influences on air quality.
- Spatial and temporal comparison of fire activity with atmospheric conditions.
Combined with wind and atmospheric information, fire observations can contribute to the investigation of potential smoke movement.

**Important limitation**

A satellite active-fire detection is not a direct measurement of the amount of smoke reaching a particular ground location.

Cloud cover, sensor characteristics, overpass timing, detection limits, and atmospheric conditions affect the observations.

Active-fire detections should not be confused with complete burned-area inventories or confirmed ground-level pollution measurements.

For scientific applications, users should distinguish near-real-time products from standard science-quality products, since the underlying processing can differ.

Source: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/)

---

## 3.7 WorldPop
**Data category:** Population and demographic geography

**Provider:** WorldPop, University of Southampton and associated research collaborators.

**Official source:** [https://www.worldpop.org/](https://www.worldpop.org/)

WorldPop produces gridded population datasets designed to represent the geographical distribution of population.

Clima-Core uses population information to investigate the potential human exposure associated with environmental conditions.

**Dataset characteristics**

- Gridded population information.
- Approximately 100-metre products, depending on the dataset.
- Geographical coverage across India.
- Different reference years and modelled population estimates.
The Clima-Core v2.0 technical dossier identifies a 2020 population baseline for its existing exposure workflow. The exact active raster version should be confirmed through the deployed dataset manifest.

**How Clima-Core uses it**

WorldPop supports:

- Population distribution analysis.
- Population-density estimation.
- Geographical exposure assessment.
- Population-weighted environmental screening.
- Identification of potentially exposed areas.
The information can help contextualise environmental hazards in terms of the population potentially located within relevant geographical areas.

**Important limitation**

A gridded population estimate does not represent an exact real-time count of people present in every grid cell.

Population estimates depend on the reference year, modelling methodology, census inputs, and geographical assumptions.

Population located near a hazard is not necessarily population directly affected by it.

Source: [WorldPop](https://www.worldpop.org/) | [WorldPop Data API and Documentation](https://api.worldpop.org/v3/documentation)

---

## 3.8 OpenStreetMap (OSM)
**Data category:** Geospatial infrastructure and geographic features

**Provider:** OpenStreetMap contributors.

**Official source:** [https://www.openstreetmap.org/](https://www.openstreetmap.org/)

OpenStreetMap is a collaborative geographical database containing mapped features contributed and maintained by its global community.

Clima-Core uses selected OSM information to analyse the geographical distribution of infrastructure and other relevant features.

**Relevant feature categories may include**

- Roads and transport networks.
- Bridges.
- Hospitals and healthcare facilities.
- Electrical substations.
- Railway infrastructure.
- Water-related facilities.
- Canals and mapped drainage features.
- Other geographical points and lines.
**How Clima-Core uses it**

OSM information supports:

- Infrastructure exposure screening.
- Distance calculations.
- Geographical intersections.
- Infrastructure proximity analysis.
- Environmental risk contextualisation.
For example, an environmental risk assessment may consider the proximity of a location to mapped hospitals, bridges, transport networks, and electrical infrastructure.

**Important limitation**

OpenStreetMap completeness varies by location and feature category.

The absence of a mapped feature does not establish that the feature does not exist.

Mapped drainage canals should not automatically be interpreted as underground drainage pipes or complete urban stormwater systems.

**Licensing and attribution**

OpenStreetMap data is licensed under the Open Data Commons Open Database License (ODbL).

Clima-Core acknowledges OpenStreetMap and its contributors. Attribution and other applicable licence obligations must be respected when using, modifying, or distributing OSM-derived data.

Source: [OpenStreetMap](https://www.openstreetmap.org/) | [OpenStreetMap Copyright and Licence](https://www.openstreetmap.org/copyright)

---

## 3.9 HydroRIVERS and HydroSHEDS
**Data category:** River networks and hydrological geography

**Provider:** HydroSHEDS, developed through the WWF-supported HydroSHEDS initiative.

**Official source:** [https://www.hydrosheds.org/products/hydrorivers](https://www.hydrosheds.org/products/hydrorivers)

HydroRIVERS provides a structured geographical representation of river networks.

It consists of vectorised river reaches derived from the HydroSHEDS hydrological data framework.

The source dataset represents a global river network with an underlying terrain-derived resolution of approximately 15 arc-seconds.

**Relevant information includes**

- River-reach geometry.
- River-network structure.
- Stream order.
- Reach length.
- Upstream and downstream network characteristics.
- Estimated long-term average discharge.
- Associated sub-basin identifiers.
**How Clima-Core uses it**

HydroRIVERS contributes to:

- River proximity analysis.
- Catchment-related geographical features.
- Stream order identification.
- Flood-risk feature engineering.
- Hydrological connectivity investigations.
- River-related exposure assessment.
The project dossier reports the use of more than 531,000 Indian river reaches within its extracted network.

**Important limitation**

HydroRIVERS is designed to represent a particular scale of natural river networks. It does not include every small stream, urban drain, canal, underground pipe, or local watercourse.

Proximity to a river reach is a geographical indicator and does not independently establish flood inundation or water-level conditions.

Source: [HydroRIVERS Official Documentation](https://www.hydrosheds.org/products/hydrorivers)

---

## 3.10 Grid-India and Historical Electricity Data
**Data category:** Electricity demand and grid-related information

**Source identified in project records:** POSOCO / Grid-India and associated electricity-sector data sources.

Grid-India operates within India's power-system coordination and grid-management framework.

Clima-Core incorporates historical electricity-system information to study relationships between environmental conditions and electricity demand.

**Relevant information in the project dataset includes**

- Regional peak electricity demand.
- Historical load characteristics.
- Regional electricity-system indicators.
- Available supply and deficit-related information.
**How Clima-Core uses it**

These data support:

- PowerGrid demand prediction.
- Environmental load sensitivity analysis.
- Electricity-demand modelling.
- Deficit classification.
- Investigation of the relationship between thermal stress and electricity-system conditions.
Temperature, humidity, and seasonal conditions can contribute to variations in electricity demand, particularly where cooling requirements are significant.

**Important limitation**

Regional electricity data do not automatically describe individual substations, transformers, distribution feeders, or local grid assets.

The model's geographical and temporal scope is determined by the actual telemetry available in the project registry.

Predicted grid deficits are estimates and must not be interpreted as confirmed grid failures or official power-cut announcements.

---

## 3.11 Agricultural Statistics and Crop Information
**Data category:** Agricultural production, crop suitability, and agro-climatic information

**Sources identified in project records:** ICAR, the Directorate of Economics and Statistics, Ministry of Agriculture and Farmers Welfare, and relevant crop-calendar research resources where used.

Agricultural datasets provide information about crop production, agricultural seasons, historical yield, and geographical variation in cultivation.

The Directorate of Economics and Statistics provides official area, production, and yield reporting at state and district levels.

**Relevant information may include**

- Crop identity.
- Agricultural season.
- Cultivated area.
- Production.
- Crop yield.
- District-level crop statistics.
- Soil and agro-climatic characteristics.
- Sowing and harvesting calendar information.
**How Clima-Core uses it**

Agricultural information supports:

- AgriSense crop suitability modelling.
- Yield anomaly analysis.
- Drought-related agricultural stress assessment.
- Crop classification.
- Agro-climatic feature engineering.
- Seasonal agricultural analysis.
Where historical agricultural data are combined with climate and soil information, the system can investigate relationships between environmental conditions and crop outcomes.

**Important limitation**

Historical crop cultivation does not necessarily represent optimal crop suitability.

Historical yield statistics are influenced by multiple factors, including irrigation, farming practices, input availability, pests, market conditions, and measurement methods.

Model-based crop suggestions should therefore be interpreted as analytical guidance rather than guaranteed recommendations.

Source: [Directorate of Economics and Statistics, Agriculture](https://data.desagri.gov.in/)

---

## 3.12 Central Ground Water Board (CGWB)
**Data category:** Groundwater chemistry and quality observations

**Provider:** Central Ground Water Board, Ministry of Jal Shakti, Government of India.

**Official source:** [https://cgwb.gov.in/en/ground-water-quality](https://cgwb.gov.in/en/ground-water-quality)

CGWB undertakes groundwater sampling and chemical analysis through its regional chemical laboratories.

Its published groundwater-quality information contains records and reports relating to chemical characteristics of groundwater.

**Relevant variables may include**

- pH.
- Electrical Conductivity.
- Fluoride.
- Arsenic.
- Nitrate.
- Iron.
- Total Hardness.
- Other available chemical parameters.
The Clima-Core technical dossier identifies an ingested collection of 18,249 groundwater chemical assays. The active sample count and coverage should be displayed from the latest verified dataset manifest.

**How Clima-Core uses it**

The groundwater information supports:

- WaterQuality model development.
- Contamination screening.
- Groundwater chemical analysis.
- Potability-related assessment.
- Historical geographical comparisons.
- Deterministic water-quality calculations where sufficient laboratory measurements are provided.
**Important limitation**

Groundwater-quality observations are based on specific sample locations and collection dates.

They do not represent continuous, real-time water-quality measurements throughout an entire district or state.

A model-generated potability estimate is not equivalent to laboratory confirmation.

Users should obtain appropriate laboratory testing before making drinking-water safety decisions.

Source: [CGWB Groundwater Quality](https://cgwb.gov.in/en/ground-water-quality)

---

## 3.13 CAMELS-IND v2.2
**Data category:** Hydrological observations and catchment characteristics

**Provider:** Research teams associated with IIT Roorkee and the Indian Institute of Science, Bengaluru.

**Official source:** [https://zenodo.org/records/14999580](https://zenodo.org/records/14999580)

CAMELS-IND stands for Catchment Attributes and MEteorology for Large-sample Studies in India.

It provides hydrometeorological time series and catchment attributes for Indian catchments.

**Dataset characteristics**

- Version: 2.2.
- Publication date: 13 March 2025.
- Geographical coverage: 472 catchments in Peninsular India.
- Historical period: 1980 to 2020.
- Temporal resolution: Daily.
- Observed streamflow available for more than 30% of the period in 242 catchments.
The dataset contains information relating to:

- Observed streamflow.
- Precipitation.
- Air temperature.
- Solar radiation.
- Relative humidity.
- Wind speed.
- Potential evapotranspiration.
- Actual evapotranspiration.
- Soil moisture.
- Topography.
- Climate characteristics.
- Land cover.
- Soil and geology.
- Catchment hydrological attributes.
The collection also includes regionally trained LSTM-predicted streamflow series. These are model-generated products, not independent observed discharge.

**How Clima-Core uses it**

CAMELS-IND is especially important for BasinFlow development.

It supports:

- Gauged catchment modelling.
- Rainfall-runoff model calibration.
- Streamflow prediction.
- Hydrological model comparison.
- Ungauged basin transfer research.
- Evaluation of physical and machine learning-based hydrological methods.
The availability of observed streamflow provides a valuable basis for evaluating hydrological models against measured targets.

**Important limitation**

Not all 472 catchments have equally complete observed streamflow records.

Data availability, measurement quality, geographical coverage, and the selection of catchments for experiments affect the results.

Clima-Core distinguishes observed streamflow from model-generated streamflow and uses independent observations as ground truth for the relevant validation tasks.

**Research acknowledgement**

The dataset should be cited using its associated scientific publication:

Mangukiya, N. K., Kanneganti, B. K., Dey, P., Sharma, S., Bejagam, V., Mujumdar, P. P., and Sharma, A. (2025). CAMELS-IND: hydrometeorological time series and catchment attributes for 228 catchments in Peninsular India. Earth System Science Data, 17, 461–491.

DOI: [https://doi.org/10.5194/essd-17-461-2025](https://doi.org/10.5194/essd-17-461-2025)

Consult the official dataset record and the original data providers' policies for applicable usage, citation, and redistribution requirements.

Source: [CAMELS-IND v2.2 on Zenodo](https://zenodo.org/records/14999580)

---

## 3.14 Open-Meteo Previous Runs and Numerical Weather Prediction
**Data category:** Weather forecast archives

**Service provider:** Open-Meteo.

**Official source:** [https://open-meteo.com/en/docs/previous-runs-api](https://open-meteo.com/en/docs/previous-runs-api)

Clima-Core uses archived numerical weather prediction information to investigate forecast performance and improve forecast processing.

The Previous Runs API provides forecast values from earlier model initialisations at specified lead-time offsets.

Depending on the available archive and the active extraction configuration, these may include forecasts from:

- NOAA Global Forecast System.
- DWD ICON.
- Japan Meteorological Agency models.
- ECMWF forecast products.
- Other supported numerical weather prediction providers.
**Relevant variables may include**

- 2-metre temperature.
- Precipitation.
- Wind speed and components.
- Dew-point temperature.
- Other available atmospheric variables.
**How Clima-Core uses it**

These archived forecasts support ForecastFusion.

They are used to:

- Compare forecast providers.
- Evaluate forecast accuracy at specific lead times.
- Analyse systematic forecast bias.
- Develop ensemble post-processing methods.
- Train probabilistic forecast calibration models.
**Important limitation**

Different forecast systems have different initialization times, spatial resolutions, forecast horizons, and archival periods.

The availability of a model in an API does not establish that its historical forecasts are available for every date.

Previous-run forecasts must be aligned with their actual forecast initialization and valid times when evaluated against observations.

Open-Meteo distributes forecast information produced by the originating weather models. Forecast products should not be described as direct measurements.

Source: [Open-Meteo Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)

---

## 3.15 Central Water Commission (CWC) Reservoir Storage
**Data category:** Reservoir and water-resource observations

**Provider:** Central Water Commission, Ministry of Jal Shakti, Government of India.

**Official source:** [https://nwdp.nwic.gov.in/](https://nwdp.nwic.gov.in/)

Clima-Core's canonical dataset registry includes a CWC reservoir-storage data collection.

The National Water Data Portal provides reservoir information collected through CWC observation systems.

Relevant records may contain:

- Reservoir identification.
- Observation dates.
- Geographical hierarchy.
- Reservoir water levels.
- Storage values.
- Other available station information.
**How Clima-Core uses it**

Where the required records are available and integrated, reservoir information can support:

- Water-resource analysis.
- Historical storage assessment.
- Hydrological context.
- Investigation of seasonal water availability.
- Reservoir-related environmental assessments.
**Important limitation**

Reservoir storage observations describe recorded reservoir conditions at specific observation times.

They should not be interpreted as river discharge, catchment runoff, groundwater availability, or a direct prediction of future reservoir levels.

Coverage and observation frequency depend on the specific reservoir dataset.

Source: [CWC Reservoir Storage Data, National Water Data Portal](https://nwdp.nwic.gov.in/en/dataset/reservoir-water-level-manual-daily-centre-water-commission_cwc)

---

# 4. How Clima-Core Integrates These Sources
Environmental datasets are not used independently in every situation.

Depending on the analytical objective, different sources are combined to construct features, support physical calculations, or evaluate model outputs.

The following examples illustrate how the source families contribute to Clima-Core.

## Flood and Hydrological Intelligence
Relevant information:

- IMD gridded rainfall.
- ERA5-Land soil moisture and atmospheric variables.
- CAMELS-IND observed streamflow and catchment attributes.
- India Flood Inventory v3.
- HydroRIVERS.
- CWC reservoir information, where relevant.
These sources support rainfall analysis, catchment modelling, flood-event classification, and hydrological assessment.

## Heat Intelligence
Relevant information:

- IMD gridded temperature.
- ERA5-Land atmospheric and land-surface variables.
- Historical meteorological inputs.
- Supporting geographical information.
These sources support temperature predictions, heat-related calculations, and regional environmental analysis.

## Air-Quality Intelligence
Relevant information:

- CPCB monitoring observations.
- CAMS atmospheric composition.
- NASA FIRMS active fire observations.
- Weather and wind-related variables.
These sources support pollutant estimation and the investigation of atmospheric conditions associated with smoke and particulate matter.

## Agricultural Intelligence
Relevant information:

- Agricultural production and yield records.
- Soil information.
- Climate and rainfall variables.
- Crop-season information.
- Catchment and geographical characteristics where relevant.
These sources support crop suitability, agricultural stress analysis, and yield-related modelling.

## Water-Quality Intelligence
Relevant information:

- CGWB groundwater chemical observations.
- Available chemical measurements.
- Soil and geographical information.
- Relevant historical environmental variables.
These sources support groundwater quality screening and deterministic water-quality assessments.

## Electricity and Infrastructure Intelligence
Relevant information:

- Historical grid demand and supply indicators.
- Temperature and humidity.
- Calendar and seasonal information.
- OpenStreetMap infrastructure features.
These sources support electricity demand analysis and geographical infrastructure exposure assessment.

## Compound Environmental Risk
Relevant information may be combined from several source families to examine statistical relationships between different environmental conditions.

The variables used depend on the specific compound-risk pathway, available overlapping historical records, and the validity of the underlying statistical assumptions.

---

# 5. Data Processing and Quality Assurance
Raw environmental data cannot always be used directly.

Different providers may publish information using different units, file structures, spatial grids, coordinate systems, timestamps, and measurement conventions.

Clima-Core therefore applies appropriate data preparation and validation procedures before using information in analytical pipelines.

Depending on the dataset, these procedures may include the following.

### 5.1 Format and Schema Validation
Checking file structure, expected fields, variable types, and data compatibility.

### 5.2 Unit Harmonisation
Converting compatible measurements into consistent units before combining or analysing them.

### 5.3 Missing-Value Analysis
Identifying incomplete observations and evaluating whether missing information affects model suitability.

### 5.4 Duplicate Detection
Detecting repeated records and checking their effect on historical analysis.

### 5.5 Temporal Alignment
Ensuring that observations and forecasts correspond to the correct timestamps, dates, and prediction periods.

### 5.6 Geographical Alignment
Checking coordinate systems, geographical identifiers, grid locations, and spatial compatibility.

### 5.7 Feature Engineering
Preparing derived variables such as rainfall accumulations, antecedent moisture indicators, temperature differences, and other task-specific features.

### 5.8 Data Provenance
Recording source information, dataset versions, processing details, and relevant checksums where supported by the project registry.

The exact quality checks depend on the nature and intended use of the source data.

---

# 6. Understanding Data Resolution and Coverage
Every environmental dataset has limitations relating to its spatial resolution, temporal frequency, geographical coverage, and observation period.

These limitations are important when interpreting the outputs generated by Clima-Core.

## Spatial Resolution
Spatial resolution describes the geographical detail represented by a dataset.

For example, a 0.25° rainfall grid and a 1° temperature grid represent different geographical scales.

A finer spatial grid does not automatically guarantee greater measurement accuracy at every location.

## Temporal Resolution
Temporal resolution describes how frequently information is recorded or represented.

Examples include:

- Hourly observations.
- Daily observations.
- Monthly records.
- Seasonal summaries.
Daily rainfall totals cannot independently establish the precise hourly intensity or timing of a rainfall event.

## Historical Coverage
Historical coverage describes the period for which data are available.

Some environmental datasets cover several decades, while others have more limited observation histories.

The amount of historical information available may also differ across geographical locations and variables.

## Geographical Coverage
A national dataset may not contain equally complete observations for every district, city, or monitoring location.

Model coverage depends on the actual records available for the relevant geographical area.

Clima-Core considers these characteristics when selecting inputs and interpreting environmental predictions.

---

# 7. Data Licensing, Attribution and Responsible Use
Clima-Core recognises that environmental datasets are made available under different terms and conditions.

The fact that a dataset is publicly accessible does not automatically mean that it can be redistributed, modified, or used commercially without restrictions.

Different providers may specify requirements relating to:

- Attribution.
- Citation.
- Redistribution.
- Commercial use.
- Modification of derived databases.
- Registration or access permissions.
- Publication of derived products.
Some examples include:

**OpenStreetMap**

OpenStreetMap data is distributed under the Open Data Commons Open Database License (ODbL). Attribution and applicable share-alike obligations must be respected.

**WorldPop**

WorldPop dataset products specify their own usage and citation terms. The applicable licence must be checked for the specific version used.

**CAMELS-IND**

The dataset has an associated scientific publication and source-specific citation guidance. Its dataset record, individual source data policies, and applicable permissions should be checked before redistribution or commercial use.

**IMD, CPCB, CWC, CGWB and other government sources**

Users should follow the relevant source-provider conditions, attribution requirements, access rules, and published data disclaimers.

Clima-Core aims to preserve source attribution and respect applicable data-use conditions.

---

# 8. Data Freshness and Updates
Environmental datasets are updated at different frequencies.

Some sources provide historical archives, some publish periodic updates, and others provide forecast information or monitoring observations at shorter intervals.

For this reason, Clima-Core does not assume that every dataset is continuously updated or that all outputs represent current environmental conditions.

Where available, relevant dataset metadata may include:

- Original source.
- Dataset version.
- Observation period.
- Extraction or download date.
- Last successful update.
- Processing date.
- Geographic coverage.
- Current availability status.
The timestamp associated with an environmental observation must be distinguished from the time at which the dataset was downloaded or processed.

Users should consult the displayed timestamp and source information when interpreting a result.

---

# 9. What Our Data Does Not Represent
To ensure responsible interpretation, it is important to understand what environmental datasets can and cannot establish.

### Historical observations are not future forecasts.
Past rainfall, temperature, and river-flow records describe historical conditions. Predicting future conditions requires an appropriate forecasting or modelling process.

### Reanalysis is not a direct station measurement.
Reanalysis products combine observations with numerical modelling systems and should be interpreted according to their product characteristics.

### Satellite detections are not complete ground observations.
Satellite-derived fire and atmospheric products provide valuable information, but they have sensor-specific limitations.

### Population estimates are not real-time population counts.
Gridded population datasets represent estimated population distributions for specified reference periods.

### River proximity is not proof of flood inundation.
Geographical proximity to a river does not independently establish whether a location will experience flooding.

### Synthetic data are not measured events.
Scenario-generated rainfall profiles and simulated environmental outcomes are possible representations produced by modelling methods.

They must not be mistaken for confirmed observations or forecasts.

### Model estimates are not professional measurements.
Predicted water quality, pollutant concentrations, temperature, or other environmental values do not replace appropriate monitoring or laboratory measurements.

---

# 10. Source Acknowledgement and Research References
Clima-Core acknowledges the government agencies, scientific institutions, researchers, open-data communities, and international organisations whose work contributes to the platform's environmental data foundation.

We encourage interested users and researchers to consult the original sources rather than relying exclusively on processed outputs.

Important research and source references include:

- India Meteorological Department gridded rainfall and temperature documentation.
- Copernicus ERA5-Land documentation.
- Copernicus Atmosphere Monitoring Service documentation.
- Central Pollution Control Board air-quality resources.
- NASA FIRMS data documentation.
- WorldPop dataset documentation.
- OpenStreetMap copyright and licence information.
- HydroSHEDS and HydroRIVERS documentation.
- Central Ground Water Board groundwater-quality resources.
- Directorate of Economics and Statistics agricultural reports.
- Central Water Commission datasets.
- Open-Meteo Previous Runs API documentation.
- CAMELS-IND scientific publication and dataset record.
Dataset-specific citations and acknowledgements should be provided wherever required by the original data providers and research publications.

---

# 11. Data Transparency and Continuous Improvement
Clima-Core treats data quality and source transparency as essential parts of environmental intelligence.

As the platform evolves, new datasets may be evaluated and incorporated where they provide relevant scientific information and meet the requirements of the intended analytical task.

Existing sources may also be updated, replaced, or reprocessed as new versions become available.

Changes to dataset coverage, versions, or processing configurations may influence model behaviour and prediction results.

We therefore aim to maintain a documented relationship between the environmental data used by the platform and the analytical outputs generated from it.

Our commitment is to improve data traceability, recognise the limitations of available information, and use environmental data responsibly.

---

# 12. Explore the Original Sources
Users interested in exploring the underlying environmental information can visit the original providers and research resources referenced throughout this page.

For individual datasets, the relevant source documentation should be treated as the authoritative reference for:

- Original data definitions.
- Measurement methods.
- Dataset versions.
- Access conditions.
- Licensing.
- Attribution.
- Technical limitations.

## City Autocomplete Catalogue
The city search suggestions use a 500-place catalogue derived from the GeoNames India dataset. State labels are mapped from GeoNames administrative-region codes, and Udaipur, Rajasthan is included explicitly.

- **Source:** [GeoNames India extract](https://download.geonames.org/export/dump/IN.zip)
- **Provider:** [GeoNames](https://www.geonames.org/)
- **Licence:** [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

For questions about the use of environmental datasets within Clima-Core, users may contact us through the official contact channel.

**Project:** Clima-Core

**Data Catalogue Version:** 2.1

**Last Updated:** 22 November 2026

**Related Pages:** Methodology | Model Performance | About Us | Privacy Policy

---

**Clima-Core. Intelligence for a changing environment.**