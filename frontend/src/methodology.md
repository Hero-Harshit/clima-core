# Our Methodology

## An Integrated Approach to Environmental Intelligence
Clima-Core combines artificial intelligence, environmental science, statistical modelling, hydrological simulations, and geospatial analysis to develop a comprehensive understanding of environmental conditions and their potential impacts.

Environmental challenges rarely exist in isolation. Extreme temperatures can influence electricity demand, intense rainfall can affect river systems and urban infrastructure, prolonged soil moisture deficits can contribute to agricultural stress, and atmospheric conditions can influence air quality.

Understanding these relationships requires more than analysing individual environmental measurements. It requires an integrated approach that considers observations, predictive models, physical processes, geographical conditions, and uncertainty.

Clima-Core follows a modular four-layer methodology designed to transform diverse environmental datasets into meaningful analytical outputs and decision-support information.

Our methodology is built around five fundamental principles:

- **Scientific Foundation:** Use established scientific equations, statistical techniques, and environmental modelling methods wherever appropriate.
- **Data-Driven Intelligence:** Apply machine learning to identify patterns and relationships within historical and current environmental information.
- **Physical Consistency:** Complement statistical predictions with physical calculations and process-based models.
- **Geographical Context:** Consider the locations, populations, infrastructure, and natural systems potentially affected by environmental conditions.
- **Responsible Interpretation:** Communicate uncertainty, data limitations, and the distinction between observations, predictions, calculations, and simulated scenarios.

---

# 1. The Clima-Core Architecture
Clima-Core operates through four interconnected intelligence layers. Each layer has a distinct responsibility, and the resulting information is combined where the relevant data and analytical methods are available.

## Layer 1: Predictive Intelligence
The Predictive Intelligence Layer uses machine learning and statistical models to analyse environmental information and estimate future or otherwise unobserved conditions.

Environmental datasets often contain complex relationships that cannot be adequately represented through simple thresholds or linear calculations. Machine learning helps identify nonlinear patterns, interactions between variables, seasonal behaviour, and regional differences.

Depending on the prediction task, Clima-Core uses algorithms and modelling techniques such as:

- LightGBM gradient boosting.
- XGBoost.
- CatBoost.
- Ridge regression.
- Natural Gradient Boosting.
- Quantile regression.
- Long Short-Term Memory networks.
- Mass-Conserving LSTM architectures.
- Statistical ensemble models.
- Probabilistic classification and regression.
Different models are assigned to different environmental problems. A model designed to estimate temperature is not automatically suitable for predicting streamflow or air pollution.

### Major predictive systems
**FloodRisk**

FloodRisk analyses rainfall conditions, antecedent moisture, hydrological indicators, and geographical characteristics to estimate the probability of flood-related events.

Its machine learning models incorporate rainfall-derived features and, where supported, physical runoff calculations and river-network information.

The output is an estimated flood occurrence probability, not a guaranteed declaration that flooding will happen at a particular location.

**HeatRisk**

HeatRisk analyses temperature-related and meteorological variables to estimate thermal conditions.

It incorporates LightGBM models, regionalised Mixture of Experts modelling, and probabilistic prediction techniques.

The system considers regional climatic differences rather than assuming that a single national relationship will describe every geographical area equally well.

Its outputs can support further calculations involving thermal stress indicators and human exposure.

**PowerGrid**

PowerGrid analyses historical electricity demand and environmental conditions to estimate electricity demand and identify potential supply-shortfall conditions.

Its predictive features may include temperature, humidity, cooling demand indicators, seasonal variables, calendar information, and historical electricity load.

The system explores the relationship between environmental stress and electricity-system demand. Its predictions should be interpreted within the geographical and historical coverage of the available grid data.

**AgriSense**

AgriSense evaluates relationships between agricultural conditions, climate variables, and soil characteristics.

Its capabilities include crop suitability classification, agricultural stress assessment, yield anomaly modelling, and drought-related crop impact estimation.

The outputs provide analytical support for understanding agro-climatic conditions. They should not be interpreted as guaranteed crop yield or agricultural success predictions.

**AirQuality**

AirQuality uses meteorological conditions, historical monitoring observations, and relevant atmospheric information to estimate pollutant concentrations.

The system includes particulate matter modelling and SmokePath capabilities that investigate the relationship between fire activity, atmospheric movement, and particulate pollution.

Its predictions are subject to the availability and geographical coverage of monitoring data.

**WaterQuality**

WaterQuality combines deterministic water-quality calculations with machine learning-based screening models.

Where sufficient laboratory measurements are provided, the deterministic calculation engine evaluates relevant water-quality indicators.

Machine learning models provide supplementary estimates when working with available historical and environmental features.

Predicted contamination or potability classifications should not be treated as substitutes for laboratory testing.

---

# 2. Layer 2: Physical and Statistical Intelligence
Machine learning can identify patterns from historical data, but environmental systems are governed by physical processes and scientific relationships.

The Physical and Statistical Intelligence Layer complements predictive modelling with scientific equations, process-based calculations, and statistical methods.

This layer helps interpret environmental conditions using established mathematical formulations.

## 2.1 Hydrological Modelling
Hydrological processes describe how rainfall interacts with soil, vegetation, catchments, rivers, and groundwater systems.

Clima-Core incorporates multiple approaches to understand rainfall-runoff behaviour and streamflow.

### GR4J Conceptual Hydrological Model
GR4J is a four-parameter conceptual rainfall-runoff model.

It represents important catchment processes through simplified production and routing mechanisms.

The model uses meteorological inputs, including precipitation and potential evapotranspiration, to simulate catchment streamflow.

Its relatively compact structure makes it suitable for hydrological calibration and comparison across gauged catchments.

### HBV-Light
HBV-Light is a conceptual hydrological modelling approach that represents processes such as:

- Soil moisture storage.
- Snow and rainfall-related processes where applicable.
- Upper groundwater storage.
- Lower groundwater storage.
- Runoff generation.
- Flow routing.
Its structure provides an alternative representation of catchment response and can be compared with other hydrological models.

### Mass-Conserving LSTM Models
Clima-Core also investigates mass-conserving recurrent neural network architectures.

These models incorporate constraints intended to represent the movement and partitioning of water through internal model states.

The dual-state MC-LSTM architecture represents faster runoff-related responses and slower storage-related responses separately.

This provides a different approach to streamflow estimation, particularly when investigating catchments with limited or unavailable streamflow observations.

### Hybrid GR4J Modelling
Clima-Core also explores a hybrid approach combining conceptual hydrological modelling with machine learning-based residual correction.

The conceptual model provides a physical baseline, while the residual learner attempts to capture systematic prediction errors.

The purpose is to investigate whether combining physical structure with data-driven correction can improve streamflow estimates without unnecessarily replacing the underlying hydrological model.

Different models are evaluated for different tasks, including gauged streamflow forecasting, rainfall-runoff simulation, and prediction in ungauged catchments.

## 2.2 Runoff and Flood Hydrology
For rainfall-related assessments, Clima-Core uses hydrological calculations to investigate how precipitation may translate into runoff.

One of the methods used is the Soil Conservation Service Curve Number approach, commonly referred to as SCS-CN.

The method estimates direct runoff based on rainfall and a catchment retention parameter known as the Curve Number.

A simplified formulation is:

[
Q =
\begin{cases}
0, & P \leq I_a \
\frac{(P-I_a)^2}{P-I_a+S}, & P>I_a
\end{cases}
]

Where:

- (Q) is estimated direct runoff.
- (P) is event precipitation.
- (I_a) is initial abstraction.
- (S) is potential maximum retention.
The method can incorporate antecedent moisture conditions and catchment characteristics.

Clima-Core also incorporates unit hydrograph and channel-routing concepts where supported by the available inputs.

These calculations help connect rainfall conditions with potential hydrological responses. They do not independently establish an actual flood extent or inundation depth.

## 2.3 Biometeorological Calculations
Environmental temperature alone does not fully describe the thermal conditions experienced by people.

Humidity, wind, radiation, and other meteorological variables influence how heat is experienced.

Clima-Core incorporates several biometeorological calculations, including:

- Heat Index.
- Universal Thermal Climate Index.
- Wet-bulb temperature estimation.
- Wet-Bulb Globe Temperature estimation.
- Nocturnal heat-related indicators.
These calculations provide additional context for understanding thermal stress.

Different indices describe different physical quantities and are not interchangeable. Their interpretation depends on the inputs, assumptions, and formulations used.

## 2.4 Drought Regime Modelling
Drought is a gradual and evolving environmental process rather than a single-day event.

Clima-Core uses a Hidden Markov Model-based approach to analyse soil moisture conditions and identify possible drought regimes.

The current framework represents four broad states:

1. Equilibrium.
2. Developing Deficit.
3. Severe Drought Cascade.
4. Recovery.
The model uses causal forward filtering to estimate the probability of different hidden environmental states using information available up to the current time.

This approach helps investigate transitions between environmental regimes without requiring future observations to determine the current state.

The resulting state probabilities represent model-estimated conditions and should be interpreted alongside independent drought observations where available.

## 2.5 Extreme Climate and Statistical Modelling
Extreme environmental events require specialised statistical techniques.

Clima-Core investigates extreme-value methods such as:

- Generalized Extreme Value distributions.
- Generalized Pareto distributions.
- Peaks-Over-Threshold analysis.
- Event declustering.
- Stationary and non-stationary extreme-value models.
These methods help analyse the statistical behaviour of unusually high rainfall, temperature, and other environmental variables.

Where adequate historical data exist, non-stationary formulations can investigate whether distribution characteristics change over time.

For example, a time-varying GEV location parameter may be represented as:

[
\mu(t)=\mu_0+\mu_1t
]

This allows the model to investigate a temporal trend in the distribution's location parameter.

However, an observed trend does not automatically establish climate-driven change. Model assumptions, sampling uncertainty, data quality, and statistical significance must all be considered.

Return-period estimates are statistical quantities, not predictions of the exact date when an extreme event will occur.

---

# 3. Layer 3: Exposure and Geospatial Intelligence
Environmental hazards are inherently geographical.

A rainfall event, heatwave, or river-flow increase may have very different consequences depending on the terrain, surrounding population, infrastructure, and environmental characteristics of the affected region.

The Exposure Intelligence Layer combines environmental outputs with geospatial datasets to assess their geographical context.

## 3.1 Terrain Analysis
Terrain characteristics influence surface water movement, drainage behaviour, and geographical susceptibility.

Where suitable elevation data are available, Clima-Core investigates:

- Elevation.
- Slope.
- Aspect.
- Flow direction.
- Flow accumulation.
- Topographic Wetness Index.
The Topographic Wetness Index is commonly expressed as:

[
TWI=\ln\left(\frac{a}{\tan\beta}\right)
]

Where:

- (a) represents the specific contributing area.
- (\beta) represents the local slope angle.
This index provides an indication of the tendency of terrain locations to accumulate water under the assumptions of the underlying terrain model.

It is a terrain-derived indicator, not a direct measurement of soil saturation or flood depth.

## 3.2 River Network Intelligence
Clima-Core incorporates HydroRIVERS river-network information to investigate the spatial relationship between geographical locations and natural river systems.

Relevant characteristics may include:

- Distance to nearby river reaches.
- Stream order.
- River-network structure.
- Catchment relationships.
- Hydrological connectivity where supported by the data.
These features help provide geographical context for flood and river-related assessments.

However, proximity to a natural river does not automatically establish flood exposure. Actual inundation depends on additional factors, including terrain, river discharge, water levels, channel characteristics, and local drainage conditions.

## 3.3 Population Exposure
Clima-Core uses population datasets, including WorldPop, to investigate the relationship between environmental conditions and human settlement patterns.

Population information can help estimate the number of people potentially located within relevant geographical areas.

Depending on the available hazard footprint and population resolution, analysis may include:

- Population distribution.
- Population density.
- Population within selected geographical buffers.
- Population associated with potentially exposed areas.
A population estimate within a selected radius should not be interpreted as the exact number of people affected by an environmental event.

## 3.4 Infrastructure Exposure
Infrastructure networks play an important role in environmental vulnerability.

Where suitable OpenStreetMap information is available, Clima-Core can analyse the geographical distribution of features such as:

- Hospitals.
- Roads.
- Bridges.
- Electrical substations.
- Railway infrastructure.
- Water-related facilities.
- Drainage and canal features.
Spatial analysis can help identify nearby infrastructure and potential geographical relationships with environmental hazards.

The completeness and accuracy of infrastructure analysis depend on the underlying map data.

An infrastructure feature's proximity to a hazard does not necessarily mean that the asset is damaged, inaccessible, or physically exposed.

## 3.5 Geospatial Vulnerability Indicators
Clima-Core also explores combining different geographical characteristics into composite screening indicators.

For example, terrain conditions, hazard estimates, and river proximity may be combined to prioritise areas for further investigation.

Such indicators provide a structured way to compare geographical susceptibility.

Unless independently validated and calibrated against suitable observations, composite scores should be understood as screening indicators rather than direct physical probabilities of damage.

---

# 4. Layer 4: Decision Intelligence
Predictive and physical calculations become more valuable when their outputs can be interpreted in the context of real-world responsibilities.

The Decision Intelligence Layer transforms relevant environmental information into contextual insights for different stakeholder groups.

Clima-Core supports thirteen domain-specific decision personas.

These include:

1. **Disaster Management Official:** Environmental risk assessment, geographical exposure, and preparedness information.
2. **City Commissioner:** Urban environmental conditions, infrastructure exposure, and municipal planning considerations.
3. **Power Grid Dispatcher:** Environmental stress indicators relevant to electricity demand and potential grid shortfalls.
4. **Farmer Advisory:** Agro-climatic conditions, crop suitability, soil moisture, and agricultural stress information.
5. **Heat Health Officer:** Thermal conditions and indicators relevant to public-health preparedness.
6. **Water Supply Engineer:** Water-quality information, groundwater indicators, and relevant hydrological assessments.
7. **Port Authority Officer:** Relevant coastal weather and environmental conditions where supporting data are available.
8. **Construction Manager:** Thermal and meteorological conditions relevant to construction-site planning.
9. **School Administrator:** Environmental conditions relevant to educational facilities and student safety.
10. **Insurance Underwriter:** Statistical extreme-event information and relevant environmental indicators.
11. **Transport Logistics:** Geographical exposure of transport infrastructure to environmental hazards.
12. **Public Health Epidemiologist:** Environmental conditions that may be relevant to public-health surveillance and preparedness.
13. **Citizen Resident:** Understandable environmental information and general awareness guidance.
Each persona is intended to interpret the same underlying evidence according to its relevant operational context.

The decision layer is designed to support informed human decisions. It does not independently issue official emergency declarations, evacuation orders, school closures, infrastructure shutdowns, or regulatory instructions.

---

# 5. The End-to-End Prediction Process
Clima-Core uses an ordered, modular workflow to transform input information into environmental intelligence.

Not every assessment requires every model or every layer. The specific processing path depends on the requested environmental analysis, available data, geographical coverage, and supported model inputs.

The general workflow consists of the following stages.

## Step 1: Data Collection and Input Identification
The process begins by identifying the relevant geographical location, environmental variables, and requested assessment.

Input information may come from:

- User-selected geographical locations.
- Historical environmental datasets.
- Weather forecast products.
- Environmental monitoring observations.
- Satellite and atmospheric products.
- Geospatial datasets.
Every input has a source, temporal context, spatial resolution, and set of limitations.

## Step 2: Data Validation and Preprocessing
Before model inference, relevant information is checked and prepared.

Depending on the dataset, this may involve:

- Unit harmonisation.
- Date and timestamp alignment.
- Missing-value handling.
- Duplicate detection.
- Geographic coordinate processing.
- Feature engineering.
- Temporal aggregation.
- Spatial alignment.
- Historical feature construction.
For time-dependent models, input features must respect the information that would actually have been available at the prediction time.

Data from future dates must not be used to construct historical predictions.

## Step 3: Predictive Model Inference
The relevant predictive model or models process the prepared input features.

The selected models depend on the environmental variable being evaluated.

Examples include:

- FloodRisk for flood-event probability estimation.
- HeatRisk for temperature-related predictions.
- AirQuality for pollutant concentration estimation.
- ForecastFusion for weather forecast post-processing.
- PowerGrid for electricity demand and deficit assessment.
- AgriSense for crop-related predictions.
- BasinFlow for hydrological modelling.
Model outputs may be point estimates, probability scores, predictive distributions, or other task-specific results.

## Step 4: Physical and Statistical Analysis
Where relevant, model outputs and input variables are processed through physical and statistical engines.

Examples include:

- Rainfall-runoff calculations.
- Catchment streamflow simulations.
- Biometeorological indices.
- Drought-regime probabilities.
- Extreme-event analysis.
- Compound hazard dependence modelling.
- Water-quality calculations.
This stage provides additional scientific context and may produce results that are distinct from the machine learning predictions.

## Step 5: Geospatial Exposure Assessment
Relevant environmental outputs are combined with geographical information.

The system may analyse:

- Terrain characteristics.
- River-network relationships.
- Population distribution.
- Infrastructure proximity.
- Catchment boundaries.
- Geographic exposure indicators.
The resulting information helps describe where potential environmental impacts may be relevant.

The accuracy of this stage depends on the resolution and completeness of the geographical datasets.

## Step 6: Uncertainty Assessment
Environmental predictions are not perfectly certain.

Clima-Core uses suitable probabilistic and calibration methods to communicate uncertainty where the model supports it.

Depending on the prediction task, this may involve:

- NGBoost predictive distributions.
- Quantile regression.
- Ensemble forecast distributions.
- Conformal prediction intervals.
- Adaptive conformal calibration.
- Classification probability calibration.
- Monte Carlo scenario analysis.
The uncertainty method is selected according to the type of output and the available validation evidence.

Not every model produces the same type of uncertainty information.

## Step 7: Decision Support and Result Generation
Finally, the relevant results are organised into interpretable information.

Depending on the requested functionality, the output may include:

- Environmental predictions.
- Risk estimates.
- Geospatial exposure information.
- Statistical indicators.
- Predictive uncertainty.
- Model explanations.
- Persona-specific guidance.
The final output should preserve its data provenance and should distinguish predictions from calculations and scenarios.

---

# 6. Weather Forecast Processing
Clima-Core incorporates ForecastFusion to investigate the combination and post-processing of numerical weather prediction information.

Numerical weather prediction systems use mathematical representations of atmospheric processes to estimate future weather conditions.

ForecastFusion processes available forecast information from providers such as:

- NOAA Global Forecast System.
- DWD ICON.
- Japan Meteorological Agency forecast products.
- Other supported forecast sources.
Where valid independent verification observations are available, the system can evaluate and calibrate forecast outputs.

## 6.1 Ensemble Model Output Statistics
Extended Ensemble Model Output Statistics, or EMOS, is used to investigate systematic forecast biases and uncertainty.

Rather than assuming that a raw forecast is perfectly calibrated, EMOS estimates a predictive distribution using forecast information and observed historical relationships.

A simplified model structure can be expressed as:

\alpha+\sum_{m=1}^{M}\beta_mT_{\text{raw},m}
]

Where:

- (\hat T_{\text{corr}}) is the corrected temperature estimate.
- (T_{\text{raw},m}) is the forecast from provider (m).
- (\alpha) is an estimated intercept.
- (\beta_m) represents the contribution of the relevant provider.
The actual model formulation may include additional spread or distributional terms depending on the forecasting configuration.

## 6.2 Precipitation Forecasting
Precipitation forecasting involves two related problems:

1. Estimating whether rainfall will occur.
2. Estimating how much rainfall may occur, conditional on rainfall.
Clima-Core uses a two-part hurdle modelling approach to distinguish these components.

The occurrence model estimates rainfall probability, while the amount model estimates conditional rainfall depth.

This structure recognises that rainfall occurrence and rainfall magnitude have different statistical characteristics.

## 6.3 Forecast Verification
Forecasts are evaluated against appropriately aligned observations.

Important considerations include:

- Forecast initialization time.
- Valid time.
- Prediction lead time.
- Geographical location.
- Observation quality.
- Missing data.
- Independent verification periods.
A forecast generated for a particular valid date must not be evaluated against an incorrectly aligned observation.

Forecast performance is assessed using suitable deterministic and probabilistic metrics.

---

# 7. Sub-Daily Rainfall and Scenario Modelling
Daily rainfall totals do not reveal precisely how precipitation was distributed across individual hours.

For example, the same daily rainfall total could result from relatively continuous rainfall or a short period of intense precipitation.

These two rainfall patterns can have different hydrological implications.

Clima-Core incorporates sub-daily rainfall disaggregation methods to investigate plausible hourly rainfall distributions when appropriate daily input data are available.

The current methodology includes:

- Huff rainfall temporal distributions.
- Method of Fragments.
These methods can generate synthetic temporal rainfall profiles that preserve the supplied daily rainfall volume.

The basic volume-preservation requirement is:

[
\sum_{i=1}^{24}h_i=P_{\text{daily}}
]

Where:

- (h_i) represents rainfall assigned to hour (i).
- (P_{\text{daily}}) is the original daily rainfall total.
Generated rainfall profiles are scenarios and are not equivalent to measured hourly observations.

Where authentic hourly or sub-hourly observations are available, they should be preferred for event verification and calibration.

Synthetic rainfall scenarios may support exploratory runoff analysis and sensitivity assessments, but they must not be presented as confirmed reconstructions of historical cloudbursts or as actual hourly weather forecasts.

---

# 8. Compound Environmental Risk Analysis
Environmental hazards may occur together or influence one another.

For instance:

- Heavy rainfall can coincide with high antecedent soil moisture and elevated river discharge.
- Extreme heat can be associated with increased electricity demand.
- Persistent dry conditions can occur alongside elevated vegetation and fire-related risks.
Analysing these variables independently may overlook important dependencies.

Clima-Core uses statistical dependence modelling to investigate relationships between environmental variables.

## Copula-Based Analysis
A copula is a statistical function that describes the dependence structure between random variables separately from their marginal distributions.

This makes it possible to study the probability of multiple environmental conditions occurring together.

Clima-Core's CompoundRisk engine investigates Archimedean copulas and regular vine copula structures.

The vine approach uses combinations of bivariate and conditional dependence functions to represent more flexible multivariable relationships.

Potential applications include:

- Deluge and hydrological conditions.
- Thermal and electricity-system stress.
- Drought-related environmental conditions.
- Other supported compound hazard combinations.
The output can include estimated joint probabilities and return-period information under the selected model assumptions.

Copula analysis describes statistical dependence. It does not independently prove that one environmental variable causes another or that a complete infrastructure failure pathway will occur.

Joint-risk estimates are sensitive to sample size, historical coverage, variable selection, marginal distributions, and model assumptions.

---

# 9. Uncertainty Estimation and Calibration
Environmental models cannot eliminate uncertainty.

Predictions may be affected by incomplete observations, changing weather patterns, measurement errors, model assumptions, and geographical limitations.

Clima-Core investigates uncertainty using several complementary methods.

## 9.1 Probabilistic Prediction
Some models estimate distributions rather than producing only one predicted value.

For example, a predictive temperature model may provide a distribution represented by a mean and standard deviation.

Such outputs offer additional information about the range of plausible outcomes under the model.

## 9.2 Quantile Prediction
Quantile regression estimates different points of a predictive distribution.

For example:

- Lower quantile.
- Median estimate.
- Upper quantile.
These values can be used to construct prediction intervals and communicate potential variability.

## 9.3 Conformal Prediction
Conformal prediction uses calibration observations to construct prediction intervals or sets.

It can provide useful distribution-free marginal coverage guarantees under the assumptions of the selected method, including exchangeability for classical split conformal prediction.

## 9.4 Adaptive Calibration
Environmental distributions may change over time.

Clima-Core investigates adaptive conformal methods that update calibration using newly available observations.

Rolling calibration and adaptive threshold methods can help the system respond to changing forecast errors.

However, adaptive calibration does not guarantee accurate conditional coverage for every location, environmental regime, or extreme event.

Prediction intervals must be interpreted alongside their historical validation coverage and width.

## 9.5 Classification Calibration
For classification tasks, predicted probabilities can be evaluated and adjusted using calibration methods such as:

- Platt scaling.
- Isotonic regression.
Calibration diagnostics help investigate whether predictions assigned a particular probability correspond to approximately similar observed event frequencies over the relevant evaluation population.

Uncertainty methods are selected according to the output type. Regression intervals and classification probabilities are not interchangeable.

---

# 10. Model Explainability
Clima-Core aims to make predictive outputs easier to interpret.

For supported tree-based models, feature-attribution approaches such as SHAP can be used to examine how individual features contribute to a prediction relative to a model's baseline output.

For example, a flood-risk model may be influenced by variables associated with:

- Recent rainfall.
- Antecedent precipitation.
- Soil moisture.
- Catchment characteristics.
- River-network features.
Feature attribution helps explain the behaviour of the model.

It does not establish that an individual feature caused the environmental event.

Explainability results must be interpreted within the model's feature definitions, training data, and assumptions.

---

# 11. Data Sources and Processing
Clima-Core combines environmental datasets from scientific institutions, government agencies, research projects, and other recognised data providers.

Its data foundation includes information relating to:

- Meteorological observations.
- Numerical weather prediction.
- Atmospheric composition.
- Air-quality monitoring.
- Hydrological observations.
- River networks.
- Soil and agricultural conditions.
- Population distribution.
- Infrastructure.
- Groundwater quality.
- Electricity demand.
Examples of data sources used within the platform include:

- India Meteorological Department.
- ERA5-Land.
- Copernicus Atmospheric Monitoring Service.
- Central Pollution Control Board.
- NASA satellite observation products.
- WorldPop.
- OpenStreetMap.
- HydroSHEDS and HydroRIVERS.
- CAMELS-IND.
- Central Ground Water Board.
- Grid-related historical datasets.
The exact dataset used depends on the environmental problem and geographical coverage.

## 11.1 Data Quality
Before environmental information is incorporated into analytical pipelines, relevant quality checks are applied.

These may include:

- Missing-value analysis.
- Duplicate detection.
- Unit verification.
- Timestamp consistency.
- Spatial alignment.
- Data range checks.
- Source traceability.
Historical datasets may have different spatial resolutions, temporal coverage, and measurement characteristics.

Combining datasets requires appropriate alignment and consideration of those differences.

## 11.2 Geographic and Temporal Resolution
Environmental predictions cannot provide greater observational detail than the underlying evidence justifies.

For example, coarse-resolution meteorological data should not automatically be interpreted as precise neighbourhood-level measurements.

Likewise, a daily rainfall dataset cannot establish the actual hourly timing of a cloudburst.

Clima-Core considers the spatial and temporal resolution of the input datasets when interpreting analytical outputs.

Further information about individual data providers, dataset coverage, and applicable attribution requirements is available through the Data Sources section.

---

# 12. Model Training and Scientific Validation
Training a model is only one step in developing a reliable predictive system.

A model that performs well on its training data may not necessarily generalise to new dates, cities, districts, or environmental conditions.

For this reason, Clima-Core evaluates models using appropriate validation strategies.

## 12.1 Training and Validation
Depending on the problem, the evaluation methodology may include:

- Training datasets.
- Validation datasets.
- Independent chronological test periods.
- Rolling-origin evaluation.
- Geographic holdout evaluation.
- Catchment-level holdout evaluation.
- Monitoring-station holdout evaluation.
- Event-based evaluation.
For time-dependent environmental modelling, historical information must be separated chronologically where appropriate.

For spatial generalisation, entire locations or catchments may be held out to investigate how models perform in areas not represented during training.

## 12.2 Performance Metrics
Different environmental tasks require different evaluation metrics.

Some of the metrics used across Clima-Core include:

**Regression Metrics**

- Mean Absolute Error.
- Root Mean Squared Error.
- Coefficient of Determination.
- Bias.
**Classification Metrics**

- Accuracy.
- Precision.
- Recall.
- F1 Score.
- Receiver Operating Characteristic Area Under the Curve.
- Precision-Recall Area Under the Curve.
- Brier Score.
**Hydrological Metrics**

- Nash-Sutcliffe Efficiency.
- Kling-Gupta Efficiency.
- Logarithmic NSE.
- High-flow bias.
- Low-flow bias.
- Water-balance error.
**Probabilistic Metrics**

- Continuous Ranked Probability Score.
- Brier Score.
- Prediction interval coverage.
- Prediction interval width.
- Reliability diagnostics.
A metric is selected according to the scientific objective of the model. A higher value is not necessarily better for every metric.

## 12.3 Baseline Comparison
New algorithms are compared against relevant existing methods and simpler baselines.

A more complex model is not automatically promoted simply because it achieves a better result on one evaluation metric.

Model selection considers predictive performance, generalisation, calibration, physical consistency, resource requirements, and uncertainty.

Rejected experiments remain useful for understanding the limitations of different modelling approaches.

## 12.4 Reproducibility
Clima-Core follows reproducible experimentation practices where applicable.

Model development records may include:

- Dataset version.
- Training period.
- Validation period.
- Test period.
- Feature configuration.
- Model parameters.
- Random seeds.
- Evaluation metrics.
- Model artifact version.
- Computational requirements.
Performance figures published on the platform should be associated with their relevant model version and evaluation methodology.

Historical benchmark results should not be interpreted as guarantees of future performance.

---

# 13. Responsible Interpretation of Environmental Outputs
Clima-Core is designed to support environmental awareness, research, and informed decision-making.

However, environmental analysis has important limitations.

To make these limitations clear, the platform distinguishes between different types of outputs.

### Measured
Information obtained directly from an observation or monitoring source.

Examples include recorded rainfall, measured temperature, and laboratory water-quality measurements.

### Predicted
An estimate produced by a predictive model using available input information.

Examples include predicted temperature, estimated pollutant concentration, and flood-event probability.

### Calculated
An output generated using a specified scientific equation or deterministic calculation.

Examples include hydrological indices, thermal indices, and deterministic water-quality calculations.

### Scenario
A hypothetical result produced through a simulation, stochastic generation process, or what-if analysis.

Examples include synthetic hourly rainfall distributions and simulated hydrological responses.

### Conceptual
A proposed environmental relationship or risk pathway that has not been sufficiently established through independent empirical validation.

These categories help users understand the nature and evidential status of each result.

---

# 14. Limitations and Important Considerations
Although Clima-Core integrates multiple analytical approaches, its results remain subject to the limitations of environmental data and modelling.

Important considerations include:

- Historical datasets may contain missing values or measurement limitations.
- Spatial datasets may not represent every location at the same resolution.
- Environmental conditions can change over time.
- Different models may produce different estimates.
- Predictive uncertainty can increase under unusual or previously unseen conditions.
- Synthetic scenarios do not establish that a particular event will occur.
- Statistical associations do not automatically establish physical causation.
- Geospatial proximity does not necessarily establish actual hazard exposure.
- Model performance may vary across regions, seasons, and environmental regimes.
Where data are insufficient, an assessment may be incomplete or unavailable.

Clima-Core aims to communicate relevant limitations rather than imply that every output has the same level of certainty.

---

# 15. Responsible Use and Decision Support
Clima-Core provides analytical information to support environmental understanding and decision-making.

It is not intended to replace:

- Official weather forecasts and warnings.
- Government disaster-management communications.
- Emergency-response instructions.
- Professional hydrological assessments.
- Accredited environmental laboratory testing.
- Medical or public-health advice.
- Regulatory or infrastructure safety assessments.
Users should consult the relevant official authorities and qualified professionals before making decisions that involve public safety, critical infrastructure, health, or emergency response.

Automated outputs should be interpreted alongside their source information, uncertainty, and limitations.

---

# 16. Continuous Research and Improvement
Environmental intelligence is an evolving field.

New observations, scientific research, improved algorithms, and changing environmental conditions create opportunities to improve predictive systems.

Clima-Core follows an iterative development approach involving:

1. Identifying scientific and operational limitations.
2. Investigating appropriate data sources and methods.
3. Developing candidate models or calculations.
4. Evaluating them against appropriate benchmarks.
5. Comparing results with existing implementations.
6. Reviewing uncertainty and physical consistency.
7. Integrating validated improvements.
8. Documenting changes and known limitations.
New methods are not automatically adopted because they are more complex or recently developed.

They are evaluated based on their relevance, reliability, computational feasibility, and demonstrated contribution to the platform.

---

# 17. Our Methodological Commitment
At Clima-Core, we believe that reliable environmental intelligence requires a balance between data-driven prediction, scientific modelling, geographical context, and responsible interpretation.

Machine learning provides powerful analytical capabilities, but it is not a replacement for physical understanding.

Physical models provide scientific structure, but they also depend on assumptions and available observations.

Geospatial analysis provides essential context, but its usefulness depends on the quality and resolution of geographic data.

Uncertainty estimation helps communicate limitations, but it does not eliminate the uncertainty inherent in environmental systems.

Our methodology brings these complementary disciplines together within a unified framework.

**Our objective is not simply to generate environmental predictions. It is to make those predictions more understandable, scientifically grounded, geographically relevant, and useful for informed decision-making.**

**Clima-Core. Intelligence for a changing environment.**

---

**Methodology Version:** 2.1
**Last Updated:** [Insert Date]
**Related Pages:** Data Sources | Model Performance | About Us