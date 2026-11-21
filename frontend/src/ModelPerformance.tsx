import {
  Activity,
  BadgeCheck,
  ChevronDown,
  Clock3,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  Layers3,
  ShieldCheck,
} from "lucide-react";

type ModelRecord = {
  name: string;
  version: string;
  algorithm: string;
  features: string;
  target: string;
  primary: string;
  secondary: string;
  training: string;
  memory: string;
  status: string;
  statusKind: "champion" | "challenger" | "benchmark";
};

const models: ModelRecord[] = [
  {
    name: "FloodRisk V3",
    version: "flood_risk_challenger_scs_cn",
    algorithm: "LightGBM GBDT + USDA NRCS SCS-CN",
    features: "23",
    target: "District flood disaster occurrence (binary)",
    primary: "ROC-AUC 0.8011",
    secondary: "PR-AUC 0.1224 · Brier 0.1354 · Recall@0.35 73.2%",
    training: "8.09 s",
    memory: "280.2 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "FloodRisk XGBoost",
    version: "flood_risk_xgboost_23feat",
    algorithm: "XGBoost Hist + Optuna HPO",
    features: "23",
    target: "District flood disaster occurrence (binary)",
    primary: "ROC-AUC 0.8014",
    secondary: "PR-AUC 0.1077 · Brier 0.1374",
    training: "8.10 s",
    memory: "280.2 MB",
    status: "Challenger",
    statusKind: "challenger",
  },
  {
    name: "HeatRisk V4",
    version: "heat_risk_v3",
    algorithm: "LightGBM Regressor + Conformal",
    features: "13",
    target: "Next-day maximum temperature (°C)",
    primary: "R² 0.9510",
    secondary: "MAE 1.023 °C · RMSE 1.417 °C · MedAE 0.761 °C",
    training: "8.40 s",
    memory: "271.4 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "HeatRisk Soft MoE",
    version: "heat_risk_v5_moe",
    algorithm: "Soft-gated regularized MoE (5 experts)",
    features: "7",
    target: "Next-day maximum temperature (°C)",
    primary: "R² 0.9511",
    secondary: "MAE 1.023 °C · RMSE 1.416 °C · λ shrinkage 0.60",
    training: "5.30 s",
    memory: "342.7 MB",
    status: "Co-champion",
    statusKind: "champion",
  },
  {
    name: "HeatRisk Quantiles",
    version: "heat_risk_quantile_suite",
    algorithm: "LightGBM Pinball Loss (Q10, Q50, Q90)",
    features: "6",
    target: "Marginal prediction interval (Q10 → Q90)",
    primary: "Coverage 76.82% (80% nominal)",
    secondary: "Average width 3.01 °C · Crossing rate 0.00%",
    training: "15.18 s",
    memory: "275.8 MB",
    status: "Champion · uncertainty",
    statusKind: "champion",
  },
  {
    name: "ForecastFusion EMOS",
    version: "forecast_fusion_v1_emos",
    algorithm: "Ensemble Model Output Statistics",
    features: "4 NWP sources",
    target: "24-hour calibrated maximum temperature (°C)",
    primary: "CRPS 0.4764",
    secondary: "MAE 0.624 °C · 21% lower error vs raw GFS/ICON · Coverage 95.56%",
    training: "3.20 s",
    memory: "210.5 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "PowerGrid Load",
    version: "power_grid_v2_reg",
    algorithm: "LightGBM Regressor",
    features: "12",
    target: "Peak transmission load (MW)",
    primary: "R² 0.8758",
    secondary: "MAE 1,519.2 MW · RMSE 2,336.2 MW · MAPE 18.02%",
    training: "2.53 s",
    memory: "267.8 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "PowerGrid Deficit",
    version: "power_grid_v2_clf",
    algorithm: "LightGBM Binary Classifier",
    features: "8",
    target: "Grid deficit day (unmet demand > 0)",
    primary: "ROC-AUC 0.9521",
    secondary: "Recall 96.41% · Precision 88.42% · Brier 0.0412",
    training: "4.52 s",
    memory: "289.7 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "PowerGrid TCN",
    version: "power_grid_tcn_v4",
    algorithm: "PyTorch 1D causal dilated CNN (CPU)",
    features: "12 · 30-day lookback",
    target: "Peak transmission load (MW)",
    primary: "R² 0.5385",
    secondary: "15,545 parameters · 31-day receptive field",
    training: "444.5 s",
    memory: "843.5 MB",
    status: "Benchmark",
    statusKind: "benchmark",
  },
  {
    name: "AirQuality PM2.5",
    version: "air_quality_v2_pm25",
    algorithm: "LightGBM Regressor + inversion physics",
    features: "12",
    target: "Next-day surface PM2.5 (µg/m³)",
    primary: "R² 0.8725",
    secondary: "MAE 6.38 µg/m³ · RMSE 9.63 µg/m³ · CPCB accuracy 80.74%",
    training: "6.26 s",
    memory: "360.7 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "SmokePath V2",
    version: "smokepath_v2_airquality",
    algorithm: "LightGBM + satellite FRP advection",
    features: "7",
    target: "Stagnation-adjusted PM2.5 (µg/m³)",
    primary: "R² 0.8738",
    secondary: "MAE 6.34 µg/m³ · RMSE 9.58 µg/m³",
    training: "7.15 s",
    memory: "352.1 MB",
    status: "Challenger",
    statusKind: "challenger",
  },
  {
    name: "AgriSense V2",
    version: "agrisense_v2",
    algorithm: "Multi-class LightGBM GBDT",
    features: "19",
    target: "Dominant optimal crop (1 of 12)",
    primary: "Top-1 accuracy 82.49%",
    secondary: "Top-3 accuracy 97.62% · Weighted F1 0.8250",
    training: "14.20 s",
    memory: "312.4 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "DroughtCascade V2",
    version: "drought_cascade_v2",
    algorithm: "Gaussian HMM (causal forward filter)",
    features: "4",
    target: "Four latent soil drought regimes",
    primary: "Causal agreement 93.93%",
    secondary: "Severe drought F1 0.9129 · Recall 90.18%",
    training: "23.50 s",
    memory: "231.5 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "BasinFlow MC-LSTM",
    version: "basinflow_v2_mc_lstm",
    algorithm: "Mass-conserving deep LSTM (PyTorch)",
    features: "3 hydrological",
    target: "Daily river discharge Q (mm/day)",
    primary: "Median KGE 0.589",
    secondary: "Median NSE 0.563 · Mass error < 0.001%",
    training: "185.2 s",
    memory: "468.2 MB",
    status: "Champion · deep hydro",
    statusKind: "champion",
  },
  {
    name: "BasinFlow Hybrid-GR4J",
    version: "basinflow_v2_gr4j",
    algorithm: "Physical GR4J + ridge residual learner",
    features: "2 physical",
    target: "Daily river discharge Q (mm/day)",
    primary: "Median KGE 0.5003",
    secondary: "Median NSE 0.4913 · Peak error −0.45%",
    training: "12.40 s",
    memory: "195.0 MB",
    status: "Champion · conceptual",
    statusKind: "champion",
  },
  {
    name: "WaterQuality WQI",
    version: "water_quality_wqi_regressor_v1",
    algorithm: "LightGBM Regressor (BIS IS 10500)",
    features: "5 chemical ions",
    target: "Water Quality Index (0–100)",
    primary: "R² 0.9410",
    secondary: "MAE 3.82 points · RMSE 5.14 points",
    training: "1.85 s",
    memory: "182.0 MB",
    status: "Champion",
    statusKind: "champion",
  },
  {
    name: "WaterQuality Hazard",
    version: "water_quality_hazard_v1",
    algorithm: "LightGBM Binary Classifier",
    features: "5 chemical ions",
    target: "Potable safety flag (binary)",
    primary: "ROC-AUC 0.9620",
    secondary: "PR-AUC 0.9480 · F1 0.8950",
    training: "1.72 s",
    memory: "179.5 MB",
    status: "Champion",
    statusKind: "champion",
  },
];

const floodFeatures = [
  {
    label: "Rainfall antecedents",
    values: ["rainfall_mm", "rain_rolling_3d", "rain_rolling_7d", "rain_rolling_14d", "rain_rolling_30d", "rain_max_3d", "api_14d", "precipitation_hours"],
  },
  {
    label: "Soil moisture & evaporation",
    values: ["et0_fao_evapotranspiration", "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean", "topsoil_moisture_anomaly", "rootzone_moisture_anomaly", "soil_moisture_deficit_index"],
  },
  {
    label: "Civil exposure & bottlenecks",
    values: ["drainage_canals_15km", "bridges_15km", "transit_stations_15km", "population_15km", "dense_cluster_pop_15km", "wind_gusts_10m_max"],
  },
  {
    label: "Hydrological runoff",
    values: ["scs_runoff_depth_mm", "scs_runoff_coefficient", "scs_potential_retention_s"],
  },
];

const heatFeatures = [
  "temperature_2m_max_lag1",
  "temperature_2m_min_lag1",
  "apparent_temp_max_lag1",
  "relative_humidity_lag1",
  "surface_solar_radiation_sum",
  "wind_speed_10m_max",
  "urban_canopy_fraction",
  "elevation_m",
  "day_of_year_sin",
  "day_of_year_cos",
  "surface_thermal_inertia",
  "bioclimatic_zone_id",
  "nocturnal_uhi_intensity_c",
];

const powerFeatures = [
  "temp_max_c",
  "temp_min_c",
  "apparent_temp_max_c",
  "wet_bulb_temp_c",
  "cooling_degree_days_base22",
  "load_lag_24h_mw",
  "load_lag_168h_mw",
  "is_weekend",
  "is_statutory_holiday",
  "heavy_industrial_curfew_flag",
  "regional_wind_solar_mw",
  "transmission_capacity_mw",
];

const airQualityFeatures = [
  "pm2_5_lag1",
  "pm2_5_rolling_3d",
  "temp_max_c",
  "temp_min_c",
  "wind_speed_kmh",
  "planetary_boundary_layer_height_m",
  "ventilation_index_m2_s",
  "relative_humidity_pct",
  "thermal_inversion_flag",
  "satellite_frp_advection_mw",
  "wind_u_vector",
  "wind_v_vector",
];

function FeatureTags({ features }: { features: string[] }) {
  return <div className="model-perf-tags">{features.map((feature) => <code key={feature}>{feature}</code>)}</div>;
}

function FeatureGroups({ groups }: { groups: { label: string; values: string[] }[] }) {
  return (
    <div className="model-perf-feature-groups">
      {groups.map((group) => (
        <div className="model-perf-feature-group" key={group.label}>
          <strong>{group.label}</strong>
          <FeatureTags features={group.values} />
        </div>
      ))}
    </div>
  );
}

function ModelPerformance() {
  return (
    <div className="model-performance">
      <div className="model-perf-intro">
        <div>
          <span className="model-perf-eyebrow"><Activity size={13} /> VALIDATION SNAPSHOT</span>
          <h3>Built for reliable, efficient inference.</h3>
          <p>Evaluation results across ClimaCore’s environmental risk models, reported against held-out data.</p>
        </div>
        <span className="model-perf-snapshot"><ShieldCheck size={14} /> Zero-leakage holdouts</span>
      </div>

      <section className="model-perf-summary" aria-label="System performance summary">
        <article className="model-perf-summary-card">
          <span className="model-perf-summary-icon"><Layers3 size={17} /></span>
          <span className="model-perf-summary-label">Models in registry</span>
          <strong>24</strong>
          <small>Canonical production models</small>
        </article>
        <article className="model-perf-summary-card">
          <span className="model-perf-summary-icon"><Gauge size={17} /></span>
          <span className="model-perf-summary-label">Warm inference</span>
          <strong>&lt; 0.85 <i>ms</i></strong>
          <small>Per query on CPU</small>
        </article>
        <article className="model-perf-summary-card">
          <span className="model-perf-summary-icon"><Cpu size={17} /></span>
          <span className="model-perf-summary-label">Compute mode</span>
          <strong>100%</strong>
          <small>CPU · 0 CUDA calls</small>
        </article>
        <article className="model-perf-summary-card">
          <span className="model-perf-summary-icon"><HardDrive size={17} /></span>
          <span className="model-perf-summary-label">Peak training memory</span>
          <strong>843.48 <i>MB</i></strong>
          <small>Under the 5 GB ceiling</small>
        </article>
      </section>

      <section className="model-perf-section">
        <div className="model-perf-section-heading">
          <div>
            <span className="model-perf-kicker">EVALUATION SCOPE</span>
            <h4>Zero-leakage test holdouts</h4>
          </div>
          <span className="model-perf-subtle"><Database size={14} /> Independent evaluation samples</span>
        </div>
        <div className="model-perf-holdouts">
          <div><strong>36,500</strong><span>Flood samples</span></div>
          <div><strong>15,827</strong><span>Power-grid samples</span></div>
          <div><strong>15,100</strong><span>Weather &amp; air quality</span></div>
          <div><strong>230,118</strong><span>Drought days</span></div>
          <div><strong>24,200</strong><span>BasinFlow samples</span></div>
          <div><strong>3,650</strong><span>Water quality samples</span></div>
        </div>
      </section>

      <section className="model-perf-section">
        <div className="model-perf-section-heading">
          <div>
            <span className="model-perf-kicker">MODEL REGISTRY</span>
            <h4>Performance matrix</h4>
          </div>
          <span className="model-perf-count"><BadgeCheck size={14} /> 17 reported model records</span>
        </div>
        <div className="model-perf-table-note">24 models are registered; the supplied benchmark details below cover 17 of them.</div>
        <div className="model-perf-table-wrap" role="region" aria-label="Model performance matrix" tabIndex={0}>
          <table className="model-perf-table">
            <thead>
              <tr>
                <th scope="col">Model &amp; version</th>
                <th scope="col">Algorithm</th>
                <th scope="col">Features</th>
                <th scope="col">Target variable</th>
                <th scope="col">Validation metrics</th>
                <th scope="col">Training</th>
                <th scope="col">Peak RAM</th>
                <th scope="col">Status</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => (
                <tr key={model.version}>
                  <td className="model-perf-name">
                    <strong>{model.name}</strong>
                    <code>{model.version}</code>
                  </td>
                  <td>{model.algorithm}</td>
                  <td>{model.features}</td>
                  <td>{model.target}</td>
                  <td className="model-perf-metrics">
                    <strong>{model.primary}</strong>
                    <span>{model.secondary}</span>
                  </td>
                  <td className="model-perf-nowrap">{model.training}</td>
                  <td className="model-perf-nowrap">{model.memory}</td>
                  <td><span className={`model-perf-status ${model.statusKind}`}>{model.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="model-perf-section model-perf-deep-dive">
        <div className="model-perf-section-heading">
          <div>
            <span className="model-perf-kicker">UNDER THE HOOD</span>
            <h4>Feature breakdown by domain</h4>
          </div>
          <span className="model-perf-subtle"><ChevronDown size={14} /> Expand a domain</span>
        </div>

        <details className="model-perf-domain" open>
          <summary>
            <span className="model-perf-domain-icon flood"><Database size={16} /></span>
            <span><strong>FloodRisk V3</strong><small>Pluvial &amp; fluvial hydrology · 23 features</small></span>
            <ChevronDown className="model-perf-chevron" size={17} />
          </summary>
          <div className="model-perf-domain-body">
            <FeatureGroups groups={floodFeatures} />
            <div className="model-perf-output"><strong>Output</strong><span>District-level flood-disaster probability [0, 1] and calibrated binary breach alert.</span></div>
            <div className="model-perf-validation"><strong>Evaluation</strong><span>36,500 held-out samples across 50 Indian cities (2022–2023).</span></div>
          </div>
        </details>

        <details className="model-perf-domain">
          <summary>
            <span className="model-perf-domain-icon heat"><Activity size={16} /></span>
            <span><strong>HeatRisk V4 &amp; Soft MoE</strong><small>Thermal biometeorology · 13 features</small></span>
            <ChevronDown className="model-perf-chevron" size={17} />
          </summary>
          <div className="model-perf-domain-body">
            <FeatureTags features={heatFeatures} />
            <div className="model-perf-output"><strong>Output</strong><span>Next-day Tmax in °C, with 80% and 90% conformal coverage intervals.</span></div>
            <div className="model-perf-regions">
              <strong>Soft MoE regional MAE</strong>
              <span>Coastal <b>0.874 °C</b></span>
              <span>Deccan Plateau <b>0.856 °C</b></span>
              <span>Indo-Gangetic Plains <b>1.147 °C</b></span>
              <span>Semi-Arid Central <b>0.968 °C</b></span>
              <span>Montane Sub-Himalayan <b>1.213 °C</b></span>
            </div>
          </div>
        </details>

        <details className="model-perf-domain">
          <summary>
            <span className="model-perf-domain-icon power"><Clock3 size={16} /></span>
            <span><strong>PowerGrid Load &amp; Deficit</strong><small>Critical energy infrastructure · 12 load features</small></span>
            <ChevronDown className="model-perf-chevron" size={17} />
          </summary>
          <div className="model-perf-domain-body">
            <FeatureTags features={powerFeatures} />
            <div className="model-perf-output"><strong>Outputs</strong><span>Continuous peak transmission load in MW, and grid-stress / unmet-deficit probability P(Deficit) ∈ [0, 1].</span></div>
          </div>
        </details>

        <details className="model-perf-domain">
          <summary>
            <span className="model-perf-domain-icon air"><Activity size={16} /></span>
            <span><strong>AirQuality &amp; SmokePath</strong><small>Atmospheric pollution · 12 AirQuality features</small></span>
            <ChevronDown className="model-perf-chevron" size={17} />
          </summary>
          <div className="model-perf-domain-body">
            <FeatureTags features={airQualityFeatures} />
            <div className="model-perf-output"><strong>Output</strong><span>Next-day 24-hour average PM2.5 concentration (µg/m³) and CPCB six-tier category: Good, Satisfactory, Moderate, Poor, Very Poor, or Severe.</span></div>
          </div>
        </details>

        <details className="model-perf-domain">
          <summary>
            <span className="model-perf-domain-icon water"><Activity size={16} /></span>
            <span><strong>BasinFlow hydrological benchmark</strong><small>CAMELS-IND catchments · 3 hydrological features</small></span>
            <ChevronDown className="model-perf-chevron" size={17} />
          </summary>
          <div className="model-perf-domain-body">
            <div className="model-perf-feature-groups">
              <div className="model-perf-feature-group">
                <strong>Inputs</strong>
                <FeatureTags features={["daily_precipitation_mm", "potential_evapotranspiration_mm", "catchment_surface_area_km2"]} />
              </div>
            </div>
            <div className="model-perf-output"><strong>Output</strong><span>Streamflow discharge Q in mm/day and m³/s, decomposed into quickflow and baseflow.</span></div>
            <div className="model-perf-validation"><strong>Mass balance parity</strong><span>Error &lt; 0.001% strictly enforced.</span></div>
          </div>
        </details>
      </section>

      <footer className="model-perf-footer">
        <ShieldCheck size={15} />
        <span>Reported evaluation snapshot · Metrics reflect held-out validation results supplied for this model registry.</span>
      </footer>
    </div>
  );
}

export default ModelPerformance;
