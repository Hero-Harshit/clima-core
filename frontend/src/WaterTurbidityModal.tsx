import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Droplets,
  Filter,
  RefreshCw,
  ShieldCheck,
  Waves,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type WaterTurbidityResponse = {
  city: string;
  raw_water_source: string;
  predicted_raw_turbidity_ntu: number;
  coagulant_dose_ppm: number;
  total_daily_coagulant_kg: number;
  estimated_settling_floc_time_mins: number;
  filter_backwash_interval_hours: number;
  sludge_volume_dry_tons: number;
  treatment_risk_level: string;
  operational_advisories: string[];
  ai_water_guidance: string;
  provider: string;
  latency_ms?: number;
};

type WaterTurbidityModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

const SOURCE_OPTIONS = [
  { id: "surface_river_run_of_the_river", label: "Surface River (Run-of-the-River Catchment)" },
  { id: "lake_catchment", label: "Natural Lake / Urban Wetland Catchment" },
  { id: "canal_gravity_fed", label: "Gravity-Fed Irrigation / Raw Water Canal" },
  { id: "large_storage_reservoir", label: "Deep Masonry / Earthen Storage Reservoir" },
];

const COAGULANT_OPTIONS = [
  { id: "polyaluminium_chloride_pac", label: "Polyaluminium Chloride (PAC, High Basicity)" },
  { id: "commercial_alum_aluminum_sulfate", label: "Commercial Hydrated Alum (Al₂SO₄)" },
  { id: "ferric_chloride_polymer_blend", label: "Ferric Chloride & Polymer Cationic Blend" },
];

const CAPACITY_OPTIONS = [
  { mld: 50, label: "50 MLD" },
  { mld: 100, label: "100 MLD" },
  { mld: 250, label: "250 MLD" },
  { mld: 500, label: "500 MLD" },
];

export default function WaterTurbidityModal({ city, onClose }: WaterTurbidityModalProps) {
  const [source, setSource] = useState<string>("surface_river_run_of_the_river");
  const [coagulant, setCoagulant] = useState<string>("polyaluminium_chloride_pac");
  const [capacityMld, setCapacityMld] = useState<number>(100);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<WaterTurbidityResponse | null>(null);

  const fetchDosingPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/water-turbidity`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          raw_water_source: source,
          coagulant_type: coagulant,
          plant_capacity_mld: capacityMld,
        }),
      });
      if (!res.ok) {
        throw new Error(`Water treatment service returned HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to compute coagulant dosing parameters.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchDosingPrediction();
  }, [city, source, coagulant, capacityMld]);

  const getRiskBadgeClass = (level?: string) => {
    switch (level) {
      case "SEVERE_TURBIDITY_FLOOD_OVERLOAD":
        return "hazard-badge-critical";
      case "ELEVATED_COAGULANT_REGIME":
        return "hazard-badge-warning";
      case "STABLE_CLARIFICATION":
        return "hazard-badge-moderate";
      default:
        return "hazard-badge-safe";
    }
  };

  return (
    <div className="utility-modal-backdrop" onClick={onClose}>
      <div
        className="utility-modal-card botanical-card"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-labelledby="water-modal-title"
      >
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn water-turbidity-circle-btn" style={{ width: 36, height: 36 }}>
              <Droplets size={18} />
            </div>
            <div>
              <h2 id="water-modal-title" className="utility-modal-heading">
                Water Treatment Turbidity & Coagulant Dosing Predictor
              </h2>
              <p className="utility-modal-subtitle">
                Catchment runoff silt surge, jar-test coagulant dosing, and filter backwash scheduling for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Water Treatment Modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="source-select">Raw Water Inflow Source</label>
            <UtilityCustomSelect
              id="source-select"
              value={source}
              options={SOURCE_OPTIONS}
              onChange={setSource}
              ariaLabel="Select raw water source"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="coagulant-select">Active Coagulant</label>
            <UtilityCustomSelect
              id="coagulant-select"
              value={coagulant}
              options={COAGULANT_OPTIONS}
              onChange={setCoagulant}
              ariaLabel="Select active coagulant"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Plant Output Rating</label>
            <div className="utility-chips-row">
              {CAPACITY_OPTIONS.map((c) => (
                <button
                  key={c.mld}
                  type="button"
                  className={`utility-chip-btn ${capacityMld === c.mld ? "selected" : ""}`}
                  onClick={() => setCapacityMld(c.mld)}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Content Body */}
        <div className="utility-modal-content">
          {loading && (
            <div className="utility-loading-state">
              <RefreshCw className="animate-spin" size={24} />
              <span>Calibrating jar-test flocculation curve and clarifier detention time...</span>
            </div>
          )}

          {error && !loading && (
            <div className="utility-error-banner">
              <AlertTriangle size={18} />
              <span>{error}</span>
              <button type="button" className="utility-btn-secondary" onClick={() => void fetchDosingPrediction()}>
                Retry
              </button>
            </div>
          )}

          {data && !loading && (
            <>
              {/* Primary KPI Row */}
              <div className="utility-kpi-grid">
                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Waves size={14} /> Raw Inflow Turbidity
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#d97706" }}>
                    {data.predicted_raw_turbidity_ntu} NTU
                  </div>
                  <div className="utility-kpi-meta">
                    <span className={`hazard-badge ${getRiskBadgeClass(data.treatment_risk_level || "OPTIMAL")}`}>
                      {(data.treatment_risk_level || "OPTIMAL").replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Droplets size={14} /> Optimum Dose Rate
                  </div>
                  <div className="utility-kpi-value">
                    {data.coagulant_dose_ppm} ppm
                  </div>
                  <div className="utility-kpi-meta">
                    Daily Requirement: {data.total_daily_coagulant_kg.toLocaleString()} kg/day
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Clock size={14} /> Clarifier Floc Time
                  </div>
                  <div className="utility-kpi-value">
                    {data.estimated_settling_floc_time_mins} min
                  </div>
                  <div className="utility-kpi-meta">
                    Flash Mix Speed: 120 RPM
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Filter size={14} /> Filter Backwash
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#0891b2" }}>
                    Every {data.filter_backwash_interval_hours} hrs
                  </div>
                  <div className="utility-kpi-meta">
                    Dry Sludge: {data.sludge_volume_dry_tons} Tons/day
                  </div>
                </div>
              </div>

              {/* AI Guidance */}
              <div className="utility-ai-guidance-card">
                <div className="utility-ai-header">
                  <span>ClimaCore Municipal Water Treatment & Chemistry Synthesis</span>
                  <span className="utility-provider-badge">ClimaCore Intelligence</span>
                </div>
                <p className="utility-ai-text">{data.ai_water_guidance}</p>
              </div>

              {/* Advisories List */}
              <div className="utility-advisories-card">
                <h3 className="utility-section-title">
                  <ShieldCheck size={16} /> Plant Superintendent Operational Protocols
                </h3>
                <ul className="utility-checklist">
                  {data.operational_advisories.map((item, idx) => (
                    <li key={idx} className="utility-check-item">
                      <CheckCircle2 size={16} className="utility-check-icon" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
