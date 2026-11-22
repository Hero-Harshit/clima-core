import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Factory,
  Flame,
  RefreshCw,
  ShieldCheck,
  Thermometer,
  Wind,
  X,
  Zap,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type FactoryTinRoofResponse = {
  city: str;
  roof_type: str;
  shift_type: str;
  ambient_temp_c: number;
  estimated_tin_surface_temp_c: number;
  shopfloor_wbgt_c: number;
  worker_heat_strain_risk: str;
  hvac_thermal_load_kw: number;
  peak_demand_cost_inr_hr: number;
  load_shedding_recommended_tr: number;
  sprinkler_roof_cooling_effectiveness_c: number;
  operational_advisories: string[];
  ai_industrial_guidance: string;
  provider: string;
  latency_ms?: number;
};

type str = string;

type FactoryTinRoofModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const ROOF_OPTIONS = [
  { id: "uninsulated_corrugated_tin", label: "Corrugated Galvanized Tin (Uninsulated)" },
  { id: "polycarbonate_skylight", label: "Polycarbonate Skylight Sheeting" },
  { id: "insulated_sandwich_puff", label: "Insulated Sandwich PUFF Panel (50mm)" },
  { id: "cool_roof_reflective_coating", label: "High-Albedo Reflective Cool Roof" },
];

const SHIFT_OPTIONS = [
  { id: "morning_shift", label: "Morning Shift (06:00 - 14:00)" },
  { id: "afternoon_general", label: "General Peak Shift (10:00 - 18:00)" },
  { id: "night_shift", label: "Night Shift (22:00 - 06:00)" },
];

const AREA_OPTIONS = [
  { sqm: 1000, label: "1,000 m²" },
  { sqm: 2500, label: "2,500 m²" },
  { sqm: 5000, label: "5,000 m²" },
  { sqm: 10000, label: "10,000 m²" },
];

const HVAC_OPTIONS = [
  { tr: 50, label: "50 TR" },
  { tr: 150, label: "150 TR" },
  { tr: 300, label: "300 TR" },
  { tr: 600, label: "600 TR" },
];

export default function FactoryTinRoofModal({ city, onClose }: FactoryTinRoofModalProps) {
  const [roofType, setRoofType] = useState<string>("uninsulated_corrugated_tin");
  const [shiftType, setShiftType] = useState<string>("afternoon_general");
  const [areaSqm, setAreaSqm] = useState<number>(2500);
  const [hvacTr, setHvacTr] = useState<number>(150);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FactoryTinRoofResponse | null>(null);

  const fetchAssessment = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/factory-tin-roof`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          roof_type: roofType,
          shift_type: shiftType,
          shopfloor_area_sqm: areaSqm,
          hvac_capacity_tr: hvacTr,
        }),
      });
      if (!res.ok) {
        throw new Error(`Factory assessment service returned HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to compute factory tin roof thermal stress.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchAssessment();
  }, [city, roofType, shiftType, areaSqm, hvacTr]);

  const getStrainBadgeClass = (risk?: string) => {
    switch (risk) {
      case "CRITICAL_HEAT_CESSATION":
        return "hazard-badge-critical";
      case "HIGH_MANDATORY_REST_ROTATION":
        return "hazard-badge-warning";
      case "MODERATE_CAUTION":
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
        aria-labelledby="factory-modal-title"
      >
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn tin-roof-circle-btn" style={{ width: 36, height: 36 }}>
              <Factory size={18} />
            </div>
            <div>
              <h2 id="factory-modal-title" className="utility-modal-heading">
                Factory Tin Roof Heat Stress & HVAC Load Shedder
              </h2>
              <p className="utility-modal-subtitle">
                Shopfloor WBGT thermal radiation, peak ToD tariff shedding, and worker hydration protocols for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Factory Heat Stress Modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="roof-select">Roof Material</label>
            <UtilityCustomSelect
              id="roof-select"
              value={roofType}
              options={ROOF_OPTIONS}
              onChange={setRoofType}
              ariaLabel="Select roof material"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="shift-select">Operating Shift</label>
            <UtilityCustomSelect
              id="shift-select"
              value={shiftType}
              options={SHIFT_OPTIONS}
              onChange={setShiftType}
              ariaLabel="Select operating shift"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Shed Floor Area</label>
            <div className="utility-chips-row">
              {AREA_OPTIONS.map((a) => (
                <button
                  key={a.sqm}
                  type="button"
                  className={`utility-chip-btn ${areaSqm === a.sqm ? "selected" : ""}`}
                  onClick={() => setAreaSqm(a.sqm)}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>HVAC Plant Capacity</label>
            <div className="utility-chips-row">
              {HVAC_OPTIONS.map((h) => (
                <button
                  key={h.tr}
                  type="button"
                  className={`utility-chip-btn ${hvacTr === h.tr ? "selected" : ""}`}
                  onClick={() => setHvacTr(h.tr)}
                >
                  {h.label}
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
              <span>Simulating factory tin roof thermal radiation and chiller demand...</span>
            </div>
          )}

          {error && !loading && (
            <div className="utility-error-banner">
              <AlertTriangle size={18} />
              <span>{error}</span>
              <button type="button" className="utility-btn-secondary" onClick={() => void fetchAssessment()}>
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
                    <Thermometer size={14} /> Estimated Tin Roof Skin
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#ea580c" }}>
                    {data.estimated_tin_surface_temp_c}°C
                  </div>
                  <div className="utility-kpi-meta">
                    Ambient Outdoor: {data.ambient_temp_c}°C
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Flame size={14} /> Shopfloor WBGT
                  </div>
                  <div className="utility-kpi-value">
                    {data.shopfloor_wbgt_c}°C
                  </div>
                  <div className="utility-kpi-meta">
                    <span className={`hazard-badge ${getStrainBadgeClass(data.worker_heat_strain_risk)}`}>
                      {data.worker_heat_strain_risk.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Zap size={14} /> HVAC Chiller Draw
                  </div>
                  <div className="utility-kpi-value">
                    {data.hvac_thermal_load_kw} kW
                  </div>
                  <div className="utility-kpi-meta">
                    Peak ToD Tariff: ₹{data.peak_demand_cost_inr_hr}/hr
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Wind size={14} /> Load Shedding Potential
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#16a34a" }}>
                    {data.load_shedding_recommended_tr} TR
                  </div>
                  <div className="utility-kpi-meta">
                    Roof Misting Drop: -{data.sprinkler_roof_cooling_effectiveness_c}°C
                  </div>
                </div>
              </div>

              {/* AI Guidance */}
              <div className="utility-ai-guidance-card">
                <div className="utility-ai-header">
                  <span>ClimaCore Plant Engineering & Occupational Health Synthesis</span>
                  <span className="utility-provider-badge">ClimaCore Intelligence</span>
                </div>
                <p className="utility-ai-text">{data.ai_industrial_guidance}</p>
              </div>

              {/* Advisories List */}
              <div className="utility-advisories-card">
                <h3 className="utility-section-title">
                  <ShieldCheck size={16} /> Standard Operating Procedures & Worker Safety
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
