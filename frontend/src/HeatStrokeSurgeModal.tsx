import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Droplets,
  Flame,
  HeartPulse,
  RefreshCw,
  ShieldCheck,
  Stethoscope,
  Thermometer,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type HeatStrokeSurgeResponse = {
  city: string;
  facility_type: string;
  surge_risk_score: number;
  surge_alert_level: string;
  projected_daily_heat_admissions: number;
  wet_bulb_c: number;
  heat_index_c: number;
  pediatric_geriatric_vulnerability: string;
  daily_iv_fluid_demand_liters: number;
  ors_packets_needed: number;
  er_triage_advisories: string[];
  ai_clinical_guidance: string;
  provider: string;
  latency_ms?: number;
};

type HeatStrokeSurgeModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const FACILITY_OPTIONS = [
  { id: "tertiary_hospital", label: "Tertiary Teaching Hospital (Multi-Specialty)" },
  { id: "district_hospital", label: "District Civil Hospital" },
  { id: "urban_primary_clinic", label: "Urban Primary Health Center (UPHC)" },
];

const BED_OPTIONS = [
  { beds: 20, label: "20 ER Beds" },
  { beds: 40, label: "40 ER Beds" },
  { beds: 80, label: "80 ER Beds" },
  { beds: 120, label: "120 ER Beds" },
];

export default function HeatStrokeSurgeModal({ city, onClose }: HeatStrokeSurgeModalProps) {
  const [facilityType, setFacilityType] = useState<string>("tertiary_hospital");
  const [erBeds, setErBeds] = useState<number>(40);
  const [coolingWardActive] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<HeatStrokeSurgeResponse | null>(null);

  const fetchSurgePrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/heat-stroke-surge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          facility_type: facilityType,
          er_capacity_beds: erBeds,
          cooling_ward_active: coolingWardActive,
        }),
      });
      if (!res.ok) {
        throw new Error(`Heat stroke surge prediction failed (${res.status})`);
      }
      const json = (await res.json()) as HeatStrokeSurgeResponse;
      setData(json);
    } catch (err) {
      console.error("Heat Stroke Surge error:", err);
      setError(err instanceof Error ? err.message : "Failed to load heat surge forecast");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSurgePrediction();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getAlertBadgeClass = (alert: string) => {
    switch (alert) {
      case "LOW":
        return "hazard-badge-clear";
      case "MODERATE":
        return "hazard-badge-caution";
      default:
        return "hazard-badge-critical";
    }
  };

  return (
    <div className="utility-modal-scrim" onClick={onClose} role="dialog" aria-modal="true">
      <div className="utility-modal-window" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn heat-surge-circle-btn" style={{ width: 36, height: 36 }}>
              <Flame size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <HeartPulse size={13} className="inline-icon" /> EMERGENCY CLINICAL SURGE PREDICTOR
              </div>
              <h2>Heat Stroke & Dehydration Surge Predictor</h2>
              <p className="utility-modal-subtitle">
                Emergency room casualty surge forecasting, wet-bulb thermal strain, and hydration buffer calculator for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Heat Surge Predictor"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="facility-select">Facility Tier</label>
            <UtilityCustomSelect
              id="facility-select"
              value={facilityType}
              options={FACILITY_OPTIONS}
              onChange={setFacilityType}
              ariaLabel="Select facility tier"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Emergency Department Capacity</label>
            <div className="utility-chips-row">
              {BED_OPTIONS.map((b) => (
                <button
                  key={b.beds}
                  type="button"
                  className={`utility-chip-btn ${erBeds === b.beds ? "selected" : ""}`}
                  onClick={() => setErBeds(b.beds)}
                >
                  {b.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-refresh-btn"
              onClick={fetchSurgePrediction}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              <span>Recalculate</span>
            </button>
          </div>
        </div>

        {/* Modal Body */}
        <div className="utility-modal-body">
          {error && (
            <div className="utility-error-banner">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}

          {data && (
            <>
              {/* Verdict Banner */}
              <div className="utility-verdict-banner">
                <div className="utility-verdict-title">
                  <Thermometer size={16} className="inline-icon" /> Surge Alert Status: {data.surge_alert_level.replace(/_/g, " ")}
                  <span className={`utility-status-pill ${getAlertBadgeClass(data.surge_alert_level)}`} style={{ marginLeft: 8 }}>
                    Risk Index {data.surge_risk_score}/100
                  </span>
                </div>
              </div>

              {/* Status Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Stethoscope size={13} className="inline-icon" /> Daily Heat Casualty Influx
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">~{data.projected_daily_heat_admissions}</span>
                    <span className="utility-metric-unit">patients</span>
                    <span className="utility-status-pill hazard-badge-caution">PROJECTED</span>
                  </div>
                  <div className="utility-metric-sub">
                    Wet-Bulb Globe Temp: {data.wet_bulb_c} deg C (Heat Index: {data.heat_index_c} deg C)
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Droplets size={13} className="inline-icon" /> Emergency IV Fluids Buffer
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.daily_iv_fluid_demand_liters}</span>
                    <span className="utility-metric-unit">Liters</span>
                  </div>
                  <div className="utility-metric-sub">
                    Stock {data.ors_packets_needed} Oral Rehydration Salts (ORS) sachets for outpatient triage
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <AlertTriangle size={13} className="inline-icon" /> Vulnerability Demographic
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val" style={{ fontSize: "13px" }}>
                      High Influx Imminent
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    {data.pediatric_geriatric_vulnerability}
                  </div>
                </div>
              </div>

              {/* AI Clinical Guidance */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    CLINICAL TRIAGE ADVISORY
                  </div>
                  <span className="utility-provider-badge">
                    {data.provider.includes("gemini") ? "Powered by Gemini Intelligence" : "Deterministic Engine"}
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_clinical_guidance}</p>
              </div>

              {/* ER Triage Protocols */}
              <div className="utility-section-title">
                Emergency Department Heatstroke Protocol Checklist
              </div>
              <div className="utility-checklist-grid">
                {data.er_triage_advisories.map((rec, i) => (
                  <div key={i} className="utility-checklist-item">
                    <CheckCircle2 size={15} className="utility-check-icon" />
                    <span>{rec}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Action Footer */}
        <div className="utility-action-footer">
          <div className="utility-footer-meta">
            <ShieldCheck size={14} className="inline-icon" /> Grounded in wet-bulb globe thermal strain & IDSP disease surveillance
          </div>
        </div>
      </div>
    </div>
  );
}
