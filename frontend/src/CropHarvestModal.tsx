import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Droplets,
  Gauge,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Sprout,
  X,
} from "lucide-react";

type CropStageRisk = {
  stage_name: string;
  risk_level: string;
  estimated_moisture_pct: number;
  mold_incubation_risk: string;
  advisory: string;
};

type CropHarvestResponse = {
  city: string;
  crop_type: string;
  harvest_stage: string;
  storage_type: string;
  spoilage_risk_score: number;
  status_level: string;
  harvest_window_status: string;
  estimated_field_moisture_pct: number;
  dew_point_spread_c: number;
  fungal_mold_incubation_risk: string;
  safe_ventilation_hours: string;
  drying_recommendations: string[];
  stages_breakdown: CropStageRisk[];
  ai_agronomy_guidance: string;
  provider: string;
  latency_ms?: number;
};

type CropHarvestModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const CROP_OPTIONS = [
  { id: "paddy", label: "Paddy (Rice)", desc: "High water absorption & sprouting sensitivity" },
  { id: "wheat", label: "Wheat", desc: "Post-maturity shattering & mold vulnerability" },
  { id: "maize", label: "Maize (Corn)", desc: "Ear-rot & mycotoxin incubation risk" },
  { id: "cotton", label: "Cotton", desc: "Boll rot & fiber discoloration hazard" },
  { id: "soybean", label: "Soybean", desc: "Pod shattering & high moisture oil loss" },
  { id: "pulses", label: "Pulses / Lentils", desc: "Rapid beetle & fungal susceptibility" },
];

const STAGE_OPTIONS = [
  { id: "mature_standing", label: "Standing Mature Crop" },
  { id: "field_cut", label: "Cut in Field (Windrowed)" },
  { id: "thrashing_yard", label: "Open Threshing Yard" },
  { id: "bagged_godown", label: "Bagged in Rural Godown" },
];

const STORAGE_OPTIONS = [
  { id: "traditional_tarpaulin", label: "Traditional Tarpaulin Cover" },
  { id: "covered_shed", label: "Covered Farm Shed" },
  { id: "hermetic_bag", label: "Hermetic Moisture-Barrier Bags" },
  { id: "ventilated_metal_silo", label: "Ventilated Metal Silo" },
];

export default function CropHarvestModal({ city, onClose }: CropHarvestModalProps) {
  const [cropType, setCropType] = useState<string>("paddy");
  const [harvestStage, setHarvestStage] = useState<string>("mature_standing");
  const [storageType, setStorageType] = useState<string>("traditional_tarpaulin");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CropHarvestResponse | null>(null);

  const fetchCropSpoilageAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/crop-spoilage`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          crop_type: cropType,
          harvest_stage: harvestStage,
          storage_type: storageType,
        }),
      });
      if (!res.ok) {
        throw new Error(`Crop harvest spoilage analysis failed (${res.status})`);
      }
      const json = (await res.json()) as CropHarvestResponse;
      setData(json);
    } catch (err) {
      console.error("Crop Harvest Spoilage error:", err);
      setError(err instanceof Error ? err.message : "Failed to load crop spoilage analysis");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCropSpoilageAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "SAFE_WINDOW":
        return "hazard-badge-clear";
      case "MODERATE_RISK":
        return "hazard-badge-caution";
      default:
        return "hazard-badge-critical";
    }
  };

  const getMoldBadgeClass = (mold: string) => {
    switch (mold) {
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
            <div className="utility-tool-circle-btn spoilage-circle-btn" style={{ width: 36, height: 36 }}>
              <Sprout size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Sprout size={13} className="inline-icon" /> AGRICULTURE & GRAIN PRESERVATION
              </div>
              <h2>Crop Harvest Spoilage Radar</h2>
              <p className="utility-modal-subtitle">
                Live grain moisture, dew condensation, and post-harvest drying risk calculator for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Crop Harvest Spoilage Radar"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="crop-select">Crop Type</label>
            <select
              id="crop-select"
              className="utility-select"
              value={cropType}
              onChange={(e) => setCropType(e.target.value)}
            >
              {CROP_OPTIONS.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label}
                </option>
              ))}
            </select>
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Harvest Stage</label>
            <div className="utility-chips-row">
              {STAGE_OPTIONS.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={`utility-chip-btn ${harvestStage === s.id ? "selected" : ""}`}
                  onClick={() => setHarvestStage(s.id)}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-control-group">
            <label htmlFor="storage-select">Storage Facility</label>
            <select
              id="storage-select"
              className="utility-select"
              value={storageType}
              onChange={(e) => setStorageType(e.target.value)}
            >
              {STORAGE_OPTIONS.map((st) => (
                <option key={st.id} value={st.id}>
                  {st.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            className="utility-refresh-btn"
            onClick={fetchCropSpoilageAnalysis}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? "spin" : ""} />
            <span>Recalculate</span>
          </button>
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
              {/* Status & Key Metrics Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Gauge size={13} className="inline-icon" /> Spoilage Threat Index
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.spoilage_risk_score}</span>
                    <span className="utility-metric-unit">/100</span>
                    <span className={`utility-status-pill ${getStatusBadgeClass(data.status_level)}`}>
                      {data.status_level.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    {data.spoilage_risk_score > 60
                      ? "High atmospheric humidity and condensation threat"
                      : "Favorable conditions for field curing & drying"}
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Droplets size={13} className="inline-icon" /> Grain Moisture & Dew Point
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.estimated_field_moisture_pct}%</span>
                    <span className="utility-metric-unit">moisture</span>
                    <span className={`utility-status-pill ${getMoldBadgeClass(data.fungal_mold_incubation_risk)}`}>
                      {data.fungal_mold_incubation_risk} MOLD RISK
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Dew point spread: {data.dew_point_spread_c} deg C (Nocturnal condensation risk)
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Clock size={13} className="inline-icon" /> Safe Ventilation Hours
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val">{data.safe_ventilation_hours}</span>
                  </div>
                  <div className="utility-metric-sub">
                    Run aeration blowers strictly during low ambient relative humidity
                  </div>
                </div>
              </div>

              {/* Harvest Safety Window Banner */}
              <div className="utility-verdict-banner">
                <div className="utility-verdict-title">
                  <Sprout size={16} className="inline-icon" /> {data.harvest_window_status}
                </div>
              </div>

              {/* AI Agronomy Intelligence Guidance */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    <Sparkles size={13} className="inline-icon" /> POST-HARVEST AGRONOMY ADVISORY
                  </div>
                  <span className="utility-provider-badge">
                    {data.provider.includes("gemini") ? "Powered by Gemini Intelligence" : "Deterministic Agronomy Engine"}
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_agronomy_guidance}</p>
              </div>

              {/* Stage-by-Stage Vulnerability Breakdown */}
              <div className="utility-section-title">
                Harvest & Storage Stage Vulnerability Radar
              </div>
              <div className="utility-segments-list">
                {data.stages_breakdown.map((stg, i) => (
                  <div key={i} className="utility-segment-row">
                    <div className="utility-segment-left">
                      <div className="utility-segment-hazard-badge">
                        <Droplets size={14} className="hazard-icon-rain" />
                      </div>
                      <div className="utility-segment-info">
                        <div className="utility-segment-title">{stg.stage_name}</div>
                        <div className="utility-segment-advisory">{stg.advisory}</div>
                      </div>
                    </div>
                    <div className="utility-segment-right">
                      <span className="utility-speed-pill">
                        {stg.estimated_moisture_pct}% Moisture
                      </span>
                      <span className={`utility-status-pill ${getStatusBadgeClass(stg.risk_level)}`}>
                        {stg.risk_level}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Drying and Aeration Protocol Checklist */}
              <div className="utility-section-title">
                Actionable Grain Drying & Protection Checklist
              </div>
              <div className="utility-checklist-grid">
                {data.drying_recommendations.map((rec, i) => (
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
            <ShieldCheck size={14} className="inline-icon" /> Grounded in real-time localized agrometeorological sensors
          </div>
          <button type="button" className="utility-btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
