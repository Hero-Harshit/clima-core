import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Droplets,
  Gauge,
  GraduationCap,
  RefreshCw,
  ShieldCheck,
  Umbrella,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type RouteChokePoint = {
  choke_name: string;
  hazard_type: string;
  risk_level: string;
  transit_delay_mins: number;
  advisory: string;
};

type MonsoonExamResponse = {
  city: string;
  exam_shift: string;
  transit_mode: string;
  buffer_level: string;
  commute_risk_score: number;
  status_level: string;
  recommended_departure_advance_mins: number;
  target_arrival_time: string;
  inundation_zones: RouteChokePoint[];
  safe_arrival_checklist: string[];
  transit_advisory: string;
  ai_commute_guidance: string;
  provider: string;
  latency_ms?: number;
};

type MonsoonExamModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const SHIFT_OPTIONS = [
  { id: "morning", label: "Morning (09:00 AM - 12:00 PM)", desc: "Gate arrival by 08:30 AM" },
  { id: "afternoon", label: "Afternoon (02:00 PM - 05:00 PM)", desc: "Gate arrival by 01:30 PM" },
  { id: "evening", label: "Evening (05:30 PM - 08:30 PM)", desc: "Gate arrival by 05:00 PM" },
];

const MODE_OPTIONS = [
  { id: "public_bus_metro", label: "Public Bus & Metro", desc: "Grade-separated rail bypasses street waterlogging" },
  { id: "two_wheeler", label: "Two-Wheeler / Scooter", desc: "High water spray & road slippage hazard" },
  { id: "auto_cab", label: "Auto-Rickshaw / App Cab", desc: "Arterial bottleneck & surge delay risk" },
  { id: "walking", label: "Walking / Foot Transit", desc: "Open storm drain & curb submergence hazard" },
];

const BUFFER_OPTIONS = [
  { id: "standard", label: "Standard (+20 min)" },
  { id: "moderate", label: "Moderate Caution (+45 min)" },
  { id: "maximum", label: "Maximum Monsoon Buffer (+75 min)" },
];

export default function MonsoonExamModal({ city, onClose }: MonsoonExamModalProps) {
  const [examShift, setExamShift] = useState<string>("morning");
  const [transitMode, setTransitMode] = useState<string>("public_bus_metro");
  const [bufferLevel, setBufferLevel] = useState<string>("moderate");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<MonsoonExamResponse | null>(null);

  const fetchExamTransitAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/monsoon-exam`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          exam_shift: examShift,
          transit_mode: transitMode,
          buffer_level: bufferLevel,
        }),
      });
      if (!res.ok) {
        throw new Error(`Monsoon exam transit analysis failed (${res.status})`);
      }
      const json = (await res.json()) as MonsoonExamResponse;
      setData(json);
    } catch (err) {
      console.error("Monsoon Exam Commute error:", err);
      setError(err instanceof Error ? err.message : "Failed to load exam commute analysis");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExamTransitAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "ON_SCHEDULE":
        return "hazard-badge-clear";
      case "MODERATE_DELAYS":
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
            <div className="utility-tool-circle-btn monsoon-exam-circle-btn" style={{ width: 36, height: 36 }}>
              <GraduationCap size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <GraduationCap size={13} className="inline-icon" /> EDUCATION & STUDENT TRANSIT SAFETY
              </div>
              <h2>Monsoon Exam Commute Planner</h2>
              <p className="utility-modal-subtitle">
                Exam shift waterlogging transit planner, inundation delay buffer, and safe arrival checklist for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Monsoon Exam Commute Planner"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="shift-select">Exam Shift</label>
            <UtilityCustomSelect
              id="shift-select"
              value={examShift}
              options={SHIFT_OPTIONS}
              onChange={setExamShift}
              ariaLabel="Select exam shift"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="mode-select">Transit Mode</label>
            <UtilityCustomSelect
              id="mode-select"
              value={transitMode}
              options={MODE_OPTIONS}
              onChange={setTransitMode}
              ariaLabel="Select transit mode"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Monsoon Safety Buffer</label>
            <div className="utility-chips-row">
              {BUFFER_OPTIONS.map((b) => (
                <button
                  key={b.id}
                  type="button"
                  className={`utility-chip-btn ${bufferLevel === b.id ? "selected" : ""}`}
                  onClick={() => setBufferLevel(b.id)}
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
              onClick={fetchExamTransitAnalysis}
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
              {/* Status & Key Metrics Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Gauge size={13} className="inline-icon" /> Transit Disruption Index
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.commute_risk_score}</span>
                    <span className="utility-metric-unit">/100</span>
                    <span className={`utility-status-pill ${getStatusBadgeClass(data.status_level)}`}>
                      {data.status_level.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    {data.commute_risk_score > 60
                      ? "Substantial street flooding and transit delays projected"
                      : "Manageable conditions with standard rain gear"}
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Clock size={13} className="inline-icon" /> Recommended Early Departure
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">+{data.recommended_departure_advance_mins}</span>
                    <span className="utility-metric-unit">mins earlier</span>
                  </div>
                  <div className="utility-metric-sub">
                    Target Gate Arrival: {data.target_arrival_time}
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Umbrella size={13} className="inline-icon" /> Transit Priority Guidance
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val">{data.transit_advisory}</span>
                  </div>
                  <div className="utility-metric-sub">
                    Prioritize grade-separated rail to avoid low-lying street blockages
                  </div>
                </div>
              </div>

              {/* AI Student Transit Intelligence Box */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    STUDENT COMMUTE INTELLIGENCE
                  </div>
                  <span className="utility-provider-badge">
                    {data.provider.includes("gemini") ? "Powered by Gemini Intelligence" : "Deterministic Transit Engine"}
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_commute_guidance}</p>
              </div>

              {/* Route Inundation & Choke Points */}
              <div className="utility-section-title">
                Route Inundation Choke Points & Detours
              </div>
              <div className="utility-segments-list">
                {data.inundation_zones.map((zone, i) => (
                  <div key={i} className="utility-segment-row">
                    <div className="utility-segment-left">
                      <div className="utility-segment-hazard-badge">
                        <Droplets size={14} className="hazard-icon-rain" />
                      </div>
                      <div className="utility-segment-info">
                        <div className="utility-segment-title">{zone.choke_name}</div>
                        <div className="utility-segment-advisory">{zone.advisory}</div>
                      </div>
                    </div>
                    <div className="utility-segment-right">
                      <span className="utility-speed-pill">
                        +{zone.transit_delay_mins} min delay
                      </span>
                      <span className={`utility-status-pill ${getStatusBadgeClass(zone.risk_level)}`}>
                        {zone.risk_level}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Safe Arrival Exam Kit Checklist */}
              <div className="utility-section-title">
                Safe Arrival Exam Kit & Checklist
              </div>
              <div className="utility-checklist-grid">
                {data.safe_arrival_checklist.map((item, i) => (
                  <div key={i} className="utility-checklist-item">
                    <CheckCircle2 size={15} className="utility-check-icon" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* Action Footer */}
        <div className="utility-action-footer">
          <div className="utility-footer-meta">
            <ShieldCheck size={14} className="inline-icon" /> Real-time municipal storm drain telemetry & transit buffers
          </div>
        </div>
      </div>
    </div>
  );
}
