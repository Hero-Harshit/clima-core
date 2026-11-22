import { useEffect, useState } from "react";
import {
  AlertTriangle,
  BadgeAlert,
  CheckCircle2,
  Coins,
  History,
  Percent,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type ParametricTriggerResponse = {
  city: string;
  peril_type: string;
  strike_threshold_value: number;
  current_observed_index_value: number;
  trigger_status: string;
  strike_breach_probability_pct: number;
  estimated_payout_inr_crores: number;
  payout_ratio_pct: number;
  basis_risk_score: string;
  historical_event_benchmark: string;
  underwriter_advisories: string[];
  ai_actuarial_guidance: string;
  provider: string;
  latency_ms?: number;
};

type ParametricTriggerModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const PERIL_OPTIONS = [
  { id: "excess_rainfall_flood", label: "Excess Rainfall Deluge & Cloudburst" },
  { id: "severe_heatwave_degree_days", label: "Extreme Heatwave Degree Days (HDD)" },
  { id: "cyclonic_storm_wind_gust", label: "Cyclonic Storm Gale Wind Gust" },
  { id: "drought_consecutive_dry_days", label: "Agricultural Drought (Consecutive Dry Days)" },
];

const LIMIT_OPTIONS = [
  { limit: 5.0, label: "₹5 Crores" },
  { limit: 25.0, label: "₹25 Crores" },
  { limit: 100.0, label: "₹100 Crores" },
  { limit: 500.0, label: "₹500 Crores" },
];

export default function ParametricTriggerModal({ city, onClose }: ParametricTriggerModalProps) {
  const [peril, setPeril] = useState<string>("excess_rainfall_flood");
  const [coverageLimit, setCoverageLimit] = useState<number>(25.0);
  const [tenorDays, setTenorDays] = useState<number>(30);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ParametricTriggerResponse | null>(null);

  const fetchSimulation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/parametric-trigger`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          peril_type: peril,
          coverage_limit_inr_crores: coverageLimit,
          policy_tenor_days: tenorDays,
        }),
      });
      if (!res.ok) {
        throw new Error(`Parametric trigger service returned HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to run parametric contract simulation.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchSimulation();
  }, [city, peril, coverageLimit, tenorDays]);

  const getStatusBadgeClass = (status?: string) => {
    switch (status) {
      case "TRIGGER_BREACHED_PAYOUT_ACTIVE":
        return "hazard-badge-critical";
      case "IMMINENT_NEAR_TRIGGER_WARN":
        return "hazard-badge-warning";
      case "ACCUMULATING_BELOW_STRIKE":
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
        aria-labelledby="insurance-modal-title"
      >
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn parametric-trigger-circle-btn" style={{ width: 36, height: 36 }}>
              <ShieldAlert size={18} />
            </div>
            <div>
              <h2 id="insurance-modal-title" className="utility-modal-heading">
                Parametric Weather Contract Trigger Simulator
              </h2>
              <p className="utility-modal-subtitle">
                Climate derivative tranche strikes, loss payout simulations, and actuarial basis risk for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Parametric Trigger Simulator"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="peril-select">Underwritten Peril</label>
            <UtilityCustomSelect
              id="peril-select"
              value={peril}
              options={PERIL_OPTIONS}
              onChange={setPeril}
              ariaLabel="Select underwritten peril"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Policy Tranche Capacity</label>
            <div className="utility-chips-row">
              {LIMIT_OPTIONS.map((l) => (
                <button
                  key={l.limit}
                  type="button"
                  className={`utility-chip-btn ${coverageLimit === l.limit ? "selected" : ""}`}
                  onClick={() => setCoverageLimit(l.limit)}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Contract Risk Tenor</label>
            <div className="utility-chips-row">
              {[15, 30, 60, 90].map((days) => (
                <button
                  key={days}
                  type="button"
                  className={`utility-chip-btn ${tenorDays === days ? "selected" : ""}`}
                  onClick={() => setTenorDays(days)}
                >
                  {days} Days
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
              <span>Simulating probabilistic contract trigger curves and payout tranches...</span>
            </div>
          )}

          {error && !loading && (
            <div className="utility-error-banner">
              <AlertTriangle size={18} />
              <span>{error}</span>
              <button type="button" className="utility-btn-secondary" onClick={() => void fetchSimulation()}>
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
                    <BadgeAlert size={14} /> Contract Strike Status
                  </div>
                  <div className="utility-kpi-value" style={{ fontSize: 16, lineHeight: 1.25 }}>
                    <span className={`hazard-badge ${getStatusBadgeClass(data.trigger_status)}`}>
                      {data.trigger_status.replace(/_/g, " ")}
                    </span>
                  </div>
                  <div className="utility-kpi-meta">
                    Observed: {data.current_observed_index_value} / Strike: {data.strike_threshold_value}
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Percent size={14} /> Breach Probability
                  </div>
                  <div className="utility-kpi-value" style={{ color: data.strike_breach_probability_pct > 50 ? "#e11d48" : "#ca8a04" }}>
                    {data.strike_breach_probability_pct}%
                  </div>
                  <div className="utility-kpi-meta">
                    Exceedance over {tenorDays}-day risk window
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Coins size={14} /> Modeled Loss Settlement
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#16a34a" }}>
                    ₹{data.estimated_payout_inr_crores.toFixed(2)} Cr
                  </div>
                  <div className="utility-kpi-meta">
                    Payout Ratio: {data.payout_ratio_pct}% of ₹{coverageLimit} Cr
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <History size={14} /> Benchmark Precedent
                  </div>
                  <div className="utility-kpi-value" style={{ fontSize: 13, lineHeight: 1.35, fontWeight: 500 }}>
                    {data.historical_event_benchmark}
                  </div>
                  <div className="utility-kpi-meta">
                    {data.basis_risk_score}
                  </div>
                </div>
              </div>

              {/* AI Guidance */}
              <div className="utility-ai-guidance-card">
                <div className="utility-ai-header">
                  <span>ClimaCore Climate Insurance & Actuarial Synthesis</span>
                  <span className="utility-provider-badge">ClimaCore Intelligence</span>
                </div>
                <p className="utility-ai-text">{data.ai_actuarial_guidance}</p>
              </div>

              {/* Advisories List */}
              <div className="utility-advisories-card">
                <h3 className="utility-section-title">
                  <ShieldCheck size={16} /> Underwriter Reserve & Risk Transfer Protocol
                </h3>
                <ul className="utility-checklist">
                  {data.underwriter_advisories.map((item, idx) => (
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
