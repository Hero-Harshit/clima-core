import { useEffect, useState } from "react";
import {
  AlertTriangle,
  BatteryCharging,
  BatteryWarning,
  CheckCircle2,
  Clock,
  Fuel,
  RefreshCw,
  ShieldCheck,
  Zap,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type HospitalGridFuelResponse = {
  city: string;
  hospital_tier: string;
  grid_failure_risk_score: number;
  grid_stability_level: string;
  full_load_runtime_hours: number;
  fuel_burn_rate_lph: number;
  tanker_replenishment_delay_hours: number;
  safe_fuel_buffer_hours: number;
  priority_load_shedding_order: string[];
  generator_readiness_checklist: string[];
  ai_engineering_advisory: string;
  provider: string;
  latency_ms?: number;
};

type HospitalGridFuelModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

const TIER_OPTIONS = [
  { id: "tertiary_specialty", label: "Tertiary Multi-Specialty Hospital (850 kW)" },
  { id: "district_civil", label: "District Civil Hospital (380 kW)" },
  { id: "community_health_center", label: "Community Health Center (CHC, 120 kW)" },
  { id: "primary_health_center", label: "Primary Health Center (PHC, 40 kW)" },
];

const FUEL_OPTIONS = [
  { liters: 1500, label: "1,500 L" },
  { liters: 3500, label: "3,500 L" },
  { liters: 6000, label: "6,000 L" },
  { liters: 12000, label: "12,000 L" },
];

export default function HospitalGridFuelModal({ city, onClose }: HospitalGridFuelModalProps) {
  const [tier, setTier] = useState<string>("tertiary_specialty");
  const [dieselReserve, setDieselReserve] = useState<number>(6000);
  const [ventilators] = useState<number>(45);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<HospitalGridFuelResponse | null>(null);

  const fetchGridFuelAssessment = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/hospital-grid-fuel`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          hospital_tier: tier,
          diesel_reserve_liters: dieselReserve,
          critical_icu_ventilators: ventilators,
        }),
      });
      if (!res.ok) {
        throw new Error(`Hospital grid fuel calculation failed (${res.status})`);
      }
      const json = (await res.json()) as HospitalGridFuelResponse;
      setData(json);
    } catch (err) {
      console.error("Hospital Grid Fuel error:", err);
      setError(err instanceof Error ? err.message : "Failed to load backup fuel assessment");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGridFuelAssessment();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getStabilityPillClass = (status: string) => {
    switch (status) {
      case "STABLE_GRID":
        return "hazard-badge-clear";
      case "ELEVATED_BROWNOUT_RISK":
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
            <div className="utility-tool-circle-btn grid-fuel-circle-btn" style={{ width: 36, height: 36 }}>
              <BatteryWarning size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Zap size={13} className="inline-icon" /> CRITICAL HEALTHCARE INFRASTRUCTURE RESILIENCE
              </div>
              <h2>Hospital Grid Outage & Backup Generator Fuel Buffer</h2>
              <p className="utility-modal-subtitle">
                Power grid stability assessment, ICU life-support run-hours, and diesel logistics buffer for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Hospital Grid Fuel Buffer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="tier-select">Hospital Tier</label>
            <UtilityCustomSelect
              id="tier-select"
              value={tier}
              options={TIER_OPTIONS}
              onChange={setTier}
              ariaLabel="Select hospital tier"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Diesel Reserve Tank Volume</label>
            <div className="utility-chips-row">
              {FUEL_OPTIONS.map((f) => (
                <button
                  key={f.liters}
                  type="button"
                  className={`utility-chip-btn ${dieselReserve === f.liters ? "selected" : ""}`}
                  onClick={() => setDieselReserve(f.liters)}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-refresh-btn"
              onClick={fetchGridFuelAssessment}
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
                  <Zap size={16} className="inline-icon" /> Grid Reliability: {(data.grid_stability_level || "STABLE").replace(/_/g, " ")}
                  <span className={`utility-status-pill ${getStabilityPillClass(data.grid_stability_level || "STABLE")}`} style={{ marginLeft: 8 }}>
                    Grid Vulnerability {data.grid_failure_risk_score}/100
                  </span>
                </div>
              </div>

              {/* Status Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Clock size={13} className="inline-icon" /> Generator Full Load Runtime
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.full_load_runtime_hours}</span>
                    <span className="utility-metric-unit">hours</span>
                    <span className="utility-status-pill hazard-badge-clear">DIESEL RESERVE</span>
                  </div>
                  <div className="utility-metric-sub">
                    Consumption rate: {data.fuel_burn_rate_lph} Liters/hour at rated ICU & HVAC load
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Fuel size={13} className="inline-icon" /> Net Safe Buffer Margin
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">+{data.safe_fuel_buffer_hours}</span>
                    <span className="utility-metric-unit">hours</span>
                  </div>
                  <div className="utility-metric-sub">
                    Accounts for {data.tanker_replenishment_delay_hours}h supply delivery delay due to flooded access routes
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <BatteryCharging size={13} className="inline-icon" /> Critical ICU Ventilators
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val">
                      {ventilators} Units Active
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Instantaneous ATS automatic failover backed by UPS battery isolation
                  </div>
                </div>
              </div>

              {/* AI Engineering Advisory */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    FACILITY ENGINEERING DIRECTIVE
                  </div>
                  <span className="utility-provider-badge">
                    "Powered by ClimaCore Intelligence"
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_engineering_advisory}</p>
              </div>

              {/* Load Shedding Order */}
              <div className="utility-section-title">
                Emergency Circuit Load Shedding Hierarchy
              </div>
              <div className="utility-segments-list">
                {data.priority_load_shedding_order.map((shed, i) => (
                  <div key={i} className="utility-segment-row">
                    <div className="utility-segment-left">
                      <div className="utility-segment-hazard-badge">
                        <Zap size={14} className="hazard-icon-heat" />
                      </div>
                      <div className="utility-segment-info">
                        <div className="utility-segment-title">Tier {i + 1} Circuit Protocol</div>
                        <div className="utility-segment-advisory">{shed}</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Generator Readiness Checklist */}
              <div className="utility-section-title">
                Standby Generator & Substation Readiness Checklist
              </div>
              <div className="utility-checklist-grid">
                {data.generator_readiness_checklist.map((rec, i) => (
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
            <ShieldCheck size={14} className="inline-icon" /> Grounded in regional transmission grid load telemetry & IEEE 446 healthcare power standards
          </div>
        </div>
      </div>
    </div>
  );
}
