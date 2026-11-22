import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Anchor,
  CheckCircle2,
  RefreshCw,
  ShieldCheck,
  Ship,
  Waves,
  Wind,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type CoastalSquallResponse = {
  coastal_zone: string;
  craft_type: string;
  sea_state_code: number;
  sea_state_label: string;
  significant_wave_height_m: number;
  swell_period_seconds: number;
  wind_speed_knots: number;
  squall_gust_knots: number;
  fishermen_safety_status: string;
  squall_eta_window: string;
  marine_hazard_bulletins: string[];
  ai_maritime_advisory: string;
  provider: string;
  latency_ms?: number;
};

type CoastalSquallModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const COASTAL_ZONES = [
  { id: "Mumbai Coast", label: "Mumbai & Konkan Coast" },
  { id: "Gujarat Gulf of Khambhat", label: "Gujarat & Gulf of Khambhat" },
  { id: "Kerala Malabar Coast", label: "Kerala & Malabar Coast" },
  { id: "Tamil Nadu Coromandel", label: "Tamil Nadu & Coromandel Coast" },
  { id: "Odisha Paradip Coast", label: "Odisha & Paradip Outer Roads" },
  { id: "Bengal Digha Coast", label: "West Bengal & Digha Coast" },
];

const CRAFT_OPTIONS = [
  { id: "traditional_motorized", label: "Motorized Fiber Craft (FRP, OBM)" },
  { id: "non_motorized_catamaran", label: "Non-Motorized Traditional Catamaran" },
  { id: "mechanized_trawler", label: "Mechanized Gillnetter / Trawler (IBM)" },
  { id: "deep_sea_longliner", label: "Deep-Sea Commercial Longliner" },
];

const DISTANCE_OPTIONS = [
  { dist: 5, label: "5 NM (Nearshore)" },
  { dist: 12, label: "12 NM (Territorial)" },
  { dist: 25, label: "25 NM (Contiguous)" },
  { dist: 45, label: "45 NM (EEZ Deep Sea)" },
];

export default function CoastalSquallModal({ city, onClose }: CoastalSquallModalProps) {
  const [coastalZone, setCoastalZone] = useState<string>("Mumbai Coast");
  const [craftType, setCraftType] = useState<string>("traditional_motorized");
  const [distanceNm, setDistanceNm] = useState<number>(12);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CoastalSquallResponse | null>(null);

  const fetchSquallAdvisory = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/coastal-squall`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          coastal_zone: coastalZone,
          craft_type: craftType,
          distance_offshore_nm: distanceNm,
        }),
      });
      if (!res.ok) {
        throw new Error(`Coastal squall evaluation failed (${res.status})`);
      }
      const json = (await res.json()) as CoastalSquallResponse;
      setData(json);
    } catch (err) {
      console.error("Coastal Squall error:", err);
      setError(err instanceof Error ? err.message : "Failed to load marine advisory");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSquallAdvisory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getSafetyBadgeClass = (status: string) => {
    switch (status) {
      case "SAFE_FOR_ALL":
        return "hazard-badge-clear";
      case "CAUTION_SMALL_CRAFT":
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
            <div className="utility-tool-circle-btn coastal-squall-circle-btn" style={{ width: 36, height: 36 }}>
              <Anchor size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Ship size={13} className="inline-icon" /> COASTAL FISHERIES & SEA STATE TELEMETRY
              </div>
              <h2>Coastal Squall & Small Craft Sea State Advisory</h2>
              <p className="utility-modal-subtitle">
                Douglas sea state scale, wave heights, and convective squall approach warnings for {coastalZone}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Coastal Squall Advisory"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="coast-select">Coastal Zone</label>
            <UtilityCustomSelect
              id="coast-select"
              value={coastalZone}
              options={COASTAL_ZONES}
              onChange={setCoastalZone}
              ariaLabel="Select coastal zone"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="craft-select">Vessel Class</label>
            <UtilityCustomSelect
              id="craft-select"
              value={craftType}
              options={CRAFT_OPTIONS}
              onChange={setCraftType}
              ariaLabel="Select vessel class"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Offshore Distance</label>
            <div className="utility-chips-row">
              {DISTANCE_OPTIONS.map((d) => (
                <button
                  key={d.dist}
                  type="button"
                  className={`utility-chip-btn ${distanceNm === d.dist ? "selected" : ""}`}
                  onClick={() => setDistanceNm(d.dist)}
                >
                  {d.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-refresh-btn"
              onClick={fetchSquallAdvisory}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              <span>Broadcast</span>
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
                  <Waves size={16} className="inline-icon" /> Sea State: Douglas Scale {data.sea_state_code} ({data.sea_state_label})
                  <span className={`utility-status-pill ${getSafetyBadgeClass(data.fishermen_safety_status)}`} style={{ marginLeft: 8 }}>
                    {data.fishermen_safety_status.replace(/_/g, " ")}
                  </span>
                </div>
              </div>

              {/* Status Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Waves size={13} className="inline-icon" /> Significant Wave Height
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.significant_wave_height_m}</span>
                    <span className="utility-metric-unit">meters</span>
                    <span className="utility-status-pill hazard-badge-caution">
                      {data.swell_period_seconds}s SWELL
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Dominant wave energy period: {data.swell_period_seconds} seconds
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Wind size={13} className="inline-icon" /> Coastal Wind & Gusts
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.wind_speed_knots}</span>
                    <span className="utility-metric-unit">kts</span>
                    <span className="utility-status-pill hazard-badge-critical">
                      GUSTS {data.squall_gust_knots} KTS
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Beaufort Force: {Math.min(12, Math.round(data.wind_speed_knots / 3.5))}
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <AlertTriangle size={13} className="inline-icon" /> Squall Approach ETA
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val" style={{ fontSize: "13px" }}>
                      {data.squall_eta_window}
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Maintain continuous listening watch on VHF Marine Channel 16
                  </div>
                </div>
              </div>

              {/* AI Maritime Advisory */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    FISHERIES WEATHER BROADCAST
                  </div>
                  <span className="utility-provider-badge">
                    "Powered by ClimaCore Intelligence"
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_maritime_advisory}</p>
              </div>

              {/* Bulletins */}
              <div className="utility-section-title">
                Navigational Safety Bulletins & Bar Warnings
              </div>
              <div className="utility-checklist-grid">
                {data.marine_hazard_bulletins.map((rec, i) => (
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
            <ShieldCheck size={14} className="inline-icon" /> Grounded in INCOIS ocean state forecasts & IMD coastal radar telemetry
          </div>
        </div>
      </div>
    </div>
  );
}
