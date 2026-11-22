import { useEffect, useState } from "react";
import {
  AlertTriangle,
  Car,
  CheckCircle2,
  Clock,
  RefreshCw,
  ShieldCheck,
  Train,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type MetroRoadComparison = {
  mode_name: string;
  travel_time_mins: number;
  delay_variance_mins: number;
  water_hazard_status: string;
  comfort_reliability_rating: string;
  advisory: string;
};

type MetroVsRoadResponse = {
  city: string;
  commute_corridor: string;
  departure_window: string;
  recommended_mode: string;
  confidence_score: number;
  metro_delay_mins: number;
  road_delay_mins: number;
  waterlogging_depth_max_cm: number;
  road_gridlock_index: number;
  metro_egress_safety: string;
  route_comparisons: MetroRoadComparison[];
  driver_transit_advisory: string;
  ai_transit_verdict: string;
  provider: string;
  latency_ms?: number;
};

type MetroVsRoadModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

const CORRIDOR_OPTIONS = [
  { id: "central_to_suburban", label: "Central City to Suburban Radial" },
  { id: "ring_road_arterial", label: "Arterial Ring Road Circular Bypass" },
  { id: "tech_corridor_bypass", label: "Tech Corridor Expressway to IT Hub" },
];

const WINDOW_OPTIONS = [
  { id: "peak_morning", label: "Morning Peak (08:30 - 10:30 AM)" },
  { id: "midday_lull", label: "Midday Lull (01:00 - 03:00 PM)" },
  { id: "peak_evening", label: "Evening Peak (05:30 - 08:30 PM)" },
  { id: "night_shift", label: "Late Night Shift (10:00 PM - 01:00 AM)" },
];

const VEHICLE_OPTIONS = [
  { id: "car_cab", label: "Private Car / App Cab" },
  { id: "two_wheeler", label: "Two-Wheeler / Scooter" },
  { id: "public_bus", label: "Public Surface Bus" },
];

export default function MetroVsRoadModal({ city, onClose }: MetroVsRoadModalProps) {
  const [corridor, setCorridor] = useState<string>("central_to_suburban");
  const [departureWindow, setDepartureWindow] = useState<string>("peak_evening");
  const [vehicleMode, setVehicleMode] = useState<string>("car_cab");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<MetroVsRoadResponse | null>(null);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/metro-vs-road`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          commute_corridor: corridor,
          departure_window: departureWindow,
          personal_vehicle: vehicleMode,
        }),
      });
      if (!res.ok) {
        throw new Error(`Metro vs Road evaluation failed (${res.status})`);
      }
      const json = (await res.json()) as MetroVsRoadResponse;
      setData(json);
    } catch (err) {
      console.error("Metro vs Road error:", err);
      setError(err instanceof Error ? err.message : "Failed to load transit comparison");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComparison();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getRatingPillClass = (rating: string) => {
    switch (rating) {
      case "EXCELLENT":
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
            <div className="utility-tool-circle-btn metro-road-circle-btn" style={{ width: 36, height: 36 }}>
              <Train size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Train size={13} className="inline-icon" /> TRANSIT INTUITION & DELUGE ROUTING
              </div>
              <h2>Metro vs Road Flood Decision Engine</h2>
              <p className="utility-modal-subtitle">
                Comparative transit evaluation under street waterlogging and arterial traffic stress for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Metro vs Road Decision Engine"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="corridor-select">Commute Corridor</label>
            <UtilityCustomSelect
              id="corridor-select"
              value={corridor}
              options={CORRIDOR_OPTIONS}
              onChange={setCorridor}
              ariaLabel="Select commute corridor"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="window-select">Departure Window</label>
            <UtilityCustomSelect
              id="window-select"
              value={departureWindow}
              options={WINDOW_OPTIONS}
              onChange={setDepartureWindow}
              ariaLabel="Select departure window"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="vehicle-select">Surface Mode</label>
            <UtilityCustomSelect
              id="vehicle-select"
              value={vehicleMode}
              options={VEHICLE_OPTIONS}
              onChange={setVehicleMode}
              ariaLabel="Select surface vehicle mode"
            />
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-refresh-btn"
              onClick={fetchComparison}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              <span>Evaluate</span>
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
                  <Train size={16} className="inline-icon" /> Recommended Mode: {data.recommended_mode.replace(/_/g, " ")}
                  <span className="utility-status-pill hazard-badge-clear" style={{ marginLeft: 8 }}>
                    {data.confidence_score}% Confidence
                  </span>
                </div>
              </div>

              {/* Status Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Train size={13} className="inline-icon" /> Metro Transit Delay
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">+{data.metro_delay_mins}</span>
                    <span className="utility-metric-unit">min</span>
                    <span className="utility-status-pill hazard-badge-clear">INSULATED</span>
                  </div>
                  <div className="utility-metric-sub">
                    Elevated track & tunnel dry. Concourse safety: {data.metro_egress_safety}
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Car size={13} className="inline-icon" /> Surface Road Delay
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">+{data.road_delay_mins}</span>
                    <span className="utility-metric-unit">min</span>
                    <span className="utility-status-pill hazard-badge-critical">
                      {data.road_gridlock_index}/100 GRIDLOCK
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Underpass & curb runoff reaches {data.waterlogging_depth_max_cm} cm
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Clock size={13} className="inline-icon" /> Net Time Differential
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val">
                      Save {Math.max(0, data.road_delay_mins - data.metro_delay_mins)} mins on Metro
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Avoid arterial bottleneck crawl during active monsoon downpour
                  </div>
                </div>
              </div>

              {/* AI Transit Verdict */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    TRANSIT DECISION VERDICT
                  </div>
                  <span className="utility-provider-badge">
                    "Powered by ClimaCore Intelligence"
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_transit_verdict}</p>
              </div>

              {/* Route Mode Comparisons */}
              <div className="utility-section-title">
                Detailed Mode Reliability & Hazard Comparison
              </div>
              <div className="utility-segments-list">
                {data.route_comparisons.map((c, i) => (
                  <div key={i} className="utility-segment-row">
                    <div className="utility-segment-left">
                      <div className="utility-segment-hazard-badge">
                        {c.mode_name.includes("Metro") ? <Train size={14} /> : <Car size={14} />}
                      </div>
                      <div className="utility-segment-info">
                        <div className="utility-segment-title">{c.mode_name}</div>
                        <div className="utility-segment-advisory">{c.advisory}</div>
                        <div style={{ fontSize: "11.5px", color: "#556b5a", marginTop: 3 }}>
                          {c.water_hazard_status}
                        </div>
                      </div>
                    </div>
                    <div className="utility-segment-right">
                      <span className="utility-speed-pill">{c.travel_time_mins} min total</span>
                      <span className={`utility-status-pill ${getRatingPillClass(c.comfort_reliability_rating)}`}>
                        {c.comfort_reliability_rating}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Driver & Transit Guidance */}
              <div className="utility-section-title">
                Commuter Advisory & Detour Guidance
              </div>
              <div className="utility-checklist-grid">
                <div className="utility-checklist-item">
                  <CheckCircle2 size={15} className="utility-check-icon" />
                  <span>{data.driver_transit_advisory}</span>
                </div>
                <div className="utility-checklist-item">
                  <CheckCircle2 size={15} className="utility-check-icon" />
                  <span>Park personal vehicles at elevated station park-and-ride lots to avoid street submergence.</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Action Footer */}
        <div className="utility-action-footer">
          <div className="utility-footer-meta">
            <ShieldCheck size={14} className="inline-icon" /> Grounded in municipal flood telemetry & metro schedule feeds
          </div>
        </div>
      </div>
    </div>
  );
}
