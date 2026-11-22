import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Gauge,
  Navigation,
  RefreshCw,
  ShieldCheck,
  Thermometer,
  Truck,
  Wind,
  X,
  Droplets,
} from "lucide-react";
import UtilityCustomSelect, { UtilitySelectOption } from "./UtilityCustomSelect";

type CorridorSegmentRisk = {
  segment_name: string;
  hazard_type: string;
  risk_level: string;
  advisory: string;
  speed_reduction_pct: number;
};

type CorridorRadarResponse = {
  city: string;
  origin: string;
  destination: string;
  vehicle_type: string;
  corridor_risk_score: number;
  status_level: string;
  estimated_delay_mins: number;
  surface_runoff_mm: number;
  max_wind_gust_kmh: number;
  asphalt_temp_c: number;
  operational_guidance: string;
  segments: CorridorSegmentRisk[];
  vehicle_recommendations: string[];
  ai_insights: string;
  provider: string;
  latency_ms?: number;
};

type CorridorRadarModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const VEHICLE_OPTIONS: UtilitySelectOption[] = [
  { id: "two_wheeler", label: "Two-Wheeler (Bike/EV)", desc: "High crosswind & slip sensitivity" },
  { id: "ev_van", label: "Electric Delivery Van", desc: "Battery thermal & traction control" },
  { id: "light_commercial", label: "Light Commercial (Pickup)", desc: "Braking & underpass clearance" },
  { id: "heavy_truck", label: "Heavy Freight Truck", desc: "Bridge stability & jackknife safety" },
];

export default function CorridorRadarModal({ city, onClose }: CorridorRadarModalProps) {
  const [origin, setOrigin] = useState<string>("City Center Hub");
  const [destination, setDestination] = useState<string>("East Cargo Terminal");
  const [vehicleType, setVehicleType] = useState<string>("two_wheeler");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CorridorRadarResponse | null>(null);

  const fetchCorridorAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/corridor-radar`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          origin,
          destination,
          vehicle_type: vehicleType,
        }),
      });
      if (!res.ok) {
        throw new Error(`Corridor radar analysis failed (${res.status})`);
      }
      const json = (await res.json()) as CorridorRadarResponse;
      setData(json);
    } catch (err) {
      console.error("Corridor Radar error:", err);
      setError(err instanceof Error ? err.message : "Failed to load corridor analysis");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCorridorAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "CLEAR":
        return "hazard-badge-clear";
      case "CAUTION":
        return "hazard-badge-caution";
      default:
        return "hazard-badge-critical";
    }
  };

  const getSegmentHazardIcon = (hazard: string) => {
    switch (hazard) {
      case "WATERLOGGING":
        return <Droplets size={14} className="hazard-icon-rain" />;
      case "CROSSWIND":
        return <Wind size={14} className="hazard-icon-wind" />;
      case "ASPHALT_HEAT":
        return <Thermometer size={14} className="hazard-icon-heat" />;
      default:
        return <CheckCircle2 size={14} className="hazard-icon-clear" />;
    }
  };

  return (
    <div className="utility-modal-scrim" onClick={onClose} role="dialog" aria-modal="true">
      <div className="utility-modal-window" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn corridor-circle-btn" style={{ width: 36, height: 36 }}>
              <Truck size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Truck size={13} className="inline-icon" /> LOGISTICS & FREIGHT TELEMETRY
              </div>
              <h2>Delivery Corridor Radar</h2>
              <p className="utility-modal-subtitle">
                Live route waterlogging, flyover crosswinds, and asphalt thermal stress navigator for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Delivery Corridor Radar"
          >
            <X size={18} />
          </button>
        </div>

        {/* Route Controls */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="corridor-origin">Origin</label>
            <div className="utility-input-wrapper">
              <Navigation size={14} className="utility-input-icon" />
              <input
                id="corridor-origin"
                type="text"
                value={origin}
                onChange={(e) => setOrigin(e.target.value)}
                placeholder="e.g. City Center Hub"
              />
            </div>
          </div>

          <div className="utility-control-group">
            <label htmlFor="corridor-dest">Destination</label>
            <div className="utility-input-wrapper">
              <Navigation size={14} className="utility-input-icon" />
              <input
                id="corridor-dest"
                type="text"
                value={destination}
                onChange={(e) => setDestination(e.target.value)}
                placeholder="e.g. Cargo Terminal"
              />
            </div>
          </div>

          <div className="utility-control-group vehicle-select-group">
            <label htmlFor="corridor-vehicle">Fleet Vehicle</label>
            <UtilityCustomSelect
              id="corridor-vehicle"
              value={vehicleType}
              options={VEHICLE_OPTIONS}
              onChange={setVehicleType}
              ariaLabel="Select fleet vehicle"
            />
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-action-btn"
              onClick={fetchCorridorAnalysis}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "spin-icon" : ""} />
              <span>Analyze Route</span>
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="utility-modal-body">
          {error && (
            <div className="utility-error-banner">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}

          {loading && !data && (
            <div className="utility-loading-state">
              <RefreshCw size={24} className="spin-icon" />
              <p>Evaluating waypoint hydrology, wind shear, and asphalt heat...</p>
            </div>
          )}

          {data && (
            <>
              {/* Telemetry Summary Card */}
              <div className="utility-score-grid">
                <div className="utility-score-dial-card">
                  <div className="utility-dial-wrapper">
                    <span className="utility-dial-val">{data.corridor_risk_score}</span>
                    <span className="utility-dial-max">/100</span>
                  </div>
                  <div className="utility-dial-caption">
                    <span className="utility-dial-title">Corridor Risk</span>
                    <span className={`utility-status-pill ${getStatusBadgeClass(data.status_level)}`}>
                      {data.status_level.replace("_", " ")}
                    </span>
                  </div>
                </div>

                <div className="utility-stats-cards">
                  <div className="utility-stat-chip">
                    <Clock size={16} className="stat-icon-blue" />
                    <div>
                      <div className="utility-stat-num">+{data.estimated_delay_mins} min</div>
                      <div className="utility-stat-label">Weather Delay</div>
                    </div>
                  </div>

                  <div className="utility-stat-chip">
                    <Droplets size={16} className="stat-icon-cyan" />
                    <div>
                      <div className="utility-stat-num">{data.surface_runoff_mm} mm</div>
                      <div className="utility-stat-label">Surface Runoff</div>
                    </div>
                  </div>

                  <div className="utility-stat-chip">
                    <Wind size={16} className="stat-icon-teal" />
                    <div>
                      <div className="utility-stat-num">{data.max_wind_gust_kmh} km/h</div>
                      <div className="utility-stat-label">Peak Gust</div>
                    </div>
                  </div>

                  <div className="utility-stat-chip">
                    <Thermometer size={16} className="stat-icon-orange" />
                    <div>
                      <div className="utility-stat-num">{data.asphalt_temp_c}°C</div>
                      <div className="utility-stat-label">Asphalt Temp</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Operational Guidance */}
              <div className="utility-guidance-card">
                <div className="utility-guidance-header">
                  <ShieldCheck size={16} className="guidance-icon" />
                  <strong>DISPATCH & ROUTING DIRECTIVE</strong>
                </div>
                <p>{data.operational_guidance}</p>
              </div>

              {/* Segments Breakdown */}
              <div className="utility-section-card">
                <div className="utility-card-kicker">WAYPOINT HAZARD BREAKDOWN</div>
                <div className="utility-segments-list">
                  {data.segments.map((seg, idx) => (
                    <div key={idx} className="utility-segment-row">
                      <div className="utility-segment-meta">
                        <div className="utility-segment-title-wrap">
                          {getSegmentHazardIcon(seg.hazard_type)}
                          <span className="utility-segment-title">{seg.segment_name}</span>
                        </div>
                        <div className="utility-segment-badges">
                          <span className="utility-hazard-tag">{seg.hazard_type.replace("_", " ")}</span>
                          <span className={`utility-risk-pill ${getStatusBadgeClass(seg.risk_level)}`}>
                            {seg.risk_level}
                          </span>
                          {seg.speed_reduction_pct > 0 && (
                            <span className="utility-speed-pill">
                              <Gauge size={12} /> -{seg.speed_reduction_pct}% Spd
                            </span>
                          )}
                        </div>
                      </div>
                      <p className="utility-segment-advisory">{seg.advisory}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Fleet Recommendations & AI Insights Grid */}
              <div className="utility-dual-grid">
                <div className="utility-section-card">
                  <div className="utility-card-kicker">VEHICLE CLASS ADVISORIES</div>
                  <ul className="utility-checklist">
                    {data.vehicle_recommendations.map((tip, idx) => (
                      <li key={idx}>
                        <CheckCircle2 size={15} className="check-icon" />
                        <span>{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="utility-section-card ai-concierge-card">
                  <div className="utility-card-kicker">
                    CLIMACORE LOGISTICS INTELLIGENCE
                  </div>
                  <p className="utility-ai-prose">{data.ai_insights}</p>
                  <div className="utility-provider-badge">
                    <span>Powered by ClimaCore Intelligence</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="utility-modal-footer">
          <div className="utility-footer-meta">
            <span>Spatial Engine: High-Resolution ClimaCore Ingestion</span>
            {data?.latency_ms && <span> · Latency: {data.latency_ms} ms</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
