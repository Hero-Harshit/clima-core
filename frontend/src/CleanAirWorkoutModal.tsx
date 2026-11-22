import { useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  Dumbbell,
  RefreshCw,
  ShieldAlert,
  Thermometer,
  Wind,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type AirQualityHourEstimate = {
  time_slot: string;
  aqi: number;
  pm25: number;
  inversion_trapping: boolean;
  athletic_score: number;
  is_optimal: boolean;
};

type CleanAirWorkoutResponse = {
  city: string;
  activity_type: string;
  athletic_suitability_score: number;
  status_level: string;
  recommended_window: string;
  inversion_status: string;
  current_aqi: number;
  current_pm25: number;
  wbgt_thermal_strain_c: number;
  hourly_forecast: AirQualityHourEstimate[];
  coaching_guidance: string;
  respiratory_precautions: string[];
  indoor_alternative: string;
  provider: string;
  latency_ms?: number;
};

type CleanAirWorkoutModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

const ACTIVITY_OPTIONS = [
  { id: "running", label: "Running / Jogging", desc: "High tidal breathing volume" },
  { id: "cycling", label: "Road Cycling", desc: "Extended aerobic exposure & windage" },
  { id: "brisk_walking", label: "Brisk Walking", desc: "Moderate cardiovascular load" },
  { id: "hiit", label: "Outdoor Functional HIIT", desc: "Peak respiratory rate spikes" },
];

const INTENSITY_OPTIONS = [
  { id: "light", label: "Zone 1-2 (Active Recovery)", desc: "Gentle aerobic session" },
  { id: "moderate", label: "Zone 3 (Aerobic Base Training)", desc: "Sustained cardiovascular pace" },
  { id: "vigorous", label: "Zone 4-5 (Threshold / VO2 Max)", desc: "High cardiovascular exertion" },
];

const WINDOW_OPTIONS = [
  { id: "any", label: "Auto-Detect Best Window", desc: "Select lowest PM2.5 period" },
  { id: "early_morning", label: "Early Morning Focus", desc: "Dawn to 08:00 AM" },
  { id: "evening", label: "Late Afternoon / Evening", desc: "17:00 PM onwards" },
];

export default function CleanAirWorkoutModal({ city, onClose }: CleanAirWorkoutModalProps) {
  const [activityType, setActivityType] = useState<string>("running");
  const [intensity, setIntensity] = useState<string>("moderate");
  const [preferredTime, setPreferredTime] = useState<string>("any");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<CleanAirWorkoutResponse | null>(null);

  const fetchWorkoutAnalysis = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/clean-air-workout`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          activity_type: activityType,
          intensity,
          preferred_time: preferredTime,
        }),
      });
      if (!res.ok) {
        throw new Error(`Clean air workout calculation failed (${res.status})`);
      }
      const json = (await res.json()) as CleanAirWorkoutResponse;
      setData(json);
    } catch (err) {
      console.error("Clean Air Workout error:", err);
      setError(err instanceof Error ? err.message : "Failed to load athletic air analysis");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchWorkoutAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "OPTIMAL":
        return "hazard-badge-clear";
      case "ACCEPTABLE":
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
            <div className="utility-tool-circle-btn workout-circle-btn" style={{ width: 36, height: 36 }}>
              <Activity size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Activity size={13} className="inline-icon" /> ATHLETIC PHYSIOLOGY & AIR QUALITY
              </div>
              <h2>Clean Air Athletic Window</h2>
              <p className="utility-modal-subtitle">
                Atmospheric inversion dissipation, particulate trapping, and outdoor fitness radar for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Clean Air Athletic Window"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="workout-activity">Sport Activity</label>
            <UtilityCustomSelect
              id="workout-activity"
              value={activityType}
              options={ACTIVITY_OPTIONS}
              onChange={setActivityType}
              ariaLabel="Select sport activity"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="workout-intensity">Intensity Zone</label>
            <UtilityCustomSelect
              id="workout-intensity"
              value={intensity}
              options={INTENSITY_OPTIONS}
              onChange={setIntensity}
              ariaLabel="Select workout intensity"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="workout-time">Target Window</label>
            <UtilityCustomSelect
              id="workout-time"
              value={preferredTime}
              options={WINDOW_OPTIONS}
              onChange={setPreferredTime}
              ariaLabel="Select target workout window"
            />
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-action-btn"
              onClick={fetchWorkoutAnalysis}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "spin-icon" : ""} />
              <span>Radar Scan</span>
            </button>
          </div>
        </div>

        {/* Body */}
        <div className="utility-modal-body">
          {error && (
            <div className="utility-error-banner">
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}

          {loading && !data && (
            <div className="utility-loading-state">
              <RefreshCw size={24} className="spin-icon" />
              <p>Simulating nocturnal inversion layers, turbulent convective mixing, and lung ventilation...</p>
            </div>
          )}

          {data && (
            <>
              {/* Score & Recommended Window Card */}
              <div className="utility-score-grid">
                <div className="utility-score-dial-card">
                  <div className="utility-dial-wrapper">
                    <span className="utility-dial-val">{data.athletic_suitability_score}</span>
                    <span className="utility-dial-max">/100</span>
                  </div>
                  <div className="utility-dial-caption">
                    <span className="utility-dial-title">Athletic Index</span>
                    <span className={`utility-status-pill ${getStatusBadgeClass(data.status_level)}`}>
                      {data.status_level.replace("_", " ")}
                    </span>
                  </div>
                </div>

                <div className="utility-stats-cards">
                  <div className="utility-stat-chip highlight-chip">
                    <Clock size={16} className="stat-icon-green" />
                    <div>
                      <div className="utility-stat-num">{data.recommended_window}</div>
                      <div className="utility-stat-label">Optimal Training Window</div>
                    </div>
                  </div>

                  <div className="utility-stat-chip">
                    <Wind size={16} className="stat-icon-teal" />
                    <div>
                      <div className="utility-stat-num">AQI {data.current_aqi}</div>
                      <div className="utility-stat-label">PM2.5: {data.current_pm25} µg/m³</div>
                    </div>
                  </div>

                  <div className="utility-stat-chip">
                    <Thermometer size={16} className="stat-icon-orange" />
                    <div>
                      <div className="utility-stat-num">{data.wbgt_thermal_strain_c}°C</div>
                      <div className="utility-stat-label">WBGT Thermal Strain</div>
                    </div>
                  </div>

                  <div className="utility-stat-chip">
                    <ShieldAlert size={16} className="stat-icon-blue" />
                    <div>
                      <div className="utility-stat-num-small">
                        {data.inversion_status === "SURFACE_TRAPPED" ? "Trapping Active" : "Ventilated"}
                      </div>
                      <div className="utility-stat-label">Inversion Boundary Layer</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Hourly Inversion & Air Quality Forecast */}
              <div className="utility-section-card">
                <div className="utility-card-kicker">HOURLY DISPERSION & FITNESS FORECAST</div>
                <div className="utility-hourly-grid">
                  {data.hourly_forecast.map((slot, idx) => (
                    <div
                      key={idx}
                      className={`utility-hour-card ${slot.is_optimal ? "optimal-window-card" : ""}`}
                    >
                      <div className="utility-hour-header">
                        <span className="utility-hour-time">{slot.time_slot}</span>
                        {slot.is_optimal && (
                          <span className="optimal-badge">
                            <CheckCircle2 size={11} /> OPTIMAL
                          </span>
                        )}
                      </div>

                      <div className="utility-hour-metrics">
                        <div className="hour-metric-row">
                          <span className="metric-name">AQI</span>
                          <span className="metric-val">{slot.aqi}</span>
                        </div>
                        <div className="hour-metric-row">
                          <span className="metric-name">PM2.5</span>
                          <span className="metric-val">{slot.pm25} µg/m³</span>
                        </div>
                        <div className="hour-metric-row">
                          <span className="metric-name">Inversion</span>
                          <span className={`metric-val ${slot.inversion_trapping ? "trapped-text" : "clear-text"}`}>
                            {slot.inversion_trapping ? "Trapped" : "Clear"}
                          </span>
                        </div>
                      </div>

                      <div className="utility-hour-score">
                        <span>Fitness Score</span>
                        <strong>{slot.athletic_score}/100</strong>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Precautions & Indoor Alternative Dual Grid */}
              <div className="utility-dual-grid">
                <div className="utility-section-card">
                  <div className="utility-card-kicker">RESPIRATORY SAFETY PRECAUTIONS</div>
                  <ul className="utility-checklist">
                    {data.respiratory_precautions.map((item, idx) => (
                      <li key={idx}>
                        <CheckCircle2 size={15} className="check-icon" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="utility-section-card">
                  <div className="utility-card-kicker">
                    <Dumbbell size={13} className="inline-icon" /> INDOOR SANCTUARY ALTERNATIVE
                  </div>
                  <p className="utility-indoor-prose">{data.indoor_alternative}</p>
                </div>
              </div>

              {/* ClimaCore Coaching Briefing */}
              <div className="utility-section-card ai-concierge-card">
                <div className="utility-card-kicker">
                  CLIMACORE ATHLETIC COACH PHYSIOLOGY BRIEFING
                </div>
                <p className="utility-ai-prose">{data.coaching_guidance}</p>
                <div className="utility-provider-badge">
                  <span>Powered by ClimaCore Intelligence</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="utility-modal-footer">
          <div className="utility-footer-meta">
            <span>Atmospheric Physics: Boundary-Layer Convective Mixing & Respiratory Models</span>
            {data?.latency_ms && <span> · Latency: {data.latency_ms} ms</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
