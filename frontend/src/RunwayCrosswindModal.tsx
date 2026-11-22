import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Compass,
  Navigation,
  Plane,
  RefreshCw,
  ShieldCheck,
  Wind,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type RunwayCrosswindResponse = {
  airport_identifier: string;
  active_runway_designator: string;
  surface_wind_knots: number;
  surface_wind_dir_deg: number;
  crosswind_component_knots: number;
  headwind_tailwind_component_knots: number;
  crosswind_safety_status: string;
  convective_divergence_hazard: string;
  go_around_probability_pct: number;
  air_density_altitude_ft: number;
  pilot_atc_advisories: string[];
  ai_aviation_guidance: string;
  provider: string;
  latency_ms?: number;
};

type RunwayCrosswindModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const AIRPORT_OPTIONS = [
  { id: "BOM", code: "BOM", name: "Mumbai (CSMI Airport - BOM / VABB)", label: "Mumbai (CSMI Airport - BOM / VABB)", defaultRwy: 270 },
  { id: "DEL", code: "DEL", name: "Delhi (Indira Gandhi Intl - DEL / VIDP)", label: "Delhi (Indira Gandhi Intl - DEL / VIDP)", defaultRwy: 280 },
  { id: "BLR", code: "BLR", name: "Bengaluru (Kempegowda Intl - BLR / VOBL)", label: "Bengaluru (Kempegowda Intl - BLR / VOBL)", defaultRwy: 270 },
  { id: "CCU", code: "CCU", name: "Kolkata (Netaji Subhash Intl - CCU / VECC)", label: "Kolkata (Netaji Subhash Intl - CCU / VECC)", defaultRwy: 190 },
  { id: "MAA", code: "MAA", name: "Chennai (Chennai Intl - MAA / VOMM)", label: "Chennai (Chennai Intl - MAA / VOMM)", defaultRwy: 250 },
  { id: "HYD", code: "HYD", name: "Hyderabad (Rajiv Gandhi Intl - HYD / VOHS)", label: "Hyderabad (Rajiv Gandhi Intl - HYD / VOHS)", defaultRwy: 270 },
  { id: "PNQ", code: "PNQ", name: "Pune (Pune Airport - PNQ / VAPO)", label: "Pune (Pune Airport - PNQ / VAPO)", defaultRwy: 280 },
  { id: "AMD", code: "AMD", name: "Ahmedabad (Sardar Patel Intl - AMD / VAAH)", label: "Ahmedabad (Sardar Patel Intl - AMD / VAAH)", defaultRwy: 230 },
];

const AIRCRAFT_OPTIONS = [
  { id: "code_c_narrowbody_a320_b737", label: "Code C Narrowbody (Airbus A320 / Boeing 737)" },
  { id: "code_e_widebody_b777_a350", label: "Code E Widebody (Boeing 777 / Airbus A350)" },
  { id: "atr72_q400_turboprop", label: "Regional Turboprop (ATR 72-600 / Q400)" },
];

const SURFACE_OPTIONS = [
  { id: "dry_clean", label: "Dry & Clean Runway Friction" },
  { id: "damp_wet_water_patches", label: "Damp / Standing Water Patches" },
  { id: "standing_water_hydroplaning", label: "Contaminated (> 3mm Water, Hydroplaning Risk)" },
];

export default function RunwayCrosswindModal({ city, onClose }: RunwayCrosswindModalProps) {
  const [airportCode, setAirportCode] = useState<string>("BOM");
  const [runwayHeading, setRunwayHeading] = useState<number>(270);
  const [aircraftCategory, setAircraftCategory] = useState<string>("code_c_narrowbody_a320_b737");
  const [surfaceCondition, setSurfaceCondition] = useState<string>("dry_clean");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<RunwayCrosswindResponse | null>(null);

  const fetchAssessment = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/runway-crosswind`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          airport_code_or_city: airportCode,
          runway_heading_deg: runwayHeading,
          aircraft_category: aircraftCategory,
          runway_surface_condition: surfaceCondition,
        }),
      });
      if (!res.ok) {
        throw new Error(`Aviation radar service returned HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to calculate runway crosswind vectors.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchAssessment();
  }, [airportCode, runwayHeading, aircraftCategory, surfaceCondition]);

  const handleAirportChange = (code: string) => {
    setAirportCode(code);
    const match = AIRPORT_OPTIONS.find((a) => a.code === code);
    if (match) {
      setRunwayHeading(match.defaultRwy);
    }
  };

  const getSafetyBadgeClass = (status?: string) => {
    switch (status) {
      case "LIMIT_EXCEEDED_GO_AROUND_MANDATORY":
        return "hazard-badge-critical";
      case "CAUTION_APPROACHING_LIMITS":
        return "hazard-badge-warning";
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
        aria-labelledby="aviation-modal-title"
      >
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn runway-crosswind-circle-btn" style={{ width: 36, height: 36 }}>
              <Plane size={18} />
            </div>
            <div>
              <h2 id="aviation-modal-title" className="utility-modal-heading">
                Runway Crosswind & Convective Divergent Radar
              </h2>
              <p className="utility-modal-subtitle">
                Crosswind vectors, low-level wind shear divergence, and density altitude dispatch for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Runway Crosswind Modal"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="airport-select">Target Aerodrome</label>
            <UtilityCustomSelect
              id="airport-select"
              value={airportCode}
              options={AIRPORT_OPTIONS}
              onChange={(val) => handleAirportChange(val)}
              ariaLabel="Select target aerodrome"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="aircraft-select">Aircraft Fleet Type</label>
            <UtilityCustomSelect
              id="aircraft-select"
              value={aircraftCategory}
              options={AIRCRAFT_OPTIONS}
              onChange={setAircraftCategory}
              ariaLabel="Select aircraft fleet type"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="surface-select">Runway Surface State</label>
            <UtilityCustomSelect
              id="surface-select"
              value={surfaceCondition}
              options={SURFACE_OPTIONS}
              onChange={setSurfaceCondition}
              ariaLabel="Select runway surface condition"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Runway Heading</label>
            <div className="utility-chips-row">
              {[90, 140, 230, 270, 280].map((hdg) => (
                <button
                  key={hdg}
                  type="button"
                  className={`utility-chip-btn ${runwayHeading === hdg ? "selected" : ""}`}
                  onClick={() => setRunwayHeading(hdg)}
                >
                  RWY {Math.round(hdg / 10).toString().padStart(2, "0")} ({hdg}°)
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
              <span>Resolving runway magnetic vectors, microburst shear, and density altitude...</span>
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
                    <Compass size={14} /> Active Threshold
                  </div>
                  <div className="utility-kpi-value">
                    {data.active_runway_designator}
                  </div>
                  <div className="utility-kpi-meta">
                    Wind: {data.surface_wind_knots} kts @ {data.surface_wind_dir_deg}°
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Wind size={14} /> Crosswind Vector
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#0284c7" }}>
                    {data.crosswind_component_knots} kts
                  </div>
                  <div className="utility-kpi-meta">
                    <span className={`hazard-badge ${getSafetyBadgeClass(data.crosswind_safety_status)}`}>
                      {data.crosswind_safety_status.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Navigation size={14} /> Convective Divergence
                  </div>
                  <div className="utility-kpi-value" style={{ fontSize: 16, lineHeight: 1.25 }}>
                    {data.convective_divergence_hazard.replace(/_/g, " ")}
                  </div>
                  <div className="utility-kpi-meta">
                    Headwind: {data.headwind_tailwind_component_knots} kts
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Plane size={14} /> Missed Approach Risk
                  </div>
                  <div className="utility-kpi-value" style={{ color: data.go_around_probability_pct > 30 ? "#e11d48" : "#16a34a" }}>
                    {data.go_around_probability_pct}%
                  </div>
                  <div className="utility-kpi-meta">
                    Density Altitude: {data.air_density_altitude_ft.toLocaleString()} ft
                  </div>
                </div>
              </div>

              {/* AI Guidance */}
              <div className="utility-ai-guidance-card">
                <div className="utility-ai-header">
                  <span>Gemini Aviation Meteorology & Flight Dispatch Synthesis</span>
                  <span className="utility-provider-badge">{data.provider}</span>
                </div>
                <p className="utility-ai-text">{data.ai_aviation_guidance}</p>
              </div>

              {/* Advisories List */}
              <div className="utility-advisories-card">
                <h3 className="utility-section-title">
                  <ShieldCheck size={16} /> Pilot & Air Traffic Control Operating Advisories
                </h3>
                <ul className="utility-checklist">
                  {data.pilot_atc_advisories.map((item, idx) => (
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
