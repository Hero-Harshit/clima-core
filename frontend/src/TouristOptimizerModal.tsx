import { useEffect, useState } from "react";
import {
  Calendar,
  CheckCircle2,
  Clock,
  Compass,
  Footprints,
  MapPin,
  RefreshCw,
  Sparkles,
  Sun,
  Umbrella,
  X,
  Droplet,
} from "lucide-react";

type ItineraryHourSlot = {
  time_window: string;
  period_label: string;
  activity_title: string;
  venue_type: string;
  weather_suitability: string;
  temp_c: number;
  uv_index: number;
  aqi_category: string;
  recommendation: string;
};

type TouristOptimizerResponse = {
  city: string;
  travel_date: string;
  pace: string;
  comfort_score: number;
  overall_verdict: string;
  weather_summary: string;
  itinerary: ItineraryHourSlot[];
  packing_checklist: string[];
  local_hydrated_spots: string[];
  ai_concierge_briefing: string;
  provider: string;
  latency_ms?: number;
};

type TouristOptimizerModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const AVAILABLE_INTERESTS = [
  { id: "heritage", label: "Historical Monuments" },
  { id: "culinary", label: "Street Food & Dining" },
  { id: "parks", label: "Botanical Parks & Gardens" },
  { id: "museums", label: "Air-Conditioned Museums" },
  { id: "bazaar", label: "Open-Air Bazaars" },
];

export default function TouristOptimizerModal({ city, onClose }: TouristOptimizerModalProps) {
  const [travelDate, setTravelDate] = useState<string>("Today");
  const [pace, setPace] = useState<string>("balanced");
  const [selectedInterests, setSelectedInterests] = useState<string[]>(["heritage", "culinary", "parks"]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<TouristOptimizerResponse | null>(null);

  const toggleInterest = (id: string) => {
    if (selectedInterests.includes(id)) {
      if (selectedInterests.length > 1) {
        setSelectedInterests(selectedInterests.filter((x) => x !== id));
      }
    } else {
      setSelectedInterests([...selectedInterests, id]);
    }
  };

  const fetchOptimizedSchedule = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/tourist-optimizer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          travel_date: travelDate,
          interests: selectedInterests,
          pace,
        }),
      });
      if (!res.ok) {
        throw new Error(`Tourist schedule optimization failed (${res.status})`);
      }
      const json = (await res.json()) as TouristOptimizerResponse;
      setData(json);
    } catch (err) {
      console.error("Tourist Optimizer error:", err);
      setError(err instanceof Error ? err.message : "Failed to load tourist schedule");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchOptimizedSchedule();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  const getSuitabilityClass = (suitability: string) => {
    switch (suitability) {
      case "EXCELLENT":
        return "suitability-pill-excellent";
      case "ACCEPTABLE":
        return "suitability-pill-acceptable";
      default:
        return "suitability-pill-unfavorable";
    }
  };

  const getVenueBadge = (venue: string) => {
    switch (venue) {
      case "OUTDOOR":
        return <span className="utility-venue-badge venue-outdoor">Outdoor Excursion</span>;
      case "INDOOR":
        return <span className="utility-venue-badge venue-indoor">AC Indoor Sanctuary</span>;
      default:
        return <span className="utility-venue-badge venue-sheltered">Covered / Semi-Shaded</span>;
    }
  };

  return (
    <div className="utility-modal-scrim" onClick={onClose} role="dialog" aria-modal="true">
      <div className="utility-modal-window" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-header-info">
            <div className="utility-modal-kicker">
              <Compass size={13} className="inline-icon" /> TOURISM & RECREATION PLANNING
            </div>
            <h2>Tourist Itinerary Optimizer</h2>
            <p className="utility-modal-subtitle">
              Diurnal weather curve, UV index, and heat avoidance scheduler for {city}
            </p>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Tourist Itinerary Optimizer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="tourist-date">Visit Date</label>
            <div className="utility-input-wrapper">
              <Calendar size={14} className="utility-input-icon" />
              <select
                id="tourist-date"
                value={travelDate}
                onChange={(e) => setTravelDate(e.target.value)}
                className="utility-select"
              >
                <option value="Today">Today (Live Forecast)</option>
                <option value="Tomorrow">Tomorrow</option>
                <option value="Weekend">Upcoming Weekend</option>
              </select>
            </div>
          </div>

          <div className="utility-control-group">
            <label htmlFor="tourist-pace">Tour Pace</label>
            <div className="utility-input-wrapper">
              <Footprints size={14} className="utility-input-icon" />
              <select
                id="tourist-pace"
                value={pace}
                onChange={(e) => setPace(e.target.value)}
                className="utility-select"
              >
                <option value="relaxed">Relaxed (Shaded Leisures)</option>
                <option value="balanced">Balanced (Optimal Windows)</option>
                <option value="intensive">Intensive (Full Day Active)</option>
              </select>
            </div>
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Travel Interests</label>
            <div className="utility-chips-row">
              {AVAILABLE_INTERESTS.map((item) => {
                const isSelected = selectedInterests.includes(item.id);
                return (
                  <button
                    key={item.id}
                    type="button"
                    className={`utility-chip-btn ${isSelected ? "selected" : ""}`}
                    onClick={() => toggleInterest(item.id)}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>

          <button
            type="button"
            className="utility-action-btn"
            onClick={fetchOptimizedSchedule}
            disabled={loading}
          >
            <RefreshCw size={14} className={loading ? "spin-icon" : ""} />
            <span>Plan Itinerary</span>
          </button>
        </div>

        {/* Body */}
        <div className="utility-modal-body">
          {error && (
            <div className="utility-error-banner">
              <X size={16} />
              <span>{error}</span>
            </div>
          )}

          {loading && !data && (
            <div className="utility-loading-state">
              <RefreshCw size={24} className="spin-icon" />
              <p>Analyzing diurnal temperature arcs, UV radiation indices, and comfort zones...</p>
            </div>
          )}

          {data && (
            <>
              {/* Comfort Score & Summary */}
              <div className="utility-score-grid">
                <div className="utility-score-dial-card">
                  <div className="utility-dial-wrapper">
                    <span className="utility-dial-val">{data.comfort_score}</span>
                    <span className="utility-dial-max">/100</span>
                  </div>
                  <div className="utility-dial-caption">
                    <span className="utility-dial-title">Comfort Index</span>
                    <span className="utility-status-pill hazard-badge-clear">
                      {data.overall_verdict}
                    </span>
                  </div>
                </div>

                <div className="utility-summary-card">
                  <div className="utility-card-kicker">ATMOSPHERIC RECONNAISSANCE</div>
                  <p className="utility-summary-text">{data.weather_summary}</p>
                </div>
              </div>

              {/* Hour-by-Hour Timeline */}
              <div className="utility-section-card">
                <div className="utility-card-kicker">CHRONOLOGICAL SIGHTSEEING SCHEDULE</div>
                <div className="utility-timeline-container">
                  {data.itinerary.map((slot, idx) => (
                    <div key={idx} className="utility-timeline-slot">
                      <div className="utility-slot-sidebar">
                        <span className="utility-slot-time">
                          <Clock size={12} className="inline-icon" /> {slot.time_window}
                        </span>
                        <span className="utility-slot-period">{slot.period_label}</span>
                      </div>

                      <div className="utility-slot-content">
                        <div className="utility-slot-header">
                          <h4 className="utility-slot-title">{slot.activity_title}</h4>
                          <div className="utility-slot-tags">
                            {getVenueBadge(slot.venue_type)}
                            <span className={`utility-suitability-badge ${getSuitabilityClass(slot.weather_suitability)}`}>
                              {slot.weather_suitability}
                            </span>
                          </div>
                        </div>

                        <div className="utility-slot-metrics">
                          <span className="slot-metric-pill">
                            <Sun size={13} /> {slot.temp_c}°C
                          </span>
                          <span className="slot-metric-pill">
                            UV Index: {slot.uv_index}
                          </span>
                          <span className="slot-metric-pill">
                            AQI: {slot.aqi_category}
                          </span>
                        </div>

                        <p className="utility-slot-advisory">{slot.recommendation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Dual Grid: Packing & Hydrated Hubs */}
              <div className="utility-dual-grid">
                <div className="utility-section-card">
                  <div className="utility-card-kicker">
                    <Umbrella size={13} className="inline-icon" /> RECOMMENDED PACKING CHECKLIST
                  </div>
                  <ul className="utility-checklist">
                    {data.packing_checklist.map((item, idx) => (
                      <li key={idx}>
                        <CheckCircle2 size={15} className="check-icon" />
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="utility-section-card">
                  <div className="utility-card-kicker">
                    <Droplet size={13} className="inline-icon" /> MUNICIPAL SHADE & HYDRATION HUBS
                  </div>
                  <ul className="utility-checklist">
                    {data.local_hydrated_spots.map((spot, idx) => (
                      <li key={idx}>
                        <MapPin size={15} className="location-icon" />
                        <span>{spot}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* AI Concierge Briefing */}
              <div className="utility-section-card ai-concierge-card">
                <div className="utility-card-kicker">
                  <Sparkles size={13} className="sparkle-icon" /> GEMINI AI CONCIERGE BRIEFING
                </div>
                <p className="utility-ai-prose">{data.ai_concierge_briefing}</p>
                <div className="utility-provider-badge">
                  <span>Concierge Model: {data.provider}</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="utility-modal-footer">
          <div className="utility-footer-meta">
            <span>Thermal Comfort Model: ClimaCore Wet-Bulb & Solar Flux Analysis</span>
            {data?.latency_ms && <span> · Latency: {data.latency_ms} ms</span>}
          </div>
          <button type="button" className="utility-close-btn" onClick={onClose}>
            Close Planner
          </button>
        </div>
      </div>
    </div>
  );
}
