import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Fish,
  Package,
  RefreshCw,
  ShieldCheck,
  Snowflake,
  Thermometer,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type IceToFishRatioResponse = {
  coastal_zone: string;
  target_species: string;
  ambient_deck_temp_c: number;
  solar_thermal_flux_index: string;
  recommended_ice_to_fish_ratio: string;
  total_ice_required_kg: number;
  estimated_shelf_life_hours: number;
  spoilage_speed_factor: number;
  chilling_method_recommendations: string[];
  ai_preservation_guidance: string;
  provider: string;
  latency_ms?: number;
};

type IceToFishRatioModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

const SPECIES_OPTIONS = [
  { id: "mackerel_sardine", label: "Indian Mackerel & Oil Sardine (High Fat)" },
  { id: "pomfret", label: "Silver & Black Pomfret (High Export Value)" },
  { id: "kingfish_seer", label: "Kingfish / Seer Fish (Surmai)" },
  { id: "prawns_shrimp", label: "Tiger Prawns & White Shrimp (Rapid Melanosis)" },
  { id: "tuna", label: "Yellowfin & Skipjack Tuna (High Core Temp)" },
];

const INSULATION_OPTIONS = [
  { id: "fiberglass_insulated", label: "Fiberglass (FRP) Insulated Fish Box" },
  { id: "thermocol_lined", label: "Expanded Polystyrene (Thermocol Lined)" },
  { id: "uninsulated_wooden", label: "Traditional Uninsulated Wooden Hold" },
  { id: "refrigerated_seawater", label: "Refrigerated Seawater Slurry (RSW)" },
];

const TONNAGE_OPTIONS = [
  { kg: 300, label: "300 kg" },
  { kg: 800, label: "800 kg" },
  { kg: 1500, label: "1.5 Tonnes" },
  { kg: 3000, label: "3.0 Tonnes" },
];

export default function IceToFishRatioModal({ city, onClose }: IceToFishRatioModalProps) {
  const [species, setSpecies] = useState<string>("mackerel_sardine");
  const [tonnageKg, setTonnageKg] = useState<number>(800);
  const [durationHours] = useState<number>(18);
  const [insulation, setInsulation] = useState<string>("fiberglass_insulated");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<IceToFishRatioResponse | null>(null);

  const fetchThermalCalculation = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/ice-to-fish-ratio`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          coastal_zone: city || "Mumbai Coast",
          target_species: species,
          catch_tonnage_kg: tonnageKg,
          voyage_duration_hours: durationHours,
          hold_insulation: insulation,
        }),
      });
      if (!res.ok) {
        throw new Error(`Ice-to-fish calculation failed (${res.status})`);
      }
      const json = (await res.json()) as IceToFishRatioResponse;
      setData(json);
    } catch (err) {
      console.error("Ice-to-Fish error:", err);
      setError(err instanceof Error ? err.message : "Failed to calculate chilling requirements");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchThermalCalculation();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [city]);

  return (
    <div className="utility-modal-scrim" onClick={onClose} role="dialog" aria-modal="true">
      <div className="utility-modal-window" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn ice-ratio-circle-btn" style={{ width: 36, height: 36 }}>
              <Fish size={18} />
            </div>
            <div className="utility-modal-header-info">
              <div className="utility-modal-kicker">
                <Snowflake size={13} className="inline-icon" /> MARINE COLD-CHAIN & POST-CATCH THERMODYNAMICS
              </div>
              <h2>Catch Freshness & Ice-to-Fish Thermal Ratio Calculator</h2>
              <p className="utility-modal-subtitle">
                Deck solar heat flux, hold insulation thermodynamics, and ice preservation ratio optimizer for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Ice-to-Fish Ratio Calculator"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls Bar */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="species-select">Target Catch</label>
            <UtilityCustomSelect
              id="species-select"
              value={species}
              options={SPECIES_OPTIONS}
              onChange={setSpecies}
              ariaLabel="Select target catch"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="insulation-select">Hold Insulation</label>
            <UtilityCustomSelect
              id="insulation-select"
              value={insulation}
              options={INSULATION_OPTIONS}
              onChange={setInsulation}
              ariaLabel="Select hold insulation"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Catch Tonnage</label>
            <div className="utility-chips-row">
              {TONNAGE_OPTIONS.map((t) => (
                <button
                  key={t.kg}
                  type="button"
                  className={`utility-chip-btn ${tonnageKg === t.kg ? "selected" : ""}`}
                  onClick={() => setTonnageKg(t.kg)}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-action-row-center">
            <button
              type="button"
              className="utility-refresh-btn"
              onClick={fetchThermalCalculation}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? "spin" : ""} />
              <span>Calculate</span>
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
                  <Snowflake size={16} className="inline-icon" /> Required Chilling Ratio: {data.recommended_ice_to_fish_ratio}
                  <span className="utility-status-pill hazard-badge-clear" style={{ marginLeft: 8 }}>
                    Total Ice: {data.total_ice_required_kg} kg
                  </span>
                </div>
              </div>

              {/* Status Grid */}
              <div className="utility-status-grid">
                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Clock size={13} className="inline-icon" /> Grade-A Shelf Life
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.estimated_shelf_life_hours}</span>
                    <span className="utility-metric-unit">hours</span>
                    <span className="utility-status-pill hazard-badge-clear">PRIME FRESHNESS</span>
                  </div>
                  <div className="utility-metric-sub">
                    Enzymatic spoilage factor: {data.spoilage_speed_factor}x acceleration above 4 deg C
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Thermometer size={13} className="inline-icon" /> Deck Heat & Solar Flux
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-number">{data.ambient_deck_temp_c}</span>
                    <span className="utility-metric-unit">deg C</span>
                    <span className="utility-status-pill hazard-badge-caution">
                      {data.solar_thermal_flux_index} FLUX
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Solar thermal absorption through deck skin accounts for 35% of hold heat gain
                  </div>
                </div>

                <div className="utility-status-card">
                  <div className="utility-metric-label">
                    <Package size={13} className="inline-icon" /> Voyage Duration
                  </div>
                  <div className="utility-metric-main">
                    <span className="utility-metric-text-val">
                      {durationHours} Hours At Sea
                    </span>
                  </div>
                  <div className="utility-metric-sub">
                    Calibrated for tropical maritime humidity and open sea evaporative loss
                  </div>
                </div>
              </div>

              {/* AI Preservation Advice */}
              <div className="utility-ai-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    THERMAL PRESERVATION PROTOCOL
                  </div>
                  <span className="utility-provider-badge">
                    "Powered by ClimaCore Intelligence"
                  </span>
                </div>
                <p className="utility-ai-text">{data.ai_preservation_guidance}</p>
              </div>

              {/* Chilling Methods */}
              <div className="utility-section-title">
                Cold Chain Chilling & Slurry Protocols
              </div>
              <div className="utility-checklist-grid">
                {(data.chilling_method_recommendations || []).map((rec, i) => (
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
            <ShieldCheck size={14} className="inline-icon" /> Grounded in FAO fish handling guidelines & tropical marine thermodynamics
          </div>
        </div>
      </div>
    </div>
  );
}
