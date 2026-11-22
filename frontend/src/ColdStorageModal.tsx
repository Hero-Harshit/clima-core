import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Fuel,
  RefreshCw,
  ShieldCheck,
  Snowflake,
  Thermometer,
  X,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";

type ColdStorageBlackoutResponse = {
  city: string;
  commodity_type: string;
  ambient_temp_c: number;
  ambient_heat_ingress_rate_kw: number;
  passive_thermal_flywheel_hours: number;
  recommended_pre_cooling_temperature_c: number;
  critical_spoilage_breach_time_hours: number;
  diesel_genset_fuel_burn_liters_hr: number;
  blackout_defense_readiness_level: string;
  logistics_advisories: string[];
  ai_cold_chain_guidance: string;
  provider: string;
  latency_ms?: number;
};

type ColdStorageModalProps = {
  city: string;
  onClose: () => void;
};

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const COMMODITY_OPTIONS = [
  { id: "vaccines_pharmaceuticals_2_8c", label: "Vaccines & Biologicals (Target 2°C - 8°C)" },
  { id: "dairy_milk_cheese_4c", label: "Dairy, Milk & Fresh Cheeses (Target 4°C)" },
  { id: "fresh_fruits_table_grapes_0_2c", label: "Table Grapes & Export Pome Fruits (Target 0°C - 2°C)" },
  { id: "potatoes_storage_tubers_8_10c", label: "Bulk Storage Seed Potatoes (Target 8°C - 10°C)" },
];

const INSULATION_OPTIONS = [
  { id: "high_density_pir_150mm", label: "High-Density PIR Panel (150mm, U = 0.15 W/m²K)" },
  { id: "standard_pu_panel_100mm", label: "Standard Polyurethane PUF (100mm, U = 0.22 W/m²K)" },
  { id: "legacy_expanded_polystyrene_thermocol", label: "Legacy Expanded Polystyrene (Thermocol, U = 0.45 W/m²K)" },
];

const TONNAGE_OPTIONS = [
  { mt: 50, label: "50 MT" },
  { mt: 200, label: "200 MT" },
  { mt: 500, label: "500 MT" },
  { mt: 1500, label: "1,500 MT" },
];

const DIESEL_OPTIONS = [
  { liters: 500, label: "500 L" },
  { liters: 2000, label: "2,000 L" },
  { liters: 5000, label: "5,000 L" },
  { liters: 10000, label: "10,000 L" },
];

export default function ColdStorageModal({ city, onClose }: ColdStorageModalProps) {
  const [commodity, setCommodity] = useState<string>("vaccines_pharmaceuticals_2_8c");
  const [insulation, setInsulation] = useState<string>("standard_pu_panel_100mm");
  const [capacityMt, setCapacityMt] = useState<number>(500);
  const [dieselStock, setDieselStock] = useState<number>(2000);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ColdStorageBlackoutResponse | null>(null);

  const fetchDefensePlan = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/tools/cold-storage-defense`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          commodity_type: commodity,
          insulation_rating: insulation,
          facility_capacity_metric_tons: capacityMt,
          backup_diesel_available_liters: dieselStock,
        }),
      });
      if (!res.ok) {
        throw new Error(`Cold storage defense service returned HTTP ${res.status}`);
      }
      const json = await res.json();
      setData(json);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to compute cold chain blackout defense plan.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchDefensePlan();
  }, [city, commodity, insulation, capacityMt, dieselStock]);

  const getReadinessBadgeClass = (readiness?: string) => {
    switch (readiness) {
      case "CRITICAL_RAPID_WARMING_HAZARD":
        return "hazard-badge-critical";
      case "MODERATE_THERMAL_BUFFER":
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
        aria-labelledby="cold-chain-modal-title"
      >
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn cold-storage-circle-btn" style={{ width: 36, height: 36 }}>
              <Snowflake size={18} />
            </div>
            <div>
              <h2 id="cold-chain-modal-title" className="utility-modal-heading">
                Cold Storage Pre-Cooling & Grid Blackout Defense Planner
              </h2>
              <p className="utility-modal-subtitle">
                Chamber heat ingress, thermal flywheel retention, and generator diesel fuel preservation for {city}
              </p>
            </div>
          </div>
          <button
            type="button"
            className="utility-modal-close"
            onClick={onClose}
            aria-label="Close Cold Storage Defense Planner"
          >
            <X size={18} />
          </button>
        </div>

        {/* Controls */}
        <div className="utility-controls-bar">
          <div className="utility-control-group">
            <label htmlFor="commodity-select">Chamber Cargo</label>
            <UtilityCustomSelect
              id="commodity-select"
              value={commodity}
              options={COMMODITY_OPTIONS}
              onChange={setCommodity}
              ariaLabel="Select chamber cargo"
            />
          </div>

          <div className="utility-control-group">
            <label htmlFor="insulation-select">Insulation Specification</label>
            <UtilityCustomSelect
              id="insulation-select"
              value={insulation}
              options={INSULATION_OPTIONS}
              onChange={setInsulation}
              ariaLabel="Select insulation specification"
            />
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Storage Chamber Capacity</label>
            <div className="utility-chips-row">
              {TONNAGE_OPTIONS.map((t) => (
                <button
                  key={t.mt}
                  type="button"
                  className={`utility-chip-btn ${capacityMt === t.mt ? "selected" : ""}`}
                  onClick={() => setCapacityMt(t.mt)}
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>

          <div className="utility-control-group interests-toggle-group">
            <label>Diesel Reserve</label>
            <div className="utility-chips-row">
              {DIESEL_OPTIONS.map((d) => (
                <button
                  key={d.liters}
                  type="button"
                  className={`utility-chip-btn ${dieselStock === d.liters ? "selected" : ""}`}
                  onClick={() => setDieselStock(d.liters)}
                >
                  {d.label}
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
              <span>Modeling chamber heat transfer flux, thermal battery flywheel, and fuel runtime...</span>
            </div>
          )}

          {error && !loading && (
            <div className="utility-error-banner">
              <AlertTriangle size={18} />
              <span>{error}</span>
              <button type="button" className="utility-btn-secondary" onClick={() => void fetchDefensePlan()}>
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
                    <Thermometer size={14} /> Ambient Heat Ingress
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#e11d48" }}>
                    {data.ambient_heat_ingress_rate_kw} kW
                  </div>
                  <div className="utility-kpi-meta">
                    Outdoor Ambient: {data.ambient_temp_c}°C
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Clock size={14} /> Passive Flywheel Hours
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#16a34a" }}>
                    {data.passive_thermal_flywheel_hours} hrs
                  </div>
                  <div className="utility-kpi-meta">
                    <span className={`hazard-badge ${getReadinessBadgeClass(data.blackout_defense_readiness_level)}`}>
                      {data.blackout_defense_readiness_level.replace(/_/g, " ")}
                    </span>
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Snowflake size={14} /> Night Pre-Cool Setpoint
                  </div>
                  <div className="utility-kpi-value" style={{ color: "#0284c7" }}>
                    {data.recommended_pre_cooling_temperature_c}°C
                  </div>
                  <div className="utility-kpi-meta">
                    Deep sub-cooling before peak tariff
                  </div>
                </div>

                <div className="utility-kpi-card">
                  <div className="utility-kpi-label">
                    <Fuel size={14} /> Total Blackout Endurance
                  </div>
                  <div className="utility-kpi-value">
                    {data.critical_spoilage_breach_time_hours} hrs
                  </div>
                  <div className="utility-kpi-meta">
                    Generator Burn: {data.diesel_genset_fuel_burn_liters_hr} L/hr
                  </div>
                </div>
              </div>

              {/* AI Guidance */}
              <div className="utility-ai-guidance-card">
                <div className="utility-ai-header">
                  <span>ClimaCore Cold Chain Engineering & Thermodynamics Synthesis</span>
                  <span className="utility-provider-badge">ClimaCore Intelligence</span>
                </div>
                <p className="utility-ai-text">{data.ai_cold_chain_guidance}</p>
              </div>

              {/* Advisories List */}
              <div className="utility-advisories-card">
                <h3 className="utility-section-title">
                  <ShieldCheck size={16} /> Facility Blackout Defense & Cargo Preservation Protocol
                </h3>
                <ul className="utility-checklist">
                  {data.logistics_advisories.map((item, idx) => (
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
