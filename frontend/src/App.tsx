import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown from "react-markdown";
import logoUrl from "../Logo.png";
import AboutUs from "./AboutUs";
import CitySearch from "./CitySearch";
import DataSources from "./DataSources";
import Methodology from "./Methodology";
import PrivacyPolicy from "./PrivacyPolicy";
import {
  Activity, AlertTriangle, ArrowUpRight, BadgeCheck,
  BatteryWarning, CalendarDays, Check, ChevronDown, ChevronRight,
  BarChart3, CircleHelp, CloudDrizzle, CloudSun, Database, Droplets, Factory, Flame,
  Globe2, History, Info, Layers, Send,
  LocateFixed, LockKeyhole, MapPin, RotateCcw,
  ShieldCheck, Sprout, Sun, Waves, Wind, X, Zap,
} from "lucide-react";

const MapCanvas = lazy(() => import("./MapCanvas"));
const MapBackdrop = lazy(() => import("./MapBackdrop"));
const ConformalChart = lazy(() => import("./ConformalChart"));
const ModelPerformance = lazy(() => import("./ModelPerformance"));
const CrisisCouncilModal = lazy(() => import("./CrisisCouncilModal"));
const DisasterReplayModal = lazy(() => import("./DisasterReplayModal"));

type Persona = { id: string; key?: string; title: string; icon?: string; description?: string };
type PersonasResponse = Persona[] | { personas?: Persona[] };
const additionalPages = [
  { id: "methodology", title: "Methodology", Icon: Layers },
  { id: "data-sources", title: "Data Sources", Icon: Database },
  { id: "model-performance", title: "Model Performance", Icon: BarChart3 },
  { id: "faq", title: "FAQ", Icon: CircleHelp },
] as const;
type AdditionalPageId = (typeof additionalPages)[number]["id"];
type ActiveModal = "provenance" | "privacy" | "about" | AdditionalPageId | null;
type Day = { date: string; day_name?: string; temp_max?: number; precipitation_mm?: number; compound_risk_score?: number; [key: string]: unknown };
type IntervalMetric = "temperature_c" | "rainfall_mm" | "peak_grid_load_mw" | "pm2_5_ug_m3";
type Prediction = {
  city?: string; state?: string; latitude?: number; longitude?: number;
  retrieved_at_utc?: string; peak_compound_risk?: number; peak_risk_category?: string;
  daily_predictions?: Day[]; spatial_resolution_metadata?: Record<string, unknown>;
  physical_units_manifest?: Record<string, string>; [key: string]: unknown;
};
type CityGeocodingResult = {
  name: string;
  latitude: number;
  longitude: number;
  country_code?: string;
  admin1?: string;
};
type AssistantMessage = { role: "user" | "assistant"; content: string };
type AssistantChatResponse = { response: string; suggested_followups?: string[] };

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const fallbackCityCoordinates: Record<string, { latitude: number; longitude: number }> = {
  mumbai: { latitude: 19.076, longitude: 72.8777 },
  delhi: { latitude: 28.6139, longitude: 77.209 },
  bengaluru: { latitude: 12.9716, longitude: 77.5946 },
  hyderabad: { latitude: 17.385, longitude: 78.4867 },
  ahmedabad: { latitude: 23.0225, longitude: 72.5714 },
  chennai: { latitude: 13.0827, longitude: 80.2707 },
  kolkata: { latitude: 22.5726, longitude: 88.3639 },
  surat: { latitude: 21.1702, longitude: 72.8311 },
  pune: { latitude: 18.5204, longitude: 73.8567 },
  jaipur: { latitude: 26.9124, longitude: 75.7873 },
  kanpur: { latitude: 26.4499, longitude: 80.3319 },
  "navi mumbai": { latitude: 19.033, longitude: 73.0297 },
  lucknow: { latitude: 26.8467, longitude: 80.9462 },
  nagpur: { latitude: 21.1458, longitude: 79.0882 },
  coimbatore: { latitude: 11.0168, longitude: 76.9558 },
  visakhapatnam: { latitude: 17.6868, longitude: 83.2185 },
  bhopal: { latitude: 23.2599, longitude: 77.4126 },
  patna: { latitude: 25.5941, longitude: 85.1376 },
  vadodara: { latitude: 22.3072, longitude: 73.1812 },
  ludhiana: { latitude: 30.901, longitude: 75.8573 },
  agra: { latitude: 27.1767, longitude: 78.0081 },
  nashik: { latitude: 19.9975, longitude: 73.7898 },
  madurai: { latitude: 9.9252, longitude: 78.1198 },
  varanasi: { latitude: 25.3176, longitude: 82.9739 },
  amritsar: { latitude: 31.634, longitude: 74.8723 },
  thane: { latitude: 19.2183, longitude: 72.9781 },
  ranchi: { latitude: 23.3441, longitude: 85.3096 },
  guwahati: { latitude: 26.1445, longitude: 91.7362 },
  mysuru: { latitude: 12.2958, longitude: 76.6394 },
  kochi: { latitude: 9.9312, longitude: 76.2673 },
};
const demoDays: Day[] = Array.from({ length: 7 }, (_, index) => {
  const date = new Date();
  date.setDate(date.getDate() + index);
  return {
    date: date.toISOString().slice(0, 10),
    day_name: index === 0 ? "Today" : new Intl.DateTimeFormat("en", { weekday: "short" }).format(date),
    temp_max: [34, 33, 32, 35, 36, 33, 31][index],
    precipitation_mm: [12, 18, 8, 2, 0, 22, 15][index],
    compound_risk_score: [68, 74, 58, 42, 49, 82, 71][index],
  };
});
const demoPersonas: Persona[] = [
  { id: "farmer", title: "Farmer & Agriculture", icon: "Wheat" },
  { id: "student", title: "Student & Education", icon: "GraduationCap" },
  { id: "gig_worker", title: "Gig Worker & Outdoor Labor", icon: "Bike" },
  { id: "commuter", title: "Daily Commuter & Transit", icon: "Car" },
  { id: "healthcare", title: "Healthcare & Hospital Ops", icon: "Stethoscope" },
  { id: "city_official", title: "City Official & Disaster Mgmt", icon: "Building2" },
  { id: "maritime", title: "Maritime & Fishermen", icon: "Ship" },
  { id: "industry", title: "Industry & Manufacturing", icon: "Factory" },
  { id: "aviation", title: "Aviation & Airports", icon: "Plane" },
  { id: "water", title: "Water Utility & Stormwater", icon: "Droplets" },
  { id: "insurance", title: "Insurance & Catastrophe Actuary", icon: "ShieldAlert" },
  { id: "cold_chain", title: "Cold Chain Logistics", icon: "Truck" },
  { id: "tourism", title: "Tourism & Hospitality", icon: "Palmtree" },
];
const demoPrediction: Prediction = {
  city: "Mumbai", state: "Maharashtra", latitude: 19.076, longitude: 72.8777,
  peak_compound_risk: 68, peak_risk_category: "HIGH", daily_predictions: demoDays,
};
const intervalConfigs: Record<IntervalMetric, { label: string; unit: string; keys: string[]; fallback: number; spread: number }> = {
  temperature_c: { label: "Temperature", unit: "°C", keys: ["temp_max", "day_temp_max_corrected_c"], fallback: 32, spread: 3 },
  rainfall_mm: { label: "Rainfall", unit: "mm", keys: ["precipitation_mm", "rainfall_mm"], fallback: 8, spread: 5 },
  peak_grid_load_mw: { label: "Power demand", unit: "MW", keys: ["peak_load_mw", "projected_peak_grid_load_mw"], fallback: 4200, spread: 500 },
  pm2_5_ug_m3: { label: "PM2.5", unit: "μg/m³", keys: ["pm2_5", "concentration_ug_m3"], fallback: 32, spread: 20 },
};
const titleCase = (s: string) => s.replace(/[_-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const riskTone = (risk: number) => risk >= 80 ? "critical" : risk >= 60 ? "high" : risk >= 40 ? "moderate" : "low";
const getSeasonLabel = (month: number) => month >= 6 && month <= 9
  ? "SOUTHWEST MONSOON"
  : month >= 10 && month <= 11
    ? "POST-MONSOON"
    : month >= 3 && month <= 5
      ? "PRE-MONSOON"
      : "WINTER";
function findMetric(source: unknown, names: string[], depth = 0): unknown {
  if (!source || typeof source !== "object" || depth > 5) return undefined;
  const record = source as Record<string, unknown>;
  const wanted = new Set(names.map((name) => name.toLowerCase().replace(/[^a-z0-9]/g, "")));
  for (const [key, value] of Object.entries(record)) {
    if (wanted.has(key.toLowerCase().replace(/[^a-z0-9]/g, "")) && (typeof value === "number" || typeof value === "string")) return value;
  }
  for (const value of Object.values(record)) {
    if (value && typeof value === "object") {
      const found = findMetric(value, names, depth + 1);
      if (found !== undefined) return found;
    }
  }
  return undefined;
}
function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

type DropdownOption = { value: string; label: string };
type StyledDropdownProps = {
  ariaLabel: string;
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
};

function StyledDropdown({ ariaLabel, value, options, onChange, disabled = false }: StyledDropdownProps) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const selectedIndex = Math.max(0, options.findIndex((option) => option.value === value));
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(selectedIndex);
  const selectedOption = options.find((option) => option.value === value);

  useEffect(() => {
    if (!isOpen) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setIsOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    menuRef.current?.querySelector<HTMLElement>(`[data-option-index="${activeIndex}"]`)?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, isOpen]);

  const selectOption = (option: DropdownOption) => {
    onChange(option.value);
    setIsOpen(false);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === "Escape" && isOpen) {
      event.preventDefault();
      setIsOpen(false);
    } else if (event.key === "Tab") {
      setIsOpen(false);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!isOpen) {
        setActiveIndex(selectedIndex);
        setIsOpen(true);
      } else if (options.length) {
        const offset = event.key === "ArrowDown" ? 1 : -1;
        setActiveIndex((index) => (index + offset + options.length) % options.length);
      }
    } else if (event.key === "Home" || event.key === "End") {
      if (!isOpen || !options.length) return;
      event.preventDefault();
      setActiveIndex(event.key === "Home" ? 0 : options.length - 1);
    } else if ((event.key === "Enter" || event.key === " ") && isOpen) {
      event.preventDefault();
      const option = options[activeIndex];
      if (option) selectOption(option);
    }
  };

  return (
    <div className="styled-dropdown" ref={rootRef}>
      <button
        type="button"
        className="styled-dropdown-trigger"
        role="combobox"
        aria-label={ariaLabel}
        aria-valuetext={selectedOption?.label}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={`dropdown-list-${id}`}
        aria-activedescendant={isOpen ? `dropdown-option-${id}-${activeIndex}` : undefined}
        disabled={disabled}
        onClick={() => {
          setActiveIndex(selectedIndex);
          setIsOpen((open) => !open);
        }}
        onKeyDown={handleKeyDown}
      >
        <span className="styled-dropdown-value">{selectedOption?.label ?? value}</span>
        <ChevronDown size={14} aria-hidden="true" />
      </button>
      {isOpen && (
        <div className="styled-dropdown-menu" id={`dropdown-list-${id}`} role="listbox" ref={menuRef}>
          {options.map((option, index) => (
            <div
              key={option.value}
              id={`dropdown-option-${id}-${index}`}
              className={`styled-dropdown-option ${activeIndex === index ? "active" : ""}`}
              data-option-index={index}
              role="option"
              aria-selected={value === option.value}
              onMouseDown={(event) => event.preventDefault()}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => selectOption(option)}
            >
              <span>{option.label}</span>
              {value === option.value && <Check size={14} aria-hidden="true" />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function EmptyInfoPageModal({ title, onClose, children }: { title: string; onClose: () => void; children?: ReactNode }) {
  return (
    <section className="provenance-modal" role="dialog" aria-modal="true" aria-labelledby="empty-page-title">
      <header className="drawer-header">
        <div>
          <span className="section-kicker">CLIMA-CORE</span>
          <h2 id="empty-page-title">{title}</h2>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label={`Close ${title}`}><X size={18} /></button>
      </header>
      {children}
    </section>
  );
}

function getModelIcon(name: string) {
  const key = name.toLowerCase();
  if (key.includes("flood")) return Droplets;
  if (key.includes("heat")) return Sun;
  if (/grid|power|electric/.test(key)) return Zap;
  if (key.includes("air")) return Wind;
  if (key.includes("water")) return Waves;
  if (/agri|crop|soil/.test(key)) return Sprout;
  return Activity;
}

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json() as Promise<T>;
}

async function requestPrediction(city: string, persona: string, force: boolean): Promise<Prediction> {
  const response = await fetch(`${API_BASE}/api/predict/city`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ city, requested_personas: [persona], force_refresh: force }),
  });
  if (!response.ok) throw new Error(`Prediction request failed (${response.status})`);
  return response.json() as Promise<Prediction>;
}

function App() {
  const [city, setCity] = useState("Mumbai");
  const [persona, setPersona] = useState("farmer");
  const [activeDay, setActiveDay] = useState(0);
  const [intervalMetric, setIntervalMetric] = useState<IntervalMetric>("temperature_c");
  const [mapReset, setMapReset] = useState(0);
  const [realismEnabled, setRealismEnabled] = useState(false);
  const [assistantInput, setAssistantInput] = useState("");
  const [assistantMessages, setAssistantMessages] = useState<AssistantMessage[]>([]);
  const [assistantFollowups, setAssistantFollowups] = useState<string[]>([]);
  const [assistantError, setAssistantError] = useState<string | null>(null);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [isAssistantOpen, setIsAssistantOpen] = useState(false);
  const [activeModal, setActiveModal] = useState<ActiveModal>(null);
  const [showAllAdvisories, setShowAllAdvisories] = useState(false);
  const [isPageScrolled, setIsPageScrolled] = useState(false);
  const [isNavBarExpanded, setIsNavBarExpanded] = useState(false);
  const [isCrisisCouncilOpen, setIsCrisisCouncilOpen] = useState(false);
  const [isDisasterReplayOpen, setIsDisasterReplayOpen] = useState(false);

  useEffect(() => {
    let lastScrollY = window.scrollY;
    const updateNavBarOnScroll = () => {
      const currentScrollY = window.scrollY;
      setIsPageScrolled((prevScrolled) => {
        if (!prevScrolled && currentScrollY > 90) {
          setIsAssistantOpen(false);
          setIsNavBarExpanded(false);
          return true;
        } else if (prevScrolled && currentScrollY <= 40) {
          setIsNavBarExpanded(false);
          return false;
        }
        return prevScrolled;
      });

      if (Math.abs(currentScrollY - lastScrollY) > 80) {
        setIsNavBarExpanded(false);
        lastScrollY = currentScrollY;
      }
    };

    updateNavBarOnScroll();
    window.addEventListener("scroll", updateNavBarOnScroll, { passive: true });
    return () => window.removeEventListener("scroll", updateNavBarOnScroll);
  }, []);

  useEffect(() => {
    if (!activeModal) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setActiveModal(null);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [activeModal]);

  const personasQuery = useQuery({
    queryKey: ["personas"],
    queryFn: async () => {
      const response = await getJson<PersonasResponse>("/api/personas");
      return Array.isArray(response) ? response : response.personas ?? [];
    },
  });
  const predictionQuery = useQuery({
    queryKey: ["prediction", city, persona],
    queryFn: () => requestPrediction(city, persona, false),
  });
  const selectedCityName = city.split(",")[0].trim();
  const selectedStateName = city.split(",")[1]?.trim().toLocaleLowerCase();
  const selectedCityFallback = fallbackCityCoordinates[selectedCityName.toLocaleLowerCase()];
  const hasPredictionCoordinates = predictionQuery.data?.latitude != null
    && predictionQuery.data?.longitude != null
    && Number.isFinite(Number(predictionQuery.data.latitude))
    && Number.isFinite(Number(predictionQuery.data?.longitude));
  const mapLocationQuery = useQuery({
    queryKey: ["map-location", city],
    queryFn: async () => {
      const params = new URLSearchParams({ name: selectedCityName, count: "10", language: "en", format: "json" });
      const response = await fetch(`https://geocoding-api.open-meteo.com/v1/search?${params}`);
      if (!response.ok) throw new Error(`City geocoding failed (${response.status})`);
      const payload = await response.json() as { results?: CityGeocodingResult[] };
      const results = payload.results ?? [];
      const indiaResults = results.filter((result) => result.country_code?.toLocaleLowerCase() === "in");
      const selectedResult = (selectedStateName
        ? indiaResults.find((result) => result.admin1?.toLocaleLowerCase() === selectedStateName)
        : undefined)
        ?? indiaResults.find((result) => result.name.toLocaleLowerCase() === selectedCityName.toLocaleLowerCase())
        ?? indiaResults[0]
        ?? results[0];
      if (!selectedResult || !Number.isFinite(selectedResult.latitude) || !Number.isFinite(selectedResult.longitude)) {
        throw new Error(`No map coordinates found for ${city}`);
      }
      return { latitude: selectedResult.latitude, longitude: selectedResult.longitude };
    },
    enabled: Boolean(selectedCityName)
      && !hasPredictionCoordinates
      && !selectedCityFallback
      && (predictionQuery.isError || Boolean(predictionQuery.data)),
    staleTime: 86_400_000,
  });
  const prediction = predictionQuery.data ?? demoPrediction;
  const people = (personasQuery.data?.length ? personasQuery.data : demoPersonas).map((item) =>
    item.title.toLowerCase() === "student & school"
      ? { ...item, title: "Student & Education" }
      : item,
  );
  const days = prediction.daily_predictions?.length ? prediction.daily_predictions : demoDays;
  const activeForecast = days[activeDay] ?? days[0];
  const intervalConfig = intervalConfigs[intervalMetric];
  const intervalData = days.slice(0, 7).map((day, index) => {
    const bounds = asRecord(asRecord(day.conformal_bounds)?.[intervalMetric]);
    const pointEstimate = Number(bounds?.point_estimate ?? findMetric(day, intervalConfig.keys));
    const point = Number.isFinite(pointEstimate) ? pointEstimate : intervalConfig.fallback + index * 2;
    const lowerValue = Number(bounds?.lower_bound);
    const upperValue = Number(bounds?.upper_bound);
    const lower = Number.isFinite(lowerValue) ? lowerValue : Math.max(0, point - intervalConfig.spread);
    const upper = Number.isFinite(upperValue) ? upperValue : point + intervalConfig.spread;
    return {
      date: day.day_name ?? day.date,
      point,
      lower,
      band: Math.max(0, upper - lower),
      upper,
      hasBounds: Number.isFinite(lowerValue) && Number.isFinite(upperValue),
    };
  });
  const intervalMin = Math.min(...intervalData.map((item) => item.lower));
  const intervalMax = Math.max(...intervalData.map((item) => item.upper));
  const intervalPadding = (intervalMax - intervalMin) * 0.12 || 1;
  const selectedInterval = intervalData[activeDay] ?? intervalData[0];
  const activeBounds = asRecord(asRecord(activeForecast?.conformal_bounds)?.[intervalMetric]);
  const intervalNote = typeof activeBounds?.coverage_guarantee === "string"
    ? activeBounds.coverage_guarantee
    : predictionQuery.data ? "Conformal bounds are not available for every forecast variable."
      : "Demo interval only; live conformal bounds are not connected.";
  const flagship = asRecord(activeForecast?.flagship_intelligence);
  const floodRisk = asRecord(flagship?.flood_risk);
  const heatRisk = asRecord(flagship?.heat_risk);
  const powerGrid = asRecord(flagship?.power_grid);
  const airQuality = asRecord(flagship?.air_quality);
  const agriSense = asRecord(flagship?.agrisense);
  const waterQuality = asRecord(activeForecast?.water_quality_risk) ?? asRecord(flagship?.water_quality);
  const cascadeRisk = asRecord(activeForecast?.cascading_risk);
  const activeChains = Array.isArray(cascadeRisk?.active_chains) ? cascadeRisk.active_chains : [];
  const activeChain = asRecord(activeChains[0]);
  const cascadeNodes = Array.isArray(activeChain?.governing_nodes)
    ? activeChain.governing_nodes.map(asRecord).filter((node): node is Record<string, unknown> => Boolean(node))
    : [];
  const temperatureBounds = asRecord(asRecord(activeForecast?.conformal_bounds)?.temperature_c);
  const personaDataMap = asRecord(activeForecast?.personas);
  const personaId = persona.toLowerCase().replace(/[^a-z0-9]/g, "");
  const activePersonaData = asRecord(Object.entries(personaDataMap ?? {}).find(([key, value]) => {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9]/g, "");
    const returnedId = String(findMetric(value, ["persona_id"]) ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
    return normalizedKey === personaId || normalizedKey.startsWith(personaId) || returnedId === personaId;
  })?.[1]);
  const riskValue = Number(findMetric(prediction, ["peak_compound_risk"]) ?? findMetric(activeForecast, ["compound_risk_score"]) ?? 68);
  const risk = Math.round(Number.isFinite(riskValue) ? riskValue : 68);
  const peakRiskCategory = String(prediction.peak_risk_category ?? titleCase(riskTone(risk))).toUpperCase();
  const tone = peakRiskCategory.includes("CRITICAL") ? "critical"
    : peakRiskCategory.includes("HIGH") ? "high"
      : peakRiskCategory.includes("MODERATE") ? "moderate"
        : "low";
  const metric = (names: string[], fallback: number, source: unknown = activeForecast) => {
    const raw = findMetric(source, names) ?? findMetric(prediction, names);
    const value = Number(raw);
    return Number.isFinite(value) ? value : fallback;
  };
  const textMetric = (names: string[], fallback: string, source: unknown = activeForecast) => {
    const value = findMetric(source, names);
    return typeof value === "string" && value.trim() ? value : fallback;
  };
  const pollutant = (key: string, fallback: number): number | null => {
    const pollutantData = asRecord(asRecord(airQuality?.pollutants)?.[key]);
    if (!pollutantData) return predictionQuery.data ? null : fallback;
    const value = Number(findMetric(pollutantData, ["concentration_ug_m3"]));
    return Number.isFinite(value) ? value : predictionQuery.data ? null : fallback;
  };
  const runoff = metric(["effective_surface_runoff_mm", "runoff_depth_mm"], 18.4, floodRisk);
  const floodProbability = metric(["hydrological_hazard_probability", "flood_probability"], 0.42, floodRisk);
  const antecedentRain = metric(["antecedent_precipitation_index_14d"], 0, asRecord(floodRisk?.antecedent_hydrology));
  const dryBulb = metric(["day_temp_max_corrected_c", "temp_max"], Number(activeForecast?.temp_max ?? 34), heatRisk);
  const apparentTemp = metric(["apparent_temp_max_c", "apparent_temp_max", "feels_like"], 39, heatRisk);
  const wetBulb = metric(["wet_bulb_temp_c", "wet_bulb_temperature_c", "wet_bulb_temperature"], 29);
  const heatAlert = textMetric(["heatwave_alert_tier", "alert_tier"], "None", heatRisk);
  const heatAlertLabel = /^(none|normal)|no heatwave/i.test(heatAlert) ? "NO ALERT" : heatAlert.replace(/_/g, " ");
  const uhiDelta = metric(["uhi_temperature_delta_c", "uhi_delta_c"], 2.1, heatRisk);
  const peakLoad = metric(["peak_load_mw", "projected_peak_grid_load_mw"], 4280, powerGrid);
  const coolingSurge = metric(["cooling_demand_surge_ratio"], 1.18, powerGrid);
  const blackoutProbability = metric(["unmet_supply_stress_probability"], 0.04, powerGrid);
  const aqi = metric(["naqi", "national_aqi", "aqi"], 84, airQuality);
  const aqiCategory = textMetric(["naqi_category", "aqi_category"], "Moderate", airQuality);
  const aqiStatusClass = /severe|very poor|poor/i.test(aqiCategory) ? "critical" : /moderate/i.test(aqiCategory) ? "moderate" : "good";
  const pm25 = pollutant("pm2_5", 32);
  const pm10 = pollutant("pm10", 48);
  const no2 = pollutant("no2", 19);
  const so2 = pollutant("so2", 8);
  const co = pollutant("co", 0.7);
  const ozone = pollutant("o3", 22);
  const atmosphericPhysics = asRecord(airQuality?.atmospheric_physics);
  const stagnationFlag = Number(findMetric(atmosphericPhysics, ["stagnation_flag"]));
  const stagnation = stagnationFlag > 0
    ? "Flagged"
    : textMetric(["ventilation_tier"], "Low", atmosphericPhysics);
  const smdi = metric(["soil_moisture_deficit_index", "smdi"], -0.8, activePersonaData);
  const cropStage = textMetric(["growth_stage", "crop_stage", "phenological_stage"], "Vegetative", agriSense);
  const droughtAlert = textMetric(["flash_drought_alert"], "NONE", activePersonaData);
  const droughtIsActive = !/^(none|no|false|inactive)$/i.test(droughtAlert.trim());
  const droughtLabel = droughtIsActive ? (/critical/i.test(droughtAlert) ? "CRITICAL DROUGHT" : "DROUGHT ALERT") : "NO FLASH DROUGHT";
  const wqi = metric(["predicted_wqi", "exact_formula_wqi", "water_quality_index", "wqi"], 78, waterQuality);
  const confidenceCoverage = metric(["confidence_level_pct"], 90, temperatureBounds);
  const calibrationSamples = metric(["calibration_sample_size"], 900, temperatureBounds);
  const potability = textMetric(["potability_status"], "POTABLE", waterQuality);
  const waterGrade = titleCase(textMetric(["wqi_grade"], "Good quality", waterQuality));
  const exceedances = asRecord(waterQuality)?.bis_standard_exceedances;
  const waterLimitingParameter = Array.isArray(exceedances) && typeof exceedances[0] === "string"
    ? `Exceedance: ${exceedances[0].split("(")[0].trim()}`
    : "Fluoride 0.8 mg/L";
  const isPotable = /^potable$/i.test(potability.trim());
  const potabilityLabel = /non.?potable/i.test(potability)
    ? "NON-POTABLE"
    : /treatment/i.test(potability) ? "TREATMENT" : isPotable ? "POTABLE" : titleCase(potability).toUpperCase();
  const statusValue = activePersonaData?.status_level;
  const personaStatus = typeof statusValue === "string"
    ? statusValue
    : risk >= 80 ? "CRITICAL" : risk >= 60 ? "CAUTION" : "NORMAL";
  const personaStatusClass = personaStatus.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const impactScore = Number(findMetric(activePersonaData, ["impact_score"]));
  const suppliedWorkability = Number(findMetric(activePersonaData, ["workability_score"]));
  const workability = Math.round(Number.isFinite(suppliedWorkability)
    ? suppliedWorkability
    : Number.isFinite(impactScore) ? 100 - impactScore : 100 - risk);
  const restrictions = Array.isArray(activePersonaData?.operational_restrictions)
    ? activePersonaData.operational_restrictions.filter((item): item is string => typeof item === "string")
    : [];
  const advisories = Array.isArray(activePersonaData?.advisories)
    ? activePersonaData.advisories.filter((item): item is string => typeof item === "string")
    : [];
  const noActiveRestrictions = restrictions.length > 0 && restrictions.every((item) => /no .*restrictions? active|no restrictions/i.test(item));
  const restrictionText = restrictions[0] ?? (predictionQuery.data
    ? "No persona-specific restrictions were returned."
    : "Delay pesticide spraying during afternoon showers. Wet foliage and shifting winds may reduce application effectiveness.");
  const actionItems = advisories.length
    ? advisories
    : predictionQuery.data
      ? ["No persona-specific advisories were returned."]
      : ["Check local rainfall before field work", "Keep drainage channels clear"];
  const visibleActionItems = showAllAdvisories ? actionItems : actionItems.slice(0, 2);
  const hospitalCount = metric(["hospitals_15km", "hospital_count"], 42, floodRisk);
  const schoolCount = metric(["schools_15km", "school_count"], 318, floodRisk);
  const powerAssetCount = metric(["power_substations_15km", "substations_15km", "power_assets_15km"], 16, prediction.spatial_resolution_metadata);
  const modelVersions = asRecord(asRecord(prediction.system_provenance)?.champion_models);
  const provenanceModels = Object.entries(modelVersions ?? {
    flood_risk: "FloodRisk V3",
    heat_risk: "HeatRisk V4",
    hydrological_model: "Hybrid-GR4J v2.1",
  });
  const cascadeDescription = typeof findMetric(activeChain, ["trigger_condition"]) === "string"
    ? String(findMetric(activeChain, ["trigger_condition"]))
    : predictionQuery.data ? "No active cascade chains reported for this forecast."
      : "How today’s conditions could ripple across critical systems.";
  const cascadeDisplayNodes: Record<string, unknown>[] = cascadeNodes.length
    ? cascadeNodes.slice(0, 3)
    : predictionQuery.data ? []
      : [
          { system: "Heavy rainfall", status: "12 mm expected" },
          { system: "Urban runoff", status: "Drainage pressure" },
          { system: "Transit disruption", status: "Localized impact" },
        ];
  const selectedPersona = people.find((item) => (item.id ?? item.key) === persona) ?? people[0];
  const cityName = predictionQuery.data?.city ?? selectedCityName;
  const cityState = predictionQuery.data?.state ?? city.split(",")[1]?.trim();
  const coordinates = useMemo(() => ({
    latitude: Number(predictionQuery.data?.latitude ?? mapLocationQuery.data?.latitude ?? selectedCityFallback?.latitude ?? demoPrediction.latitude),
    longitude: Number(predictionQuery.data?.longitude ?? mapLocationQuery.data?.longitude ?? selectedCityFallback?.longitude ?? demoPrediction.longitude),
  }), [
    mapLocationQuery.data?.latitude,
    mapLocationQuery.data?.longitude,
    predictionQuery.data?.latitude,
    predictionQuery.data?.longitude,
    selectedCityFallback?.latitude,
    selectedCityFallback?.longitude,
  ]);
  const mapLocationUnavailable = !hasPredictionCoordinates && !selectedCityFallback && mapLocationQuery.isError;
  const activeInfoPage = additionalPages.find((page) => page.id === activeModal);
  const openNavigationPage = (page: ActiveModal) => {
    setActiveModal(page);
  };
  const isCompactNavBar = isPageScrolled && !isNavBarExpanded;
  const handleAssistantSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const message = assistantInput.trim();
    if (!message || assistantLoading) return;

    setAssistantInput("");
    setIsAssistantOpen(true);
    setAssistantError(null);
    setAssistantFollowups([]);
    setAssistantMessages((messages) => [...messages, { role: "user", content: message }]);
    setAssistantLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/assistant/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          city: selectedCityName,
          persona,
          history: assistantMessages.slice(-8),
          force_refresh: false,
        }),
      });
      if (!response.ok) throw new Error(`Assistant request failed (${response.status})`);
      const result = await response.json() as AssistantChatResponse;
      if (typeof result.response !== "string" || !result.response.trim()) {
        throw new Error("The assistant returned an empty response.");
      }
      setAssistantMessages((messages) => [...messages, { role: "assistant", content: result.response }]);
      setAssistantFollowups(result.suggested_followups ?? []);
    } catch (error) {
      setAssistantError(error instanceof TypeError
        ? "Could not reach the assistant backend. Check that it is running and try again."
        : error instanceof Error ? error.message : "Unable to reach the assistant. Please try again.");
    } finally {
      setAssistantLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <Suspense fallback={null}><MapBackdrop {...coordinates} /></Suspense>
      <main className="main-content" id="top">
        <header className={`topbar${isCompactNavBar ? " is-compact" : ""}`}>
          <a
            className="brand topbar-brand"
            href="#top"
            aria-label={isPageScrolled ? isNavBarExpanded ? "Collapse navigation bar" : "Expand navigation bar" : "ClimaCore home"}
            aria-expanded={isPageScrolled ? isNavBarExpanded : undefined}
            title={isPageScrolled ? isNavBarExpanded ? "Collapse navigation" : "Expand navigation" : "ClimaCore home"}
            onClick={(event) => {
              if (!isPageScrolled) return;
              event.preventDefault();
              setIsAssistantOpen(false);
              setIsNavBarExpanded((expanded) => !expanded);
            }}
          >
            <img className="brand-logo" src={logoUrl} alt="ClimaCore" />
            <div className="topbar-brand-text">
              <strong>Clima<span>Core</span></strong>
              <small>SUPER SMART WEATHER ASSISTANT</small>
            </div>
          </a>
          <form
            className="assistant-search"
            onSubmit={handleAssistantSubmit}
            role="search"
          >
            <Globe2 size={16} aria-hidden="true" />
            <input
              type="search"
              aria-label="Ask the Clima-Core assistant"
              aria-controls="assistant-panel"
              aria-expanded={isAssistantOpen}
              placeholder="How may I help you today?"
              value={assistantInput}
              onChange={(event) => setAssistantInput(event.target.value)}
            />
            <button
              type="button"
              className="assistant-history-toggle"
              aria-label={isAssistantOpen ? "Close chat history" : "Show previous chats"}
              aria-controls="assistant-panel"
              aria-expanded={isAssistantOpen}
              title="Show previous chats"
              onClick={() => setIsAssistantOpen((open) => !open)}
            >
              <History size={15} aria-hidden="true" />
            </button>
            <button type="submit" aria-label="Send query" disabled={assistantLoading || !assistantInput.trim()}>
              <Send size={15} aria-hidden="true" />
            </button>
          </form>
          <div className="topbar-modal-triggers">
            <button
              type="button"
              className="disaster-replay-circle-btn"
              onClick={() => setIsDisasterReplayOpen(true)}
              title="Disaster Replay Time Machine (Catastrophe Backtester)"
              aria-label="Disaster Replay Time Machine"
              tabIndex={isCompactNavBar ? -1 : 0}
            >
              <RotateCcw size={14} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="war-room-circle-btn"
              onClick={() => setIsCrisisCouncilOpen(true)}
              title="Crisis Council War Room"
              aria-label="Crisis Council War Room"
              tabIndex={isCompactNavBar ? -1 : 0}
            >
              !
            </button>
          </div>
          {isAssistantOpen && !isCompactNavBar && (
            <section className="assistant-panel" id="assistant-panel" aria-label="Clima-Core assistant">
              <div className="assistant-panel-header">
                <div><strong>Previous chats</strong><span>Clima-Core assistant · Grounded in live environmental data</span></div>
                <button type="button" className="assistant-close" aria-label="Close assistant" onClick={() => setIsAssistantOpen(false)}><X size={16} /></button>
              </div>
              <div className="assistant-conversation" aria-live="polite">
                {assistantMessages.length === 0 && <p className="assistant-welcome">No previous chats yet. Ask about weather or environmental risks for {selectedCityName} to get started.</p>}
                {assistantMessages.slice(-8).map((entry, index) => (
                  <div className={`assistant-message ${entry.role}`} key={`${index}-${entry.role}`}>
                    {entry.role === "assistant" ? <ReactMarkdown>{entry.content}</ReactMarkdown> : <p>{entry.content}</p>}
                  </div>
                ))}
                {assistantLoading && <p className="assistant-status" role="status">Checking current conditions…</p>}
                {assistantError && <p className="assistant-error" role="alert">{assistantError}</p>}
              </div>
              {assistantFollowups.length > 0 && (
                <div className="assistant-followups" aria-label="Suggested follow-up questions">
                  {assistantFollowups.map((followup) => (
                    <button type="button" key={followup} onClick={() => setAssistantInput(followup)}>{followup}</button>
                  ))}
                </div>
              )}
            </section>
          )}
        </header>

        <div className="page-wrap">
          <div className="weather-hero-region">
            <section className="page-heading">
              <div>
                <div className="eyebrow"><CalendarDays className="eyebrow-calendar" size={14} strokeWidth={2} aria-hidden="true" /> {new Intl.DateTimeFormat("en-IN", { weekday: "long", day: "2-digit", month: "long", year: "numeric" }).format(new Date()).toUpperCase()} <span className="eyebrow-separator">/</span> {getSeasonLabel(new Date().getMonth())}</div>
                <h1>Environmental overview</h1>
                <p className="heading-subtitle">A clear view of today’s climate risks and what comes next.</p>
              </div>
            </section>

            <div className="view-controls">
              <div className="city-select-wrap">
                <span>VIEW IN</span>
                <CitySearch
                  value={city}
                  onSelect={(value) => {
                    setCity(value);
                    setActiveDay(0);
                  }}
                />
              </div>
              <div className="persona-select-wrap">
                <span>VIEW FOR</span>
                <StyledDropdown
                  ariaLabel="Select persona"
                  value={persona}
                  options={people.map((item) => ({ value: item.id ?? item.key ?? "", label: item.title }))}
                  onChange={setPersona}
                />
              </div>
            </div>

            <section className={`threat-banner ${tone}`}>
              <div className="threat-left">
                <div><div className="threat-kicker">COMPOUND RISK · {cityName.toUpperCase()}</div><div className="threat-title">{tone === "critical" ? "Critical conditions emerging" : tone === "high" ? "Elevated risks across the region" : tone === "moderate" ? "Moderate conditions to monitor" : "Conditions are within normal range"}</div></div>
              </div>
              <div className="threat-right">
                <div className="risk-score"><strong>{risk}</strong><span>/100</span></div>
                <span className="risk-pill"><span />{peakRiskCategory}</span>
              </div>
            </section>
          </div>

          <section className="section-block">
            <div className="section-heading">
              <div><h2>Today’s operational outlook</h2></div>
            </div>
            <div className="top-grid">
              <article className="card map-card" id="map-section">
                <div className="card-topline map-title">
                  <div className="map-title-content">
                    <span className="card-label">GEOSPATIAL INTELLIGENCE</span>
                    <div className="map-location-row">
                      <h3><MapPin size={16} /><span>{cityName}{cityState && cityState.toLowerCase() !== cityName.toLowerCase() ? `, ${cityState}` : ""}</span></h3>
                      <button
                        type="button"
                        className={`realism-toggle ${realismEnabled ? "is-on" : ""}`}
                        role="switch"
                        aria-label="Realism"
                        aria-checked={realismEnabled}
                        onClick={() => setRealismEnabled((enabled) => !enabled)}
                      >
                        <span>Realism</span>
                        <span className="realism-toggle-track" aria-hidden="true"><span /></span>
                      </button>
                    </div>
                  </div>
                </div>
                <div className="map-frame">
                  <Suspense fallback={<div className="map-loading">Loading geospatial view…</div>}>
                    <MapCanvas key={mapReset} {...coordinates} realismEnabled={realismEnabled} />
                  </Suspense>
                  <div className="map-overlay"><span className="map-live-dot" /> RISK EXPOSURE</div>
                  {mapLocationUnavailable && <div className="map-location-warning" role="status">Could not locate {cityName}; showing the default map center.</div>}
                  <div className="map-legend"><span><i className="legend-core" />Urban core</span><span><i className="legend-ring" />Exposure radius</span><span><i className="legend-marker" />City center</span></div>
                  <div className="map-attribution">{realismEnabled ? "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community" : "© OpenStreetMap contributors © OpenFreeMap"}</div>
                </div>
                <div className="map-stats"><div><span className="stat-icon hospital">+</span><span><strong>{Math.round(hospitalCount).toLocaleString()}</strong><small>Hospitals</small></span></div><div><span className="stat-icon school">S</span><span><strong>{Math.round(schoolCount).toLocaleString()}</strong><small>Schools</small></span></div><div><span className="stat-icon power"><Activity size={14} /></span><span><strong>{Math.round(powerAssetCount).toLocaleString()}</strong><small>Power assets</small></span></div><button className="text-button" onClick={() => setMapReset((value) => value + 1)}>Recenter map <LocateFixed size={14} /></button></div>
              </article>

              <article className="card tactical-card">
                <div className="card-topline"><span className="card-label">PERSONA BRIEFING</span><span className="date-chip"><CalendarDays size={13} /> TODAY</span></div>
                <div className="briefing-layout">
                  <div className="briefing-summary">
                    <div className="briefing-title-row"><div><h3>{selectedPersona?.title ?? "Farmer"}’s outlook</h3><p>Conditions may affect your day-to-day operations.</p></div></div>
                    <div className="briefing-score">
                      <div className="workability" style={{ "--score": `${Math.max(0, workability)}%` } as React.CSSProperties}><span>{workability}</span></div>
                      <div className="briefing-score-meta"><span className="briefing-score-label">WORKABILITY SCORE</span><span className={`status-tag status-${personaStatusClass}`}><span />{personaStatus.replace(/_/g, " ")}</span></div>
                    </div>
                  </div>
                  <div className="briefing-content">
                    <div className={`restriction-box ${noActiveRestrictions ? "no-restriction" : ""}`}><div className="restriction-heading">{noActiveRestrictions ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />} {noActiveRestrictions ? "NO ACTIVE RESTRICTIONS" : "OPERATIONAL GUIDANCE"}</div><p>{restrictionText}</p></div>
                    <div className="advisory-panel">
                      <div className="action-checklist"><span>SECTOR ADVISORIES</span>{visibleActionItems.map((item, index) => <div key={`${item}-${index}`}><Check size={12} /> {item}</div>)}</div>
                      {actionItems.length > 2 && (
                        <div className="briefing-footer">
                          <button onClick={() => setShowAllAdvisories((value) => !value)} aria-label={showAllAdvisories ? "Show fewer advisories" : "Show all advisories"}>
                            {showAllAdvisories ? "Less" : "More"} <ChevronRight size={14} />
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </article>
            </div>
          </section>

          <section className="section-block forecast-section" id="forecast">
            <div className="section-heading forecast-heading">
              <div><div className="section-kicker">7-DAY FORECAST</div><h2>Risk outlook</h2></div>
              <div className="forecast-legend"><span><i className="legend-risk" /> Compound risk</span><span><i className="legend-rain" /> Rainfall</span></div>
            </div>
            <div className="card forecast-card">
              <div className="day-strip">
                {days.slice(0, 7).map((day, index) => {
                  const score = Math.round(Number(findMetric(day, ["compound_risk_score"]) ?? (index === 0 ? risk : 42 + (index * 9) % 40)));
                  const dayTemperature = Number(findMetric(day, ["temp_max", "temperature_max"]) ?? 33);
                  const dayRainfall = Number(findMetric(day, ["precipitation_mm", "rainfall_mm"]) ?? 8);
                  return <button key={day.date ?? index} className={`day-item ${activeDay === index ? "selected" : ""}`} onClick={() => setActiveDay(index)}>
                    <span className="day-name">{day.day_name ?? (index === 0 ? "Today" : new Intl.DateTimeFormat("en", { weekday: "short" }).format(new Date(`${day.date}T12:00:00`)))}</span>
                    <span className={`day-icon ${dayRainfall > 1 ? "rainy" : ""}`}>{dayRainfall > 1 ? <CloudDrizzle size={19} /> : dayRainfall === 0 ? <Sun size={19} /> : <CloudSun size={19} />}</span>
                    <span className="day-temperature">{Math.round(dayTemperature)}°</span>
                    <span className="day-rain"><Droplets size={12} />{Math.round(dayRainfall)} mm</span>
                    <span className={`day-risk ${riskTone(score)}`}><i />{score}</span>
                  </button>;
                })}
              </div>
              <div className="interval-tabs" aria-label="Conformal forecast variable">
                {(Object.keys(intervalConfigs) as IntervalMetric[]).map((key) => <button key={key} className={intervalMetric === key ? "active" : ""} onClick={() => setIntervalMetric(key)}>{intervalConfigs[key].label}</button>)}
              </div>
              <div className="forecast-chart-header"><div><strong>{intervalConfig.label} · 90% prediction interval</strong><span>{activeForecast?.day_name ?? "Selected day"} point estimate and uncertainty range</span></div><div className="temp-summary"><strong>{selectedInterval.point.toFixed(1)} {intervalConfig.unit}</strong><span>{selectedInterval.lower.toFixed(1)}–{selectedInterval.upper.toFixed(1)} {intervalConfig.unit}</span></div></div>
              <div className="chart-wrap">
                <Suspense fallback={<div className="map-loading">Loading prediction chart…</div>}>
                  <ConformalChart data={intervalData.map(({ date, point, lower, band, upper }) => ({ date: String(date ?? ""), point, lower, band, upper }))} unit={intervalConfig.unit} label={intervalConfig.label} domain={[Math.max(0, intervalMin - intervalPadding), intervalMax + intervalPadding]} />
                </Suspense>
              </div>
              <div className="chart-footnote"><BadgeCheck size={14} /> {intervalNote}</div>
            </div>
          </section>

          <section className="section-block intelligence-section">
            <div className="section-heading"><div><div className="section-kicker">ENVIRONMENTAL INTELLIGENCE</div><h2>What’s driving risk today</h2></div><button className="text-button" onClick={() => document.getElementById("intel-grid")?.scrollIntoView({ behavior: "smooth" })}>View all signals <ArrowUpRight size={14} /></button></div>
            <div className="intel-grid" id="intel-grid">
              <article className="card intel-card">
                <div className="intel-head"><div className="intel-icon flood"><Droplets size={17} /></div><div><h3>Flood risk</h3><span>HYDROLOGICAL MODEL</span></div><span className={`mini-status ${String(findMetric(floodRisk, ["hazard_tier"]) ?? "WATCH").toLowerCase() === "normal" ? "good" : "watch"}`}>{String(findMetric(floodRisk, ["hazard_tier"]) ?? "WATCH").replace(/_/g, " ")}</span></div>
                <div className="metric-row"><div><strong>{runoff.toFixed(1)}</strong><span>mm runoff</span></div><div className="mini-chart flood-chart"><span style={{ height: "28%" }} /><span style={{ height: "48%" }} /><span style={{ height: "37%" }} /><span style={{ height: "64%" }} /><span style={{ height: "52%" }} /><span style={{ height: "82%" }} /><span style={{ height: "70%" }} /></div></div>
                <div className="intel-footer"><span>API-14 <b>{antecedentRain.toFixed(1)} mm</b></span><span>Hazard probability <b>{floodProbability.toFixed(2)}</b></span></div>
              </article>
              <article className="card intel-card">
                <div className="intel-head"><div className="intel-icon heat"><Sun size={17} /></div><div><h3>Heat stress</h3><span>THERMAL EXPOSURE</span></div><span className={`mini-status ${heatAlertLabel === "NO ALERT" ? "good" : "moderate"}`}>{heatAlertLabel}</span></div>
                <div className="heat-metrics"><div><strong>{dryBulb.toFixed(0)}°</strong><span>Dry bulb</span></div><div><strong>{apparentTemp.toFixed(0)}°</strong><span>Feels like</span></div><div><strong>{wetBulb.toFixed(0)}°</strong><span>Wet bulb</span></div></div>
                <div className="intel-footer"><span>Urban heat island delta</span><span className="up-trend"><ArrowUpRight size={13} /> +{uhiDelta.toFixed(1)}°</span></div>
              </article>
              <article className="card intel-card">
                <div className="intel-head"><div className="intel-icon power"><BatteryWarning size={17} /></div><div><h3>Power grid</h3><span>DEMAND FORECAST</span></div><span className="mini-status watch">{textMetric(["grid_stability_tier"], "ELEVATED", powerGrid).replace(/_/g, " ")}</span></div>
                <div className="metric-row"><div><strong>{Math.round(peakLoad).toLocaleString()}</strong><span>MW peak load</span></div><div className="demand-pill"><ArrowUpRight size={14} />{coolingSurge.toFixed(2)}× <small>baseline</small></div></div>
                <div className="load-meter"><span style={{ width: `${Math.min(100, Math.max(0, blackoutProbability * 100))}%` }} /></div>
                <div className="intel-footer"><span>Blackout probability</span><b>{(blackoutProbability * 100).toFixed(1)}%</b></div>
              </article>
              <article className="card intel-card">
                <div className="intel-head"><div className="intel-icon air"><Wind size={17} /></div><div><h3>Air quality</h3><span>AMBIENT CONDITIONS</span></div><span className={`mini-status ${aqiStatusClass}`}>{aqiCategory.toUpperCase()}</span></div>
                <div className="aqi-row"><div className="aqi-number">{Math.round(aqi)}<span> AQI</span></div><div className="pollutant-bars">{[["PM2.5", pm25], ["PM10", pm10], ["NO₂", no2], ["SO₂", so2], ["CO", co], ["O₃", ozone]].map(([label, amount]) => { const value = typeof amount === "number" ? amount : null; return <div key={String(label)}><span>{label}</span><i><b style={{ width: `${value === null ? 0 : Math.min(100, Math.max(5, value / 250 * 100))}%` }} /></i><small>{value === null ? "—" : value.toFixed(0)}</small></div>; })}</div></div>
                <div className="intel-footer"><span>Boundary-layer stagnation</span><span className="inline-good"><Check size={13} /> {stagnation}</span></div>
              </article>
              <article className="card intel-card">
                <div className="intel-head"><div className="intel-icon agri"><Sprout size={17} /></div><div><h3>AgriSense</h3><span>AGRICULTURAL CONDITIONS</span></div><span className={`mini-status ${droughtIsActive ? "critical" : "good"}`}>{droughtIsActive ? "DROUGHT ALERT" : "FAVORABLE"}</span></div>
                <div className="soil-row"><div><strong>{smdi.toFixed(1)}</strong><span>Soil moisture deficit</span></div><div className="soil-scale"><div className="soil-track"><i style={{ left: `${Math.min(95, Math.max(5, ((smdi + 3) / 6) * 100))}%` }} /></div><div className="soil-labels"><span>Dry</span><span>Balanced</span><span>Wet</span></div></div></div>
                <div className="intel-footer"><span>{textMetric(["active_crop"], "Crop", agriSense)} · {droughtLabel}</span><b>{cropStage}</b></div>
              </article>
              <article className="card intel-card">
                <div className="intel-head"><div className="intel-icon water"><CloudDrizzle size={17} /></div><div><h3>Water quality</h3><span>GROUNDWATER STATUS</span></div><span className={`mini-status ${isPotable ? "good" : "critical"}`}>{potabilityLabel}</span></div>
                <div className="water-score-row"><div className="wqi-circle"><span>{Math.round(wqi)}</span><small>WQI</small></div><div className="water-copy"><strong>{waterGrade}</strong><span>{isPotable ? "Within recommended limits" : "Treatment recommended"}</span><small>{waterLimitingParameter}</small></div></div>
                <div className="intel-footer"><span>Source: CGWB</span><span className={isPotable ? "inline-good" : "inline-risk"}><ShieldCheck size={13} /> {isPotable ? "Safe" : "Treat"}</span></div>
              </article>
            </div>
          </section>

          <section className="bottom-grid">
            <article className="card cascade-card">
              <div className="section-heading compact-heading"><div><div className="section-kicker">CASCADING IMPACT</div><h2>Connected risk pathways</h2></div></div>
              <p className="card-description">{cascadeDescription}</p>
              {cascadeDisplayNodes.length > 0
                ? <div className="cascade-flow">{cascadeDisplayNodes.map((node, index) => <div className="cascade-flow-step" key={String(node.node_id ?? node.system ?? index)}>
                    <div className={`cascade-node ${index === 0 ? "hazard-node" : ""}`}><span>{index === 0 ? <Flame size={15} /> : index === 1 ? <Droplets size={15} /> : <Factory size={15} />}</span><div><strong>{String(node.system ?? node.name ?? `Impact ${index + 1}`)}</strong><small>{String(node.status ?? node.estimated_time_to_failure ?? "")}</small></div></div>
                    {index < cascadeDisplayNodes.length - 1 && <div className="cascade-connector"><i /><i /><i /><ChevronRight size={13} /></div>}
                  </div>)}</div>
                : <div className="no-cascade"><ShieldCheck size={15} /> No cascading failures are forecast for this period.</div>}
            </article>
            <article className="card sources-card">
              <div className="section-heading compact-heading"><div><div className="section-kicker">MODEL CONFIDENCE</div><h2>Signal quality</h2></div><span className="confidence-icon"><BadgeCheck size={17} /></span></div>
              <div className="confidence-score"><strong>{Math.round(confidenceCoverage)}<span>%</span></strong><span>Conformal coverage</span><span className="confidence-trend">n = {Math.round(calibrationSamples)}</span></div>
              <div className="confidence-bar"><span style={{ width: `${Math.min(100, Math.max(0, confidenceCoverage))}%` }} /></div>
              <div className="source-list"><span>Data sources</span><div><i>IMD</i><i>CWC</i><i>CGWB</i><b>+4</b></div></div>
            </article>
          </section>

          <footer className="home-footer">
            <div className="home-footer-heading">
              <span className="section-kicker">EXPLORE CLIMA-CORE</span>
              <span>Pages and resources</span>
            </div>
            <nav className="footer-page-nav" aria-label="Site pages">
              <button type="button" className="footer-page-button" onClick={() => openNavigationPage("provenance")} title="Transparency & Trust">
                <span className="footer-page-icon"><ShieldCheck size={19} aria-hidden="true" /></span>
                <span className="footer-page-label">Transparency &amp; Trust</span>
              </button>
              <button type="button" className="footer-page-button" onClick={() => openNavigationPage("privacy")} title="Privacy policy">
                <span className="footer-page-icon"><LockKeyhole size={19} aria-hidden="true" /></span>
                <span className="footer-page-label">Privacy policy</span>
              </button>
              <button type="button" className="footer-page-button" onClick={() => openNavigationPage("about")} title="About Clima-Core">
                <span className="footer-page-icon"><Info size={19} aria-hidden="true" /></span>
                <span className="footer-page-label">About Clima-Core</span>
              </button>
              {additionalPages.map(({ id, title, Icon }) => (
                <button key={id} type="button" className="footer-page-button" onClick={() => openNavigationPage(id)} title={title}>
                  <span className="footer-page-icon"><Icon size={19} aria-hidden="true" /></span>
                  <span className="footer-page-label">{title}</span>
                </button>
              ))}
            </nav>
          </footer>
        </div>
      </main>

      {activeModal && <div className="drawer-scrim" onMouseDown={(e) => { if (e.target === e.currentTarget) setActiveModal(null); }}>
        {activeModal === "provenance" ? (
          <section className="provenance-modal" role="dialog" aria-modal="true" aria-labelledby="transparency-title">
            <div className="drawer-header"><div><span className="section-kicker">ENVIRONMENTAL INTELLIGENCE</span><h2 id="transparency-title">Transparency &amp; Trust</h2></div><button className="icon-btn" onClick={() => setActiveModal(null)} aria-label="Close Transparency & Trust"><X size={18} /></button></div>
            <p className="drawer-intro">Trace the models and source data behind this environmental intelligence briefing.</p>
            <div className="drawer-section"><h3>ACTIVE MODEL ENGINES</h3><div className="model-engine-grid">{provenanceModels.map(([name, model]) => {
              const ModelIcon = getModelIcon(name);
              return <div className="provenance-row" key={name}><span><i className="engine-icon"><ModelIcon size={15} /></i>{titleCase(name)}</span><b title={String(model)}>{String(model)}</b></div>;
            })}</div><div className="hash-note"><ShieldCheck size={17} /><div><strong>Model versions listed</strong><span>SHA-256 hashes are not included in the available prediction API response.</span></div></div></div>
            <div className="drawer-section"><h3>CANONICAL DATA SOURCES</h3><div className="source-chips">{["IMD", "CWC", "CGWB", "CAMELS-IND", "HydroRIVERS", "WorldPop"].map((item) => <span key={item}><Check size={12} />{item}</span>)}</div></div>
            <div className="drawer-section"><h3>PHYSICAL UNITS</h3><div className="physical-units-grid">{Object.entries(prediction.physical_units_manifest ?? { temperature: "°C", precipitation: "mm", wind_speed: "m/s", peak_load: "MW", air_quality: "μg/m³" }).map(([key, unit]) => <div className="provenance-row" key={key}><span>{titleCase(key)}</span><b>{unit}</b></div>)}</div></div>
            <div className="drawer-footer">CLIMA-CORE · SCIENCE YOU CAN TRACE</div>
          </section>
        ) : activeModal === "privacy" ? <PrivacyPolicy onClose={() => setActiveModal(null)} />
          : activeModal === "about" ? <AboutUs onClose={() => setActiveModal(null)} />
            : activeInfoPage ? <EmptyInfoPageModal title={activeInfoPage.title} onClose={() => setActiveModal(null)}>
              {activeInfoPage.id === "methodology" ? <Methodology />
                : activeInfoPage.id === "data-sources" ? <DataSources />
                  : activeInfoPage.id === "model-performance" ? (
                    <Suspense fallback={<div className="drawer-intro">Loading model performance…</div>}>
                      <ModelPerformance />
                    </Suspense>
                  )
                  : null}
            </EmptyInfoPageModal>
              : null}
      </div>}

      {isCrisisCouncilOpen && (
        <Suspense fallback={null}>
          <CrisisCouncilModal
            city={selectedCityName}
            initialTempMax={dryBulb}
            initialRainfall={runoff}
            initialGridLoad={peakLoad}
            onClose={() => setIsCrisisCouncilOpen(false)}
          />
        </Suspense>
      )}

      {isDisasterReplayOpen && (
        <Suspense fallback={null}>
          <DisasterReplayModal
            currentCity={selectedCityName}
            onClose={() => setIsDisasterReplayOpen(false)}
          />
        </Suspense>
      )}
    </div>
  );
}

export default App;
