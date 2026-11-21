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
  BatteryWarning, Building2, CalendarDays, Check, ChevronDown, ChevronRight,
  BarChart3, CircleHelp, CloudDrizzle, CloudRain, CloudSun, Compass, Database, Droplets, Factory, Flame,
  Globe2, History, Info, Layers, Send,
  LocateFixed, LockKeyhole, MapPin, MoreHorizontal, RotateCcw,
  ShieldCheck, Sprout, Sun, Thermometer, Waves, Wind, X, Zap,
} from "lucide-react";

const MapCanvas = lazy(() => import("./MapCanvas"));
const MapBackdrop = lazy(() => import("./MapBackdrop"));
const ConformalChart = lazy(() => import("./ConformalChart"));
const ModelPerformance = lazy(() => import("./ModelPerformance"));
const CrisisCouncilModal = lazy(() => import("./CrisisCouncilModal"));
const DisasterReplayModal = lazy(() => import("./DisasterReplayModal"));

type Persona = { id: string; key?: string; title: string; icon?: string; description?: string };
type PersonasResponse = Persona[] | { personas?: Persona[] };
type PersonaBriefing = {
  city: string;
  persona_id: string;
  persona_title: string;
  operational_guidance: string;
  advisories: string[];
  workability_score?: number;
  status_level?: string;
  provider?: string;
};
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
  placeholder?: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  disabled?: boolean;
};

function StyledDropdown({ ariaLabel, value, placeholder = "Select occupation", options, onChange, disabled = false }: StyledDropdownProps) {
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
        <span className={`styled-dropdown-value ${!selectedOption ? "is-placeholder" : ""}`}>{selectedOption?.label ?? (value || placeholder)}</span>
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
    body: JSON.stringify({
      city,
      requested_personas: persona.trim() ? [persona.trim()] : [],
      force_refresh: force,
    }),
  });
  if (!response.ok) throw new Error(`Prediction request failed (${response.status})`);
  return response.json() as Promise<Prediction>;
}

function App() {
  const [city, setCity] = useState("");
  const [persona, setPersona] = useState("");
  const [activeDay, setActiveDay] = useState(0);
  const [intervalMetric, setIntervalMetric] = useState<IntervalMetric>("temperature_c");
  const [mapReset, setMapReset] = useState(0);
  const [realismEnabled, setRealismEnabled] = useState(true);
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
  const [isToolsMenuOpen, setIsToolsMenuOpen] = useState(false);
  const toolsMenuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isToolsMenuOpen) return;
    const handlePointerDown = (event: MouseEvent) => {
      if (toolsMenuRef.current && !toolsMenuRef.current.contains(event.target as Node)) {
        setIsToolsMenuOpen(false);
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsToolsMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("mousedown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isToolsMenuOpen]);

  useEffect(() => {
    let lastScrollY = window.scrollY;
    const updateNavBarOnScroll = () => {
      const currentScrollY = window.scrollY;
      setIsPageScrolled((prevScrolled) => {
        if (!prevScrolled && currentScrollY > 90) {
          setIsAssistantOpen(false);
          setIsToolsMenuOpen(false);
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
    enabled: Boolean(city.trim()),
  });
  const personaBriefingQuery = useQuery({
    queryKey: ["persona-briefing", city, persona, activeDay],
    queryFn: async () => {
      const response = await fetch(`${API_BASE}/api/persona/briefing`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          city,
          persona,
          day_offset: activeDay,
        }),
      });
      if (!response.ok) {
        throw new Error("Failed to fetch persona briefing");
      }
      return response.json() as Promise<PersonaBriefing>;
    },
    enabled: Boolean(city.trim() && persona.trim()),
    staleTime: 60_000,
  });
  const hasSelectedLocation = Boolean(city.trim());
  const hasSelectedLocationAndOccupation = Boolean(city.trim() && persona.trim());
  const selectedCityName = city ? city.split(",")[0].trim() : "";
  const selectedStateName = city.includes(",") ? city.split(",")[1]?.trim().toLocaleLowerCase() : undefined;
  const selectedCityFallback = selectedCityName ? fallbackCityCoordinates[selectedCityName.toLocaleLowerCase()] : undefined;
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
  const prediction = predictionQuery.data ?? (hasSelectedLocation ? demoPrediction : ({} as Prediction));
  const people = (personasQuery.data?.length ? personasQuery.data : demoPersonas).map((item) =>
    item.title.toLowerCase() === "student & school"
      ? { ...item, title: "Student & Education" }
      : item,
  );
  const days = hasSelectedLocation ? (prediction.daily_predictions?.length ? prediction.daily_predictions : demoDays) : [];
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
  const intervalMin = intervalData.length ? Math.min(...intervalData.map((item) => item.lower)) : 0;
  const intervalMax = intervalData.length ? Math.max(...intervalData.map((item) => item.upper)) : 100;
  const intervalPadding = (intervalMax - intervalMin) * 0.12 || 1;
  const selectedInterval = intervalData[activeDay] ?? intervalData[0] ?? { point: 0, lower: 0, upper: 0 };
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
  const riskValue = hasSelectedLocation
    ? Number(findMetric(prediction, ["peak_compound_risk"]) ?? findMetric(activeForecast, ["compound_risk_score"]) ?? 68)
    : null;
  const risk = riskValue != null && Number.isFinite(riskValue) ? Math.round(riskValue) : null;
  const peakRiskCategory = hasSelectedLocation
    ? String(prediction.peak_risk_category ?? (risk != null ? titleCase(riskTone(risk)) : "LOW")).toUpperCase()
    : "AWAITING SELECTION";
  const tone = hasSelectedLocation && risk != null
    ? (peakRiskCategory.includes("CRITICAL") ? "critical"
      : peakRiskCategory.includes("HIGH") ? "high"
        : peakRiskCategory.includes("MODERATE") ? "moderate"
          : "low")
    : "low";
  const metric = (names: string[], fallback: number, source: unknown = activeForecast): number | null => {
    if (!hasSelectedLocation) return null;
    const raw = findMetric(source, names) ?? findMetric(prediction, names);
    const value = Number(raw);
    return Number.isFinite(value) ? value : fallback;
  };
  const textMetric = (names: string[], fallback: string, source: unknown = activeForecast): string => {
    if (!hasSelectedLocation) return "";
    const value = findMetric(source, names);
    return typeof value === "string" && value.trim() ? value : fallback;
  };
  const pollutant = (key: string, fallback: number): number | null => {
    if (!hasSelectedLocation) return null;
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
  const galeGust = metric(["gale_wind_gust_kmh", "wind_gusts_10m_max"], 24);
  const windSpeed = metric(["wind_speed_kmh", "wind_speed_10m_max"], 14);
  const histPercentile = metric(["historical_percentile", "climatology_percentile"], 52);
  const ventilationTier = textMetric(["ventilation_tier"], "Moderate", atmosphericPhysics);
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

  // Transboundary Smoke Model
  const isBurningSeason = [4, 5, 10, 11].includes(new Date().getMonth() + 1);
  const isNorthPlainsCity = /delhi|punjab|haryana|lucknow|kanpur|chandigarh|jaipur|agra|varanasi|patna|noida|gurugram|ghaziabad|amritsar|ludhiana/i.test(city);
  const smokeAlignmentScore = hasSelectedLocation
    ? (isNorthPlainsCity && isBurningSeason ? 0.84 : isNorthPlainsCity ? 0.46 : 0.18)
    : null;
  const smokeAlert = hasSelectedLocation
    ? (isNorthPlainsCity && isBurningSeason ? "CORRIDOR ACTIVE" : isNorthPlainsCity ? "MODERATE DRIFT" : "MINIMAL DRIFT")
    : "AWAITING LOCATION";
  const smokeStatusClass = isNorthPlainsCity && isBurningSeason ? "critical" : isNorthPlainsCity ? "watch" : "good";

  // UHI Nocturnal Canopy Model
  const uhiCanopyDelta = uhiDelta != null ? uhiDelta : 2.4;
  const uhiAlert = hasSelectedLocation
    ? (uhiCanopyDelta >= 3.0 ? "INTENSE DOME" : uhiCanopyDelta >= 1.8 ? "MODERATE DOME" : "LOW CANOPY")
    : "AWAITING LOCATION";
  const uhiStatusClass = uhiCanopyDelta >= 3.0 ? "critical" : uhiCanopyDelta >= 1.8 ? "watch" : "good";

  // Bioclimatic Regime Classification (44-Yr GMM)
  const getBioclimaticRegime = (cityName: string) => {
    const c = cityName.toLowerCase();
    if (/mumbai|chennai|kolkata|kochi|cochin|visakhapatnam|surat|mangalore|panaji|goa|puri|bhubaneswar|trivandrum|thiruvananthapuram/i.test(c)) {
      return {
        title: "Humid Tropical Coastal",
        code: "Zone A · Maritime Humid",
        primaryHazard: "Cyclone & Maritime Humidity",
        confidence: "98.2%",
      };
    }
    if (/shimla|dehradun|srinagar|gangtok|guwahati|shillong|dharamsala|rishikesh|leh|jammu/i.test(c)) {
      return {
        title: "Sub-Himalayan Montane",
        code: "Zone D · High Elevation",
        primaryHazard: "Flash Hydrology & Slope Shear",
        confidence: "97.5%",
      };
    }
    if (/bengaluru|bangalore|hyderabad|pune|nagpur|coimbatore|mysuru|mysore|indore|nashik|aurangabad/i.test(c)) {
      return {
        title: "Deccan Plateau Ventilation",
        code: "Zone B · Semi-Arid Tableland",
        primaryHazard: "Strong Seasonal Wind Shear",
        confidence: "96.4%",
      };
    }
    return {
      title: "Indo-Gangetic Inversion",
      code: "Zone C · Alluvial Basin",
      primaryHazard: "Thermal Extremes & Smog Trapping",
      confidence: "95.8%",
    };
  };
  const bioclimate = getBioclimaticRegime(city);

  // Convective Cloudburst Burst Rate
  const precipMm = metric(["precipitation_mm", "precipitation_sum"], 0);
  const rainProb = metric(["rain_probability"], 20);
  const convectiveBurstRate = hasSelectedLocation && precipMm != null ? Number((precipMm * 1.6).toFixed(1)) : null;
  const cloudburstAlert = hasSelectedLocation
    ? (precipMm != null && precipMm >= 40 ? "BURST WARNING" : precipMm != null && precipMm >= 15 ? "CONVECTIVE CELL" : "NORMAL RATE")
    : "AWAITING LOCATION";
  const cloudburstStatusClass = precipMm != null && precipMm >= 40 ? "critical" : precipMm != null && precipMm >= 15 ? "watch" : "good";

  // Physiological Heat Stress (WBGT ISO 7243)
  const dryBulbVal = dryBulb != null ? dryBulb : 32;
  const wetBulbVal = wetBulb != null ? wetBulb : 26;
  const wbgt = hasSelectedLocation ? Number((0.7 * wetBulbVal + 0.3 * dryBulbVal).toFixed(1)) : null;
  const workRestCycle = wbgt != null
    ? (wbgt >= 32 ? "15m work / 45m rest" : wbgt >= 30 ? "30m work / 30m rest" : wbgt >= 28 ? "45m work / 15m rest" : "Continuous safe")
    : "--";
  const wbgtAlert = hasSelectedLocation
    ? (wbgt != null && wbgt >= 32 ? "REST RESTRICTION" : wbgt != null && wbgt >= 30 ? "CAUTION" : "NORMAL CAPACITY")
    : "AWAITING LOCATION";
  const wbgtStatusClass = wbgt != null && wbgt >= 32 ? "critical" : wbgt != null && wbgt >= 30 ? "watch" : "good";
  const briefingData = personaBriefingQuery.data;
  const statusValue = briefingData?.status_level ?? activePersonaData?.status_level;
  const personaStatus = typeof statusValue === "string"
    ? statusValue
    : risk != null && risk >= 80 ? "CRITICAL" : risk != null && risk >= 60 ? "CAUTION" : "NORMAL";
  const personaStatusClass = personaStatus.toLowerCase().replace(/[^a-z0-9]+/g, "-");
  const impactScore = Number(findMetric(activePersonaData, ["impact_score"]));
  const suppliedWorkability = briefingData?.workability_score != null
    ? briefingData.workability_score
    : Number(findMetric(activePersonaData, ["workability_score"]));
  const workability = Math.round(Number.isFinite(suppliedWorkability)
    ? suppliedWorkability
    : Number.isFinite(impactScore) ? 100 - impactScore : (risk != null ? 100 - risk : 100));

  const rawRestrictions = briefingData?.operational_guidance
    ? [briefingData.operational_guidance]
    : Array.isArray(activePersonaData?.operational_restrictions)
      ? activePersonaData.operational_restrictions.filter((item): item is string => typeof item === "string")
      : [];

  const rawAdvisories = briefingData?.advisories && briefingData.advisories.length > 0
    ? briefingData.advisories
    : Array.isArray(activePersonaData?.advisories)
      ? activePersonaData.advisories.filter((item): item is string => typeof item === "string")
      : [];

  const noActiveRestrictions = !hasSelectedLocationAndOccupation || (rawRestrictions.length > 0 && rawRestrictions.every((item) => /no .*restrictions? active|no restrictions/i.test(item)));
  const restrictionText = !hasSelectedLocationAndOccupation
    ? "Select a location and occupation above to evaluate active restrictions and operational guidance."
    : (rawRestrictions.length
      ? rawRestrictions.join(" ")
      : "Mandate scheduled shaded hydration intervals and adjust outdoor shift intensity based on current thermal conditions.");

  const fallbackAdvisories = [
    "Hydration Protocol: Drink 250ml water/electrolytes every 20-30 minutes; do not wait until thirsty.",
    "Shift Gaps: Plan shaded rest intervals during peak afternoon sun between 12:00 PM and 3:30 PM.",
    "Protective Gear: Equip UV-blocking breathable apparel, sunglasses, and cooling neck gaiters.",
    "Route Safety: Watch for sudden road asphalt slickness and localized congestion corridors."
  ];

  const actionItems = !hasSelectedLocationAndOccupation
    ? ["Select a location and occupation above to view customized sector advisories."]
    : (rawAdvisories.length
      ? rawAdvisories
      : fallbackAdvisories);

  const visibleActionItems = showAllAdvisories ? actionItems : actionItems.slice(0, 4);
  const hospitalCount = metric(["hospitals_15km", "hospital_count"], 42, floodRisk);
  const schoolCount = metric(["schools_15km", "school_count"], 318, floodRisk);
  const powerAssetCount = metric(["power_substations_15km", "substations_15km", "power_assets_15km"], 16, prediction.spatial_resolution_metadata);
  const modelVersions = asRecord(asRecord(prediction.system_provenance)?.champion_models);
  const provenanceModels = Object.entries(modelVersions ?? {
    flood_risk: "FloodRisk V3",
    heat_risk: "HeatRisk V4",
    hydrological_model: "Hybrid-GR4J v2.1",
  });
  const cascadeDescription = !hasSelectedLocation
    ? "Select a location above to compute cross-sector infrastructure cascade simulations."
    : typeof findMetric(activeChain, ["trigger_condition"]) === "string"
      ? String(findMetric(activeChain, ["trigger_condition"]))
      : predictionQuery.data ? "No active cascade chains reported for this forecast."
        : "How today’s conditions could ripple across critical systems.";
  const cascadeDisplayNodes: Record<string, unknown>[] = hasSelectedLocation
    ? (cascadeNodes.length
      ? cascadeNodes.slice(0, 3)
      : predictionQuery.data ? []
        : [
            { system: "Heavy rainfall", status: "12 mm expected" },
            { system: "Urban runoff", status: "Drainage pressure" },
            { system: "Transit disruption", status: "Localized impact" },
          ])
    : [];
  const selectedPersona = people.find((item) => (item.id ?? item.key) === persona);
  const cityName = predictionQuery.data?.city ?? selectedCityName;
  const cityState = predictionQuery.data?.state ?? (city.includes(",") ? city.split(",")[1]?.trim() : "");
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
    <div className={`app-shell${hasSelectedLocationAndOccupation ? " has-selection" : " is-white-bg"}`}>
      {hasSelectedLocationAndOccupation && (
        <Suspense fallback={null}><MapBackdrop {...coordinates} /></Suspense>
      )}
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
              setIsToolsMenuOpen(false);
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
          <div className="topbar-modal-triggers" ref={toolsMenuRef}>
            <button
              type="button"
              className={`topbar-menu-circle-btn${isToolsMenuOpen ? " active" : ""}`}
              onClick={() => setIsToolsMenuOpen((open) => !open)}
              title="Simulations & emergency tools"
              aria-label="Simulations and emergency tools"
              aria-expanded={isToolsMenuOpen}
              tabIndex={isCompactNavBar ? -1 : 0}
            >
              <MoreHorizontal size={17} aria-hidden="true" />
            </button>
            {isToolsMenuOpen && !isCompactNavBar && (
              <div className="tools-menu-popover" role="dialog" aria-label="Simulations and emergency tools">
                <div className="tools-menu-header">
                  <span>TOOLS & SIMULATIONS</span>
                </div>
                <div className="tools-menu-list">
                  <button
                    type="button"
                    className="tools-menu-item"
                    onClick={() => {
                      setIsDisasterReplayOpen(true);
                      setIsToolsMenuOpen(false);
                    }}
                  >
                    <span className="disaster-replay-circle-btn" aria-hidden="true">
                      <RotateCcw size={14} />
                    </span>
                    <div className="tools-menu-item-info">
                      <strong>Disaster Replay</strong>
                      <small>Historical catastrophe time machine</small>
                    </div>
                  </button>

                  <button
                    type="button"
                    className="tools-menu-item"
                    onClick={() => {
                      setIsCrisisCouncilOpen(true);
                      setIsToolsMenuOpen(false);
                    }}
                  >
                    <span className="war-room-circle-btn" aria-hidden="true">
                      !
                    </span>
                    <div className="tools-menu-item-info">
                      <strong>Crisis Council</strong>
                      <small>Multi-agent emergency war room</small>
                    </div>
                  </button>
                </div>
              </div>
            )}
          </div>
          {isAssistantOpen && !isCompactNavBar && (
            <section className="assistant-panel" id="assistant-panel" aria-label="Clima-Core assistant">
              <div className="assistant-panel-header">
                <div><strong>Previous chats</strong><span>Clima-Core assistant · Grounded in live environmental data</span></div>
                <button type="button" className="assistant-close" aria-label="Close assistant" onClick={() => setIsAssistantOpen(false)}><X size={16} /></button>
              </div>
              <div className="assistant-conversation" aria-live="polite">
                {assistantMessages.length === 0 && <p className="assistant-welcome">No previous chats yet. Ask for weather assistance or environmental guidance for {selectedCityName} to get started.</p>}
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
                <p className="heading-subtitle">A clear view of today’s weather assistance and what comes next.</p>
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
                  placeholder="Select occupation"
                  options={people.map((item) => ({ value: item.id ?? item.key ?? "", label: item.title }))}
                  onChange={setPersona}
                />
              </div>
            </div>

            <section className={`threat-banner ${hasSelectedLocationAndOccupation ? tone : "is-unselected"}`}>
              <div className="threat-left">
                <div>
                  <div className="threat-kicker">WEATHER ASSISTANCE{cityName ? ` · ${cityName.toUpperCase()}` : ""}</div>
                  <div className="threat-title">
                    {!hasSelectedLocationAndOccupation
                      ? "Select your location and occupation above to activate live weather assistance"
                      : tone === "critical"
                        ? "Critical weather assistance required"
                        : tone === "high"
                          ? "Active weather assistance advised across the region"
                          : tone === "moderate"
                            ? "Moderate weather assistance to monitor"
                            : "Conditions are within normal range"}
                  </div>
                </div>
              </div>
              <div className="threat-right">
                <div className="risk-score">
                  <strong>{hasSelectedLocationAndOccupation && risk != null ? risk : "--"}</strong>
                  <span>/100</span>
                </div>
                <span className={`risk-pill ${!hasSelectedLocationAndOccupation ? "is-unselected" : tone}`}>
                  <span />
                  {hasSelectedLocationAndOccupation ? peakRiskCategory : "AWAITING SELECTION"}
                </span>
              </div>
            </section>
          </div>

          <section className="section-block">
            <div className="section-heading">
              <div>
                <div className="section-kicker">OPERATIONAL INTELLIGENCE</div>
                <h2>Today’s operational outlook</h2>
              </div>
            </div>
            <div className="top-grid">
              <article className="card map-card" id="map-section">
                <div className="card-topline map-title">
                  <div className="map-title-content">
                    <span className="card-label">GEOSPATIAL INTELLIGENCE</span>
                    <div className="map-location-row">
                      <h3><MapPin size={16} /><span>{cityName || "Select Location"}{cityState && cityState.toLowerCase() !== cityName.toLowerCase() ? `, ${cityState}` : ""}</span></h3>
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
                  {city.trim() ? (
                    <>
                      <Suspense fallback={<div className="map-loading">Loading geospatial view…</div>}>
                        <MapCanvas key={mapReset} {...coordinates} realismEnabled={realismEnabled} />
                      </Suspense>
                      <div className="map-overlay"><span className="map-live-dot" /> WEATHER ASSISTANCE</div>
                      {mapLocationUnavailable && <div className="map-location-warning" role="status">Could not locate {cityName || "selected city"}; showing default coordinates.</div>}
                      <div className="map-legend"><span><i className="legend-core" />Urban core</span><span><i className="legend-ring" />Exposure radius</span><span><i className="legend-marker" />City center</span></div>
                      <div className="map-attribution">{realismEnabled ? "Imagery © Esri, Maxar, Earthstar Geographics, and the GIS User Community" : "© OpenStreetMap contributors © OpenFreeMap"}</div>
                    </>
                  ) : (
                    <div className="map-blank-placeholder">
                      <div className="map-blank-icon">
                        <MapPin size={28} strokeWidth={1.8} aria-hidden="true" />
                      </div>
                      <div className="map-blank-text">
                        <strong>No Location Selected</strong>
                        <p>Search and select a city in &ldquo;VIEW IN&rdquo; above to load live satellite terrain and geospatial asset telemetry.</p>
                      </div>
                    </div>
                  )}
                </div>
                <div className="map-stats">
                  <div><span className="stat-icon hospital">+</span><span><strong>{hasSelectedLocation && hospitalCount != null ? Math.round(hospitalCount).toLocaleString() : "--"}</strong><small>Hospitals</small></span></div>
                  <div><span className="stat-icon school">S</span><span><strong>{hasSelectedLocation && schoolCount != null ? Math.round(schoolCount).toLocaleString() : "--"}</strong><small>Schools</small></span></div>
                  <div><span className="stat-icon power"><Activity size={14} /></span><span><strong>{hasSelectedLocation && powerAssetCount != null ? Math.round(powerAssetCount).toLocaleString() : "--"}</strong><small>Power assets</small></span></div>
                  <button className="text-button" disabled={!hasSelectedLocation} onClick={() => setMapReset((value) => value + 1)}>
                    Recenter map <LocateFixed size={14} />
                  </button>
                </div>
              </article>

              <article className="card tactical-card">
                <div className="card-topline">
                  <span className="card-label">PERSONA BRIEFING</span>
                  <span className="date-chip"><CalendarDays size={13} /> TODAY</span>
                </div>
                <div className="briefing-layout">
                  <div className="briefing-summary">
                    <div className="briefing-title-row">
                      <div>
                        <h3>{selectedPersona?.title ? `${selectedPersona.title}’s outlook` : "Operational outlook"}</h3>
                        <p>{selectedPersona ? "Conditions may affect your day-to-day operations." : "Select an occupation above to view targeted weather assistance."}</p>
                      </div>
                    </div>
                    <div className="briefing-score">
                      <div className="workability" style={{ "--score": `${Math.max(0, hasSelectedLocationAndOccupation ? workability : 0)}%` } as React.CSSProperties}>
                        <span>{hasSelectedLocationAndOccupation ? workability : "--"}</span>
                      </div>
                      <div className="briefing-score-meta">
                        <span className="briefing-score-label">WORKABILITY SCORE</span>
                        <span className={`status-tag status-${hasSelectedLocationAndOccupation ? personaStatusClass : "normal"}`}>
                          <span />{hasSelectedLocationAndOccupation ? personaStatus.replace(/_/g, " ") : "Awaiting selection"}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="briefing-content">
                    <div className={`restriction-box ${noActiveRestrictions ? "no-restriction" : ""}`}><div className="restriction-heading">{noActiveRestrictions ? <ShieldCheck size={16} /> : <AlertTriangle size={16} />} {noActiveRestrictions ? "NO ACTIVE RESTRICTIONS" : "OPERATIONAL GUIDANCE"}</div><p>{restrictionText}</p></div>
                    <div className="advisory-panel">
                      <div className="action-checklist"><span>SECTOR ADVISORIES</span>{visibleActionItems.map((item, index) => <div key={`${item}-${index}`}><Check size={13} /> {item}</div>)}</div>
                      {actionItems.length > 4 && (
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
              <div><div className="section-kicker">7-DAY FORECAST</div><h2>Weather assistance outlook</h2></div>
              {hasSelectedLocation && <div className="forecast-legend"><span><i className="legend-risk" /> Weather assistance index</span><span><i className="legend-rain" /> Rainfall</span></div>}
            </div>
            <div className="card forecast-card">
              {hasSelectedLocation && days.length > 0 ? (
                <>
                  <div className="day-strip">
                    {days.slice(0, 7).map((day, index) => {
                      const score = Math.round(Number(findMetric(day, ["compound_risk_score"]) ?? (index === 0 && risk != null ? risk : 42 + (index * 9) % 40)));
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
                </>
              ) : (
                <div className="forecast-blank-placeholder">
                  <div className="forecast-blank-icon">
                    <BarChart3 size={28} strokeWidth={1.8} aria-hidden="true" />
                  </div>
                  <div className="forecast-blank-text">
                    <strong>No Forecast Telemetry Available</strong>
                    <p>Search and select a city in &ldquo;VIEW IN&rdquo; above to calculate 7-day weather assistance projections and 90% conformal prediction intervals.</p>
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="section-block intelligence-section">
            <div className="section-heading"><div><div className="section-kicker">ENVIRONMENTAL INTELLIGENCE</div><h2>What’s driving weather assistance today</h2></div><button className="text-button" disabled={!hasSelectedLocation} onClick={() => document.getElementById("intel-grid")?.scrollIntoView({ behavior: "smooth" })}>View all signals <ArrowUpRight size={14} /></button></div>
            <div className="intel-grid" id="intel-grid">
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon flood"><Droplets size={17} /></div>
                  <div><h3>Flood assistance</h3><span>HYDROLOGICAL MODEL</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (String(findMetric(floodRisk, ["hazard_tier"]) ?? "WATCH").toLowerCase() === "normal" ? "good" : "watch") : "watch"}`}>
                    {hasSelectedLocation ? String(findMetric(floodRisk, ["hazard_tier"]) ?? "WATCH").replace(/_/g, " ") : "AWAITING LOCATION"}
                  </span>
                </div>
                <div className="metric-row">
                  <div><strong>{hasSelectedLocation && runoff != null ? runoff.toFixed(1) : "--"}</strong><span>{hasSelectedLocation ? " mm runoff" : ""}</span></div>
                  <div className="mini-chart flood-chart">
                    <span style={{ height: hasSelectedLocation ? "28%" : "0%" }} />
                    <span style={{ height: hasSelectedLocation ? "48%" : "0%" }} />
                    <span style={{ height: hasSelectedLocation ? "37%" : "0%" }} />
                    <span style={{ height: hasSelectedLocation ? "64%" : "0%" }} />
                    <span style={{ height: hasSelectedLocation ? "52%" : "0%" }} />
                    <span style={{ height: hasSelectedLocation ? "82%" : "0%" }} />
                    <span style={{ height: hasSelectedLocation ? "70%" : "0%" }} />
                  </div>
                </div>
                <div className="intel-footer">
                  <span>API-14 <b>{hasSelectedLocation && antecedentRain != null ? `${antecedentRain.toFixed(1)} mm` : "--"}</b></span>
                  <span>Hazard probability <b>{hasSelectedLocation && floodProbability != null ? floodProbability.toFixed(2) : "--"}</b></span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon heat"><Sun size={17} /></div>
                  <div><h3>Heat stress</h3><span>THERMAL EXPOSURE</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (heatAlertLabel === "NO ALERT" ? "good" : "moderate") : "good"}`}>
                    {hasSelectedLocation ? heatAlertLabel : "AWAITING LOCATION"}
                  </span>
                </div>
                <div className="heat-metrics">
                  <div><strong>{hasSelectedLocation && dryBulb != null ? `${dryBulb.toFixed(0)}°` : "--"}</strong><span>Dry bulb</span></div>
                  <div><strong>{hasSelectedLocation && apparentTemp != null ? `${apparentTemp.toFixed(0)}°` : "--"}</strong><span>Feels like</span></div>
                  <div><strong>{hasSelectedLocation && wetBulb != null ? `${wetBulb.toFixed(0)}°` : "--"}</strong><span>Wet bulb</span></div>
                </div>
                <div className="intel-footer">
                  <span>Urban heat island delta</span>
                  <span className="up-trend">{hasSelectedLocation && uhiDelta != null ? <><ArrowUpRight size={13} /> +{uhiDelta.toFixed(1)}°</> : "--"}</span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon power"><BatteryWarning size={17} /></div>
                  <div><h3>Power grid</h3><span>DEMAND FORECAST</span></div>
                  <span className="mini-status watch">{hasSelectedLocation ? textMetric(["grid_stability_tier"], "ELEVATED", powerGrid).replace(/_/g, " ") : "AWAITING LOCATION"}</span>
                </div>
                <div className="metric-row">
                  <div><strong>{hasSelectedLocation && peakLoad != null ? Math.round(peakLoad).toLocaleString() : "--"}</strong><span>{hasSelectedLocation ? " MW peak load" : ""}</span></div>
                  <div className="demand-pill">{hasSelectedLocation && coolingSurge != null ? <><ArrowUpRight size={14} />{coolingSurge.toFixed(2)}× <small>baseline</small></> : "--"}</div>
                </div>
                <div className="load-meter"><span style={{ width: `${hasSelectedLocation && blackoutProbability != null ? Math.min(100, Math.max(0, blackoutProbability * 100)) : 0}%` }} /></div>
                <div className="intel-footer">
                  <span>Blackout probability</span>
                  <b>{hasSelectedLocation && blackoutProbability != null ? `${(blackoutProbability * 100).toFixed(1)}%` : "--"}</b>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon air"><Wind size={17} /></div>
                  <div><h3>Air quality</h3><span>AMBIENT CONDITIONS</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? aqiStatusClass : "good"}`}>{hasSelectedLocation ? aqiCategory.toUpperCase() : "AWAITING LOCATION"}</span>
                </div>
                <div className="aqi-row">
                  <div className="aqi-number">{hasSelectedLocation && aqi != null ? Math.round(aqi) : "--"}<span>{hasSelectedLocation ? " AQI" : ""}</span></div>
                  <div className="pollutant-bars">
                    {[["PM2.5", pm25], ["PM10", pm10], ["NO₂", no2], ["SO₂", so2], ["CO", co], ["O₃", ozone]].map(([label, amount]) => {
                      const value = typeof amount === "number" ? amount : null;
                      return (
                        <div key={String(label)}>
                          <span>{label}</span>
                          <i><b style={{ width: `${value === null ? 0 : Math.min(100, Math.max(5, (value / 250) * 100))}%` }} /></i>
                          <small>{value === null ? "--" : value.toFixed(0)}</small>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Boundary-layer stagnation</span>
                  <span className="inline-good">{hasSelectedLocation ? <><Check size={13} /> {stagnation}</> : "--"}</span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon agri"><Sprout size={17} /></div>
                  <div><h3>AgriSense</h3><span>AGRICULTURAL CONDITIONS</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (droughtIsActive ? "critical" : "good") : "good"}`}>{hasSelectedLocation ? (droughtIsActive ? "DROUGHT ALERT" : "FAVORABLE") : "AWAITING LOCATION"}</span>
                </div>
                <div className="soil-row">
                  <div><strong>{hasSelectedLocation && smdi != null ? smdi.toFixed(1) : "--"}</strong><span>Soil moisture deficit</span></div>
                  <div className="soil-scale">
                    <div className="soil-track"><i style={{ left: `${hasSelectedLocation && smdi != null ? Math.min(95, Math.max(5, ((smdi + 3) / 6) * 100)) : 50}%`, opacity: hasSelectedLocation ? 1 : 0 }} /></div>
                    <div className="soil-labels"><span>Dry</span><span>Balanced</span><span>Wet</span></div>
                  </div>
                </div>
                <div className="intel-footer">
                  <span>{hasSelectedLocation ? `${textMetric(["active_crop"], "Crop", agriSense)} · ${droughtLabel}` : "Crop · Awaiting location"}</span>
                  <b>{hasSelectedLocation ? cropStage : "--"}</b>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon water"><CloudDrizzle size={17} /></div>
                  <div><h3>Water quality</h3><span>GROUNDWATER STATUS</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (isPotable ? "good" : "critical") : "good"}`}>{hasSelectedLocation ? potabilityLabel : "AWAITING LOCATION"}</span>
                </div>
                <div className="water-score-row">
                  <div className="wqi-circle"><span>{hasSelectedLocation && wqi != null ? Math.round(wqi) : "--"}</span><small>WQI</small></div>
                  <div className="water-copy">
                    <strong>{hasSelectedLocation ? waterGrade : "Awaiting location"}</strong>
                    <span>{hasSelectedLocation ? (isPotable ? "Within recommended limits" : "Treatment recommended") : "Select a city to evaluate groundwater potability"}</span>
                    <small>{hasSelectedLocation ? waterLimitingParameter : "BIS standards evaluation"}</small>
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Source: CGWB</span>
                  <span className={hasSelectedLocation ? (isPotable ? "inline-good" : "inline-risk") : "inline-good"}>{hasSelectedLocation ? <><ShieldCheck size={13} /> {isPotable ? "Safe" : "Treat"}</> : "--"}</span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon wind"><Zap size={17} /></div>
                  <div><h3>Gale & wind gust</h3><span>KINETIC HAZARD MODEL</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (galeGust != null && galeGust >= 50 ? "watch" : "good") : "good"}`}>
                    {hasSelectedLocation ? (galeGust != null && galeGust >= 60 ? "GALE WARNING" : galeGust != null && galeGust >= 45 ? "ELEVATED GUSTS" : "NORMAL GUSTS") : "AWAITING LOCATION"}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong>{hasSelectedLocation && galeGust != null ? Math.round(galeGust) : "--"}</strong>
                    <span>{hasSelectedLocation ? " km/h peak gust" : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation && windSpeed != null ? <><Wind size={13} /> {Math.round(windSpeed)} km/h <small>sustained</small></> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Beaufort rating</span>
                  <b>{hasSelectedLocation && galeGust != null ? (galeGust > 61 ? "Force 8 (Gale)" : galeGust > 49 ? "Force 7 (Near Gale)" : galeGust > 28 ? "Force 5-6 (Breeze)" : "Force 3-4 (Gentle)") : "--"}</b>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon climate"><Globe2 size={17} /></div>
                  <div><h3>Climate anomaly</h3><span>44-YEAR CLIMATOLOGY ENGINE</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (histPercentile != null && (histPercentile >= 85 || histPercentile <= 15) ? "watch" : "good") : "good"}`}>
                    {hasSelectedLocation ? (histPercentile != null && histPercentile >= 85 ? "WARM ANOMALY" : histPercentile != null && histPercentile <= 15 ? "COLD ANOMALY" : "HISTORICAL NORMAL") : "AWAITING LOCATION"}
                  </span>
                </div>
                <div className="soil-row">
                  <div>
                    <strong>{hasSelectedLocation && histPercentile != null ? `${Math.round(histPercentile)}th` : "--"}</strong>
                    <span>{hasSelectedLocation ? " percentile" : "Climatological baseline"}</span>
                  </div>
                  <div className="soil-scale">
                    <div className="soil-track">
                      <i style={{ left: `${hasSelectedLocation && histPercentile != null ? Math.min(95, Math.max(5, histPercentile)) : 50}%`, opacity: hasSelectedLocation ? 1 : 0 }} />
                    </div>
                    <div className="soil-labels"><span>Cool (P10)</span><span>Normal (P50)</span><span>Warm (P90)</span></div>
                  </div>
                </div>
                <div className="intel-footer">
                  <span>ERA5 reanalysis departure</span>
                  <b>{hasSelectedLocation && histPercentile != null ? (histPercentile >= 75 ? `+${((histPercentile - 50) / 15).toFixed(1)}σ warmer` : histPercentile <= 25 ? `-${((50 - histPercentile) / 15).toFixed(1)}σ cooler` : "Within normal band") : "--"}</b>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon stagnation"><Layers size={17} /></div>
                  <div><h3>Atmospheric stability</h3><span>BOUNDARY-LAYER VENTILATION</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? (stagnationFlag > 0 ? "watch" : "good") : "good"}`}>
                    {hasSelectedLocation ? (stagnationFlag > 0 ? "SURFACE INVERSION" : "DISPERSIVE") : "AWAITING LOCATION"}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong>{hasSelectedLocation ? (stagnationFlag > 0 ? "Trapped" : "Ventilated") : "--"}</strong>
                    <span>{hasSelectedLocation ? " mixing layer" : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation ? <><Check size={13} /> {ventilationTier} <small>ventilation</small></> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Thermal inversion status</span>
                  <span className={hasSelectedLocation ? (stagnationFlag > 0 ? "inline-risk" : "inline-good") : "inline-good"}>
                    {hasSelectedLocation ? (stagnationFlag > 0 ? <><AlertTriangle size={13} /> Inversion active</> : <><Check size={13} /> Well ventilated</>) : "--"}
                  </span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon smoke"><Flame size={17} /></div>
                  <div><h3>Transboundary smoke</h3><span>BIOMASS PLUME ADVECTION</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? smokeStatusClass : "good"}`}>
                    {smokeAlert}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong>{hasSelectedLocation && smokeAlignmentScore != null ? smokeAlignmentScore.toFixed(2) : "--"}</strong>
                    <span>{hasSelectedLocation ? " alignment index" : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation ? <><Wind size={13} /> {isBurningSeason ? "Stubble season" : "Baseline season"}</> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>NW advection trajectory</span>
                  <span className={hasSelectedLocation ? (smokeAlignmentScore != null && smokeAlignmentScore >= 0.7 ? "inline-risk" : "inline-good") : "inline-good"}>
                    {hasSelectedLocation ? (smokeAlignmentScore != null && smokeAlignmentScore >= 0.7 ? <><AlertTriangle size={13} /> Corridor aligned</> : <><Check size={13} /> Clean / Dispersed</>) : "--"}
                  </span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon uhi"><Building2 size={17} /></div>
                  <div><h3>Urban heat island</h3><span>NOCTURNAL CANOPY MODEL</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? uhiStatusClass : "good"}`}>
                    {uhiAlert}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong>{hasSelectedLocation && uhiCanopyDelta != null ? `+${uhiCanopyDelta.toFixed(1)}°` : "--"}</strong>
                    <span>{hasSelectedLocation ? "C canopy excess" : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation && uhiCanopyDelta != null ? <><Sun size={13} /> {(uhiCanopyDelta * 1.35).toFixed(1)}× <small>rural baseline</small></> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Oke thermal morphology</span>
                  <b>{hasSelectedLocation ? "Concrete thermal mass" : "--"}</b>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon bioclimate"><Compass size={17} /></div>
                  <div><h3>Bioclimatic regime</h3><span>44-YEAR CLUSTERING (GMM)</span></div>
                  <span className="mini-status good">
                    {hasSelectedLocation ? "REGIME IDENTIFIED" : "AWAITING LOCATION"}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong style={{ fontSize: "17.5px", letterSpacing: "-0.3px" }}>{hasSelectedLocation ? bioclimate.title : "--"}</strong>
                    <span style={{ display: "block", fontSize: "10.5px", marginTop: "2px", fontWeight: 500, color: "#435b4b" }}>{hasSelectedLocation ? bioclimate.code : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation ? <><Check size={13} /> {bioclimate.confidence} <small>fit</small></> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Primary hazard class</span>
                  <b>{hasSelectedLocation ? bioclimate.primaryHazard : "--"}</b>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon cloudburst"><CloudRain size={17} /></div>
                  <div><h3>Convective cloudburst</h3><span>HIGH-DENSITY IMD GAUGES</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? cloudburstStatusClass : "good"}`}>
                    {cloudburstAlert}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong>{hasSelectedLocation && convectiveBurstRate != null ? convectiveBurstRate.toFixed(1) : "--"}</strong>
                    <span>{hasSelectedLocation ? " mm/h peak burst" : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation && rainProb != null ? <><Droplets size={13} /> {Math.round(rainProb)}% <small>rain prob</small></> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Cloudburst alert tier</span>
                  <span className={hasSelectedLocation ? (precipMm != null && precipMm >= 40 ? "inline-risk" : "inline-good") : "inline-good"}>
                    {hasSelectedLocation ? (precipMm != null && precipMm >= 40 ? <><AlertTriangle size={13} /> High burst risk</> : <><Check size={13} /> Stable convection</>) : "--"}
                  </span>
                </div>
              </article>
              <article className="card intel-card">
                <div className="intel-head">
                  <div className="intel-icon wbgt"><Thermometer size={17} /></div>
                  <div><h3>Physiological heat stress</h3><span>ISO 7243 WBGT LABOR INDEX</span></div>
                  <span className={`mini-status ${hasSelectedLocation ? wbgtStatusClass : "good"}`}>
                    {wbgtAlert}
                  </span>
                </div>
                <div className="metric-row">
                  <div>
                    <strong>{hasSelectedLocation && wbgt != null ? `${wbgt.toFixed(1)}°` : "--"}</strong>
                    <span>{hasSelectedLocation ? "C WBGT index" : ""}</span>
                  </div>
                  <div className="demand-pill">
                    {hasSelectedLocation && wbgt != null ? <><Activity size={13} /> {workRestCycle}</> : "--"}
                  </div>
                </div>
                <div className="intel-footer">
                  <span>Occupational labor guidance</span>
                  <b>{hasSelectedLocation && wbgt != null ? (wbgt >= 32 ? "Mandatory cooling breaks" : wbgt >= 30 ? "Hydration & shaded rest" : "Safe continuous labor") : "--"}</b>
                </div>
              </article>
            </div>
          </section>

          <section className="bottom-grid">
            <article className="card cascade-card">
              <div className="section-heading compact-heading"><div><div className="section-kicker">CASCADING IMPACT</div><h2>Connected assistance pathways</h2></div></div>
              <p className="card-description">{cascadeDescription}</p>
              {cascadeDisplayNodes.length > 0
                ? <div className="cascade-flow">{cascadeDisplayNodes.map((node, index) => <div className="cascade-flow-step" key={String(node.node_id ?? node.system ?? index)}>
                    <div className={`cascade-node ${index === 0 ? "hazard-node" : ""}`}><span>{index === 0 ? <Flame size={15} /> : index === 1 ? <Droplets size={15} /> : <Factory size={15} />}</span><div><strong>{String(node.system ?? node.name ?? `Impact ${index + 1}`)}</strong><small>{String(node.status ?? node.estimated_time_to_failure ?? "")}</small></div></div>
                    {index < cascadeDisplayNodes.length - 1 && <div className="cascade-connector"><i /><i /><i /><ChevronRight size={13} /></div>}
                  </div>)}</div>
                : <div className="no-cascade"><ShieldCheck size={15} /> {hasSelectedLocation ? "No cascading failures are forecast for this period." : "Select a location to model cascading systemic failures."}</div>}
            </article>
            <article className="card sources-card">
              <div className="section-heading compact-heading"><div><div className="section-kicker">MODEL CONFIDENCE</div><h2>Signal quality</h2></div><span className="confidence-icon"><BadgeCheck size={17} /></span></div>
              <div className="confidence-score">
                <strong>{hasSelectedLocation && confidenceCoverage != null ? Math.round(confidenceCoverage) : "--"}<span>{hasSelectedLocation ? "%" : ""}</span></strong>
                <span>Conformal coverage</span>
                <span className="confidence-trend">{hasSelectedLocation && calibrationSamples != null ? `n = ${Math.round(calibrationSamples)}` : "Awaiting location"}</span>
              </div>
              <div className="confidence-bar"><span style={{ width: `${hasSelectedLocation && confidenceCoverage != null ? Math.min(100, Math.max(0, confidenceCoverage)) : 0}%` }} /></div>
              <div className="source-list"><span>Data sources</span><div>{hasSelectedLocation ? <><i>IMD</i><i>CWC</i><i>CGWB</i><b>+4</b></> : <i>Awaiting location</i>}</div></div>
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
            initialTempMax={dryBulb ?? undefined}
            initialRainfall={runoff ?? undefined}
            initialGridLoad={peakLoad ?? undefined}
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
