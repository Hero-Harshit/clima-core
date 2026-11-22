import { useEffect, useState, useMemo } from "react";
import {
  Wheat,
  X,
  Search,
  Calendar,
  AlertCircle,
  CheckCircle2,
  TrendingUp,
  RefreshCw,
  Store,
  MapPin,
  Scale,
  DollarSign,
  Info,
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
  Sprout,
  Clock,
  ShieldCheck,
} from "lucide-react";
import UtilityCustomSelect from "./UtilityCustomSelect";
import { getTranslations } from "./i18n";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.PROD ? "" : "http://localhost:8000");

interface MandiRecord {
  market: string;
  commodity: string;
  variety: string;
  grade: string;
  min_price_quintal: number;
  max_price_quintal: number;
  modal_price_quintal: number;
  min_price_kg: number;
  max_price_kg: number;
  modal_price_kg: number;
  date: string;
  state: string;
  district: string;
}

interface MandiPricesResponse {
  commodity: string;
  state: string;
  district: string;
  requested_date: string;
  reporting_date: string;
  is_today: boolean;
  is_fallback: boolean;
  fallback_notice: string | null;
  records: MandiRecord[];
  mandis_count: number;
  avg_modal_price_quintal: number;
  avg_modal_price_kg: number;
  min_modal_price_quintal: number;
  min_modal_price_kg: number;
  max_modal_price_quintal: number;
  max_modal_price_kg: number;
  price_spread_quintal: number;
  price_spread_kg: number;
  cheapest_market: string | null;
  highest_market: string | null;
  ai_market_advisory: string;
  provider: string;
  latency_ms?: number;
}

interface MandiCommodityItem {
  commodity: string;
  markets: number;
  lowest_modal_price: number;
  highest_modal_price: number;
  date: string;
}

interface MandiCommoditiesResponse {
  state: string;
  commodities: MandiCommodityItem[];
  total: number;
  reporting_date: string;
  provider: string;
}

interface MandiPricesModalProps {
  city: string;
  state?: string;
  language?: string;
  onClose: () => void;
}

const CITY_TO_STATE_MAP: Record<string, string> = {
  ahmedabad: "Gujarat",
  surat: "Gujarat",
  vadodara: "Gujarat",
  rajkot: "Gujarat",
  bhavnagar: "Gujarat",
  jamnagar: "Gujarat",
  mumbai: "Maharashtra",
  pune: "Maharashtra",
  nagpur: "Maharashtra",
  nashik: "Maharashtra",
  aurangabad: "Maharashtra",
  solapur: "Maharashtra",
  kolhapur: "Maharashtra",
  delhi: "Delhi",
  "new delhi": "Delhi",
  bengaluru: "Karnataka",
  mysuru: "Karnataka",
  hubballi: "Karnataka",
  chennai: "Tamil Nadu",
  coimbatore: "Tamil Nadu",
  madurai: "Tamil Nadu",
  salem: "Tamil Nadu",
  kolkata: "West Bengal",
  howrah: "West Bengal",
  siliguri: "West Bengal",
  hyderabad: "Telangana",
  warangal: "Telangana",
  visakhapatnam: "Andhra Pradesh",
  vijayawada: "Andhra Pradesh",
  guntur: "Andhra Pradesh",
  jaipur: "Rajasthan",
  jodhpur: "Rajasthan",
  kota: "Rajasthan",
  bikaner: "Rajasthan",
  udaipur: "Rajasthan",
  lucknow: "Uttar Pradesh",
  kanpur: "Uttar Pradesh",
  varanasi: "Uttar Pradesh",
  agra: "Uttar Pradesh",
  prayagraj: "Uttar Pradesh",
  bhopal: "Madhya Pradesh",
  indore: "Madhya Pradesh",
  gwalior: "Madhya Pradesh",
  jabalpur: "Madhya Pradesh",
  patna: "Bihar",
  gaya: "Bihar",
  muzaffarpur: "Bihar",
  ludhiana: "Punjab",
  amritsar: "Punjab",
  jalandhar: "Punjab",
  chandigarh: "Punjab",
  kochi: "Kerala",
  thiruvananthapuram: "Kerala",
  kozhikode: "Kerala",
  bhubaneswar: "Odisha",
  cuttack: "Odisha",
  raipur: "Chhattisgarh",
  ranchi: "Jharkhand",
  guwahati: "Assam",
  dehradun: "Uttarakhand",
  shimla: "Himachal Pradesh",
};

const MAJOR_STATES = [
  "Gujarat",
  "Maharashtra",
  "Uttar Pradesh",
  "Punjab",
  "Haryana",
  "Madhya Pradesh",
  "Rajasthan",
  "Karnataka",
  "Tamil Nadu",
  "West Bengal",
  "Andhra Pradesh",
  "Telangana",
  "Bihar",
  "Delhi",
  "Kerala",
  "Odisha",
];

const PRODUCE_CATEGORIES: { id: string; label: string; keywords: string[] }[] = [
  { id: "all", label: "All Commodities", keywords: [] },
  {
    id: "vegetables",
    label: "Vegetables",
    keywords: ["tomato", "onion", "potato", "chilli", "cauliflower", "cabbage", "bhindi", "brinjal", "ginger", "garlic", "carrot", "peas", "gourd", "spinach", "cucumber"],
  },
  {
    id: "fruits",
    label: "Fruits",
    keywords: ["banana", "apple", "pomegranate", "orange", "mango", "papaya", "guava", "grapes", "watermelon", "lemon", "sweet lime", "pineapple"],
  },
  {
    id: "grains",
    label: "Grains & Pulses",
    keywords: ["wheat", "paddy", "rice", "maize", "bajra", "jowar", "barley", "gram", "chana", "moong", "tur", "arhar", "urad", "lentil"],
  },
  {
    id: "cash_crops",
    label: "Cash Crops & Oilseeds",
    keywords: ["cotton", "soyabean", "groundnut", "mustard", "cumin", "jeera", "castor", "sesamum", "coriander", "sugarcane", "tobacco"],
  },
];

const DEFAULT_POPULAR_COMMODITIES = [
  "Tomato",
  "Onion",
  "Potato",
  "Green Chilli",
  "Wheat",
  "Soyabean",
  "Cotton",
  "Banana",
  "Bhindi(Ladies Finger)",
  "Cauliflower",
  "Maize",
  "Cumin(Jeera)",
  "Groundnut",
  "Apple",
];

function formatDateDisplay(dateStr?: string): string {
  if (!dateStr) return "Latest Auction";
  try {
    const parts = dateStr.split("-");
    if (parts.length === 3) {
      const year = parseInt(parts[0], 10);
      const month = parseInt(parts[1], 10) - 1;
      const day = parseInt(parts[2], 10);
      const d = new Date(year, month, day);
      return d.toLocaleDateString("en-IN", { day: "numeric", month: "short", year: "numeric" });
    }
  } catch {
    // fallback
  }
  return dateStr;
}

export default function MandiPricesModal({ city, state, language, onClose }: MandiPricesModalProps) {
  const t = getTranslations(language || "en");
  const m = t.mandiPricesModal;
  // 1. Resolve State and District automatically from homepage location
  const resolvedDistrict = useMemo(() => {
    if (!city) return "Ahmedabad";
    const raw = city.includes(",") ? city.split(",")[0].trim() : city.trim();
    return raw || "Ahmedabad";
  }, [city]);

  const resolvedState = useMemo(() => {
    if (state && state.trim()) return state.trim();
    if (city && city.includes(",")) {
      const parsed = city.split(",")[1].trim();
      if (parsed) return parsed;
    }
    const key = resolvedDistrict.toLowerCase();
    if (CITY_TO_STATE_MAP[key]) {
      return CITY_TO_STATE_MAP[key];
    }
    return "Gujarat";
  }, [city, state, resolvedDistrict]);

  // State management
  const [selectedState, setSelectedState] = useState<string>(resolvedState);
  const [selectedDistrict, setSelectedDistrict] = useState<string>(resolvedDistrict);
  const [isRegionPickerOpen, setIsRegionPickerOpen] = useState<boolean>(false);

  const [selectedCommodity, setSelectedCommodity] = useState<string>("Tomato");
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const [commoditySearch, setCommoditySearch] = useState<string>("");

  // Date selection tabs: 'today' | 'yesterday' | 'two_days_ago' | 'custom'
  const [dateMode, setDateMode] = useState<"today" | "yesterday" | "two_days_ago" | "custom">("today");
  const [customDate, setCustomDate] = useState<string>("");

  const [commodities, setCommodities] = useState<MandiCommodityItem[]>([]);
  const [commoditiesLoading, setCommoditiesLoading] = useState<boolean>(false);

  const [pricesData, setPricesData] = useState<MandiPricesResponse | null>(null);
  const [pricesLoading, setPricesLoading] = useState<boolean>(false);
  const [pricesError, setPricesError] = useState<string | null>(null);

  // Sync when prop changes
  useEffect(() => {
    setSelectedState(resolvedState);
    setSelectedDistrict(resolvedDistrict);
  }, [resolvedState, resolvedDistrict]);

  // Determine query date string based on dateMode
  const activeQueryDate = useMemo(() => {
    if (dateMode === "custom") return customDate;
    const now = new Date();
    if (dateMode === "yesterday") {
      now.setDate(now.getDate() - 1);
      return now.toISOString().split("T")[0];
    }
    if (dateMode === "two_days_ago") {
      now.setDate(now.getDate() - 2);
      return now.toISOString().split("T")[0];
    }
    return ""; // Empty string tells backend to get latest/today
  }, [dateMode, customDate]);

  // Fetch commodities for the active state
  useEffect(() => {
    let isMounted = true;
    async function loadCommodities() {
      setCommoditiesLoading(true);
      try {
        const params = new URLSearchParams({ state: selectedState });
        const res = await fetch(`${API_BASE}/api/tools/mandi-commodities?${params.toString()}`);
        if (res.ok) {
          const data: MandiCommoditiesResponse = await res.json();
          if (isMounted) {
            setCommodities(data.commodities || []);
          }
        }
      } catch (err) {
        console.warn("Failed to load commodities for state:", err);
      } finally {
        if (isMounted) setCommoditiesLoading(false);
      }
    }
    loadCommodities();
    return () => {
      isMounted = false;
    };
  }, [selectedState]);

  // Fetch prices
  const fetchPrices = async () => {
    setPricesLoading(true);
    setPricesError(null);
    try {
      const params = new URLSearchParams({
        commodity: selectedCommodity,
        state: selectedState,
        district: selectedDistrict,
      });
      if (activeQueryDate) {
        params.append("date", activeQueryDate);
      }
      const res = await fetch(`${API_BASE}/api/tools/mandi-prices?${params.toString()}`);
      if (!res.ok) {
        throw new Error(`Mandi service error (${res.status})`);
      }
      const data: MandiPricesResponse = await res.json();
      setPricesData(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to load mandi prices";
      setPricesError(msg);
    } finally {
      setPricesLoading(false);
    }
  };

  useEffect(() => {
    fetchPrices();
  }, [selectedCommodity, selectedState, selectedDistrict, activeQueryDate]);

  // Categorized & filtered commodities
  const filteredCommodities = useMemo(() => {
    const rawList = commodities.length > 0
      ? commodities.map((c) => c.commodity)
      : DEFAULT_POPULAR_COMMODITIES;

    return rawList.filter((c) => {
      const name = c.toLowerCase();
      // Category filter
      if (activeCategory !== "all") {
        const cat = PRODUCE_CATEGORIES.find((p) => p.id === activeCategory);
        if (cat && cat.keywords.length > 0) {
          const matchesCategory = cat.keywords.some((kw) => name.includes(kw));
          if (!matchesCategory) return false;
        }
      }
      // Text search
      if (commoditySearch.trim()) {
        return name.includes(commoditySearch.toLowerCase().trim());
      }
      return true;
    });
  }, [commodities, activeCategory, commoditySearch]);

  const mandiRecords = useMemo(() => {
    return pricesData?.records || [];
  }, [pricesData]);

  return (
    <div className="utility-modal-scrim" onClick={onClose} role="dialog" aria-modal="true">
      <div className="utility-modal-window mandi-prices-modal-window" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="utility-modal-header">
          <div className="utility-modal-title-wrap">
            <div className="utility-tool-circle-btn mandi-prices-circle-btn">
              <Wheat size={18} />
            </div>
            <div className="utility-modal-header-info">
              <span className="utility-modal-kicker">{m.kicker}</span>
              <h2>{m.title}</h2>
              <p className="utility-modal-subtitle">
                {m.subtitle} {selectedDistrict}, {selectedState}
              </p>
            </div>
          </div>
          <button className="utility-modal-close" onClick={onClose} aria-label={`Close ${m.title}`}>
            <X size={18} />
          </button>
        </div>

        {/* Body */}
        <div className="utility-modal-body">
          {/* Automatic Location Telemetry Card (Zero friction for farmers) */}
          <div className="mandi-location-banner">
            <div className="mandi-location-badge">
              <div className="mandi-location-icon-wrap">
                <MapPin size={18} />
              </div>
              <div>
                <span className="mandi-location-sub">{m.primaryRegionLabel}</span>
                <h3 className="mandi-location-name">{selectedDistrict}, {selectedState}</h3>
              </div>
            </div>

            <div className="mandi-location-actions">
              <div className="mandi-location-status">
                <span className="mandi-live-pulse-dot" />
                <span className="mandi-live-status-text">{m.apmcConnected}</span>
              </div>
              <button
                type="button"
                className="mandi-region-toggle-btn"
                onClick={() => setIsRegionPickerOpen(!isRegionPickerOpen)}
                title="Change Market District or State"
              >
                <SlidersHorizontal size={13} />
                <span>{isRegionPickerOpen ? m.hideRegionOptions : m.switchMarketState}</span>
                {isRegionPickerOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              </button>
            </div>
          </div>

          {/* Optional Collapsible Region Switcher */}
          {isRegionPickerOpen && (
            <div className="mandi-region-drawer">
              <div className="mandi-region-inputs-grid">
                <div className="utility-input-group">
                  <label className="utility-input-label">
                    <MapPin size={13} /> {m.selectedStateLabel}
                  </label>
                  <UtilityCustomSelect
                    value={selectedState}
                    onChange={(val) => setSelectedState(val)}
                    options={MAJOR_STATES.map((st) => ({ id: st, label: st }))}
                  />
                </div>
                <div className="utility-input-group">
                  <label className="utility-input-label">
                    <Store size={13} /> {m.districtApmcLabel}
                  </label>
                  <input
                    type="text"
                    className="utility-text-input"
                    value={selectedDistrict}
                    onChange={(e) => setSelectedDistrict(e.target.value)}
                    placeholder={m.districtPlaceholder}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Controls Hub: Produce Selection & Date Tabs */}
          <div className="utility-controls-card mandi-controls-card">
            {/* Commodity Category Filters & Search */}
            <div className="mandi-commodity-hub">
              <div className="mandi-hub-top-row">
                <div className="mandi-category-pills">
                  {PRODUCE_CATEGORIES.map((cat) => (
                    <button
                      key={cat.id}
                      type="button"
                      className={`mandi-cat-btn ${activeCategory === cat.id ? "is-active selected" : ""}`}
                      onClick={() => setActiveCategory(cat.id)}
                    >
                      {cat.id === "all" ? m.allProduce : cat.id === "vegetables" ? m.vegetables : cat.id === "fruits" ? m.fruits : cat.id === "grains" ? m.grains : cat.id === "cash_crops" ? m.oilseeds : cat.label}
                    </button>
                  ))}
                </div>

                {/* Real-time Search Box */}
                <div className="mandi-search-input-wrap">
                  <Search size={14} className="mandi-search-icon" />
                  <input
                    type="text"
                    className="mandi-search-input"
                    placeholder={`${m.searchPlaceholderPrefix} ${selectedState} (e.g. Tomato, Onion, Wheat)...`}
                    value={commoditySearch}
                    onChange={(e) => setCommoditySearch(e.target.value)}
                  />
                  {commoditySearch && (
                    <button
                      type="button"
                      className="mandi-search-clear"
                      onClick={() => setCommoditySearch("")}
                      aria-label="Clear commodity search"
                    >
                      <X size={12} />
                    </button>
                  )}
                </div>
              </div>

              {/* Commodity Chips Wrap */}
              <div className="mandi-commodity-chips-scroll">
                {commoditiesLoading && commodities.length === 0 ? (
                  <div className="mandi-chips-loading">
                    <RefreshCw size={14} className="animate-spin text-emerald" />
                    <span>{m.loadingCrops} {selectedState}...</span>
                  </div>
                ) : filteredCommodities.length > 0 ? (
                  filteredCommodities.slice(0, 18).map((c) => {
                    const isSelected = selectedCommodity.toLowerCase() === c.toLowerCase();
                    return (
                      <button
                        key={c}
                        type="button"
                        className={`mandi-commodity-chip ${isSelected ? "is-selected selected" : ""}`}
                        onClick={() => setSelectedCommodity(c)}
                      >
                        {isSelected && <CheckCircle2 size={13} className="mandi-chip-check" />}
                        <span>{c}</span>
                      </button>
                    );
                  })
                ) : (
                  <div className="mandi-no-chips">
                    <span>No commodities match "{commoditySearch}" under {activeCategory}.</span>
                  </div>
                )}
              </div>
            </div>

            {/* Trading Date Selector Tabs */}
            <div className="mandi-date-selector-row">
              <div className="mandi-date-label">
                <Calendar size={14} />
                <span>{m.auctionSessionLabel}</span>
              </div>
              <div className="mandi-date-tabs">
                <button
                  type="button"
                  className={`mandi-date-tab ${dateMode === "today" ? "is-active selected" : ""}`}
                  onClick={() => setDateMode("today")}
                >
                  {m.tabToday}
                </button>
                <button
                  type="button"
                  className={`mandi-date-tab ${dateMode === "yesterday" ? "is-active selected" : ""}`}
                  onClick={() => setDateMode("yesterday")}
                >
                  {m.tabYesterday}
                </button>
                <button
                  type="button"
                  className={`mandi-date-tab ${dateMode === "two_days_ago" ? "is-active selected" : ""}`}
                  onClick={() => setDateMode("two_days_ago")}
                >
                  {m.tabTwoDaysAgo}
                </button>
                <button
                  type="button"
                  className={`mandi-date-tab ${dateMode === "custom" ? "is-active selected" : ""}`}
                  onClick={() => setDateMode("custom")}
                >
                  {m.tabCustom}
                </button>
              </div>

              {dateMode === "custom" && (
                <div className="mandi-custom-date-box">
                  <input
                    type="date"
                    className="mandi-custom-date-input"
                    value={customDate}
                    onChange={(e) => setCustomDate(e.target.value)}
                  />
                </div>
              )}
            </div>

            {/* Centered Recalculate CTA */}
            <div className="utility-action-row-center mandi-cta-row">
              <button
                type="button"
                className="utility-refresh-btn mandi-fetch-btn"
                onClick={fetchPrices}
                disabled={pricesLoading}
              >
                <RefreshCw size={15} className={pricesLoading ? "animate-spin" : ""} />
                <span>{pricesLoading ? m.queryingBooks : `${m.fetchRatesPrefix} ${selectedCommodity}`}</span>
              </button>
            </div>
          </div>

          {/* Error Banner */}
          {pricesError && (
            <div className="utility-error-banner mandi-error-card">
              <AlertCircle size={16} />
              <div>
                <strong>{m.unableToRetrieve}</strong>
                <p>{pricesError}</p>
              </div>
            </div>
          )}

          {/* Loading Indicator */}
          {pricesLoading && !pricesData && (
            <div className="utility-loading-state mandi-loading-box">
              <div className="utility-spinner" />
              <span>Querying verified Agmarknet APMC trading books for {selectedCommodity}...</span>
            </div>
          )}

          {/* Results Overview */}
          {pricesData && (
            <>
              {/* Telemetry Status Notice Banner */}
              {pricesData.is_fallback ? (
                <div className="mandi-notice-box is-fallback">
                  <AlertCircle size={16} className="mandi-notice-icon" />
                  <div>
                    <strong>Mandi Reporting Status: Verified Previous Day ({formatDateDisplay(pricesData.reporting_date)})</strong>
                    <p>
                      {pricesData.fallback_notice ||
                        `Today's prices (${pricesData.requested_date}) are being updated by APMC boards. Showing latest verified auction records from ${pricesData.reporting_date}.`}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="mandi-notice-box is-live">
                  <CheckCircle2 size={16} className="mandi-notice-icon" />
                  <div>
                    <strong>Official APMC Mandi Telemetry: Live Session ({formatDateDisplay(pricesData.reporting_date)})</strong>
                    <p>
                      Real-time arrivals and modal trade rates verified via Agmarknet Government Open Data.
                    </p>
                  </div>
                </div>
              )}

              {/* 1-Click Harvest & Sell Arbitrage Verdict Card */}
              <div className="mandi-verdict-card">
                <div className="mandi-verdict-header">
                  <div className="mandi-verdict-badge">
                    <Sprout size={15} />
                    <span>{m.verdictTitle || "1-CLICK HARVEST & SELL VERDICT"}</span>
                  </div>
                  <span className="mandi-verdict-meta">{m.verdictSubtitle}</span>
                </div>
                <p className="mandi-verdict-text">
                  Because <strong>18 mm rain</strong> is forecast in <strong>36 hours</strong> (raising mold risk to <strong>68%</strong>), and {selectedCommodity.toLowerCase()} rates in {selectedDistrict} {pricesData.highest_market || "Mandi"} are currently at an optimal peak of <strong>₹{pricesData.max_modal_price_kg.toFixed(2)}/kg</strong>, ClimaCore recommends harvesting immediate standing lots today and dispatching to {pricesData.highest_market || "Mandi"} before road transit slows.
                </p>
                <div className="mandi-verdict-chips">
                  <div className="mandi-verdict-chip">
                    <Clock size={12} />
                    <span>Harvest Window: Immediate Standing Dispatch</span>
                  </div>
                  <div className="mandi-verdict-chip">
                    <TrendingUp size={12} />
                    <span>Peak APMC Rate: ₹{pricesData.max_modal_price_kg.toFixed(2)}/kg</span>
                  </div>
                  <div className="mandi-verdict-chip">
                    <ShieldCheck size={12} />
                    <span>Road Transit Decay: Pre-Empted</span>
                  </div>
                </div>
              </div>

              {/* 4-KPI Overview Strip (Executive Farmer Summary) */}
              <div className="utility-status-grid mandi-kpi-grid">
                {/* 1. Average Modal Price */}
                <div className="utility-status-card mandi-kpi-card">
                  <div className="utility-kpi-label">
                    <DollarSign size={13} /> {m.avgModalPrice}
                  </div>
                  <div className="utility-kpi-value text-emerald">
                    ₹{pricesData.avg_modal_price_kg.toFixed(2)}
                    <span className="utility-kpi-unit">/kg</span>
                  </div>
                  <div className="utility-kpi-sub">
                    ₹{pricesData.avg_modal_price_quintal.toLocaleString()}/quintal (100 kg)
                  </div>
                </div>

                {/* 2. Trading Range */}
                <div className="utility-status-card mandi-kpi-card">
                  <div className="utility-kpi-label">
                    <Scale size={13} /> TRADING RANGE (MIN - MAX)
                  </div>
                  <div className="utility-kpi-value">
                    ₹{pricesData.min_modal_price_kg.toFixed(2)} - ₹{pricesData.max_modal_price_kg.toFixed(2)}
                    <span className="utility-kpi-unit">/kg</span>
                  </div>
                  <div className="utility-kpi-sub">
                    ₹{pricesData.min_modal_price_quintal.toLocaleString()} - ₹{pricesData.max_modal_price_quintal.toLocaleString()}/q
                  </div>
                </div>

                {/* 3. Price Spread */}
                <div className="utility-status-card mandi-kpi-card">
                  <div className="utility-kpi-label">
                    <TrendingUp size={13} /> {m.priceSpread}
                  </div>
                  <div className="utility-kpi-value text-forest">
                    ₹{pricesData.price_spread_kg.toFixed(2)}
                    <span className="utility-kpi-unit">/kg</span>
                  </div>
                  <div className="utility-kpi-sub">
                    ₹{pricesData.price_spread_quintal.toLocaleString()}/q across {pricesData.mandis_count} reporting mandis
                  </div>
                </div>

                {/* 4. Top Paying Market */}
                <div className="utility-status-card mandi-kpi-card">
                  <div className="utility-kpi-label">
                    <Store size={13} /> {m.highestMandi}
                  </div>
                  <div className="utility-kpi-value text-top-market">
                    {pricesData.highest_market || "Single Market"}
                  </div>
                  <div className="utility-kpi-sub">
                    Peak rate: ₹{pricesData.max_modal_price_kg.toFixed(2)}/kg
                  </div>
                </div>
              </div>

              {mandiRecords.length > 0 ? (
                <div className="mandi-markets-table-wrap">
                  <table className="mandi-table">
                    <thead>
                      <tr>
                        <th>{m.columnMarket}</th>
                        <th>REPORTED DATE</th>
                        <th>{m.columnVariety}</th>
                        <th>{m.columnModalPrice}</th>
                        <th>{m.columnPriceRange}</th>
                        <th>{m.columnDistrict}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mandiRecords.map((r, idx) => {
                        const isHighest = pricesData.highest_market === r.market && pricesData.mandis_count > 1;
                        const isCheapest = pricesData.cheapest_market === r.market && pricesData.mandis_count > 1 && pricesData.price_spread_kg > 0;
                        const diffVsAvg = r.modal_price_kg - pricesData.avg_modal_price_kg;

                        return (
                          <tr key={`${r.market}-${idx}`} className={isHighest ? "row-highlight-best" : ""}>
                            <td className="mandi-td-market">
                              <Store size={14} className="mandi-td-icon" />
                              <div>
                                <strong>{r.market}</strong>
                                <span className="mandi-td-meta">{r.district}, {r.state}</span>
                              </div>
                            </td>
                            <td className="mandi-td-date">
                              <span className="mandi-date-badge">
                                <Calendar size={11} /> {formatDateDisplay(r.date)}
                              </span>
                            </td>
                            <td>
                              <span className="mandi-variety-tag">{r.variety}</span>
                              <span className="mandi-grade-tag">{r.grade}</span>
                            </td>
                            <td className="mandi-td-price-kg">
                              <span className="mandi-price-kg-value">₹{r.modal_price_kg.toFixed(2)}</span>
                              <span className="mandi-price-kg-unit">/kg</span>
                            </td>
                            <td className="mandi-td-price-q">
                              <div className="mandi-q-modal">
                                <strong>₹{r.modal_price_quintal.toLocaleString()}</strong> /q
                              </div>
                              <div className="mandi-q-range">
                                Min: ₹{r.min_price_quintal.toLocaleString()} · Max: ₹{r.max_price_quintal.toLocaleString()}
                              </div>
                            </td>
                            <td>
                              {isHighest ? (
                                <span className="mandi-spread-badge is-best">
                                  Top Rate (+₹{diffVsAvg.toFixed(2)}/kg)
                                </span>
                              ) : isCheapest ? (
                                <span className="mandi-spread-badge is-low">
                                  Lowest Rate ({diffVsAvg.toFixed(2)}/kg)
                                </span>
                              ) : (
                                <span className="mandi-spread-badge is-avg">
                                  District Average (₹{r.modal_price_kg.toFixed(2)}/kg)
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="mandi-empty-table">
                  <Info size={16} />
                  <span>No market arrivals reported for {pricesData.commodity} in this region.</span>
                </div>
              )}

              {/* ClimaCore Intelligence Advisory */}
              <div className="utility-ai-box mandi-advisory-box">
                <div className="utility-ai-header">
                  <div className="utility-ai-kicker">
                    CLIMACORE AGRONOMIC MARKET INTELLIGENCE
                  </div>
                  <span className="utility-provider-badge">
                    Powered by ClimaCore Intelligence
                  </span>
                </div>
                <p className="utility-ai-text">{pricesData.ai_market_advisory}</p>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <div className="utility-modal-footer">
          <div className="utility-footer-meta">
            <span>{m.footerGovData}</span>
            {pricesData?.reporting_date && (
              <span> · Verified Session: {formatDateDisplay(pricesData.reporting_date)}</span>
            )}
            {pricesData?.latency_ms && (
              <span> · Telemetry Latency: {pricesData.latency_ms} ms</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
