import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Search } from "lucide-react";

type CitySuggestion = { city: string; state: string };

function CitySearch({
  value,
  onSelect,
  placeholder = "Search city or state",
}: {
  value: string;
  onSelect: (value: string) => void;
  placeholder?: string;
}) {
  const id = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState(value);
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [isFiltering, setIsFiltering] = useState(false);
  const [cities, setCities] = useState<CitySuggestion[] | null>(null);
  const [isLoadingCities, setIsLoadingCities] = useState(false);
  const [cityLoadFailed, setCityLoadFailed] = useState(false);
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const suggestions = useMemo(() => {
    const cityList = cities ?? [];
    const matches = !isFiltering || !normalizedQuery
      ? cityList
      : cityList.filter(({ city, state }) =>
        `${city}, ${state}`.toLocaleLowerCase().includes(normalizedQuery),
      );
    return matches;
  }, [cities, isFiltering, normalizedQuery]);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  useEffect(() => {
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setIsOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, []);

  useEffect(() => {
    if (!isOpen || activeIndex < 0) return;
    menuRef.current
      ?.querySelector<HTMLElement>(`[data-option-index="${activeIndex}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [activeIndex, isOpen]);

  const loadCityList = () => {
    setIsOpen(true);
    if (cities || isLoadingCities || cityLoadFailed) return;
    setIsLoadingCities(true);
    void import("./indian-cities.json")
      .then(({ default: cityList }) => setCities(cityList))
      .catch((error: unknown) => {
        console.error("Unable to load Indian city suggestions.", error);
        setCityLoadFailed(true);
      })
      .finally(() => setIsLoadingCities(false));
  };

  const chooseCity = (city: CitySuggestion) => {
    const location = `${city.city}, ${city.state}`;
    setQuery(location);
    setIsFiltering(false);
    setIsOpen(false);
    onSelect(location);
  };

  const submitQuery = () => {
    const location = query.trim();
    if (!location) return;
    setQuery(location);
    setIsOpen(false);
    onSelect(location);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Escape") {
      setIsOpen(false);
    } else if (event.key === "ArrowDown" && suggestions.length) {
      event.preventDefault();
      setIsOpen(true);
      setActiveIndex((index) => index < 0 ? 0 : (index + 1) % suggestions.length);
    } else if (event.key === "ArrowUp" && suggestions.length) {
      event.preventDefault();
      setIsOpen(true);
      setActiveIndex((index) => index < 0 ? suggestions.length - 1 : (index - 1 + suggestions.length) % suggestions.length);
    } else if (event.key === "Enter") {
      event.preventDefault();
      const activeSuggestion = isOpen ? suggestions[activeIndex] : undefined;
      if (activeSuggestion) chooseCity(activeSuggestion);
      else submitQuery();
    } else if (event.key === "Tab") {
      setIsOpen(false);
    }
  };

  return (
    <div className="city-search" ref={rootRef}>
      <Search size={14} aria-hidden="true" />
      <input
        type="text"
        role="combobox"
        aria-label="View in city"
        aria-autocomplete="list"
        aria-expanded={isOpen}
        aria-controls={`city-suggestions-${id}`}
        aria-activedescendant={isOpen && activeIndex >= 0 ? `city-suggestion-${id}-${activeIndex}` : undefined}
        autoComplete="off"
        placeholder={placeholder}
        value={query}
        onFocus={(event) => {
          setIsFiltering(false);
          event.currentTarget.select();
          loadCityList();
        }}
        onChange={(event) => {
          setQuery(event.currentTarget.value);
          setIsFiltering(true);
          setActiveIndex(-1);
          loadCityList();
        }}
        onKeyDown={handleKeyDown}
      />
      {isOpen && (
        <div className="city-search-menu" id={`city-suggestions-${id}`} role="listbox" ref={menuRef}>
          {isLoadingCities ? (
            <div className="city-search-empty" role="status">Loading city suggestions…</div>
          ) : cityLoadFailed ? (
            <div className="city-search-empty" role="status">Suggestions are unavailable. Press Enter to search a custom town.</div>
          ) : suggestions.length ? suggestions.map((city, index) => (
            <div
              key={`${city.city}-${city.state}`}
              id={`city-suggestion-${id}-${index}`}
              className={`city-search-option ${activeIndex === index ? "active" : ""}`}
              role="option"
              aria-selected={activeIndex === index}
              data-option-index={index}
              onMouseDown={(event) => event.preventDefault()}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => chooseCity(city)}
            >
              <span>{city.city}, <small>{city.state}</small></span>
            </div>
          )) : (
            <div className="city-search-empty" role="status">
              No matches. Press Enter to search “{query.trim()}”.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default CitySearch;
