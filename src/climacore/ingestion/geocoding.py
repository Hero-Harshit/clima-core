"""
City Geolocation & Spatial Coordinate Resolver for Indian Urban Centers.
Provides instantaneous O(1) in-memory resolution for the standard 50 Indian cities (plus historic aliases),
with dynamic API fallback for arbitrary towns and coordinates across the Indian subcontinent.
"""

from __future__ import annotations

import re
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import httpx

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.ingestion.geocoding")

# Common historical and colloquial Indian city name variants
INDIAN_CITY_ALIASES: Dict[str, str] = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "madras": "Chennai",
    "chennai": "Chennai",
    "baroda": "Vadodara",
    "vadodara": "Vadodara",
    "banaras": "Varanasi",
    "benares": "Varanasi",
    "kashi": "Varanasi",
    "varanasi": "Varanasi",
    "allahabad": "Prayagraj",
    "prayagraj": "Prayagraj",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "gurgaon": "Delhi",
    "gurugram": "Delhi",
    "noida": "Delhi",
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "gauhati": "Guwahati",
    "guwahati": "Guwahati",
    "poona": "Pune",
    "pune": "Pune",
    "ahmadabad": "Ahmedabad",
    "ahmedabad": "Ahmedabad",
}

_CITIES_REGISTRY: Optional[Dict[str, Dict[str, Any]]] = None


def _load_standard_cities() -> Dict[str, Dict[str, Any]]:
    """Load standard 50 Indian cities from local repository metadata."""
    global _CITIES_REGISTRY
    if _CITIES_REGISTRY is not None:
        return _CITIES_REGISTRY

    registry: Dict[str, Dict[str, Any]] = {}
    root = get_project_root()
    cities_file = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"

    if cities_file.exists():
        df = pd.read_csv(cities_file)
        for _, row in df.iterrows():
            c_name = str(row["city"]).strip()
            state = str(row["state"]).strip() if pd.notna(row.get("state")) else "India"
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            entry = {
                "city": c_name,
                "state": state,
                "latitude": lat,
                "longitude": lon,
                "is_standard_50": True
            }
            registry[c_name.lower()] = entry
            # Also register canonical name
            registry[c_name] = entry

    _CITIES_REGISTRY = registry
    return registry


def parse_coordinates_string(query: str) -> Optional[Tuple[float, float]]:
    """Check if query is a coordinate pair, e.g. '28.61, 77.20' or '28.61N, 77.20E'."""
    s = str(query).strip().upper()
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*[NnSs]?\s*[,/ ]\s*([+-]?\d+(?:\.\d+)?)\s*[EeWw]?$", s)
    if m:
        try:
            lat = float(m.group(1))
            lon = float(m.group(2))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                return lat, lon
        except ValueError:
            pass
    return None


def resolve_location(query: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """Resolve a city name, alias, or lat/lon coordinates into a standardized Indian location object.

    Returns:
        dict with keys: 'city', 'state', 'latitude', 'longitude', 'is_standard_50'
    """
    clean_q = str(query).strip()
    if not clean_q:
        raise ValueError("Location query cannot be empty.")

    # 1. Check if user provided direct coordinates
    coords = parse_coordinates_string(clean_q)
    if coords is not None:
        lat, lon = coords
        return {
            "city": f"Location ({lat:.3f}, {lon:.3f})",
            "state": "Custom Coordinates",
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "is_standard_50": False
        }

    # 2. Check alias map
    q_lower = clean_q.lower()
    canonical_name = INDIAN_CITY_ALIASES.get(q_lower, clean_q)

    # 3. Check 50 standard cities registry
    reg = _load_standard_cities()
    if canonical_name.lower() in reg:
        return dict(reg[canonical_name.lower()])
    if q_lower in reg:
        return dict(reg[q_lower])

    # 4. Dynamic Geocoding API Fallback for arbitrary Indian towns
    logger.info(f"Query '{clean_q}' not in standard 50 registry. Calling Open-Meteo Geocoding API...")
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={clean_q}&count=15&language=en&format=json"

    try:
        with httpx.Client(timeout=timeout_seconds) as client:
            resp = client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    # Filter and prioritize Indian locations
                    in_results = [
                        r for r in results
                        if str(r.get("country_code", "")).upper() == "IN" or "india" in str(r.get("country", "")).lower()
                    ]
                    best = in_results[0] if in_results else results[0]
                    return {
                        "city": best.get("name", clean_q),
                        "state": best.get("admin1", best.get("country", "India")),
                        "latitude": round(float(best["latitude"]), 4),
                        "longitude": round(float(best["longitude"]), 4),
                        "is_standard_50": False
                    }
    except Exception as e:
        logger.warning(f"Geocoding API failed for '{clean_q}': {e}")

    raise ValueError(f"Could not resolve location for '{clean_q}'. Please provide a valid Indian city name or 'lat, lon' coordinates.")
