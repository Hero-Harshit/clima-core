"""
Unit tests for Clima-Core Live Ingestion & 7-Day Prediction Pipeline.
Tests spatial geocoding, schema normalization, psychrometric feature derivations,
local TTL cache lifecycle, and end-to-end live forecast orchestration.
"""

from datetime import datetime, timezone
import pytest
from climacore.ingestion.geocoding import resolve_location, parse_coordinates_string
from climacore.ingestion.models import LiveDayForecast, LiveCityForecast
from climacore.ingestion.normalizer import normalize_live_bundle
from climacore.ingestion.cache import LiveForecastCache
from climacore.ingestion.engine import LiveForecastEngine


def test_geocoding_standard_cities():
    """Verify standard 50 Indian cities and historic aliases resolve correctly."""
    # Canonical cities
    delhi = resolve_location("Delhi")
    assert delhi["city"] == "Delhi"
    assert abs(delhi["latitude"] - 28.6139) < 0.1
    assert delhi["is_standard_50"] is True

    # Historic aliases
    bangalore = resolve_location("Bangalore")
    assert bangalore["city"] == "Bengaluru"
    assert bangalore["is_standard_50"] is True

    bombay = resolve_location("Bombay")
    assert bombay["city"] == "Mumbai"

    calcutta = resolve_location("Calcutta")
    assert calcutta["city"] == "Kolkata"

    madras = resolve_location("Madras")
    assert madras["city"] == "Chennai"


def test_geocoding_coordinate_parsing():
    """Verify coordinate strings are parsed cleanly."""
    coords = parse_coordinates_string("28.6139, 77.2090")
    assert coords is not None
    assert abs(coords[0] - 28.6139) < 1e-4
    assert abs(coords[1] - 77.2090) < 1e-4

    loc = resolve_location("19.076, 72.877")
    assert "Location" in loc["city"]
    assert abs(loc["latitude"] - 19.076) < 1e-3


@pytest.fixture
def sample_raw_bundle():
    """Construct a synthetic raw Open-Meteo bundle covering 3 days."""
    dates = ["2026-11-16", "2026-11-17", "2026-11-18"]
    hourly_times = []
    for d in dates:
        for h in range(24):
            hourly_times.append(f"{d}T{h:02d}:00")

    weather_raw = {
        "latitude": 28.61,
        "longitude": 77.21,
        "timezone": "Asia/Kolkata",
        "elevation": 216.0,
        "daily": {
            "time": dates,
            "temperature_2m_max": [31.0, 32.5, 41.0],
            "temperature_2m_min": [18.0, 19.0, 26.0],
            "apparent_temperature_max": [33.0, 34.5, 44.0],
            "precipitation_sum": [0.0, 12.5, 0.0],
            "precipitation_probability_max": [5.0, 80.0, 10.0],
            "wind_speed_10m_max": [8.0, 24.0, 14.0],
            "wind_gusts_10m_max": [12.0, 45.0, 22.0],
            "shortwave_radiation_sum": [17.5, 12.0, 22.0],
            "et0_fao_evapotranspiration": [4.0, 3.2, 5.8],
        },
        "hourly": {
            "time": hourly_times,
            "temperature_2m": [25.0] * len(hourly_times),
            "relative_humidity_2m": [55.0] * len(hourly_times),
            "soil_moisture_0_to_7cm": [0.22] * len(hourly_times),
            "soil_moisture_7_to_28cm": [0.25] * len(hourly_times),
        }
    }

    aq_raw = {
        "hourly": {
            "time": hourly_times,
            "pm2_5": [115.0] * len(hourly_times),
            "pm10": [210.0] * len(hourly_times),
            "nitrogen_dioxide": [35.0] * len(hourly_times),
            "sulphur_dioxide": [12.0] * len(hourly_times),
            "carbon_monoxide": [650.0] * len(hourly_times),
            "ozone": [45.0] * len(hourly_times),
        }
    }

    return {"weather": weather_raw, "air_quality": aq_raw}


def test_normalizer_schema_and_physics(sample_raw_bundle):
    """Test data normalization, wet-bulb derivation, and physical alerts."""
    loc_meta = {"city": "Delhi", "state": "Delhi", "latitude": 28.61, "longitude": 77.21}
    forecast = normalize_live_bundle(loc_meta, sample_raw_bundle)

    assert isinstance(forecast, LiveCityForecast)
    assert forecast.city == "Delhi"
    assert len(forecast.days) == 3

    day1 = forecast.days[0]
    assert day1.date == "2026-11-16"
    assert day1.temp_max == 31.0
    assert day1.precipitation_mm == 0.0
    assert day1.pm2_5 == 115.0
    assert day1.wet_bulb_temp_c is not None
    assert day1.wet_bulb_temp_c < day1.temp_max  # Thermodynamic law: Twb <= Tdry

    # Day 3 has Tmax=41.0°C (Yellow Heatwave alert)
    day3 = forecast.days[2]
    assert day3.temp_max == 41.0
    assert day3.imd_heatwave_alert >= 1


def test_cache_lifecycle(tmp_path, sample_raw_bundle):
    """Test writing, reading, and invalidating the local forecast cache."""
    cache = LiveForecastCache(cache_dir=tmp_path, default_ttl_seconds=3600)

    loc_meta = {"city": "Pune", "state": "Maharashtra", "latitude": 18.52, "longitude": 73.85}
    forecast = normalize_live_bundle(loc_meta, sample_raw_bundle)

    # Initial get should be miss
    assert cache.get("Pune") is None

    # Write to cache
    cache.set("Pune", forecast)

    # Second get should be hit
    cached = cache.get("Pune")
    assert cached is not None
    assert cached.city == "Pune"
    assert cached.is_cached is True
    assert cached.cached_at_utc is not None

    # Invalidate
    assert cache.invalidate("Pune") is True
    assert cache.get("Pune") is None


def test_live_forecast_engine_simulation(tmp_path, sample_raw_bundle):
    """Test full engine orchestration using mock cache injection."""
    cache = LiveForecastCache(cache_dir=tmp_path)
    loc_meta = {"city": "Nagpur", "state": "Maharashtra", "latitude": 21.1458, "longitude": 79.0882}
    forecast = normalize_live_bundle(loc_meta, sample_raw_bundle)
    cache.set("Nagpur", forecast)

    engine = LiveForecastEngine()
    engine.cache = cache

    # Predict for farmer persona
    res = engine.predict(query="Nagpur", persona="farmer", days=3)

    assert res.city == "Nagpur"
    assert res.is_cached is True
    assert res.selected_persona == "farmer"
    assert len(res.daily_predictions) == 3

    # Check Day 1 metrics
    day1_pred = res.daily_predictions[0]
    assert day1_pred.date == "2026-11-16"
    assert "farmer_kisan" in day1_pred.personas
    farmer_data = day1_pred.personas["farmer_kisan"]
    assert "crop_water_balance" in farmer_data["key_metrics"]
    assert "irrigation_requirement" in farmer_data["key_metrics"]
    assert res.peak_compound_risk > 0.0
