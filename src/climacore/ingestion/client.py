"""
Live Data Ingestion Client for Open-Meteo High-Resolution Weather & Air Quality APIs.
Provides resilient, authenticated-optional REST client with exponential backoff, connection pooling,
and graceful timeout management across the Indian subcontinent.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional
import httpx

from climacore.common.logging import get_logger

logger = get_logger("climacore.ingestion.client")

USER_AGENT = "ClimaCore-EnvironmentalIntelligence/0.2.0 (Disaster Risk Platform; contact@climacore.org)"

WEATHER_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_ENDPOINT = "https://air-quality-api.open-meteo.com/v1/air-quality"


class LiveIngestionClient:
    """Resilient HTTP client for fetching real-time 7-day weather and atmospheric chemistry forecasts."""

    def __init__(self, timeout_seconds: float = 12.0, max_retries: int = 3):
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.headers = {"User-Agent": USER_AGENT}

    def _execute_get_with_retry(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP GET with exponential backoff on network errors or rate-limits (HTTP 429)."""
        backoff = 1.0
        last_err: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    resp = client.get(url, params=params)

                    if resp.status_code == 200:
                        return resp.json()
                    elif resp.status_code == 429:
                        logger.warning(f"Rate limited (HTTP 429) on attempt {attempt}/{self.max_retries}. Backing off {backoff:.1f}s...")
                        time.sleep(backoff)
                        backoff *= 2.0
                    elif resp.status_code >= 500:
                        logger.warning(f"Server error (HTTP {resp.status_code}) on attempt {attempt}/{self.max_retries}. Backing off {backoff:.1f}s...")
                        time.sleep(backoff)
                        backoff *= 1.5
                    else:
                        resp.raise_for_status()

            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                last_err = e
                logger.warning(f"Network request to {url} failed on attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= 2.0

        raise ConnectionError(f"Failed to fetch data from {url} after {self.max_retries} attempts: {last_err}")

    def fetch_weather_forecast(
        self, latitude: float, longitude: float, days: int = 7, timezone: str = "Asia/Kolkata"
    ) -> Dict[str, Any]:
        """Fetch 7-day numerical weather forecast including surface fluxes, wind gusts, and hydrology."""
        params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "apparent_temperature_max,apparent_temperature_min,"
                "precipitation_sum,rain_sum,precipitation_probability_max,"
                "wind_speed_10m_max,wind_gusts_10m_max,"
                "shortwave_radiation_sum,et0_fao_evapotranspiration"
            ),
            "hourly": (
                "temperature_2m,relative_humidity_2m,surface_pressure,"
                "wind_speed_10m,soil_moisture_0_to_7cm,soil_moisture_7_to_28cm"
            ),
            "forecast_days": min(14, max(1, days)),
            "timezone": timezone,
        }
        return self._execute_get_with_retry(WEATHER_ENDPOINT, params)

    def fetch_air_quality_forecast(
        self, latitude: float, longitude: float, days: int = 7, timezone: str = "Asia/Kolkata"
    ) -> Dict[str, Any]:
        """Fetch 7-day atmospheric chemistry forecast (particulate matter and trace gas pollutants)."""
        params = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone,dust",
            "forecast_days": min(14, max(1, days)),
            "timezone": timezone,
        }
        return self._execute_get_with_retry(AIR_QUALITY_ENDPOINT, params)

    def fetch_city_bundle(
        self, latitude: float, longitude: float, days: int = 7, timezone: str = "Asia/Kolkata"
    ) -> Dict[str, Any]:
        """Fetch coordinated weather and air quality payloads for a location."""
        t0 = time.time()
        logger.info(f"Fetching coordinated weather & AQ live bundle for ({latitude:.3f}, {longitude:.3f}) [{days} days]...")

        weather_data = self.fetch_weather_forecast(latitude, longitude, days=days, timezone=timezone)
        aq_data = self.fetch_air_quality_forecast(latitude, longitude, days=days, timezone=timezone)

        logger.info(f"Successfully retrieved live environmental bundle in {time.time()-t0:.2f}s.")
        return {
            "weather": weather_data,
            "air_quality": aq_data,
        }
