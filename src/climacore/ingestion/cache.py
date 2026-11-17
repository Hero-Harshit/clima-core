"""
Local High-Performance Cache Layer for Real-Time Forecast Streams.
Prevents redundant external API calls, respects Open-Meteo usage limits, and enables
sub-millisecond instant responses for repeated city lookups across configured TTL windows.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import re
from typing import Optional

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root
from climacore.ingestion.models import LiveCityForecast

logger = get_logger("climacore.ingestion.cache")

DEFAULT_CACHE_TTL_SECONDS = 7200  # 2 Hours


def _slugify(name: str) -> str:
    """Create a clean filesystem-safe slug from city name or coordinate string."""
    s = str(name).strip().lower()
    s = re.sub(r"[^\w.-]", "_", s)
    return s


class LiveForecastCache:
    """File-backed local cache for validated 7-day city environmental forecasts."""

    def __init__(self, cache_dir: Optional[Path] = None, default_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS):
        root = get_project_root()
        self.cache_dir = cache_dir or (root / "data" / "cache" / "live_forecasts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl_seconds

    def _get_path(self, city_name: str) -> Path:
        return self.cache_dir / f"{_slugify(city_name)}.json"

    def get(self, city_name: str) -> Optional[LiveCityForecast]:
        """Retrieve cached live forecast if still valid within TTL window."""
        cache_path = self._get_path(city_name)
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                entry = json.load(f)

            expires_at = datetime.fromisoformat(entry["expires_at_utc"])
            now_utc = datetime.now(timezone.utc)

            if now_utc < expires_at:
                logger.info(f"Cache HIT for '{city_name}' (Valid until {expires_at.strftime('%H:%M:%S UTC')}).")
                forecast_data = entry["data"]
                forecast = LiveCityForecast.model_validate(forecast_data)
                forecast.is_cached = True
                forecast.cached_at_utc = entry["cached_at_utc"]
                return forecast
            else:
                logger.debug(f"Cache expired for '{city_name}' (Expired at {expires_at.strftime('%H:%M:%S UTC')}).")
                return None
        except Exception as e:
            logger.warning(f"Failed reading cache for '{city_name}': {e}")
            return None

    def set(
        self, city_name: str, forecast: LiveCityForecast, ttl_seconds: Optional[int] = None
    ) -> None:
        """Store freshly retrieved forecast into local cache with expiration timestamp."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        now_utc = datetime.now(timezone.utc)
        expires_at = now_utc + timedelta(seconds=ttl)

        cache_path = self._get_path(city_name)
        entry = {
            "city": city_name,
            "cached_at_utc": now_utc.isoformat(),
            "expires_at_utc": expires_at.isoformat(),
            "ttl_seconds": ttl,
            "data": forecast.model_dump(),
        }

        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2)
            logger.debug(f"Saved live forecast for '{city_name}' to cache at {cache_path}.")
        except Exception as e:
            logger.warning(f"Failed writing cache for '{city_name}': {e}")

    def invalidate(self, city_name: str) -> bool:
        """Delete cached forecast for a city."""
        cache_path = self._get_path(city_name)
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"Invalidated cache for '{city_name}'.")
                return True
            except Exception as e:
                logger.warning(f"Error deleting cache file {cache_path}: {e}")
        return False

    def clear_all(self) -> int:
        """Clear all cached forecast files in directory."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
        logger.info(f"Cleared {count} cached forecast files.")
        return count
