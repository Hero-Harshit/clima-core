"""
Data Normalization & Atmospheric Feature Extractor for Live Weather and Air Quality Streams.
Aligns heterogeneous hourly and daily Open-Meteo payloads into clean, schema-validated
7-day forecasting structures with derived psychrometric, thermodynamic, and air quality metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np

from climacore.common.logging import get_logger
from climacore.ingestion.models import LiveDayForecast, LiveCityForecast
from climacore.personas.industry import estimate_stull_wet_bulb

logger = get_logger("climacore.ingestion.normalizer")


def _safe_mean(values: List[Any], default: float = 0.0) -> float:
    """Compute mean over valid non-null numeric items."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(clean)) if clean else default


def _safe_max(values: List[Any], default: float = 0.0) -> float:
    """Compute max over valid non-null numeric items."""
    clean = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.max(clean)) if clean else default


def normalize_live_bundle(
    location_meta: Dict[str, Any], raw_bundle: Dict[str, Any]
) -> LiveCityForecast:
    """Transform raw Open-Meteo weather and air quality responses into a validated LiveCityForecast."""
    weather = raw_bundle.get("weather", {})
    aq = raw_bundle.get("air_quality", {})

    daily = weather.get("daily", {})
    hourly_w = weather.get("hourly", {})
    hourly_aq = aq.get("hourly", {})

    dates = daily.get("time", [])
    if not dates:
        raise ValueError("Invalid weather payload: no daily timestamps found.")

    city_name = location_meta.get("city", "Unknown City")
    state_name = location_meta.get("state")
    lat = float(location_meta.get("latitude", weather.get("latitude", 0.0)))
    lon = float(location_meta.get("longitude", weather.get("longitude", 0.0)))

    # Index hourly data by date prefix (YYYY-MM-DD)
    hourly_w_times = hourly_w.get("time", [])
    hourly_aq_times = hourly_aq.get("time", [])

    day_forecasts: List[LiveDayForecast] = []

    for d_idx, d_str in enumerate(dates):
        # 1. Direct Daily Variables
        t_max = float(daily.get("temperature_2m_max", [32.0])[d_idx])
        t_min = float(daily.get("temperature_2m_min", [22.0])[d_idx])
        app_max = float(daily.get("apparent_temperature_max", [t_max + 2.0])[d_idx])
        rain_sum = float(daily.get("precipitation_sum", [0.0])[d_idx])
        rain_prob = float(daily.get("precipitation_probability_max", [10.0])[d_idx]) / 100.0
        wind_max = float(daily.get("wind_speed_10m_max", [12.0])[d_idx])
        gust_max = float(daily.get("wind_gusts_10m_max", [wind_max * 1.35])[d_idx])
        solar_rad = float(daily.get("shortwave_radiation_sum", [18.0])[d_idx])
        et0 = float(daily.get("et0_fao_evapotranspiration", [4.5])[d_idx])

        # 2. Extract Matching Hourly Subsets for this Date
        w_indices = [i for i, t in enumerate(hourly_w_times) if str(t).startswith(d_str)]
        aq_indices = [i for i, t in enumerate(hourly_aq_times) if str(t).startswith(d_str)]

        # Relative Humidity
        rh_vals = [hourly_w.get("relative_humidity_2m", [])[i] for i in w_indices if i < len(hourly_w.get("relative_humidity_2m", []))]
        mean_rh = round(_safe_mean(rh_vals, default=50.0), 1)

        # Soil Moisture
        sm_top_vals = [hourly_w.get("soil_moisture_0_to_7cm", [])[i] for i in w_indices if i < len(hourly_w.get("soil_moisture_0_to_7cm", []))]
        sm_root_vals = [hourly_w.get("soil_moisture_7_to_28cm", [])[i] for i in w_indices if i < len(hourly_w.get("soil_moisture_7_to_28cm", []))]
        topsoil_sm = round(_safe_mean(sm_top_vals, default=0.22), 3) if sm_top_vals else None
        rootzone_sm = round(_safe_mean(sm_root_vals, default=0.24), 3) if sm_root_vals else None

        # Air Quality Pollutants
        pm25_vals = [hourly_aq.get("pm2_5", [])[i] for i in aq_indices if i < len(hourly_aq.get("pm2_5", []))]
        pm10_vals = [hourly_aq.get("pm10", [])[i] for i in aq_indices if i < len(hourly_aq.get("pm10", []))]
        no2_vals = [hourly_aq.get("nitrogen_dioxide", [])[i] for i in aq_indices if i < len(hourly_aq.get("nitrogen_dioxide", []))]
        so2_vals = [hourly_aq.get("sulphur_dioxide", [])[i] for i in aq_indices if i < len(hourly_aq.get("sulphur_dioxide", []))]
        co_vals = [hourly_aq.get("carbon_monoxide", [])[i] for i in aq_indices if i < len(hourly_aq.get("carbon_monoxide", []))]
        o3_vals = [hourly_aq.get("ozone", [])[i] for i in aq_indices if i < len(hourly_aq.get("ozone", []))]

        mean_pm25 = round(_safe_mean(pm25_vals, default=45.0), 1)
        mean_pm10 = round(_safe_mean(pm10_vals, default=mean_pm25 * 1.7), 1)
        mean_no2 = round(_safe_mean(no2_vals, default=22.0), 1)
        mean_so2 = round(_safe_mean(so2_vals, default=8.0), 1)
        mean_co = round(_safe_mean(co_vals, default=450.0), 1)
        mean_o3 = round(_safe_mean(o3_vals, default=35.0), 1)

        # 3. Physical Derived Metrics
        twb = estimate_stull_wet_bulb(t_max, mean_rh)

        # Atmospheric Stagnation (Winter months + low wind)
        dt = datetime.strptime(d_str, "%Y-%m-%d")
        is_winter = dt.month in [11, 12, 1, 2]
        stagnation = 1 if (wind_max <= 9.0 and (is_winter or mean_rh >= 70.0)) else 0

        # IMD Heatwave Alert
        if t_max >= 45.0:
            heatwave_alert = 3  # Red Alert
        elif t_max >= 43.0:
            heatwave_alert = 2  # Orange Alert
        elif t_max >= 40.0:
            heatwave_alert = 1  # Yellow Alert
        else:
            heatwave_alert = 0

        # Soil Moisture Deficit Index (SMDI): Reference baseline ≈ 0.25 m³/m³
        if topsoil_sm is not None:
            smdi = round(float(np.clip((topsoil_sm - 0.25) / 0.08, -3.0, 3.0)), 2)
            flash_drought = 1 if (topsoil_sm <= 0.14 and (et0 - rain_sum) >= 3.5) else 0
        else:
            smdi = None
            flash_drought = 0

        # Historical percentiles estimate (Summer vs Winter seasonality)
        hist_pct = min(99.5, max(10.0, 50.0 + (t_max - 35.0) * 3.5)) if dt.month in [4, 5, 6] else 50.0

        day_forecasts.append(
            LiveDayForecast(
                date=d_str,
                day_offset=d_idx,
                temp_max=t_max,
                temp_min=t_min,
                apparent_temp_max=app_max,
                precipitation_mm=rain_sum,
                rain_probability=rain_prob,
                wind_speed_kmh=wind_max,
                gale_wind_gust_kmh=gust_max,
                relative_humidity=mean_rh,
                solar_radiation=solar_rad,
                et0_evapotranspiration=et0,
                pm2_5=mean_pm25,
                pm10=mean_pm10,
                no2=mean_no2,
                so2=mean_so2,
                co=mean_co,
                ozone=mean_o3,
                soil_moisture_0_to_7cm=topsoil_sm,
                soil_moisture_7_to_28cm=rootzone_sm,
                wet_bulb_temp_c=twb,
                imd_heatwave_alert=heatwave_alert,
                atmospheric_stagnation=stagnation,
                historical_percentile=round(hist_pct, 1),
                soil_moisture_deficit_index=smdi,
                flash_drought_alert=flash_drought,
                cyclone_distance_km=None,
                cyclone_grade=None
            )
        )

    return LiveCityForecast(
        city=city_name,
        state=state_name,
        latitude=lat,
        longitude=lon,
        elevation_m=weather.get("elevation"),
        timezone=weather.get("timezone", "Asia/Kolkata"),
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
        is_cached=False,
        cached_at_utc=None,
        days=day_forecasts
    )
