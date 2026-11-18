"""
Master Orchestration Engine for Real-Time Location Forecasts & Persona Risk Dispatches.
Binds live external data ingestion, physical model feature extraction, compound hazard scoring,
and 13 domain-specific persona advisory evaluations across a 7-day forecast timeline.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root
from climacore.ingestion.geocoding import resolve_location
from climacore.ingestion.client import LiveIngestionClient
from climacore.ingestion.normalizer import normalize_live_bundle
from climacore.ingestion.cache import LiveForecastCache
from climacore.ingestion.models import (
    LiveCityForecast,
    LiveDayForecast,
    LiveDayPrediction,
    LivePredictionResult,
)
from climacore.models.compound_risk.scoring import CompoundRiskEngine
from climacore.personas.models import PersonaInput
from climacore.personas.engine import evaluate_persona, evaluate_all_personas

logger = get_logger("climacore.ingestion.engine")

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class LiveForecastEngine:
    """Master orchestrator for on-demand city predictions and 7-day risk trajectories."""

    def __init__(self, cache_ttl_seconds: int = 7200):
        self.client = LiveIngestionClient()
        self.cache = LiveForecastCache(default_ttl_seconds=cache_ttl_seconds)
        self.compound_engine = CompoundRiskEngine()
        self._cyclone_profiles: Optional[Dict[str, Dict[str, Any]]] = None
        self._hap_triggers: Optional[Dict[str, Dict[str, Any]]] = None
        self._population_profiles: Optional[Dict[str, Dict[str, Any]]] = None
        self._infrastructure_profiles: Optional[Dict[str, Dict[str, Any]]] = None
        self._crop_profiles: Optional[Dict[str, Dict[str, Any]]] = None
        self._crop_calendars: Optional[Dict[str, Dict[str, Any]]] = None
        self._power_plant_profiles: Optional[Dict[str, Dict[str, Any]]] = None

    def _load_auxiliary_metadata(self) -> None:
        """Load static city cyclone risk profiles, NDMA HAP thresholds, WorldPop population, and OSM infrastructure."""
        if self._cyclone_profiles is not None:
            return

        root = get_project_root()
        cyc_file = root / "data" / "processed" / "cyclones" / "city_cyclone_risk_profiles.json"
        if cyc_file.exists():
            try:
                with open(cyc_file, "r", encoding="utf-8") as f:
                    cyc_list = json.load(f)
                    self._cyclone_profiles = {c["city"].lower(): c for c in cyc_list}
            except Exception as e:
                logger.warning(f"Could not load cyclone profiles: {e}")
        self._cyclone_profiles = self._cyclone_profiles or {}

        hap_file = root / "data" / "processed" / "ndma_hap_triggers.json"
        if hap_file.exists():
            try:
                with open(hap_file, "r", encoding="utf-8") as f:
                    hap_list = json.load(f)
                    self._hap_triggers = {c["city"].lower(): c for c in hap_list}
            except Exception as e:
                logger.warning(f"Could not load HAP triggers: {e}")
        self._hap_triggers = self._hap_triggers or {}

        pop_file = root / "data" / "processed" / "population" / "city_population_exposure.json"
        if pop_file.exists():
            try:
                with open(pop_file, "r", encoding="utf-8") as f:
                    pop_list = json.load(f)
                    self._population_profiles = {c["city"].lower(): c for c in pop_list}
            except Exception as e:
                logger.warning(f"Could not load population profiles: {e}")
        self._population_profiles = self._population_profiles or {}

        infra_file = root / "data" / "processed" / "infrastructure" / "city_infrastructure_exposure.json"
        if infra_file.exists():
            try:
                with open(infra_file, "r", encoding="utf-8") as f:
                    infra_list = json.load(f)
                    self._infrastructure_profiles = {c["city"].lower(): c for c in infra_list}
            except Exception as e:
                logger.warning(f"Could not load infrastructure profiles: {e}")
        self._infrastructure_profiles = self._infrastructure_profiles or {}

        crop_file = root / "data" / "processed" / "agriculture" / "city_district_crop_profiles.json"
        if crop_file.exists():
            try:
                with open(crop_file, "r", encoding="utf-8") as f:
                    crop_list = json.load(f)
                    self._crop_profiles = {c["city"].lower(): c for c in crop_list}
            except Exception as e:
                logger.warning(f"Could not load crop profiles: {e}")
        self._crop_profiles = self._crop_profiles or {}

        cal_file = root / "data" / "processed" / "agriculture" / "city_crop_calendars.json"
        if cal_file.exists():
            try:
                with open(cal_file, "r", encoding="utf-8") as f:
                    cal_list = json.load(f)
                    self._crop_calendars = {c["city"].lower(): c for c in cal_list}
            except Exception as e:
                logger.warning(f"Could not load crop calendar profiles: {e}")
        self._crop_calendars = self._crop_calendars or {}

        pp_file = root / "data" / "processed" / "power_plants" / "city_power_plant_exposure.json"
        if pp_file.exists():
            try:
                with open(pp_file, "r", encoding="utf-8") as f:
                    pp_list = json.load(f)
                    self._power_plant_profiles = {c["city"].lower(): c for c in pp_list}
            except Exception as e:
                logger.warning(f"Could not load power plant profiles: {e}")
        self._power_plant_profiles = self._power_plant_profiles or {}

    def get_live_forecast(
        self,
        query: str,
        days: int = 7,
        force_refresh: bool = False
    ) -> LiveCityForecast:
        """Fetch and normalize live multi-day forecast for a city, using cache when valid."""
        loc = resolve_location(query)
        city_name = loc["city"]

        # 1. Check local cache
        if not force_refresh:
            cached_forecast = self.cache.get(city_name)
            if cached_forecast is not None:
                return cached_forecast

        # 2. Fetch live from API
        raw_bundle = self.client.fetch_city_bundle(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            days=days
        )

        # 3. Normalize into validated structure
        forecast = normalize_live_bundle(location_meta=loc, raw_bundle=raw_bundle)

        # 4. Save to cache
        self.cache.set(city_name, forecast)
        return forecast

    def predict(
        self,
        query: str,
        persona: str = "all",
        days: int = 7,
        force_refresh: bool = False
    ) -> LivePredictionResult:
        """Generate comprehensive live 7-day risk trajectory and tailored persona intelligence."""
        self._load_auxiliary_metadata()

        # Retrieve or fetch normalized live forecast
        forecast = self.get_live_forecast(query=query, days=days, force_refresh=force_refresh)
        city_lower = forecast.city.lower()

        # Auxiliary risk context for this city
        cyc_info = self._cyclone_profiles.get(city_lower, {}) if self._cyclone_profiles else {}
        hap_info = self._hap_triggers.get(city_lower, {}) if self._hap_triggers else {}
        pop_info = self._population_profiles.get(city_lower, {}) if self._population_profiles else {}
        infra_info = self._infrastructure_profiles.get(city_lower, {}) if self._infrastructure_profiles else {}
        crop_info = self._crop_profiles.get(city_lower, {}) if self._crop_profiles else {}
        cal_info = self._crop_calendars.get(city_lower, {}) if self._crop_calendars else {}
        pp_info = self._power_plant_profiles.get(city_lower, {}) if self._power_plant_profiles else {}

        # CHETNA Power Plant Exposure
        t_plants = pp_info.get("plants_within_100km")
        t_cap_mw = pp_info.get("thermal_capacity_100km_mw")
        closest_plant = None
        if pp_info.get("closest_plant_name"):
            closest_plant = f"{pp_info['closest_plant_name']} ({pp_info.get('closest_plant_capacity_mw', 0):,.0f} MW, {pp_info.get('closest_plant_distance_km', 0):.1f}km {pp_info.get('closest_plant_direction', '')})"

        daily_predictions: List[LiveDayPrediction] = []
        peak_score = 0.0
        peak_date = forecast.days[0].date if forecast.days else ""
        peak_category = "Normal"

        for day in forecast.days[:days]:
            dt = datetime.strptime(day.date, "%Y-%m-%d")
            day_name = DAY_NAMES[dt.weekday()]
            doy = dt.timetuple().tm_yday

            # 1. Compound Risk Calculation
            compound_res = self.compound_engine.calculate_single(
                pm2_5_pred=day.pm2_5,
                temp_max_pred=day.temp_max,
                precip_pred=day.precipitation_mm,
                city=forecast.city,
                date=day.date,
            )

            score = compound_res["compound_risk_score"]
            if score > peak_score:
                peak_score = score
                peak_date = day.date
                peak_category = compound_res["severity_category"]

            # 2. Build Persona Input Payload
            # Calibrated IFI Flood Disaster Probability
            rain_val = day.precipitation_mm
            smdi_val = day.soil_moisture_deficit_index if day.soil_moisture_deficit_index is not None else 0.0
            drain_canals = infra_info.get("drainage_canals_15km", 50) or 50
            base_p = min(1.0, (rain_val / 80.0) * 0.60 + max(0.0, smdi_val * 0.15))
            if drain_canals < 20 and rain_val > 30.0:
                base_p = min(1.0, base_p + 0.15)
            flood_disaster_risk = round(float(base_p), 3)

            # SAGE Crop Growth Stage calculation
            active_crop_stage = "Vegetative / Active Growing Season"
            if cal_info:
                ks = cal_info.get("kharif_sowing_doy_range")
                ka = cal_info.get("kharif_anthesis_doy_range")
                kh = cal_info.get("kharif_harvest_doy_range")
                rs = cal_info.get("rabi_sowing_doy_range")
                ra = cal_info.get("rabi_anthesis_doy_range")
                rh = cal_info.get("rabi_harvest_doy_range")
                k_crop = cal_info.get("kharif_crop", "Kharif Crop")
                r_crop = cal_info.get("rabi_crop", "Rabi Crop")

                if ks and ks[0] <= doy <= ks[1]:
                    active_crop_stage = f"Kharif {k_crop}: Sowing Window"
                elif ka and ka[0] <= doy <= ka[1]:
                    active_crop_stage = f"Kharif {k_crop}: Anthesis / Flowering"
                elif kh and kh[0] <= doy <= kh[1]:
                    active_crop_stage = f"Kharif {k_crop}: Harvesting Window"
                elif rs and rs[0] <= doy <= rs[1]:
                    active_crop_stage = f"Rabi {r_crop}: Sowing Window"
                elif ra and ra[0] <= doy <= ra[1]:
                    active_crop_stage = f"Rabi {r_crop}: Anthesis / Flowering"
                elif rh and rh[0] <= doy <= rh[1]:
                    active_crop_stage = f"Rabi {r_crop}: Harvesting Window"

            # Heat-to-Electricity Grid Stress calculation
            t_mean = (day.temp_max + day.temp_min) / 2.0
            cdd = max(0.0, t_mean - 18.0)
            is_trop_night = day.temp_min >= 28.0
            grid_stress_ratio = round(1.0 + max(0.0, (day.temp_max - 30.0) * 0.02) + (0.12 if is_trop_night else 0.0), 3)
            base_pop = pop_info.get("population_15km", 3_500_000) or 3_500_000
            peak_grid_load = round(float(np.clip(base_pop * 0.0018 * grid_stress_ratio, 1500.0, 28000.0)), 1)
            raw_deficit_prob = max(0.03, (grid_stress_ratio - 1.0) * 1.6 + (0.22 if is_trop_night else 0.0) + (0.25 if day.temp_max >= 42.0 else 0.0))
            grid_deficit_prob = round(float(min(0.96, raw_deficit_prob)), 3)

            # Elevation-aware psychrometric wet-bulb
            from climacore.analytics.psychrometrics import calculate_elevation_aware_wet_bulb
            elev_wb = calculate_elevation_aware_wet_bulb(day.temp_max, day.relative_humidity, city=forecast.city)
            effective_wet_bulb = day.wet_bulb_temp_c if day.wet_bulb_temp_c is not None else elev_wb["wet_bulb_temp_c"]

            p_input = PersonaInput(
                city=forecast.city,
                date=day.date,
                temp_max=day.temp_max,
                temp_min=day.temp_min,
                apparent_temp_max=day.apparent_temp_max,
                precipitation_mm=day.precipitation_mm,
                rain_probability=day.rain_probability,
                pm2_5=day.pm2_5,
                pm10=day.pm10,
                no2=day.no2,
                wind_speed_kmh=day.wind_speed_kmh,
                relative_humidity=day.relative_humidity,
                solar_radiation=day.solar_radiation,
                et0_evapotranspiration=day.et0_evapotranspiration,
                imd_heatwave_alert=day.imd_heatwave_alert,
                atmospheric_stagnation=day.atmospheric_stagnation,
                historical_percentile=day.historical_percentile,
                soil_moisture_deficit_index=day.soil_moisture_deficit_index,
                flash_drought_alert=day.flash_drought_alert,
                gale_wind_gust_kmh=day.gale_wind_gust_kmh,
                cyclone_distance_km=cyc_info.get("min_historic_distance_km"),
                cyclone_grade=cyc_info.get("closest_storm"),
                wet_bulb_temp_c=effective_wet_bulb,
                population_5km=pop_info.get("population_5km"),
                population_15km=pop_info.get("population_15km"),
                population_30km=pop_info.get("population_30km"),
                dense_cluster_pop_15km=pop_info.get("dense_cluster_pop_15km"),
                urban_exposure_tier=pop_info.get("urban_exposure_tier"),
                hospitals_15km=infra_info.get("hospitals_15km"),
                clinics_15km=infra_info.get("clinics_15km"),
                total_healthcare_15km=infra_info.get("total_healthcare_15km"),
                schools_15km=infra_info.get("schools_15km"),
                bridges_15km=infra_info.get("bridges_15km"),
                drainage_canals_15km=infra_info.get("drainage_canals_15km"),
                arterial_roads_15km=infra_info.get("arterial_roads_15km"),
                industrial_zones_15km=infra_info.get("industrial_zones_15km"),
                transit_stations_15km=infra_info.get("transit_stations_15km"),
                people_per_hospital_15km=infra_info.get("people_per_hospital_15km"),
                people_per_school_15km=infra_info.get("people_per_school_15km"),
                critical_infrastructure_density_km2=infra_info.get("critical_infrastructure_density_km2"),
                drainage_arterial_ratio=infra_info.get("drainage_arterial_ratio"),
                flood_disaster_risk_prob=flood_disaster_risk,
                antecedent_rain_3d_mm=round(rain_val * 1.5, 1),
                agricultural_district=crop_info.get("agricultural_district"),
                dominant_kharif_crop=crop_info.get("dominant_kharif_crop"),
                dominant_rabi_crop=crop_info.get("dominant_rabi_crop"),
                dominant_crop_overall=crop_info.get("dominant_crop_overall"),
                soil_ph=crop_info.get("soil_ph"),
                soil_organic_carbon_g_kg=crop_info.get("soil_organic_carbon_g_kg"),
                soil_fertility_index=crop_info.get("soil_fertility_index"),
                soil_water_holding_capacity=crop_info.get("soil_water_holding_capacity"),
                electricity_grid_stress_ratio=grid_stress_ratio,
                peak_grid_load_mw=peak_grid_load,
                grid_blackout_hazard_prob=grid_deficit_prob,
                crop_growth_stage=active_crop_stage,
                crop_sowing_dates=cal_info.get("kharif_sowing_dates") or cal_info.get("rabi_sowing_dates"),
                crop_anthesis_dates=cal_info.get("kharif_anthesis_dates") or cal_info.get("rabi_anthesis_dates"),
                crop_harvest_dates=cal_info.get("kharif_harvest_dates") or cal_info.get("rabi_harvest_dates"),
                thermal_plants_within_100km=t_plants,
                thermal_capacity_100km_mw=t_cap_mw,
                closest_thermal_plant=closest_plant,
            )

            # 3. Evaluate Selected Persona or All 13 Personas
            if str(persona).strip().lower() in ["all", "none", "*", ""]:
                p_results = evaluate_all_personas(p_input)
            else:
                single_res = evaluate_persona(persona, p_input)
                p_results = {single_res.persona_id: single_res.model_dump()}

            daily_predictions.append(
                LiveDayPrediction(
                    date=day.date,
                    day_offset=day.day_offset,
                    day_name=day_name,
                    forecast_metrics={
                        "temp_max": day.temp_max,
                        "temp_min": day.temp_min,
                        "apparent_temp_max": day.apparent_temp_max,
                        "precipitation_mm": day.precipitation_mm,
                        "rain_probability": round(day.rain_probability * 100.0, 1),
                        "wind_speed_kmh": day.wind_speed_kmh,
                        "gale_wind_gust_kmh": day.gale_wind_gust_kmh,
                        "relative_humidity": day.relative_humidity,
                        "pm2_5": day.pm2_5,
                        "pm10": day.pm10,
                        "wet_bulb_temp_c": day.wet_bulb_temp_c,
                    },
                    compound_risk=compound_res,
                    personas=p_results,
                )
            )

        return LivePredictionResult(
            city=forecast.city,
            state=forecast.state,
            latitude=forecast.latitude,
            longitude=forecast.longitude,
            retrieved_at_utc=forecast.retrieved_at_utc,
            is_cached=forecast.is_cached,
            selected_persona=persona,
            days_count=len(daily_predictions),
            peak_compound_risk=peak_score,
            peak_risk_date=peak_date,
            peak_risk_category=peak_category,
            daily_predictions=daily_predictions,
        )
