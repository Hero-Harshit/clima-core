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
    PHYSICAL_UNITS_MANIFEST,
)
from climacore.models.compound_risk.scoring import CompoundRiskEngine
from climacore.models.compound_risk.cascading import CascadingRiskEngine
from climacore.models.flood_risk.engine import FloodRiskEngine
from climacore.models.heat_risk.engine import HeatRiskEngine
from climacore.models.electricity_grid.engine import PowerGridEngine
from climacore.models.crop_resilience.engine import AgriSenseEngine
from climacore.models.air_quality.engine import AirQualityEngine
from climacore.analytics.conformal import (
    calibrate_temperature_conformal,
    calibrate_rainfall_conformal,
    calibrate_power_demand_conformal,
    calibrate_pm25_conformal,
)
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
        self.cascading_engine = CascadingRiskEngine()
        self.flood_engine = FloodRiskEngine()
        self.heat_engine = HeatRiskEngine()
        self.power_engine = PowerGridEngine()
        self.agri_engine = AgriSenseEngine()
        self.aq_engine = AirQualityEngine()
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

            # 2. Flagship Models Execution (Phase 3 Architecture)
            # Flagship 1: FloodRisk V2
            flood_eval = self.flood_engine.evaluate_flood_risk(
                city=forecast.city,
                rainfall_mm=day.precipitation_mm,
                rain_rolling_3d=round(day.precipitation_mm * 1.5, 1),
                smdi=day.soil_moisture_deficit_index,
                drainage_canals_15km=infra_info.get("drainage_canals_15km"),
                arterial_roads_15km=infra_info.get("arterial_roads_15km"),
                bridges_15km=infra_info.get("bridges_15km"),
                population_15km=pop_info.get("population_15km"),
                dense_cluster_pop_15km=pop_info.get("dense_cluster_pop_15km"),
                hospitals_15km=infra_info.get("hospitals_15km"),
                schools_15km=infra_info.get("schools_15km"),
            )
            flood_disaster_risk = flood_eval["hydrological_hazard_probability"]

            # Flagship 2: HeatRisk V3
            heat_eval = self.heat_engine.evaluate_heat_risk(
                city=forecast.city,
                temp_max_c=day.temp_max,
                temp_min_c=day.temp_min,
                relative_humidity=day.relative_humidity,
                dense_cluster_pop=pop_info.get("dense_cluster_pop_15km"),
            )

            # Flagship 3: PowerGrid V2
            power_eval = self.power_engine.evaluate_power_grid_risk(
                temp_max_c=day.temp_max,
                temp_min_c=day.temp_min,
                state=forecast.state,
                population_15km=pop_info.get("population_15km"),
                thermal_plants_within_100km=t_plants,
                thermal_capacity_100km_mw=t_cap_mw,
            )
            grid_stress_ratio = power_eval["cooling_demand_surge_ratio"]
            peak_grid_load = power_eval["peak_load_mw"]
            grid_deficit_prob = power_eval["unmet_supply_stress_probability"]

            # Flagship 4: AgriSense V2
            t_mean = (day.temp_max + day.temp_min) / 2.0
            agri_eval = self.agri_engine.evaluate_agrisense(
                city=forecast.city,
                doy=doy,
                temp_max_c=day.temp_max,
                temp_min_c=day.temp_min,
                temp_mean_c=t_mean,
                rainfall_mm=day.precipitation_mm,
                humidity_pct=day.relative_humidity,
                solar_mj_m2_day=day.solar_radiation,
                et0_evapotranspiration=day.et0_evapotranspiration,
                elevation_m=forecast.latitude,
                smdi=day.soil_moisture_deficit_index,
                flash_drought_alert=day.flash_drought_alert or 0,
            )
            active_crop_stage = agri_eval["phenology"]["growth_stage"]

            # Flagship 5: AirQuality V2
            aq_eval = self.aq_engine.evaluate_air_quality(
                city=forecast.city,
                pm2_5_ug_m3=day.pm2_5,
                pm10_ug_m3=day.pm10,
                no2_ug_m3=day.no2,
                wind_speed_kmh=day.wind_speed_kmh,
                temp_max_c=day.temp_max,
                temp_min_c=day.temp_min,
                precipitation_mm=day.precipitation_mm,
                month=dt.month,
                day_of_year=doy,
                population_15km=pop_info.get("population_15km"),
                schools_15km=infra_info.get("schools_15km"),
                hospitals_15km=infra_info.get("hospitals_15km"),
            )

            # Elevation-aware psychrometric wet-bulb
            from climacore.analytics.psychrometrics import calculate_elevation_aware_wet_bulb
            elev_wb = calculate_elevation_aware_wet_bulb(day.temp_max, day.relative_humidity, city=forecast.city)
            effective_wet_bulb = day.wet_bulb_temp_c if day.wet_bulb_temp_c is not None else elev_wb["wet_bulb_temp_c"]
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
                antecedent_rain_3d_mm=round(day.precipitation_mm * 1.5, 1),
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

            # 3. Probabilistic Conformal Prediction Bounds (>= 90% finite-sample coverage)
            conf_temp = calibrate_temperature_conformal(day.temp_max, alpha=0.10)
            conf_rain = calibrate_rainfall_conformal(day.precipitation_mm, alpha=0.10)
            conf_power = calibrate_power_demand_conformal(power_eval["peak_load_mw"], alpha=0.10)
            conf_pm25 = calibrate_pm25_conformal(day.pm2_5, alpha=0.10)
            conformal_bounds = {
                "temperature_c": conf_temp,
                "rainfall_mm": conf_rain,
                "peak_grid_load_mw": conf_power,
                "pm2_5_ug_m3": conf_pm25,
            }

            # 4. Cascading Multi-Hazard Infrastructure Dependency Failure Chains
            cascade_eval = self.cascading_engine.evaluate_cascading_chains(
                city=forecast.city,
                temp_max_c=day.temp_max,
                temp_min_c=day.temp_min,
                rainfall_mm=day.precipitation_mm,
                pm2_5_ug_m3=day.pm2_5,
                is_tropical_night=heat_eval["is_tropical_night"],
                grid_deficit_prob=power_eval["unmet_supply_stress_probability"],
                flood_hazard_prob=flood_disaster_risk,
                stagnation_flag=aq_eval["atmospheric_physics"]["stagnation_flag"],
                population_15km=pop_info.get("population_15km"),
                dense_cluster_pop_15km=pop_info.get("dense_cluster_pop_15km"),
                hospitals_15km=infra_info.get("hospitals_15km"),
                bridges_15km=infra_info.get("bridges_15km"),
                drainage_canals_15km=infra_info.get("drainage_canals_15km"),
                arterial_roads_15km=infra_info.get("arterial_roads_15km"),
            )

            # 5. Evaluate Selected Persona or All 13 Personas
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
                    flagship_intelligence={
                        "flood_risk": flood_eval,
                        "heat_risk": heat_eval,
                        "power_grid": power_eval,
                        "agrisense": agri_eval,
                        "air_quality": aq_eval,
                    },
                    conformal_bounds=conformal_bounds,
                    cascading_risk=cascade_eval,
                )
            )

        provenance = {
            "engine_version": "v1.0.0",
            "phase_architecture": "Phase 4 Probabilistic & Compound Environmental Intelligence",
            "champion_models": {
                "flood_risk": "IFI LightGBM v1 (AUC 0.7931)",
                "heat_risk": "HeatRisk V4 Dual Diurnal GEV",
                "electricity_grid": "POSOCO LightGBM Dual Regressor/Classifier (R2 0.8758, AUC 0.9521)",
                "agrisense": "AgriSense LightGBM Multi-Class GBDT (Top-1 82.49%)",
                "air_quality": "CPCB NAQI Multi-Pollutant Transfer Calibrator",
            },
            "tournament_manifest": "data/manifests/champion_challenger_tournament.json",
            "canonical_registry": "data/manifests/canonical_model_registry.json",
        }

        spatial_meta = {
            "concentric_buffers": ["5km (immediate urban core)", "15km (commuter / lifeline infrastructure)", "30km (hinterland)"],
            "demographic_resolution": "WorldPop 100m spatial population grid",
            "reanalysis_grid_resolution": "ERA5-Land 0.1° (~9km), IMD 0.25° gridded reanalysis",
            "infrastructure_topology": "OpenStreetMap vector topology (hospitals, bridges, canals, roads)",
        }

        return LivePredictionResult(
            schema_version="v1.0.0",
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
            system_provenance=provenance,
            spatial_resolution_metadata=spatial_meta,
            physical_units_manifest=PHYSICAL_UNITS_MANIFEST,
        )
