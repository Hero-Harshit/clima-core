"""
FastAPI Serverless & Edge-Ready Inference Application for Clima-Core (Phase 5).
Optimized for Vercel Python Functions and local production serving.
Decoupled architecture: Loads only lightweight booster artifacts and cached bundles.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Dict, List, Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

from climacore.common.logging import get_logger
from climacore.ingestion.engine import LiveForecastEngine
from climacore.ingestion.models import LivePredictionResult
from climacore.monitoring.service import get_monitoring_service
from climacore.evaluation.gatekeeper import DeploymentGatekeeper, GatekeeperCriteria
from climacore.registry import get_model_registry, get_dataset_registry
from climacore.service.assistant import (
    AssistantEngine,
    AssistantChatRequest,
    AssistantChatResponse,
)

logger = get_logger("climacore.service.app")

# Global engine cache (singleton for warm container reuse)
_LIVE_ENGINE: Optional[LiveForecastEngine] = None
_ASSISTANT_ENGINE: Optional[AssistantEngine] = None
STARTUP_TIME = datetime.now(timezone.utc)


def get_live_engine() -> LiveForecastEngine:
    global _LIVE_ENGINE
    if _LIVE_ENGINE is None:
        _LIVE_ENGINE = LiveForecastEngine()
    return _LIVE_ENGINE


def get_assistant_engine() -> AssistantEngine:
    global _ASSISTANT_ENGINE
    if _ASSISTANT_ENGINE is None:
        _ASSISTANT_ENGINE = AssistantEngine()
    return _ASSISTANT_ENGINE


# Pydantic Schemas for Request & Response
class CityPredictionRequest(BaseModel):
    city: str = Field(..., description="Indian city name (e.g. Delhi, Mumbai, Kolkata, Bengaluru)")
    requested_personas: Optional[List[str]] = Field(default=None, description="Optional list of personas to evaluate")
    force_refresh: bool = Field(default=False, description="Bypass local cache and query upstream APIs")


class CustomPredictionRequest(BaseModel):
    city_name: str = Field(default="Custom Coordinates")
    latitude: float = Field(default=28.61, ge=-90.0, le=90.0)
    longitude: float = Field(default=77.23, ge=-180.0, le=180.0)
    temperature_max: float = Field(..., description="Daily maximum temperature in degC")
    temperature_min: float = Field(..., description="Daily minimum temperature in degC")
    rainfall_mm: float = Field(..., ge=0.0, description="Daily precipitation in mm")
    relative_humidity: float = Field(default=50.0, ge=0.0, le=100.0)
    wind_speed_kmh: float = Field(default=12.0, ge=0.0)
    pm25: float = Field(default=60.0, ge=0.0, description="PM2.5 in ug/m3")
    pm10: Optional[float] = Field(default=None, ge=0.0)
    soil_saturation_ratio: float = Field(default=0.40, ge=0.0, le=1.0)
    requested_personas: Optional[List[str]] = None


class WaterQualityRequest(BaseModel):
    latitude: float = Field(default=19.0760, ge=-90.0, le=90.0)
    longitude: float = Field(default=72.8777, ge=-180.0, le=180.0)
    city: str = Field(default="Mumbai")
    ph: Optional[float] = Field(default=None, ge=0.0, le=14.0)
    electric_conductivity: Optional[float] = Field(default=None, ge=0.0)
    total_hardness: Optional[float] = Field(default=None, ge=0.0)
    chloride: Optional[float] = Field(default=None, ge=0.0)
    sulphate: Optional[float] = Field(default=None, ge=0.0)
    calcium: Optional[float] = Field(default=None, ge=0.0)
    magnesium: Optional[float] = Field(default=None, ge=0.0)
    sodium: Optional[float] = Field(default=None, ge=0.0)
    is_drought_stressed: bool = False
    is_flood_inundated: bool = False


class VerificationRecordRequest(BaseModel):
    target_name: str = Field(..., description="Target variable name (e.g. temperature_max, rainfall_mm)")
    y_true: float = Field(..., description="Actual verifying ground truth observation")
    interval_lower: float = Field(..., description="Conformal interval lower bound")
    interval_upper: float = Field(..., description="Conformal interval upper bound")


class CrisisCouncilRequest(BaseModel):
    city: str = Field(..., description="Target Indian city for emergency incident command")
    temp_max_c: Optional[float] = Field(default=None, description="Optional simulated maximum temperature in Celsius")
    temp_min_c: Optional[float] = Field(default=None, description="Optional simulated minimum temperature in Celsius")
    rainfall_mm: Optional[float] = Field(default=None, description="Optional simulated 24h rainfall in millimeters")
    wet_bulb_c: Optional[float] = Field(default=None, description="Optional wet-bulb temperature in Celsius")
    pm25: Optional[float] = Field(default=None, description="Optional PM2.5 in ug/m3")
    peak_grid_load_mw: Optional[float] = Field(default=None, description="Optional baseline grid peak demand in MW")


class DisasterReplayRequest(BaseModel):
    disaster_id: str = Field(default="MUMBAI_2005_CLOUDBURST", description="Historical disaster ID, e.g. 'MUMBAI_2005_CLOUDBURST', 'CHENNAI_2015_DELUGE', 'NORTH_INDIA_2022_HEATWAVE'")
    city_override: Optional[str] = None


class GatekeeperEvaluationRequest(BaseModel):
    model_name: str
    champion_metric: float
    challenger_metric: float
    champion_version: str = "v2.0.0"
    challenger_version: str = "v3.0.0"
    conformal_coverage: Optional[float] = None
    loczo_zone_scores: Optional[Dict[str, float]] = None
    expected_features: Optional[List[str]] = None
    candidate_features: Optional[List[str]] = None
    higher_is_better: bool = True
    min_metric_delta: float = 0.0


CANONICAL_PERSONAS: List[Dict[str, Any]] = [
    {
        "id": "farmer",
        "title": "Farmer & Agriculture",
        "category": "Agriculture & Rural",
        "icon": "Wheat",
        "description": "Crop heat stress, soil moisture deficit, irrigation scheduling, and rainfall deluge alerts.",
        "primary_concerns": ["Soil saturation", "Crop anthesis heat", "Pesticide drift", "Runoff erosion"],
    },
    {
        "id": "student",
        "title": "Student & School",
        "category": "Education & Youth",
        "icon": "GraduationCap",
        "description": "School closure advisories, heatwave exposure, commute safety, and indoor AQI alerts.",
        "primary_concerns": ["Extreme heat commute", "Severe AQI school warnings", "Flash flood transport safety"],
    },
    {
        "id": "gig_worker",
        "title": "Gig Worker & Outdoor Labor",
        "category": "Urban Labor & Logistics",
        "icon": "Bike",
        "description": "Continuous outdoor wet-bulb heat stress, delivery hazard index, and hydration rest stops.",
        "primary_concerns": ["Wet-bulb globe temperature", "Peak midday heat index", "Flooded arterial roads"],
    },
    {
        "id": "commuter",
        "title": "Daily Commuter & Transit",
        "category": "Transportation & Public",
        "icon": "Car",
        "description": "Road waterlogging, suburban rail disruption, localized urban flooding, and storm wind gusts.",
        "primary_concerns": ["Urban road waterlogging", "Transit delays", "Gale wind visibility"],
    },
    {
        "id": "healthcare",
        "title": "Healthcare & Hospital Ops",
        "category": "Public Health & Emergency",
        "icon": "Stethoscope",
        "description": "Hospital heatstroke surge preparedness, respiratory emergency admissions, and ICU power backup.",
        "primary_concerns": ["Heat-related illness influx", "Severe air quality exacerbation", "Hospital grid stability"],
    },
    {
        "id": "city_official",
        "title": "City Official & Disaster Mgmt",
        "category": "Governance & Civil Defense",
        "icon": "Building2",
        "description": "NDMA Heat Action Plan (HAP) stage triggers, municipal flood alert levels, and shelter deployment.",
        "primary_concerns": ["NDMA HAP alerts", "CWC reservoir levels", "Drainage outfall capacity", "Evacuation corridors"],
    },
    {
        "id": "maritime",
        "title": "Maritime & Fishermen",
        "category": "Coastal & Marine",
        "icon": "Ship",
        "description": "Coastal sea state, cyclone proximity, gale wind alerts, and rough sea fishing bans.",
        "primary_concerns": ["Cyclone track distance", "Beaufort gale force", "High astronomical tides"],
    },
    {
        "id": "industry",
        "title": "Industry & Manufacturing",
        "category": "Manufacturing & Power",
        "icon": "Factory",
        "description": "Cooling tower heat dissipation, peak electrical demand surcharges, and factory floor thermal limits.",
        "primary_concerns": ["Wet-bulb cooling tower limits", "Grid peak load shedding", "Worker shift restructuring"],
    },
    {
        "id": "aviation",
        "title": "Aviation & Airports",
        "category": "Aviation & Logistics",
        "icon": "Plane",
        "description": "Crosswind runway limits, convective thunderstorms, air density altitude, and ramp ground stops.",
        "primary_concerns": ["Surface wind shear", "Convective storm cells", "Density altitude thrust margins"],
    },
    {
        "id": "water",
        "title": "Water Utility & Stormwater",
        "category": "Municipal Infrastructure",
        "icon": "Droplets",
        "description": "Stormwater pumping station capacity, reservoir storage management, and sewer overflow hazards.",
        "primary_concerns": ["SCS runoff peak volume", "Reservoir buffer capacity", "Urban drainage canal surcharge"],
    },
    {
        "id": "insurance",
        "title": "Insurance & Catastrophe Actuary",
        "category": "Financial Services",
        "icon": "ShieldAlert",
        "description": "Parametric weather derivative triggers, crop insurance claims, and catastrophe portfolio exposure.",
        "primary_concerns": ["Parametric payout triggers", "Multi-hazard joint probabilities", "Asset exposure density"],
    },
    {
        "id": "cold_chain",
        "title": "Cold Chain Logistics",
        "category": "Supply Chain & Food Safety",
        "icon": "Truck",
        "description": "Reefer transport thermal breach probability, ambient cooling deficit, and vaccine/dairy preservation.",
        "primary_concerns": ["Ambient thermal breach", "Grid refrigeration downtime", "Perishable cargo spoilage"],
    },
    {
        "id": "tourism",
        "title": "Tourism & Hospitality",
        "category": "Hospitality & Leisure",
        "icon": "Palmtree",
        "description": "Outdoor tourist comfort index, pilgrimage mountain/monsoon route risk, and hotel cooling stress.",
        "primary_concerns": ["Outdoor bioclimatic comfort", "Pilgrimage cloudburst safety", "Hotel power continuity"],
    },
]


def create_app() -> FastAPI:
    """Application factory for Clima-Core FastAPI service."""
    app = FastAPI(
        title="Clima-Core Environmental Intelligence API",
        description="Unified Multi-Source Probabilistic Environmental Risk API for Indian Cities",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> Dict[str, Any]:
        return {
            "platform": "Clima-Core Environmental Intelligence System",
            "version": "1.0.0",
            "schema_contract": "v1.0.0",
            "status": "OPERATIONAL",
            "docs": "/docs",
            "endpoints": [
                "/api/health",
                "/api/cities",
                "/api/personas",
                "/api/assistant/chat",
                "/api/predict/city",
                "/api/predict/custom",
                "/api/predict/water_quality",
                "/api/monitoring/health",
                "/api/monitoring/verify",
                "/api/tournament/status",
                "/api/gatekeeper/evaluate",
                "/api/crisis-council/deliberate",
                "/api/disaster-replay/events",
                "/api/disaster-replay/run",
            ],
        }

    @app.get("/api/personas")
    def list_personas() -> Dict[str, Any]:
        """List all 13 canonical personas with metadata, icons, and sector descriptions."""
        return {
            "total": len(CANONICAL_PERSONAS),
            "personas": CANONICAL_PERSONAS,
        }

    @app.get("/api/cities")
    def list_cities(q: Optional[str] = Query(default=None, description="Optional search prefix or substring")) -> Dict[str, Any]:
        """List standard Indian cities with coordinates, supporting optional prefix/substring filtering."""
        from climacore.ingestion.geocoding import _load_standard_cities
        raw_cities = _load_standard_cities()
        unique_map = {}
        for c in raw_cities.values():
            if c.get("is_standard_50"):
                unique_map[c["city"]] = {
                    "city": c["city"],
                    "state": c.get("state", "India"),
                    "latitude": c["latitude"],
                    "longitude": c["longitude"],
                }
        cities_list = sorted(list(unique_map.values()), key=lambda x: x["city"])
        if q:
            q_lower = q.strip().lower()
            cities_list = [c for c in cities_list if q_lower in c["city"].lower() or q_lower in c["state"].lower()]
        return {
            "total": len(cities_list),
            "cities": cities_list,
        }

    @app.get("/health")
    @app.get("/api/health")
    def health_check() -> Dict[str, Any]:
        monitor = get_monitoring_service()
        model_reg = get_model_registry()
        uptime_sec = (datetime.now(timezone.utc) - STARTUP_TIME).total_seconds()

        return {
            "status": "HEALTHY",
            "schema_version": "v1.0.0",
            "uptime_seconds": round(uptime_sec, 1),
            "registered_models_count": len(model_reg),
            "flagship_champions": [
                "FloodRisk V3 (Hydro-Atmospheric Cascade + HydroRIVERS)",
                "HeatRisk V4 (Dual Diurnal + Oke UHI + Conformal)",
                "PowerGrid V3 (Lags + Regressor + Conformal)",
                "AgriSense V3 (Suitability GBDT + Yield Quantiles + SAGE)",
                "AirQuality V3 (Ventilation + Nocturnal Inversion)",
                "WaterQuality V1 (CGWB Chemical Potability GBDT)",
            ],
            "latency_profile": monitor.latency_profiler.get_summary(),
        }

    @app.post("/api/predict/city")
    def predict_city(req: CityPredictionRequest) -> Dict[str, Any]:
        monitor = get_monitoring_service()
        t0 = time.perf_counter()
        try:
            engine = get_live_engine()
            persona_arg = req.requested_personas[0] if (req.requested_personas and len(req.requested_personas) == 1) else "all"
            res = engine.predict(
                query=req.city,
                persona=persona_arg,
                force_refresh=req.force_refresh,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=False)
            return res.model_dump()
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=True)
            logger.error(f"Error predicting city {req.city}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed for city {req.city}: {str(e)}")

    @app.post("/api/assistant/chat", response_model=AssistantChatResponse)
    async def chat_with_assistant(req: AssistantChatRequest) -> AssistantChatResponse:
        """Cognitive AI Assistant grounded in verified physical environmental data."""
        monitor = get_monitoring_service()
        t0 = time.perf_counter()
        try:
            live_engine = get_live_engine()
            assistant_engine = get_assistant_engine()

            # Retrieve or fetch live prediction contract for context grounding
            canonical_ids = {p["id"] for p in CANONICAL_PERSONAS}
            persona_to_fetch = req.persona if req.persona in canonical_ids else "all"
            pred = live_engine.predict(
                query=req.city,
                persona=persona_to_fetch,
                force_refresh=req.force_refresh,
            )

            # Generate grounded response
            chat_resp = await assistant_engine.chat(req=req, prediction_result=pred, live_engine=live_engine)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=False)
            return chat_resp
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=True)
            logger.error(f"Error in assistant chat for {req.city}: {e}")
            raise HTTPException(status_code=500, detail=f"Assistant chat failed: {str(e)}")

    @app.post("/api/crisis-council/deliberate")
    def deliberate_crisis_council(req: CrisisCouncilRequest) -> Dict[str, Any]:
        """Autonomous multi-agent Crisis Council War Room deliberation."""
        from climacore.analytics.crisis_council import simulate_crisis_council_deliberation
        monitor = get_monitoring_service()
        t0 = time.perf_counter()
        try:
            live_engine = get_live_engine()
            t_max = req.temp_max_c
            t_min = req.temp_min_c
            rain = req.rainfall_mm
            tw = req.wet_bulb_c
            pm25 = req.pm25
            grid_mw = req.peak_grid_load_mw

            if t_max is None or rain is None:
                pred = live_engine.predict(query=req.city, persona="city_official")
                today = pred.daily_predictions[0]
                m = today.forecast_metrics
                if t_max is None:
                    t_max = float(m.get("temp_max", 35.0))
                if t_min is None:
                    t_min = float(m.get("temp_min", 25.0))
                if rain is None:
                    rain = float(m.get("precipitation_mm", 0.0))
                if tw is None:
                    tw = float(m.get("wet_bulb_temp_c", 26.0) or 26.0)
                if pm25 is None:
                    pm25 = float(m.get("pm2_5", 55.0))
                if grid_mw is None:
                    grid_info = today.flagship_intelligence.get("power_grid_stress", {}) if today.flagship_intelligence else {}
                    grid_mw = float(grid_info.get("peak_load_mw", 8500.0) or 8500.0)

            res = simulate_crisis_council_deliberation(
                city=req.city,
                temp_max_c=t_max,
                temp_min_c=t_min or max(15.0, t_max - 10.0),
                rainfall_mm=rain,
                wet_bulb_c=tw,
                pm25=pm25,
                peak_grid_load_mw=grid_mw,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=False)
            data = res.model_dump()
            data["latency_ms"] = round(lat_ms, 2)
            return data
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=True)
            logger.error(f"Error in crisis council deliberation for {req.city}: {e}")
            raise HTTPException(status_code=500, detail=f"Crisis Council deliberation failed: {str(e)}")

    @app.get("/api/disaster-replay/events")
    def get_disaster_replay_events() -> Dict[str, Any]:
        """List all historical disaster benchmarks available for backtesting replay."""
        from climacore.analytics.disaster_replay import list_historical_disasters
        events = list_historical_disasters()
        return {
            "total_events": len(events),
            "events": events,
        }

    @app.post("/api/disaster-replay/run")
    def run_historical_disaster_replay(req: DisasterReplayRequest) -> Dict[str, Any]:
        """Execute full chronological backtesting replay for a selected historical catastrophe."""
        from climacore.analytics.disaster_replay import run_disaster_replay
        monitor = get_monitoring_service()
        t0 = time.perf_counter()
        try:
            res = run_disaster_replay(disaster_id=req.disaster_id, city_override=req.city_override)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=False)
            data = res.model_dump()
            data["latency_ms"] = round(lat_ms, 2)
            return data
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=True)
            logger.error(f"Error running disaster replay for {req.disaster_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Disaster replay failed: {str(e)}")

    @app.post("/api/predict/custom")
    def predict_custom(req: CustomPredictionRequest) -> Dict[str, Any]:
        monitor = get_monitoring_service()
        t0 = time.perf_counter()
        try:
            engine = get_live_engine()

            # Evaluate heat risk
            heat_res = engine.heat_engine.evaluate_heat_risk(
                city=req.city_name,
                temp_max_c=req.temperature_max,
                temp_min_c=req.temperature_min,
                relative_humidity_pct=req.relative_humidity,
                wind_speed_kmh=req.wind_speed_kmh,
            )
            # Evaluate flood risk
            flood_res = engine.flood_engine.evaluate_flood_risk(
                city=req.city_name,
                rainfall_mm=req.rainfall_mm,
                rain_rolling_3d=round(req.rainfall_mm * 1.5, 1),
                smdi=req.soil_saturation_ratio,
            )
            # Evaluate grid risk
            grid_res = engine.power_engine.evaluate_power_grid_risk(
                temp_max_c=req.temperature_max,
                temp_min_c=req.temperature_min,
                city=req.city_name,
            )
            # Evaluate air quality risk
            air_res = engine.aq_engine.evaluate_air_quality(
                city=req.city_name,
                pm2_5_ug_m3=req.pm25,
                t_max=req.temperature_max,
                t_min=req.temperature_min,
                wind_speed_kmh=req.wind_speed_kmh,
            )
            # Conformal calibration intervals
            from climacore.analytics.conformal import (
                calibrate_temperature_conformal,
                calibrate_rainfall_conformal,
                calibrate_power_demand_conformal,
                calibrate_pm25_conformal,
            )
            conf_t = calibrate_temperature_conformal(req.temperature_max)
            conf_r = calibrate_rainfall_conformal(req.rainfall_mm)
            conf_p = calibrate_power_demand_conformal(grid_res["peak_load_mw"])
            conf_a = calibrate_pm25_conformal(req.pm25)

            # Cascading compound risks
            cascades = engine.cascading_engine.evaluate_cascading_chains(
                city=req.city_name,
                temp_max_c=req.temperature_max,
                temp_min_c=req.temperature_min,
                rainfall_mm=req.rainfall_mm,
                pm2_5_ug_m3=req.pm25,
                is_tropical_night=heat_res.get("is_tropical_night", False),
                grid_deficit_prob=grid_res.get("unmet_supply_stress_probability", 0.05),
                flood_hazard_prob=flood_res.get("hydrological_hazard_probability", 0.05),
                stagnation_flag=air_res.get("atmospheric_physics", {}).get("stagnation_flag", 0),
            )

            # Evaluate groundwater chemical quality
            wq_res = engine.water_quality_engine.evaluate_water_quality(
                latitude=req.latitude,
                longitude=req.longitude,
                city=req.city_name,
                is_drought_stressed=bool(req.temperature_max >= 42.0 and req.rainfall_mm == 0.0),
                is_flood_inundated=bool(flood_res.get("hydrological_hazard_probability", 0.0) >= 0.40),
            )

            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=False)

            return {
                "schema_version": "v1.0.0",
                "location": {
                    "city_name": req.city_name,
                    "latitude": req.latitude,
                    "longitude": req.longitude,
                },
                "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                "predictions": {
                    "heat_risk": heat_res,
                    "flood_risk": flood_res,
                    "power_grid_stress": grid_res,
                    "air_quality": air_res,
                    "water_quality": wq_res,
                },
                "conformal_bounds": {
                    "temperature": conf_t,
                    "rainfall": conf_r,
                    "power_demand": conf_p,
                    "pm25": conf_a,
                },
                "cascading_risk": cascades,
                "latency_ms": round(lat_ms, 2),
            }
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=True)
            logger.error(f"Error in custom prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Custom prediction failed: {str(e)}")

    @app.post("/api/predict/water_quality")
    def predict_water_quality(req: WaterQualityRequest) -> Dict[str, Any]:
        monitor = get_monitoring_service()
        t0 = time.perf_counter()
        try:
            engine = get_live_engine()
            res = engine.water_quality_engine.evaluate_water_quality(
                latitude=req.latitude,
                longitude=req.longitude,
                city=req.city,
                ph=req.ph,
                electric_conductivity=req.electric_conductivity,
                total_hardness=req.total_hardness,
                chloride=req.chloride,
                sulphate=req.sulphate,
                calcium=req.calcium,
                magnesium=req.magnesium,
                sodium=req.sodium,
                is_drought_stressed=req.is_drought_stressed,
                is_flood_inundated=req.is_flood_inundated,
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=False)
            res["latency_ms"] = round(lat_ms, 2)
            return res
        except Exception as e:
            lat_ms = (time.perf_counter() - t0) * 1000.0
            monitor.latency_profiler.record(lat_ms, is_error=True)
            logger.error(f"Error predicting water quality: {e}")
            raise HTTPException(status_code=500, detail=f"Water quality prediction failed: {str(e)}")

    @app.get("/api/monitoring/health")
    def get_monitoring_health() -> Dict[str, Any]:
        monitor = get_monitoring_service()
        # Sample check of default baseline features against a normal operational batch
        np.random.seed(int(time.time()) % 10000)
        sample_prod = {
            "temperature_max": np.random.normal(loc=33.0, scale=5.8, size=100),
            "rainfall_mm": np.random.exponential(scale=11.5, size=100),
            "peak_grid_load_mw": np.random.normal(loc=12600.0, scale=2400.0, size=100),
            "pm25_concentration": np.random.lognormal(mean=3.85, sigma=0.75, size=100),
        }
        return monitor.evaluate_production_batch(sample_prod)

    @app.post("/api/monitoring/verify")
    def verify_observation(rec: VerificationRecordRequest) -> Dict[str, Any]:
        monitor = get_monitoring_service()
        covered = monitor.calibration_monitor.record_verification(
            target_name=rec.target_name,
            y_true=rec.y_true,
            interval_lower=rec.interval_lower,
            interval_upper=rec.interval_upper,
        )
        return {
            "recorded": True,
            "covered": covered,
            "current_health": monitor.calibration_monitor.get_health_status(),
        }

    @app.get("/api/tournament/status")
    def get_tournament_status() -> Dict[str, Any]:
        from climacore.config.loader import get_project_root
        import json
        manifest_file = get_project_root() / "data" / "manifests" / "champion_challenger_tournament.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"status": "MANIFEST_NOT_FOUND", "note": "Run tournament suite to generate manifest"}

    @app.post("/api/gatekeeper/evaluate")
    def evaluate_gatekeeper(req: GatekeeperEvaluationRequest) -> Dict[str, Any]:
        criteria = GatekeeperCriteria(
            min_metric_delta=req.min_metric_delta,
            higher_is_better=req.higher_is_better,
        )
        gatekeeper = DeploymentGatekeeper(criteria=criteria)
        report = gatekeeper.evaluate_promotion(
            model_name=req.model_name,
            champion_metric=req.champion_metric,
            challenger_metric=req.challenger_metric,
            champion_version=req.champion_version,
            challenger_version=req.challenger_version,
            conformal_coverage=req.conformal_coverage,
            loczo_zone_scores=req.loczo_zone_scores,
            expected_schema_features=req.expected_features,
            candidate_features=req.candidate_features,
        )
        return report.to_dict()

    return app


# Default ASGI application instance
app = create_app()
