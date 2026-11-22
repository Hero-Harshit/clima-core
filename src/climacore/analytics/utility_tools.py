"""
Utility Tools and Simulation Engines for ClimaCore.
Powers:
1. Delivery Corridor Radar (Route waterlogging, crosswinds, and asphalt thermal stress)
2. Tourist Itinerary Optimizer (Diurnal weather curve, UV index, and sightseeing scheduler)
3. Clean Air Athletic Window (Inversion layer dissipation, AQI, and workout timefinder)
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger("climacore.analytics.utility_tools")

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# In-memory LRU cache for utility tool results
_TOOL_CACHE: Dict[str, Any] = {}


# ============================================================================
# Schemas: 1. Delivery Corridor Radar
# ============================================================================

class CorridorSegmentRisk(BaseModel):
    segment_name: str
    hazard_type: str  # WATERLOGGING, CROSSWIND, ASPHALT_HEAT, LOW_VISIBILITY, CLEAR
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    advisory: str
    speed_reduction_pct: int


class CorridorRadarRequest(BaseModel):
    city: str = Field(default="Delhi")
    origin: str = Field(default="City Center")
    destination: str = Field(default="Logistics Hub")
    vehicle_type: str = Field(default="two_wheeler")  # two_wheeler, ev_van, light_commercial, heavy_truck
    language: Optional[str] = Field(default="en")


class CorridorRadarResponse(BaseModel):
    city: str
    origin: str
    destination: str
    vehicle_type: str
    corridor_risk_score: int
    status_level: str
    estimated_delay_mins: int
    surface_runoff_mm: float
    max_wind_gust_kmh: float
    asphalt_temp_c: float
    operational_guidance: str
    segments: List[CorridorSegmentRisk]
    vehicle_recommendations: List[str]
    ai_insights: str
    provider: str


# ============================================================================
# Schemas: 2. Tourist Itinerary Optimizer
# ============================================================================

class ItineraryHourSlot(BaseModel):
    time_window: str
    period_label: str
    activity_title: str
    venue_type: str  # OUTDOOR, INDOOR, SHELTERED
    weather_suitability: str  # EXCELLENT, ACCEPTABLE, UNFAVORABLE
    temp_c: float
    uv_index: int
    aqi_category: str
    recommendation: str


class TouristOptimizerRequest(BaseModel):
    city: str = Field(default="Delhi")
    travel_date: Optional[str] = Field(default=None)
    interests: List[str] = Field(default_factory=lambda: ["heritage", "culinary"])
    pace: str = Field(default="balanced")  # relaxed, balanced, intensive
    language: Optional[str] = Field(default="en")


class TouristOptimizerResponse(BaseModel):
    city: str
    travel_date: str
    pace: str
    comfort_score: int
    overall_verdict: str
    weather_summary: str
    itinerary: List[ItineraryHourSlot]
    packing_checklist: List[str]
    local_hydrated_spots: List[str]
    ai_concierge_briefing: str
    provider: str


# ============================================================================
# Schemas: 3. Clean Air Athletic Window
# ============================================================================

class AirQualityHourEstimate(BaseModel):
    time_slot: str
    aqi: int
    pm25: float
    inversion_trapping: bool
    athletic_score: int
    is_optimal: bool


class CleanAirWorkoutRequest(BaseModel):
    city: str = Field(default="Delhi")
    activity_type: str = Field(default="running")  # running, cycling, brisk_walking, hiit
    preferred_time: str = Field(default="any")  # early_morning, midday, evening, any
    intensity: str = Field(default="moderate")  # light, moderate, vigorous
    language: Optional[str] = Field(default="en")


class CleanAirWorkoutResponse(BaseModel):
    city: str
    activity_type: str
    athletic_suitability_score: int
    status_level: str  # OPTIMAL, ACCEPTABLE, CAUTION, INDOORS_ONLY
    recommended_window: str
    inversion_status: str
    current_aqi: int
    current_pm25: float
    wbgt_thermal_strain_c: float
    hourly_forecast: List[AirQualityHourEstimate]
    coaching_guidance: str
    respiratory_precautions: List[str]
    indoor_alternative: str
    provider: str


# ============================================================================
# Implementation: 1. Delivery Corridor Radar
# ============================================================================

async def analyze_delivery_corridor(
    req: CorridorRadarRequest,
    current_weather: Optional[Dict[str, Any]] = None,
) -> CorridorRadarResponse:
    city = req.city.strip() or "Delhi"
    origin = req.origin.strip() or "City Center"
    destination = req.destination.strip() or "Logistics Hub"
    v_type = (req.vehicle_type or "two_wheeler").strip().lower()
    norm_lang = (req.language or "en").strip().lower()

    # Extract or estimate base metrics
    metrics = current_weather or {}
    t_max = float(metrics.get("temp_max") or metrics.get("temperature_max") or 33.0)
    precip_mm = float(metrics.get("precipitation_mm") or metrics.get("runoff_mm") or 1.2)
    wind_kmh = float(metrics.get("wind_speed_kmh") or metrics.get("gale_gust_kmh") or 22.0)
    asphalt_temp = round(t_max + 8.5, 1)

    # Compute baseline risk
    risk_score = 25
    if precip_mm > 15:
        risk_score += 35
    elif precip_mm > 5:
        risk_score += 20
    if wind_kmh > 45:
        risk_score += 25
    elif wind_kmh > 30:
        risk_score += 15
    if asphalt_temp > 42:
        risk_score += 15
    risk_score = min(95, max(15, risk_score))

    status_level = "CLEAR" if risk_score < 40 else "CAUTION" if risk_score < 70 else "HIGH_HAZARD"
    delay_mins = int(risk_score * 0.4)

    # Generate segments
    segments = [
        CorridorSegmentRisk(
            segment_name=f"{origin} to Ring Road Expressway",
            hazard_type="CROSSWIND" if wind_kmh > 35 else "CLEAR",
            risk_level="MODERATE" if wind_kmh > 35 else "LOW",
            advisory="Flyover exposed to side gusts. Maintain firm handlebar grip." if v_type == "two_wheeler" else "Maintain steady lane discipline across bridge spans.",
            speed_reduction_pct=15 if wind_kmh > 35 else 0,
        ),
        CorridorSegmentRisk(
            segment_name="Low-Lying Underpass & Drainage Choke",
            hazard_type="WATERLOGGING" if precip_mm > 8 else "CLEAR",
            risk_level="HIGH" if precip_mm > 8 else "LOW",
            advisory="Water buildup depth > 15cm possible in underpass dips. Use elevated ramp detour." if precip_mm > 8 else "Catch-basins flowing freely. Normal clearance.",
            speed_reduction_pct=35 if precip_mm > 8 else 5,
        ),
        CorridorSegmentRisk(
            segment_name=f"Industrial Freight Access to {destination}",
            hazard_type="ASPHALT_HEAT" if asphalt_temp > 42 else "CLEAR",
            risk_level="MODERATE" if asphalt_temp > 42 else "LOW",
            advisory=f"Road surface heat reaching {asphalt_temp}°C. Inspect tire pressure and EV battery temperature." if v_type == "ev_van" else f"Asphalt temperature at {asphalt_temp}°C. Watch for tire tread wear and brake fade.",
            speed_reduction_pct=10 if asphalt_temp > 42 else 0,
        ),
    ]

    vehicle_tips = {
        "two_wheeler": [
            "Wear a certified high-visibility reflective rain jacket and non-slip boots.",
            "Avoid low-lying underpasses if standing water exceeds curb height (10 cm).",
            "Reduce speed by 20 km/h on wet road expansion joints and painted lane markings.",
        ],
        "ev_van": [
            "Monitor high-voltage battery pack temperatures during stop-and-go city traffic.",
            "Regenerative braking efficiency decreases on wet asphalt; increase following distance.",
            "Pre-condition cabin cooling while plugged in to preserve 12-15% driving range.",
        ],
        "light_commercial": [
            "Check cargo tie-downs against crosswind turbulence on elevated bypass flyovers.",
            "Inspect drum brake clearance after wading through water accumulation spots.",
            "Maintain coolant levels and check radiator fins for monsoon leaf debris.",
        ],
        "heavy_truck": [
            "Avoid abrupt braking on wet descending ramps to prevent trailer jackknifing.",
            "Verify low-clearance railway bridges and flood gauge boards prior to entering bypass roads.",
            "Schedule heavy freight transit outside peak afternoon urban heat hours.",
        ],
    }

    v_recs = vehicle_tips.get(v_type, vehicle_tips["two_wheeler"])
    op_guidance = (
        f"Corridor transit between {origin} and {destination} is under {status_level.replace('_', ' ')}. "
        f"Anticipate approximately {delay_mins} minutes of weather-related transit latency."
    )
    ai_insights = (
        f"ClimaCore predictive models indicate road surface temperatures of {asphalt_temp}°C with peak wind "
        f"gusts reaching {wind_kmh} km/h along the {origin} corridor. Vehicles of type '{v_type}' should exercise "
        f"caution around elevated flyovers and potential underpass runoff accumulation."
    )

    # Try Gemini call if available
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        sys_prompt = (
            "You are ClimaCore's Logistics Corridor Weather Radar AI. "
            "Generate actionable driver advice for an urban freight corridor in an Indian city.\n"
            "Respond ONLY with a valid JSON object matching:\n"
            "{\n"
            '  "operational_guidance": "string",\n'
            '  "ai_insights": "string",\n'
            '  "vehicle_recommendations": ["string", "string", "string"]\n'
            "}"
        )
        user_prompt = (
            f"City: {city}, Route: {origin} to {destination}, Vehicle: {v_type}.\n"
            f"Surface Runoff: {precip_mm} mm, Max Gust: {wind_kmh} km/h, Asphalt Temp: {asphalt_temp}°C, "
            f"Corridor Risk Score: {risk_score}/100, Language: {norm_lang}.\n"
            "Provide tailored logistics safety advice now."
        )
        for model in GEMINI_MODELS:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": sys_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 450,
                    "response_mime_type": "application/json",
                },
            }
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        parsed = json.loads(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
                        if parsed.get("operational_guidance"):
                            op_guidance = parsed["operational_guidance"]
                        if parsed.get("ai_insights"):
                            ai_insights = parsed["ai_insights"]
                        if parsed.get("vehicle_recommendations"):
                            v_recs = parsed["vehicle_recommendations"][:3]
                        return CorridorRadarResponse(
                            city=city,
                            origin=origin,
                            destination=destination,
                            vehicle_type=v_type,
                            corridor_risk_score=risk_score,
                            status_level=status_level,
                            estimated_delay_mins=delay_mins,
                            surface_runoff_mm=precip_mm,
                            max_wind_gust_kmh=wind_kmh,
                            asphalt_temp_c=asphalt_temp,
                            operational_guidance=op_guidance,
                            segments=segments,
                            vehicle_recommendations=v_recs,
                            ai_insights=ai_insights,
                            provider=f"gemini ({model})",
                        )
            except Exception as e:
                logger.debug("Gemini corridor radar failed with model %s: %s", model, e)
                continue

    return CorridorRadarResponse(
        city=city,
        origin=origin,
        destination=destination,
        vehicle_type=v_type,
        corridor_risk_score=risk_score,
        status_level=status_level,
        estimated_delay_mins=delay_mins,
        surface_runoff_mm=precip_mm,
        max_wind_gust_kmh=wind_kmh,
        asphalt_temp_c=asphalt_temp,
        operational_guidance=op_guidance,
        segments=segments,
        vehicle_recommendations=v_recs,
        ai_insights=ai_insights,
        provider="deterministic_engine",
    )


# ============================================================================
# Implementation: 2. Tourist Itinerary Optimizer
# ============================================================================

async def optimize_tourist_schedule(
    req: TouristOptimizerRequest,
    current_weather: Optional[Dict[str, Any]] = None,
) -> TouristOptimizerResponse:
    city = req.city.strip() or "Delhi"
    t_date = req.travel_date or "Today"
    pace = req.pace.lower()
    norm_lang = (req.language or "en").strip().lower()

    metrics = current_weather or {}
    t_max = float(metrics.get("temp_max") or metrics.get("temperature_max") or 34.0)
    t_min = float(metrics.get("temp_min") or metrics.get("temperature_min") or 24.0)
    precip_mm = float(metrics.get("precipitation_mm") or metrics.get("runoff_mm") or 0.5)
    aqi = int(metrics.get("aqi") or 85)

    comfort_score = 78
    if t_max > 38:
        comfort_score -= 25
    elif t_max > 34:
        comfort_score -= 12
    if aqi > 200:
        comfort_score -= 20
    elif aqi > 120:
        comfort_score -= 10
    if precip_mm > 20:
        comfort_score -= 25
    comfort_score = max(20, min(95, comfort_score))

    verdict = (
        "IDEAL SIGHTSEEING" if comfort_score >= 75
        else "MODERATE CONDITIONS" if comfort_score >= 50
        else "HIGH THERMAL/RAIN CAUTION"
    )

    weather_summary = (
        f"Temperatures range from {round(t_min)}°C to {round(t_max)}°C with an overall comfort index of {comfort_score}/100. "
        f"Air quality is rated at AQI {aqi}. Plan outdoor excursions for early morning and evening."
    )

    itinerary_slots = [
        ItineraryHourSlot(
            time_window="06:30 AM - 09:30 AM",
            period_label="Morning Fresh Air Window",
            activity_title=f"Outdoor Heritage Walk & Botanical Exploration in {city}",
            venue_type="OUTDOOR",
            weather_suitability="EXCELLENT",
            temp_c=round(t_min + 2.0, 1),
            uv_index=2,
            aqi_category="MODERATE" if aqi > 100 else "GOOD",
            recommendation="Take advantage of cooler ambient temperatures and lower solar radiation before ground heating accelerates.",
        ),
        ItineraryHourSlot(
            time_window="10:00 AM - 01:00 PM",
            period_label="Mid-Morning Transition",
            activity_title=f"Cultural Art Galleries & Covered Architectural Monuments in {city}",
            venue_type="SHELTERED",
            weather_suitability="ACCEPTABLE",
            temp_c=round(t_min + 6.5, 1),
            uv_index=6,
            aqi_category="MODERATE",
            recommendation="Wear UV-protective sunglasses and carry a refillable thermal water flask.",
        ),
        ItineraryHourSlot(
            time_window="01:30 PM - 04:00 PM",
            period_label="Midday Thermal Peak Sanctuary",
            activity_title=f"Air-Conditioned Museum Tour, Science Center, or Regional Dining",
            venue_type="INDOOR",
            weather_suitability="UNFAVORABLE",
            temp_c=t_max,
            uv_index=9,
            aqi_category="MODERATE",
            recommendation=f"Peak solar heating reaches {t_max}°C. Stay indoors in air-conditioned environments to prevent heat fatigue.",
        ),
        ItineraryHourSlot(
            time_window="04:30 PM - 07:30 PM",
            period_label="Golden Hour & Sunset Riverfront",
            activity_title=f"Bazaar Exploration, Street Food Tasting & Panoramic Sunset Points in {city}",
            venue_type="OUTDOOR",
            weather_suitability="EXCELLENT",
            temp_c=round(t_max - 3.5, 1),
            uv_index=3,
            aqi_category="MODERATE",
            recommendation="Atmospheric boundary layer begins cooling. Comfortable window for evening street photography.",
        ),
    ]

    packing = [
        "Lightweight, breathable cotton or linen clothing in light shades.",
        "UV400 polarized sunglasses and broad-brim sun hat.",
        "Vacuum-insulated thermal water bottle to maintain chilled hydration.",
        "Compact travel umbrella or lightweight rain poncho.",
        "Electrolyte sachets (ORS) and broad-spectrum SPF 50 sunscreen.",
    ]

    hydrated_spots = [
        "Municipal Heritage Center air-conditioned visitors lounge",
        "Public shaded gazebos with clean filtered drinking water kiosks",
        "Botanical garden shaded pergolas with mist cooling stations",
    ]

    concierge_briefing = (
        f"For your tour in {city}, ClimaCore has structured your daily itinerary to shield you from peak midday "
        f"thermal stress ({t_max}°C) while maximizing your enjoyment of outdoor historical sights during "
        f"the golden early morning and sunset hours."
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        sys_prompt = (
            "You are ClimaCore's Smart Tourist Concierge AI. "
            "Generate personalized, hourly sightseeing advice adapted to today's climate conditions in an Indian city.\n"
            "Respond ONLY with a valid JSON object matching:\n"
            "{\n"
            '  "weather_summary": "string",\n'
            '  "ai_concierge_briefing": "string",\n'
            '  "packing_checklist": ["string", "string", "string", "string"],\n'
            '  "local_hydrated_spots": ["string", "string", "string"]\n'
            "}"
        )
        user_prompt = (
            f"City: {city}, Date: {t_date}, Pace: {pace}, Interests: {', '.join(req.interests)}.\n"
            f"Weather: Max {t_max}°C, Min {t_min}°C, AQI: {aqi}, Rain: {precip_mm} mm, Language: {norm_lang}.\n"
            "Provide tailored sightseeing advice."
        )
        for model in GEMINI_MODELS:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": sys_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 450,
                    "response_mime_type": "application/json",
                },
            }
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        parsed = json.loads(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
                        if parsed.get("weather_summary"):
                            weather_summary = parsed["weather_summary"]
                        if parsed.get("ai_concierge_briefing"):
                            concierge_briefing = parsed["ai_concierge_briefing"]
                        if parsed.get("packing_checklist"):
                            packing = parsed["packing_checklist"][:5]
                        if parsed.get("local_hydrated_spots"):
                            hydrated_spots = parsed["local_hydrated_spots"][:3]
                        return TouristOptimizerResponse(
                            city=city,
                            travel_date=t_date,
                            pace=pace,
                            comfort_score=comfort_score,
                            overall_verdict=verdict,
                            weather_summary=weather_summary,
                            itinerary=itinerary_slots,
                            packing_checklist=packing,
                            local_hydrated_spots=hydrated_spots,
                            ai_concierge_briefing=concierge_briefing,
                            provider=f"gemini ({model})",
                        )
            except Exception as e:
                logger.debug("Gemini tourist optimizer failed with model %s: %s", model, e)
                continue

    return TouristOptimizerResponse(
        city=city,
        travel_date=t_date,
        pace=pace,
        comfort_score=comfort_score,
        overall_verdict=verdict,
        weather_summary=weather_summary,
        itinerary=itinerary_slots,
        packing_checklist=packing,
        local_hydrated_spots=hydrated_spots,
        ai_concierge_briefing=concierge_briefing,
        provider="deterministic_engine",
    )


# ============================================================================
# Implementation: 3. Clean Air Athletic Window
# ============================================================================

async def calculate_clean_air_workout(
    req: CleanAirWorkoutRequest,
    current_weather: Optional[Dict[str, Any]] = None,
) -> CleanAirWorkoutResponse:
    city = req.city.strip() or "Delhi"
    act_type = req.activity_type.lower()
    intensity = req.intensity.lower()
    norm_lang = (req.language or "en").strip().lower()

    metrics = current_weather or {}
    aqi = int(metrics.get("aqi") or 118)
    pm25 = float(metrics.get("pm25") or 46.0)
    wind_kmh = float(metrics.get("wind_speed_kmh") or 14.0)
    wbgt = float(metrics.get("wbgt") or 27.5)

    # Inversion layer status: early morning inversions trap pollutants near ground
    inversion_active = aqi > 140 or (wind_kmh < 8.0)
    inversion_status = "SURFACE_TRAPPED" if inversion_active else "DISPERSIVE_VENTILATED"

    # Athletic score (0 to 100)
    athletic_score = 80
    if aqi > 250:
        athletic_score -= 50
    elif aqi > 150:
        athletic_score -= 30
    elif aqi > 90:
        athletic_score -= 15

    if wbgt > 32:
        athletic_score -= 35
    elif wbgt > 29:
        athletic_score -= 20
    elif wbgt > 27:
        athletic_score -= 8

    athletic_score = max(15, min(95, athletic_score))

    status_level = (
        "OPTIMAL" if athletic_score >= 75
        else "ACCEPTABLE" if athletic_score >= 55
        else "CAUTION" if athletic_score >= 35
        else "INDOORS_ONLY"
    )

    recommended_window = (
        "05:15 PM - 06:45 PM" if inversion_active
        else "06:00 AM - 07:30 AM"
    )

    hourly_slots = [
        AirQualityHourEstimate(
            time_slot="06:00 AM - 07:30 AM",
            aqi=int(aqi * 1.25) if inversion_active else int(aqi * 0.9),
            pm25=round(pm25 * 1.2, 1) if inversion_active else round(pm25 * 0.9, 1),
            inversion_trapping=inversion_active,
            athletic_score=max(20, athletic_score - 15) if inversion_active else min(95, athletic_score + 10),
            is_optimal=not inversion_active,
        ),
        AirQualityHourEstimate(
            time_slot="12:00 PM - 01:30 PM",
            aqi=int(aqi * 0.85),
            pm25=round(pm25 * 0.85, 1),
            inversion_trapping=False,
            athletic_score=max(20, athletic_score - 25),  # High thermal stress
            is_optimal=False,
        ),
        AirQualityHourEstimate(
            time_slot="05:15 PM - 06:45 PM",
            aqi=int(aqi * 0.95),
            pm25=round(pm25 * 0.95, 1),
            inversion_trapping=False,
            athletic_score=min(95, athletic_score + 5),
            is_optimal=inversion_active,
        ),
        AirQualityHourEstimate(
            time_slot="08:00 PM - 09:30 PM",
            aqi=int(aqi * 1.15),
            pm25=round(pm25 * 1.15, 1),
            inversion_trapping=True,
            athletic_score=max(20, athletic_score - 10),
            is_optimal=False,
        ),
    ]

    coaching = (
        f"For {act_type} at {intensity} intensity in {city}, today's atmospheric profile is rated {status_level}. "
        f"The best outdoor athletic window is {recommended_window} when thermal mixing clears surface particulate build-up."
    )

    precautions = [
        "Avoid running or cycling within 50 meters of high-density arterial vehicular corridors.",
        f"Target heart rate zones should be adjusted 5 to 10 BPM lower to compensate for {wbgt}°C WBGT thermal strain.",
        "Perform deep nasal breathing during warm-up; switch to indoor training if throat tightness occurs.",
        "Hydrate with 200ml cold water or electrolyte drink every 20 minutes of sustained exertion.",
    ]

    indoor_alt = (
        f"If ground-level air stagnation persists, substitute outdoor {act_type} with a 45-minute indoor treadmill tempo "
        "session or functional bodyweight HIIT in an air-filtered environment."
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        sys_prompt = (
            "You are ClimaCore's Clean Air Athletic Coach AI. "
            "Generate actionable outdoor fitness recommendations analyzing atmospheric inversion and particulate metrics.\n"
            "Respond ONLY with a valid JSON object matching:\n"
            "{\n"
            '  "coaching_guidance": "string",\n'
            '  "respiratory_precautions": ["string", "string", "string", "string"],\n'
            '  "indoor_alternative": "string"\n'
            "}"
        )
        user_prompt = (
            f"City: {city}, Activity: {act_type}, Intensity: {intensity}.\n"
            f"AQI: {aqi}, PM2.5: {pm25} µg/m³, WBGT: {wbgt}°C, Wind: {wind_kmh} km/h, Inversion: {inversion_status}, Language: {norm_lang}.\n"
            "Provide athletic guidance now."
        )
        for model in GEMINI_MODELS:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
            payload = {
                "system_instruction": {"parts": [{"text": sys_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 450,
                    "response_mime_type": "application/json",
                },
            }
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        parsed = json.loads(resp.json()["candidates"][0]["content"]["parts"][0]["text"])
                        if parsed.get("coaching_guidance"):
                            coaching = parsed["coaching_guidance"]
                        if parsed.get("respiratory_precautions"):
                            precautions = parsed["respiratory_precautions"][:4]
                        if parsed.get("indoor_alternative"):
                            indoor_alt = parsed["indoor_alternative"]
                        return CleanAirWorkoutResponse(
                            city=city,
                            activity_type=act_type,
                            athletic_suitability_score=athletic_score,
                            status_level=status_level,
                            recommended_window=recommended_window,
                            inversion_status=inversion_status,
                            current_aqi=aqi,
                            current_pm25=pm25,
                            wbgt_thermal_strain_c=wbgt,
                            hourly_forecast=hourly_slots,
                            coaching_guidance=coaching,
                            respiratory_precautions=precautions,
                            indoor_alternative=indoor_alt,
                            provider=f"gemini ({model})",
                        )
            except Exception as e:
                logger.debug("Gemini clean air workout failed with model %s: %s", model, e)
                continue

    return CleanAirWorkoutResponse(
        city=city,
        activity_type=act_type,
        athletic_suitability_score=athletic_score,
        status_level=status_level,
        recommended_window=recommended_window,
        inversion_status=inversion_status,
        current_aqi=aqi,
        current_pm25=pm25,
        wbgt_thermal_strain_c=wbgt,
        hourly_forecast=hourly_slots,
        coaching_guidance=coaching,
        respiratory_precautions=precautions,
        indoor_alternative=indoor_alt,
        provider="deterministic_engine",
    )
