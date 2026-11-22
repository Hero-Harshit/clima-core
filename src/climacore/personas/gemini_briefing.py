"""
Gemini-powered Dynamic Persona Briefing and Actionable Advisory Engine.
Synthesizes hyper-personalized, occupation-specific operational guidance,
hydration protocols, rest gaps, equipment checks, and hazard countermeasures
using Google Gemini API with fallback to deterministic domain synthesis.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import httpx
from climacore.personas.briefing_translations import (
    get_localized_persona_title,
    get_localized_briefing_content,
)

logger = logging.getLogger("climacore.personas.gemini_briefing")

# Prioritize fastest lightweight model with active quota
GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-3.8-flash",
]
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# In-memory LRU cache: (city, persona_id, date) -> Dict[str, Any]
_BRIEFING_CACHE: Dict[str, Dict[str, Any]] = {}


PERSONA_TITLES: Dict[str, str] = {
    "gig_worker": "Gig Worker & Outdoor Labor",
    "gig_worker_labor": "Gig Worker & Outdoor Labor",
    "farmer": "Farmer & Agriculture",
    "farmer_kisan": "Farmer & Agriculture",
    "student": "Student & Parent",
    "student_parent": "Student & Parent",
    "commuter": "Daily Commuter",
    "daily_commuter": "Daily Commuter",
    "healthcare": "Healthcare & Vulnerable",
    "healthcare_vulnerable": "Healthcare & Vulnerable",
    "city_official": "City Official & Infrastructure",
    "city_official_infrastructure": "City Official & Infrastructure",
    "maritime": "Maritime & Coastal",
    "maritime_coastal": "Maritime & Coastal",
    "industry": "Industry & Manufacturing",
    "industry_manufacturing": "Industry & Manufacturing",
    "aviation": "Aviation & Airport Operations",
    "aviation_airport": "Aviation & Airport Operations",
    "water": "Water Utilities",
    "water_utilities": "Water Utilities",
    "insurance": "Insurance & Actuarial",
    "insurance_actuarial": "Insurance & Actuarial",
    "cold_chain": "Cold Chain Logistics",
    "cold_chain_logistics": "Cold Chain Logistics",
    "tourism": "Tourism & Hospitality",
    "tourism_hospitality": "Tourism & Hospitality",
}


def _get_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def generate_deterministic_briefing(
    city: str,
    persona_id: str,
    metrics: Dict[str, Any],
    language: str = "en",
) -> Dict[str, Any]:
    """
    Produce rich, multi-point, occupation-specific operational guidance and advisories
    strictly calibrated to verified weather metrics (temperature, WBGT, rain, humidity, AQI).
    Ensures the UI ALWAYS displays 4 rich, actionable advisories including hydration,
    gap intervals, PPE, and operational timing. Supports full localization into 7 languages.
    """
    norm_lang = (language or "en").strip().lower()
    t_max = float(metrics.get("temp_max") or 32.0)
    wbgt = float(metrics.get("wbgt") or (t_max * 0.85))
    precip = float(metrics.get("precipitation_mm") or 0.0)
    rain_prob = float(metrics.get("rain_probability") or 0.0)
    pm25 = float(metrics.get("pm2_5") or 45.0)

    p_norm = persona_id.lower().replace("-", "_")
    p_title = get_localized_persona_title(persona_id, language=norm_lang) if norm_lang != "en" else PERSONA_TITLES.get(p_norm, "Outdoor Professional")

    advisories: List[str] = []
    operational_guidance: str = ""

    is_hot = wbgt >= 30.0 or t_max >= 35.0
    is_extreme_hot = wbgt >= 33.0 or t_max >= 40.0
    is_rainy = precip >= 15.0 or rain_prob >= 50.0

    if norm_lang != "en":
        cond = "extreme_hot" if is_extreme_hot else "hot" if is_hot else "rainy" if is_rainy else "default"
        operational_guidance, advisories = get_localized_briefing_content(persona_id, cond, language=norm_lang)
    elif "gig_worker" in p_norm or "labor" in p_norm:
        if is_extreme_hot:
            operational_guidance = "Mandate 45 minutes of shaded rest per hour between 12:00 PM and 4:00 PM. Platforms must suspend delivery SLA penalties during peak heat."
            advisories = [
                "Hydration Gap: Drink at least 250–300 ml of electrolyte-rich water (ORS) every 20 minutes; do not wait until thirst sets in.",
                "Shaded Rest Intervals: Take a mandatory 15-minute cooling break in shaded or air-conditioned rest hubs after every 45 minutes of riding.",
                "Protective Gear: Wear breathable light-colored UV-blocking arm sleeves, polarized visor, and cooling neck gaiters to combat solar irradiance.",
                "Route Safety: Watch for softened asphalt on arterial roads and plan rest stops at verified public water kiosks."
            ]
        elif is_hot:
            operational_guidance = "Mandate 30 minutes of shaded rest per hour between 12:30 PM and 3:30 PM to prevent cumulative thermal strain."
            advisories = [
                "Hydration Timing: Consume at least 750ml–1L per hour of water supplemented with electrolytes; set regular 30-minute hydration reminders.",
                "Work-Rest Gap: Schedule delivery clusters in the early morning and late afternoon, avoiding peak sun between 1:00 PM and 3:00 PM.",
                "Rider Attire: Use moisture-wicking undershirts, well-ventilated helmets, and sunscreen with SPF 50+ on exposed neck and forearms.",
                "Vehicle Checks: Check tire pressure in the morning as highway road surface temperatures can trigger sudden blowout risks."
            ]
        elif is_rainy:
            operational_guidance = "Exercise extreme transit caution. Limit delivery speed to 30 km/h and avoid known flood-prone underpasses."
            advisories = [
                "Waterproofing Protocol: Seal smartphones, power banks, and billing devices in double-layered waterproof pouches before dispatch.",
                "Braking Safety: Increase following distance to at least 4 vehicle lengths to compensate for reduced wet asphalt braking friction.",
                "Hydration & Nutrition: Maintain regular fluid intake with warm tea or water; maintain dry footwear between delivery shifts.",
                "Flood Navigation: Divert around standing water exceeding 15cm to prevent engine stalling and hidden open-drain hazards."
            ]
        else:
            operational_guidance = "Favorable ambient operational conditions. Maintain standard delivery cadence and steady hydration."
            advisories = [
                "Steady Hydration: Drink 200–250ml of clean water every 45 minutes to maintain optimal alertness throughout your shift.",
                "Rest Gaps: Take short 10-minute relaxation intervals between delivery batches to stretch and avoid physical fatigue.",
                "Equipment Check: Ensure vehicle mirrors, headlights, and battery charging cables are securely inspected before peak hours.",
                "Corridor Awareness: Stay mindful of localized traffic stagnation and air quality along major ring roads."
            ]

    elif "farmer" in p_norm or "kisan" in p_norm:
        if is_hot:
            operational_guidance = "Shift field labor to split shifts (6:00 AM–10:30 AM and 4:30 PM–7:00 PM). Avoid heavy manual tillage during peak midday sun."
            advisories = [
                "Field Hydration: Drink cool water infused with lime or electrolytes every 30 minutes; carry at least 4 liters of water to the field.",
                "Spraying Window: Suspend chemical spraying between 11:00 AM and 4:00 PM to prevent rapid evaporation and inhalation risks.",
                "Soil Moisture: Prioritize evening or drip irrigation to minimize high evaporative losses and conserve groundwater.",
                "Livestock Care: Provide shaded shelter, continuous fresh drinking water, and ventilation for farm animals."
            ]
        elif is_rainy:
            operational_guidance = "Postpone chemical spraying and fertilizer broadcasting to avoid wash-off. Open field drainage channels immediately."
            advisories = [
                "Drainage Management: Clear bund outlets and check lateral furrows to prevent root-zone waterlogging in standing crops.",
                "Machinery Safety: Avoid operating heavy tractors on water-saturated clay soils to prevent severe soil compaction.",
                "Field Hydration & Gear: Wear waterproof gumboots and rain protection; maintain clean drinking water to prevent waterborne illness.",
                "Post-Harvest Storage: Move harvested produce to elevated, leak-proof tarpaulin-covered sheds immediately."
            ]
        else:
            operational_guidance = "Optimal weather window for scheduled agricultural tasks, weeding, and standard agronomic management."
            advisories = [
                "Regular Hydration: Maintain consistent water intake during weeding and field scouting activities.",
                "Irrigation Timing: Apply measured irrigation during early morning hours to maximize soil absorption.",
                "Pest Surveillance: Conduct morning crop pest inspections while atmospheric conditions remain calm and clear.",
                "Tool Maintenance: Sharpen and inspect hand implements and drip lines during afternoon lull intervals."
            ]

    elif "student" in p_norm or "parent" in p_norm:
        if is_hot:
            operational_guidance = "Schools should restrict outdoor physical education and assemblies between 10:30 AM and 3:30 PM. Ensure active classroom ventilation."
            advisories = [
                "Hydration Bottle: Ensure students carry a refillable water bottle with a minimum target of 200ml per 45-minute class interval.",
                "Sun Protection: Encourage wearing wide-brim hats, cotton caps, and applying sunscreen during the school commute.",
                "Classroom Gaps: Mandate 5-minute water breaks between lecture periods to ensure steady child hydration.",
                "Commute Safety: School buses and vans must keep windows cracked or use air-conditioning to prevent heat buildup inside vehicles."
            ]
        elif is_rainy:
            operational_guidance = "Allow flexible morning school arrival times to accommodate monsoon traffic delays and localized waterlogging."
            advisories = [
                "Rain Gear: Equip students with sturdy umbrellas, hooded raincoats, and water-resistant school bag covers.",
                "Safe Footwear: Wear non-skid waterproof shoes to avoid slippery staircases and corridor tile accidents.",
                "Clean Drinking Water: Drink only boiled or filtered water from home; avoid roadside open beverage vendors.",
                "Commute Monitoring: Parents should track real-time school transit updates and avoid low-lying school pickup routes."
            ]
        else:
            operational_guidance = "Ideal conditions for academic activities, outdoor sports, and regular student schedules."
            advisories = [
                "Routine Hydration: Maintain 4 to 6 glasses of water intake throughout the school day.",
                "Outdoor Activity: Favorable weather for school field sports and extracurricular physical development.",
                "Nutritious Gaps: Include fresh fruits and balanced snacks during recess intervals to maintain energy.",
                "Commute Ease: Standard travel conditions across all school transit and walking corridors."
            ]

    elif "commuter" in p_norm:
        operational_guidance = "Plan travel with a 15-minute buffer during peak hours. Carry personal hydration and monitor transport alerts."
        advisories = [
            "Transit Hydration: Carry a personal insulated water flask and sip steadily while waiting on metro platforms or bus stops.",
            "Travel Gap Planning: Avoid rush-hour overcrowding during peak afternoon heat; opt for early or late commute windows where feasible.",
            "Footwear & Grip: Wear slip-resistant footwear to navigate metro station stairs, wet footbridges, and bus footboards safely.",
            "Air Quality Shield: Wear a protective mask during transit through congested road intersections and diesel bus corridors."
        ]

    elif "healthcare" in p_norm or "vulnerable" in p_norm:
        operational_guidance = "Activate hospital heat and respiratory surveillance protocols. Ensure cooling wards and medical oxygen backups are verified."
        advisories = [
            "Patient Fluid Monitoring: Monitor elderly and pediatric fluid intake closely; administer oral rehydration solutions at scheduled intervals.",
            "Medicine Cold Chain: Check pharmacy refrigeration daily to ensure temperature-sensitive drugs remain strictly between 2°C and 8°C.",
            "Triage Preparedness: Screen outpatient walk-ins immediately for early signs of heat exhaustion, dizziness, or asthma aggravation.",
            "Indoor Air Quality: Run indoor HEPA purifiers in critical care wards to minimize ambient particulate penetration."
        ]

    elif "city_official" in p_norm or "infrastructure" in p_norm:
        operational_guidance = "Deploy municipal water tankers to water-stressed informal settlements and place emergency pump crews on high alert."
        advisories = [
            "Cooling Centers: Open public air-conditioned community centers and shaded resting zones across major transit hubs.",
            "Drainage Clearance: Inspect stormwater pump stations and clear road catch-basins to prevent flash waterlogging.",
            "Grid Coordination: Liaise with power distribution companies to manage transformer cooling loads during peak afternoon surges.",
            "Public Alerts: Broadcast hourly weather advisories and safety bulletins across digital road signs and local media."
        ]

    elif "maritime" in p_norm or "coastal" in p_norm:
        operational_guidance = "Monitor coastal wave swell and wind gusts. Instruct small fishing crafts to stay within safe harbor perimeters."
        advisories = [
            "Crew Hydration: Mandate 15-minute water breaks in shaded wheelhouses for deckhands every hour during high solar exposure.",
            "Vessel Mooring: Double check mooring lines and cargo lashings against anticipated tidal surges and gusty squalls.",
            "Navigational Watch: Maintain continuous VHF Channel 16 radio watch for localized coastal weather warnings.",
            "Safety Equipment: Ensure all crew members wear life jackets and non-slip deck shoes during harbor operations."
        ]

    elif "industry" in p_norm or "manufacturing" in p_norm:
        operational_guidance = "Implement heat-stress rotation for boiler and furnace operators. Monitor industrial chiller plant loads."
        advisories = [
            "Cooling Gaps: Mandate 20-minute shaded/air-conditioned recovery intervals for heavy equipment and furnace line workers.",
            "Electrolyte Stations: Install accessible oral rehydration dispensers at all plant work cells and assembly bays.",
            "Equipment Monitoring: Inspect hydraulic fluid temperatures and transformer oil levels to avoid thermal shutdowns.",
            "Chemical Storage: Ensure volatile solvent tanks and pressurized gas cylinders remain shielded from direct sunlight."
        ]

    elif "aviation" in p_norm or "airport" in p_norm:
        operational_guidance = "Rotate ramp and tarmac personnel frequently. Monitor high density altitude impacts on aircraft takeoff performance."
        advisories = [
            "Tarmac Worker Gaps: Provide air-conditioned rest vans with mandatory 15-minute hydration gaps between aircraft turnarounds.",
            "Surface Protection: Tarmac surface heat requires high-durability heat-resistant work boots and polarized eye protection.",
            "Baggage Crew Safety: Enforce buddy systems and heat-exhaustion symptom checks during heavy baggage handling shifts.",
            "Cabin Pre-Cooling: Maximize ground power unit usage for cabin air-conditioning to reduce auxiliary engine emissions."
        ]

    elif "water" in p_norm or "utilities" in p_norm:
        operational_guidance = "Optimize water treatment chemical dosing and maintain distribution pipe pressure against peak summer drawdowns."
        advisories = [
            "Field Crew Safety: Schedule pipeline maintenance and leak detection in early morning hours with mandatory water breaks.",
            "Turbidity Surveillance: Monitor raw intake water turbidity and adjust alum/chlorine dosing to meet BIS 10500 standards.",
            "Pumping Efficiency: Shift high-power raw water intake pumping to off-peak electrical tariff hours where possible.",
            "Contingency Reserves: Keep mobile water tankers on standby for rapid deployment to low-pressure tail-end zones."
        ]

    elif "insurance" in p_norm or "actuarial" in p_norm:
        operational_guidance = "Track parametric weather triggers and monitor real-time crop, heat, and flash-flood claim accumulation."
        advisories = [
            "Index Monitoring: Track temperature and rainfall thresholds for automated parametric insurance policy activations.",
            "Claim Readiness: Pre-alert field adjusters and automated claim triage pipelines in zones experiencing compound stress.",
            "Risk Exposure: Cross-reference spatial flood inundation models with commercial property concentration in low-lying sectors.",
            "Policyholder Guidance: Issue proactive SMS loss-mitigation advisories to insured agricultural and commercial clients."
        ]

    elif "cold_chain" in p_norm or "logistics" in p_norm:
        operational_guidance = "Enforce pre-cooling of reefer trucks and minimize loading dock door open duration to under 90 seconds."
        advisories = [
            "Reefer Setpoint: Lower transport refrigeration setpoint by 1.0°C during peak ambient heat to buffer temperature excursions.",
            "Dock Door Gaps: Implement strict door-seal protocols during pallet transfer to prevent ambient warm air infiltration.",
            "Driver Hydration: Provide drivers with thermal water jugs and schedule deliveries to avoid midday highway traffic bottlenecks.",
            "Telemetry Alarms: Set automated SMS alerts for refrigeration compressor power failures or cargo temperature rises."
        ]

    elif "tourism" in p_norm or "hospitality" in p_norm:
        operational_guidance = "Reschedule outdoor heritage walking tours to early morning. Provide air-conditioned guest transit and welcome refreshments."
        advisories = [
            "Guest Hydration: Set up complimentary chilled electrolyte and infused water stations at hotel lobbies and tour departure gates.",
            "Tour Timing Gaps: Advise tourists to complete outdoor monument visits by 10:30 AM or after 4:30 PM, visiting indoor museums midday.",
            "Outdoor Amenities: Activate patio misting fans, pool umbrellas, and shaded cabanas across hotel outdoor dining spaces.",
            "Staff Protection: Ensure doormen, bellhops, and pool lifeguards rotate shifts every 45 minutes with shaded rest intervals."
        ]

    else:
        operational_guidance = f"Tailored occupational recommendations for {p_title} under current ambient conditions."
        advisories = [
            "Hydration Protocol: Drink 250ml of water every 30 to 45 minutes; do not wait until thirst signals dehydration.",
            "Work-Rest Intervals: Take 10-15 minute rest intervals in shaded areas during peak ambient temperatures.",
            "Protective Apparel: Wear breathable cotton clothing, UV sunglasses, and head protection during outdoor activities.",
            "Environmental Alert: Keep updated with real-time local weather forecasts and adjust daily tasks accordingly."
        ]

    # Calculate calibrated workability score
    heat_deduction = max(0.0, (wbgt - 24.0) * 4.5)
    rain_deduction = min(35.0, precip * 0.8)
    pm25_deduction = max(0.0, (pm25 - 60.0) * 0.15)
    workability = max(20, min(100, int(round(100.0 - heat_deduction - rain_deduction - pm25_deduction))))

    status_level = "CRITICAL" if workability < 40 else "HIGH_ALERT" if workability < 60 else "CAUTION" if workability < 80 else "NORMAL"

    return {
        "city": city,
        "persona_id": persona_id,
        "persona_title": p_title,
        "operational_guidance": operational_guidance,
        "advisories": advisories,
        "workability_score": workability,
        "status_level": status_level,
        "provider": "climacore-deterministic-synthesis",
        "language": norm_lang,
    }


async def generate_gemini_persona_briefing(
    city: str,
    persona_id: str,
    metrics: Dict[str, Any],
    date_str: str = "",
    language: str = "en",
) -> Dict[str, Any]:
    """
    Generate tailored, rich operational guidance and 4 persona-specific advisories
    using Google Gemini API with fallback to deterministic domain synthesis.
    Supports localization across all 7 languages.
    """
    norm_lang = (language or "en").strip().lower()
    city_str = (city or "").strip().lower()
    persona_str = (persona_id or "general").strip().lower()
    date_clean = (date_str or "").strip()
    cache_key = f"{city_str}:{persona_str}:{date_clean}:{norm_lang}"
    if cache_key in _BRIEFING_CACHE:
        return _BRIEFING_CACHE[cache_key]

    api_key = _get_api_key()
    p_norm = persona_id.lower().replace("-", "_") if persona_id else "general"
    p_title = get_localized_persona_title(persona_id, language=norm_lang) if norm_lang != "en" else PERSONA_TITLES.get(p_norm, persona_id.replace("_", " ").title() if persona_id else "General")

    t_max = metrics.get("temp_max", 32.0)
    t_min = metrics.get("temp_min", 24.0)
    feels_like = metrics.get("feels_like", t_max)
    wbgt = metrics.get("wbgt", t_max * 0.85)
    precip = metrics.get("precipitation_mm", 0.0)
    rain_prob = metrics.get("rain_probability", 0.0)
    humidity = metrics.get("relative_humidity", 65.0)
    pm25 = metrics.get("pm2_5", 40.0)
    wind_kmh = metrics.get("wind_speed_kmh", 12.0)

    if not api_key:
        logger.info("No Gemini API key found, generating deterministic persona briefing.")
        result = generate_deterministic_briefing(city, persona_id, metrics, language=norm_lang)
        _BRIEFING_CACHE[cache_key] = result
        return result

    lang_directive = ""
    LANGUAGE_DIRECTIVES = {
        "hi": "MANDATORY: Generate 'operational_guidance' and all 'advisories' in Hindi (हिन्दी) in Devanagari script. Ensure numeric quantities and units (°C, mm, µg/m³, km/h) are maintained.",
        "bn": "MANDATORY: Generate 'operational_guidance' and all 'advisories' in Bengali (বাংলা) in Bengali script. Ensure numeric quantities and units (°C, mm, µg/m³, km/h) are maintained.",
        "ta": "MANDATORY: Generate 'operational_guidance' and all 'advisories' in Tamil (தமிழ்) in Tamil script. Ensure numeric quantities and units (°C, mm, µg/m³, km/h) are maintained.",
        "te": "MANDATORY: Generate 'operational_guidance' and all 'advisories' in Telugu (తెలుగు) in Telugu script. Ensure numeric quantities and units (°C, mm, µg/m³, km/h) are maintained.",
        "mr": "MANDATORY: Generate 'operational_guidance' and all 'advisories' in Marathi (मराठी) in Devanagari script. Ensure numeric quantities and units (°C, mm, µg/m³, km/h) are maintained.",
        "gu": "MANDATORY: Generate 'operational_guidance' and all 'advisories' in Gujarati (ગુજરાતી) in Gujarati script. Ensure numeric quantities and units (°C, mm, µg/m³, km/h) are maintained.",
    }
    if norm_lang != "en" and norm_lang in LANGUAGE_DIRECTIVES:
        lang_directive = f"\n7. {LANGUAGE_DIRECTIVES[norm_lang]}"

    system_prompt = (
        "You are ClimaCore's Occupational Environmental Safety Intelligence Engine. "
        "Your task is to generate actionable, highly practical operational guidance and 4 sector advisories "
        "tailored specifically to the worker's exact occupation and today's environmental metrics in an Indian city.\n"
        "Rules:\n"
        "1. Advisory 1 MUST be an exact Hydration Protocol with specific time intervals/gaps (e.g. 'drink 250ml every 20 minutes', electrolytes, ORS).\n"
        "2. Advisory 2 MUST be specific Shift & Gap Planning (shaded rest intervals, peak sun pause windows, shift splits).\n"
        "3. Advisory 3 MUST be Protective Equipment & Attire (PPE, clothing, UV gear, cooling neck gaiter, rainwear, dust masks).\n"
        "4. Advisory 4 MUST be Workplace, Route & Hazard Countermeasures (slick roads, equipment overheating, transit safety).\n"
        "5. 'operational_guidance' must be a concise, authoritative 1-2 sentence directive.\n"
        "6. Provide a calculated 'workability_score' (integer 0-100) and 'status_level' ('NORMAL', 'CAUTION', 'HIGH_ALERT', or 'CRITICAL')."
        f"{lang_directive}\n"
        "Return ONLY a valid JSON object matching the schema:\n"
        "{\n"
        '  "operational_guidance": "string",\n'
        '  "advisories": ["string", "string", "string", "string"],\n'
        '  "workability_score": number,\n'
        '  "status_level": "NORMAL" | "CAUTION" | "HIGH_ALERT" | "CRITICAL"\n'
        "}"
    )

    user_prompt = (
        f"City: {city}\n"
        f"Occupation/Persona: {p_title} ({persona_id})\n"
        f"Weather Metrics: Max Temp: {t_max}°C, Min Temp: {t_min}°C, Feels Like: {feels_like}°C, "
        f"WBGT Labor Index: {wbgt}°C, Humidity: {humidity}%, Rain: {precip}mm (Prob: {rain_prob}%), "
        f"PM2.5: {pm25} µg/m³, Wind: {wind_kmh} km/h.\n"
        "Generate the structured JSON briefing now."
    )

    # Try Gemini models in priority order
    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE_URL}/{model}:generateContent?key={api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 600,
                "response_mime_type": "application/json"
            }
        }

        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        parsed = json.loads(raw_text)
                        advisories_list = parsed.get("advisories", [])
                        op_guidance = parsed.get("operational_guidance", "")
                        if advisories_list and len(advisories_list) >= 3 and op_guidance:
                            result = {
                                "city": city,
                                "persona_id": persona_id,
                                "persona_title": p_title,
                                "operational_guidance": op_guidance,
                                "advisories": advisories_list[:5],
                                "workability_score": int(parsed.get("workability_score", 70)),
                                "status_level": str(parsed.get("status_level", "CAUTION")).upper(),
                                "provider": f"gemini ({model})",
                                "language": norm_lang,
                            }
                            _BRIEFING_CACHE[cache_key] = result
                            logger.info(f"Generated Gemini persona briefing for {city}/{persona_id} via {model}")
                            return result
                else:
                    logger.warning(f"Model {model} returned HTTP {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            logger.warning(f"Failed to query {model}: {e}")

    # Fallback to high-quality deterministic synthesizer
    logger.info("Falling back to deterministic persona synthesis.")
    result = generate_deterministic_briefing(city, persona_id, metrics, language=norm_lang)
    _BRIEFING_CACHE[cache_key] = result
    return result
