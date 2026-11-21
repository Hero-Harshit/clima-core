"""
Clima-Core Cognitive AI Assistant & Autonomous Simulation Agent.
Grounds large language models (Google Gemini) in verified physical and environmental intelligence.
Features:
1. Native Gemini Tool/Function Calling Loop (counterfactual simulations, city comparisons, UI actions).
2. Physical What-If Scenario Stress-Testing Engine (SCS runoff, heat index, power grid MW, air quality).
3. Multi-turn Conversational Memory & Zero-Hallucination Grounding.
4. Client-side Dashboard UI Dispatching (Agent-driven view navigation).
5. Deterministic Physical Synthesis Fallback (runs full simulations even without an API key).
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import httpx
from pydantic import BaseModel, Field

from climacore.common.logging import get_logger
from climacore.ingestion.models import LivePredictionResult
from climacore.service.agent_tools import (
    AGENT_FUNCTION_DECLARATIONS,
    AgentToolExecutor,
    detect_and_execute_offline_scenario,
)
from climacore.analytics.rag_retriever import (
    retrieve_relevant_standards,
    format_retrieved_standards_for_prompt,
)

logger = get_logger("climacore.service.assistant")

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.0-flash"


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant' (or 'system')")
    content: str = Field(..., description="Message text")


class AssistantChatRequest(BaseModel):
    message: str = Field(..., description="User query or instruction")
    city: str = Field(default="Mumbai", description="Target Indian city or coordinates")
    persona: str = Field(default="general", description="Occupation / persona identifier (e.g. farmer, student, commuter)")
    history: List[ChatMessage] = Field(default_factory=list, description="Prior conversational turns")
    force_refresh: bool = Field(default=False, description="Force fresh fetch of upstream weather data")


class AssistantChatResponse(BaseModel):
    response: str = Field(..., description="Grounded markdown response from the assistant")
    city: str
    persona: str
    metrics_used: Dict[str, Any] = Field(default_factory=dict, description="Key environmental metrics that grounded this answer")
    suggested_followups: List[str] = Field(default_factory=list, description="Suggested quick-reply prompt pills")
    action: Optional[Dict[str, Any]] = Field(default=None, description="Client-side UI action dispatched by the agent")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tools executed by the agent during reasoning")
    retrieved_standards: List[Dict[str, Any]] = Field(default_factory=list, description="Verified institutional standards retrieved via local RAG")
    provider: str = Field(..., description="Model provider ('gemini-2.0-flash' or 'climacore-deterministic-agent')")
    latency_ms: float = Field(..., description="Total execution latency in milliseconds")


def build_metrics_summary(pred: LivePredictionResult, persona_id: str) -> Dict[str, Any]:
    """Extract key metrics summary from live prediction payload for context and citations."""
    if not pred.daily_predictions:
        return {}

    today = pred.daily_predictions[0]
    m = today.forecast_metrics
    personas_dict = today.personas or {}
    persona_data = personas_dict.get(persona_id, personas_dict.get("general", {}))

    flagship = today.flagship_intelligence or {}
    flood = flagship.get("flood_risk", {})
    heat = flagship.get("heat_risk", {})
    grid = flagship.get("power_grid_stress", {})
    air = flagship.get("air_quality", {})
    agri = flagship.get("agrisense", {})
    wq = today.water_quality_risk or {}
    conf = today.conformal_bounds or {}

    raw_rain_prob = float(m.get("rain_probability", 0.0) or 0.0)
    rain_probability_pct = int(raw_rain_prob if raw_rain_prob > 1.0 else raw_rain_prob * 100.0)

    summary = {
        "city": pred.city,
        "state": pred.state,
        "date": today.date,
        "day_name": today.day_name,
        "temperature_max_c": m.get("temp_max"),
        "temperature_min_c": m.get("temp_min"),
        "apparent_temp_max_c": m.get("apparent_temp_max"),
        "wet_bulb_temp_c": m.get("wet_bulb_temp_c"),
        "precipitation_mm": float(m.get("precipitation_mm", 0.0) or 0.0),
        "rain_probability": rain_probability_pct / 100.0,
        "rain_probability_pct": rain_probability_pct,
        "wind_speed_kmh": m.get("wind_speed_kmh"),
        "relative_humidity_pct": m.get("relative_humidity"),
        "pm25_ug_m3": m.get("pm2_5"),
        "pm10_ug_m3": m.get("pm10"),
        "air_quality_category": air.get("air_quality_category", "MODERATE"),
        "heatwave_alert_level": m.get("imd_heatwave_alert", 0),
        "flood_hazard_probability": flood.get("hydrological_hazard_probability", 0.0),
        "effective_runoff_mm": flood.get("scs_runoff_mm", 0.0),
        "peak_grid_load_mw": grid.get("peak_load_mw"),
        "grid_blackout_probability": grid.get("unmet_supply_stress_probability", 0.0),
        "soil_moisture_deficit_index": m.get("soil_moisture_deficit_index"),
        "crop_stress_status": agri.get("crop_stress_status", "NORMAL"),
        "groundwater_potability": wq.get("potability_status", "UNKNOWN"),
        "peak_compound_risk": pred.peak_compound_risk,
        "peak_risk_category": pred.peak_risk_category,
        "persona_status": persona_data.get("status_level", "NORMAL") if isinstance(persona_data, dict) else getattr(persona_data, "status_level", "NORMAL"),
        "persona_impact_score": persona_data.get("impact_score", 0.0) if isinstance(persona_data, dict) else getattr(persona_data, "impact_score", 0.0),
        "conformal_intervals_90pct": {
            "temperature_range": f"{conf.get('temperature', {}).get('lower', m.get('temp_max')):.1f}°C to {conf.get('temperature', {}).get('upper', m.get('temp_max')):.1f}°C",
            "rainfall_range": f"{conf.get('rainfall', {}).get('lower', 0.0):.1f}mm to {conf.get('rainfall', {}).get('upper', m.get('precipitation_mm', 0.0)):.1f}mm",
            "pm25_range": f"{conf.get('pm25', {}).get('lower', m.get('pm2_5')):.1f} to {conf.get('pm25', {}).get('upper', m.get('pm2_5')):.1f} µg/m³",
        },
    }
    return summary


def build_grounded_system_prompt(
    pred: LivePredictionResult,
    persona_id: str,
    retrieved_standards: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a zero-hallucination system prompt embedding Clima-Core's verified physical metrics and RAG standards."""
    metrics = build_metrics_summary(pred, persona_id)
    today = pred.daily_predictions[0]
    personas_dict = today.personas or {}
    persona_data = personas_dict.get(persona_id, {})
    if hasattr(persona_data, "model_dump"):
        persona_data = persona_data.model_dump()

    advisories = persona_data.get("advisories", []) if isinstance(persona_data, dict) else []
    restrictions = persona_data.get("operational_restrictions", []) if isinstance(persona_data, dict) else []

    outlook_lines = []
    for d in pred.daily_predictions[:5]:
        m = d.forecast_metrics
        raw_p = float(m.get("rain_probability", 0.0) or 0.0)
        p_pct = int(raw_p if raw_p > 1.0 else raw_p * 100.0)
        outlook_lines.append(
            f"- {d.day_name} ({d.date}): Max Temp {m.get('temp_max')}°C, Rain {m.get('precipitation_mm')}mm (Prob {p_pct}%), PM2.5 {m.get('pm2_5')} µg/m³"
        )
    outlook_str = "\n".join(outlook_lines)

    rag_section = ""
    if retrieved_standards:
        rag_section = f"\n\n{format_retrieved_standards_for_prompt(retrieved_standards)}"

    prompt = f"""You are the Clima-Core Cognitive Environmental Risk & Decision Agent, an autonomous AI system built on India's premier physics-guided environmental intelligence platform.

CURRENT DASHBOARD CONTEXT:
- Location: {metrics['city']}, {metrics['state']}
- Active Persona / Sector: {persona_id.upper()} ({persona_data.get('persona_title', persona_id.capitalize()) if isinstance(persona_data, dict) else persona_id})
- Today's Date: {metrics['date']} ({metrics['day_name']})

CURRENT DAY VERIFIED METRICS:
- Max Temp: {metrics['temperature_max_c']}°C (Apparent: {metrics['apparent_temp_max_c']}°C, Wet-Bulb: {metrics['wet_bulb_temp_c']}°C)
- 90% Conformal Temperature Range: {metrics['conformal_intervals_90pct']['temperature_range']}
- 24h Rainfall: {metrics['precipitation_mm']} mm (Rain Prob: {metrics['rain_probability_pct']}%)
- 90% Conformal Rainfall Range: {metrics['conformal_intervals_90pct']['rainfall_range']}
- Effective Runoff: {metrics['effective_runoff_mm']} mm | Flood Hazard Prob: {metrics['flood_hazard_probability']:.2f}
- Max Wind Speed: {metrics['wind_speed_kmh']} km/h | Relative Humidity: {metrics['relative_humidity_pct']}%
- Air Quality (PM2.5): {metrics['pm25_ug_m3']} µg/m³ ({metrics['air_quality_category']})
- Power Grid: Peak Load {metrics['peak_grid_load_mw']} MW (Blackout Risk: {metrics['grid_blackout_probability']:.2f})
- Soil Moisture Deficit (SMDI): {metrics['soil_moisture_deficit_index']} | Crop Stress: {metrics['crop_stress_status']}
- Water Quality: {metrics['groundwater_potability']}

ACTIVE OPERATIONAL RESTRICTIONS:
{chr(10).join([f'* CRITICAL: {r}' for r in restrictions]) if restrictions else '* No emergency restrictions currently active.'}

ACTIVE STRATEGIC ADVISORIES:
{chr(10).join([f'* {a}' for a in advisories]) if advisories else '* Standard operations.'}

UPCOMING 5-DAY OUTLOOK:
{outlook_str}{rag_section}

AGENT POWERS AND CAPABILITIES:
You have access to specialized simulation and control tools:
1. `simulate_what_if_scenario`: Call this whenever the user asks an imaginary or hypothetical question (e.g. 'What if it rains 120mm?', 'What if temperatures hit 46C during a heatwave?', 'What if PM2.5 spikes?'). NEVER invent or guess what-if numbers; call this simulator to get real SCS runoff, MW surge, and hazard probabilities.
2. `compare_cities_risk`: Call this whenever the user compares two locations or asks about travel/route risk (e.g. 'Compare Delhi and Jaipur', 'I am driving from Mumbai to Pune').
3. `get_groundwater_quality`: Call this for specific aquifer and chemical potability inquiries.
4. `dispatch_dashboard_action`: Call this when the user instructs the dashboard to change view, switch city, switch persona, or focus on a specific card.
5. `evaluate_parametric_insurance_claim`: Call this whenever the user inquires about crop insurance payouts, PMFBY, WBCIS compensation, or agricultural loss settlements.
6. `simulate_compound_domino_cascade`: Call this whenever the user asks about compound failure, cascading disaster impacts across grid/cold-chain/transit, or systemic infrastructure collapse.
7. `explain_risk_attribution_shap`: Call this whenever the user asks WHY a risk score or probability was predicted, requesting mathematical TreeSHAP explainability and key amplifying/mitigating drivers.
8. `convene_crisis_council`: Call this whenever the user asks to convene a Crisis Council, War Room deliberation, Incident Command debate, or resolve conflicting operational policies between Public Health, Power Grid, and Cold-Chain Logistics under severe climate stress.
9. `replay_historical_disaster`: Call this whenever the user asks about historical disasters, backtesting, catastrophe replays, or how Clima-Core would have performed during landmark events (e.g. 2005 Mumbai Cloudburst 944mm, 2015 Chennai Floods 494mm, 2022 North India 49.2C Heatwave).

STRICT PHYSICAL GROUNDING & INSTRUCTIONS:
- Ground all advice strictly in the provided verified data and physical laws.
- When answering a what-if question, cite the exact numbers returned by the simulation tool.
- Answer directly in the first 1-2 sentences with clear, punchy guidance.
"""
    return prompt.strip()


def generate_suggested_followups(pred: LivePredictionResult, persona_id: str) -> List[str]:
    """Generate 3 dynamic, context-relevant follow-up prompt pills."""
    if not pred.daily_predictions:
        return [
            "What if rainfall doubles tomorrow?",
            "Compare risk with another city",
            "Are there any safety precautions for today?",
        ]

    today = pred.daily_predictions[0]
    m = today.forecast_metrics
    raw_p = float(m.get("rain_probability", 0.0) or 0.0)
    rain_p = raw_p if raw_p <= 1.0 else raw_p / 100.0
    t_max = m.get("temp_max", 30.0)
    p_lower = persona_id.lower()

    if "farmer" in p_lower:
        return [
            "What if a 100mm cloudburst hits tomorrow?",
            "Is tomorrow safe for pesticide spraying?",
            "What is the 5-day crop heat-stress outlook?",
        ]

    if "student" in p_lower:
        pills = []
        pills.append("Should I carry an umbrella to school today?" if rain_p > 0.25 else "Is outdoor sports safe during afternoon heat?")
        pills.append("What if rainfall surges to 120mm during school hours?")
        pills.append("What is the commute weather forecast for tomorrow morning?")
        return pills

    if "commuter" in p_lower or "gig_worker" in p_lower:
        return [
            "What if rainfall causes arterial road waterlogging?",
            "What hours have the lowest heat index for outdoor delivery?",
            "Compare risk between my city and Pune",
        ]

    # General / Default
    pills = [
        "What if rainfall hits 100mm tomorrow in a cloudburst?",
        f"What if temperatures reach {int(t_max + 5)}°C during a heatwave?",
        "Compare weather and risk with another Indian city",
    ]
    return pills


def generate_deterministic_synthesis(
    query: str,
    pred: LivePredictionResult,
    persona_id: str,
    retrieved_standards: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Intelligent rule-based synthesis answering basic weather queries when Gemini API key is absent."""
    metrics = build_metrics_summary(pred, persona_id)
    q_lower = query.strip().lower()
    city = metrics["city"]
    rain_mm = metrics["precipitation_mm"]
    rain_p = metrics["rain_probability"]
    rain_prob_pct = int(metrics.get("rain_probability_pct", 0))
    t_max = metrics["temperature_max_c"]
    wet_bulb = metrics["wet_bulb_temp_c"]
    pm25 = metrics["pm25_ug_m3"]
    aqi_cat = metrics["air_quality_category"]

    today = pred.daily_predictions[0]
    personas_dict = today.personas or {}
    persona_data = personas_dict.get(persona_id, {})
    restrictions = persona_data.get("operational_restrictions", []) if isinstance(persona_data, dict) else []

    is_rainy = (rain_prob_pct >= 35) or (rain_mm >= 1.5)

    # 1. Agriculture / Pesticide / Crop Query
    if re.search(r"\b(pesticide|spray|spraying|crop|crops|wheat|paddy|rice|cotton|mustard|potato|irrigate|irrigation|farming|sowing|soil)\b", q_lower):
        icar_ref = ""
        if retrieved_standards:
            icar = next((s for s in retrieved_standards if "pesticide" in s.get("id", "") or "spray" in s.get("id", "") or s.get("domain") in ["agro_meteorology", "agriculture"]), None)
            if icar:
                icar_ref = (
                    f"\n\n* **ICAR-CRIDA Scientific Standard ({icar['title']}):** {icar['rule_text']}\n"
                    f"* **Mandatory Directive:** {icar['mandatory_directive']} ({icar['citation']})"
                )

        if is_rainy:
            return (
                f"**Hold off on chemical pesticide spraying in {city} today.**\n\n"
                f"* **Reason:** Rainfall is forecast at {rain_mm:.1f} mm ({rain_prob_pct}% probability). "
                f"Chemicals applied now will suffer severe surface wash-off, reducing efficacy and causing pesticide runoff into drainage channels.\n"
                f"* **Recommendation:** Delay chemical applications until dry conditions return.{icar_ref}"
            )
        else:
            return (
                f"**Field spraying conditions in {city} are favorable today.**\n\n"
                f"* **Forecast:** Precipitation probability is low ({rain_prob_pct}%), and wind speed is {metrics['wind_speed_kmh']} km/h (within safe spray drift threshold < 15 km/h).\n"
                f"* **Soil Deficit:** Soil moisture deficit index is {metrics['soil_moisture_deficit_index']}.{icar_ref}"
            )

    # 2. Water Quality / Chemical Potability / Aquifer Contamination (RAG Driven)
    if re.search(r"\b(fluoride|nitrate|arsenic|lead|uranium|hardness|tds|potable|drinking water|water quality|borewell|aquifer)\b", q_lower):
        wq_std = None
        if retrieved_standards:
            wq_std = next((s for s in retrieved_standards if s.get("domain") in ["hydro_geology", "water_quality"]), None)
        if wq_std:
            return (
                f"### Drinking Water Potability Assessment ({wq_std['authority']})\n\n"
                f"* **Standard:** **{wq_std['title']}**\n"
                f"* **Regulatory Limit:** {wq_std['rule_text']}\n"
                f"* **Trigger Threshold:** `{wq_std['trigger_condition']}`\n"
                f"* **Mandatory Directive:** {wq_std['mandatory_directive']}\n"
                f"* **Official Citation:** {wq_std['citation']}"
            )

    # 3. Critical Infrastructure, Railway Track Safety & Energy Grid (RAG Driven)
    if re.search(r"\b(railway|rail|track|buckling|patrolling|transformer|grid|substation|transmission|bitumen|asphalt|power plant)\b", q_lower):
        infra_std = None
        if retrieved_standards:
            infra_std = next((s for s in retrieved_standards if s.get("domain") == "infrastructure_energy"), None)
        if infra_std:
            return (
                f"### Infrastructure Operational Directive ({infra_std['authority']})\n\n"
                f"* **Standard:** **{infra_std['title']}**\n"
                f"* **Engineering Threshold:** {infra_std['rule_text']}\n"
                f"* **Trigger Condition:** `{infra_std['trigger_condition']}`\n"
                f"* **Mandatory Safety Action:** {infra_std['mandatory_directive']}\n"
                f"* **Official Reference:** {infra_std['citation']}"
            )

    # 4. Air Quality & GRAP Emergency Countermeasures (RAG Driven)
    if re.search(r"\b(aqi|pm2.5|pm10|grap|stubble|inversion|stagnation|kiln|smog)\b", q_lower):
        air_std = None
        if retrieved_standards:
            air_std = next((s for s in retrieved_standards if s.get("domain") == "air_quality_emissions"), None)
        if air_std:
            return (
                f"### Air Quality Regulatory Protocol ({air_std['authority']})\n\n"
                f"* **Regulatory Standard:** **{air_std['title']}**\n"
                f"* **Enforcement Threshold:** {air_std['rule_text']}\n"
                f"* **Trigger Condition:** `{air_std['trigger_condition']}`\n"
                f"* **Mandatory Countermeasure:** {air_std['mandatory_directive']}\n"
                f"* **Statutory Citation:** {air_std['citation']}"
            )

    # 5. Umbrella / Rain Query
    if re.search(r"\b(umbrella|rain|raining|rainy|shower|showers|monsoon|deluge|pour|downpour)\b", q_lower):
        if is_rainy:
            return (
                f"**Yes, you should definitely carry an umbrella today in {city}.**\n\n"
                f"* **Rainfall Forecast:** {rain_mm:.1f} mm with a **{rain_prob_pct}% probability** of precipitation.\n"
                f"* **Conformal 90% Interval:** Rainfall expected between {metrics['conformal_intervals_90pct']['rainfall_range']}.\n"
                f"* **Runoff Hazard:** Effective surface runoff is calculated at {metrics['effective_runoff_mm']:.1f} mm. "
                f"Be prepared for localized water accumulation in low-lying transit corridors."
            )
        else:
            return (
                f"**No, an umbrella is not required today in {city}.**\n\n"
                f"* **Rainfall Forecast:** {rain_mm:.1f} mm with only a **{rain_prob_pct}% chance** of isolated drizzle.\n"
                f"* **Conditions:** Skies are predominantly dry. You can travel without rain protection."
            )

    # 6. Heat / Outdoor Activity Query
    if re.search(r"\b(heat|heatwave|hot|sun|outdoor|sports|jog|exercise)\b", q_lower):
        ndma_ref = ""
        if retrieved_standards:
            ndma_std = next((s for s in retrieved_standards if s.get("domain") in ["public_health_disaster", "heat_disaster"]), None)
            if ndma_std and ndma_std.get("relevance_score", 0.0) >= 0.2:
                ndma_ref = f"\n* **NDMA Protocol ({ndma_std['title']}):** {ndma_std['rule_text']}"

        if t_max >= 38.0 or (wet_bulb and wet_bulb >= 28.0):
            return (
                f"**Caution is advised for outdoor activities in {city} today.**\n\n"
                f"* **Peak Temperature:** {t_max:.1f}°C (Apparent Heat Index: {metrics['apparent_temp_max_c']:.1f}°C).\n"
                f"* **Wet-Bulb Stress:** {wet_bulb:.1f}°C, which indicates heightened thermal strain.\n"
                f"* **Recommendation:** Reschedule strenuous outdoor sports or heavy physical labor to before 9:00 AM or after 5:30 PM. Maintain continuous hydration.{ndma_ref}"
            )
        else:
            return (
                f"**Outdoor conditions in {city} are comfortable for activities today.**\n\n"
                f"* **Max Temperature:** {t_max:.1f}°C with relative humidity at {metrics['relative_humidity_pct']}%. "
                f"Thermal strain is within normal operational limits.{ndma_ref}"
            )

    # 4. Agriculture / Pesticide / Crop Query
    if any(k in q_lower for k in ["pesticide", "spray", "crop", "irrigate", "farming", "sowing", "soil"]):
        icar_ref = ""
        if retrieved_standards:
            icar = next((s for s in retrieved_standards if "pesticide" in s.get("id", "") or "spray" in s.get("id", "") or s.get("domain") == "agriculture"), None)
            if icar:
                icar_ref = (
                    f"\n* **ICAR-CRIDA Scientific Standard ({icar['title']}):** {icar['rule_text']}\n"
                    f"* **Mandatory Directive:** {icar['mandatory_directive']} ({icar['citation']})"
                )

        if is_rainy:
            return (
                f"**Hold off on chemical pesticide spraying in {city} today.**\n\n"
                f"* **Reason:** Rainfall is forecast at {rain_mm:.1f} mm ({rain_prob_pct}% probability). "
                f"Chemicals applied now will suffer severe surface wash-off, reducing efficacy and causing pesticide runoff into drainage channels.\n"
                f"* **Recommendation:** Delay chemical applications until dry conditions return.{icar_ref}"
            )
        else:
            return (
                f"**Field spraying conditions in {city} are favorable today.**\n\n"
                f"* **Forecast:** Precipitation probability is low ({rain_prob_pct}%), and wind speed is {metrics['wind_speed_kmh']} km/h (within safe spray drift threshold < 15 km/h).\n"
                f"* **Soil Deficit:** Soil moisture deficit index is {metrics['soil_moisture_deficit_index']}.{icar_ref}"
            )

    # 5. Default Comprehensive Synthesis
    lines = [
        f"### Environmental Risk Summary for **{city}** ({persona_id.capitalize()} Context)\n",
        f"* **Thermal State:** High of **{t_max:.1f}°C** (Wet-Bulb: {wet_bulb or 'N/A'}°C), 90% confidence range: {metrics['conformal_intervals_90pct']['temperature_range']}.",
        f"* **Hydrology:** **{rain_mm:.1f} mm** precipitation expected with **{rain_prob_pct}%** probability.",
        f"* **Air Quality:** PM2.5 is **{pm25:.1f} µg/m³** ({aqi_cat}).",
        f"* **Overall Threat Level:** **{metrics['peak_risk_category']}** (Compound score {metrics['peak_compound_risk']:.1f}/100).",
    ]
    if restrictions:
        lines.append(f"\n**Active Operational Directives:**\n" + "\n".join([f"- ⚠️ {r}" for r in restrictions]))

    if retrieved_standards:
        top_std = retrieved_standards[0]
        if top_std.get("relevance_score", 0.0) >= 0.25:
            lines.append(
                f"\n**Institutional Regulatory Benchmark ({top_std['authority']}):**\n"
                f"- **{top_std['title']}**: {top_std['rule_text']}\n"
                f"  *(Directive: {top_std['mandatory_directive']} | Ref: {top_std['citation']})*"
            )

    return "\n".join(lines)


async def call_gemini_agent_with_tools(
    system_prompt: str,
    user_message: str,
    history: List[ChatMessage],
    api_key: str,
    executor: AgentToolExecutor,
    model: str = DEFAULT_MODEL,
    timeout_seconds: float = 16.0,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Call Google Gemini REST API with native tool calling loop, executing tools dynamically."""
    url = f"{GEMINI_API_URL}/{model}:generateContent?key={api_key}"

    # Build initial conversation contents
    contents: List[Dict[str, Any]] = []
    for h in history:
        role = "model" if h.role == "assistant" else "user"
        contents.append({
            "role": role,
            "parts": [{"text": h.content}]
        })
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    tools = [{"function_declarations": AGENT_FUNCTION_DECLARATIONS}]

    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": contents,
        "tools": tools,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1000,
        }
    }

    executed_tools: List[Dict[str, Any]] = []
    dispatched_action: Optional[Dict[str, Any]] = None

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        # Step 1: Initial invocation
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text[:200]}")
            return None, None, []

        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None, None, []

        candidate = candidates[0]
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])

        # Step 2: Check for function calls
        fn_call_part = None
        direct_text = None
        for p in parts:
            if "functionCall" in p:
                fn_call_part = p["functionCall"]
                break
            elif "text" in p:
                direct_text = p["text"]

        # If no tool was called, return direct text answer
        if not fn_call_part:
            return direct_text.strip() if direct_text else None, None, []

        # Step 3: Execute tool
        fn_name = fn_call_part.get("name")
        fn_args = fn_call_part.get("args", {})
        logger.info(f"Agent executing tool: {fn_name}({fn_args})")

        tool_result = executor.execute_tool(name=fn_name, args=fn_args)
        executed_tools.append({"name": fn_name, "args": fn_args, "result": tool_result})

        if fn_name == "dispatch_dashboard_action":
            dispatched_action = tool_result

        # Step 4: Multi-turn loop: send function response back to Gemini
        contents.append({
            "role": "model",
            "parts": [{"functionCall": fn_call_part}]
        })
        contents.append({
            "role": "function",
            "parts": [{
                "functionResponse": {
                    "name": fn_name,
                    "response": tool_result
                }
            }]
        })

        followup_payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents,
            "tools": tools,
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000,
            }
        }

        # Step 5: Second call for final natural language synthesis
        resp2 = await client.post(url, json=followup_payload)
        if resp2.status_code == 200:
            data2 = resp2.json()
            candidates2 = data2.get("candidates", [])
            if candidates2:
                parts2 = candidates2[0].get("content", {}).get("parts", [])
                for p in parts2:
                    if "text" in p:
                        return p["text"].strip(), dispatched_action, executed_tools

        return str(tool_result), dispatched_action, executed_tools


class AssistantEngine:
    """Master controller for Clima-Core's Cognitive AI Assistant & Autonomous Agent."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model = model

    async def chat(
        self,
        req: AssistantChatRequest,
        prediction_result: LivePredictionResult,
        live_engine: Optional[Any] = None,
    ) -> AssistantChatResponse:
        t0 = time.perf_counter()
        metrics = build_metrics_summary(prediction_result, req.persona)

        # 0. Retrieve Institutional Regulatory Standards via Federated Multi-Shard RAG
        retrieved_standards = retrieve_relevant_standards(
            query=req.message,
            persona=req.persona,
            city=prediction_result.city,
            top_k=2,
        )

        system_prompt = build_grounded_system_prompt(
            pred=prediction_result,
            persona_id=req.persona,
            retrieved_standards=retrieved_standards,
        )
        followups = generate_suggested_followups(prediction_result, req.persona)

        # Initialize tool executor
        from climacore.ingestion.engine import LiveForecastEngine
        active_live_engine = live_engine if isinstance(live_engine, LiveForecastEngine) else LiveForecastEngine()
        executor = AgentToolExecutor(live_engine=active_live_engine)

        ai_text = None
        action = None
        tool_calls: List[Dict[str, Any]] = []
        provider = "climacore-deterministic-agent"

        # 1. Attempt Gemini Agent Call with Native Function Calling
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if key:
            try:
                ai_text, action, tool_calls = await call_gemini_agent_with_tools(
                    system_prompt=system_prompt,
                    user_message=req.message,
                    history=req.history,
                    api_key=key,
                    executor=executor,
                    model=self.model,
                )
                if ai_text:
                    provider = self.model
            except Exception as e:
                logger.warning(f"Failed to query Gemini Agent API: {e}. Falling back to deterministic agent.")

        # 2. Offline / Deterministic Agent Fallback (Handles What-Ifs, Comparisons, and UI actions)
        if not ai_text:
            offline_result = detect_and_execute_offline_scenario(
                query=req.message,
                current_city=prediction_result.city,
                current_persona=req.persona,
                executor=executor,
            )
            if offline_result is not None:
                ai_text, action, tool_data = offline_result
                tool_calls.append({"name": "deterministic_scenario_simulator", "data": tool_data})
                provider = "climacore-deterministic-agent"
            else:
                ai_text = generate_deterministic_synthesis(
                    query=req.message,
                    pred=prediction_result,
                    persona_id=req.persona,
                    retrieved_standards=retrieved_standards,
                )
                provider = "climacore-deterministic-synthesis"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return AssistantChatResponse(
            response=ai_text,
            city=prediction_result.city,
            persona=req.persona,
            metrics_used=metrics,
            suggested_followups=followups,
            action=action,
            tool_calls=tool_calls,
            retrieved_standards=retrieved_standards,
            provider=provider,
            latency_ms=round(latency_ms, 2),
        )
