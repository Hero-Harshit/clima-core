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
DEFAULT_MODEL = "gemini-3-flash-preview"


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant' (or 'system')")
    content: str = Field(..., description="Message text")


class AssistantChatRequest(BaseModel):
    message: Optional[str] = Field(default=None, description="User query or instruction")
    query: Optional[str] = Field(default=None, description="Alternative user query field")
    prompt: Optional[str] = Field(default=None, description="Alternative prompt field")
    city: str = Field(default="Mumbai", description="Target Indian city or coordinates")
    persona: str = Field(default="general", description="Occupation / persona identifier (e.g. farmer, student, commuter)")
    history: List[ChatMessage] = Field(default_factory=list, description="Prior conversational turns")
    force_refresh: bool = Field(default=False, description="Force fresh fetch of upstream weather data")
    language: Optional[str] = Field(default="en", description="Target language code (en, hi, bn, ta, te, mr, gu)")


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
    language: str = Field(default="en", description="Response language code")


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
    language: str = "en",
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

    localization_rule = ""
    LANGUAGE_INSTRUCTIONS = {
        "hi": "MANDATORY LOCALIZATION DIRECTIVE: You MUST respond in Hindi (हिन्दी) using the Devanagari script. Ensure numeric quantities and scientific units (°C, mm, µg/m³, km/h, MW) remain accurate.",
        "bn": "MANDATORY LOCALIZATION DIRECTIVE: You MUST respond in Bengali (বাংলা) using the Bengali script. Ensure numeric quantities and scientific units (°C, mm, µg/m³, km/h, MW) remain accurate.",
        "ta": "MANDATORY LOCALIZATION DIRECTIVE: You MUST respond in Tamil (தமிழ்) using the Tamil script. Ensure numeric quantities and scientific units (°C, mm, µg/m³, km/h, MW) remain accurate.",
        "te": "MANDATORY LOCALIZATION DIRECTIVE: You MUST respond in Telugu (తెలుగు) using the Telugu script. Ensure numeric quantities and scientific units (°C, mm, µg/m³, km/h, MW) remain accurate.",
        "mr": "MANDATORY LOCALIZATION DIRECTIVE: You MUST respond in Marathi (मराठी) using the Devanagari script. Ensure numeric quantities and scientific units (°C, mm, µg/m³, km/h, MW) remain accurate.",
        "gu": "MANDATORY LOCALIZATION DIRECTIVE: You MUST respond in Gujarati (ગુજરાતી) using the Gujarati script. Ensure numeric quantities and scientific units (°C, mm, µg/m³, km/h, MW) remain accurate.",
    }
    lang = (language or "en").strip().lower()
    if lang != "en" and lang in LANGUAGE_INSTRUCTIONS:
        localization_rule = f"\n- {LANGUAGE_INSTRUCTIONS[lang]}"

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

OPERATIONAL COMMAND PROTOCOLS:
The user or assistant toolbar may invoke specialized operational command modes:
1. `/plan [role/scenario]` (Operational SOP Generator):
   You MUST output a structured, professional Standard Operating Procedure (SOP).
   Format with chronological phases:
   - Phase 1: Immediate Action (T-0 to T+2h) with actionable checkmarks (`- [ ]`)
   - Phase 2: Operational Defense (T+2 to T+6h) with actionable checkmarks (`- [ ]`)
   - Phase 3: Recovery & Verification (T+6h+) with actionable checkmarks (`- [ ]`)
   Ground the steps in the active persona, current weather risks, and official Indian regulatory standards (NDMA, IMD, CPCB).
2. `/whatif [delta/event]` (Extreme Scenario Stress-Tester):
   You MUST ground the query in ClimaCore physical hydrology, thermal stress, and supply chain models.
   You MUST explicitly project these physical metrics:
   - Drainage Capacity Breach Time
   - Underpass Runoff Depth (in mm)
   - Fleet Delay Advance (in minutes)
   - Mandi Arrival Impact (tonnage/arrival delay)
3. `/compare [cityA] [cityB]` (Multi-City Arbitrage & Disruption):
   You MUST pull telemetry or Mandi pricing for both locations and format a side-by-side comparison table including:
   - Risk Score
   - Disruption Level
   - Key Bottleneck
   - Mandi Commodity Price Difference (if Mandi or commodities are specified)

STRICT ENFORCEMENT: Never output em dashes (unicode U+2014). Use hyphens (-), colons (:), or bullets (*).

STRICT PHYSICAL GROUNDING & INSTRUCTIONS:
- Ground all advice strictly in the provided verified data and physical laws.
- When answering a what-if question, cite the exact numbers returned by the simulation tool.
- Answer directly in the first 1-2 sentences with clear, punchy guidance.{localization_rule}
"""
    return prompt.strip()


def generate_suggested_followups(
    pred: LivePredictionResult,
    persona_id: str,
    language: str = "en",
) -> List[str]:
    """Generate 3 dynamic, context-relevant follow-up prompt pills localized across 7 languages."""
    lang = (language or "en").strip().lower()
    if not pred.daily_predictions:
        defaults = {
            "en": ["What if rainfall doubles tomorrow?", "Compare risk with another city", "Are there any safety precautions for today?"],
            "hi": ["यदि कल बारिश दोगुनी हो जाए तो क्या होगा?", "किसी अन्य शहर के साथ जोखिम की तुलना करें", "क्या आज के लिए कोई सुरक्षा सावधानियां हैं?"],
            "bn": ["কাল যদি বৃষ্টি দ্বিগুণ হয় তবে কী হবে?", "অন্য একটি শহরের সাথে ঝুঁকির তুলনা করুন", "আজকের জন্য কি কোনো নিরাপত্তা সতর্কতা আছে?"],
            "ta": ["நாளை மழை இருமடங்கானால் என்ன நடக்கும்?", "மற்றொரு நகரத்துடன் அபாயத்தை ஒப்பிடுங்கள்", "இன்றைக்கு ஏதேனும் பாதுகாப்பு முன்னெச்சரிக்கைகள் உள்ளதா?"],
            "te": ["రేపు వర్షపాతం రెట్టింపు అయితే ఏమిటి?", "మరొక నగరంతో ప్రమాదాన్ని పోల్చండి", "ఈరోజు కోసం ఏవైనా భద్రతా జాగ్రత్తలు ఉన్నాయా?"],
            "mr": ["उद्या पाऊस दुप्पट झाल्यास काय होईल?", "दुसऱ्या शहराशी जोखमीची तुलना करा", "आजसाठी काही सुरक्षा खबरदारी आहे का?"],
            "gu": ["જો આવતીકાલે વરસાદ બમણો થાય તો શું થશે?", "અન્ય શહેર સાથે જોખમની સરખામણી કરો", "શું આજના દિવસ માટે કોઈ સલામતી સાવચેતી છે?"],
        }
        return defaults.get(lang, defaults["en"])

    today = pred.daily_predictions[0]
    m = today.forecast_metrics
    raw_p = float(m.get("rain_probability", 0.0) or 0.0)
    rain_p = raw_p if raw_p <= 1.0 else raw_p / 100.0
    t_max = m.get("temp_max", 30.0)
    p_lower = persona_id.lower()

    if "farmer" in p_lower:
        farmer_pills = {
            "en": [
                "What if a 100mm cloudburst hits tomorrow?",
                "Is tomorrow safe for pesticide spraying?",
                "What is the 5-day crop heat-stress outlook?",
            ],
            "hi": [
                "क्या कल 100 मिमी बादल फटने का खतरा है?",
                "क्या कल सुबह कीटनाशक छिड़काव सुरक्षित है?",
                "फसल पर 5 दिनों का ताप-तनाव पूर्वानुमान क्या है?",
            ],
            "bn": [
                "কাল কি ১০০ মিমি মেঘভাঙা বৃষ্টির সম্ভাবনা আছে?",
                "কাল সকালে কি কীটনাশক স্প্রে করা নিরাপদ?",
                "ফসলের ৫ দিনের তাপ-চাপ পূর্বাভাস কী?",
            ],
            "ta": [
                "நாளை 100 மி.மீ மேகவெடிப்பு மழை பெய்யுமா?",
                "நாளை காலை பூச்சிக்கொல்லி தெளிப்பது பாதுகாப்பானதா?",
                "அடுத்த 5 நாட்களுக்கான பயிர் வெப்ப அழுத்தம் என்ன?",
            ],
            "te": [
                "రేపు 100 మి.మీ క్లౌడ్‌బర్స్ట్ వచ్చే అవకాశం ఉందా?",
                "రేపు ఉదయం పురుగుమందుల పిచికారీ సురక్షితమేనా?",
                "పంటపై 5 రోజుల వేడి ఒత్తిడి అంచనా ఏమిటి?",
            ],
            "mr": [
                "उद्या १०० मिमी ढगफुटीची शक्यता आहे का?",
                "उद्या सकाळी कीटकनाशक फवारणी सुरक्षित आहे का?",
                "पिकांवरील ५ दिवसांचा उष्णता ताण अंदाज काय आहे?",
            ],
            "gu": [
                "શું આવતીકાલે ૧૦૦ મીમી વાદળ ફાટવાનું જોખમ છે?",
                "શું આવતીકાલે સવારે જંતુનાશક દવાનો છંટકાવ સુરક્ષિત છે?",
                "પાક પર આગામી ૫ દિવસના તાપમાનના તણાવની આગાહી શું છે?",
            ],
        }
        return farmer_pills.get(lang, farmer_pills["en"])

    if "student" in p_lower:
        if lang == "hi":
            p1 = "क्या आज मुझे स्कूल के लिए छाता ले जाना चाहिए?" if rain_p > 0.25 else "क्या दोपहर की धूप में खेलकूद सुरक्षित है?"
            return [p1, "यदि स्कूल के समय 120 मिमी बारिश हो जाए तो क्या होगा?", "कल सुबह स्कूल यात्रा का मौसम कैसा रहेगा?"]
        elif lang == "bn":
            p1 = "আজ কি স্কুলে ছাতা নিয়ে যাওয়া উচিত?" if rain_p > 0.25 else "দুপুরের রোদে কি খেলাধুলা করা নিরাপদ?"
            return [p1, "স্কুলের সময় যদি ১২০ মিমি ভারী বৃষ্টি হয় তবে কী হবে?", "আগামীকাল সকালে স্কুল যাতায়াতের আবহাওয়া কেমন থাকবে?"]
        elif lang == "ta":
            p1 = "இன்று பள்ளிக்கு குடை எடுத்துச் செல்ல வேண்டுமா?" if rain_p > 0.25 else "மதிய வெயிலில் விளையாடுவது பாதுகாப்பானதா?"
            return [p1, "பள்ளி நேரத்தில் மழை 120 மி.மீ ஆக உயர்ந்தால் என்ன செய்வது?", "நாளை காலை பள்ளிப் பயணத்திற்கான வானிலை முன்னறிவிப்பு என்ன?"]
        elif lang == "te":
            p1 = "ఈరోజు నేను పాఠశాలకు గొడుగు తీసుకెళ్లాలా?" if rain_p > 0.25 else "మధ్యాహ్నం ఎండలో ఆరుబయట ఆటలు ఆడటం సురక్షితమేనా?"
            return [p1, "పాఠశాల సమయంలో వర్షం 120 మి.మీకి పెరిగితే ఏమిటి?", "రేపు ఉదయం ప్రయాణ వాతావరణ సూచన ఏమిటి?"]
        elif lang == "mr":
            p1 = "आज शाळेत जाताना छत्री सोबत न्यावी का?" if rain_p > 0.25 else "दुपारच्या उन्हात मैदानी खेळ खेळणे सुरक्षित आहे का?"
            return [p1, "शाळेच्या वेळेत १२० मिमी मुसळधार पाऊस पडल्यास काय होईल?", "उद्या सकाळच्या प्रवासासाठी हवामानाचा अंदाज काय आहे?"]
        elif lang == "gu":
            p1 = "શું મારે આજે શાળાએ છત્રી લઈ જવી જોઈએ?" if rain_p > 0.25 else "બપોરના તડકામાં બહારની રમતો રમવી સલામત છે?"
            return [p1, "જો શાળા સમય દરમિયાન ૧૨૦ મીમી વરસાદ પડે તો શું થશે?", "આવતીકાલે સવારે મુસાફરી માટે હવામાન કેવું રહેશે?"]
        else:
            pills = []
            pills.append("Should I carry an umbrella to school today?" if rain_p > 0.25 else "Is outdoor sports safe during afternoon heat?")
            pills.append("What if rainfall surges to 120mm during school hours?")
            pills.append("What is the commute weather forecast for tomorrow morning?")
            return pills

    if "commuter" in p_lower or "gig_worker" in p_lower:
        commuter_pills = {
            "en": [
                "What if rainfall causes arterial road waterlogging?",
                "What hours have the lowest heat index for outdoor delivery?",
                "Compare risk between my city and Pune",
            ],
            "hi": [
                "क्या बारिश से मुख्य सड़कों पर जलभराव होगा?",
                "आउटडोर डिलीवरी के लिए सबसे कम गर्मी के घंटे कौन से हैं?",
                "मेरे शहर और पुणे के जोखिम की तुलना करें",
            ],
            "bn": [
                "বৃষ্টিতে কি প্রধান সড়কে জলজট হতে পারে?",
                "বাইরে ডেলিভারির জন্য সবচেয়ে কম তাপমাত্রার সময় কোনটি?",
                "আমার শহর এবং পুনের ঝুঁকির তুলনা করুন",
            ],
            "ta": [
                "மழையால் பிரதான சாலைகளில் வெள்ளம் தேங்குமா?",
                "வெளிப்புற டெலிவரிக்கு உகந்த குறைந்த வெப்ப நேரம் எது?",
                "என் நகரத்திற்கும் புனேவிற்கும் இடையிலான அபாயத்தை ஒப்பிடுங்கள்",
            ],
            "te": [
                "వర్షం వల్ల ప్రధాన రహదారులపై నీరు నిలుస్తుందా?",
                "డెలివరీల కోసం తక్కువ వేడి ఉండే సమయాలు ఏవి?",
                "నా నగరానికి మరియు పూణేకు మధ్య ప్రమాదాలను పోల్చండి",
            ],
            "mr": [
                "पावसामुळे मुख्य रस्त्यांवर पाणी साचेल का?",
                "डिलिव्हरीसाठी सर्वात कमी उन्हाची वेळ कोणती?",
                "माझ्या शहराची आणि पुण्याची जोखीम तुलना करा",
            ],
            "gu": [
                "શું વરસાદથી મુખ્ય રસ્તાઓ પર પાણી ભરાશે?",
                "ડિલિવરી માટે સૌથી ઓછા તાપમાનવાળા કલાકો કયા છે?",
                "મારા શહેર અને પુણેના જોખમની સરખામણી કરો",
            ],
        }
        return commuter_pills.get(lang, commuter_pills["en"])

    # General / Default
    if lang == "hi":
        return ["यदि कल 100 मिमी बादल फटे तो क्या होगा?", f"यदि तापमान बढ़कर {int(t_max + 5)}°C हो जाए तो क्या प्रभाव होगा?", "अन्य भारतीय शहर के साथ मौसम और जोखिम की तुलना करें"]
    elif lang == "bn":
        return ["কাল যদি ১০০ মিমি মেঘভাঙা বৃষ্টি হয় তবে কী হবে?", f"তাপমাত্রা বেড়ে {int(t_max + 5)}°C হলে কী प्रभाव পড়বে?", "অন্য ভারতীয় শহরের সাথে আবহাওয়া ও ঝুঁকির তুলনা করুন"]
    elif lang == "ta":
        return ["நாளை 100 மி.மீ மேகவெடிப்பு மழை பெய்தால் என்ன நடக்கும்?", f"வெப்பநிலை {int(t_max + 5)}°C ஆக உயர்ந்தால் ஏற்படும் விளைவு என்ன?", "மற்றொரு இந்திய நகரத்துடன் வானிலை மற்றும் அபாயத்தை ஒப்பிடுங்கள்"]
    elif lang == "te":
        return ["రేపు 100 மி.మీ క్లౌడ్‌బర్స్ట్ సంభవిస్తే ఏమిటి?", f"ఉష్ణోగ్రత {int(t_max + 5)}°Cకి చేరితే ఎలాంటి ప్రభావం ఉంటుంది?", "మరొక భారతీయ నగరంతో వాతావరణం మరియు ప్రమాదాలను పోల్చండి"]
    elif lang == "mr":
        return ["उद्या १०० मिमी ढगफुटी झाल्यास काय होईल?", f"उष्णतेच्या लाटेत तापमान {int(t_max + 5)}°C पर्यंत पोहोचल्यास काय होईल?", "दुसऱ्या भारतीय शहराशी हवामान आणि जोखमीची तुलना करा"]
    elif lang == "gu":
        return ["જો આવતીકાલે ૧૦૦ મીમી વાદળ ફાટે તો શું થશે?", f"જો તાપમાન વધીને {int(t_max + 5)}°C થાય તો શું અસર થશે?", "અન્ય ભારતીય શહેર સાથે હવામાન અને જોખમની સરખામણી કરો"]

    return [
        "What if rainfall hits 100mm tomorrow in a cloudburst?",
        f"What if temperatures reach {int(t_max + 5)}°C during a heatwave?",
        "Compare weather and risk with another Indian city",
    ]


def _generate_localized_deterministic_synthesis(
    query: str,
    pred: LivePredictionResult,
    persona_id: str,
    metrics: Dict[str, Any],
    is_rainy: bool,
    language: str,
) -> str:
    lang = (language or "en").strip().lower()
    if lang not in {"hi", "bn", "ta", "te", "mr", "gu"}:
        return generate_deterministic_synthesis(
            query=query,
            pred=pred,
            persona_id=persona_id,
            language="en",
        )
    q_lower = query.strip().lower()
    city = metrics["city"]
    rain_mm = metrics["precipitation_mm"]
    rain_prob_pct = int(metrics.get("rain_probability_pct", 0))
    t_max = metrics["temperature_max_c"]
    wet_bulb = metrics["wet_bulb_temp_c"]
    pm25 = metrics["pm25_ug_m3"]
    aqi_cat = metrics["air_quality_category"]

    # 1. Greetings & Introductions
    if re.search(r"\b(hi|hello|hey|namaste|pranam|greetings|help|shuru)\b", q_lower) and len(q_lower.split()) <= 6:
        greetings = {
            "hi": f"**नमस्ते! मैं आपका ClimaCore परिचालन खुफिया सहायक हूँ।**\n\nमैं **{city}** के लिए वास्तविक समय के मौसम और पर्यावरणीय जोखिमों की निगरानी कर रहा हूँ।\n* **आज का मौसम:** अधिकतम तापमान **{t_max:.1f}°C**, बारिश की संभावना **{rain_prob_pct}%**, वायु गुणवत्ता PM2.5 **{pm25:.1f} µg/m³** ({aqi_cat})।\n* आप मुझसे कल के मौसम, बारिश, या छाते की आवश्यकता के बारे में पूछ सकते हैं।",
            "bn": f"**নমস্কার! আমি আপনার ClimaCore অপারেশনাল ইন্টেলিজেন্স সহকারী।**\n\nআমি **{city}**-এর রিয়েল-টাইম আবহাওয়া ও পরিবেশগত ঝুঁকি পর্যবেক্ষণ করছি।\n* **আজকের আবহাওয়া:** সর্বোচ্চ তাপমাত্রা **{t_max:.1f}°C**, বৃষ্টির সম্ভাবনা **{rain_prob_pct}%**, বাতাসের মান PM2.5 **{pm25:.1f} µg/m³** ({aqi_cat})।\n* আপনি কালকের আবহাওয়া, বৃষ্টি বা ছাতার প্রয়োজনীয়তা সম্পর্কে আমাকে জিজ্ঞাসা করতে পারেন।",
            "ta": f"**வணக்கம்! நான் உங்கள் ClimaCore செயல்பாட்டு நுண்ணறிவு உதவியாளர்.**\n\nநான் **{city}**-க்கான நிகழ்நேர வானிலை மற்றும் சுற்றுச்சூழல் அபாயங்களைக் கண்காணித்து வருகிறேன்.\n* **இன்றைய வானிலை:** அதிகபட்ச வெப்பநிலை **{t_max:.1f}°C**, மழை வாய்ப்பு **{rain_prob_pct}%**, காற்றின் தரம் PM2.5 **{pm25:.1f} µg/m³** ({aqi_cat}).\n* நாளைய வானிலை, மழை அல்லது பாதுகாப்பு வழிமுறைகள் பற்றி நீங்கள் கேட்கலாம்.",
            "te": f"**నమస్కారం! నేను మీ ClimaCore ఆపరేషనల్ ఇంటెలిజెన్స్ అసిస్టెంట్.**\n\nనేను **{city}** కోసం నిజ-సమయ వాతావరణం మరియు పర్యావరణ ప్రమాదాలను పర్యవేక్షిస్తున్నాను.\n* **నేటి వాతావరణం:** గరిష్ట ఉష్ణోగ్రత **{t_max:.1f}°C**, వర్షం అవకాశం **{rain_prob_pct}%**, గాలి నాణ్యత PM2.5 **{pm25:.1f} µg/m³** ({aqi_cat}).\n* రేపటి వాతావరణం లేదా గొడుగు అవసరం గురించి మీరు నన్ను అడగవచ్చు.",
            "mr": f"**नमस्कार! मी तुमचा ClimaCore ऑपरेशनल इंटेलिजन्स सहाय्यक आहे.**\n\nमी **{city}** साठी रिअल-टाइम हवामान आणि पर्यावरणीय जोखमींवर लक्ष ठेवून आहे.\n* **आजचे हवामान:** कमाल तापमान **{t_max:.1f}°C**, पावसाची शक्यता **{rain_prob_pct}%**, हवेची गुणवत्ता PM2.5 **{pm25:.1f} µg/m³** ({aqi_cat}).\n* आपण मला उद्याचे हवामान, पाऊस किंवा सुरक्षेविषयी विचारू शकता.",
            "gu": f"**નમસ્તે! હું તમારો ClimaCore ઓપરેશનલ ઇન્ટેલિજન્સ સહાયક છું.**\n\nહું **{city}** માટે રીઅલ-ટાઇમ હવામાન અને પર્યાવરણીય જોખમો પર નજર રાખી રહ્યો છું.\n* **આજનું હવામાન:** મહત્તમ તાપમાન **{t_max:.1f}°C**, વરસાદની શક્યતા **{rain_prob_pct}%**, હવાની ગુણવત્તા PM2.5 **{pm25:.1f} µg/m³** ({aqi_cat}).\n* તમે મને આવતીકાલના હવામાન અથવા વરસાદ વિશે પૂછી શકો છો.",
        }
        return greetings.get(lang, greetings["hi"])

    # 2. About ClimaCore / Platform Inquiries
    if re.search(r"\b(clima[- ]?core|climacore|about|platform|features|kya hai)\b", q_lower):
        about_dict = {
            "hi": f"### ClimaCore इंटेलिजेंस प्लेटफॉर्म\n\n**ClimaCore** भारत के प्रमुख शहरों, बुनियादी ढांचे और उच्च-जोखिम वाले कार्यक्षेत्रों के लिए एक भौतिकी-निर्देशित जलवायु जोखिम खुफिया मंच है।\n\n* **भौतिकी-आधारित जोखिम मॉडल:** SCS सतही जलभराव, वेट-बल्ब ग्लोब तापमान (WBGT) हीट स्ट्रेस, CPCB वायु गुणवत्ता, और पावर ग्रिड लोड पूर्वानुमान।\n* **विशिष्ट कार्यक्षेत्र परामर्श:** किसान, गिग वर्कर्स, दैनिक यात्री, छात्र और उद्योग जगत के लिए अनुकूलित प्रत्यक्ष परिचालन दिशा-निर्देश।\n* **सिमुलेशन उपकरण:** आपातकालीन योजना के लिए `/plan`, अप्रत्याशित चरम मौसम के लिए `/whatif`, और दो शहरों के जोखिम तुलना के लिए `/compare` का उपयोग करें।",
            "bn": f"### ClimaCore ইন্টেলিজেন্স প্ল্যাটফর্ম\n\n**ClimaCore** ভারতের প্রধান শহর, অবকাঠামো ও শ্রমজীবী সম্প্রদায়ের জন্য নির্মিত একটি পদার্থবিজ্ঞান-নির্দেশিত জলবায়ু ঝুঁকি বিশ্লেষণ প্ল্যাটফর্ম।\n\n* **পদার্থবিজ্ঞান মডেল:** SCS সারফেস রানঅফ, ওয়েট-বাল্ব হিট স্ট্রেস, CPCB বায়ু মান এবং পাওয়ার গ্রিড লোড পূর্বাভাস।\n* **পেশাভিত্তিক পরামর্শ:** কৃষক, গিগ কর্মী, যাত্রী, শিক্ষার্থী এবং শিল্পের জন্য সরাসরি অপারেশনাল নির্দেশিকা।\n* **সিমুলেশন টুল:** জরুরি পরিকল্পনার জন্য `/plan`, আবহাওয়ার চরম পরিস্থিতির জন্য `/whatif` ব্যবহার করুন।",
            "ta": f"### ClimaCore நுண்ணறிவு தளம்\n\n**ClimaCore** என்பது இந்தியாவின் முக்கிய நகரங்கள் மற்றும் உள்கட்டமைப்புகளுக்காக வடிவமைக்கப்பட்ட காலநிலை இடர் பகுப்பாய்வு அமைப்பாகும்.\n\n* **இயற்பியல் மாதிரிகள்:** மேற்பரப்பு நீர் வடிகால் (SCS), ஈரக்குமிழ் வெப்ப அழுத்தம் (WBGT), மற்றும் காற்றின் தரம் (CPCB).\n* **துறைசார் ஆலோசனைகள்:** விவசாயிகள், டெலிவரி தொழிலாளர்கள், மாணவர்கள் மற்றும் தொழில்துறைகளுக்கான பாதுகாப்பு நெறிமுறைகள்.",
            "te": f"### ClimaCore ఇంటెలిజెన్స్ ప్లాట్‌ఫామ్\n\n**ClimaCore** అనేది భారతీయ నగరాలు మరియు కీలక మౌలిక సదుపాయాల కోసం రూపొందించబడిన వాతావరణ ముప్పు విశ్లేషణ వేదిక.\n\n* **భౌతిక నమూనాలు:** ఉపరితల నీటి ప్రవాహం (SCS), వెట్-బల్బ్ వేడి ఒత్తిడి (WBGT), మరియు గాలి నాణ్యత సూచిక.",
            "mr": f"### ClimaCore इंटेलिजन्स प्लॅटफॉर्म\n\n**ClimaCore** ही भारतातील शहरे आणि पायाभूत सुविधांसाठी भौतिकशास्त्र-आधारित हवामान जोखीम बुद्धिमत्ता प्रणाली आहे.\n\n* **भौतिक मॉडेल:** पृष्ठभागावरील पाण्याचा निचरा (SCS), वेट-बल्ब उष्णता ताण (WBGT), आणि हवेची गुणवत्ता.",
            "gu": f"### ClimaCore ઇન્ટેલિજન્સ પ્લેટફોર્મ\n\n**ClimaCore** એ ભારતીય શહેરો અને પાયાની સુવિધાઓ માટે ભૌતિકશાસ્ત્ર આધારિત આબોહવા જોખમ વિશ્લેષણ સિસ્ટમ છે.",
        }
        return about_dict.get(lang, about_dict["hi"])

    # 3. Tomorrow Weather / Sunny / Umbrella Query
    if re.search(r"\b(tomorrow|tommorow|tomorow|kal|aavtikaal|naalai|repu|udya)\b", q_lower):
        tomorrow = pred.daily_predictions[1] if len(pred.daily_predictions) > 1 else pred.daily_predictions[0]
        m1 = tomorrow.forecast_metrics
        t_max1 = float(m1.get("temp_max", 30.0) or 30.0)
        rain_mm1 = float(m1.get("precipitation_mm", 0.0) or 0.0)
        raw_p1 = float(m1.get("rain_probability", 0.0) or 0.0)
        rain_p1 = int(raw_p1 if raw_p1 > 1.0 else raw_p1 * 100.0)
        is_sunny1 = (rain_p1 < 25 and rain_mm1 < 1.0)
        is_rainy1 = (rain_p1 >= 35 or rain_mm1 >= 1.5)

        # Umbrella specific check for tomorrow
        if re.search(r"\b(umbrella|chata|chatri|kodai|godugu)\b", q_lower):
            if is_rainy1:
                tomorrow_umb_yes = {
                    "hi": f"**हाँ, कल ({tomorrow.day_name}) {city} में आपको छाता साथ रखना चाहिए।**\n\n* **पूर्वानुमान:** {rain_mm1:.1f} मिमी बारिश की **{rain_p1}% संभावना** है।\n* **परामर्श:** यात्रा के दौरान भीगने से बचने के लिए छाता अवश्य रखें।",
                    "bn": f"**হ্যাঁ, কাল ({tomorrow.day_name}) {city}-তে আপনার সাথে ছাতা রাখা উচিত।**\n\n* **পূর্বাভাস:** {rain_mm1:.1f} মিমি বৃষ্টির **{rain_p1}% সম্ভাবনা** রয়েছে।",
                    "ta": f"**ஆம், நாளை ({tomorrow.day_name}) {city}-ல் குடை எடுத்துச் செல்ல வேண்டும்.**\n\n* **முன்னறிவிப்பு:** {rain_mm1:.1f} மி.மீ மழை பெய்ய **{rain_p1}% வாய்ப்பு** உள்ளது.",
                    "te": f"**అవును, రేపు ({tomorrow.day_name}) {city}లో గొడుగు తీసుకెళ్లాలి.**\n\n* **సూచన:** {rain_mm1:.1f} మి.మీ వర్షం అవకాశం **{rain_p1}%**.",
                    "mr": f"**होय, उद्या ({tomorrow.day_name}) {city} मध्ये छत्री सोबत ठेवावी.**\n\n* **अंदाज:** {rain_mm1:.1f} मिमी पावसाची **{rain_p1}% शक्यता** आहे.",
                    "gu": f"**હા, આવતીકાલે ({tomorrow.day_name}) {city}માં છત્રી સાથે રાખવી પડશે.**\n\n* **આગાહી:** {rain_mm1:.1f} મીમી વરસાદની **{rain_p1}% શક્યતા** છે.",
                }
                return tomorrow_umb_yes.get(lang, tomorrow_umb_yes["hi"])
            else:
                tomorrow_umb_no = {
                    "hi": f"**नहीं, कल ({tomorrow.day_name}) {city} में छाते की कोई आवश्यकता नहीं होगी।**\n\n* **पूर्वानुमान:** मौसम मुख्य रूप से शुष्क रहेगा और बारिश की संभावना केवल **{rain_p1}%** ({rain_mm1:.1f} मिमी) है।\n* **परिस्थिति:** आसमान साफ रहेगा और धूप खिली रहेगी।",
                    "bn": f"**না, কাল ({tomorrow.day_name}) {city}-তে ছাতার কোনো প্রয়োজন নেই।**\n\n* **পূর্বাভাস:** আকাশ প্রধানত শুষ্ক ও পরিষ্কার থাকবে, বৃষ্টির সম্ভাবনা মাত্র **{rain_p1}%**।",
                    "ta": f"**இல்லை, நாளை ({tomorrow.day_name}) {city}-ல் குடை தேவையில்லை.**\n\n* **முன்னறிவிப்பு:** வானம் வறண்டு காணப்படும், மழை வாய்ப்பு **{rain_p1}%** மட்டுமே.",
                    "te": f"**లేదు, రేపు ({tomorrow.day_name}) {city}లో గొడుగు అవసరం లేదు.**\n\n* **పరిస్థితులు:** వాతావరణం పొడిగా ఉంటుంది, వర్షం అవకాశం **{rain_p1}%** మాత్రమే.",
                    "mr": f"**नाही, उद्या ({tomorrow.day_name}) {city} मध्ये छत्रीची गरज नाही.**\n\n* **अंदाज:** हवामान कोरडे राहील, पावसाची शक्यता केवळ **{rain_p1}%** आहे.",
                    "gu": f"**ના, આવતીકાલે ({tomorrow.day_name}) {city}માં છત્રીની જરૂર નથી.**\n\n* **આગાહી:** આકાશ સ્વચ્છ રહેશે, વરસાદની શક્યતા માત્ર **{rain_p1}%** છે.",
                }
                return tomorrow_umb_no.get(lang, tomorrow_umb_no["hi"])

        if is_sunny1:
            tomorrow_res = {
                "hi": f"**हाँ, कल ({tomorrow.day_name}) {city} में मुख्य रूप से धूप खिली रहेगी और आसमान साफ रहेगा!**\n\n* **तापमान:** अधिकतम तापमान **{t_max1:.1f}°C** तक पहुंचेगा।\n* **वर्षा:** बारिश की संभावना केवल **{rain_p1}%** ({rain_mm1:.1f} मिमी) है। मौसम पूरी तरह शुष्क रहेगा।\n* **परामर्श:** बाहरी गतिविधियों और यात्रा के लिए परिस्थितियां अनुकूल हैं।",
                "bn": f"**হ্যাঁ, কাল ({tomorrow.day_name}) {city}-তে প্রধানত রৌদ্রোজ্জ্বল ও পরিষ্কার আকাশ থাকবে!**\n\n* **তাপমাত্রা:** সর্বোচ্চ তাপমাত্রা **{t_max1:.1f}°C** পর্যন্ত হতে পারে।\n* **বৃষ্টিপাত:** বৃষ্টির সম্ভাবনা মাত্র **{rain_p1}%** ({rain_mm1:.1f} মিমি)। আবহাওয়া শুষ্ক থাকবে।\n* **পরামর্শ:** বাইরের কাজকর্মের জন্য পরিস্থিতি অনুকূল।",
                "ta": f"**ஆம், நாளை ({tomorrow.day_name}) {city}-ல் பெரும்பாலும் வெயில் மற்றும் தெளிவான வானிலை நிலவும்!**\n\n* **வெப்பநிலை:** அதிகபட்ச வெப்பநிலை **{t_max1:.1f}°C** வரை இருக்கும்.\n* **மழைப்பொழிவு:** மழைக்கான வாய்ப்பு மிகக் குறைவு **{rain_p1}%** மட்டுமே ({rain_mm1:.1f} மி.மீ).\n* **பரிந்துரை:** வெளிப்புற பணிகளுக்கு சாதகமான சூழல் உள்ளது.",
                "te": f"**అవును, రేపు ({tomorrow.day_name}) {city}లో ప్రధానంగా ఎండగా మరియు స్పష్టమైన ఆకాశం ఉంటుంది!**\n\n* **ఉష్ణోగ్రత:** గరిష్ట ఉష్ణోగ్రత **{t_max1:.1f}°C** చేరుకుంటుంది.\n* **వర్షపాతం:** వర్షం అవకాశం కేవలం **{rain_p1}%** మాత్రమే ({rain_mm1:.1f} మి.మీ).",
                "mr": f"**होय, उद्या ({tomorrow.day_name}) {city} मध्ये मुख्यतः निरभ्र आणि सूर्यप्रकाशित हवामान राहील!**\n\n* **तापमान:** कमाल तापमान **{t_max1:.1f}°C** पर्यंत राहील.\n* **पाऊस:** पावसाची शक्यता केवळ **{rain_p1}%** ({rain_mm1:.1f} मिमी) आहे.",
                "gu": f"**હા, આવતીકાલે ({tomorrow.day_name}) {city}માં મુખ્યત્વે તડકો અને સ્વચ્છ આકાશ રહેશે!**\n\n* **તાપમાન:** મહત્તમ તાપમાન **{t_max1:.1f}°C** રહેશે.\n* **વરસાદ:** વરસાદની શક્યતા માત્ર **{rain_p1}%** છે.",
            }
            return tomorrow_res.get(lang, tomorrow_res["hi"])
        elif is_rainy1:
            tomorrow_rain = {
                "hi": f"**कल ({tomorrow.day_name}) {city} में बादल छाए रहेंगे और बारिश की संभावना है।**\n\n* **तापमान:** अधिकतम तापमान **{t_max1:.1f}°C** रहेगा।\n* **वर्षा:** **{rain_mm1:.1f} मिमी** बारिश की **{rain_p1}% संभावना** है।\n* **परामर्श:** यात्रा के समय छाता साथ रखें और निचले रास्तों पर जलभराव से बचें।",
                "bn": f"**কাল ({tomorrow.day_name}) {city}-তে মেঘলা আকাশ এবং বৃষ্টির সম্ভাবনা রয়েছে।**\n\n* **তাপমাত্রা:** সর্বোচ্চ তাপমাত্রা **{t_max1:.1f}°C**।\n* **বৃষ্টিপাত:** **{rain_mm1:.1f} মিমি** বৃষ্টির **{rain_p1}% সম্ভাবনা** রয়েছে।\n* **পরামর্শ:** সাথে ছাতা রাখুন।",
                "ta": f"**நாளை ({tomorrow.day_name}) {city}-ல் மேகமூட்டத்துடன் மழை பெய்ய வாய்ப்புள்ளது.**\n\n* **அதிகபட்ச வெப்பநிலை:** **{t_max1:.1f}°C**.\n* **மழை:** **{rain_mm1:.1f} மி.மீ** மழை பெய்ய **{rain_p1}% வாய்ப்பு** உள்ளது.",
                "te": f"**రేపు ({tomorrow.day_name}) {city}లో మేఘావృతమై వర్షం పడే అవకాశం ఉంది.**\n\n* **గరిష్ట ఉష్ణోగ్రత:** **{t_max1:.1f}°C**.\n* **వర్షం:** **{rain_mm1:.1f} మి.మీ** వర్షం అవకాశం **{rain_p1}%**.",
                "mr": f"**उद्या ({tomorrow.day_name}) {city} मध्ये ढगाळ हवामान आणि पावसाची शक्यता आहे.**\n\n* **कमाल तापमान:** **{t_max1:.1f}°C**.\n* **पाऊस:** **{rain_mm1:.1f} मिमी** पावसाची **{rain_p1}% शक्यता**.",
                "gu": f"**આવતીકાલે ({tomorrow.day_name}) {city}માં વાદળછાયું વાતાવરણ અને વરસાદની સંભાવના છે.**\n\n* **મહત્તમ તાપમાન:** **{t_max1:.1f}°C**.\n* **વરસાદ:** **{rain_mm1:.1f} મીમી** વરસાદની **{rain_p1}% શક્યતા**.",
            }
            return tomorrow_rain.get(lang, tomorrow_rain["hi"])

    # 4. Umbrella / Rain Query
    if re.search(r"\b(umbrella|rain|raining|rainy|shower|showers|monsoon|deluge|pour|downpour)\b", q_lower):
        if is_rainy:
            umbrella_yes = {
                "hi": f"**हाँ, आज {city} में आपको निश्चित रूप से छाता ले जाना चाहिए।**\n\n* **वर्षा पूर्वानुमान:** {rain_mm:.1f} मिमी बारिश की **{rain_prob_pct}% संभावना** है।\n* **पूर्वानुमान 90% अंतराल:** वर्षा सीमा {metrics['conformal_intervals_90pct']['rainfall_range']} के बीच रहने की संभावना है।\n* **सतही जलभराव जोखिम:** लगभग {metrics['effective_runoff_mm']:.1f} मिमी जल अपवाह का अनुमान है। निचले इलाकों में जलभराव के प्रति सचेत रहें।",
                "bn": f"**হ্যাঁ, আজ {city}-তে অবশ্যই আপনার সাথে ছাতা রাখা উচিত।**\n\n* **বৃষ্টির পূর্বাভাস:** {rain_mm:.1f} মিমি বৃষ্টিপাতের **{rain_prob_pct}% সম্ভাবনা** রয়েছে।\n* **৯০% কনফরমাল ইন্টারভাল:** সম্ভাব্য বৃষ্টিপাতের পরিসীমা {metrics['conformal_intervals_90pct']['rainfall_range']}।\n* **জলজট ঝুঁকি:** ভূপৃষ্ঠে আনুমানিক {metrics['effective_runoff_mm']:.1f} মিমি রানঅফ হতে পারে। নিচু এলাকায় চলাচলে সতর্কতা অবলম্বন করুন।",
                "ta": f"**ஆம், இன்று {city}-ல் நீங்கள் கண்டிப்பாக குடை எடுத்துச் செல்ல வேண்டும்.**\n\n* **மழை முன்னறிவிப்பு:** {rain_mm:.1f} மி.மீ மழைப் பொழிவிற்கு **{rain_prob_pct}% வாய்ப்பு** உள்ளது.\n* **90% முன்னறிவிப்பு வரம்பு:** மழைப்பொழிவு {metrics['conformal_intervals_90pct']['rainfall_range']} இடையே இருக்கும்.\n* **வெள்ள அபாயம்:** பயனுள்ள மேற்பரப்பு நீர் ஓட்டம் {metrics['effective_runoff_mm']:.1f} மி.மீ என கணக்கிடப்பட்டுள்ளது. தாழ்வான பகுதிகளில் கவனமாக இருங்கள்.",
                "te": f"**అవును, ఈరోజు {city}లో మీరు ఖచ్చితంగా గొడుగు తీసుకెళ్లాలి.**\n\n* **వర్షపాతం సూచన:** {rain_mm:.1f} మి.మీ వర్షపాతం పడేందుకు **{rain_prob_pct}% సంభావ్యత** ఉంది.\n* **90% విశ్వసనీయ పరిధి:** వర్షపాతం {metrics['conformal_intervals_90pct']['rainfall_range']} మధ్య ఉండే అవకాశం ఉంది.\n* **మురుగునీటి ముప్పు:** ఉపరితల ప్రవాహం {metrics['effective_runoff_mm']:.1f} మి.మీగా అంచనా వేయబడింది. లోతట్టు ప్రాంతాలలో జాగ్రత్తగా ఉండండి.",
                "mr": f"**होय, आज {city} मध्ये तुम्ही नक्कीच छत्री सोबत घेतली पाहिजे.**\n\n* **पावसाचा अंदाज:** {rain_mm:.1f} मिमी पावसाची **{rain_prob_pct}% शक्यता** आहे.\n* **९०% कॉन्फॉर्मल श्रेणी:** पाऊस {metrics['conformal_intervals_90pct']['rainfall_range']} दरम्यान पडण्याची शक्यता आहे.\n* **पाणी साचण्याचा धोका:** पृष्ठभागावरील पाण्याचा निचरा {metrics['effective_runoff_mm']:.1f} मिमी अपेक्षित आहे. सखल भागात सावध राहा.",
                "gu": f"**હા, આજે {city}માં તમારે ચોક્કસપણે છત્રી સાથે રાખવી જોઈએ.**\n\n* **વરસાદની આગાહી:** {rain_mm:.1f} મીમી વરસાદની **{rain_prob_pct}% સંભાવના** છે.\n* **૯૦% અનુમાન મર્યાદા:** વરસાદ {metrics['conformal_intervals_90pct']['rainfall_range']} વચ્ચે રહેવાની અપેક્ષા છે.\n* **પાણી ભરાવાનું જોખમ:** સપાટી પર {metrics['effective_runoff_mm']:.1f} મીમી પાણી વહેવાની ગણતરી છે. નીચાણવાળા વિસ્તારોમાં સાવચેતી રાખો.",
            }
            return umbrella_yes.get(lang, umbrella_yes["hi"])
        else:
            umbrella_no = {
                "hi": f"**नहीं, आज {city} में छाते की आवश्यकता नहीं है।**\n\n* **वर्षा पूर्वानुमान:** केवल {rain_mm:.1f} मिमी और छिटपुट बूंदाबांदी की **{rain_prob_pct}% संभावना** है।\n* **मौसम की स्थिति:** आसमान मुख्य रूप से शुष्क रहेगा। आप बिना बारिश सुरक्षा के यात्रा कर सकते हैं।",
                "bn": f"**না, আজ {city}-তে ছাতার প্রয়োজন নেই।**\n\n* **বৃষ্টির পূর্বাভাস:** মাত্র {rain_mm:.1f} মিমি এবং হালকা গুঁড়ি গুঁড়ি বৃষ্টির **{rain_prob_pct}% সম্ভাবনা** রয়েছে।\n* **পরিস্থিতি:** আকাশ প্রধানত শুষ্ক থাকবে। আপনি কোনো বৃষ্টি সুরক্ষা ছাড়াই যাতায়াত করতে পারেন।",
                "ta": f"**இல்லை, இன்று {city}-ல் குடை தேவையில்லை.**\n\n* **மழை முன்னறிவிப்பு:** {rain_mm:.1f} மி.மீ மழை மற்றும் லேசான தூறலுக்கு **{rain_prob_pct}% மட்டுமே வாய்ப்பு** உள்ளது.\n* **வானிலை நிலை:** வானம் பெரும்பாலும் வறண்டு காணப்படும். நீங்கள் தாராளமாக பயணிக்கலாம்.",
                "te": f"**లేదు, ఈరోజు {city}లో గొడుగు అవసరం లేదు.**\n\n* **వర్షపాతం సూచన:** కేవలం {rain_mm:.1f} మి.మీ మరియు తేలికపాటి జల్లులకు **{rain_prob_pct}% మాత్రమే అవకాశం** ఉంది.\n* **పరిస్థితులు:** వాతావరణం పొడిగా ఉంటుంది. వర్ష రక్షణ లేకుండా ప్రయాణించవచ్చు.",
                "mr": f"**नाही, आज {city} मध्ये छत्रीची गरज नाही.**\n\n* **पावसाचा अंदाज:** केवळ {rain_mm:.1f} मिमी आणि तुरळक सरींची **{rain_prob_pct}% शक्यता** आहे.\n* **परिस्थिती:** हवामान मुख्यतः कोरडे राहील. आपण छत्रीशिवाय प्रवास करू शकता.",
                "gu": f"**ના, આજે {city}માં છત્રીની જરૂર નથી.**\n\n* **વરસાદની આગાહી:** માત્ર {rain_mm:.1f} મીમી અને હળવા ઝાપટાંની **{rain_prob_pct}% સંભાવના** છે.\n* **પરિસ્થિતિ:** આકાશ મુખ્યત્વે સૂકું રહેશે. તમે વરસાદના સાધનો વિના મુસાફરી કરી શકો છો.",
            }
            return umbrella_no.get(lang, umbrella_no["hi"])

    # 2. Agriculture / Pesticide / Crop Query
    if re.search(r"\b(pesticide|spray|spraying|crop|crops|wheat|paddy|rice|cotton|mustard|potato|irrigate|irrigation|farming|sowing|soil)\b", q_lower):
        if is_rainy:
            pesticide_delay = {
                "hi": f"**आज {city} में रासायनिक कीटनाशकों का छिड़काव रोक दें।**\n\n* **कारण:** {rain_mm:.1f} मिमी बारिश ({rain_prob_pct}% संभावना) का अनुमान है। कीटनाशक बह जाने से प्रभाव समाप्त हो जाएगा।\n* **सिफारिश:** शुष्क मौसम लौटने तक छिड़काव टालें।",
                "bn": f"**আজ {city}-তে রাসায়নিক কীটনাশক স্প্রে করা স্থগিত রাখুন।**\n\n* **কারণ:** {rain_mm:.1f} মিমি বৃষ্টি ({rain_prob_pct}% সম্ভাবনা) পূর্বাভাসিত। রাসায়নিক ধুয়ে যাওয়ায় কার্যকারিতা নষ্ট হবে।\n* **সুপারিশ:** আবহাওয়া শুষ্ক না হওয়া পর্যন্ত প্রয়োগ বিলম্বিত করুন।",
                "ta": f"**இன்று {city}-ல் ரசாயன பூச்சிக்கொல்லி தெளிப்பதைத் தவிர்க்கவும்.**\n\n* **காரணம்:** {rain_mm:.1f} மி.மீ மழை ({rain_prob_pct}% வாய்ப்பு) கணிக்கப்பட்டுள்ளது. மழைநீரில் மருந்து அடித்துச் செல்லப்படும்.\n* **பரிந்துரை:** வறண்ட வானிலை திரும்பும் வரை தெளிப்பதைத் தள்ளிப்போடுங்கள்.",
                "te": f"**ఈరోజు {city}లో రసాయన పురుగుమందుల పిచికారీని వాయిదా వేయండి.**\n\n* **కారణం:** {rain_mm:.1f} మి.మీ వర్షం ({rain_prob_pct}% సంభావ్యత) అంచనా వేయబడింది. వర్షానికి మందు కొట్టుకుపోయే ప్రమాదం ఉంది.\n* **సిఫార్సు:** పొడి వాతావరణం ఏర్పడే వరకు పిచికారీ చేయవద్దు.",
                "mr": f"**आज {city} मध्ये रासायनिक कीटकनाशक फवारणी थांबवा.**\n\n* **कारण:** {rain_mm:.1f} मिमी पाऊस ({rain_prob_pct}% शक्यता) अपेक्षित आहे. फवारलेले औषध वाहून जाऊन नुकसान होईल.\n* **शिफारस:** हवामान कोरडे होईपर्यंत फवारणी पुढे ढकला.",
                "gu": f"**આજે {city}માં રાસાયણિક જંતુનાશકોનો છંટકાવ મોકૂફ રાખો.**\n\n* **કારણ:** {rain_mm:.1f} મીમી વરસાદ ({rain_prob_pct}% સંભાવના) ની આગાહી છે. દવા ધોવાઈ જવાથી અસર થશે નહીં.\n* **ભલામણ:** હવામાન સૂકું ન થાય ત્યાં સુધી છંટકાવ મોકૂફ રાખો.",
            }
            return pesticide_delay.get(lang, pesticide_delay["hi"])
        else:
            pesticide_ok = {
                "hi": f"**आज {city} में खेत में कीटनाशक छिड़काव की स्थिति अनुकूल है।**\n\n* **पूर्वानुमान:** बारिश की संभावना कम ({rain_prob_pct}%) है और हवा की गति {metrics['wind_speed_kmh']} किमी/घंटा है (सुरक्षित सीमा < 15 किमी/घंटा)।\n* **मृदा नमी:** मृदा नमी घाटा सूचकांक {metrics['soil_moisture_deficit_index']} है।",
                "bn": f"**আজ {city}-তে জমিতে স্প্রে করার অনুকূল পরিস্থিতি রয়েছে।**\n\n* **পূর্বাভাস:** বৃষ্টির সম্ভাবনা কম ({rain_prob_pct}%), বাতাসের গতি {metrics['wind_speed_kmh']} কিমি/ঘণ্টা যা নিরাপদ সীমার মধ্যে রয়েছে।\n* **মাটির আর্দ্রতা:** মাটির আর্দ্রতা ঘাটতি সূচক {metrics['soil_moisture_deficit_index']}।",
                "ta": f"**இன்று {city}-ல் வயல்வெளி தெளிப்புக்கு சாதகமான சூழல் நிலவுகிறது.**\n\n* **முன்னறிவிப்பு:** மழைக்கான வாய்ப்பு குறைவு ({rain_prob_pct}%), காற்றின் வேகம் {metrics['wind_speed_kmh']} கி.மீ/மணி (பாதுகாப்பான வரம்பிற்குள் உள்ளது).\n* **மண் ஈரப்பதம்:** மண் ஈரப்பதம் பற்றாக்குறை குறியீடு {metrics['soil_moisture_deficit_index']} ஆகும்.",
                "te": f"**ఈరోజు {city}లో పొలంలో పిచికారీ చేయడానికి అనుకూలమైన పరిస్థితులు ఉన్నాయి.**\n\n* **సూచన:** వర్షపాతం సంభావ్యత తక్కువ ({rain_prob_pct}%), గాలి వేగం {metrics['wind_speed_kmh']} కి.మీ/గం (సురక్షిత పరిమితిలో ఉంది).\n* **నేల తేమ:** నేల తేమ లోటు సూచిక {metrics['soil_moisture_deficit_index']}.",
                "mr": f"**आज {city} मध्ये शेतात फवारणीसाठी अनुकूल परिस्थिती आहे.**\n\n* **अंदाज:** पावसाची शक्यता कमी ({rain_prob_pct}%) आहे आणि वाऱ्याचा वेग {metrics['wind_speed_kmh']} किमी/तास आहे (सुरक्षित मर्यादेत).\n* **जमिनीतील ओलावा:** मातीचा ओलावा निर्देशांक {metrics['soil_moisture_deficit_index']} आहे.",
                "gu": f"**આજે {city}માં ખેતરમાં છંટકાવ માટે સાનુકૂળ પરિસ્થિતિ છે.**\n\n* **આગાહી:** વરસાદની શક્યતા ઓછી ({rain_prob_pct}%) છે અને પવનની ગતિ {metrics['wind_speed_kmh']} કિમી/કલાક છે (સલામત મર્યાદામાં).\n* **જમીનનો ભેજ:** જમીન ભેજ ખાધ સૂચકાંક {metrics['soil_moisture_deficit_index']} છે.",
            }
            return pesticide_ok.get(lang, pesticide_ok["hi"])

    # 3. Heat / Outdoor Activity Query
    if re.search(r"\b(heat|heatwave|hot|sun|outdoor|sports|jog|exercise)\b", q_lower):
        if t_max >= 38.0 or (wet_bulb and wet_bulb >= 28.0):
            heat_caution = {
                "hi": f"**आज {city} में बाहरी गतिविधियों के लिए सावधानी बरतने की सलाह दी जाती है।**\n\n* **उच्चतम तापमान:** {t_max:.1f}°C (महसूस होने वाला तापमान: {metrics['apparent_temp_max_c']:.1f}°C)।\n* **वेट-बल्ब तनाव:** {wet_bulb:.1f}°C, जो अत्यधिक थर्मल तनाव का संकेत देता है।\n* **परामर्श:** भारी श्रम या खेलकूद सुबह 9:00 बजे से पहले या शाम 5:30 बजे के बाद पुनर्निर्धारित करें।",
                "bn": f"**আজ {city}-তে বাইরের কর্মকাণ্ডের জন্য সতর্কতা অবলম্বন করার পরামর্শ দেওয়া হচ্ছে।**\n\n* **সর্বোচ্চ তাপমাত্রা:** {t_max:.1f}°C (অনুভূত তাপমাত্রা: {metrics['apparent_temp_max_c']:.1f}°C)।\n* **ওয়েট-বাল্ব চাপ:** {wet_bulb:.1f}°C, যা উচ্চ তাপজনিত চাপের ইঙ্গিত দেয়।\n* **পরামর্শ:** সকাল ৯:০০ এর আগে বা বিকেল ৫:৩০ এর পরে ভারী কাজকর্ম স্থানান্তর করুন।",
                "ta": f"**இன்று {city}-ல் வெளிப்புற நடவடிக்கைகளுக்கு எச்சரிக்கை தேவை.**\n\n* **அதிகபட்ச வெப்பநிலை:** {t_max:.1f}°C (உணரப்படும் வெப்பம்: {metrics['apparent_temp_max_c']:.1f}°C).\n* **ஈரக்குமிழ் அழுத்தம்:** {wet_bulb:.1f}°C, இது கடுமையான வெப்ப அழுத்தத்தைக் குறிக்கிறது.\n* **பரிந்துரை:** கடினமான பணிகளை காலை 9:00 மணி அல்லது மாலை 5:30 மணிக்கு திட்டமிடுங்கள்.",
                "te": f"**ఈరోజు {city}లో బహిరంగ కార్యకలాపాల కోసం జాగ్రత్త వహించాలని సూచించడమైనది.**\n\n* **గరిష్ట ఉష్ణోగ్రత:** {t_max:.1f}°C (అనిపించే వేడి: {metrics['apparent_temp_max_c']:.1f}°C).\n* **వెట్-బల్బ్ ఒత్తిడి:** {wet_bulb:.1f}°C, ఇది అధిక వేడి ఒత్తిడిని సూచిస్తుంది.\n* **సిఫార్సు:** కష్టమైన పనులను ఉదయం 9:00 లోపు లేదా సాయంత్రం 5:30 తర్వాత షెడ్యూల్ చేయండి.",
                "mr": f"**आज {city} मध्ये बाहेरील कामांसाठी काळजी घेण्याचा सल्ला दिला जातो.**\n\n* **कमाल तापमान:** {t_max:.1f}°C (भासमान तापमान: {metrics['apparent_temp_max_c']:.1f}°C).\n* **वेट-बल्ब ताण:** {wet_bulb:.1f}°C, जे वाढत्या उष्णतेच्या ताणाचे संकेत देते.\n* **सल्ला:** कष्टाची कामे सकाळी ९:०० पूर्वी किंवा संध्याकाळी ५:३० नंतर ठेवावीत.",
                "gu": f"**આજે {city}માં આઉટડોર પ્રવૃત્તિઓ માટે સાવચેતી રાખવાની સલાહ આપવામાં આવે છે.**\n\n* **મહત્તમ તાપમાન:** {t_max:.1f}°C (અનુભવાતું તાપમાન: {metrics['apparent_temp_max_c']:.1f}°C).\n* **વેટ-બલ્બ તણાવ:** {wet_bulb:.1f}°C, જે ઉચ્ચ થર્મલ તણાવ દર્શાવે છે.\n* **ભલામણ:** ભારે શ્રમ અથવા રમતગમત સવારે ૯:૦૦ પહેલાં અથવા સાંજે ૫:૩૦ પછી ગોઠવો.",
            }
            return heat_caution.get(lang, heat_caution["hi"])
        else:
            heat_ok = {
                "hi": f"**आज {city} में बाहरी गतिविधियां करने के लिए मौसम आरामदायक है।**\n\n* **अधिकतम तापमान:** {t_max:.1f}°C और सापेक्ष आर्द्रता {metrics['relative_humidity_pct']}% है। थर्मल तनाव सामान्य सीमा में है।",
                "bn": f"**আজ {city}-তে বাইরের কর্মকাণ্ডের জন্য আবহাওয়া আরামদায়ক।**\n\n* **সর্বোচ্চ তাপমাত্রা:** {t_max:.1f}°C এবং আপেক্ষিক আর্দ্রতা {metrics['relative_humidity_pct']}%। তাপীয় চাপ স্বাভাবিক সীমায় রয়েছে।",
                "ta": f"**இன்று {city}-ல் வெளிப்புற நடவடிக்கைகளுக்கு வானிலை இதமாக உள்ளது.**\n\n* **அதிகபட்ச வெப்பநிலை:** {t_max:.1f}°C மற்றும் ஈரப்பதம் {metrics['relative_humidity_pct']}%. வெப்ப அழுத்தம் இயல்பான வரம்பில் உள்ளது.",
                "te": f"**ఈరోజు {city}లో బహిరంగ కార్యకలాపాలకు వాతావరణం అనుకూలంగా ఉంది.**\n\n* **గరిష్ట ఉష్ణోగ్రత:** {t_max:.1f}°C మరియు తేమ {metrics['relative_humidity_pct']}%. వేడి ఒత్తిడి సాధారణ పరిమితిలోనే ఉంది.",
                "mr": f"**आज {city} मध्ये बाहेरील कामांसाठी हवामान आल्हाददायक आहे.**\n\n* **कमाल तापमान:** {t_max:.1f}°C आणि हवेतील ओलावा {metrics['relative_humidity_pct']}% आहे. उष्णतेचा ताण नियंत्रणात आहे.",
                "gu": f"**આજે {city}માં બહારની પ્રવૃત્તિઓ માટે હવામાન આરામદાયક છે.**\n\n* **મહત્તમ તાપમાન:** {t_max:.1f}°C અને સાપેક્ષ ભેજ {metrics['relative_humidity_pct']}% છે. ગરમી સામાન્ય મર્યાદામાં છે.",
            }
            return heat_ok.get(lang, heat_ok["hi"])

    # 4. Default Environmental Risk Summary
    summaries = {
        "hi": (
            f"### **{city}** के लिए पर्यावरणीय जोखिम सारांश ({persona_id} संदर्भ)\n\n"
            f"* **थर्मल स्थिति:** अधिकतम तापमान **{t_max:.1f}°C** (वेट-बल्ब: {wet_bulb or 'N/A'}°C), 90% अंतराल: {metrics['conformal_intervals_90pct']['temperature_range']}।\n"
            f"* **जल विज्ञान:** **{rain_mm:.1f} मिमी** वर्षा अपेक्षित है (**{rain_prob_pct}%** संभावना)।\n"
            f"* **वायु गुणवत्ता:** PM2.5 स्तर **{pm25:.1f} µg/m³** ({aqi_cat}) है।\n"
            f"* **समग्र जोखिम स्तर:** **{metrics['peak_risk_category']}** (कंपाउंड स्कोर {metrics['peak_compound_risk']:.1f}/100)।"
        ),
        "bn": (
            f"### **{city}**-এর পরিবেশগত ঝুঁকি সারসংক্ষেপ ({persona_id} প্রেক্ষাপট)\n\n"
            f"* **তাপমাত্রা:** সর্বোচ্চ **{t_max:.1f}°C** (ওয়েট-বাল্ব: {wet_bulb or 'N/A'}°C), ৯০% ইন্টারভাল: {metrics['conformal_intervals_90pct']['temperature_range']}।\n"
            f"* **বৃষ্টিপাত:** **{rain_mm:.1f} মিমি** বৃষ্টিপাত প্রত্যাশিত (**{rain_prob_pct}%** সম্ভাবনা)।\n"
            f"* **বাতাসের মান:** PM2.5 ঘনত্ব **{pm25:.1f} µg/m³** ({aqi_cat})।\n"
            f"* **সামগ্রিক ঝুঁকি:** **{metrics['peak_risk_category']}** (যৌগিক স্কোর {metrics['peak_compound_risk']:.1f}/100)।"
        ),
        "ta": (
            f"### **{city}**-க்கான சுற்றுச்சூழல் அபாய சுருக்கம் ({persona_id} சூழல்)\n\n"
            f"* **வெப்ப நிலை:** அதிகபட்சம் **{t_max:.1f}°C** (ஈரக்குமிழ்: {wet_bulb or 'N/A'}°C), 90% வரம்பு: {metrics['conformal_intervals_90pct']['temperature_range']}.\n"
            f"* **மழைப்பொழிவு:** **{rain_mm:.1f} மி.மீ** மழை எதிர்பார்க்கப்படுகிறது (**{rain_prob_pct}%** வாய்ப்பு).\n"
            f"* **காற்றின் தரம்:** PM2.5 அளவு **{pm25:.1f} µg/m³** ({aqi_cat}).\n"
            f"* **ஒட்டுமொத்த அபாய நிலை:** **{metrics['peak_risk_category']}** (கூட்டு மதிப்பெண் {metrics['peak_compound_risk']:.1f}/100)."
        ),
        "te": (
            f"### **{city}** కోసం పర్యావరణ ప్రమాద సారాంశం ({persona_id} సందర్భం)\n\n"
            f"* **ఉష్ణోగ్రత:** గరిష్టంగా **{t_max:.1f}°C** (వెట్-బల్బ్: {wet_bulb or 'N/A'}°C), 90% పరిధి: {metrics['conformal_intervals_90pct']['temperature_range']}.\n"
            f"* **వర్షపాతం:** **{rain_mm:.1f} మి.మీ** వర్షం అంచనా వేయబడింది (**{rain_prob_pct}%** అవకాశం).\n"
            f"* **గాలి నాణ్యత:** PM2.5 స్థాయి **{pm25:.1f} µg/m³** ({aqi_cat}).\n"
            f"* **మొత్తం ముప్పు స్థాయి:** **{metrics['peak_risk_category']}** (కాంపౌండ్ స్కోర్ {metrics['peak_compound_risk']:.1f}/100)."
        ),
        "mr": (
            f"### **{city}** साठी पर्यावरणीय जोखीम सारांश ({persona_id} संदर्भ)\n\n"
            f"* **तापमान स्थिती:** कमाल **{t_max:.1f}°C** (वेट-बल्ब: {wet_bulb or 'N/A'}°C), ९०% श्रेणी: {metrics['conformal_intervals_90pct']['temperature_range']}.\n"
            f"* **पाऊस:** **{rain_mm:.1f} मिमी** पाऊस अपेक्षित आहे (**{rain_prob_pct}%** शक्यता).\n"
            f"* **हवेची गुणवत्ता:** PM2.5 पातळी **{pm25:.1f} µg/m³** ({aqi_cat}) आहे.\n"
            f"* **एकूण जोखीम पातळी:** **{metrics['peak_risk_category']}** (संयुक्त स्कोअर {metrics['peak_compound_risk']:.1f}/100)."
        ),
        "gu": (
            f"### **{city}** માટે પર્યાવરણીય જોખમ સારાંશ ({persona_id} સંદર્ભ)\n\n"
            f"* **તાપમાન:** મહત્તમ **{t_max:.1f}°C** (વેટ-બલ્બ: {wet_bulb or 'N/A'}°C), ૯૦% મર્યાદા: {metrics['conformal_intervals_90pct']['temperature_range']}.\n"
            f"* **વરસાદ:** **{rain_mm:.1f} મીમી** વરસાદની અપેક્ષા છે (**{rain_prob_pct}%** સંભાવના).\n"
            f"* **હવાની ગુણવત્તા:** PM2.5 સ્તર **{pm25:.1f} µg/m³** ({aqi_cat}) છે.\n"
            f"* **એકંદરે જોખમ સ્તર:** **{metrics['peak_risk_category']}** (સંયુક્ત સ્કોર {metrics['peak_compound_risk']:.1f}/100)."
        ),
    }
    return summaries.get(lang, summaries["hi"])


def generate_deterministic_synthesis(
    query: str,
    pred: LivePredictionResult,
    persona_id: str,
    retrieved_standards: Optional[List[Dict[str, Any]]] = None,
    language: str = "en",
) -> str:
    """Intelligent rule-based synthesis answering conversational & weather queries when Gemini API quota is unavailable."""
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
    tomorrow = pred.daily_predictions[1] if len(pred.daily_predictions) > 1 else pred.daily_predictions[0]
    personas_dict = today.personas or {}
    persona_data = personas_dict.get(persona_id, {})
    persona_title = persona_data.get("persona_title", persona_id.replace("_", " ").title()) if isinstance(persona_data, dict) else persona_id.replace("_", " ").title()
    restrictions = persona_data.get("operational_restrictions", []) if isinstance(persona_data, dict) else []

    is_rainy = (rain_prob_pct >= 35) or (rain_mm >= 1.5)

    norm_lang = (language or "en").strip().lower()
    if norm_lang in {"hi", "bn", "ta", "te", "mr", "gu"}:
        return _generate_localized_deterministic_synthesis(
            query=query,
            pred=pred,
            persona_id=persona_id,
            metrics=metrics,
            is_rainy=is_rainy,
            language=norm_lang,
        )

    # 1. Platform Overview / "About ClimaCore" / "What is Clima Core"
    if re.search(r"\b(clima[- ]?core|climacore|about|what is this|tell me about|platform|features|capabilities|how does it work)\b", q_lower):
        return (
            f"### ClimaCore Intelligence Platform\n\n"
            f"**ClimaCore** is an institutional-grade, physics-guided climate and weather intelligence system built specifically for Indian cities, critical infrastructure, and high-vulnerability communities.\n\n"
            f"**Key Capabilities for {city} ({persona_title}):**\n"
            f"* **Physics-Calibrated Risk Models:** Real-time SCS surface runoff hydrology, Wet-Bulb Globe Temperature (WBGT) heat strain, CPCB-standard multipollutant air quality, and power grid peak load surge modeling.\n"
            f"* **Sector-Specific Directives:** Personalized risk metrics and actionable standard operating procedures calibrated for Farmers, Gig Workers, Commuters, Students, and Industrial Operations.\n"
            f"* **Counterfactual Scenario Stress-Testing:** Type `/whatif` to simulate cloudbursts or heat spikes, `/compare` to assess risk between Indian cities, or `/plan` to generate emergency operational SOPs.\n"
            f"* **Verified Environmental Telemetry:** Grounded in high-resolution IMD, ECMWF, and CPCB observational feeds with conformal 90% confidence uncertainty bounds."
        )

    # 2. Greeting / Introduction Queries ("Hi", "Hello", "Hey", "Good morning", "Help")
    if re.search(r"\b(hi|hello|hey|greetings|namaste|good morning|good afternoon|good evening|who are you|what can you do|help)\b", q_lower) and len(q_lower.split()) <= 7:
        return (
            f"**Hello! I am your ClimaCore Operational Intelligence Assistant.**\n\n"
            f"I am actively monitoring live physical environmental telemetry for **{city}** calibrated for the **{persona_title}** sector.\n\n"
            f"**Current Status for {city}:**\n"
            f"* **Today's Weather:** High of **{t_max:.1f}°C**, Precipitation: **{rain_mm:.1f} mm** ({rain_prob_pct}% probability), PM2.5: **{pm25:.1f} µg/m³** ({aqi_cat}).\n"
            f"* **Threat Level:** **{metrics['peak_risk_category']}** (Compound score {metrics['peak_compound_risk']:.1f}/100).\n\n"
            f"Feel free to ask me questions like *'Will it be sunny tomorrow?'*, *'Should I carry an umbrella today?'*, *'What is the air quality outlook?'*, or use operational commands like `/plan`, `/whatif`, or `/compare`."
        )

    # 3. Tomorrow / Future Weather Queries ("Will it be sunny tomorrow?", "Tomorrow weather", "Rain tomorrow")
    if re.search(r"\b(tomorrow|tommorow|tomorow|next day|kal)\b", q_lower):
        m1 = tomorrow.forecast_metrics
        t_max1 = float(m1.get("temp_max", 30.0) or 30.0)
        t_min1 = float(m1.get("temp_min", 20.0) or 20.0)
        apparent1 = float(m1.get("apparent_temp_max", t_max1) or t_max1)
        rain_mm1 = float(m1.get("precipitation_mm", 0.0) or 0.0)
        raw_p1 = float(m1.get("rain_probability", 0.0) or 0.0)
        rain_p1 = int(raw_p1 if raw_p1 > 1.0 else raw_p1 * 100.0)
        wind1 = float(m1.get("wind_speed_kmh", 12.0) or 12.0)
        pm25_1 = float(m1.get("pm2_5", 35.0) or 35.0)
        is_sunny_tomorrow = (rain_p1 < 25 and rain_mm1 < 1.0)
        is_rainy_tomorrow = (rain_p1 >= 35 or rain_mm1 >= 1.5)

        # Specific query intentions for tomorrow
        if re.search(r"\b(umbrella|carry.*umbrella|need.*umbrella)\b", q_lower):
            if is_rainy_tomorrow:
                lead = f"**Yes, you should definitely carry an umbrella tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city}.**"
            else:
                lead = f"**No, you will not need an umbrella tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city}.**"
        elif re.search(r"\b(rain|raining|rainy|shower|showers|downpour|drizzle|precipitation)\b", q_lower):
            if is_rainy_tomorrow:
                lead = f"**Yes, rain is expected tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city}.**"
            else:
                lead = f"**No, rain is not expected tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city}.**"
        elif re.search(r"\b(sunny|sun|clear|bright)\b", q_lower):
            if is_sunny_tomorrow:
                lead = f"**Yes, tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city} will be predominantly sunny and clear!**"
            else:
                lead = f"**No, tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city} will be mostly overcast with a chance of precipitation.**"
        elif re.search(r"\b(hot|cold|temperature|temp|degrees)\b", q_lower):
            lead = f"**Tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city}: High of {t_max1:.1f}°C and low of {t_min1:.1f}°C.**"
        elif is_rainy_tomorrow:
            lead = f"**Tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city} will see overcast skies and expected showers.**"
        else:
            lead = f"**Tomorrow ({tomorrow.day_name}, {tomorrow.date}) in {city} will have generally clear and stable conditions.**"

        condition_desc = "Sunny and clear skies" if is_sunny_tomorrow else ("Scattered rain and overcast" if is_rainy_tomorrow else "Partly cloudy with dry spells")

        return (
            f"{lead}\n\n"
            f"* **Sky Condition:** {condition_desc}\n"
            f"* **Temperature Range:** High of **{t_max1:.1f}°C** (Low: **{t_min1:.1f}°C**, Feels like: **{apparent1:.1f}°C**)\n"
            f"* **Precipitation:** **{rain_mm1:.1f} mm** expected (**{rain_p1}%** rain probability)\n"
            f"* **Wind & Atmosphere:** Wind speed around **{wind1:.1f} km/h**; PM2.5 is **{pm25_1:.1f} µg/m³**\n"
            f"* **Operational Advisory ({persona_title}):** "
            + ("Dry, sunny conditions favor normal field and commute operations." if is_sunny_tomorrow else ("Plan for moisture protection and check low-lying transit routes." if is_rainy_tomorrow else "Standard operational parameters apply."))
        )

    # 4. Sunny / Clear / General Weather Outlook for Today
    if re.search(r"\b(sunny|clear sky|clear skies|how is the weather|weather today|weather outlook|is it hot|is it warm|climate today)\b", q_lower):
        if not is_rainy and rain_prob_pct < 25:
            lead = f"**Yes, conditions in {city} are sunny and clear today!**"
            condition_desc = "predominantly sunny with clear visibility"
        elif is_rainy:
            lead = f"**Conditions in {city} today are overcast with rain expected.**"
            condition_desc = f"cloudy with {rain_mm:.1f} mm of rainfall ({rain_prob_pct}% probability)"
        else:
            lead = f"**Conditions in {city} today are mild and partly cloudy.**"
            condition_desc = "partly cloudy with dry conditions"

        return (
            f"{lead}\n\n"
            f"* **Current State:** Skies are {condition_desc}.\n"
            f"* **Temperature:** High of **{t_max:.1f}°C** (Apparent feel: **{metrics['apparent_temp_max_c']:.1f}°C**, Wet-Bulb: {wet_bulb or 'N/A'}°C).\n"
            f"* **Precipitation Risk:** **{rain_mm:.1f} mm** ({rain_prob_pct}% probability).\n"
            f"* **Air Quality:** PM2.5 is **{pm25:.1f} µg/m³** ({aqi_cat}).\n"
            f"* **Sector Context ({persona_title}):** Overall operational threat level is **{metrics['peak_risk_category']}** (Compound score {metrics['peak_compound_risk']:.1f}/100)."
        )

    # 5. Agriculture / Pesticide / Crop Query
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

    # 6. Water Quality / Chemical Potability / Aquifer Contamination (RAG Driven)
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

    # 7. Critical Infrastructure, Railway Track Safety & Energy Grid (RAG Driven)
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

    # 8. Air Quality & GRAP Emergency Countermeasures (RAG Driven)
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

    # 9. Umbrella / Rain Query
    if re.search(r"\b(umbrella|rain|raining|rainy|shower|showers|monsoon|deluge|pour|downpour|drizzle|precipitation)\b", q_lower):
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

    # 10. Heat / Outdoor Activity Query
    if re.search(r"\b(heat|heatwave|hot|sun|thermal|outdoor|sports|jog|exercise)\b", q_lower):
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

    # 11. Default Comprehensive Synthesis (only when query has no specific match)
    lines = [
        f"### Environmental Risk Summary for **{city}** ({persona_title} Context)\n",
        f"* **Thermal State:** High of **{t_max:.1f}°C** (Wet-Bulb: {wet_bulb or 'N/A'}°C), 90% confidence range: {metrics['conformal_intervals_90pct']['temperature_range']}.",
        f"* **Hydrology:** **{rain_mm:.1f} mm** precipitation expected with **{rain_prob_pct}%** probability.",
        f"* **Air Quality:** PM2.5 is **{pm25:.1f} µg/m³** ({aqi_cat}).",
        f"* **Overall Threat Level:** **{metrics['peak_risk_category']}** (Compound score {metrics['peak_compound_risk']:.1f}/100).",
    ]
    if restrictions:
        lines.append(f"\n**Active Operational Directives:**\n" + "\n".join([f"- ⚠️ {r}" for r in restrictions]))

    if retrieved_standards:
        top_std = retrieved_standards[0]
        if top_std.get("relevance_score", 0.0) >= 0.50:
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
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": fn_name,
                    "response": {"output": tool_result}
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

        # Fallback formatting if second call is throttled or empty
        if fn_name == "convene_crisis_council" and isinstance(tool_result, dict):
            c_lines = [
                f"### Autonomous Multi-Agent Crisis Council (ICS War Room): **{tool_result.get('city', '')}**\n",
                f"* **Incident Command Reference:** `{tool_result.get('incident_id', '')}` ({tool_result.get('ics_form_reference', '')})",
                f"* **Cryptographic Verification Hash:** `{tool_result.get('verification_hash', '')}` (SHA-256)",
                f"* **Incident Type & Severity:** **{tool_result.get('incident_type', '').replace('_', ' ')}** | `{tool_result.get('severity_tier', '')}`",
                f"* **Life Preservation Score:** **{tool_result.get('life_preservation_index', 0):.1f}/100** | **Economic Continuity Score:** **{tool_result.get('economic_continuity_index', 0):.1f}/100**\n",
            ]
            if tool_result.get("mermaid_sequence_diagram"):
                c_lines.append("```mermaid")
                c_lines.append(tool_result["mermaid_sequence_diagram"])
                c_lines.append("```\n")

            c_lines.append("\n**Executive Incident Commander Directives (Approved ICS-201 Policy):**")
            for d in tool_result.get("executive_directives", []):
                c_lines.append(f"- 📜 **{d.get('directive_id')} ({d.get('title')}):** {d.get('mandate')}")

            return "\n".join(c_lines), dispatched_action, executed_tools

        if fn_name == "replay_historical_disaster" and isinstance(tool_result, dict):
            d_lines = [
                f"### Historical Disaster Replay Time Machine: **{tool_result.get('name', '')}**\n",
                f"* **Benchmark ID:** `{tool_result.get('disaster_id', '')}` | **Date:** {tool_result.get('historical_date', '')}",
                f"* **Advance Warning Delivered:** **+{tool_result.get('total_advance_warning_hours', 0):.0f} Hours** in advance lead time",
                f"* **Lives Preserved Estimate:** **{tool_result.get('lives_protected_estimate', 0):,} citizens**",
                f"* **Parametric DBT Claims Disbursed:** **₹{tool_result.get('simulated_parametric_payout_inr_cr', 0):.0f} Crores** instant voucher\n",
            ]
            if tool_result.get("mermaid_timeline_chart"):
                d_lines.append(f"\n```mermaid\n{tool_result['mermaid_timeline_chart']}\n```")
            return "\n".join(d_lines), dispatched_action, executed_tools

        if fn_name == "compare_cities_risk" and isinstance(tool_result, dict):
            ca = tool_result.get("city_a", {})
            cb = tool_result.get("city_b", {})
            kd = tool_result.get("key_differences", {})
            comp_lines = [
                f"### Comparative Environmental Risk Evaluation: **{ca.get('name')}** vs **{cb.get('name')}**\n",
                f"* **{ca.get('name')}:** Max Temp: {ca.get('temperature_max_c')}°C | Rain: {ca.get('rainfall_mm')} mm | PM2.5: {ca.get('pm25_ug_m3')} µg/m³ | Risk: **{ca.get('risk_category')}** ({ca.get('compound_risk_score')})",
                f"* **{cb.get('name')}:** Max Temp: {cb.get('temperature_max_c')}°C | Rain: {cb.get('rainfall_mm')} mm | PM2.5: {cb.get('pm25_ug_m3')} µg/m³ | Risk: **{cb.get('risk_category')}** ({cb.get('compound_risk_score')})",
                f"* **Key Differences:** Δ Temp: {kd.get('temp_delta_c')}°C | Δ Rain: {kd.get('rain_delta_mm')} mm | Δ PM2.5: {kd.get('pm25_delta')} µg/m³\n",
            ]
            return "\n".join(comp_lines), dispatched_action, executed_tools

        if fn_name == "simulate_what_if_scenario" and isinstance(tool_result, dict):
            city = tool_result.get("city", "")
            inputs = tool_result.get("inputs", {})
            impacts = tool_result.get("simulated_impacts", {})
            w_lines = [
                f"### Counterfactual Simulation: **{city}**\n",
                f"* **Physical Inputs:** Max Temp: {inputs.get('temperature_max_c')}°C | Rainfall: {inputs.get('rainfall_mm')} mm | PM2.5: {inputs.get('pm25_ug_m3')} µg/m³",
                f"* **SCS Surface Runoff:** {impacts.get('effective_surface_runoff_mm', 0):.1f} mm ({impacts.get('runoff_surge_multiplier', '')})",
                f"* **Hydrological Hazard Probability:** {impacts.get('flood_hazard_probability', 0):.1%}",
                f"* **Wet-Bulb Temperature:** {impacts.get('wet_bulb_temp_c', 0):.1f}°C (Apparent Heat Index: {impacts.get('apparent_heat_index_c', 0):.1f}°C)",
                f"* **Power Grid Stress & Peak Load:** {impacts.get('peak_grid_load_mw', 0):,.0f} MW (Δ: {impacts.get('grid_load_delta_mw', 0):+,.0f} MW | Blackout Risk: {impacts.get('blackout_stress_probability', 0):.1%})",
                f"* **Air Quality Category:** `{impacts.get('air_quality_category', 'MODERATE')}`\n",
            ]
            persona_sim = tool_result.get("persona_simulation")
            if persona_sim:
                w_lines.append(f"**Persona Impact Level:** `{persona_sim.get('status_level')}` (Score: {persona_sim.get('impact_score')}/100)")
                for adv in persona_sim.get("advisories", []):
                    w_lines.append(f"- {adv}")
            return "\n".join(w_lines), dispatched_action, executed_tools

        if fn_name == "evaluate_parametric_crop_claim" and isinstance(tool_result, dict):
            v_lines = [
                f"### Parametric Insurance Settlement Voucher: **{tool_result.get('voucher_id', '')}**\n",
                f"* **Payout Status:** `{tool_result.get('payout_status', '')}`",
                f"* **Crop & Season:** **{tool_result.get('crop', '')}** ({tool_result.get('season', '')})",
                f"* **Net Claim Payout:** **₹{tool_result.get('net_claim_payout_inr', 0):,.2f}** ({tool_result.get('payout_percentage', 0)}% of sum insured)",
                f"* **Statistical Guarantee:** {tool_result.get('statistical_guarantee', '')}",
                f"* **Cryptographic Verification Hash:** `{tool_result.get('verification_hash', '')}` (SHA-256)\n",
            ]
            return "\n".join(v_lines), dispatched_action, executed_tools

        if fn_name == "simulate_compound_domino_cascade" and isinstance(tool_result, dict):
            dom_lines = [
                f"### Compound Domino Cascade Simulation: **{tool_result.get('city', '')}**\n",
                f"* **System Vulnerability Index:** **{tool_result.get('system_vulnerability_index', 0):.1f}/100**",
                f"* **Cascade Stages:** {len(tool_result.get('stages', []))} Sequential Failure Propagation steps\n",
            ]
            if tool_result.get("mermaid_flowchart"):
                dom_lines.append(f"```mermaid\n{tool_result['mermaid_flowchart']}\n```\n")
            return "\n".join(dom_lines), dispatched_action, executed_tools

        if fn_name == "explain_risk_attribution_shap" and isinstance(tool_result, dict):
            shap_lines = [
                f"### TreeSHAP Mathematical Risk Attribution: **{tool_result.get('city', '')}**\n",
                f"* **Model ID:** `{tool_result.get('model_id', '')}` | Target: `{tool_result.get('target_unit', '')}`",
                f"* **Additivity Error:** `{tool_result.get('additivity_error', '0.0')}`",
                f"\n**Top Amplifying Risk Drivers:**",
            ]
            for feat in tool_result.get("top_amplifying_features", []):
                shap_lines.append(f"- **{feat.get('feature_name')}:** +{feat.get('shap_value', 0):.3f} (Val: {feat.get('feature_value')})")
            return "\n".join(shap_lines), dispatched_action, executed_tools

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
        req.message = (req.message or req.query or req.prompt or "").strip()
        metrics = build_metrics_summary(prediction_result, req.persona)

        # 0. Retrieve Institutional Regulatory Standards via Federated Multi-Shard RAG
        retrieved_standards = retrieve_relevant_standards(
            query=req.message,
            persona=req.persona,
            city=prediction_result.city,
            top_k=2,
        )

        norm_lang = (req.language or "en").strip().lower()

        system_prompt = build_grounded_system_prompt(
            pred=prediction_result,
            persona_id=req.persona,
            retrieved_standards=retrieved_standards,
            language=norm_lang,
        )
        followups = generate_suggested_followups(prediction_result, req.persona, language=norm_lang)

        # Initialize tool executor
        from climacore.ingestion.engine import LiveForecastEngine
        active_live_engine = live_engine if isinstance(live_engine, LiveForecastEngine) else LiveForecastEngine()
        executor = AgentToolExecutor(live_engine=active_live_engine)

        ai_text = None
        action = None
        tool_calls: List[Dict[str, Any]] = []
        provider = "climacore-deterministic-agent"

        # Explicit operational command tools (/plan, /whatif, /compare)
        norm_cmd = req.message.strip().lower()
        if norm_cmd.startswith(("/plan", "/whatif", "/compare")):
            cmd_result = detect_and_execute_offline_scenario(
                query=req.message,
                current_city=prediction_result.city,
                current_persona=req.persona,
                executor=executor,
            )
            if cmd_result is not None:
                ai_text, action, tool_data = cmd_result
                tool_calls.append({"name": "operational_command_engine", "data": tool_data})
                provider = "climacore-intelligence-engine"

        # 1. Attempt Gemini Agent Call with Native Function Calling
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not ai_text and key:
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
                    language=norm_lang,
                )
                provider = "climacore-deterministic-synthesis"

        latency_ms = (time.perf_counter() - t0) * 1000.0

        if ai_text:
            ai_text = ai_text.replace(chr(8212), " - ")
            ai_text = ai_text.replace("Gemini 1.5 Flash", "ClimaCore Intelligence").replace("Gemini", "ClimaCore Intelligence")

        clean_provider = "ClimaCore Intelligence"

        return AssistantChatResponse(
            response=ai_text or "ClimaCore Intelligence synthesized operational response.",
            city=prediction_result.city,
            persona=req.persona,
            metrics_used=metrics,
            suggested_followups=followups,
            action=action,
            tool_calls=tool_calls,
            retrieved_standards=retrieved_standards,
            provider=clean_provider,
            latency_ms=round(latency_ms, 2),
            language=norm_lang,
        )
