"""
Clima-Core Agent Tool Declarations and Execution Engine.
Empowers the AI Assistant to act as an autonomous environmental risk agent with capabilities for:
1. Counterfactual / "What-If" Scenario Simulation (rain, heat, wind, PM2.5).
2. Multi-City Comparison & Route Risk Analysis.
3. Groundwater & Aquifer Chemical Potability Deep Dives.
4. Client-side Dashboard Action & Viewport Navigation.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from climacore.common.logging import get_logger
from climacore.ingestion.engine import LiveForecastEngine
from climacore.ingestion.geocoding import resolve_location
from climacore.personas.models import PersonaInput
from climacore.personas.engine import evaluate_persona, PERSONA_REGISTRY

logger = get_logger("climacore.service.agent_tools")

# -------------------------------------------------------------------------
# 1. Native Gemini Function Declarations (JSON Schema)
# -------------------------------------------------------------------------
AGENT_FUNCTION_DECLARATIONS = [
    {
        "name": "simulate_what_if_scenario",
        "description": "Simulate counterfactual environmental and climate risk scenarios for any Indian city under hypothetical extreme temperature, cloudburst rainfall, wind gusts, humidity, or air pollution. Runs physical SCS-CN runoff, heat index, power grid stress, and persona impact calculations.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {
                    "type": "STRING",
                    "description": "Name of the Indian city (e.g. 'Mumbai', 'Delhi', 'Nagpur')"
                },
                "hypothetical_rainfall_mm": {
                    "type": "NUMBER",
                    "description": "Hypothetical 24h precipitation in mm (e.g. 100.0 or 150.0 for a cloudburst, 0.0 for drought)"
                },
                "hypothetical_temp_max_c": {
                    "type": "NUMBER",
                    "description": "Hypothetical daily maximum air temperature in °C (e.g. 45.0 or 48.0 for a severe heatwave)"
                },
                "hypothetical_temp_min_c": {
                    "type": "NUMBER",
                    "description": "Hypothetical daily minimum air temperature in °C (e.g. 32.0 for tropical nights)"
                },
                "hypothetical_relative_humidity": {
                    "type": "NUMBER",
                    "description": "Hypothetical relative humidity in % (e.g. 85.0 for muggy heat stress)"
                },
                "hypothetical_wind_speed_kmh": {
                    "type": "NUMBER",
                    "description": "Hypothetical 10m wind speed in km/h (e.g. 75.0 for gale force winds)"
                },
                "hypothetical_pm25": {
                    "type": "NUMBER",
                    "description": "Hypothetical PM2.5 concentration in µg/m³ (e.g. 350.0 for hazardous smog crisis)"
                },
                "persona": {
                    "type": "STRING",
                    "description": "Occupation/sector to evaluate (e.g. 'farmer', 'student', 'commuter', 'healthcare', 'gig_worker')"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "compare_cities_risk",
        "description": "Compare real-time multi-hazard environmental risk, weather, and sector workability between two Indian cities (ideal for travel, logistics, supply chains, or regional analysis).",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city_a": {
                    "type": "STRING",
                    "description": "First Indian city name (e.g. 'Delhi')"
                },
                "city_b": {
                    "type": "STRING",
                    "description": "Second Indian city name (e.g. 'Jaipur')"
                },
                "persona": {
                    "type": "STRING",
                    "description": "Target occupation/persona for comparison (e.g. 'commuter', 'farmer', 'general')"
                }
            },
            "required": ["city_a", "city_b"]
        }
    },
    {
        "name": "get_groundwater_quality",
        "description": "Evaluate CGWB groundwater chemical potability, water quality index (WQI), and BIS IS 10500 compliance for any Indian district.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {
                    "type": "STRING",
                    "description": "Indian city or district name"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "dispatch_dashboard_action",
        "description": "Control the user interface of the frontend dashboard: navigate to a specific city, switch persona, focus on a specific hazard card (flood_risk, heat_risk, power_grid, air_quality, water_quality, agrisense), or zoom the map.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "target_city": {
                    "type": "STRING",
                    "description": "City to switch the dashboard to (e.g. 'Chennai')"
                },
                "target_persona": {
                    "type": "STRING",
                    "description": "Persona to switch the dashboard to (e.g. 'farmer')"
                },
                "active_card": {
                    "type": "STRING",
                    "description": "Card to focus on: 'flood_risk', 'heat_risk', 'power_grid', 'air_quality', 'water_quality', 'agrisense'"
                },
                "explanation": {
                    "type": "STRING",
                    "description": "Brief confirmation message of what UI change was triggered"
                }
            }
        }
    }
]


# -------------------------------------------------------------------------
# 2. Tool Execution Implementation
# -------------------------------------------------------------------------
class AgentToolExecutor:
    """Executes agent tool calls using Clima-Core's live models and physical engines."""

    def __init__(self, live_engine: LiveForecastEngine):
        self.engine = live_engine

    def execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously execute a requested tool by name."""
        logger.info(f"Agent executing tool: {name} with args: {args}")

        if name == "simulate_what_if_scenario":
            return self.simulate_what_if_scenario(
                city=args.get("city", "Mumbai"),
                hypothetical_rainfall_mm=args.get("hypothetical_rainfall_mm"),
                hypothetical_temp_max_c=args.get("hypothetical_temp_max_c"),
                hypothetical_temp_min_c=args.get("hypothetical_temp_min_c"),
                hypothetical_relative_humidity=args.get("hypothetical_relative_humidity"),
                hypothetical_wind_speed_kmh=args.get("hypothetical_wind_speed_kmh"),
                hypothetical_pm25=args.get("hypothetical_pm25"),
                persona=args.get("persona", "general"),
            )
        elif name == "compare_cities_risk":
            return self.compare_cities_risk(
                city_a=args.get("city_a", "Delhi"),
                city_b=args.get("city_b", "Jaipur"),
                persona=args.get("persona", "general"),
            )
        elif name == "get_groundwater_quality":
            return self.get_groundwater_quality(city=args.get("city", "Mumbai"))
        elif name == "dispatch_dashboard_action":
            return self.dispatch_dashboard_action(
                target_city=args.get("target_city"),
                target_persona=args.get("target_persona"),
                active_card=args.get("active_card"),
                explanation=args.get("explanation", "Updating dashboard view"),
            )
        else:
            return {"error": f"Unknown tool: '{name}'"}

    def simulate_what_if_scenario(
        self,
        city: str,
        hypothetical_rainfall_mm: Optional[float] = None,
        hypothetical_temp_max_c: Optional[float] = None,
        hypothetical_temp_min_c: Optional[float] = None,
        hypothetical_relative_humidity: Optional[float] = None,
        hypothetical_wind_speed_kmh: Optional[float] = None,
        hypothetical_pm25: Optional[float] = None,
        persona: str = "general",
    ) -> Dict[str, Any]:
        """Run counterfactual physical simulation under altered conditions."""
        # 1. Fetch baseline real forecast
        base_pred = self.engine.predict(query=city, persona=persona)
        base_day = base_pred.daily_predictions[0]
        base_m = base_day.forecast_metrics

        # 2. Merge baseline with hypothetical parameters
        t_max = float(hypothetical_temp_max_c if hypothetical_temp_max_c is not None else base_m.get("temp_max", 32.0))
        t_min = float(hypothetical_temp_min_c if hypothetical_temp_min_c is not None else base_m.get("temp_min", 24.0))
        rain_mm = max(0.0, float(hypothetical_rainfall_mm if hypothetical_rainfall_mm is not None else base_m.get("precipitation_mm", 0.0)))
        rh = float(hypothetical_relative_humidity if hypothetical_relative_humidity is not None else base_m.get("relative_humidity", 50.0))
        wind = float(hypothetical_wind_speed_kmh if hypothetical_wind_speed_kmh is not None else base_m.get("wind_speed_kmh", 12.0))
        pm25 = float(hypothetical_pm25 if hypothetical_pm25 is not None else base_m.get("pm2_5", 55.0))

        # 3. Run physical model simulators
        from climacore.analytics.psychrometrics import calculate_elevation_aware_wet_bulb
        wb_res = calculate_elevation_aware_wet_bulb(t_max, rh, city=city)
        hypo_wet_bulb = float(wb_res.get("wet_bulb_temp_c", 24.0))

        heat_res = self.engine.heat_engine.evaluate_heat_risk(
            city=city,
            temp_max_c=t_max,
            temp_min_c=t_min,
            relative_humidity_pct=rh,
            wind_speed_kmh=wind,
        )
        hypo_apparent_t = float(heat_res.get("apparent_temp_max_c", t_max))

        flood_res = self.engine.flood_engine.evaluate_flood_risk(
            city=city,
            rainfall_mm=rain_mm,
            rain_rolling_3d=round(rain_mm * 1.5, 1),
            smdi=base_m.get("soil_moisture_deficit_index", 0.0),
        )
        grid_res = self.engine.power_engine.evaluate_power_grid_risk(
            temp_max_c=t_max,
            temp_min_c=t_min,
            city=city,
        )
        air_res = self.engine.aq_engine.evaluate_air_quality(
            city=city,
            pm2_5_ug_m3=pm25,
            t_max=t_max,
            t_min=t_min,
            wind_speed_kmh=wind,
        )

        # Persona impact under hypothetical scenario
        norm_persona = persona.lower().strip()
        persona_result = None
        if norm_persona in PERSONA_REGISTRY:
            p_inp = PersonaInput(
                city=city,
                date=base_day.date,
                temp_max=t_max,
                temp_min=t_min,
                precipitation_mm=rain_mm,
                rain_probability=1.0 if rain_mm > 5.0 else 0.1,
                pm2_5=pm25,
                wind_speed_kmh=wind,
                relative_humidity=rh,
                apparent_temp_max=hypo_apparent_t,
                wet_bulb_temp_c=hypo_wet_bulb,
            )
            p_res = evaluate_persona(norm_persona, p_inp)
            persona_result = {
                "status_level": p_res.status_level,
                "impact_score": p_res.impact_score,
                "advisories": p_res.advisories[:3],
                "operational_restrictions": p_res.operational_restrictions[:3],
            }

        # Calculate deltas relative to baseline
        base_flood = base_day.flagship_intelligence.get("flood_risk", {}) if base_day.flagship_intelligence else {}
        base_grid = base_day.flagship_intelligence.get("power_grid_stress", {}) if base_day.flagship_intelligence else {}
        base_heat = base_day.flagship_intelligence.get("heat_risk", {}) if base_day.flagship_intelligence else {}

        baseline_runoff = float(base_flood.get("scs_runoff_mm", 0.0))
        hypo_runoff = float(flood_res.get("scs_runoff_mm", 0.0))
        runoff_multiplier = round(hypo_runoff / max(baseline_runoff, 0.5), 1)

        baseline_peak_mw = float(base_grid.get("peak_load_mw", 10000.0) or 10000.0)
        hypo_peak_mw = float(grid_res.get("peak_load_mw", baseline_peak_mw) or baseline_peak_mw)
        grid_mw_delta = round(hypo_peak_mw - baseline_peak_mw, 1)

        return {
            "scenario": "COUNTERFACTUAL_SIMULATION",
            "city": city,
            "inputs": {
                "temperature_max_c": t_max,
                "temperature_min_c": t_min,
                "rainfall_mm": rain_mm,
                "relative_humidity_pct": rh,
                "wind_speed_kmh": wind,
                "pm25_ug_m3": pm25,
            },
            "baseline_comparison": {
                "baseline_temp_max_c": base_m.get("temp_max"),
                "baseline_rainfall_mm": base_m.get("precipitation_mm"),
                "baseline_runoff_mm": baseline_runoff,
                "baseline_peak_mw": baseline_peak_mw,
            },
            "simulated_impacts": {
                "effective_surface_runoff_mm": hypo_runoff,
                "runoff_surge_multiplier": f"{runoff_multiplier}x baseline",
                "flood_hazard_probability": float(flood_res.get("hydrological_hazard_probability", 0.0) or 0.0),
                "wet_bulb_temp_c": hypo_wet_bulb,
                "apparent_heat_index_c": hypo_apparent_t,
                "heatwave_alert_level": heat_res.get("heatwave_severity_code", 0),
                "peak_grid_load_mw": hypo_peak_mw,
                "grid_load_delta_mw": grid_mw_delta,
                "blackout_stress_probability": float(grid_res.get("unmet_supply_stress_probability", 0.0) or 0.0),
                "air_quality_category": air_res.get("air_quality_category", "MODERATE"),
            },
            "persona_simulation": persona_result,
        }

    def compare_cities_risk(self, city_a: str, city_b: str, persona: str = "general") -> Dict[str, Any]:
        """Fetch and structure side-by-side environmental risk comparisons between two cities."""
        res_a = self.engine.predict(query=city_a, persona=persona)
        res_b = self.engine.predict(query=city_b, persona=persona)

        day_a = res_a.daily_predictions[0]
        day_b = res_b.daily_predictions[0]

        ma = day_a.forecast_metrics
        mb = day_b.forecast_metrics

        return {
            "comparison": "MULTI_CITY_HAZARD_EVALUATION",
            "city_a": {
                "name": res_a.city,
                "state": res_a.state,
                "temperature_max_c": ma.get("temp_max"),
                "wet_bulb_temp_c": ma.get("wet_bulb_temp_c"),
                "rainfall_mm": ma.get("precipitation_mm"),
                "rain_probability_pct": int(float(ma.get("rain_probability", 0.0) or 0.0) * (1 if float(ma.get("rain_probability", 0.0) or 0.0) > 1.0 else 100)),
                "pm25_ug_m3": ma.get("pm2_5"),
                "compound_risk_score": res_a.peak_compound_risk,
                "risk_category": res_a.peak_risk_category,
            },
            "city_b": {
                "name": res_b.city,
                "state": res_b.state,
                "temperature_max_c": mb.get("temp_max"),
                "wet_bulb_temp_c": mb.get("wet_bulb_temp_c"),
                "rainfall_mm": mb.get("precipitation_mm"),
                "rain_probability_pct": int(float(mb.get("rain_probability", 0.0) or 0.0) * (1 if float(mb.get("rain_probability", 0.0) or 0.0) > 1.0 else 100)),
                "pm25_ug_m3": mb.get("pm2_5"),
                "compound_risk_score": res_b.peak_compound_risk,
                "risk_category": res_b.peak_risk_category,
            },
            "key_differences": {
                "temp_delta_c": round(float(ma.get("temp_max", 0)) - float(mb.get("temp_max", 0)), 1),
                "rain_delta_mm": round(float(ma.get("precipitation_mm", 0)) - float(mb.get("precipitation_mm", 0)), 1),
                "pm25_delta": round(float(ma.get("pm2_5", 0)) - float(mb.get("pm2_5", 0)), 1),
            }
        }

    def get_groundwater_quality(self, city: str) -> Dict[str, Any]:
        """Evaluate CGWB chemical groundwater quality for a target city."""
        loc = resolve_location(city)
        wq_res = self.engine.water_quality_engine.evaluate_water_quality(
            latitude=loc["latitude"],
            longitude=loc["longitude"],
            city=loc["city"],
        )
        return {
            "city": loc["city"],
            "state": loc.get("state", "India"),
            "water_quality": wq_res,
        }

    def dispatch_dashboard_action(
        self,
        target_city: Optional[str] = None,
        target_persona: Optional[str] = None,
        active_card: Optional[str] = None,
        explanation: str = "Updating dashboard state",
    ) -> Dict[str, Any]:
        """Dispatch a client-side dashboard UI action."""
        action: Dict[str, Any] = {
            "type": "SET_DASHBOARD_STATE",
            "explanation": explanation,
        }
        if target_city:
            action["target_city"] = target_city
        if target_persona:
            action["target_persona"] = target_persona.lower().strip()
        if active_card:
            action["active_card"] = active_card
        return action


# -------------------------------------------------------------------------
# 3. Deterministic Fallback Parser (For Offline / Zero-Key Execution)
# -------------------------------------------------------------------------
def detect_and_execute_offline_scenario(
    query: str,
    current_city: str,
    current_persona: str,
    executor: AgentToolExecutor,
) -> Optional[Tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]]:
    """Pattern match what-if scenarios, city comparisons, and UI commands when running without an LLM API key."""
    q = query.strip().lower()

    # 1. UI Navigation Commands
    # e.g. "show me Chennai as a farmer" or "switch to Delhi" or "focus on flood risk"
    if any(k in q for k in ["switch to", "show me", "navigate to", "open", "change city", "change persona"]):
        target_city = None
        target_persona = None
        active_card = None

        # Check for card keywords
        for card_key in ["flood", "heat", "power", "air", "water", "agri"]:
            if card_key in q:
                active_card = f"{card_key}_risk" if card_key in ["flood", "heat"] else f"{card_key}_quality" if card_key == "water" else card_key
                break

        # Check for persona keywords
        for p_key in PERSONA_REGISTRY.keys():
            if p_key in q:
                target_persona = p_key
                break

        # Check for city names in standard list
        from climacore.ingestion.geocoding import _load_standard_cities
        for c_cand in _load_standard_cities().keys():
            if len(c_cand) > 3 and c_cand in q:
                target_city = c_cand.capitalize()
                break

        if target_city or target_persona or active_card:
            action = executor.dispatch_dashboard_action(
                target_city=target_city,
                target_persona=target_persona,
                active_card=active_card,
                explanation=f"Navigating to {target_city or current_city} with {target_persona or current_persona} profile.",
            )
            text = (
                f"**Dashboard View Updated:**\n\n"
                f"* **Target Location:** {target_city or current_city}\n"
                f"* **Active Persona:** {(target_persona or current_persona).capitalize()}\n"
                f"* **Focused Risk Module:** {active_card or 'Overview'}\n\n"
                f"The interactive dashboard and map have been synced to this view."
            )
            return text, action, {"action_dispatched": action}

    # 2. Multi-City Comparison
    # e.g. "compare Delhi and Jaipur" or "compare Delhi and Jaipur for travel tomorrow"
    from climacore.ingestion.geocoding import _load_standard_cities, INDIAN_CITY_ALIASES
    reg_cities = _load_standard_cities()

    if any(k in q for k in ["compare", "comparison", "between", "versus", "vs"]):
        # Find which known cities are mentioned in the query
        words = re.findall(r"\b[a-zA-Z]+\b", q)
        found_cities = []
        stop_words = {"compare", "comparison", "between", "versus", "vs", "and", "with", "to", "for", "tomorrow", "travel", "weather", "risk"}
        for w in words:
            w_lower = w.lower()
            if w_lower in stop_words:
                continue
            if w_lower in reg_cities or w_lower in INDIAN_CITY_ALIASES:
                c_name = INDIAN_CITY_ALIASES.get(w_lower, w_lower).capitalize()
                if c_name not in found_cities:
                    found_cities.append(c_name)

        if len(found_cities) >= 2:
            city_1 = found_cities[0]
            city_2 = found_cities[1]
            try:
                comp_res = executor.compare_cities_risk(city_a=city_1, city_b=city_2, persona=current_persona)
                ca = comp_res["city_a"]
                cb = comp_res["city_b"]
                diff = comp_res["key_differences"]
                text = (
                    f"### Comparative Environmental Risk: **{ca['name']}** vs **{cb['name']}**\n\n"
                    f"| Metric | {ca['name']} | {cb['name']} | Difference |\n"
                    f"| :--- | :--- | :--- | :--- |\n"
                    f"| **Max Temperature** | {ca['temperature_max_c']}°C | {cb['temperature_max_c']}°C | {diff['temp_delta_c']:+.1f}°C |\n"
                    f"| **Wet-Bulb Temp** | {ca['wet_bulb_temp_c']}°C | {cb['wet_bulb_temp_c']}°C | — |\n"
                    f"| **24h Rainfall** | {ca['rainfall_mm']} mm ({ca['rain_probability_pct']}%) | {cb['rainfall_mm']} mm ({cb['rain_probability_pct']}%) | {diff['rain_delta_mm']:+.1f} mm |\n"
                    f"| **Air Quality (PM2.5)** | {ca['pm25_ug_m3']} µg/m³ | {cb['pm25_ug_m3']} µg/m³ | {diff['pm25_delta']:+.1f} µg/m³ |\n"
                    f"| **Compound Risk Score** | {ca['compound_risk_score']:.1f} ({ca['risk_category']}) | {cb['compound_risk_score']:.1f} ({cb['risk_category']}) | — |\n\n"
                    f"**Strategic Assessment:** "
                    f"{ca['name']} presents a {ca['risk_category'].lower()} hazard tier, while {cb['name']} is {cb['risk_category'].lower()}. "
                    f"Plan departures and activities around the less volatile window."
                )
                return text, None, comp_res
            except Exception as e:
                logger.warning(f"Offline city comparison failed: {e}")

    # 3. Counterfactual / What-If Scenarios
    # e.g. "what if rainfall is 120mm" or "what if temp reaches 46C"
    is_what_if = any(k in q for k in ["what if", "suppose", "imagine", "scenario", "hypothetical", "reaches", "surges to"])
    if is_what_if:
        hypo_rain = None
        hypo_temp = None
        hypo_pm25 = None

        # Extract rain (e.g. "120 mm" or "100mm")
        m_rain = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|millimeter)", q)
        if m_rain:
            hypo_rain = float(m_rain.group(1))

        # Extract temp (e.g. "45 C", "45 deg", "45 degrees")
        m_temp = re.search(r"(\d+(?:\.\d+)?)\s*(?:c|deg|degree|°c)", q)
        if m_temp:
            hypo_temp = float(m_temp.group(1))

        # Extract PM2.5
        m_pm = re.search(r"(\d+(?:\.\d+)?)\s*(?:pm|ug|µg)", q)
        if m_pm:
            hypo_pm25 = float(m_pm.group(1))

        if hypo_rain is not None or hypo_temp is not None or hypo_pm25 is not None:
            sim_res = executor.simulate_what_if_scenario(
                city=current_city,
                hypothetical_rainfall_mm=hypo_rain,
                hypothetical_temp_max_c=hypo_temp,
                hypothetical_pm25=hypo_pm25,
                persona=current_persona,
            )
            simp = sim_res["simulated_impacts"]
            base = sim_res["baseline_comparison"]

            lines = [
                f"### Counterfactual Simulation Results for **{current_city}**\n",
                f"Simulated under: **{f'{hypo_rain} mm Rain' if hypo_rain is not None else ''} {f'{hypo_temp}°C Temperature' if hypo_temp is not None else ''} {f'{hypo_pm25} µg/m³ PM2.5' if hypo_pm25 is not None else ''}**\n",
            ]

            if hypo_rain is not None:
                lines.append(
                    f"* **SCS Surface Runoff:** Surges from **{base['baseline_runoff_mm']:.1f} mm $\\to$ {simp['effective_surface_runoff_mm']:.1f} mm** ({simp['runoff_surge_multiplier']})."
                )
                lines.append(
                    f"* **Hydrological Hazard Probability:** Climbs to **{simp['flood_hazard_probability']:.2f}** (High Risk of urban canal surcharge)."
                )

            if hypo_temp is not None:
                wb_val = float(simp.get("wet_bulb_temp_c") or 24.0)
                app_val = float(simp.get("apparent_heat_index_c") or hypo_temp)
                bo_prob = float(simp.get("blackout_stress_probability") or 0.0)
                lines.append(
                    f"* **Wet-Bulb Temperature:** Reaches **{wb_val:.1f}°C** (Heat Index: {app_val:.1f}°C)."
                )
                lines.append(
                    f"* **Power Grid Cooling Demand:** Increases peak demand by **+{simp['grid_load_delta_mw']:.0f} MW** to **{simp['peak_grid_load_mw']:.0f} MW** (Blackout Risk: {bo_prob:.2f})."
                )

            if sim_res.get("persona_simulation"):
                p_sim = sim_res["persona_simulation"]
                lines.append(f"\n**Impact on {current_persona.capitalize()}:** Status changes to **{p_sim['status_level']}** (Impact Score: {p_sim['impact_score']:.1f}/100).")
                if p_sim["operational_restrictions"]:
                    lines.append(f"- ⚠️ **Direct Restriction:** {p_sim['operational_restrictions'][0]}")

            return "\n".join(lines), None, sim_res

    return None
