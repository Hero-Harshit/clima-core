"""
Farmer (Kisan) Agro-Climate Persona Adapter.
Evaluates crop water deficit, irrigation scheduling, pesticide spraying windows,
fungal blight risk, and terminal heat shock on flowering crops.
"""

from typing import Dict, Any, List
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_farmer_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate agro-meteorological risks tailored for agricultural cultivators."""
    advisories = []
    restrictions = []

    # 1. Daily Crop Water Balance (FAO-56 Penman-Monteith Evapotranspiration vs Rain)
    et0 = inp.et0_evapotranspiration
    rain = inp.precipitation_mm
    water_balance = round(float(rain - et0), 2)  # positive = surplus, negative = deficit

    # 2. Pesticide & Chemical Spraying Suitability Window
    # Favorable: Wind <= 14 km/h, rain probability < 20%, humidity <= 75%
    wind = inp.wind_speed_kmh
    rain_prob = inp.rain_probability
    rh = inp.relative_humidity

    if rain >= 5.0 or rain_prob >= 0.5:
        spraying_window = "UNFAVORABLE_RAIN_WASHOFF"
        restrictions.append("Suspend foliar pesticide, fungicide, and urea top-dressing to prevent chemical washoff.")
    elif wind >= 20.0:
        spraying_window = "UNFAVORABLE_HIGH_DRIFT"
        restrictions.append("Excessive wind (>20 km/h) will cause spray drift to non-target areas.")
    elif wind <= 14.0 and rain_prob <= 0.2:
        spraying_window = "OPTIMAL_WINDOW"
        advisories.append("Ideal conditions for foliar nutrient and pest spraying between 7:00 AM and 10:30 AM.")
    else:
        spraying_window = "MODERATE"

    # 3. Fungal Blight / Foliar Disease Risk Index
    # Warm temperatures (22-30°C) with sustained high relative humidity (>75%) foster fungal spores
    fungal_risk = "LOW"
    if 22.0 <= inp.temp_max <= 32.0 and rh >= 75.0:
        fungal_risk = "HIGH"
        advisories.append("High humidity and warm conditions favor fungal spore multiplication (blast/rust/mildew). Scout crops closely.")
    elif rh >= 65.0:
        fungal_risk = "MODERATE"

    # 4. Terminal Heat Shock (Wheat/Mustard grain filling)
    terminal_heat_shock = False
    if inp.temp_max >= 36.0:
        terminal_heat_shock = True
        restrictions.append("Extreme temperature (>=36°C) threatens grain weight and premature crop ripening.")
        advisories.append("Apply light sprinkler irrigation in evening to induce canopy evaporative cooling.")

    # 5. Soil Moisture Deficit & Agricultural Flash Drought
    smdi = inp.soil_moisture_deficit_index
    flash_drought = inp.flash_drought_alert == 1
    drought_stress_score = 0.0

    if flash_drought:
        restrictions.append("FLASH DROUGHT WARNING: Topsoil moisture collapsed to critical decile (<P10). Severe wilting imminent.")
        advisories.append("Trigger emergency deficit irrigation or mulching immediately to preserve root-zone moisture.")
        drought_stress_score = 95.0
    elif smdi is not None and smdi <= -1.5:
        advisories.append(f"Pronounced root-zone moisture deficit (SMDI: {smdi:+.2f}). Crop water extraction impaired.")
        drought_stress_score = 75.0
    elif smdi is not None and smdi < -0.5:
        advisories.append(f"Mild soil moisture stress (SMDI: {smdi:+.2f}). Monitor soil drying rate.")
        drought_stress_score = 40.0

    # 6. AgriSense District Soil & Crop-Specific Advisory
    if inp.dominant_rabi_crop and "wheat" in inp.dominant_rabi_crop.lower() and inp.temp_max >= 32.0:
        dist_name = inp.agricultural_district or inp.city
        advisories.append(f"AgriSense Wheat Alert: Daytime temperature ({inp.temp_max:.1f}°C) in {dist_name} hinterland exceeds optimal grain-filling threshold. Apply evening sprinkler irrigation to mitigate terminal heat shock.")
    elif inp.dominant_kharif_crop and "rice" in inp.dominant_kharif_crop.lower() and water_balance <= -4.0:
        dist_name = inp.agricultural_district or inp.city
        advisories.append(f"AgriSense Paddy Alert: Deficit of {abs(water_balance):.1f} mm/day in {dist_name} kharif belt requires maintaining 2-3 cm standing water depth in nursery/puddled fields.")

    if inp.soil_water_holding_capacity is not None and inp.soil_water_holding_capacity < 18.0 and water_balance < -2.0:
        advisories.append(f"Soil Texture Warning: Porous soil profile (WHC: {inp.soil_water_holding_capacity:.1f}) causes accelerated root-zone moisture depletion. Prioritize mulch and micro-irrigation.")

    if inp.soil_ph is not None and (inp.soil_ph > 8.0 or inp.soil_ph < 6.0):
        advisories.append(f"Soil Chemistry Note: District soil pH is {inp.soil_ph:.1f}. Monitor micronutrient bioavailability (zinc/iron) during moisture stress.")

    # Compute Overall Agro Impact Score (0 to 100)
    # High score means severe agro-climatic stress (drought, extreme storm, or heat shock)
    deficit_score = min(100.0, max(0.0, (-water_balance / 8.0) * 100.0)) if water_balance < 0 else min(100.0, (water_balance / 40.0) * 100.0)
    heat_shock_score = 100.0 if terminal_heat_shock else (50.0 if inp.temp_max >= 33.0 else 10.0)
    base_agro = 0.35 * deficit_score + 0.30 * heat_shock_score + 0.20 * (80.0 if fungal_risk == "HIGH" else 20.0)
    agro_score = round(float(max(base_agro, 0.40 * drought_stress_score + 0.60 * base_agro)), 1)
    agro_score = min(100.0, max(0.0, agro_score))

    if agro_score >= 75.0 or terminal_heat_shock or rain >= 65.0 or flash_drought:
        status = "CRITICAL"
    elif agro_score >= 50.0 or fungal_risk == "HIGH" or (smdi is not None and smdi <= -1.5):
        status = "HIGH_ALERT"
    elif agro_score >= 30.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    if water_balance <= -4.0:
        advisories.append(f"Significant moisture deficit of {abs(water_balance):.1f} mm/day. Schedule irrigation within 24 hours.")
    elif water_balance >= 25.0:
        advisories.append("Heavy rainfall surplus expected. Ensure drainage channels are clear to prevent water stagnation in root zone.")

    return PersonaResult(
        persona_id="farmer_kisan",
        persona_title="Farmer & Agricultural Cultivator",
        status_level=status,
        impact_score=agro_score,
        key_metrics={
            "crop_water_balance": f"{water_balance:+.1f} mm/day",
            "soil_moisture_deficit_index": f"{smdi:+.2f}" if smdi is not None else "N/A",
            "flash_drought_alert": "ACTIVE_CRITICAL" if flash_drought else "NONE",
            "irrigation_requirement": "EMERGENCY" if flash_drought else ("URGENT" if water_balance <= -4.0 else ("MODERATE" if water_balance < 0 else "NONE_RAIN_SUFFICIENT")),
            "spraying_suitability": spraying_window,
            "fungal_disease_risk": fungal_risk,
            "terminal_heat_shock": terminal_heat_shock,
            "evapotranspiration_et0": f"{et0:.1f} mm/day",
            "forecasted_rain": f"{rain:.1f} mm",
            "agricultural_hinterland": f"{inp.agricultural_district or inp.city} ({inp.dominant_crop_overall or 'Dominant'})",
            "dominant_cropping_pattern": f"Kharif: {inp.dominant_kharif_crop or 'Rice'} | Rabi: {inp.dominant_rabi_crop or 'Wheat'}",
            "soil_fertility_and_whc": f"Fertility: {inp.soil_fertility_index or 'N/A'}/100 | WHC: {inp.soil_water_holding_capacity or 'N/A'}",
            "soil_ph": f"{inp.soil_ph:.1f}" if inp.soil_ph is not None else "N/A",
        },
        advisories=advisories or ["Normal agro-meteorological conditions. Standard farm operations may proceed."],
        operational_restrictions=restrictions or ["No farming restrictions active."],
    )
