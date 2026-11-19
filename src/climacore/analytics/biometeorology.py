"""Biometeorological Heat Intelligence Engine V1.

Provides separate, rigorous bioclimatic heat stress pathways:
1. Rothfusz Heat Index (NWS HI Apparent Temperature)
2. Universal Thermal Climate Index (UTCI Bioclimatic Heat Budget)
3. Estimated Wet Bulb Globe Temperature (Estimated WBGT - outdoor & indoor)
4. Multi-Night Heat Exposure & Nocturnal Thermal Recovery Shortfall Indicator
5. Sourced Persona Dispatches (HeatHealthOfficer, OutdoorWorkerAdvisory)

CRITICAL ARCHITECTURAL BOUNDARY & ETHICAL/CLINICAL DISCLAIMERS:
1. NON-INTERCHANGEABILITY:
   Heat Index (HI), UTCI, and WBGT represent distinct physical and biometeorological
   quantities and MUST NEVER be treated as interchangeable:
   - Heat Index measures apparent temperature in shaded, light-wind conditions.
   - UTCI models full dynamic human heat exchange including wind and mean radiant temperature.
   - WBGT measures environmental heat load on working individuals (convection, radiation, evaporation).

2. NO UNIVERSAL MANDATORY WORK-STOP BAN FROM SINGLE HI THRESHOLD:
   Work-rest cycles and occupational safety depend on metabolic work rate (Watts),
   acclimatization state, clothing/PPE insulation, and radiant solar exposure. Universal
   mandatory work-stop rules cannot be legally or physically issued from a single HI number.

3. NON-CLINICAL DISCLAIMER:
   The nocturnal heat recovery index and cumulative cooling debt indicators are
   meteorological indicators of environmental cooling shortfall and do NOT constitute
   medical diagnosis, clinical treatment advice, or direct physiological strain monitoring.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field

from climacore.analytics.psychrometrics import (
    estimate_stull_wet_bulb,
    saturation_vapor_pressure_hpa,
)

logger = logging.getLogger(__name__)


class HeatIndexCategory(str, Enum):
    """NWS Heat Index Alert Categories."""
    NORMAL = "Normal"                    # HI < 27°C (< 80°F)
    CAUTION = "Caution"                  # 27 <= HI < 32°C (80-90°F): Fatigue possible with prolonged exposure
    EXTREME_CAUTION = "Extreme Caution"  # 32 <= HI < 39°C (90-103°F): Heat stroke/cramps possible
    DANGER = "Danger"                    # 39 <= HI < 51°C (103-124°F): Heat cramps/exhaustion likely
    EXTREME_DANGER = "Extreme Danger"    # HI >= 51°C (>= 125°F): Heat stroke highly likely


class UTCICategory(str, Enum):
    """Universal Thermal Climate Index Stress Categories (Bröde et al., 2012)."""
    EXTREME_COLD_STRESS = "Extreme Cold Stress"    # UTCI < -40°C
    VERY_STRONG_COLD = "Very Strong Cold Stress"   # -40 <= UTCI < -27°C
    STRONG_COLD = "Strong Cold Stress"             # -27 <= UTCI < -13°C
    MODERATE_COLD = "Moderate Cold Stress"         # -13 <= UTCI < 0°C
    SLIGHT_COLD = "Slight Cold Stress"             # 0 <= UTCI < +9°C
    NO_THERMAL_STRESS = "No Thermal Stress"        # +9 <= UTCI < +26°C
    MODERATE_HEAT_STRESS = "Moderate Heat Stress"  # +26 <= UTCI < +32°C
    STRONG_HEAT_STRESS = "Strong Heat Stress"      # +32 <= UTCI < +38°C
    VERY_STRONG_HEAT = "Very Strong Heat Stress"   # +38 <= UTCI < +46°C
    EXTREME_HEAT_STRESS = "Extreme Heat Stress"    # UTCI >= +46°C


class WBGTWorkRestCycle(str, Enum):
    """OSHA / ISO 7243 Recommended Work-Rest Allocation for Acclimatized Workers."""
    CONTINUOUS = "Continuous Work (No mandatory rest)"  # Normal work
    WORK_45_REST_15 = "45 min Work / 15 min Rest"      # Moderate stress
    WORK_30_REST_30 = "30 min Work / 30 min Rest"      # High stress
    WORK_15_REST_45 = "15 min Work / 45 min Rest"      # Very high stress
    STOP_WORK_RECOMMENDED = "Activity Suspended (Except Emergency)"  # Extreme stress


class RothfuszHeatIndexResult(BaseModel):
    """Rothfusz Heat Index result."""
    temperature_c: float
    relative_humidity_pct: float
    heat_index_c: float = Field(..., description="Calculated apparent temperature [°C]")
    heat_index_f: float = Field(..., description="Apparent temperature in Fahrenheit [°F]")
    category: HeatIndexCategory
    advisory: str
    is_shade_assumed: bool = True
    sun_exposure_adder_c: float = Field(
        default=4.4,
        description="Full direct sunshine exposure may add up to ~4.4°C (8°F) to apparent temperature."
    )

    model_config = {"frozen": True}


class UTCIResult(BaseModel):
    """Universal Thermal Climate Index result."""
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_10m_ms: float
    mean_radiant_temp_c: float
    utci_c: float = Field(..., description="UTCI equivalent bioclimatic temperature [°C]")
    category: UTCICategory
    bioclimatic_assessment: str

    model_config = {"frozen": True}


class EstimatedWBGTResult(BaseModel):
    """Estimated Wet Bulb Globe Temperature (explicitly non-instrument)."""
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_10m_ms: float
    solar_radiation_wm2: float
    estimated_wet_bulb_c: float
    estimated_globe_temp_c: float
    estimated_wbgt_outdoor_c: float = Field(..., description="0.7*Tw + 0.2*Tg + 0.1*Ta in full sunshine [°C]")
    estimated_wbgt_shaded_c: float = Field(..., description="0.7*Tw + 0.3*Ta in shade [°C]")
    light_work_cycle: WBGTWorkRestCycle
    moderate_work_cycle: WBGTWorkRestCycle
    heavy_work_cycle: WBGTWorkRestCycle
    fluid_replacement_liters_per_hour: float
    is_instrument_measured: bool = False
    disclaimer: str = Field(
        default=(
            "Calculated via psychrometric approximations (Stull & Liljegren models). "
            "Not an instrument black-globe measurement. Work-rest allocations require "
            "onsite evaluation of worker acclimatization, PPE, and hydration."
        )
    )

    model_config = {"frozen": True}


class MultiNightHeatExposureResult(BaseModel):
    """Multi-night heat exposure and nocturnal recovery shortfall indicator."""
    window_days: int
    nightly_min_temps_c: List[float]
    tropical_nights_count: int = Field(..., description="Count of nights with Tmin >= 25.0°C")
    severe_tropical_nights_count: int = Field(..., description="Count of nights with Tmin >= 28.0°C")
    cumulative_nocturnal_debt_deg_c: float = Field(
        ...,
        description="Cumulative Nocturnal Cooling Shortfall: sum(max(0, Tmin - 25.0°C))"
    )
    recovery_quality_score: float = Field(
        ...,
        description="Nocturnal thermal recovery score (0-100, where 100 is complete nighttime relief)"
    )
    prolonged_heat_warning: bool
    clinical_disclaimer: str = Field(
        default=(
            "Environmental indicator of nocturnal cooling deficit. "
            "Does not constitute clinical or medical diagnosis."
        )
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Pathway A: Rothfusz Heat Index
# ---------------------------------------------------------------------------

def calculate_rothfusz_heat_index(temp_c: float, rh_pct: float) -> RothfuszHeatIndexResult:
    """Calculate Steadman-Rothfusz Heat Index according to NWS SR 90-23.

    Valid for air temperatures above ~20°C (68°F).
    """
    t_c = float(temp_c)
    rh = float(np.clip(rh_pct, 1.0, 100.0))
    t_f = (t_c * 9.0 / 5.0) + 32.0

    # 1. Simple Steadman formula
    hi_simple = 0.5 * (t_f + 61.0 + ((t_f - 68.0) * 1.2) + (rh * 0.094))

    if (hi_simple + t_f) / 2.0 < 80.0:
        hi_f = hi_simple
    else:
        # Full 16-parameter Rothfusz regression
        c1 = -42.379
        c2 = 2.04901523
        c3 = 10.14333127
        c4 = -0.22475541
        c5 = -0.00683783
        c6 = -0.05481717
        c7 = 0.00122874
        c8 = 0.00085282
        c9 = -0.00000199

        hi_f = (
            c1
            + c2 * t_f
            + c3 * rh
            + c4 * t_f * rh
            + c5 * (t_f ** 2)
            + c6 * (rh ** 2)
            + c7 * (t_f ** 2) * rh
            + c8 * t_f * (rh ** 2)
            + c9 * (t_f ** 2) * (rh ** 2)
        )

        # Adjustment for low relative humidity
        if rh < 13.0 and (80.0 <= t_f <= 112.0):
            adj = ((13.0 - rh) / 4.0) * math.sqrt((17.0 - abs(t_f - 95.0)) / 17.0)
            hi_f -= adj

        # Adjustment for high relative humidity
        elif rh > 85.0 and (80.0 <= t_f <= 87.0):
            adj = ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0)
            hi_f += adj

    hi_c = (hi_f - 32.0) * 5.0 / 9.0

    # Categorization
    if hi_c < 27.0:
        cat = HeatIndexCategory.NORMAL
        adv = "Normal conditions; no specific heat restrictions."
    elif hi_c < 32.0:
        cat = HeatIndexCategory.CAUTION
        adv = "Caution: Fatigue possible with prolonged exposure and activity. Maintain hydration."
    elif hi_c < 39.0:
        cat = HeatIndexCategory.EXTREME_CAUTION
        adv = "Extreme Caution: Heat cramps and heat exhaustion possible. Plan frequent rest breaks."
    elif hi_c < 51.0:
        cat = HeatIndexCategory.DANGER
        adv = "Danger: Heat cramps or heat exhaustion likely; heat stroke possible with continued exertion."
    else:
        cat = HeatIndexCategory.EXTREME_DANGER
        adv = "Extreme Danger: Heat stroke highly likely with continued exposure. Suspend non-essential strenuous outdoor activity."

    return RothfuszHeatIndexResult(
        temperature_c=round(t_c, 2),
        relative_humidity_pct=round(rh, 1),
        heat_index_c=round(hi_c, 2),
        heat_index_f=round(hi_f, 2),
        category=cat,
        advisory=adv,
    )


# ---------------------------------------------------------------------------
# Pathway B: Universal Thermal Climate Index (UTCI)
# ---------------------------------------------------------------------------

def calculate_utci(
    temp_c: float,
    rh_pct: float,
    wind_speed_10m_ms: float = 1.0,
    mean_radiant_temp_c: Optional[float] = None,
) -> UTCIResult:
    """Calculate Universal Thermal Climate Index (UTCI) using the standard bioclimatic polynomial.

    References:
        Bröde et al. (2012), 'Deriving the operational procedure for the Universal Thermal Climate Index (UTCI)',
        International Journal of Biometeorology.
    """
    t_a = float(temp_c)
    rh = float(np.clip(rh_pct, 1.0, 100.0))
    # Wind speed at 10m height, clipped to valid UTCI domain [0.5, 17.0 m/s]
    va = float(np.clip(wind_speed_10m_ms, 0.5, 17.0))

    # Mean Radiant Temperature Tmrt defaults to Ta if not provided (indoor/shaded baseline)
    tmrt = float(mean_radiant_temp_c) if mean_radiant_temp_c is not None else t_a
    d_tmrt = tmrt - t_a

    # Water vapour pressure in kPa
    es_hpa = saturation_vapor_pressure_hpa(t_a)
    e_kpa = (es_hpa * (rh / 100.0)) / 10.0

    # Standard operational UTCI regression polynomial approximation
    # Delta_UTCI = f(Ta, va, D_Tmrt, e_kPa)
    d_utci = (
        0.607562052
        - 0.0227712343 * t_a
        + 8.06470249e-4 * (t_a ** 2)
        - 1.54271372e-4 * (t_a ** 3)
        - 3.24651986e-6 * (t_a ** 4)
        + 7.32602852e-8 * (t_a ** 5)
        + 1.35959073e-9 * (t_a ** 6)
        - 2.25836520 * va
        + 0.0880326035 * t_a * va
        + 0.00216844454 * (t_a ** 2) * va
        - 1.53347087e-5 * (t_a ** 3) * va
        - 5.72983704e-7 * (t_a ** 4) * va
        - 2.55090145e-9 * (t_a ** 5) * va
        - 0.751269505 * (va ** 2)
        - 0.0040292211 * t_a * (va ** 2)
        + 3.64000000e-4 * (t_a ** 2) * (va ** 2)
        + 0.089352833 * (va ** 3)
        + 0.60653435 * d_tmrt
        - 0.0168188565 * t_a * d_tmrt
        + 0.17498194 * (d_tmrt ** 2) / 100.0
        + 0.043542252 * e_kpa * 10.0
        - 0.0045866164 * t_a * e_kpa * 10.0
    )

    utci_c = t_a + d_utci

    if utci_c < 9.0:
        cat = UTCICategory.SLIGHT_COLD
        assess = "Slight to moderate cold stress."
    elif utci_c < 26.0:
        cat = UTCICategory.NO_THERMAL_STRESS
        assess = "Comfort zone / no thermal stress. Human thermal equilibrium maintained."
    elif utci_c < 32.0:
        cat = UTCICategory.MODERATE_HEAT_STRESS
        assess = "Moderate heat stress. Vasodilation and mild perspiration active."
    elif utci_c < 38.0:
        cat = UTCICategory.STRONG_HEAT_STRESS
        assess = "Strong heat stress. Significant thermoregulatory strain; heavy sweating required."
    elif utci_c < 46.0:
        cat = UTCICategory.VERY_STRONG_HEAT
        assess = "Very strong heat stress. High risk of hyperthermia, heat syncope, and core temperature rise."
    else:
        cat = UTCICategory.EXTREME_HEAT_STRESS
        assess = "Extreme heat stress. Critical life hazard; core body temperature increases rapidly."

    return UTCIResult(
        temperature_c=round(t_a, 2),
        relative_humidity_pct=round(rh, 1),
        wind_speed_10m_ms=round(va, 2),
        mean_radiant_temp_c=round(tmrt, 2),
        utci_c=round(utci_c, 2),
        category=cat,
        bioclimatic_assessment=assess,
    )


# ---------------------------------------------------------------------------
# Pathway C: Estimated Wet Bulb Globe Temperature (WBGT)
# ---------------------------------------------------------------------------

def calculate_estimated_wbgt(
    temp_c: float,
    rh_pct: float,
    wind_speed_10m_ms: float = 1.0,
    solar_radiation_wm2: float = 0.0,
    is_outdoor_sun: bool = True,
) -> EstimatedWBGTResult:
    """Calculate Estimated Wet Bulb Globe Temperature (psychrometric approximation).

    Uses Stull (2011) for wet-bulb temperature Tw, and Liljegren/BOM approximation for globe Tg.
    Explicitly labeled as Estimated WBGT.
    """
    t_a = float(temp_c)
    rh = float(np.clip(rh_pct, 1.0, 100.0))
    ws = max(0.2, float(wind_speed_10m_ms))
    s_rad = max(0.0, float(solar_radiation_wm2))

    # 1. Natural wet bulb approximation (Stull 2011)
    tw = estimate_stull_wet_bulb(t_a, rh)

    # 2. Globe temperature approximation (Australian BOM / Liljegren empirical model)
    # Tg = Ta + 0.0149 * S_rad - 0.05 * ws
    if is_outdoor_sun and s_rad > 0.0:
        tg = t_a + (0.0149 * s_rad) / (1.0 + 0.2 * math.sqrt(ws))
    else:
        tg = t_a

    # 3. WBGT formulas
    wbgt_outdoor = 0.7 * tw + 0.2 * tg + 0.1 * t_a
    wbgt_shaded = 0.7 * tw + 0.3 * t_a

    active_wbgt = wbgt_outdoor if is_outdoor_sun else wbgt_shaded

    # 4. OSHA / ISO 7243 Work-Rest Schedules by Metabolic Work Rate
    # Categories for acclimatized workers:
    # Light (<200W): continuous up to 30.0°C; 45/15 up to 31.0°C; 30/30 up to 32.0°C; 15/45 up to 33.0°C
    # Moderate (200-350W): continuous up to 28.0°C; 45/15 up to 29.0°C; 30/30 up to 30.5°C; 15/45 up to 31.5°C
    # Heavy (>350W): continuous up to 26.0°C; 45/15 up to 27.5°C; 30/30 up to 29.0°C; 15/45 up to 30.0°C
    def get_cycle(wbgt_val: float, c_lim: float, w45_lim: float, w30_lim: float, w15_lim: float) -> WBGTWorkRestCycle:
        if wbgt_val < c_lim:
            return WBGTWorkRestCycle.CONTINUOUS
        elif wbgt_val < w45_lim:
            return WBGTWorkRestCycle.WORK_45_REST_15
        elif wbgt_val < w30_lim:
            return WBGTWorkRestCycle.WORK_30_REST_30
        elif wbgt_val < w15_lim:
            return WBGTWorkRestCycle.WORK_15_REST_45
        else:
            return WBGTWorkRestCycle.STOP_WORK_RECOMMENDED

    light_cycle = get_cycle(active_wbgt, 30.0, 31.0, 32.0, 33.0)
    mod_cycle = get_cycle(active_wbgt, 28.0, 29.0, 30.5, 31.5)
    heavy_cycle = get_cycle(active_wbgt, 26.0, 27.5, 29.0, 30.0)

    # Fluid replacement guideline (liters per hour for active work)
    if active_wbgt < 26.0:
        fluid_l_hr = 0.5
    elif active_wbgt < 29.0:
        fluid_l_hr = 0.75
    elif active_wbgt < 31.0:
        fluid_l_hr = 1.0
    else:
        fluid_l_hr = 1.25

    return EstimatedWBGTResult(
        temperature_c=round(t_a, 2),
        relative_humidity_pct=round(rh, 1),
        wind_speed_10m_ms=round(ws, 2),
        solar_radiation_wm2=round(s_rad, 1),
        estimated_wet_bulb_c=round(tw, 2),
        estimated_globe_temp_c=round(tg, 2),
        estimated_wbgt_outdoor_c=round(wbgt_outdoor, 2),
        estimated_wbgt_shaded_c=round(wbgt_shaded, 2),
        light_work_cycle=light_cycle,
        moderate_work_cycle=mod_cycle,
        heavy_work_cycle=heavy_cycle,
        fluid_replacement_liters_per_hour=fluid_l_hr,
        is_instrument_measured=False,
    )


# ---------------------------------------------------------------------------
# Pathway D: Multi-Night Heat Exposure & Recovery Shortfall
# ---------------------------------------------------------------------------

def calculate_multi_night_heat_index(
    nightly_min_temps_c: List[float],
    tropical_threshold_c: float = 25.0,
    severe_threshold_c: float = 28.0,
) -> MultiNightHeatExposureResult:
    """Calculate multi-night cumulative heat debt and nocturnal recovery shortfall.

    Cumulative Nocturnal Debt = sum(max(0, Tmin - 25.0°C)) across the monitoring window.
    Recovery Quality Score: 100 - penalization for consecutive uncooled nights.
    """
    if not nightly_min_temps_c:
        raise ValueError("Must provide at least one nocturnal minimum temperature.")

    temps = [float(t) for t in nightly_min_temps_c]
    window_days = len(temps)

    tropical_count = sum(1 for t in temps if t >= tropical_threshold_c)
    severe_count = sum(1 for t in temps if t >= severe_threshold_c)

    # Cumulative nocturnal shortfall
    nocturnal_debt = sum(max(0.0, t - tropical_threshold_c) for t in temps)

    # Recovery Quality Score: starts at 100, drops by 8 pts per °C of cumulative debt,
    # and drops by 10 pts per severe tropical night.
    penalty = (nocturnal_debt * 8.0) + (severe_count * 10.0)
    recovery_score = max(0.0, min(100.0, 100.0 - penalty))

    # Warning flag: >= 3 consecutive tropical nights OR cumulative debt >= 10.0 °C-days
    prolonged_warning = (tropical_count >= 3) or (nocturnal_debt >= 10.0) or (severe_count >= 2)

    return MultiNightHeatExposureResult(
        window_days=window_days,
        nightly_min_temps_c=[round(t, 2) for t in temps],
        tropical_nights_count=tropical_count,
        severe_tropical_nights_count=severe_count,
        cumulative_nocturnal_debt_deg_c=round(nocturnal_debt, 2),
        recovery_quality_score=round(recovery_score, 1),
        prolonged_heat_warning=prolonged_warning,
    )


# ---------------------------------------------------------------------------
# Persona Adapters
# ---------------------------------------------------------------------------

def dispatch_heat_health_officer(
    hi_res: RothfuszHeatIndexResult,
    utci_res: UTCIResult,
    multi_night_res: Optional[MultiNightHeatExposureResult] = None,
) -> Dict[str, Any]:
    """Generate municipal public health actions for HeatHealthOfficer persona."""
    actions = []
    alert_level = "GREEN"

    if hi_res.category in (HeatIndexCategory.EXTREME_DANGER, HeatIndexCategory.DANGER) or utci_res.category in (UTCICategory.EXTREME_HEAT_STRESS, UTCICategory.VERY_STRONG_HEAT):
        alert_level = "RED"
        actions.extend([
            "Activate municipal Heat Wave Action Plan (HAP) Level-3 Red Alert.",
            "Open air-conditioned municipal cooling centers in high-density wards.",
            "Deploy emergency potable water tankers to transit hubs and informal settlements.",
            "Order primary health centers to stock IV fluids and ORS; cancel non-emergency medical staff leave.",
        ])
    elif hi_res.category == HeatIndexCategory.EXTREME_CAUTION or utci_res.category == UTCICategory.STRONG_HEAT_STRESS:
        alert_level = "ORANGE"
        actions.extend([
            "Issue Level-2 Orange Advisory for elderly, infants, and chronic respiratory patients.",
            "Ensure shading and drinking water stations along major commercial corridors.",
            "Reschedule municipal outdoor maintenance to pre-10:00 AM hours.",
        ])
    elif hi_res.category == HeatIndexCategory.CAUTION:
        alert_level = "YELLOW"
        actions.append("Issue routine Level-1 Yellow Heat Awareness bulletin.")

    if multi_night_res and multi_night_res.prolonged_heat_warning:
        actions.append(
            f"PROLONGED NOCTURNAL HEAT ALERT: {multi_night_res.tropical_nights_count} tropical nights "
            f"detected with cumulative cooling shortfall of {multi_night_res.cumulative_nocturnal_debt_deg_c}°C-days. "
            "Inspect residential care facilities without mechanical air conditioning."
        )

    return {
        "persona": "HeatHealthOfficer",
        "alert_level": alert_level,
        "recommended_actions": actions,
        "heat_index_c": hi_res.heat_index_c,
        "utci_c": utci_res.utci_c,
        "disclaimer": "Municipal operations advisory only. Verify with local IMD official warnings.",
    }


def dispatch_outdoor_worker_advisory(
    wbgt_res: EstimatedWBGTResult,
    work_intensity: str = "moderate",
    is_acclimatized: bool = True,
) -> Dict[str, Any]:
    """Generate occupational safety guidance for OutdoorWorkerAdvisory persona.

    Uses sourced ISO 7243 work-rest allocations based on Estimated WBGT.
    Never issues universal blanket work-stops from a single Heat Index threshold.
    """
    intensity = work_intensity.lower()
    if intensity == "light":
        cycle = wbgt_res.light_work_cycle
    elif intensity == "heavy":
        cycle = wbgt_res.heavy_work_cycle
    else:
        cycle = wbgt_res.moderate_work_cycle

    guidance = [
        f"Recommended Work-Rest Schedule: {cycle.value} for {intensity.upper()} metabolic intensity.",
        f"Fluid Replacement: Minimum {wbgt_res.fluid_replacement_liters_per_hour:.2f} liters/hour of cool water / electrolyte solution.",
        "Implement mandatory buddy system for early detection of heat exhaustion (dizziness, nausea, cessation of sweating).",
    ]

    if not is_acclimatized:
        guidance.insert(
            0,
            "UNACCLIMATIZED WORKER NOTICE: Increase rest break duration by 50% for workers in their first 7 days on site."
        )

    if wbgt_res.estimated_wbgt_outdoor_c >= 32.0:
        guidance.append(
            "HIGH THERMAL LOAD: Move strenuous tasks to early morning (06:00-09:00) or post-sunset (after 18:00)."
        )

    return {
        "persona": "OutdoorWorkerAdvisory",
        "work_intensity": intensity,
        "is_acclimatized": is_acclimatized,
        "estimated_wbgt_outdoor_c": wbgt_res.estimated_wbgt_outdoor_c,
        "estimated_wbgt_shaded_c": wbgt_res.estimated_wbgt_shaded_c,
        "recommended_work_rest_cycle": cycle.value,
        "guidance": guidance,
        "disclaimer": (
            "Guidance reflects ISO 7243 / OSHA occupational heat recommendations based on Estimated WBGT. "
            "Onsite competent person must verify individual health, PPE, and local statutory labor regulations."
        ),
    }
