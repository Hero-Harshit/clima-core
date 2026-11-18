"""
Parametric Climate Insurance, Actuarial Risk & Catastrophe Underwriting Persona Adapter.
Evaluates zero-dispute parametric policy triggers (root-zone SMDI drought breach, 74-year heatwave return periods,
cyclone landfall radius buffers, and pluvial flood loss reserve multipliers).
"""

from typing import Dict, Any, List
import numpy as np
from climacore.personas.models import PersonaInput, PersonaResult


def evaluate_insurance_persona(inp: PersonaInput) -> PersonaResult:
    """Evaluate parametric insurance claim triggers, loss severity multipliers, and capital reserve adequacy."""
    advisories: List[str] = []
    restrictions: List[str] = []

    t_max = inp.temp_max
    rain_mm = inp.precipitation_mm
    pct = inp.historical_percentile
    smdi = inp.soil_moisture_deficit_index if inp.soil_moisture_deficit_index is not None else 0.0
    flash_drought = inp.flash_drought_alert or 0
    cyc_dist = inp.cyclone_distance_km
    wind_kmh = inp.wind_speed_kmh
    gust_kmh = inp.gale_wind_gust_kmh if inp.gale_wind_gust_kmh is not None else wind_kmh * 1.35

    # 1. Parametric Agricultural Drought Trigger (PMFBY Index)
    # SMDI <= -2.0 represents severe root-zone water depletion (bottom 5th percentile)
    if smdi <= -2.2 or (flash_drought == 1 and smdi <= -1.8):
        drought_trigger = "PARAMETRIC_BREACH_CONFIRMED (Total Payout Triggered)"
        drought_payout_pct = 100.0
        restrictions.append("Underwriters: Lock claim reserve funds for automated direct benefit transfer (DBT) to insured farmers.")
        advisories.append("Parametric root-zone drought threshold breached; loss adjustment zero-dispute protocol active.")
    elif smdi <= -1.5:
        drought_trigger = "APPROACHING_TRIGGER_ZONE (50% Partial Payout Window)"
        drought_payout_pct = 50.0
        advisories.append("Soil moisture deficit index nearing critical contract threshold; monitor 7-day rolling balance.")
    else:
        drought_trigger = "NO_DROUGHT_CLAIM_BREACH"
        drought_payout_pct = 0.0

    # 2. Extreme Value Theory (EVT) Parametric Heat & Rainfall Policy Triggers
    try:
        from climacore.analytics.extreme_value import evaluate_return_period
        evt_t = evaluate_return_period(inp.city, t_max, hazard_type="temperature")
        evt_r = evaluate_return_period(inp.city, rain_mm, hazard_type="rainfall")
        t_ret_heat = evt_t.get("estimated_return_period_years") or 1.0
        t_ret_rain = evt_r.get("estimated_return_period_years") or 1.0
    except Exception:
        t_ret_heat = 1.0
        t_ret_rain = 1.0

    # Parametric Heat Trigger (Grounded in 74-Year IMD Return Levels)
    if t_ret_heat >= 10.0 or pct >= 98.5 or t_max >= 45.0:
        heat_trigger = f"PARAMETRIC_HEAT_STRIKE (>= 1-in-10 Year Extreme, T={t_ret_heat:.1f}yr)"
        heat_payout_pct = 100.0
        restrictions.append("Commercial Property & Business Interruption: Activate heatwave parametric payout protocol.")
    elif t_ret_heat >= 5.0 or pct >= 95.0 or t_max >= 43.0:
        heat_trigger = f"ELEVATED_HEAT_RISK (1-in-5 Year Watchlist, T={t_ret_heat:.1f}yr)"
        heat_payout_pct = 25.0
    else:
        heat_trigger = "NORMAL_THERMAL_RISK"
        heat_payout_pct = 0.0

    # 3. Cyclone Landfall & Windstorm Catastrophe Trigger
    cyc_payout_pct = 0.0
    if cyc_dist is not None:
        if cyc_dist <= 100.0 or gust_kmh >= 120.0:
            cyc_trigger = f"CATASTROPHIC_STRIKE ({cyc_dist:.1f} km, Gusts {gust_kmh:.1f} km/h)"
            cyc_payout_pct = 100.0
            restrictions.append("Catastrophe Reinsurers: Release initial liquidity tranches for coastal commercial property and port facilities.")
        elif cyc_dist <= 250.0 or gust_kmh >= 85.0:
            cyc_trigger = f"GALE_BUFFER_EXPOSURE ({cyc_dist:.1f} km)"
            cyc_payout_pct = 50.0
        else:
            cyc_trigger = f"DISTANT_TRACK ({cyc_dist:.1f} km)"
    else:
        cyc_trigger = "NO_ACTIVE_CYCLONE_THREAT"

    # 4. Parametric Pluvial Flood Trigger (Grounded in 46-Year IMD Return Levels)
    if t_ret_rain >= 10.0 or rain_mm >= 115.0:
        flood_trigger = f"PARAMETRIC_FLOOD_STRIKE (>= 1-in-10 Year Pluvial Extreme, T={t_ret_rain:.1f}yr)"
        flood_payout_pct = 100.0
        restrictions.append("Underwriters: Trigger automated municipal inundation parametric policy settlements.")
        advisories.append("Severe pluvial flood return period breached; loss-adjustment fast-track protocol active.")
    elif t_ret_rain >= 5.0 or rain_mm >= 65.0:
        flood_trigger = f"ELEVATED_FLOOD_RISK (1-in-5 Year Watchlist, T={t_ret_rain:.1f}yr)"
        flood_payout_pct = 40.0
        advisories.append("Anticipate high loss-adjustment claim frequency for basement flooding and motor vehicle hydrolock.")
    else:
        flood_trigger = "BELOW_FLOOD_INDEX_THRESHOLD"
        flood_payout_pct = 0.0

    # 5. Composite Portfolio Loss Reserve Score (0 to 100)
    score = round(float(
        0.35 * drought_payout_pct +
        0.30 * max(heat_payout_pct, cyc_payout_pct) +
        0.20 * flood_payout_pct +
        0.15 * min(100.0, (pct / 100.0) * 100.0)
    ), 1)

    if score >= 70.0 or cyc_payout_pct >= 100.0 or drought_payout_pct >= 100.0:
        status = "CRITICAL"
    elif score >= 40.0:
        status = "HIGH_ALERT"
    elif score >= 20.0:
        status = "CAUTION"
    else:
        status = "NORMAL"

    metrics: Dict[str, Any] = {
        "parametric_drought_status": drought_trigger,
        "soil_moisture_deficit_index": f"{smdi:.2f}",
        "parametric_heat_status": heat_trigger,
        "evt_temperature_return_period": f"{t_ret_heat:.1f} Years",
        "evt_rainfall_return_period": f"{t_ret_rain:.1f} Years",
        "74y_climate_rarity_percentile": f"{pct:.1f} %",
        "cyclone_catastrophe_status": cyc_trigger,
        "pluvial_flood_exposure": flood_trigger,
        "estimated_portfolio_payout_load": f"{score:.1f} / 100",
        "claims_processing_mode": "AUTOMATED_PARAMETRIC_EXECUTION" if (score >= 60.0 or cyc_payout_pct >= 100.0 or drought_payout_pct >= 100.0 or heat_payout_pct >= 100.0) else "STANDARD_MANUAL_ADJUSTMENT"
    }

    if inp.population_15km is not None:
        pop_15k = inp.population_15km
        dense_pop = inp.dense_cluster_pop_15km or 0
        loss_multiplier = 1.0 + min(1.0, (pop_15k / 15_000_000.0) * 0.4 + (dense_pop / 5_000_000.0) * 0.4)
        metrics["portfolio_population_exposure"] = f"{pop_15k:,} individuals"
        metrics["informal_settlement_claim_potential"] = f"{dense_pop:,} residents"
        metrics["urban_concentration_loss_multiplier"] = f"{loss_multiplier:.2f}x"

        if loss_multiplier >= 1.25 and score >= 40.0:
            advisories.append(
                f"Catastrophe concentration alert: Urban density loss multiplier {loss_multiplier:.2f}x due to {pop_15k:,} exposed population. Augment secondary reinsurance liquidity reserves."
            )

    return PersonaResult(
        persona_id="insurance_actuarial",
        persona_title="Parametric Climate Insurance, Actuarial Risk & Catastrophe Underwriting",
        status_level=status,
        impact_score=score,
        key_metrics=metrics,
        advisories=advisories or ["Underwritten portfolio risks operating within standard actuarial variance."],
        operational_restrictions=restrictions or ["No capital reserve calls or catastrophe stop-loss mandates active."]
    )
