"""
Audit module for Open-Meteo Previous Runs API.
Queries a sample of historical forecast runs and verifies availability, lead times,
and alignment feasibility for Indian cities.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import httpx

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.audit_forecast_api")


def audit_previous_runs_api(
    city_sample: str = "Delhi",
    lat: float = 28.6139,
    lon: float = 77.2090,
    start_date: str = "2024-06-01",
    end_date: str = "2024-06-07",
    output_dir: Path | str = None,
) -> Dict[str, Any]:
    """Query Open-Meteo Previous Runs API for sample dates and lead times."""
    if output_dir is None:
        output_dir = get_project_root() / "reports" / "data_audit"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "temperature_2m_previous_day1",
            "temperature_2m_previous_day2",
            "precipitation",
            "precipitation_previous_day1",
            "relative_humidity_2m",
            "wind_speed_10m",
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Kolkata",
    }

    logger.info(f"Querying Open-Meteo Previous Runs API for {city_sample} ({start_date} to {end_date})...")

    report: Dict[str, Any] = {
        "api_name": "Open-Meteo Previous Runs API",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_parameters": {
            "city": city_sample,
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Asia/Kolkata",
            "variables_requested": params["hourly"],
        },
        "response_status": None,
        "hourly_variables_returned": [],
        "sample_data_count": 0,
        "lead_time_alignment_verified": False,
        "terms_and_limits": {
            "free_tier_daily_call_limit": 10000,
            "attribution": "CC BY 4.0; Open-Meteo",
            "commercial_use_note": "Requires paid API key for commercial deployment; free tier is non-commercial/fair-use.",
        },
        "feasibility_assessment": "",
    }

    try:
        response = httpx.get(url, params=params, timeout=15.0)
        report["response_status"] = response.status_code
        if response.status_code == 200:
            data = response.json()
            hourly = data.get("hourly", {})
            returned_vars = list(hourly.keys())
            report["hourly_variables_returned"] = returned_vars
            time_steps = len(hourly.get("time", []))
            report["sample_data_count"] = time_steps

            # Verify presence of lead offsets
            has_prev_day1 = "temperature_2m_previous_day1" in hourly
            has_prev_day2 = "temperature_2m_previous_day2" in hourly
            has_target = "temperature_2m" in hourly

            if has_prev_day1 and has_target and time_steps > 0:
                report["lead_time_alignment_verified"] = True
                report["feasibility_assessment"] = (
                    "FEASIBLE: Previous Runs API provides paired valid_time, analyzing target "
                    "temperature_2m and forecast lead-time offsets (day1, day2). "
                    "However, daily bulk queries across 50 cities for multiple years requires "
                    "throttled batch downloads due to rate limits."
                )

            # Save fixture sample
            fixture_path = out_path / "open_meteo_previous_runs_sample.json"
            with open(fixture_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        else:
            report["feasibility_assessment"] = f"FAILED: API returned status code {response.status_code}: {response.text}"
    except Exception as exc:
        report["feasibility_assessment"] = f"ERROR: Exception occurred while querying API: {exc}"

    report_path = out_path / "open_meteo_previous_runs_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Open-Meteo Previous Runs API audit saved to {report_path}")
    return report


if __name__ == "__main__":
    audit_previous_runs_api()
