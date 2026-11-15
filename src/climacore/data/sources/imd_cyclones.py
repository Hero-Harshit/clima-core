"""IMD North Indian Ocean Tropical Cyclone Best Track Ingestion & City Risk Engine.

Ingests the official IMD RSMC New Delhi Cyclone eAtlas best track records (1982-2026),
decodes 3-hourly/6-hourly storm trajectories, intensity classifications, and pressure drops,
and calculates geodesic exposure buffers (100km, 250km, 500km) across all 50 Indian cities.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import openpyxl
import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.imd_cyclones")

EARTH_RADIUS_KM = 6371.0


def haversine_distance_vectorized(
    lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Compute great-circle distances in km from a single point to an array of points."""
    phi1 = np.radians(lat1)
    lam1 = np.radians(lon1)
    phi2 = np.radians(lats)
    lam2 = np.radians(lons)

    dphi = phi2 - phi1
    dlam = lam2 - lam1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return EARTH_RADIUS_KM * c


def clean_coord(val: Any, is_lat: bool = True) -> Optional[float]:
    """Parse coordinate string or float (handling '14.0N', '82.5E', ' 7.5', etc.)
    with strict meteorological domain quality control for North Indian Ocean (RSMC New Delhi).
    """
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    s = re.sub(r"[^\d.-]", "", s)
    try:
        f = float(s)
        if is_lat:
            # North Indian Ocean basin physical bounds: 0°N to 36°N
            return f if 0.0 <= f <= 36.0 else None
        else:
            # North Indian Ocean basin physical bounds: 40°E to 105°E
            return f if 40.0 <= f <= 105.0 else None
    except ValueError:
        return None


def clean_numeric(val: Any) -> Optional[float]:
    """Clean numeric values (pressure, wind, pressure drop)."""
    if pd.isna(val):
        return None
    s = str(val).strip()
    s = re.sub(r"[^\d.-]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def classify_imd_grade(wind_kt: Optional[float], text_grade: Any) -> str:
    """Standardize IMD Cyclone Intensity Scale Grade."""
    t_str = str(text_grade).strip().upper() if pd.notna(text_grade) else ""
    if "SUCS" in t_str or "SUPER" in t_str:
        return "SuCS"
    if "ESCS" in t_str or "EXTREMELY" in t_str:
        return "ESCS"
    if "VSCS" in t_str or "VERY SEVERE" in t_str:
        return "VSCS"
    if "SCS" in t_str or "SEVERE" in t_str:
        return "SCS"
    if "CS" in t_str or "CYCLONIC" in t_str:
        return "CS"
    if "DD" in t_str or "DEEP" in t_str:
        return "DD"
    if "D" in t_str or "DEPRESSION" in t_str:
        return "D"

    # Fallback to wind speed criteria
    if wind_kt is not None:
        if wind_kt >= 120.0:
            return "SuCS"
        if wind_kt >= 90.0:
            return "ESCS"
        if wind_kt >= 64.0:
            return "VSCS"
        if wind_kt >= 48.0:
            return "SCS"
        if wind_kt >= 34.0:
            return "CS"
        if wind_kt >= 28.0:
            return "DD"
        if wind_kt >= 17.0:
            return "D"

    return "LOW"


def parse_time_str(time_val: Any) -> str:
    """Robustly extract valid HH:MM:00 from messy meteorological time cells."""
    if pd.isna(time_val):
        return "00:00:00"

    # Numeric float/int handling, e.g. 1200.0 or 300 or 0.0
    if isinstance(time_val, (int, float)):
        try:
            val_int = int(round(time_val))
            if val_int >= 10000:
                val_int = val_int // 10
            hh = val_int // 100
            mm = val_int % 100
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return f"{hh:02d}:{mm:02d}:00"
        except Exception:
            pass

    s = str(time_val).strip()

    # Look for 4 consecutive digits (e.g. 0300, 1200, 1800)
    m = re.search(r"\b([012]?\d)([0-5]\d)\b", s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh <= 23:
            return f"{hh:02d}:{mm:02d}:00"

    # Numeric string fallback
    digits = re.sub(r"\D", "", s)
    if len(digits) == 5:
        digits = digits[:4]
    if len(digits) == 4:
        hh, mm = int(digits[:2]), int(digits[2:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}:00"
    if len(digits) == 3:
        hh, mm = int(digits[:1]), int(digits[1:])
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return f"{hh:02d}:{mm:02d}:00"

    return "00:00:00"


def ingest_cyclone_best_tracks() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ingest, clean, and enrich full 1982-2026 IMD Cyclone Best Tracks."""
    root = get_project_root()
    raw_file = root / "data" / "raw" / "cyclones" / "Cyclone_Best_Tracks__Data__1982-2026_.xlsx"
    cities_file = root / "data" / "raw" / "india_weather_air_quality" / "cities.csv"
    proc_dir = root / "data" / "processed" / "cyclones"
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        raise FileNotFoundError(f"Cyclone best tracks file not found at {raw_file}")

    cities_df = pd.read_csv(cities_file)
    city_lats = cities_df["latitude"].to_numpy()
    city_lons = cities_df["longitude"].to_numpy()
    city_names = cities_df["city"].to_numpy()

    logger.info(f"Loading Cyclone Best Tracks Excel workbook from {raw_file}...")
    wb = openpyxl.load_workbook(raw_file, read_only=True)
    sheets = wb.sheetnames
    wb.close()

    logger.info(f"Found {len(sheets)} yearly sheets in Cyclone Best Tracks workbook ({sheets[0]} to {sheets[-1]}).")

    all_tracks: List[pd.DataFrame] = []
    t0 = time.time()

    for s_idx, sheet in enumerate(sheets):
        # Determine header row
        df_raw = pd.read_excel(raw_file, sheet_name=sheet, header=None, nrows=10)
        header_row = 0
        for idx in range(min(5, len(df_raw))):
            row_str = " ".join([str(x).lower() for x in df_raw.iloc[idx].dropna()])
            if "lat" in row_str and ("date" in row_str or "time" in row_str or "yyyy" in row_str or "dd" in row_str):
                header_row = idx
                break

        df = pd.read_excel(raw_file, sheet_name=sheet, header=header_row)

        # Normalize column mapping
        col_map = {}
        for c in df.columns:
            cl = str(c).lower().strip()
            if "serial" in cl:
                col_map[c] = "serial_no"
            elif "basin" in cl or "barbin" in cl:
                col_map[c] = "basin"
            elif cl == "name":
                col_map[c] = "cyclone_name"
            elif "date" in cl:
                col_map[c] = "date_raw"
            elif "time" in cl:
                col_map[c] = "time_raw"
            elif "lat" in cl:
                col_map[c] = "lat_raw"
            elif "lon" in cl:
                col_map[c] = "lon_raw"
            elif "central pressure" in cl or "e.c.p" in cl:
                col_map[c] = "central_pressure_hpa"
            elif "wind" in cl:
                col_map[c] = "max_wind_kt"
            elif "pressure drop" in cl or "delta p" in cl:
                col_map[c] = "pressure_drop_hpa"
            elif "grade" in cl:
                col_map[c] = "grade_raw"

        df = df.rename(columns=col_map)
        needed = ["serial_no", "basin", "cyclone_name", "date_raw", "time_raw", "lat_raw", "lon_raw"]
        for col in needed:
            if col not in df.columns:
                df[col] = np.nan

        # Forward-fill system metadata within year
        df["serial_no"] = df["serial_no"].ffill()
        df["basin"] = df["basin"].ffill()
        df["cyclone_name"] = df["cyclone_name"].ffill()

        # Clean coordinates with basin bounds QC
        df["latitude"] = df["lat_raw"].apply(lambda x: clean_coord(x, is_lat=True))
        df["longitude"] = df["lon_raw"].apply(lambda x: clean_coord(x, is_lat=False))

        # Drop rows where coordinates are missing
        df = df[df["latitude"].notnull() & df["longitude"].notnull()].copy()
        if len(df) == 0:
            continue

        # Parse year and system ID
        try:
            year_int = int(re.sub(r"\D", "", sheet)[:4])
        except Exception:
            year_int = 2000
        df["year"] = year_int

        # Clean numeric fields
        df["central_pressure_hpa"] = df["central_pressure_hpa"].apply(clean_numeric) if "central_pressure_hpa" in df.columns else np.nan
        df["max_wind_kt"] = df["max_wind_kt"].apply(clean_numeric) if "max_wind_kt" in df.columns else np.nan
        df["pressure_drop_hpa"] = df["pressure_drop_hpa"].apply(clean_numeric) if "pressure_drop_hpa" in df.columns else np.nan

        # Wind speed in km/h
        df["max_wind_kmh"] = (df["max_wind_kt"] * 1.852).round(1)

        # Grade standardization
        df["grade"] = [
            classify_imd_grade(w, g)
            for w, g in zip(df["max_wind_kt"], df.get("grade_raw", [np.nan] * len(df)))
        ]

        # Standardize dates
        parsed_dates = []
        for d in df["date_raw"]:
            if isinstance(d, pd.Timestamp):
                parsed_dates.append(d.strftime("%Y-%m-%d"))
            else:
                try:
                    dt = pd.to_datetime(d, dayfirst=True)
                    parsed_dates.append(dt.strftime("%Y-%m-%d"))
                except Exception:
                    parsed_dates.append(f"{year_int}-01-01")
        df["date"] = parsed_dates
        df["time_str"] = df["time_raw"].apply(parse_time_str)
        df["timestamp_utc"] = pd.to_datetime(df["date"] + " " + df["time_str"], errors="coerce").fillna(pd.to_datetime(df["date"], errors="coerce"))

        # System identifier
        df["serial_str"] = df["serial_no"].apply(lambda s: f"{int(s):02d}" if pd.notna(s) and isinstance(s, (int, float)) else "01")
        df["basin_str"] = df["basin"].apply(lambda b: str(b).strip().upper() if pd.notna(b) else "BOB")
        df["name_str"] = df["cyclone_name"].apply(lambda n: str(n).strip().upper() if pd.notna(n) and str(n).strip().upper() not in ["NAN", "NONE", ""] else "UNNAMED")
        df["system_id"] = df["year"].astype(str) + "_" + df["basin_str"] + "_" + df["serial_str"]

        clean_cols = [
            "system_id", "year", "basin_str", "name_str",
            "timestamp_utc", "date", "latitude", "longitude",
            "central_pressure_hpa", "max_wind_kt", "max_wind_kmh",
            "pressure_drop_hpa", "grade"
        ]
        all_tracks.append(df[clean_cols].copy())

    master_tracks = pd.concat(all_tracks, ignore_index=True)
    master_tracks = master_tracks.rename(columns={"basin_str": "basin", "name_str": "name"})
    master_tracks = master_tracks.sort_values(by=["timestamp_utc", "system_id"]).reset_index(drop=True)

    logger.info(f"Parsed {len(master_tracks):,} cyclone trajectory points across {master_tracks['system_id'].nunique()} storms in {time.time()-t0:.1f}s.")

    # Spatial Proximity Analysis for All 50 Indian Cities
    logger.info("Computing spatial proximity buffers (100km, 250km, 500km) for all 50 cities...")
    track_lats = master_tracks["latitude"].to_numpy()
    track_lons = master_tracks["longitude"].to_numpy()

    # Precompute pairwise distances: (N_tracks, 50)
    # Vectorized across tracks and cities
    city_risk_records = []
    severe_grades = {"VSCS", "ESCS", "SuCS"}

    for c_idx, c_name in enumerate(city_names):
        c_lat = city_lats[c_idx]
        c_lon = city_lons[c_idx]

        dists = haversine_distance_vectorized(c_lat, c_lon, track_lats, track_lons)
        within_500 = master_tracks[dists <= 500.0].copy()
        within_500["distance_to_city_km"] = np.round(dists[dists <= 500.0], 1)

        total_storms_500km = within_500["system_id"].nunique()
        within_250 = within_500[within_500["distance_to_city_km"] <= 250.0]
        total_storms_250km = within_250["system_id"].nunique()
        within_100 = within_500[within_500["distance_to_city_km"] <= 100.0]
        total_direct_hits_100km = within_100["system_id"].nunique()

        severe_storms_250km = within_250[within_250["grade"].isin(severe_grades)]["system_id"].nunique()
        min_distance_km = float(np.round(np.min(dists), 1))

        # Peak historical wind speed within 250km
        peak_wind_kmh = float(within_250["max_wind_kmh"].max()) if len(within_250) > 0 and within_250["max_wind_kmh"].notnull().any() else 0.0

        # Most notable storms
        closest_row = within_500.sort_values(by="distance_to_city_km").iloc[0] if len(within_500) > 0 else None
        closest_storm = f"{closest_row['name']} ({closest_row['year']})" if closest_row is not None and closest_row['name'] != "UNNAMED" else (f"{closest_row['system_id']}" if closest_row is not None else "None")

        # Vulnerability Tier
        if total_direct_hits_100km >= 5 or total_storms_250km >= 15:
            tier = "EXTREME_COASTAL_VULNERABILITY"
        elif total_storms_250km >= 6:
            tier = "HIGH_CYCLONE_HAZARD"
        elif total_storms_500km >= 10:
            tier = "MODERATE_INLAND_DEPRESSION_RISK"
        else:
            tier = "LOW_SHELTERED"

        city_risk_records.append({
            "city": c_name,
            "latitude": c_lat,
            "longitude": c_lon,
            "cyclone_hazard_tier": tier,
            "total_storms_500km": int(total_storms_500km),
            "total_storms_250km": int(total_storms_250km),
            "direct_strikes_100km": int(total_direct_hits_100km),
            "severe_cyclones_250km": int(severe_storms_250km),
            "min_historic_distance_km": min_distance_km,
            "peak_historic_wind_kmh": peak_wind_kmh,
            "closest_storm": closest_storm,
        })

    city_risk_df = pd.DataFrame(city_risk_records)
    city_risk_df = city_risk_df.sort_values(by=["total_storms_250km", "direct_strikes_100km"], ascending=False).reset_index(drop=True)

    # Export Parquet datasets
    tracks_parquet = proc_dir / "north_indian_ocean_cyclone_tracks_1982_2026.parquet"
    city_risk_parquet = proc_dir / "city_cyclone_risk_profiles.parquet"
    city_risk_json = proc_dir / "city_cyclone_risk_profiles.json"

    master_tracks.to_parquet(tracks_parquet, index=False)
    city_risk_df.to_parquet(city_risk_parquet, index=False)

    with open(city_risk_json, "w", encoding="utf-8") as f:
        json.dump(city_risk_records, f, indent=2)

    logger.info(f"Exported Master Cyclone Tracks ({len(master_tracks):,} rows) to {tracks_parquet}.")
    logger.info(f"Exported City Cyclone Risk Profiles ({len(city_risk_df)} cities) to {city_risk_parquet}.")

    # Write Manifest
    manifest_path = root / "data" / "manifests" / "cyclones_manifest.json"
    manifest = {
        "source_id": "imd_cyclone_best_tracks_1982_2026",
        "display_name": "IMD RSMC New Delhi North Indian Ocean Tropical Cyclone Best Tracks (1982-2026)",
        "provider": "India Meteorological Department (Cyclone eAtlas & RSMC New Delhi)",
        "spatial_coverage": "North Indian Ocean (Bay of Bengal, Arabian Sea, Coastal India)",
        "temporal_coverage": "1982 to 2026 (45 years)",
        "total_cyclonic_systems": int(master_tracks["system_id"].nunique()),
        "total_trajectory_points": len(master_tracks),
        "cities_assessed": len(city_risk_df),
        "raw_source_file": raw_file.name,
        "sha256_raw": compute_sha256(raw_file),
        "sha256_tracks_parquet": compute_sha256(tracks_parquet),
        "sha256_city_risk_parquet": compute_sha256(city_risk_parquet),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "cyclone_intensity_scale": {
            "D": "Depression (17-27 kt, 31-49 km/h)",
            "DD": "Deep Depression (28-33 kt, 50-61 km/h)",
            "CS": "Cyclonic Storm (34-47 kt, 62-88 km/h)",
            "SCS": "Severe Cyclonic Storm (48-63 kt, 89-117 km/h)",
            "VSCS": "Very Severe Cyclonic Storm (64-89 kt, 118-166 km/h)",
            "ESCS": "Extremely Severe Cyclonic Storm (90-119 kt, 167-221 km/h)",
            "SuCS": "Super Cyclonic Storm (>= 120 kt, >= 222 km/h)"
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Cyclone manifest written successfully.")
    return master_tracks, city_risk_df


if __name__ == "__main__":
    ingest_cyclone_best_tracks()
