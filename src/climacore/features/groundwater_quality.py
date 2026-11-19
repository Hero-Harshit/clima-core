"""
Groundwater Chemical Quality Feature Extraction & BIS IS 10500:2012 Standards Engine.
Ingests Central Ground Water Board (CGWB) chemical quality monitoring data,
cleans geochemical concentrations, calculates Weighted Arithmetic Water Quality Index (WQI),
and flags chemical standard exceedances for drinking water safety.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.features.groundwater_quality")

# BIS IS 10500:2012 Drinking Water Standards (Acceptable & Permissible Limits in mg/L, except pH and EC)
BIS_STANDARDS: Dict[str, Dict[str, float]] = {
    "ph": {"acceptable_min": 6.5, "acceptable_max": 8.5, "permissible_min": 6.5, "permissible_max": 8.5, "unit_weight": 0.12},
    "electric_conductivity": {"acceptable": 750.0, "permissible": 2250.0, "unit_weight": 0.10},  # uS/cm
    "total_dissolved_solids": {"acceptable": 500.0, "permissible": 2000.0, "unit_weight": 0.15},  # mg/L
    "total_hardness": {"acceptable": 200.0, "permissible": 600.0, "unit_weight": 0.12},  # mg/L as CaCO3
    "chloride": {"acceptable": 250.0, "permissible": 1000.0, "unit_weight": 0.10},  # mg/L
    "sulphate": {"acceptable": 200.0, "permissible": 400.0, "unit_weight": 0.10},  # mg/L
    "calcium": {"acceptable": 75.0, "permissible": 200.0, "unit_weight": 0.08},  # mg/L
    "magnesium": {"acceptable": 30.0, "permissible": 100.0, "unit_weight": 0.08},  # mg/L
    "total_alkalinity": {"acceptable": 200.0, "permissible": 600.0, "unit_weight": 0.08},  # mg/L
    "iron": {"acceptable": 1.0, "permissible": 1.0, "unit_weight": 0.05},  # mg/L
    "uranium": {"acceptable": 0.03, "permissible": 0.03, "unit_weight": 0.02},  # mg/L (30 ug/L WHO/AERB)
}


def compute_water_quality_index(row: Dict[str, float]) -> Tuple[float, str]:
    """
    Compute Weighted Arithmetic Water Quality Index (WQI) based on Brown et al. (1970).
      q_i = 100 * (V_i - V_ideal) / (S_i - V_ideal)
      WQI = sum(W_i * q_i) / sum(W_i)
    Classification:
      WQI < 25: EXCELLENT (Grade A)
      25 - 50: GOOD (Grade B)
      51 - 75: POOR (Grade C)
      76 - 100: VERY POOR (Grade D)
      > 100: UNSUITABLE FOR DRINKING (Grade E)
    """
    q_values = []
    weights = []

    # pH
    ph_val = row.get("ph")
    if ph_val is not None and not np.isnan(ph_val) and 0.0 < ph_val < 14.0:
        # Ideal pH is 7.0, acceptable limit 8.5
        q_ph = 100.0 * abs(ph_val - 7.0) / (8.5 - 7.0)
        q_values.append(q_ph * BIS_STANDARDS["ph"]["unit_weight"])
        weights.append(BIS_STANDARDS["ph"]["unit_weight"])

    # Continuous chemical ions
    std_keys = [
        ("total_dissolved_solids", "total_dissolved_solids"),
        ("total_hardness", "total_hardness"),
        ("chloride", "chloride"),
        ("sulphate", "sulphate"),
        ("calcium", "calcium"),
        ("magnesium", "magnesium"),
    ]

    for param_name, std_key in std_keys:
        val = row.get(param_name)
        if val is not None and not np.isnan(val) and val >= 0.0:
            std = BIS_STANDARDS[std_key]["acceptable"]
            weight = BIS_STANDARDS[std_key]["unit_weight"]
            q_i = 100.0 * (val / std)
            q_values.append(q_i * weight)
            weights.append(weight)

    if not weights or sum(weights) == 0:
        return np.nan, "UNKNOWN"

    wqi = float(sum(q_values) / sum(weights))

    if wqi < 25.0:
        status = "EXCELLENT"
    elif wqi < 50.0:
        status = "GOOD"
    elif wqi < 75.0:
        status = "POOR"
    elif wqi <= 100.0:
        status = "VERY_POOR"
    else:
        status = "UNSUITABLE_FOR_DRINKING"

    return round(wqi, 2), status


def load_and_clean_cgwb_data(
    raw_csv_path: Optional[Path] = None,
    output_parquet_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Ingest, clean, and enrich CGWB groundwater quality records.
    Estimates missing TDS from EC, evaluates BIS exceedances, and calculates WQI.
    """
    root = get_project_root()
    if raw_csv_path is None:
        raw_csv_path = root / "data" / "raw" / "cgwb_groundwater_quality" / "gwq_chemical_parameter_manual_cgwb_mh_1961_2025.csv"
    if output_parquet_path is None:
        output_parquet_path = root / "data" / "processed" / "cgwb_groundwater_quality_cleaned.parquet"

    if not raw_csv_path.exists():
        raise FileNotFoundError(f"CGWB CSV not found at {raw_csv_path}")

    logger.info(f"Loading CGWB groundwater quality data from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path, low_memory=False)

    col_map = {
        "Station": "station_name",
        "District": "district",
        "Tehsil": "tehsil",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Data Acquisition Time": "acquisition_date",
        "Potential of Hydrogen (pH)": "ph",
        "Electric Conductivity (μS/cm)": "electric_conductivity",
        "Total Dissolved Solids (mg/L)": "total_dissolved_solids",
        "Total Hardness (mgCaCO3/L)": "total_hardness",
        "Chloride (mg/L)": "chloride",
        "Sulphate (mg/L)": "sulphate",
        "Calcium (mg/L)": "calcium",
        "Magnesium (mg/L)": "magnesium",
        "Sodium (mg/L)": "sodium",
        "Total Alkalinity (mg/L as CaCO3)": "total_alkalinity",
        "Iron(mg/L)": "iron",
        "Uranium(mg/L)": "uranium",
    }

    clean_df = pd.DataFrame()
    for raw_c, clean_c in col_map.items():
        if raw_c in df.columns:
            clean_df[clean_c] = df[raw_c]

    # Clean numeric columns
    numeric_cols = [
        "latitude", "longitude", "ph", "electric_conductivity", "total_dissolved_solids",
        "total_hardness", "chloride", "sulphate", "calcium", "magnesium", "sodium",
        "total_alkalinity", "iron", "uranium",
    ]
    for nc in numeric_cols:
        if nc in clean_df.columns:
            clean_df[nc] = pd.to_numeric(clean_df[nc], errors="coerce")

    # Drop invalid coordinates (India latitude ~6 to 38, longitude ~68 to 98)
    clean_df = clean_df.dropna(subset=["latitude", "longitude"]).copy()
    clean_df = clean_df[
        (clean_df["latitude"].between(6.0, 38.0)) &
        (clean_df["longitude"].between(68.0, 98.0))
    ].copy()

    # Empirical TDS estimation from EC when TDS is NaN: TDS ~ 0.64 * EC (standard limnology conversion)
    missing_tds = clean_df["total_dissolved_solids"].isna() & clean_df["electric_conductivity"].notna()
    clean_df.loc[missing_tds, "total_dissolved_solids"] = clean_df.loc[missing_tds, "electric_conductivity"] * 0.64

    # Remove extreme physical outliers (e.g. pH > 14 or < 0)
    clean_df.loc[~clean_df["ph"].between(4.0, 11.0), "ph"] = np.nan

    # Calculate BIS Exceedance Flags
    clean_df["tds_acceptable_exceeded"] = clean_df["total_dissolved_solids"] > BIS_STANDARDS["total_dissolved_solids"]["acceptable"]
    clean_df["tds_permissible_exceeded"] = clean_df["total_dissolved_solids"] > BIS_STANDARDS["total_dissolved_solids"]["permissible"]
    clean_df["hardness_acceptable_exceeded"] = clean_df["total_hardness"] > BIS_STANDARDS["total_hardness"]["acceptable"]
    clean_df["hardness_permissible_exceeded"] = clean_df["total_hardness"] > BIS_STANDARDS["total_hardness"]["permissible"]
    clean_df["chloride_exceeded"] = clean_df["chloride"] > BIS_STANDARDS["chloride"]["acceptable"]
    clean_df["uranium_toxic_exceeded"] = clean_df["uranium"] > BIS_STANDARDS["uranium"]["acceptable"]

    # Calculate WQI per sample
    wqi_scores = []
    wqi_categories = []
    for _, row in clean_df.iterrows():
        wqi, cat = compute_water_quality_index(row.to_dict())
        wqi_scores.append(wqi)
        wqi_categories.append(cat)

    clean_df["water_quality_index"] = wqi_scores
    clean_df["wqi_category"] = wqi_categories

    # Define binary non-potable hazard label (WQI >= 75 or any permissible exceedance)
    clean_df["is_non_potable_hazard"] = (
        (clean_df["water_quality_index"] >= 75.0) |
        clean_df["tds_permissible_exceeded"] |
        clean_df["hardness_permissible_exceeded"] |
        clean_df["uranium_toxic_exceeded"]
    ).astype(int)

    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_parquet(output_parquet_path, index=False)
    logger.info(f"Cleaned CGWB data saved to {output_parquet_path} ({len(clean_df)} records).")

    return clean_df
