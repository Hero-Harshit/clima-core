"""
Validation Tri-Split Engine for Clima-Core (Phase 1 Scientific Audit).
Provides three rigorous partitioning strategies to prevent spatial and temporal data leakage:
  1. Temporal Validation (chronological time-series with optional purge embargo)
  2. Spatial Validation (Leave-One-Climate-Zone-Out across 5 Indian Bioclimatic Zones)
  3. Event-Safe Partitioning (atomically isolates complete disaster episodes/cyclone tracks)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Generator, Optional
import numpy as np
import pandas as pd

# Canonical 5 Indian Bioclimatic Zones across all 50 cities
BIOCLIMATIC_ZONES: Dict[str, List[str]] = {
    "INDO_GANGETIC_PLAINS": [
        "delhi", "lucknow", "kanpur", "patna", "agra", "varanasi", "prayagraj", "amritsar", "ludhiana", "chandigarh"
    ],
    "DECCAN_PLATEAU_SEMI_ARID": [
        "hyderabad", "bengaluru", "pune", "nagpur", "indore", "bhopal", "jaipur", "jodhpur", "gwalior", "raipur",
        "ranchi", "coimbatore", "madurai", "vijayawada", "mysuru", "vadodara", "nashik"
    ],
    "WESTERN_COASTAL": [
        "mumbai", "surat", "kochi", "thiruvananthapuram", "panaji", "mangaluru"
    ],
    "EASTERN_COASTAL": [
        "chennai", "kolkata", "visakhapatnam", "bhubaneswar", "puducherry", "port blair"
    ],
    "HIMALAYAN_HILL_NE": [
        "srinagar", "jammu", "dehradun", "shimla", "leh", "guwahati", "shillong", "imphal", "agartala", "gangtok"
    ],
}

_CITY_TO_ZONE_MAP: Dict[str, str] = {}
for zone_name, cities in BIOCLIMATIC_ZONES.items():
    for c in cities:
        _CITY_TO_ZONE_MAP[c.lower()] = zone_name


def assign_city_climate_zone(city_name: str) -> str:
    """Deterministically resolve any Indian city to its canonical bioclimatic zone."""
    norm = city_name.strip().lower()
    norm = norm.replace("bangalore", "bengaluru").replace("calcutta", "kolkata").replace("madras", "chennai")
    return _CITY_TO_ZONE_MAP.get(norm, "DECCAN_PLATEAU_SEMI_ARID")


def chronological_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    val_start: str = "2025-06-01",
    test_start: str = "2026-01-01",
    purge_days: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train, validation, and test subsets with optional purge embargo.
    - Train: date < (val_start - purge_days)
    - Validation: val_start <= date < (test_start - purge_days)
    - Test: date >= test_start
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    val_dt = pd.to_datetime(val_start)
    test_dt = pd.to_datetime(test_start)

    purge_td = pd.Timedelta(days=purge_days)
    train_end = val_dt - purge_td
    val_end = test_dt - purge_td

    train_df = df[df[date_col] < train_end].copy()
    val_df = df[(df[date_col] >= val_dt) & (df[date_col] < val_end)].copy()
    test_df = df[df[date_col] >= test_dt].copy()

    return train_df, val_df, test_df


def leave_one_climate_zone_out_split(
    df: pd.DataFrame,
    city_col: str = "city",
) -> Generator[Tuple[str, pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generator yielding 5 spatial leave-one-climate-zone-out folds:
    (held_out_zone_name, train_df, test_df)
    Ensures model is evaluated on entire geographical regions it has never seen during training.
    """
    df = df.copy()
    if "climate_zone" not in df.columns:
        df["climate_zone"] = df[city_col].apply(assign_city_climate_zone)

    for zone in BIOCLIMATIC_ZONES.keys():
        test_mask = df["climate_zone"] == zone
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        yield zone, train_df, test_df


def event_safe_train_val_test_split(
    df: pd.DataFrame,
    event_id_col: str = "event_id",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partition dataset by grouping entire disaster episodes or storm tracks atomically.
    Guarantees no individual event (e.g. Cyclone Fani or 2018 Kerala Floods) has observations
    straddling both training and evaluation partitions.
    """
    df = df.copy()
    unique_events = df[event_id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    shuffled_events = rng.permutation(unique_events)

    n_events = len(shuffled_events)
    n_train = int(n_events * train_ratio)
    n_val = int(n_events * val_ratio)

    train_events = set(shuffled_events[:n_train])
    val_events = set(shuffled_events[n_train:n_train + n_val])
    test_events = set(shuffled_events[n_train + n_val:])

    train_df = df[df[event_id_col].isin(train_events)].copy()
    val_df = df[df[event_id_col].isin(val_events)].copy()
    test_df = df[df[event_id_col].isin(test_events)].copy()

    return train_df, val_df, test_df
