"""CAMELS-IND v2.2 Dataset Ingestion, Catchment Selection, and Temporal-Spatial Splits.
Extracts genuine observed streamflow ground-truth, daily hydrometeorological forcing,
and static catchment attributes across Peninsular Indian catchments.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

RAW_CAMELS_DIR = "data/raw/camels_ind/extracted"
PROCESSED_CAMELS_META = "data/processed/camels_ind_selected_catchments.json"

# Date split boundaries
TRAIN_START = "1980-01-01"
TRAIN_END = "2008-12-31"    # 29 years
VAL_START = "2009-01-01"
VAL_END = "2013-12-31"      # 5 years
TEST_START = "2014-01-01"
TEST_END = "2018-12-31"     # 5 years


def select_and_partition_catchments(
    min_coverage_pct: float = 70.0,
    random_state: int = 42,
    ungauged_holdout_count: int = 20,
) -> Dict[str, Any]:
    """Selects catchments meeting minimum coverage in all 3 splits and partitions into gauged / ungauged."""
    obs_file = os.path.join(RAW_CAMELS_DIR, "streamflow_timeseries", "streamflow_observed.csv")
    name_file = os.path.join(RAW_CAMELS_DIR, "attributes_csv", "camels_ind_name.csv")
    topo_file = os.path.join(RAW_CAMELS_DIR, "attributes_csv", "camels_ind_topo.csv")
    clim_file = os.path.join(RAW_CAMELS_DIR, "attributes_csv", "camels_ind_clim.csv")
    soil_file = os.path.join(RAW_CAMELS_DIR, "attributes_csv", "camels_ind_soil.csv")
    land_file = os.path.join(RAW_CAMELS_DIR, "attributes_csv", "camels_ind_land.csv")

    obs_df = pd.read_csv(obs_file)
    dates = pd.to_datetime(obs_df[["year", "month", "day"]])
    m_tr = (dates >= TRAIN_START) & (dates <= TRAIN_END)
    m_val = (dates >= VAL_START) & (dates <= VAL_END)
    m_te = (dates >= TEST_START) & (dates <= TEST_END)

    gauge_cols = [c for c in obs_df.columns if c not in ["year", "month", "day"]]

    name_df = pd.read_csv(name_file).set_index("gauge_id")
    topo_df = pd.read_csv(topo_file).set_index("gauge_id")
    clim_df = pd.read_csv(clim_file).set_index("gauge_id")
    soil_df = pd.read_csv(soil_file).set_index("gauge_id")
    land_df = pd.read_csv(land_file).set_index("gauge_id")

    qualifying_gauges = []
    catchment_info = {}

    for gid_str in gauge_cols:
        gid = int(gid_str)
        s = obs_df[gid_str]
        
        p_tr = (s[m_tr].notnull().sum() / m_tr.sum()) * 100.0
        p_val = (s[m_val].notnull().sum() / m_val.sum()) * 100.0
        p_te = (s[m_te].notnull().sum() / m_te.sum()) * 100.0
        p_tot = (s.notnull().sum() / len(obs_df)) * 100.0
        
        if p_tr >= min_coverage_pct and p_val >= min_coverage_pct and p_te >= min_coverage_pct:
            qualifying_gauges.append(gid)
            area_km2 = float(topo_df.loc[gid, "ghi_area"]) if gid in topo_df.index else 1000.0
            basin = str(name_df.loc[gid, "river_basin"]) if gid in name_df.index else "Other"
            aridity = float(clim_df.loc[gid, "aridity_p_pet"]) if gid in clim_df.index else 1.0
            elev = float(topo_df.loc[gid, "elev_mean"]) if gid in topo_df.index else 300.0
            slope = float(topo_df.loc[gid, "slope_mean"]) if gid in topo_df.index else 2.0
            soil_cond = float(soil_df.loc[gid, "soil_conductivity_top"]) if gid in soil_df.index else 10.0
            forest_frac = float(land_df.loc[gid, "trees_frac"]) if gid in land_df.index else 0.2
            crops_frac = float(land_df.loc[gid, "crops_frac"]) if gid in land_df.index else 0.5
            
            catchment_info[str(gid)] = {
                "gauge_id": gid,
                "area_km2": round(area_km2, 2),
                "river_basin": basin,
                "aridity_p_pet": round(aridity, 3),
                "elev_mean_m": round(elev, 1),
                "slope_mean_deg": round(slope, 2),
                "soil_conductivity_top": round(soil_cond, 2),
                "forest_frac": round(forest_frac, 3),
                "crops_frac": round(crops_frac, 3),
                "coverage_pct": {
                    "total": round(p_tot, 1),
                    "train": round(p_tr, 1),
                    "val": round(p_val, 1),
                    "test": round(p_te, 1),
                },
            }

    print(f"Total qualifying catchments (>= {min_coverage_pct}% coverage across all 3 splits): {len(qualifying_gauges)}")

    # Stratified split into 89 gauged (training/seen) and 20 ungauged (spatial holdout/unseen)
    rng = np.random.default_rng(random_state)
    
    # Stratify by major basins
    basin_groups = {}
    for gid in qualifying_gauges:
        b = catchment_info[str(gid)]["river_basin"]
        basin_groups.setdefault(b, []).append(gid)

    heldout_ungauged = []
    # Pick 1-2 from each substantial basin to guarantee geographic diversity
    for b, g_list in sorted(basin_groups.items(), key=lambda x: len(x[1]), reverse=True):
        if len(heldout_ungauged) < ungauged_holdout_count and len(g_list) >= 3:
            chosen = rng.choice(g_list, size=min(2, len(g_list) // 3), replace=False).tolist()
            heldout_ungauged.extend(chosen)

    # Fill remainder randomly if needed
    remaining_candidates = [g for g in qualifying_gauges if g not in heldout_ungauged]
    needed = ungauged_holdout_count - len(heldout_ungauged)
    if needed > 0:
        extra = rng.choice(remaining_candidates, size=needed, replace=False).tolist()
        heldout_ungauged.extend(extra)
        
    heldout_ungauged = sorted(heldout_ungauged[:ungauged_holdout_count])
    gauged_seen = sorted([g for g in qualifying_gauges if g not in heldout_ungauged])

    manifest = {
        "dataset_name": "CAMELS-IND v2.2 Streamflow Sufficient",
        "version": "2.2",
        "zenodo_record": "14999580",
        "license": "CC-BY-4.0",
        "total_archive_catchments": len(gauge_cols),
        "qualifying_catchments_count": len(qualifying_gauges),
        "min_coverage_threshold_pct": min_coverage_pct,
        "date_splits": {
            "train": [TRAIN_START, TRAIN_END],
            "val": [VAL_START, VAL_END],
            "test": [TEST_START, TEST_END],
        },
        "spatial_partition": {
            "gauged_seen_count": len(gauged_seen),
            "ungauged_holdout_count": len(heldout_ungauged),
            "gauged_seen_ids": [str(g) for g in gauged_seen],
            "ungauged_holdout_ids": [str(g) for g in heldout_ungauged],
        },
        "catchments": catchment_info,
    }

    os.makedirs(os.path.dirname(PROCESSED_CAMELS_META), exist_ok=True)
    with open(PROCESSED_CAMELS_META, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved partition manifest to {PROCESSED_CAMELS_META} (Gauged={len(gauged_seen)}, Ungauged={len(heldout_ungauged)})")

    return manifest


def load_catchment_timeseries(gauge_id: int) -> pd.DataFrame:
    """Loads daily meteorology and streamflow for a single catchment.
    Returns DataFrame with columns: date, prcp, pet, tavg, q_obs_m3s, q_obs_mm
    """
    gid_str_5 = f"{gauge_id:05d}"
    gid_str = str(gauge_id)
    
    forcing_file = os.path.join(RAW_CAMELS_DIR, "catchment_mean_forcings", f"{gid_str_5}.csv")
    if not os.path.exists(forcing_file):
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

    f_df = pd.read_csv(forcing_file)
    f_df["date"] = pd.to_datetime(f_df[["year", "month", "day"]])
    
    # Load observed streamflow
    obs_file = os.path.join(RAW_CAMELS_DIR, "streamflow_timeseries", "streamflow_observed.csv")
    # Read only date and target column for memory efficiency
    cols = ["year", "month", "day", gid_str]
    obs_df = pd.read_csv(obs_file, usecols=cols)
    obs_df["date"] = pd.to_datetime(obs_df[["year", "month", "day"]])
    
    # Topo area for specific discharge conversion
    topo_file = os.path.join(RAW_CAMELS_DIR, "attributes_csv", "camels_ind_topo.csv")
    topo_df = pd.read_csv(topo_file).set_index("gauge_id")
    area_km2 = float(topo_df.loc[gauge_id, "ghi_area"]) if gauge_id in topo_df.index else 1000.0

    df = pd.DataFrame({
        "date": f_df["date"],
        "prcp": f_df["prcp(mm/day)"].astype(np.float32),
        "pet": f_df["pet(mm/day)"].astype(np.float32),
        "tavg": f_df["tavg(C)"].astype(np.float32),
        "q_obs_m3s": obs_df[gid_str].astype(np.float32),
    })

    # Robust imputation for forcing fields (e.g. Year 1980 PET missing in CAMELS-IND)
    if df["pet"].isna().any():
        doy_pet = df.groupby(df["date"].dt.dayofyear)["pet"].transform("mean")
        df["pet"] = df["pet"].fillna(doy_pet).bfill().ffill().astype(np.float32)
    if df["prcp"].isna().any():
        df["prcp"] = df["prcp"].fillna(0.0).astype(np.float32)
    if df["tavg"].isna().any():
        df["tavg"] = df["tavg"].ffill().bfill().astype(np.float32)
    
    # Convert m3/s to mm/day: q_mm = (q_m3s * 86.4) / area_km2
    df["q_obs_mm"] = (df["q_obs_m3s"] * 86.4 / area_km2).astype(np.float32)
    df["area_km2"] = area_km2
    
    return df


def get_static_attributes(gauge_id: int) -> np.ndarray:
    """Returns normalized static attribute vector: [log(area), elev, slope, aridity, forest, crops, soil_cond]."""
    meta_path = PROCESSED_CAMELS_META
    if not os.path.exists(meta_path):
        select_and_partition_catchments()
    with open(meta_path) as f:
        meta = json.load(f)
    info = meta["catchments"].get(str(gauge_id), {})
    
    vec = np.array([
        np.log1p(max(info.get("area_km2", 1000.0), 10.0)) / 10.0,
        info.get("elev_mean_m", 300.0) / 1000.0,
        info.get("slope_mean_deg", 2.0) / 10.0,
        min(info.get("aridity_p_pet", 1.0), 3.0) / 3.0,
        info.get("forest_frac", 0.2),
        info.get("crops_frac", 0.5),
        min(info.get("soil_conductivity_top", 10.0), 50.0) / 50.0,
    ], dtype=np.float32)
    return vec


class StreamflowSequenceDataset(Dataset):
    """Memory-efficient PyTorch Dataset creating sliding sequences on the fly."""

    def __init__(
        self,
        catchment_dfs: Dict[int, pd.DataFrame],
        seq_len: int = 180,
        static_attrs: Optional[Dict[int, np.ndarray]] = None,
        date_mask: Optional[Tuple[str, str]] = None,
        stride: int = 10,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.samples = []  # List of (gauge_id, start_idx)
        self.data_dict = {}
        self.static_dict = static_attrs or {}

        for gid, df in catchment_dfs.items():
            if date_mask is not None:
                m = (df["date"] >= date_mask[0]) & (df["date"] <= date_mask[1])
                sub_df = df[m].copy().reset_index(drop=True)
            else:
                sub_df = df.copy().reset_index(drop=True)

            if len(sub_df) < seq_len + 10:
                continue

            # Normalized features: [prcp / 50.0, pet / 10.0, tavg / 40.0]
            forcing = np.column_stack([
                sub_df["prcp"].values / 50.0,
                sub_df["pet"].values / 10.0,
                (sub_df["tavg"].values - 20.0) / 20.0,
            ]).astype(np.float32)
            
            # Target streamflow in mm/day
            q_obs = sub_df["q_obs_mm"].values.astype(np.float32)
            
            # Valid target mask (non-NaN, non-negative)
            valid_mask = (~np.isnan(q_obs)) & (q_obs >= 0.0)
            
            self.data_dict[gid] = {
                "forcing": forcing,
                "q_obs": q_obs,
                "valid_mask": valid_mask,
                "raw_prcp": sub_df["prcp"].values.astype(np.float32),
                "raw_pet": sub_df["pet"].values.astype(np.float32),
            }
            
            # Step size for sequence indexing
            n_steps = len(sub_df)
            for start_idx in range(0, n_steps - seq_len, self.stride):
                target_idx = start_idx + seq_len - 1
                if valid_mask[target_idx]:  # Only train where ground truth exists!
                    self.samples.append((gid, start_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gid, start_idx = self.samples[idx]
        d = self.data_dict[gid]
        end_idx = start_idx + self.seq_len
        
        x_seq = d["forcing"][start_idx:end_idx]  # Shape (seq_len, 3)
        raw_p_seq = d["raw_prcp"][start_idx:end_idx]
        raw_pet_seq = d["raw_pet"][start_idx:end_idx]
        
        static_vec = self.static_dict.get(gid, np.zeros(7, dtype=np.float32))
        # Tile static vector across time steps
        static_seq = np.tile(static_vec, (self.seq_len, 1))
        
        x_full = np.hstack([x_seq, static_seq])  # Shape (seq_len, 10)
        target_q = d["q_obs"][end_idx - 1]
        
        return {
            "x": torch.from_numpy(x_full),
            "raw_p": torch.from_numpy(raw_p_seq),
            "raw_pet": torch.from_numpy(raw_pet_seq),
            "target_q": torch.tensor(target_q, dtype=torch.float32),
            "gauge_id": torch.tensor(gid, dtype=torch.long),
        }
