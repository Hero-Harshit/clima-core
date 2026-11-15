"""NWIC / India-WRIS Automated Water Ingestion Pipeline.

Automates the retrieval of Central Ground Water Board (CGWB) groundwater levels,
Central Water Commission (CWC) river discharge/levels, and national reservoir
storage data from the National Water Data Portal (NWIC CKAN API) without manual clicking.
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request

import pandas as pd

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.nwic_water")

NWIC_CKAN_BASE = "https://nwdp.nwic.gov.in/api/3/action"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def query_ckan_package(package_query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search for packages matching a query on NWIC CKAN API."""
    encoded_query = urllib.parse.quote(package_query)
    url = f"{NWIC_CKAN_BASE}/package_search?q={encoded_query}&rows={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("success"):
                return data["result"]["results"]
    except Exception as e:
        logger.error(f"Failed to query NWIC CKAN API for '{package_query}': {e}")
    return []


def download_single_csv(res_meta: Dict[str, Any], output_dir: Path) -> Optional[Path]:
    """Download a single resource CSV file and save to disk."""
    name = res_meta["name"]
    url = res_meta["url"]
    clean_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_")
    if not clean_name.lower().endswith(".csv"):
        clean_name += ".csv"

    out_file = output_dir / clean_name
    if out_file.exists() and out_file.stat().st_size > 1024:
        logger.info(f"Already cached: {out_file.name} ({out_file.stat().st_size:,} bytes)")
        return out_file

    logger.info(f"Downloading: {clean_name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            if len(content) > 100:
                with open(out_file, "wb") as f:
                    f.write(content)
                logger.info(f"Saved {clean_name} ({len(content):,} bytes)")
                return out_file
    except Exception as e:
        logger.warning(f"Could not download {clean_name} ({url}): {e}")
    return None


def fetch_and_process_groundwater_data(max_states: int = 25) -> pd.DataFrame:
    """Download and compile CGWB groundwater records across Indian states."""
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "india_wris" / "groundwater"
    proc_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching NWIC CKAN catalog for CGWB Groundwater datasets...")
    packages = query_ckan_package("Central Ground Water Board Ground Water Level Manual Quarterly", limit=5)
    
    csv_resources = []
    for pkg in packages:
        for r in pkg.get("resources", []):
            if r.get("format", "").upper() == "CSV":
                # Prefer 2021-2025 and 1991-2020 files
                csv_resources.append(r)

    logger.info(f"Found {len(csv_resources)} CGWB groundwater state CSV resources. Downloading...")

    downloaded_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(download_single_csv, r, raw_dir): r for r in csv_resources[:max_states]}
        for fut in concurrent.futures.as_completed(futures):
            res_file = fut.result()
            if res_file:
                downloaded_files.append(res_file)

    logger.info(f"Successfully downloaded {len(downloaded_files)} groundwater CSV state archives.")

    # Parse and combine into a unified dataframe
    dfs = []
    for f in downloaded_files:
        try:
            df_state = pd.read_csv(f, low_memory=False)
            dfs.append(df_state)
        except Exception as e:
            logger.warning(f"Error reading {f.name}: {e}")

    if not dfs:
        logger.error("No groundwater dataframes could be parsed.")
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined Groundwater Dataset: {len(combined_df):,} records across {combined_df.get('State', pd.Series()).nunique()} states.")

    # Save processed high-speed parquet
    out_parquet = proc_dir / "cgwb_groundwater_levels.parquet"
    combined_df.to_parquet(out_parquet, index=False)
    logger.info(f"Exported combined groundwater parquet to {out_parquet} ({out_parquet.stat().st_size:,} bytes)")
    return combined_df


def fetch_and_process_reservoir_storage(max_files: int = 15) -> pd.DataFrame:
    """Download and compile CWC reservoir storage records."""
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "india_wris" / "reservoirs"
    proc_dir = root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching NWIC CKAN catalog for CWC Reservoir Water Storage datasets...")
    packages = query_ckan_package("Reservoir Water Storage Level Central Water Commission", limit=5)

    csv_resources = []
    for pkg in packages:
        for r in pkg.get("resources", []):
            if r.get("format", "").upper() == "CSV":
                csv_resources.append(r)

    logger.info(f"Found {len(csv_resources)} CWC reservoir storage CSV resources. Downloading...")

    downloaded_files = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_single_csv, r, raw_dir): r for r in csv_resources[:max_files]}
        for fut in concurrent.futures.as_completed(futures):
            res_file = fut.result()
            if res_file:
                downloaded_files.append(res_file)

    dfs = []
    for f in downloaded_files:
        try:
            df_res = pd.read_csv(f, low_memory=False)
            dfs.append(df_res)
        except Exception as e:
            logger.warning(f"Error reading reservoir CSV {f.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    out_parquet = proc_dir / "cwc_reservoir_storage.parquet"
    combined_df.to_parquet(out_parquet, index=False)
    logger.info(f"Exported combined reservoir storage parquet to {out_parquet} ({len(combined_df):,} rows)")
    return combined_df


def run_nwic_water_pipeline():
    """Execute complete automated water ingestion pipeline."""
    gw_df = fetch_and_process_groundwater_data()
    res_df = fetch_and_process_reservoir_storage()

    root = get_project_root()
    manifest = {
        "source_id": "nwic_india_water_resources",
        "display_name": "National Water Data Portal (NWIC / CGWB / CWC)",
        "groundwater": {
            "records": len(gw_df),
            "columns": list(gw_df.columns) if not gw_df.empty else [],
            "parquet_file": "data/processed/cgwb_groundwater_levels.parquet",
            "sha256": compute_sha256(root / "data" / "processed" / "cgwb_groundwater_levels.parquet") if not gw_df.empty else None
        },
        "reservoirs": {
            "records": len(res_df),
            "columns": list(res_df.columns) if not res_df.empty else [],
            "parquet_file": "data/processed/cwc_reservoir_storage.parquet",
            "sha256": compute_sha256(root / "data" / "processed" / "cwc_reservoir_storage.parquet") if not res_df.empty else None
        }
    }

    manifest_path = root / "data" / "manifests" / "india_wris_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("NWIC water pipeline complete. Manifest written to data/manifests/india_wris_manifest.json.")


if __name__ == "__main__":
    run_nwic_water_pipeline()
