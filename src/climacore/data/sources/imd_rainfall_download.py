"""Automated Downloader for IMD Pune 0.25° Daily Gridded Rainfall NetCDF Archive.

Downloads historical yearly NetCDF files (1980-2020) directly from IMD Pune's
gridded data portal (RF25.php) to complement existing 2021-2025 observations.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import ssl
import time
from typing import Dict, List, Optional
import urllib.parse
import urllib.request

from climacore.common.logging import get_logger
from climacore.common.provenance import compute_sha256
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.imd_rainfall_download")

IMD_URL = "https://imdpune.gov.in/cmpg/Griddata/RF25.php"
REFERER = "https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClimaCore/0.2.0"


def download_single_year(
    year: int,
    raw_dir: Path,
    max_retries: int = 5,
) -> Optional[Path]:
    """Download a single year NetCDF file from IMD Pune portal with verification."""
    dest_file = raw_dir / f"RF25_ind{year}_rfp25.nc"

    # Skip if valid and > 20 MB
    if dest_file.exists() and dest_file.stat().st_size > 20_000_000:
        logger.info(f"[{year}] Already downloaded ({dest_file.stat().st_size:,} bytes). Skipping.")
        return dest_file

    temp_file = raw_dir / f"RF25_ind{year}_rfp25.nc.tmp"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    data = urllib.parse.urlencode({"RF25": str(year)}).encode("utf-8")
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": REFERER,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    req = urllib.request.Request(IMD_URL, data=data, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            logger.info(f"[{year}] Starting download (attempt {attempt}/{max_retries})...")

            with urllib.request.urlopen(req, context=ctx, timeout=90) as resp:
                if resp.status != 200:
                    raise IOError(f"HTTP {resp.status} response from IMD server.")

                with open(temp_file, "wb") as f_out:
                    while True:
                        chunk = resp.read(1024 * 1024)  # 1MB buffer
                        if not chunk:
                            break
                        f_out.write(chunk)

            size = temp_file.stat().st_size
            elapsed = time.time() - t0

            # NetCDF files for full year are ~25-26 MB (non-leap 365 days or leap 366 days)
            if size < 20_000_000:
                temp_file.unlink(missing_ok=True)
                raise ValueError(f"Incomplete download for {year}: only {size:,} bytes received.")

            # Atomic rename
            temp_file.replace(dest_file)
            logger.info(f"[{year}] Successfully downloaded {size:,} bytes in {elapsed:.1f}s ({dest_file.name}).")
            return dest_file

        except Exception as e:
            temp_file.unlink(missing_ok=True)
            wait = attempt * 4.0
            logger.warning(f"[{year}] Download failed (attempt {attempt}): {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"[{year}] FAILED to download after {max_retries} attempts.")
    return None


def download_imd_rainfall_archive(
    start_year: int = 1980,
    end_year: int = 2020,
    max_workers: int = 3,
) -> List[Path]:
    """Download gridded rainfall archive for specified year range using parallel workers."""
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "imd_rainfall"
    raw_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(start_year, end_year + 1))
    logger.info(f"Initiating IMD Pune 0.25° Rainfall download for {len(years)} years ({start_year} to {end_year})...")

    results: List[Path] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {
            executor.submit(download_single_year, y, raw_dir): y
            for y in years
        }

        for future in concurrent.futures.as_completed(future_to_year):
            y = future_to_year[future]
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as exc:
                logger.error(f"Year {y} generated an unhandled exception: {exc}")

    logger.info(f"Download batch completed: {len(results)}/{len(years)} years successfully stored on disk.")
    return sorted(results)


if __name__ == "__main__":
    download_imd_rainfall_archive(1980, 2020, max_workers=3)
