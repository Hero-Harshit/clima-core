"""Automated Downloader for IMD 74-Year Daily Gridded Temperature Archives (1951-2024).

Retrieves 1.0° binary grid files (.GRD) for both:
1. Daily Maximum Temperature (MaxT_YYYY.GRD) via maxtemp.php
2. Daily Minimum Temperature (MinT_YYYY.GRD) via mintemp.php
"""

from __future__ import annotations

import calendar
import concurrent.futures
from pathlib import Path
import ssl
import time
from typing import List, Optional
import urllib.parse
import urllib.request

from climacore.common.logging import get_logger
from climacore.config.loader import get_project_root

logger = get_logger("climacore.data.sources.imd_temperature_download")

MAX_URL = "https://imdpune.gov.in/cmpg/Griddata/maxtemp.php"
MIN_URL = "https://imdpune.gov.in/cmpg/Griddata/mintemp.php"
MAX_REFERER = "https://imdpune.gov.in/cmpg/Griddata/Max_1_Bin.html"
MIN_REFERER = "https://imdpune.gov.in/cmpg/Griddata/Min_1_Bin.html"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClimaCore/0.2.0"

# IMD 1.0° grid parameters
GRID_COLS = 31  # 67.5°E to 97.5°E
GRID_ROWS = 31  # 7.5°N to 37.5°N
BYTES_PER_FLOAT = 4
BYTES_PER_DAY = GRID_COLS * GRID_ROWS * BYTES_PER_FLOAT  # 3,844 bytes


def get_expected_file_size(year: int) -> int:
    """Calculate exact expected binary size in bytes based on calendar days."""
    days = 366 if calendar.isleap(year) else 365
    return days * BYTES_PER_DAY


def download_temperature_file(
    var_type: str,  # "max" or "min"
    year: int,
    output_dir: Path,
    max_retries: int = 5,
) -> Optional[Path]:
    """Download a single year's binary temperature grid with byte-exact verification."""
    prefix = "MaxT" if var_type == "max" else "MinT"
    dest_file = output_dir / f"{prefix}_{year}.GRD"
    expected_size = get_expected_file_size(year)

    # Check if existing file is exact size
    if dest_file.exists() and dest_file.stat().st_size == expected_size:
        return dest_file

    temp_file = output_dir / f"{prefix}_{year}.GRD.tmp"
    url = MAX_URL if var_type == "max" else MIN_URL
    referer = MAX_REFERER if var_type == "max" else MIN_REFERER
    field_name = "maxtemp" if var_type == "max" else "mintemp"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    data = urllib.parse.urlencode({field_name: str(year)}).encode("utf-8")
    headers = {
        "User-Agent": USER_AGENT,
        "Referer": referer,
        "Content-Type": "application/x-www-form-urlencoded",
    }
    req = urllib.request.Request(url, data=data, headers=headers)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                if resp.status != 200:
                    raise IOError(f"HTTP {resp.status} response from server.")

                with open(temp_file, "wb") as f_out:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        f_out.write(chunk)

            size = temp_file.stat().st_size
            elapsed = time.time() - t0

            if size != expected_size:
                temp_file.unlink(missing_ok=True)
                raise ValueError(f"Size mismatch for {prefix} {year}: got {size:,} bytes, expected {expected_size:,} bytes.")

            temp_file.replace(dest_file)
            logger.info(f"[{var_type.upper()} {year}] Downloaded {size:,} bytes in {elapsed:.1f}s ({dest_file.name}).")
            return dest_file

        except Exception as e:
            temp_file.unlink(missing_ok=True)
            wait = attempt * 3.0
            logger.warning(f"[{var_type.upper()} {year}] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"[{var_type.upper()} {year}] Failed to download after {max_retries} attempts.")
    return None


def download_imd_temperature_archive(
    start_year: int = 1951,
    end_year: int = 2024,
    max_workers: int = 4,
) -> Dict[str, List[Path]]:
    """Download full 74-year daily Max and Min temperature binary grid archives."""
    root = get_project_root()
    max_dir = root / "data" / "raw" / "imd_temperature" / "max"
    min_dir = root / "data" / "raw" / "imd_temperature" / "min"
    max_dir.mkdir(parents=True, exist_ok=True)
    min_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(start_year, end_year + 1))
    logger.info(f"Initiating IMD 74-Year Temperature Grid download ({start_year} to {end_year}, 2 x {len(years)} files)...")

    results = {"max": [], "min": []}

    tasks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for y in years:
            tasks.append(executor.submit(download_temperature_file, "max", y, max_dir))
            tasks.append(executor.submit(download_temperature_file, "min", y, min_dir))

        for future in concurrent.futures.as_completed(tasks):
            try:
                res = future.result()
                if res is not None:
                    if "MaxT" in res.name:
                        results["max"].append(res)
                    else:
                        results["min"].append(res)
            except Exception as e:
                logger.error(f"Worker task error: {e}")

    logger.info(f"Download complete: {len(results['max'])} MaxT files and {len(results['min'])} MinT files ready.")
    return results


if __name__ == "__main__":
    download_imd_temperature_archive(1951, 2024, max_workers=4)
