"""Download and verify CAMELS-IND v2.2 Streamflow Sufficient Archive from Zenodo."""

import os
import hashlib
import urllib.request
import zipfile
import time

FILES_TO_DOWNLOAD = [
    {
        "filename": "00_CAMELS_IND_Data_Description.pdf",
        "url": "https://zenodo.org/api/records/14999580/files/00_CAMELS_IND_Data_Description.pdf/content",
        "expected_md5": "19cf924c9a0a5417163ba773432ac274",
        "expected_size_mb": 0.52,
    },
    {
        "filename": "CAMELS_IND_Catchments_Streamflow_Sufficient.zip",
        "url": "https://zenodo.org/api/records/14999580/files/CAMELS_IND_Catchments_Streamflow_Sufficient.zip/content",
        "expected_md5": "3993c25ba7d7b86df0541de91e094f39",
        "expected_size_mb": 169.89,
    },
]

RAW_DIR = "data/raw/camels_ind"
EXTRACT_DIR = os.path.join(RAW_DIR, "extracted")


def compute_file_md5(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(item: dict) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    dest_path = os.path.join(RAW_DIR, item["filename"])
    
    if os.path.exists(dest_path):
        actual_md5 = compute_file_md5(dest_path)
        if actual_md5 == item["expected_md5"]:
            print(f"File {item['filename']} already downloaded and verified (MD5: {actual_md5}).")
            return dest_path
        else:
            print(f"Checksum mismatch for existing {item['filename']}. Re-downloading...")

    print(f"Downloading {item['filename']} (~{item['expected_size_mb']} MB) from Zenodo...")
    t0 = time.time()
    
    req = urllib.request.Request(item["url"], headers={"User-Agent": "ClimaCore/2.0 Research Client"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        downloaded = 0
        total_size = int(resp.headers.get("content-length", 0))
        while chunk := resp.read(1024 * 512):
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0 and downloaded % (1024 * 1024 * 20) < (1024 * 512):
                pct = (downloaded / total_size) * 100
                print(f"  {item['filename']}: {downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB ({pct:.1f}%)")

    elapsed = time.time() - t0
    actual_md5 = compute_file_md5(dest_path)
    print(f"Finished {item['filename']} in {elapsed:.1f}s. Verified MD5: {actual_md5}")
    assert actual_md5 == item["expected_md5"], f"MD5 mismatch: got {actual_md5}, expected {item['expected_md5']}"
    return dest_path


def extract_zip(zip_path: str):
    print(f"Extracting {zip_path} to {EXTRACT_DIR}...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    for item in FILES_TO_DOWNLOAD:
        path = download_file(item)
        if item["filename"].endswith(".zip"):
            extract_zip(path)
    
    print("\nExtraction complete. Listing top-level extracted contents:")
    for root, dirs, files in os.walk(EXTRACT_DIR):
        rel = os.path.relpath(root, EXTRACT_DIR)
        print(f"[{rel}] Dirs: {len(dirs)}, Files: {len(files)}")
        if len(files) > 0 and len(files) <= 10:
            for f in files:
                print(f"   {f}")
        elif len(files) > 10:
            print(f"   Sample: {files[:5]} ... ({len(files)} files)")
