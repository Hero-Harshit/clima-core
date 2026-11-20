"""Clima-Core v2.1 Phase Zero Baseline Audit Script."""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.getcwd())

import glob
import json
import hashlib
from pathlib import Path
import src.climacore.registry.engine as reg

def main():
    mr = reg.get_model_registry()
    dr = reg.get_dataset_registry()
    tr = reg.get_physical_thresholds_registry()
    
    print("=" * 80)
    print("PHASE ZERO BASELINE AUDIT — CLIMA-CORE V2.0 -> V2.1")
    print("=" * 80)
    
    print(f"\n1. CANONICAL MODEL REGISTRY (engine.py): {len(mr)} records")
    suites = {}
    for m_id, m in mr.items():
        suites.setdefault(m.flagship_suite, []).append((m_id, m.name, m.operational_status, m.algorithm))
        
    for s, m_list in sorted(suites.items()):
        print(f"\n--- Suite: {s} ({len(m_list)} models) ---")
        for m_id, name, status, algo in m_list:
            print(f"  * {m_id:<32} | {status:<12} | {algo:<35} | {name}")
            
    print(f"\n2. CANONICAL DATASET REGISTRY (engine.py): {len(dr)} records")
    for d_id, d in dr.items():
        exists = os.path.exists(d.processed_file_path) if d.processed_file_path else False
        sz_mb = round(os.path.getsize(d.processed_file_path) / (1024 * 1024), 2) if exists else 0.0
        print(f"  * {d_id:<28} | Exists: {str(exists):<5} ({sz_mb:>6.1f} MB) | Agency: {d.source_agency:<35} | {d.name}")
        
    print(f"\n3. PHYSICAL THRESHOLD REGISTRY: {len(tr)} domains")
    for t_id, t in tr.items():
        print(f"  * {t_id:<32} | Value: {t.threshold_value} {t.unit:<6} | Authority: {t.source_authority}")
        
    print(f"\n4. ON-DISK MODEL ARTIFACTS (artifacts/models/**):")
    art_files = [f for f in glob.glob("artifacts/models/**", recursive=True) if os.path.isfile(f)]
    print(f"Total artifact files: {len(art_files)}")
    for af in sorted(art_files):
        sz_kb = round(os.path.getsize(af) / 1024, 1)
        print(f"  * {af:<70} ({sz_kb} KB)")

if __name__ == "__main__":
    main()
