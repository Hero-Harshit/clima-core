"""Analyze missingness and data coverage across all 242 catchments in CAMELS-IND v2.2."""

import pandas as pd
import numpy as np

obs_df = pd.read_csv("data/raw/camels_ind/extracted/streamflow_timeseries/streamflow_observed.csv")
name_df = pd.read_csv("data/raw/camels_ind/extracted/attributes_csv/camels_ind_name.csv")
topo_df = pd.read_csv("data/raw/camels_ind/extracted/attributes_csv/camels_ind_topo.csv")
clim_df = pd.read_csv("data/raw/camels_ind/extracted/attributes_csv/camels_ind_clim.csv")

total_days = len(obs_df)
print(f"Total time steps: {total_days} days (1980-2020)")

# Create datetime index
dates = pd.to_datetime(obs_df[["year", "month", "day"]])
train_mask = (dates >= "1980-01-01") & (dates <= "2010-12-31")
val_mask = (dates >= "2011-01-01") & (dates <= "2015-12-31")
test_mask = (dates >= "2016-01-01") & (dates <= "2020-12-31")

print(f"Split days: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")

gauge_cols = [c for c in obs_df.columns if c not in ["year", "month", "day"]]
print(f"Total catchments in observed streamflow: {len(gauge_cols)}")

records = []
for gid in gauge_cols:
    s = obs_df[gid]
    valid_total = s.notnull().sum()
    pct_total = (valid_total / total_days) * 100.0
    
    valid_train = s[train_mask].notnull().sum()
    pct_train = (valid_train / train_mask.sum()) * 100.0
    
    valid_val = s[val_mask].notnull().sum()
    pct_val = (valid_val / val_mask.sum()) * 100.0
    
    valid_test = s[test_mask].notnull().sum()
    pct_test = (valid_test / test_mask.sum()) * 100.0
    
    # Negative values check
    neg_count = (s < 0).sum()
    
    records.append({
        "gauge_id": int(gid),
        "valid_total": int(valid_total),
        "pct_total": round(float(pct_total), 2),
        "pct_train": round(float(pct_train), 2),
        "pct_val": round(float(pct_val), 2),
        "pct_test": round(float(pct_test), 2),
        "neg_count": int(neg_count),
    })

df_cov = pd.DataFrame(records)

# Threshold distribution
print("\nCatchment Distribution by Total Coverage:")
for thresh in [90, 80, 70, 60, 50, 40, 30, 20, 10]:
    count = (df_cov["pct_total"] >= thresh).sum()
    print(f"  >= {thresh}% coverage: {count} catchments ({count / len(gauge_cols):.1%})")

# Coverage across all three splits simultaneously
# A rigorous benchmark requires valid observations in train, val, and test periods!
cov_all_70 = df_cov[(df_cov["pct_train"] >= 50) & (df_cov["pct_val"] >= 50) & (df_cov["pct_test"] >= 50)]
print(f"\nCatchments with >= 50% in Train, Val, AND Test: {len(cov_all_70)}")

cov_70_strict = df_cov[(df_cov["pct_train"] >= 70) & (df_cov["pct_val"] >= 70) & (df_cov["pct_test"] >= 70)]
print(f"Catchments with >= 70% in Train, Val, AND Test: {len(cov_70_strict)}")

cov_total_70 = df_cov[df_cov["pct_total"] >= 70]
print(f"Catchments with >= 70% Total Coverage: {len(cov_total_70)}")

# Save coverage report
df_cov.to_csv("scratch/camels_ind_catchment_coverage.csv", index=False)
print("Saved detailed catchment coverage to scratch/camels_ind_catchment_coverage.csv")
