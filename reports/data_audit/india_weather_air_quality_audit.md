# Data Audit Report: India Daily Weather & Air Quality (50 Cities)

**Audited At:** 2026-11-15T16:00:50.923225+00:00
**Validation Status:** PASSED

## 1. File Integrity & Provenance

- **weather_daily.csv**: 9,034,526 bytes | SHA-256: `6fe74082c230dcc90effaf1fea37f89050278cdf53f22a099fa1b9f4f1c58509`
- **air_quality_daily.csv**: 8,656,946 bytes | SHA-256: `dd753585283ecb449d4021b37ab1bf20c89dca13af90b21a8040520e6b10dfe5`
- **cities.csv**: 2,418 bytes | SHA-256: `b21222072aeda9571c6250d4223999ef5d62f322eff3b57ea8d5d70895e4aab0`

## 2. Coverage & Shape Summary

- **Cities Count:** 50 cities across India
- **Weather Records:** 124,750 rows (2020-01-01 to 2026-10-30)
- **Air Quality Records:** 77,400 rows (2022-08-05 to 2026-10-30)
- **Primary Key Duplicates:** Weather: 0, AQ: 0
- **City Alignment:** All 50 cities match perfectly across all 3 tables

## 3. Physical Sanity Checks

- Negative Rain: 0
- Negative PM2.5: 0
- Extreme Temperature Flag (>55°C or <-15°C): 91
