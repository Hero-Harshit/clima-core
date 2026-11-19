"""ForecastCorrectionEngine V1: Statistical & Machine Learning Forecast Calibration.

Implements matched historical forecast-observation pair analysis for numerical
weather prediction (NWP) outputs across Indian climate zones.

Capabilities:
1. Mean Bias Error (MBE) Baseline (additive for temperature, multiplicative/bounded for precipitation)
2. Empirical Quantile Mapping (EQM) with wet-day thresholding for precipitation
3. ML Residual-Correction Challenger (LightGBM regressor predicting NWP residual error)
4. Rigorous Chronological Holdout Benchmark Suite

Architectural Boundary Note:
    Bias correction and spatial downscaling are mathematically distinct tasks.
    Bias correction eliminates systematic model parameterization drift and
    instrument/grid offset at a point/station. Spatial downscaling, by contrast,
    interpolates coarse grid topographies to high-resolution DEM elevations.
    This module strictly performs bias and distribution calibration on paired records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger(__name__)


class ForecastVariable(str, Enum):
    """Supported meteorological forecast variables."""
    T2M_MAX = "t2m_max"
    T2M_MIN = "t2m_min"
    T2M_MEAN = "t2m_mean"
    PRECIPITATION = "precipitation"
    WIND_SPEED = "wind_speed"
    RELATIVE_HUMIDITY = "relative_humidity"


class CorrectionMethod(str, Enum):
    """Calibration and correction methodologies."""
    RAW = "raw"
    MEAN_BIAS = "mean_bias"
    QUANTILE_MAPPING = "quantile_mapping"
    ML_RESIDUAL = "ml_residual"


class CorrectionMetrics(BaseModel):
    """Statistical evaluation metrics computed on chronological holdout."""
    method: CorrectionMethod
    variable: str
    sample_count: int
    mae: float = Field(..., description="Mean Absolute Error in physical units")
    rmse: float = Field(..., description="Root Mean Squared Error in physical units")
    mbe: float = Field(..., description="Mean Bias Error (fcst - obs)")
    correlation_r: float = Field(..., description="Pearson correlation coefficient")
    skill_score: float = Field(..., description="Skill Score vs Raw (1 - MSE_corr / MSE_raw)")

    model_config = {"frozen": True}


class CorrectionResult(BaseModel):
    """Result of applying forecast correction to a point forecast."""
    variable: str
    lead_days: int
    raw_value: float
    corrected_value: float
    delta: float = Field(..., description="Correction applied (corrected - raw)")
    method: CorrectionMethod
    unit: str
    lower_ci_90: float
    upper_ci_90: float

    model_config = {"frozen": True}


class MeanBiasCorrection:
    """Additive or multiplicative Mean Bias Error (MBE) corrector."""

    def __init__(self, variable: ForecastVariable, additive: bool = True):
        self.variable = variable
        self.additive = additive
        self.global_offset: float = 0.0
        self.strata_offsets: Dict[Tuple[str, str], float] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, fcst_col: str, obs_col: str) -> MeanBiasCorrection:
        """Fit bias offset on training DataFrame with optional (city, season) stratification."""
        if self.additive:
            self.global_offset = float(np.mean(df[fcst_col] - df[obs_col]))
            if "city" in df.columns and "season" in df.columns:
                grouped = (df[fcst_col] - df[obs_col]).groupby([df["city"], df["season"]]).mean()
                self.strata_offsets = {k: float(v) for k, v in grouped.items()}
        else:
            # Multiplicative ratio for non-negative variables like precipitation
            sum_obs = float(df[obs_col].sum())
            sum_fcst = float(df[fcst_col].sum())
            ratio = (sum_obs + 1e-3) / (sum_fcst + 1e-3)
            # Clip ratio to prevent numerical instability in extreme dry cases
            self.global_offset = float(np.clip(ratio, 0.2, 5.0))
            if "city" in df.columns and "season" in df.columns:
                for (city, season), grp in df.groupby(["city", "season"]):
                    s_obs = float(grp[obs_col].sum())
                    s_fcst = float(grp[fcst_col].sum())
                    r = float(np.clip((s_obs + 1e-3) / (s_fcst + 1e-3), 0.2, 5.0))
                    self.strata_offsets[(city, season)] = r
        self.is_fitted = True
        return self

    def predict(self, fcst_val: float, city: Optional[str] = None, season: Optional[str] = None) -> float:
        """Apply bias correction to a single forecast value."""
        if not self.is_fitted:
            return fcst_val
        offset = self.global_offset
        if city and season and (city, season) in self.strata_offsets:
            offset = self.strata_offsets[(city, season)]

        if self.additive:
            return float(fcst_val - offset)
        else:
            return float(max(0.0, fcst_val * offset))

    def predict_series(self, df: pd.DataFrame, fcst_col: str) -> np.ndarray:
        """Apply bias correction across a pandas DataFrame."""
        if not self.is_fitted:
            return df[fcst_col].values
        res = []
        for _, row in df.iterrows():
            city = row.get("city")
            season = row.get("season")
            val = row[fcst_col]
            res.append(self.predict(val, city, season))
        return np.array(res, dtype=float)


class QuantileMappingCorrection:
    """Non-parametric Empirical Quantile Mapping (EQM) with wet-day thresholding."""

    def __init__(self, variable: ForecastVariable, n_quantiles: int = 100, wet_threshold: float = 0.1):
        self.variable = variable
        self.n_quantiles = n_quantiles
        self.wet_threshold = wet_threshold
        self.probabilities: np.ndarray = np.linspace(0.01, 0.99, n_quantiles)
        self.q_fcst: np.ndarray = np.array([])
        self.q_obs: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def fit(self, fcst_values: np.ndarray, obs_values: np.ndarray) -> QuantileMappingCorrection:
        """Fit empirical quantiles on training values."""
        clean_mask = (~np.isnan(fcst_values)) & (~np.isnan(obs_values))
        f_clean = fcst_values[clean_mask]
        o_clean = obs_values[clean_mask]

        if self.variable == ForecastVariable.PRECIPITATION:
            # Apply wet-day thresholding to avoid drizzle inflation
            f_clean = f_clean[f_clean >= self.wet_threshold]
            o_clean = o_clean[o_clean >= self.wet_threshold]

        if len(f_clean) < 10 or len(o_clean) < 10:
            logger.warning("Insufficient samples for Quantile Mapping; identity fallback.")
            self.q_fcst = self.probabilities
            self.q_obs = self.probabilities
        else:
            self.q_fcst = np.quantile(f_clean, self.probabilities)
            self.q_obs = np.quantile(o_clean, self.probabilities)

        self.is_fitted = True
        return self

    def predict(self, val: float) -> float:
        """Transfer forecast quantile to observed quantile space."""
        if not self.is_fitted:
            return val
        if self.variable == ForecastVariable.PRECIPITATION and val < self.wet_threshold:
            return 0.0
        # Map forecast value to quantile space, then interpolate into observation quantiles
        mapped = float(np.interp(val, self.q_fcst, self.q_obs))
        if self.variable == ForecastVariable.PRECIPITATION:
            mapped = max(0.0, mapped)
        return mapped

    def predict_array(self, values: np.ndarray) -> np.ndarray:
        """Apply EQM across a 1D numpy array."""
        if not self.is_fitted:
            return values
        if self.variable == ForecastVariable.PRECIPITATION:
            out = np.zeros_like(values, dtype=float)
            wet_mask = values >= self.wet_threshold
            out[wet_mask] = np.interp(values[wet_mask], self.q_fcst, self.q_obs)
            return np.maximum(0.0, out)
        else:
            return np.interp(values, self.q_fcst, self.q_obs)


class LightGBMResidualChallenger:
    """ML Challenger predicting residual error Delta = obs - fcst using companion weather features."""

    def __init__(self, variable: ForecastVariable):
        self.variable = variable
        self.booster: Optional[Any] = None
        self.feature_names: List[str] = []
        self.residual_std: float = 1.0
        self.is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        fcst_col: str,
        obs_col: str,
        feature_cols: List[str],
    ) -> LightGBMResidualChallenger:
        """Fit LightGBM regressor on residual error."""
        if not HAS_LIGHTGBM:
            logger.warning("LightGBM not installed; residual challenger disabled.")
            return self

        self.feature_names = feature_cols
        X = df[feature_cols].copy()
        y = (df[obs_col] - df[fcst_col]).values

        dataset = lgb.Dataset(X, label=y, free_raw_data=False)
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 20,
            "verbosity": -1,
            "random_state": 42,
            "force_col_wise": True,
        }
        self.booster = lgb.train(params, dataset, num_boost_round=80)
        residuals = y - self.booster.predict(X)
        self.residual_std = float(np.std(residuals))
        self.is_fitted = True
        return self

    def predict(self, raw_fcst: float, features: Dict[str, float]) -> Tuple[float, float, float]:
        """Predict calibrated value along with 90% confidence interval."""
        if not self.is_fitted or self.booster is None:
            return raw_fcst, raw_fcst - 1.645, raw_fcst + 1.645

        x_row = pd.DataFrame([{f: features.get(f, 0.0) for f in self.feature_names}])
        delta = float(self.booster.predict(x_row)[0])
        corrected = raw_fcst + delta

        if self.variable == ForecastVariable.PRECIPITATION:
            corrected = max(0.0, corrected)

        lower_ci = corrected - 1.645 * self.residual_std
        upper_ci = corrected + 1.645 * self.residual_std
        if self.variable == ForecastVariable.PRECIPITATION:
            lower_ci = max(0.0, lower_ci)

        return corrected, lower_ci, upper_ci

    def predict_df(self, df: pd.DataFrame, fcst_col: str) -> np.ndarray:
        """Predict corrected series for DataFrame."""
        if not self.is_fitted or self.booster is None:
            return df[fcst_col].values
        X = df[self.feature_names].copy()
        deltas = self.booster.predict(X)
        corrected = df[fcst_col].values + deltas
        if self.variable == ForecastVariable.PRECIPITATION:
            corrected = np.maximum(0.0, corrected)
        return corrected


class ForecastCorrectionEngine:
    """Production coordination engine for meteorological forecast verification and bias calibration."""

    VARIABLE_UNITS = {
        ForecastVariable.T2M_MAX: "°C",
        ForecastVariable.T2M_MIN: "°C",
        ForecastVariable.T2M_MEAN: "°C",
        ForecastVariable.PRECIPITATION: "mm",
        ForecastVariable.WIND_SPEED: "km/h",
        ForecastVariable.RELATIVE_HUMIDITY: "%",
    }

    def __init__(self, artifacts_dir: Union[str, Path] = "artifacts/models/forecast_correction/v0.1.0"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.mbe_models: Dict[ForecastVariable, MeanBiasCorrection] = {}
        self.qm_models: Dict[ForecastVariable, QuantileMappingCorrection] = {}
        self.ml_models: Dict[ForecastVariable, LightGBMResidualChallenger] = {}
        self.benchmarks: List[CorrectionMetrics] = []

    @staticmethod
    def compute_metrics(
        method: CorrectionMethod,
        variable: str,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        y_raw: np.ndarray,
    ) -> CorrectionMetrics:
        """Compute standardized meteorological validation metrics."""
        mask = (~np.isnan(y_obs)) & (~np.isnan(y_pred)) & (~np.isnan(y_raw))
        obs = y_obs[mask]
        pred = y_pred[mask]
        raw = y_raw[mask]

        n = len(obs)
        if n == 0:
            return CorrectionMetrics(
                method=method, variable=variable, sample_count=0,
                mae=0.0, rmse=0.0, mbe=0.0, correlation_r=0.0, skill_score=0.0
            )

        err = pred - obs
        raw_err = raw - obs

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        mbe = float(np.mean(err))

        # Correlation
        if np.std(pred) > 1e-6 and np.std(obs) > 1e-6:
            r = float(np.corrcoef(pred, obs)[0, 1])
        else:
            r = 0.0

        # Skill Score vs raw MSE
        mse_pred = np.mean(err**2)
        mse_raw = np.mean(raw_err**2)
        skill_score = float(1.0 - (mse_pred / (mse_raw + 1e-9)))

        return CorrectionMetrics(
            method=method,
            variable=variable,
            sample_count=n,
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            mbe=round(mbe, 4),
            correlation_r=round(r, 4),
            skill_score=round(skill_score, 4),
        )

    def train_and_benchmark(
        self,
        df: pd.DataFrame,
        variable: ForecastVariable,
        fcst_col: str,
        obs_col: str,
        feature_cols: List[str],
        train_cutoff_date: str = "2024-07-31",
    ) -> List[CorrectionMetrics]:
        """Train all correctors and benchmark on chronological holdout."""
        logger.info("Training and benchmarking %s (split: %s)...", variable.value, train_cutoff_date)
        train_df = df[df["date"] <= train_cutoff_date].copy().reset_index(drop=True)
        test_df = df[df["date"] > train_cutoff_date].copy().reset_index(drop=True)

        if len(train_df) == 0 or len(test_df) == 0:
            raise ValueError(f"Insufficient chronological split data for cutoff {train_cutoff_date}")

        y_test_obs = test_df[obs_col].values
        y_test_raw = test_df[fcst_col].values

        results = []

        # 1. Raw Baseline
        raw_metrics = self.compute_metrics(
            CorrectionMethod.RAW, variable.value, y_test_obs, y_test_raw, y_test_raw
        )
        results.append(raw_metrics)

        # 2. Mean Bias Baseline
        is_additive = variable != ForecastVariable.PRECIPITATION
        mbe_model = MeanBiasCorrection(variable, additive=is_additive)
        mbe_model.fit(train_df, fcst_col, obs_col)
        self.mbe_models[variable] = mbe_model
        y_test_mbe = mbe_model.predict_series(test_df, fcst_col)
        mbe_metrics = self.compute_metrics(
            CorrectionMethod.MEAN_BIAS, variable.value, y_test_obs, y_test_mbe, y_test_raw
        )
        results.append(mbe_metrics)

        # 3. Quantile Mapping
        qm_model = QuantileMappingCorrection(variable)
        qm_model.fit(train_df[fcst_col].values, train_df[obs_col].values)
        self.qm_models[variable] = qm_model
        y_test_qm = qm_model.predict_array(y_test_raw)
        qm_metrics = self.compute_metrics(
            CorrectionMethod.QUANTILE_MAPPING, variable.value, y_test_obs, y_test_qm, y_test_raw
        )
        results.append(qm_metrics)

        # 4. LightGBM Residual Challenger
        ml_model = LightGBMResidualChallenger(variable)
        ml_model.fit(train_df, fcst_col, obs_col, feature_cols)
        self.ml_models[variable] = ml_model
        y_test_ml = ml_model.predict_df(test_df, fcst_col)
        ml_metrics = self.compute_metrics(
            CorrectionMethod.ML_RESIDUAL, variable.value, y_test_obs, y_test_ml, y_test_raw
        )
        results.append(ml_metrics)

        self.benchmarks.extend(results)
        return results

    def calibrate(
        self,
        variable: ForecastVariable,
        raw_val: float,
        features: Optional[Dict[str, float]] = None,
        method: CorrectionMethod = CorrectionMethod.ML_RESIDUAL,
        lead_days: int = 1,
    ) -> CorrectionResult:
        """Calibrate an operational forecast using the requested method."""
        unit = self.VARIABLE_UNITS.get(variable, "")
        features = features or {}

        if method == CorrectionMethod.RAW:
            return CorrectionResult(
                variable=variable.value,
                lead_days=lead_days,
                raw_value=raw_val,
                corrected_value=raw_val,
                delta=0.0,
                method=CorrectionMethod.RAW,
                unit=unit,
                lower_ci_90=raw_val,
                upper_ci_90=raw_val,
            )

        elif method == CorrectionMethod.MEAN_BIAS and variable in self.mbe_models:
            model = self.mbe_models[variable]
            city = features.get("city_name") if isinstance(features.get("city_name"), str) else None
            season = features.get("season") if isinstance(features.get("season"), str) else None
            corr = model.predict(raw_val, city, season)
            return CorrectionResult(
                variable=variable.value,
                lead_days=lead_days,
                raw_value=raw_val,
                corrected_value=round(corr, 3),
                delta=round(corr - raw_val, 3),
                method=CorrectionMethod.MEAN_BIAS,
                unit=unit,
                lower_ci_90=round(corr - 1.5, 3),
                upper_ci_90=round(corr + 1.5, 3),
            )

        elif method == CorrectionMethod.QUANTILE_MAPPING and variable in self.qm_models:
            model = self.qm_models[variable]
            corr = model.predict(raw_val)
            return CorrectionResult(
                variable=variable.value,
                lead_days=lead_days,
                raw_value=raw_val,
                corrected_value=round(corr, 3),
                delta=round(corr - raw_val, 3),
                method=CorrectionMethod.QUANTILE_MAPPING,
                unit=unit,
                lower_ci_90=round(corr - 1.5, 3),
                upper_ci_90=round(corr + 1.5, 3),
            )

        elif method == CorrectionMethod.ML_RESIDUAL and variable in self.ml_models:
            model = self.ml_models[variable]
            corr, lower, upper = model.predict(raw_val, features)
            return CorrectionResult(
                variable=variable.value,
                lead_days=lead_days,
                raw_value=raw_val,
                corrected_value=round(corr, 3),
                delta=round(corr - raw_val, 3),
                method=CorrectionMethod.ML_RESIDUAL,
                unit=unit,
                lower_ci_90=round(lower, 3),
                upper_ci_90=round(upper, 3),
            )

        else:
            # Fallback to raw if model not trained
            return self.calibrate(variable, raw_val, features, method=CorrectionMethod.RAW, lead_days=lead_days)

    def save_state(self, path: Optional[Union[str, Path]] = None) -> None:
        """Persist benchmark metrics and model metadata to JSON."""
        save_path = Path(path) if path else (self.artifacts_dir / "forecast_correction_report.json")
        data = {
            "version": "1.0.0",
            "benchmarks": [m.model_dump() for m in self.benchmarks],
            "trained_variables": [v.value for v in self.ml_models.keys()],
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved forecast correction report to %s", save_path)
