"""
Baseline models for Clima-Core.
Implements persistence, moving average, and training-fitted seasonal climatology baselines.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


class PersistenceBaseline:
    """Predicts next-day value will equal the current day observation (lag 0 / lag 1)."""
    def __init__(self, observation_col: str = "pm2_5_mean"):
        self.observation_col = observation_col

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.observation_col].to_numpy()


class RollingMeanBaseline:
    """Predicts next-day value will equal the 7-day rolling average."""
    def __init__(self, rolling_col: str = "pm2_5_mean_roll_mean_7d"):
        self.rolling_col = rolling_col

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.rolling_col].to_numpy()


class SeasonalClimatologyBaseline:
    """
    Fits average target value by (city, month) strictly on the training set.
    Predicts using this lookup table (falls back to global train mean for unknown pairs).
    """
    def __init__(self, group_cols=("city", "month"), target_col="target_pm2_5_next_day"):
        self.group_cols = list(group_cols)
        self.target_col = target_col
        self.lookup: Dict[Tuple, float] = {}
        self.global_mean: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "SeasonalClimatologyBaseline":
        self.global_mean = float(train_df[self.target_col].mean())
        grouped = train_df.groupby(self.group_cols)[self.target_col].mean()
        self.lookup = grouped.to_dict()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in df.iterrows():
            key = tuple(row[col] for col in self.group_cols)
            preds.append(self.lookup.get(key, self.global_mean))
        return np.array(preds)
