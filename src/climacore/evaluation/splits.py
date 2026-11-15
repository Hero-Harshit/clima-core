"""
Chronological split utilities for Clima-Core.
Strictly prohibits random row shuffling to preserve temporal causality and avoid data leakage.
"""

from typing import Tuple
import pandas as pd


def chronological_train_val_test_split(
    df: pd.DataFrame,
    date_col: str = "date",
    val_start: str = "2025-06-01",
    test_start: str = "2026-01-01",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically into train, validation, and test subsets.
    - Train: date < val_start
    - Validation: val_start <= date < test_start
    - Test: date >= test_start
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    val_dt = pd.to_datetime(val_start)
    test_dt = pd.to_datetime(test_start)

    train_df = df[df[date_col] < val_dt].copy()
    val_df = df[(df[date_col] >= val_dt) & (df[date_col] < test_dt)].copy()
    test_df = df[df[date_col] >= test_dt].copy()

    return train_df, val_df, test_df
