import pandas as pd
import numpy as np
from typing import List


def combine_features(df: pd.DataFrame, cols: list[str], method: str = "mean", new_name: str = "Combined") -> pd.DataFrame:
    """
    Combine multiple site columns (e.g., Site-1 Temp, Site-2 Temp...) into one aggregated feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    cols : list[str]
        List of columns to combine.
    method : str, default="mean"
        Aggregation method: "mean", "median", "sum", or "weighted".
    new_name : str, default="Combined"
        Base name for the resulting column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new combined feature.
    """

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    
    if method == "mean":
        combined = df[cols].mean(axis=1)
    elif method == "median":
        combined = df[cols].median(axis=1)
    elif method == "sum":
        combined = df[cols].sum(axis=1)
    elif method == "weighted":
        weights = np.arange(1, len(cols) + 1)
        combined = df[cols].multiply(weights, axis=1).sum(axis=1) / weights.sum()
    else:
        raise ValueError("Method must be one of 'mean', 'median', 'sum', or 'weighted'.")
    
    return pd.DataFrame({
        new_name: combined
    })

def create_target_lag(df: pd.DataFrame, target_col: str, lags: List[int]) -> pd.DataFrame:
    """
    Create lag features for a specified column.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    col : str
        Column to create lag features for.
    lags : List[int]
        List of lag periods.

    Returns
    -------
    pd.DataFrame
        DataFrame with new lag features added.
    """

    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")
    
    feature = {}
    
    for lag in lags:
        feature[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return pd.DataFrame(feature)


def create_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Create cyclical time-based features from a DatetimeIndex.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The time index of your data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
    """
    hour = index.hour
    hour_sin = np.sin( 2 * np.pi * hour / 24)
    hour_cos = np.cos( 2 * np.pi * hour / 24)

    dow = index.dayofweek
    dow_sin = np.sin( 2 * np.pi * dow  / 7)
    dow_cos = np.cos( 2 * np.pi * dow / 7)

    is_weekend = (dow >= 5).astype(int)

    return pd.DataFrame({
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "is_weekend": is_weekend
    }, index=index)

    

def make_fourier_features(index: pd.DatetimeIndex, period: int, order: int, prefix: str) -> pd.DataFrame:
    """
    Create Fourier features for capturing smooth seasonality.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Time index.
    period : int
        Seasonal period (e.g., 24 for daily, 168 for weekly).
    order : int
        Number of Fourier pairs to generate.
    prefix : str
        Prefix for column names (e.g., 'day', 'week').

    Returns
    -------
    pd.DataFrame
        DataFrame containing sin/cos Fourier terms.
    """
    t = np.arange(len(index))
    features = {}
    for k in range(1, order + 1):
        features[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    
    return pd.DataFrame(features, index=index)
    
def make_exog_lags(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """Create lag features for exogenous variables."""

    feature = {}

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in columns")
    
        for lag in lags:
            feature[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return pd.DataFrame(feature)
    

def make_exog_deltas(df: pd.DataFrame, cols: list[str], steps: list[int]) -> pd.DataFrame:
    """Create delta features for exogenous variables (current - lagged)."""

    feature = {}

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in columns")
    
        for step in steps:
            feature[f"{col}_delta_{step}"] = df[col] - df[col].shift(step)

    return pd.DataFrame(feature)
        
    

def build_feature_matrix(df: pd.DataFrame, target_col: str, base: float = 18) -> pd.DataFrame:
    """
    Build complete feature matrix for time series forecasting.

    Steps:
    1. Compute combined exogenous variables (Temp, GHI).
    2. Create lags & deltas for exog.
    3. Create CDH/HDH.
    4. Create target lags & rolling stats.
    5. Add calendar & Fourier features.
    6. Align and drop NaNs.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable aligned with X.
    """

    combinedTemp = combine_features(df, [c for c in df.columns if c.endswith("Temp")], method="mean", new_name="avg_temp")
    combinedGHI = combine_features(df, [c for c in df.columns if c.endswith("GHI")], method="mean", new_name="avg_ghi")
    targetLag = create_target_lag(df, target_col, [1, 24])
    tempLag = make_exog_lags(combinedTemp, ["avg_temp"], [1, 24])
    ghiLag = make_exog_lags(combinedGHI, ["avg_ghi"], [1, 24])
    tempDelta = make_exog_deltas(combinedTemp, ["avg_temp"], [1, 24])
    ghiDelta = make_exog_deltas(combinedGHI, ["avg_ghi"], [1, 24])
    hdh = np.maximum(base - combinedTemp, 0)
    cdh = np.maximum(combinedTemp - base, 0)
    cdh.columns = ["cdh"]
    hdh.columns = ["hdh"]
    timeFeatures = create_time_features(df.index)
    fourierDaily = make_fourier_features(df.index, period=24, order=3, prefix="day")
    fourierWeekly = make_fourier_features(df.index, period=168, order=3, prefix="week")

    feature_blocks = [
        combinedTemp,
        combinedGHI,
        targetLag,
        tempLag,
        ghiLag,
        tempDelta,
        ghiDelta,
        cdh,
        hdh,
        timeFeatures,
        fourierDaily,
        fourierWeekly
    ]

    X = pd.concat(feature_blocks, axis=1)
    y = df[target_col]

    data = pd.concat([X, y], axis=1).dropna()
    return data

    