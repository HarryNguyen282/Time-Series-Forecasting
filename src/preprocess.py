
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

def enforce_regular_frequency(df: pd.DataFrame, freq: str, method: str = "asfreq") -> pd.DataFrame:
    """Reindex to a fixed grid. Use 'asfreq' or 'resample' (mean)."""
    if method == "asfreq":
        return df.asfreq(freq)
    elif method == "resample":
        return df.resample(freq).mean()
    else:
        raise ValueError("method must be 'asfreq' or 'resample'")

def impute_missing(df: pd.DataFrame, strategy: str = "ffill", limit: int | None = None) -> pd.DataFrame:
    """Fill missing values with ffill/bfill/linear."""
    if strategy == "ffill":
        return df.ffill(limit=limit).bfill(limit=limit)
    if strategy == "bfill":
        return df.bfill(limit=limit).ffill(limit=limit)
    if strategy == "linear":
        return df.interpolate(method="time").ffill().bfill()
    raise ValueError("Unsupported strategy")


def scale_features(df: pd.DataFrame):
    """Scale features; returns (scaled_df, scalers_dict, inverse_fn)."""
    from sklearn.preprocessing import StandardScaler
    Scaler =  StandardScaler
    scalers: Dict[str, object] = {}
    df_scaled = df.copy()
    for c in df.columns:
        s = Scaler()
        vals = df[[c]].values
        mask = np.isfinite(vals).ravel()
        s.fit(vals[mask].reshape(-1, 1))
        df_scaled[c] = s.transform(vals).ravel()
        scalers[c] = s
    def inverse_fn(df_scaled_in: pd.DataFrame) -> pd.DataFrame:
        out = df_scaled_in.copy()
        for c, s in scalers.items():
            out[c] = s.inverse_transform(out[[c]].values).ravel()
        return out
    return df_scaled, scalers, inverse_fn

def preprocess(df: pd.DataFrame, freq: str, impute_strategy: str = "ffill") -> pd.DataFrame:

    df_clean = df.copy()
    df_clean = enforce_regular_frequency(df_clean, freq=freq, method="asfreq")
    df_clean = impute_missing(df_clean, strategy=impute_strategy)
    return df_clean
    
    