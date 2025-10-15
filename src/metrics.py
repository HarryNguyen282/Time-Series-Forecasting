
from __future__ import annotations
from typing import Dict, Callable
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    make_scorer,
    get_scorer,
)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_pred - y_true) / denom)))

def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), eps, None)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))

def r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))

def make_scorers(primary: str = "neg_root_mean_squared_error") -> Dict[str, Callable]:
    """Return sklearn scorer objects; keep names compatible with sklearn."""
    scorers: Dict[str, Callable] = {
        "neg_root_mean_squared_error": make_scorer(rmse, greater_is_better=False),
        "neg_mean_absolute_error":     get_scorer("neg_mean_absolute_error"),
        "neg_mean_absolute_percentage_error": get_scorer("neg_mean_absolute_percentage_error"),
        "r2":                          get_scorer("r2"),
        "neg_smape":                   make_scorer(smape, greater_is_better=False),
    }
    if primary not in scorers:
        raise ValueError(f"Unknown primary '{primary}'. Choose from {list(scorers)}")
    return scorers
