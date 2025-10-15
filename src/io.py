from __future__ import annotations
import pandas as pd
from pandas.tseries.frequencies import to_offset
from pathlib import Path
from typing import Tuple


from .config import DATA


def _build_timestamp(n, start_date, freq='h') -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, periods=n, freq=freq)


def load_data(file_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(f'{DATA.PATHS.raw_data}/{file_name}')
    if not path.exists():
        raise FileNotFoundError(f"Excel not found at {path}")


    xls = pd.read_excel(path, sheet_name=["Training", "Testing"])
    if not {"Training", "Testing"}.issubset(set(xls.keys())):
        raise ValueError("Expected sheets 'Training' and 'Testing' in the Excel file.")

    freq = 'h'

    train = xls["Training"].copy()
    test = xls["Testing"].copy()

    train_start = pd.Timestamp("2023-01-01 00:00:00")
    train["Timestamp"] = _build_timestamp(len(train), train_start, freq)

    test_start = train["Timestamp"].iloc[-1] + to_offset(freq)
    test["Timestamp"] = _build_timestamp(len(test), test_start, freq)


    train = train.set_index("Timestamp").sort_index()
    test = test.set_index("Timestamp").sort_index()

    drop_cols = ['Year', 'Month', 'Day', 'Hour']
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)


    return train, test