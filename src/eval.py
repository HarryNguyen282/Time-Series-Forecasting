import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from joblib import load

from .config import DATA as C
from .io import load_data
from .preprocess import preprocess, scale_features
from .features import build_feature_matrix
from .utils import  to_csv
from .metrics import rmse, mae, mape, smape, r2
from .models import LSTMRegressor

def _prepare_test_features() -> Tuple[pd.DataFrame, pd.Series]:
    train_raw, test_raw = load_data('dataset.xlsx')
    
    target = C.COLS.target
 
    scaler_path = Path(f'{C.PATHS.artifacts}/scalers/scalers.joblib')
    scalers = load(scaler_path)
    train_pp = preprocess(train_raw, freq='h')
    test_pp  = preprocess(test_raw,  freq='h')

    # scale using train-fitted scalers
    Xtr_pp = train_pp.copy()
    Xte_pp = test_pp.copy()
    for col, s in scalers.items():
        if col in Xtr_pp.columns:
            vals = Xtr_pp[col].to_numpy()
            m = np.isfinite(vals)
            if m.any():
                vals[m] = s.transform(vals[m].reshape(-1,1)).ravel()
            Xtr_pp[col] = vals
        if col in Xte_pp.columns:
            vals = Xte_pp[col].to_numpy()
            m = np.isfinite(vals)
            if m.any():
                vals[m] = s.transform(vals[m].reshape(-1,1)).ravel()
            Xte_pp[col] = vals

    # put back the (unscaled) target for FE if your FE expects it present
    Xtr_pp[target] = train_pp[target]
    Xte_pp[target]  = test_pp[target]

    # 3) provide context from train before FE
    # choose K >= max lag used in your FE (e.g., 168 for one week of hourly lags)
    K = getattr(C, "FE", None) and getattr(C.FE, "context_rows", 168) or 168
    K = min(K, len(Xtr_pp))  # can’t take more than train length
    combo = pd.concat([Xtr_pp.tail(K), Xte_pp], axis=0)

    # 4) FE on the combined frame
    ds = build_feature_matrix(combo, target)

    # 5) keep only the test part after FE (last len(test_pp) rows)
    X_test_fe = ds.drop(columns=[target]).tail(len(Xte_pp))
    y_test = ds[target].tail(len(Xte_pp))

    return X_test_fe, y_test

def main():
    
    model_dir = Path(C.PATHS.models_dir)
    report_dir = Path(C.PATHS.reports)
    report_dir.mkdir(parents=True, exist_ok=True)

    X_test_fe, y_test = _prepare_test_features()
    joblib_paths = list(model_dir.glob("*.joblib"))
    if not joblib_paths:
        raise FileNotFoundError(f"No models found in {model_dir}")
    
    rows = []
    for mp in joblib_paths:
        name = mp.stem
        print(f"Evaluating for {name}")
        model = load(mp)
        y_hat = model.predict(X_test_fe)

        rows.append({
            "model": name,
            "rmse": rmse(y_test, y_hat),
            "mae": mae(y_test, y_hat),
            "mape": mape(y_test, y_hat),
            "smape": smape(y_test, y_hat),
            "r2": r2(y_test, y_hat)
        })

    #lstm
    lstm = LSTMRegressor.load(Path(C.PATHS.models_dir)/"lstm.pt")
    y_pred = lstm.predict(X_test_fe)
    mask = ~np.isnan(y_pred)
    y_true_eval = y_test.values.astype(np.float32)[mask]
    y_pred_eval = y_pred.astype(np.float32)[mask]
    rows.append({
        "model": "lstm",
        "rmse":  rmse(y_true_eval, y_pred_eval),
        "mae":   mae(y_true_eval, y_pred_eval),
        "mape":  mape(y_true_eval, y_pred_eval),
        "smape": smape(y_true_eval, y_pred_eval),
        "r2":    r2(y_true_eval, y_pred_eval),
    })

    df = pd.DataFrame(rows).sort_values("rmse")
    to_csv(df, Path(f'{C.PATHS.reports}/evaluation.csv'))
    print(df)

if __name__ == "__main__":
    main()