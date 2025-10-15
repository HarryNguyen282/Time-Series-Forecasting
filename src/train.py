import argparse
from pathlib import Path
from joblib import dump
import importlib.util
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import cross_val_score, TimeSeriesSplit

from .models import LSTMRegressor
from .io import load_data
from .preprocess import preprocess, scale_features
from .features import build_feature_matrix
from .utils import ensure_path, save_joblib, save_best_params, load_best_params, to_csv
from .tuning import tune_all_models, build_model_space

from datetime import datetime
    

def main(argv=None):

    # Load Config
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', help="Path to config.py (positional)")
    parser.add_argument('--config', dest='config_opt', help='Path to config.py (flag)')
    parser.set_defaults(config_opt=None)
    args = parser.parse_args(argv)

    config_path = args.config or args.config_opt or 'src/config.py'
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path}')

    spec = importlib.util.spec_from_file_location('cfg', str(path))
    cfg = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(cfg)
    C = cfg.DATA
    ensure_path(C.PATHS.artifacts, C.PATHS.raw_data, C.PATHS.models_dir)

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Load training and testing data, preprocess training data
    train_raw, test_raw = load_data('dataset.xlsx')
    target = C.COLS.target
    train_pp = preprocess(train_raw, 'h')
    X_pp, scalers, inverse_fn = scale_features(train_pp)
    X_pp[target] = train_pp[target]
    ds_fe = build_feature_matrix(X_pp, target)
    X_fe = ds_fe.drop([target], axis=1)
    y = ds_fe[target]

    # save scaler to use later for transforming test data
    scaler_dir = Path(f'{C.PATHS.artifacts}/scalers')
    scaler_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = scaler_dir / 'scalers.joblib'
    dump(scalers, scaler_path)
    print(f'Saved scalers -> {scaler_path}\n')

    # hyperparameter tuning
    param_path = Path(f'{C.PATHS.artifacts}/tuning/best_params.json')
    if param_path.exists():
        print("Loading save parameters for ML model...\n")
        best_params = load_best_params(param_path)
    else:
        best_params = tune_all_models(X_fe, y)
        save_best_params(best_params=best_params, artifact_dir=C.PATHS.artifacts)

    # initalize LM models
    model_space = build_model_space()
    models = {}
    for model, (est, _grid) in model_space.items():
        params = best_params.get(model)
        if params:
            est.set_params(**params)
        models[model] = est

    # LSTM model
    lstm = LSTMRegressor(
        n_features=len(list(X_fe.columns)),
        hidden_size=C.LSTM.hidden_size,
        num_layers=C.LSTM.num_layers,
        seq_size=C.LSTM.seq_size,
        lr=C.LSTM.lr,
        epochs=C.LSTM.epochs,
        batch_size=C.LSTM.batch_size,
        dropout=C.LSTM.dropout
    )

    print("Training lstm\n")
    lstm.fit(X_fe.values, y.values)
    lstm.save()
    

    tscv = TimeSeriesSplit(
        n_splits=C.CV.n_splits | 5,
        test_size=C.CV.test_size_per_split | 7 * 24,
        gap=C.CV.gap | 0
    )

    cv_rows = []

    for name, est in models.items():
        print(f"Training model: {name}\n")
        start = datetime.now()
        cvres = cross_val_score(
            est, X_fe, y,
            scoring=C.CV.scoring,
            n_jobs=C.CV.n_jobs | -1,
            cv=tscv,
        )

        rmse_mean = -float(cvres.mean())
        rmse_std = float(cvres.std())
        cv_rows.append({
            "model": name,
            "rmse_cv_mean": rmse_mean,
            "rmse_cv_std": rmse_std
        })

        est.fit(X_fe, y)
        save_joblib(est, name, "1")
        end = datetime.now()
        print(f"Training {name} done, {(end-start).total_seconds():.4f} s\n")

    cv_df = pd.DataFrame(cv_rows).sort_values("rmse_cv_mean")
    to_csv(cv_df, Path(f'{C.PATHS.reports}/csv_summary.csv'))

if __name__ == "__main__":
    main()
    
