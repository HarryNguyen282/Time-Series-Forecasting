import os
import pandas as pd
import numpy as np
from typing import Sequence, Mapping, Tuple, Any, Dict, OrderedDict

from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA

from datetime import datetime

cfg = DATA
RANDOM_STATE = cfg.RANDOM_STATE
learning_rate = cfg.learning_rate
n_jobs = cfg.CV.n_jobs
scoring = cfg.CV.scoring
verbose = cfg.CV.verbose


def space_linear() -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    from sklearn.linear_model import LinearRegression
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True)),
         ("model", LinearRegression())
    ])

    param_grid: Dict[str, Sequence] = {
        'model__fit_intercept': [True, False],
        'model__positive': [False, True]
    }
    return pipe, param_grid

def space_lasso() -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    from sklearn.linear_model import Lasso
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', Lasso(max_iter=50000, tol=1e-4, selection='cyclic'))
    ])

    param_grid = {
        'model__alpha': [0.1, 0.5, 1]
    }
    return pipe, param_grid

def space_ridge() -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    from sklearn.linear_model import Ridge
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', Ridge(max_iter=50000, tol=1e-4))
    ])

    param_grid = {
        'model__alpha': [0.1, 0.5, 1]
    }
    return pipe, param_grid

def space_RF() -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    from sklearn.ensemble import RandomForestRegressor
    pipe = Pipeline(steps=[
        ('model', RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=cfg.CV.n_jobs,
                    bootstrap=True,
                    criterion="squared_error"))
    ])

    param_grid = {
        'model__n_estimators': [100, 200],
    }
    return pipe, param_grid

def space_HGB() -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    pipe = Pipeline(steps=[
        ('model', HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            early_stopping=True
        ))
    ])

    param_grid = {
        'model__max_iter': [100, 200],
        'model__max_depth': [6, 12]
    }
    return pipe, param_grid

def space_svr() -> Tuple[BaseEstimator, Dict[str, Sequence[Any]]]:
    from sklearn.svm import SVR
    pipe = Pipeline(steps=[
         ('scaler', StandardScaler()),
         ('model', SVR())
    ])

    param_grid = {
        'model__C': [0.1, 1],
    }
    return pipe, param_grid

def build_model_space() -> Dict[str, Tuple[BaseEstimator, Dict[str, Sequence[Any]]]]:
    
    return {
        'lr': space_linear(),
        'lasso': space_lasso(),
        'ridge': space_ridge(),
        'rf': space_RF(),
        'hgb': space_HGB(),
        'svr': space_svr()
    }

def run_search(
        estimator: BaseEstimator,
        param_grid: Mapping[str, Sequence[Any]],
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: TimeSeriesSplit | None = None,
        scoring: str = scoring,
        n_jobs: int = n_jobs,
        verbose: int = verbose):
    tscv = cv
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        verbose=verbose,
        n_jobs=n_jobs,
        refit=True
    )
    gs.fit(X, y)
    return gs.best_estimator_, dict(gs.best_params_)

def tune_all_models(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        cv: TimeSeriesSplit | None = None,
        scoring: str = scoring,
        n_jobs: int = n_jobs,
        verbose: int = verbose) -> Dict[str, Any]:
    
    model_space = build_model_space()
    tscv = cv
    best_params = dict()
    
    for model in model_space.keys():
        print(f'Tuning for {model}')
        start = datetime.now()
        est, grid = model_space[model]
        _, params = run_search(estimator=est,
                               param_grid=grid,
                               X=X,
                               y=y,
                               cv=tscv,
                               scoring=scoring,
                               n_jobs=n_jobs,
                               verbose=verbose)
        best_params[model] = params   
        end = datetime.now()
        elapsed_time = end - start
        print(f'--Running time: {elapsed_time.total_seconds():.4f} s') 
    return best_params

