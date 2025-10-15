# Time-Series Forecasting (Hourly)

End-to-end, reproducible pipeline for forecasting an hourly target using classical ML and a PyTorch LSTM.
Best model on test: Random Forest — RMSE ≈ 67.7, R² ≈ 0.948.

## Results (test set)

|   model |        rmse |      mae |     mape |    smape |           r2 |
| ------: | ----------: | -------: | -------: | -------: | -----------: |
|    rf_1 | **67.7129** |  48.5695 | 0.022089 | 0.022107 | **0.948109** |
|    lstm |     71.0325 |  56.6335 | 0.025365 | 0.025736 |     0.925414 |
|   hgb_1 |     74.1843 |  56.5238 | 0.025784 | 0.025746 |     0.937716 |
| lasso_1 |    107.4883 |  84.9037 | 0.038592 | 0.038559 |     0.869241 |
| ridge_1 |    113.1374 |  90.3750 | 0.041052 | 0.041020 |     0.855135 |
|    lr_1 |    113.1948 |  90.4317 | 0.041078 | 0.041046 |     0.854988 |
|   svr_1 |    211.0431 | 179.5371 | 0.078233 | 0.081216 |     0.495928 |

Summary: Tree ensembles (RF/HGB) and the LSTM capture non-linear relationships and temporal effects better than purely linear models. SVR under-performs with the current kernel/scale.

## What’s in the box

Goal: Forecast a continuous hourly target (e.g., “Load”) from weather + calendar signals.

Stack: pandas, numpy, scikit-learn, PyTorch, joblib.

Design: Single pipeline for preprocessing → feature engineering (FE) → hyperparameter tuning (GridSearchCV + TimeSeriesSplit) → training → evaluation.

Reproducibility: Tuned params, fitted scalers, and trained models are saved and reused.

Data note: dataset.xlsx is gitignored. Use your own file; adjust src/config.py as needed.

## Project Structure

.
├─ src/
│ ├─ config.py # DATA.CV, PATHS, COLS, LSTM hyperparams
│ ├─ data.py # load_data('dataset.xlsx')
│ ├─ preprocess.py # preprocess(), scale_features()
│ ├─ features.py # build_feature_matrix()
│ ├─ tuning.py # model spaces + GridSearchCV + TimeSeriesSplit
│ ├─ models.py # LSTMRegressor (fit/predict/save/load), \_LSTM, seqDataset
│ ├─ metrics.py # rmse, mae, mape, smape, r2 (+ scorer helpers)
│ ├─ train.py # training: tune/load params, fit, save
│ └─ eval.py # evaluation: build test FE with context, score
├─ artifacts/
│ ├─ tuning/best_params.json # tuned hyperparams (or latest.json)
│ └─ scalers/scalers.joblib # StandardScalers fitted on train
├─ models/
│ ├─ rf_1.joblib, hgb_1.joblib, ... # trained sklearn pipelines
│ └─ lstm.pt # LSTM weights + scalers + meta
└─ data/
└─ raw/dataset.xlsx # (gitignored)

## Target & Features (high level)

Target: a continuous hourly series (e.g., energy demand “Load”).

Raw drivers: temperature, irradiance (GHI), timestamps, calendar info, etc.

Engineered features (examples):

Lags: recent history for target and drivers (e.g., t-1, t-24).

Deltas: first differences on key drivers (e.g., Δtemp, ΔGHI).

Degree-hours: heating/cooling degree hours.

Aggregations: combined temp/GHI signals.

Calendar: hour of day, day of week, weekend/holiday flags.

Seasonality: Fourier terms to represent daily/weekly cycles.

## Preprocessing

Timestamp normalization; ensure a continuous hourly index.

Column-wise StandardScaler fitted only on train and saved to artifacts/scalers/scalers.joblib.

Test FE context: To compute lags on the test set, we prepend the last K rows of train (K ≥ max lag, e.g., 168 for a week of hourly data) before running FE, then we keep only the true test rows. This prevents the “0 rows” issue from all-NaN lagged features.

## Models

#### Classic ML (scikit-learn)

Linear Regression, Lasso, Ridge — pipelines with scaling; tuned with small grids.

Random Forest (RF) — tuned n*estimators, max_depth, min_samples*\*, max_features.

HistGradientBoosting (HGB) — tuned learning_rate, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization, max_iter.

SVR — RBF kernel baseline; tuned C, epsilon, gamma.

#### Deep Learning (PyTorch)

LSTMRegressor

Windowed inputs of shape (seq_len, n_features); predicts y_t.

Uses StandardScaler on X and y (fit on train), MSELoss, Adam, GPU if available.

Saved to models/lstm.pt (weights + scalers + hyperparams) via save() / load().

## Training & Tuning

CV: TimeSeriesSplit forward-chaining (gap/test_size from config.py).

Search: GridSearchCV with scoring="neg_root_mean_squared_error" (kept simple).

Persistence: If artifacts/tuning/best_params.json exists, it’s loaded; otherwise tuning runs and saves it. Trained models are stored in models/.

```# Train (tunes if no best_params.json yet; fits models; saves artifacts)
python -m src.train --config src/config.py

# Evaluate on test (rebuilds FE with train-tail context and uses train scalers)
python -m src.eval  --config src/config.py
```

## Metrics (plain-English)

RMSE: Root-mean-squared error; penalizes larger errors more heavily. Lower is better.

MAE: Mean absolute error; average absolute deviation. Lower is better.

MAPE/sMAPE: Scale-free percentage-type errors. Lower is better.

R²: Proportion of variance explained by the model. Higher is better.

## Analysis of Results

Random Forest is the winner with the lowest RMSE and highest R². It handles non-linearities and interactions across weather/time features with minimal fuss.

HistGradientBoosting is close; additional tuning (depth/regularization/bins) might surpass RF on some datasets.

LSTM is competitive and benefits from longer windows, more epochs, and careful regularization/early-stopping. Great when sequential dependencies dominate.

Linear models (LR/Lasso/Ridge) under-fit here; they’re fast and interpretable but miss non-linear structure.

SVR lags with current settings; may improve with tighter kernel/scale choices or a linear kernel on these features.

Operationally: RF offers strong accuracy and fast inference with low maintenance. HGB/LSTM can be explored for incremental gains depending on constraints and data scale.

## Key Takeaways

Provide history for test FE: prepend train tail before feature engineering to avoid empty test matrices.

Train-only scaling: fit scalers on train; reuse them to transform test (never refit on test).

Persist everything: tuned params, scalers, models → deterministic, repeatable runs.

Start with RF: strong baseline for hourly forecasting; add HGB or LSTM for extra performance if needed.

Keep config in one place: paths, CV settings, and LSTM hyperparams live in src/config.py.

## Configuration

Edit src/config.py:

DATA.PATHS: artifacts_dir, models_dir, reports_dir, dataset location.

DATA.COLS: target column name (and any key inputs you reference).

DATA.CV: n_splits, gap, test_size_per_split, n_jobs.

DATA.LSTM: seq_size, hidden_size, num_layers, dropout, lr, epochs, batch_size.

## Future Work

Add LightGBM/XGBoost (optionally GPU) for stronger tree baselines.

Automated holiday/season encoders; more weather-derived non-linear transforms.

Prediction intervals (quantile boosting or conformal).

MLflow logging and small unit tests for FE invariants.
