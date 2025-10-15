from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Paths:
    raw_data: str = 'data/raw/'
    artifacts: str = 'artifacts'
    models_dir: str = 'artifacts/models'
    reports: str = "reports/"
    

@dataclass
class Columns:
    target: str = 'Load'
    timestamp: str = 'Timestamp'

@dataclass
class CV:
    n_splits: int = 5
    gap: int = 0
    test_size_per_split : int = 7 * 24
    n_jobs = -1
    max_train_size: int | None = None
    scoring: str = "neg_root_mean_squared_error"
    verbose: int = 0
@dataclass
class LSTMParams:
    seq_size: int = 24
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 20

@dataclass
class DATA:
    PATHS: Paths = Paths()
    COLS: Columns = Columns()
    CV = CV()
    LSTM: LSTMParams = LSTMParams()
    RANDOM_STATE: int = 42
    learning_rate: float = 1e-3

DATA = DATA()