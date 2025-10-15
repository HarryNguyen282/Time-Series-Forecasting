
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from .config import DATA as C

class SeasonalNaive(BaseEstimator, RegressorMixin):
    def __init__(self, seasonal_lag: int = 24):
        self.seasonal_lag = seasonal_lag
        self.history_ = None
    
    def fit(self, X, y):
        self.history_ = np.asarray(y)

    def predict(self, X):
        n = len(X)
        s = self.seasonal_lag
        if self.history_ is None or len(self.history_) < s:
            last = self.history_[-1] if (self.history_ is not None and len(self.history_)>0) else 0.0
            return np.repeat(last, n)        
        preds = []
        for i in range(n):
            preds.append(self.history_[-s+i] if i < s else preds[i-s])
        
        return np.asarray(preds)
    
class seqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    
    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        xseq = self.X[idx: idx + self.seq_len]
        yval = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(xseq), torch.tensor(yval)
    
class _LSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, X):
        out, _ = self.lstm(X)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)

class LSTMRegressor:
    def __init__(self, n_features: int, hidden_size: int = 64, seq_size: int = 24,
                num_layers: int = 1, dropout:float = 0.0, 
                lr: float = 1e-3, epochs: int = 20, batch_size: int = 128):
        self.seq_size = seq_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = _LSTM(n_features=n_features, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(self.device)
        self.model.to(self.device)
        self.scaler_X = None
        self.scaler_y = None
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def fit(self, X, y):
        self.scaler_X = StandardScaler().fit(X)
        self.scaler_y = StandardScaler().fit(y.reshape(-1,1))
        
        Xs = self.scaler_X.transform(X).astype(np.float32)
        ys = self.scaler_y.transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        ds = seqDataset(Xs, ys, self.seq_size)
        dl = DataLoader(ds, self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()

        for _ in range(self.epochs):
            
            for xb, yb in dl:
                xb, yb = xb.to(self.device), yb.to(self.device)

                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
        
        return self
    
    def predict(self, X):
        Xs = self.scaler_X.transform(X)
        ds = seqDataset(Xs, np.zeros(len(Xs)), self.seq_size)
        dl = DataLoader(ds, self.batch_size, shuffle=False)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl:
                xb = xb.to(self.device)
                out = self.model(xb).cpu().numpy()
                preds.append(out)
        
        preds = np.concatenate(preds)
        pad = self.seq_size - 1
        preds = np.concatenate([np.full(pad, np.nan), preds])
        preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()
        
        return preds

    def save(self, name:str = "lstm"):
        out_dir = Path(C.PATHS.models_dir)
        path = out_dir/f"{name}.pt"
        payload = {
            "state_dict": self.model.state_dict(),
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "meta": {
                "n_features": self.n_features,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "epochs": self.epochs,
                "seq_size": self.seq_size,
                "lr": self.lr,
                "batch_size": self.batch_size
            }
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device: str | None = None) -> "LSTMRegressor":
        path = Path(path)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

        m = ckpt["meta"]
        obj = cls(
            n_features=m["n_features"],
            hidden_size=m["hidden_size"],
            seq_size=m["seq_size"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
            lr=m["lr"],
            epochs=m["epochs"],
            batch_size=m["batch_size"],
        )
        obj.model.load_state_dict(ckpt["state_dict"])
        obj.model.to(device).eval()
        obj.device = device
        obj.scaler_X = ckpt["scaler_X"]
        obj.scaler_y = ckpt["scaler_y"]
        return obj
