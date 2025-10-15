import joblib
import os
import random
import numpy as np
import datetime
import json
from typing import Dict, Any
from pathlib import Path
from datetime import datetime
from .config import DATA as C

def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_path(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def save_joblib(obj: object, model_name: str, version: str):
    path = Path(f'{C.PATHS.models_dir}/{model_name}_{version}.joblib')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path

def save_metadata(path: str, metrics, version, description: str = "") -> None:
    meta = {
        "version": version,
        "description": description,
        "metrics": metrics,
        "timestamp": datetime.datetime.now().isoformat()
    }
    with open(path.replace("model.joblib", "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)    


def to_csv(df, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def save_best_params(best_params: Dict[str, Dict[str, Any]],
                     artifact_dir: str | Path,
                     ):
    out_dir = Path(artifact_dir)/"tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir/f'best_params.json'
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, sort_keys=True)

def load_best_params(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)