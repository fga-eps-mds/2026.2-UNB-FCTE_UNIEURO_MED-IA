"""Generic helpers: seeding, config loading, metrics, checkpointing."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int) -> None:
    """Seed python/numpy/torch (CPU+CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(y_true_hard: np.ndarray, y_prob_mci: np.ndarray) -> dict[str, float]:
    """Accuracy/F1 at the paper's p>=0.5 cutoff, plus AUC on the raw probability."""
    y_pred = (y_prob_mci >= 0.5).astype(int)
    acc = accuracy_score(y_true_hard, y_pred)
    f1 = f1_score(y_true_hard, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true_hard, y_prob_mci)
    except ValueError:
        auc = float("nan")  # only one class present, e.g. tiny smoke-test subsets
    return {"accuracy": float(acc), "f1": float(f1), "auc": float(auc)}


def mean_std_table(per_seed_metrics: list[dict[str, float]]) -> dict[str, str]:
    """Format mean +/- std across repetitions, matching the paper's Table 1 style."""
    keys = per_seed_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed_metrics])
        out[k] = f"{vals.mean():.4f} ± {vals.std():.4f}"
    return out
