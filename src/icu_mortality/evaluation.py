"""Threshold selection and metrics for an imbalanced binary target (~8.6% deaths)."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Accuracy / precision / recall / F1 at each probability cutoff."""
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.01)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def best_f1_threshold(y_true, y_prob, thresholds=None) -> float:
    """Cutoff that maximizes F1. Pick it on validation data, never on test."""
    sweep = threshold_sweep(y_true, y_prob, thresholds)
    return float(sweep.loc[sweep["f1"].idxmax(), "threshold"])


def score(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Headline metrics at a fixed cutoff."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": threshold,
        "auc": roc_auc_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
