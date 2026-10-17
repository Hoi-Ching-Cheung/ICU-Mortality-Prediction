"""Regression checks against the saved artifacts behind the README numbers.

These need the (git-ignored) data/ folder and are skipped without it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from icu_mortality.evaluation import best_f1_threshold, score

DATA = Path(__file__).resolve().parents[1] / "data"
SPLITS = DATA / "preprocessed" / "12-01"
TRACE = DATA / "bayesian_model_checkpoints" / "pooled_trace.nc"

needs_data = pytest.mark.skipif(
    not (SPLITS.exists() and TRACE.exists()), reason="data/ not downloaded"
)


def _load(split):
    X = pd.read_csv(SPLITS / f"X_{split}.csv")
    y = pd.read_csv(SPLITS / f"y_{split}.csv").iloc[:, 0].to_numpy()
    return X, y


@needs_data
def test_splits_are_leak_free_and_test_set_untouched_by_smote():
    X_train, y_train = _load("train")
    X_test, y_test = _load("test")

    assert list(X_train.columns) == list(X_test.columns)
    assert y_train.mean() == pytest.approx(0.5)            # SMOTE-balanced
    assert y_test.mean() == pytest.approx(0.086, abs=0.002)  # real prevalence
    assert len(X_test) == 18_343


@needs_data
def test_pooled_bayes_reproduces_reported_numbers():
    az = pytest.importorskip("arviz")
    from icu_mortality.bayes import posterior_mean_proba

    trace = az.from_netcdf(TRACE)

    def proba(split):
        X, y = _load(split)
        X = X.drop(columns=["hospital_id", "icu_id"])
        X.insert(0, "intercept", 1.0)
        return posterior_mean_proba(trace, X.to_numpy(np.float32)), y

    p_val, y_val = proba("val")
    t = best_f1_threshold(y_val, p_val)
    assert t == pytest.approx(0.79, abs=0.005)
    assert score(y_val, p_val, t)["f1"] == pytest.approx(0.4574, abs=1e-3)

    # Threshold chosen on validation, applied once to test.
    p_test, y_test = proba("test")
    test = score(y_test, p_test, t)
    assert test["f1"] == pytest.approx(0.452, abs=2e-3)
    assert test["auc"] == pytest.approx(0.862, abs=2e-3)
