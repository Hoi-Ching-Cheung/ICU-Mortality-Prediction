import numpy as np
import pytest

from icu_mortality.evaluation import best_f1_threshold, score, threshold_sweep


def test_perfectly_separable_scores_hit_f1_of_one():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    sweep = threshold_sweep(y, p)
    assert sweep["f1"].max() == pytest.approx(1.0)
    assert 0.3 < best_f1_threshold(y, p) <= 0.8


def test_all_negative_predictions_do_not_crash():
    # At a high cutoff nothing is flagged; precision is undefined, report 0.
    y = np.array([0, 1, 0, 1])
    p = np.full(4, 0.2)
    row = threshold_sweep(y, p, thresholds=[0.5]).iloc[0]
    assert row["precision"] == 0 and row["recall"] == 0 and row["f1"] == 0


def test_majority_class_baseline_looks_good_on_accuracy_only():
    # Why we don't use accuracy: predicting "survive" for everyone at the
    # dataset's 8.6% mortality rate is ~91% accurate and useless.
    rng = np.random.default_rng(0)
    y = (rng.random(10_000) < 0.086).astype(int)
    p = np.zeros_like(y, dtype=float)
    row = threshold_sweep(y, p, thresholds=[0.5]).iloc[0]
    assert row["accuracy"] > 0.9
    assert row["f1"] == 0


def test_lower_threshold_trades_precision_for_recall():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2_000)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, y.size), 0, 1)
    lo, hi = score(y, p, 0.3), score(y, p, 0.7)
    assert lo["recall"] > hi["recall"]
    assert lo["precision"] < hi["precision"]
