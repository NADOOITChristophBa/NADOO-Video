import pytest
import numpy as np
import os, sys
# Ensure formulas module is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from formulas import (
    activity_score,
    complexity_worst,
    complexity_expected,
    prune_threshold,
    apply_prune_mask,
    chunk_topk,
)

def test_activity_score():
    x = np.array([-1, 2, -3, 4])
    assert activity_score(x) == pytest.approx((1 + 2 + 3 + 4) / 4)


def test_complexity_worst():
    assert complexity_worst(3, 2) == 3 * (2 + 2 * 2)


def test_complexity_expected():
    expected = (1 - 0.5) * 3 * 2 * 2 + 3 * 2
    assert complexity_expected(3, 2, 0.5) == pytest.approx(expected)


def test_prune_threshold_and_mask():
    W = np.array([-1, 2, -3, 4])
    tau = 0.5
    thresh = prune_threshold(W, tau)
    assert thresh == pytest.approx(0.5 * 4)
    masked = apply_prune_mask(W, tau)
    expected = np.array([0, 2, -3, 4])
    assert np.array_equal(masked, expected)


def test_chunk_topk():
    weights = np.arange(1, 9)
    idx1 = chunk_topk(weights, 4, 1)
    assert np.array_equal(idx1, np.array([1]))
    idx2 = chunk_topk(weights, 4, 2)
    assert set(idx2.tolist()) == {0, 1}
