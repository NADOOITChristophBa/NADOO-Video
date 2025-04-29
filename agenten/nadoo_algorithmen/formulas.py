"""
formulas.py

Mathematische Formeln für Nadoo Algorithmen.
"""
import numpy as np

def activity_score(x: np.ndarray) -> float:
    """Berechnet den Aktivitäts-Score A(x) = mean(abs(x))."""
    return float(np.mean(np.abs(x)))

def complexity_worst(B: int, d: int) -> float:
    """Worst-Case Kosten C_worst = B * (d + d^2)."""
    return float(B * (d + d * d))

def complexity_expected(B: int, d: int, s: float) -> float:
    """Erwartete Kosten C_expected = (1 - s) * B * d^2 + B * d."""
    return float((1.0 - s) * B * d * d + B * d)

def prune_threshold(W: np.ndarray, tau: float) -> float:
    """Berechnet die Pruning-Schwelle tau * max(abs(W))."""
    return float(tau * np.max(np.abs(W)))

def apply_prune_mask(W: np.ndarray, tau: float) -> np.ndarray:
    """Wendet die Pruning-Maske an: setzt Gewichte unterhalb der Schwelle auf 0."""
    thresh = prune_threshold(W, tau)
    mask = np.abs(W) >= thresh
    return W * mask

def chunk_topk(weights: np.ndarray, chunk_size: int, k: int) -> np.ndarray:
    """Wählt Indizes der k Chunks mit der höchsten L1-Norm aus."""
    assert weights.ndim == 1, "weights must be a 1D array"
    n_chunks = len(weights) // chunk_size
    chunks = weights[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
    scores = np.sum(np.abs(chunks), axis=1)
    topk_idx = np.argsort(scores)[-k:][::-1]
    return topk_idx
