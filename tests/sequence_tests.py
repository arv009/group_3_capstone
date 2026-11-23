"""Unit tests for sequence.py / sequence_model.py (no Kaggle data needed).

This file must not import LightGBM (secondlook.modeling): PyTorch and LightGBM clash on macOS."""
import numpy as np
import pytest
import torch

from secondlook.sequence import standardize
from secondlook.sequence_model import GRUScorer, out_of_fold_scores, predict, train_gru


def _toy(n=600, max_len=10, n_feat=3, seed=0):
    """Defaulters have late payments near the END of their sequence; repayers do not."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(3, max_len + 1, n)
    y = rng.integers(0, 2, n)
    X = np.zeros((n, max_len, n_feat), np.float32)
    for i in range(n):
        X[i, : lengths[i]] = rng.normal(0, 0.3, (lengths[i], n_feat))
        if y[i]:
            X[i, lengths[i] - 2: lengths[i], 0] += 2.0
    return X, lengths.astype(np.int64), y


def test_standardize_ignores_padding():
    X = np.zeros((2, 4, 1), np.float32)
    X[0, :2, 0] = [1, 3]
    X[1, :1, 0] = [2]
    Xs, mean, std = standardize(X, np.array([2, 1]))
    assert mean[0] == pytest.approx(2.0)
    assert (Xs[0, 2:] == 0).all() and (Xs[1, 1:] == 0).all()


def test_padding_does_not_change_the_score():
    torch.manual_seed(0)
    model = GRUScorer(3)
    X, lengths, _ = _toy(n=5)
    padded = np.concatenate([X, np.full((5, 6, 3), 9.0, np.float32)], axis=1)  # junk after the real steps
    assert predict(model, X, lengths) == pytest.approx(predict(model, padded, lengths), abs=1e-5)


def test_gru_learns_recent_lateness():
    X, L, y = _toy(n=800)
    Xv, Lv, yv = _toy(n=300, seed=1)
    _, hist = train_gru(X, L, y, Xv, Lv, yv, epochs=6, batch=64, lr=5e-3, verbose=False)
    assert max(hist) > 0.9


def test_out_of_fold_scores_shapes():
    X, L, y = _toy(n=300)
    Xv, Lv, yv = _toy(n=100, seed=1)
    Xt, Lt, _ = _toy(n=50, seed=2)
    oof, outside, hist = out_of_fold_scores(X, L, y, Xv, Lv, yv, {"test": (Xt, Lt)}, n_folds=3,
                                            epochs=1, batch=64, verbose=False)
    assert oof.shape == (300,) and outside["test"].shape == (50,) and outside["valid"].shape == (100,)
    assert len(hist) == 3 and not np.isnan(oof).any()
