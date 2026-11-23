"""GRU sequence scorer (PyTorch). See sequence.py for the data side and the leakage design.

Do NOT import this module in a process that also trains LightGBM: on macOS their OpenMP runtimes
clash (segfault or deadlock). scripts/train_sequence_scores.py runs it in a separate process.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch import nn


class GRUScorer(nn.Module):
    """GRU over the installment sequence -> small dense layer -> default logit."""

    def __init__(self, n_features: int, hidden: int = 32, embed: int = 16):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, embed), nn.ReLU(), nn.Dropout(0.1), nn.Linear(embed, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        return self.head(h[-1]).squeeze(-1)


def predict(model: GRUScorer, X, lengths, batch: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]), torch.from_numpy(lengths[i:i + batch])).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_gru(X, lengths, y, X_val, len_val, y_val, epochs: int = 6, batch: int = 512, lr: float = 2e-3,
              patience: int = 1, seed: int = 42, verbose: bool = True):
    """Adam + BCE, early stopping on validation AUC. Returns (best model, history)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GRUScorer(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    best, best_state, history, bad = -1.0, None, [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(X))
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(X[b]), torch.from_numpy(lengths[b])), y_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = roc_auc_score(y_val, predict(model, X_val, len_val))
        history.append(auc)
        if verbose:
            print(f"    epoch {epoch}: valid AUC {auc:.4f}")
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad > patience:
                break
    model.load_state_dict(best_state)
    return model, history


def out_of_fold_scores(X, lengths, y, X_val, len_val, y_val, others: dict, n_folds: int = 5,
                       seed: int = 42, **train_kw):
    """Out-of-fold GRU logits for the training rows; mean of the fold models for X_val and every
    array in `others` ({name: (X, lengths)}). Returns (oof, {"valid": ..., name: ...}, histories)."""
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=np.float32)
    outside = {"valid": (X_val, len_val), **others}
    sums = {k: np.zeros(len(v[0])) for k, v in outside.items()}
    histories = []
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (fit, hold) in enumerate(folds.split(X, y), 1):
        print(f"  fold {k}/{n_folds}")
        model, hist = train_gru(X[fit], lengths[fit], y[fit], X_val, len_val, y_val, seed=seed + k, **train_kw)
        histories.append(hist)
        oof[hold] = predict(model, X[hold], lengths[hold])
        for name, (Xo, lo) in outside.items():
            sums[name] += predict(model, Xo, lo)
    return oof, {k: v / n_folds for k, v in sums.items()}, histories
