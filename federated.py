"""Toy federated learning task and the end-to-end training loop.

Each round composes the three phases exactly as in Section 4:
  local training -> Phase II sanitisation (clip, noise, quantise) on the client
  -> Phase III secure aggregation over the Phase I QKD transport
  -> centred de-quantisation (Eq. 8) -> FedAvg step on the server.

The model is a 2-feature logistic regression (m = 3 parameters) on synthetic,
non-IID client data, small enough to show the decision boundary move.  A
plaintext, non-private FedAvg baseline on the same data and dropouts is
trained alongside for comparison.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

import sanitize as S
import secagg_transport as T


# ---- toy data -----------------------------------------------------------------
@dataclass
class ToyData:
    client_X: dict
    client_y: dict
    X_test: np.ndarray
    y_test: np.ndarray


def make_toy_data(client_ids, samples_per_client=80, seed=0):
    """Two Gaussian classes in 2-D; each client sees a different class balance (non-IID)."""
    rng = np.random.default_rng(seed)
    mu = {0: np.array([2.5, 3.5]), 1: np.array([4.5, 3.0])}

    def draw(n, frac_pos):
        y = (rng.random(n) < frac_pos).astype(int)
        X = np.stack([rng.normal(mu[c], 0.8) for c in y])
        return X, y

    cx, cy = {}, {}
    fracs = np.linspace(0.15, 0.85, len(client_ids))
    rng.shuffle(fracs)
    for cid, f in zip(client_ids, fracs):
        cx[cid], cy[cid] = draw(samples_per_client, f)
    Xt, yt = draw(1000, 0.5)
    return ToyData(cx, cy, Xt, yt)


# ---- model ----------------------------------------------------------------------
def _aug(X):
    return np.hstack([X, np.ones((len(X), 1))])


def predict_proba(w, X):
    return 1.0 / (1.0 + np.exp(-_aug(X) @ w))


def evaluate(w, X, y):
    p = np.clip(predict_proba(w, X), 1e-9, 1 - 1e-9)
    loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    acc = float(np.mean((p >= 0.5) == y))
    return acc, loss


def local_update(w_global, X, y, epochs, lr):
    """Full-batch gradient descent from the global model; returns w_u = w_local - w_global."""
    w = w_global.copy()
    A = _aug(X)
    for _ in range(epochs):
        w -= lr * A.T @ (predict_proba(w, X) - y) / len(y)
    return w - w_global


# ---- training loop ---------------------------------------------------------------
@dataclass
class Config:
    threshold: int
    rounds: int = 25
    local_epochs: int = 1
    lr: float = 0.2
    C: float = 0.2
    sigma: float = 0.3
    delta_scale: int = 2**16
    dp_delta: float = 1e-5
    dropout_prob: float = 0.0
    seed: int = 0


@dataclass
class RoundRecord:
    r: int
    U3: list
    dropped: list
    ok: bool
    abort_reason: str
    acc: float
    loss: float
    base_acc: float
    eps: float
    sigma_eff: float | None
    agg_error: float | None          # |de-quantised SecAgg sum - plaintext sum of sanitised updates|
    norms: dict = field(default_factory=dict)
    w: np.ndarray | None = None
    trace: T.RoundTrace | None = None
    client_view: dict = field(default_factory=dict)   # sanitised x_u (never leaves the client)


def train(server, clients, data: ToyData, cfg: Config):
    """Generator yielding one RoundRecord per federated round."""
    rng = np.random.default_rng(cfg.seed)
    random.seed(cfg.seed)
    w = np.zeros(3)
    w_base = np.zeros(3)
    sigma_effs = []
    ids = [c.ID for c in clients]
    for r in range(1, cfg.rounds + 1):
        dropped = sorted(cid for cid in ids if rng.random() < cfg.dropout_prob)
        updates = {cid: local_update(w, data.client_X[cid], data.client_y[cid], cfg.local_epochs, cfg.lr)
                   for cid in ids}
        vectors = {cid: S.sanitize_update(u, cfg.C, cfg.sigma, cfg.delta_scale, rng=rng)
                   for cid, u in updates.items()}
        agg, trace = T.run_secure_aggregation_over_qkd(server, clients, vectors, cfg.threshold,
                                                       drop_after_shares=dropped)
        # Plaintext FedAvg baseline (no clipping, no noise) with the same survivors.
        base_survivors = [cid for cid in ids if cid not in dropped]
        if len(base_survivors) >= cfg.threshold:
            base_upd = [local_update(w_base, data.client_X[c], data.client_y[c], cfg.local_epochs, cfg.lr)
                        for c in base_survivors]
            w_base = w_base + np.mean(base_upd, axis=0)

        rec = RoundRecord(r=r, U3=trace.U3, dropped=dropped, ok=agg is not None,
                          abort_reason=trace.aborted, acc=0, loss=0, base_acc=0, eps=0,
                          sigma_eff=None, agg_error=None, trace=trace,
                          norms={c: float(np.linalg.norm(u)) for c, u in updates.items()},
                          client_view={c: np.array(v, dtype=np.uint64) for c, v in vectors.items()})
        if agg is not None:
            z_hat = S.dequantize_aggregate(agg, cfg.delta_scale)
            plain = S.dequantize_aggregate(
                np.sum([np.array(vectors[c], dtype=np.uint64) for c in trace.U3], axis=0) % S.R,
                cfg.delta_scale)
            rec.agg_error = float(np.max(np.abs(z_hat - plain)))
            w = w + z_hat / len(trace.U3)          # unweighted FedAvg (rho_u = 1)
            H = len(trace.U3)                      # all simulated clients are honest
            rec.sigma_eff = float(np.sqrt(H) * cfg.sigma)
            sigma_effs.append(rec.sigma_eff)
        rec.eps = S.rdp_epsilon(sigma_effs, cfg.dp_delta)[0]
        rec.acc, rec.loss = evaluate(w, data.X_test, data.y_test)
        rec.base_acc = evaluate(w_base, data.X_test, data.y_test)[0]
        rec.w = w.copy()
        yield rec
