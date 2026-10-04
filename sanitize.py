"""Phase II - client-level update sanitisation (paper Section 4.3).

``sanitize_update``, ``dequantize_aggregate`` and ``clip_only`` are taken
verbatim from the benchmark notebook (QKD_SecAgg_Benchmarks.ipynb, Figure 8).
The wrap-around bound (Eq. 5) and the Renyi-DP accountant follow Section 4.3.
"""

from __future__ import annotations

import math

import numpy as np

R = 2**32


# ---- Eq. (2)-(4): clip -> Gaussian noise -> fixed-point quantisation -------
def sanitize_update(w, C, sigma, delta, R=2**32, rng=None):
    """Clip to L2 norm C, add N(0,(sigma*C)^2) per coordinate, quantize into Z_R."""
    rng = rng or np.random.default_rng()
    w = np.asarray(w, dtype=np.float64)
    norm = np.linalg.norm(w)
    if norm > C:
        w = w * (C / norm)
    noised = w + rng.normal(0.0, sigma * C, size=w.shape)
    return [int(v) for v in np.rint(delta * noised).astype(np.int64) % R]


# ---- Eq. (8): centred lift and de-quantisation ------------------------------
def dequantize_aggregate(agg, delta, R=2**32):
    """Map the modular aggregate back to signed reals (centered lift)."""
    a = np.asarray(agg, dtype=np.int64)
    return np.where(a >= R // 2, a - R, a) / delta


def clip_only(w, C):
    w = np.asarray(w, dtype=np.float64)
    norm = np.linalg.norm(w)
    return w * (C / norm) if norm > C else w


# ---- Eq. (5): no-wraparound condition ---------------------------------------
def tau(sigma, C, n, m, delta_wrap):
    """tau = sigma*C*sqrt(2 ln(2nm/delta_wrap)): bound on every noise sample w.p. >= 1-delta_wrap."""
    return sigma * C * math.sqrt(2 * math.log(2 * n * m / delta_wrap))


def wraparound_ok(n, m, C, sigma, delta, delta_wrap=1e-9, R=2**32):
    """Check n*(Delta*(C+tau) + 1/2) < R/2.  Returns (ok, lhs, rhs, tau)."""
    t = tau(sigma, C, n, m, delta_wrap)
    lhs = n * (delta * (C + t) + 0.5)
    return lhs < R / 2, lhs, R / 2, t


# ---- Client-level Renyi-DP accounting ---------------------------------------
def rdp_epsilon(sigma_effs, delta):
    """(eps, delta)-DP after T rounds from per-round effective multipliers.

    Each round is a Gaussian mechanism with L2 sensitivity 2C and noise sigma_eff*C,
    i.e. (alpha, 2*alpha/sigma_eff^2)-RDP.  Composition over rounds and conversion:
        eps = sum_r 2*alpha/sigma_eff_r^2 + log(1/delta)/(alpha-1),
    minimised over alpha > 1 (closed form: alpha* = 1 + sqrt(log(1/delta)/S)).
    Returns (eps, alpha*).
    """
    if not sigma_effs:
        return 0.0, None
    if any(s <= 0 for s in sigma_effs):
        return math.inf, None
    S = sum(2.0 / (s * s) for s in sigma_effs)
    L = math.log(1.0 / delta)
    alpha = 1.0 + math.sqrt(L / S)
    return S * alpha + L / (alpha - 1.0), alpha
