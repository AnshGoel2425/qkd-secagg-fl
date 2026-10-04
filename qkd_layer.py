"""Phase I - QKD transport-key establishment (paper Section 4.2).

Thin orchestration around the reference implementation in ``qkd_secagg.py``.
The quantum simulation, sifting, QBER test, Cascade reconciliation, Toeplitz
privacy amplification and key derivation are all executed by the original
``Server._qkd_establish_pair_transport_keys``.  This module only

* runs one cluster-state session per client pair and records per-pair results,
* lets an eavesdropper (dephasing strength p) tap selected pairs,
* enforces the paper's final-length rule "the session aborts unless l >= 256"
  (Section 4.2, step 6), which the reference code does not check itself,
* installs the resulting 256-bit keys exactly as
  ``Server.qkd_establish_transport_keys`` does.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
from dataclasses import dataclass, field

import qkd_secagg as M

QBER_LIMIT = 0.0968          # abort tolerance inherited from Younes et al. [42]
MIN_FINAL_KEY_BITS = 256     # "The session aborts unless l >= 256"


def analytic_qber(p: float) -> float:
    """Closed-form mean test-round error rate under the dephasing model, (3p - p^2)/5."""
    return (3 * p - p * p) / 5.0


def p_star(limit: float = QBER_LIMIT) -> float:
    """Attack strength at which the mean QBER meets the abort threshold (paper: 0.171)."""
    # solve p^2 - 3p + 5*limit = 0, smaller root
    return (3 - (9 - 20 * limit) ** 0.5) / 2


@dataclass
class LinkResult:
    client_id: str
    raw_len: int = 0          # l_raw
    leaked: int = 0           # l_leak (Cascade parity bits disclosed)
    final_len: int = 0        # l after privacy amplification
    key_fingerprint: str = ""  # first 8 hex chars of SHA-256(key); never the key itself


@dataclass
class PairResult:
    pair: tuple
    attacked: bool
    p: float
    status: str = "pending"   # "established" | "aborted"
    reason: str = ""
    qber: float | None = None
    test_blocks: int = 0
    raw_blocks: int = 0
    case_counts: dict = field(default_factory=dict)
    links: list = field(default_factory=list)
    seconds: float = 0.0


def _instrumented(fn_name, sink):
    """Wrap a module-level function of qkd_secagg to record its inputs/outputs.

    The wrapped function is called unchanged; we only observe the leak count and
    the amplified length so they can be displayed and checked against l >= 256.
    """
    original = getattr(M, fn_name)

    def wrapper(*args, **kwargs):
        result = original(*args, **kwargs)
        label = kwargs.get("label", args[-1] if args and isinstance(args[-1], str) else "")
        sink.append((fn_name, label, args, kwargs, result))
        return result

    return original, wrapper


def establish_pair(server: M.Server, first: M.Client, second: M.Client, raw_bits: int,
                   p: float = 0.0, qber_limit: float = QBER_LIMIT) -> PairResult:
    """Run one three-party cluster-state session (two independent client-server keys)."""
    import time

    result = PairResult(pair=(first.ID, second.ID), attacked=p > 0, p=p)
    calls: list = []
    saved = {}
    for name in ("reconcile_keys", "privacy_amplification"):
        saved[name], wrapper = _instrumented(name, calls)
        setattr(M, name, wrapper)
    start_choice = len(server.gate_choice)
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            keys, meta = server._qkd_establish_pair_transport_keys(
                first, second, raw_bits, qber_limit, p_a=p, p_b=p)
    except RuntimeError as exc:
        keys, meta = None, None
        result.status, result.reason = "aborted", str(exc)
    finally:
        for name, fn in saved.items():
            setattr(M, name, fn)
        result.seconds = time.perf_counter() - t0

    choices = server.gate_choice[start_choice:]
    result.case_counts = {c: choices.count(c) for c in ("00", "01", "10", "11")}

    # Recover per-link statistics from the observed calls.
    per_link = {}
    for fn_name, label, args, kwargs, out in calls:
        link = per_link.setdefault(label, {})
        if fn_name == "reconcile_keys":
            link["raw_len"] = len(args[0])
            link["leaked"] = out[2]
        elif fn_name == "privacy_amplification" and out[0] is not None:
            link["final_len"] = len(out[0])
    for client in (first, second):
        label = f"{client.ID}-{server.ID}"
        info = per_link.get(label, {})
        result.links.append(LinkResult(client.ID, info.get("raw_len", 0),
                                       info.get("leaked", 0), info.get("final_len", 0)))

    if meta is not None:
        result.qber = meta["qber"]
        result.test_blocks = meta["test_blocks"]
        result.raw_blocks = meta["raw_indices"]
    else:
        # The reference code raises before returning metadata; recompute QBER from the
        # abort message so the UI can still show it.
        if "QBER" in result.reason:
            try:
                result.qber = float(result.reason.split("QBER ")[1].split("%")[0]) / 100
            except (IndexError, ValueError):
                pass

    if keys is not None:
        short = [l for l in result.links if l.final_len < MIN_FINAL_KEY_BITS]
        if short:
            result.status = "aborted"
            result.reason = (f"final key length l={min(l.final_len for l in short)} < "
                             f"{MIN_FINAL_KEY_BITS} bits (increase raw cluster states)")
            keys = None
    if keys is not None:
        # Install keys exactly as Server.qkd_establish_transport_keys does.
        for client in (first, second):
            server.qkd_transport_keys[client.ID] = keys[client.ID]
            client.qkd_transport_keys[server.ID] = keys[client.ID]
            client._qkd_transport_send_counter = 0
            client._qkd_transport_seen_counters.clear()
        server.qkd_pair_sessions[(first.ID, second.ID)] = meta
        for link in result.links:
            link.key_fingerprint = hashlib.sha256(keys[link.client_id]).hexdigest()[:8]
        result.status = "established"
    return result


def bootstrap(server: M.Server, clients: list, raw_bits: int, eve_p: float = 0.0,
              attacked_pairs: set | None = None, progress=None) -> list:
    """Pair the (even) roster and run one session per pair.  Returns PairResults."""
    if len(clients) < 2 or len(clients) % 2:
        raise ValueError("Phase I pairs clients for each cluster-state session: n must be even")
    attacked_pairs = attacked_pairs or set()
    server.qkd_transport_keys.clear(); server.qkd_pair_sessions.clear()
    server._qkd_transport_send_counters.clear(); server._qkd_transport_seen_counters.clear()
    results = []
    pairs = [tuple(clients[i:i + 2]) for i in range(0, len(clients), 2)]
    for k, (first, second) in enumerate(pairs):
        p = eve_p if k in attacked_pairs else 0.0
        results.append(establish_pair(server, first, second, raw_bits, p=p))
        if progress:
            progress(k + 1, len(pairs), results[-1])
    return results
