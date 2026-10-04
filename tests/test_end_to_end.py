"""End-to-end checks of the three phases.  Run:  python -m pytest -q"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import federated as F            # noqa: E402
import qkd_layer as Q            # noqa: E402
import qkd_secagg as M           # noqa: E402
import sanitize as S             # noqa: E402
import secagg_transport as T     # noqa: E402


@pytest.fixture(scope="module")
def network():
    ids = M.make_client_ids(4)
    clients = [M.Client(i) for i in ids]
    server = M.Server("Charlie")
    results = Q.bootstrap(server, clients, raw_bits=1600)
    return server, clients, results


def test_phase1_keys_established_and_distinct(network):
    server, clients, results = network
    assert all(r.status == "established" for r in results)
    keys = [server.qkd_transport_keys[c.ID] for c in clients]
    assert all(len(k) == 32 for k in keys) and len(set(keys)) == len(keys)
    assert all(c.qkd_transport_keys[server.ID] == server.qkd_transport_keys[c.ID] for c in clients)
    assert all(l.final_len >= Q.MIN_FINAL_KEY_BITS for r in results for l in r.links)


def test_phase1_eavesdropper_aborts():
    a, b, s = M.Client("A"), M.Client("B"), M.Server("S")
    r = Q.establish_pair(s, a, b, raw_bits=1600, p=0.5)
    assert r.status == "aborted" and "QBER" in r.reason
    assert "A" not in s.qkd_transport_keys


def test_phase1_short_key_aborts():
    a, b, s = M.Client("A"), M.Client("B"), M.Server("S")
    r = Q.establish_pair(s, a, b, raw_bits=400)
    assert r.status == "aborted" and "256" in r.reason


def test_phase3_exact_with_dropout(network):
    server, clients, _ = network
    rng = np.random.default_rng(0)
    raw = {c.ID: rng.normal(0, 0.5, 5) for c in clients}
    vec = {k: S.sanitize_update(v, 1.0, 0.1, 2**16, rng=rng) for k, v in raw.items()}
    agg, trace = T.run_secure_aggregation_over_qkd(server, clients, vec, 3,
                                                   drop_after_shares={clients[1].ID})
    assert agg == M.expected_sum(vec, trace.U3)
    assert clients[1].ID not in trace.U3 and len(trace.U3) == 3
    assert sum(trace.envelopes.values()) > 0


def test_phase3_aborts_below_threshold(network):
    server, clients, _ = network
    vec = {c.ID: [1, 2] for c in clients}
    agg, trace = T.run_secure_aggregation_over_qkd(server, clients, vec, 3,
                                                   drop_after_shares={clients[0].ID, clients[1].ID})
    assert agg is None and trace.aborted


def test_transport_rejects_tampering(network):
    server, clients, _ = network
    env = clients[0].qkd_transport_encrypt_for_server(server.ID, {"type": "ping", "round_id": None})
    env["ciphertext"] = bytes([env["ciphertext"][0] ^ 1]) + env["ciphertext"][1:]
    with pytest.raises(ValueError):
        server.qkd_transport_decrypt_from_client(env)


def test_phase2_dequantisation_roundtrip():
    w = np.array([0.3, -0.4, 1.5])
    x = S.sanitize_update(w, C=1.0, sigma=0.0, delta=2**16)
    np.testing.assert_allclose(S.dequantize_aggregate(x, 2**16), S.clip_only(w, 1.0), atol=1e-4)
    assert S.wraparound_ok(n=8, m=64, C=1.0, sigma=1.0, delta=2**16)[0]


def test_rdp_accountant():
    e1, _ = S.rdp_epsilon([2.0] * 10, 1e-5)
    e2, _ = S.rdp_epsilon([4.0] * 10, 1e-5)
    assert e2 < e1 and np.isinf(S.rdp_epsilon([0.0], 1e-5)[0])


def test_training_improves(network):
    server, clients, _ = network
    data = F.make_toy_data([c.ID for c in clients], seed=1)
    recs = list(F.train(server, clients, data, F.Config(threshold=3, rounds=15, sigma=0.3, C=0.2, lr=0.2)))
    assert recs[-1].acc > 0.75
    assert all(r.agg_error == 0 for r in recs if r.ok)
