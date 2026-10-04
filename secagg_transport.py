"""Phase III - secure aggregation carried over the QKD transport (paper Section 4.4).

"Every client-server message is carried over the channel created in phase 1."
The reference ``run_secure_aggregation`` in ``qkd_secagg.py`` passes messages
between objects directly.  This module runs the same Rounds 0-4 with the same
Client/Server methods, but every client->server and server->client message is
wrapped in a ChaCha20-Poly1305 envelope under that client's QKD transport key
(``qkd_transport_encrypt_for_server`` / ``..._decrypt_from_client`` etc.), and
the transport-authenticated sender is checked against the inner ``sender_id``.

Messages that the reference code exchanges as bare lists (the key directory,
the share relay, the confirmation relay) are wrapped in a typed dict, because
the transport layer only carries schema-valid messages with a ``type`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import qkd_secagg as M

ROUND_NAMES = {0: "AdvertiseKeys", 1: "ShareKeys", 2: "MaskedInputCollection",
               3: "ConsistencyCheck", 4: "Unmasking"}


@dataclass
class RoundTrace:
    round_id: str = ""
    U1: list = field(default_factory=list)
    U2: list = field(default_factory=list)
    U3: list = field(default_factory=list)
    U4: list = field(default_factory=list)
    U5: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    envelopes: dict = field(default_factory=lambda: {k: 0 for k in ROUND_NAMES})
    ciphertext_bytes: dict = field(default_factory=lambda: {k: 0 for k in ROUND_NAMES})
    masked_updates: dict = field(default_factory=dict)  # what the server actually received
    sample_envelope: dict | None = None
    aggregate: list | None = None
    aborted: str = ""


class _Wire:
    """All traffic between one server and its clients, encrypted under QKD keys."""

    def __init__(self, server, clients_by_id, trace):
        self.server, self.clients, self.trace = server, clients_by_id, trace

    def up(self, rnd, client, message):
        env = client.qkd_transport_encrypt_for_server(self.server.ID, message)
        self._count(rnd, env)
        inner = self.server.qkd_transport_decrypt_from_client(env)
        if inner.get("sender_id") != env["sender_id"]:
            raise ValueError("inner sender_id does not match the QKD-authenticated sender")
        return inner

    def down(self, rnd, client_id, message):
        env = self.server.qkd_transport_encrypt_for_client(client_id, message)
        self._count(rnd, env)
        return self.clients[client_id].qkd_transport_decrypt_from_server(self.server.ID, env)

    def _count(self, rnd, env):
        self.trace.envelopes[rnd] += 1
        self.trace.ciphertext_bytes[rnd] += len(env["ciphertext"])
        if self.trace.sample_envelope is None and rnd == 2:
            self.trace.sample_envelope = {k: (v.hex() if isinstance(v, bytes) else v)
                                          for k, v in env.items()}


def run_secure_aggregation_over_qkd(server, clients, vectors, threshold, *,
                                    vector_length=None, modulus=2**32,
                                    drop_after_shares=None, round_id=None):
    """One Bonawitz et al. round (Rounds 0-4) with QKD-protected transport.

    Mirrors ``qkd_secagg.run_secure_aggregation``; ``drop_after_shares`` are clients
    that complete Round 1 but never submit a masked update (dropout before U3).
    Returns (aggregate in Z_R^m, RoundTrace).
    """
    clients_by_id = {c.ID: c for c in clients}
    roster = tuple(c.ID for c in clients)
    for c in clients:
        if c.ID not in server.qkd_transport_keys:
            raise ValueError(f"{c.ID} has no QKD transport key; run Phase I first")
    if set(vectors) != set(roster):
        raise ValueError("vectors must provide exactly one entry per client")
    vector_length = vector_length or len(next(iter(vectors.values())))
    drop_after_shares = set(drop_after_shares or ())
    trace = RoundTrace(dropped=sorted(drop_after_shares))
    wire = _Wire(server, clients_by_id, trace)

    params = server.secure_agg_create_round(clients, threshold, vector_length,
                                            modulus=modulus, round_id=round_id)
    rid, signing_keys = params["round_id"], params["signing_public_keys"]
    trace.round_id = rid

    # Round 0 - AdvertiseKeys
    for c in clients:
        c.secure_agg_start_round(rid, roster, threshold, vector_length,
                                 modulus=modulus, signing_public_keys=signing_keys)
        server.secure_agg_collect_advertised_keys(wire.up(0, c, c.secure_agg_advertise_keys()))
    directory = server.secure_agg_relay_advertised_keys()
    trace.U1 = sorted(server.U1)
    for cid, entries in directory.items():
        msg = wire.down(0, cid, {"type": "advertise-directory", "round_id": rid, "directory": entries})
        clients_by_id[cid].secure_agg_verify_advertised_keys(msg["directory"])

    # Round 1 - ShareKeys
    for c in clients:
        bundle = wire.up(1, c, {"type": "share-bundle", "round_id": rid, "sender_id": c.ID,
                                "shares": c.secure_agg_create_and_encrypt_shares()})
        if any(s.get("sender_id") != c.ID for s in bundle["shares"]):
            raise ValueError("share bundle contains a share from another sender")
        server.secure_agg_collect_and_relay_encrypted_shares(bundle["shares"])
    relayed = server.secure_agg_collect_and_relay_encrypted_shares(finalize=True)
    trace.U2 = sorted(server.U2)
    for cid, shares in relayed.items():
        msg = wire.down(1, cid, {"type": "share-relay", "round_id": rid,
                                 "u2": sorted(relayed), "shares": shares})
        clients_by_id[cid].secure_agg_receive_and_decrypt_shares(msg["shares"], u2=msg["u2"])

    # Round 2 - MaskedInputCollection
    survivors = [c for c in clients if c.ID not in drop_after_shares]
    if len(survivors) < threshold:
        server.secure_agg_abort_round("fewer than threshold masked updates")
        for c in clients:
            c.secure_agg_abort_and_clear_state()
        trace.aborted = f"only {len(survivors)} survivors < threshold t={threshold}"
        return None, trace
    for c in survivors:
        server.secure_agg_collect_masked_updates(wire.up(2, c, c.secure_agg_create_masked_update(vectors[c.ID])))
    trace.masked_updates = {k: np.array(v, dtype=np.uint64) for k, v in server.secure_agg_masked_updates.items()}
    survivor_msgs = server.secure_agg_collect_masked_updates(finalize=True)
    trace.U3 = sorted(server.U3)

    # Round 3 - ConsistencyCheck
    for c in survivors:
        survivor_set = wire.down(3, c.ID, survivor_msgs[c.ID])
        server.secure_agg_collect_survivor_confirmations(wire.up(3, c, c.secure_agg_confirm_survivor_set(survivor_set)))
    conf_relay = server.secure_agg_collect_survivor_confirmations(finalize=True)
    trace.U4 = sorted(server.U4)
    for c in survivors:
        msg = wire.down(3, c.ID, {"type": "confirmation-relay", "round_id": rid,
                                  "confirmations": conf_relay[c.ID]})
        c.secure_agg_verify_survivor_confirmations(msg["confirmations"])

    # Round 4 - Unmasking
    for c in survivors:
        server.secure_agg_collect_unmasking_shares(wire.up(4, c, c.secure_agg_create_unmasking_response()))
    aggregate = server.secure_agg_collect_unmasking_shares(finalize=True)
    trace.U5 = sorted(server.U5)
    for c in clients:
        c.secure_agg_finalize_and_clear_state()
    trace.aggregate = aggregate
    return aggregate, trace
