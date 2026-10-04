import random
import numpy as np
import hashlib
import json
import secrets
import struct
import time
from collections import deque
from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.quantum_info import Statevector
    from qiskit.circuit.library import HGate, IGate
except ImportError:  # Secure aggregation can be used without the QKD simulator installed.
    QuantumCircuit = QuantumRegister = ClassicalRegister = Statevector = HGate = IGate = None
from IPython.display import display, Math

def cluster_state_generation():
    """Circuit to generate a 4-qubit cluster state."""
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is required for the QKD simulation")
    circuit = QuantumCircuit(4)
    circuit.h(3)
    circuit.cx(3, 0)
    circuit.h(2)
    circuit.cx(2, 1)
    circuit.cz(3, 2)
    return circuit.to_gate(label="Cluster_Gen")


# Secure-aggregation helpers are intentionally stateless.  They encode data
# canonically rather than serializing cryptographic objects or using pickle.
_SA_VERSION = 1
_SA_PRIME = 2**256 - 2**32 - 977  # secp256k1 prime; safely holds 32-byte secrets.


def _sa_encode(value):
    """Canonical, type-tagged binary encoding for signed/AEAD-bound messages."""
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("canonical secure-aggregation integers must be non-negative")
        raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
        return b"i" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, bytes):
        return b"y" + struct.pack(">I", len(value)) + value
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"s" + struct.pack(">I", len(raw)) + raw
    if isinstance(value, (tuple, list)):
        return b"l" + struct.pack(">I", len(value)) + b"".join(_sa_encode(v) for v in value)
    if isinstance(value, dict):
        items = []
        for key in sorted(value):
            if not isinstance(key, str):
                raise ValueError("canonical secure-aggregation map keys must be strings")
            items.append(_sa_encode(key) + _sa_encode(value[key]))
        return b"d" + struct.pack(">I", len(items)) + b"".join(items)
    raise TypeError("unsupported secure-aggregation canonical value")


def _sa_decode(data, offset=0):
    """Inverse for the small canonical format used inside encrypted shares."""
    if offset >= len(data):
        raise ValueError("truncated canonical data")
    tag = data[offset:offset + 1]; offset += 1
    if tag == b"n":
        return None, offset
    if tag == b"b":
        if offset >= len(data) or data[offset] not in (0, 1):
            raise ValueError("malformed boolean")
        return bool(data[offset]), offset + 1
    if tag in (b"i", b"y", b"s", b"l", b"d"):
        if offset + 4 > len(data):
            raise ValueError("truncated canonical length")
        length = struct.unpack(">I", data[offset:offset + 4])[0]; offset += 4
        if tag == b"i":
            if length == 0 or offset + length > len(data):
                raise ValueError("malformed integer")
            return int.from_bytes(data[offset:offset + length], "big"), offset + length
        if tag == b"y":
            if offset + length > len(data):
                raise ValueError("truncated byte string")
            return data[offset:offset + length], offset + length
        if tag == b"s":
            if offset + length > len(data):
                raise ValueError("truncated string")
            try:
                return data[offset:offset + length].decode("utf-8"), offset + length
            except UnicodeDecodeError as exc:
                raise ValueError("invalid UTF-8 string") from exc
        if tag == b"l":
            result = []
            for _ in range(length):
                item, offset = _sa_decode(data, offset); result.append(item)
            return result, offset
        result = {}
        previous = None
        for _ in range(length):
            key, offset = _sa_decode(data, offset)
            value, offset = _sa_decode(data, offset)
            if not isinstance(key, str):
                raise ValueError("non-string map key")
            if previous is not None and key <= previous:
                raise ValueError("non-canonical map ordering")
            previous = key
            result[key] = value
        return result, offset
    raise ValueError("unknown canonical type tag")


def _sa_decode_share_plaintext(data):
    value, offset = _sa_decode(data)
    required = {"round_id", "sender", "recipient", "x_s", "s_share", "x_b", "b_share"}
    if offset != len(data) or not isinstance(value, dict) or set(value) != required:
        raise ValueError("malformed encrypted-share plaintext")
    if (not all(isinstance(value[key], str) for key in ("round_id", "sender", "recipient"))
            or not all(isinstance(value[key], int) and value[key] > 0 for key in ("x_s", "x_b"))
            or not all(isinstance(value[key], bytes) and len(value[key]) == 32 for key in ("s_share", "b_share"))):
        raise ValueError("invalid encrypted-share fields")
    return value


def _sa_decode_message(data):
    """Decode one complete canonical transport payload without pickle/JSON ambiguity."""
    value, offset = _sa_decode(data)
    if offset != len(data) or not isinstance(value, dict):
        raise ValueError("malformed canonical transport payload")
    return value


def _sa_hash(value):
    return hashlib.sha256(_sa_encode(value)).digest()


def _sa_raw_public_key(key):
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def _sa_raw_private_key(key):
    return key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                             serialization.NoEncryption())


def _sa_hkdf(shared_secret, round_id, info):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=round_id.encode("utf-8"),
                info=info).derive(shared_secret)


def _sa_prg_vector(seed, length, modulus, label, context=b""):
    """SHA-256 counter-mode PRG.  With R=2**32 each 32-bit word is unbiased."""
    if modulus != 2**32:
        raise ValueError("this implementation intentionally supports modulus R=2**32 only")
    if length < 0:
        raise ValueError("vector length must be non-negative")
    output = bytearray()
    counter = 0
    domain = _sa_encode(["secure-agg/prg", label, context])
    while len(output) < 4 * length:
        output.extend(hashlib.sha256(domain + seed + struct.pack(">Q", counter)).digest())
        counter += 1
    return np.frombuffer(bytes(output[:4 * length]), dtype=">u4").astype(np.uint64)


def _sa_random_field_secret():
    while True:
        raw = secrets.token_bytes(32)
        if int.from_bytes(raw, "big") < _SA_PRIME:
            return raw


def _sa_shamir_split(secret_bytes, threshold, share_count):
    """t-of-n Shamir sharing over the documented 256-bit prime field."""
    if not (2 <= threshold <= share_count):
        raise ValueError("invalid Shamir threshold")
    secret = int.from_bytes(secret_bytes, "big")
    if secret >= _SA_PRIME:
        raise ValueError("secret is outside the Shamir field")
    coefficients = [secret] + [secrets.randbelow(_SA_PRIME) for _ in range(threshold - 1)]
    shares = []
    for x in range(1, share_count + 1):
        y = 0
        for coefficient in reversed(coefficients):
            y = (y * x + coefficient) % _SA_PRIME
        shares.append((x, y.to_bytes(32, "big")))
    return shares


def _sa_shamir_reconstruct(shares, threshold):
    if len(shares) < threshold:
        raise ValueError("not enough Shamir shares")
    unique = {}
    for x, y in shares:
        if not isinstance(x, int) or not 1 <= x <= 2**32 or len(y) != 32:
            raise ValueError("malformed Shamir share")
        if x in unique and unique[x] != y:
            raise ValueError("inconsistent duplicate Shamir share")
        unique[x] = y
    points = list(unique.items())
    if len(points) < threshold:
        raise ValueError("not enough unique Shamir shares")
    points = points[:threshold]
    secret = 0
    for i, (x_i, y_i) in enumerate(points):
        numerator, denominator = 1, 1
        for j, (x_j, _) in enumerate(points):
            if i != j:
                numerator = (numerator * (-x_j)) % _SA_PRIME
                denominator = (denominator * (x_i - x_j)) % _SA_PRIME
        secret = (secret + int.from_bytes(y_i, "big") * numerator * pow(denominator, -1, _SA_PRIME)) % _SA_PRIME
    return secret.to_bytes(32, "big")


def _sa_advertise_body(round_id, client_id, c_pk, s_pk, roster, threshold, modulus, vector_length):
    return _sa_encode([_SA_VERSION, "advertise", round_id, client_id, c_pk, s_pk,
                       list(roster), threshold, modulus, vector_length])


def _sa_confirmation_body(round_id, client_id, survivor_hash, roster, threshold, modulus, vector_length):
    return _sa_encode([_SA_VERSION, "survivor-confirmation", round_id, client_id,
                       survivor_hash, list(roster), threshold, modulus, vector_length])

class Client():
    def __init__(self, ID):
        self.ID = ID
        self.measurements = []
        # Long-term identity key.  It is never used for encryption or ECDH.
        self.signing_private_key = ed25519.Ed25519PrivateKey.generate()
        self.signing_public_key = self.signing_private_key.public_key()
        self.secure_agg_round_id = None
        self.secure_agg_participants = ()
        self.secure_agg_threshold = None
        self.secure_agg_modulus = None
        self.secure_agg_vector_length = None
        self.share_enc_private_key = None
        self.share_enc_public_key = None
        self.mask_private_key = None
        self.mask_public_key = None
        self.self_mask_seed = None
        self.peer_advertised_keys = {}
        self.received_share_ciphertexts = {}
        self.received_shares = {}
        self.secure_agg_cleanup_tracking = {"wiped": True, "reason": None}
        self._sa_registry = {}
        self._sa_initial_roster = ()
        self._sa_u1 = ()
        self._sa_u2 = ()
        self._sa_u3 = ()
        self._sa_confirmations = {}
        self._sa_seen_messages = set()
        # Populated only by a completed QKD run with the aggregation server.
        # This is the sole client<->server transport-key material.
        self.qkd_transport_keys = {}
        self._qkd_transport_send_counter = 0
        self._qkd_transport_seen_counters = set()
        

    def apply_basis_gate(self, bit, circuit, index):
        if bit == 0:
            circuit.append(IGate(), [index])
        elif bit == 1:
            circuit.append(HGate(), [index])
        else:
            raise ValueError("The bit argument must be either 0 or 1.")

    def secure_agg_start_round(self, round_id, participants, threshold, vector_length,
                               modulus=2**32, signing_public_keys=None):
        """Install verified public round parameters and generate fresh X25519 keys."""
        participants = tuple(participants)
        if (not isinstance(round_id, str) or not round_id or len(set(participants)) != len(participants)
                or self.ID not in participants or any(not isinstance(p, str) or not p for p in participants)):
            raise ValueError("invalid secure-aggregation roster")
        if not (2 <= threshold <= len(participants)) or vector_length < 0 or modulus != 2**32:
            raise ValueError("invalid secure-aggregation round parameters")
        if signing_public_keys is None or set(signing_public_keys) != set(participants):
            raise ValueError("a trusted signing-key registry for exactly the roster is required")
        registry = {}
        for participant, public_key in signing_public_keys.items():
            try:
                registry[participant] = (public_key if isinstance(public_key, ed25519.Ed25519PublicKey)
                                         else ed25519.Ed25519PublicKey.from_public_bytes(public_key))
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid signing public key in trusted registry") from exc
        self.secure_agg_abort_and_clear_state()
        self.secure_agg_round_id = round_id
        self._sa_initial_roster = participants
        self.secure_agg_participants = participants
        self.secure_agg_threshold = threshold
        self.secure_agg_modulus = modulus
        self.secure_agg_vector_length = vector_length
        self._sa_registry = registry
        self.share_enc_private_key = x25519.X25519PrivateKey.generate()
        self.share_enc_public_key = self.share_enc_private_key.public_key()
        self.mask_private_key = x25519.X25519PrivateKey.generate()
        self.mask_public_key = self.mask_private_key.public_key()
        self.secure_agg_cleanup_tracking = {"wiped": False, "reason": None}

    def secure_agg_advertise_keys(self):
        if self.secure_agg_round_id is None or self.share_enc_public_key is None:
            raise ValueError("no active secure-aggregation round")
        c_pk, s_pk = _sa_raw_public_key(self.share_enc_public_key), _sa_raw_public_key(self.mask_public_key)
        body = _sa_advertise_body(self.secure_agg_round_id, self.ID, c_pk, s_pk,
                                  self._sa_initial_roster, self.secure_agg_threshold,
                                  self.secure_agg_modulus, self.secure_agg_vector_length)
        return {"version": _SA_VERSION, "type": "advertise", "round_id": self.secure_agg_round_id,
                "sender_id": self.ID, "c_pk": c_pk, "s_pk": s_pk,
                "signature": self.signing_private_key.sign(body)}

    def secure_agg_verify_advertised_keys(self, advertised_keys):
        """Verify the signed U1 directory relayed by the server."""
        if self.secure_agg_round_id is None:
            raise ValueError("no active secure-aggregation round")
        if not isinstance(advertised_keys, (list, tuple)):
            raise ValueError("malformed advertised-key directory")
        directory, c_seen, s_seen = {}, set(), set()
        for message in advertised_keys:
            required = {"version", "type", "round_id", "sender_id", "c_pk", "s_pk", "signature"}
            if not isinstance(message, dict) or set(message) != required:
                raise ValueError("malformed advertised-key message")
            sender = message["sender_id"]
            if (message["version"] != _SA_VERSION or message["type"] != "advertise"
                    or message["round_id"] != self.secure_agg_round_id or sender not in self._sa_initial_roster
                    or sender in directory or len(message["c_pk"]) != 32 or len(message["s_pk"]) != 32):
                raise ValueError("invalid advertised-key message")
            body = _sa_advertise_body(self.secure_agg_round_id, sender, message["c_pk"], message["s_pk"],
                                      self._sa_initial_roster, self.secure_agg_threshold,
                                      self.secure_agg_modulus, self.secure_agg_vector_length)
            try:
                self._sa_registry[sender].verify(message["signature"], body)
                c_key = x25519.X25519PublicKey.from_public_bytes(message["c_pk"])
                s_key = x25519.X25519PublicKey.from_public_bytes(message["s_pk"])
            except (InvalidSignature, TypeError, ValueError) as exc:
                self.secure_agg_abort_and_clear_state()
                raise ValueError("invalid advertised-key signature or key") from exc
            if message["c_pk"] in c_seen or message["s_pk"] in s_seen:
                raise ValueError("duplicate ephemeral public key")
            c_seen.add(message["c_pk"]); s_seen.add(message["s_pk"])
            directory[sender] = {"share_enc_public_key": c_key, "mask_public_key": s_key,
                                 "c_pk": message["c_pk"], "s_pk": message["s_pk"]}
        if self.ID not in directory or len(directory) < self.secure_agg_threshold:
            raise ValueError("U1 is invalid or too small")
        self.peer_advertised_keys = directory
        self._sa_u1 = tuple(sorted(directory))
        self.secure_agg_participants = self._sa_u1

    def secure_agg_create_and_encrypt_shares(self):
        if not self._sa_u1 or self.ID not in self._sa_u1:
            raise ValueError("advertised keys have not been verified")
        self.self_mask_seed = _sa_random_field_secret()
        mask_private_raw = _sa_raw_private_key(self.mask_private_key)
        member_count = len(self._sa_u1)
        s_shares = _sa_shamir_split(mask_private_raw, self.secure_agg_threshold, member_count)
        b_shares = _sa_shamir_split(self.self_mask_seed, self.secure_agg_threshold, member_count)
        roster_hash = _sa_hash(list(self._sa_u1))
        messages = []
        for index, recipient in enumerate(self._sa_u1):
            x_s, s_share = s_shares[index]
            x_b, b_share = b_shares[index]
            plaintext = _sa_encode({"round_id": self.secure_agg_round_id, "sender": self.ID,
                                    "recipient": recipient, "x_s": x_s, "s_share": s_share,
                                    "x_b": x_b, "b_share": b_share})
            shared = self.share_enc_private_key.exchange(self.peer_advertised_keys[recipient]["share_enc_public_key"])
            key = _sa_hkdf(shared, self.secure_agg_round_id,
                           _sa_encode(["secure-agg/share-aead", self.ID, recipient]))
            aad = _sa_encode([_SA_VERSION, "encrypted-share", self.secure_agg_round_id, self.ID, recipient,
                              self.secure_agg_threshold, self.secure_agg_modulus,
                              self.secure_agg_vector_length, roster_hash])
            nonce = secrets.token_bytes(12)
            messages.append({"version": _SA_VERSION, "type": "encrypted-share", "round_id": self.secure_agg_round_id,
                             "sender_id": self.ID, "recipient_id": recipient, "nonce": nonce,
                             "ciphertext": ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)})
        return messages

    def secure_agg_receive_and_decrypt_shares(self, ciphertexts, u2=None):
        if not self._sa_u1:
            raise ValueError("share phase was not initialized")
        u2 = tuple(sorted(u2 if u2 is not None else self._sa_u1))
        if self.ID not in u2 or len(u2) < self.secure_agg_threshold or not set(u2).issubset(self._sa_u1):
            raise ValueError("invalid U2")
        expected = set(u2)
        received = {}
        roster_hash = _sa_hash(list(self._sa_u1))
        for message in ciphertexts:
            sender = message.get("sender_id") if isinstance(message, dict) else None
            if (not isinstance(message, dict) or message.get("version") != _SA_VERSION
                    or message.get("type") != "encrypted-share" or message.get("round_id") != self.secure_agg_round_id
                    or message.get("recipient_id") != self.ID or sender not in expected or sender in received
                    or len(message.get("nonce", b"")) != 12):
                raise ValueError("invalid encrypted-share message")
            shared = self.share_enc_private_key.exchange(self.peer_advertised_keys[sender]["share_enc_public_key"])
            key = _sa_hkdf(shared, self.secure_agg_round_id,
                           _sa_encode(["secure-agg/share-aead", sender, self.ID]))
            aad = _sa_encode([_SA_VERSION, "encrypted-share", self.secure_agg_round_id, sender, self.ID,
                              self.secure_agg_threshold, self.secure_agg_modulus,
                              self.secure_agg_vector_length, roster_hash])
            try:
                # The plaintext is re-encoded only after a strict local decoding below.
                plaintext = ChaCha20Poly1305(key).decrypt(message["nonce"], message["ciphertext"], aad)
                decoded = _sa_decode_share_plaintext(plaintext)
            except (InvalidTag, ValueError, TypeError) as exc:
                self.secure_agg_abort_and_clear_state()
                raise ValueError("encrypted share failed authentication") from exc
            if (decoded["round_id"] != self.secure_agg_round_id or decoded["sender"] != sender
                    or decoded["recipient"] != self.ID):
                raise ValueError("encrypted-share metadata mismatch")
            received[sender] = decoded
        if set(received) != expected:
            raise ValueError("incomplete incoming encrypted-share bundle")
        self.received_share_ciphertexts = {m["sender_id"]: m for m in ciphertexts}
        self.received_shares = received
        self._sa_u2 = u2

    def secure_agg_create_masked_update(self, update):
        if not self.self_mask_seed or set(self.received_shares) != set(self._sa_u2):
            raise ValueError("complete authenticated shares are required before masking")
        if len(update) != self.secure_agg_vector_length:
            raise ValueError("incorrect update vector length")
        try:
            values = [int(v) for v in update]
        except (TypeError, ValueError) as exc:
            raise ValueError("updates must be integer vectors") from exc
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               or v < 0 or v >= self.secure_agg_modulus for v in update):
            raise ValueError("updates must be elements of Z_R, not floats or out-of-range integers")
        masked = np.array(values, dtype=np.uint64)
        context = _sa_encode([self.secure_agg_round_id, list(self._sa_u2)])
        for peer in self._sa_u2:
            if peer == self.ID:
                continue
            shared = self.mask_private_key.exchange(self.peer_advertised_keys[peer]["mask_public_key"])
            left, right = sorted((self.ID, peer))
            seed = _sa_hkdf(shared, self.secure_agg_round_id,
                            _sa_encode(["secure-agg/pairwise-mask", left, right]))
            mask = _sa_prg_vector(seed, self.secure_agg_vector_length, self.secure_agg_modulus,
                                  "pairwise", context + _sa_encode([left, right]))
            masked = ((masked - mask) if self.ID < peer else (masked + mask)) % self.secure_agg_modulus
        masked = (masked + _sa_prg_vector(self.self_mask_seed, self.secure_agg_vector_length,
                                          self.secure_agg_modulus, "self", context)) % self.secure_agg_modulus
        return {"version": _SA_VERSION, "type": "masked-update", "round_id": self.secure_agg_round_id,
                "sender_id": self.ID, "vector": masked.astype(np.uint64).tolist()}

    def secure_agg_confirm_survivor_set(self, survivor_ids):
        if isinstance(survivor_ids, dict):
            required = {"version", "type", "round_id", "sender_id", "survivors"}
            if (set(survivor_ids) != required or survivor_ids["version"] != _SA_VERSION
                    or survivor_ids["type"] != "survivor-set" or survivor_ids["round_id"] != self.secure_agg_round_id
                    or not isinstance(survivor_ids["sender_id"], str)):
                self.secure_agg_abort_and_clear_state()
                raise ValueError("malformed survivor-set message")
            survivor_ids = survivor_ids["survivors"]
        survivor_ids = tuple(survivor_ids)
        if (self.ID not in survivor_ids or len(survivor_ids) < self.secure_agg_threshold
                or tuple(sorted(survivor_ids)) != survivor_ids or not set(survivor_ids).issubset(self._sa_u2)):
            raise ValueError("invalid survivor set")
        self._sa_u3 = survivor_ids
        digest = _sa_hash(list(survivor_ids))
        body = _sa_confirmation_body(self.secure_agg_round_id, self.ID, digest, self._sa_initial_roster,
                                     self.secure_agg_threshold, self.secure_agg_modulus,
                                     self.secure_agg_vector_length)
        return {"version": _SA_VERSION, "type": "survivor-confirmation", "round_id": self.secure_agg_round_id,
                "sender_id": self.ID, "survivor_hash": digest,
                "signature": self.signing_private_key.sign(body)}

    def secure_agg_verify_survivor_confirmations(self, confirmations, survivor_ids=None):
        survivor_ids = tuple(survivor_ids if survivor_ids is not None else self._sa_u3)
        digest = _sa_hash(list(survivor_ids))
        seen = {}
        for message in confirmations:
            sender = message.get("sender_id") if isinstance(message, dict) else None
            if (not isinstance(message, dict) or message.get("version") != _SA_VERSION
                    or message.get("type") != "survivor-confirmation" or message.get("round_id") != self.secure_agg_round_id
                    or sender not in survivor_ids or sender in seen or message.get("survivor_hash") != digest):
                raise ValueError("invalid survivor confirmation")
            body = _sa_confirmation_body(self.secure_agg_round_id, sender, digest, self._sa_initial_roster,
                                         self.secure_agg_threshold, self.secure_agg_modulus,
                                         self.secure_agg_vector_length)
            try:
                self._sa_registry[sender].verify(message["signature"], body)
            except (InvalidSignature, KeyError) as exc:
                self.secure_agg_abort_and_clear_state()
                raise ValueError("invalid survivor confirmation signature") from exc
            seen[sender] = message
        if self.ID not in seen or len(seen) < self.secure_agg_threshold:
            raise ValueError("not enough consistent survivor confirmations")
        self._sa_confirmations = seen

    def secure_agg_create_unmasking_response(self):
        if not self._sa_confirmations or self.ID not in self._sa_u3:
            raise ValueError("consistent survivor confirmations are required")
        targets = []
        dropped = set(self._sa_u2) - set(self._sa_u3)
        for target in self._sa_u2:
            if target not in self.received_shares:
                raise ValueError("missing decrypted share")
            share = self.received_shares[target]
            if target in dropped:
                targets.append({"target_id": target, "kind": "mask_private", "x": share["x_s"], "share": share["s_share"]})
            else:
                targets.append({"target_id": target, "kind": "self_mask", "x": share["x_b"], "share": share["b_share"]})
        # One type per target is enforced above and bound by a signature.
        unsigned = {"version": _SA_VERSION, "type": "unmasking-response", "round_id": self.secure_agg_round_id,
                    "sender_id": self.ID, "shares": targets}
        return dict(unsigned, signature=self.signing_private_key.sign(_sa_encode(unsigned)))

    def secure_agg_abort_and_clear_state(self):
        self.share_enc_private_key = self.share_enc_public_key = None
        self.mask_private_key = self.mask_public_key = None
        self.self_mask_seed = None
        self.peer_advertised_keys.clear(); self.received_share_ciphertexts.clear(); self.received_shares.clear()
        self._sa_u1 = self._sa_u2 = self._sa_u3 = ()
        self._sa_confirmations.clear(); self._sa_seen_messages.clear()
        self.secure_agg_cleanup_tracking = {"wiped": True, "reason": "abort"}

    def secure_agg_finalize_and_clear_state(self):
        self.secure_agg_abort_and_clear_state()
        self.secure_agg_cleanup_tracking = {"wiped": True, "reason": "finalize"}

    def qkd_transport_encrypt_for_server(self, server_id, message):
        """AEAD-wrap one protocol message using this client's QKD-derived key only."""
        key = self.qkd_transport_keys.get(server_id)
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("no QKD-derived transport key is available for this server")
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ValueError("a schema-valid protocol message is required")
        counter = self._qkd_transport_send_counter
        if counter >= 2**88:
            raise ValueError("QKD transport nonce space exhausted; establish fresh QKD key material")
        self._qkd_transport_send_counter += 1
        nonce = b"\x00" + counter.to_bytes(11, "big")  # direction-separated client -> server nonce
        inner_type, inner_round = message["type"], message.get("round_id")
        aad = _sa_encode([_SA_VERSION, "qkd-transport", self.ID, server_id, counter, inner_type, inner_round])
        return {"version": _SA_VERSION, "type": "qkd-transport", "sender_id": self.ID,
                "recipient_id": server_id, "counter": counter, "inner_type": inner_type,
                "inner_round_id": inner_round, "nonce": nonce,
                "ciphertext": ChaCha20Poly1305(key).encrypt(nonce, _sa_encode(message), aad)}

    def qkd_transport_decrypt_from_server(self, server_id, envelope):
        key = self.qkd_transport_keys.get(server_id)
        required = {"version", "type", "sender_id", "recipient_id", "counter", "inner_type",
                    "inner_round_id", "nonce", "ciphertext"}
        if (not isinstance(key, bytes) or len(key) != 32 or not isinstance(envelope, dict)
                or set(envelope) != required or envelope["version"] != _SA_VERSION
                or envelope["type"] != "qkd-transport" or envelope["sender_id"] != server_id
                or envelope["recipient_id"] != self.ID or not isinstance(envelope["counter"], int)
                or envelope["counter"] < 0 or envelope["counter"] in self._qkd_transport_seen_counters
                or envelope["nonce"] != b"\x01" + envelope["counter"].to_bytes(11, "big")):
            raise ValueError("invalid or replayed QKD transport envelope")
        aad = _sa_encode([_SA_VERSION, "qkd-transport", server_id, self.ID, envelope["counter"],
                          envelope["inner_type"], envelope["inner_round_id"]])
        try:
            message = _sa_decode_message(ChaCha20Poly1305(key).decrypt(envelope["nonce"], envelope["ciphertext"], aad))
        except (InvalidTag, ValueError, TypeError) as exc:
            raise ValueError("QKD transport authentication failed") from exc
        if message.get("type") != envelope["inner_type"] or message.get("round_id") != envelope["inner_round_id"]:
            raise ValueError("QKD transport inner-message binding failed")
        self._qkd_transport_seen_counters.add(envelope["counter"])
        return message

class Server():
    def __init__(self, ID):
        self.ID = ID
        self.gate_choice = []
        self.measurements = []
        self.secure_agg_round_id = None
        self.secure_agg_phase = "idle"
        self.secure_agg_threshold = None
        self.secure_agg_participants = ()
        self.U1 = set(); self.U2 = set(); self.U3 = set(); self.U4 = set(); self.U5 = set()
        self.secure_agg_advertised_public_keys = {}
        self.secure_agg_advertised_signatures = {}
        self.secure_agg_relayed_ciphertexts = {}
        self.secure_agg_masked_updates = {}
        self.secure_agg_survivor_confirmations = {}
        self.secure_agg_unmasking_shares = {}
        self.secure_agg_modulus = None
        self.secure_agg_vector_length = None
        self.secure_agg_deadlines = {}
        self.secure_agg_replay_protection = set()
        self._sa_registry = {}
        self._sa_roster_hash = None
        self._sa_abort_reason = None
        # One distinct final QKD key per directly connected client.
        self.qkd_transport_keys = {}
        self.qkd_pair_sessions = {}
        self._qkd_transport_send_counters = {}
        self._qkd_transport_seen_counters = {}

    def generate_single_cluster_state(self):
        """Generates a single 4-qubit block to bypass memory limits."""
        circuit = QuantumCircuit(4, 4)
        circuit.append(cluster_state_generation(), [0, 1, 2, 3])
        circuit.barrier()
        return circuit

    def distribute_apply_gate_single(self, circuit, alice, bob, p_a = 0.0, p_b = 0.0):
        """Randomly applies Alice and Bob's choices for a single block.
        Eve's Z-basis intercept-resend is modeled as dephasing (Z with prob p/2)
        on the flying qubits, applied in-channel: after cluster creation,
        before the clients' basis gates."""
        # --- Eve's channel attack (trajectory form of the dephasing channel) ---
        if random.random() < p_a and random.random() < 0.5:
            circuit.z(0)   # Alice's flying qubit
        if random.random() < p_b and random.random() < 0.5:
            circuit.z(1)   # Bob's flying qubit
        # --- clients receive and operate ---
        a_choice = random.randint(0, 1)
        b_choice = random.randint(0, 1)
        alice.apply_basis_gate(a_choice, circuit, 0)
        bob.apply_basis_gate(b_choice, circuit, 1)
        return circuit, a_choice, b_choice
    
    def apply_basis_gate(self, bit, circuit, index):
        if bit == 0:
            circuit.append(IGate(), [index])
        elif bit == 1:
            circuit.append(HGate(), [index])

    def server_applies_gate_single(self, circuit, a_choice, b_choice):
        """Charlie applies his gates based on Alice and Bob's choices."""
        if a_choice == 1 and b_choice == 1:
            self.apply_basis_gate(0, circuit, 3)
            self.apply_basis_gate(0, circuit, 2)
        else:
            self.apply_basis_gate(a_choice, circuit, 3)
            self.apply_basis_gate(b_choice, circuit, 2)

        combined_choice = f"{a_choice}{b_choice}"
        self.gate_choice.append(combined_choice)
        return circuit, combined_choice

    def measure_single_block(self, circuit, alice, bob, choice, block_idx):
        """Measures the single block, extracts results, and logs detailed debug info."""

        # 1. Apply Bell basis transformation for '11' gate choices
        if choice == "11":
            circuit.cx(2, 3)
            circuit.h(2)

        # 2. Extract Statevector before classical measurement breaks the pure state
        sv = Statevector(circuit)

        # 3. Simulate measurement
        shot_string = sv.sample_memory(1)[0]

        # Qiskit formatting: shot_string is little-endian.
        # For a 4-qubit circuit: index 0 is q3, index 1 is q2, index 2 is q1, index 3 is q0.
        a_bit = shot_string[3]   # Qubit 0 (Alice)
        b_bit = shot_string[2]   # Qubit 1 (Bob)
        c1_bit = shot_string[1]  # Qubit 2 (Charlie_B, corresponds to Bob)
        c2_bit = shot_string[0]  # Qubit 3 (Charlie_A, corresponds to Alice)

        alice.measurements.append(a_bit)
        bob.measurements.append(b_bit)
        self.measurements.append([c1_bit, c2_bit])

    def qkd_establish_transport_keys(self, clients, raw_bits=512, qber_limit=0.0968, p_a = 0.0, p_b = 0.0):
        """Run independent QKD sessions for an even roster, two clients per cluster session.

        A four-qubit cluster session produces *two different* final keys: one
        for the first client<->server link and one for the second
        client<->server link.  It never derives one client's key from another.
        """
        clients = tuple(clients)
        if (len(clients) < 2 or len(clients) % 2 or any(not isinstance(client, Client) for client in clients)
                or len({client.ID for client in clients}) != len(clients) or raw_bits <= 0):
            raise ValueError("QKD transport setup requires an even, unique roster of at least two Clients")
        if QuantumCircuit is None:
            raise RuntimeError("Qiskit is required to establish QKD transport keys")
        self.qkd_transport_keys.clear(); self.qkd_pair_sessions.clear()
        self._qkd_transport_send_counters.clear(); self._qkd_transport_seen_counters.clear()
        for offset in range(0, len(clients), 2):
            first, second = clients[offset:offset + 2]
            keys, metadata = self._qkd_establish_pair_transport_keys(first, second, raw_bits, qber_limit, p_a = p_a, p_b = p_b)
            for client_id, key in keys.items():
                self.qkd_transport_keys[client_id] = key
            first.qkd_transport_keys[self.ID] = keys[first.ID]
            second.qkd_transport_keys[self.ID] = keys[second.ID]
            first._qkd_transport_send_counter = second._qkd_transport_send_counter = 0
            first._qkd_transport_seen_counters.clear(); second._qkd_transport_seen_counters.clear()
            self.qkd_pair_sessions[(first.ID, second.ID)] = metadata
        return {client_id: "established" for client_id in self.qkd_transport_keys}

    def _qkd_establish_pair_transport_keys(self, first, second, raw_bits, qber_limit, p_a = 0.0, p_b = 0.0):
        """Execute the original two-client QKD flow for one roster pair."""
        start_gate, start_measurement = len(self.gate_choice), len(self.measurements)
        first_start, second_start = len(first.measurements), len(second.measurements)
        for block_idx in range(raw_bits):
            circuit = self.generate_single_cluster_state()
            circuit, first_choice, second_choice = self.distribute_apply_gate_single(circuit, first, second, p_a = p_a, p_b = p_b)
            circuit, choice = self.server_applies_gate_single(circuit, first_choice, second_choice)
            self.measure_single_block(circuit, first, second, choice, block_idx)
        choices = self.gate_choice[start_gate:]
        measurements = self.measurements[start_measurement:]
        first_bits, second_bits = first.measurements[first_start:], second.measurements[second_start:]
        cases = {"00": [], "01": [], "10": [], "11": []}
        for index, choice in enumerate(choices):
            cases[choice].append(index)
        test_indices, raw_indices = [], []
        for choice, indices in cases.items():
            if choice == "11":
                test_indices.extend(indices)
            else:
                indices = list(indices); random.shuffle(indices)
                midpoint = len(indices) // 2
                test_indices.extend(indices[:midpoint]); raw_indices.extend(indices[midpoint:])
        errors = 0
        for index in test_indices:
            choice = choices[index]
            first_bit, second_bit = int(first_bits[index]), int(second_bits[index])
            c1_bit, c2_bit = int(measurements[index][0]), int(measurements[index][1])
            if choice == "00":
                valid = first_bit == c2_bit and second_bit == c1_bit
            elif choice == "01":
                valid = first_bit == c2_bit and (second_bit == c1_bit if first_bit == 0 else second_bit == (c1_bit ^ 1))
            elif choice == "10":
                valid = second_bit == c1_bit and (first_bit == c2_bit if second_bit == 0 else first_bit == (c2_bit ^ 1))
            else:
                charlie_bits = measurements[index][1] + measurements[index][0]
                valid = (first_bit == second_bit) if charlie_bits in ("01", "10") else (first_bit != second_bit)
            errors += not valid
        qber = errors / len(test_indices) if test_indices else 1.0
        if qber > qber_limit:
            raise RuntimeError(f"QKD abort for {first.ID}/{second.ID}: QBER {qber:.2%} exceeds limit")
        first_raw, server_first_raw, second_raw, server_second_raw = [], [], [], []
        for index in raw_indices:
            if choices[index] in ("00", "01"):
                first_raw.append(int(first_bits[index])); server_first_raw.append(int(measurements[index][1]))
            if choices[index] in ("00", "10"):
                second_raw.append(int(second_bits[index])); server_second_raw.append(int(measurements[index][0]))
        def final_link_key(client_raw, server_raw, label):
            if not client_raw:
                raise RuntimeError(f"QKD abort for {label}: no raw key material")
            corrected, _, leaked = reconcile_keys(client_raw, server_raw, qber, num_passes=4, label=label)
            # This public Toeplitz seed is for this one link only; it does not
            # become transport key material and is not shared with another link.
            toeplitz_seed = secrets.randbelow(2**31)
            client_amplified, _ = privacy_amplification(client_raw, qber, leaked, seed=toeplitz_seed, label=label)
            server_amplified, _ = privacy_amplification(corrected, qber, leaked, seed=toeplitz_seed, label=label)
            if client_amplified is None or server_amplified is None:
                raise RuntimeError(f"QKD abort for {label}: insufficient post-amplification entropy")
            client_key = derive_symmetric_key(client_amplified, label=f"{label} client")
            server_key = derive_symmetric_key(server_amplified, label=f"{label} server")
            if client_key != server_key:
                raise RuntimeError(f"QKD abort for {label}: final key mismatch")
            return client_key
        first_key = final_link_key(first_raw, server_first_raw, f"{first.ID}-{self.ID}")
        second_key = final_link_key(second_raw, server_second_raw, f"{second.ID}-{self.ID}")
        return {first.ID: first_key, second.ID: second_key}, {
            "qber": qber, "raw_bits": raw_bits, "test_blocks": len(test_indices),
            "raw_indices": len(raw_indices), "links": (first.ID, second.ID),
        }

    def qkd_transport_decrypt_from_client(self, envelope):
        """Authenticate/decrypt a client message using only that client's QKD key."""
        required = {"version", "type", "sender_id", "recipient_id", "counter", "inner_type",
                    "inner_round_id", "nonce", "ciphertext"}
        sender = envelope.get("sender_id") if isinstance(envelope, dict) else None
        key = self.qkd_transport_keys.get(sender)
        if (not isinstance(key, bytes) or len(key) != 32 or not isinstance(envelope, dict)
                or set(envelope) != required or envelope["version"] != _SA_VERSION
                or envelope["type"] != "qkd-transport" or envelope["recipient_id"] != self.ID
                or not isinstance(envelope["counter"], int) or not 0 <= envelope["counter"] < 2**88
                or envelope["counter"] in self._qkd_transport_seen_counters.setdefault(sender, set())
                or envelope["nonce"] != b"\x00" + envelope["counter"].to_bytes(11, "big")):
            raise ValueError("invalid or replayed QKD transport envelope")
        aad = _sa_encode([_SA_VERSION, "qkd-transport", sender, self.ID, envelope["counter"],
                          envelope["inner_type"], envelope["inner_round_id"]])
        try:
            message = _sa_decode_message(ChaCha20Poly1305(key).decrypt(envelope["nonce"], envelope["ciphertext"], aad))
        except (InvalidTag, ValueError, TypeError) as exc:
            raise ValueError("QKD transport authentication failed") from exc
        if message.get("type") != envelope["inner_type"] or message.get("round_id") != envelope["inner_round_id"]:
            raise ValueError("QKD transport inner-message binding failed")
        self._qkd_transport_seen_counters[sender].add(envelope["counter"])
        return message

    def qkd_transport_encrypt_for_client(self, recipient_id, message):
        key = self.qkd_transport_keys.get(recipient_id)
        if not isinstance(key, bytes) or len(key) != 32:
            raise ValueError("no QKD-derived transport key is available for this client")
        if not isinstance(message, dict) or not isinstance(message.get("type"), str):
            raise ValueError("a schema-valid protocol message is required")
        counter = self._qkd_transport_send_counters.get(recipient_id, 0)
        if counter >= 2**88:
            raise ValueError("QKD transport nonce space exhausted; establish fresh QKD key material")
        self._qkd_transport_send_counters[recipient_id] = counter + 1
        nonce = b"\x01" + counter.to_bytes(11, "big")  # direction-separated server -> client nonce
        inner_type, inner_round = message["type"], message.get("round_id")
        aad = _sa_encode([_SA_VERSION, "qkd-transport", self.ID, recipient_id, counter, inner_type, inner_round])
        return {"version": _SA_VERSION, "type": "qkd-transport", "sender_id": self.ID,
                "recipient_id": recipient_id, "counter": counter, "inner_type": inner_type,
                "inner_round_id": inner_round, "nonce": nonce,
                "ciphertext": ChaCha20Poly1305(key).encrypt(nonce, _sa_encode(message), aad)}

    # ---- Bonawitz et al. secure aggregation relay/aggregation protocol ----
    def secure_agg_create_round(self, participants, threshold, vector_length, modulus=2**32,
                                round_id=None, signing_public_keys=None, timeouts=None):
        """Create a fresh round. `participants` may be Clients or identity strings."""
        if self.secure_agg_phase not in ("idle", "complete", "aborted"):
            raise ValueError("a secure-aggregation round is already active")
        identities = tuple(client.ID if isinstance(client, Client) else client for client in participants)
        if (not identities or len(set(identities)) != len(identities) or any(not isinstance(i, str) or not i for i in identities)
                or not (2 <= threshold <= len(identities)) or vector_length < 0 or modulus != 2**32):
            raise ValueError("invalid secure-aggregation round parameters")
        if signing_public_keys is None:
            if not all(isinstance(client, Client) for client in participants):
                raise ValueError("trusted signing public keys are required for identity-only rosters")
            signing_public_keys = {client.ID: client.signing_public_key for client in participants}
        if set(signing_public_keys) != set(identities):
            raise ValueError("trusted signing-key registry must match the roster")
        registry = {}
        for identity, public_key in signing_public_keys.items():
            try:
                registry[identity] = public_key if isinstance(public_key, ed25519.Ed25519PublicKey) else ed25519.Ed25519PublicKey.from_public_bytes(public_key)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid trusted signing key") from exc
        self.secure_agg_clear_round_state()
        self.secure_agg_round_id = round_id or secrets.token_hex(16)
        self.secure_agg_phase = "advertise"
        self.secure_agg_threshold = threshold
        self.secure_agg_participants = identities
        self.secure_agg_vector_length = vector_length
        self.secure_agg_modulus = modulus
        self._sa_registry = registry
        self._sa_roster_hash = _sa_hash(list(identities))
        now = time.monotonic()
        supplied = timeouts or {}
        self.secure_agg_deadlines = {phase: now + float(supplied.get(phase, 300))
                                    for phase in ("advertise", "share", "masked-update", "confirmation", "unmask")}
        self._sa_abort_reason = None
        return {"round_id": self.secure_agg_round_id, "participants": identities, "threshold": threshold,
                "modulus": modulus, "vector_length": vector_length,
                "signing_public_keys": {p: _sa_raw_public_key(k) for p, k in registry.items()}}

    def _sa_require_phase(self, phase):
        if self.secure_agg_phase != phase:
            raise ValueError("secure-aggregation message received in the wrong phase")
        if time.monotonic() > self.secure_agg_deadlines.get(phase, float("inf")):
            self.secure_agg_abort_round("phase timeout")
            raise ValueError("secure-aggregation phase timed out")

    def _sa_verify_advertisement(self, message):
        required = {"version", "type", "round_id", "sender_id", "c_pk", "s_pk", "signature"}
        if not isinstance(message, dict) or set(message) != required:
            raise ValueError("malformed advertised-key message")
        sender = message["sender_id"]
        if (message["version"] != _SA_VERSION or message["type"] != "advertise" or message["round_id"] != self.secure_agg_round_id
                or sender not in self.secure_agg_participants or len(message["c_pk"]) != 32 or len(message["s_pk"]) != 32):
            raise ValueError("invalid advertised-key message")
        body = _sa_advertise_body(self.secure_agg_round_id, sender, message["c_pk"], message["s_pk"],
                                  self.secure_agg_participants, self.secure_agg_threshold,
                                  self.secure_agg_modulus, self.secure_agg_vector_length)
        try:
            self._sa_registry[sender].verify(message["signature"], body)
            x25519.X25519PublicKey.from_public_bytes(message["c_pk"])
            x25519.X25519PublicKey.from_public_bytes(message["s_pk"])
        except (InvalidSignature, ValueError, TypeError) as exc:
            raise ValueError("bad advertised-key signature or key") from exc

    def secure_agg_collect_advertised_keys(self, message):
        self._sa_require_phase("advertise")
        try:
            self._sa_verify_advertisement(message)
        except ValueError:
            self.secure_agg_abort_round("invalid advertised key")
            raise
        sender = message["sender_id"]
        if sender in self.secure_agg_advertised_public_keys:
            raise ValueError("replayed or duplicate advertised-key message")
        if any(message["c_pk"] == data["c_pk"] or message["s_pk"] == data["s_pk"]
               for data in self.secure_agg_advertised_public_keys.values()):
            self.secure_agg_abort_round("duplicate ephemeral public key")
            raise ValueError("duplicate ephemeral public key")
        self.secure_agg_advertised_public_keys[sender] = {"c_pk": message["c_pk"], "s_pk": message["s_pk"]}
        self.secure_agg_advertised_signatures[sender] = message["signature"]
        self.secure_agg_replay_protection.add(("advertise", sender))

    def secure_agg_relay_advertised_keys(self):
        self._sa_require_phase("advertise")
        self.U1 = set(self.secure_agg_advertised_public_keys)
        if len(self.U1) < self.secure_agg_threshold:
            self.secure_agg_abort_round("fewer than threshold advertisements")
            raise ValueError("not enough advertised clients")
        ordered = tuple(sorted(self.U1))
        directory = [{"version": _SA_VERSION, "type": "advertise", "round_id": self.secure_agg_round_id,
                      "sender_id": sender, "c_pk": self.secure_agg_advertised_public_keys[sender]["c_pk"],
                      "s_pk": self.secure_agg_advertised_public_keys[sender]["s_pk"],
                      "signature": self.secure_agg_advertised_signatures[sender]} for sender in ordered]
        self.secure_agg_phase = "share"
        return {recipient: list(directory) for recipient in ordered}

    def secure_agg_collect_and_relay_encrypted_shares(self, messages=None, finalize=False):
        self._sa_require_phase("share")
        if messages is not None:
            if isinstance(messages, dict): messages = [messages]
            for message in messages:
                required = {"version", "type", "round_id", "sender_id", "recipient_id", "nonce", "ciphertext"}
                sender = message.get("sender_id") if isinstance(message, dict) else None
                recipient = message.get("recipient_id") if isinstance(message, dict) else None
                token = ("encrypted-share", sender, recipient)
                if (not isinstance(message, dict) or set(message) != required or message.get("version") != _SA_VERSION
                        or message.get("type") != "encrypted-share" or message.get("round_id") != self.secure_agg_round_id
                        or sender not in self.U1 or recipient not in self.U1 or token in self.secure_agg_replay_protection
                        or len(message.get("nonce", b"")) != 12 or not isinstance(message.get("ciphertext"), bytes)):
                    raise ValueError("invalid, nonmember, or replayed encrypted-share message")
                self.secure_agg_relayed_ciphertexts[(sender, recipient)] = message
                self.secure_agg_replay_protection.add(token)
        if not finalize:
            return None
        complete = {sender for sender in self.U1 if all((sender, recipient) in self.secure_agg_relayed_ciphertexts for recipient in self.U1)}
        self.U2 = complete
        if len(self.U2) < self.secure_agg_threshold:
            self.secure_agg_abort_round("fewer than threshold complete share bundles")
            raise ValueError("not enough complete share bundles")
        ordered = tuple(sorted(self.U2))
        self.secure_agg_phase = "masked-update"
        return {recipient: [self.secure_agg_relayed_ciphertexts[(sender, recipient)] for sender in ordered]
                for recipient in ordered}

    def secure_agg_collect_masked_updates(self, message=None, finalize=False):
        self._sa_require_phase("masked-update")
        if message is not None:
            required = {"version", "type", "round_id", "sender_id", "vector"}
            sender = message.get("sender_id") if isinstance(message, dict) else None
            vector = message.get("vector") if isinstance(message, dict) else None
            if (not isinstance(message, dict) or set(message) != required or message.get("version") != _SA_VERSION
                    or message.get("type") != "masked-update" or message.get("round_id") != self.secure_agg_round_id
                    or sender not in self.U2 or sender in self.secure_agg_masked_updates
                    or not isinstance(vector, list) or len(vector) != self.secure_agg_vector_length
                    or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 0 or v >= self.secure_agg_modulus for v in vector)):
                raise ValueError("invalid, replayed, or malformed masked update")
            self.secure_agg_masked_updates[sender] = np.array(vector, dtype=np.uint64)
            self.secure_agg_replay_protection.add(("masked-update", sender))
        if not finalize:
            return None
        self.U3 = set(self.secure_agg_masked_updates)
        if len(self.U3) < self.secure_agg_threshold:
            self.secure_agg_abort_round("fewer than threshold masked updates")
            raise ValueError("not enough masked updates")
        return self.secure_agg_start_consistency_check()

    def secure_agg_start_consistency_check(self):
        self._sa_require_phase("masked-update")
        if len(self.U3) < self.secure_agg_threshold:
            raise ValueError("U3 is below threshold")
        self.secure_agg_phase = "confirmation"
        survivors = tuple(sorted(self.U3))
        message = {"version": _SA_VERSION, "type": "survivor-set", "round_id": self.secure_agg_round_id,
                   "sender_id": self.ID, "survivors": survivors}
        return {client: dict(message) for client in survivors}

    def secure_agg_collect_survivor_confirmations(self, message=None, finalize=False):
        self._sa_require_phase("confirmation")
        if message is not None:
            required = {"version", "type", "round_id", "sender_id", "survivor_hash", "signature"}
            sender = message.get("sender_id") if isinstance(message, dict) else None
            digest = _sa_hash(sorted(self.U3))
            if (not isinstance(message, dict) or set(message) != required or message.get("version") != _SA_VERSION
                    or message.get("type") != "survivor-confirmation" or message.get("round_id") != self.secure_agg_round_id
                    or sender not in self.U3 or sender in self.secure_agg_survivor_confirmations
                    or message.get("survivor_hash") != digest):
                raise ValueError("invalid or replayed survivor confirmation")
            body = _sa_confirmation_body(self.secure_agg_round_id, sender, digest, self.secure_agg_participants,
                                         self.secure_agg_threshold, self.secure_agg_modulus, self.secure_agg_vector_length)
            try:
                self._sa_registry[sender].verify(message["signature"], body)
            except InvalidSignature as exc:
                self.secure_agg_abort_round("bad survivor confirmation signature")
                raise ValueError("bad survivor confirmation signature") from exc
            self.secure_agg_survivor_confirmations[sender] = message
            self.secure_agg_replay_protection.add(("survivor-confirmation", sender))
        if not finalize:
            return None
        self.U4 = set(self.secure_agg_survivor_confirmations)
        if len(self.U4) < self.secure_agg_threshold:
            self.secure_agg_abort_round("fewer than threshold survivor confirmations")
            raise ValueError("not enough survivor confirmations")
        self.secure_agg_phase = "unmask"
        confirmations = [self.secure_agg_survivor_confirmations[p] for p in sorted(self.U4)]
        return {client: list(confirmations) for client in sorted(self.U4)}

    def secure_agg_collect_unmasking_shares(self, message=None, finalize=False):
        self._sa_require_phase("unmask")
        if message is not None:
            required = {"version", "type", "round_id", "sender_id", "shares", "signature"}
            sender = message.get("sender_id") if isinstance(message, dict) else None
            if (not isinstance(message, dict) or set(message) != required or message.get("version") != _SA_VERSION
                    or message.get("type") != "unmasking-response" or message.get("round_id") != self.secure_agg_round_id
                    or sender not in self.U4 or sender in self.secure_agg_unmasking_shares or not isinstance(message.get("shares"), list)):
                raise ValueError("invalid or replayed unmasking response")
            unsigned = {key: message[key] for key in required - {"signature"}}
            try:
                self._sa_registry[sender].verify(message["signature"], _sa_encode(unsigned))
            except InvalidSignature as exc:
                self.secure_agg_abort_round("bad unmasking response signature")
                raise ValueError("bad unmasking response signature") from exc
            expected = {target: ("mask_private" if target in self.U2 - self.U3 else "self_mask") for target in self.U2}
            provided = {}
            for share in message["shares"]:
                if (not isinstance(share, dict) or set(share) != {"target_id", "kind", "x", "share"}
                        or share["target_id"] not in expected or share["target_id"] in provided
                        or share["kind"] != expected[share["target_id"]] or not isinstance(share["x"], int)
                        or not isinstance(share["share"], bytes) or len(share["share"]) != 32):
                    raise ValueError("malformed or forbidden unmasking share")
                provided[share["target_id"]] = share
            if set(provided) != set(expected):
                raise ValueError("unmasking response omitted or added targets")
            self.secure_agg_unmasking_shares[sender] = provided
            self.secure_agg_replay_protection.add(("unmasking-response", sender))
        if finalize:
            self.U5 = set(self.secure_agg_unmasking_shares)
            if len(self.U5) < self.secure_agg_threshold:
                self.secure_agg_abort_round("fewer than threshold unmasking replies")
                raise ValueError("not enough unmasking replies")
            return self.secure_agg_reconstruct_and_aggregate()
        return None

    def secure_agg_reconstruct_and_aggregate(self):
        self._sa_require_phase("unmask")
        if len(self.U5) < self.secure_agg_threshold:
            raise ValueError("not enough unmasking replies")
        # The server starts with only masked vectors; no unmasked input is retained.
        aggregate = np.zeros(self.secure_agg_vector_length, dtype=np.uint64)
        for vector in self.secure_agg_masked_updates.values():
            aggregate = (aggregate + vector) % self.secure_agg_modulus
        context = _sa_encode([self.secure_agg_round_id, sorted(self.U2)])
        responders = sorted(self.U5)
        for target in sorted(self.U3):
            shares = [(self.secure_agg_unmasking_shares[responder][target]["x"],
                       self.secure_agg_unmasking_shares[responder][target]["share"])
                      for responder in responders]
            seed = _sa_shamir_reconstruct(shares, self.secure_agg_threshold)
            aggregate = (aggregate - _sa_prg_vector(seed, self.secure_agg_vector_length, self.secure_agg_modulus,
                                                     "self", context)) % self.secure_agg_modulus
        # Only private mask keys for U2-U3 are reconstructed.  Private keys for U3 and
        # self-mask seeds for dropped clients are never requested or reconstructed.
        for dropped in sorted(self.U2 - self.U3):
            shares = [(self.secure_agg_unmasking_shares[responder][dropped]["x"],
                       self.secure_agg_unmasking_shares[responder][dropped]["share"])
                      for responder in responders]
            raw_private = _sa_shamir_reconstruct(shares, self.secure_agg_threshold)
            private_key = x25519.X25519PrivateKey.from_private_bytes(raw_private)
            for survivor in sorted(self.U3):
                peer_key = x25519.X25519PublicKey.from_public_bytes(self.secure_agg_advertised_public_keys[survivor]["s_pk"])
                left, right = sorted((dropped, survivor))
                seed = _sa_hkdf(private_key.exchange(peer_key), self.secure_agg_round_id,
                                _sa_encode(["secure-agg/pairwise-mask", left, right]))
                mask = _sa_prg_vector(seed, self.secure_agg_vector_length, self.secure_agg_modulus,
                                      "pairwise", context + _sa_encode([left, right]))
                # Survivor contribution is -mask if survivor < dropped, otherwise +mask.
                aggregate = ((aggregate + mask) if survivor < dropped else (aggregate - mask)) % self.secure_agg_modulus
        self.secure_agg_phase = "complete"
        return aggregate.astype(np.uint64).tolist()

    def secure_agg_abort_round(self, reason="aborted"):
        self._sa_abort_reason = reason
        self.secure_agg_phase = "aborted"
        return None

    def secure_agg_clear_round_state(self):
        self.secure_agg_round_id = None
        self.secure_agg_phase = "idle"
        self.secure_agg_threshold = None; self.secure_agg_participants = ()
        self.U1.clear(); self.U2.clear(); self.U3.clear(); self.U4.clear(); self.U5.clear()
        self.secure_agg_advertised_public_keys.clear(); self.secure_agg_advertised_signatures.clear()
        self.secure_agg_relayed_ciphertexts.clear(); self.secure_agg_masked_updates.clear()
        self.secure_agg_survivor_confirmations.clear(); self.secure_agg_unmasking_shares.clear()
        self.secure_agg_modulus = self.secure_agg_vector_length = None
        self.secure_agg_deadlines.clear(); self.secure_agg_replay_protection.clear(); self._sa_registry = {}
        self._sa_roster_hash = None

def calculate_qber_and_extract_raw_key(server, alice, bob):
    print("\n========== INITIATING EAVESDROPPING CHECK (QBER) ==========")
    case_indices = {"00": [], "01": [], "10": [], "11": []}
    for idx, choice in enumerate(server.gate_choice):
        case_indices[choice].append(idx)

    test_indices = []
    raw_key_indices = []

    for choice, indices in case_indices.items():
        if choice == "11":
            test_indices.extend(indices)
        else:
            random.shuffle(indices)
            half_point = len(indices) // 2
            test_indices.extend(indices[:half_point])
            raw_key_indices.extend(indices[half_point:])

    errors = 0
    total_tested = len(test_indices)

    for i in test_indices:
        choice = server.gate_choice[i]
        a_bit = int(alice.measurements[i])
        b_bit = int(bob.measurements[i])
        c1_bit = int(server.measurements[i][0])
        c2_bit = int(server.measurements[i][1])

        is_match = True

        if choice == "00":
            if a_bit != c2_bit or b_bit != c1_bit:
                is_match = False
        elif choice == "01":
            if a_bit != c2_bit:
                is_match = False
            elif a_bit == 0 and b_bit != c1_bit:
                is_match = False
            elif a_bit == 1 and b_bit != (c1_bit ^ 1):
                is_match = False
        elif choice == "10":
            if b_bit != c1_bit:
                is_match = False
            elif b_bit == 0 and a_bit != c2_bit:
                is_match = False
            elif b_bit == 1 and a_bit != (c2_bit ^ 1):
                is_match = False
        elif choice == "11":
            charlie_str = server.measurements[i][1] + server.measurements[i][0]
            if charlie_str in ["01", "10"]:
                if a_bit != b_bit:
                    is_match = False
            elif charlie_str in ["00", "11"]:
                if a_bit == b_bit:
                    is_match = False

        if not is_match:
            errors += 1

    qber = errors / total_tested if total_tested > 0 else 0

    print(f"\n========== QBER SUMMARY ==========")
    print(f"Total blocks generated: {len(server.gate_choice)}")
    print(f"Blocks tested (cases 00/01/10: 50% split; case 11: 100% disclosed): {total_tested}")
    print(f"Errors detected: {errors}")
    print(f"Calculated QBER: {qber:.2%}")
    print(f"Remaining blocks kept for Raw Key: {len(raw_key_indices)}")

    return qber, raw_key_indices


# ==========================================
# STEP 1 - ERROR CORRECTION (RECONCILIATION)
# ==========================================
# IMPORTANT TOPOLOGY NOTE:
# This is a trusted-node / hub setup. Charlie (the server) shares ONE key
# with Alice and a SEPARATE, independent key with Bob. Alice and Bob never
# share a key directly with each other -- there is no physical
# correlation between Alice's bit and Bob's bit in this protocol; the
# only guaranteed correlations are (Alice's bit <-> Charlie's c2 bit) and
# (Bob's bit <-> Charlie's c1 bit). So reconciliation must run TWICE,
# independently, once per link:
#   Link 1: Alice  <-> Charlie   (using Alice's bits + Charlie's c2 bits)
#   Link 2: Bob    <-> Charlie   (using Bob's bits   + Charlie's c1 bits)

def bit_list_to_str(bits):
    return ''.join(str(int(b)) for b in bits)

def compute_parity(bits):
    """XOR parity of a list of 0/1 bits."""
    parity = 0
    for b in bits:
        parity ^= int(b)
    return parity

def binary_search_fix_block(a_block, b_block, leaked_bits_counter):
    """
    Finds and corrects ONE erroneous bit in a block known to have odd
    parity mismatch (standard Cascade binary search). Returns
    (corrected_b_block, flipped_local_index_or_None).
    """
    a_block = list(a_block)
    b_block = list(b_block)
    n = len(a_block)

    if n == 1:
        if a_block[0] != b_block[0]:
            return [a_block[0]], 0
        return b_block, None

    mid = n // 2
    a_left, a_right = a_block[:mid], a_block[mid:]
    b_left, b_right = b_block[:mid], b_block[mid:]

    leaked_bits_counter[0] += 1
    if compute_parity(a_left) != compute_parity(b_left):
        b_left, flipped = binary_search_fix_block(a_left, b_left, leaked_bits_counter)
        return b_left + b_right, flipped
    else:
        # Right half's parity is derived (block parity XOR left parity),
        # not disclosed -- no extra leak needed.
        b_right, flipped = binary_search_fix_block(a_right, b_right, leaked_bits_counter)
        if flipped is not None:
            flipped += mid
        return b_left + b_right, flipped

def reconcile_keys(reference_key, other_key, qber, num_passes=4, min_block_size=2, label=""):
    """
    Multi-pass Cascade reconciliation for ONE link. Corrects a working
    copy of other_key to match reference_key, cascading corrections back
    through previously-checked blocks whenever a bit flips.
    """
    n = len(reference_key)
    leaked_bits_counter = [0]
    total_corrections = 0
    tag = f"[{label}] " if label else ""

    print(f"\n========== ERROR CORRECTION (MULTI-PASS CASCADE) {tag}==========")
    print(f"{tag}Reconciling {n} bits over {num_passes} passes...")

    if n == 0:
        return [], 0, 0

    effective_qber = qber if qber > 0.001 else 0.01
    k1 = max(min_block_size, int(round(0.73 / effective_qber)))
    k1 = min(k1, n)

    working_other = list(other_key)

    bit_to_blocks = {i: [] for i in range(n)}
    pass_blocks = []

    for p in range(num_passes):
        block_size = min(k1 * (2 ** p), n)
        if block_size < 1:
            block_size = n

        perm = list(range(n))
        if p > 0:
            random.shuffle(perm)

        blocks = []
        for start in range(0, n, block_size):
            block_positions = perm[start:start + block_size]
            block_id = len(blocks)
            blocks.append(block_positions)
            for orig_idx in block_positions:
                bit_to_blocks[orig_idx].append((p, block_id))
        pass_blocks.append(blocks)

        queue = deque((p, bid) for bid in range(len(blocks)))

        while queue:
            pass_idx, block_id = queue.popleft()
            block_positions = pass_blocks[pass_idx][block_id]

            ref_vals = [reference_key[i] for i in block_positions]
            oth_vals = [working_other[i] for i in block_positions]

            leaked_bits_counter[0] += 1
            if compute_parity(ref_vals) != compute_parity(oth_vals):
                corrected_vals, flipped_local = binary_search_fix_block(
                    ref_vals, oth_vals, leaked_bits_counter
                )

                if flipped_local is not None:
                    orig_idx = block_positions[flipped_local]
                    if working_other[orig_idx] != reference_key[orig_idx]:
                        working_other[orig_idx] = reference_key[orig_idx]
                        total_corrections += 1

                        for (pp, bb) in bit_to_blocks[orig_idx]:
                            if (pp, bb) != (pass_idx, block_id):
                                queue.append((pp, bb))

    total_leaked = leaked_bits_counter[0]
    still_mismatched = sum(1 for i in range(n) if reference_key[i] != working_other[i])

    print(f"{tag}Passes run: {num_passes} (initial block size k1={k1})")
    print(f"{tag}Total bit corrections applied: {total_corrections}")
    print(f"{tag}Parity bits disclosed (leaked to public channel): {total_leaked}")
    print(f"{tag}Reference key (first 32 bits): {bit_list_to_str(reference_key[:32])}")
    print(f"{tag}Other key before correction (first 32 bits): {bit_list_to_str(other_key[:32])}")
    print(f"{tag}Other key after correction  (first 32 bits): {bit_list_to_str(working_other[:32])}")
    print(f"{tag}Remaining mismatches after all passes: {still_mismatched}")
    print(f"{tag}Reference key == corrected other key: {reference_key == working_other}")

    return working_other, total_corrections, total_leaked


# ==========================================
# STEP 2 - PRIVACY AMPLIFICATION
# ==========================================

def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def generate_toeplitz_matrix(seed, n_in, n_out):
    """Builds an n_out x n_in binary Toeplitz matrix from a shared seed."""
    rng = np.random.RandomState(seed)
    first_col = rng.randint(0, 2, size=n_out)
    first_row = rng.randint(0, 2, size=n_in)
    first_row[0] = first_col[0]

    T = np.zeros((n_out, n_in), dtype=np.uint8)
    for i in range(n_out):
        for j in range(n_in):
            T[i, j] = first_col[i - j] if i - j >= 0 else first_row[j - i]
    return T

def privacy_amplification(key_bits, qber, leaked_bits, safety_bits=32, seed=None, label=""):
    """
    Shrinks key_bits to remove estimated eavesdropper information for
    ONE link. Returns (amplified_bits, seed_used). amplified_bits is
    None if there is not enough surplus entropy left to produce any
    secure output.
    """
    tag = f"[{label}] " if label else ""
    n_in = len(key_bits)
    eve_info_estimate = n_in * binary_entropy(qber)
    n_out = int(n_in - leaked_bits - eve_info_estimate - safety_bits)
    n_out = max(n_out, 0)

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    print(f"\n========== PRIVACY AMPLIFICATION {tag}==========")
    print(f"{tag}Input key length: {n_in} bits")
    print(f"{tag}Estimated eavesdropper information from QBER: {eve_info_estimate:.2f} bits")
    print(f"{tag}Bits leaked during reconciliation: {leaked_bits}")
    print(f"{tag}Safety margin: {safety_bits} bits")
    print(f"{tag}Output key length after compression: {n_out} bits")
    print(f"{tag}Public Toeplitz seed (shared ONLY over this link's classical channel): {seed}")

    if n_out <= 0:
        print(f"{tag}[ALERT] Not enough entropy left to produce a secure key for this link. "
              "Generate a longer raw key (increase NO_OF_RAW_BITS) and retry.")
        return None, seed

    T = generate_toeplitz_matrix(seed, n_in, n_out)
    key_vec = np.array([int(b) for b in key_bits], dtype=np.uint8)
    amplified = (T @ key_vec) % 2

    print(f"{tag}Privacy-amplified key (first 32 bits): {bit_list_to_str(amplified[:32])}")

    return amplified, seed


# ==========================================
# STEP 3 - FINAL SYMMETRIC KEY (HASHING)
# ==========================================

def derive_symmetric_key(amplified_bits, key_length_bytes=32, label=""):
    bit_str = bit_list_to_str(amplified_bits)
    byte_len = (len(bit_str) + 7) // 8
    padded = bit_str.ljust(byte_len * 8, '0')
    raw_bytes = int(padded, 2).to_bytes(byte_len, byteorder='big') if byte_len > 0 else b''

    digest = hashlib.sha256(raw_bytes).digest()
    final_key = digest[:key_length_bytes]

    print(f"\n========== FINAL SYMMETRIC KEY ({label}) ==========")
    print(f"Final key (hex, {key_length_bytes * 8}-bit nominal): {final_key.hex()}")

    return final_key


# ==========================================
# EVEN-N ORCHESTRATION HELPERS
# ==========================================

def make_client_ids(n, names=None):
    """Return n unique client identity strings. n must be even and >= 2."""
    if names is not None:
        ids = tuple(names)
        if len(ids) != n:
            raise ValueError("names length must equal n")
    else:
        width = max(2, len(str(max(n - 1, 0))))
        ids = tuple(f"Client_{i:0{width}d}" for i in range(n))
    if n < 2 or n % 2 != 0:
        raise ValueError("client count must be an even integer >= 2 (4-qubit cluster packs 2 links per session)")
    if len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
        raise ValueError("client identities must be unique non-empty strings")
    return ids


def bootstrap_qkd(n, raw_bits=128, qber_limit=0.0968, client_ids=None, server_id="Charlie", p_a=0.0, p_b=0.0):
    """Create server + even roster and establish independent QKD transport keys.

    Returns (server, clients) where each client has a 32-byte key shared only
    with the server (never with another client).
    """
    ids = make_client_ids(n, names=client_ids)
    if QuantumCircuit is None:
        raise RuntimeError("Qiskit is required for QKD transport setup")
    server = Server(server_id)
    clients = [Client(cid) for cid in ids]
    established = server.qkd_establish_transport_keys(clients, raw_bits=raw_bits, qber_limit=qber_limit, p_a=p_a, p_b=p_b)
    for client in clients:
        key = client.qkd_transport_keys.get(server.ID)
        if not isinstance(key, bytes) or len(key) != 32:
            raise RuntimeError(f"missing 32-byte QKD key for {client.ID}")
        if server.qkd_transport_keys.get(client.ID) != key:
            raise RuntimeError(f"server/client QKD key mismatch for {client.ID}")
    print(f"QKD transport keys established for {len(established)} clients: {', '.join(established)}")
    return server, clients


def expected_sum(vectors, survivor_ids, modulus=2**32):
    """Element-wise sum of survivor vectors mod R."""
    survivor_ids = list(survivor_ids)
    if not survivor_ids:
        raise ValueError("empty survivor set")
    length = len(next(iter(vectors.values())))
    total = [0] * length
    for sid in survivor_ids:
        vec = vectors[sid]
        if len(vec) != length:
            raise ValueError("vector length mismatch")
        for i, v in enumerate(vec):
            total[i] = (total[i] + int(v)) % modulus
    return total


def run_secure_aggregation(server, clients, vectors, threshold, *,
                           vector_length=None, modulus=2**32,
                           drop_after_shares=None, round_id=None):
    """Run one Bonawitz-style secure aggregation round for an arbitrary roster.

    drop_after_shares: optional set of client IDs that complete share exchange
    but do not submit masked updates (simulated dropout before U3).
    """
    clients_by_id = {c.ID: c for c in clients}
    roster = tuple(c.ID for c in clients)
    if set(vectors) != set(roster):
        raise ValueError("vectors must provide exactly one entry per client")
    if vector_length is None:
        vector_length = len(next(iter(vectors.values())))
    for cid, vec in vectors.items():
        if len(vec) != vector_length:
            raise ValueError(f"vector length mismatch for {cid}")
    drop_after_shares = set(drop_after_shares or ())
    if not drop_after_shares.issubset(roster):
        raise ValueError("drop_after_shares contains unknown client ids")

    params = server.secure_agg_create_round(
        clients, threshold, vector_length, modulus=modulus, round_id=round_id
    )
    signing_keys = params["signing_public_keys"]
    round_id = params["round_id"]

    # Round 0 - AdvertiseKeys
    for client in clients:
        client.secure_agg_start_round(
            round_id, roster, threshold, vector_length,
            modulus=modulus, signing_public_keys=signing_keys,
        )
        server.secure_agg_collect_advertised_keys(client.secure_agg_advertise_keys())
    directory_by_recipient = server.secure_agg_relay_advertised_keys()
    for client in clients:
        client.secure_agg_verify_advertised_keys(directory_by_recipient[client.ID])

    # Round 1 - ShareKeys
    share_messages = []
    for client in clients:
        share_messages.extend(client.secure_agg_create_and_encrypt_shares())
    relayed = server.secure_agg_collect_and_relay_encrypted_shares(share_messages, finalize=True)
    for client in clients:
        client.secure_agg_receive_and_decrypt_shares(relayed[client.ID], u2=sorted(relayed))

    # Round 2 - MaskedInputCollection (optional dropouts)
    survivors = [c for c in clients if c.ID not in drop_after_shares]
    if len(survivors) < threshold:
        raise ValueError("too many dropouts: survivors below threshold")
    for client in survivors:
        msg = client.secure_agg_create_masked_update(vectors[client.ID])
        server.secure_agg_collect_masked_updates(msg)
    survivor_set_msgs = server.secure_agg_collect_masked_updates(finalize=True)

    # Round 3 - ConsistencyCheck (survivor confirmations)
    for client in survivors:
        conf = client.secure_agg_confirm_survivor_set(survivor_set_msgs[client.ID])
        server.secure_agg_collect_survivor_confirmations(conf)
    conf_relay = server.secure_agg_collect_survivor_confirmations(finalize=True)
    for client in survivors:
        client.secure_agg_verify_survivor_confirmations(conf_relay[client.ID])

    # Round 4 - Unmasking
    for client in survivors:
        server.secure_agg_collect_unmasking_shares(client.secure_agg_create_unmasking_response())
    aggregate = server.secure_agg_collect_unmasking_shares(finalize=True)
    for client in survivors:
        client.secure_agg_finalize_and_clear_state()
    return aggregate
