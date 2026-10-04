# QKD-Assisted Secure Aggregation with Client-Level Differential Privacy for Federated Learning

Reference implementation and interactive demo of the three-phase protocol from the paper
*"QKD-Assisted Secure Aggregation with Client-Level Differential Privacy for Federated Learning"* (Ansh Goel).

| Phase | What it does | Where |
|---|---|---|
| **I. QKD transport keys** | Three-party lightweight QKD on four-qubit cluster states (Younes et al.). Clients need only a Hadamard gate and Z-basis measurement. One session per client pair gives two independent 256-bit client–server keys (sifting, QBER test at 9.68%, Cascade, Toeplitz privacy amplification, ℓ ≥ 256). | `qkd_secagg.py`, `qkd_layer.py` |
| **II. Sanitisation** | Clip ‖w_u‖ ≤ C, add N(0, σ²C²), quantise with scale Δ into Z_R (R = 2³²). Client-level Rényi-DP accounting with σ_eff = √\|H\|·σ. | `sanitize.py` |
| **III. Secure aggregation** | Bonawitz et al. Rounds 0–4 (X25519, Ed25519, Shamir t-of-n, pairwise + self masks, consistency check, unmasking), with **every client–server message in a ChaCha20-Poly1305 envelope under the Phase I key**. | `qkd_secagg.py`, `secagg_transport.py` |

`federated.py` ties them together in a federated learning loop on a toy, non-IID 2-D dataset with logistic
regression, alongside a non-private FedAvg baseline.

## Run the demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

1. **Phase I:** choose the number of clients (even), the number of cluster states per pair, and optionally an eavesdropper of strength *p* on selected pairs, then distribute keys. Tapped pairs above p* ≈ 0.171 abort.
2. **Training:** set the threshold *t*, C, σ, Δ, dropout rate and rounds, then train. You can watch accuracy, the privacy budget ε, survivors, and the decision boundary update round by round.
3. **What the server sees:** compare the client's sanitised update with the uniformly random masked vector the server receives, see the aggregate it recovers, and see how much traffic was encrypted under QKD keys.

## Tests

```bash
python -m pytest -q
```

## Scope and limitations (as stated in the paper)

* The quantum layer is a Qiskit statevector simulation; the eavesdropper is a phase-flip (dephasing) channel, not a general attack.
* QKD protects client–server transport only. Peer masking keys, share encryption and signatures remain classical, so the protocol is **QKD-assisted, not fully quantum-secure**.
* Final key length uses the operational estimate ℓ = ℓ_raw − ℓ_leak − ℓ_raw·h(Q) − s_PA, not a composable finite-key bound.
* Honest-but-curious server; privacy accounting assumes all survivors are honest (|H| = |U3|) and claims no subsampling amplification.

## Files

```
qkd_secagg.py        reference protocol implementation (Phase I and III primitives), unchanged
qkd_layer.py         Phase I orchestration: per-pair sessions, eavesdropper, l >= 256 rule
sanitize.py          Phase II: clip / noise / quantise, Eq. (5) wrap-around check, RDP accountant
secagg_transport.py  Phase III Rounds 0-4 carried over the QKD transport
federated.py         toy data, logistic regression, end-to-end training loop
app.py               Streamlit interface
tests/               end-to-end tests
```
