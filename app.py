"""Interactive demo: QKD-Assisted Secure Aggregation with Client-Level DP for FL.

Run:  streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import federated as F
import qkd_layer as Q
import qkd_secagg as M
import sanitize as S
from secagg_transport import ROUND_NAMES

st.set_page_config(page_title="QKD-SecAgg-DP Federated Learning", page_icon="🔐", layout="wide")

BLUE, ORANGE, GREEN, RED, GREY = "#2563eb", "#ea580c", "#16a34a", "#dc2626", "#94a3b8"
ss = st.session_state

# ------------------------------------------------------------------ header
st.title("🔐 QKD-Assisted Secure Aggregation with Client-Level Differential Privacy")
st.caption("Live demo of the three-phase protocol, running the real implementation: Qiskit cluster-state QKD, "
           "X25519 / Ed25519 / ChaCha20-Poly1305, Shamir sharing, Bonawitz et al. secure aggregation, "
           "and a toy federated learning task.  QKD secures the client–server transport keys; "
           "peer masking, share encryption and signatures remain classical.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("① Network & quantum layer")
    n = st.select_slider("Number of clients n (even: 2 per cluster-state session)",
                         options=list(range(2, 17, 2)), value=6)
    raw_bits = st.slider("Cluster states per pair  N_raw", 1100, 4000, 1600, 100,
                         help="N_raw = 4ℓ(1+γ). Too few states cannot reach ℓ ≥ 256 and the session aborts.")
    eve_p = st.slider("Eavesdropper strength p (dephasing)", 0.0, 0.5, 0.0, 0.01,
                      help=f"Mean QBER = (3p − p²)/5; it crosses the 9.68% threshold at p* = {Q.p_star():.3f}.")
    pairs = [f"Pair {k + 1}: Client_{2 * k:02d} & Client_{2 * k + 1:02d}" for k in range(n // 2)]
    tapped = st.multiselect("Pairs Eve taps", pairs, default=[], disabled=eve_p == 0)
    run_qkd = st.button("Run Phase I — distribute keys", type="primary", width="stretch")

    st.header("②③ Federated training")
    rounds = st.slider("Rounds T", 1, 50, 25)
    dropout = st.slider("Per-round dropout probability", 0.0, 0.6, 0.1, 0.05,
                        help="Each client independently fails to send its masked update (after Round 1).")
    C = st.slider("Clipping bound C", 0.05, 2.0, 0.2, 0.05, help="Set near the typical update norm (~0.1–0.2 here); larger C means more noise per coordinate.")
    sigma = st.slider("Noise multiplier σ", 0.0, 3.0, 0.3, 0.05)
    log_delta = st.slider("Fixed-point scale Δ = 2^k", 8, 24, 16)
    dp_delta = st.select_slider("DP δ", options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], value=1e-5)
    with st.expander("Local training"):
        lr = st.slider("Learning rate", 0.01, 1.0, 0.2, 0.01)
        epochs = st.slider("Local epochs", 1, 10, 1)
        seed = st.number_input("Seed (data, dropouts, noise)", 0, 10_000, 1)
        speed = st.slider("Animation delay per round (s)", 0.0, 1.0, 0.25, 0.05)

# ------------------------------------------------------------------ Phase I
if run_qkd or ss.get("n") != n:
    ss.n = n
    ids = M.make_client_ids(n)
    ss.server = M.Server("Charlie")
    ss.clients = [M.Client(i) for i in ids]
    ss.qkd = None
    ss.records = []
if run_qkd:
    bar = st.progress(0.0, text="Preparing cluster states…")
    attacked = {pairs.index(p) for p in tapped} if eve_p > 0 else set()

    def _progress(k, total, res):
        bar.progress(k / total, text=f"Session {k}/{total}: {res.pair[0]} & {res.pair[1]} → {res.status}")

    ss.qkd = Q.bootstrap(ss.server, ss.clients, raw_bits, eve_p=eve_p, attacked_pairs=attacked, progress=_progress)
    ss.qkd_params = dict(raw_bits=raw_bits, p=eve_p)
    ss.records = []
    bar.empty()

keyed = [c for c in ss.clients if c.ID in ss.server.qkd_transport_keys] if ss.get("qkd") else []

NAV = ["① Quantum key distribution", "②③ Private federated training", "🔍 What the server sees", "ℹ️ About"]
if run_qkd or "nav" not in ss or ss.nav is None:
    ss.nav = NAV[0]
nav = st.segmented_control("Section", NAV, key="nav", label_visibility="collapsed") or NAV[0]
st.divider()

# ================================================================== TAB 1
if nav == NAV[0]:
    if not ss.get("qkd"):
        st.info("Choose the network in the sidebar and press **Run Phase I — distribute keys**. "
                "The server prepares four-qubit cluster states |C₄⟩ = ½(|0000⟩+|0011⟩+|1100⟩−|1111⟩); "
                "each client only applies a Hadamard (or not) and measures in Z.")
    else:
        res = ss.qkd
        ok = [r for r in res if r.status == "established"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions established", f"{len(ok)} / {len(res)}")
        c2.metric("Clients with a 256-bit key", f"{len(keyed)} / {len(ss.clients)}")
        c3.metric("Abort threshold", f"{Q.QBER_LIMIT:.2%}")
        c4.metric("Analytic QBER at p", f"{Q.analytic_qber(ss.qkd_params['p']):.2%}",
                  help="(3p − p²)/5 for the tapped pairs")

        rows = []
        for r in res:
            for l in r.links:
                rows.append({"Session": f"{r.pair[0]} & {r.pair[1]}", "Link": f"{l.client_id} ↔ Charlie",
                             "Eve p": r.p, "QBER": None if r.qber is None else round(r.qber * 100, 2),
                             "ℓ_raw": l.raw_len, "ℓ_leak (Cascade)": l.leaked, "ℓ final": l.final_len,
                             "Key fingerprint": l.key_fingerprint or "—",
                             "Status": "✅ key established" if r.status == "established" else "❌ " + r.reason,
                             "Time (s)": round(r.seconds, 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"QBER": st.column_config.NumberColumn("QBER (%)")})

        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            names = [f"{r.pair[0][-2:]}&{r.pair[1][-2:]}" for r in res]
            fig.add_bar(x=names, y=[(r.qber or 0) * 100 for r in res],
                        marker_color=[RED if r.status != "established" else (ORANGE if r.attacked else BLUE) for r in res],
                        text=[f"{(r.qber or 0):.1%}" for r in res], textposition="outside")
            fig.add_hline(y=Q.QBER_LIMIT * 100, line_dash="dash", line_color=RED,
                          annotation_text="abort threshold 9.68%")
            fig.update_layout(title="Measured test QBER per session", yaxis_title="QBER (%)",
                              xaxis_title="client pair", height=360, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")
        with right:
            ps = np.linspace(0, 0.5, 101)
            fig = go.Figure()
            fig.add_scatter(x=ps, y=[Q.analytic_qber(p) * 100 for p in ps], name="analytic (3p−p²)/5",
                            line=dict(color=GREY))
            fig.add_scatter(x=[r.p for r in res], y=[(r.qber or 0) * 100 for r in res], mode="markers",
                            name="this run", marker=dict(size=11, color=BLUE))
            fig.add_hline(y=Q.QBER_LIMIT * 100, line_dash="dash", line_color=RED)
            fig.add_vline(x=Q.p_star(), line_dash="dot", annotation_text=f"p* = {Q.p_star():.3f}")
            fig.update_layout(title="QBER vs attack strength", xaxis_title="p", yaxis_title="QBER (%)",
                              height=360, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")

        with st.expander("How a session works, and one cluster-state block"):
            st.markdown(
                "1. **Distribution** – Charlie prepares N_raw cluster states, sends qubit 1 to the first client and "
                "qubit 2 to the second, keeps qubits 3–4.\n"
                "2. **Client operations** – each client flips a coin per qubit: measure in Z, or apply H then measure.\n"
                "3. **Server operations** – Charlie mirrors the announced choices (Bell-basis measurement when both "
                "chose H), giving cases 00, 01, 10, 11.\n"
                "4. **Sifting & eavesdropper test** – half of cases 00/01/10 and all of case 11 are disclosed; "
                "abort if QBER > 9.68%.\n"
                "5. **Cascade reconciliation** per link (leaked parity bits counted).\n"
                "6. **Toeplitz privacy amplification** to ℓ = ℓ_raw − ℓ_leak − ℓ_raw·h(Q) − 32; abort unless ℓ ≥ 256.\n"
                "7. **Key derivation** – SHA-256 to the 256-bit transport key k_u.")
            counts = res[0].case_counts
            st.write("Sifting cases in session 1:", counts)
            try:
                st.code(str(ss.server.generate_single_cluster_state().decompose().draw("text")), language=None)
            except Exception:
                pass

# ================================================================== TAB 2
if nav == NAV[1]:
    if not ss.get("qkd"):
        st.info("Run Phase I first: every message in Phase III travels over the QKD-keyed channel.")
    elif len(keyed) < 2:
        st.error("Fewer than two clients obtained a QKD key, so aggregation cannot run. Lower p or untap pairs.")
    else:
        excluded = sorted(set(c.ID for c in ss.clients) - set(c.ID for c in keyed))
        if excluded:
            st.warning(f"No transport key (session aborted): {', '.join(excluded)}. "
                       "They are excluded from training until a new session succeeds.")
        k = len(keyed)
        t = st.slider("Shamir threshold t (tolerates up to n − t dropouts per round)", 2, k,
                      min(k, max(2, k // 2 + 1)))
        ok_wrap, lhs, rhs, tau = S.wraparound_ok(k, 3, C, sigma, 2 ** log_delta)
        cols = st.columns(4)
        cols[0].metric("Clients training", k)
        cols[1].metric("Model size m", 3, help="2-feature logistic regression: w₁, w₂, bias")
        cols[2].metric("σ_eff (all survive)", f"{np.sqrt(k) * sigma:.2f}", help="√|H|·σ")
        cols[3].metric("No-wraparound (Eq. 5)", "✅ holds" if ok_wrap else "❌ violated",
                       help=f"n(Δ(C+τ)+½) = {lhs:.3g}  <  R/2 = {rhs:.3g}  (δ_wrap = 1e-9)")
        go_train = st.button("▶ Train", type="primary")

        m1, m2, m3, m4 = st.columns(4)
        ph_acc, ph_eps, ph_surv, ph_ok = m1.empty(), m2.empty(), m3.empty(), m4.empty()
        cl, cr = st.columns(2)
        ph_curve, ph_boundary = cl.empty(), cr.empty()
        ph_log = st.empty()
        data = F.make_toy_data([c.ID for c in ss.clients], seed=int(seed))

        def draw(records):
            last = records[-1]
            ph_acc.metric("Test accuracy", f"{last.acc:.1%}", f"{last.acc - last.base_acc:+.1%} vs non-private")
            ph_eps.metric(f"Privacy spent ε (δ={dp_delta:g})", "∞ (σ = 0)" if np.isinf(last.eps) else f"{last.eps:.2f}")
            ph_surv.metric("Survivors this round", f"{len(last.U3)} / {k}" if last.ok else "round aborted")
            ph_ok.metric("Aggregate = Σ sanitised updates", "✅ exact" if last.agg_error is not None
                         and last.agg_error == 0 else ("—" if last.agg_error is None else f"err {last.agg_error:.1e}"))
            rs = [r.r for r in records]
            fig = go.Figure()
            fig.add_scatter(x=rs, y=[r.acc for r in records], name="QKD-SecAgg + DP (this protocol)",
                            line=dict(color=BLUE, width=3))
            fig.add_scatter(x=rs, y=[r.base_acc for r in records], name="plain FedAvg (no privacy)",
                            line=dict(color=GREY, dash="dash"))
            fig.update_layout(title="Global model accuracy", xaxis_title="round", yaxis_title="test accuracy",
                              yaxis_range=[0, 1], xaxis_range=[0.5, rounds + 0.5], height=380,
                              legend=dict(orientation="h", y=-0.25), margin=dict(t=50, b=40))
            ph_curve.plotly_chart(fig, width="stretch", key=f"curve{len(records)}")

            Xt, yt = data.X_test[:400], data.y_test[:400]
            fig = go.Figure()
            for cls, col in ((0, BLUE), (1, ORANGE)):
                sel = yt == cls
                fig.add_scatter(x=Xt[sel, 0], y=Xt[sel, 1], mode="markers", name=f"class {cls}",
                                marker=dict(color=col, size=5, opacity=0.55))
            w = last.w
            xs = np.array([0.0, 7.0])
            if abs(w[1]) > 1e-9:
                fig.add_scatter(x=xs, y=-(w[0] * xs + w[2]) / w[1], mode="lines", name="decision boundary",
                                line=dict(color="black", width=3))
            fig.update_layout(title=f"Decision boundary after round {last.r}", xaxis_range=[0, 7],
                              yaxis_range=[0, 7], height=380, legend=dict(orientation="h", y=-0.25),
                              margin=dict(t=50, b=40))
            ph_boundary.plotly_chart(fig, width="stretch", key=f"bound{len(records)}")

            ph_log.dataframe(pd.DataFrame([{
                "Round": r.r, "Survivors |U3|": len(r.U3), "Dropped": ", ".join(d[-2:] for d in r.dropped) or "—",
                "Status": "✅" if r.ok else "❌ " + r.abort_reason, "σ_eff": None if r.sigma_eff is None else round(r.sigma_eff, 3),
                "ε so far": None if np.isinf(r.eps) else round(r.eps, 2), "Accuracy": round(r.acc, 3),
                "QKD envelopes": sum(r.trace.envelopes.values()),
                "Ciphertext (KB)": round(sum(r.trace.ciphertext_bytes.values()) / 1024, 1),
            } for r in reversed(records)]), hide_index=True, width="stretch", height=260)

        if go_train:
            cfg = F.Config(threshold=t, rounds=rounds, local_epochs=epochs, lr=lr, C=C, sigma=sigma,
                           delta_scale=2 ** log_delta, dp_delta=dp_delta, dropout_prob=dropout, seed=int(seed))
            ss.records = []
            for rec in F.train(ss.server, keyed, data, cfg):
                ss.records.append(rec)
                draw(ss.records)
                time.sleep(speed)
        elif ss.get("records"):
            draw(ss.records)
        else:
            st.caption("Each round: local gradient descent → clip ‖w_u‖ ≤ C → add N(0, σ²C²) → quantise "
                       "into Z_R with scale Δ → masked secure aggregation over the QKD channel → centred "
                       "de-quantisation → FedAvg step.")

# ================================================================== TAB 3
if nav == NAV[2]:
    recs = [r for r in ss.get("records", []) if r.ok]
    if not recs:
        st.info("Train at least one round to inspect the traffic.")
    else:
        rsel = st.select_slider("Round", options=[r.r for r in recs], value=recs[-1].r)
        rec = next(r for r in recs if r.r == rsel)
        tr = rec.trace
        st.markdown(f"**Round id** `{tr.round_id}` · U1 = {len(tr.U1)}, U2 = {len(tr.U2)}, "
                    f"U3 = {len(tr.U3)}, U4 = {len(tr.U4)}, U5 = {len(tr.U5)}"
                    + (f" · dropped after Round 1: {', '.join(tr.dropped)}" if tr.dropped else ""))
        delta_scale = 2 ** log_delta
        a, b = st.columns(2)
        with a:
            st.subheader("On the client (never leaves the device)")
            st.caption("Sanitised update x_u, de-quantised to reals: clipped + noised.")
            st.dataframe(pd.DataFrame({cid: S.dequantize_aggregate(v, delta_scale) for cid, v in rec.client_view.items()
                                       if cid in tr.U3}, index=["w₁", "w₂", "bias"]).T.round(4),
                         width="stretch")
        with b:
            st.subheader("What the server receives (Round 2)")
            st.caption("Masked update y_u = x_u + PRG(b_u) + Σ±PRG(s_uv)  mod 2³²: indistinguishable from uniform.")
            st.dataframe(pd.DataFrame({cid: v.astype(np.int64) for cid, v in tr.masked_updates.items()},
                                      index=["y[0]", "y[1]", "y[2]"]).T, width="stretch")
        agg = S.dequantize_aggregate(tr.aggregate, delta_scale)
        st.success(f"After Round 4 the server unmasks only the **sum**: ẑ = ({agg[0]:.4f}, {agg[1]:.4f}, {agg[2]:.4f}), "
                   f"identical to Σ x_u over U3 — it never learns any single x_u.")

        c, d = st.columns(2)
        with c:
            allm = np.concatenate([r.trace.masked_updates and np.concatenate(list(r.trace.masked_updates.values()))
                                   for r in recs]).astype(np.float64) / 2**32
            fig = go.Figure(go.Histogram(x=allm, nbinsx=20, marker_color=BLUE))
            fig.update_layout(title=f"All masked coordinates the server saw, y/R ({len(allm)} values)",
                              xaxis_title="y / 2³²", yaxis_title="count", height=330, margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")
        with d:
            names = [f"R{k} {v}" for k, v in ROUND_NAMES.items()]
            fig = go.Figure(go.Bar(x=names, y=[tr.ciphertext_bytes[k] / 1024 for k in ROUND_NAMES],
                                   text=[f"{tr.envelopes[k]} env." for k in ROUND_NAMES], textposition="outside",
                                   marker_color=GREEN))
            fig.update_layout(title="Traffic encrypted under QKD transport keys", yaxis_title="KB", height=330,
                              margin=dict(t=50, b=40))
            st.plotly_chart(fig, width="stretch")
        with st.expander("One QKD transport envelope (Round 2, as seen on the wire)"):
            env = dict(tr.sample_envelope or {})
            if "ciphertext" in env:
                env["ciphertext"] = env["ciphertext"][:96] + f"… ({len(env['ciphertext']) // 2} bytes)"
            st.json(env)

# ================================================================== TAB 4
if nav == NAV[3]:
    st.markdown(r"""
### The protocol
| Phase | What happens | Security |
|---|---|---|
| **I. QKD transport keys** | Three-party lightweight QKD (Younes et al.) on four-qubit cluster states; clients need only a Hadamard gate and Z-basis measurement. One session per client pair gives two independent 256-bit client–server keys. | Information-theoretic key establishment, given an authenticated classical channel |
| **II. Sanitisation** | Each client clips its update to ‖w‖ ≤ C, adds N(0, σ²C²), and quantises with scale Δ into Z_R, R = 2³². | Client-level (ε, δ)-DP via Rényi accounting with σ_eff = √\|H\|·σ |
| **III. Secure aggregation** | Bonawitz et al. Rounds 0–4 (key advertisement, Shamir-shared keys, pairwise + self masks, survivor consistency check, unmasking), every message in a ChaCha20-Poly1305 envelope under the Phase I key. | Server learns only the sum; t-of-n dropout tolerance |

### What this demo does and does not claim
* The quantum layer is a **Qiskit statevector simulation**; the eavesdropper is a phase-flip (dephasing) channel of strength p, not a general attack.
* QKD protects **client–server transport**. Pairwise masking keys, share encryption and signatures remain **classical** (X25519, Ed25519), so the protocol is QKD-assisted, not fully quantum-secure.
* The final key length uses the operational estimate ℓ = ℓ_raw − ℓ_leak − ℓ_raw·h(Q) − 32, not a composable finite-key bound.
* Privacy accounting assumes every surviving client is honest (|H| = |U3|) and uses no subsampling amplification.
* The server is honest-but-curious; dropouts are simulated after Round 1.

### Files
`qkd_secagg.py` reference protocol (unchanged) · `qkd_layer.py` Phase I runner · `sanitize.py` Phase II ·
`secagg_transport.py` Phase III over QKD transport · `federated.py` toy data and training loop · `app.py` this UI.
""")
