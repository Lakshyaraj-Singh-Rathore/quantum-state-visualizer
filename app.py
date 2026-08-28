"""
app.py — Quantum State Visualizer
Streamlit dashboard. Logic stays in engine.py / noise.py / visualizer.py.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import csv
import io
import time

import numpy as np
import streamlit as st

from engine import GATE_CATALOG, QuantumEngine, one_qubit_matrix
from noise import NoiseParams, compare_ideal_noisy
from visualizer import (
    CORAL,
    GOLD,
    PLOT_CONFIG,
    TEAL,
    amplitude_table,
    circuit_text,
    format_state_ket,
    hud,
    plot_circuit,
    plot_density_matrix_from_engine,
    plot_entanglement,
    plot_exact_vs_shots,
    plot_fidelity,
    plot_gauge,
    plot_ideal_vs_noisy,
    plot_probabilities,
    plot_qsphere,
)

st.set_page_config(
    page_title="Quantum State Visualizer",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

GATE_LABELS = {
    "H": "H — Hadamard",
    "X": "X — Pauli-X",
    "Y": "Y — Pauli-Y",
    "Z": "Z — Pauli-Z",
    "S": "S — π/2 phase",
    "SDG": "S† — −π/2 phase",
    "T": "T — π/4 phase",
    "TDG": "T† — −π/4 phase",
    "RX": "RX(θ)",
    "RY": "RY(θ)",
    "RZ": "RZ(θ)",
    "CNOT": "CNOT (CX)",
    "SWAP": "SWAP",
    "TOFFOLI": "Toffoli (CCX)",
}

PRESETS = [
    ("Superposition", "superposition"),
    ("HZH ≡ X", "hzh"),
    ("T-phase", "t_phase"),
    ("T-demo", "t_demo"),
    ("RX(π)", "rx_pi"),
    ("Bell", "bell"),
    ("GHZ", "ghz"),
    ("Toffoli AND", "toffoli"),
    ("SWAP demo", "swap_demo"),
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0e1117; }
        [data-testid="stHeader"] { background: rgba(14,17,23,0.9); }
        .hero {
            background: linear-gradient(90deg, #161b26 0%, #1a1430 55%, #0e1117 100%);
            border: 1px solid #2a3144;
            border-radius: 16px;
            padding: 1.05rem 1.3rem 1.15rem 1.3rem;
            margin-bottom: 0.9rem;
        }
        .hero h1 {
            color: #e6edf3; font-size: 1.55rem; margin: 0 0 0.2rem 0;
            letter-spacing: 0.02em;
        }
        .hero p { color: #8b9bb4; margin: 0; font-size: 0.95rem; }
        .narration {
            background: #161b26;
            border-left: 4px solid #7c5cbf;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            color: #e6edf3;
            font-size: 1.02rem;
            line-height: 1.45;
            margin: 0.35rem 0 0.8rem 0;
        }
        .ketbox {
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            background: #121722;
            border: 1px solid #2a3144;
            border-radius: 10px;
            padding: 0.7rem 0.9rem;
            color: #3dd6c6;
            font-size: 0.98rem;
        }
        .limit { color: #8b9bb4; font-size: 0.88rem; line-height: 1.45; }
        .measure-hit {
            font-size: 1.55rem; color: #ffd166; font-weight: 700;
            font-family: ui-monospace, Menlo, Consolas, monospace;
        }
        div[data-testid="stMetricValue"] {
            font-family: ui-monospace, Menlo, Consolas, monospace;
            font-size: 1.15rem !important;
        }
        div[data-testid="stMetricValue"] > div {
            font-size: 1.15rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    ss = st.session_state
    if "engine" not in ss:
        ss.engine = QuantumEngine(2)
    ss.setdefault("step", 0)
    ss.setdefault("playing", False)
    ss.setdefault("rx_theta", float(np.pi))
    ss.setdefault("last_measure", None)
    ss.setdefault("flash", "")
    ss.setdefault("cmp", None)
    ss.setdefault("cmp_key", None)
    ss.setdefault("shots", 1024)
    ss.setdefault("noise_demo", False)
    ss.setdefault("noise_on", False)
    ss.setdefault("t1", 50.0)
    ss.setdefault("t2", 30.0)
    ss.setdefault("readout_pct", 2.0)


def flash(msg: str) -> None:
    st.session_state.flash = msg
def apply_noise_demo() -> None:
    """Runs before widgets on the next rerun — required by Streamlit."""
    st.session_state.noise_on = True
    st.session_state.t1 = 8.0
    st.session_state.t2 = 6.0
    st.session_state.readout_pct = 8.0
    st.session_state.noise_demo = True
    st.session_state.flash = (
        "Strong model noise for the demo — not a chip datasheet."
    )

def clamp_step() -> int:
    eng: QuantumEngine = st.session_state.engine
    st.session_state.step = max(0, min(int(st.session_state.step), eng.max_step))
    return st.session_state.step


def available_gates(n: int) -> list[str]:
    names = ["H", "X", "Y", "Z", "S", "SDG", "T", "TDG", "RX", "RY", "RZ"]
    if n >= 2:
        names += ["CNOT", "SWAP"]
    if n >= 3:
        names += ["TOFFOLI"]
    return names


def pretty_matrix(name: str, theta: float | None) -> str:
    try:
        m = one_qubit_matrix(name, theta)
    except Exception:
        return ""
    lines = []
    for r in range(m.shape[0]):
        cells = []
        for c in range(m.shape[1]):
            z = m[r, c]
            re, im = float(np.real(z)), float(np.imag(z))
            if abs(re) < 1e-10:
                re = 0.0
            if abs(im) < 1e-10:
                im = 0.0
            if abs(im) < 1e-10:
                cells.append(f"{re: .3f}")
            elif abs(re) < 1e-10:
                cells.append(f"{im: .3f}j")
            else:
                sign = "+" if im >= 0 else "-"
                cells.append(f"{re: .3f}{sign}{abs(im):.3f}j")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def ops_signature(engine: QuantumEngine) -> tuple:
    return tuple(
        (op.name, tuple(op.qubits), tuple(float(x) for x in op.params))
        for op in engine.operations
    )


def plotly_show(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def load_preset(key: str) -> None:
    ok, msg = st.session_state.engine.load_preset(key)
    flash(msg + "  Rewound to step 0 — press Next.")
    st.session_state.step = 0
    st.session_state.playing = False
    st.session_state.last_measure = None
    st.session_state.cmp = None
    st.session_state.cmp_key = None


def amplitude_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    fields = ["State", "Re", "Im", "Probability", "Percent", "Phase_rad", "Phase_deg"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def render_sidebar(engine: QuantumEngine) -> NoiseParams:
    st.sidebar.markdown("### Circuit builder")
    n = st.sidebar.slider(
        "Number of qubits",
        1,
        4,
        value=engine.n_qubits,
        help="3 qubits unlock GHZ and Toffoli.",
    )
    if n != engine.n_qubits:
        engine.set_n_qubits(n, clear=True)
        st.session_state.step = 0
        st.session_state.last_measure = None
        flash(f"Register resized to {n} qubit(s). Circuit cleared.")

    gates = available_gates(engine.n_qubits)
    gate = st.sidebar.selectbox("Gate", gates, format_func=lambda g: GATE_LABELS.get(g, g))
    spec = GATE_CATALOG[gate]
    st.sidebar.caption(f"**{spec['title']}** — {spec['blurb']}")

    if spec["params"]:
        c1, c2, c3, c4 = st.sidebar.columns(4)
        if c1.button("π/4"):
            st.session_state.rx_theta = float(np.pi / 4)
            st.rerun()
        if c2.button("π/2"):
            st.session_state.rx_theta = float(np.pi / 2)
            st.rerun()
        if c3.button("π"):
            st.session_state.rx_theta = float(np.pi)
            st.rerun()
        if c4.button("2π"):
            st.session_state.rx_theta = float(2 * np.pi)
            st.rerun()
        st.sidebar.slider("Angle θ (radians)", 0.0, float(2 * np.pi), key="rx_theta")
        theta = float(st.session_state.rx_theta)
        st.sidebar.caption(
            f"θ = {theta / np.pi:.3g}π · live if last gate is this RX/RY/RZ · "
            "Add gate only the first time"
        )
        mat = pretty_matrix(gate, theta)
        if mat:
            st.sidebar.code(mat, language="text")
    else:
        theta = float(st.session_state.rx_theta)
        if spec["arity"] == 1:
            mat = pretty_matrix(gate, None)
            if mat:
                st.sidebar.code(mat, language="text")

    q_opts = list(range(engine.n_qubits))
    qubits: list[int] = []
    params = None
    if spec["arity"] == 1:
        target = st.sidebar.selectbox("Target qubit", q_opts, format_func=lambda i: f"q{i}")
        qubits = [int(target)]
        if spec["params"]:
            params = theta
    elif gate == "CNOT":
        c = st.sidebar.selectbox("Control", q_opts, format_func=lambda i: f"q{i}", key="cx_c")
        t_opts = [q for q in q_opts if q != c]
        t = st.sidebar.selectbox("Target", t_opts, format_func=lambda i: f"q{i}", key="cx_t")
        qubits = [int(c), int(t)]
    elif gate == "SWAP":
        a = st.sidebar.selectbox("Qubit A", q_opts, format_func=lambda i: f"q{i}", key="sw_a")
        b_opts = [q for q in q_opts if q != a]
        b = st.sidebar.selectbox("Qubit B", b_opts, format_func=lambda i: f"q{i}", key="sw_b")
        qubits = [int(a), int(b)]
    else:  # TOFFOLI
        c1 = st.sidebar.selectbox("Control 1", q_opts, format_func=lambda i: f"q{i}", key="ccx_c1")
        c2_opts = [q for q in q_opts if q != c1]
        c2 = st.sidebar.selectbox("Control 2", c2_opts, format_func=lambda i: f"q{i}", key="ccx_c2")
        t_opts = [q for q in q_opts if q not in (c1, c2)]
        t = st.sidebar.selectbox("Target", t_opts, format_func=lambda i: f"q{i}", key="ccx_t")
        qubits = [int(c1), int(c2), int(t)]

    # Live knob: dragging θ edits the last RX/RY/RZ if it matches this gate + qubit
    if spec["params"] and engine.operations:
        last = engine.operations[-1]
        if last.name == gate and list(last.qubits) == list(qubits):
            ok, msg = engine.set_last_rotation_angle(float(theta))
            if ok:
                st.sidebar.caption(f"Live angle — {msg}. Drag θ; don’t Add again.")

    if st.sidebar.button("Add gate", type="primary", use_container_width=True):
        ok, msg = engine.add_gate(gate, qubits, params)
        flash(msg)
        if ok:
            st.session_state.step = engine.max_step
            st.session_state.last_measure = None

    u1, u2 = st.sidebar.columns(2)
    if u1.button("Undo", use_container_width=True):
        _, msg = engine.undo()
        flash(msg)
        st.session_state.step = engine.max_step
        st.session_state.last_measure = None
    if u2.button("Clear circuit", use_container_width=True):
        engine.clear()
        st.session_state.step = 0
        st.session_state.last_measure = None
        flash("Circuit cleared.")

    st.sidebar.markdown("### Presets")
    st.sidebar.caption("Loads the circuit and **rewinds to step 0**. Press Next to evolve.")
    cols = st.sidebar.columns(2)
    for i, (label, key) in enumerate(PRESETS):
        if cols[i % 2].button(label, use_container_width=True):
            load_preset(key)
            st.rerun()

    st.sidebar.markdown("### Noise model")
    enabled = st.sidebar.toggle("Enable T₁ / T₂ / readout", key="noise_on")
    t1 = st.sidebar.slider("T₁ (μs)", 1.0, 200.0, key="t1")
    t2 = st.sidebar.slider("T₂ (μs)", 1.0, 200.0, key="t2")
    readout_pct = st.sidebar.slider("Readout error (%)", 0.0, 20.0, key="readout_pct")
    shots = st.sidebar.select_slider(
        "Shots", [128, 256, 512, 1024, 2048, 4096, 8192], key="shots"
    )

    st.sidebar.button(
    "Demo: make noise obvious",
    use_container_width=True,
    on_click=apply_noise_demo,
    key="noise_demo_btn",
    )

    if st.session_state.noise_demo:
        st.sidebar.info("Demo noise preset is active until you move T₁/T₂/readout.")

    with st.sidebar.expander("Advanced gate times"):
        st.caption("Longer pulses → more T₁/T₂ per gate. Z/S/T/RZ stay virtual (no thermal error).")
        g1 = st.slider("1-qubit pulse (μs)", 0.02, 2.0, 0.20, 0.02)
        g2 = st.slider("2-qubit pulse (μs)", 0.05, 4.0, 0.80, 0.05)
        g3 = st.slider("Toffoli pulse (μs)", 0.10, 6.0, 1.50, 0.10)

    if st.session_state.noise_demo:
        g1, g2, g3 = 0.60, 2.00, 3.00

    st.sidebar.markdown("### Demo path (2–3 min)")
    st.sidebar.markdown(
        "1. Superposition (H)\n"
        "2. HZH vs X — same bars, **phase table**\n"
        "3. Bell — Next until S goes **0 → 1**\n"
        "4. Toffoli AND\n"
        "5. RX(θ)\n"
        "6. Bell + Enable noise → **fidelity**\n"
        "7. *T₁/T₂ are model knobs, not IBM calibration.*"
    )

    return NoiseParams(
        enabled=bool(enabled),
        t1_us=float(t1),
        t2_us=float(t2),
        readout_error=float(readout_pct) / 100.0,
        gate_time_1q_us=float(g1),
        gate_time_2q_us=float(g2),
        gate_time_3q_us=float(g3),
    ), int(shots)


def render_stepper(engine: QuantumEngine) -> None:
    max_step = engine.max_step
    c0, c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1, 2])
    if c0.button("⏮ Reset", use_container_width=True):
        st.session_state.step = 0
        st.session_state.playing = False
        st.session_state.last_measure = None
    if c1.button("◀ Prev", use_container_width=True):
        st.session_state.step = max(0, st.session_state.step - 1)
        st.session_state.playing = False
        st.session_state.last_measure = None
    if c2.button("Next ▶", use_container_width=True):
        st.session_state.step = min(max_step, st.session_state.step + 1)
        st.session_state.playing = False
        st.session_state.last_measure = None
    if c3.button("⏭ End", use_container_width=True):
        st.session_state.step = max_step
        st.session_state.playing = False
    play_label = "⏸ Pause" if st.session_state.playing else "▶ Play"
    if c4.button(play_label, use_container_width=True):
        if st.session_state.playing:
            st.session_state.playing = False
        else:
            if st.session_state.step >= max_step:
                st.session_state.step = 0
            st.session_state.playing = True
    c5.markdown(
        f"<div style='padding-top:0.45rem;color:#8b9bb4'>Step "
        f"<b style='color:#ffd166'>{st.session_state.step}</b> / {max_step}</div>",
        unsafe_allow_html=True,
    )
    if max_step > 0:
        st.session_state.step = st.slider(
            "Timeline",
            0,
            max_step,
            st.session_state.step,
            label_visibility="collapsed",
        )


def get_comparison(engine: QuantumEngine, step: int, params: NoiseParams, shots: int):
    key = (
        ops_signature(engine),
        engine.n_qubits,
        int(step),
        bool(params.enabled),
        round(params.t1_us, 4),
        round(params.t2_us, 4),
        round(params.readout_error, 6),
        round(params.gate_time_1q_us, 4),
        round(params.gate_time_2q_us, 4),
        round(params.gate_time_3q_us, 4),
        int(shots),
    )
    if st.session_state.cmp_key == key and st.session_state.cmp is not None:
        return st.session_state.cmp
    try:
        with st.spinner("Running Aer noise model…"):
            result = compare_ideal_noisy(
                engine, step=step, params=params, shots=shots, seed=42
            )
        st.session_state.cmp = result
        st.session_state.cmp_key = key
        return result
    except Exception as exc:
        st.error(f"Noisy simulation failed: {exc}")
        return None


def main() -> None:
    inject_css()
    init_state()
    engine: QuantumEngine = st.session_state.engine
    params, shots = render_sidebar(engine)
    clamp_step()

    st.markdown(
        """
        <div class="hero">
          <h1>⚛ Quantum State Visualizer</h1>
          <p>Step through every gate · watch amplitudes and entanglement · compare ideal vs T₁/T₂/readout noise</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.flash:
        st.caption(st.session_state.flash)

    render_stepper(engine)
    step = clamp_step()

    if st.session_state.playing:
        if step < engine.max_step:
            time.sleep(0.65)
            st.session_state.step = step + 1
            st.rerun()
        else:
            st.session_state.playing = False

    snap = engine.get_snapshot(step)
    info = hud(engine, step)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Qubits", info["qubits"])
    m2.metric("Step", f"{info['step']} / {info['max_step']}")
    m3.metric("Gates", info["gates"])
    m4.metric("Depth", info["depth"])
    m5.metric("Entanglement S", f"{info['entanglement']:.3f}")
    m6.metric("Status", "ENTANGLED" if info["entangled"] else "separable")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Quantum circuit")
        plotly_show(plot_circuit(engine, step))
        with st.expander("Qiskit text diagram"):
            st.code(circuit_text(engine, step), language="text")
    with right:
        st.subheader("Current operation")
        st.markdown(f"<div class='narration'>{snap.narration}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ketbox'>{format_state_ket(snap)}</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button("Measure once (collapse)", use_container_width=True):
            st.session_state.last_measure = engine.measure_once(step=step)
        if b2.button("Clear measurement", use_container_width=True):
            st.session_state.last_measure = None
        if st.session_state.last_measure is not None:
            st.markdown(
                f"<div class='measure-hit'>Measured |{st.session_state.last_measure}⟩</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "One ideal Born-rule sample at this step. The register is **not** left collapsed — "
                "Reset/Next still show the pre-measurement state. Teaching control, not hardware."
            )

    st.subheader("Probability distribution")
    st.caption("Bar **color = relative phase**. Z / S / T / RZ often change color, not height.")
    plotly_show(plot_probabilities(snap))

    comparison = None
    if params.enabled:
        comparison = get_comparison(engine, step, params, shots)

    g1, g2, g3 = st.columns(3)
    with g1:
        plotly_show(plot_entanglement(snap))
    with g2:
        fid = 1.0 if comparison is None else comparison.fidelity
        pur = None if comparison is None else comparison.purity
        plotly_show(plot_fidelity(fid, pur))
    with g3:
        if comparison is None:
            st.info("Enable noise to see **purity** of the mixed state and shot leakage.")
        else:
            plotly_show(
                plot_gauge(
                    comparison.purity,
                    "Purity",
                    color=(
                        TEAL
                        if comparison.purity > 0.97
                        else GOLD
                        if comparison.purity > 0.85
                        else CORAL
                    ),
                    subtitle="Tr(ρ²)  ·  1 = still pure",
                )
            )
    tabs = st.tabs(
        [
            "Ideal vs noisy",
            "Born vs shots",
            "Amplitude / phase table",
            "Density matrix |ρ|",
            "Q-sphere",
            "Export",
            "Limitations",
        ]
    )

    with tabs[0]:
        if not params.enabled:
            st.info("Turn on **Enable T₁ / T₂ / readout** in the sidebar.")
            st.caption("Readout changes **shots**. T₁/T₂ change the density matrix and fidelity.")
        elif comparison is None:
            st.warning("Noisy comparison unavailable — see the error above.")
        else:
            st.caption(comparison.caption)
            plotly_show(plot_ideal_vs_noisy(comparison))
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Fidelity", f"{comparison.fidelity:.3f}")
            k2.metric("Purity", f"{comparison.purity:.3f}")
            k3.metric("TV (state)", f"{comparison.total_variation_state:.3f}")
            k4.metric("Shot leakage", f"{comparison.leakage_shots * 100:.1f}%")
            for w in comparison.warnings:
                st.caption("Note: " + w)

    with tabs[1]:
        plotly_show(plot_exact_vs_shots(engine, step, shots=shots, seed=0))
        st.caption("Finite shots = sampling error. That is not the same thing as T₁/T₂.")

    with tabs[2]:
        rows = amplitude_table(snap)
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with tabs[3]:
        plotly_show(plot_density_matrix_from_engine(engine, step))
        st.caption("Ideal |ρ|. Bell lights up |00⟩⟨11| corners. This tab is noiseless on purpose.")

    with tabs[4]:
        st.markdown(
            "<p style='color:#8b9bb4;font-size:1.02rem;line-height:1.5;margin:0.2rem 0 0.8rem 0'>"
            "Latitude is Hamming weight (how many 1s). North pole is |0…0⟩, south pole is |1…1⟩. "
            "Blob size is probability; color is relative phase (same scale as the bar chart). "
            "This is the <b>ideal</b> statevector at this step — not a Bloch sphere, and not the noisy mixed state."
            "</p>",
            unsafe_allow_html=True,
        )
        plotly_show(plot_qsphere(snap))

    with tabs[5]:
        qasm = engine.to_qasm(step)
        st.download_button(
            "Download OpenQASM (up to this step)",
            qasm,
            file_name="circuit.qasm",
            mime="text/plain",
        )
        st.download_button(
            "Download amplitude CSV",
            amplitude_csv(amplitude_table(snap)),
            file_name="state.csv",
            mime="text/csv",
        )
        st.code(qasm, language="text")

    with tabs[6]:
        st.markdown(
            """
<div class="limit">

**Scope**

This app runs entirely on your machine. Circuits are stepped with Qiskit’s statevector simulator; the optional noise layer uses Qiskit Aer. Nothing here is submitted to cloud hardware.

**Noise model**

T₁, T₂ and readout error are parameters you set. They reproduce the *kind* of errors seen on superconducting devices (relaxation, dephasing, misreported bits), but they are not taken from any particular chip’s calibration.

Z, S, S†, T, T† and RZ are treated as instantaneous. That matches how those phases are usually applied in software (virtual-Z) rather than as extra microwave pulses, so they do not pick up extra T₁/T₂ in this model.

**State display**

Basis labels are written |q0 q1 q2…⟩ with **q0 on the left**. That is the usual textbook order; it is the reverse of Qiskit’s little-endian bitstrings.

The entanglement reading is the largest single-qubit von Neumann entropy after the other qubits are traced out (in bits). On two qubits, concurrence is shown as well. Those numbers describe the *ideal* snapshot. Once noise is on, **fidelity** and **purity** are the quantities that track how far the mixed state has drifted.

A Bloch sphere is omitted on purpose. It fully describes one qubit; it does not describe a general entangled register.

The Q-sphere is a 3D layout of computational-basis amplitudes (IBM-style). It is not a Bloch sphere and does not represent a mixed state after T₁/T₂.

</div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
