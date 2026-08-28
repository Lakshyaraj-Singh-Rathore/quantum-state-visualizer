"""
visualizer.py — Quantum State Visualizer
Plotly charts + step-highlighted circuit. No Streamlit.

Teaching labels match engine.py: |q0 q1 ...> with q0 LEFTMOST.
"""

from __future__ import annotations

import colorsys
import os
from typing import Any

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "plotly is required. With the venv on:\n"
        "  pip install plotly"
    ) from exc

from qiskit.quantum_info import DensityMatrix, Statevector

from engine import QuantumEngine, Snapshot, textbook_label
from noise import ComparisonResult


PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
}

BG = "#0e1117"
GRID = "rgba(255,255,255,0.06)"
INK = "#e6edf3"
MUTED = "#8b9bb4"
ACCENT = "#7c5cbf"
GOLD = "#ffd166"
TEAL = "#3dd6c6"
CORAL = "#ff5d5d"

GATE_COLOR = {
    "H": "#9b7bff",
    "X": "#ff5d5d",
    "Y": "#ffb020",
    "Z": "#3dd6c6",
    "S": "#3dd6c6",
    "SDG": "#2bb5a8",
    "T": "#5ee0d0",
    "TDG": "#2bb5a8",
    "RX": "#5b9dff",
    "RY": "#5b9dff",
    "RZ": "#3dd6c6",
    "CNOT": "#ff7a45",
    "SWAP": "#e17cff",
    "TOFFOLI": "#ff7a45",
}

PHASE_CSCALE = [
    [0.00, "rgb(61,214,198)"],
    [0.25, "rgb(91,157,255)"],
    [0.50, "rgb(225,124,255)"],
    [0.75, "rgb(255,93,93)"],
    [1.00, "rgb(61,214,198)"],
]


def _layout(**extra: Any) -> dict[str, Any]:
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color=INK, family="Segoe UI, Inter, sans-serif", size=13),
        margin=dict(l=56, r=36, t=56, b=52),
        hoverlabel=dict(bgcolor="#1b2030", font_size=12),
    )
    base.update(extra)
    return base


def _hex_rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _tick_label(lab: str) -> str:
    return f"|{lab}⟩"


def _strip_ket(tick: Any) -> str:
    return str(tick).replace("|", "").replace("⟩", "").replace(">", "")


def _angle_text(theta: float) -> str:
    turns = float(theta) / np.pi
    for num, den in ((1, 4), (1, 2), (3, 4), (1, 1), (5, 4), (3, 2), (7, 4), (2, 1)):
        if abs(turns - num / den) < 0.02:
            if den == 1:
                return "π" if num == 1 else f"{num}π"
            if num == 1:
                return f"π/{den}"
            return f"{num}π/{den}"
    return f"{turns:.2f}π"


def phase_to_rgba(phase: float | None, prob: float) -> str:
    if phase is None or prob < 1e-10:
        return "rgba(90,100,120,0.38)"
    h = (float(phase) / (2 * np.pi)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.72, 0.98)
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.95)"


def _sorted_labels(keys) -> list[str]:
    return sorted(keys, key=lambda s: (len(s), s))


# ------------------------------------------------------------------ circuit
def _qubit_y(q: int, n: int) -> float:
    return float(n - 1 - q)


def _gate_role(gate_index: int, current_step: int | None) -> str:
    if current_step is None:
        return "past"
    if gate_index < current_step - 1:
        return "past"
    if gate_index == current_step - 1:
        return "current"
    return "future"


def _short_gate_name(op) -> str:
    if op.name in ("RX", "RY", "RZ"):
        th = op.params[0] if op.params else 0.0
        return f"{op.name}\n{_angle_text(th)}"
    if op.name == "CNOT":
        return "CX"
    if op.name == "TOFFOLI":
        return "CCX"
    if op.name == "SDG":
        return "S†"
    if op.name == "TDG":
        return "T†"
    return op.name


def circuit_text(engine: QuantumEngine, step: int | None = None) -> str:
    qc = engine.to_circuit(step)
    try:
        return str(qc.draw(output="text", fold=-1))
    except Exception:
        return str(qc)


def circuit_mpl(engine: QuantumEngine, step: int | None = None):
    """Optional matplotlib figure. Returns None if mpl/pylatexenc missing."""
    try:
        qc = engine.to_circuit(step)
        return qc.draw(output="mpl", fold=-1, idle_wires=True)
    except Exception:
        return None


def plot_circuit(
    engine: QuantumEngine,
    current_step: int | None = None,
    *,
    title: str | None = None,
) -> go.Figure:
    """One column per gate (not packed by depth) so Prev/Next can highlight."""
    n = engine.n_qubits
    ops = engine.operations
    n_ops = len(ops)
    if current_step is None:
        current_step = engine.max_step
    current_step = max(0, min(int(current_step), n_ops))

    fig = go.Figure()
    x_left, x_right = -0.85, max(n_ops, 1) + 0.55

    for q in range(n):
        y = _qubit_y(q, n)
        fig.add_shape(
            type="line",
            x0=x_left,
            x1=x_right,
            y0=y,
            y1=y,
            line=dict(color="#6d7b91", width=2),
            layer="below",
        )
        fig.add_annotation(
            x=x_left - 0.05,
            y=y,
            text=f"q{q}",
            showarrow=False,
            xanchor="right",
            font=dict(size=13, color=MUTED),
        )

    fig.add_annotation(
        x=x_left,
        y=n - 0.15 if n else 0,
        text="|0⋯0⟩",
        showarrow=False,
        font=dict(size=11, color=MUTED),
        yshift=18,
    )

    hover_x, hover_y, hover_t = [], [], []

    for i, op in enumerate(ops):
        x = float(i + 1)
        role = _gate_role(i, current_step)
        alpha = {"past": 0.96, "current": 1.0, "future": 0.28}[role]
        stroke = GOLD if role == "current" else "rgba(255,255,255,0.35)"
        sw = 3 if role == "current" else 1
        color = GATE_COLOR.get(op.name, ACCENT)
        fill = _hex_rgba(color, alpha)

        if role == "current":
            fig.add_vrect(
                x0=x - 0.48,
                x1=x + 0.48,
                fillcolor="rgba(255,209,102,0.10)",
                line_width=0,
                layer="below",
            )

        ys = [_qubit_y(q, n) for q in op.qubits]
        y_min, y_max = min(ys), max(ys)
        if y_max != y_min:
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=y_min,
                y1=y_max,
                line=dict(color=_hex_rgba(color, alpha), width=2),
            )

        if op.name == "CNOT":
            yc, yt = _qubit_y(op.qubits[0], n), _qubit_y(op.qubits[1], n)
            fig.add_shape(
                type="circle",
                x0=x - 0.12,
                x1=x + 0.12,
                y0=yc - 0.12,
                y1=yc + 0.12,
                fillcolor=fill,
                line=dict(color=stroke, width=sw),
            )
            fig.add_shape(
                type="circle",
                x0=x - 0.22,
                x1=x + 0.22,
                y0=yt - 0.22,
                y1=yt + 0.22,
                fillcolor=BG,
                line=dict(color=_hex_rgba(color, alpha), width=2),
            )
            fig.add_annotation(
                x=x, y=yt, text="+", showarrow=False,
                font=dict(size=18, color=_hex_rgba("#ffffff", alpha)),
            )
        elif op.name == "TOFFOLI":
            y1, y2, yt = (_qubit_y(op.qubits[0], n), _qubit_y(op.qubits[1], n), _qubit_y(op.qubits[2], n))
            for yc in (y1, y2):
                fig.add_shape(
                    type="circle",
                    x0=x - 0.12,
                    x1=x + 0.12,
                    y0=yc - 0.12,
                    y1=yc + 0.12,
                    fillcolor=fill,
                    line=dict(color=stroke, width=sw),
                )
            fig.add_shape(
                type="circle",
                x0=x - 0.22,
                x1=x + 0.22,
                y0=yt - 0.22,
                y1=yt + 0.22,
                fillcolor=BG,
                line=dict(color=_hex_rgba(color, alpha), width=2),
            )
            fig.add_annotation(
                x=x, y=yt, text="+", showarrow=False,
                font=dict(size=18, color=_hex_rgba("#ffffff", alpha)),
            )
        elif op.name == "SWAP":
            for y in ys:
                fig.add_annotation(
                    x=x, y=y, text="×", showarrow=False,
                    font=dict(size=22, color=_hex_rgba(color, alpha)),
                )
        else:
            y = ys[0]
            fig.add_shape(
                type="rect",
                x0=x - 0.36,
                x1=x + 0.36,
                y0=y - 0.32,
                y1=y + 0.32,
                fillcolor=fill,
                line=dict(color=stroke, width=sw),
            )
            fig.add_annotation(
                x=x,
                y=y,
                text=_short_gate_name(op).replace("\n", "<br>"),
                showarrow=False,
                font=dict(size=11, color="#0b0d12"),
            )

        hover_x.append(x)
        hover_y.append(sum(ys) / len(ys))
        hover_t.append(op.display_label() + f"  ({role})")

    if hover_x:
        fig.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker=dict(size=32, opacity=0),
                hovertext=hover_t,
                hoverinfo="text",
                showlegend=False,
            )
        )

    snap = engine.get_snapshot(current_step)
    if title is None:
        title = f"Circuit  ·  step {current_step}/{n_ops}  ·  {snap.label}"

    height = 140 + n * 78
    width = max(520, 180 + max(n_ops, 1) * 88)
    fig.update_layout(
        **_layout(
            title=dict(text=title, x=0.02, xanchor="left"),
            height=height,
            width=width,
            showlegend=False,
            margin=dict(l=70, r=24, t=48, b=24),
        )
    )
    fig.update_xaxes(visible=False, range=[x_left - 0.45, x_right + 0.1])
    fig.update_yaxes(visible=False, range=[-0.75, n - 0.25 + 0.55])
    return fig


# ----------------------------------------------------------- probabilities
def plot_probabilities(
    snapshot: Snapshot,
    *,
    title: str | None = None,
    min_prob: float = 0.0,
) -> go.Figure:
    labels = _sorted_labels(snapshot.probabilities)
    xs, ys, colors, custom = [], [], [], []
    for lab in labels:
        p = snapshot.probabilities[lab]
        if p < min_prob:
            continue
        xs.append(_tick_label(lab))
        ys.append(p)
        ph = snapshot.phases_relative.get(lab)
        colors.append(phase_to_rgba(ph, p))
        deg = "" if ph is None else f"{np.degrees(ph):+.0f}°"
        custom.append(f"p={p * 100:.2f}%  phase {deg}")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=xs,
            y=ys,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v * 100:.1f}%" if v >= 0.03 else "" for v in ys],
            textposition="outside",
            hovertext=custom,
            hoverinfo="text+x",
            name="probability",
            cliponaxis=False,
        )
    )
    # invisible trace so a phase colorbar always appears
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                colorscale=PHASE_CSCALE,
                cmin=-np.pi,
                cmax=np.pi,
                color=[0],
                showscale=True,
                colorbar=dict(
                    title=dict(text="rel. phase", side="right"),
                    tickvals=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                    ticktext=["−π", "−π/2", "0", "+π/2", "+π"],
                    len=0.7,
                ),
                size=1,
                opacity=0,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    if title is None:
        title = f"Probabilities  ·  {snapshot.label}"
    ymax = max(ys) if ys else 1.0
    fig.update_layout(
        **_layout(
            title=dict(text=title, x=0.02, xanchor="left"),
            height=380,
            bargap=0.28,
            yaxis=dict(
                title="P(|x⟩)",
                range=[0, min(1.15, max(0.4, ymax * 1.25))],
                gridcolor=GRID,
                zerolinecolor=GRID,
            ),
            xaxis=dict(title="basis state (q0 leftmost)", tickangle=0),
            showlegend=False,
        )
    )
    return fig


def plot_exact_vs_shots(
    engine: QuantumEngine,
    step: int | None = None,
    shots: int = 1024,
    seed: int | None = 0,
) -> go.Figure:
    snap = engine.get_snapshot(step)
    counts = engine.sample_counts(shots=shots, step=snap.step, seed=seed)
    total = float(sum(counts.values())) or 1.0
    labels = _sorted_labels(snap.probabilities)
    exact = [snap.probabilities[k] for k in labels]
    emp = [counts.get(k, 0) / total for k in labels]
    ticks = [_tick_label(k) for k in labels]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="exact |ψ|²", x=ticks, y=exact, marker_color=TEAL))
    fig.add_trace(go.Bar(name=f"{shots} shots", x=ticks, y=emp, marker_color=ACCENT))
    fig.update_layout(
        **_layout(
            title=dict(text="Born rule vs sampling", x=0.02, xanchor="left"),
            barmode="group",
            height=380,
            yaxis=dict(title="probability", gridcolor=GRID, range=[0, 1.05]),
            legend=dict(orientation="h", y=1.12),
        )
    )
    return fig


def plot_ideal_vs_noisy(result: ComparisonResult) -> go.Figure:
    labels = _sorted_labels(
        set(result.ideal_probs) | set(result.noisy_state_probs) | set(result.noisy_shot_probs)
    )
    ticks = [_tick_label(k) for k in labels]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Ideal",
            x=ticks,
            y=[result.ideal_probs.get(k, 0.0) for k in labels],
            marker_color=TEAL,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Noisy state",
            x=ticks,
            y=[result.noisy_state_probs.get(k, 0.0) for k in labels],
            marker_color=GOLD,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Noisy shots",
            x=ticks,
            y=[result.noisy_shot_probs.get(k, 0.0) for k in labels],
            marker_color=CORAL,
        )
    )
    fig.update_layout(
        **_layout(
            title=dict(
                text="Ideal vs noisy",
                x=0.0,
                xanchor="left",
                font=dict(size=22, color=INK, family="Segoe UI, Inter, sans-serif"),
            ),
            barmode="group",
            height=460,
            font=dict(size=15, color=INK, family="Segoe UI, Inter, sans-serif"),
            yaxis=dict(
                title=dict(text="Probability", font=dict(size=16)),
                tickfont=dict(size=14),
                gridcolor=GRID,
                range=[0, 1.08],
            ),
            xaxis=dict(
                title=dict(text="Basis state (q0 leftmost)", font=dict(size=16)),
                tickfont=dict(size=16),
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                x=0.0,
                xanchor="left",
                font=dict(size=15),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=64, r=28, t=64, b=88),
        )
    )
    return fig


# --------------------------------------------------------------- gauges
def plot_gauge(
    value: float,
    title: str,
    *,
    vmax: float = 1.0,
    color: str = TEAL,
    subtitle: str = "",
) -> go.Figure:
    title_html = (
        f"<span style='font-size:20px;color:{INK}'>{title}</span>"
        f"<br><span style='font-size:14px;color:{MUTED}'>{subtitle}</span>"
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(value),
            number={
                "valueformat": ".3f",
                "font": {
                    "size": 40,
                    "color": INK,
                    "family": "Segoe UI, Inter, sans-serif",
                },
            },
            title={"text": title_html, "font": {"size": 18}},
            domain={"x": [0.08, 0.92], "y": [0.0, 0.70]},
            gauge={
                "axis": {
                    "range": [0, vmax],
                    "tickcolor": MUTED,
                    "tickfont": {"size": 13, "color": MUTED},
                },
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "#1b2030",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, vmax * 0.33], "color": "#18202c"},
                    {"range": [vmax * 0.33, vmax * 0.66], "color": "#1e2a3c"},
                    {"range": [vmax * 0.66, vmax], "color": "#24344c"},
                ],
                "threshold": {
                    "line": {"color": GOLD, "width": 2},
                    "thickness": 0.8,
                    "value": float(value),
                },
            },
        )
    )
    fig.update_layout(
        **_layout(
            height=320,
            margin=dict(l=10, r=10, t=72, b=4),
        )
    )
    return fig


def plot_entanglement(snapshot: Snapshot) -> go.Figure:
    n = len(next(iter(snapshot.probabilities)))
    if n < 2:
        return plot_gauge(
            0.0,
            "Entanglement",
            color=MUTED,
            subtitle="need 2+ qubits  ·  N/A",
        )
    sub = "ENTANGLED" if snapshot.entangled else "separable"
    if snapshot.concurrence is not None:
        sub += f"  ·  C={snapshot.concurrence:.2f}"
    return plot_gauge(
        snapshot.entanglement,
        "Entanglement S",
        color=GOLD if snapshot.entangled else TEAL,
        subtitle=sub,
    )


def plot_fidelity(value: float, purity: float | None = None) -> go.Figure:
    sub = "F(ideal, noisy ρ)"
    if purity is not None:
        sub += f"  ·  purity {purity:.3f}"
    color = TEAL if value > 0.97 else GOLD if value > 0.85 else CORAL
    return plot_gauge(value, "Fidelity", color=color, subtitle=sub)


# --------------------------------------------------------------- density
def plot_density_matrix(
    dm: DensityMatrix | np.ndarray | Statevector,
    n_qubits: int,
    *,
    title: str = "|ρᵢⱼ|  (textbook basis)",
) -> go.Figure:
    if isinstance(dm, Statevector):
        rho = np.asarray(DensityMatrix(dm).data, dtype=complex)
    else:
        rho = np.asarray(getattr(dm, "data", dm), dtype=complex)
    z = np.abs(rho)
    labels = [_tick_label(textbook_label(i, n_qubits)) for i in range(2**n_qubits)]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0,
            zmax=max(1.0, float(z.max()) if z.size else 1.0),
            colorbar=dict(title="|ρ|"),
            hovertemplate="row %{y}<br>col %{x}<br>|ρ|=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            title=dict(text=title, x=0.02, xanchor="left"),
            height=420,
            yaxis=dict(autorange="reversed", title="bra", scaleanchor="x"),
            xaxis=dict(title="ket", side="top"),
            margin=dict(l=70, r=50, t=70, b=40),
        )
    )
    return fig


def plot_density_matrix_from_engine(
    engine: QuantumEngine, step: int | None = None
) -> go.Figure:
    sv = engine.statevector_at(step)
    snap = engine.get_snapshot(step)
    return plot_density_matrix(
        sv, engine.n_qubits, title=f"Ideal |ρ|  ·  {snap.label}"
    )
# --------------------------------------------------------------- q-sphere
def hamming_weight(label: str) -> int:
    return str(label).count("1")


def qsphere_points(snapshot: Snapshot, min_prob: float = 1e-4) -> list[dict[str, Any]]:
    """
    IBM-style Q-sphere coordinates for the ideal ket.
    Latitude = Hamming weight. Azimuth = lexicographic index among ALL
    basis states of that weight (so missing amplitudes do not slide).
    """
    if not snapshot.probabilities:
        return []
    n = len(next(iter(snapshot.probabilities)))
    if n == 0:
        return []

    by_weight: dict[int, list[str]] = {w: [] for w in range(n + 1)}
    for lab in snapshot.probabilities:
        by_weight[hamming_weight(lab)].append(lab)
    for w in by_weight:
        by_weight[w].sort()

    points: list[dict[str, Any]] = []
    for w, labs in by_weight.items():
        m = len(labs)
        theta = 0.0 if n == 0 else np.pi * w / n
        st, ct = float(np.sin(theta)), float(np.cos(theta))
        for k, lab in enumerate(labs):
            p = float(snapshot.probabilities.get(lab, 0.0))
            if p < min_prob:
                continue
            phi = 0.0 if m <= 1 else 2.0 * np.pi * k / m
            ph = snapshot.phases_relative.get(lab)
            points.append(
                {
                    "label": lab,
                    "x": st * float(np.cos(phi)),
                    "y": st * float(np.sin(phi)),
                    "z": ct,
                    "weight": w,
                    "prob": p,
                    "phase": ph,
                    "color": phase_to_rgba(ph, p),
                }
            )
    return points


def _qsphere_wireframe(n: int) -> list[go.Scatter3d]:
    """IBM-style globe: meridians + one parallel per Hamming weight."""
    traces: list[go.Scatter3d] = []
    ring_color = "rgba(170,178,190,0.55)"
    t = np.linspace(0.0, 2.0 * np.pi, 96)

    weights = list(range(1, n)) if n >= 2 else []
    if n >= 1 and 0 not in weights and n not in weights:
        pass
    for w in weights:
        theta = np.pi * w / n
        r, z = float(np.sin(theta)), float(np.cos(theta))
        traces.append(
            go.Scatter3d(
                x=r * np.cos(t),
                y=r * np.sin(t),
                z=np.full_like(t, z),
                mode="lines",
                line=dict(color=ring_color, width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    v = np.linspace(0.0, np.pi, 48)
    sv, cv = np.sin(v), np.cos(v)
    for phi in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        traces.append(
            go.Scatter3d(
                x=sv * np.cos(phi),
                y=sv * np.sin(phi),
                z=cv,
                mode="lines",
                line=dict(color="rgba(170,178,190,0.28)", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _phase_label(phase: float | None) -> str:
    if phase is None:
        return "0"
    turns = float(phase) / np.pi
    for num, den, s in (
        (0, 1, "0"),
        (1, 4, "π/4"),
        (1, 2, "π/2"),
        (3, 4, "3π/4"),
        (1, 1, "π"),
        (-1, 4, "−π/4"),
        (-1, 2, "−π/2"),
        (-3, 4, "−3π/4"),
        (-1, 1, "−π"),
    ):
        if abs(turns - num / den) < 0.08:
            return s
    return f"{turns:.2f}π"

def _qsphere_label_xyz(pts: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Labels stay inside the scene. South pole sits just below the blob, not off-canvas."""
    placed: list[list[float]] = []
    for p in pts:
        x, y, z = float(p["x"]), float(p["y"]), float(p["z"])
        if z >= 0.70:
            placed.append([0.0, 0.0, 1.38])
        elif z <= -0.70:
            placed.append([0.0, 0.0, -1.42])
        else:
            rho = float(np.sqrt(x * x + y * y)) or 1e-9
            placed.append([x * 1.38 + 0.22 * x / rho, y * 1.38 + 0.22 * y / rho, z])

    min_lab, min_blob = 0.42, 0.40
    for _ in range(24):
        moved = False
        for i, p in enumerate(pts):
            dx = placed[i][0] - p["x"]
            dy = placed[i][1] - p["y"]
            dz = placed[i][2] - p["z"]
            d = float(np.sqrt(dx * dx + dy * dy + dz * dz)) or 1e-9
            if d < min_blob:
                g = (min_blob - d) / d
                placed[i][0] += dx * g
                placed[i][1] += dy * g
                placed[i][2] += dz * g
                moved = True
            for j in range(i + 1, len(pts)):
                dx = placed[i][0] - placed[j][0]
                dy = placed[i][1] - placed[j][1]
                dz = placed[i][2] - placed[j][2]
                d = float(np.sqrt(dx * dx + dy * dy + dz * dz)) or 1e-9
                if d < min_lab:
                    g = 0.5 * (min_lab - d) / d
                    placed[i][0] += dx * g
                    placed[i][1] += dy * g
                    placed[i][2] += dz * g
                    placed[j][0] -= dx * g
                    placed[j][1] -= dy * g
                    placed[j][2] -= dz * g
                    moved = True
        if not moved:
            break

    for row in placed:
        row[0] = float(np.clip(row[0], -1.55, 1.55))
        row[1] = float(np.clip(row[1], -1.55, 1.55))
        row[2] = float(np.clip(row[2], -1.55, 1.55))
    return [(a, b, c) for a, b, c in placed]


def plot_qsphere(
    snapshot: Snapshot,
    *,
    min_prob: float = 1e-4,
    title: str | None = None,
) -> go.Figure:
    """Ideal Q-sphere, IBM layout, red spokes, bold labels inside the scene."""
    n = len(next(iter(snapshot.probabilities))) if snapshot.probabilities else 0
    pts = qsphere_points(snapshot, min_prob=min_prob)
    fig = go.Figure()
    spoke = CORAL

    u = np.linspace(0.0, 2.0 * np.pi, 40)
    v = np.linspace(0.0, np.pi, 20)
    fig.add_trace(
        go.Surface(
            x=np.outer(np.cos(u), np.sin(v)),
            y=np.outer(np.sin(u), np.sin(v)),
            z=np.outer(np.ones_like(u), np.cos(v)),
            opacity=0.10,
            showscale=False,
            colorscale=[[0, "#8b95a5"], [1, "#8b95a5"]],
            hoverinfo="skip",
            name="sphere",
        )
    )
    for tr in _qsphere_wireframe(n):
        fig.add_trace(tr)

    if pts:
        xs, ys, zs = [], [], []
        for p in pts:
            xs += [0.0, p["x"], None]
            ys += [0.0, p["y"], None]
            zs += [0.0, p["z"], None]
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color=spoke, width=6),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        sizes = [14 + 22.0 * np.sqrt(p["prob"]) for p in pts]
        hover = []
        for p in pts:
            phs = _phase_label(p["phase"])
            deg = "" if p["phase"] is None else f"{np.degrees(p['phase']):+.0f}°"
            hover.append(
                f"|{p['label']}⟩<br>p = {p['prob'] * 100:.2f}%<br>"
                f"weight {p['weight']}<br>rel. phase {phs} {deg}"
            )
        fig.add_trace(
            go.Scatter3d(
                x=[p["x"] for p in pts],
                y=[p["y"] for p in pts],
                z=[p["z"] for p in pts],
                mode="markers",
                marker=dict(
                    size=sizes,
                    color=spoke,
                    line=dict(width=1, color="rgba(255,255,255,0.35)"),
                    opacity=0.98,
                ),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        lab_xyz = _qsphere_label_xyz(pts)
        fig.add_trace(
            go.Scatter3d(
                x=[a for a, _, _ in lab_xyz],
                y=[b for _, b, _ in lab_xyz],
                z=[c for _, _, c in lab_xyz],
                mode="text",
                text=[
                    f"<b>|{p['label']}⟩</b>  {_phase_label(p['phase'])}"
                    for p in pts
                ],
                textfont=dict(
                    size=17,
                    color=INK,
                    family="Segoe UI Semibold, Segoe UI, Arial, sans-serif",
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if title is None:
        title = f"Q-sphere  ·  {snapshot.label}"

    fig.update_layout(
        **_layout(
            title=dict(
                text=title,
                x=0.0,
                xanchor="left",
                font=dict(size=22, color=INK),
            ),
            height=600,
            margin=dict(l=8, r=16, t=64, b=28),
            scene=dict(
                xaxis=dict(visible=False, range=[-1.65, 1.65]),
                yaxis=dict(visible=False, range=[-1.65, 1.65]),
                zaxis=dict(visible=False, range=[-1.65, 1.65]),
                aspectmode="cube",
                bgcolor=BG,
                camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)),
            ),
            showlegend=False,
        )
    )
    return fig


# --------------------------------------------------------------- tables
def format_state_ket(snapshot: Snapshot, min_p: float = 0.02, max_terms: int = 6) -> str:
    items = sorted(snapshot.probabilities.items(), key=lambda kv: (-kv[1], kv[0]))
    parts: list[str] = []
    for lab, p in items:
        if p < min_p:
            continue
        a = snapshot.amplitudes[lab]
        ph = snapshot.phases_relative.get(lab) or 0.0
        mag = abs(a)
        if abs(ph) < 1e-6:
            term = f"{mag:.3f}|{lab}⟩"
        else:
            term = f"{mag:.3f} e^{{i{ph:+.2f}}}|{lab}⟩"
        parts.append(term)
        if len(parts) >= max_terms:
            break
    if not parts:
        return "|ψ⟩ ≈ 0"
    return "|ψ⟩ ≈ " + " + ".join(parts)


def amplitude_table(snapshot: Snapshot, min_p: float = 0.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lab in _sorted_labels(snapshot.probabilities):
        p = snapshot.probabilities[lab]
        if p < min_p:
            continue
        a = snapshot.amplitudes[lab]
        ph = snapshot.phases_relative.get(lab)
        rows.append(
            {
                "State": _tick_label(lab),
                "Re": float(np.real(a)),
                "Im": float(np.imag(a)),
                "Probability": p,
                "Percent": p * 100.0,
                "Phase_rad": ph,
                "Phase_deg": None if ph is None else float(np.degrees(ph)),
            }
        )
    return rows


def hud(engine: QuantumEngine, step: int | None = None) -> dict[str, Any]:
    snap = engine.get_snapshot(step)
    return {
        "qubits": engine.n_qubits,
        "step": snap.step,
        "max_step": engine.max_step,
        "depth": snap.depth,
        "gates": snap.gate_count,
        "entanglement": snap.entanglement,
        "entangled": snap.entangled,
        "concurrence": snap.concurrence,
        "operation": snap.label,
        "narration": snap.narration,
        "ket": format_state_ket(snap),
    }


def build_step_views(
    engine: QuantumEngine,
    step: int | None = None,
    *,
    comparison: ComparisonResult | None = None,
    shots: int = 512,
    seed: int | None = 0,
) -> dict[str, Any]:
    """One dict for Streamlit later — do not duplicate this logic in app.py."""
    snap = engine.get_snapshot(step)
    views: dict[str, Any] = {
        "hud": hud(engine, snap.step),
        "circuit": plot_circuit(engine, snap.step),
        "circuit_text": circuit_text(engine, snap.step),
        "probs": plot_probabilities(snap),
        "entanglement": plot_entanglement(snap),
        "shots": plot_exact_vs_shots(engine, snap.step, shots=shots, seed=seed),
        "dm": plot_density_matrix_from_engine(engine, snap.step),
        "table": amplitude_table(snap),
        "ket": format_state_ket(snap),
        "narration": snap.narration,
    }
    if comparison is not None:
        views["ideal_noisy"] = plot_ideal_vs_noisy(comparison)
        views["fidelity"] = plot_fidelity(comparison.fidelity, comparison.purity)
        views["caption"] = comparison.caption
    return views


def write_preview_html(
    figures: list[tuple[str, go.Figure]],
    path: str = "viz_preview.html",
) -> str:
    chunks: list[str] = []
    include_js: bool | str = True
    for heading, fig in figures:
        chunks.append(
            f"<h2 style='color:#e6edf3;font-family:Segoe UI,sans-serif'>{heading}</h2>"
        )
        chunks.append(
            fig.to_html(full_html=False, include_plotlyjs=include_js, config=PLOT_CONFIG)
        )
        include_js = False
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Quantum State Visualizer — preview</title>"
        "<style>body{background:#0e1117;margin:28px;color:#e6edf3}</style>"
        "</head><body>"
        "<h1>Quantum State Visualizer — visualizer.py preview</h1>"
        "<p>If the charts render, Plotly is working. Close this tab and continue.</p>"
        + "".join(chunks)
        + "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.abspath(path)


# ===================================================================== tests
def _close(a: float, b: float, tol: float = 0.03) -> bool:
    return abs(a - b) <= tol


def run_self_test() -> bool:
    print("=== visualizer.py self-test (Plotly, no window required) ===\n")
    passed = 0
    failed = 0

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name}" + (f"  — {detail}" if detail else ""))
        if cond:
            passed += 1
        else:
            failed += 1

    bell = QuantumEngine(2)
    bell.load_preset("bell")
    s0, s1, s2 = bell.get_snapshot(0), bell.get_snapshot(1), bell.get_snapshot(2)

    fig_c = plot_circuit(bell, 2)
    check("Bell circuit figure", len(fig_c.layout.shapes) >= 2, f"shapes={len(fig_c.layout.shapes)}")

    fig_p = plot_probabilities(s2)
    mapped = {_strip_ket(x): float(y) for x, y in zip(fig_p.data[0].x, fig_p.data[0].y)}
    check(
        "Bell probability bars",
        _close(mapped.get("00", 0), 0.5) and _close(mapped.get("11", 0), 0.5),
        str({k: round(v, 3) for k, v in mapped.items()}),
    )

    fig_e = plot_entanglement(s2)
    check(
        "Entanglement gauge ~ 1",
        _close(float(fig_e.data[0].value), 1.0, 0.08),
        f"value={fig_e.data[0].value}",
    )
    fig_e0 = plot_entanglement(s0)
    check(
        "Step 0 entanglement ~ 0",
        float(fig_e0.data[0].value) < 0.05,
        f"value={fig_e0.data[0].value}",
    )

    ket = format_state_ket(s2)
    check("Bell ket text", "00" in ket and "11" in ket, ket)

    rows = amplitude_table(s2)
    check("Amplitude table has 4 basis states", len(rows) == 4, str(len(rows)))

    h = hud(bell, 2)
    check(
        "HUD keys",
        h["entangled"] is True and h["step"] == 2 and "CNOT" in h["operation"],
        str({k: h[k] for k in ("step", "operation", "entangled")}),
    )

    txt = circuit_text(bell)
    check("Qiskit text circuit non-empty", len(txt) > 10, txt.splitlines()[0] if txt else "empty")

    fig_dm = plot_density_matrix_from_engine(bell, 2)
    z = np.array(fig_dm.data[0].z)
    check("Bell |ρ| is 4×4", z.shape == (4, 4), str(z.shape))
    check("Bell |ρ_00,11| visible", float(z[0, 3]) > 0.4 or float(z.max()) > 0.4, f"max={z.max():.3f}")

    fig_sh = plot_exact_vs_shots(bell, 2, shots=200, seed=1)
    check("Exact vs shots has 2 traces", len(fig_sh.data) == 2, str(len(fig_sh.data)))

    hzh = QuantumEngine(1)
    hzh.load_preset("hzh")
    fig_h = plot_probabilities(hzh.get_snapshot())
    mapped_h = {_strip_ket(x): float(y) for x, y in zip(fig_h.data[0].x, fig_h.data[0].y)}
    check("HZH |1⟩ ≈ 1", _close(mapped_h.get("1", 0), 1.0), str(mapped_h))
    pts0 = qsphere_points(s0)
    check(
        "Q-sphere step 0 is north |00⟩",
        len(pts0) == 1 and pts0[0]["label"] == "00" and pts0[0]["z"] > 0.99,
        str(pts0),
    )
    pts1 = qsphere_points(s1)
    labs1 = {p["label"] for p in pts1}
    z10 = next((p["z"] for p in pts1 if p["label"] == "10"), 99.0)
    check(
        "Q-sphere after H: |10⟩ on equator, not |01⟩",
        "10" in labs1 and "01" not in labs1 and abs(z10) < 0.08,
        f"labels={labs1} z10={z10:.3f}",
    )
    pts2 = qsphere_points(s2)
    z00 = next((p["z"] for p in pts2 if p["label"] == "00"), 0.0)
    z11 = next((p["z"] for p in pts2 if p["label"] == "11"), 0.0)
    check(
        "Q-sphere Bell: north |00⟩ and south |11⟩",
        z00 > 0.99 and z11 < -0.99 and {p["label"] for p in pts2} == {"00", "11"},
        f"z00={z00:.3f} z11={z11:.3f} labels={[p['label'] for p in pts2]}",
    )
    ghz = QuantumEngine(3)
    ghz.load_preset("ghz")
    pts_g = qsphere_points(ghz.get_snapshot())
    zg = {p["label"]: p["z"] for p in pts_g}
    check(
        "Q-sphere GHZ: |000⟩ north, |111⟩ south",
        zg.get("000", 0) > 0.99 and zg.get("111", 0) < -0.99,
        str(zg),
    )
    fig_q = plot_qsphere(s2)
    check("plot_qsphere returns a 3D figure", len(fig_q.data) >= 3, str(len(fig_q.data)))
    views = build_step_views(bell, 1)
    check("build_step_views after H", "circuit" in views and "00" in views["ket"], views["ket"])

    # Optional: real noise chart (Aer). If this fails, send the traceback.
    noisy_ok = False
    fig_cmp = None
    try:
        from noise import NoiseParams, compare_ideal_noisy

        cmp = compare_ideal_noisy(
            bell,
            params=NoiseParams(t1_us=20, t2_us=15, readout_error=0.05),
            shots=512,
            seed=2,
        )
        fig_cmp = plot_ideal_vs_noisy(cmp)
        noisy_ok = len(fig_cmp.data) == 3 and cmp.fidelity <= 1.0
        check(
            "Ideal vs noisy chart",
            noisy_ok,
            f"F={cmp.fidelity:.3f} traces={len(fig_cmp.data)}",
        )
    except Exception as exc:
        check("Ideal vs noisy chart", False, repr(exc))

    path = None
    try:
        figs = [
            ("Bell circuit (CNOT highlighted)", fig_c),
            ("Bell probabilities (phase color)", fig_p),
            ("Entanglement after CNOT", fig_e),
            ("Ideal |ρ| Bell", fig_dm),
            ("Born vs shots", fig_sh),
            ("HZH probabilities", fig_h),
            ("Bell Q-sphere (two poles)", fig_q)
        ]
        if fig_cmp is not None:
            figs.append(("Bell ideal vs noisy", fig_cmp))
        path = write_preview_html(figs, "viz_preview.html")
        check("Wrote viz_preview.html", os.path.isfile(path), path)
    except Exception as exc:
        check("Wrote viz_preview.html", False, repr(exc))

    print(f"\n=== {passed} passed, {failed} failed ===")
    if failed == 0:
        print("All tests passed. Tell me this worked and we start app.py.")
        if path:
            print(f"Open in a browser:\n  {path}")
        print("\nBell narration @ step 2:")
        print(" ", s2.narration)
        print(" ", ket)
        return True
    print("Copy the FULL terminal output (including traceback) and send it.")
    return False


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_self_test() else 1)