"""
engine.py — Quantum State Visualizer
Ideal circuit, step snapshots, metrics. No Aer / no Streamlit.

Teaching bit order: |q0 q1 q2 ...>  (q0 LEFTMOST).
Qiskit is little-endian internally; we convert on every readout by labeling basis
states using q0 as the least-significant bit in the computational basis index.

This file provides:
- Generic gate builder with validation
- Step snapshots (step 0 = |0...0>)
- Amplitude/probability/phase + relative phase (global phase stripped)
- Entanglement meter (max single-qubit Von Neumann entropy; optional concurrence for 2q)
- Narration text per step
- CLI self-test: `python engine.py` -> must print "All tests passed."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Qiskit is required. From the project folder, with the venv on:\n"
        '  python -m pip install "qiskit>=1.0,<3" numpy'
    ) from exc


MAX_QUBITS = 5
EPS = 1e-10
ENTANGLE_THRESHOLD = 0.05

# Canonical names used everywhere after normalization
SINGLE_NO_PARAM = ("H", "X", "Y", "Z", "S", "SDG", "T", "TDG")
SINGLE_PARAM = ("RX", "RY", "RZ")
TWO_QUBIT = ("CNOT", "SWAP")
THREE_QUBIT = ("TOFFOLI",)

ALIASES = {
    "CX": "CNOT",
    "CXGATE": "CNOT",
    "CCX": "TOFFOLI",
    "CCNOT": "TOFFOLI",
    "TOF": "TOFFOLI",
    "S†": "SDG",
    "S_DAG": "SDG",
    "SDAG": "SDG",
    "S_DG": "SDG",
    "T†": "TDG",
    "T_DAG": "TDG",
    "TDAG": "TDG",
    "T_DG": "TDG",
}

GATE_CATALOG: dict[str, dict[str, Any]] = {
    "H": {"arity": 1, "params": 0, "title": "Hadamard", "blurb": "Creates superposition."},
    "X": {"arity": 1, "params": 0, "title": "Pauli-X", "blurb": "Bit flip |0⟩↔|1⟩."},
    "Y": {"arity": 1, "params": 0, "title": "Pauli-Y", "blurb": "Bit+phase flip (π about Y)."},
    "Z": {"arity": 1, "params": 0, "title": "Pauli-Z", "blurb": "Phase flip on |1⟩."},
    "S": {"arity": 1, "params": 0, "title": "S", "blurb": "Phase +π/2 on |1⟩."},
    "SDG": {"arity": 1, "params": 0, "title": "S†", "blurb": "Phase −π/2 on |1⟩."},
    "T": {"arity": 1, "params": 0, "title": "T", "blurb": "Phase +π/4 on |1⟩ (non-Clifford)."},
    "TDG": {"arity": 1, "params": 0, "title": "T†", "blurb": "Phase −π/4 on |1⟩."},
    "RX": {"arity": 1, "params": 1, "title": "RX(θ)", "blurb": "Rotate around X by θ."},
    "RY": {"arity": 1, "params": 1, "title": "RY(θ)", "blurb": "Rotate around Y by θ."},
    "RZ": {"arity": 1, "params": 1, "title": "RZ(θ)", "blurb": "Rotate around Z by θ (phase)."},
    "CNOT": {"arity": 2, "params": 0, "title": "CNOT", "blurb": "Flip target iff control=1."},
    "SWAP": {"arity": 2, "params": 0, "title": "SWAP", "blurb": "Swap two qubits."},
    "TOFFOLI": {"arity": 3, "params": 0, "title": "Toffoli", "blurb": "Flip target iff both controls=1."},
}


def normalize_gate_name(name: str) -> str:
    key = name.strip().upper().replace(" ", "")
    return ALIASES.get(key, key)


def textbook_label(index: int, n_qubits: int) -> str:
    """
    Integer basis index -> |q0 q1 q2 ...> string (q0 leftmost).
    We map q0 to the least significant bit of the index.
    """
    return "".join(str((index >> q) & 1) for q in range(n_qubits))


def one_qubit_matrix(name: str, theta: float | None = None) -> np.ndarray:
    """2×2 matrix for a sidebar cheat-sheet (teaching; not used for simulation)."""
    n = normalize_gate_name(name)
    s2 = 1 / np.sqrt(2)
    if n == "H":
        return np.array([[s2, s2], [s2, -s2]], dtype=complex)
    if n == "X":
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if n == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if n == "Z":
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if n == "S":
        return np.array([[1, 0], [0, 1j]], dtype=complex)
    if n == "SDG":
        return np.array([[1, 0], [0, -1j]], dtype=complex)
    if n == "T":
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    if n == "TDG":
        return np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)

    if theta is None:
        theta = np.pi

    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if n == "RX":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if n == "RY":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if n == "RZ":
        return np.array(
            [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
            dtype=complex,
        )
    raise ValueError(f"No 1-qubit matrix for {name}")


def _von_neumann_entropy_bits(rho: np.ndarray) -> float:
    """Entropy S(ρ) in bits for a density matrix ρ."""
    evals = np.linalg.eigvals(rho)
    evals = np.real_if_close(evals, tol=1e-10).astype(float)
    evals = np.clip(evals, 0.0, 1.0)
    nz = evals[evals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log2(nz)))


def _concurrence_2q(rho4: np.ndarray) -> float:
    """
    Wootters concurrence for a 2-qubit density matrix (4x4).
    Returns in [0, 1].
    """
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    yy = np.kron(sy, sy)
    r_tilde = yy @ rho4.conj() @ yy
    R = rho4 @ r_tilde
    evals = np.linalg.eigvals(R)
    # sort descending by real part (numerical noise can add tiny imaginary components)
    evals = np.real_if_close(evals, tol=1e-10)
    evals = np.real(evals)
    evals = np.clip(evals, 0.0, None)
    s = np.sort(np.sqrt(evals))[::-1]
    c = float(max(0.0, s[0] - s[1] - s[2] - s[3]))
    return float(min(1.0, max(0.0, c)))


def _relative_phases(
    amps: dict[str, complex], probs: dict[str, float]
) -> dict[str, float | None]:
    """Strip global phase using the first significant basis state (sorted label)."""
    ref = 0.0
    for lab in sorted(amps):
        if probs[lab] > EPS:
            ref = float(np.angle(amps[lab]))
            break
    out: dict[str, float | None] = {}
    for lab, a in amps.items():
        if probs[lab] > EPS:
            ph = float(np.angle(a * np.exp(-1j * ref)))
            if abs(ph) < 1e-8:
                ph = 0.0
            out[lab] = ph
        else:
            out[lab] = None
    return out


def decompose_statevector(
    sv: Statevector, n_qubits: int
) -> tuple[dict[str, complex], dict[str, float], dict[str, float | None], dict[str, float | None]]:
    data = np.asarray(sv.data, dtype=complex)
    amps: dict[str, complex] = {}
    probs: dict[str, float] = {}
    phases: dict[str, float | None] = {}

    for i, a in enumerate(data):
        lab = textbook_label(i, n_qubits)
        amps[lab] = complex(a)
        p = float(abs(a) ** 2)
        if p < 1e-16:
            p = 0.0
        probs[lab] = p
        phases[lab] = float(np.angle(a)) if p > EPS else None

    total = sum(probs.values())
    if total > 0 and abs(total - 1.0) > 1e-8:
        probs = {k: v / total for k, v in probs.items()}

    rel = _relative_phases(amps, probs)
    return amps, probs, phases, rel


def entanglement_metrics(sv: Statevector, n_qubits: int) -> dict[str, Any]:
    """
    Pure-state entanglement meter.
    We compute the reduced density matrix of each single qubit and take the
    maximum Von Neumann entropy across qubits (bits).

    For Bell/GHZ: ~1
    For product states: ~0
    """
    if n_qubits < 2:
        return {
            "entropy": 0.0,
            "qubit_entropies": [0.0],
            "concurrence": None,
            "entangled": False,
        }

    dm = DensityMatrix(sv)
    entropies: list[float] = []
    for q in range(n_qubits):
        traced = [i for i in range(n_qubits) if i != q]
        rho = partial_trace(dm, traced)  # keeps qubit q
        s = _von_neumann_entropy_bits(np.asarray(rho.data, dtype=complex))
        s = 0.0 if s < 1e-10 else min(1.0, max(0.0, float(s)))
        entropies.append(float(s))

    max_s = float(max(entropies)) if entropies else 0.0

    conc = None
    if n_qubits == 2:
        try:
            conc = _concurrence_2q(np.asarray(dm.data, dtype=complex))
        except Exception:
            conc = None

    return {
        "entropy": max_s,
        "qubit_entropies": entropies,
        "concurrence": conc,
        "entangled": max_s > ENTANGLE_THRESHOLD,
    }


def top_states_text(probs: dict[str, float], min_p: float = 0.02, k: int = 4) -> str:
    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    parts = [f"|{lab}⟩ {p * 100:.1f}%" for lab, p in items if p >= min_p][:k]
    return ", ".join(parts) if parts else "all amplitudes ~ 0"


def apply_operation(qc: QuantumCircuit, op: "Operation") -> None:
    n, qs, pr = op.name, op.qubits, op.params

    if n == "H":
        qc.h(qs[0])
    elif n == "X":
        qc.x(qs[0])
    elif n == "Y":
        qc.y(qs[0])
    elif n == "Z":
        qc.z(qs[0])
    elif n == "S":
        qc.s(qs[0])
    elif n == "SDG":
        qc.sdg(qs[0])
    elif n == "T":
        qc.t(qs[0])
    elif n == "TDG":
        qc.tdg(qs[0])
    elif n == "RX":
        qc.rx(pr[0], qs[0])
    elif n == "RY":
        qc.ry(pr[0], qs[0])
    elif n == "RZ":
        qc.rz(pr[0], qs[0])
    elif n == "CNOT":
        qc.cx(qs[0], qs[1])  # control, target
    elif n == "SWAP":
        qc.swap(qs[0], qs[1])
    elif n == "TOFFOLI":
        qc.ccx(qs[0], qs[1], qs[2])  # c1, c2, target
    else:
        raise ValueError(f"Unsupported gate in apply_operation: {n}")


@dataclass
class Operation:
    name: str
    qubits: list[int]
    params: list[float] = field(default_factory=list)

    def display_label(self) -> str:
        q = self.qubits
        if self.name in SINGLE_PARAM:
            th = self.params[0] if self.params else 0.0
            turns = th / np.pi
            return f"{self.name}(q{q[0]}, θ={turns:.3g}π)"
        if self.name == "CNOT":
            return f"CNOT(q{q[0]} → q{q[1]})"
        if self.name == "SWAP":
            return f"SWAP(q{q[0]}, q{q[1]})"
        if self.name == "TOFFOLI":
            return f"TOFFOLI(c={q[0]},{q[1]} t={q[2]})"
        return f"{self.name}(q{q[0]})"


@dataclass
class Snapshot:
    step: int
    label: str
    operation: Optional[Operation]
    amplitudes: dict[str, complex]
    probabilities: dict[str, float]
    phases: dict[str, float | None]
    phases_relative: dict[str, float | None]
    entanglement: float
    qubit_entropies: list[float]
    concurrence: float | None
    entangled: bool
    narration: str
    depth: int
    gate_count: int

    def to_display(self) -> dict[str, Any]:
        rows = []
        for lab in sorted(self.probabilities):
            p = self.probabilities[lab]
            a = self.amplitudes[lab]
            rows.append(
                {
                    "state": lab,
                    "re": float(np.real(a)),
                    "im": float(np.imag(a)),
                    "prob": float(p),
                    "percent": float(p) * 100.0,
                    "phase_rad": self.phases[lab],
                    "phase_rel_rad": self.phases_relative[lab],
                }
            )
        return {
            "step": self.step,
            "label": self.label,
            "narration": self.narration,
            "entanglement": self.entanglement,
            "entangled": self.entangled,
            "concurrence": self.concurrence,
            "depth": self.depth,
            "gate_count": self.gate_count,
            "rows": rows,
        }


class QuantumEngine:
    """Ideal circuit + one snapshot after every gate (step 0 = |0...0⟩)."""

    def __init__(self, n_qubits: int = 2):
        self.n_qubits = 0
        self.operations: list[Operation] = []
        self.snapshots: list[Snapshot] = []
        self.set_n_qubits(n_qubits)

    # ------------------------------------------------------------------ setup
    def set_n_qubits(self, n: int, clear: bool = True) -> tuple[bool, str]:
        if not isinstance(n, int) or n < 1 or n > MAX_QUBITS:
            return False, f"n_qubits must be an integer 1–{MAX_QUBITS}."
        self.n_qubits = n
        if clear:
            self.operations = []
        else:
            self.operations = [op for op in self.operations if self._qubits_ok(op.qubits)]
        self._rebuild()
        return True, f"Using {n} qubit(s)."

    def clear(self) -> None:
        self.operations = []
        self._rebuild()

    def undo(self) -> tuple[bool, str]:
        if not self.operations:
            return False, "Nothing to undo."
        removed = self.operations.pop()
        self._rebuild()
        return True, f"Removed {removed.display_label()}."

    # ----------------------------------------------------------------- gates
    def add_gate(
        self,
        name: str,
        qubits: int | list[int],
        params: float | list[float] | None = None,
    ) -> tuple[bool, str]:
        ok, msg, op = self._make_operation(name, qubits, params)
        if not ok or op is None:
            return False, msg
        self.operations.append(op)
        self._rebuild()
        return True, f"Added {op.display_label()}."
    def set_last_rotation_angle(self, theta: float) -> tuple[bool, str]:
        """Live-update θ on the last RX/RY/RZ. No extra gate is appended."""
        if not self.operations:
            return False, "No gate to update."
        op = self.operations[-1]
        if op.name not in SINGLE_PARAM:
            return False, "Last gate has no angle."
        th = float(theta)
        if op.params and abs(op.params[0] - th) < 1e-15:
            return True, op.display_label()
        op.params = [th]
        self._rebuild()
        return True, f"Updated {op.display_label()}."
    def load_preset(self, name: str) -> tuple[bool, str]:
        key = name.strip().lower().replace(" ", "_").replace("-", "_")
        presets = {
            "superposition": (1, [("H", [0], None)]),
            "hzh": (1, [("H", [0], None), ("Z", [0], None), ("H", [0], None)]),
            "t_phase": (1, [("H", [0], None), ("T", [0], None)]),
            "t_demo": (1, [("H", [0], None), ("T", [0], None), ("H", [0], None)]),
            "rx_pi": (1, [("RX", [0], np.pi)]),
            "bell": (2, [("H", [0], None), ("CNOT", [0, 1], None)]),
            "ghz": (3, [("H", [0], None), ("CNOT", [0, 1], None), ("CNOT", [1, 2], None)]),
            "toffoli": (3, [("X", [0], None), ("X", [1], None), ("TOFFOLI", [0, 1, 2], None)]),
            "swap_demo": (2, [("X", [0], None), ("SWAP", [0, 1], None)]),
        }
        if key not in presets:
            return False, f"Unknown preset '{name}'. Try: {', '.join(presets)}."

        n, ops = presets[key]
        self.set_n_qubits(n, clear=True)
        for g, qs, pr in ops:
            ok, msg = self.add_gate(g, qs, pr)
            if not ok:
                return False, msg
        return True, f"Loaded preset '{key}' ({n} qubits, {len(ops)} gates)."

    # ---------------------------------------------------------------- circuit
    def to_circuit(self, up_to_step: int | None = None) -> QuantumCircuit:
        if up_to_step is None:
            up_to_step = len(self.operations)
        up_to_step = max(0, min(up_to_step, len(self.operations)))
        qc = QuantumCircuit(self.n_qubits, name="visualizer")
        for op in self.operations[:up_to_step]:
            apply_operation(qc, op)
        return qc

    def to_qasm(self, up_to_step: int | None = None) -> str:
        qc = self.to_circuit(up_to_step)
        try:
            from qiskit.qasm2 import dumps

            return dumps(qc)
        except Exception:
            try:
                return qc.qasm()  # type: ignore[attr-defined]
            except Exception:
                return str(qc)

    def gate_count(self) -> int:
        return len(self.operations)

    def depth(self) -> int:
        return int(self.to_circuit().depth() or 0)

    # -------------------------------------------------------------- snapshots
    @property
    def max_step(self) -> int:
        return len(self.operations)

    def get_snapshot(self, step: int | None = None) -> Snapshot:
        if not self.snapshots:
            self._rebuild()
        if step is None:
            step = self.max_step
        step = max(0, min(int(step), self.max_step))
        return self.snapshots[step]

    def sample_counts(
        self,
        shots: int = 1024,
        step: int | None = None,
        seed: int | None = None,
    ) -> dict[str, int]:
        """Exact Born-rule sampling of the ideal state (no hardware noise)."""
        snap = self.get_snapshot(step)
        labels = list(snap.probabilities.keys())
        p = np.array([snap.probabilities[k] for k in labels], dtype=float)
        s = float(p.sum())
        if s <= 0:
            return {}
        p = p / s
        rng = np.random.default_rng(seed)
        draws = rng.choice(len(labels), size=int(shots), p=p)
        counts: dict[str, int] = {}
        for i in draws:
            lab = labels[int(i)]
            counts[lab] = counts.get(lab, 0) + 1
        return counts

    def measure_once(self, step: int | None = None, seed: int | None = None) -> str:
        counts = self.sample_counts(shots=1, step=step, seed=seed)
        return next(iter(counts))

    def fidelity_between_steps(self, step_a: int, step_b: int) -> float:
        sv_a = Statevector.from_instruction(self.to_circuit(step_a))
        sv_b = Statevector.from_instruction(self.to_circuit(step_b))
        return float(np.real(state_fidelity(sv_a, sv_b)))

    def statevector_at(self, step: int | None = None) -> Statevector:
        if step is None:
            step = self.max_step
        return Statevector.from_instruction(self.to_circuit(step))

    # --------------------------------------------------------------- internal
    def _qubits_ok(self, qubits: list[int]) -> bool:
        return all(0 <= q < self.n_qubits for q in qubits)

    def _make_operation(
        self,
        name: str,
        qubits: int | list[int],
        params: float | list[float] | None,
    ) -> tuple[bool, str, Optional[Operation]]:
        n = normalize_gate_name(name)
        if n not in GATE_CATALOG:
            known = ", ".join(GATE_CATALOG)
            return False, f"Unknown gate '{name}'. Use: {known}.", None

        if isinstance(qubits, (int, np.integer)):
            qubits = [int(qubits)]
        else:
            qubits = [int(q) for q in qubits]

        spec = GATE_CATALOG[n]
        if len(qubits) != spec["arity"]:
            return False, f"{n} needs {spec['arity']} qubit index(es), got {len(qubits)}.", None
        if len(set(qubits)) != len(qubits):
            return False, "A gate cannot use the same qubit twice.", None
        if not self._qubits_ok(qubits):
            return False, f"Qubit index out of range. Valid: 0–{self.n_qubits - 1}.", None

        plist: list[float] = []
        if spec["params"]:
            if params is None:
                plist = [float(np.pi)]
            elif isinstance(params, (int, float, np.floating)):
                plist = [float(params)]
            else:
                plist = [float(x) for x in params]
            if len(plist) != spec["params"]:
                return False, f"{n} needs {spec['params']} angle(s).", None

        return True, "ok", Operation(name=n, qubits=qubits, params=plist)

    def _narrate(
        self,
        op: Optional[Operation],
        prev_ent: float,
        metrics: dict[str, Any],
        probs: dict[str, float],
    ) -> str:
        top = top_states_text(probs)
        if op is None:
            ket = "0" * self.n_qubits
            return (
                f"Step 0 — register prepared in |{ket}⟩ (100%). "
                "No gates yet. Entanglement = 0."
            )

        q = op.qubits
        n = op.name
        if n == "H":
            core = f"Hadamard on q{q[0]} creates superposition."
        elif n == "X":
            core = f"Pauli-X on q{q[0]} flips |0⟩↔|1⟩."
        elif n == "Y":
            core = f"Pauli-Y on q{q[0]}: bit flip plus phase."
        elif n == "Z":
            core = f"Pauli-Z on q{q[0]} flips phase of |1⟩ (probabilities may not change)."
        elif n == "S":
            core = f"S on q{q[0]} applies a +π/2 phase to |1⟩."
        elif n == "SDG":
            core = f"S† on q{q[0]} applies a −π/2 phase to |1⟩."
        elif n == "T":
            core = f"T on q{q[0]} applies a +π/4 phase to |1⟩ (non-Clifford)."
        elif n == "TDG":
            core = f"T† on q{q[0]} applies a −π/4 phase to |1⟩."
        elif n == "RX":
            th = op.params[0] / np.pi
            core = f"RX({th:.3g}π) on q{q[0]} rotates around X."
        elif n == "RY":
            th = op.params[0] / np.pi
            core = f"RY({th:.3g}π) on q{q[0]} rotates around Y."
        elif n == "RZ":
            th = op.params[0] / np.pi
            core = f"RZ({th:.3g}π) on q{q[0]} rotates around Z (phase-only)."
        elif n == "CNOT":
            core = f"CNOT: if control q{q[0]} is |1⟩, target q{q[1]} flips."
        elif n == "SWAP":
            core = f"SWAP exchanges q{q[0]} and q{q[1]}."
        elif n == "TOFFOLI":
            core = f"Toffoli flips target q{q[2]} iff q{q[0]} and q{q[1]} are both |1⟩."
        else:
            core = op.display_label()

        ent = float(metrics["entropy"])
        extra = ""
        if self.n_qubits >= 2:
            if (prev_ent <= ENTANGLE_THRESHOLD) and metrics["entangled"]:
                extra = f" Entanglement {prev_ent:.2f} → {ent:.2f} (ENTANGLED)."
            else:
                extra = f" Entanglement = {ent:.2f}."
        return f"{core} Now: {top}.{extra}"

    def _rebuild(self) -> None:
        n = self.n_qubits
        snaps: list[Snapshot] = []
        prev_ent = 0.0

        for step in range(0, len(self.operations) + 1):
            qc = self.to_circuit(step)
            sv = Statevector.from_label("0" * n).evolve(qc)

            amps, probs, phases, rel = decompose_statevector(sv, n)
            metrics = entanglement_metrics(sv, n)

            op = None if step == 0 else self.operations[step - 1]
            label = "Initial |0…0⟩" if op is None else op.display_label()
            narration = self._narrate(op, prev_ent, metrics, probs)
            depth = int(qc.depth() or 0)

            snaps.append(
                Snapshot(
                    step=step,
                    label=label,
                    operation=op,
                    amplitudes=amps,
                    probabilities=probs,
                    phases=phases,
                    phases_relative=rel,
                    entanglement=float(metrics["entropy"]),
                    qubit_entropies=list(metrics["qubit_entropies"]),
                    concurrence=metrics["concurrence"],
                    entangled=bool(metrics["entangled"]),
                    narration=narration,
                    depth=depth,
                    gate_count=step,
                )
            )
            prev_ent = float(metrics["entropy"])

        self.snapshots = snaps


# ===================================================================== tests

def _close(a: float, b: float, tol: float = 0.02) -> bool:
    return abs(float(a) - float(b)) <= tol


def _prob(engine: QuantumEngine, ket: str, step: int | None = None) -> float:
    return float(engine.get_snapshot(step).probabilities.get(ket, 0.0))


def run_self_test() -> bool:
    print("=== QuantumEngine self-test (textbook |q0 q1 …⟩) ===\n")
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

    # 1. H
    e = QuantumEngine(1)
    e.add_gate("H", 0)
    check(
        "H superposition",
        _close(_prob(e, "0"), 0.5) and _close(_prob(e, "1"), 0.5),
        f"|0|={_prob(e,'0'):.3f} |1|={_prob(e,'1'):.3f}",
    )

    # 2. X
    e = QuantumEngine(1)
    e.add_gate("X", 0)
    check(
        "X bit flip",
        _close(_prob(e, "1"), 1.0) and _close(_prob(e, "0"), 0.0),
        f"|1|={_prob(e,'1'):.3f}",
    )

    # 3. RX(π) probabilities match X
    e = QuantumEngine(1)
    e.add_gate("RX", 0, np.pi)
    check(
        "RX(π) ≈ X (probabilities)",
        _close(_prob(e, "1"), 1.0),
        f"|1|={_prob(e,'1'):.3f} (phase may differ; OK)",
    )

    # 4. Bell
    e = QuantumEngine(2)
    e.load_preset("bell")
    s = e.get_snapshot()
    check(
        "Bell probabilities",
        _close(_prob(e, "00"), 0.5)
        and _close(_prob(e, "11"), 0.5)
        and _close(_prob(e, "01"), 0.0)
        and _close(_prob(e, "10"), 0.0),
        f"|00|={_prob(e,'00'):.3f} |11|={_prob(e,'11'):.3f} leak01={_prob(e,'01'):.3f} leak10={_prob(e,'10'):.3f}",
    )
    check(
        "Bell entanglement ~ 1",
        s.entangled and _close(s.entanglement, 1.0, 0.05),
        f"S={s.entanglement:.3f} concurrence={s.concurrence}",
    )
    s1 = e.get_snapshot(1)
    check(
        "After H only, still separable",
        (not s1.entangled) and _close(_prob(e, "00", 1), 0.5) and _close(_prob(e, "10", 1), 0.5),
        f"step1 |00|={_prob(e,'00',1):.3f} |10|={_prob(e,'10',1):.3f} S={s1.entanglement:.3f}",
    )

    # 5. GHZ
    e = QuantumEngine(3)
    e.load_preset("ghz")
    check(
        "GHZ probabilities",
        _close(_prob(e, "000"), 0.5) and _close(_prob(e, "111"), 0.5),
        f"|000|={_prob(e,'000'):.3f} |111|={_prob(e,'111'):.3f} S={e.get_snapshot().entanglement:.3f}",
    )

    # 6. Toffoli |110> → |111>
    e = QuantumEngine(3)
    e.load_preset("toffoli")
    check(
        "Toffoli |110⟩ → |111⟩",
        _close(_prob(e, "111"), 1.0),
        f"|111|={_prob(e,'111'):.3f}",
    )

    # 7. SWAP demo: X(q0) then SWAP(q0,q1): |10⟩ → |01⟩
    e = QuantumEngine(2)
    e.load_preset("swap_demo")
    check(
        "SWAP demo",
        _close(_prob(e, "01"), 1.0),
        f"|01|={_prob(e,'01'):.3f} |10|={_prob(e,'10'):.3f}",
    )

    # 8. HZH ≡ X on |0>
    e = QuantumEngine(1)
    e.load_preset("hzh")
    check("HZH |0⟩ → |1⟩ (same as X)", _close(_prob(e, "1"), 1.0), f"|1|={_prob(e,'1'):.3f}")

    # 9. Validation
    e = QuantumEngine(2)
    ok, msg = e.add_gate("CNOT", [0, 0])
    check("Reject CNOT with duplicate qubits", ok is False, msg)
    ok, msg = e.add_gate("TOFFOLI", [0, 1, 2])
    check("Reject Toffoli on 2 qubits", ok is False, msg)
    ok, msg = e.add_gate("H", 0)
    check("Accept H", ok is True, msg)

    # 10. Sampling sums to shots
    e = QuantumEngine(1)
    e.add_gate("H", 0)
    counts = e.sample_counts(shots=200, seed=7)
    check("sample_counts sums to shots", sum(counts.values()) == 200, str(counts))

    print(f"\n=== {passed} passed, {failed} failed ===")
    if failed == 0:
        print("All tests passed.")
        return True
    return False


# Backward/typo compatibility if anything calls the plural name
run_self_tests = run_self_test


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_self_test() else 1)
