"""
noise.py — Quantum State Visualizer
Configurable T1 / T2 / readout noise via Qiskit Aer.

These numbers are SIMULATION PARAMETERS, not calibration from a
specific IBM chip. The UI must say that.

Quantum noise (T1/T2) is applied to pulse-like gates only:
  1q pulses: H, X, Y, RX, RY
  2q:        CNOT, SWAP
  3q:        Toffoli
Virtual-Z family (Z, S, S†, T, T†, RZ) is treated as zero-duration,
which matches how superconducting hardware usually implements them.

Readout error is a *measurement assignment* error. It shows up in
shot histograms, not in state fidelity.

Teaching bit order matches engine.py: |q0 q1 ...> with q0 LEFTMOST.
Aer count strings are reversed into that convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Qiskit is required. With the venv on:\n"
        '  pip install "qiskit>=1.0,<3" numpy'
    ) from exc

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "qiskit-aer is required for noise.py. With the venv on:\n"
        "  pip install qiskit-aer"
    ) from exc

from engine import QuantumEngine, textbook_label


# Pulse-like instructions (Qiskit names) that get thermal relaxation.
PULSE_1Q = ("h", "x", "y", "rx", "ry", "id", "sx", "u", "u2", "u3")
PULSE_2Q = ("cx", "swap", "cz")
PULSE_3Q = ("ccx",)

# Keep these un-decomposed so one user gate ≈ one noise event.
BASIS_GATES = [
    "h", "x", "y", "z", "s", "sdg", "t", "tdg",
    "rx", "ry", "rz", "id", "sx", "p", "u", "u1", "u2", "u3",
    "cx", "swap", "cz", "ccx",
    "measure", "reset", "barrier", "delay",
]


def _as_percent_prob(p: float) -> float:
    """Accept 0.02 or 2 (meaning 2%) and clip to [0, 0.5]."""
    x = float(p)
    if x > 1.0:
        x = x / 100.0
    return min(0.5, max(0.0, x))


def _tensor_copies(err, n: int):
    out = err
    for _ in range(n - 1):
        nxt = err.copy() if hasattr(err, "copy") else err
        if hasattr(out, "tensor"):
            out = out.tensor(nxt)
        else:
            out = out.expand(nxt)
    return out


def qiskit_bitstring_to_textbook(bitstr: str, n_qubits: int) -> str:
    bits = str(bitstr).replace(" ", "").replace("_", "")
    if len(bits) > n_qubits:
        bits = bits[-n_qubits:]
    bits = bits.zfill(n_qubits)
    return bits[::-1]


def counts_to_textbook(counts: dict, n_qubits: int) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, val in dict(counts).items():
        lab = qiskit_bitstring_to_textbook(key, n_qubits)
        out[lab] = out.get(lab, 0) + int(val)
    return out


def counts_to_probs(counts: dict[str, int]) -> dict[str, float]:
    tot = float(sum(counts.values()))
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in counts.items()}


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def support_leakage(
    ideal: dict[str, float],
    noisy: dict[str, float],
    thresh: float = 0.02,
) -> float:
    support = {k for k, v in ideal.items() if v >= thresh}
    return float(sum(v for k, v in noisy.items() if k not in support))


def dm_to_probs(dm: DensityMatrix, n_qubits: int) -> dict[str, float]:
    rho = np.asarray(dm.data, dtype=complex)
    diag = np.real(np.diag(rho))
    diag = np.clip(diag, 0.0, None)
    s = float(diag.sum())
    if s > 0:
        diag = diag / s
    return {textbook_label(i, n_qubits): float(diag[i]) for i in range(2**n_qubits)}


def _result_payload(result) -> dict:
    for getter in (
        lambda: dict(result.data(0)),
        lambda: dict(result.data()),
        lambda: dict(result.results[0].data),
    ):
        try:
            return getter()
        except Exception:
            continue
    raise RuntimeError("Could not read Aer result data().")


def _extract_density_matrix(result) -> DensityMatrix:
    data = _result_payload(result)
    for key in ("density_matrix", "dm", "rho"):
        if key in data:
            return DensityMatrix(data[key])
    raise RuntimeError(
        "Aer result has no density matrix. Keys: " + ", ".join(map(str, data.keys()))
    )


def _attach_save_dm(qc: QuantumCircuit) -> QuantumCircuit:
    out = qc.copy()
    n = out.num_qubits
    try:
        from qiskit_aer.library import SaveDensityMatrix

        out.append(SaveDensityMatrix(n, label="density_matrix"), list(range(n)))
        return out
    except Exception:
        pass
    if hasattr(out, "save_density_matrix"):
        out.save_density_matrix(label="density_matrix")
        return out
    raise RuntimeError("Cannot save density matrix. Upgrade qiskit-aer.")


def _circuit_with_measures(qc: QuantumCircuit) -> QuantumCircuit:
    n = qc.num_qubits
    out = QuantumCircuit(n, n, name=qc.name or "noisy_meas")
    if n:
        out.compose(qc, qubits=list(range(n)), inplace=True)
        out.measure(range(n), range(n))
    return out


def _add_errors(nm: NoiseModel, err, gates: tuple[str, ...] | list[str]) -> list[str]:
    added: list[str] = []
    for g in gates:
        e = err.copy() if hasattr(err, "copy") else err
        try:
            nm.add_all_qubit_quantum_error(e, [g])
            added.append(g)
        except Exception:
            continue
    return added


@dataclass
class NoiseParams:
    """Physical knobs. All times in microseconds."""

    enabled: bool = True
    t1_us: float = 50.0
    t2_us: float = 30.0
    readout_error: float = 0.02
    gate_time_1q_us: float = 0.10
    gate_time_2q_us: float = 0.40
    gate_time_3q_us: float = 1.00

    def clamped(self) -> tuple["NoiseParams", list[str]]:
        notes: list[str] = []
        t1 = max(float(self.t1_us), 1e-6)
        t2 = max(float(self.t2_us), 1e-6)
        if t2 > 2.0 * t1:
            t2 = 2.0 * t1
            notes.append(f"T2 clamped to 2·T1 = {t2:.4g} μs (Aer requirement).")
        ro = _as_percent_prob(self.readout_error)
        if ro != float(self.readout_error) and float(self.readout_error) > 1.0:
            notes.append(f"Readout {self.readout_error} interpreted as {ro * 100:.2f}%.")
        out = NoiseParams(
            enabled=bool(self.enabled),
            t1_us=t1,
            t2_us=t2,
            readout_error=ro,
            gate_time_1q_us=max(float(self.gate_time_1q_us), 0.0),
            gate_time_2q_us=max(float(self.gate_time_2q_us), 0.0),
            gate_time_3q_us=max(float(self.gate_time_3q_us), 0.0),
        )
        return out, notes


@dataclass
class ComparisonResult:
    n_qubits: int
    shots: int
    params: NoiseParams
    warnings: list[str]
    ideal_probs: dict[str, float]
    noisy_state_probs: dict[str, float]
    noisy_counts: dict[str, int]
    noisy_shot_probs: dict[str, float]
    fidelity: float
    purity: float
    total_variation_state: float
    total_variation_shots: float
    leakage_state: float
    leakage_shots: float
    caption: str

    def to_display(self) -> dict[str, Any]:
        labels = sorted(set(self.ideal_probs) | set(self.noisy_state_probs) | set(self.noisy_shot_probs))
        rows = []
        for lab in labels:
            rows.append(
                {
                    "state": lab,
                    "ideal": self.ideal_probs.get(lab, 0.0),
                    "noisy_state": self.noisy_state_probs.get(lab, 0.0),
                    "noisy_shots": self.noisy_shot_probs.get(lab, 0.0),
                }
            )
        return {
            "fidelity": self.fidelity,
            "purity": self.purity,
            "tv_state": self.total_variation_state,
            "tv_shots": self.total_variation_shots,
            "leakage_state": self.leakage_state,
            "leakage_shots": self.leakage_shots,
            "caption": self.caption,
            "warnings": self.warnings,
            "rows": rows,
        }


def build_noise_model(params: NoiseParams, include_readout: bool = True) -> tuple[NoiseModel, list[str]]:
    p, notes = params.clamped()
    nm = NoiseModel()

    t1, t2 = p.t1_us, p.t2_us
    if p.gate_time_1q_us > 0:
        err_1q = thermal_relaxation_error(t1, t2, p.gate_time_1q_us)
        added = _add_errors(nm, err_1q, PULSE_1Q)
        if not added:
            notes.append("Warning: no 1-qubit thermal errors were attached.")
    if p.gate_time_2q_us > 0:
        e = thermal_relaxation_error(t1, t2, p.gate_time_2q_us)
        err_2q = _tensor_copies(e, 2)
        _add_errors(nm, err_2q, PULSE_2Q)
    if p.gate_time_3q_us > 0:
        e = thermal_relaxation_error(t1, t2, p.gate_time_3q_us)
        err_3q = _tensor_copies(e, 3)
        _add_errors(nm, err_3q, PULSE_3Q)

    if include_readout and p.readout_error > 0:
        r = p.readout_error
        ro = ReadoutError([[1.0 - r, r], [r, 1.0 - r]])
        try:
            nm.add_all_qubit_readout_error(ro)
        except Exception as exc:
            notes.append(f"Readout error not attached: {exc}")

    return nm, notes


def _make_simulator(noise_model: NoiseModel, method: str) -> AerSimulator:
    kwargs: dict[str, Any] = {"method": method, "noise_model": noise_model}
    try:
        return AerSimulator(basis_gates=list(BASIS_GATES), **kwargs)
    except Exception:
        return AerSimulator(**kwargs)


def _transpile_for(sim: AerSimulator, qc: QuantumCircuit) -> QuantumCircuit:
    try:
        return transpile(
            qc,
            backend=sim,
            optimization_level=0,
            basis_gates=list(BASIS_GATES),
        )
    except Exception:
        try:
            return transpile(qc, backend=sim, optimization_level=0)
        except Exception:
            return qc


def run_noisy_density_matrix(
    qc: QuantumCircuit,
    params: NoiseParams,
) -> tuple[DensityMatrix, list[str]]:
    """T1/T2 only (no readout). Deterministic mixed state after the circuit."""
    nm, notes = build_noise_model(params, include_readout=False)
    saved = _attach_save_dm(qc)
    sim = _make_simulator(nm, "density_matrix")
    tqc = _transpile_for(sim, saved)
    result = sim.run(tqc, shots=1).result()
    if not result.success:
        raise RuntimeError(f"Density-matrix sim failed: {result.status}")
    return _extract_density_matrix(result), notes


def run_noisy_shots(
    qc: QuantumCircuit,
    params: NoiseParams,
    shots: int = 2048,
    seed: int | None = 42,
) -> tuple[dict[str, int], list[str]]:
    """T1/T2 + readout. Returns textbook-order counts."""
    if shots <= 0:
        return {}, []
    nm, notes = build_noise_model(params, include_readout=True)
    meas = _circuit_with_measures(qc)
    sim = _make_simulator(nm, "automatic")
    tqc = _transpile_for(sim, meas)
    run_kw: dict[str, Any] = {"shots": int(shots)}
    if seed is not None:
        run_kw["seed_simulator"] = int(seed)
    result = sim.run(tqc, **run_kw).result()
    if not result.success:
        raise RuntimeError(f"Shot sim failed: {result.status}")
    raw = result.get_counts()
    if isinstance(raw, list):
        raw = raw[0]
    return counts_to_textbook(raw, qc.num_qubits), notes


def _caption(params: NoiseParams, fid: float, purity: float, leak: float) -> str:
    if not params.enabled:
        return "Noise is OFF. Histograms follow the exact ideal state (Born rule)."
    p, _ = params.clamped()
    return (
        f"Hardware-style model (not a specific IBM chip): "
        f"T1={p.t1_us:g} μs, T2={p.t2_us:g} μs, "
        f"readout={p.readout_error * 100:.2f}%. "
        f"State fidelity vs ideal = {fid:.3f}, purity = {purity:.3f}, "
        f"leakage = {leak * 100:.1f}%."
    )


def compare_ideal_noisy(
    engine: QuantumEngine,
    step: int | None = None,
    params: NoiseParams | None = None,
    shots: int = 2048,
    seed: int | None = 42,
) -> ComparisonResult:
    """Main API for the app: one ideal snapshot vs one noisy run of that prefix."""
    if params is None:
        params = NoiseParams()
    p, clamp_notes = params.clamped()
    snap = engine.get_snapshot(step)
    n = engine.n_qubits
    ideal_probs = dict(snap.probabilities)
    qc = engine.to_circuit(snap.step)

    if not p.enabled:
        counts = engine.sample_counts(shots=max(shots, 0) or 0, step=snap.step, seed=seed)
        shot_probs = counts_to_probs(counts) if counts else dict(ideal_probs)
        return ComparisonResult(
            n_qubits=n,
            shots=int(shots),
            params=p,
            warnings=clamp_notes,
            ideal_probs=ideal_probs,
            noisy_state_probs=dict(ideal_probs),
            noisy_counts=counts,
            noisy_shot_probs=shot_probs,
            fidelity=1.0,
            purity=1.0,
            total_variation_state=0.0,
            total_variation_shots=total_variation(ideal_probs, shot_probs),
            leakage_state=0.0,
            leakage_shots=support_leakage(ideal_probs, shot_probs),
            caption=_caption(p, 1.0, 1.0, 0.0),
        )

    dm, notes_dm = run_noisy_density_matrix(qc, p)
    noisy_state = dm_to_probs(dm, n)
    counts, notes_sh = run_noisy_shots(qc, p, shots=shots, seed=seed)
    shot_probs = counts_to_probs(counts)

    ideal_sv = Statevector.from_instruction(qc)
    fid = float(np.real(state_fidelity(ideal_sv, dm)))
    fid = min(1.0, max(0.0, fid))
    try:
        purity = float(np.real(dm.purity()))
    except Exception:
        rho = np.asarray(dm.data)
        purity = float(np.real(np.trace(rho @ rho)))
    purity = min(1.0, max(0.0, purity))

    leak_s = support_leakage(ideal_probs, noisy_state)
    leak_c = support_leakage(ideal_probs, shot_probs) if shot_probs else 0.0
    notes = clamp_notes + notes_dm + notes_sh

    return ComparisonResult(
        n_qubits=n,
        shots=int(shots),
        params=p,
        warnings=notes,
        ideal_probs=ideal_probs,
        noisy_state_probs=noisy_state,
        noisy_counts=counts,
        noisy_shot_probs=shot_probs,
        fidelity=fid,
        purity=purity,
        total_variation_state=total_variation(ideal_probs, noisy_state),
        total_variation_shots=total_variation(ideal_probs, shot_probs) if shot_probs else 0.0,
        leakage_state=leak_s,
        leakage_shots=leak_c,
        caption=_caption(p, fid, purity, leak_s),
    )


def format_comparison(result: ComparisonResult) -> str:
    labels = sorted(set(result.ideal_probs) | set(result.noisy_state_probs) | set(result.noisy_shot_probs))
    lines = [
        result.caption,
        f"{'state':<8} {'ideal':>10} {'noisy ρ':>10} {'shots':>10}",
        "-" * 42,
    ]
    for lab in labels:
        i = result.ideal_probs.get(lab, 0.0)
        s = result.noisy_state_probs.get(lab, 0.0)
        c = result.noisy_shot_probs.get(lab, 0.0)
        if i < 1e-4 and s < 1e-3 and c < 1e-3:
            continue
        lines.append(f"|{lab}⟩".ljust(8) + f"{i * 100:9.2f}% {s * 100:9.2f}% {c * 100:9.2f}%")
    lines.append("-" * 42)
    lines.append(
        f"F={result.fidelity:.4f}  purity={result.purity:.4f}  "
        f"TV_state={result.total_variation_state:.4f}  "
        f"leak_shots={result.leakage_shots * 100:.2f}%"
    )
    for w in result.warnings:
        lines.append("note: " + w)
    return "\n".join(lines)


# ===================================================================== tests
def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def run_self_test() -> bool:
    print("=== Noise module self-test (Aer) ===\n")
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

    e = QuantumEngine(2)
    ok, msg = e.load_preset("bell")
    check("Load Bell preset", ok, msg)

    # 1. Model builds
    nm, notes = build_noise_model(NoiseParams())
    check("build_noise_model", isinstance(nm, NoiseModel), str(notes))

    # 2. T2 clamp
    clamped, notes = NoiseParams(t1_us=10.0, t2_us=100.0).clamped()
    check(
        "T2 clamped to 2·T1",
        _close(clamped.t2_us, 20.0, 1e-9),
        f"T2={clamped.t2_us}  {notes}",
    )

    # 3. Readout 2 → 2%
    clamped, notes = NoiseParams(readout_error=2).clamped()
    check(
        "Readout 2 means 2%",
        _close(clamped.readout_error, 0.02, 1e-9),
        f"p={clamped.readout_error}  {notes}",
    )

    # 4. Noise OFF
    off = compare_ideal_noisy(e, params=NoiseParams(enabled=False), shots=512, seed=1)
    check("Noise OFF fidelity = 1", _close(off.fidelity, 1.0, 1e-12), f"F={off.fidelity}")

    # 5. Huge T1/T2, zero readout ≈ ideal
    almost_ideal = NoiseParams(
        enabled=True,
        t1_us=1e9,
        t2_us=1e9,
        readout_error=0.0,
        gate_time_1q_us=0.05,
        gate_time_2q_us=0.2,
        gate_time_3q_us=0.4,
    )
    r = compare_ideal_noisy(e, params=almost_ideal, shots=256, seed=1)
    check(
        "Tiny decoherence, F ≈ 1",
        r.fidelity > 0.995,
        f"F={r.fidelity:.6f} purity={r.purity:.6f}",
    )
    check(
        "Ideal Bell still ~50/50 on noisy ρ",
        r.noisy_state_probs.get("00", 0) > 0.45 and r.noisy_state_probs.get("11", 0) > 0.45,
        f"00={r.noisy_state_probs.get('00', 0):.3f} 11={r.noisy_state_probs.get('11', 0):.3f}",
    )

    # 6. Readout does NOT lower state fidelity, DOES leak shots
    readout_only = NoiseParams(
        enabled=True,
        t1_us=1e9,
        t2_us=1e9,
        readout_error=0.20,
        gate_time_1q_us=0.05,
        gate_time_2q_us=0.2,
        gate_time_3q_us=0.4,
    )
    r = compare_ideal_noisy(e, params=readout_only, shots=4096, seed=7)
    check(
        "Readout-only: state fidelity still ≈ 1",
        r.fidelity > 0.995,
        f"F={r.fidelity:.6f} (readout is measurement error, not a channel on ρ)",
    )
    check(
        "Readout-only: shots leak into |01⟩/|10⟩",
        r.leakage_shots > 0.15,
        f"leak_shots={r.leakage_shots:.3f}  counts={r.noisy_counts}",
    )

    # 7. Strong T1/T2, no readout: fidelity drops, mixedness rises
    decohere = NoiseParams(
        enabled=True,
        t1_us=1.0,
        t2_us=1.0,
        readout_error=0.0,
        gate_time_1q_us=0.5,
        gate_time_2q_us=2.0,
        gate_time_3q_us=3.0,
    )
    r = compare_ideal_noisy(e, params=decohere, shots=2048, seed=3)
    check(
        "Strong T1/T2: fidelity drops",
        r.fidelity < 0.97,
        f"F={r.fidelity:.4f} purity={r.purity:.4f} leak_state={r.leakage_state:.3f}",
    )
    check(
        "Strong T1/T2: purity < 1 (mixed state)",
        r.purity < 0.999,
        f"purity={r.purity:.4f}",
    )

    # 8. Toffoli still runs
    t = QuantumEngine(3)
    t.load_preset("toffoli")
    mild = NoiseParams(t1_us=50, t2_us=30, readout_error=0.02)
    try:
        rt = compare_ideal_noisy(t, params=mild, shots=256, seed=1)
        check("Toffoli + noise runs", rt.fidelity >= 0.0, f"F={rt.fidelity:.4f}")
    except Exception as exc:
        check("Toffoli + noise runs", False, repr(exc))

    # 9. Demo print — this is the viva slide
    demo = NoiseParams(
        enabled=True,
        t1_us=20.0,
        t2_us=15.0,
        readout_error=0.05,
        gate_time_1q_us=0.3,
        gate_time_2q_us=1.0,
        gate_time_3q_us=2.0,
    )
    demo_r = compare_ideal_noisy(e, params=demo, shots=2048, seed=11)
    print("\n--- Bell: ideal vs noisy (demo table) ---")
    print(format_comparison(demo_r))
    print("----------------------------------------\n")

    print(f"=== {passed} passed, {failed} failed ===")
    if failed == 0:
        print("All tests passed. Tell me this worked and we start visualizer.py.")
        return True
    print("Copy the FULL terminal output (including traceback) and send it.")
    return False


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_self_test() else 1)