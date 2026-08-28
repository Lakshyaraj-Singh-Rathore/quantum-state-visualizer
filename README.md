# Quantum State Visualizer

Interactive lab for building small quantum circuits, stepping through every gate, and watching the state evolve. Optional Qiskit Aer noise (T₁, T₂, readout) sits next to the ideal statevector so you can compare them.

The app runs **entirely on your machine**. There is no cloud backend and no IBM Quantum account.

**Author:** LAKSHYARAJ SINGH RATHORE 
**Project:** Quantum State Visualizer  
**Year:** 2026

---

## What it does

- Build circuits with H, X, Y, Z, S, S†, T, T†, RX / RY / RZ, CNOT, SWAP, and Toffoli
- Step **Reset / Prev / Next / Play** through a snapshot after each gate
- Probability bars colored by **relative phase**
- Entanglement meter (von Neumann entropy of a reduced qubit; concurrence when n = 2)
- Ideal vs noisy histograms, **fidelity**, **purity**, and shot leakage
- Amplitude / phase table, density-matrix heatmap, OpenQASM + CSV export
- **Q-sphere** of the *ideal* ket (IBM-style layout: latitude = Hamming weight)

Presets: Superposition, HZH, T-phase, T-demo, RX(π), Bell, GHZ, Toffoli AND, SWAP demo.

---

## Requirements

- Python 3.10 or newer (recommended)
- Windows, macOS, or Linux

Packages (see `requirements.txt`):

```text
qiskit>=1.0,<3
qiskit-aer
numpy
plotly
streamlit
```

---

## Run

```bash
cd quantum-visualizer
python -m venv .venv
```

### Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**.

On first launch Streamlit may ask for an email. Leave it blank and press Enter.

Keep the terminal open while you use the app. Stop with **Ctrl+C**.

Optional dark theme: `.streamlit/config.toml` (not required).

---

## Project layout

| File | Role |
|------|------|
| `engine.py` | Circuits, gates, ideal snapshots, entanglement, presets |
| `noise.py` | Aer T₁ / T₂ / readout; fidelity vs the ideal circuit prefix |
| `visualizer.py` | Plotly charts, circuit drawing, Q-sphere |
| `app.py` | Streamlit UI only — no second simulator |
| `requirements.txt` | Dependencies |

Simulation lives in `engine.py` and `noise.py`. Charts live in `visualizer.py`. `app.py` wires them together.

---

## Bit order

Basis labels are textbook **|q0 q1 q2…⟩ with q0 on the left**. That is the reverse of Qiskit’s little-endian bitstrings.

After H on q0 of |00⟩ you should see **|00⟩ and |10⟩**, not |01⟩. That is intentional.

---

## Suggested demo (2–3 minutes)

1. **Superposition** — H maps |0⟩ to a 50/50 mix.
2. **HZH ≡ X** — final |1⟩, but the Z step changes **phase**, not bar height. Open the amplitude / phase table.
3. **Bell** — Next after H (still separable: |00⟩ + |10⟩). Next after CNOT: |00⟩ + |11⟩, entanglement **S → 1**. Open the **Q-sphere**: two poles.
4. **Toffoli AND** or **RX(θ)** if there is time.
5. Stay on Bell. Turn on **Enable T₁ / T₂ / readout**, or **Demo: make noise obvious**. Tab **Ideal vs noisy**: fidelity, purity, extra |01⟩ / |10⟩ on **shots**.
6. **Limitations** — T₁/T₂ are sliders, not a named chip; the Q-sphere is not a Bloch sphere.

If asked about hardware:

> This is a configurable Aer model of relaxation, dephasing, and readout error — not calibration from a particular IBM processor.

---

## Noise

| Control | What it affects |
|---------|-----------------|
| T₁, T₂, gate times | Mixed state after the circuit (density matrix). Lowers **fidelity** and **purity**. |
| Readout error | Mis-assigned measurement bits. Appears in **shots**, not in state fidelity. |
| Shots | Sampling of the (possibly noisy) measurement distribution. |

Z, S, S†, T, T†, and RZ are treated as **instantaneous** (virtual-Z). They do not pick up extra T₁/T₂ in this model.

The Q-sphere always shows the **ideal** snapshot at the current step. A mixed state has no unique amplitudes, so noise is not drawn on the sphere.

---

## Q-sphere

- Latitude = Hamming weight (number of 1s)
- North pole = |0…0⟩, south pole = |1…1⟩
- Blob size = probability
- Labels include relative phase
- Red spokes from the origin (IBM-style layout)

A Bell state is **two poles**. That is the correct Q-sphere for two qubits. It is **not** a Bloch sphere (a Bloch sphere only describes one qubit).

---

## Module tests (optional)

With the venv active, from the project folder:

```bash
python engine.py
python noise.py
python visualizer.py
```

Each script should print `All tests passed`.

---

## Scope

**This is** a local teaching lab: Qiskit statevector steps plus optional Aer noise.

**This is not** a real QPU, a pulse-level simulator, or IBM Quantum Composer. T₁, T₂, and readout are parameters you set. They reproduce the *kind* of errors seen on superconducting devices; they are not taken from any particular chip’s calibration.

Entanglement **S** is the largest single-qubit von Neumann entropy after tracing out the rest (in bits). On two qubits, concurrence is shown as well. Those numbers describe the **ideal** snapshot. Under noise, use **fidelity** and **purity**.

A Bloch sphere is omitted on purpose. The Q-sphere is an IBM-style layout of computational-basis amplitudes — not a Bloch sphere, and not the mixed state after T₁/T₂.

---

## Author

YOUR NAME

This project was built as a student / course project. Simulation uses Qiskit and Qiskit Aer; charts use Plotly; the UI uses Streamlit. Those libraries keep their own licenses.

© 2026 Lakshyaraj Singh Rathore. No affiliation with IBM is implied.
```