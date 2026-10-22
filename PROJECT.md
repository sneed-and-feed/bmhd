# Project: bmhd — Stylo Diagnostic Bridge & Theorem 2.8 Extension

## Architecture
- **Real-Time Control Loop SWaP-C Envelope**: `< 3.5 MB` hard arena ceiling (`~2.33 MB` active working set: `1.48–1.60 MB` BMHD Z-slab + `692,416 B` Stylo static arena + `< 64 KB` bridge buffer) resident in CPU L3 cache with 0 GPU dependencies and 0 per-frame heap allocations.
- **Standalone GitHub Repository**: `bmhd` is completely self-contained under `CC0-1.0` with 0 external package dependencies. Pure NumPy vectorization fallback with optional `#![no_std]` Rust SIMD C-ABI auto-detection.
- **Mathematical Invariants**:
  - Theorems 2.1–2.7: Skew-symmetry (`0.00e+0`), tie-breaking, energy conservation, symmetric homotopy reduction (`0.00e+0`), discrete Morse impulse recovery (`1.0000`), $\mathrm{SO}(6)$ Kelvin-Mandel rotor conservation ($\le 1.14\times 10^{-13}$).
  - Theorem Family 2.8: $\mathbf{B}\cdot\nabla\psi = 0 \iff \langle v_{\mathrm{flux}}, B_W(u_{\mathrm{occ}}, v_{\mathrm{flux}}) \rangle = 0$ ($\le 1.0\times 10^{-8}$ residual), exact skew-symmetric tie-breaking margin with 0 ID swaps.
- **Kinematic & Control Laws**:
  - Scenario 04: Cryogenic $\mathrm{D}_2$ pellet core deposition depth $\psi_{\mathrm{dep}}(t) \le 0.80\text{ px}$ RMSE under 10% glare.
  - Scenario 05: 100% molten glass flare rejection, nozzle steering $\mathbf{slew\_cmd} = (-0.15 v_x, -0.15 v_y)$, Olhoeft-Strangway flux throttling $\mathrm{throttle\_gain} = \max(0.35, 6.5 / \varepsilon_{\mathrm{local}})$.
- **Tactical 4-QUAD & Fused HUD**: Embedded WebGL 3D Studio + 512x512 vector HUD over 128x128 sensor arena with MIL-STD-2525D symbology, STANAG 4609 KLV .JSON & 6-DOF .CSV export.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `StyloTracker` & `TrackerStateArena` vendoring | 692,416 B arena layout in pure NumPy with C-ABI auto-detection | M1 | R1 |
| 2 | Stage 1 Mask-First MAD & Morse Impulse Gate | 256-sample strided MAD on $(1-M_{\mathrm{clock}})(1-M_{\mathrm{washout}})$ bitplanes | M1 | R1 |
| 3 | Stage 2 Continuity Splitter & Ring Buffer | $\partial_t \rho_c + \nabla \cdot \mathbf{J}_c = \sigma_c$, notch out AC ripple | M1 | R1 |
| 4 | Stage 3 $\mathfrak{so}(C)$ Skew Triad & $K=16$ Cascade | $B_W(u_{\mathrm{occ}}, v_{\mathrm{comp}})$ tie-breaking with 0 ID swaps | M1 | R1 |
| 5 | Stage 4 $\mathrm{SE}(2)$ Kinematics & Causal Ratio | $\eta_{\mathrm{nonlocal}} \ge 0.50$ separating $\mathcal{G}_{\mathrm{act}}$ from $\Omega_{\mathrm{reg}}$ | M1 | R1 |
| 6 | Theorem 2.8 Runtime Audit in `engine.py` | Add `theorem_2_8` audit block to `BMHDModel.verify_formal_theorems()` | M1 | R2 |
| 7 | Sightline Projections in `scenarios.py` | 8 tangential/poloidal $128 \times 128$ Y8 $Z$-slab sightline projections | M2 | R2 |
| 8 | Scenario 04 Pellet/ELM Separation & Deposition | Pedestal ELMs vs inward pellet, $\psi_{\mathrm{dep}}(t) \le 0.80\text{ px}$ RMSE | M1 | R2 |
| 9 | Scenario 05 Flare Rejection & Nozzle Control | 100% flare rejection, slew damping, Olhoeft-Strangway dielectric throttling | M1 | R2 |
| 10 | Scenarios 01–03 Flux-Locked Tracking | `step_scenario_flux_locked()` with 0 ID swaps across $\tau_0, \tau_1$, SPEC | M1 | R2 |
| 11 | Tactical 4-QUAD / FUSED HUD in `studio.py` | Quads A–D, vector overlays, thermal palettes, URL deep-linking | M3 | R3 |
| 12 | Telemetry Export (STANAG 4609 & 6-DOF CSV) | Client-side KLV JSON and 6-DOF CSV export in portable studio | M3 | R3 |
| 13 | Bridge Benchmark Suite (`benchmark.py`) | `run_stylo_bridge_benchmark()` across all 5 benchmark scenarios | M4 | R4 |
| 14 | Evaluation CLI Script (`evaluate_bmhd.py`) | Export `results/bmhd_stylo_bridge_summary.json` & update evaluation JSON | M4 | R4 |
| 15 | Unit & Adversarial Test Suite | 8 tests in `tests/test_stylo_diagnostic_bridge.py` + `tests/test_bmhd.py` | M4 | R4 |
| 16 | Headless Chrome Visual Verification | Capture `results/studio_screenshot.png` and `results/studio_stylo_hud_screenshot.png` | M4 | R4 |
| 17 | Literature & Architecture Documentation | Update `README.md` and `docs/ARCHITECTURE.md` with operator bijections | M4 | R0 |
| 18 | KaTeX Math Compliance Audit | `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` (0 errors) | M4 | R4 |
| 19 | Git Delivery & Push | Commit and push to `https://github.com/sneed-and-feed/bmhd` | M5 | R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Core Bridge & Theorem 2.8 | `bmhd/stylo_diagnostic_bridge.py`, `bmhd/engine.py` | none | DONE (`BMHDStyloDiagnosticBridge`, Thm 2.8) |
| M2 | Scenarios Sightline Streaming | `bmhd/scenarios.py` | none | DONE (`get_sightline_projections()`) |
| M3 | Studio 4-QUAD & Fused HUD | `bmhd/studio.py`, `results/bmhd_portable_studio.html` | M1, M2 | DONE (4-QUAD / Fused HUD, STANAG/CSV export) |
| M4 | Benchmarks, Tests & Documentation | `bmhd/benchmark.py`, `scripts/evaluate_bmhd.py`, `tests/`, `README.md`, `docs/ARCHITECTURE.md` | M1, M2 | DONE (44 tests pass, summaries exported, KaTeX clean) |
| M5 | Multi-Party Verification & Git Push | Reviewers, Challengers, Forensic Auditor, Git commit & push | M1–M4 | IN_PROGRESS (Gate 2 Passed CLEAN, Delivery in progress) |

---

## Code Layout & Exclusive Write Ownership
| Workstream | Worker Role | Owned Files (Exclusive Write Access) |
| :--- | :--- | :--- |
| **WS1 (M1)** | `worker_bridge_core` | `bmhd/stylo_diagnostic_bridge.py`, `bmhd/engine.py` |
| **WS2 (M2)** | `worker_scenarios` | `bmhd/scenarios.py` |
| **WS3 (M3)** | `worker_studio_hud` | `bmhd/studio.py`, `results/bmhd_portable_studio.html` |
| **WS4 (M4)** | `worker_eval_tests_docs`| `bmhd/benchmark.py`, `scripts/evaluate_bmhd.py`, `tests/test_stylo_diagnostic_bridge.py`, `tests/test_bmhd.py`, `README.md`, `docs/ARCHITECTURE.md` |

---

## Interface Contracts

### `BMHDStyloDiagnosticBridge` ↔ `BMHDModel` & `scenarios`
```python
class BMHDStyloDiagnosticBridge:
    ARENA_SIZE_BYTES: int = 692416
    COMBINED_ARENA_CEILING_BYTES: int = 3500000

    def __init__(self, use_c_abi: bool = True): ...
    def step_scenario_04_pellet_elm(self, scenario_data: Any, frame_idx: int = 0) -> Dict[str, Any]: ...
    def step_scenario_05_spallation_torch(self, scenario_data: Any, frame_idx: int = 0) -> Dict[str, Any]: ...
    def step_scenario_flux_locked(self, scenario_data: Any, frame_idx: int = 0) -> Dict[str, Any]: ...
```

### `scenarios.py` Sightline Helper
```python
def get_sightline_projections(scenario_data: ScenarioData, num_chords: int = 8, resolution: Tuple[int, int] = (128, 128)) -> Dict[str, np.ndarray]:
    """
    Returns:
        dict with keys:
        - "projections": np.ndarray of shape (num_chords, 128, 128) float32 in [0, 1]
        - "grad_psi": np.ndarray of shape (num_chords, 128, 128, 2) float32
        - "mask_washout": np.ndarray of shape (num_chords, 128, 128) bool
        - "mask_clock": np.ndarray of shape (num_chords, 128, 128) bool
        - "dielectric": Optional np.ndarray of shape (num_chords, 128, 128) float32
    """
```
