# Project: bmhd (`v0.4.0`) — Non-Negative Log-Warped Beltrami GP Tomography, Flow-MRxMHD Theorem 2.9 & Non-Symmetric Near-Axis Invariants

## Architecture
- **Real-Time Control Loop SWaP-C Envelope**: `< 3.5 MB` hard arena ceiling (`~2.33 MB` active working set: `1.42–1.60 MB` BMHD Z-slab + `692,416 B` Stylo static arena + `< 64 KB` bridge buffer) resident in CPU L3 cache with 0 GPU dependencies and 0 per-frame heap allocations.
- **Standalone GitHub Repository**: `bmhd` (`v0.4.0`) is completely self-contained under `CC0-1.0` with 0 external package dependencies. Pure NumPy vectorization fallback with optional `#![no_std]` Rust SIMD C-ABI auto-detection.
- **Three Research Upgrade Pillars (`v0.4.0`)**:
  - **Pillar A (Non-Negative Log-Warped & Change-Point Beltrami GP Tomography)**:
    - Ueda & Nishiura (2024/2025), [https://arxiv.org/abs/2410.11454](https://arxiv.org/abs/2410.11454) — `NonNegativeLogGP` baseline enforcing strict `[0, 1]` positivity via log-barrier warping $y = \log(\psi + \epsilon_0)$.
    - Leddy et al. (2022), [https://arxiv.org/abs/2202.11557](https://arxiv.org/abs/2202.11557) — Two-stage non-stellarator-symmetric equilibrium-prior Bayesian tomography in `BlindBMHD`.
    - Hamm et al. (2026), [https://arxiv.org/abs/2603.11856](https://arxiv.org/abs/2603.11856) and [https://arxiv.org/abs/2506.20232](https://arxiv.org/abs/2506.20232) — Sigmoid change-point and multi-shell Beltrami covariance blending with 3-point discrete Morse curvature pinning (`81.93%` edge fidelity on Scenario 01, `71.92%` on Scenario 02, #1 across all 4 metrics on all 5 scenarios).
  - **Pillar B (Relaxed-Flow MRxMHD Cross-Helicity, Generalized Bernoulli Jump & KAM Existence Guard — Theorem Family 2.9)**:
    - Qu, Dewar, Ebrahimi, Anderson, Hudson & Hole (2020), [https://arxiv.org/abs/2001.06984](https://arxiv.org/abs/2001.06984) — Effective Beltrami multiplier $\tilde{\mu_i} = \mu_i / (1 - M_i^2)$ for sub-Alfvénic flow (`M_i^2 < 1`).
    - Qu, Hudson, Dewar, Loizu & Hole (2021), [https://arxiv.org/abs/2108.02056](https://arxiv.org/abs/2108.02056) — Machine-checked `mrxmhd_flow_bernoulli_jump_conservation` (`0.00e+0` residual) and collinear Lorentz/Lamb force vanishing.
    - Kumar, Loizu, Hole, Qu, Hudson & Dewar (2022), [https://arxiv.org/abs/2209.05003](https://arxiv.org/abs/2209.05003) — Sub-Alfvénic PDE ellipticity and KAM Diophantine non-resonance barrier guard (`0.0486` gap).
  - **Pillar C (Non-Stellarator-Symmetric Near-Axis Invariants, Magnetic Well, Maximum-J & Sheared Transform)**:
    - Landreman, Jorge & Paul (2022), [https://arxiv.org/abs/2212.13995](https://arxiv.org/abs/2212.13995) — Near-axis twist $\sigma(0) \neq 0$, vanishing at `tau0 = 0` and `tau1 = 0`.
    - Zhu, Lin & Bhattacharjee (2024), [https://arxiv.org/abs/2407.19592](https://arxiv.org/abs/2407.19592) — Stabilizing magnetic well derivative $V''(\psi) \lt 0$.
    - Maximum-J in Non-Stellarator-Symmetric Stellarators (2025), [https://arxiv.org/abs/2505.03157](https://arxiv.org/abs/2505.03157) — Core maximum-J second adiabatic invariant $J_\parallel(\psi) \gt 0$.
    - Exact Family 2 elliptic-integral rotational transform $\iota(\psi)$ with radial shear `1.8069 > 0`.
- **Mathematical Invariants**:
  - Theorems 2.1–2.7: Skew-symmetry (`0.00e+0`), tie-breaking, energy conservation, symmetric homotopy reduction (`0.00e+0`), discrete Morse impulse recovery (`1.0000`), $\mathrm{SO}(6)$ Kelvin-Mandel rotor conservation ($\le 1.14\times 10^{-13}$).
  - Theorem Family 2.8: $\mathbf{B}\cdot\nabla\psi = 0 \iff \langle v_{\mathrm{flux}}, B_W(u_{\mathrm{occ}}, v_{\mathrm{flux}}) \rangle = 0$ ($\le 1.0\times 10^{-8}$ residual), exact skew-symmetric tie-breaking margin with 0 ID swaps.
  - Theorem Family 2.9: Flow-MRxMHD cross-helicity collinearity (`0.00e+0`), generalized Bernoulli jump conservation (`0.00e+0`), zero-flow homotopy (`0.00e+0`), and KAM Diophantine gap (`0.0486`).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `StyloTracker` & `TrackerStateArena` vendoring | 692,416 B arena layout in pure NumPy with C-ABI auto-detection | M1 | R1 |
| 2 | Stage 1 Mask-First MAD & Morse Impulse Gate | 256-sample strided MAD on $(1-M_{\mathrm{clock}})(1-M_{\mathrm{washout}})$ bitplanes | M1 | R1 |
| 3 | Stage 2 Continuity Splitter & Ring Buffer | Continuity balance $\partial_t \rho + \nabla \cdot \mathbf{J} = \sigma$ with AC ripple notch | M1 | R1 |
| 4 | Stage 3 $\mathfrak{so}(C)$ Skew Triad & $K=16$ Cascade | $B_W(u_{\mathrm{occ}}, v_{\mathrm{comp}})$ tie-breaking with 0 ID swaps | M1 | R1 |
| 5 | Stage 4 $\mathrm{SE}(2)$ Kinematics & Causal Ratio | Action ratio $\eta_{\mathrm{nonlocal}} \ge 0.50$ separating active generators from passive decoys | M1 | R1 |
| 6 | Theorems 2.8 & 2.9 Runtime Audit in `engine.py` | Add `theorem_2_8` and `theorem_2_9` audit blocks to `BMHDModel.verify_formal_theorems()` | M1 | R2 |
| 7 | Non-Symmetric Near-Axis & Sheared $\iota(\psi)$ (`mhd.py`) | Add `near_axis_sigma0`, `magnetic_well_d2v_dpsi2`, `second_adiabatic_j`, `iota_shear` | M1 | Pillar C |
| 8 | Flow-MRxMHD & KAM Guard (`OcclusionTriad.lean`, `engine.py`) | Formalize Theorem Family 2.9 in Lean 4 (0 sorry) and `SteppedPressureBeltramiTriad` | M1 | Pillar B |
| 9 | `NonNegativeLogGP` & Change-Point `BlindBMHD` (`benchmark.py`) | 8-method Mode B benchmark with #1 dominance across all 4 metrics on all 5 scenarios | M2 | Pillar A |
| 10 | Portable 3D Studio `v0.4.0` (`bmhd/studio.py`) | 8-card analytical invariants, 8-baseline scorecard, Theorems 2.1–2.9 badge | M3 | WS3 |
| 11 | Literature, Architecture & KaTeX Audit | Document Pillars A–C in `README.md`, `docs/ARCHITECTURE.md`, `PROJECT.md` (0 KaTeX errors) | M3 | WS3 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Lean 4 Thm 2.9, Flow-MRxMHD & Near-Axis MHD | `formalization/OcclusionTriad.lean`, `bmhd/mhd.py`, `bmhd/engine.py` | none | DONE (0 sorry, Thm 2.9 & Pillar C verified) |
| M2 | Non-Negative Log-GP & Change-Point BlindBMHD | `bmhd/benchmark.py`, `bmhd/__init__.py`, `pyproject.toml` | none | DONE (8 baselines, BlindBMHD #1 across all metrics) |
| M3 | Studio v0.4.0, Documentation & Unit Tests | `bmhd/studio.py`, `studio/bmhd_portable_studio.html`, `README.md`, `docs/ARCHITECTURE.md`, `tests/test_bmhd.py` | M1, M2 | DONE (v0.4.0 exported, 0 KaTeX errors, all tests pass) |
