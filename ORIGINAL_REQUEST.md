# Original User Request

## 2026-10-21T13:28:15Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: 4-worker parallel build team with preliminary arXiv research phase (full team)

Build `bmhd` (`v0.1.0`, **B**eltrami **M**HD & Non-Symmetric 3D Stellarator Engine, `CC0-1.0` Public Domain) from a clean repository into a complete, standalone 3D/4D magnetohydrodynamic (MHD) flux-surface solver, stepped-pressure Beltrami shell relaxer, and sparse diagnostic laser-chord tomography workstation powered by the machine-checked invariants of `OcclusionTriad.lean`, beginning with a preliminary arXiv literature research phase and culminating in a verified git push to `https://github.com/sneed-and-feed/bmhd`.

Working directory: `c:\Users\x\Documents\antigravity\bmhd`
Integrity mode: development

Reference repositories in workspace:
- `c:\Users\x\Documents\antigravity\g3vm`
- `c:\Users\x\Documents\antigravity\darksnail` (`projectdarksnail`)
- `c:\Users\x\Documents\antigravity\oatse`
- Target GitHub repository: `https://github.com/sneed-and-feed/bmhd`

## Requirements

### R0. Preliminary arXiv Literature Research Phase
- Before or alongside core implementation, conduct a preliminary arXiv research phase covering:
  1. Analytic 3D non-axisymmetric, non-stellarator-symmetric ideal MHD equilibria ($\mathbf{J} \times \mathbf{B} = \nabla p$, $\nabla \cdot \mathbf{B} = 0$, $\mathbf{B} \cdot \nabla \psi = 0$), including stretched Solov'ev equilibria with odd-in-$z$ symmetry breaking ($\tau_0$) and elliptic-coordinate sheared-$\iota(\psi)$ equilibria with phase shifts ($\tau_1$) (e.g., Issan, Citrin, Landreman & Tracey, `arXiv:2610.07304v1`, and related stellarator symmetry-breaking literature).
  2. Stepped-Pressure Equilibrium Code (SPEC) / multi-region relaxed MHD (MRxMHD) Beltrami shells ($\nabla \times \mathbf{B} = \mu_i \mathbf{B}$), resonant magnetic island chains ($\beta_1$ enclosures), Internal Transport Barriers (ITBs), and Edge Localized Mode (ELM) filament dynamics.
  3. Sparse diagnostic laser-chord tomography (Thomson scattering, $\text{CO}_2$ interferometry, Faraday polarimetry, Mirnov coils), Bremsstrahlung / divertor arc-glare washout rejection, and coupled Beltrami MHD plasma-nozzle dielectric/TTI rock spallation.
- Synthesize relevant arXiv preprints and canonical fusion/MHD literature to inform the physical parametrization and cite them in `README.md` and `docs/ARCHITECTURE.md`.

### R1. Architectural Objective, Hardware Invariants & 4-Worker Parallel Topology
Every module must enforce the hardware and mathematical invariants established across `projectdarksnail` (`c:\Users\x\Documents\antigravity\darksnail`), `g3vm` (`c:\Users\x\Documents\antigravity\g3vm`), and `oatse` (`c:\Users\x\Documents\antigravity\oatse`):
- **Real-Time Control Loop SWaP-C Envelope (`< 35 MB` Hard Ceiling, `~1.30–1.60 MB` Active Heap):** Evaluate all 3D toroidal spectral bases, multi-bounce diagnostic chord rays, $\beta_1$ magnetic-island `cavity_core` enclosures, and $\mathfrak{so}(C)$ triad operators in $Z$-slabs (`chunk_z = 16`) and stream large 3D field volumes via out-of-core Inline-slab `np.memmap` windowing so the entire working set fits inside a 4 MB L2/L3 SRAM cache.
- **Machine-Checked `OcclusionTriad.lean` Runtime Invariants (`model.verify_formal_theorems()`):**
  1. `Theorem 2.1 & 2.2` (`skew_triad_swap_neg`, `occlusion_triad_breaks_symmetric_tie`): Exact skew-symmetry (`0.00e+0` residual) and deterministic symmetry-breaking across non-stellarator-symmetric odd-parity vertical drift ($\tau_0, \tau_1 \neq 0$), Internal Transport Barriers (ITBs), and stepped-pressure Beltrami interfaces (preventing smooth Fourier/RBF solvers from zeroing out odd-parity harmonics or smearing pressure jumps).
  2. `Theorem 2.3 & 2.6` (`darksnail_readout_energy_conservation`, `darksnail_multistep_spherical_invariance`): Pointwise tangent orthogonality $\langle \mathbf{v}, B_W(\mathbf{u}, \mathbf{v}) \rangle = 0 \iff \mathbf{B} \cdot \nabla \psi = 0$ (`<= 1.0e-8` float32 residual) and `0.000e+00` radial flux-surface escape across `K in {1, 4, 8, 16}` unrolled Poincare / field-line steps.
  3. `Theorem 2.4` (`darksnail_step0_system1_homotopy`): Exact reduction to the stellarator-symmetric baseline equilibrium ($\psi(R, -\phi, -z) = \psi(R, \phi, z)$, `0.00e+0` residual) when the symmetry-breaking parameters vanish ($\tau_0 = 0, \tau_1 = 0, \delta z = \mathbf{0}$).
  4. `Theorem 2.5` (`rope2d_medoid_relative_invariance` & Non-Circular `BeltramiChordShell`): `1.0000` (100.0% clean) exterior-curl + 3-point discrete Morse ($\Delta_- v \cdot \Delta_+ v > 0$) spike recovery of thin Edge Localized Mode (ELM) filaments, resonant current sheets, and ITB gradients along 1D Thomson scattering and interferometer laser chords against Equal-Budget ($K = 16$) uniform pooling under 10% Bremsstrahlung / arc-glare washout.
  5. `Theorem 2.7` ($\mathrm{SO}(6)$ Kelvin-Mandel Tensor Rotor): Exact trace and Frobenius norm conservation (`<= 1.14e-13` residual) when rotating 3D anisotropic Braginskii magnetothermal stress and gyro-viscous stiffness tensors along helical magnetic field lines.

#### 4-Worker Parallel Topology & Strict File Ownership

| Agent ID | Role | Owned Files (Exclusive Write Access) | Synchronous Self-Verification Command |
| :--- | :--- | :--- | :--- |
| **`agent-1-core`** | Analytic Non-Symmetric 3D MHD Engine (`arXiv:2610.07304v1`), SPEC Beltrami Shells, $\mathfrak{so}(C)$ Flux Triad & 4D ELM/Sawtooth Splitter | `pyproject.toml`, `LICENSE`, `.gitignore`, `formalization/OcclusionTriad.lean`, `bmhd/__init__.py`, `bmhd/mhd.py`, `bmhd/engine.py` | `python -c "from bmhd.engine import BMHDModel; m = BMHDModel(); print(m.verify_formal_theorems())"` |
| **`agent-2-scenarios-io`** | 5-Scenario Fusion & Plasma Deck (`01`–`05`), Diagnostic Chord `.las`/`.dev` & Out-of-Core `.segy` Memmap I/O + CLI | `bmhd/scenarios.py`, `bmhd/io.py`, `bmhd/cli.py`, `data/stellarator_chords/issan_nonsym_solovev.las`, `data/stellarator_chords/thomson_chord1.dev`, `data/stellarator_chords/interferometer_chord2.dev` | `python -c "from bmhd.scenarios import build_all_scenarios; s = build_all_scenarios(); assert len(s) == 5"` |
| **`agent-3-evaluation`** | Mode A + Mode B Sparse-Chord Reconstruction (`SNR=20dB`), Noise/Jitter Sweeps & Pytest Suite | `bmhd/benchmark.py`, `scripts/evaluate_bmhd.py`, `tests/test_bmhd.py`, `results/bmhd_evaluation_summary.json`, `results/bmhd_blind_summary.json` | `python -m pytest tests/test_bmhd.py -v && python scripts/evaluate_bmhd.py` |
| **`agent-4-studio-docs`** | Zero-Install WebGL 3D Flux-Surface & Poincare Studio, Math Linter, CI Workflow & Documentation | `bmhd/studio.py`, `results/bmhd_portable_studio.html`, `scripts/audit_gfm_math.py`, `.github/workflows/ci-and-release.yml`, `README.md`, `docs/ARCHITECTURE.md` | `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` |

### R2. Workstream 1 (`agent-1-core`): `bmhd/mhd.py` & `bmhd/engine.py`
1. **`IssanNonSymmetricEquilibriumEngine` (`bmhd/mhd.py`):**
   - Implement the two closed-form 3D non-axisymmetric, non-stellarator-symmetric ideal MHD equilibrium families ($\mathbf{J} \times \mathbf{B} = \nabla p, \; \nabla \cdot \mathbf{B} = 0, \; \mathbf{B} \cdot \nabla \psi = 0$) from Issan, Citrin, Landreman & Tracey (`arXiv:2610.07304v1`):
     - **Family 1 — Stretched Solov'ev with Odd-in-$z$ Symmetry Breaking ($\tau_0 \in (-1, 1)$):** Breaks stellarator involution $\mathbf{S} = \mathrm{diag}(1, -1, -1)$ via odd-parity vertical drift couplings ($2\tau_0 z$ and $-4\tau_0 s z$), reducing identically to the stellarator-symmetric baseline at $\tau_0 = 0$ (Theorem 2.4).
     - **Family 2 — Elliptic-Coordinate Sheared-$\iota(\psi)$ with Phase Shift ($\tau_1 \in \mathbb{R}$):** Non-stellarator-symmetric 3D equilibrium with non-uniform rotational transform $\iota(\psi)$ and magnetic shear, reducing identically to the symmetric baseline at $\tau_1 = 0$.
   - Expose inspectable 3D physical cubes across `(Z=32, Y=24, X=24)` toroidal/cylindrical cross-section grids:
     - `bmhd_psi`: Normalized 3D Magnetic Flux $\psi(R, \phi, z)$ (`dimensionless`).
     - `bmhd_b_mod`: Magnetic Field Strength $\Vert{}\mathbf{B}(R, \phi, z)\Vert{}_2$ (`Tesla`, $\text{T}$).
     - `bmhd_p_kpa`: Stepped Plasma Pressure Profile $p(\psi)$ (`kPa`).
     - `bmhd_j_curl`: Beltrami Current Density & Exterior-Curl Norm $\Vert{}\mathbf{J}\Vert{}_2 = \Vert{}\nabla \times \mathbf{B}\Vert{}_2 / \mu_0$ ($\text{MA/m}^2$).
     - `bmhd_iota`: Rotational Transform Profile $\iota(\psi)$ (`dimensionless`).
2. **`SteppedPressureBeltramiTriad` & $Z$-Slab Engine (`bmhd/engine.py`):**
   - Partition the 3D plasma volume into topological shells where $\nabla \times \mathbf{B} = \mu_i \mathbf{B}$ inside each shell, separated by sharp stepped-pressure interfaces across Internal Transport Barriers (ITBs) and island separatrices enforced by $\mathfrak{so}(C)$ skew-symmetric tie-breaking (`SkewSymmetricFluxTriad`, Theorem 2.2) inside `chunk_z = 16` slabs (`< 35 MB` heap).
   - Use 6-way Jordan `cavity_core` ($\beta_1$) basis channels to isolate enclosed 3D magnetic island chains and hollow X-point divertor / plasma-torch cavities without numerical blowup, and $\mathrm{SO}(6)$ Kelvin-Mandel rotors (Theorem 2.7) to rotate anisotropic Braginskii stress/conductivity tensors along the local field vector $\hat{\mathbf{b}} = \mathbf{B} / \Vert{}\mathbf{B}\Vert{}_2$.
3. **Pure Non-Circular `BeltramiChordShell` ($K = 8 + 8 = 16$ Tokens):**
   - Along 1D Thomson scattering laser, interferometer, and soft X-ray chords, select Tier-1 spike tokens strictly via coarse-fine Beltrami exterior-curl salience, fine-shell magnitude $\vert{}u_{\mathrm{fine}}(s)\vert{}$, and 3-point discrete Morse impulse $\Delta_- v(s)\cdot\Delta_+ v(s) > 0$, paired with a 10%–20% Bremsstrahlung / arc-glare washout gate.
4. **4D Sawtooth / Coil-Ripple Continuity Splitter (`TemporalPlasmaContinuitySplitter`):**
   - Decompose 4D time-resolved magnetic/density reconstructions into $\mathcal{H}_{\mathrm{clock}} \oplus \mathcal{H}_{\mathrm{nope}}$, quarantining reversible AC power-supply coil ripple and benign limit-cycle oscillations (`continuity_cycle_score > 0`) while recovering `1.0000` (`100.0%`) of irreversible radiative collapse / disruption precursors or cryogenic pellet-injection fronts in $\mathcal{H}_{\mathrm{nope}}$.

### R3. Workstream 2 (`agent-2-scenarios-io`): `bmhd/scenarios.py`, `bmhd/io.py` & `bmhd/cli.py`
Implement the **5-Scenario 3D Stellarator, Tokamak & Beltrami Plasma Benchmark Deck** on `(Z=32, Y=24, X=24)` grids (`18,432` voxels) with 8 synthetic 1D diagnostic chords (Thomson scattering, $\text{CO}_2$ interferometer, Faraday polarimetry, and Mirnov probe arrays) per scenario:
1. **`01` `issan_nonsym_solovev_tau0` (`arXiv:2610.07304v1` Section 2):** Non-stellarator-symmetric 3D stretched Solov'ev equilibrium ($\tau_0 = 0.38$) with broken $\mathbf{S} = \mathrm{diag}(1, -1, -1)$ parity, stepped H-mode pedestal pressure barrier, and thin radial ELM filament spikes.
2. **`02` `issan_sheared_elliptic_tau1` (`arXiv:2610.07304v1` Section 3):** Non-stellarator-symmetric elliptic-coordinate equilibrium ($\tau_1 = 0.42$) with non-uniform sheared rotational transform $\iota(\psi)$ and an Internal Transport Barrier (ITB) step jump where smooth global Fourier/RBF surrogates smear the shear gradient.
3. **`03` `spec_magnetic_island_chain` (Stepped-Pressure Beltrami Shells & $\beta_1$ Island Enclosures):** Multi-shell relaxed Beltrami plasma ($\nabla \times \mathbf{B} = \mu_i \mathbf{B}$) containing a resonant $m/n = 2/1$ magnetic island chain ($\beta_1$ `cavity_core` flattened-pressure pockets) bounded by razor-sharp helicity-conserving current sheets.
4. **`04` `tokamak_elm_pellet_4d` (4D Fast Control Loop Reconstruction):** Tokamak H-mode pedestal with a sharp scrape-off-layer (SOL) X-point separatrix, sub-centimeter Edge Localized Mode (ELM) filaments, 10% divertor Bremsstrahlung glare washout, and 4D cryogenic $\mathrm{D}_2$ pellet injection vs. reversible coil ripple.
5. **`05` `beltrami_plasma_spallation_torch` (Coupled Beltrami MHD Nozzle & Dielectric/TTI Rock Spallation):** Magnetically nozzle-steered Beltrami plasma vortex ($\nabla \times \mathbf{B} = \mu \mathbf{B}$) impinging on anisotropic crystalline granite/basalt with a thin water-filled fracture stringer ($\varepsilon_r' = 9.85$), coupling magnetic nozzle flux confinement with `OlhoeftStrangwayDielectricEngine` breakdown throttling and $\mathrm{SO}(6)$ thermal spallation stress $\boldsymbol{\sigma}_{\mathrm{therm}} = \mathbf{C}_{\mathrm{TTI}} : \boldsymbol{\alpha}\Delta T$ across a self-vitrifying $\beta_1$ glass tunnel liner.
6. **Out-of-Core `.segy` / Field-Cube Memmap & `.las` Diagnostic Chord I/O (`bmhd/io.py` & `bmhd/cli.py`):**
   - Implement `stream_field_volume()` using Inline-slab `np.memmap` windowing (`< 35 MB` peak heap on a `64 MB` `400x400x100` volume) with 3D Beltrami exterior-curl downsampling (`--roi`, `--max-grid`), plus CWLS `.las` 2.0 (`PSI`, `B_MOD`, `P_KPA`, `J_CURL`, `IOTA`) and minimum-curvature `.dev` laser-chord geometry parsers.

### R4. Workstream 3 (`agent-3-evaluation`): `bmhd/benchmark.py`, `scripts/evaluate_bmhd.py` & `tests/test_bmhd.py`
Build the complete dual-mode verification and sparse-chord inversion suite:
1. **Mode A (Analytical Forward & Operator Verification) & Mode B (Blind Sparse-Chord & Noisy Tomography Reconstruction at `SNR = 20 dB`, Zero Target Leakage):**
   - In Mode B, reconstruct the 3D flux $\psi$, field strength $\Vert{}\mathbf{B}\Vert{}_2$, and stepped pressure $p$ strictly from training diagnostic laser chords `C_train` + $3\times 3\times 3$ smoothed noisy magnetics tomography at `SNR = 20 dB` with **zero access to ground-truth shell labels or held-out chords**.
   - Compare `NaiveBox`, **`TunedAnisotropicKriging`** (inner Leave-One-Chord-Out CV-tuned 3D Matérn-3/2), **`SymmetricFourierVMECBaseline`** (stellarator-symmetric harmonic projection that zeroes out odd-parity $\tau_0, \tau_1$ terms), **`ShellPartitionedKriging`**, and **`BlindBMHD`** across:
     - Leave-One-Chord-Out (`LOCO`) vertical chords,
     - Unseen 3D angled Thomson / interferometer laser chords,
     - Whole-Torus $\psi$ & $\Vert{}\mathbf{B}\Vert{}_2$ RMSE, and
     - Symmetric **Bounded Edge Gradient Fidelity** (`bmhd_edge_gradient_fidelity` $\in [0\%, 100\%]$) across ITB and island separatrix steps.
2. **Equal-Budget ($K=16$) `BeltramiChordShell`, SNR Sweep ($\infty, 20, 10, 5\text{ dB}$) & Separatrix-Pick Jitter ($\Delta x_{\mathrm{jitter}} \in \{0, 1, 2, 3\}$ Voxels):**
   - Benchmark Peak / All-Filaments / Thin-Current-Sheet F1 against Equal-Budget ($K=16$) and Coarse (`Z//8`) pooling across Clean, `20 dB`, `15 dB + 10% Bremsstrahlung glare washout`, and `10 dB + 20% washout`.
3. **Pytest Suite (`tests/test_bmhd.py`):**
   - Assert all `OcclusionTriad.lean` runtime theorems (`Thm 2.1–2.7`), exact zero-perturbation reduction at $\tau_0 = \tau_1 = 0$ (`0.00e+0`), `0.000e+00` multistep field-line radial drift, `chunk_z = 8 / 16 / 32` bit-identity (`< 1e-6`), and `tracemalloc` peak heap `< 35 MB`.

### R5. Workstream 4 (`agent-4-studio-docs`): `bmhd/studio.py`, `results/bmhd_portable_studio.html`, Docs & Git Push
1. **Zero-Install HTML5/WebGL 3D Stellarator Flux-Surface & Poincare Studio (`results/bmhd_portable_studio.html`):**
   - Single-file dark-mode Three.js / WebGL workstation supporting `file://` and GitHack CDN deep-linking (`?scenario=issan_nonsym_solovev_tau0&field=bmhd_psi`) across all 5 scenarios (`01`–`05`).
   - 3D chair-cut torus/block viewer + interactive Poincare puncture plot inspector with selectable fields: `Magnetic Flux psi`, `Field Strength |B| (Tesla)`, `Stepped Pressure p (kPa)`, `Beltrami Current Density |J|`, and `Rotational Transform iota`.
   - Interactive symmetry-breaking slider ($\tau_0, \tau_1 \in [-0.8, +0.8]$, snapping to the exact symmetric baseline at `0.00`), 3D laser-chord tubes displaying locked Beltrami ELM/ITB filament spikes (yellow rings) and rejected Bremsstrahlung glare washouts (purple squares), and a client-side **`.las` / `.dev` / `.segy` `ArrayBuffer` Drag-and-Drop Inspector**.
2. **Documentation & Math Audit (`README.md`, `docs/ARCHITECTURE.md`):**
   - Document the exact equation-by-equation mapping between Issan et al. (`arXiv:2610.07304v1`), PPPL SPEC Beltrami shells, `OcclusionTriad.lean`, and `bmhd`, incorporate citations from the preliminary arXiv research phase, include the full Mode A and Mode B 5-scenario tables, and verify `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` passes with `0` errors.
3. **Git Initialization & Push (`https://github.com/sneed-and-feed/bmhd`):**
   - Initialize the git repository in `c:\Users\x\Documents\antigravity\bmhd`, stage all project files, commit `v0.1.0`, and push to `https://github.com/sneed-and-feed/bmhd`.

## Acceptance Criteria

### Formal & Runtime Verification
- [ ] `python -c "from bmhd.engine import BMHDModel; m = BMHDModel(); print(m.verify_formal_theorems())"` passes all Theorems 2.1–2.7 (`0.00e+0` skew-symmetry residual, `0.000e+00` multistep radial flux-surface escape, `<= 1.0e-8` tangent orthogonality, `0.00e+0` symmetric homotopy residual at $\tau_0 = \tau_1 = 0$, `1.0000` clean spike recovery under 10% washout, `<= 1.14e-13` $\mathrm{SO}(6)$ rotor conservation).
- [ ] `python -c "from bmhd.scenarios import build_all_scenarios; s = build_all_scenarios(); assert len(s) == 5"` succeeds with all 5 scenarios (`01`–`05`) and bundled diagnostic chord files in `data/stellarator_chords/`.

### Test Suite, Benchmarks, Studio & Git Delivery
- [ ] `python -m pytest tests/test_bmhd.py -v` passes 100% with `0` warnings, `0.000e+00` radial field-line escape, `0.00e+0` symmetric homotopy residual at $\tau_0 = \tau_1 = 0$, `<= 1.14e-13` $\mathrm{SO}(6)$ rotor residual, and `< 35 MB` (`~1.5 MB` active engine) peak heap.
- [ ] `python scripts/evaluate_bmhd.py` generates `results/bmhd_evaluation_summary.json` and `results/bmhd_blind_summary.json` across all 5 scenarios (`01`–`05`) in both Mode A and Mode B (`SNR = 20 dB`).
- [ ] `results/bmhd_portable_studio.html` opens directly in any browser with zero external network calls, rendering all 5 stellarator/tokamak/plasma-torch scenarios, diagnostic laser chords, and client-side `.las`/`.segy` dropzone inspection.
- [ ] `python scripts/audit_gfm_math.py README.md docs/ARCHITECTURE.md` passes with `0` issues.
- [ ] Repository `c:\Users\x\Documents\antigravity\bmhd` is committed and pushed to `https://github.com/sneed-and-feed/bmhd`.

## Follow-up — 2026-10-21T13:43:23Z

User Scope Addition (Include Both Official Benchmark Extensions):

Please incorporate BOTH of the following official baseline benchmark additions into `PROJECT.md` and the worker dispatches (`agent-1-core`, `agent-3-evaluation`, `agent-4-studio-docs`):

1. **Exact `arXiv:2610.07304v1` Analytical Invariant Checks (`bmhd/mhd.py`, `bmhd/benchmark.py`, `tests/test_bmhd.py`):**
   - Verify the exact analytical enclosed volume $V(\delta)$, volume-averaged plasma beta $\beta_V(\tau_0, \tau_1)$, constant Jacobian determinant $-\frac{ab}{\Omega^3}$ (Family 1), and Hamada condition line integrals $\oint \mathrm{d}\ell / \lvert\mathbf{B}\rvert$ (constant $2\pi/\Omega$ for Family 1, complete elliptic integral of the first kind for Family 2) from Sections 2.3 and 3.3–3.4 of Issan, Citrin, Landreman & Tracey (`arXiv:2610.07304v1`).

2. **Cerfon–Freidberg (2010) & Minimum-Fisher Official Baselines (`bmhd/mhd.py`, `bmhd/benchmark.py`, `scripts/evaluate_bmhd.py`, `tests/test_bmhd.py`, `README.md`, `docs/ARCHITECTURE.md`):**
   - **Cerfon & Freidberg (2010) *"One size fits all"* Analytic Grad–Shafranov Baseline** (*Phys. Plasmas* 17, 032502): Include the canonical axisymmetric tokamak X-point / H-mode Grad–Shafranov analytical equilibrium benchmark (supporting Scenario `04` `tokamak_elm_pellet_4d` and Mode A/B comparisons).
   - **Tikhonov Minimum-Fisher Regularization (`MinimumFisherTomography`)**: Include the standard fusion plasma diagnostic chord tomography baseline (used on JET, ASDEX Upgrade, and W7-X) in `bmhd/benchmark.py` Mode B sparse-chord inversion comparisons alongside `NaiveBox`, `TunedAnisotropicKriging`, `SymmetricFourierVMECBaseline`, `ShellPartitionedKriging`, and `BlindBMHD`.


## 2026-10-21T14:53:17Z

User Request Note: When finished, we will also capture a headless Chrome screenshot of `results/bmhd_portable_studio.html` demonstrating primary functionality (3D WebGL flux-surface & laser-chord viewer + Poincaré puncture plot + HUD metrics) and visually audit it. Please ensure `results/bmhd_portable_studio.html` renders cleanly in headless Chromium/Chrome with 0 console/page errors during your Victory Audit.
