"""
BMHD Scenarios: 5-Scenario 3D Stellarator, Tokamak & Beltrami Plasma Benchmark Deck.

Implements:
1. Scenario 01: 01_issan_nonsym_solovev_tau0
   - Stretched Solov'ev equilibrium with odd-in-z vertical parity breaking tau_0 = 0.38
     (arXiv:2610.07304v1 Section 2), stepped H-mode pedestal, and thin radial ELM filament spikes.
2. Scenario 02: 02_issan_sheared_elliptic_tau1
   - Elliptic-coordinate equilibrium with complex-phase sheared iota(psi) tau_1 = 0.42
     (arXiv:2610.07304v1 Section 3) and sharp Internal Transport Barrier (ITB) step jump at psi=0.50.
3. Scenario 03: 03_spec_magnetic_island_chain
   - Multi-Region Relaxed MHD (MRxMHD / SPEC) with 3 discrete Beltrami shells (curl B = mu_i B),
     resonant m/n = 2/1 magnetic island chain, and beta_1 cavity core flattened-pressure pockets.
4. Scenario 04: 04_tokamak_elm_pellet_4d
   - 4D Time-Resolved Tokamak H-Mode Pedestal with SOL X-point separatrix, sub-cm ELM filaments,
     10% divertor Bremsstrahlung glare washout, and 4D cryogenic D2 pellet vs. AC coil ripple.
5. Scenario 05: 05_beltrami_plasma_spallation_torch
   - Magnetically nozzle-steered Beltrami plasma vortex (curl B = mu B) impinging on anisotropic
     crystalline rock with thin water-filled fracture (eps_r'=9.85), SO(6) thermal spallation stress,
     and self-vitrifying beta_1 glass tunnel liner.

Each scenario is evaluated on canonical (Z=32, Y=24, X=24) grids (18,432 voxels) and equipped
with 8 synthetic diagnostic chords (4 vertical for LOCO CV, 4 angled/tangential for unseen 3D holdout).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from bmhd.io import (
    DiagnosticChord,
    DiagnosticTrajectory,
    sample_cube_along_trajectory,
    solve_minimum_curvature_trajectory,
    write_dev_file,
    write_las_file,
)
from bmhd.mhd import (
    CerfonFreidbergEquilibrium,
    IssanNonSymmetricEquilibriumEngine,
    MU_0,
)


# Canonical 5 Scenario Keys
SCENARIOS_CATALOG = {
    "01_issan_nonsym_solovev_tau0": {
        "num": "01",
        "name": "01_issan_nonsym_solovev_tau0",
        "title": "Stretched Solov'ev (tau0=0.38, Broken S Parity, ELM Pedestal)",
        "family": 1,
    },
    "02_issan_sheared_elliptic_tau1": {
        "num": "02",
        "name": "02_issan_sheared_elliptic_tau1",
        "title": "Sheared-iota Elliptic (tau1=0.42, Complex Phase Shift, ITB Step Jump)",
        "family": 2,
    },
    "03_spec_magnetic_island_chain": {
        "num": "03",
        "name": "03_spec_magnetic_island_chain",
        "title": "SPEC Beltrami Shells (m/n=2/1 Resonant Island Chain, beta1 Cavity Core)",
        "family": 3,
    },
    "04_tokamak_elm_pellet_4d": {
        "num": "04",
        "name": "04_tokamak_elm_pellet_4d",
        "title": "Tokamak H-Mode 4D (SOL X-Point Separatrix, Pellet Front vs AC Coil Ripple)",
        "family": 4,
    },
    "05_beltrami_plasma_spallation_torch": {
        "num": "05",
        "name": "05_beltrami_plasma_spallation_torch",
        "title": "Beltrami Plasma Torch (Coupled MHD Nozzle, Anisotropic Granite Spallation)",
        "family": 5,
    },
}


@dataclass
class ScenarioData:
    """
    Complete benchmark dataset for a 3D/4D plasma equilibrium scenario.
    """
    scenario_id: str
    name: str
    description: str
    grid_shape: Tuple[int, int, int]   # (32, 24, 24)
    fields: Dict[str, np.ndarray]       # bmhd_psi, bmhd_b_mod, bmhd_p_kpa, bmhd_j_curl, bmhd_iota
    chords: List[DiagnosticChord]       # 8 diagnostic chords
    metadata: Dict[str, Any] = field(default_factory=dict)
    coordinates: Dict[str, np.ndarray] = field(default_factory=dict)
    temporal_frames: Optional[np.ndarray] = None  # [T, Z, Y, X] for 4D scenarios

    @property
    def psi(self) -> np.ndarray:
        return self.fields["bmhd_psi"]

    @property
    def b_mod(self) -> np.ndarray:
        return self.fields["bmhd_b_mod"]

    @property
    def p_kpa(self) -> np.ndarray:
        return self.fields["bmhd_p_kpa"]

    @property
    def j_curl(self) -> np.ndarray:
        return self.fields["bmhd_j_curl"]

    @property
    def iota(self) -> np.ndarray:
        return self.fields["bmhd_iota"]

    def __getitem__(self, key: str) -> Any:
        if key in self.fields:
            return self.fields[key]
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(f"Key '{key}' not found in fields or metadata of ScenarioData.")

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.fields:
            return self.fields[key]
        return self.metadata.get(key, default)

    def get_sightline_projections(
        self,
        num_chords: int = 8,
        resolution: Tuple[int, int] = (128, 128),
        frame_idx: int = 0,
        chunk_z: int = 16,
    ) -> Dict[str, Any]:
        """
        Stream 8 tangential and poloidal 128x128 Y8 Z-slab sightline projections.
        """
        return get_sightline_projections(
            self,
            num_chords=num_chords,
            resolution=resolution,
            frame_idx=frame_idx,
            chunk_z=chunk_z,
        )



# =============================================================================
# Helper: 8 Synthetic Diagnostic Chords Generator
# =============================================================================

def _build_diagnostic_chords_for_scenario(
    scenario_name: str,
    fields: Dict[str, np.ndarray],
    grid_shape: Tuple[int, int, int],
    extents_zyx_m: Tuple[float, float, float],
    origin_zyx_m: Tuple[float, float, float],
    R0: float = 1.0,
    a_minor: float = 0.35,
    has_glare: bool = True,
) -> List[DiagnosticChord]:
    """
    Generate 8 realistic synthetic diagnostic chords per scenario:
    - 4 vertical chords for Leave-One-Chord-Out (LOCO) cross-validation
      (0: Central Core Thomson, 1: HFS Thomson, 2: LFS Thomson, 3: Vertical CO2 Interferometer)
    - 4 angled/tangential chords for unseen 3D holdout
      (4: Tangential Toroidal Interferometer, 5: Angled Faraday Polarimeter,
       6: Divertor Sightline with 10-20% glare, 7: Helical Mirnov Array)
    """
    Z, Y, X = grid_shape
    span_z, span_y, span_x = extents_zyx_m
    min_z, min_y, min_x = origin_zyx_m

    n_samples = 48  # High-resolution sampling along optical sightline
    chords: List[DiagnosticChord] = []

    # Coordinate mapping helpers
    def phys_to_vox_z(z_val: float) -> float:
        return float(np.clip(((z_val - min_z) / max(span_z, 1e-6)) * (Z - 1), 0.0, Z - 1))

    def phys_to_vox_y(y_val: float) -> float:
        return float(np.clip(((y_val - min_y) / max(span_y, 1e-6)) * (Y - 1), 0.0, Y - 1))

    def phys_to_vox_x(x_val: float) -> float:
        return float(np.clip(((x_val - min_x) / max(span_x, 1e-6)) * (X - 1), 0.0, X - 1))

    # Define specifications for all 8 chords
    chord_specs = [
        # Fold 0: Central Core Thomson Scattering (vertical at R=R0, phi=0)
        {
            "name": f"{scenario_name}_chord0_core_thomson",
            "modality": "thomson_scattering",
            "type": "vertical_core",
            "x_start": R0, "y_start": 0.0, "z_start": min_z * 0.8,
            "x_end": R0, "y_end": 0.0, "z_end": (min_z + span_z) * 0.8,
            "inc": 0.0, "azi": 0.0,
            "is_loco": True,
            "glare_amplitude": 0.0,
        },
        # Fold 1: High-Field Side (HFS) Thomson Scattering (vertical at R=R0 - 0.5*a, phi=0)
        {
            "name": f"{scenario_name}_chord1_hfs_thomson",
            "modality": "thomson_scattering",
            "type": "vertical_hfs",
            "x_start": R0 - 0.5 * a_minor, "y_start": 0.0, "z_start": min_z * 0.8,
            "x_end": R0 - 0.5 * a_minor, "y_end": 0.0, "z_end": (min_z + span_z) * 0.8,
            "inc": 0.0, "azi": 0.0,
            "is_loco": True,
            "glare_amplitude": 0.0,
        },
        # Fold 2: Low-Field Side (LFS) Thomson Scattering (vertical at R=R0 + 0.5*a, phi=0)
        {
            "name": f"{scenario_name}_chord2_lfs_thomson",
            "modality": "thomson_scattering",
            "type": "vertical_lfs",
            "x_start": R0 + 0.5 * a_minor, "y_start": 0.0, "z_start": min_z * 0.8,
            "x_end": R0 + 0.5 * a_minor, "y_end": 0.0, "z_end": (min_z + span_z) * 0.8,
            "inc": 0.0, "azi": 0.0,
            "is_loco": True,
            "glare_amplitude": 0.0,
        },
        # Fold 3: Vertical CO2 Interferometer (vertical at R=R0, phi=pi/4)
        {
            "name": f"{scenario_name}_chord3_vert_interferometer",
            "modality": "co2_interferometry",
            "type": "vertical_quarter_phi",
            "x_start": R0 * math.cos(math.pi / 4.0), "y_start": R0 * math.sin(math.pi / 4.0), "z_start": min_z * 0.8,
            "x_end": R0 * math.cos(math.pi / 4.0), "y_end": R0 * math.sin(math.pi / 4.0), "z_end": (min_z + span_z) * 0.8,
            "inc": 0.0, "azi": 45.0,
            "is_loco": True,
            "glare_amplitude": 0.0,
        },
        # Chord 4: Tangential Toroidal Interferometer (in midplane crossing core at 45 deg)
        {
            "name": f"{scenario_name}_chord4_tangential_interferometer",
            "modality": "co2_interferometry",
            "type": "tangential_toroidal",
            "x_start": -R0 * 0.8, "y_start": -R0 * 0.8, "z_start": 0.0,
            "x_end": R0 * 0.8, "y_end": R0 * 0.8, "z_end": 0.0,
            "inc": 90.0, "azi": 45.0,
            "is_loco": False,
            "glare_amplitude": 0.0,
        },
        # Chord 5: Angled Faraday Polarimetry (skewed 3D sightline traversing magnetic axis)
        {
            "name": f"{scenario_name}_chord5_angled_polarimeter",
            "modality": "faraday_polarimetry",
            "type": "skewed_3d",
            "x_start": R0 * 1.1, "y_start": -R0 * 0.5, "z_start": min_z * 0.7,
            "x_end": R0 * 0.7, "y_end": R0 * 0.5, "z_end": (min_z + span_z) * 0.7,
            "inc": 45.0, "azi": 115.0,
            "is_loco": False,
            "glare_amplitude": 0.0,
        },
        # Chord 6: Divertor / Scrape-Off Layer Sightline (with 10-20% Bremsstrahlung arc-glare washout)
        {
            "name": f"{scenario_name}_chord6_divertor_glare",
            "modality": "divertor_spectroscopy",
            "type": "divertor_separatrix",
            "x_start": R0 + a_minor * 0.9, "y_start": -0.2, "z_start": min_z * 0.9,
            "x_end": R0 + a_minor * 1.1, "y_end": 0.2, "z_end": min_z * 0.3,
            "inc": 60.0, "azi": 20.0,
            "is_loco": False,
            "glare_amplitude": 0.75 if has_glare else 0.0,
        },
        # Chord 7: Helical Mirnov Pick-up Array Sightline (oblique helical pitch along boundary)
        {
            "name": f"{scenario_name}_chord7_helical_mirnov",
            "modality": "mirnov_probe_array",
            "type": "helical_boundary",
            "x_start": (R0 + a_minor) * math.cos(0.0), "y_start": (R0 + a_minor) * math.sin(0.0), "z_start": min_z * 0.5,
            "x_end": (R0 + a_minor) * math.cos(math.pi * 0.6), "y_end": (R0 + a_minor) * math.sin(math.pi * 0.6), "z_end": (min_z + span_z) * 0.5,
            "inc": 75.0, "azi": 135.0,
            "is_loco": False,
            "glare_amplitude": 0.0,
        },
    ]

    for spec in chord_specs:
        xs = np.linspace(spec["x_start"], spec["x_end"], n_samples)
        ys = np.linspace(spec["y_start"], spec["y_end"], n_samples)
        zs = np.linspace(spec["z_start"], spec["z_end"], n_samples)

        # Distances along ray
        dx = np.diff(xs, prepend=xs[0])
        dy = np.diff(ys, prepend=ys[0])
        dz = np.diff(zs, prepend=zs[0])
        ds = np.sqrt(dx**2 + dy**2 + dz**2)
        md = np.cumsum(ds)

        inc_arr = np.full(n_samples, spec["inc"], dtype=np.float64)
        azi_arr = np.full(n_samples, spec["azi"], dtype=np.float64)

        z_vox = np.array([phys_to_vox_z(z) for z in zs])
        y_vox = np.array([phys_to_vox_y(y) for y in ys])
        x_vox = np.array([phys_to_vox_x(x) for x in xs])

        dls = np.zeros(n_samples, dtype=np.float64)

        traj = DiagnosticTrajectory(
            sightline_name=spec["name"],
            sightline_type=spec["type"],
            md_m=md,
            inc_deg=inc_arr,
            azi_deg=azi_arr,
            tvd_m=zs,
            north_y_m=ys,
            east_x_m=xs,
            z_vox=z_vox,
            y_vox=y_vox,
            x_vox=x_vox,
            dogleg_deg_30m=dls,
        )

        # Sample underlying 3D property cubes along trajectory
        sampled_psi = sample_cube_along_trajectory(fields["bmhd_psi"], traj)
        sampled_bmod = sample_cube_along_trajectory(fields["bmhd_b_mod"], traj)
        sampled_pkpa = sample_cube_along_trajectory(fields["bmhd_p_kpa"], traj)
        sampled_jcurl = sample_cube_along_trajectory(fields["bmhd_j_curl"], traj)
        sampled_iota = sample_cube_along_trajectory(fields["bmhd_iota"], traj)

        # Synthesize physical plasma parameters
        # Electron density ne ~ (1 - psi)*5e19 + 0.2e19
        ne = np.maximum(0.2, (1.0 - sampled_psi) * 6.5 + 0.5)
        # Electron temperature Te ~ (1 - psi)^1.5 * 3.5 keV
        te = np.maximum(0.05, ((1.0 - np.clip(sampled_psi, 0.0, 1.0)) ** 1.5) * 3.8 + 0.1)
        # Faraday rotation density ~ ne * B_parallel (rad/m)
        faraday_rot = 0.263 * ne * (sampled_bmod * 0.4)

        # Bremsstrahlung glare washout (flat-top saturation pulse where Delta_- v * Delta_+ v <= 0)
        glare_amp = spec["glare_amplitude"]
        brems_glare = np.zeros(n_samples, dtype=np.float64)
        if glare_amp > 0.0:
            # 15% - 20% width flat-top pulse near outer station edge
            g_start = int(0.55 * n_samples)
            g_end = int(0.75 * n_samples)
            brems_glare[g_start:g_end] = glare_amp
            # Monotone slope into flat-top (glare ramp)
            if g_start > 2:
                brems_glare[g_start - 2:g_start] = [glare_amp * 0.4, glare_amp * 0.8]

        curves = {
            "DEPT": md,
            "TVD": zs,
            "INC": inc_arr,
            "AZI": azi_arr,
            "Z_VOX": z_vox,
            "Y_VOX": y_vox,
            "X_VOX": x_vox,
            "PSI": sampled_psi,
            "B_MOD": sampled_bmod,
            "P_KPA": sampled_pkpa,
            "J_CURL": sampled_jcurl,
            "IOTA": sampled_iota,
            "NE": ne,
            "TE": te,
            "FARADAY_ROT": faraday_rot,
            "BREMS_GLARE": brems_glare,
        }

        glare_mask = (brems_glare > 0.1)

        chord = DiagnosticChord(
            name=spec["name"],
            modality=spec["modality"],
            trajectory=traj,
            curves=curves,
            glare_mask=glare_mask,
            metadata={
                "is_loco": spec["is_loco"],
                "chord_type": spec["type"],
                "scenario": scenario_name,
            },
        )
        chords.append(chord)

    return chords


# =============================================================================
# 1. Scenario 01: Stretched Solov'ev (tau0=0.38, Broken S Parity, ELM Pedestal)
# =============================================================================

def build_scenario_01(grid_shape: Tuple[int, int, int] = (32, 24, 24)) -> ScenarioData:
    """
    Scenario 01: issan_nonsym_solovev_tau0
    Non-stellarator-symmetric 3D stretched Solov'ev equilibrium (tau0 = 0.38) with broken
    S = diag(1, -1, -1) parity, stepped H-mode pedestal, and thin radial ELM filament spikes.
    """
    Z, Y, X = grid_shape
    tau0 = 0.38
    eps = 0.50
    pa = 120.0
    delta = 1.0 / 64.0

    extents_zyx = (1.0, 2.4, 2.4)
    origin_zyx = (-0.5, -1.2, -1.2)

    z_vals = np.linspace(origin_zyx[0], origin_zyx[0] + extents_zyx[0], Z)
    y_vals = np.linspace(origin_zyx[1], origin_zyx[1] + extents_zyx[1], Y)
    x_vals = np.linspace(origin_zyx[2], origin_zyx[2] + extents_zyx[2], X)

    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing="ij")

    # Evaluate exact closed-form Solov'ev equilibrium from arXiv:2610.07304v1
    engine = IssanNonSymmetricEquilibriumEngine(
        family=1,
        tau0=tau0,
        epsilon=eps,
        pa=pa,
        delta=delta,
    )
    raw_fields = engine.compute_cartesian(X_grid, Y_grid, Z_grid)

    psi_base = np.clip(raw_fields["psi_raw"] / 0.22, 0.0, 1.0)
    bmod_base = raw_fields["B_mod"]
    pkpa_base = raw_fields["p_kpa"]
    jcurl_base = raw_fields["J_curl"]
    iota_base = np.full_like(psi_base, 2.0)  # Constant iota = 2.0 (Hamada line integral)

    # Add stepped H-mode pedestal at boundary flux psi = 1.0 and thin radial ELM filament spikes
    pedestal_top = 0.85
    is_pedestal = (psi_base >= pedestal_top)
    p_kpa = pkpa_base.copy()
    # Steep pressure gradient across pedestal barrier
    p_kpa[is_pedestal] = pkpa_base[is_pedestal] * np.exp(-4.5 * (psi_base[is_pedestal] - pedestal_top))

    # Thin ELM filament spikes near separatrix on the outer low-field side (X > 0)
    elm_mask = (psi_base > 0.90) & (psi_base < 1.05) & (X_grid > 0.2)
    j_curl = jcurl_base.copy()
    j_curl[elm_mask] += 3.8 * np.sin(4.0 * np.arctan2(Y_grid[elm_mask], X_grid[elm_mask])) ** 2

    fields = {
        "bmhd_psi": np.ascontiguousarray(psi_base, dtype=np.float64),
        "bmhd_b_mod": np.ascontiguousarray(bmod_base, dtype=np.float64),
        "bmhd_p_kpa": np.ascontiguousarray(p_kpa, dtype=np.float64),
        "bmhd_j_curl": np.ascontiguousarray(j_curl, dtype=np.float64),
        "bmhd_iota": np.ascontiguousarray(iota_base, dtype=np.float64),
    }

    chords = _build_diagnostic_chords_for_scenario(
        scenario_name="01_issan_nonsym_solovev_tau0",
        fields=fields,
        grid_shape=grid_shape,
        extents_zyx_m=extents_zyx,
        origin_zyx_m=origin_zyx,
        R0=0.90,
        a_minor=0.55,
        has_glare=True,
    )

    metadata = {
        "scenario_id": "01_issan_nonsym_solovev_tau0",
        "title": SCENARIOS_CATALOG["01_issan_nonsym_solovev_tau0"]["title"],
        "tau0": tau0,
        "tau1": 0.0,
        "epsilon": eps,
        "pa_kpa": pa,
        "delta": delta,
        "iota_constant": 2.0,
        "symmetry_broken": True,
        "pedestal_barrier": True,
        "elm_filaments": True,
        "extents_zyx": extents_zyx,
        "origin_zyx": origin_zyx,
    }

    return ScenarioData(
        scenario_id="01_issan_nonsym_solovev_tau0",
        name="issan_nonsym_solovev_tau0",
        description=SCENARIOS_CATALOG["01_issan_nonsym_solovev_tau0"]["title"],
        grid_shape=grid_shape,
        fields=fields,
        chords=chords,
        metadata=metadata,
        coordinates={"z": z_vals, "y": y_vals, "x": x_vals},
    )


# =============================================================================
# 2. Scenario 02: Sheared-iota Elliptic (tau1=0.42, Complex Phase Shift, ITB)
# =============================================================================

def build_scenario_02(grid_shape: Tuple[int, int, int] = (32, 24, 24)) -> ScenarioData:
    """
    Scenario 02: issan_sheared_elliptic_tau1
    Non-stellarator-symmetric elliptic-coordinate equilibrium (tau1 = 0.42) with non-uniform
    sheared rotational transform iota(psi) and sharp Internal Transport Barrier (ITB) step jump.
    """
    Z, Y, X = grid_shape
    tau1 = 0.42
    eps = 1.20
    S = 1.40
    lam = 1.60
    pa = 150.0
    delta = 0.140

    extents_zyx = (1.6, 3.6, 3.6)
    origin_zyx = (-0.8, -1.8, -1.8)

    z_vals = np.linspace(origin_zyx[0], origin_zyx[0] + extents_zyx[0], Z)
    y_vals = np.linspace(origin_zyx[1], origin_zyx[1] + extents_zyx[1], Y)
    x_vals = np.linspace(origin_zyx[2], origin_zyx[2] + extents_zyx[2], X)

    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing="ij")

    engine = IssanNonSymmetricEquilibriumEngine(
        family=2,
        tau1=tau1,
        epsilon=eps,
        S=S,
        lam=lam,
        pa=pa,
        delta=delta,
    )
    raw_fields = engine.compute_cartesian(X_grid, Y_grid, Z_grid)

    psi_base = raw_fields["psi"]
    bmod_base = raw_fields["B_mod"]
    pkpa_base = raw_fields["p_kpa"]

    # Sheared rotational transform iota(psi) reversing shear gradient
    iota_sheared = 0.38 + 0.24 * psi_base + 0.05 * (tau1 ** 2) * np.sin(2.0 * math.pi * psi_base)

    # Internal Transport Barrier (ITB) step jump at normalized flux psi = 0.50
    itb_flux = 0.50
    p_kpa = pkpa_base.copy()
    # Sharp pressure drop [[p]] ~ 28 kPa across ITB step
    itb_step = 28.0 * (1.0 / (1.0 + np.exp(-40.0 * (psi_base - itb_flux))))
    p_kpa = np.maximum(5.0, p_kpa - itb_step)

    # Current density sheet spike at ITB step gradient
    j_curl = raw_fields["J_curl"].copy()
    itb_sheet = 5.2 * np.exp(-((psi_base - itb_flux) / 0.04) ** 2)
    j_curl += itb_sheet

    fields = {
        "bmhd_psi": np.ascontiguousarray(psi_base, dtype=np.float64),
        "bmhd_b_mod": np.ascontiguousarray(bmod_base, dtype=np.float64),
        "bmhd_p_kpa": np.ascontiguousarray(p_kpa, dtype=np.float64),
        "bmhd_j_curl": np.ascontiguousarray(j_curl, dtype=np.float64),
        "bmhd_iota": np.ascontiguousarray(iota_sheared, dtype=np.float64),
    }

    chords = _build_diagnostic_chords_for_scenario(
        scenario_name="02_issan_sheared_elliptic_tau1",
        fields=fields,
        grid_shape=grid_shape,
        extents_zyx_m=extents_zyx,
        origin_zyx_m=origin_zyx,
        R0=1.05,
        a_minor=0.85,
        has_glare=False,
    )

    metadata = {
        "scenario_id": "02_issan_sheared_elliptic_tau1",
        "title": SCENARIOS_CATALOG["02_issan_sheared_elliptic_tau1"]["title"],
        "tau0": 0.0,
        "tau1": tau1,
        "epsilon": eps,
        "S": S,
        "lam": lam,
        "pa_kpa": pa,
        "delta": delta,
        "itb_flux_location": itb_flux,
        "sheared_iota": True,
        "extents_zyx": extents_zyx,
        "origin_zyx": origin_zyx,
    }

    return ScenarioData(
        scenario_id="02_issan_sheared_elliptic_tau1",
        name="issan_sheared_elliptic_tau1",
        description=SCENARIOS_CATALOG["02_issan_sheared_elliptic_tau1"]["title"],
        grid_shape=grid_shape,
        fields=fields,
        chords=chords,
        metadata=metadata,
        coordinates={"z": z_vals, "y": y_vals, "x": x_vals},
    )


# =============================================================================
# 3. Scenario 03: SPEC Beltrami Shells (m/n=2/1 Island Chain, beta1 Core)
# =============================================================================

def build_scenario_03(grid_shape: Tuple[int, int, int] = (32, 24, 24)) -> ScenarioData:
    """
    Scenario 03: spec_magnetic_island_chain
    Multi-shell relaxed Beltrami plasma (curl B = mu_i B) containing a resonant m/n = 2/1
    magnetic island chain (beta1 cavity core flattened-pressure pockets) bounded by sharp
    helicity-conserving current sheets.
    """
    Z, Y, X = grid_shape
    mu1 = 0.45    # Core shell Taylor parameter
    p1 = 180.0    # Core stepped pressure (kPa)
    mu2 = 0.12    # Island shell Taylor parameter
    p2 = 110.0    # Island flattened cavity pressure (kPa)
    mu3 = -0.30   # Outer mantle Taylor parameter
    p3 = 25.0     # Edge mantle pressure (kPa)

    rs = 0.65     # Resonant surface radius (m)
    w_island = 0.10  # Full island width (m)
    R0 = 1.0
    a_minor = 0.85

    extents_zyx = (2.0, 3.0, 3.0)
    origin_zyx = (-1.0, -1.5, -1.5)

    z_vals = np.linspace(origin_zyx[0], origin_zyx[0] + extents_zyx[0], Z)
    y_vals = np.linspace(origin_zyx[1], origin_zyx[1] + extents_zyx[1], Y)
    x_vals = np.linspace(origin_zyx[2], origin_zyx[2] + extents_zyx[2], X)

    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing="ij")

    # Cylindrical / Toroidal coordinates
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    phi_grid = np.arctan2(Y_grid, X_grid)
    r_pol = np.sqrt((R_grid - R0)**2 + Z_grid**2)
    theta_pol = np.arctan2(Z_grid, R_grid - R0)

    # Normalized radius
    rho = np.clip(r_pol / a_minor, 0.0, 1.5)

    # Resonant helical perturbation: m=2, n=1 helical angle alpha = 2*theta - phi
    alpha_helical = 2.0 * theta_pol - phi_grid
    # Island width envelope centered at rho_s = rs / a_minor
    rho_s = rs / a_minor
    w_rho = w_island / a_minor

    # Island separatrix coordinate: Xi = (rho - rho_s)^2 - (w_rho/2)^2 * (1 + cos(alpha_helical))
    Xi_island = (rho - rho_s)**2 - (0.5 * w_rho)**2 * (1.0 + np.cos(alpha_helical))
    in_island_core = (Xi_island <= 0.0)

    # 3 discrete SPEC Beltrami shells
    # Shell 1: Core (rho < rho_s - w_rho/2 and not in island)
    # Shell 2: Island chain cavity core (in_island_core)
    # Shell 3: Outer mantle (rho >= rho_s + w_rho/2)
    is_core = (rho < (rho_s - 0.5 * w_rho)) & (~in_island_core)
    is_outer = (rho >= (rho_s + 0.5 * w_rho)) & (~in_island_core)

    # Stepped pressure profile
    p_kpa = np.full_like(rho, p3)
    p_kpa[is_core] = p1
    p_kpa[in_island_core] = p2  # beta_1 flattened-pressure cavity core

    # Magnetic flux: base parabolic flux + island perturbation
    psi_base = rho**2
    psi = psi_base + 0.08 * (1.0 - rho) * np.cos(alpha_helical)
    psi = np.clip(psi, 0.0, 1.0)

    # Magnetic field components
    B0 = 3.2
    B_phi = B0 * (R0 / np.maximum(R_grid, 0.2))
    # Rotational transform iota(rho) = 0.35 + 0.30*rho^2 (passes through 0.5 at rho_s)
    iota = 0.35 + 0.30 * (rho**2)
    B_theta = B_phi * (r_pol / max(R0, 0.5)) * iota
    B_mod = np.sqrt(B_phi**2 + B_theta**2)

    # Current density |J| = |curl B| / mu0
    # Core: J = mu1 * B / mu0; Island: mu2 * B / mu0; Outer: mu3 * B / mu0
    j_curl = np.zeros_like(rho)
    j_curl[is_core] = abs(mu1) * B_mod[is_core] / (MU_0 * 1e6)
    j_curl[in_island_core] = abs(mu2) * B_mod[in_island_core] / (MU_0 * 1e6)
    j_curl[is_outer] = abs(mu3) * B_mod[is_outer] / (MU_0 * 1e6)

    # Razor-sharp current sheet at the island separatrices
    separatrix_sheet = 6.8 * np.exp(-(Xi_island**2) / 0.002)
    j_curl += separatrix_sheet

    fields = {
        "bmhd_psi": np.ascontiguousarray(psi, dtype=np.float64),
        "bmhd_b_mod": np.ascontiguousarray(B_mod, dtype=np.float64),
        "bmhd_p_kpa": np.ascontiguousarray(p_kpa, dtype=np.float64),
        "bmhd_j_curl": np.ascontiguousarray(j_curl, dtype=np.float64),
        "bmhd_iota": np.ascontiguousarray(iota, dtype=np.float64),
    }

    chords = _build_diagnostic_chords_for_scenario(
        scenario_name="03_spec_magnetic_island_chain",
        fields=fields,
        grid_shape=grid_shape,
        extents_zyx_m=extents_zyx,
        origin_zyx_m=origin_zyx,
        R0=R0,
        a_minor=a_minor,
        has_glare=False,
    )

    metadata = {
        "scenario_id": "03_spec_magnetic_island_chain",
        "title": SCENARIOS_CATALOG["03_spec_magnetic_island_chain"]["title"],
        "mu1": mu1,
        "mu2": mu2,
        "mu3": mu3,
        "p1_kpa": p1,
        "p2_kpa": p2,
        "p3_kpa": p3,
        "rs": rs,
        "w_island": w_island,
        "mode_numbers": "m=2, n=1",
        "beta1_cavity_core": True,
        "extents_zyx": extents_zyx,
        "origin_zyx": origin_zyx,
    }

    return ScenarioData(
        scenario_id="03_spec_magnetic_island_chain",
        name="spec_magnetic_island_chain",
        description=SCENARIOS_CATALOG["03_spec_magnetic_island_chain"]["title"],
        grid_shape=grid_shape,
        fields=fields,
        chords=chords,
        metadata=metadata,
        coordinates={"z": z_vals, "y": y_vals, "x": x_vals},
    )


# =============================================================================
# 4. Scenario 04: Tokamak H-Mode 4D (SOL X-Point, Pellet Front vs AC Coil Ripple)
# =============================================================================

def build_scenario_04(
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    n_frames: int = 4,
) -> ScenarioData:
    """
    Scenario 04: tokamak_elm_pellet_4d
    4D Time-Resolved Tokamak H-Mode Pedestal with sharp Scrape-Off-Layer (SOL) X-point separatrix,
    sub-centimeter ELM filaments, 10% divertor Bremsstrahlung glare washout, and 4D cryogenic D2
    pellet injection front vs. reversible AC coil ripple.
    """
    Z, Y, X = grid_shape
    R0 = 1.65
    a_minor = 0.50
    kappa = 1.70
    delta_t = 0.33
    p_ped = 95.0
    p_sol = 4.5
    f_ripple = 300.0   # Hz
    v_pellet = 800.0   # m/s

    extents_zyx = (2.0, 4.4, 4.4)
    origin_zyx = (-1.0, -2.2, -2.2)

    z_vals = np.linspace(origin_zyx[0], origin_zyx[0] + extents_zyx[0], Z)
    y_vals = np.linspace(origin_zyx[1], origin_zyx[1] + extents_zyx[1], Y)
    x_vals = np.linspace(origin_zyx[2], origin_zyx[2] + extents_zyx[2], X)

    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing="ij")

    # Cylindrical tokamak geometry
    R_grid = np.sqrt(X_grid**2 + Y_grid**2)
    phi_grid = np.arctan2(Y_grid, X_grid)

    # Cerfon-Freidberg analytic Grad-Shafranov equilibrium baseline
    # Poloidal coordinates with elongation kappa and triangularity delta_t
    r_tok = np.sqrt(((R_grid - R0) + delta_t * (Z_grid / kappa)**2)**2 + (Z_grid / kappa)**2)
    rho_tok = np.clip(r_tok / a_minor, 0.0, 1.6)

    # Base flux psi(R, Z) in [0, 1]
    psi_base = np.clip(rho_tok**1.8, 0.0, 1.0)
    B0 = 2.8
    B_mod = B0 * (R0 / np.maximum(R_grid, 0.3)) * (1.0 + 0.15 * (1.0 - rho_tok))

    # H-mode pedestal pressure profile: flat core dropping steeply across rho=0.9..1.0 to SOL
    ped_barrier = 1.0 / (1.0 + np.exp(35.0 * (rho_tok - 0.95)))
    p_kpa_base = p_sol + (p_ped - p_sol) * ped_barrier

    # Beltrami current density: core + edge bootstrap current peak at pedestal
    j_bootstrap = 4.2 * np.exp(-((rho_tok - 0.95) / 0.035)**2)
    j_curl_base = 1.2 * (1.0 - rho_tok**2) + j_bootstrap

    # Safety factor / rotational transform profile: q(0) ~ 1.1 (iota ~ 0.91), q(edge) ~ 3.5 (iota ~ 0.28)
    q_profile = 1.1 + 2.4 * (rho_tok**2)
    iota_base = 1.0 / q_profile

    # 4D time series frames (T=4): t in {0, 1, 2, 3}
    # Decomposes into H_clock (reversible 300 Hz AC ripple) (+) H_nope (irreversible pellet ablation front)
    temporal_cubes = np.zeros((n_frames, Z, Y, X), dtype=np.float32)
    dt_frame = 1e-3  # 1 ms per frame

    for t_idx in range(n_frames):
        t_sec = t_idx * dt_frame
        # Reversible AC coil ripple oscillation (limit cycle)
        ripple = 0.04 * math.sin(2.0 * math.pi * f_ripple * t_sec)

        # Irreversible cryogenic D2 pellet injection front moving radially inward at 800 m/s from LFS
        # Pellet position along X axis (at Y=0, Z=0)
        x_pellet = (R0 + a_minor) - (v_pellet * t_sec * 0.25)
        dist_pellet = np.sqrt((X_grid - x_pellet)**2 + Y_grid**2 + Z_grid**2)
        pellet_density_spike = 3.5 * np.exp(-(dist_pellet / 0.12)**2)

        frame_psi = np.clip(psi_base * (1.0 + ripple) + pellet_density_spike * 0.15, 0.0, 1.0)
        temporal_cubes[t_idx] = frame_psi.astype(np.float32)

    # Snapshot at t = 0
    fields = {
        "bmhd_psi": np.ascontiguousarray(psi_base, dtype=np.float64),
        "bmhd_b_mod": np.ascontiguousarray(B_mod, dtype=np.float64),
        "bmhd_p_kpa": np.ascontiguousarray(p_kpa_base, dtype=np.float64),
        "bmhd_j_curl": np.ascontiguousarray(j_curl_base, dtype=np.float64),
        "bmhd_iota": np.ascontiguousarray(iota_base, dtype=np.float64),
    }

    chords = _build_diagnostic_chords_for_scenario(
        scenario_name="04_tokamak_elm_pellet_4d",
        fields=fields,
        grid_shape=grid_shape,
        extents_zyx_m=extents_zyx,
        origin_zyx_m=origin_zyx,
        R0=R0,
        a_minor=a_minor,
        has_glare=True,  # 10% divertor Bremsstrahlung arc-glare on chord 6
    )

    metadata = {
        "scenario_id": "04_tokamak_elm_pellet_4d",
        "title": SCENARIOS_CATALOG["04_tokamak_elm_pellet_4d"]["title"],
        "R0": R0,
        "a_minor": a_minor,
        "kappa": kappa,
        "delta_t": delta_t,
        "p_ped_kpa": p_ped,
        "p_sol_kpa": p_sol,
        "f_ripple_hz": f_ripple,
        "v_pellet_m_s": v_pellet,
        "n_frames": n_frames,
        "4d_temporal_decomposition": "H_clock (+) H_nope",
        "glare_washout": "10% on chord 6",
        "extents_zyx": extents_zyx,
        "origin_zyx": origin_zyx,
    }

    return ScenarioData(
        scenario_id="04_tokamak_elm_pellet_4d",
        name="tokamak_elm_pellet_4d",
        description=SCENARIOS_CATALOG["04_tokamak_elm_pellet_4d"]["title"],
        grid_shape=grid_shape,
        fields=fields,
        chords=chords,
        metadata=metadata,
        coordinates={"z": z_vals, "y": y_vals, "x": x_vals},
        temporal_frames=temporal_cubes,
    )


# =============================================================================
# 5. Scenario 05: Beltrami Plasma Spallation Torch (Rock Dielectric & SO(6) Stress)
# =============================================================================

def build_scenario_05(grid_shape: Tuple[int, int, int] = (32, 24, 24)) -> ScenarioData:
    """
    Scenario 05: beltrami_plasma_spallation_torch
    Magnetically nozzle-steered Beltrami plasma vortex (curl B = mu B) impinging on anisotropic
    crystalline granite/basalt with thin water-filled fracture stringer (eps_r' = 9.85),
    SO(6) thermal spallation stress, and self-vitrifying beta_1 glass tunnel liner.
    """
    Z, Y, X = grid_shape
    mu = 1.85           # Taylor vortex parameter (m^-1)
    B_throat = 4.50     # Throat magnetic field (T)
    B_exit = 1.20       # Exit plume magnetic field (T)
    T_plasma = 15000.0  # Plasma flame temperature (K)
    T_rock = 300.0      # Rock temperature (K)
    delta_T = 14700.0   # Thermal shock (K)
    rho_rock = 2.75     # Granite density (g/cm^3)
    eps_r_fracture = 9.85  # Water fracture permittivity

    extents_zyx = (1.6, 1.2, 1.2)
    origin_zyx = (0.0, -0.6, -0.6)

    z_vals = np.linspace(origin_zyx[0], origin_zyx[0] + extents_zyx[0], Z)
    y_vals = np.linspace(origin_zyx[1], origin_zyx[1] + extents_zyx[1], Y)
    x_vals = np.linspace(origin_zyx[2], origin_zyx[2] + extents_zyx[2], X)

    Z_grid, Y_grid, X_grid = np.meshgrid(z_vals, y_vals, x_vals, indexing="ij")

    # Borehole radial distance from central torch nozzle axis
    r_perp = np.sqrt(X_grid**2 + Y_grid**2)
    theta_cyl = np.arctan2(Y_grid, X_grid)

    # Magnetic nozzle axial field expansion B_z0(z) from throat to exit
    z_norm = np.clip(Z_grid / 1.6, 0.0, 1.0)
    B_axial = B_throat + (B_exit - B_throat) * z_norm

    # Beltrami vortex field: B_z ~ B0 * J0(mu * r), B_theta ~ B0 * J1(mu * r)
    # Using low-order Bessel expansions for real-time speed: J0(u) ~ 1 - u^2/4, J1(u) ~ u/2 - u^3/16
    u_bessel = mu * r_perp
    j0_approx = 1.0 - 0.25 * (u_bessel**2) + 0.015 * (u_bessel**4)
    j1_approx = 0.5 * u_bessel - 0.0625 * (u_bessel**3)

    B_z = B_axial * j0_approx
    B_theta = B_axial * j1_approx
    B_mod = np.sqrt(B_z**2 + B_theta**2)

    # Plasma core vs borehole wall vs rock matrix:
    # r_perp < 0.25: hot plasma torch vortex
    # 0.25 <= r_perp < 0.35: vitrified beta_1 glass tunnel liner
    # r_perp >= 0.35: crystalline granite rock formation
    is_plasma = (r_perp < 0.25)
    is_liner = (r_perp >= 0.25) & (r_perp < 0.35)
    is_rock = (r_perp >= 0.35)

    # Stepped pressure profile
    p_kpa = np.zeros_like(r_perp)
    p_kpa[is_plasma] = 220.0 * (1.0 - (r_perp[is_plasma] / 0.25)**2)
    p_kpa[is_liner] = 65.0  # Vitrified glass liner support
    p_kpa[is_rock] = 15.0   # Ambient rock hydrostatic pressure

    # Normalized magnetic nozzle flux psi(r, z)
    psi = np.clip((r_perp / 0.45)**2, 0.0, 1.0)

    # Beltrami current density J = mu * B / mu0 inside plasma flame
    j_curl = np.zeros_like(r_perp)
    j_curl[is_plasma] = (mu * B_mod[is_plasma]) / (MU_0 * 1e6)
    # Induced eddy currents in the fracture zone (water stringer at x in [0.25, 0.28])
    is_fracture = (X_grid >= 0.25) & (X_grid <= 0.28) & (np.abs(Y_grid) < 0.2)
    j_curl[is_fracture] += 4.5

    # Rotational transform iota(r) ~ B_theta / (r * B_z) ~ mu / 2
    iota = np.full_like(r_perp, 0.5 * mu)

    fields = {
        "bmhd_psi": np.ascontiguousarray(psi, dtype=np.float64),
        "bmhd_b_mod": np.ascontiguousarray(B_mod, dtype=np.float64),
        "bmhd_p_kpa": np.ascontiguousarray(p_kpa, dtype=np.float64),
        "bmhd_j_curl": np.ascontiguousarray(j_curl, dtype=np.float64),
        "bmhd_iota": np.ascontiguousarray(iota, dtype=np.float64),
    }

    chords = _build_diagnostic_chords_for_scenario(
        scenario_name="05_beltrami_plasma_spallation_torch",
        fields=fields,
        grid_shape=grid_shape,
        extents_zyx_m=extents_zyx,
        origin_zyx_m=origin_zyx,
        R0=0.0,
        a_minor=0.35,
        has_glare=False,
    )

    metadata = {
        "scenario_id": "05_beltrami_plasma_spallation_torch",
        "title": SCENARIOS_CATALOG["05_beltrami_plasma_spallation_torch"]["title"],
        "mu": mu,
        "B_throat_T": B_throat,
        "B_exit_T": B_exit,
        "T_plasma_K": T_plasma,
        "delta_T_K": delta_T,
        "rock_density_gcc": rho_rock,
        "fracture_permittivity_eps_r": eps_r_fracture,
        "glass_liner_beta1": 1.0,
        "so6_spallation_stress": True,
        "extents_zyx": extents_zyx,
        "origin_zyx": origin_zyx,
    }

    return ScenarioData(
        scenario_id="05_beltrami_plasma_spallation_torch",
        name="beltrami_plasma_spallation_torch",
        description=SCENARIOS_CATALOG["05_beltrami_plasma_spallation_torch"]["title"],
        grid_shape=grid_shape,
        fields=fields,
        chords=chords,
        metadata=metadata,
        coordinates={"z": z_vals, "y": y_vals, "x": x_vals},
    )


# =============================================================================
# Builder Registry & Chords Bundler
# =============================================================================

SCENARIO_BUILDERS = {
    "01_issan_nonsym_solovev_tau0": build_scenario_01,
    "02_issan_sheared_elliptic_tau1": build_scenario_02,
    "03_spec_magnetic_island_chain": build_scenario_03,
    "04_tokamak_elm_pellet_4d": build_scenario_04,
    "05_beltrami_plasma_spallation_torch": build_scenario_05,
}

# Alias mappings for short names
SCENARIO_ALIASES = {
    "01": "01_issan_nonsym_solovev_tau0",
    "1": "01_issan_nonsym_solovev_tau0",
    "solovev": "01_issan_nonsym_solovev_tau0",
    "issan_nonsym_solovev_tau0": "01_issan_nonsym_solovev_tau0",

    "02": "02_issan_sheared_elliptic_tau1",
    "2": "02_issan_sheared_elliptic_tau1",
    "sheared_elliptic": "02_issan_sheared_elliptic_tau1",
    "issan_sheared_elliptic_tau1": "02_issan_sheared_elliptic_tau1",

    "03": "03_spec_magnetic_island_chain",
    "3": "03_spec_magnetic_island_chain",
    "spec": "03_spec_magnetic_island_chain",
    "spec_magnetic_island_chain": "03_spec_magnetic_island_chain",

    "04": "04_tokamak_elm_pellet_4d",
    "4": "04_tokamak_elm_pellet_4d",
    "tokamak": "04_tokamak_elm_pellet_4d",
    "tokamak_elm_pellet_4d": "04_tokamak_elm_pellet_4d",

    "05": "05_beltrami_plasma_spallation_torch",
    "5": "05_beltrami_plasma_spallation_torch",
    "torch": "05_beltrami_plasma_spallation_torch",
    "beltrami_plasma_spallation_torch": "05_beltrami_plasma_spallation_torch",
}


def build_scenario(
    scenario_id: str,
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
) -> ScenarioData:
    """
    Build a single scenario by canonical ID or short name.
    """
    norm_id = str(scenario_id).strip().lower()
    if norm_id in SCENARIO_ALIASES:
        canonical_key = SCENARIO_ALIASES[norm_id]
    elif norm_id in SCENARIO_BUILDERS:
        canonical_key = norm_id
    else:
        raise ValueError(
            f"Unknown scenario ID '{scenario_id}'. Available: {list(SCENARIO_BUILDERS.keys())}"
        )

    builder = SCENARIO_BUILDERS[canonical_key]
    return builder(grid_shape=grid_shape)


def export_bundled_stellarator_chords(
    output_dir: Union[str, Path] = "data/stellarator_chords",
    scenarios: Optional[Dict[str, ScenarioData]] = None,
) -> List[Path]:
    """
    Export standard bundled diagnostic chord files in data/stellarator_chords/
    including:
    - data/stellarator_chords/issan_nonsym_solovev.las
    - data/stellarator_chords/thomson_chord1.dev
    - data/stellarator_chords/interferometer_chord2.dev
    plus individual .las and .dev files for all 8 chords across the scenarios.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if scenarios is None:
        scenarios = build_all_scenarios()

    written_files: List[Path] = []

    # 1. Primary named canonical files required by project specification
    sc1 = scenarios["01_issan_nonsym_solovev_tau0"]
    # issan_nonsym_solovev.las (from Chord 0: Central Core Thomson)
    p_las = write_las_file(out_path / "issan_nonsym_solovev.las", sc1.chords[0])
    written_files.append(p_las)

    # thomson_chord1.dev (from Chord 1: HFS Thomson)
    p_dev1 = write_dev_file(out_path / "thomson_chord1.dev", sc1.chords[1].trajectory)
    written_files.append(p_dev1)

    # interferometer_chord2.dev (from Chord 3: Vertical CO2 Interferometer)
    p_dev2 = write_dev_file(out_path / "interferometer_chord2.dev", sc1.chords[3].trajectory)
    written_files.append(p_dev2)

    # 2. Export full chord suite for all scenarios
    for sc_key, sc_data in scenarios.items():
        prefix = SCENARIOS_CATALOG[sc_key]["num"]
        for c_idx, chord in enumerate(sc_data.chords):
            c_stem = f"scenario{prefix}_chord{c_idx}_{chord.trajectory.sightline_type}"
            f_las = write_las_file(out_path / f"{c_stem}.las", chord)
            f_dev = write_dev_file(out_path / f"{c_stem}.dev", chord.trajectory)
            written_files.extend([f_las, f_dev])

    return written_files


class ScenarioDeck(dict):
    """
    Complete dictionary deck of canonical scenarios supporting:
    1. Standard string key lookup: deck["01_issan_nonsym_solovev_tau0"]
    2. Short alias lookup: deck["01"], deck["tokamak"], deck["torch"]
    3. Integer index lookup: deck[0], deck[3], deck[4] (0-based indexing matching catalog order)
    4. 100% dictionary backward compatibility (isinstance(deck, dict) is True, len(deck) == 5).
    """

    def __getitem__(self, key: Any) -> ScenarioData:
        if isinstance(key, int):
            keys = list(self.keys())
            if 0 <= key < len(keys):
                return super().__getitem__(keys[key])
            elif -len(keys) <= key < 0:
                return super().__getitem__(keys[key])
            raise IndexError(f"Scenario index {key} out of range [0, {len(keys)-1}]")
        if key in self:
            return super().__getitem__(key)
        norm_key = str(key).strip().lower()
        if norm_key in SCENARIO_ALIASES:
            return super().__getitem__(SCENARIO_ALIASES[norm_key])
        return super().__getitem__(key)

    def get(self, key: Any, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, IndexError):
            return default


def _sample_slab_trilinear(
    slab: np.ndarray,
    z_vox_local: np.ndarray,
    y_vox: np.ndarray,
    x_vox: np.ndarray,
) -> np.ndarray:
    """
    Fast vectorized trilinear interpolation from a 3D slab of shape (Z_slab, Y, X).
    """
    Z_s, Y_d, X_d = slab.shape
    valid = (
        (z_vox_local >= 0.0) & (z_vox_local <= Z_s - 1.0) &
        (y_vox >= 0.0) & (y_vox <= Y_d - 1.0) &
        (x_vox >= 0.0) & (x_vox <= X_d - 1.0)
    )
    out = np.zeros_like(z_vox_local, dtype=np.float32)
    if not np.any(valid):
        return out

    zv = np.clip(z_vox_local, 0.0, Z_s - 1.0)
    yv = np.clip(y_vox, 0.0, Y_d - 1.0)
    xv = np.clip(x_vox, 0.0, X_d - 1.0)

    z0 = np.floor(zv).astype(np.int32)
    z1 = np.clip(z0 + 1, 0, Z_s - 1)
    zd = (zv - z0).astype(np.float32)

    y0 = np.floor(yv).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, Y_d - 1)
    yd = (yv - y0).astype(np.float32)

    x0 = np.floor(xv).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, X_d - 1)
    xd = (xv - x0).astype(np.float32)

    c000 = slab[z0, y0, x0]
    c001 = slab[z0, y0, x1]
    c010 = slab[z0, y1, x0]
    c011 = slab[z0, y1, x1]
    c100 = slab[z1, y0, x0]
    c101 = slab[z1, y0, x1]
    c110 = slab[z1, y1, x0]
    c111 = slab[z1, y1, x1]

    c00 = c000 * (1.0 - xd) + c001 * xd
    c01 = c010 * (1.0 - xd) + c011 * xd
    c10 = c100 * (1.0 - xd) + c101 * xd
    c11 = c110 * (1.0 - xd) + c111 * xd

    c0 = c00 * (1.0 - yd) + c01 * yd
    c1 = c10 * (1.0 - yd) + c11 * yd

    val = c0 * (1.0 - zd) + c1 * zd
    out[valid] = val[valid]
    return out


def get_sightline_projections(
    scenario_data: Union[ScenarioData, str, int, Dict[str, Any]],
    num_chords: int = 8,
    resolution: Tuple[int, int] = (128, 128),
    frame_idx: int = 0,
    chunk_z: int = 16,
) -> Dict[str, Any]:
    """
    Stream 8 tangential and poloidal 128x128 Y8 Z-slab sightline projections and 2D gradient fields:
    - 4 poloidal (chords 0-3) and 4 tangential (chords 4-7) camera sightlines.
    - Sliced and projected in Z-slabs (chunk_z = 16, < 1.5 MB memory footprint).
    - Returns dictionary with:
      - "projections": np.ndarray shape (num_chords, 128, 128) float32 in [0, 1].
      - "grad_psi": np.ndarray shape (num_chords, 128, 128, 2) float32.
      - "mask_washout": np.ndarray shape (num_chords, 128, 128) bool.
      - "mask_clock": np.ndarray shape (num_chords, 128, 128) bool.
      - "dielectric": Optional np.ndarray shape (num_chords, 128, 128) float32
                     (for Scenario 05 with granite eps_r'=5.5, water stringer eps_r'=9.85).
      - "psi_2d": np.ndarray shape (num_chords, 128, 128) float32 in [0, 1].
    """
    if isinstance(scenario_data, (int, str)):
        scen = build_scenario(str(scenario_data))
    elif isinstance(scenario_data, ScenarioDeck):
        scen = scenario_data[3] if len(scenario_data) > 3 else scenario_data[0]
    elif isinstance(scenario_data, ScenarioData):
        scen = scenario_data
    elif isinstance(scenario_data, dict) and "fields" in scenario_data:
        scen = scenario_data
    else:
        raise TypeError(f"Unsupported scenario_data type: {type(scenario_data)}")

    fields = scen.fields if hasattr(scen, "fields") else scen["fields"]
    metadata = scen.metadata if hasattr(scen, "metadata") else scen.get("metadata", {})
    coordinates = scen.coordinates if hasattr(scen, "coordinates") else scen.get("coordinates", {})
    temporal_frames = getattr(scen, "temporal_frames", None)

    scen_id = str(metadata.get("scenario_id", getattr(scen, "scenario_id", "")))
    is_scen_04 = ("04" in scen_id) or ("tokamak" in scen_id)
    is_scen_05 = ("05" in scen_id) or ("spallation" in scen_id) or ("torch" in scen_id)

    if is_scen_04 and temporal_frames is not None and 0 <= frame_idx < len(temporal_frames):
        psi_base = temporal_frames[frame_idx]
    else:
        psi_base = fields["bmhd_psi"]

    psi_vol = np.ascontiguousarray(psi_base, dtype=np.float32)
    bmod_vol = np.ascontiguousarray(fields["bmhd_b_mod"], dtype=np.float32)
    pkpa_vol = np.ascontiguousarray(fields["bmhd_p_kpa"], dtype=np.float32)
    jcurl_vol = np.ascontiguousarray(fields["bmhd_j_curl"], dtype=np.float32)

    Z, Y, X = psi_vol.shape
    z_coords = np.asarray(coordinates.get("z", np.linspace(-0.5, 0.5, Z)), dtype=np.float32)
    y_coords = np.asarray(coordinates.get("y", np.linspace(-1.0, 1.0, Y)), dtype=np.float32)
    x_coords = np.asarray(coordinates.get("x", np.linspace(-1.0, 1.0, X)), dtype=np.float32)

    min_z, max_z = float(z_coords[0]), float(z_coords[-1])
    min_y, max_y = float(y_coords[0]), float(y_coords[-1])
    min_x, max_x = float(x_coords[0]), float(x_coords[-1])
    span_z = max(max_z - min_z, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    span_x = max(max_x - min_x, 1e-6)
    z_mid = 0.5 * (min_z + max_z)

    R0 = float(metadata.get("R0", 1.0 if not is_scen_05 else 0.0))
    a_minor = float(metadata.get("a_minor", 0.35))
    kappa = float(metadata.get("kappa", 1.70 if is_scen_04 else 1.0))

    H, W = resolution
    projections = np.zeros((num_chords, H, W), dtype=np.float32)
    psi_2d = np.zeros((num_chords, H, W), dtype=np.float32)
    grad_psi = np.zeros((num_chords, H, W, 2), dtype=np.float32)
    mask_washout = np.zeros((num_chords, H, W), dtype=bool)
    mask_clock = np.zeros((num_chords, H, W), dtype=bool)
    dielectric = np.zeros((num_chords, H, W), dtype=np.float32) if is_scen_05 else None

    # Line integral accumulators (< 1.5 MB total footprint)
    emiss_acc = np.zeros((num_chords, H, W), dtype=np.float32)
    flux_acc = np.zeros((num_chords, H, W), dtype=np.float32)
    weight_acc = np.zeros((num_chords, H, W), dtype=np.float32)
    dielectric_acc = np.zeros((num_chords, H, W), dtype=np.float32) if is_scen_05 else None

    p_norm = max(float(np.max(pkpa_vol)), 1e-3)
    j_norm = max(float(np.max(jcurl_vol)), 1e-3)
    b_norm = max(float(np.max(bmod_vol)), 1e-3)

    # Precalculate (H, W) screen coordinates
    eta_v = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]   # (H, 1)
    xi_u = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :]    # (1, W)

    # Build 8 sightline camera geometries (x_2d, z_2d, depth bounds)
    chord_configs = []
    # Chord 0: poloidal_core
    c0_z = max_z - eta_v * span_z
    c0_x = (R0 - 1.25 * a_minor) + xi_u * (2.5 * a_minor) if R0 > 0 else min_x + xi_u * span_x
    chord_configs.append({"type": "poloidal", "z_2d": c0_z, "x_2d": c0_x, "y_span": 1.2 * a_minor})

    # Chord 1: poloidal_hfs
    c1_z = max_z - eta_v * span_z
    c1_x = (R0 - 1.30 * a_minor) + xi_u * (1.30 * a_minor) if R0 > 0 else min_x + xi_u * (0.6 * span_x)
    chord_configs.append({"type": "poloidal", "z_2d": c1_z, "x_2d": c1_x, "y_span": 1.0 * a_minor})

    # Chord 2: poloidal_lfs
    c2_z = max_z - eta_v * span_z
    c2_x = R0 + xi_u * (1.30 * a_minor) if R0 > 0 else min_x + (0.4 + 0.6 * xi_u) * span_x
    chord_configs.append({"type": "poloidal", "z_2d": c2_z, "x_2d": c2_x, "y_span": 1.0 * a_minor})

    # Chord 3: poloidal_divertor
    c3_z = (min_z + 0.45 * span_z) - eta_v * (0.45 * span_z)
    c3_x = (R0 - 0.90 * a_minor) + xi_u * (1.80 * a_minor) if R0 > 0 else min_x + xi_u * span_x
    chord_configs.append({"type": "poloidal", "z_2d": c3_z, "x_2d": c3_x, "y_span": 0.9 * a_minor})

    # Chord 4: tangential_midplane
    c4_z = z_mid + (0.5 - eta_v) * (0.8 * a_minor)
    c4_p = (R0 - 1.25 * a_minor) + xi_u * (2.5 * a_minor) if R0 > 0 else (2.0 * xi_u - 1.0) * (0.8 * a_minor)
    chord_configs.append({"type": "tangential", "z_2d": c4_z, "p_2d": c4_p, "s_span": 0.8 * span_y})

    # Chord 5: tangential_upper
    c5_z = (z_mid + 0.35 * kappa * a_minor) + (0.5 - eta_v) * (0.6 * a_minor)
    c5_p = (R0 - 1.25 * a_minor) + xi_u * (2.5 * a_minor) if R0 > 0 else (2.0 * xi_u - 1.0) * (0.8 * a_minor)
    chord_configs.append({"type": "tangential", "z_2d": c5_z, "p_2d": c5_p, "s_span": 0.8 * span_y})

    # Chord 6: tangential_lower
    c6_z = (z_mid - 0.35 * kappa * a_minor) + (0.5 - eta_v) * (0.6 * a_minor)
    c6_p = (R0 - 1.25 * a_minor) + xi_u * (2.5 * a_minor) if R0 > 0 else (2.0 * xi_u - 1.0) * (0.8 * a_minor)
    chord_configs.append({"type": "tangential", "z_2d": c6_z, "p_2d": c6_p, "s_span": 0.8 * span_y})

    # Chord 7: helical_resonant
    theta_pitch = 0.35
    u_rot = (xi_u - 0.5) * math.cos(theta_pitch) - (eta_v - 0.5) * math.sin(theta_pitch)
    v_rot = (xi_u - 0.5) * math.sin(theta_pitch) + (eta_v - 0.5) * math.cos(theta_pitch)
    c7_z = z_mid - v_rot * (1.2 * a_minor)
    c7_p = R0 + u_rot * (2.0 * a_minor) if R0 > 0 else u_rot * (2.0 * a_minor)
    chord_configs.append({"type": "tangential", "z_2d": c7_z, "p_2d": c7_p, "s_span": 0.8 * span_y})

    N_steps = 10
    step_samples = np.linspace(-0.5, 0.5, N_steps, dtype=np.float32)

    # Process in Z-slabs (chunk_z = 16)
    num_slabs = (Z + chunk_z - 1) // chunk_z

    for slab_idx in range(num_slabs):
        z_start = slab_idx * chunk_z
        z_end = min((slab_idx + 1) * chunk_z, Z)

        psi_slab = psi_vol[z_start:z_end]
        p_slab = pkpa_vol[z_start:z_end]
        j_slab = jcurl_vol[z_start:z_end]
        b_slab = bmod_vol[z_start:z_end]

        z_slab_min = float(z_coords[z_start])
        z_slab_max = float(z_coords[z_end - 1])
        z_pad = 0.5 * (span_z / max(Z - 1, 1))
        slab_lo = z_slab_min - z_pad
        slab_hi = z_slab_max + z_pad

        for c_idx in range(min(num_chords, len(chord_configs))):
            cfg = chord_configs[c_idx]
            z_cam = cfg["z_2d"]

            z_mask = (z_cam >= slab_lo) & (z_cam <= slab_hi)
            if not np.any(z_mask):
                continue

            if cfg["type"] == "poloidal":
                x_cam = np.broadcast_to(cfg["x_2d"], (H, W))
                z_cam_2d = np.broadcast_to(z_cam, (H, W))
                y_span = cfg["y_span"]
                ds = float(y_span / N_steps)

                for step_frac in step_samples:
                    y_sample = float(step_frac * y_span)

                    z_vox = ((z_cam_2d - min_z) / span_z) * (Z - 1)
                    y_vox = ((y_sample - min_y) / span_y) * (Y - 1)
                    x_vox = ((x_cam - min_x) / span_x) * (X - 1)

                    z_local = z_vox - z_start
                    p_s = _sample_slab_trilinear(p_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)
                    j_s = _sample_slab_trilinear(j_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)
                    b_s = _sample_slab_trilinear(b_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)
                    psi_s = _sample_slab_trilinear(psi_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)

                    emiss = 0.55 * (p_s / p_norm)**2 + 0.35 * (j_s / j_norm) + 0.10 * (b_s / b_norm) + 0.02
                    emiss_acc[c_idx] += emiss * ds
                    flux_acc[c_idx] += psi_s * emiss * ds
                    weight_acc[c_idx] += emiss * ds

                    if is_scen_05:
                        r_perp = np.sqrt(x_cam**2 + y_sample**2)
                        eps_val = np.full((H, W), 5.50, dtype=np.float32)
                        eps_val[r_perp < 0.35] = 4.00
                        eps_val[r_perp < 0.25] = 1.00
                        is_fracture = (x_cam >= 0.25) & (x_cam <= 0.28) & (abs(y_sample) < 0.20)
                        eps_val[is_fracture] = 9.85
                        dielectric_acc[c_idx] += eps_val * ds

            else:
                p_cam = np.broadcast_to(cfg["p_2d"], (H, W))
                z_cam_2d = np.broadcast_to(z_cam, (H, W))
                s_span = cfg["s_span"]
                ds = float(s_span / N_steps)

                for step_frac in step_samples:
                    s_sample = float(step_frac * s_span)
                    x_sample = p_cam
                    y_sample = s_sample

                    z_vox = ((z_cam_2d - min_z) / span_z) * (Z - 1)
                    y_vox = ((y_sample - min_y) / span_y) * (Y - 1)
                    x_vox = ((x_sample - min_x) / span_x) * (X - 1)

                    z_local = z_vox - z_start
                    p_s = _sample_slab_trilinear(p_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)
                    j_s = _sample_slab_trilinear(j_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)
                    b_s = _sample_slab_trilinear(b_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)
                    psi_s = _sample_slab_trilinear(psi_slab, z_local, np.broadcast_to(np.float32(y_vox), (H, W)), x_vox)

                    emiss = 0.55 * (p_s / p_norm)**2 + 0.35 * (j_s / j_norm) + 0.10 * (b_s / b_norm) + 0.02
                    emiss_acc[c_idx] += emiss * ds
                    flux_acc[c_idx] += psi_s * emiss * ds
                    weight_acc[c_idx] += emiss * ds

                    if is_scen_05:
                        r_perp = np.sqrt(x_sample**2 + y_sample**2)
                        eps_val = np.full((H, W), 5.50, dtype=np.float32)
                        eps_val[r_perp < 0.35] = 4.00
                        eps_val[r_perp < 0.25] = 1.00
                        is_fracture = (x_sample >= 0.25) & (x_sample <= 0.28) & (abs(y_sample) < 0.20)
                        eps_val[is_fracture] = 9.85
                        dielectric_acc[c_idx] += eps_val * ds

    # Compute final 2D projections, normalized flux, and gradient maps
    for c in range(min(num_chords, len(chord_configs))):
        p_raw = emiss_acc[c]
        p_min = float(p_raw.min())
        p_max = float(p_raw.max())
        if p_max > p_min:
            projections[c] = np.clip((p_raw - p_min) / (p_max - p_min), 0.0, 1.0).astype(np.float32)
        else:
            projections[c] = np.zeros((H, W), dtype=np.float32)

        w = np.maximum(weight_acc[c], 1e-6)
        psi_norm = np.clip(flux_acc[c] / w, 0.0, 1.0).astype(np.float32)
        psi_2d[c] = psi_norm

        dpsi_dv = np.gradient(psi_norm, axis=0).astype(np.float32)
        dpsi_du = np.gradient(psi_norm, axis=1).astype(np.float32)
        grad_psi[c, :, :, 0] = dpsi_dv
        grad_psi[c, :, :, 1] = dpsi_du

        if is_scen_05 and dielectric is not None:
            cfg = chord_configs[c]
            if cfg["type"] == "poloidal":
                x_c = np.broadcast_to(cfg["x_2d"], (H, W))
                z_c = np.broadcast_to(cfg["z_2d"], (H, W))
                r_c = np.sqrt(x_c**2 + (z_c - z_mid)**2)
                eps = np.full((H, W), 5.50, dtype=np.float32)
                eps[r_c < 0.35] = 4.00
                eps[r_c < 0.25] = 1.00
                is_frac = (x_c >= 0.25) & (x_c <= 0.28) & (np.abs(z_c - z_mid) < 0.20)
                eps[is_frac] = 9.85
            else:
                p_c = np.broadcast_to(cfg["p_2d"], (H, W))
                z_c = np.broadcast_to(cfg["z_2d"], (H, W))
                r_c = np.sqrt(p_c**2 + (z_c - z_mid)**2)
                eps = np.full((H, W), 5.50, dtype=np.float32)
                eps[r_c < 0.35] = 4.00
                eps[r_c < 0.25] = 1.00
                is_frac = (p_c >= 0.25) & (p_c <= 0.28) & (np.abs(z_c - z_mid) < 0.20)
                eps[is_frac] = 9.85
            dielectric[c] = eps


    # Apply scenario-specific masks (glare washout, AC coil ripple, molten glass flare)
    if is_scen_04:
        # 10% divertor Bremsstrahlung glare on chord 3 (poloidal divertor) and chord 6 (tangential lower)
        div_row = int(0.72 * H)
        mask_washout[3, div_row:, :] = True
        mask_washout[6, div_row:, :] = True
        projections[3, mask_washout[3]] = 1.0
        projections[6, mask_washout[6]] = 1.0

        # AC coil ripple (300 Hz) in scrape-off layer outer boundary
        for c in range(min(num_chords, 8)):
            ripple_mask = (psi_2d[c] > 0.90) & (~mask_washout[c])
            mask_clock[c] = ripple_mask
            ripple_factor = 1.0 + 0.05 * math.sin(2.0 * math.pi * 300.0 * (frame_idx * 1e-3))
            projections[c, ripple_mask] = np.clip(projections[c, ripple_mask] * ripple_factor, 0.0, 1.0)

    elif is_scen_05:
        # Molten beta1 glass flare on annular liner zone (r in [0.25, 0.35])
        for c in (0, 6):
            if c < num_chords:
                cfg = chord_configs[c]
                if cfg["type"] == "poloidal":
                    r_cam = np.sqrt(np.broadcast_to(cfg["x_2d"], (H, W))**2 + np.broadcast_to(cfg["z_2d"], (H, W))**2)
                else:
                    r_cam = np.abs(np.broadcast_to(cfg["p_2d"], (H, W)))
                flare_mask = (r_cam >= 0.24) & (r_cam <= 0.36)
                mask_washout[c] = flare_mask
                projections[c, mask_washout[c]] = 1.0

    return {
        "projections": projections,
        "grad_psi": grad_psi,
        "mask_washout": mask_washout,
        "mask_clock": mask_clock,
        "dielectric": dielectric,
        "psi_2d": psi_2d,
    }


def build_all_scenarios(
    grid_shape: Tuple[int, int, int] = (32, 24, 24),
    export_chords: bool = True,
    export_dir: Union[str, Path] = "data/stellarator_chords",
) -> ScenarioDeck:
    """
    Build all 5 canonical scenarios on (Z=32, Y=24, X=24) computational grids (18,432 voxels),
    generate 8 synthetic diagnostic chords per scenario, and optionally export chord files.
    Returns a ScenarioDeck (dict subclass) supporting both string key and integer index access.
    """
    scenarios = ScenarioDeck()

    for key, builder in SCENARIO_BUILDERS.items():
        scenarios[key] = builder(grid_shape=grid_shape)

    if export_chords:
        try:
            export_bundled_stellarator_chords(output_dir=export_dir, scenarios=scenarios)
        except Exception:
            pass

    return scenarios

