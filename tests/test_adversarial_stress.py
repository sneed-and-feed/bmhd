"""
tests/test_adversarial_stress.py: Empirical Adversarial Stress Test Suite for BMHD.

Adversarial Stress Verification of Physical Invariants & Core Engine:
1. test_stress_divergence_and_force_balance:
   Numerical div(B) and force balance (J x B) - grad(p) across 150+ random points
   and multiple grid resolutions for Family 1 (Solov'ev) and Family 2 (sheared elliptic).
2. test_stress_field_line_radial_drift_k32:
   Multistep unrolled spherical cascade across K in {1, 4, 8, 16, 32} unroll steps
   asserting exact 0.000e+00 radial flux escape.
3. test_stress_homotopy_zero_reduction_100plus:
   Exact 0.00e+0 reduction to stellarator symmetry at tau_0 = 0 and tau_1 = 0
   across 200+ random points.
4. test_stress_so6_rotor_500_rotations:
   Lifted SO(6) Kelvin-Mandel stiffness rotor invariance asserting trace drift
   and Frobenius norm drift <= 1.14e-13 across 500 random 3D rotations.
5. test_stress_heap_memory_ceiling:
   tracemalloc peak heap ceiling verification strictly < 35 MB during full
   scenario evaluation and 64 MB out-of-core memmap streaming.
"""

from __future__ import annotations

import math
import os
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import torch

from bmhd.engine import (
    BMHDModel,
    SkewSymmetricFluxTriad,
    SlowFastPlasmaCascade,
    build_kelvin_mandel_so6_rotor,
    rotate_stiffness_so6,
)
from bmhd.mhd import (
    IssanNonSymmetricEquilibriumEngine,
    CerfonFreidbergEquilibrium,
)
from bmhd.scenarios import build_all_scenarios
from bmhd.io import stream_field_volume
from bmhd.benchmark import run_mode_b_evaluation


# =============================================================================
# 1. Numerical div(B) and Force Balance (J x B) - grad(p)
# =============================================================================

def test_stress_divergence_and_force_balance():
    """
    Adversarially evaluate div(B), B . grad(psi), and (J x B) - grad(p) across
    150+ randomly sampled interior points and multiple grid resolutions.
    """
    np.random.seed(1337)
    h = 1e-5

    # Family 1 across tau_0 in {-0.7, -0.38, 0.0, 0.38, 0.7}
    tau0_candidates = [-0.7, -0.38, 0.0, 0.38, 0.7]
    for tau0 in tau0_candidates:
        engine1 = IssanNonSymmetricEquilibriumEngine(family=1, tau0=tau0)
        expected_ratio = 2.0 / (engine1.Omega**2)

        pts_tested = 0
        while pts_tested < 25:
            x = np.random.uniform(0.6, 1.2)
            y = np.random.uniform(-0.3, 0.3)
            z = np.random.uniform(-0.15, 0.15)

            s = x**2 / engine1.a**2 + y**2 / engine1.b**2
            f_sq = 1.0 - (1.0 - s)**2 - 4.0 * z**2 - 4.0 * tau0 * s * z
            if f_sq < 0.05:
                continue

            def get_f1(px, py, pz):
                r = engine1.compute_cartesian(np.array([px]), np.array([py]), np.array([pz]))
                return np.array([r["Bx"][0], r["By"][0], r["Bz"][0]]), r["psi_raw"][0]

            B0, psi0 = get_f1(x, y, z)

            # Divergence
            dBx_dx = (get_f1(x + h, y, z)[0][0] - get_f1(x - h, y, z)[0][0]) / (2 * h)
            dBy_dy = (get_f1(x, y + h, z)[0][1] - get_f1(x, y - h, z)[0][1]) / (2 * h)
            dBz_dz = (get_f1(x, y, z + h)[0][2] - get_f1(x, y, z - h)[0][2]) / (2 * h)
            divB = dBx_dx + dBy_dy + dBz_dz
            assert abs(divB) < 1e-7, f"div(B) {divB} exceeded threshold at ({x},{y},{z})"

            # Tangent orthogonality B . grad(psi) = 0
            dpsi_dx = (get_f1(x + h, y, z)[1] - get_f1(x - h, y, z)[1]) / (2 * h)
            dpsi_dy = (get_f1(x, y + h, z)[1] - get_f1(x, y - h, z)[1]) / (2 * h)
            dpsi_dz = (get_f1(x, y, z + h)[1] - get_f1(x, y, z - h)[1]) / (2 * h)
            grad_psi = np.array([dpsi_dx, dpsi_dy, dpsi_dz])
            b_dot_grad = float(np.dot(B0, grad_psi))
            assert abs(b_dot_grad) < 1e-7, f"B . grad(psi) {b_dot_grad} exceeded threshold"

            # Force balance (curl B) x B = (2/Omega^2) grad(psi)
            curl_Bx = (get_f1(x, y, z + h)[0][1] - get_f1(x, y, z - h)[0][1]) / (2 * h) - (
                get_f1(x, y + h, z)[0][2] - get_f1(x, y - h, z)[0][2]
            ) / (2 * h)
            curl_By = (get_f1(x + h, y, z)[0][2] - get_f1(x - h, y, z)[0][2]) / (2 * h) - (
                get_f1(x, y, z + h)[0][0] - get_f1(x, y, z - h)[0][0]
            ) / (2 * h)
            curl_Bz = (get_f1(x, y + h, z)[0][0] - get_f1(x, y - h, z)[0][0]) / (2 * h) - (
                get_f1(x + h, y, z)[0][1] - get_f1(x - h, y, z)[0][1]
            ) / (2 * h)
            curl_B = np.array([curl_Bx, curl_By, curl_Bz])
            curl_B_cross_B = np.cross(curl_B, B0)
            expected_force = expected_ratio * grad_psi
            force_err = np.linalg.norm(curl_B_cross_B - expected_force)
            force_norm = max(np.linalg.norm(expected_force), 1e-6)
            assert force_err / force_norm < 1e-5, f"Force balance rel err {force_err / force_norm} exceeded limit"

            pts_tested += 1

    # Family 2 across tau_1 in {-0.5, 0.0, 0.42, 0.8}
    for tau1 in [-0.5, 0.0, 0.42, 0.8]:
        engine2 = IssanNonSymmetricEquilibriumEngine(family=2, tau1=tau1)
        expected_ratio2 = 1.0 / (engine2.lam**2)

        pts_tested = 0
        while pts_tested < 25:
            x = np.random.uniform(0.7, 1.3)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(-0.2, 0.2)

            def get_f2(px, py, pz):
                r = engine2.compute_cartesian(np.array([px]), np.array([py]), np.array([pz]))
                return np.array([r["Bx"][0], r["By"][0], r["Bz"][0]]), r["psi_raw"][0]

            B0, psi0 = get_f2(x, y, z)
            dBx_dx = (get_f2(x + h, y, z)[0][0] - get_f2(x - h, y, z)[0][0]) / (2 * h)
            dBy_dy = (get_f2(x, y + h, z)[0][1] - get_f2(x, y - h, z)[0][1]) / (2 * h)
            dBz_dz = (get_f2(x, y, z + h)[0][2] - get_f2(x, y, z - h)[0][2]) / (2 * h)
            divB2 = dBx_dx + dBy_dy + dBz_dz
            assert abs(divB2) < 1e-7

            dpsi_dx = (get_f2(x + h, y, z)[1] - get_f2(x - h, y, z)[1]) / (2 * h)
            dpsi_dy = (get_f2(x, y + h, z)[1] - get_f2(x, y - h, z)[1]) / (2 * h)
            dpsi_dz = (get_f2(x, y, z + h)[1] - get_f2(x, y, z - h)[1]) / (2 * h)
            grad_psi2 = np.array([dpsi_dx, dpsi_dy, dpsi_dz])
            assert abs(float(np.dot(B0, grad_psi2))) < 1e-7

            curl_Bx = (get_f2(x, y, z + h)[0][1] - get_f2(x, y, z - h)[0][1]) / (2 * h) - (
                get_f2(x, y + h, z)[0][2] - get_f2(x, y - h, z)[0][2]
            ) / (2 * h)
            curl_By = (get_f2(x + h, y, z)[0][2] - get_f2(x - h, y, z)[0][2]) / (2 * h) - (
                get_f2(x, y, z + h)[0][0] - get_f2(x, y, z - h)[0][0]
            ) / (2 * h)
            curl_Bz = (get_f2(x, y + h, z)[0][0] - get_f2(x, y - h, z)[0][0]) / (2 * h) - (
                get_f2(x + h, y, z)[0][1] - get_f2(x - h, y, z)[0][1]
            ) / (2 * h)
            curl_B2 = np.array([curl_Bx, curl_By, curl_Bz])
            curl_B_cross_B2 = np.cross(curl_B2, B0)
            expected_force2 = expected_ratio2 * grad_psi2
            force_err2 = np.linalg.norm(curl_B_cross_B2 - expected_force2)
            force_norm2 = max(np.linalg.norm(expected_force2), 1e-6)
            assert force_err2 / force_norm2 < 1e-5

            pts_tested += 1

    # Multi-resolution volume test
    for grid in [(16, 12, 12), (32, 24, 24), (64, 48, 48)]:
        Z, Y, X = grid
        r_line = np.linspace(0.8, 1.2, X)
        phi_line = np.linspace(0, 2 * math.pi, Y, endpoint=False)
        z_line = np.linspace(-0.15, 0.15, Z)
        res_grid = engine1.compute_fields(r_line, phi_line, z_line)
        assert res_grid["B_cart"].shape == (3, Z, Y, X)
        assert np.all(np.isfinite(res_grid["B_cart"]))


# =============================================================================
# 2. Multistep Cascade Field-Line Radial Drift for K in {1, 4, 8, 16, 32}
# =============================================================================

def test_stress_field_line_radial_drift_k32():
    """
    Stress-test unrolled Lie-algebraic spherical cascade over K in {1, 4, 8, 16, 32} steps.
    Asserts exact 0.000e+00 radial drift and <= 1.0e-8 tangent orthogonality.
    """
    torch.manual_seed(999)
    model = BMHDModel()

    dim = 8
    num_samples = 256
    u_cas = torch.randn(num_samples, dim, dtype=torch.float32)
    v_cas = torch.randn(num_samples, dim, dtype=torch.float32)
    W_rec = 0.35 * torch.randn(dim, dim, dim, dtype=torch.float32)

    for K in [1, 4, 8, 16, 32]:
        v_out, diag = model.cascade_op(u_cas, v_cas, W_rec, unroll_steps=K)
        assert diag["max_spherical_radial_drift"] == 0.000e+00, f"K={K} radial drift non-zero"
        assert diag["max_tangent_orthogonality_error"] <= 1.0e-8, f"K={K} ortho err exceeded 1e-8"

        # Independent double-check on squared norm equality
        in_norm_sq = (v_cas * v_cas).sum(dim=-1)
        out_norm_sq = (v_out * v_out).sum(dim=-1)
        emp_drift = float(torch.max(torch.abs(out_norm_sq - in_norm_sq)).item())
        assert emp_drift == 0.000e+00, f"Empirical drift non-zero for K={K}: {emp_drift}"


# =============================================================================
# 3. Symmetry-Breaking Homotopy Exact Reduction across 200+ Points
# =============================================================================

def test_stress_homotopy_zero_reduction_100plus():
    """
    Assert exact 0.00e+0 reduction to stellarator-symmetric baseline at tau_0=0 and tau_1=0
    across 200+ randomly sampled coordinates (x, y, z).
    """
    np.random.seed(2026)
    engine_f1_zero = IssanNonSymmetricEquilibriumEngine(family=1, tau0=0.0)
    engine_f2_zero = IssanNonSymmetricEquilibriumEngine(family=2, tau1=0.0)

    # 100 points for Family 1
    for _ in range(100):
        x = float(np.random.uniform(0.65, 1.25))
        y = float(np.random.uniform(-0.35, 0.35))
        z = float(np.random.uniform(-0.25, 0.25))
        sym = engine_f1_zero.check_stellarator_symmetry(x=x, y=y, z=z)
        assert sym["max_residual"] == 0.0, f"Family 1 residual at ({x},{y},{z}): {sym}"

    # 100 points for Family 2
    for _ in range(100):
        x = float(np.random.uniform(0.70, 1.30))
        y = float(np.random.uniform(-0.40, 0.40))
        z = float(np.random.uniform(-0.30, 0.30))
        sym2 = engine_f2_zero.check_stellarator_symmetry(x=x, y=y, z=z)
        assert sym2["max_residual"] == 0.0, f"Family 2 residual at ({x},{y},{z}): {sym2}"


# =============================================================================
# 4. SO(6) Kelvin-Mandel Rotor Invariance across 500 Random Rotations
# =============================================================================

def test_stress_so6_rotor_500_rotations():
    """
    Assert trace drift and Frobenius norm drift <= 1.14e-13 across 500 random 3D rotations.
    """
    np.random.seed(777)
    I6 = np.eye(6, dtype=np.float64)

    for i in range(500):
        # Generate random SO(3) matrix via QR decomposition of Gaussian 3x3
        G = np.random.randn(3, 3)
        Q, R = np.linalg.qr(G)
        d = np.diagonal(R)
        ph = d / np.abs(d)
        Q = Q @ np.diag(ph)
        if np.linalg.det(Q) < 0:
            Q[:, 0] = -Q[:, 0]
        R_so3 = Q

        # Lift to SO(6)
        M6 = build_kelvin_mandel_so6_rotor(R_so3)

        # SO(6) orthogonality
        ortho_err = float(np.max(np.abs(M6.T @ M6 - I6)))
        assert ortho_err <= 1.14e-13, f"Rotation {i} SO(6) orthogonality error: {ortho_err}"

        # Anisotropic stiffness tensor
        A = np.random.randn(6, 6)
        C0 = A @ A.T + 2.0 * I6

        # Rotate C_rot = M C0 M^T
        C_rot = rotate_stiffness_so6(M6, C0)

        tr_drift = abs(float(np.trace(C_rot) - np.trace(C0)))
        assert tr_drift <= 1.14e-13, f"Rotation {i} trace drift: {tr_drift}"

        fr_drift = abs(float(np.linalg.norm(C_rot, "fro") - np.linalg.norm(C0, "fro")))
        assert fr_drift <= 1.14e-13, f"Rotation {i} Frobenius drift: {fr_drift}"


# =============================================================================
# 5. Heap Memory Ceiling (< 35 MB Hard Limit) under tracemalloc
# =============================================================================

def test_stress_heap_memory_ceiling(tmp_path: Path):
    """
    Measure peak heap memory under tracemalloc during full model evaluation,
    Mode B reconstruction, and 64 MB out-of-core memmap streaming (< 35 MB limit).
    """
    tracemalloc.start()
    tracemalloc.reset_peak()

    # 1. Model formal theorem verification
    model = BMHDModel()
    audit = model.verify_formal_theorems()
    assert audit.all_theorems_verified is True
    _, peak1 = tracemalloc.get_traced_memory()
    assert peak1 / (1024 * 1024) < 35.0

    # 2. Build 5 scenarios
    scenarios = build_all_scenarios()
    assert len(scenarios) == 5
    _, peak2 = tracemalloc.get_traced_memory()
    assert peak2 / (1024 * 1024) < 35.0

    # 3. Mode B 7-baseline reconstruction
    sc01 = scenarios["01_issan_nonsym_solovev_tau0"]
    res_b = run_mode_b_evaluation(sc01, snr_db=20.0, chunk_z=16)
    assert res_b["performance"]["heap_ceiling_passed"] is True
    _, peak3 = tracemalloc.get_traced_memory()
    assert peak3 / (1024 * 1024) < 35.0

    # 4. Out-of-core np.memmap streaming on a 64 MB volume (100 x 400 x 400 float32)
    vol_file = tmp_path / "synthetic_64mb_volume.npy"
    fp = np.lib.format.open_memmap(vol_file, mode="w+", dtype="float32", shape=(100, 400, 400))
    fp[:] = 1.0
    fp.flush()
    del fp

    tracemalloc.reset_peak()
    streamer = stream_field_volume(vol_file, chunk_ky=16, target_grid=(32, 24, 24))
    slabs = list(streamer)
    assert len(slabs) > 0

    _, peak4 = tracemalloc.get_traced_memory()
    peak4_mb = peak4 / (1024 * 1024)
    assert peak4_mb < 35.0, f"Memmap streaming peak heap {peak4_mb:.2f} MB exceeded 35 MB"

    tracemalloc.stop()
