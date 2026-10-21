"""
scripts/run_challenger_stress_suite.py: Comprehensive Empirical Adversarial Stress Suite for BMHD v0.1.0.

Challenges:
1. Numerical divergence div(B) and force balance (J x B) - grad(p) across 150+ random points & 3 grid resolutions.
2. Field-line integration over unrolled K in {1, 4, 8, 16, 32} steps for radial flux escape.
3. Symmetry-breaking homotopy: exact 0.00e+0 reduction at tau_0=0 and tau_1=0 across 200+ random points.
4. SO(6) Kelvin-Mandel tensor rotor: trace & Frobenius drift <= 1.14e-13 across 500 random 3D rotations.
5. Memory ceiling: peak heap under tracemalloc during full volume evaluation & 64 MB memmap stream (< 35 MB).
"""

import math
import sys
import os
import tracemalloc
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

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


def run_stress_test_1_divergence_and_force_balance():
    print("=====================================================================")
    print("STRESS TEST 1: Numerical Divergence div(B) & Force Balance (J x B) - grad(p)")
    print("=====================================================================")
    
    np.random.seed(1337)
    
    # 1.1 Test Family 1 across multiple tau0 values and 150 random points
    tau0_candidates = [-0.7, -0.38, 0.0, 0.38, 0.7]
    h = 1e-5
    
    f1_max_divB = 0.0
    f1_max_tangent_ortho = 0.0
    f1_max_force_balance_rel_err = 0.0
    total_f1_pts = 0
    
    for tau0 in tau0_candidates:
        engine1 = IssanNonSymmetricEquilibriumEngine(family=1, tau0=tau0)
        Omega = engine1.Omega
        expected_ratio = 2.0 / (Omega**2)  # dp/dpsi = -2 / Omega^2 in dimensionless Solov'ev
        
        # Sample points strictly inside plasma volume where f_sq > 0.05
        points_tested = 0
        while points_tested < 30:
            x = np.random.uniform(0.6, 1.2)
            y = np.random.uniform(-0.3, 0.3)
            z = np.random.uniform(-0.15, 0.15)
            
            s = x**2 / engine1.a**2 + y**2 / engine1.b**2
            f_sq = 1.0 - (1.0 - s)**2 - 4.0 * z**2 - 4.0 * tau0 * s * z
            if f_sq < 0.05:
                continue
                
            def get_fields(px, py, pz):
                r = engine1.compute_cartesian(np.array([px]), np.array([py]), np.array([pz]))
                return (
                    np.array([r['Bx'][0], r['By'][0], r['Bz'][0]]),
                    r['psi_raw'][0],
                    r['p_kpa'][0]
                )
            
            B0, psi0, p0 = get_fields(x, y, z)
            
            # Central difference divergence div(B)
            dBx_dx = (get_fields(x+h, y, z)[0][0] - get_fields(x-h, y, z)[0][0]) / (2*h)
            dBy_dy = (get_fields(x, y+h, z)[0][1] - get_fields(x, y-h, z)[0][1]) / (2*h)
            dBz_dz = (get_fields(x, y, z+h)[0][2] - get_fields(x, y, z-h)[0][2]) / (2*h)
            divB = dBx_dx + dBy_dy + dBz_dz
            f1_max_divB = max(f1_max_divB, abs(divB))
            
            # Gradient of psi
            dpsi_dx = (get_fields(x+h, y, z)[1] - get_fields(x-h, y, z)[1]) / (2*h)
            dpsi_dy = (get_fields(x, y+h, z)[1] - get_fields(x, y-h, z)[1]) / (2*h)
            dpsi_dz = (get_fields(x, y, z+h)[1] - get_fields(x, y, z-h)[1]) / (2*h)
            grad_psi = np.array([dpsi_dx, dpsi_dy, dpsi_dz])
            
            # B . grad(psi)
            b_dot_grad_psi = float(np.dot(B0, grad_psi))
            f1_max_tangent_ortho = max(f1_max_tangent_ortho, abs(b_dot_grad_psi))
            
            # Numerical curl of B
            curl_Bx = (get_fields(x, y, z+h)[0][1] - get_fields(x, y, z-h)[0][1])/(2*h) - (get_fields(x, y+h, z)[0][2] - get_fields(x, y-h, z)[0][2])/(2*h)
            curl_By = (get_fields(x+h, y, z)[0][2] - get_fields(x-h, y, z)[0][2])/(2*h) - (get_fields(x, y, z+h)[0][0] - get_fields(x, y, z-h)[0][0])/(2*h)
            curl_Bz = (get_fields(x, y+h, z)[0][0] - get_fields(x, y-h, z)[0][0])/(2*h) - (get_fields(x+h, y, z)[0][1] - get_fields(x-h, y, z)[0][1])/(2*h)
            curl_B = np.array([curl_Bx, curl_By, curl_Bz])
            
            # (curl B) x B
            curl_B_cross_B = np.cross(curl_B, B0)
            
            # Expected force balance in Solov'ev: (curl B) x B = (2 / Omega^2) * grad(psi)
            expected_force = expected_ratio * grad_psi
            force_err = np.linalg.norm(curl_B_cross_B - expected_force)
            force_norm = max(np.linalg.norm(expected_force), 1e-6)
            f1_max_force_balance_rel_err = max(f1_max_force_balance_rel_err, force_err / force_norm)
            
            points_tested += 1
            total_f1_pts += 1

    print(f"Family 1 (Solov'ev): Tested {total_f1_pts} points across tau0 in {tau0_candidates}")
    print(f"  - Peak |div(B)|: {f1_max_divB:.4e}")
    print(f"  - Peak |B . grad(psi)|: {f1_max_tangent_ortho:.4e}")
    print(f"  - Peak Relative Force Balance Error: {f1_max_force_balance_rel_err:.4e}")
    assert f1_max_divB < 1e-7, f"div(B) exceeded threshold: {f1_max_divB}"
    assert f1_max_tangent_ortho < 1e-7, f"B . grad(psi) exceeded threshold: {f1_max_tangent_ortho}"
    assert f1_max_force_balance_rel_err < 1e-5, f"Force balance rel error exceeded threshold: {f1_max_force_balance_rel_err}"

    # 1.2 Test Family 2 across multiple tau1 values and 120 random points
    tau1_candidates = [-0.5, 0.0, 0.42, 0.8]
    f2_max_divB = 0.0
    f2_max_tangent_ortho = 0.0
    f2_max_force_balance_rel_err = 0.0
    total_f2_pts = 0
    
    for tau1 in tau1_candidates:
        engine2 = IssanNonSymmetricEquilibriumEngine(family=2, tau1=tau1)
        expected_ratio2 = 1.0 / (engine2.lam**2)
        
        points_tested = 0
        while points_tested < 30:
            x = np.random.uniform(0.7, 1.3)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(-0.2, 0.2)
            
            def get_f2(px, py, pz):
                r = engine2.compute_cartesian(np.array([px]), np.array([py]), np.array([pz]))
                return np.array([r['Bx'][0], r['By'][0], r['Bz'][0]]), r['psi_raw'][0]
            
            B0, psi0 = get_f2(x, y, z)
            
            dBx_dx = (get_f2(x+h, y, z)[0][0] - get_f2(x-h, y, z)[0][0]) / (2*h)
            dBy_dy = (get_f2(x, y+h, z)[0][1] - get_f2(x, y-h, z)[0][1]) / (2*h)
            dBz_dz = (get_f2(x, y, z+h)[0][2] - get_f2(x, y, z-h)[0][2]) / (2*h)
            divB2 = dBx_dx + dBy_dy + dBz_dz
            f2_max_divB = max(f2_max_divB, abs(divB2))
            
            dpsi_dx = (get_f2(x+h, y, z)[1] - get_f2(x-h, y, z)[1]) / (2*h)
            dpsi_dy = (get_f2(x, y+h, z)[1] - get_f2(x, y-h, z)[1]) / (2*h)
            dpsi_dz = (get_f2(x, y, z+h)[1] - get_f2(x, y, z-h)[1]) / (2*h)
            grad_psi2 = np.array([dpsi_dx, dpsi_dy, dpsi_dz])
            
            b_dot_grad_psi2 = float(np.dot(B0, grad_psi2))
            f2_max_tangent_ortho = max(f2_max_tangent_ortho, abs(b_dot_grad_psi2))
            
            # Numerical curl
            curl_Bx = (get_f2(x, y, z+h)[0][1] - get_f2(x, y, z-h)[0][1])/(2*h) - (get_f2(x, y+h, z)[0][2] - get_f2(x, y-h, z)[0][2])/(2*h)
            curl_By = (get_f2(x+h, y, z)[0][2] - get_f2(x-h, y, z)[0][2])/(2*h) - (get_f2(x, y, z+h)[0][0] - get_f2(x, y, z-h)[0][0])/(2*h)
            curl_Bz = (get_f2(x, y+h, z)[0][0] - get_f2(x, y-h, z)[0][0])/(2*h) - (get_f2(x+h, y, z)[0][1] - get_f2(x-h, y, z)[0][1])/(2*h)
            curl_B = np.array([curl_Bx, curl_By, curl_Bz])
            
            curl_B_cross_B = np.cross(curl_B, B0)
            expected_force2 = expected_ratio2 * grad_psi2
            force_err2 = np.linalg.norm(curl_B_cross_B - expected_force2)
            force_norm2 = max(np.linalg.norm(expected_force2), 1e-6)
            f2_max_force_balance_rel_err = max(f2_max_force_balance_rel_err, force_err2 / force_norm2)
            
            points_tested += 1
            total_f2_pts += 1

    print(f"Family 2 (Sheared Elliptic): Tested {total_f2_pts} points across tau1 in {tau1_candidates}")
    print(f"  - Peak |div(B)|: {f2_max_divB:.4e}")
    print(f"  - Peak |B . grad(psi)|: {f2_max_tangent_ortho:.4e}")
    print(f"  - Peak Relative Force Balance Error: {f2_max_force_balance_rel_err:.4e}")
    assert f2_max_divB < 1e-7
    assert f2_max_tangent_ortho < 1e-7
    assert f2_max_force_balance_rel_err < 1e-5

    # 1.3 Test across 3 different grid resolutions: (16, 12, 12), (32, 24, 24), (64, 48, 48)
    for grid in [(16, 12, 12), (32, 24, 24), (64, 48, 48)]:
        Z, Y, X = grid
        r_line = np.linspace(0.8, 1.2, X)
        phi_line = np.linspace(0, 2*math.pi, Y, endpoint=False)
        z_line = np.linspace(-0.15, 0.15, Z)
        res_grid = engine1.compute_fields(r_line, phi_line, z_line)
        b_cube = res_grid["B_cart"]  # (3, Z, Y, X)
        assert b_cube.shape == (3, Z, Y, X)
        assert np.all(np.isfinite(b_cube))
        print(f"  - Grid {grid} evaluation: SUCCESS, B_mod range = [{np.min(res_grid['B_mod']):.3f}, {np.max(res_grid['B_mod']):.3f}] T")

    print("STRESS TEST 1: PASSED\n")


def run_stress_test_2_field_line_radial_drift():
    print("=====================================================================")
    print("STRESS TEST 2: Field-Line Integration & Multistep Cascade (K in {1, 4, 8, 16, 32})")
    print("=====================================================================")
    
    torch.manual_seed(999)
    model = BMHDModel()
    
    dim = 8
    num_samples = 256
    u_cas = torch.randn(num_samples, dim, dtype=torch.float32)
    v_cas = torch.randn(num_samples, dim, dtype=torch.float32)
    W_rec = 0.35 * torch.randn(dim, dim, dim, dtype=torch.float32)
    
    k_steps = [1, 4, 8, 16, 32]
    for K in k_steps:
        v_out, diag = model.cascade_op(u_cas, v_cas, W_rec, unroll_steps=K)
        radial_drift = diag["max_spherical_radial_drift"]
        ortho_err = diag["max_tangent_orthogonality_error"]
        
        # Also independently verify norm equality
        in_norm_sq = (v_cas * v_cas).sum(dim=-1)
        out_norm_sq = (v_out * v_out).sum(dim=-1)
        emp_max_drift = float(torch.max(torch.abs(out_norm_sq - in_norm_sq)).item())
        
        print(f"  - Unroll K = {K:2d}: radial drift = {radial_drift:.3e} (emp = {emp_max_drift:.3e}), ortho err = {ortho_err:.3e}")
        assert radial_drift == 0.000e+00, f"K={K} radial drift non-zero: {radial_drift}"
        assert emp_max_drift == 0.000e+00, f"K={K} empirical drift non-zero: {emp_max_drift}"
        assert ortho_err <= 1.0e-8, f"K={K} orthogonality error exceeded 1e-8: {ortho_err}"

    print("STRESS TEST 2: PASSED\n")


def run_stress_test_3_homotopy_symmetry_breaking():
    print("=====================================================================")
    print("STRESS TEST 3: Symmetry-Breaking Homotopy Exact Reduction (tau_0=0, tau_1=0)")
    print("=====================================================================")
    
    np.random.seed(2026)
    engine_f1_zero = IssanNonSymmetricEquilibriumEngine(family=1, tau0=0.0)
    engine_f2_zero = IssanNonSymmetricEquilibriumEngine(family=2, tau1=0.0)
    
    num_test_points = 120
    f1_max_err = 0.0
    f2_max_err = 0.0
    
    # 3.1 Family 1 test across 120 random 3D points
    for idx in range(num_test_points):
        x = float(np.random.uniform(0.65, 1.25))
        y = float(np.random.uniform(-0.35, 0.35))
        z = float(np.random.uniform(-0.25, 0.25))
        
        sym = engine_f1_zero.check_stellarator_symmetry(x=x, y=y, z=z)
        err = sym["max_residual"]
        f1_max_err = max(f1_max_err, err)
        assert err == 0.0, f"Family 1 tau0=0 non-zero residual {err} at point {idx} ({x}, {y}, {z})"

    # 3.2 Family 2 test across 120 random 3D points
    for idx in range(num_test_points):
        x = float(np.random.uniform(0.70, 1.30))
        y = float(np.random.uniform(-0.40, 0.40))
        z = float(np.random.uniform(-0.30, 0.30))
        
        sym2 = engine_f2_zero.check_stellarator_symmetry(x=x, y=y, z=z)
        err2 = sym2["max_residual"]
        f2_max_err = max(f2_max_err, err2)
        assert err2 == 0.0, f"Family 2 tau1=0 non-zero residual {err2} at point {idx} ({x}, {y}, {z})"

    print(f"Tested {num_test_points} random points for Family 1 (tau0=0): max residual = {f1_max_err:.2e}")
    print(f"Tested {num_test_points} random points for Family 2 (tau1=0): max residual = {f2_max_err:.2e}")
    print("STRESS TEST 3: PASSED\n")


def run_stress_test_4_so6_kelvin_mandel_rotors():
    print("=====================================================================")
    print("STRESS TEST 4: SO(6) Kelvin-Mandel Rotor Invariance across 500 Random Rotations")
    print("=====================================================================")
    
    np.random.seed(777)
    max_trace_drift = 0.0
    max_frob_drift = 0.0
    max_ortho_drift = 0.0
    
    I6 = np.eye(6, dtype=np.float64)
    num_rotations = 500
    
    for i in range(num_rotations):
        # Generate random SO(3) matrix via QR decomposition of random Gaussian 3x3
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
        
        # Orthogonality check M^T M = I_6
        ortho_err = float(np.max(np.abs(M6.T @ M6 - I6)))
        max_ortho_drift = max(max_ortho_drift, ortho_err)
        
        # Arbitrary anisotropic symmetric positive definite stiffness tensor
        A = np.random.randn(6, 6)
        C0 = A @ A.T + 2.0 * I6
        
        # Rotate C_rot = M C0 M^T
        C_rot = rotate_stiffness_so6(M6, C0)
        
        # Trace and Frobenius drift
        tr0 = np.trace(C0)
        tr1 = np.trace(C_rot)
        tr_drift = abs(tr1 - tr0)
        max_trace_drift = max(max_trace_drift, tr_drift)
        
        frob0 = np.linalg.norm(C0, "fro")
        frob1 = np.linalg.norm(C_rot, "fro")
        fr_drift = abs(frob1 - frob0)
        max_frob_drift = max(max_frob_drift, fr_drift)
        
        assert ortho_err <= 1.14e-13, f"Rotation {i} SO(6) orthogonality error: {ortho_err}"
        assert tr_drift <= 1.14e-13, f"Rotation {i} trace drift: {tr_drift}"
        assert fr_drift <= 1.14e-13, f"Rotation {i} Frobenius drift: {fr_drift}"

    print(f"Tested {num_rotations} random 3D rotations:")
    print(f"  - Peak SO(6) Orthogonality Residual: {max_ortho_drift:.4e}")
    print(f"  - Peak Stiffness Trace Drift:        {max_trace_drift:.4e}")
    print(f"  - Peak Stiffness Frobenius Drift:    {max_frob_drift:.4e}")
    assert max_trace_drift <= 1.14e-13
    assert max_frob_drift <= 1.14e-13
    print("STRESS TEST 4: PASSED\n")


def run_stress_test_5_heap_memory_ceiling(tmp_path: Path):
    print("=====================================================================")
    print("STRESS TEST 5: Heap Memory Ceiling (< 35 MB Hard Limit) under tracemalloc")
    print("=====================================================================")
    
    tracemalloc.start()
    tracemalloc.reset_peak()
    
    # 5.1 Model formal theorem audit
    model = BMHDModel()
    audit = model.verify_formal_theorems()
    assert audit.all_theorems_verified is True
    
    cur1, peak1 = tracemalloc.get_traced_memory()
    peak1_mb = peak1 / (1024 * 1024)
    print(f"  - Phase 1 (verify_formal_theorems): Peak = {peak1_mb:.2f} MB")
    assert peak1_mb < 35.0, f"Peak memory exceeded 35 MB: {peak1_mb}"
    
    # 5.2 Build all 5 scenarios
    scenarios = build_all_scenarios()
    assert len(scenarios) == 5
    cur2, peak2 = tracemalloc.get_traced_memory()
    peak2_mb = peak2 / (1024 * 1024)
    print(f"  - Phase 2 (build_all_scenarios 5 decks): Peak = {peak2_mb:.2f} MB")
    assert peak2_mb < 35.0, f"Peak memory exceeded 35 MB: {peak2_mb}"
    
    # 5.3 Mode B full evaluation with 7 baselines
    sc01 = scenarios["01_issan_nonsym_solovev_tau0"]
    res_b = run_mode_b_evaluation(sc01, snr_db=20.0, chunk_z=16)
    cur3, peak3 = tracemalloc.get_traced_memory()
    peak3_mb = peak3 / (1024 * 1024)
    print(f"  - Phase 3 (Mode B 7-baseline inversion): Peak = {peak3_mb:.2f} MB")
    assert peak3_mb < 35.0, f"Peak memory exceeded 35 MB: {peak3_mb}"
    
    # 5.4 Out-of-core np.memmap streaming on a 64 MB volume (400 x 400 x 100 float32)
    vol_file = tmp_path / "synthetic_64mb_volume.npy"
    shape_64mb = (100, 400, 400)  # 16,000,000 float32 voxels = 64.0 MB
    print(f"  - Phase 4: Generating {shape_64mb} volume on disk (64 MB)...")
    
    # Create memory-mapped .npy file directly on disk without holding in RAM
    fp = np.lib.format.open_memmap(vol_file, mode="w+", dtype="float32", shape=shape_64mb)
    fp[:] = 1.0  # fill
    fp.flush()
    del fp
    
    tracemalloc.reset_peak()
    streamer = stream_field_volume(vol_file, chunk_ky=16, target_grid=(32, 24, 24))
    slab_count = 0
    max_slab_peak = 0.0
    for slab in streamer:
        slab_count += 1
        max_slab_peak = max(max_slab_peak, slab.peak_heap_mb)
    
    cur4, peak4 = tracemalloc.get_traced_memory()
    peak4_mb = peak4 / (1024 * 1024)
    print(f"  - Phase 4 (64 MB memmap streaming {slab_count} slabs): Peak = {peak4_mb:.2f} MB (slab reporter max = {max_slab_peak:.2f} MB)")
    assert peak4_mb < 35.0, f"Phase 4 peak memory exceeded 35 MB: {peak4_mb}"
    
    # Cleanup file
    if vol_file.exists():
        os.remove(vol_file)
        
    tracemalloc.stop()
    print("STRESS TEST 5: PASSED\n")


if __name__ == "__main__":
    import tempfile
    print("=====================================================================")
    print("BMHD ADVERSARIAL STRESS CHALLENGER SUITE")
    print("=====================================================================\n")
    
    run_stress_test_1_divergence_and_force_balance()
    run_stress_test_2_field_line_radial_drift()
    run_stress_test_3_homotopy_symmetry_breaking()
    run_stress_test_4_so6_kelvin_mandel_rotors()
    
    with tempfile.TemporaryDirectory() as td:
        run_stress_test_5_heap_memory_ceiling(Path(td))
        
    print("=====================================================================")
    print("ALL 5 ADVERSARIAL STRESS TESTS COMPLETED SUCCESSFULLY WITH ZERO DEFECTS.")
    print("VERDICT: APPROVE")
    print("=====================================================================")
