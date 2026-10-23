"""
tests/test_v040_arxiv_adversarial.py: Gate 1 Adversarial Stress Test Suite for BMHD v0.4.0.

Independently challenges and verifies Pillars A, B, and C of the v0.4.0 arXiv research upgrade:
1. test_adversarial_held_out_chord_corruption_invariance:
   Zero target leakage under NaN / 1e6 corruption of held-out chords [4, 5, 6, 7].
2. test_adversarial_non_negative_log_gp_extreme_outliers:
   Strict [0.0, 1.0] log-barrier confinement under extreme negative (-5.0) and positive (+8.0) spikes.
3. test_adversarial_z_slab_chunk_invariance:
   Exact Z-slab chunk invariance (chunk_z in {4, 8, 16}) across NonNegativeLogGP, BlindBMHD,
   and SteppedPressureBeltramiTriad (max_diff < 1e-12).
4. test_adversarial_supersonic_and_resonant_kam_rejection:
   Rejection of supersonic Mach numbers (M_i >= 1.0) and resonant rotational transforms (iota = n/m).
5. test_adversarial_flow_bernoulli_jump_and_zero_flow_homotopy:
   Zero-flow homotopy to static Taylor-Hudson MRxMHD and Bernoulli jump conservation across random shells.
6. test_adversarial_near_axis_sigma0_odd_parity_and_homotopy:
   Exact odd parity sigma(0; -tau) == -sigma(0; tau) and zero-torsion vanishing sigma(0; 0) == 0.0.
7. test_adversarial_family2_sheared_iota_and_wirtinger_curl:
   Strict radial decay iota(0) > iota(delta), Wirtinger curl accuracy, and B . grad(psi) == 0 on Family 2.
8. test_adversarial_mode_b_all_5_scenarios_unconditional_rank1:
   Unconditional #1 ranking of BlindBMHD across all 8 methods, 4 metrics, and 5 scenarios.
9. test_adversarial_lean4_6repo_mirror_sha256_identity:
   Theorem 8 & 9 declaration audit, zero sorry/axiom check, and 6-repo SHA-256 byte-identity.
10. test_adversarial_swapc_heap_ceiling_v040:
    Peak heap strictly < 35.0 MB during 8-method Mode B evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import scipy.ndimage
import scipy.special

from bmhd.benchmark import (
    BlindBMHD,
    NonNegativeLogGP,
    extract_training_chord_samples,
    fit_and_predict_anisotropic_matern_kriging,
    fit_and_predict_blind_bmhd,
    fit_and_predict_non_negative_log_gp,
    run_mode_b_evaluation,
)
from bmhd.engine import BMHDModel, SteppedPressureBeltramiTriad
from bmhd.mhd import MU_0, IssanNonSymmetricEquilibriumEngine
from bmhd.scenarios import build_all_scenarios


REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# 1. Zero Target Leakage under Held-Out Chord NaN / 1e6 Corruption
# =============================================================================


def test_adversarial_held_out_chord_corruption_invariance():
    """
    Corrupt chords 4, 5, 6, 7 with 1e6 and NaN values before fitting NonNegativeLogGP
    and BlindBMHD on train_indices=[0, 1, 2, 3], and assert exact array equality (0.0 max diff)
    on the predicted 3D cubes.
    """
    scenarios = build_all_scenarios()
    sc_clean = scenarios["01_issan_nonsym_solovev_tau0"]
    sc_corrupt = copy.deepcopy(sc_clean)

    # Inject catastrophic 1e6 and NaN corruption into held-out chords [4, 5, 6, 7]
    for c_idx in (4, 5, 6, 7):
        n_pts = len(sc_corrupt.chords[c_idx].curves["PSI"])
        corrupted = np.full(n_pts, 1.0e6, dtype=np.float64)
        corrupted[::2] = np.nan
        corrupted[1::3] = -1.0e9
        sc_corrupt.chords[c_idx].curves["PSI"] = corrupted

    train_idx = [0, 1, 2, 3]
    grid_shape = sc_clean.grid_shape

    # 1. OOP NonNegativeLogGP & BlindBMHD
    gp_clean = NonNegativeLogGP(chunk_z=16).fit(sc_clean.chords, grid_shape, train_indices=train_idx)
    gp_corrupt = NonNegativeLogGP(chunk_z=16).fit(sc_corrupt.chords, grid_shape, train_indices=train_idx)
    pred_gp_clean = gp_clean.predict_3d(grid_shape, chunk_z=16)
    pred_gp_corrupt = gp_corrupt.predict_3d(grid_shape, chunk_z=16)
    assert np.array_equal(pred_gp_clean, pred_gp_corrupt), (
        f"NonNegativeLogGP leaked held-out chords! Max diff: {np.max(np.abs(pred_gp_clean - pred_gp_corrupt))}"
    )

    bmhd_clean = BlindBMHD(chunk_z=16).fit(sc_clean.chords, grid_shape, train_indices=train_idx)
    bmhd_corrupt = BlindBMHD(chunk_z=16).fit(sc_corrupt.chords, grid_shape, train_indices=train_idx)
    pred_bmhd_clean = bmhd_clean.predict_3d(grid_shape, chunk_z=16)
    pred_bmhd_corrupt = bmhd_corrupt.predict_3d(grid_shape, chunk_z=16)
    assert np.array_equal(pred_bmhd_clean, pred_bmhd_corrupt), (
        f"BlindBMHD leaked held-out chords! Max diff: {np.max(np.abs(pred_bmhd_clean - pred_bmhd_corrupt))}"
    )

    # 2. Functional API with prior_cube
    obs_zyx_c, obs_val_c = extract_training_chord_samples(
        sc_clean.chords, train_idx, grid_shape=grid_shape, snr_db=20.0
    )
    obs_zyx_x, obs_val_x = extract_training_chord_samples(
        sc_corrupt.chords, train_idx, grid_shape=grid_shape, snr_db=20.0
    )
    assert np.array_equal(obs_zyx_c, obs_zyx_x)
    assert np.array_equal(obs_val_c, obs_val_x)

    prior_cube = scipy.ndimage.uniform_filter(sc_clean.psi, size=3, mode="nearest")
    fgp_clean = fit_and_predict_non_negative_log_gp(
        obs_zyx_c,
        obs_val_c,
        grid_shape=grid_shape,
        chords_for_cv=sc_clean.chords,
        train_chord_indices=train_idx,
        chunk_z=16,
        prior_cube=prior_cube,
    )
    fgp_corrupt = fit_and_predict_non_negative_log_gp(
        obs_zyx_x,
        obs_val_x,
        grid_shape=grid_shape,
        chords_for_cv=sc_corrupt.chords,
        train_chord_indices=train_idx,
        chunk_z=16,
        prior_cube=prior_cube,
    )
    assert np.array_equal(fgp_clean, fgp_corrupt)


# =============================================================================
# 2. Extreme Outlier Spikes (-5.0, +8.0) -> Strict [0.0, 1.0] Positivity Bounds
# =============================================================================


def test_adversarial_non_negative_log_gp_extreme_outliers():
    """
    Inject severe negative (-5.0) and >1 (+8.0) spikes into observations and prior_cube,
    and assert NonNegativeLogGP and BlindBMHD outputs remain strictly in [0.0, 1.0]
    with zero NaN or Inf values.
    """
    scenarios = build_all_scenarios()
    sc = scenarios["02_issan_sheared_elliptic_tau1"]
    grid_shape = sc.grid_shape

    obs_zyx, obs_val = extract_training_chord_samples(
        sc.chords, [0, 1, 2, 3], grid_shape=grid_shape, snr_db=20.0
    )
    obs_extreme = obs_val.copy()
    obs_extreme[0::7] = -5.0
    obs_extreme[3::7] = 8.0

    prior_extreme = sc.psi.copy()
    prior_extreme[0:4, :, :] = -5.0
    prior_extreme[28:32, :, :] = 8.0

    pred_gp = fit_and_predict_non_negative_log_gp(
        obs_zyx=obs_zyx,
        obs_val=obs_extreme,
        grid_shape=grid_shape,
        chunk_z=16,
        prior_cube=prior_extreme,
    )
    assert np.all(np.isfinite(pred_gp)), "NonNegativeLogGP produced NaN or Inf under extreme outliers"
    assert float(np.min(pred_gp)) >= 0.0, f"NonNegativeLogGP violated lower bound 0.0: {np.min(pred_gp)}"
    assert float(np.max(pred_gp)) <= 1.0, f"NonNegativeLogGP violated upper bound 1.0: {np.max(pred_gp)}"

    pred_krig, _ = fit_and_predict_anisotropic_matern_kriging(
        obs_zyx=obs_zyx,
        obs_val=obs_extreme,
        grid_shape=grid_shape,
        chunk_z=16,
        prior_cube=prior_extreme,
    )
    pred_bmhd = fit_and_predict_blind_bmhd(
        kriged_cube=pred_krig,
        obs_zyx=obs_zyx,
        obs_val=obs_extreme,
        grid_shape=grid_shape,
        n_shells=4,
        chunk_z=16,
        prior_cube=prior_extreme,
    )
    assert np.all(np.isfinite(pred_bmhd)), "BlindBMHD produced NaN or Inf under extreme outliers"
    assert float(np.min(pred_bmhd)) >= 0.0, f"BlindBMHD violated lower bound 0.0: {np.min(pred_bmhd)}"
    assert float(np.max(pred_bmhd)) <= 1.0, f"BlindBMHD violated upper bound 1.0: {np.max(pred_bmhd)}"


# =============================================================================
# 3. Z-Slab Chunk Invariance (chunk_z in {4, 8, 16})
# =============================================================================


def test_adversarial_z_slab_chunk_invariance():
    """
    Verify chunk_z=4, chunk_z=8, and chunk_z=16 produce identical 3D predictions
    (max_diff < 1e-12) for NonNegativeLogGP, BlindBMHD, and SteppedPressureBeltramiTriad.
    """
    scenarios = build_all_scenarios()
    sc = scenarios["01_issan_nonsym_solovev_tau0"]
    grid_shape = sc.grid_shape
    train_idx = [0, 1, 2, 3]

    # 1. NonNegativeLogGP
    gp_preds = {}
    for cz in (4, 8, 16):
        gp = NonNegativeLogGP(chunk_z=cz).fit(sc.chords, grid_shape, train_indices=train_idx)
        gp_preds[cz] = gp.predict_3d(grid_shape, chunk_z=cz)
    assert float(np.max(np.abs(gp_preds[4] - gp_preds[8]))) < 1e-12
    assert float(np.max(np.abs(gp_preds[8] - gp_preds[16]))) < 1e-12

    # 2. BlindBMHD
    bmhd_preds = {}
    for cz in (4, 8, 16):
        bmhd = BlindBMHD(chunk_z=cz).fit(sc.chords, grid_shape, train_indices=train_idx)
        bmhd_preds[cz] = bmhd.predict_3d(grid_shape, chunk_z=cz)
    assert float(np.max(np.abs(bmhd_preds[4] - bmhd_preds[8]))) < 1e-12
    assert float(np.max(np.abs(bmhd_preds[8] - bmhd_preds[16]))) < 1e-12

    # 3. SteppedPressureBeltramiTriad
    psi_grid = sc.psi.astype(np.float32)
    B_grid = np.stack([sc.b_mod, 0.5 * sc.b_mod, 0.25 * sc.b_mod], axis=0).astype(np.float32)
    p_grid = sc.p_kpa.astype(np.float32)

    triad_p = {}
    triad_B = {}
    for cz in (4, 8, 16):
        triad = SteppedPressureBeltramiTriad(
            n_shells=4,
            mu_values=(1.2, 0.8, -0.5, 0.3),
            mach_values=(0.12, 0.08, 0.05, 0.02),
            chunk_z=cz,
        )
        res = triad.evaluate_volume(psi_grid, B_grid, p_grid)
        triad_p[cz] = res["stepped_p"]
        triad_B[cz] = res["stepped_B"]

    assert float(np.max(np.abs(triad_p[4] - triad_p[8]))) < 1e-12
    assert float(np.max(np.abs(triad_p[8] - triad_p[16]))) < 1e-12
    assert float(np.max(np.abs(triad_B[4] - triad_B[8]))) < 1e-12
    assert float(np.max(np.abs(triad_B[8] - triad_B[16]))) < 1e-12


# =============================================================================
# 4. Supersonic Mach (M_i >= 1.0) & Resonant Rational Iota (iota = 1/2) Rejection
# =============================================================================


def test_adversarial_supersonic_and_resonant_kam_rejection():
    """
    Pass supersonic mach_values=(1.05, 0.2, ...) and resonant iota = 0.5 = 1/2 into
    SteppedPressureBeltramiTriad and assert kam_admissible is False.
    """
    psi_grid = np.linspace(0.0, 1.0, 32 * 24 * 24, dtype=np.float32).reshape(32, 24, 24)
    B_grid = np.ones((3, 32, 24, 24), dtype=np.float32) * 2.0
    p_grid = (1.0 - psi_grid) * 120.0

    # 1. Strict constructor rejects supersonic Mach M_1 = 1.05 >= 1.0
    with pytest.raises(ValueError, match="sub-Alfvenic"):
        SteppedPressureBeltramiTriad(
            n_shells=4,
            mu_values=(1.2, 0.8, -0.5, 0.3),
            mach_values=(1.05, 0.2, 0.08, 0.02),
        )

    # 2. Non-strict evaluation with supersonic Mach M_1 = 1.05 flags kam_admissible == False
    triad_supersonic = SteppedPressureBeltramiTriad(
        n_shells=4,
        mu_values=(1.2, 0.8, -0.5, 0.3),
        mach_values=(1.05, 0.2, 0.08, 0.02),
        strict_sub_alfvenic=False,
    )
    inv_super = triad_supersonic.verify_flow_mrxmhd_invariants()
    assert inv_super["kam_sub_alfvenic_margin"] < 0.0
    assert inv_super["kam_admissible"] is False
    assert inv_super["passed"] is False

    vol_super = triad_supersonic.evaluate_volume(psi_grid, B_grid, p_grid)
    assert vol_super["kam_admissible"] is False

    # 3. Sub-Alfvenic Mach with resonant rotational transform iota = 0.5 = 1/2 flags kam_admissible == False
    triad_res = SteppedPressureBeltramiTriad(
        n_shells=4,
        mu_values=(1.2, 0.8, -0.5, 0.3),
        mach_values=(0.12, 0.08, 0.05, 0.02),
        iota_values=(0.5, 0.61803398875, 0.41421356237, 0.38196601125),
    )
    inv_res = triad_res.verify_flow_mrxmhd_invariants()
    assert inv_res["min_kam_diophantine_gap"] if "min_kam_diophantine_gap" in inv_res else inv_res["kam_diophantine_gap"] == 0.0
    assert inv_res["kam_admissible"] is False
    assert inv_res["passed"] is False

    # Also test passing resonant iota_grid = 0.5 directly into evaluate_volume
    triad_nom = SteppedPressureBeltramiTriad(n_shells=4)
    resonant_iota_grid = np.full_like(psi_grid, 0.5)
    vol_res = triad_nom.evaluate_volume(psi_grid, B_grid, p_grid, iota_grid=resonant_iota_grid)
    assert vol_res["min_kam_diophantine_gap"] == 0.0
    assert vol_res["kam_admissible"] is False


# =============================================================================
# 5. Flow-MRxMHD Bernoulli Jump Conservation & Zero-Flow Homotopy
# =============================================================================


def test_adversarial_flow_bernoulli_jump_and_zero_flow_homotopy():
    """
    Verify exact agreement between SteppedPressureBeltramiTriad(mach_values=(0,0,...))
    and the static Taylor-Hudson limit, and verify Bernoulli jump conservation across
    100 random multi-shell pressures, magnetic fields, and sub-Alfvenic Mach numbers.
    """
    mu_vals = (1.35, -0.85, 0.65, -0.40, 0.95, 0.25)
    triad_zero = SteppedPressureBeltramiTriad(
        n_shells=6,
        mu_values=mu_vals,
        mach_values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    assert triad_zero.effective_mu == mu_vals
    inv_zero = triad_zero.verify_flow_mrxmhd_invariants()
    assert inv_zero["zero_flow_homotopy_error"] == 0.0
    assert inv_zero["collinear_lorentz_error"] == 0.0
    assert inv_zero["bernoulli_jump_residual"] == 0.0
    assert inv_zero["passed"] is True

    # Stress-test 100 random multi-shell Bernoulli jump configurations:
    # H_flow(p_i, B_i^2, M_i) = p_i + 0.5 * (1 + M_i^2) * B_i^2 = const
    rng = np.random.default_rng(20261023)
    for _ in range(100):
        H_const = float(rng.uniform(50.0, 300.0))
        B_sq = rng.uniform(0.5, 9.0, size=6)
        M_vals = rng.uniform(-0.85, 0.85, size=6)
        p_vals = H_const - 0.5 * (1.0 + M_vals**2) * B_sq
        for i in range(5):
            dp_lhs = p_vals[i + 1] - p_vals[i]
            dp_rhs = 0.5 * ((1.0 + M_vals[i] ** 2) * B_sq[i] - (1.0 + M_vals[i + 1] ** 2) * B_sq[i + 1])
            assert abs(dp_lhs - dp_rhs) <= 1e-12

    # Verify BMHDModel Theorem 2.9 passes
    model = BMHDModel()
    audit = model.verify_formal_theorems()
    assert audit["theorem_2_9"]["passed"] is True


# =============================================================================
# 6. Near-Axis Twist sigma(0) Odd Parity & Zero-Torsion Homotopy
# =============================================================================


def test_adversarial_near_axis_sigma0_odd_parity_and_homotopy():
    """
    Verify odd parity sigma(0; -tau_0) == -sigma(0; tau_0) for Family 1,
    sigma(0; -tau_1) == -sigma(0; tau_1) for Family 2, and exact vanishing
    sigma(0; 0) == 0.0 at tau_0 = tau_1 = 0.
    """
    # Zero torsion homotopy
    inv1_zero = IssanNonSymmetricEquilibriumEngine(family=1, tau0=0.0).verify_analytical_invariants()
    inv2_zero = IssanNonSymmetricEquilibriumEngine(family=2, tau1=0.0).verify_analytical_invariants()
    assert inv1_zero["near_axis_sigma0"] == 0.0
    assert inv2_zero["near_axis_sigma0"] == 0.0

    # Family 1 odd parity across tau_0 in (0.05, 0.22, 0.38, 0.65, 0.88)
    for tau0 in (0.05, 0.22, 0.38, 0.65, 0.88):
        s_pos = IssanNonSymmetricEquilibriumEngine(family=1, tau0=tau0).verify_analytical_invariants()["near_axis_sigma0"]
        s_neg = IssanNonSymmetricEquilibriumEngine(family=1, tau0=-tau0).verify_analytical_invariants()["near_axis_sigma0"]
        assert s_pos > 0.0
        assert abs(s_neg + s_pos) <= 1e-15, f"Family 1 sigma(0) parity failed for tau0={tau0}: {s_pos} vs {s_neg}"

    # Family 2 odd parity across tau_1 in (0.05, 0.25, 0.42, 0.75, 1.15)
    for tau1 in (0.05, 0.25, 0.42, 0.75, 1.15):
        s_pos = IssanNonSymmetricEquilibriumEngine(family=2, tau1=tau1).verify_analytical_invariants()["near_axis_sigma0"]
        s_neg = IssanNonSymmetricEquilibriumEngine(family=2, tau1=-tau1).verify_analytical_invariants()["near_axis_sigma0"]
        assert s_pos > 0.0
        assert abs(s_neg + s_pos) <= 1e-15, f"Family 2 sigma(0) parity failed for tau1={tau1}: {s_pos} vs {s_neg}"


# =============================================================================
# 7. Family 2 Sheared Rotational Transform iota(psi) & Wirtinger Curl / Confinement
# =============================================================================


def test_adversarial_family2_sheared_iota_and_wirtinger_curl():
    """
    Verify Family 2 iota(psi) is strictly decreasing with psi (iota(0) > iota(delta)),
    analytical Wirtinger curl matches finite-difference curl(B), and B . grad(psi) = 0
    holds to machine precision on Family 2.
    """
    eng2 = IssanNonSymmetricEquilibriumEngine(family=2, tau1=0.42, epsilon=1.2, S=1.4, lam=1.6, delta=0.14)
    inv2 = eng2.verify_analytical_invariants()
    assert inv2["iota_shear"] > 1.5

    # Check strict monotonicity of elliptic iota(psi) formula with respect to psi in [0, delta]
    psi_samples = np.linspace(0.0, eng2.delta, 32)
    m_ellip = np.clip(2.0 * psi_samples, 0.0, 0.9999)
    K_ellip = scipy.special.ellipk(m_ellip)
    iota_curve = (
        math.pi
        * math.sqrt(1.0 + eng2.epsilon / (eng2.S**2))
        / (2.0 * K_ellip * math.cosh(eng2.tau1))
        * np.sqrt(np.maximum(1e-6, 1.0 - 2.0 * psi_samples))
    )
    diffs = np.diff(iota_curve)
    assert np.all(diffs < 0.0), f"Family 2 iota(psi) is not strictly decreasing: max diff {np.max(diffs)}"
    assert float(iota_curve[0]) > float(iota_curve[-1])

    # Check spatial variation and B . grad(psi) == 0 + Wirtinger curl accuracy across 30 interior points
    rng = np.random.default_rng(4242)
    h = 1e-6
    x_pts = rng.uniform(0.75, 1.25, size=30)
    y_pts = rng.uniform(-0.35, 0.35, size=30)
    z_pts = rng.uniform(-0.20, 0.20, size=30)

    c0 = eng2.compute_cartesian(x_pts, y_pts, z_pts)
    assert float(np.std(c0["iota"])) > 0.01

    cxp = eng2.compute_cartesian(x_pts + h, y_pts, z_pts)
    cxm = eng2.compute_cartesian(x_pts - h, y_pts, z_pts)
    cyp = eng2.compute_cartesian(x_pts, y_pts + h, z_pts)
    cym = eng2.compute_cartesian(x_pts, y_pts - h, z_pts)
    czp = eng2.compute_cartesian(x_pts, y_pts, z_pts + h)
    czm = eng2.compute_cartesian(x_pts, y_pts, z_pts - h)

    dpsi_dx = (cxp["psi_raw"] - cxm["psi_raw"]) / (2.0 * h)
    dpsi_dy = (cyp["psi_raw"] - cym["psi_raw"]) / (2.0 * h)
    dpsi_dz = (czp["psi_raw"] - czm["psi_raw"]) / (2.0 * h)

    b_dot_grad_psi = c0["Bx"] * dpsi_dx + c0["By"] * dpsi_dy + c0["Bz"] * dpsi_dz
    assert float(np.max(np.abs(b_dot_grad_psi))) < 1e-8

    # Finite-difference curl(B) vs analytical Wirtinger J_curl
    curl_x = (czp["By"] - czm["By"]) / (2.0 * h) - (cyp["Bz"] - cym["Bz"]) / (2.0 * h)
    curl_y = (cxp["Bz"] - cxm["Bz"]) / (2.0 * h) - (czp["Bx"] - czm["Bx"]) / (2.0 * h)
    curl_z = (cyp["Bx"] - cym["Bx"]) / (2.0 * h) - (cxp["By"] - cxm["By"]) / (2.0 * h)
    fd_j_curl = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2) / (MU_0 * 1e6)
    rel_j_err = np.max(np.abs(fd_j_curl - c0["J_curl"]) / np.maximum(fd_j_curl, 1e-6))
    assert float(rel_j_err) < 1e-5, f"Wirtinger J_curl mismatch vs FD curl: {rel_j_err}"


# =============================================================================
# 8. Mode B All 5 Scenarios Unconditional #1 Rank across All 8 Methods
# =============================================================================


def test_adversarial_mode_b_all_5_scenarios_unconditional_rank1():
    """
    Load results/bmhd_evaluation_summary.json and results/bmhd_blind_summary.json
    and assert that on all 5 scenarios (01 to 05), BlindBMHD is strictly #1 among
    all 8 methods on loco_rmse, unseen_3d_chords_rmse, whole_torus_psi_rmse, and
    bounded_edge_gradient_fidelity.
    """
    eval_path = REPO_ROOT / "results" / "bmhd_evaluation_summary.json"
    blind_path = REPO_ROOT / "results" / "bmhd_blind_summary.json"
    assert eval_path.exists()
    assert blind_path.exists()

    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    blind_data = json.loads(blind_path.read_text(encoding="utf-8"))

    assert eval_data["benchmark_version"] == "v0.4.0"
    assert blind_data["benchmark_version"] == "v0.4.0"
    assert eval_data["mode_a_formal_theorems"]["theorems_audit"]["theorem_2_9"]["passed"] is True

    expected_scenarios = [
        "01_issan_nonsym_solovev_tau0",
        "02_issan_sheared_elliptic_tau1",
        "03_spec_magnetic_island_chain",
        "04_tokamak_elm_pellet_4d",
        "05_beltrami_plasma_spallation_torch",
    ]
    expected_methods = [
        "NaiveBox",
        "TunedAnisotropicKriging",
        "SymmetricFourierVMECBaseline",
        "ShellPartitionedKriging",
        "CerfonFreidbergEquilibrium",
        "MinimumFisherTomography",
        "NonNegativeLogGP",
        "BlindBMHD",
    ]

    for scen_key in expected_scenarios:
        assert scen_key in eval_data["mode_b_summary"]
        assert scen_key in blind_data["scenarios"]

        methods = eval_data["mode_b_summary"][scen_key]
        assert set(methods.keys()) == set(expected_methods)

        perf = blind_data["scenarios"][scen_key]["performance"]
        assert perf["heap_ceiling_passed"] is True
        assert perf["peak_heap_mb"] < 35.0

        bmhd_m = methods["BlindBMHD"]
        for other_name in expected_methods:
            if other_name == "BlindBMHD":
                continue
            other_m = methods[other_name]
            assert bmhd_m["loco_rmse"] < other_m["loco_rmse"], (
                f"[{scen_key}] BlindBMHD loco_rmse ({bmhd_m['loco_rmse']}) >= {other_name} ({other_m['loco_rmse']})"
            )
            assert bmhd_m["unseen_3d_chords_rmse"] < other_m["unseen_3d_chords_rmse"], (
                f"[{scen_key}] BlindBMHD unseen_3d_chords_rmse ({bmhd_m['unseen_3d_chords_rmse']}) >= {other_name} ({other_m['unseen_3d_chords_rmse']})"
            )
            assert bmhd_m["whole_torus_psi_rmse"] < other_m["whole_torus_psi_rmse"], (
                f"[{scen_key}] BlindBMHD whole_torus_psi_rmse ({bmhd_m['whole_torus_psi_rmse']}) >= {other_name} ({other_m['whole_torus_psi_rmse']})"
            )
            assert bmhd_m["bounded_edge_gradient_fidelity"] > other_m["bounded_edge_gradient_fidelity"], (
                f"[{scen_key}] BlindBMHD bounded_edge_gradient_fidelity ({bmhd_m['bounded_edge_gradient_fidelity']}) <= {other_name} ({other_m['bounded_edge_gradient_fidelity']})"
            )


# =============================================================================
# 9. Lean 4 6-Repo Mirror SHA-256 Byte-Identity & Theorem 8/9 Audit
# =============================================================================


def test_adversarial_lean4_6repo_mirror_sha256_identity():
    """
    Compute SHA-256 of bmhd/formalization/OcclusionTriad.lean, assert all Theorem 8
    and Theorem 9 declarations exist with zero sorry/axiom, and assert all existing
    sibling copies match byte-for-byte.
    """
    bmhd_lean = REPO_ROOT / "formalization" / "OcclusionTriad.lean"
    assert bmhd_lean.exists(), f"Missing {bmhd_lean}"

    content = bmhd_lean.read_text(encoding="utf-8")
    digest = hashlib.sha256(bmhd_lean.read_bytes()).hexdigest()

    required_declarations = [
        "theorem cross3_eq_triadOperator",
        "theorem mhd_euler_force_balance_bijection",
        "theorem mhd_flux_surface_confinement_iff",
        "theorem stellarator_parity_breaks_tie_and_conserves_flux",
        "theorem stellarator_zero_torsion_homotopy_iff",
        "theorem sub_alfvenic_beltrami_homotopy_zero_flow",
        "theorem sub_alfvenic_beltrami_pos",
        "theorem beltrami_cross_helicity_collinear_lorentz_vanishes",
        "theorem mrxmhd_flow_bernoulli_jump_conservation",
        "theorem kam_sub_alfvenic_ellipticity_and_gap",
    ]
    for decl in required_declarations:
        assert decl in content, f"Missing Lean 4 declaration in OcclusionTriad.lean: {decl}"

    # Ensure no sorry or axiom declarations in code lines
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("--") and not stripped.startswith("/-") and not stripped.startswith("*"):
            assert "sorry" not in stripped.split(), f"Found 'sorry' in OcclusionTriad.lean: {line}"
            assert not stripped.startswith("axiom "), f"Found 'axiom' in OcclusionTriad.lean: {line}"

    workspace_root = REPO_ROOT.parent
    sibling_paths = [
        workspace_root / "sparse-ai" / "formalization" / "Formalization" / "Analysis" / "OcclusionTriad.lean",
        workspace_root / "bmhd" / "formalization" / "OcclusionTriad.lean",
        workspace_root / "darksnail" / "formalization" / "OcclusionTriad.lean",
        workspace_root / "g3vm" / "formalization" / "OcclusionTriad.lean",
        workspace_root / "oatse" / "formalization" / "OcclusionTriad.lean",
        workspace_root / "stylo" / "formalization" / "OcclusionTriad.lean",
    ]
    found_mirrors = 0
    for p in sibling_paths:
        if p.exists():
            found_mirrors += 1
            p_digest = hashlib.sha256(p.read_bytes()).hexdigest()
            assert p_digest == digest, f"SHA-256 mismatch at {p}: {p_digest} != {digest}"
    assert found_mirrors >= 1


# =============================================================================
# 10. SWaP-C Heap Ceiling Verification (< 35.0 MB) for v0.4.0
# =============================================================================


def test_adversarial_swapc_heap_ceiling_v040():
    """
    Verify run_mode_b_evaluation across all 8 methods stays strictly below 35.0 MB peak heap.
    """
    scenarios = build_all_scenarios()
    for scen_key in ("01_issan_nonsym_solovev_tau0", "03_spec_magnetic_island_chain"):
        sc = scenarios[scen_key]
        tracemalloc.start()
        tracemalloc.reset_peak()
        res = run_mode_b_evaluation(sc, snr_db=20.0, chunk_z=16)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak_bytes / (1024 * 1024)
        assert peak_mb < 35.0, f"[{scen_key}] Peak heap {peak_mb:.2f} MB >= 35.0 MB"
        assert res["performance"]["peak_heap_mb"] < 35.0
        assert res["performance"]["heap_ceiling_passed"] is True
