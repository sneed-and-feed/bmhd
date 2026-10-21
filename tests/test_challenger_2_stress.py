"""
tests/test_challenger_2_stress.py: Adversarial Stress & Robustness Suite (Challenger 2).

Empirical verification and adversarial challenge covering:
1. Bounded Edge Gradient Fidelity strictly within [0%, 100%] under adversarial perturbations.
2. Discrete Morse impulse and 100.0% clean spike recovery under monotonic glare ramps and flat-topped washout.
3. Zero target leakage empirical proof across all 7 baselines.
4. Mode B sparse-chord reconstruction under SNR in {inf, 20, 10, 5 dB} and jitter in {0, 1, 2, 3 voxels}.
5. Out-of-core field-cube streaming on large 64 MB volumes with chunk_ky windowing (< 35 MB heap).
6. Baseline OOP class interface audit.
"""

from __future__ import annotations

import copy
import math
import os
import shutil
import tempfile
import tracemalloc
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from bmhd.engine import (
    BeltramiChordShell,
    BeltramiChordShellResult,
    SteppedPressureBeltramiTriad,
)
from bmhd.scenarios import build_all_scenarios
from bmhd.io import stream_field_volume
from bmhd.benchmark import (
    compute_edge_gradient_fidelity,
    compute_whole_torus_rmse,
    compute_unseen_chords_rmse,
    add_deterministic_noise,
    extract_training_chord_samples,
    run_mode_b_evaluation,
    run_snr_sweep,
    run_jitter_sweep,
    run_glare_ablation,
    fit_and_predict_naive_box,
    fit_and_predict_anisotropic_matern_kriging,
    fit_and_predict_symmetric_vmec,
    fit_and_predict_shell_partitioned_kriging,
    fit_and_predict_cerfon_freidberg,
    fit_and_predict_minimum_fisher_tomography,
    fit_and_predict_blind_bmhd,
    NaiveBox,
    TunedAnisotropicKriging,
    SymmetricFourierVMECBaseline,
    ShellPartitionedKriging,
    CerfonFreidbergEquilibrium,
    MinimumFisherTomography,
    BlindBMHD,
)


# =============================================================================
# 1. Bounded Edge Gradient Fidelity: Strictly in [0.0, 1.0] ([0%, 100%])
# =============================================================================

class TestBoundedEdgeGradientFidelity:
    """Adversarial stress-testing of Bounded Edge Gradient Fidelity."""

    def test_fidelity_bounds_under_extreme_adversarial_inputs(self):
        rng = np.random.default_rng(2026)
        true_cube = rng.standard_normal((32, 24, 24))

        # Case 1: Identical prediction -> Exactly 1.0 (100.0%)
        fid_ident = compute_edge_gradient_fidelity(true_cube, true_cube)
        assert fid_ident == 1.0, f"Identical cubes did not give 1.0: {fid_ident}"

        # Case 2: Sign-inverted prediction -> Finite and in [0, 1]
        fid_inv = compute_edge_gradient_fidelity(-true_cube, true_cube)
        assert 0.0 <= fid_inv <= 1.0, f"Inverted cube fidelity outside bounds: {fid_inv}"

        # Case 3: Extreme blowout scaling (1e6x, 1e12x) -> Must be clamped to 0.0, not negative
        for scale in [1e3, 1e6, 1e12]:
            fid_blowout = compute_edge_gradient_fidelity(scale * true_cube, true_cube)
            assert fid_blowout == 0.0, f"Blown-out scale {scale} did not clamp to 0.0: {fid_blowout}"

        # Case 4: Zero prediction -> Clamped to 0.0
        fid_zero = compute_edge_gradient_fidelity(np.zeros_like(true_cube), true_cube)
        assert fid_zero == 0.0, f"Zero prediction did not give 0.0: {fid_zero}"

        # Case 5: Uniform flat ground truth (zero gradient everywhere) -> Safe fallback 1.0
        flat_true = np.full((32, 24, 24), 42.0)
        fid_flat = compute_edge_gradient_fidelity(rng.standard_normal((32, 24, 24)), flat_true)
        assert fid_flat == 1.0, f"Flat ground truth did not yield safe 1.0: {fid_flat}"

        # Case 6: Pure random noise against structured step jump
        stepped_true = np.zeros((32, 24, 24))
        stepped_true[:, :, 12:] = 10.0
        for seed in range(50):
            noise_pred = rng.standard_normal((32, 24, 24)) * 5.0
            fid_noise = compute_edge_gradient_fidelity(noise_pred, stepped_true)
            assert 0.0 <= fid_noise <= 1.0, f"Noise seed {seed} produced invalid fidelity: {fid_noise}"


# =============================================================================
# 2. Discrete Morse Impulse: Glare Ramps & Flat-Topped Washout
# =============================================================================

class TestDiscreteMorseImpulseAndSpikeRecovery:
    """Adversarial stress-testing of 3-point discrete Morse impulse and spike recovery."""

    def test_monotonic_glare_ramps_rejected(self):
        """Verify that monotonic ramps produce Morse impulse <= 0 everywhere."""
        Z = 64
        # Linear upward and downward ramps
        ramp_up = torch.linspace(0.0, 50.0, Z)
        ramp_down = torch.linspace(50.0, 0.0, Z)
        m_up = BeltramiChordShell.compute_3point_discrete_morse_impulse(ramp_up)
        m_down = BeltramiChordShell.compute_3point_discrete_morse_impulse(ramp_down)

        assert torch.all(m_up <= 1e-6), f"Upward linear ramp produced positive Morse impulse: {m_up.max()}"
        assert torch.all(m_down <= 1e-6), f"Downward linear ramp produced positive Morse impulse: {m_down.max()}"

        # Quadratic ramp
        ramp_quad = (torch.linspace(0.0, 10.0, Z) ** 2)
        m_quad = BeltramiChordShell.compute_3point_discrete_morse_impulse(ramp_quad)
        assert torch.all(m_quad <= 1e-6), f"Quadratic ramp produced positive Morse impulse: {m_quad.max()}"

        # Exponential glare ramp
        ramp_exp = torch.exp(torch.linspace(0.0, 3.0, Z))
        m_exp = BeltramiChordShell.compute_3point_discrete_morse_impulse(ramp_exp)
        assert torch.all(m_exp <= 1e-6), f"Exponential ramp produced positive Morse impulse: {m_exp.max()}"

    def test_flat_topped_washout_plateau_rejected(self):
        """Verify that flat-topped saturation plateaus produce exactly zero Morse impulse."""
        Z = 64
        # Ramp entering a saturated flat plateau and staying flat
        plateau = torch.cat([
            torch.linspace(1.0, 8.0, 20),
            torch.full((24,), 8.0),  # Saturated plateau
            torch.linspace(8.0, 2.0, 20),
        ])
        m_plat = BeltramiChordShell.compute_3point_discrete_morse_impulse(plateau)
        # Interior of the plateau (indices 22 to 42) must have exactly 0.0 Morse impulse
        assert torch.all(m_plat[22:42] == 0.0), f"Plateau interior had non-zero Morse impulse: {m_plat[22:42]}"

    def test_clean_spike_recovery_with_thin_filaments_and_current_sheets(self):
        """Verify 100.0% clean spike recovery of thin ELM filaments and current sheets under glare."""
        shell = BeltramiChordShell(outlier_budget=8, condensed_budget=8)

        # Adversarial waveform: H-mode pedestal + monotonic glare ramp + flat plateau + 2 sharp spikes
        Z = 64
        base_trace = torch.linspace(2.5, 0.4, Z)
        # Glare ramp
        base_trace[10:25] += torch.linspace(0.0, 0.5, 15)
        # Saturated divertor washout plateau
        base_trace[28:38] = 2.0
        # Thin ELM filament at z=12 (1-voxel width)
        base_trace[12] += 3.5
        # Resonant current sheet spike at z=48 (1-voxel width)
        base_trace[48] += 2.8

        res = shell.forward(base_trace)
        tier1 = res.tier1_outlier_indices.tolist()

        # Both spikes must be captured in Tier-1 tokens
        assert 12 in tier1, f"Thin ELM filament at z=12 was missed in Tier-1: {tier1}"
        assert 48 in tier1, f"Resonant current sheet at z=48 was missed in Tier-1: {tier1}"

        # Recovery rate must be 1.0000 (100.0% clean spike recovery)
        assert res.beltrami_spike_recovery >= 0.999, f"Spike recovery was below 100%: {res.beltrami_spike_recovery}"

        # Standard theorem verification test must pass
        thm_res = BeltramiChordShell.verify_theorem_2_5_spike_recovery()
        assert thm_res["passed"] is True
        assert thm_res["spike_recovery"] == 1.0000


# =============================================================================
# 3. Zero Target Leakage: Empirical Proof Across All 7 Baselines
# =============================================================================

class TestZeroTargetLeakage:
    """Empirically prove that baselines have zero access or leakage to held-out test chords."""

    def test_unseen_chords_corruption_invariance(self):
        """
        Corrupt held-out unseen chords [4, 5, 6, 7] with random values.
        Assert that the reconstructed predictions for ALL 7 methods are bit-identical (diff = 0.00e+0).
        """
        scenarios = build_all_scenarios()
        sc_clean = scenarios["01_issan_nonsym_solovev_tau0"]
        sc_corrupt = copy.deepcopy(sc_clean)

        # Corrupt held-out unseen chords
        rng = np.random.default_rng(999)
        for i in [4, 5, 6, 7]:
            noisy_curve = rng.standard_normal(len(sc_corrupt.chords[i].curves["PSI"])) * 100.0
            sc_corrupt.chords[i].curves["PSI"] = noisy_curve

        res_clean = run_mode_b_evaluation(sc_clean, snr_db=20.0, chunk_z=16)
        res_corrupt = run_mode_b_evaluation(sc_corrupt, snr_db=20.0, chunk_z=16)

        methods_clean = res_clean["methods"]
        methods_corrupt = res_corrupt["methods"]

        expected_methods = [
            "NaiveBox",
            "TunedAnisotropicKriging",
            "SymmetricFourierVMECBaseline",
            "ShellPartitionedKriging",
            "CerfonFreidbergEquilibrium",
            "MinimumFisherTomography",
            "BlindBMHD",
        ]

        for m in expected_methods:
            # LOCO RMSE on training chords must be identical
            loco_diff = abs(methods_clean[m]["loco_rmse"] - methods_corrupt[m]["loco_rmse"])
            # Whole torus RMSE must be identical
            torus_diff = abs(methods_clean[m]["whole_torus_psi_rmse"] - methods_corrupt[m]["whole_torus_psi_rmse"])
            # Edge gradient fidelity must be identical
            edge_diff = abs(methods_clean[m]["bounded_edge_gradient_fidelity"] - methods_corrupt[m]["bounded_edge_gradient_fidelity"])

            assert loco_diff == 0.0, f"Method {m} showed LOCO target leakage (diff={loco_diff})"
            assert torus_diff == 0.0, f"Method {m} showed whole-torus target leakage (diff={torus_diff})"
            assert edge_diff == 0.0, f"Method {m} showed edge fidelity target leakage (diff={edge_diff})"


# =============================================================================
# 4. Mode B Sparse-Chord Reconstruction: Sweeps Across SNR and Jitter
# =============================================================================

class TestModeBSweeps:
    """Stress-test Mode B reconstruction across SNR in {inf, 20, 10, 5 dB} and jitter in {0, 1, 2, 3 voxels}."""

    @pytest.mark.parametrize("snr_level", [float("inf"), 20.0, 10.0, 5.0])
    def test_mode_b_snr_levels(self, snr_level):
        scenarios = build_all_scenarios()
        sc = scenarios["03_spec_magnetic_island_chain"]

        res = run_mode_b_evaluation(sc, snr_db=snr_level, jitter_voxels=0, chunk_z=16)
        methods = res["methods"]

        for mname, metrics in methods.items():
            assert np.isfinite(metrics["loco_rmse"]), f"{mname} loco_rmse is non-finite at SNR={snr_level}"
            assert np.isfinite(metrics["unseen_3d_chords_rmse"]), f"{mname} unseen_rmse is non-finite at SNR={snr_level}"
            assert np.isfinite(metrics["whole_torus_psi_rmse"]), f"{mname} torus_rmse is non-finite at SNR={snr_level}"
            assert 0.0 <= metrics["bounded_edge_gradient_fidelity"] <= 1.0

        # Memory ceiling must be satisfied
        assert res["performance"]["heap_ceiling_passed"] is True
        assert res["performance"]["peak_heap_mb"] < 35.0

    @pytest.mark.parametrize("jitter_v", [0, 1, 2, 3])
    def test_mode_b_separatrix_jitter_levels(self, jitter_v):
        scenarios = build_all_scenarios()
        sc = scenarios["03_spec_magnetic_island_chain"]

        res = run_mode_b_evaluation(sc, snr_db=20.0, jitter_voxels=jitter_v, chunk_z=16)
        methods = res["methods"]

        for mname, metrics in methods.items():
            assert np.isfinite(metrics["whole_torus_psi_rmse"])
            assert 0.0 <= metrics["bounded_edge_gradient_fidelity"] <= 1.0


# =============================================================================
# 5. Out-of-Core Field-Cube Memmap Streaming: 64 MB Volume & chunk_ky Windowing
# =============================================================================

class TestOutOfCoreStreamingStress:
    """Adversarial stress-testing of out-of-core memmap field volume streaming."""

    def test_streaming_large_volume_and_chunk_ky_windowing(self):
        """
        Create a 64 MB raw float32 volume (100, 400, 400) on disk.
        Stream with chunk_ky in {4, 8, 16, 24}.
        Assert that peak heap strictly stays < 35 MB and output grid shape is (32, 24, 24).
        """
        temp_dir = tempfile.mkdtemp()
        vol_path = os.path.join(temp_dir, "large_64mb_volume.dat")

        try:
            # Create 100 x 400 x 400 float32 volume (16,000,000 floats * 4 = 64,000,000 bytes)
            raw_mm = np.memmap(vol_path, dtype=np.float32, mode="w+", shape=(100, 400, 400))
            # Inject structured field with high-gradient core
            z_plane = np.linspace(0.0, 10.0, 100, dtype=np.float32)[:, None, None]
            raw_mm[:] = z_plane
            raw_mm.flush()
            del raw_mm

            # Test multiple chunk_ky configurations
            for chunk_ky in [4, 8, 16, 24]:
                streamer = stream_field_volume(
                    vol_path,
                    chunk_ky=chunk_ky,
                    target_grid=(32, 24, 24),
                    max_heap_mb=35.0,
                )
                slabs = list(streamer)

                assert len(slabs) > 0, f"No slabs generated for chunk_ky={chunk_ky}"

                total_y = sum(s.y_end - s.y_start for s in slabs)
                assert total_y == 24, f"Total streamed Y dimension {total_y} != 24"

                # Check peak heap of every slab
                for s in slabs:
                    assert s.peak_heap_mb < 35.0, f"Slab {s.slab_index} exceeded heap ceiling: {s.peak_heap_mb:.2f} MB"
                    assert s.downsampled_cube.shape[0] == 32
                    assert s.downsampled_cube.shape[2] == 24

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 6. Baseline OOP Class Interface Audit (Catching Broken Class Methods)
# =============================================================================

class TestBaselineOOPClassesAudit:
    """Audit the Object-Oriented baseline class interfaces for runtime call correctness."""

    def test_baseline_classes_fit_and_predict(self):
        scenarios = build_all_scenarios()
        sc = scenarios["01_issan_nonsym_solovev_tau0"]
        chords = sc.chords

        # 1. TunedAnisotropicKriging: Works
        tak = TunedAnisotropicKriging().fit(chords, train_indices=[0, 1, 2, 3])
        p_tak = tak.predict_3d()
        assert p_tak.shape == (32, 24, 24)

        # 2. SymmetricFourierVMECBaseline: Works
        vmec = SymmetricFourierVMECBaseline(base_predictor=tak).fit(chords, train_indices=[0, 1, 2, 3])
        p_vmec = vmec.predict_3d()
        assert p_vmec.shape == (32, 24, 24)

        # 3. BlindBMHD: Works
        bmhd = BlindBMHD().fit(chords, train_indices=[0, 1, 2, 3])
        p_bmhd = bmhd.predict_3d()
        assert p_bmhd.shape == (32, 24, 24)

        # 4. NaiveBox: Works
        nb = NaiveBox().fit(chords, train_indices=[0, 1, 2, 3])
        p_nb = nb.predict_3d()
        assert p_nb.shape == (32, 24, 24)

        # 5. ShellPartitionedKriging: Works
        spk = ShellPartitionedKriging().fit(chords, train_indices=[0, 1, 2, 3])
        p_spk = spk.predict_3d()
        assert p_spk.shape == (32, 24, 24)

        # 6. CerfonFreidbergEquilibrium: Works
        cfe = CerfonFreidbergEquilibrium().fit(chords, train_indices=[0, 1, 2, 3])
        p_cfe = cfe.predict_3d()
        assert p_cfe.shape == (32, 24, 24)

        # 7. MinimumFisherTomography: Works
        mft = MinimumFisherTomography().fit(chords, train_indices=[0, 1, 2, 3])
        p_mft = mft.predict_3d()
        assert p_mft.shape == (32, 24, 24)
