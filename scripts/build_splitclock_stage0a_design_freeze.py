#!/usr/bin/env python3
"""Create the label/physics-only SPLITCLOCK design freeze before clean IQ access."""

from __future__ import annotations

import argparse
from pathlib import Path

from gnss_doppler_lab.splitclock_stage0a import (
    BASE_SHA,
    BRANCH,
    EPOCH_RATE_HZ,
    FIT_FRACTION,
    GALILEO_E1_WAVELENGTH_M,
    REFERENCE_ADAPTER_BLOB,
    REFERENCE_CONFIG_BLOB,
    REFERENCE_RECEIVER_SHA256,
    REFERENCE_RECEIVER_SIZE,
    REFERENCE_RUNNER_BLOB,
    REFERENCE_SHA,
    REQUIRED_OBSERVABLES,
    SEED,
    STUDENT_T_DF,
    VERDICTS,
    WINDOW_SECONDS,
    sha256_file,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receiver-binary", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    receiver = args.receiver_binary.resolve()
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "figures").mkdir(exist_ok=True)
    receiver_size = receiver.stat().st_size
    receiver_sha = sha256_file(receiver)
    if receiver_size != REFERENCE_RECEIVER_SIZE or receiver_sha != REFERENCE_RECEIVER_SHA256:
        raise SystemExit("REFERENCE_RECEIVER_BINDING_FAILURE")

    write_json(artifact / "source_binding.json", {
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "reference_receiver_support_sha": REFERENCE_SHA,
        "status": "PASS",
    })
    write_json(artifact / "receiver_source_binding.json", {
        "reference_only": True,
        "merge_performed": False,
        "receiver_binary": {"path_from_cli": str(receiver), "size_bytes": receiver_size, "sha256": receiver_sha},
        "reference_git_blobs": {
            "config_template": REFERENCE_CONFIG_BLOB,
            "runner": REFERENCE_RUNNER_BLOB,
            "adapter": REFERENCE_ADAPTER_BLOB,
        },
        "v3_parameters": {"concurrent_acquisition_channels": 12, "coherent_integration_ms": 8, "pfa": "0.00001"},
        "qset_feature_threshold_score_reused": False,
        "status": "PASS",
    })
    observable = {
        "required": list(REQUIRED_OBSERVABLES),
        "model_feature_exclusions": [
            "PRN_identity", "filename", "scenario", "absolute_sample_index",
            "C/N0", "lock", "PRN_count",
        ],
        "audit_only": ["decoded_satellite_id", "C/N0", "lock", "PRN_count"],
        "galileo_e1_wavelength_m": GALILEO_E1_WAVELENGTH_M,
        "pseudorange_unit": "meter",
        "carrier_source_unit": "radian_or_cycle",
        "carrier_increment_unit": "meter",
        "doppler_source_unit": "hertz",
        "range_rate_unit": "meter_per_second",
        "doppler_to_range_rate": "range_rate_mps = - wavelength_m * doppler_hz",
        "carrier_increment": "delta_range_m = - wavelength_m * delta(cycle-consistent accumulated_phase_rad)/(2*pi)",
        "time_basis": "receiver-relative 1 Hz epoch after alignment",
        "raw_tracking_cadence_ms": 8,
        "epoch_rate_hz": EPOCH_RATE_HZ,
        "alignment_tolerance_epochs": 1,
        "real_sign_checks_required_before_score": True,
        "unavailable_policy": "do not manufacture approximations; STOP_SPLITCLOCK_OBSERVABLES_UNAVAILABLE",
    }
    write_json(artifact / "observable_contract.json", observable)
    write_json(artifact / "design_freeze.json", {
        "status": "PRE_RAW_DESIGN_FREEZE",
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "clean_sources": {
            "C-1": {"role": "development_and_state_model_fitting", "size_bytes": 29_999_832_000, "md5": "4ff0e86938792bf3150c30d5f1481917"},
            "C-3": {"role": "calibration_then_chronological_holdout", "size_bytes": 29_999_832_000, "md5": "1b7c99c754faec3c8fa625849ef70014"},
        },
        "clean_format": {"endianness": "big", "component_dtype": "signed_int16", "interleaving": "I,Q", "bytes_per_complex_sample": 4, "sample_rate_hz": 50_000_000, "nominal_duration_s": 149.99916},
        "forbidden_access": ["SS-1", "SS-3", "SS-5", "SS-11", "SS-12", "SS-13", "SS-33", "TEXBAT_attack", "OAKBAT_attack", "Jammertest_raw"],
        "receiver": {"reference_sha": REFERENCE_SHA, "configuration": "V3", "concurrent_acquisition_channels": 12, "coherent_integration_ms": 8, "pfa": "0.00001"},
        "observable_contract": observable,
        "support_gates": {"minimum_prns_each_clean": 5, "minimum_continuous_m_ge_5_seconds": 60, "minimum_required_observable_finite_coverage": 0.95, "maximum_alignment_error_epochs": 1},
        "model": {
            "primary": "non-neural robust K=1 vs K=2 state-space baseline",
            "observation_likelihood": "Student-t",
            "student_t_df": STUDENT_T_DF,
            "state_evolution": "shared constant drift",
            "window_seconds": WINDOW_SECONDS,
            "fit_fraction": FIT_FRACTION,
            "heldout_fraction": 1.0 - FIT_FRACTION,
            "assignment": "residual-history only, fixed throughout window as persistence prior",
            "minimum_prns_per_cluster": 2,
            "label_canonicalization": "ascending clock-path fitted range intercept; no PRN or truth label",
            "score": "heldout_loglik_K2 - heldout_loglik_K1 - 0.5*delta_parameter_count*log(valid_observation_count)",
            "shared_noise_scale": True,
        },
        "split": {
            "C-1": "first stable M>=5 interval after acquisition plus 10 s guard; maximum 80 s; never score-selected",
            "C-3": "after acquisition plus 10 s guard: first usable chronological half calibration, 10 s guard, remainder untouched holdout",
        },
        "threshold": {"source": "C-3 calibration non-overlapping 10 s block maxima", "quantile": 0.99, "method": "higher", "persistence_consecutive_exceedances": 3, "positive_control_tuning": False},
        "synthetic_positive": {
            "subset_sizes": {"primary": [3, 5], "diagnostic": [1]},
            "subset_order": "SHA-256 frozen order, never PRN identity order",
            "onset_count": 3,
            "duration_minimum_seconds": 30,
            "grids": {
                "mild_ramp": {"d0_m": 0, "v_mps": 0.10, "a_mps2": 0},
                "moderate_ramp": {"d0_m": 0, "v_mps": 0.50, "a_mps2": 0},
                "accelerating": {"d0_m": 0, "v_mps": 0.05, "a_mps2": 0.02},
                "delayed_offset_plus_ramp": {"d0_m": 10, "v_mps": 0.10, "a_mps2": 0},
            },
            "coherent_modalities": ["pseudorange", "Doppler/range-rate", "carrier increment"],
        },
        "negative_controls": [f"N{number}" for number in range(1, 13)],
        "boundary_controls": ["B1_all_PRN_coherent_false_path", "B2_below_noise_K2"],
        "ablations": ["A0_K1_energy_CUSUM", "A1_no_temporal_persistence", "A2_pseudorange_only", "A3_Doppler_only", "A4_carrier_only", "A5_full_primary"],
        "destructions": ["D1_epochwise_assignment_shuffle", "D2_temporal_membership_destruction_with_modal_coherence"],
        "bootstrap": {"unit": "independent_10_second_parent_block", "replicates": 2000},
        "seeds": {"global": SEED, "bootstrap": SEED + 1, "subset_order": SEED + 2},
        "verdicts": sorted(VERDICTS),
        "verdict_precedence": ["observable_unavailable", "panel_unsupported", "execution_or_provenance", "clean_false_alarms", "negative_controls", "synthetic_identifiability", "ready"],
        "raw_feature_bytes_read": 0,
        "score_operations": 0,
        "attack_access": {"stats": 0, "hashes": 0, "mmaps": 0, "opens": 0, "bytes_read": 0},
    })
    write_json(artifact / "access_audit.json", {
        "phase": "DESIGN_FREEZE",
        "clean_raw": {"stats": 0, "hashes": 0, "mmaps": 0, "opens": 0, "bytes_read": 0},
        "attack": {"stats": 0, "hashes": 0, "mmaps": 0, "opens": 0, "bytes_read": 0},
        "jammertest_raw": {"stats": 0, "hashes": 0, "mmaps": 0, "opens": 0, "bytes_read": 0},
        "status": "PASS",
    })
    (artifact / "README.md").write_text(
        "# SPLITCLOCK-GNSS Stage-0A clean identifiability\n\n"
        "Status: pre-raw design freeze. No clean raw feature or score and no attack path access has occurred.\n",
        encoding="utf-8",
    )
    print("SPLITCLOCK_STAGE0A_DESIGN_FREEZE_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
