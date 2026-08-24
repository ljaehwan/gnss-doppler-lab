"""Frozen, result-independent contracts for SPLITCLOCK Stage-0A R1."""

from __future__ import annotations

BASE_SHA = "04ba01f2f01d08882e069917bfe062c93abe1585"
BRANCH = "research/splitclock-stage0a-r1-contract-model-repair"
R0_ARTIFACT = "artifacts/splitclock_stage0a_clean_identifiability"
R1_ARTIFACT = "artifacts/splitclock_stage0a_r1_contract_model_repair"
R0_DESIGN_SHA = "8b7de5722037c1269989f2ee8cbff89ac42e3773"
R0_FINAL_SHA = BASE_SHA
DESIGN_COMMIT_MESSAGE = "SPLITCLOCK_STAGE0A_R1_DESIGN_FREEZE"
IMPLEMENTATION_COMMIT_MESSAGE = "SPLITCLOCK_STAGE0A_R1_IMPLEMENTATION_FREEZE"

SPEED_OF_LIGHT_MPS = 299_792_458.0
GALILEO_E1_HZ = 1_575_420_000.0
GALILEO_E1_WAVELENGTH_M = SPEED_OF_LIGHT_MPS / GALILEO_E1_HZ
WINDOW_EPOCHS = 10
FIT_EPOCHS = 7
HOLDOUT_EPOCHS = 3
STUDENT_T_DF = 4.0
MIN_CLUSTER_MASS = 2.0
SEED = 20250824

ALLOWED_VERDICTS = (
    "STOP_SPLITCLOCK_GEOMETRY_UNAVAILABLE",
    "STOP_SPLITCLOCK_CLEAN_PANEL_UNSUPPORTED",
    "INCONCLUSIVE_SPLITCLOCK_SIGN_UNIT_CADENCE",
    "INCONCLUSIVE_SPLITCLOCK_IMPLEMENTATION_CONTRACT",
    "INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE",
    "NO_GO_SPLITCLOCK_CLEAN_FALSE_ALARMS",
    "NO_GO_SPLITCLOCK_NEGATIVE_CONTROLS",
    "NO_GO_SPLITCLOCK_SYNTHETIC_IDENTIFIABILITY",
    "READY_FOR_SPLITCLOCK_ATTACK_FREEZE",
)

FORBIDDEN_MODEL_INPUTS = (
    "prn_number",
    "filename",
    "scenario_label",
    "cn0_db_hz",
    "lock",
    "prn_count",
)


def frozen_design() -> dict:
    return {
        "schema": "gnss-doppler-lab.splitclock-stage0a-r1-design.v1",
        "status": "PRE_CLEAN_SCORE_DESIGN_FREEZE",
        "base_sha": BASE_SHA,
        "branch": BRANCH,
        "r0_binding": {
            "final_sha": R0_FINAL_SHA,
            "design_freeze_sha": R0_DESIGN_SHA,
            "artifact": R0_ARTIFACT,
            "r0_artifact_immutable": True,
        },
        "scope": {
            "allowed": ["Tuni Galileo C-1 clean", "Tuni Galileo C-3 clean", "R0-verified QSET V3 clean outputs", "synthetic-on-clean observables"],
            "forbidden_access": ["SS-1", "SS-3", "SS-5", "SS-11", "SS-12", "SS-13", "SS-33", "TEXBAT attack", "OAKBAT attack", "Jammertest raw"],
            "attack_stats_hashes_opens_mmaps_bytes": 0,
        },
        "observable_contract": {
            "alignment": "receiver-relative common 1 Hz RINEX epochs",
            "model_modalities": ["geometry_corrected_pseudorange_residual_m", "geometry_corrected_doppler_residual_mps", "geometry_corrected_carrier_increment_m"],
            "metadata_only": ["satellite_id", "valid_mask", "cycle_slip", "reacquisition", "receiver_relative_epoch"],
            "forbidden_model_inputs": list(FORBIDDEN_MODEL_INPUTS),
            "carrier": "carrier_increment_m = +lambda_E1 * delta(L1B_cycles)",
            "doppler": "range_rate_mps = -lambda_E1 * doppler_hz",
            "sign_revalidation": ["carrier_increment_vs_pseudorange_increment", "carrier_increment_vs_doppler_integral", "pseudorange_increment_vs_doppler_integral"],
            "sign_pass": "all three correlations >= 0.90 independently in C-1 and C-3",
            "cadence": {
                "acquisition_coherent_integration_ms": 8,
                "native_trace_cadence_source": "parse every native TRACE header coherent_integration_s; cross-check consecutive raw_interval_start_sample deltas and frozen output manifest",
                "native_trace_cadence_must_not_be_hardcoded": True,
                "semantic_separation_required": True,
            },
        },
        "geometry": {
            "reference_frame": "WGS84 ECEF meters",
            "receiver_position": "GNSS-SDR GPX latitude/longitude/ellipsoidal-height converted to WGS84 ECEF and linearly interpolated at RINEX epochs; maximum alignment error 1.0 s",
            "broadcast_ephemeris": "Galileo RINEX 3 navigation, closest healthy record for the same satellite with |t-Toe|<=7200 s",
            "orbit": "Galileo broadcast Kepler model; solve eccentric anomaly to 1e-13 rad; harmonic corrections; velocity by centered 0.5 s finite difference",
            "satellite_clock": "af0+af1*dt+af2*dt^2+F*e*sqrt(A)*sin(E), I/NAV clock model; record applied=True",
            "signal_bias": "no additional BGD for E1B I/NAV clock model; record applied=False and preserve as per-PRN constant nuisance removed by fit-only centering",
            "earth_rotation": "rotate transmit-time satellite ECEF around z by omega_E*rho/c (Sagnac), two fixed-point iterations",
            "apparent_range_m": "sagnac_geometric_range_m - c*satellite_clock_s",
            "apparent_range_rate_mps": "centered 0.5 s derivative of apparent range with receiver-position linear interpolation",
            "residuals": {
                "code": "C1B_m - apparent_range_m",
                "carrier_increment": "+lambda_E1*delta(L1B_cycles) - delta(apparent_range_m)",
                "doppler": "-lambda_E1*D1B_hz - apparent_range_rate_mps",
            },
            "fit_only_nuisance": "subtract each eligible PRN's fit-interval median code residual and median dynamic residual; never use heldout values",
            "required_validation": ["finite_satellite_position_velocity", "receiver_position_source_and_crs", "satellite_clock", "Sagnac", "alignment", "meter_units", "meter_per_second_units", "per_scenario_residual_statistics"],
            "failure_verdict": "STOP_SPLITCLOCK_GEOMETRY_UNAVAILABLE",
        },
        "model": {
            "state": "x_k,t=[clock_range_bias_m, clock_drift_mps]",
            "transition": "x[t+1]=[[1,dt],[0,1]]@x[t]+process_noise",
            "observation_rows": {"code": [1.0, 0.0], "doppler": [0.0, 1.0], "carrier_increment": [0.0, 1.0]},
            "K1": "one shared clock state path for every eligible PRN",
            "K2": "two state paths plus one fixed soft membership probability per eligible PRN for the whole window",
            "estimator": "Student-t IRLS MAP state-space smoothing nested in deterministic soft EM",
            "student_t_df": STUDENT_T_DF,
            "shared_contract": "K1/K2 share fit-only modality scales, censoring, valid mask, eligible PRNs, process covariance, and heldout observations",
            "initialization": "three deterministic fit-only restarts: neutral 0.5 membership and +/- SVD fit-residual summaries; SVD/hard grouping is initialization only",
            "restart_selection": "maximum fit Student-t likelihood only; heldout inaccessible until selected model is frozen",
            "soft_assignment": "E-step uses per-PRN fit likelihood and equal prior; clip probabilities to [1e-6,1-1e-6]",
            "effective_mass": "sum(pi)>=2 and sum(1-pi)>=2",
            "canonicalization": "ascending fitted clock-range path mean over fit epochs; never satellite ID or synthetic truth",
            "em_iterations": 30,
            "irls_iterations": 12,
            "convergence_tolerance": 1e-8,
            "hard_clustering_final_forbidden": True,
        },
        "dynamic_panel": {
            "representation": "union-of-PRNs tensor plus per-epoch/per-PRN/per-modality valid mask",
            "window_epoch_gate": "at every epoch at least five PRNs have at least code plus one dynamic modality",
            "eligible_prn_fit_rule": "at least four fit epochs with code and at least four fit epochs with Doppler or carrier; heldout-only new PRNs excluded from both K1 and K2 heldout likelihood",
            "same_observation_mask_assertion": True,
            "missing_values": "evaluate each finite modality independently; one NaN never rejects a complete window",
            "drop_add": "allowed when epoch gate and fit-eligibility rules remain satisfied",
        },
        "score": {
            "window_epochs": WINDOW_EPOCHS,
            "fit_epochs": FIT_EPOCHS,
            "heldout_epochs": HOLDOUT_EPOCHS,
            "raw_gain": "heldout_loglik_K2-heldout_loglik_K1",
            "delta_p": "2 + M_eligible (one additional two-state initial condition plus one soft-membership logit per eligible PRN)",
            "penalty": "0.5*delta_p*log(n_valid_heldout)",
            "primary": "raw_gain-penalty",
            "diagnostic_normalized": "primary_score/n_valid_heldout",
            "heldout_policy": "prediction only from fit terminal states and fixed assignments; no heldout filtering, fitting, initialization, restart, centering, or scale estimation",
        },
        "split": {
            "stable_start": "first epoch of first continuous M>=5 run lasting at least 60 s",
            "initial_guard_epochs": 10,
            "C-1": "after initial guard, first at most 80 chronological epochs for scale/process/state fitting only",
            "C-3": "after initial guard let N be usable length; calibration=floor((N-10)/2), then 10 epoch guard, then untouched remainder holdout",
            "randomization": False,
            "threshold": "q99 method='higher' over calibration non-overlapping 10-epoch block maxima; persistence=3 consecutive score exceedances",
            "limited_duration_rule": "if independent calibration blocks <20, never claim publication-grade FPR; READY must state PROVISIONAL_LIMITED_DURATION",
            "zero_false_alarm_ci": "two-sided Clopper-Pearson 95% upper bound reported for holdout persistent-decision count",
        },
        "synthetic": {
            "subset_sizes": {"primary": [3, 5], "diagnostic": [1]},
            "subset_selection": "SHA256 ordering of audit-only satellite IDs with fixed seed; IDs never enter model",
            "onsets": "three frozen valid chronological onsets at 0%, 33%, 66% of allowable support, each with >=30 post-onset epochs",
            "trajectories": {
                "mild_ramp": {"d0_m": 0.0, "v_mps": 0.1, "a_mps2": 0.0},
                "moderate_ramp": {"d0_m": 0.0, "v_mps": 0.5, "a_mps2": 0.0},
                "accelerating": {"d0_m": 0.0, "v_mps": 0.05, "a_mps2": 0.02},
                "delayed_ramp": {"d0_m": 10.0, "v_mps": 0.1, "a_mps2": 0.0},
            },
            "injection": "code += d0*I(t=onset)+v*t+0.5*a*t^2 accumulated trajectory; carrier increment += delta(trajectory) including d0 once at onset transition; Doppler residual += v+a*t",
            "d0_duplicate_addition_forbidden": True,
        },
        "controls": {
            "negative": ["all_PRN_coherent_clock_jump", "all_PRN_coherent_clock_ramp", "common_oscillator_drift", "receiver_motion_perturbation", "one_PRN_multipath", "one_PRN_cycle_slip", "reacquisition", "PRN_drop", "PRN_add", "temporary_gap", "ephemeris_update", "independent_small_PRN_biases", "CN0_noise_change"],
            "boundary": "all-PRN coherent false path is structurally K1 and must not be tuned into a positive",
            "ablations": ["A0_K1_residual_energy_CUSUM", "A1_K2_hard_assignment", "A2_no_temporal_persistence", "A3_pseudorange_only", "A4_Doppler_only", "A5_carrier_only", "A6_full_soft_state_space"],
            "destructions": ["epoch_membership_shuffle", "per_PRN_time_shuffle", "modality_coherence_break", "time_varying_subset_membership", "random_PRN_rename"],
        },
        "primary_gates": {
            "support": {"continuous_m_ge_5_s": 60, "finite_coverage": 0.95, "alignment_error_epochs": 1},
            "clean": {"persistent_false_alarms": 0, "epoch_fpr_max": 0.01, "absolute_score_CN0_correlation_max": 0.3, "absolute_score_PRN_count_correlation_max": 0.3, "drop_add_persistent_false_alarms": 0},
            "synthetic": {"moderate_detection_rate_3_5_min": 0.90, "mild_detection_rate_3_5_min": 0.70, "accelerating_detection_rate_min": 0.80, "median_delay_s_max": 10, "localization_f1_min": 0.80},
            "negative": {"all_PRN_jump_ramp_drift_persistent_false_alarms": 0, "cycle_slip_reacquisition_persistent_false_alarms": 0, "other_persistent_fpr_max": 0.02},
            "contribution": {"A6_minus_A0_AUROC_min": 0.05, "destruction_advantage_reduction_min": 0.30, "PRN_permutation_score_difference_max": 1e-10},
        },
        "verdicts": list(ALLOWED_VERDICTS),
        "verdict_precedence": ["geometry", "panel", "sign_unit_cadence", "implementation", "execution_provenance", "clean", "negative", "synthetic", "ready"],
        "implementation_freeze_rule": "only data-independent synthetic unit tests before implementation freeze; exactly one C-1/C-3 execution after push",
        "seeds": {"model": SEED, "synthetic_subset": 20250826, "bootstrap": 20250825},
    }
