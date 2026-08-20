#!/usr/bin/env python3
"""CORA-GNSS Stage-0 freeze and evaluation entry point.

The ``freeze`` command is deliberately metadata-only: it may stat attack files
but never opens them.  The ``evaluate`` command refuses to run until the exact
freeze commit has been pushed and recorded in configuration_freeze.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/cora_stage0_cross_prn_common_origin"
BASE_SHA = "05c1f9f33661ef20584a1db4cdd8fdf7c495b419"
BRANCH = "research/cora-stage0-cross-prn-common-origin"
MAIN_SHA_AT_FREEZE = "461eb4dc7bb794e719295daf028f6811658ba37f"
FREEZE_MESSAGE = "Configuration frozen before this CORA evaluation."

DATASETS = {
    "oakbat_cleanstatic": {
        "role": "clean_model_calibration_holdout",
        "family": "OAK_CLEAN",
        "sample_rate_hz": 5_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_cleanstatic",
    },
    "oakbat_os3": {
        "role": "core_attack", "family": "OAK_OS3_OS4", "onset_s": 120.0,
        "sample_rate_hz": 5_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os3.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_os3",
    },
    "oakbat_os4": {
        "role": "core_attack", "family": "OAK_OS3_OS4", "onset_s": 120.0,
        "sample_rate_hz": 5_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os4.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_os4",
    },
    "texbat_cleanstatic": {
        "role": "clean_model_calibration_holdout",
        "family": "TEX_CLEAN",
        "sample_rate_hz": 25_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_cleanstatic",
    },
    "texbat_ds1": {
        "role": "core_attack", "family": "TEX_DS1", "onset_s": 125.0,
        "sample_rate_hz": 25_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds1.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds1-complex9/raw",
        "trace_adapter": "legacy_complex9_requires_authenticated_absolute_sample_adapter",
    },
    "texbat_ds3": {
        "role": "core_attack", "family": "TEX_DS3", "onset_s": 118.9, "pull_off_s": 195.0,
        "sample_rate_hz": 25_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_ds3",
    },
    "texbat_ds7": {
        "role": "core_attack", "family": "TEX_DS7_DS8", "onset_s": 110.0, "time_push_s": 150.0,
        "sample_rate_hz": 25_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_ds7",
    },
    "texbat_ds8": {
        "role": "core_attack", "family": "TEX_DS7_DS8", "onset_s": 110.0, "time_push_s": 150.0,
        "sample_rate_hz": 25_000_000,
        "raw": "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds8.bin",
        "trace": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/receiver/raw",
        "trace_adapter": "legacy_complex9_requires_authenticated_absolute_sample_adapter",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def dump_json(name: str, value: Any) -> None:
    path = ART / name
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def frozen_config() -> dict[str, Any]:
    return {
        "schema": "gnss-doppler-lab.cora-stage0-config.v1",
        "model": {"short_name": "CORA-GNSS", "full_name": "Common-Origin Residual Attribution for GNSS Spoofing Detection"},
        "configuration_statement": FREEZE_MESSAGE,
        "primary_alpha": 0,
        "minimum_concurrent_prns": 4,
        "raw_replica": {
            "nominal_components": ["GPS_L1_CA_code", "NAV_sign", "receiver_code_phase", "receiver_carrier_phase", "receiver_Doppler"],
            "removal": "per-epoch complex least-squares rank-1 nominal replica",
            "delay_grid_chips": [-0.25, 0.0, 0.25],
            "doppler_grid_hz": [-25.0, 0.0, 25.0],
            "residual_token": "3x3 complex raw recorrelation followed by orthogonal removal of constant/delay/Doppler tangents and L2 quotient",
            "allowed_reuse": ["CINDER raw-source binding", "absolute sample mapping", "raw recorrelation reader"],
            "prohibited_reuse": ["CINDER C4 features", "CINDER matching", "CINDER permutation", "CINDER verdict code"],
        },
        "analysis_sampling": {
            "score_window_s": 2.0,
            "windows_per_bootstrap_block": 5,
            "bootstrap_block_s": 10.0,
            "epochs_per_score_window": 32,
            "epoch_duration_ms": 1,
            "epoch_offsets_s": [round((i + 0.5) * 2.0 / 32, 9) for i in range(32)],
            "window_anchor_s": 1.0,
        },
        "clean_split": {
            "anchor_s": 31.0,
            "train_10s_blocks": list(range(0, 18)),
            "guard_10s_blocks": [18],
            "calibration_10s_blocks": list(range(19, 29)),
            "second_guard_10s_blocks": [29],
            "holdout_10s_blocks": list(range(30, 40)),
            "train_interval_s": [31.0, 211.0],
            "calibration_interval_s": [221.0, 321.0],
            "holdout_interval_s": [331.0, 431.0],
            "conditioner_cross_fit": "even/odd 10-second train-block folds; final model refit on all train blocks only",
        },
        "conditioner": {
            "fit_scope": "clean train only, separately for OAK and TEX, coefficients shared across PRNs",
            "context": ["C/N0", "raw_RMS_AGC_proxy", "residual_RMS", "token_prequotient_norm", "receiver_Doppler_increment", "concurrent_PRN_count"],
            "ridge": 0.001,
            "covariance_shrinkage": 0.2,
            "forbidden_context": ["PRN_identity", "absolute_sample_index", "absolute_time", "scenario_label", "attack_status"],
        },
        "primary_statistic": {
            "equation": "cum(x,x*,y,y*)=E|x|^2|y|^2-E|x|^2E|y|^2-|E[xy*]|^2-|E[xy]|^2",
            "estimator": "unbiased multivariate fourth k-statistic",
            "matrix_diagonal": 0,
            "H0": "diagonal/no off-diagonal common origin",
            "H1": "diagonal plus one symmetric rank-1 off-diagonal component",
            "score": "2*log-likelihood improvement minus BIC complexity penalty",
            "bic_parameters": "number of concurrent PRNs",
            "prn_policy": "sorted concurrent set; statistic equivariant to permutation and safe for missing PRNs",
        },
        "thresholds": {
            "primary": "empirical OAK/TEX clean calibration q99 computed from 50 non-overlapping 2-second scores",
            "diagnostics": ["q99.5", "highest threshold attaining <=1% calibration FPR"],
            "no_attack_calibration": True,
        },
        "synthetic_controls": {
            "seed": 20260820,
            "fit_scope": "clean train support before any attack read",
            "shared": "one non-Gaussian complex latent with per-PRN complex loadings, marginal-matched to independent control",
            "independent": "independent per-PRN non-Gaussian latents with matched per-PRN marginal amplitude and PSD",
            "receiver_nuisance": ["common_gain", "common_phase", "clock", "AGC", "AWGN"],
            "significance": "paired 10-second block bootstrap 95% CI of shared-minus-independent excludes zero",
        },
        "relations": {
            "temporal_desynchronization_offsets_s": list(range(1, 11)),
            "cross_prn_time_block_reassignment": True,
            "phase_norm_psd_surrogate": True,
            "required_attack_score_drop_fraction": 0.25,
            "required_paired_block_ci_lower": 0.0,
            "clean_shuffle_requirement": "no significant persistent increase",
        },
        "baselines": {
            "A0": "residual token norm only", "A1": "second-order cross-PRN covariance",
            "A2": "marginal fourth order only", "A3": "CORA without nuisance conditioning",
            "A4": "independent per-PRN latent alternative", "Full": "frozen CORA",
            "diagnostics": ["HSIC", "power", "C/N0", "Doppler"],
            "B0_fixed9": "UNAVAILABLE unless an actual same-raw-support rerun is executed",
        },
        "bootstrap": {"seed": 20260821, "replicates": 2000, "unit": "10-second block", "confidence": 0.95},
        "go_gates": {
            "clean_holdout_q99_fpr_max": 0.02,
            "worst_attack_preonset_fpr_max": 0.05,
            "shared_synthetic_gt_independent_both_domains": True,
            "receiver_nuisance_no_persistent_alarm": True,
            "tex_families_passing_min": 2,
            "family_pauc_min": 0.8,
            "family_detection_rate_min": 0.70,
            "oak_os3_os4_same_metrics": True,
            "full_must_beat": ["A0", "A2", "A4"],
            "destruction_drop_fraction_min": 0.25,
            "destruction_ci_excludes_zero": True,
            "leave_one_prn_out_stable": True,
            "shortcut_audit_pass": True,
            "B0_condition": "actual same-support result or explicit UNAVAILABLE; unavailable prevents GO",
        },
        "verdicts": {"GO": "GO_FOR_CORA_NEURAL_STAGE1", "NO_GO": "NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS"},
        "prohibited_core_data": ["TEXBAT cleanDynamic", "TEXBAT DS5", "TEXBAT DS6"],
        "optional_diagnostic_only": ["TEXBAT DS4 transition"],
        "datasets": DATASETS,
    }


def metadata_inventory() -> dict[str, Any]:
    rows = []
    for name, spec in DATASETS.items():
        raw = Path(spec["raw"])
        trace = Path(spec["trace"])
        rows.append({
            "dataset": name, "role": spec["role"], "family": spec["family"],
            "raw_path": str(raw), "raw_exists": raw.is_file(),
            "raw_size_bytes_from_stat_only": raw.stat().st_size if raw.is_file() else None,
            "trace_path": str(trace), "trace_exists_from_stat_only": trace.exists(),
            "payload_opened": False, "sha256_deferred_until_post_freeze": True,
        })
    return {"schema": "gnss-doppler-lab.cora-stage0-data-inventory.v1", "rows": rows,
            "attack_payload_bytes_read": 0, "inventory_operation": "os.stat/path.exists only"}


def write_readme() -> None:
    (ART / "README.md").write_text(
        "# CORA-GNSS Stage-0: cross-PRN common origin\n\n"
        + FREEZE_MESSAGE + "\n\n"
        "This first artifact commit freezes the alpha=0 complex fourth-cumulant experiment before "
        "attack TRACE or raw-IQ payloads are read. The freeze is a prospective evaluation freeze, not "
        "a claim of a fully blind preregistration; earlier TEXBAT work was developmental.\n\n"
        "The primary score tests whether clean-conditioned residual complex tokens from at least four "
        "simultaneously tracked PRNs contain a rank-1 off-diagonal fourth-cumulant component. The "
        "complete equations, data split, grids, controls, ablations, thresholds, gates, seeds, and "
        "fail-closed rules are in `config.json`.\n\n"
        "At this freeze stage no attack result, score, threshold outcome, or verdict exists. Running "
        "`python scripts/run_cora_stage0.py evaluate` is forbidden until the freeze commit is pushed "
        "and its local/remote SHA is recorded and verified.\n"
    )


def freeze() -> None:
    if git("rev-parse", "HEAD") != BASE_SHA:
        raise SystemExit(f"freeze must run at exact base {BASE_SHA}")
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit(f"freeze must run on {BRANCH}")
    ART.mkdir(parents=True, exist_ok=True)
    config = frozen_config()
    dump_json("config.json", config)
    dump_json("configuration_freeze.json", {
        "schema": "gnss-doppler-lab.cora-stage0-configuration-freeze.v1",
        "statement": FREEZE_MESSAGE,
        "base_sha": BASE_SHA, "branch": BRANCH,
        "expected_first_commit_message": "CORA_STAGE0_CONFIGURATION_FREEZE",
        "frozen_before_attack_payload_read": True,
        "attack_payload_bytes_read": 0,
        "remote_freeze_sha": None,
        "remote_freeze_verified": False,
        "evaluation_permitted": False,
        "note": "TEXBAT was previously used developmentally; this is not described as a fully blind preregistration.",
    })
    dump_json("source_commit.json", {
        "schema": "gnss-doppler-lab.cora-stage0-source-commit.v1",
        "base_branch": "origin/research/cinder-stage0a-clean-emitter-identifiability",
        "base_sha": BASE_SHA, "working_branch": BRANCH,
        "main_sha_at_freeze": MAIN_SHA_AT_FREEZE, "main_modified": False,
    })
    dump_json("data_inventory.json", metadata_inventory())
    dump_json("clean_split_audit.json", {
        "schema": "gnss-doppler-lab.cora-stage0-clean-split-audit.v1",
        "split": config["clean_split"], "overlap_samples": 0,
        "attack_preonset_used_for_fit_or_calibration": False,
        "attack_payload_read_before_freeze": False,
    })
    write_readme()
    source_hashes = {}
    for relative in ("src/gnss_doppler_lab/cora_cross_cumulant.py", "src/gnss_doppler_lab/cora_common_origin.py", "scripts/run_cora_stage0.py", "tests/test_cora_common_origin.py"):
        source_hashes[relative] = sha256_file(ROOT / relative)
    dump_json("freeze_source_hashes.json", source_hashes)
    print(json.dumps({"status": "FROZEN_LOCALLY", "artifact": str(ART), "attack_payload_bytes_read": 0}, indent=2))


def evaluate() -> None:
    freeze_path = ART / "configuration_freeze.json"
    if not freeze_path.is_file():
        raise SystemExit("missing configuration freeze")
    state = json.loads(freeze_path.read_text())
    if not state.get("remote_freeze_verified") or not state.get("evaluation_permitted"):
        raise SystemExit("evaluation forbidden until pushed freeze SHA is independently verified and recorded")
    raise SystemExit("post-freeze evaluator has not yet been activated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    if args.mode == "freeze": freeze()
    else: evaluate()


if __name__ == "__main__":
    main()
