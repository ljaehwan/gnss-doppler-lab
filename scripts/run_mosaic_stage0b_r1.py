#!/usr/bin/env python3
"""Freeze R1 receiver-in-loop preregistration; deliberately runs no injection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.mosaic_iq_injector import design_sha256  # noqa: E402
from gnss_doppler_lab.mosaic_iq_injector_int16 import validate_file_format  # noqa: E402
from gnss_doppler_lab.mosaic_receiver_in_loop import assign_case_targets  # noqa: E402

BASE = "e0993bd6b16628681b52c1abd52cf177af67e10a"
EXPECTED_DESIGN_SHA = "b1a06556f7cd67738274c132f80b0581b20914d971f72f4e4ab0b5efc9a7facf"
R0C = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
FOUNDATION = ROOT / "artifacts/mosaic_stage0ab_foundation"
ART = ROOT / "artifacts/mosaic_stage0b_r1_receiver_in_loop"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def write(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def manifest() -> None:
    files = []
    for path in sorted(ART.iterdir()):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha(path)})
    write("artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="reserved for separately authorized R1 execution")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("R1 preregistration commit forbids execution; use a separately authorized post-preregistration task")
    head = command("git", "rev-parse", "HEAD")
    if head != BASE:
        raise SystemExit(f"R1 preregistration must be generated at exact R0c base {BASE}; got {head}")
    verdict = json.loads((R0C / "final_verdict.json").read_text())
    if verdict["verdict"] != "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION":
        raise SystemExit("R0c verdict does not authorize scoped R1 preregistration")
    design_path = FOUNDATION / "injection_design.json"
    design = json.loads(design_path.read_text())
    canonical = design_sha256(design)
    if canonical != EXPECTED_DESIGN_SHA or len(design) != 72:
        raise SystemExit("frozen injection design binding mismatch")
    raw = json.loads((ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance/raw_source_binding.json").read_text())
    common = json.loads((R0C / "common_interval_validation.json").read_text())
    formats = {}
    for dataset in ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic"):
        item = raw[dataset]
        observed = validate_file_format(item["stat"]["path"])
        stat_match = observed["size_bytes"] == item["stat"]["size_bytes"]
        formats[dataset] = {**observed, "source_path": item["stat"]["path"],
            "expected_full_sha256": item["expected_sha256"], "inherited_full_sha256_status": item["status"],
            "source_stat_size_match": stat_match, "receiver_signal_source": "Ishort_To_Complex",
            "status": "PASS" if stat_match and observed["bytes_per_complex_sample"] == 4 else "FAIL"}
    provenance = json.loads((ROOT / "artifacts/mosaic_stage0a_r1_raw_recorrelation/receiver_provenance.json").read_text())
    executable = Path(provenance["receiver_executable"]["path"])
    executable_sha = sha(executable)
    if executable_sha != provenance["expected_receiver_sha256"]:
        raise SystemExit("receiver binary binding mismatch")
    prns = {dataset: common["datasets"][dataset]["included_prns"] for dataset in common["datasets"]}
    assignment = assign_case_targets(design, prns)
    ART.mkdir(parents=True, exist_ok=False)
    write("frozen_injection_design.json", design)
    write("frozen_injection_design_sha256.json", {
        "canonical_json_sha256": canonical, "required_canonical_json_sha256": EXPECTED_DESIGN_SHA,
        "source_file_sha256": sha(design_path), "case_count": len(design), "design_modified": False, "status": "PASS"})
    write("case_target_assignment.json", {"rule": {
        "single_prn": "target_prn=sorted_prns[i mod 5]",
        "four_prn": "excluded_prn=sorted_prns[i mod 5]; remaining four are targets"},
        "case_count": len(assignment), "assignments": assignment, "status": "PASS"})
    write("sample_format_validation.json", {"datasets": formats, "production_quantizer": "complex_int16_little_endian_saturating",
        "foundation_int8_quantizer_used": False, "zero_amplitude_byte_identity_required": True,
        "input_output_sample_count_preservation_required": True, "status": "PASS" if all(x["status"] == "PASS" for x in formats.values()) else "FAIL"})
    write("receiver_binary_binding.json", {"path": str(executable), "size_bytes": executable.stat().st_size,
        "expected_sha256": provenance["expected_receiver_sha256"], "observed_sha256": executable_sha,
        "receiver_configs": provenance["receiver_configs"], "status": "PASS"})
    write("r0c_input_binding.json", {"required_branch": "origin/research/mosaic-stage0b-r0c-boundary-phase-extrapolation",
        "required_commit": BASE, "r0c_manifest_sha256": sha(R0C / "artifact_manifest_sha256.json"),
        "r0c_verdict": verdict["verdict"], "common_intervals": common["datasets"],
        "corrected_mapping_sha256": sha(R0C / "corrected_bit_mapping.csv.gz"), "status": "PASS"})
    prereg = {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r1-preregistration.v1",
        "case_count": 72, "results_seen_before_freeze": False,
        "time_structure_seconds": {"identity_baseline": [0, 2], "raised_cosine_ramp": [2, 4], "target_hold": [4, 10], "smooth_release": [10, "interval_end"]},
        "state_continuity": ["absolute_raw_sample_index", "code_phase", "carrier_phase", "doppler_phase_accumulation", "NAV_bit_sign"],
        "authentic_amplitude": "complex least-squares correlation of clean baseline with target-PRN replica",
        "spoof_amplitude": "abs(alpha_auth)*10**(rho_db/20)", "realized_scer_error_required": True,
        "caf_grid": {"delay_chips": {"start": -0.35, "stop": 0.35, "step": 0.025}, "doppler_hz": {"start": -75, "stop": 75, "step": 5}},
        "hypotheses": {"H0": "authentic single-source model", "H1": "authentic + same-PRN second-source model"},
        "scores": ["residual_CAF", "recovered_delay", "recovered_doppler", "H0_RSS", "H1_RSS", "BIC_corrected_likelihood_improvement"],
        "raw_rss_improvement_is_sufficient": False,
        "collapsed_control": {"condition": "delta_tau==0 and delta_f==0", "label": "COLLAPSED_SINGLE_SOURCE_CONTROL", "detection_failure_counts_as_no_go": False},
        "strong_resolvable_subset": {"rho_db_min": -6, "abs_delta_tau_chips_min_or": .10, "abs_delta_f_hz_min_or": 25},
        "go_criteria": ["identity receiver replay", "zero-amplitude byte identity", "actual int16 format",
            "realized SCER median error <=1 dB", "OAK and TEX residual CAF observability >=75%", "delay sign accuracy >=80%",
            "delay median absolute error <=0.05 chip", "Doppler sign accuracy >=80%", "Doppler median absolute error <=10 Hz",
            "injected H1 delta-BIC significantly exceeds identity/gain/AWGN controls", "target PRN evidence exceeds non-target PRN",
            "not explained by total IQ RMS shortcut", "four-PRN strong cases >=75% recover at least 3/4", "PRN permutation invariance"],
        "future_verdicts": ["GO_FOR_MOSAIC_NEURAL_STAGE1", "INCONCLUSIVE_RECEIVER_IN_LOOP", "NO_GO_MOSAIC_INJECTOR_PHYSICS"],
        "scientific_verdict_generated": False,
    }
    write("preregistration.json", prereg)
    write("config.json", {"base_commit": BASE, "workers": 1, "chunk_streaming": True,
        "sample_format": "<i2 interleaved I,Q", "bytes_per_complex_sample": 4,
        "attack_data_accessed": False, "injection_executed": False, "results_viewed": False,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "4"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "4")})
    write("source_commit.json", {"required_base_branch": "origin/research/mosaic-stage0b-r0c-boundary-phase-extrapolation",
        "required_base_commit": BASE, "observed_generation_commit": head,
        "work_branch": command("git", "branch", "--show-current"), "base_match": True})
    write("execution_status.json", {"status": "READY_FOR_R1_EXECUTION", "injection_executed": False,
        "attack_data_accessed": False, "results_viewed": False, "receiver_replay_executed": False,
        "scientific_verdict_generated": False})
    (ART / "README.md").write_text("""# MOSAIC Stage-0B R1 receiver-in-loop preregistration

Status: **READY_FOR_R1_EXECUTION**. This commit freezes the production int16 I/Q path, stateful R0c NAV/NCO replica contract, 72-case target assignment, receiver binding, analysis grid, controls, and GO/NO-GO rules. It does not execute injection, receiver replay, or inspect results.

Both clean sources are little-endian interleaved signed int16 I/Q with four bytes per complex sample. Counterfeit amplitude is referenced to the target PRN's complex least-squares `alpha_auth`, never total clean-IQ RMS. The zero-amplitude path must preserve bytes exactly and nonzero paths preserve sample count while reporting saturation.

The R0c intervals are frozen to OAK `[150275296, 210202273)` and TEX `[817815304, 1117517038)`. Each run uses 0–2 s identity, 2–4 s raised-cosine ramp, 4–10 s hold, and a smooth release to interval end. Code/carrier/NAV/absolute-sample state may not restart at chunk or epoch boundaries.

No scientific verdict is emitted at preregistration. A future authorized execution must apply the frozen BIC-corrected H0/H1 analysis and all GO criteria. Receiver/input failures map to `INCONCLUSIVE_RECEIVER_IN_LOOP`; physical recovery failures map to `NO_GO_MOSAIC_INJECTOR_PHYSICS`.
""")
    manifest()
    print("READY_FOR_R1_EXECUTION: preregistered 72 cases; injection not executed")


if __name__ == "__main__":
    main()
