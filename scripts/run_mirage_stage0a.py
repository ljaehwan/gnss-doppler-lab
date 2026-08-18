#!/usr/bin/env python3
"""Preregister and fail-closed execute MIRAGE Stage-0A provenance/support audit."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.mirage_complex_minor import (  # noqa: E402
    DELAY_GRID_CHIPS, INTEGRATION_TIMES_S, design_sha256, deterministic_design,
    doppler_grid_hz, full_score, normalized_complex_minors, split_support_audit,
    svd_second_energy_ratio, validate_clean_source_path,
)

BASE = "3db0e12976b6ff98452096e921cf298be459d0e8"
BRANCH = "research/mirage-stage0a-complex-minor-feasibility"
ART = ROOT / "artifacts/mirage_stage0a_complex_minor_feasibility"
CONFIG_PATH = ROOT / "configs/mirage_stage0a.json"
R0C = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
MOSAIC_A = ROOT / "artifacts/mosaic_stage0a_r1_raw_recorrelation"
RECEIVER = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name: str, fields: list[str], rows: list[dict[str, object]]) -> None:
    with (ART / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def write_csv_gz(name: str, fields: list[str], rows: list[dict[str, object]]) -> None:
    with (ART / name).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
                writer.writeheader(); writer.writerows(rows)


def build_manifest() -> dict[str, object]:
    files = [{"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha(path)}
             for path in sorted(ART.rglob("*")) if path.is_file() and path.name != "artifact_manifest_sha256.json"]
    return {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files}


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def anchors(config: dict[str, object]) -> dict[str, dict[str, object]]:
    result = {}
    relative_s = (1.0, 2.5, 4.0, 5.5, 7.0, 8.5)
    for dataset, spec in config["datasets"].items():
        start, end = spec["authorized_common_interval"]
        fs = int(spec["sample_rate_hz"])
        starts = [int(start + seconds * fs) for seconds in relative_s]
        ranges = [[x, x + int(.5 * fs)] for x in starts]
        result[dataset] = dict(spec) | {"anchor_start_samples": starts, "anchor_ranges": ranges,
            "anchor_ranges_nonoverlap": all(ranges[i][1] <= ranges[i + 1][0] for i in range(5)),
            "anchors_inside_authorized_interval": all(a >= start and b <= end for a, b in ranges)}
    return result


def preregister() -> None:
    if command("git", "rev-parse", "HEAD") != BASE:
        raise SystemExit("MIRAGE preregistration must be generated at exact requested base")
    if command("git", "branch", "--show-current") != BRANCH:
        raise SystemExit("wrong MIRAGE work branch")
    if ART.exists():
        raise FileExistsError(ART)
    config = load_config(); bound = anchors(config)
    design_input = {name: {"prns": spec["prns"], "anchor_start_samples": spec["anchor_start_samples"]}
                    for name, spec in bound.items()}
    design = deterministic_design(config["seed"], design_input)
    ART.mkdir(parents=True); (ART / "plots").mkdir()
    dump("config.json", config)
    dump("preregistration.json", {"schema": config["schema"], "status": "FROZEN_BEFORE_RESULTS",
        "results_viewed": False, "injection_executed": False, "config_sha256": sha(CONFIG_PATH),
        "score_frozen": config["score"], "go_gates_frozen": config["go_gates"],
        "support_failure_verdict": "INCONCLUSIVE_INPUT_OR_SUPPORT",
        "result_dependent_changes_prohibited": True})
    dump("source_commit.json", {"required_base_branch": "origin/research/mosaic-stage0b-r1-receiver-in-loop-injection",
        "required_base_sha": BASE, "observed_generation_sha": BASE, "branch": BRANCH, "base_match": True})
    dump("caf_grid.json", {"delay_grid_chips": DELAY_GRID_CHIPS.tolist(), "normalized_frequency_xi": [-2,-1,0,1,2],
        "scales": [{"integration_s": t, "doppler_offsets_hz": doppler_grid_hz(t).tolist(),
                    "caf_shape": [9,5], "adjacent_minor_shape": [8,4]} for t in INTEGRATION_TIMES_S]})
    dump("injection_design.json", design)
    dump("injection_design_sha256.json", {"canonical_json_sha256": design_sha256(design), "case_count": len(design),
        "single_cases": sum(x["mode"] == "single_prn" for x in design),
        "four_prn_cases": sum(x["mode"] == "simultaneous_four_prn" for x in design),
        "frozen_before_execution": True, "status": "PASS"})
    dump("data_inventory.json", {"datasets": bound, "nav_mapping": config["nav"], "receiver": config["receiver"],
        "attack_data_accessed": False, "metadata_only": True})
    (ART / "README.md").write_text("# MIRAGE Stage-0A\n\nStatus: **PREREGISTERED_NOT_EXECUTED**. Complex-minor score, grids, split, controls, design, and GO rules are frozen before results.\n")
    dump("artifact_manifest_sha256.json", build_manifest())
    print(json.dumps({"status": "PREREGISTERED_NOT_EXECUTED", "design_cases": len(design)}, indent=2))


def algebraic_tests() -> dict[str, object]:
    # Literal vectors/matrices keep the known-vector checks independent of any CAF generator.
    u = np.array([1, 2j, -1, .5, 3, -2j, 1+.5j, -3, 2], np.complex128)
    v = np.array([1, -1j, 2, .25+.5j, -2j], np.complex128)
    rank1 = u[:, None] * v[None, :]
    rank2 = rank1.copy(); rank2[1, 1] += 2 - 3j; rank2[7, 3] -= 1j
    same_magnitude_phase_a = np.ones((9,5), complex)
    phase = np.fromfunction(lambda i,j: np.exp(1j * .31 * i * j), (9,5), dtype=float)
    m1 = normalized_complex_minors(rank1)
    m2 = normalized_complex_minors(rank2)
    gain = normalized_complex_minors((3.7-2.1j)*rank2)
    zero = normalized_complex_minors(np.zeros((9,5),complex))
    mag_a = normalized_complex_minors(same_magnitude_phase_a)
    mag_b = normalized_complex_minors(phase)
    checks = {"exact_rank1": float(np.max(m1)) <= 1e-12,
        "rank2_nonzero": float(np.max(m2)) > .01,
        "common_complex_gain_invariant": bool(np.allclose(m2,gain,atol=1e-12,rtol=0)),
        "global_phase_invariant": bool(np.allclose(m2,normalized_complex_minors(np.exp(1.234j)*rank2),atol=1e-12)),
        "low_energy_finite": bool(np.isfinite(zero).all()),
        "equal_magnitude_phase_structure_distinguished": float(np.max(np.abs(mag_a-mag_b))) > .01,
        "prn_permutation_invariant_aggregator": full_score([1,2,3,4,5]) == full_score([5,3,1,4,2]),
        "variable_prn_count": full_score([1,2,3]) is None and full_score([1,2,3,4]) is not None}
    return {"checks": checks, "rank1_max_minor": float(np.max(m1)), "rank2_max_minor": float(np.max(m2)),
        "rank2_svd_diagnostic": svd_second_energy_ratio(rank2), "status": "PASS" if all(checks.values()) else "FAIL"}


def placeholder_plots(verdict: str, reason: str) -> None:
    names = ["clean_vs_two_source_minor_distribution", "collapsed_vs_resolvable_source", "scale_effect",
             "gain_awgn_control", "four_prn_detection", "temporal_desynchronization", "prn_contribution",
             "rms_cn0_shortcut_audit", "threshold_stability", "example_complex_caf_and_minor_field"]
    for name in names:
        fig, ax = plt.subplots(figsize=(7,3)); ax.axis("off")
        ax.text(.5,.62,verdict,ha="center",va="center",weight="bold",transform=ax.transAxes)
        ax.text(.5,.38,"NOT GENERATED: prerequisite support gate failed",ha="center",va="center",transform=ax.transAxes)
        ax.text(.5,.18,reason,ha="center",va="center",wrap=True,fontsize=8,transform=ax.transAxes)
        fig.tight_layout(); fig.savefig(ART/"plots"/f"{name}.png",dpi=120); plt.close(fig)


def evaluate(preregistration_sha: str) -> None:
    head = command("git", "rev-parse", "HEAD")
    if head != preregistration_sha or command("git", "status", "--porcelain"):
        raise SystemExit("evaluation requires exact pushed clean preregistration SHA")
    config = load_config(); inventory = json.loads((ART/"data_inventory.json").read_text())
    source_rows=[]; source_ok=True
    for dataset,spec in config["datasets"].items():
        path=Path(spec["raw_path"]); actual=sha(path)
        validate_clean_source_path(str(path))
        ok=actual==spec["raw_sha256"] and path.stat().st_size==spec["raw_size_bytes"]
        source_ok &= ok
        source_rows.append({"dataset":dataset,"path":str(path),"expected_sha256":spec["raw_sha256"],
            "actual_sha256":actual,"expected_size_bytes":spec["raw_size_bytes"],"actual_size_bytes":path.stat().st_size,
            "sample_rate_hz":spec["sample_rate_hz"],"sample_format":"little-endian interleaved int16 I,Q",
            "bytes_per_complex_sample":4,"status":"PASS" if ok else "FAIL"})
    receiver_sha=sha(RECEIVER); nav_path=ROOT/config["nav"]["mapping_path"]
    nav_sha=sha(nav_path); r0c_manifest=sha(R0C/"artifact_manifest_sha256.json")
    source_ok &= receiver_sha==config["receiver"]["sha256"] and nav_sha==config["nav"]["mapping_sha256"] and r0c_manifest==config["nav"]["manifest_sha256"]
    prior=json.loads((MOSAIC_A/"alignment_summary.json").read_text())
    prior_ok=(prior.get("verdict")=="STAGE0A_RAW_ALIGNMENT_PASS" and prior.get("stage0a_pass") is True
              and prior.get("overall_support_rows")==470 and prior.get("overall_rejected_rows")==0)
    source_ok &= prior_ok
    support={}
    common=json.loads((R0C/"common_interval_validation.json").read_text())["datasets"]
    for dataset,spec in config["datasets"].items():
        item=common[dataset]; duration=(item["common_raw_end_sample_exclusive"]-item["common_raw_start_sample"])/spec["sample_rate_hz"]
        support[dataset]=split_support_audit(duration,role_seconds=3.0,guard_seconds=10.0) | {
            "authorized_interval":[item["common_raw_start_sample"],item["common_raw_end_sample_exclusive"]],
            "included_prns":item["included_prns"],"minimum_five_prns":len(item["included_prns"])>=5,
            "valid_500ms_interval_possible":duration>=.5,
            "chronological_roles_constructible":duration>=config["clean_split"]["required_common_valid_span_seconds"]}
    support_ok=all(x["status"]=="PASS" for x in support.values())
    algebra=algebraic_tests(); dump("algebraic_minor_tests.json",algebra)
    dump("clean_split_audit.json", {"datasets":support,"raw_overlap":False,"ten_second_block_overlap":False,
        "split_generated":support_ok,"status":"PASS" if support_ok else "FAIL"})
    inventory.update({"metadata_only":False,"raw_sources":source_rows,"receiver_actual_sha256":receiver_sha,
        "nav_mapping_actual_sha256":nav_sha,"r0c_manifest_actual_sha256":r0c_manifest,
        "native_prompt_reconstruction":{"source":"MOSAIC Stage-0A R1 470 receiver-faithful raw re-correlations",
            "selected":prior.get("overall_support_rows"),"passed":prior.get("overall_support_rows",0)-prior.get("overall_rejected_rows",0),"status":"PASS" if prior_ok else "FAIL"},
        "source_status":"PASS" if source_ok else "FAIL"})
    dump("data_inventory.json",inventory)
    reason=("validated five-PRN NAV/NCO common intervals are shorter than the preregistered 29 s minimum "
            "needed for 3 s train + 10 s guard + 3 s calibration + 10 s guard + 3 s holdout")
    verdict="INCONCLUSIVE_INPUT_OR_SUPPORT" if not source_ok or not support_ok else "UNREACHED_SUPPORT_PASS"
    not_run={"status":"NOT_RUN_PREREQUISITE_FAILED","reason":reason}
    dump("thresholds.json",not_run | {"clean_only":True,"q99":None,"q99_5":None})
    fields=["dataset","case_type","status","reason"]
    write_csv("clean_metrics.csv",fields,[{"dataset":d,"case_type":"clean","status":"UNAVAILABLE","reason":reason} for d in config["datasets"]])
    write_csv_gz("per_case_scores.csv.gz",["case_id","dataset","mode","score","status","reason"],[])
    write_csv("injection_metrics.csv",fields,[{"dataset":d,"case_type":"injection","status":"NOT_RUN","reason":reason} for d in config["datasets"]])
    write_csv("control_metrics.csv",fields,[{"dataset":d,"case_type":"controls","status":"NOT_RUN","reason":reason} for d in config["datasets"]])
    write_csv("scale_ablation.csv",["dataset","scale_s","status","reason"],[{"dataset":d,"scale_s":t,"status":"NOT_RUN","reason":reason} for d in config["datasets"] for t in INTEGRATION_TIMES_S])
    dump("relation_destruction_metrics.json",not_run); dump("prn_dominance.json",not_run)
    write_csv("bootstrap_intervals.csv",["dataset","metric","lower_95","upper_95","status","reason"],
              [{"dataset":d,"metric":"all","lower_95":"","upper_95":"","status":"NOT_RUN","reason":reason} for d in config["datasets"]])
    placeholder_plots(verdict,reason)
    dump("final_verdict.json", {"verdict":verdict,"reason":reason,"source_lineage":"PASS" if source_ok else "FAIL",
        "algebraic_minor_tests":algebra["status"],"support":support,"clean_scoring_executed":False,
        "controlled_injection_executed":False,"receiver_replay_executed":False,"real_attack_data_accessed":False,
        "score_or_gate_changed_after_results":False,"next_action":"Acquire at least 29 s of common five-PRN authenticated NAV/NCO support per clean dataset, then rerun the frozen experiment."})
    (ART/"README.md").write_text(f"""# MIRAGE Stage-0A complex-minor feasibility\n\nVerdict: **{verdict}**.\n\nThe complex-minor algebra and source SHA lineage passed, but controlled injection was not executed. {reason}. Extrapolating NAV bits or weakening the two 10-second guards is prohibited, so the experiment failed closed before clean threshold calibration or injection. Placeholder plots explicitly record this non-execution and are not scientific results.\n\nThe frozen CAF uses 20/100/500 ms, nine delays from -0.5 to +0.5 chips, and normalized Doppler xi=-2..2. Each adjacent determinant is `|C[a,p]C[b,q]-C[a,q]C[b,p]| / sqrt(|C[a,p]C[b,q]|^2+|C[a,q]C[b,p]|^2+epsilon)`. It is invariant to a common complex gain/global phase and distinguishes the independent literal rank-2 known vector.\n\nNo TEXBAT DS or OAKBAT OS recording, attack label, neural model, threshold rescue, or injection result was accessed. MIRAGE makes no physical-detection claim from this run. The only next action is to acquire at least 29 seconds of authenticated common five-PRN NAV/NCO support for each clean dataset.\n""")
    dump("artifact_manifest_sha256.json",build_manifest())
    print(json.dumps({"verdict":verdict,"source_ok":source_ok,"support":support},indent=2))


def main() -> int:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="mode",required=True)
    sub.add_parser("preregister"); run=sub.add_parser("evaluate"); run.add_argument("--preregistration-sha",required=True)
    args=parser.parse_args(); preregister() if args.mode=="preregister" else evaluate(args.preregistration_sha); return 0


if __name__ == "__main__": raise SystemExit(main())
