#!/usr/bin/env python3
"""Release and analyze the support-selected fresh CGC RF test once."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for search_root in (REPO_ROOT / "scripts", REPO_ROOT / "src"):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import run_cgc_rf_locked_test as base  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_rf_fresh_test_v2.json"
PROTOCOL_PATH = REPO_ROOT / "docs/results/cgc_rf_fresh_test_protocol_v2.md"
EXPECTED_PAIR_IDS = ["fv2-static-01", "fv2-straight-01", "fv2-sweep-01"]
RELEASE_TOKEN = "RELEASE-CGC-RF-FRESH-TEST-V2"
RELEASE_INPUTS = (
    "configs/experiments/cgc_rf_fresh_test_v2.json",
    "docs/results/cgc_rf_fresh_test_protocol_v2.md",
    "scripts/run_cgc_rf_fresh_test_v2.py",
)
EXPECTED_GATES = {
    "required_pair_count": 3,
    "positive_clock_centered_separation_pair_count": 3,
    "minimum_pair_block_auc": 0.8,
    "minimum_clock_centered_improvement_over_legacy_pair_count": 2,
    "minimum_comparison_bins_per_scenario_per_pair": 5,
    "minimum_startup_los_prns_per_pair": 8,
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_pinned(record: dict[str, str], label: str) -> tuple[Path, Any]:
    path = _repo_path(record["path"])
    observed = _sha256(path)
    if observed != record["sha256"]:
        raise ValueError(f"{label} hash mismatch: {observed}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _motion_kind(pair: dict[str, Any]) -> str:
    return "static" if pair["domain"] == "static" else str(pair["motion"]["kind"])


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") != "gnss-doppler-lab.cgc-rf-fresh-test-config" or config.get("schema_version") != 1:
        raise ValueError("unsupported fresh-test config")
    experiment = config.get("experiment", {})
    if experiment.get("name") != "cgc-rf-fresh-test-v2" or experiment.get("candidate_commit") != "60280575737ff70618c5619ff7522c516bbdc67a":
        raise ValueError("experiment identity drifted")
    if experiment.get("runner_path") != "scripts/run_cgc_rf_fresh_test_v2.py":
        raise ValueError("runner path drifted")

    frozen = config.get("frozen_candidate", {})
    for key in ("positive_train_replication", "negative_absolute_threshold_audit", "clock_centered_module", "reference_locked_runner"):
        path = _repo_path(frozen[key]["path"])
        if _sha256(path) != frozen[key]["sha256"]:
            raise ValueError(f"frozen input hash mismatch: {key}")
    if frozen.get("residual") != "SSE_full / sum(weight * (delay - weighted_mean(delay))^2)" or frozen.get("detection_score") != "negative residual":
        raise ValueError("frozen score law drifted")
    if frozen.get("absolute_threshold_applied") is not False or frozen.get("threshold_or_calibration_fitting") is not False:
        raise ValueError("threshold use is forbidden")

    pool_path, pool = _load_pinned(config["selection"]["candidate_pool"], "candidate pool")
    record_path, record = _load_pinned(config["selection"]["preflight_record"], "preflight record")
    if record.get("selected_candidate_ids") != EXPECTED_PAIR_IDS or record.get("score_accessed") is not False:
        raise ValueError("support-only selection record drifted")
    by_id = {row["paired_group_id"]: row for row in pool["candidates"]}
    pairs = config.get("pairs")
    if not isinstance(pairs, list) or [row.get("paired_group_id") for row in pairs] != EXPECTED_PAIR_IDS:
        raise ValueError("fresh pair roster drifted")
    if any(row != by_id[row["paired_group_id"]] for row in pairs):
        raise ValueError("selected pair changed after support preflight")
    if [_motion_kind(row) for row in pairs] != ["static", "straight", "parallel-sweep"]:
        raise ValueError("motion roster drifted")
    counts = config["selection"]["selected_startup_los_counts"]
    recorded_counts = {row["candidate_id"]: row["startup_los_prn_count"] for row in record["probes"]}
    if counts != {pair_id: recorded_counts[pair_id] for pair_id in EXPECTED_PAIR_IDS} or min(counts.values()) < 10:
        raise ValueError("startup LOS selection proof drifted")

    boundary = config.get("data_boundary", {})
    if boundary.get("authorized_partition") != "fresh_test" or boundary.get("allowed_pair_ids") != EXPECTED_PAIR_IDS:
        raise ValueError("fresh-test data boundary drifted")
    required_true = ("selection_used_only_startup_los_support", "post_release_pair_substitution_forbidden", "test_refit_or_adaptation_forbidden")
    if any(boundary.get(key) is not True for key in required_true) or boundary.get("pre_release_cgc_scores_accessed") is not False:
        raise ValueError("leakage boundary drifted")
    if boundary.get("texbat_recordings_in_primary_test") != []:
        raise ValueError("TEXBAT cannot enter the primary test")

    normal_path, normal_profile = _load_pinned(config["source_generation"]["normal_profile"], "normal profile")
    generator_path = _repo_path(config["source_generation"]["reference_paired_generator"]["path"])
    if _sha256(generator_path) != config["source_generation"]["reference_paired_generator"]["sha256"]:
        raise ValueError("paired generator hash mismatch")
    controlled_path, controlled = _load_pinned(config["analysis"]["controlled_template_config"], "controlled template config")
    for key in ("simulator_executable", "simulator_patch", "multipath_module"):
        record_item = config["multipath"][key]
        if _sha256(_repo_path(record_item["path"])) != record_item["sha256"]:
            raise ValueError(f"multipath input hash mismatch: {key}")
    for key in ("executable", "patch"):
        record_item = config["gnss_sdr"][key]
        if _sha256(_repo_path(record_item["path"])) != record_item["sha256"]:
            raise ValueError(f"receiver input hash mismatch: {key}")
    if config["multipath"]["seed_by_pair"] != {"fv2-static-01": 20261011, "fv2-straight-01": 20261012, "fv2-sweep-01": 20261013}:
        raise ValueError("multipath seeds drifted")
    if config["multipath"]["delay_chips_range"] != [0.12, 0.45] or config["multipath"]["amplitude_range"] != [0.20, 0.70]:
        raise ValueError("multipath law drifted")
    if {key: config["gnss_sdr"][key] for key in ("channel_count", "tracking_tap_count", "tracking_tap_spacing_chips")} != {"channel_count": 11, "tracking_tap_count": 9, "tracking_tap_spacing_chips": 0.125}:
        raise ValueError("receiver contract drifted")
    if config["analysis"].get("bin_seconds") != 1.0 or config["analysis"].get("minimum_prns") != 8:
        raise ValueError("analysis support contract drifted")
    if config["evaluation"].get("support_gates") != EXPECTED_GATES or config["evaluation"].get("threshold_fitting") is not False or config["evaluation"].get("post_release_tuning_or_retest") is not False:
        raise ValueError("evaluation contract drifted")

    profile = config["source_generation"]
    source_root = _repo_path(profile["output_root"])
    output_root = _repo_path(config["output_root"])
    if source_root != REPO_ROOT / "artifacts/simulation_v5_fresh_test_generation_v2" or output_root != REPO_ROOT / "artifacts/cgc_rf_fresh_test_v2":
        raise ValueError("fresh output roots drifted")
    if int(normal_profile["rf_profile"]["rf_sample_rate_hz"]) != 25000000 or int(normal_profile["gnss_sdr"]["tracking_tap_count"]) != 9:
        raise ValueError("normal profile differs from frozen RF contract")
    return {
        "split_path": pool_path,
        "split_record_path": record_path,
        "split": pool,
        "pairs": pairs,
        "normal_path": normal_path,
        "normal_profile": normal_profile,
        "controlled_path": controlled_path,
        "controlled": controlled,
        "source_root": source_root,
        "output_root": output_root,
    }


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _committed_release_record() -> dict[str, Any]:
    for relative in RELEASE_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=REPO_ROOT).returncode:
            raise ValueError(f"release input is not committed and clean: {relative}")
    candidate = "60280575737ff70618c5619ff7522c516bbdc67a"
    if subprocess.run(["git", "merge-base", "--is-ancestor", candidate, "HEAD"], cwd=REPO_ROOT).returncode:
        raise ValueError("frozen candidate commit is not an ancestor of HEAD")
    return {
        "head_commit": _git("rev-parse", "HEAD"),
        "candidate_commit": candidate,
        "input_commits": {relative: _git("log", "-1", "--format=%H", "--", relative) for relative in RELEASE_INPUTS},
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "reference_locked_runner_sha256": _sha256(REPO_ROOT / "scripts/run_cgc_rf_locked_test.py"),
    }


def _start_or_resume(config: dict[str, Any], config_path: Path, context: dict[str, Any], resume: bool) -> tuple[Path, dict[str, Any]]:
    state_path = context["output_root"] / "release_state.json"
    summary_path = context["output_root"] / "summary.json"
    commits = _committed_release_record()
    if not resume:
        if context["source_root"].exists() or context["output_root"].exists():
            raise FileExistsError("fresh release output root already exists")
        context["output_root"].mkdir(parents=True)
        state = {
            "schema": "gnss-doppler-lab.cgc-rf-fresh-test-release-state", "schema_version": 1,
            "started_at_utc": datetime.now(timezone.utc).isoformat(), "phase": "released_before_source_generation",
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
            "commits": commits, "authorized_pair_ids": EXPECTED_PAIR_IDS,
            "pair_substitution_forbidden": True, "metrics_emitted": False,
        }
        base._write_json(state_path, state)
        return state_path, state
    if not state_path.is_file():
        raise FileNotFoundError("resume requested without release state")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("config", {}).get("sha256") != _sha256(config_path) or state.get("commits", {}).get("runner_sha256") != commits["runner_sha256"]:
        raise ValueError("resume provenance mismatch")
    if state.get("metrics_emitted") is not False or summary_path.exists():
        raise ValueError("metrics already emitted; rerun forbidden")
    return state_path, state


def _phase(path: Path, state: dict[str, Any], phase: str) -> None:
    state["phase"] = phase
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    base._write_json(path, state)


def _emit(config: dict[str, Any], config_path: Path, context: dict[str, Any], state_path: Path, state: dict[str, Any], analysis: dict[str, Any], source_summary: Path) -> dict[str, Any]:
    analysis_root = context["output_root"] / "analysis"
    paths = {
        "delay_estimates": analysis_root / "delay_estimates.csv",
        "geometry_scores": analysis_root / "geometry_scores.csv",
        "pair_summary": analysis_root / "pair_summary.csv",
        "pair_scenario_medians": analysis_root / "pair_scenario_medians.csv",
    }
    state["phase"] = "metrics_emitted"
    state["metrics_emitted"] = True
    state["metrics_emitted_at_utc"] = datetime.now(timezone.utc).isoformat()
    base._write_json(state_path, state)
    base._write_csv(paths["delay_estimates"], analysis["delays"])
    base._write_csv(paths["geometry_scores"], analysis["geometry"])
    base._write_csv(paths["pair_summary"], analysis["pairs"])
    base._write_csv(paths["pair_scenario_medians"], analysis["pair_scenarios"])
    rows = {"delay_estimates": len(analysis["delays"]), "geometry_scores": len(analysis["geometry"]), "pair_summary": len(analysis["pairs"]), "pair_scenario_medians": len(analysis["pair_scenarios"])}
    result = {
        "schema": "gnss-doppler-lab.cgc-rf-fresh-test-result", "schema_version": 1,
        "role": config["experiment"]["role"], "execution_policy": config["experiment"]["execution_policy"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": _sha256(PROTOCOL_PATH)},
        "orchestrator_runner": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve()), "commit": state["commits"]["input_commits"]["scripts/run_cgc_rf_fresh_test_v2.py"]},
        "frozen_reference_runner": config["frozen_candidate"]["reference_locked_runner"],
        "release_state": {"path": str(state_path.resolve()), "sha256": _sha256(state_path)},
        "source_generation_summary": {"path": str(source_summary.resolve()), "sha256": _sha256(source_summary)},
        "selection": config["selection"], "frozen_candidate": config["frozen_candidate"],
        "pairs": analysis["pairs"], "primary_pair_block_evaluation": analysis["primary"], "secondary_serial_bin_auc": analysis["secondary"],
        "artifacts": {key: {"path": str(path.resolve()), "sha256": _sha256(path), "row_count": rows[key]} for key, path in paths.items()},
        "data_boundary": config["data_boundary"], "threshold_fitted": False, "absolute_threshold_applied": False,
        "pair_substitution_performed": False, "post_release_tuning_or_retest": False, "claim_boundary": config["claim_boundary"],
    }
    base._write_json(context["output_root"] / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--release-token", choices=[RELEASE_TOKEN])
    mode.add_argument("--resume-before-metrics", action="store_true")
    args = parser.parse_args(argv)
    config_path = DEFAULT_CONFIG.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = validate_config(config)
    base.EXPECTED_PAIR_IDS = EXPECTED_PAIR_IDS
    base.SOURCE_CAMPAIGN = "simulation-v5-fresh-test-v2"
    if args.validate_only:
        print("fresh v2 config and pinned inputs verified; no final signal or score accessed")
        return 0
    resume = bool(args.resume_before_metrics)
    state_path, state = _start_or_resume(config, config_path, context, resume)
    _phase(state_path, state, "source_generation")
    for pair in context["pairs"]:
        print(f"[fresh-source] {pair['paired_group_id']}", flush=True)
        base._ensure_source_pair(config, config_path, context, pair, resume=resume)
    source_summary = base._write_source_summary(config, config_path, context)
    _phase(state_path, state, "receiver_processing")
    for pair in context["pairs"]:
        print(f"[fresh-receiver] {pair['paired_group_id']}", flush=True)
        base._ensure_test_pair(config, config_path, context, pair, resume=resume)
    _phase(state_path, state, "analysis_in_memory")
    analysis = base._analyze_all_in_memory(config, config_path, context)
    _emit(config, config_path, context, state_path, state, analysis, source_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
