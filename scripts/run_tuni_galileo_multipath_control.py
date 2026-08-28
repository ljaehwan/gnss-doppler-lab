#!/usr/bin/env python3
"""Run the preregistered TUNI Galileo same-stream multipath control."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / "tuni_galileo_multipath_control_v1.json"


def _load_script(name: str, relative: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load_script("tuni_clean_preflight", "scripts/preflight_tuni_galileo_clean.py")
FREEZE = _load_script("tuni_clean_freeze", "scripts/freeze_tuni_galileo_cgc_model.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def verify_frozen_inputs(config: dict[str, Any]) -> None:
    for record in config["frozen"].values():
        expected = str(record["sha256"])
        if expected == "TO_BE_FROZEN":
            raise ValueError("protocol and runner hashes must be frozen before release")
        path = Path(record["path"])
        verify_hash(path if path.is_absolute() else ROOT / path, expected)
    for scenario in config["scenarios"]:
        iq = Path(scenario["iq_path"])
        if not iq.is_file() or iq.stat().st_size != int(scenario["bytes"]):
            raise ValueError(f"IQ byte-count mismatch for {scenario['id']}")
        verify_hash(Path(scenario["metadata_path"]), str(scenario["metadata_sha256"]))
        verify_hash(Path(scenario["readme_path"]), str(scenario["readme_sha256"]))
    receiver = config["receiver"]
    if int(receiver["source_int16_items"]) != PREFLIGHT.ishort_source_item_count(
        float(receiver["duration_seconds"]), int(receiver["input_rate_hz"])
    ):
        raise ValueError("source item count does not match duration and complex sample rate")


def git_clean_commit() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    if status.strip():
        raise RuntimeError("release requires a clean git worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def receiver_run(
    scenario: dict[str, Any], config: dict[str, Any], output_dir: Path, executable: Path
) -> dict[str, Any]:
    receiver = config["receiver"]
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    config_path = output_dir / "receiver.conf"
    config_path.write_text(
        PREFLIGHT.render_config(
            iq_path=Path(scenario["iq_path"]),
            output_dir=output_dir,
            input_samples=int(receiver["source_int16_items"]),
            channel_count=int(receiver["channel_count"]),
            tracking_tap_count=int(receiver["tap_count"]),
            input_rate_hz=int(receiver["input_rate_hz"]),
            internal_rate_hz=int(receiver["internal_rate_hz"]),
        ),
        encoding="utf-8",
    )
    command = [str(executable), f"--config_file={config_path}", "--keyboard=false"]
    completed = subprocess.run(
        command,
        cwd=output_dir,
        capture_output=True,
        text=True,
        timeout=int(receiver["timeout_seconds"]),
        check=False,
    )
    log_path = output_dir / "receiver.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    mats = sorted(raw_dir.glob("epl_tracking_ch_*.mat"))
    prns, epochs = PREFLIGHT.valid_prns(mats)
    manifest = {
        "scenario": scenario["id"],
        "command": command,
        "return_code": completed.returncode,
        "receiver_executable_sha256": sha256(executable),
        "receiver_config_sha256": sha256(config_path),
        "source_bytes": Path(scenario["iq_path"]).stat().st_size,
        "source_official_md5": scenario["md5"],
        "mat_count": len(mats),
        "valid_prns": prns,
        "valid_epochs": epochs,
    }
    (output_dir / "receiver_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if completed.returncode != 0 or not prns:
        raise RuntimeError(f"receiver failed for {scenario['id']}; see {log_path}")
    return manifest


def longest_consecutive(bin_indices: list[int]) -> int:
    if not bin_indices:
        return 0
    longest = current = 1
    for previous, value in zip(bin_indices, bin_indices[1:]):
        current = current + 1 if value == previous + 1 else 1
        longest = max(longest, current)
    return longest


def analyze_scenario(
    scenario: dict[str, Any], receiver_dir: Path, model: dict[str, Any], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    analysis = config["analysis"]
    rows = FREEZE.load_binned_shapes(receiver_dir / "raw", int(config["receiver"]["internal_rate_hz"]))
    start, end = float(analysis["stabilization_seconds"]), float(analysis["end_seconds"])
    rows = [row for row in rows if start <= float(row["time_s"]) < end]
    if not rows:
        raise ValueError(f"no analysis bins for {scenario['id']}")
    center = np.asarray(model["model"]["center"], dtype=np.float64)
    scale = np.asarray(model["model"]["robust_scale"], dtype=np.float64)
    features = np.stack([row["feature"] for row in rows])
    residuals = (features - center) / scale
    distortion = np.linalg.norm(residuals, axis=1) / np.sqrt(residuals.shape[1])
    threshold_d = float(analysis["distortion_threshold"])
    threshold_c = float(analysis["residual_cosine_threshold"])
    max_cosine = np.full(len(rows), -1.0, dtype=np.float64)
    max_joint = np.zeros(len(rows), dtype=np.float64)
    evidence = np.zeros(len(rows), dtype=bool)
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float64)
    for time_s in sorted(set(map(float, times))):
        indices = np.flatnonzero(times == time_s)
        for left_pos, left in enumerate(indices):
            for right in indices[left_pos + 1 :]:
                denominator = float(np.linalg.norm(residuals[left]) * np.linalg.norm(residuals[right]))
                cosine = float(np.dot(residuals[left], residuals[right]) / denominator) if denominator else -1.0
                joint = float(min(distortion[left], distortion[right]) * (cosine + 1.0) / 2.0)
                for index in (left, right):
                    max_cosine[index] = max(max_cosine[index], cosine)
                    max_joint[index] = max(max_joint[index], joint)
                if distortion[left] >= threshold_d and distortion[right] >= threshold_d and cosine >= threshold_c:
                    evidence[left] = evidence[right] = True
    target_set = {int(value) for value in scenario["spoofed_prns"]}
    bin_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        bin_rows.append(
            {
                "scenario": scenario["id"],
                "multipath": bool(scenario["multipath"]),
                "time_s": float(row["time_s"]),
                "prn": int(row["prn"]),
                "label": "spoof" if int(row["prn"]) in target_set else "authentic",
                "distortion_z_rms": float(distortion[index]),
                "max_partner_cosine": float(max_cosine[index]),
                "max_pair_joint_score": float(max_joint[index]),
                "coherent_evidence": bool(evidence[index]),
            }
        )
    prn_rows: list[dict[str, Any]] = []
    for prn in sorted({int(row["prn"]) for row in rows}):
        indices = [i for i, row in enumerate(rows) if int(row["prn"]) == prn]
        evidence_bins = sorted(round(float(rows[i]["time_s"]) / float(analysis["bin_seconds"])) for i in indices if evidence[i])
        longest = longest_consecutive(evidence_bins)
        score = float(np.quantile(max_joint[indices], float(analysis["continuous_score_quantile"])))
        prn_rows.append(
            {
                "scenario": scenario["id"],
                "multipath": bool(scenario["multipath"]),
                "prn": prn,
                "label": "spoof" if prn in target_set else "authentic",
                "eligible_bins": len(indices),
                "coherent_evidence_bins": len(evidence_bins),
                "longest_consecutive_evidence_bins": longest,
                "persistent_flag": longest >= int(analysis["persistence_bins"]),
                "coherent_score_q95": score,
                "distortion_q95": float(np.quantile(distortion[indices], 0.95)),
            }
        )
    return bin_rows, prn_rows


def auc(labels: list[int], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return float(wins / (len(positives) * len(negatives)))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(config: dict[str, Any], prn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    analysis = config["analysis"]
    sensitivity_ids = set(analysis["primary_sensitivity_scenarios"])
    specificity_ids = set(analysis["primary_specificity_scenarios"])
    sensitivity = [row for row in prn_rows if row["scenario"] in sensitivity_ids]
    targets = [row for row in sensitivity if row["label"] == "spoof"]
    authentic_auc = [row for row in sensitivity if row["label"] == "authentic"]
    specificity = [row for row in prn_rows if row["scenario"] in specificity_ids and row["label"] == "authentic"]
    observed_auc = auc(
        [1] * len(targets) + [0] * len(authentic_auc),
        [float(row["coherent_score_q95"]) for row in targets + authentic_auc],
    )
    target_flag_rate = sum(bool(row["persistent_flag"]) for row in targets) / len(targets) if targets else 0.0
    authentic_flag_rate = sum(bool(row["persistent_flag"]) for row in specificity) / len(specificity) if specificity else 1.0
    by_scenario = {scenario["id"]: [row for row in prn_rows if row["scenario"] == scenario["id"]] for scenario in config["scenarios"]}
    support = bool(
        {row["prn"] for row in by_scenario["SS-12"] if row["label"] == "spoof"} == {9, 31}
        and len({row["prn"] for row in by_scenario["SS-13"] if row["label"] == "spoof"}) >= int(analysis["minimum_ss13_target_support"])
        and all(
            len([row for row in by_scenario[scenario_id] if row["label"] == "authentic"])
            >= int(analysis["minimum_authentic_prns_per_mp_scenario"])
            for scenario_id in specificity_ids
        )
    )
    auc_pass = bool(np.isfinite(observed_auc) and observed_auc >= float(analysis["minimum_auc"]))
    sensitivity_pass = target_flag_rate >= float(analysis["minimum_target_flag_rate"])
    specificity_pass = authentic_flag_rate <= float(analysis["maximum_authentic_flag_rate"])
    if not support:
        decision = "INSUFFICIENT_SUPPORT"
    elif auc_pass and sensitivity_pass and specificity_pass:
        decision = "SUPPORTED"
    elif specificity_pass:
        decision = "SPECIFICITY_ONLY"
    else:
        decision = "NOT_SUPPORTED"
    return {
        "decision": decision,
        "support_sufficient": support,
        "pooled_prn_auc_ss12_ss13": observed_auc,
        "target_flag_rate_ss12_ss13": target_flag_rate,
        "authentic_flag_rate_ss11_ss12_ss13": authentic_flag_rate,
        "target_count_ss12_ss13": len(targets),
        "authentic_count_ss12_ss13": len(authentic_auc),
        "authentic_count_all_mp": len(specificity),
        "gates": {"auc": auc_pass, "sensitivity": sensitivity_pass, "specificity": specificity_pass},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--release-token", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.release_token != config["release_token"]:
        raise ValueError("release token mismatch")
    verify_frozen_inputs(config)
    release_commit = git_clean_commit()
    output_root = (ROOT / config["output_root"]).resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    release_state = {
        "schema": config["schema"], "release_commit": release_commit,
        "config_sha256": sha256(config_path), "metrics_emitted": False,
        "post_release_tuning_or_retest": False,
    }
    state_path = output_root / "release_state.json"
    state_path.write_text(json.dumps(release_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    model = json.loads((ROOT / config["frozen"]["clean_model"]["path"]).read_text(encoding="utf-8"))
    executable = (ROOT / config["frozen"]["receiver"]["path"]).resolve()
    all_bins: list[dict[str, Any]] = []
    all_prns: list[dict[str, Any]] = []
    receiver_manifests: dict[str, Any] = {}
    for scenario in config["scenarios"]:
        receiver_dir = output_root / "receiver" / scenario["id"].lower()
        receiver_dir.mkdir(parents=True)
        receiver_manifests[scenario["id"]] = receiver_run(scenario, config, receiver_dir, executable)
        bins, prns = analyze_scenario(scenario, receiver_dir, model, config)
        all_bins.extend(bins)
        all_prns.extend(prns)
        print(f"[tuni-galileo-control] {scenario['id']} complete", flush=True)
    write_csv(output_root / "bin_scores.csv", all_bins)
    write_csv(output_root / "prn_scores.csv", all_prns)
    result = aggregate(config, all_prns)
    summary = {
        "schema": config["schema"], "release_commit": release_commit,
        "config_sha256": sha256(config_path), "model_sha256": config["frozen"]["clean_model"]["sha256"],
        "receiver_sha256": config["frozen"]["receiver"]["sha256"],
        "receiver_manifests": receiver_manifests, "result": result,
        "prn_scores_sha256": sha256(output_root / "prn_scores.csv"),
        "bin_scores_sha256": sha256(output_root / "bin_scores.csv"),
        "post_release_tuning_or_retest": False,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    release_state["metrics_emitted"] = True
    release_state["summary_sha256"] = sha256(summary_path)
    state_path.write_text(json.dumps(release_state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

