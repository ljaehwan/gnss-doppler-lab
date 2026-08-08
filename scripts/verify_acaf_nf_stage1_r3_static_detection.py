#!/usr/bin/env python3
"""Independent verifier for ACAF-NF Stage-1 R3 artifacts.

This file intentionally does not import the R3 producer or R3 model module.
It mirrors the model equation and recomputes one full L20 CAF from raw IQ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import code_replica, carrier_wipeoff  # noqa: E402

FS = 25_000_000.0
SUPPORT = 25_000
DELAYS = np.arange(-1.0, 1.0001, .125)
DOPPLERS = np.arange(-250.0, 250.0001, 50.0)
GRID = np.asarray([(d / 250.0, c) for d in DOPPLERS for c in DELAYS], np.float32)
CENTER = 93
REQUIRED = {
    "README.md", "config.json", "source_binding.json", "timeline_validation.json", "normal_split.json",
    "model_manifest.json", "query_policy.json", "thresholds.json", "scenario_metrics.csv", "phase_metrics.csv",
    "family_metrics.csv", "budget_metrics.csv", "baseline_metrics.csv", "ablation_metrics.csv", "bootstrap_results.json",
    "physical_controls.json", "positive_control_results.json", "per_epoch_scores.csv", "query_traces.csv",
    "b0_comparison.json", "diagnostic_scenarios.json", "go_no_go.json", "execution_validity.json", "test_report.txt", "freeze_manifest.json",
    "model.pt", "model_no_context.pt", "pooling.json", "calibration.json", "clean_features.npz",
    "clean_features.json", "attack_features.npz", "attack_features.json", "normal_field_reference.npz", "query_policy_controls.json",
}
PLOTS = {
    "ds3_score_time.png", "ds4_score_time.png", "ds7_score_time.png", "ds8_score_time.png",
    "budget_pauc.png", "detection_correlator_tradeoff.png", "prn_score_heatmap.png",
    "active_query_coordinate_heatmap.png", "query_order.png", "dense_vs_active.png",
    "active_vs_b0.png", "gain_phase_awgn_control.png", "external_pre_onset_fpr.png",
}
FROZEN = {"model.pt", "model_context.pt", "model_no_context.pt", "normal_field_reference.npz",
          "query_policy.json", "thresholds.json", "pooling.json", "calibration.json"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class MirrorNF(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__(); h, z = config["hidden_dim"], config["latent_dim"]
        self.context_features = config["context_features"]
        self.floor, self.ceiling = config["variance_floor"], config["variance_ceiling"]
        self.point = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, z), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(z + 2 + (2 if self.context_features else 0), h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 4))

    def predict(self, values: np.ndarray, observed: list[int], targets: list[int], receiver: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.column_stack((values.real, values.imag)).astype(np.float32)
        indices = observed or [CENTER]; mask = 1.0 if observed else 0.0
        points = torch.as_tensor(np.concatenate((GRID[indices], y[indices]), axis=1))
        latent = (self.point(points) * mask).sum(0) / max(len(observed), 1)
        fields = [torch.as_tensor(GRID[targets]), latent[None].expand(len(targets), -1)]
        if self.context_features: fields.append(torch.as_tensor(receiver)[None].expand(len(targets), -1))
        output = self.decoder(torch.cat(fields, 1)); mean = output[:, :2]
        variance = torch.exp(output[:, 2:]).clamp(self.floor, self.ceiling)
        return mean.detach().numpy(), variance.detach().numpy()


def mirror_active_score(model: MirrorNF, values: np.ndarray, receiver: np.ndarray, budget: int = 9) -> tuple[float, list[int]]:
    observed, scores = [], []
    for step in range(budget):
        if step == 0: selected = CENTER
        else:
            remaining = [i for i in range(187) if i not in observed]
            _, v = model.predict(values, observed, remaining, receiver)
            selected = remaining[int(np.argmax(np.sum(np.log(v), axis=1)))]
        mean, variance = model.predict(values, observed, [selected], receiver)
        actual = np.asarray([values[selected].real, values[selected].imag])
        delta = actual - mean[0]
        scores.append(float(np.sum(delta * delta / variance[0]) + np.sum(np.log(variance[0]))))
        observed.append(selected)
    return float(np.mean(scores)), observed


def raw_surface(raw: np.memmap, state: dict[str, Any]) -> np.ndarray:
    start = int(state["raw_start_sample"])
    packed = np.asarray(raw[2 * start:2 * (start + SUPPORT)]).reshape(-1, 2)
    iq = packed[:, 0].astype(np.float64) + 1j * packed[:, 1].astype(np.float64)
    replicas = np.asarray([code_replica(int(state["prn"]), SUPPORT, FS, float(state["code_freq_chips"]),
        float(state["aux1"]), -1, float(delay), replica_direction=1)[0] for delay in DELAYS])
    wipes = np.asarray([carrier_wipeoff(SUPPORT, FS, float(state["carrier_doppler_hz"]), float(doppler), -1)[0] for doppler in DOPPLERS])
    return (wipes * iq[None]) @ replicas.T


def independent_l20(root: Path) -> dict[str, Any]:
    meta = load(root / "clean_features.json")[0]
    archive = np.load(root / "clean_features.npz"); expected = archive["surfaces"][0]
    raw_path = Path(load(root / "config.json")["raw_paths"]["cleanStatic"]); raw = np.memmap(raw_path, dtype="<i2", mode="r")
    normalized = []
    for state in meta["states"]:
        surface = raw_surface(raw, state); prompt = surface[5, 8]
        normalized.append(surface * np.exp(-1j * np.angle(prompt)) / (abs(prompt) + 1e-9))
    recomputed = np.median(np.asarray(normalized).real, axis=0) + 1j * np.median(np.asarray(normalized).imag, axis=0)
    delta = float(np.max(np.abs(recomputed.reshape(-1) - expected)))
    return {"scenario": "cleanStatic", "channel": meta["channel"], "prn": meta["prn"], "time_s": meta["time_s"],
            "raw_start_sample": meta["raw_start_sample"], "raw_end_sample": meta["raw_end_sample"],
            "max_abs_surface_delta": delta, "tolerance": 2e-5, "status": "PASS" if delta <= 2e-5 else "FAIL"}


def independent_score(root: Path) -> dict[str, Any]:
    archive = np.load(root / "clean_features.npz"); values = archive["surfaces"][0]
    meta = load(root / "clean_features.json")[0]; checkpoint = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    model = MirrorNF(checkpoint["config"]); model.load_state_dict(checkpoint["state_dict"]); model.eval()
    stats = checkpoint["context_stats"]
    receiver = (np.asarray([meta["cn0_db_hz"], meta["carrier_lock"]], np.float32) - stats["mean"]) / stats["std"]
    actual, order = mirror_active_score(model, values, receiver)
    with (root / "clean_node_scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["scenario"] == "cleanStatic" and row["method"] == "active_adaptive" and int(row["budget"]) == 9
                and int(row["channel"]) == int(meta["channel"]) and int(row["prn"]) == int(meta["prn"]) and int(row["second"]) == int(meta["second"])]
    expected = float(rows[0]["score"]) if len(rows) == 1 else float("nan")
    delta = abs(actual - expected)
    return {"actual": actual, "stored": expected, "absolute_delta": delta, "query_order": order, "pre_observation": True,
            "status": "PASS" if len(rows) == 1 and delta <= 2e-5 else "FAIL"}


def verify(root: Path) -> dict[str, Any]:
    errors = []
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing: errors.append(f"missing:{','.join(missing)}")
    missing_plots = sorted(name for name in PLOTS if not (root / "plots" / name).is_file())
    if missing_plots: errors.append(f"plots:{','.join(missing_plots)}")
    if errors:
        return {"schema": "acaf_nf_stage1_r3_verifier.v1", "status": "FAIL", "errors": errors, "independent_raw_recomputation_performed": False}
    freeze = load(root / "freeze_manifest.json")
    if set(freeze.get("files", {})) != FROZEN: errors.append("freeze_schema")
    for name in FROZEN:
        if freeze.get("files", {}).get(name) != digest(root / name): errors.append(f"freeze_drift:{name}")
    context_checkpoint = torch.load(root / "model_context.pt", map_location="cpu", weights_only=True)
    no_context_checkpoint = torch.load(root / "model_no_context.pt", map_location="cpu", weights_only=True)
    if context_checkpoint["config"].get("context_features") is not True or no_context_checkpoint["config"].get("context_features") is not False:
        errors.append("context_checkpoint_roles")
    context_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in context_checkpoint["state_dict"].values())).hexdigest()
    no_context_hash = hashlib.sha256(b"".join(value.detach().cpu().numpy().tobytes() for value in no_context_checkpoint["state_dict"].values())).hexdigest()
    if context_hash == no_context_hash: errors.append("context_checkpoints_not_distinct")
    config, binding = load(root / "config.json"), load(root / "source_binding.json")
    if binding["cleanStatic"]["status"] != "PASS" or any(binding["attacks"][x]["status"] != "PASS" for x in ("ds3", "ds4", "ds7", "ds8")): errors.append("source_binding")
    if any(binding["attacks"][x]["actual_sha256"] != config["raw_sha256"][x] for x in ("ds3", "ds4", "ds7", "ds8")): errors.append("raw_sha")
    tracker_checks = {}
    for scenario in ("cleanStatic", "ds3", "ds4", "ds7", "ds8"):
        manifest_path = Path(config["tracker_root"]) / scenario / "manifest.json"; manifest = load(manifest_path)
        stored = binding.get("tracker_mat_binding", {}).get(scenario, {})
        results = {}
        for name, expected in sorted(manifest["tracking"]["mat_inventory"].items()):
            path = Path(config["tracker_root"]) / scenario / "raw" / name; actual = digest(path) if path.is_file() else None
            results[name] = actual == expected and stored.get("files", {}).get(name, {}).get("actual_sha256") == actual
        tracker_checks[scenario] = {"files": len(results), "all_match": len(results) >= 8 and all(results.values())}
        if not tracker_checks[scenario]["all_match"] or stored.get("status") != "PASS": errors.append(f"tracker_mat_binding:{scenario}")
    split = load(root / "normal_split.json")
    if split.get("byte_overlap_status") != "PASS" or split.get("clean_after_100s_accessed") is not False: errors.append("normal_split")
    timeline = load(root / "timeline_validation.json")
    if timeline["scenarios"]["ds4"]["status"] != "FULL_DS4_UNAVAILABLE" or timeline["scenarios"]["ds7"]["official_onset_s"] != 110: errors.append("timeline")
    go = load(root / "go_no_go.json")
    if go["verdict"] != "ACAF_NF_INCONCLUSIVE" or go["ds7_ds8_independent_confirmations"] is not False: errors.append("verdict")
    b0 = load(root / "b0_comparison.json")
    if b0["historic_scores_reused"] is not False or b0["comparison_valid"] is not False: errors.append("b0")
    with (root / "family_metrics.csv").open(newline="", encoding="utf-8") as handle:
        family_rows = list(csv.DictReader(handle))
    if {x["family"] for x in family_rows} != {"ds3", "ds4", "ds7_ds8"} or not any(x["family"] == "ds7_ds8" and x["scenario_members"] == "ds7+ds8" for x in family_rows):
        errors.append("family_metrics")
    bootstrap = load(root / "bootstrap_results.json")
    required_comparisons = ("active_adaptive_K3_vs_epl_3_K3", "active_adaptive_K9_vs_fixed_delay_9_K9",
                            "active_adaptive_K9_vs_uniform_fixed_K9", "active_adaptive_K9_vs_random_fixed_K9",
                            "active_adaptive_K9_vs_dense_nf_K187", "analytic_two_source_glrt_K187_vs_dense_one_source_residual_K187",
                            "active_context_K9_vs_active_no_context_K9")
    if bootstrap.get("metric") != "normalized_partial_auc_fpr_0_05" or any(
        f"ds7_ds8:{comparison}" not in bootstrap.get("family_comparisons", {}) for comparison in required_comparisons
    ) or bootstrap.get("active_vs_b0", {}).get("status") != "INCONCLUSIVE": errors.append("bootstrap_coverage")
    controls = load(root / "physical_controls.json")
    if controls.get("fpr_unit") != "pooled variable-PRN event after frozen calibration" or controls.get("base_pooled_event_n", 0) < 1:
        errors.append("physical_control_fpr_unit")
    tests = (root / "test_report.txt").read_text(encoding="utf-8")
    if "exit_code=0" not in tests or "passed" not in tests: errors.append("tests")
    raw = independent_l20(root); score = independent_score(root)
    if raw["status"] != "PASS": errors.append("raw_recompute")
    if score["status"] != "PASS": errors.append("score_recompute")
    report = {"schema": "acaf_nf_stage1_r3_verifier.v1", "status": "PASS" if not errors else "FAIL", "errors": errors,
              "producer_imported": False, "independent_raw_recomputation_performed": True, "representative_raw_l20": raw,
              "representative_neural_score": score, "freeze_hashes_recomputed": True, "raw_tracker_binding_checked": True,
              "tracker_mat_binding": tracker_checks,
              "leakage_boundaries_checked": True, "ds7_ds8_family_checked": True, "b0_alignment_checked": True}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("artifact", type=Path); args = parser.parse_args(); root = args.artifact.resolve()
    report = verify(root); dump(root / "verification_report.json", report)
    files = {p.relative_to(root).as_posix(): {"sha256": digest(p), "size_bytes": p.stat().st_size} for p in root.rglob("*") if p.is_file() and p.name != "checksums.json"}
    dump(root / "checksums.json", {"algorithm": "sha256", "files": files})
    print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
