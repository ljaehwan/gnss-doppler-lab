#!/usr/bin/env python3
"""Fail-closed independent verifier for the real-data R2C Stage-0 artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.r2c_gnss import (  # noqa: E402
    SourceSupport, artifact_hashes, assign_attack_phase, assign_normal_split,
    build_empirical_template, empirical_template_hash, sha256_file, sustained_alarms, write_json,
)

NAMES = ("cleanStatic", "cleanDynamic", "DS1", "DS2", "DS3", "DS7", "DS8")
ONSET = {"DS1": 100.0, "DS2": 100.0, "DS3": 100.0, "DS7": 110.0, "DS8": 110.0}
DETECTORS = ("B0", "A1", "A2", "A3", "A4", "Full R2C-GNSS", "Power-only")
SOURCE_FILES = ("src/gnss_doppler_lab/r2c_gnss.py", "scripts/run_r2c_gnss_stage0.py",
                "scripts/verify_r2c_gnss_stage0.py", "configs/r2c_gnss_stage0.json",
                "configs/r2c_clean_dynamic_geometry_receiver.conf")
REQUIRED = {"README.md", "config.json", "freeze.json", "provenance.json", "input_validity.json",
    "training_summary.json", "thresholds.json", "scenario_metrics.csv", "ablation_metrics.csv",
    "per_epoch_scores.csv", "bootstrap_comparisons.json", "gain_invariance.json",
    "phase_invariance.json", "noise_control.json", "multipath_control.json",
    "second_source_injection.json", "relation_destruction.json", "decision.json",
    "verification.json", "hashes.json", "plots/relation_control.png",
    "plots/relation_control_source.csv"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def as_bool(value):
    if str(value) not in ("True", "False"):
        raise ValueError(f"invalid boolean {value}")
    return str(value) == "True"


def close(left, right, tolerance=1e-10):
    return np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance)


def roc_auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    pos, neg = scores[labels], scores[~labels]
    if not len(pos) or not len(neg):
        return None
    return float(sum((value > neg).sum() + 0.5 * (value == neg).sum() for value in pos) /
                 (len(pos) * len(neg)))


def pr_auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    if not labels.any() or labels.all():
        return None
    ranked = labels[np.argsort(-scores, kind="stable")]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / np.sum(ranked))


def pauc(labels, scores, maximum=0.05):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    ranked = labels[np.argsort(-scores, kind="stable")]
    tpr, fpr = np.r_[0.0, np.cumsum(ranked) / positives], np.r_[0.0, np.cumsum(~ranked) / negatives]
    keep = fpr <= maximum
    x, y = fpr[keep], tpr[keep]
    if x[-1] < maximum:
        index = int(np.argmax(fpr > maximum))
        value = y[-1] + (tpr[index] - y[-1]) * (maximum - x[-1]) / (fpr[index] - x[-1])
        x, y = np.r_[x, maximum], np.r_[y, value]
    return float(np.trapezoid(y, x) / maximum)


def selected_support(path):
    with np.load(path) as z:
        times = np.asarray(z["time_s"], float)
        prns = np.asarray(z["prn"], np.int64)
        bins = np.floor(times / 0.5).astype(np.int64)
        _, indices = np.unique(bins * 64 + prns, return_index=True)
        indices = np.sort(indices)
        iq = z["complex_iq"][indices]
    values = iq[:, :, 0].astype(float) + 1j * iq[:, :, 1].astype(float)
    output = {}
    for bin_index in np.unique(bins[indices]):
        group = indices[bins[indices] == bin_index]
        output[int(bin_index)] = (float(times[group].min()), float(times[group].max()))
    return output, values, times[indices], indices, prns[indices]


def row_identity(time_s, prn, value):
    digest = hashlib.sha256()
    digest.update(np.asarray([time_s], dtype="<f8").tobytes())
    digest.update(np.asarray([prn], dtype="<i8").tobytes())
    digest.update(np.column_stack((value.real, value.imag)).astype("<f8").tobytes())
    return digest.hexdigest()


def deduplicated_support(raw_support):
    _, clean_values, clean_times, _, clean_prns = raw_support["cleanStatic"]
    source = (clean_times <= 300.0) | ((clean_times >= 320.0) & (clean_times <= 400.0))
    hashes = {row_identity(t, p, y) for t, p, y in zip(clean_times[source], clean_prns[source], clean_values[source])}
    output = {}
    for name, packed in raw_support.items():
        _, values, times, indices, prns = packed
        keep = np.ones(len(times), bool) if name == "cleanStatic" else np.asarray([
            row_identity(t, p, y) not in hashes for t, p, y in zip(times, prns, values)])
        bins = np.floor(times[keep] / .5).astype(int); support = {}
        for bin_index in np.unique(bins):
            chosen = times[keep][bins == bin_index]
            support[int(bin_index)] = (float(chosen.min()), float(chosen.max()))
        output[name] = (support, values[keep], times[keep], indices[keep], prns[keep])
    return output


def compare_recomputed_artifact(artifact, reproduced):
    excluded = {"hashes.json", "verification.json", "provenance.json", "README.md",
                "plots/relation_control.png"}
    errors = []
    for relative in sorted(REQUIRED - excluded):
        if not (reproduced / relative).is_file() or sha256_file(artifact / relative) != sha256_file(reproduced / relative):
            errors.append(f"independent recomputation mismatch: {relative}")
    return errors


def verify(artifact, check_external=True, full_recompute=False):
    errors = []
    missing = sorted(name for name in REQUIRED if not (artifact / name).is_file())
    if missing:
        return [f"missing files: {missing}"]
    config = load_json(artifact / "config.json")
    freeze = load_json(artifact / "freeze.json")
    provenance = load_json(artifact / "provenance.json")
    validity = load_json(artifact / "input_validity.json")
    thresholds = load_json(artifact / "thresholds.json")
    decision = load_json(artifact / "decision.json")
    training = load_json(artifact / "training_summary.json")
    bootstrap = load_json(artifact / "bootstrap_comparisons.json")

    if load_json(artifact / "hashes.json").get("files") != artifact_hashes(artifact):
        errors.append("artifact hash mismatch")

    current_files = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    current_bundle = {"schema": "gnss-doppler-lab.r2c-source-bundle.v1", "files": current_files,
                      "bundle_sha256": canonical_hash(current_files)}
    for owner, value in (("freeze", freeze.get("source_bundle")),
                         ("provenance", provenance.get("source_bundle"))):
        if value != current_bundle:
            errors.append(f"{owner} source bundle does not exactly match current executable files")
    config_hash = sha256_file(ROOT / "configs/r2c_gnss_stage0.json")
    if freeze.get("config_sha256") != config_hash or provenance.get("config_sha256") != config_hash:
        errors.append("config hash mismatch")
    if provenance.get("freeze_sha256") != sha256_file(artifact / "freeze.json"):
        errors.append("freeze hash mismatch")
    if provenance.get("generation_policy") != "clean-source-commit-v1: exact executable source bundle and immediate artifact-only child commit":
        errors.append("invalid generation policy")
    if provenance.get("executable_source_clean") is not True:
        errors.append("artifact was not generated from clean executable source")
    if config.get("schema") != "gnss-doppler-lab.r2c-gnss-stage0-config.v1":
        errors.append("artifact config schema mismatch")
    if not freeze.get("written_before_attack_score_computation") or not freeze.get("no_attack_label_tuning"):
        errors.append("invalid attack-independent freeze")
    if not validity.get("frozen_before_attack_evaluation") or validity.get("attack_outcomes_inspected_for_tuning"):
        errors.append("input gate/tuning violation")
    if provenance.get("branch") != "research/r2c-gnss-stage0":
        errors.append("wrong branch")
    if provenance.get("frozen_base_commit") != "461eb4dc7bb794e719295daf028f6811658ba37f":
        errors.append("wrong frozen base")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    parent = subprocess.check_output(["git", "rev-parse", "HEAD^"], cwd=ROOT, text=True).strip()
    if provenance.get("source_commit_at_generation") not in (current, parent):
        errors.append("generation commit must be current HEAD or its immediate artifact-only parent")

    template = freeze.get("template", {})
    try:
        template_values = np.asarray(template["values_real"], float) + 1j * np.asarray(template["values_imag"], float)
        if empirical_template_hash(template_values) != template.get("template_sha256"):
            errors.append("empirical template hash mismatch")
    except (KeyError, ValueError, TypeError) as exc:
        errors.append(f"invalid empirical template: {exc}")
        template_values = None
    if template.get("fit_role") != "cleanStatic normal_train" or set(template.get("forbidden_fit_sources", [])) != set(NAMES[1:]):
        errors.append("template fit-role contamination")
    if training.get("DS7_fit_or_calibration_uses") != 0 or training.get("cleanDynamic_fit_or_calibration_uses") != 0:
        errors.append("DS7/cleanDynamic fit contamination")
    if training.get("template_sha256") != template.get("template_sha256"):
        errors.append("training/template hash mismatch")

    raw_support = {}
    if check_external:
        for name, dataset in validity.get("datasets", {}).items():
            path, manifest_path = Path(dataset["resolved_path"]), Path(dataset["manifest_path"])
            if not path.is_file() or sha256_file(path) != dataset.get("npz_sha256"):
                errors.append(f"external NPZ hash mismatch: {name}")
                continue
            if not manifest_path.is_file() or sha256_file(manifest_path) != dataset.get("manifest_sha256"):
                errors.append(f"external manifest hash mismatch: {name}")
                continue
            manifest = load_json(manifest_path)
            feature = manifest.get("feature_schema", {})
            if manifest.get("output", {}).get("sha256") != dataset.get("npz_sha256"):
                errors.append(f"manifest output binding mismatch: {name}")
            if feature.get("tap_order") != ["E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4"] or feature.get("component_order") != ["I", "Q"]:
                errors.append(f"external feature schema mismatch: {name}")
            receiver = Path(dataset["receiver_config_path"])
            if (not receiver.is_file() or sha256_file(receiver) != dataset.get("receiver_config_sha256") or
                    dataset.get("receiver_config_sha256") != dataset.get("manifest_receiver_config_sha256")):
                errors.append(f"receiver config provenance mismatch: {name}")
            elif "Tracking_1C.tap_spacing_chips=0.125" not in receiver.read_text():
                errors.append(f"tap spacing mismatch: {name}")
            raw_support[name] = selected_support(path)
        if "cleanStatic" in raw_support and template_values is not None:
            _, values, times, indices, _ = raw_support["cleanStatic"]
            mask = times <= 300.0
            rebuilt, metadata = build_empirical_template(values[mask], ["normal_train"] * int(mask.sum()))
            index_hash = hashlib.sha256(np.asarray(indices[mask], dtype="<i8").tobytes()).hexdigest()
            if empirical_template_hash(rebuilt) != template.get("template_sha256") or index_hash != template.get("support_selected_index_sha256"):
                errors.append("empirical template does not reproduce from cleanStatic normal_train")

        raw_support = deduplicated_support(raw_support)
    rows = list(csv.DictReader((artifact / "per_epoch_scores.csv").open()))
    scenario_rows = list(csv.DictReader((artifact / "scenario_metrics.csv").open()))
    ablation_rows = list(csv.DictReader((artifact / "ablation_metrics.csv").open()))
    if not rows or not scenario_rows or not ablation_rows:
        errors.append("empty result table")
        return errors
    if any(str(value).lower() in {"nan", "inf", "-inf"} for table in (rows, scenario_rows, ablation_rows)
           for row in table for value in row.values()):
        errors.append("nonfinite result table value")
    missing_detectors = [detector for detector in DETECTORS if detector not in rows[0]]
    if missing_detectors:
        errors.append(f"missing detector score columns: {missing_detectors}")
        return errors

    # Recompute support/phase boundaries and compare to external selected rows when available.
    for row in rows:
        name = row["scenario"]
        start, end = float(row["source_start_s"]), float(row["source_end_s"])
        support = SourceSupport(start, end, name)
        expected = (assign_normal_split(support) if name == "cleanStatic" else
                    "external_normal" if name == "cleanDynamic" else assign_attack_phase(support, ONSET[name]))
        if row["phase"] != expected or not close(row["availability_time_s"], end):
            errors.append(f"split/onset/support mismatch: {name}:{row['time_bin']}")
            break
        if row["phase"] == "normal_calibration" and start < 320.0:
            errors.append("pre-320 source entered calibration")
        if row["phase"] in ("post", "persistent") and start < ONSET[name]:
            errors.append("pre-onset source entered post")
        if name in raw_support:
            expected_support = raw_support[name][0].get(int(row["time_bin"]))
            if expected_support is None or not close(start, expected_support[0]) or not close(end, expected_support[1]):
                errors.append(f"per-epoch support not bound to selected external rows: {name}:{row['time_bin']}")
                break

    calibration = [row for row in rows if row["scenario"] == "cleanStatic" and row["phase"] == "normal_calibration"]
    if thresholds.get("source") != "cleanStatic normal_calibration only" or thresholds.get("method") != "higher" or thresholds.get("comparison") != "strict score > threshold":
        errors.append("threshold contamination/policy")
    if thresholds.get("source_epoch_count") != len(calibration):
        errors.append("threshold calibration count mismatch")
    for detector in DETECTORS:
        detector_calibration = [row for row in calibration if as_bool(row.get(detector + "_valid", "True"))]
        for quantile in (0.99, 0.995):
            expected = np.quantile([float(row[detector]) for row in detector_calibration], quantile, method="higher")
            if not close(expected, thresholds["values"][detector][str(quantile)]):
                errors.append(f"threshold derivation mismatch: {detector}:{quantile}")
        threshold = float(thresholds["values"][detector]["0.99"])
        for row in rows:
            if as_bool(row[detector + "_q99_alarm"]) != (float(row[detector]) > threshold):
                errors.append(f"strict alarm mismatch: {row['scenario']}:{detector}")
                break

    # Recompute sustained alarms with recording/gap/phase/transition resets.
    for name in NAMES:
        subset = [row for row in rows if row["scenario"] == name]
        for detector in DETECTORS:
            expected = sustained_alarms([as_bool(row[detector + "_q99_alarm"]) for row in subset],
                [name] * len(subset), [float(row["source_end_s"]) for row in subset],
                [row["phase"] for row in subset], 0.5)
            actual = np.asarray([as_bool(row[detector + "_q99_sustained"]) for row in subset])
            if not np.array_equal(expected, actual):
                errors.append(f"sustained alarm mismatch: {name}:{detector}")

    metric_map = {(row["scenario"], row["detector"]): row for row in scenario_rows}
    mandatory = {"roc_auc", "pr_auc", "normalized_pauc_fpr_lte_0.05",
                 "strict_q99_detection_rate", "sustained_q99_detection_rate",
                 "first_sustained_alarm_delay_s", "persistent_alarm_ratio"}
    for name in NAMES:
        usable = [row for row in rows if row["scenario"] == name and
                  row["phase"] not in ("transition_excluded", "excluded_guard_or_boundary")]
        for detector in DETECTORS:
            usable_detector = [row for row in usable if as_bool(row.get(detector + "_valid", "True"))]
            metric = metric_map.get((name, detector))
            if metric is None:
                errors.append(f"missing scenario metrics: {name}:{detector}")
                continue
            if metric.get("status") == "EVALUATED" and any(not metric.get(key) for key in mandatory):
                errors.append(f"EVALUATED missing mandatory metric: {name}:{detector}")
            if name.startswith("DS"):
                use = [row for row in usable_detector if row["phase"] in ("stable_pre", "post", "persistent")]
                labels = [row["phase"] in ("post", "persistent") for row in use]
                expected_metrics = {"roc_auc": roc_auc(labels, [float(row[detector]) for row in use]),
                    "pr_auc": pr_auc(labels, [float(row[detector]) for row in use]),
                    "normalized_pauc_fpr_lte_0.05": pauc(labels, [float(row[detector]) for row in use])}
                post = [row for row in use if row["phase"] in ("post", "persistent")]
                persistent = [row for row in use if row["phase"] == "persistent"]
                expected_metrics.update({
                    "strict_q99_detection_rate": np.mean([as_bool(row[detector + "_q99_alarm"]) for row in post]),
                    "sustained_q99_detection_rate": np.mean([as_bool(row[detector + "_q99_sustained"]) for row in post]),
                    "persistent_alarm_ratio": np.mean([as_bool(row[detector + "_q99_alarm"]) for row in persistent])})
                for key, value in expected_metrics.items():
                    if value is None or not close(metric[key], value):
                        errors.append(f"scenario metric mismatch: {name}:{detector}:{key}")
                alarmed = [row for row in post if as_bool(row[detector + "_q99_sustained"])]
                expected_delay = min(float(row["availability_time_s"]) for row in alarmed) - ONSET[name] if alarmed else None
                if expected_delay is None:
                    if metric["first_sustained_alarm_delay_s"] != "UNAVAILABLE_NO_SUSTAINED_ALARM":
                        errors.append(f"first alarm availability mismatch: {name}:{detector}")
                elif not close(metric["first_sustained_alarm_delay_s"], expected_delay):
                    errors.append(f"first alarm availability mismatch: {name}:{detector}")
            else:
                rate = np.mean([as_bool(row[detector + "_q99_alarm"]) for row in usable_detector])
                if not close(metric["strict_q99_detection_rate"], rate):
                    errors.append(f"normal FPR mismatch: {name}:{detector}")

    if bootstrap.get("repetitions") != 2000 or bootstrap.get("seed") != 20260803 or bootstrap.get("iid_fallback") is not False:
        errors.append("bootstrap contract mismatch")
    for key, comparison in bootstrap.get("comparisons", {}).items():
        if comparison.get("status") == "EVALUATED" and comparison.get("repetitions") != 2000:
            errors.append(f"bootstrap repetitions mismatch: {key}")
        if comparison.get("status", "").startswith("UNAVAILABLE") and comparison.get("repetitions") != 0:
            errors.append(f"unavailable bootstrap falsely reports repetitions: {key}")

    gains = load_json(artifact / "gain_invariance.json")
    if gains.get("status") != "REAL_NORMAL_RECOMPUTED_SCORES_AND_ALARMS":
        errors.append("gain scores/alarms were not recomputed")
    for gain in ("0.5", "0.75", "1.0", "1.5", "2.0"):
        for detector in ("A1", "A2"):
            entry = gains.get("by_gain", {}).get(gain, {}).get(detector, {})
            if entry.get("recomputed_epoch_count") != len(calibration) or not (0 <= entry.get("alarm_agreement_vs_gain_1", -1) <= 1):
                errors.append(f"invalid gain agreement evidence: {gain}:{detector}")

    verdict = decision.get("verdict")
    if verdict not in {"PHYSICS_SUPPORTED", "NOT_SUPPORTED", "DATA_INVALID", "INCONCLUSIVE"}:
        errors.append("invalid verdict")
    full = next((row for row in ablation_rows if row["detector"] == "Full R2C-GNSS"), None)
    if verdict == "PHYSICS_SUPPORTED" and (not full or full["status"] != "EVALUATED"):
        errors.append("PHYSICS_SUPPORTED without evaluated Full")
    if validity.get("decision") == "VALID_COMPLEX_A1_A2; FULL_GEOMETRY_DATA_INVALID" and verdict != "DATA_INVALID":
        errors.append("decision inconsistent with required LOS data invalidity")
    if decision.get("old_result_status") != "SUPERSEDED_BY_EXTERNAL_DATA_DISCOVERY":
        errors.append("superseded provenance absent")
    if full_recompute and not errors:
        runtime = config.get("runtime", {})
        command = [sys.executable, str(ROOT / "scripts/run_r2c_gnss_stage0.py"),
                   "--config", str(ROOT / "configs/r2c_gnss_stage0.json")]
        with tempfile.TemporaryDirectory(prefix="r2c-independent-") as temporary:
            reproduced = Path(temporary) / "artifact"; command += ["--output", str(reproduced)]
            for name in NAMES:
                command += ["--input", f"{name}={runtime.get('inputs', {}).get(name, '')}"]
                command += ["--geometry", f"{name}={runtime.get('geometry', {}).get(name, '')}"]
            completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
            if completed.returncode:
                errors.append("independent full recomputation failed: " + completed.stderr[-1000:])
            else:
                errors.extend(compare_recomputed_artifact(artifact, reproduced))
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/r2c_gnss_stage0")
    parser.add_argument("--write-result", action="store_true")
    parser.add_argument("--skip-external-hashes", action="store_true")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    errors = verify(artifact, not args.skip_external_hashes, full_recompute=True)
    result = {"status": "PASS" if not errors else "FAIL", "errors": errors,
              "checked_required_files": len(REQUIRED),
              "external_hashes_checked": not args.skip_external_hashes,
              "hash_policy": "artifact excludes hashes.json/verification.json; exact clean source commit and full external-input recomputation required",
              "verifier_source_sha256": sha256_file(Path(__file__))}
    if args.write_result:
        write_json(artifact / "verification.json", result)
        write_json(artifact / "hashes.json", {"algorithm": "sha256", "files": artifact_hashes(artifact)})
    print(json.dumps(result, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
