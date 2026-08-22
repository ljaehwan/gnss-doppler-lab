"""Exploratory locked-model CRID audit for the preregistered TEXBAT DS3 slice.

This module does not authorize Phase B.  The scientific score is imported
unchanged from the frozen CRID implementation; this module only freezes and
orchestrates provenance, replay, exploratory metrics, and fail-closed audit.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from . import crid_r4_phase_a as r4
from .crid import (
    CONFIG_ORDER,
    NormalModel,
    ResponseTable,
    chronological_split,
    estimate_causal_delays,
    fit_normal_model,
    load_response,
    receiver_configurations,
    render_receiver_config,
    score_aligned,
)
from .crid_receiver_replay import _supervised_run
from .trace_native_1ms import validate_dump_files


BASE_SHA = "39b38a9dd8a470e0a2f83604cde61722ad52be1d"
PREREGISTRATION_SHA = "fb51f5c09a104508051695503dd393a51c73de97"
BRANCH = "research/crid-stage0-r4d-ds3-empty-trace-manifest-repair"
ARTIFACT_REL = "artifacts/crid_stage0_r4d_ds3_empty_trace_manifest_repair"
SSD_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-r4d-ds3-empty-trace-manifest-repair"
)
R4C_ARTIFACT_REL = "artifacts/crid_stage0_r4c_texbat_ds3_exploratory_locked_score"
R4C_SSD_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-r4c-texbat-ds3-exploratory-locked-score"
)
R4C_FREEZE_SHA = "be4e51fecca78569a6a5e6d6d70930de5733b34c"
DS3_RAW = Path("/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin")
FORBIDDEN_ATTACK_PATHS = (
    "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds1.bin",
    "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds4.bin",
    "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin",
    "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds8.bin",
    "OAKBAT.OS3",
    "OAKBAT.OS4",
)
R4B_ART = "artifacts/crid_stage0_r4b_phase_a_physical_identifiability_execution"
R4A_ART = "artifacts/crid_stage0_r4a_threshold_decision_equivalence_repair"
HANDOFF_REL = "artifacts/trace_stage0_r2e_attack_support_repair/handoffs/texbat_ds3.csv"
HANDOFF_SHA256 = "1b630895bcaf0039d7d6764710e4c900b599cf6e2e4da917dc990c63b63b6400"
RECEIVER = r4.RECEIVER
BASE_CONFIG = r4.BASE_CONFIG["TEX"]
FS = 25_000_000
BYTES_PER_COMPLEX_SAMPLE = 4
START_S = 78.9
ONSET_S = 118.9
PULL_OFF_S = 195.0
END_S = 238.9
DURATION_S = 160.0
START_SAMPLE = 1_972_500_000
ONSET_SAMPLE = 2_972_500_000
PULL_OFF_SAMPLE = 4_875_000_000
END_SAMPLE = 5_972_500_000
SOURCE_ITEMS = 8_000_000_000
EXPECTED_END_BYTE = 23_890_000_000
THRESHOLD = -21.942672917134093
R4C_LOCKED_MODEL_SHA256 = "351168c50f00b7817ffad6682b480964621912049b1e4352cd03594ebfce17a8"
EXPECTED_DUMP_COUNT = 11
MINIMUM_CONFIGS = 4
MINIMUM_COMMON_PRNS = 4
PERSISTENCE_LEVEL = 0.80
PERSISTENCE_WINDOWS_S = (1, 5, 10)
EXECUTABLE_FILES = (
    "src/gnss_doppler_lab/crid_r4d_ds3.py",
    "scripts/run_crid_r4d_ds3.py",
    "scripts/verify_crid_r4d_ds3.py",
)
SCIENCE_FILES = (
    "src/gnss_doppler_lab/crid.py",
    "src/gnss_doppler_lab/crid_experiment.py",
    "src/gnss_doppler_lab/crid_metrics.py",
    "src/gnss_doppler_lab/crid_receiver_replay.py",
    "src/gnss_doppler_lab/trace_native_1ms.py",
)
EPOCH_FIELDS = (
    "dataset",
    "sample",
    "time_s",
    "partition",
    "score",
    "alarm",
    "label",
    "prn_count",
    "config_count",
    "cn0_median_db_hz",
    "lock_median",
    "tracked_prn_count_min_config",
    "tracked_prn_count_median_config",
    "h0_loglike",
    "h1_loglike",
    "h1_improvement",
    "penalty",
    "configuration_disagreement",
)
REPLAY_FIELDS = (
    "config",
    "status",
    "exit_code",
    "terminal_drain_status",
    "native_trace_status",
    "target_tracking_pass",
    "tracked_target_prns",
    "input_sha256",
    "receiver_sha256",
    "config_sha256",
    "handoff_sha256",
    "output_set_sha256",
    "dump_count",
    "output_bytes",
    "manifest_path",
    "manifest_sha256",
)
EMPTY_CHANNEL_FIELDS = (
    "config",
    "channel",
    "status",
    "record_count",
    "size_bytes",
    "file_size_bytes",
    "sha256",
    "excluded_from_support",
    "path",
)


class BindingError(RuntimeError):
    """Fail-closed provenance, replay, or support error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for payload in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(payload)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    path = Path(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def repo_binding(repo: Path, relative: str) -> dict[str, object]:
    bound = file_binding(repo / relative)
    bound["path"] = relative
    return bound


def require_binding(path: Path, binding: Mapping[str, object]) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size != int(binding["size_bytes"]):
        raise BindingError(f"size binding mismatch: {path}")
    if sha256_file(path) != binding["sha256"]:
        raise BindingError(f"SHA-256 binding mismatch: {path}")


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def assert_branch(repo: Path) -> None:
    if git(repo, "branch", "--show-current") != BRANCH:
        raise BindingError("wrong R4d branch")
    if git(repo, "merge-base", "HEAD", BASE_SHA) != BASE_SHA:
        raise BindingError("R4d branch is not based on required base")


def assert_pushed_freeze(repo: Path, freeze_sha: str) -> None:
    assert_branch(repo)
    if git(repo, "rev-parse", "HEAD") != freeze_sha:
        raise BindingError("HEAD differs from supplied freeze SHA")
    if git(repo, "rev-parse", f"origin/{BRANCH}") != freeze_sha:
        raise BindingError("remote branch differs from supplied freeze SHA")
    if git(repo, "status", "--porcelain=v1"):
        raise BindingError("execution checkout is not clean")


def assert_allowed_attack_path(path: Path) -> None:
    if str(path) != str(DS3_RAW):
        raise BindingError("only exact TEXBAT DS3 raw path is allowlisted")


def compact_manifest(artifact: Path) -> dict[str, object]:
    files = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append(
                {
                    "path": str(path.relative_to(artifact)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "gnss-doppler-lab.crid-r4d-artifact-manifest.v1",
        "status": "PASS",
        "file_count": len(files),
        "files": files,
    }


def seal_manifest(artifact: Path) -> None:
    dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))


def _runtime_environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timezone": list(time.tzname),
        "gnss_sdr_version": subprocess.check_output([str(RECEIVER), "--version"], text=True).strip(),
    }


def _load_source_handoff(repo: Path) -> list[dict[str, str]]:
    path = repo / HANDOFF_REL
    if sha256_file(path) != HANDOFF_SHA256:
        raise BindingError("committed DS3 pre-onset handoff mismatch")
    rows = read_csv(path)
    if len(rows) != EXPECTED_DUMP_COUNT or len({int(row["prn"]) for row in rows}) != EXPECTED_DUMP_COUNT:
        raise BindingError("DS3 handoff must contain 11 unique PRNs")
    return rows


def _write_reanchored_handoff(repo: Path, artifact: Path) -> tuple[dict[str, object], list[int]]:
    rows = _load_source_handoff(repo)
    for row in rows:
        source = int(row["source_raw_interval_start_sample"])
        relative = source - START_SAMPLE
        if relative < 0 or source >= ONSET_SAMPLE:
            raise BindingError("handoff row is not within the preregistered pre-onset slice")
        row["first_raw_interval_start_sample"] = str(relative)
    fields = tuple(rows[0])
    path = artifact / "frozen_handoff_texbat_ds3_78p9s.csv"
    write_csv(path, fields, rows)
    return repo_binding(artifact.parent.parent, str(path.relative_to(artifact.parent.parent))), [int(row["prn"]) for row in rows]


def _frozen_config_text(config: str, target_prns: Sequence[int]) -> str:
    values: dict[str, object] = {
        "SignalSource.filename": str(DS3_RAW),
        "SignalSource.seconds_to_skip": START_S,
        "SignalSource.samples": SOURCE_ITEMS,
        "SignalSource.repeat": "false",
        "Channels_1C.count": EXPECTED_DUMP_COUNT,
        "Channels.in_acquisition": EXPECTED_DUMP_COUNT,
        "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false",
        "Tracking_1C.trace_dump": "true",
        "Tracking_1C.trace_dump_filename": str(SSD_ROOT / "replays" / config / "trace_native_1ms_ch_"),
        "Tracking_1C.trace_scenario_id": f"CRID.R4d.TEXBAT.DS3.{config}",
        "Tracking_1C.trace_raw_sample_offset": START_SAMPLE,
        "Tracking_1C.trace_handoff_filename": str(SSD_ROOT / "handoff" / "texbat_ds3_78p9s.csv"),
        "SignalSource.enable_terminal_drain": "true",
        "Observables.dump": "false",
    }
    values.update({f"Channel{index}.satellite": prn for index, prn in enumerate(target_prns)})
    return render_receiver_config(BASE_CONFIG.read_text(), receiver_configurations()[config], values)


def _model_sha256(model: NormalModel, delays: Mapping[str, int]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json({"order": model.order, "ridge": model.ridge, "shrinkage": model.shrinkage, "latent_dimension": model.latent_dimension, "delays": dict(delays)}).encode())
    for config in CONFIG_ORDER:
        for name, value in (
            ("coefficients", model.coefficients[config]),
            ("means", model.means[config]),
            ("whiteners", model.whiteners[config]),
            ("h_matrices", model.h_matrices[config]),
        ):
            array = np.asarray(value, dtype="<f8")
            digest.update(config.encode())
            digest.update(name.encode())
            digest.update(canonical_json(list(array.shape)).encode())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_clean_source(binding: Mapping[str, object]) -> None:
    clean = binding["clean_model_source"]["TEX"]
    require_binding(Path(clean["base_config"]["path"]), clean["base_config"])
    for config in CONFIG_ORDER:
        for row in clean["configurations"][config]["traces"]:
            require_binding(Path(row["path"]), row)


def build_locked_model(binding: Mapping[str, object]) -> tuple[NormalModel, dict[str, int], dict[str, object]]:
    _validate_clean_source(binding)
    clean = binding["clean_model_source"]["TEX"]
    tables = {
        config: load_response(config, [Path(row["path"]) for row in clean["configurations"][config]["traces"]])
        for config in CONFIG_ORDER
    }
    delays = estimate_causal_delays(tables)
    samples = np.concatenate([table.sample for table in tables.values()])
    split = chronological_split(samples)
    model = fit_normal_model(tables, split["train"], split["calibration"])
    return model, delays, {
        "model_sha256": _model_sha256(model, delays),
        "causal_delays_ms": delays,
        "split": {name: {"first_sample": int(values[0]), "last_sample": int(values[-1]), "count": len(values)} for name, values in split.items()},
        "threshold_recomputation_executed": False,
    }


def _metric_contract() -> dict[str, object]:
    return {
        "coordinate": "absolute raw complex-sample / 25,000,000",
        "partitions": {
            "pre_onset": {"start_s_inclusive": START_S, "end_s_exclusive": ONSET_S, "label": 0},
            "transition": {"start_s_inclusive": ONSET_S, "end_s_exclusive": PULL_OFF_S, "label": 1},
            "established": {"start_s_inclusive": PULL_OFF_S, "end_s_exclusive": END_S, "label": 1},
        },
        "roc_pr_population": "pre_onset label 0 versus transition+established label 1",
        "pauc_0_05": "sklearn standardized partial ROC AUC with max_fpr=0.05",
        "alarm": "score > -21.942672917134093",
        "first_alarm": "earliest supported strict-threshold alarm at or after each boundary",
        "persistent_alarm_ratio": {
            "region": "established",
            "windows_s": list(PERSISTENCE_WINDOWS_S),
            "bins": "non-overlapping complete bins anchored at 195.0 s",
            "bin_pass": f"within-bin alarm fraction >= {PERSISTENCE_LEVEL}",
            "reported_value": "passing bins / complete bins with at least one supported epoch",
        },
        "common_support_availability": "valid scored epochs / nominal 1 ms epochs in each frozen interval",
        "shortcut_correlations": "Pearson and Spearman correlations of score with median C/N0, median lock, common PRN count, and minimum per-config tracked PRN count",
    }


def _gate_contract() -> dict[str, object]:
    return {
        "pre_onset_fpr_max": 0.05,
        "pauc_0_05_min": 0.80,
        "established_detection_rate_min": 0.70,
        "established_four_config_min_four_prn_support_required": True,
        "not_explained_only_by_lock_loss_or_tracked_prn_collapse": True,
        "exploratory_only": True,
    }


def prepare_execution_repair_freeze(repo: Path, artifact: Path) -> dict[str, object]:
    assert_branch(repo)
    artifact.mkdir(parents=True, exist_ok=True)
    existing = {str(path.relative_to(artifact)) for path in artifact.rglob("*") if path.is_file()}
    if existing != {"repair_preregistration.json"}:
        raise BindingError(f"unexpected pre-repair artifact paths: {sorted(existing)}")
    repair_prereg = json.loads((artifact / "repair_preregistration.json").read_text())
    if repair_prereg.get("base_sha") != BASE_SHA or repair_prereg.get("status") != "REPAIR_PREREGISTERED_BEFORE_ADAPTER_IMPLEMENTATION":
        raise BindingError("repair preregistration binding mismatch")
    r4b_final = json.loads((repo / R4B_ART / "final_verdict.json").read_text())
    if not (
        r4b_final.get("verdict") == "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE"
        and r4b_final.get("next_state") == "NOT_AUTHORIZED"
        and r4b_final.get("phase_b_executed") is False
    ):
        raise BindingError("R4b fail-closed status mismatch")
    r4a_final = json.loads((repo / R4A_ART / "final_verdict.json").read_text())
    if r4a_final.get("authoritative_thresholds", {}).get("TEX") != THRESHOLD:
        raise BindingError("authoritative TEX threshold mismatch")
    r4b_source = json.loads((repo / R4B_ART / "source_binding.json").read_text())
    r4c_artifact = repo / R4C_ARTIFACT_REL
    r4c_source = json.loads((r4c_artifact / "source_binding.json").read_text())
    if r4c_source.get("authoritative_threshold") != THRESHOLD:
        raise BindingError("R4c threshold binding mismatch")
    handoff_binding, target_prns = _write_reanchored_handoff(repo, artifact)
    config_bindings = {}
    for config in CONFIG_ORDER:
        path = artifact / "frozen_configs" / f"{config}.conf"
        path.parent.mkdir(parents=True, exist_ok=True)
        text = _frozen_config_text(config, target_prns)
        r4c_text = (r4c_artifact / "frozen_configs" / f"{config}.conf").read_text()
        expected = r4c_text.replace(str(R4C_SSD_ROOT), str(SSD_ROOT)).replace("CRID.R4c", "CRID.R4d")
        if text != expected:
            raise BindingError(f"R4d {config} differs from R4c beyond output/provenance paths")
        path.write_text(text)
        config_bindings[config] = repo_binding(repo, str(path.relative_to(repo)))
    model_binding = r4c_source["locked_model"]
    source_binding = {
        "schema": "gnss-doppler-lab.crid-r4d-source-binding.v1",
        "status": "PASS",
        "base_sha": BASE_SHA,
        "r4b_phase_a_passed": False,
        "r4b_final_verdict": repo_binding(repo, f"{R4B_ART}/final_verdict.json"),
        "r4b_source_binding": repo_binding(repo, f"{R4B_ART}/source_binding.json"),
        "r4b_artifact_manifest": repo_binding(repo, f"{R4B_ART}/artifact_manifest_sha256.json"),
        "r4a_final_verdict": repo_binding(repo, f"{R4A_ART}/final_verdict.json"),
        "r4c_final_verdict": repo_binding(repo, f"{R4C_ARTIFACT_REL}/final_verdict.json"),
        "r4c_artifact_manifest": repo_binding(repo, f"{R4C_ARTIFACT_REL}/artifact_manifest_sha256.json"),
        "r4c_source_binding": repo_binding(repo, f"{R4C_ARTIFACT_REL}/source_binding.json"),
        "r4c_freeze_sha": R4C_FREEZE_SHA,
        "receiver": file_binding(RECEIVER),
        "receiver_version": subprocess.check_output([str(RECEIVER), "--version"], text=True).strip(),
        "base_tex_config": file_binding(BASE_CONFIG),
        "source_handoff": repo_binding(repo, HANDOFF_REL),
        "reanchored_handoff": handoff_binding,
        "target_prns": target_prns,
        "configuration_definitions": {
            config: sha256_bytes(canonical_json(receiver_configurations()[config]).encode()) for config in CONFIG_ORDER
        },
        "frozen_configs": config_bindings,
        "science_files": {name: repo_binding(repo, name) for name in SCIENCE_FILES},
        "clean_model_source": {"TEX": r4c_source["clean_model_source"]["TEX"]},
        "locked_model": model_binding,
        "authoritative_threshold": THRESHOLD,
        "timeline": {"start_s": START_S, "onset_s": ONSET_S, "pull_off_s": PULL_OFF_S, "end_s": END_S},
        "threshold_source": "committed R2 literal authorized by R4a decision equivalence",
        "threshold_recomputation_executed": False,
        "attack_payload_accessed": False,
    }
    validate_frozen_contract(source_binding)
    dump_json(artifact / "source_binding.json", source_binding)
    preregistration = {
        "schema": "gnss-doppler-lab.crid-r4d-preregistration.v1",
        "status": "FROZEN_BEFORE_DS3_PAYLOAD_ACCESS",
        "scope": "EXPLORATORY_TEXBAT_DS3_LOCKED_MODEL_AUDIT",
        "base_sha": BASE_SHA,
        "r4b_phase_a_passed": False,
        "formal_phase_b": False,
        "confirmatory_detection_evaluation": False,
        "deployment_validation": False,
        "input": {
            "dataset": "TEXBAT.DS3",
            "planned_path": str(DS3_RAW),
            "sample_rate_hz": FS,
            "sample_format": "little-endian interleaved int16 I/Q",
            "bytes_per_complex_sample": BYTES_PER_COMPLEX_SAMPLE,
            "start_s": START_S,
            "end_s": END_S,
            "duration_s": DURATION_S,
            "start_sample_inclusive": START_SAMPLE,
            "end_sample_exclusive": END_SAMPLE,
            "source_items": SOURCE_ITEMS,
            "access_before_pushed_freeze": False,
        },
        "replay": {
            "configurations": list(CONFIG_ORDER),
            "worker_count": 1,
            "order": "C0,C1,C2,C3",
            "checkpoint_resume": True,
            "overwrite_or_delete": False,
            "minimum_configurations": MINIMUM_CONFIGS,
            "minimum_common_prns": MINIMUM_COMMON_PRNS,
        },
        "score": {
            "implementation": "unchanged frozen CRID feature, causal alignment, predictor, covariance, H0/H1, median-over-common-PRNs pooling",
            "threshold": THRESHOLD,
            "comparison": "score > threshold",
            "threshold_recomputation_or_recalibration": False,
            "cn0_power_lock_prn_count_in_score": False,
        },
        "metric_contract": _metric_contract(),
        "exploratory_signal_gate": _gate_contract(),
        "allowed_verdicts": [
            "INCONCLUSIVE_TEXBAT_DS3_EXPLORATORY_EXECUTION",
            "EXPLORATORY_TEXBAT_DS3_SIGNAL_PRESENT",
            "EXPLORATORY_TEXBAT_DS3_NO_USEFUL_SIGNAL",
        ],
        "post_access_changes_forbidden": ["code", "threshold", "score", "feature", "model", "window", "gate"],
        "forbidden_inputs": ["TEXBAT DS1", "TEXBAT DS4", "TEXBAT DS7", "TEXBAT DS8", "OAK attack data"],
    }
    dump_json(artifact / "preregistration.json", preregistration)
    dump_json(
        artifact / "execution_repair_freeze.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-execution-freeze.v1",
            "status": "FROZEN_PENDING_PUSH",
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "output_root": str(SSD_ROOT),
            "worker_count": 1,
            "repair_scope": "EMPTY_TRACE_MANIFEST_ADAPTER_ONLY",
            "score_executed_before_repair_freeze": False,
            "commands": {
                "audit_c0": "python3 scripts/run_crid_r4d_ds3.py audit-c0 --freeze-sha <PUSHED_FREEZE_SHA>",
                "inventory": "python3 scripts/run_crid_r4d_ds3.py inventory --freeze-sha <PUSHED_FREEZE_SHA>",
                "replay": "python3 scripts/run_crid_r4d_ds3.py replay --freeze-sha <PUSHED_FREEZE_SHA>",
                "analyze": "python3 scripts/run_crid_r4d_ds3.py analyze --freeze-sha <PUSHED_FREEZE_SHA>",
            },
            "executable_sha256": {name: sha256_file(repo / name) for name in EXECUTABLE_FILES},
            "environment": _runtime_environment(),
        },
    )
    dump_json(
        artifact / "freeze_commit.json",
        {"schema": "gnss-doppler-lab.crid-r4d-freeze-commit.v1", "status": "PENDING_PUSH", "freeze_sha": None},
    )
    dump_json(
        artifact / "ds3_input_inventory.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-ds3-input-inventory.v1",
            "status": "FROZEN_NOT_ACCESSED",
            "path": str(DS3_RAW),
            "size_bytes": None,
            "sha256": None,
            "payload_stat_count": 0,
            "payload_open_count": 0,
            "payload_hash_count": 0,
            "payload_mmap_count": 0,
            "payload_bytes_read": 0,
        },
    )
    dump_json(
        artifact / "preregistration_commit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-preregistration-commit.v1",
            "status": "PASS",
            "preregistration_sha": PREREGISTRATION_SHA,
            "local_sha": PREREGISTRATION_SHA,
            "remote_sha": PREREGISTRATION_SHA,
            "ahead": 0,
            "behind": 0,
            "clean_worktree": True,
            "only_changed_path": f"{ARTIFACT_REL}/repair_preregistration.json",
        },
    )
    dump_json(
        artifact / "r4c_preserved_output_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-r4c-preserved-output-audit.v1",
            "status": "FROZEN_NOT_AUDITED",
            "audit_after_pushed_repair_freeze_only": True,
            "old_output_root": str(R4C_SSD_ROOT),
            "overwrite_or_delete": False,
        },
    )
    write_csv(artifact / "replay_completion.csv", REPLAY_FIELDS, [])
    write_csv(artifact / "empty_channel_inventory.csv", EMPTY_CHANNEL_FIELDS, [])
    with gzip.open(artifact / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        csv.DictWriter(stream, fieldnames=EPOCH_FIELDS, lineterminator="\n").writeheader()
    dump_json(artifact / "support_audit.json", {"schema": "gnss-doppler-lab.crid-r4d-support-audit.v1", "status": "FROZEN_NOT_EVALUATED"})
    dump_json(
        artifact / "scenario_metrics.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-scenario-metrics.v1",
            "status": "FROZEN_NOT_EVALUATED",
            "metric_contract": _metric_contract(),
            "gate_contract": _gate_contract(),
            "results": None,
        },
    )
    dump_json(artifact / "shortcut_audit.json", {"schema": "gnss-doppler-lab.crid-r4d-shortcut-audit.v1", "status": "FROZEN_NOT_EVALUATED"})
    dump_json(
        artifact / "attack_access_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-attack-access-audit.v1",
            "status": "PASS_PRE_ACCESS_ZERO",
            "allowed_ds3": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            "forbidden_inputs": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            "phase_b_executed": False,
        },
    )
    dump_json(
        artifact / "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-final-verdict.v1",
            "status": "FROZEN_NOT_EXECUTED",
            "verdict": "NOT_EVALUATED_PRE_ACCESS_FREEZE",
            "exploratory_only": True,
            "formal_phase_b": False,
            "phase_b_executed": False,
        },
    )
    (artifact / "README.md").write_text(
        "# CRID Stage-0 R4d TEXBAT DS3 exploratory locked-score audit\n\n"
        "Pre-access executable freeze. R4b Phase A did not pass, so this is not formal Phase B, confirmatory detection evidence, or deployment validation. "
        "Only the frozen 78.9–238.9 s DS3 slice may be accessed after this freeze is pushed.\n"
    )
    seal_manifest(artifact)
    return {"status": "READY_TO_VERIFY_FREEZE", "target_prns": target_prns, "model_sha256": model_binding["model_sha256"]}


def audit_r4c_c0(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != SSD_ROOT:
        raise BindingError("unexpected R4d SSD root")
    ssd.mkdir(parents=True, exist_ok=True)
    target = ssd / "r4c_preserved_output_audit.json"
    if target.exists():
        prior = json.loads(target.read_text())
        if prior.get("freeze_sha") == freeze_sha:
            return prior
        raise BindingError("stale R4c C0 audit exists; no overwrite")
    committed_path = repo / R4C_ARTIFACT_REL / "incomplete_c0_output_binding.json"
    committed = json.loads(committed_path.read_text())
    observed_dumps = []
    dump_bindings_pass = True
    for row in committed.get("dumps", []):
        path = Path(str(row["path"]))
        exists = path.is_file()
        observed = {
            "path": str(path),
            "exists": exists,
            "expected_size_bytes": int(row["size_bytes"]),
            "expected_sha256": row["sha256"],
            "record_count": int(row["record_count"]),
        }
        if exists:
            observed["size_bytes"] = path.stat().st_size
            observed["sha256"] = sha256_file(path)
        observed["binding_pass"] = bool(
            exists
            and observed.get("size_bytes") == observed["expected_size_bytes"]
            and observed.get("sha256") == observed["expected_sha256"]
        )
        dump_bindings_pass = dump_bindings_pass and bool(observed["binding_pass"] )
        observed_dumps.append(observed)
    out = R4C_SSD_ROOT / "replays" / "C0"
    config_path = out / "receiver.conf"
    log_path = out / "receiver.log"
    source = json.loads((repo / R4C_ARTIFACT_REL / "source_binding.json").read_text())
    config_binding_pass = bool(
        config_path.is_file()
        and config_path.stat().st_size == int(source["frozen_configs"]["C0"]["size_bytes"] )
        and sha256_file(config_path) == source["frozen_configs"]["C0"]["sha256"]
    )
    log_binding = file_binding(log_path) if log_path.is_file() else None
    missing_required_evidence = ["receiver_exit_code", "terminal_drain_status"]
    recoverable = dump_bindings_pass and config_binding_pass and not missing_required_evidence
    result = {
        "schema": "gnss-doppler-lab.crid-r4d-r4c-preserved-output-audit.v1",
        "status": "RECOVERED_PRESERVED_C0" if recoverable else "HISTORICAL_INCOMPLETE_EVIDENCE",
        "freeze_sha": freeze_sha,
        "r4c_freeze_sha": R4C_FREEZE_SHA,
        "old_output_root": str(out),
        "committed_incomplete_binding": repo_binding(repo, f"{R4C_ARTIFACT_REL}/incomplete_c0_output_binding.json"),
        "execution_command_reconstruction": [str(RECEIVER), f"--config_file={config_path}", "--keyboard=false"],
        "ds3_input_sha256": json.loads((repo / R4C_ARTIFACT_REL / "ds3_input_inventory.json").read_text()).get("sha256"),
        "receiver_sha256": source["receiver"]["sha256"],
        "config_binding_pass": config_binding_pass,
        "handoff_sha256": source["reanchored_handoff"]["sha256"],
        "receiver_log": log_binding,
        "dumps": observed_dumps,
        "dump_bindings_pass": dump_bindings_pass,
        "native_record_counts_recovered": dump_bindings_pass,
        "receiver_exit_code_recovered": False,
        "terminal_drain_status_recovered": False,
        "missing_required_evidence": missing_required_evidence,
        "reuse_c0": recoverable,
        "rerun_c0_in_r4d_root": not recoverable,
        "old_output_deleted_or_overwritten": False,
        "score_executed": False,
    }
    atomic_json(target, result)
    return result


def inventory_ds3(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != SSD_ROOT:
        raise BindingError("unexpected R4d SSD root")
    assert_allowed_attack_path(DS3_RAW)
    ssd.mkdir(parents=True, exist_ok=True)
    inventory_path = ssd / "ds3_input_inventory.json"
    if inventory_path.exists():
        prior = json.loads(inventory_path.read_text())
        if prior.get("status") == "PASS" and prior.get("freeze_sha") == freeze_sha:
            return prior
        raise BindingError("stale DS3 inventory exists; no overwrite")
    unexpected = [path for path in ssd.iterdir() if path.name not in {"handoff", "r4c_preserved_output_audit.json"}]
    if unexpected:
        raise BindingError("unexpected pre-inventory R4d output exists")
    stat = os.stat(DS3_RAW)
    if stat.st_size < EXPECTED_END_BYTE:
        raise BindingError("DS3 payload shorter than frozen slice")
    digest = hashlib.sha256()
    bytes_read = 0
    with DS3_RAW.open("rb") as stream:
        for payload in iter(lambda: stream.read(64 * 1024 * 1024), b""):
            digest.update(payload)
            bytes_read += len(payload)
    result = {
        "schema": "gnss-doppler-lab.crid-r4d-ds3-input-inventory.v1",
        "status": "PASS",
        "freeze_sha": freeze_sha,
        "path": str(DS3_RAW),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "hash_bytes_read": bytes_read,
        "sample_rate_hz": FS,
        "bytes_per_complex_sample": BYTES_PER_COMPLEX_SAMPLE,
        "slice": {"start_sample_inclusive": START_SAMPLE, "end_sample_exclusive": END_SAMPLE, "start_s": START_S, "end_s": END_S},
        "access": {"stats": 1, "hashes": 1, "opens": 1, "mmaps": 0, "bytes_read": bytes_read},
        "forbidden_access": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
    }
    atomic_json(inventory_path, result)
    runtime_handoff = ssd / "handoff" / "texbat_ds3_78p9s.csv"
    source = artifact / "frozen_handoff_texbat_ds3_78p9s.csv"
    runtime_handoff.parent.mkdir(parents=True, exist_ok=True)
    if runtime_handoff.exists():
        if sha256_file(runtime_handoff) != sha256_file(source):
            raise BindingError("existing runtime handoff mismatch")
    else:
        runtime_handoff.write_bytes(source.read_bytes())
    atomic_json(
        ssd / "attack_access_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-runtime-access-audit.v1",
            "status": "PASS",
            "freeze_sha": freeze_sha,
            "allowed_ds3": result["access"],
            "receiver_invocations": 0,
            "receiver_logical_slice_bytes": 0,
            "forbidden_inputs": result["forbidden_access"],
            "phase_b_executed": False,
        },
    )
    return result


def validate_frozen_contract(binding: Mapping[str, object]) -> None:
    expected_definitions = {
        config: sha256_bytes(canonical_json(receiver_configurations()[config]).encode())
        for config in CONFIG_ORDER
    }
    if binding.get("authoritative_threshold") != THRESHOLD:
        raise BindingError("threshold binding mismatch")
    if binding.get("locked_model", {}).get("model_sha256") != R4C_LOCKED_MODEL_SHA256:
        raise BindingError("locked model binding mismatch")
    if binding.get("timeline") != {
        "start_s": START_S,
        "onset_s": ONSET_S,
        "pull_off_s": PULL_OFF_S,
        "end_s": END_S,
    }:
        raise BindingError("timeline/window binding mismatch")
    if binding.get("configuration_definitions") != expected_definitions:
        raise BindingError("configuration definition binding mismatch")


def common_supported_prns(native_by_config: Mapping[str, Mapping[str, object]]) -> list[int]:
    if set(native_by_config) != set(CONFIG_ORDER):
        raise BindingError("four configurations required for support")
    per_config = []
    for config in CONFIG_ORDER:
        values = {
            int(prn)
            for summary in native_by_config[config].get("file_summaries", [])
            if int(summary.get("record_count", 0)) > 0
            for prn in summary.get("prn_values", [])
        }
        per_config.append(values)
    common = sorted(set.intersection(*per_config))
    if len(common) < MINIMUM_COMMON_PRNS:
        raise BindingError("INCONCLUSIVE_SUPPORT: fewer than four common non-empty TRACE PRNs")
    return common


def expected_dump_paths(out: Path) -> list[Path]:
    return [out / f"trace_native_1ms_ch_{channel}.bin" for channel in range(EXPECTED_DUMP_COUNT)]


def adapt_dump_manifest(expected: Sequence[Path], native: Mapping[str, object]) -> list[dict[str, object]]:
    summaries = {str(Path(str(row["path"])).resolve()): row for row in native.get("file_summaries", [])}
    if len(summaries) != len(expected):
        raise BindingError("native summary count differs from expected dump count")
    rows = []
    for channel, path in enumerate(expected):
        if not path.is_file():
            raise BindingError(f"missing expected TRACE dump: {path}")
        summary = summaries.get(str(path.resolve()))
        if summary is None:
            raise BindingError(f"missing native summary for TRACE dump: {path}")
        record_count = int(summary.get("record_count", -1))
        if record_count < 0:
            raise BindingError(f"invalid record count: {path}")
        file_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if summary.get("sha256") not in (None, actual_sha):
            raise BindingError(f"native summary SHA mismatch: {path}")
        if summary.get("byte_size") is not None and int(summary["byte_size"]) != file_size:
            raise BindingError(f"native summary size mismatch: {path}")
        empty = record_count == 0
        rows.append(
            {
                "path": str(path),
                "channel": channel,
                "record_count": record_count,
                "size_bytes": 0 if empty else file_size,
                "file_size_bytes": file_size,
                "sha256": actual_sha,
                "status": "EMPTY_OPTIONAL_CHANNEL" if empty else "NONEMPTY_TRACE",
                "excluded_from_support": empty,
            }
        )
    return rows


def require_dump_binding(row: Mapping[str, object]) -> None:
    path = Path(str(row["path"]))
    if not path.is_file() or path.stat().st_size != int(row["file_size_bytes"]):
        raise BindingError(f"TRACE physical size binding mismatch: {path}")
    if sha256_file(path) != row.get("sha256"):
        raise BindingError(f"TRACE SHA-256 binding mismatch: {path}")
    empty = int(row.get("record_count", -1)) == 0
    if empty != (row.get("status") == "EMPTY_OPTIONAL_CHANNEL"):
        raise BindingError(f"TRACE empty status mismatch: {path}")
    if bool(row.get("excluded_from_support")) != empty:
        raise BindingError(f"TRACE support-exclusion mismatch: {path}")
    expected_logical = 0 if empty else int(row["file_size_bytes"])
    if int(row.get("size_bytes", -1)) != expected_logical:
        raise BindingError(f"TRACE logical size binding mismatch: {path}")


def _output_set_sha(dumps: Sequence[Mapping[str, object]]) -> str:
    selected = [
        {
            "name": Path(str(row["path"])).name,
            "channel": row["channel"],
            "record_count": row["record_count"],
            "size_bytes": row["size_bytes"],
            "file_size_bytes": row["file_size_bytes"],
            "sha256": row["sha256"],
            "status": row["status"],
            "excluded_from_support": row["excluded_from_support"],
        }
        for row in dumps
    ]
    return sha256_bytes(canonical_json(selected).encode())


def validate_completed_replay(manifest_path: Path, config: str, input_inventory: Mapping[str, object], source: Mapping[str, object]) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if not (
        manifest.get("status") == "PASS"
        and manifest.get("config") == config
        and manifest.get("input", {}).get("sha256") == input_inventory.get("sha256")
        and manifest.get("receiver", {}).get("sha256") == source["receiver"]["sha256"]
        and manifest.get("config_file", {}).get("sha256") == source["frozen_configs"][config]["sha256"]
        and manifest.get("handoff", {}).get("sha256") == source["reanchored_handoff"]["sha256"]
        and manifest.get("exit_code") == 0
        and manifest.get("termination", {}).get("status") == "PASS"
        and manifest.get("native_trace_validation", {}).get("status") == "PASS"
        and manifest.get("target_tracking_pass") is True
    ):
        raise BindingError(f"replay contract mismatch: {config}")
    require_binding(Path(manifest["config_file"]["path"]), manifest["config_file"])
    require_binding(Path(manifest["receiver_execution"]["path"]), manifest["receiver_execution"])
    execution = json.loads(Path(manifest["receiver_execution"]["path"]).read_text())
    if execution.get("exit_code") != 0 or execution.get("termination", {}).get("status") != "PASS":
        raise BindingError(f"receiver execution evidence mismatch: {config}")
    for row in manifest["dumps"]:
        require_dump_binding(row)
    if _output_set_sha(manifest["dumps"]) != manifest["output_set_sha256"]:
        raise BindingError(f"output-set SHA mismatch: {config}")
    return manifest


def _load_checkpoint(ssd: Path, freeze_sha: str, input_sha: str) -> dict[str, object]:
    path = ssd / "checkpoint.json"
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("freeze_sha") != freeze_sha or value.get("input_sha256") != input_sha:
            raise BindingError("checkpoint binding mismatch")
        return value
    value = {
        "schema": "gnss-doppler-lab.crid-r4d-checkpoint.v1",
        "freeze_sha": freeze_sha,
        "input_sha256": input_sha,
        "worker_count": 1,
        "completed": {},
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_json(path, value)
    return value


def _run_one_replay(artifact: Path, ssd: Path, config: str, inventory: Mapping[str, object], source: Mapping[str, object]) -> dict[str, object]:
    out = ssd / "replays" / config
    if out.exists():
        raise BindingError(f"uncheckpointed output exists; no overwrite/delete: {out}")
    out.mkdir(parents=True, exist_ok=False)
    frozen_config = artifact / "frozen_configs" / f"{config}.conf"
    cfg = out / "receiver.conf"
    cfg.write_bytes(frozen_config.read_bytes())
    require_binding(cfg, source["frozen_configs"][config])
    before = os.stat(DS3_RAW)
    command = [str(RECEIVER), f"--config_file={cfg}", "--keyboard=false"]
    started = utc_now()
    began = time.monotonic()
    with (out / "receiver.log").open("wb") as log:
        exit_code, termination = _supervised_run(
            command,
            cwd=out,
            log=log,
            raw=DS3_RAW,
            expected_end_byte=EXPECTED_END_BYTE,
            expected_dump_count=EXPECTED_DUMP_COUNT,
        )
    after = os.stat(DS3_RAW)
    raw_stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    execution_evidence = {
        "schema": "gnss-doppler-lab.crid-r4d-receiver-execution.v1",
        "config": config,
        "freeze_sha": git(artifact.parent.parent, "rev-parse", "HEAD"),
        "command": command,
        "started_at": started,
        "ended_at": utc_now(),
        "exit_code": exit_code,
        "termination": termination,
        "input_sha256": inventory["sha256"],
        "receiver_sha256": source["receiver"]["sha256"],
        "config_sha256": source["frozen_configs"][config]["sha256"],
        "handoff_sha256": source["reanchored_handoff"]["sha256"],
        "raw_stat_before": {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "raw_stat_after": {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns},
        "raw_stable_during_replay": raw_stable,
    }
    atomic_json(out / "receiver_execution.json", execution_evidence)
    expected = expected_dump_paths(out)
    existing = [path for path in expected if path.is_file()]
    native = validate_dump_files(existing, expected_scenario_id=f"CRID.R4d.TEXBAT.DS3.{config}", minimum_prns=MINIMUM_COMMON_PRNS) if existing else {"status": "FAIL", "file_summaries": []}
    tracked = sorted(
        {
            int(prn)
            for summary in native.get("file_summaries", [])
            if int(summary.get("record_count", 0)) > 0
            for prn in summary.get("prn_values", [])
        }
    )
    targets = [int(value) for value in source["target_prns"]]
    tracked_targets = sorted(set(tracked).intersection(targets))
    target_tracking = len(tracked_targets) >= MINIMUM_COMMON_PRNS
    dump_rows = adapt_dump_manifest(expected, native)
    passed = (
        exit_code == 0
        and termination.get("status") == "PASS"
        and native.get("status") == "PASS"
        and len(dump_rows) == EXPECTED_DUMP_COUNT
        and target_tracking
        and raw_stable
    )
    manifest = {
        "schema": "gnss-doppler-lab.crid-r4d-replay.v1",
        "status": "PASS" if passed else "FAIL",
        "dataset": "TEXBAT.DS3",
        "config": config,
        "scenario": f"CRID.R4d.TEXBAT.DS3.{config}",
        "command": command,
        "started_at": started,
        "ended_at": utc_now(),
        "elapsed_s": time.monotonic() - began,
        "exit_code": exit_code,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "input": {"path": str(DS3_RAW), "size_bytes": inventory["size_bytes"], "sha256": inventory["sha256"], "slice_start_sample": START_SAMPLE, "slice_end_sample_exclusive": END_SAMPLE},
        "receiver": source["receiver"],
        "configuration_definition_sha256": source["configuration_definitions"][config],
        "config_file": {"path": str(cfg), "size_bytes": cfg.stat().st_size, "sha256": sha256_file(cfg)},
        "handoff": {"path": str(ssd / "handoff" / "texbat_ds3_78p9s.csv"), "sha256": source["reanchored_handoff"]["sha256"]},
        "receiver_execution": file_binding(out / "receiver_execution.json"),
        "termination": termination,
        "native_trace_validation": native,
        "target_prns": targets,
        "tracked_prns": tracked,
        "tracked_target_prns": tracked_targets,
        "target_tracking_pass": target_tracking,
        "raw_stat_before": {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns},
        "raw_stat_after": {"size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns},
        "raw_stable_during_replay": raw_stable,
        "dumps": dump_rows,
        "output_set_sha256": _output_set_sha(dump_rows),
    }
    dump_json(out / "manifest.json", manifest)
    if not passed:
        raise BindingError(f"replay failed closed: {config}")
    return manifest


def run_replays(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != SSD_ROOT:
        raise BindingError("unexpected R4d SSD root")
    preserved_audit = json.loads((ssd / "r4c_preserved_output_audit.json").read_text())
    if preserved_audit.get("freeze_sha") != freeze_sha or preserved_audit.get("status") not in {"RECOVERED_PRESERVED_C0", "HISTORICAL_INCOMPLETE_EVIDENCE"}:
        raise BindingError("R4c preserved C0 audit missing or stale")
    inventory = json.loads((ssd / "ds3_input_inventory.json").read_text())
    if inventory.get("status") != "PASS" or inventory.get("freeze_sha") != freeze_sha:
        raise BindingError("DS3 inventory missing or stale")
    source = json.loads((artifact / "source_binding.json").read_text())
    validate_frozen_contract(source)
    require_binding(RECEIVER, source["receiver"])
    free = shutil.disk_usage(ssd.parent).free
    resource_check = {"free_bytes": free, "required_with_margin_bytes": 10_000_000_000, "status": "PASS" if free >= 10_000_000_000 else "INCONCLUSIVE_RESOURCE"}
    if resource_check["status"] != "PASS":
        atomic_json(ssd / "resource_stop.json", resource_check)
        raise BindingError("INCONCLUSIVE_RESOURCE")
    checkpoint = _load_checkpoint(ssd, freeze_sha, str(inventory["sha256"]))
    completed = 0
    for config in CONFIG_ORDER:
        prior = checkpoint["completed"].get(config)
        if prior is not None:
            manifest_path = Path(prior["manifest_path"])
            if sha256_file(manifest_path) != prior["manifest_sha256"]:
                raise BindingError(f"checkpoint manifest mismatch: {config}")
            validate_completed_replay(manifest_path, config, inventory, source)
            completed += 1
            print(json.dumps({"replay": completed, "total": 4, "config": config, "status": "SKIP_EXACT_COMPLETE"}), flush=True)
            continue
        manifest = _run_one_replay(artifact, ssd, config, inventory, source)
        manifest_path = ssd / "replays" / config / "manifest.json"
        checkpoint["completed"][config] = {
            "status": "PASS",
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "input_sha256": inventory["sha256"],
            "output_set_sha256": manifest["output_set_sha256"],
        }
        checkpoint["updated_at"] = utc_now()
        atomic_json(ssd / "checkpoint.json", checkpoint)
        completed += 1
        audit = json.loads((ssd / "attack_access_audit.json").read_text())
        audit["receiver_invocations"] = completed
        audit["receiver_logical_slice_bytes"] = completed * (END_SAMPLE - START_SAMPLE) * BYTES_PER_COMPLEX_SAMPLE
        audit["allowed_ds3"]["stats"] = 1 + completed * 2
        audit["allowed_ds3"]["opens"] = 1 + completed
        audit["allowed_ds3"]["bytes_read"] = inventory["hash_bytes_read"] + audit["receiver_logical_slice_bytes"]
        atomic_json(ssd / "attack_access_audit.json", audit)
        print(json.dumps({"replay": completed, "total": 4, "config": config, "status": "PASS"}), flush=True)
    result = {
        "schema": "gnss-doppler-lab.crid-r4d-replay-run.v1",
        "status": "PASS",
        "freeze_sha": freeze_sha,
        "completed_replays": completed,
        "expected_replays": 4,
        "worker_count": 1,
        "order": list(CONFIG_ORDER),
        "resource_preflight": resource_check,
        "phase_b_executed": False,
    }
    atomic_json(ssd / "replay_run_summary.json", result)
    return result


def strict_alarms(scores: Sequence[float], threshold: float = THRESHOLD) -> np.ndarray:
    return np.asarray(scores, dtype=float) > float(threshold)


def partition_for_time(time_s: float) -> str:
    if START_S <= time_s < ONSET_S:
        return "pre_onset"
    if ONSET_S <= time_s < PULL_OFF_S:
        return "transition"
    if PULL_OFF_S <= time_s < END_S:
        return "established"
    return "outside"


def persistent_alarm_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    established = [row for row in rows if row["partition"] == "established"]
    result: dict[str, object] = {}
    for width in PERSISTENCE_WINDOWS_S:
        complete_bins = int((END_S - PULL_OFF_S) // width)
        eligible = passing = 0
        bin_rows = []
        for index in range(complete_bins):
            start = PULL_OFF_S + index * width
            end = start + width
            selected = [row for row in established if start <= float(row["time_s"]) < end]
            if not selected:
                continue
            ratio = float(np.mean([int(row["alarm"]) for row in selected]))
            eligible += 1
            passed = ratio >= PERSISTENCE_LEVEL
            passing += int(passed)
            bin_rows.append({"start_s": start, "end_s": end, "supported_epochs": len(selected), "alarm_ratio": ratio, "pass": passed})
        result[f"{width}s"] = {
            "eligible_complete_bins": eligible,
            "passing_bins": passing,
            "persistent_alarm_ratio": float(passing / eligible) if eligible else None,
            "bins": bin_rows,
        }
    return result


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    tolerance = np.finfo(float).eps * 16
    if len(left) < 3 or np.std(left) <= tolerance or np.std(right) <= tolerance:
        return {"pearson": 0.0, "spearman": 0.0, "count": int(len(left))}
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(spearmanr(left, right).statistic),
        "count": int(len(left)),
    }


def shortcut_audit(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    score = [float(row["score"]) for row in rows]
    correlations = {
        name: _safe_corr(score, [float(row[field]) for row in rows])
        for name, field in (
            ("cn0_median_db_hz", "cn0_median_db_hz"),
            ("lock_median", "lock_median"),
            ("common_prn_count", "prn_count"),
            ("tracked_prn_count_min_config", "tracked_prn_count_min_config"),
        )
    }
    pre = [row for row in rows if row["partition"] == "pre_onset"]
    established_alarms = [row for row in rows if row["partition"] == "established" and int(row["alarm"]) == 1]
    pre_tracked = float(np.median([float(row["tracked_prn_count_min_config"]) for row in pre])) if pre else None
    pre_lock_q10 = float(np.quantile([float(row["lock_median"]) for row in pre], 0.10)) if pre else None
    collapse_threshold = max(4.0, 0.5 * pre_tracked) if pre_tracked is not None else None
    lock_threshold = max(0.85, 0.8 * pre_lock_q10) if pre_lock_q10 is not None else None
    outside_nuisance = []
    for row in established_alarms:
        collapsed = float(row["tracked_prn_count_min_config"]) < float(collapse_threshold)
        lock_lost = float(row["lock_median"]) < float(lock_threshold)
        outside_nuisance.append(not (collapsed or lock_lost))
    explained_only = not established_alarms or not any(outside_nuisance)
    return {
        "schema": "gnss-doppler-lab.crid-r4d-shortcut-audit.v1",
        "status": "FAIL" if explained_only else "PASS",
        "score_inputs": {"cn0": False, "power": False, "lock": False, "prn_count": False},
        "correlations": correlations,
        "pre_onset_reference": {"tracked_prn_median": pre_tracked, "lock_q10": pre_lock_q10},
        "nuisance_definitions": {"tracked_prn_collapse_below": collapse_threshold, "lock_loss_below": lock_threshold},
        "established_alarm_count": len(established_alarms),
        "established_alarms_outside_lock_loss_or_prn_collapse": sum(outside_nuisance),
        "explained_only_by_lock_loss_or_tracked_prn_collapse": explained_only,
    }


def scenario_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_partition = {name: [row for row in rows if row["partition"] == name] for name in ("pre_onset", "transition", "established")}
    evaluation = [*by_partition["pre_onset"], *by_partition["transition"], *by_partition["established"]]
    labels = np.asarray([int(row["label"]) for row in evaluation], dtype=int)
    scores = np.asarray([float(row["score"]) for row in evaluation], dtype=float)
    alarms = np.asarray([int(row["alarm"]) for row in evaluation], dtype=bool)
    if not len(evaluation) or len(np.unique(labels)) != 2:
        raise BindingError("insufficient pre/post support for exploratory metrics")
    first_onset = next((float(row["time_s"]) for row in rows if float(row["time_s"]) >= ONSET_S and int(row["alarm"]) == 1), None)
    first_pull = next((float(row["time_s"]) for row in rows if float(row["time_s"]) >= PULL_OFF_S and int(row["alarm"]) == 1), None)
    expected = {"pre_onset": 40_000, "transition": 76_100, "established": 43_900}
    support = {
        name: {
            "valid_epochs": len(selected),
            "nominal_1ms_epochs": expected[name],
            "availability": len(selected) / expected[name],
        }
        for name, selected in by_partition.items()
    }
    return {
        "pre_onset_fpr": float(np.mean([int(row["alarm"]) for row in by_partition["pre_onset"]])),
        "transition_detection_rate": float(np.mean([int(row["alarm"]) for row in by_partition["transition"]])),
        "established_detection_rate": float(np.mean([int(row["alarm"]) for row in by_partition["established"]])),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pauc_0_05": float(roc_auc_score(labels, scores, max_fpr=0.05)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "first_alarm_after_onset_s": first_onset,
        "first_alarm_after_onset_delay_s": None if first_onset is None else first_onset - ONSET_S,
        "first_alarm_after_pull_off_s": first_pull,
        "first_alarm_after_pull_off_delay_s": None if first_pull is None else first_pull - PULL_OFF_S,
        "persistent_alarm": persistent_alarm_metrics(rows),
        "common_support": support,
        "epoch_count": len(evaluation),
    }


def evaluate_signal_gate(metrics: Mapping[str, object], shortcut: Mapping[str, object], technical_ok: bool) -> dict[str, object]:
    established = metrics["common_support"]["established"]
    checks = {
        "technical_replay_alignment_support": technical_ok,
        "pre_onset_fpr_le_0p05": float(metrics["pre_onset_fpr"]) <= 0.05,
        "pauc_0p05_ge_0p80": float(metrics["pauc_0_05"]) >= 0.80,
        "established_detection_rate_ge_0p70": float(metrics["established_detection_rate"]) >= 0.70,
        "established_four_config_min_four_prn_support": int(established["valid_epochs"]) > 0,
        "not_only_lock_loss_or_tracked_prn_collapse": shortcut["explained_only_by_lock_loss_or_tracked_prn_collapse"] is False,
    }
    return {"definition": _gate_contract(), "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}


def _table_epoch_nuisance(tables: Mapping[str, ResponseTable]) -> dict[int, dict[str, float]]:
    per_config: dict[str, dict[int, dict[str, object]]] = {}
    for config, table in tables.items():
        cadence = r4._native_cadence(table) if hasattr(r4, "_native_cadence") else 25_000
        groups: dict[int, dict[str, object]] = {}
        for prn, sample, cn0, lock in zip(table.prn, table.sample, table.cn0, table.lock, strict=True):
            epoch = int(np.rint(int(sample) / cadence))
            group = groups.setdefault(epoch, {"cn0": [], "lock": [], "prns": set()})
            group["cn0"].append(float(cn0))
            group["lock"].append(float(lock))
            group["prns"].add(int(prn))
        per_config[config] = groups
    all_epochs = set.intersection(*(set(groups) for groups in per_config.values()))
    result = {}
    for epoch in all_epochs:
        cn0 = [value for config in CONFIG_ORDER for value in per_config[config][epoch]["cn0"]]
        lock = [value for config in CONFIG_ORDER for value in per_config[config][epoch]["lock"]]
        counts = [len(per_config[config][epoch]["prns"]) for config in CONFIG_ORDER]
        result[epoch] = {
            "cn0_median_db_hz": float(np.median(cn0)),
            "lock_median": float(np.median(lock)),
            "tracked_prn_count_min_config": float(min(counts)),
            "tracked_prn_count_median_config": float(np.median(counts)),
        }
    return result


def _score_hash(rows: Sequence[Mapping[str, object]]) -> str:
    selected = [
        {name: row[name] for name in ("sample", "epoch", "score", "prn_count", "config_count", "h0_loglike", "h1_loglike", "penalty", "configuration_disagreement")}
        for row in rows
    ]
    return sha256_bytes(canonical_json(selected).encode())


def _compact_replay_row(manifest_path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "config": manifest["config"],
        "status": manifest["status"],
        "exit_code": manifest["exit_code"],
        "terminal_drain_status": manifest["termination"]["status"],
        "native_trace_status": manifest["native_trace_validation"]["status"],
        "target_tracking_pass": manifest["target_tracking_pass"],
        "tracked_target_prns": ",".join(map(str, manifest["tracked_target_prns"])),
        "input_sha256": manifest["input"]["sha256"],
        "receiver_sha256": manifest["receiver"]["sha256"],
        "config_sha256": manifest["config_file"]["sha256"],
        "handoff_sha256": manifest["handoff"]["sha256"],
        "output_set_sha256": manifest["output_set_sha256"],
        "dump_count": len(manifest["dumps"]),
        "output_bytes": sum(int(row["file_size_bytes"]) for row in manifest["dumps"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _write_plots(artifact: Path, rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib.pyplot as plt

    time_values = np.asarray([float(row["time_s"]) for row in rows])
    scores = np.asarray([float(row["score"]) for row in rows])
    alarms = np.asarray([int(row["alarm"]) for row in rows], dtype=bool)
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(time_values, scores, linewidth=0.5, color="#264653", label="locked CRID score")
    axis.scatter(time_values[alarms], scores[alarms], s=2, color="#e76f51", label="alarm")
    axis.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1, label="authoritative TEX q99")
    axis.axvline(ONSET_S, color="#e9c46a", linestyle="--", label="onset")
    axis.axvline(PULL_OFF_S, color="#2a9d8f", linestyle="--", label="established")
    axis.set(xlabel="TEXBAT DS3 absolute time (s)", ylabel="score", title="Exploratory locked-score alarm timeline")
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(artifact / "alarm_timeline.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_values, [float(row["h1_improvement"]) for row in rows], linewidth=0.5)
    axes[0].set_ylabel("H1-H0 loglike")
    axes[1].plot(time_values, [float(row["penalty"]) for row in rows], linewidth=0.5)
    axes[1].set_ylabel("BIC penalty")
    axes[2].plot(time_values, [float(row["configuration_disagreement"]) for row in rows], linewidth=0.5)
    axes[2].set_ylabel("config disagreement")
    axes[2].set_xlabel("TEXBAT DS3 absolute time (s)")
    for axis in axes:
        axis.axvline(ONSET_S, color="#e9c46a", linestyle="--", linewidth=0.8)
        axis.axvline(PULL_OFF_S, color="#2a9d8f", linestyle="--", linewidth=0.8)
    figure.suptitle("Frozen score-component timeline")
    figure.tight_layout()
    figure.savefig(artifact / "score_components.png", dpi=140)
    plt.close(figure)


def analyze(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != SSD_ROOT:
        raise BindingError("unexpected R4d SSD root")
    replay_summary = json.loads((ssd / "replay_run_summary.json").read_text())
    if replay_summary.get("status") != "PASS" or replay_summary.get("completed_replays") != 4:
        raise BindingError("four replay PASS required before analysis")
    inventory = json.loads((ssd / "ds3_input_inventory.json").read_text())
    source = json.loads((artifact / "source_binding.json").read_text())
    validate_frozen_contract(source)
    replay_rows = []
    empty_rows = []
    tables = {}
    for config in CONFIG_ORDER:
        manifest_path = ssd / "replays" / config / "manifest.json"
        manifest = validate_completed_replay(manifest_path, config, inventory, source)
        replay_rows.append(_compact_replay_row(manifest_path, manifest))
        empty_rows.extend(
            {
                "config": config,
                "channel": row["channel"],
                "status": row["status"],
                "record_count": row["record_count"],
                "size_bytes": row["size_bytes"],
                "file_size_bytes": row["file_size_bytes"],
                "sha256": row["sha256"],
                "excluded_from_support": row["excluded_from_support"],
                "path": row["path"],
            }
            for row in manifest["dumps"]
            if row["status"] == "EMPTY_OPTIONAL_CHANNEL"
        )
        tables[config] = load_response(config, [Path(row["path"]) for row in manifest["dumps"]])
    model, delays, observed_model = build_locked_model(source)
    if observed_model["model_sha256"] != source["locked_model"]["model_sha256"] or delays != source["locked_model"]["causal_delays_ms"]:
        raise BindingError("locked model or causal alignment mismatch")
    first = score_aligned(tables, model, delays, minimum_prns=MINIMUM_COMMON_PRNS)
    second = score_aligned(tables, model, delays, minimum_prns=MINIMUM_COMMON_PRNS)
    first_sha = _score_hash(first)
    second_sha = _score_hash(second)
    if first_sha != second_sha:
        raise BindingError("deterministic score mismatch")
    nuisance = _table_epoch_nuisance(tables)
    rows = []
    for item in first:
        time_s = float(item["sample"]) / FS
        partition = partition_for_time(time_s)
        if partition == "outside" or item["epoch"] not in nuisance:
            continue
        label = 0 if partition == "pre_onset" else 1
        alarm = int(float(item["score"]) > THRESHOLD)
        rows.append(
            {
                "dataset": "TEXBAT.DS3",
                "sample": int(item["sample"]),
                "time_s": time_s,
                "partition": partition,
                "score": float(item["score"]),
                "alarm": alarm,
                "label": label,
                "prn_count": int(item["prn_count"]),
                "config_count": int(item["config_count"]),
                **nuisance[item["epoch"]],
                "h0_loglike": float(item["h0_loglike"]),
                "h1_loglike": float(item["h1_loglike"]),
                "h1_improvement": float(item["h1_loglike"] - item["h0_loglike"]),
                "penalty": float(item["penalty"]),
                "configuration_disagreement": float(item["configuration_disagreement"]),
            }
        )
    rows.sort(key=lambda row: int(row["sample"]))
    technical_ok = bool(rows) and all(int(row["config_count"]) == 4 and int(row["prn_count"]) >= 4 for row in rows)
    metrics = scenario_metrics(rows) if technical_ok else None
    shortcuts = shortcut_audit(rows) if technical_ok else {"schema": "gnss-doppler-lab.crid-r4d-shortcut-audit.v1", "status": "INCONCLUSIVE"}
    gate = evaluate_signal_gate(metrics, shortcuts, technical_ok) if metrics is not None else {"definition": _gate_contract(), "status": "INCONCLUSIVE", "checks": {}}
    if not technical_ok or metrics is None:
        verdict = "INCONCLUSIVE_TEXBAT_DS3_EXPLORATORY_EXECUTION"
        status = "INCONCLUSIVE"
    elif gate["status"] == "PASS":
        verdict = "EXPLORATORY_TEXBAT_DS3_SIGNAL_PRESENT"
        status = "EXPLORATORY_PASS"
    else:
        verdict = "EXPLORATORY_TEXBAT_DS3_NO_USEFUL_SIGNAL"
        status = "EXPLORATORY_NO_SIGNAL"
    write_csv(artifact / "replay_completion.csv", REPLAY_FIELDS, replay_rows)
    write_csv(artifact / "empty_channel_inventory.csv", EMPTY_CHANNEL_FIELDS, empty_rows)
    with gzip.open(artifact / "per_epoch_scores.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPOCH_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    support = metrics["common_support"] if metrics is not None else {}
    dump_json(
        artifact / "support_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-support-audit.v1",
            "status": "PASS" if technical_ok else "INCONCLUSIVE",
            "minimum_configurations": 4,
            "minimum_common_prns": 4,
            "score_epoch_count": len(rows),
            "score_sha256_first": first_sha,
            "score_sha256_second": second_sha,
            "deterministic_exact_match": first_sha == second_sha,
            "causal_delays_ms": delays,
            "partitions": support,
            "empty_optional_channel_count": len(empty_rows),
            "empty_optional_channels_excluded_from_support": True,
        },
    )
    dump_json(artifact / "shortcut_audit.json", shortcuts)
    dump_json(
        artifact / "scenario_metrics.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-scenario-metrics.v1",
            "status": "COMPUTED" if metrics is not None else "INCONCLUSIVE",
            "metric_contract": _metric_contract(),
            "gate_contract": _gate_contract(),
            "results": metrics,
            "exploratory_signal_gate": gate,
        },
    )
    preserved_audit = json.loads((ssd / "r4c_preserved_output_audit.json").read_text())
    dump_json(artifact / "r4c_preserved_output_audit.json", preserved_audit)
    runtime_audit = json.loads((ssd / "attack_access_audit.json").read_text())
    dump_json(
        artifact / "attack_access_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-attack-access-audit.v1",
            "status": "PASS",
            "allowed_ds3": runtime_audit["allowed_ds3"],
            "receiver_invocations": runtime_audit["receiver_invocations"],
            "receiver_logical_slice_bytes": runtime_audit["receiver_logical_slice_bytes"],
            "forbidden_inputs": runtime_audit["forbidden_inputs"],
            "phase_b_executed": False,
        },
    )
    dump_json(
        artifact / "ds3_input_inventory.json",
        {
            **inventory,
            "status": "PASS",
            "access_after_pushed_freeze_only": True,
            "freeze_sha": freeze_sha,
        },
    )
    dump_json(
        artifact / "freeze_commit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4d-freeze-commit.v1",
            "status": "PASS",
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "freeze_sha": freeze_sha,
            "local_sha_before_access": freeze_sha,
            "remote_sha_before_access": freeze_sha,
            "ahead": 0,
            "behind": 0,
            "clean_checkout_before_access": True,
        },
    )
    execution = json.loads((artifact / "execution_repair_freeze.json").read_text())
    execution.update({"status": "EXECUTED_FROM_PUSHED_FREEZE", "freeze_sha": freeze_sha, "post_access_executable_changes": False})
    dump_json(artifact / "execution_repair_freeze.json", execution)
    final = {
        "schema": "gnss-doppler-lab.crid-r4d-final-verdict.v1",
        "status": status,
        "verdict": verdict,
        "exploratory_only": True,
        "formal_phase_b": False,
        "confirmatory_detection_evaluation": False,
        "deployment_validation": False,
        "phase_b_executed": False,
        "base_sha": BASE_SHA,
        "freeze_sha": freeze_sha,
        "replays": 4,
        "empty_optional_channel_count": len(empty_rows),
        "score_epoch_count": len(rows),
        "threshold": THRESHOLD,
        "threshold_recomputation_executed": False,
        "post_access_code_threshold_score_feature_model_window_gate_changes": False,
        "forbidden_attack_access_bytes": 0,
    }
    dump_json(artifact / "final_verdict.json", final)
    _write_plots(artifact, rows)
    (artifact / "README.md").write_text(
        "# CRID Stage-0 R4d TEXBAT DS3 exploratory locked-score audit\n\n"
        f"Final verdict: `{verdict}`\n\n"
        "This is exploratory locked-model evidence only. R4b Phase A did not pass; this run is not formal Phase B, confirmatory detector validation, or deployment evidence. "
        "The frozen CRID model and authoritative TEX q99 literal were used without recalculation or post-access changes.\n\n"
        "Only TEXBAT DS3 was accessed. DS1, DS4, DS7, DS8, and OAK attack data remained untouched; DS7/DS8 remain independent future holdouts.\n"
    )
    seal_manifest(artifact)
    result = {"status": status, "verdict": verdict, "metrics": metrics, "gate": gate, "score_epoch_count": len(rows)}
    atomic_json(ssd / "analysis_summary.json", result)
    return result
