"""Frozen orchestration contracts for CRID Stage-0 R4 clean Phase A.

This module deliberately reuses the committed CRID feature/model/score code.
It only binds clean R3 inputs, supervises sequential receiver replays, maps
scores to truth support, evaluates the preregistered gate, and seals evidence.
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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .crid import (
    CONFIG_ORDER,
    canonical_json,
    load_response,
    receiver_configurations,
    render_receiver_config,
    score_aligned,
    sha256_bytes,
)
from .crid_control_truth import TRUTH_DTYPE
from .crid_experiment import fit_domain
from .crid_receiver_replay import _supervised_run
from .trace_native_1ms import read_records, validate_dump_files


BASE_R3B_SHA = "8cf594bde9c8e48bf80b5872e4ca13e1d0d13b0d"
BRANCH = "research/crid-stage0-r4-phase-a-physical-identifiability"
R3_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-r3-control-generator-foundation"
)
R4_SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-r4-phase-a-physical-identifiability"
)
RECEIVER = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr"
)
CLEAN_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-counterfactual-receiver-invariance/replays"
)
BASE_CONFIG = {
    "OAK": CLEAN_ROOT / "oak_clean/C0/receiver.conf",
    "TEX": CLEAN_ROOT / "tex_clean/C0/receiver.conf",
}
DOMAIN = {
    "OAK": {
        "fs": 5_000_000,
        "expected_dumps": 11,
        "clean_dataset": "oak_clean",
        "q99": -21.705587048010322,
        "holdout_fpr_q99": 0.00730093543235227,
    },
    "TEX": {
        "fs": 25_000_000,
        "expected_dumps": 10,
        "clean_dataset": "tex_clean",
        "q99": -21.942672917134093,
        "holdout_fpr_q99": 0.012708150744960562,
    },
}
R3_ART = "artifacts/crid_stage0_r3_control_generator_foundation"
R3A_ART = "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"
R3B_ART = "artifacts/crid_stage0_r3b_terminal_provenance_closure"
R2_ART = "artifacts/crid_stage0_r2_frozen_evaluation"
SCIENCE_FILES = (
    "src/gnss_doppler_lab/crid.py",
    "src/gnss_doppler_lab/crid_experiment.py",
    "src/gnss_doppler_lab/crid_metrics.py",
    "src/gnss_doppler_lab/crid_physical_controls.py",
    "src/gnss_doppler_lab/crid_receiver_replay.py",
    "src/gnss_doppler_lab/trace_native_1ms.py",
)
EXECUTABLE_FILES = (
    "src/gnss_doppler_lab/crid_r4_phase_a.py",
    "scripts/run_crid_r4_phase_a.py",
    "scripts/verify_crid_r4_phase_a.py",
)
INVENTORY_FIELDS = (
    "domain",
    "case_id",
    "family",
    "mode",
    "delay_chips",
    "power_db",
    "target_prns",
    "control_path",
    "control_size_bytes",
    "control_sha256",
    "truth_json_path",
    "truth_json_size_bytes",
    "truth_json_sha256",
    "truth_epochs_path",
    "truth_epochs_size_bytes",
    "truth_epochs_sha256",
    "package_path",
    "package_size_bytes",
    "package_sha256",
    "existing_c0_manifest_path",
    "existing_c0_manifest_size_bytes",
    "existing_c0_manifest_sha256",
    "existing_c0_status",
    "replacement_start_sample",
    "replacement_end_sample_exclusive",
)
REPLAY_FIELDS = (
    "domain",
    "case_id",
    "config",
    "status",
    "exit_code",
    "terminal_drain_status",
    "native_trace_status",
    "target_tracking_pass",
    "tracked_prns",
    "input_sha256",
    "config_sha256",
    "receiver_sha256",
    "output_set_sha256",
    "dump_count",
    "output_bytes",
    "manifest_path",
    "manifest_sha256",
)
METRIC_FIELDS = (
    "domain",
    "case_id",
    "family",
    "mode",
    "delay_chips",
    "power_db",
    "target_prns",
    "technical_status",
    "valid_full_epochs",
    "full_alarm_count",
    "alarm_ratio_q99_full_support",
    "valid_active_epochs",
    "active_alarm_count",
    "alarm_ratio_q99_active_support",
    "primary_window",
    "primary_alarm_ratio_q99",
    "threshold_q99",
    "case_gate_status",
    "score_evidence_path",
    "score_evidence_size_bytes",
    "score_evidence_sha256",
)


class BindingError(RuntimeError):
    """Fail-closed binding or checkpoint mismatch."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_bytes: int = 64 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for payload in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(payload)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    path = Path(path)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def repo_file_binding(repo: Path, name: str) -> dict[str, object]:
    path = repo / name
    return {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def require_file_binding(path: Path, size_bytes: int, sha256: str) -> None:
    path = Path(path)
    if not path.is_file() or path.stat().st_size != int(size_bytes):
        raise BindingError(f"file size binding mismatch: {path}")
    if sha256_file(path) != sha256:
        raise BindingError(f"file SHA-256 binding mismatch: {path}")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def assert_branch(repo: Path) -> None:
    if git(repo, "branch", "--show-current") != BRANCH:
        raise BindingError("wrong R4 branch")
    if git(repo, "merge-base", "HEAD", BASE_R3B_SHA) != BASE_R3B_SHA:
        raise BindingError("R4 branch is not based on the required R3b commit")


def assert_pushed_freeze(repo: Path, freeze_sha: str) -> None:
    assert_branch(repo)
    head = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    if head != freeze_sha or remote != freeze_sha:
        raise BindingError("HEAD and origin R4 branch must equal the supplied pushed freeze SHA")
    if git(repo, "status", "--porcelain=v1"):
        raise BindingError("execution checkout is not clean")


def _case_ids(repo: Path) -> list[tuple[str, str, str]]:
    art = repo / R3_ART
    rows: list[tuple[str, str, str]] = []
    for name, family in (
        ("negative_control_inventory.csv", "negative"),
        ("positive_control_inventory.csv", "positive"),
    ):
        for row in read_csv(art / name):
            rows.append((row["domain"], row["case_id"], family))
    rows.sort()
    if len(rows) != 66 or len(set(rows)) != 66:
        raise BindingError("committed R3 inventory is not exactly 66 unique controls")
    if {domain: sum(row[0] == domain for row in rows) for domain in DOMAIN} != {"OAK": 33, "TEX": 33}:
        raise BindingError("committed R3 inventory is not 33 controls per domain")
    return rows


def _assert_exact_control_path(path: Path, domain: str, case_id: str, role: str) -> None:
    expected = {
        "control": R3_ROOT / "controls" / domain / case_id / "control.bin",
        "truth_json": R3_ROOT / "controls" / domain / case_id / "truth.json",
        "truth_epochs": R3_ROOT / "controls" / domain / case_id / "truth_epochs.bin",
        "package": R3_ROOT / "smoke_prefixes" / domain / case_id / "control_45s.bin",
        "smoke_manifest": R3_ROOT / "c0_smoke" / domain / case_id / "manifest.json",
    }[role]
    if path != expected:
        raise BindingError(f"non-allowlisted {role} path for {case_id}")


def build_control_inventory(repo: Path) -> list[dict[str, object]]:
    """Hash every allowlisted R3 control/truth/package sequentially."""
    smoke_compact = json.loads((repo / R3_ART / "c0_smoke_validation.json").read_text())
    smoke_rows = {row["case_id"]: row for row in smoke_compact["cases"]}
    result = []
    for index, (domain, case_id, family) in enumerate(_case_ids(repo), start=1):
        control = R3_ROOT / "controls" / domain / case_id / "control.bin"
        truth_path = control.parent / "truth.json"
        epochs = control.parent / "truth_epochs.bin"
        package = R3_ROOT / "smoke_prefixes" / domain / case_id / "control_45s.bin"
        smoke_path = R3_ROOT / "c0_smoke" / domain / case_id / "manifest.json"
        for role, path in (
            ("control", control),
            ("truth_json", truth_path),
            ("truth_epochs", epochs),
            ("package", package),
            ("smoke_manifest", smoke_path),
        ):
            _assert_exact_control_path(path, domain, case_id, role)
            if not path.is_file():
                raise BindingError(f"missing allowlisted {role}: {path}")
        truth = json.loads(truth_path.read_text())
        smoke = json.loads(smoke_path.read_text())
        control_binding = file_binding(control)
        truth_binding = file_binding(truth_path)
        epochs_binding = file_binding(epochs)
        package_binding = file_binding(package)
        smoke_binding = file_binding(smoke_path)
        if truth["case_id"] != case_id or truth["domain"] != domain or truth["family"] != family:
            raise BindingError(f"truth identity mismatch: {case_id}")
        if control_binding["sha256"] != truth["output_sha256"] or control_binding["size_bytes"] != truth["size_bytes"]:
            raise BindingError(f"control binding mismatch: {case_id}")
        if epochs_binding["sha256"] != truth["epoch_truth"]["sha256"]:
            raise BindingError(f"truth epoch binding mismatch: {case_id}")
        if package_binding["sha256"] != smoke["raw"]["sha256"]:
            raise BindingError(f"45-second package binding mismatch: {case_id}")
        compact = smoke_rows.get(case_id, {})
        if not (
            smoke.get("status") == "PASS"
            and smoke.get("exit_code") == 0
            and smoke.get("termination", {}).get("status") == "PASS"
            and compact.get("status") == "PASS"
        ):
            raise BindingError(f"existing C0 smoke is not PASS: {case_id}")
        mode = ""
        if family == "positive":
            mode = "single" if len(truth["targets"]) == 1 else "four"
        result.append(
            {
                "domain": domain,
                "case_id": case_id,
                "family": family,
                "mode": mode,
                "delay_chips": truth["delay_chips"],
                "power_db": truth["power_db"],
                "target_prns": ",".join(map(str, compact["target_prns"])),
                "control_path": control_binding["path"],
                "control_size_bytes": control_binding["size_bytes"],
                "control_sha256": control_binding["sha256"],
                "truth_json_path": truth_binding["path"],
                "truth_json_size_bytes": truth_binding["size_bytes"],
                "truth_json_sha256": truth_binding["sha256"],
                "truth_epochs_path": epochs_binding["path"],
                "truth_epochs_size_bytes": epochs_binding["size_bytes"],
                "truth_epochs_sha256": epochs_binding["sha256"],
                "package_path": package_binding["path"],
                "package_size_bytes": package_binding["size_bytes"],
                "package_sha256": package_binding["sha256"],
                "existing_c0_manifest_path": smoke_binding["path"],
                "existing_c0_manifest_size_bytes": smoke_binding["size_bytes"],
                "existing_c0_manifest_sha256": smoke_binding["sha256"],
                "existing_c0_status": "PASS",
                "replacement_start_sample": truth["absolute_start_sample"],
                "replacement_end_sample_exclusive": truth["absolute_end_sample_exclusive"],
            }
        )
        print(json.dumps({"freeze_input_binding": index, "total": 66, "case_id": case_id, "status": "PASS"}), flush=True)
    return result


def clean_model_source_bindings() -> dict[str, object]:
    domains: dict[str, object] = {}
    for domain, spec in DOMAIN.items():
        dataset = str(spec["clean_dataset"])
        configs = {}
        for config in CONFIG_ORDER:
            directory = CLEAN_ROOT / dataset / config
            traces = sorted(directory.glob("trace_native_1ms_ch_*.bin"))
            if not traces:
                raise BindingError(f"no clean model TRACE for {domain}/{config}")
            configs[config] = {
                "directory": str(directory),
                "traces": [file_binding(path) for path in traces],
            }
        domains[domain] = {
            "dataset": dataset,
            "base_config": file_binding(BASE_CONFIG[domain]),
            "configurations": configs,
            "expected_q99": spec["q99"],
            "expected_holdout_fpr_q99": spec["holdout_fpr_q99"],
        }
    return domains


def runtime_environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "timezone": list(time.tzname),
        "gnss_sdr_version": subprocess.check_output([str(RECEIVER), "--version"], text=True).strip(),
    }


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
        "schema": "gnss-doppler-lab.crid-r4-artifact-manifest.v1",
        "file_count": len(files),
        "files": files,
        "status": "PASS",
    }


def _configuration_bindings() -> dict[str, str]:
    return {
        config: sha256_bytes(canonical_json(receiver_configurations()[config]).encode())
        for config in CONFIG_ORDER
    }


def prepare_freeze(repo: Path, artifact: Path) -> dict[str, object]:
    assert_branch(repo)
    artifact.mkdir(parents=True, exist_ok=True)
    inventory = build_control_inventory(repo)
    write_csv(artifact / "control_input_inventory.csv", INVENTORY_FIELDS, inventory)
    manifests = {
        "R3": repo_file_binding(repo, f"{R3_ART}/artifact_manifest_sha256.json"),
        "R3a": repo_file_binding(repo, f"{R3A_ART}/artifact_manifest_sha256.json"),
        "R3b": repo_file_binding(repo, f"{R3B_ART}/artifact_manifest_sha256.json"),
    }
    source_binding = {
        "schema": "gnss-doppler-lab.crid-r4-source-binding.v1",
        "status": "PASS",
        "base_r3b_sha": BASE_R3B_SHA,
        "r3_design_freeze": repo_file_binding(repo, f"{R3_ART}/design_freeze.json"),
        "r3_control_spec": repo_file_binding(repo, f"{R3_ART}/control_spec.json"),
        "artifact_manifests": manifests,
        "receiver": {**file_binding(RECEIVER), "version": runtime_environment()["gnss_sdr_version"]},
        "receiver_configuration_definitions_sha256": _configuration_bindings(),
        "scientific_code": {name: repo_file_binding(repo, name) for name in SCIENCE_FILES},
        "clean_model_source": clean_model_source_bindings(),
        "control_inventory": {
            "path": "control_input_inventory.csv",
            "rows": len(inventory),
            "OAK": sum(row["domain"] == "OAK" for row in inventory),
            "TEX": sum(row["domain"] == "TEX" for row in inventory),
        },
        "attack_bytes_read": 0,
    }
    dump_json(artifact / "source_binding.json", source_binding)
    preregistration = {
        "schema": "gnss-doppler-lab.crid-r4-preregistration.v1",
        "status": "FROZEN_BEFORE_PHASE_A_RESULTS",
        "scope": "CLEAN_ONLY_CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY",
        "base_r3b_sha": BASE_R3B_SHA,
        "inputs": {
            "controls": 66,
            "domains": {"OAK": 33, "TEX": 33},
            "configurations": list(CONFIG_ORDER),
            "total_replays": 264,
            "only_existing_r3_controls": True,
            "only_existing_45_second_packages": True,
            "control_regeneration": False,
            "overwrite": False,
            "truth_modification": False,
        },
        "score_contract": {
            "implementation": "unchanged committed CRID feature/alignment/predictor/covariance/H0/H1/median pooling",
            "minimum_configurations_per_epoch": 4,
            "minimum_common_prns_per_epoch": 4,
            "neural_model": False,
            "prn_identity_feature": False,
            "cn0_or_power_score": False,
            "score_fusion": False,
            "threshold": "committed R2 clean-only q99 after exact deterministic recomputation",
            "threshold_mismatch_verdict": "INCONCLUSIVE_THRESHOLD_BINDING",
        },
        "window_contract": {
            "coordinate": "raw sample",
            "positive_full_support": "replacement_start_sample <= score sample < replacement_end_sample_exclusive",
            "positive_active_support": "truth epoch interval whose code_delay_chips is strictly greater than zero",
            "positive_primary": "active support only",
            "negative_primary": "full replacement support",
            "full_45_second_positive_ratio_forbidden": True,
            "zero_valid_epochs": "INCONCLUSIVE_SUPPORT",
        },
        "primary_gate": primary_gate_definition(),
        "verdicts": {
            "execution_or_provenance": "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE",
            "physical_failure": "NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY",
            "pass": "CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_PASS",
            "pass_next_state": "READY_FOR_CRID_PHASE_B",
        },
        "post_result_changes_forbidden": ["code", "threshold", "window", "score", "gate", "control exclusion"],
        "phase_b_execution": False,
    }
    dump_json(artifact / "preregistration.json", preregistration)
    execution = {
        "schema": "gnss-doppler-lab.crid-r4-execution-freeze.v1",
        "status": "FROZEN_PENDING_REMOTE_COMMIT",
        "base_r3b_sha": BASE_R3B_SHA,
        "branch": BRANCH,
        "worker_count": 1,
        "sequential_order": "domain OAK then TEX; lexical case_id; C0,C1,C2,C3",
        "checkpoint_contract": {
            "completed_exact_replay": "validate and skip",
            "incomplete_output": "fail closed without overwrite or deletion",
            "hash_mismatch": "fail closed without overwrite or deletion",
            "resource_shortage": "INCONCLUSIVE_RESOURCE without deletion",
        },
        "freeze_authorization": "HEAD, origin branch, and supplied --freeze-sha must be identical",
        "commands": {
            "preflight": "python3 scripts/run_crid_r4_phase_a.py preflight --freeze-sha <PUSHED_FREEZE_SHA>",
            "threshold_check": "python3 scripts/run_crid_r4_phase_a.py threshold-check --freeze-sha <PUSHED_FREEZE_SHA>",
            "replay": "python3 scripts/run_crid_r4_phase_a.py replay --freeze-sha <PUSHED_FREEZE_SHA>",
            "analyze": "python3 scripts/run_crid_r4_phase_a.py analyze --freeze-sha <PUSHED_FREEZE_SHA>",
        },
        "environment": runtime_environment(),
        "executable_sha256": {name: sha256_file(repo / name) for name in EXECUTABLE_FILES},
    }
    dump_json(artifact / "execution_freeze.json", execution)
    threshold = {
        "schema": "gnss-doppler-lab.crid-r4-clean-threshold-binding.v1",
        "status": "FROZEN_PENDING_DETERMINISTIC_RECOMPUTATION",
        "source": "committed R2 clean-only threshold",
        "r2_thresholds": repo_file_binding(repo, f"{R2_ART}/thresholds.json"),
        "domains": {
            domain: {
                "q99": spec["q99"],
                "holdout_fpr_q99": spec["holdout_fpr_q99"],
                "recomputed_q99": None,
                "recomputed_holdout_fpr_q99": None,
                "exact_match": None,
            }
            for domain, spec in DOMAIN.items()
        },
    }
    dump_json(artifact / "clean_threshold_binding.json", threshold)
    dump_json(
        artifact / "phase_a_gate.json",
        {
            "schema": "gnss-doppler-lab.crid-r4-phase-a-gate.v1",
            "status": "FROZEN_NOT_EVALUATED",
            "definition": primary_gate_definition(),
            "results": None,
        },
    )
    dump_json(
        artifact / "attack_access_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4-attack-access-audit.v1",
            "status": "PASS",
            "explicit_allowlist_only": True,
            "allowlisted_control_rows": 66,
            "attack_paths_statted": 0,
            "attack_paths_hashed": 0,
            "attack_paths_opened": 0,
            "attack_bytes_read": 0,
            "phase_b_executed": False,
            "actual_spoofing_evaluation_executed": False,
        },
    )
    write_csv(artifact / "replay_completion.csv", REPLAY_FIELDS, [])
    write_csv(artifact / "physical_control_metrics.csv", METRIC_FIELDS, [])
    write_csv(artifact / "positive_response_surface.csv", METRIC_FIELDS, [])
    dump_json(
        artifact / "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.crid-r4-final-verdict.v1",
            "status": "FROZEN_NOT_EXECUTED",
            "verdict": "NOT_EVALUATED_PRE_RESULT_FREEZE",
            "next_state": "RUN_PHASE_A_FROM_PUSHED_FREEZE_SHA",
            "phase_a_executed": False,
            "phase_b_executed": False,
            "attack_bytes_read": 0,
        },
    )
    (artifact / "README.md").write_text(
        "# CRID Stage-0 R4 Phase A clean physical-identifiability\n\n"
        "Pre-result freeze prepared from base `8cf594bde9c8e48bf80b5872e4ca13e1d0d13b0d`. "
        "The exact 66 clean R3 controls, truth sidecars, 45-second packages, receiver, "
        "C0-C3 definitions, clean model source, threshold, code, window, and gates are bound.\n\n"
        "Phase A has not yet been executed. Phase B and actual spoofing data are out of scope.\n"
    )
    dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))
    return {"status": "READY_TO_VERIFY_FREEZE", "inventory_rows": len(inventory)}


def primary_gate_definition() -> dict[str, object]:
    return {
        "clean_holdout_fpr_max_each_domain": 0.02,
        "technical_requirements": ["replay", "terminal_drain", "provenance", "alignment", "support"],
        "negative_case_count": 30,
        "negative_alarm_ratio_q99_max_each": 0.05,
        "positive_case_count": 36,
        "positive_alarm_ratio_q99_min_each_pass": 0.70,
        "positive_group_minimum_passes": {
            "OAK-single": 1,
            "OAK-four": 1,
            "TEX-single": 1,
            "TEX-four": 1,
        },
        "positive_response_surface_complete": True,
        "legacy_any_positive_pass": "DIAGNOSTIC_ONLY_NOT_AUTHORIZATION",
    }


def verify_inventory_row(row: Mapping[str, str], hash_payload: bool = True) -> None:
    domain, case_id = row["domain"], row["case_id"]
    roles = {
        "control": (Path(row["control_path"]), int(row["control_size_bytes"]), row["control_sha256"]),
        "truth_json": (Path(row["truth_json_path"]), int(row["truth_json_size_bytes"]), row["truth_json_sha256"]),
        "truth_epochs": (Path(row["truth_epochs_path"]), int(row["truth_epochs_size_bytes"]), row["truth_epochs_sha256"]),
        "package": (Path(row["package_path"]), int(row["package_size_bytes"]), row["package_sha256"]),
        "smoke_manifest": (
            Path(row["existing_c0_manifest_path"]),
            int(row["existing_c0_manifest_size_bytes"]),
            row["existing_c0_manifest_sha256"],
        ),
    }
    for role, (path, size, expected) in roles.items():
        _assert_exact_control_path(path, domain, case_id, role)
        if not path.is_file() or path.stat().st_size != size:
            raise BindingError(f"{role} size mismatch: {case_id}")
        if hash_payload:
            require_file_binding(path, size, expected)


def preflight_inputs(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != R4_SSD:
        raise BindingError("R4 SSD output root must equal the frozen path")
    rows = read_csv(artifact / "control_input_inventory.csv")
    if len(rows) != 66:
        raise BindingError("control inventory row count mismatch")
    for index, row in enumerate(rows, start=1):
        verify_inventory_row(row, hash_payload=True)
        print(json.dumps({"preflight": index, "case_id": row["case_id"], "status": "PASS"}), flush=True)
    binding = json.loads((artifact / "source_binding.json").read_text())
    if sha256_file(RECEIVER) != binding["receiver"]["sha256"]:
        raise BindingError("receiver SHA-256 mismatch")
    for domain in DOMAIN:
        clean = binding["clean_model_source"][domain]
        if sha256_file(Path(clean["base_config"]["path"])) != clean["base_config"]["sha256"]:
            raise BindingError(f"clean base config mismatch: {domain}")
        for config in CONFIG_ORDER:
            for trace in clean["configurations"][config]["traces"]:
                path = Path(trace["path"])
                expected_dir = CLEAN_ROOT / str(DOMAIN[domain]["clean_dataset"]) / config
                if path.parent != expected_dir:
                    raise BindingError("non-allowlisted clean model path")
                if path.stat().st_size != trace["size_bytes"] or sha256_file(path) != trace["sha256"]:
                    raise BindingError(f"clean model source mismatch: {domain}/{config}/{path.name}")
    result = {
        "schema": "gnss-doppler-lab.crid-r4-input-preflight.v1",
        "status": "PASS",
        "freeze_sha": freeze_sha,
        "verified_at": utc_now(),
        "control_rows": len(rows),
        "worker_count": 1,
        "attack_bytes_read": 0,
    }
    atomic_json(ssd / "input_preflight.json", result)
    return result


def load_clean_tables(binding: Mapping[str, object], domain: str):
    clean = binding["clean_model_source"][domain]
    return {
        config: load_response(config, [Path(row["path"]) for row in clean["configurations"][config]["traces"]])
        for config in CONFIG_ORDER
    }


def recompute_thresholds(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != R4_SSD:
        raise BindingError("R4 SSD output root must equal the frozen path")
    binding = json.loads((artifact / "source_binding.json").read_text())
    domains = {}
    status = "PASS"
    for domain in DOMAIN:
        tables = load_clean_tables(binding, domain)
        _, delays, split, thresholds, _, clean = fit_domain(tables)
        q99 = thresholds["q99"]
        fpr = clean["holdout_fpr_q99"]
        exact = q99 == DOMAIN[domain]["q99"] and fpr == DOMAIN[domain]["holdout_fpr_q99"]
        if not exact:
            status = "INCONCLUSIVE_THRESHOLD_BINDING"
        domains[domain] = {
            "expected_q99": DOMAIN[domain]["q99"],
            "recomputed_q99": q99,
            "expected_holdout_fpr_q99": DOMAIN[domain]["holdout_fpr_q99"],
            "recomputed_holdout_fpr_q99": fpr,
            "exact_match": exact,
            "causal_delays_ms": delays,
            "split": {key: [int(value[0]), int(value[-1]), len(value)] for key, value in split.items()},
        }
    result = {
        "schema": "gnss-doppler-lab.crid-r4-threshold-recomputation.v1",
        "status": status,
        "freeze_sha": freeze_sha,
        "domains": domains,
        "attack_bytes_read": 0,
    }
    atomic_json(ssd / "threshold_recomputation.json", result)
    if status != "PASS":
        raise BindingError("INCONCLUSIVE_THRESHOLD_BINDING")
    return result


def _load_checkpoint(ssd: Path, freeze_sha: str, inventory_sha256: str) -> dict[str, object]:
    path = ssd / "checkpoint.json"
    if path.exists():
        checkpoint = json.loads(path.read_text())
        if checkpoint.get("freeze_sha") != freeze_sha or checkpoint.get("inventory_sha256") != inventory_sha256:
            raise BindingError("checkpoint binding mismatch")
        return checkpoint
    checkpoint = {
        "schema": "gnss-doppler-lab.crid-r4-checkpoint.v1",
        "freeze_sha": freeze_sha,
        "inventory_sha256": inventory_sha256,
        "worker_count": 1,
        "completed": {},
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }
    atomic_json(path, checkpoint)
    return checkpoint


def _aggregate_output_sha(dumps: Iterable[Mapping[str, object]]) -> str:
    value = [{"name": Path(str(row["path"])).name, "size": row["size"], "sha256": row["sha256"]} for row in dumps]
    return sha256_bytes(canonical_json(value).encode())


def validate_completed_replay(
    manifest_path: Path,
    row: Mapping[str, str],
    config: str,
    receiver_sha256: str,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text())
    if not (
        manifest.get("status") == "PASS"
        and manifest.get("domain") == row["domain"]
        and manifest.get("case_id") == row["case_id"]
        and manifest.get("config") == config
        and manifest.get("input", {}).get("sha256") == row["package_sha256"]
        and manifest.get("receiver", {}).get("sha256") == receiver_sha256
        and manifest.get("termination", {}).get("status") == "PASS"
        and manifest.get("native_trace_validation", {}).get("status") == "PASS"
        and manifest.get("target_tracking_pass") is True
    ):
        raise BindingError(f"completed replay manifest contract mismatch: {row['case_id']}/{config}")
    for dump in manifest["dumps"]:
        path = Path(dump["path"])
        if path.stat().st_size != dump["size"] or sha256_file(path) != dump["sha256"]:
            raise BindingError(f"completed replay output hash mismatch: {row['case_id']}/{config}/{path.name}")
    config_file = manifest["config_file"]
    config_path = Path(config_file["path"])
    require_file_binding(config_path, config_file["size_bytes"], config_file["sha256"])
    if _aggregate_output_sha(manifest["dumps"]) != manifest["output_set_sha256"]:
        raise BindingError(f"completed replay aggregate hash mismatch: {row['case_id']}/{config}")
    return manifest


def run_one_replay(
    row: Mapping[str, str],
    config: str,
    out: Path,
    receiver_sha256: str,
) -> dict[str, object]:
    if out.exists():
        raise BindingError(f"incomplete or uncheckpointed output exists: {out}")
    out.mkdir(parents=True, exist_ok=False)
    domain, case_id = row["domain"], row["case_id"]
    package = Path(row["package_path"])
    if package.stat().st_size != int(row["package_size_bytes"]):
        raise BindingError(f"package size changed after preflight: {case_id}")
    scenario = f"CRID.R4.{case_id}.{config}"
    fs = int(DOMAIN[domain]["fs"])
    values = {
        "SignalSource.filename": str(package),
        "SignalSource.seconds_to_skip": 0.0,
        "SignalSource.samples": int(row["package_size_bytes"]) // 2,
        "SignalSource.repeat": "false",
        "Tracking_1C.dump": "false",
        "Tracking_1C.dump_mat": "false",
        "Tracking_1C.trace_dump": "true",
        "Tracking_1C.trace_dump_filename": "trace_native_1ms_ch_",
        "Tracking_1C.trace_scenario_id": scenario,
        "Tracking_1C.trace_raw_sample_offset": 0,
        "Observables.dump": "false",
    }
    cfg = out / "receiver.conf"
    cfg.write_text(render_receiver_config(BASE_CONFIG[domain].read_text(), receiver_configurations()[config], values))
    command = [str(RECEIVER), f"--config_file={cfg}", "--keyboard=false"]
    began = utc_now()
    start = time.monotonic()
    with (out / "receiver.log").open("wb") as log:
        rc, termination = _supervised_run(
            command,
            cwd=out,
            log=log,
            raw=package,
            expected_end_byte=int(row["package_size_bytes"]),
            expected_dump_count=int(DOMAIN[domain]["expected_dumps"]),
        )
    dumps = sorted(out.glob("trace_native_1ms_ch_*.bin"))
    native = validate_dump_files(dumps, expected_scenario_id=scenario, minimum_prns=4) if dumps else {"status": "FAIL"}
    tracked = sorted({prn for summary in native.get("file_summaries", []) for prn in summary["prn_values"]})
    targets = [int(value) for value in row["target_prns"].split(",") if value]
    target_tracking = set(targets).issubset(tracked)
    dump_rows = [
        {"path": summary["path"], "size": summary["byte_size"], "sha256": summary["sha256"]}
        for summary in native.get("file_summaries", [])
    ]
    passed = (
        rc == 0
        and termination.get("status") == "PASS"
        and native.get("status") == "PASS"
        and target_tracking
        and len(dump_rows) == int(DOMAIN[domain]["expected_dumps"])
    )
    manifest = {
        "schema": "gnss-doppler-lab.crid-r4-replay.v1",
        "status": "PASS" if passed else "FAIL",
        "domain": domain,
        "case_id": case_id,
        "config": config,
        "scenario": scenario,
        "command": command,
        "started_at": began,
        "ended_at": utc_now(),
        "elapsed_s": time.monotonic() - start,
        "exit_code": rc,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "input": {
            "path": str(package),
            "size_bytes": int(row["package_size_bytes"]),
            "sha256": row["package_sha256"],
        },
        "receiver": {"path": str(RECEIVER), "sha256": receiver_sha256},
        "configuration_definition_sha256": sha256_bytes(
            canonical_json(receiver_configurations()[config]).encode()
        ),
        "config": config,
        "config_file": {"path": str(cfg), "size_bytes": cfg.stat().st_size, "sha256": sha256_file(cfg)},
        "termination": termination,
        "native_trace_validation": native,
        "target_prns": targets,
        "tracked_prns": tracked,
        "target_tracking_pass": target_tracking,
        "dumps": dump_rows,
        "output_set_sha256": _aggregate_output_sha(dump_rows),
    }
    dump_json(out / "manifest.json", manifest)
    if not passed:
        raise BindingError(f"replay failed closed: {case_id}/{config}")
    return manifest


def _resource_preflight(rows: list[dict[str, str]], ssd: Path) -> dict[str, object]:
    smoke_bytes = 0
    for row in rows:
        smoke = json.loads(Path(row["existing_c0_manifest_path"]).read_text())
        smoke_bytes += sum(int(item["size"]) for item in smoke["dumps"])
    required = smoke_bytes * 4 + 10_000_000_000
    free = shutil.disk_usage(ssd.parent).free
    result = {"free_bytes": free, "required_with_margin_bytes": required, "status": "PASS" if free >= required else "INCONCLUSIVE_RESOURCE"}
    if free < required:
        atomic_json(ssd / "resource_stop.json", result)
        raise BindingError("INCONCLUSIVE_RESOURCE")
    return result


def run_all_replays(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != R4_SSD:
        raise BindingError("R4 SSD output root must equal the frozen path")
    preflight = json.loads((ssd / "input_preflight.json").read_text())
    threshold = json.loads((ssd / "threshold_recomputation.json").read_text())
    if preflight.get("status") != "PASS" or preflight.get("freeze_sha") != freeze_sha:
        raise BindingError("input preflight missing or stale")
    if threshold.get("status") != "PASS" or threshold.get("freeze_sha") != freeze_sha:
        raise BindingError("threshold binding missing or stale")
    rows = read_csv(artifact / "control_input_inventory.csv")
    resource_status = _resource_preflight(rows, ssd)
    inventory_sha = sha256_file(artifact / "control_input_inventory.csv")
    checkpoint = _load_checkpoint(ssd, freeze_sha, inventory_sha)
    source = json.loads((artifact / "source_binding.json").read_text())
    receiver_sha = sha256_file(RECEIVER)
    if receiver_sha != source["receiver"]["sha256"]:
        raise BindingError("receiver changed after preflight")
    for domain in DOMAIN:
        base = source["clean_model_source"][domain]["base_config"]
        require_file_binding(Path(base["path"]), base["size_bytes"], base["sha256"])
    total = len(rows) * len(CONFIG_ORDER)
    completed_count = 0
    for row in rows:
        for config in CONFIG_ORDER:
            key = f"{row['domain']}|{row['case_id']}|{config}"
            out = ssd / "replays" / row["domain"] / row["case_id"] / config
            prior = checkpoint["completed"].get(key)
            if prior is not None:
                manifest_path = Path(prior["manifest_path"])
                if not manifest_path.is_file() or sha256_file(manifest_path) != prior["manifest_sha256"]:
                    raise BindingError(f"checkpoint manifest mismatch: {key}")
                validate_completed_replay(manifest_path, row, config, receiver_sha)
                completed_count += 1
                print(json.dumps({"replay": completed_count, "total": total, "key": key, "status": "SKIP_EXACT_COMPLETE"}), flush=True)
                continue
            if out.exists():
                raise BindingError(f"uncheckpointed output exists; no overwrite/delete permitted: {out}")
            manifest = run_one_replay(row, config, out, receiver_sha)
            manifest_path = out / "manifest.json"
            checkpoint["completed"][key] = {
                "status": "PASS",
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "input_sha256": row["package_sha256"],
                "output_set_sha256": manifest["output_set_sha256"],
            }
            checkpoint["updated_at"] = utc_now()
            atomic_json(ssd / "checkpoint.json", checkpoint)
            completed_count += 1
            print(json.dumps({"replay": completed_count, "total": total, "key": key, "status": "PASS"}), flush=True)
    result = {
        "schema": "gnss-doppler-lab.crid-r4-replay-run.v1",
        "status": "PASS",
        "freeze_sha": freeze_sha,
        "completed_replays": completed_count,
        "expected_replays": total,
        "worker_count": 1,
        "resource_preflight": resource_status,
        "attack_bytes_read": 0,
    }
    atomic_json(ssd / "replay_run_summary.json", result)
    return result


def _truth_active_samples(path: Path) -> np.ndarray:
    rows = np.fromfile(path, dtype=TRUTH_DTYPE)
    active = rows["code_delay_chips"] > 0.0
    return np.unique(rows["absolute_sample"][active])


def support_masks(
    score_samples: np.ndarray,
    replacement_start: int,
    replacement_end: int,
    active_samples: np.ndarray,
    cadence_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    score_samples = np.asarray(score_samples, dtype=np.int64)
    full = (score_samples >= replacement_start) & (score_samples < replacement_end)
    active = np.zeros(len(score_samples), dtype=bool)
    if len(active_samples):
        position = np.searchsorted(active_samples, score_samples, side="right") - 1
        valid = position >= 0
        selected = np.zeros(len(score_samples), dtype=np.int64)
        selected[valid] = active_samples[position[valid]]
        active = valid & (score_samples >= selected) & (score_samples < selected + cadence_samples)
    return full, active


def evaluate_primary_gate(metrics: list[Mapping[str, object]], clean: Mapping[str, object], technical_ok: bool) -> dict[str, object]:
    negative = [row for row in metrics if row["family"] == "negative"]
    positive = [row for row in metrics if row["family"] == "positive"]
    groups = {
        f"{domain}-{mode}": sum(
            row["case_gate_status"] == "PASS"
            for row in positive
            if row["domain"] == domain and row["mode"] == mode
        )
        for domain in DOMAIN
        for mode in ("single", "four")
    }
    checks = {
        "technical_complete": technical_ok,
        "clean_holdout_fpr": all(clean[domain]["holdout_fpr_q99"] <= 0.02 for domain in DOMAIN),
        "negative_count": len(negative) == 30,
        "negative_all_pass": len(negative) == 30 and all(row["case_gate_status"] == "PASS" for row in negative),
        "positive_count": len(positive) == 36,
        "positive_surface_complete": len(positive) == 36,
        "positive_group_coverage": all(value >= 1 for value in groups.values()),
    }
    physical = all(checks.values())
    return {
        "definition": primary_gate_definition(),
        "checks": checks,
        "positive_group_pass_counts": groups,
        "positive_pass_count": sum(row["case_gate_status"] == "PASS" for row in positive),
        "negative_pass_count": sum(row["case_gate_status"] == "PASS" for row in negative),
        "legacy_any_positive_pass": any(row["case_gate_status"] == "PASS" for row in positive),
        "legacy_any_positive_pass_use": "DIAGNOSTIC_ONLY_NOT_AUTHORIZATION",
        "status": "PASS" if physical else "FAIL",
    }


def _replay_compact_row(manifest_path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "domain": manifest["domain"],
        "case_id": manifest["case_id"],
        "config": manifest["config"],
        "status": manifest["status"],
        "exit_code": manifest["exit_code"],
        "terminal_drain_status": manifest["termination"]["status"],
        "native_trace_status": manifest["native_trace_validation"]["status"],
        "target_tracking_pass": manifest["target_tracking_pass"],
        "tracked_prns": ",".join(map(str, manifest["tracked_prns"])),
        "input_sha256": manifest["input"]["sha256"],
        "config_sha256": manifest["config_file"]["sha256"],
        "receiver_sha256": manifest["receiver"]["sha256"],
        "output_set_sha256": manifest["output_set_sha256"],
        "dump_count": len(manifest["dumps"]),
        "output_bytes": sum(int(row["size"]) for row in manifest["dumps"]),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def analyze_phase_a(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != R4_SSD:
        raise BindingError("R4 SSD output root must equal the frozen path")
    replay_summary = json.loads((ssd / "replay_run_summary.json").read_text())
    if replay_summary.get("status") != "PASS" or replay_summary.get("completed_replays") != 264:
        raise BindingError("all 264 replays must pass before analysis")
    rows = read_csv(artifact / "control_input_inventory.csv")
    source = json.loads((artifact / "source_binding.json").read_text())
    receiver_sha = source["receiver"]["sha256"]
    replay_rows = []
    metrics = []
    clean_results = {}
    models = {}
    for domain in DOMAIN:
        tables = load_clean_tables(source, domain)
        model, delays, _, thresholds, _, clean = fit_domain(tables)
        exact = thresholds["q99"] == DOMAIN[domain]["q99"] and clean["holdout_fpr_q99"] == DOMAIN[domain]["holdout_fpr_q99"]
        if not exact:
            raise BindingError("INCONCLUSIVE_THRESHOLD_BINDING")
        models[domain] = (model, delays)
        clean_results[domain] = {
            "q99": thresholds["q99"],
            "holdout_fpr_q99": clean["holdout_fpr_q99"],
            "causal_delays_ms": delays,
            "exact_threshold_match": True,
        }
    for index, row in enumerate(rows, start=1):
        domain, case_id = row["domain"], row["case_id"]
        require_file_binding(Path(row["truth_json_path"]), int(row["truth_json_size_bytes"]), row["truth_json_sha256"])
        require_file_binding(Path(row["truth_epochs_path"]), int(row["truth_epochs_size_bytes"]), row["truth_epochs_sha256"])
        tables = {}
        technical = True
        for config in CONFIG_ORDER:
            manifest_path = ssd / "replays" / domain / case_id / config / "manifest.json"
            manifest = validate_completed_replay(manifest_path, row, config, receiver_sha)
            replay_rows.append(_replay_compact_row(manifest_path, manifest))
            technical &= manifest["status"] == "PASS"
            tables[config] = load_response(config, [Path(item["path"]) for item in manifest["dumps"]])
        model, delays = models[domain]
        scored = score_aligned(tables, model, delays, minimum_prns=4)
        samples = np.asarray([item["sample"] for item in scored], dtype=np.int64)
        scores = np.asarray([item["score"] for item in scored], dtype=float)
        alarms = scores > float(DOMAIN[domain]["q99"])
        active_samples = _truth_active_samples(Path(row["truth_epochs_path"])) if row["family"] == "positive" else np.empty(0, np.int64)
        full, active = support_masks(
            samples,
            int(row["replacement_start_sample"]),
            int(row["replacement_end_sample_exclusive"]),
            active_samples,
            int(DOMAIN[domain]["fs"]) // 1000,
        )
        full_count = int(full.sum())
        active_count = int(active.sum())
        full_alarm = int(alarms[full].sum())
        active_alarm = int(alarms[active].sum())
        full_ratio = float(full_alarm / full_count) if full_count else None
        active_ratio = float(active_alarm / active_count) if active_count else None
        if row["family"] == "positive":
            primary_window = "active_truth_delay_support"
            primary_ratio = active_ratio
            support_ok = active_count > 0
            case_pass = support_ok and active_ratio is not None and active_ratio >= 0.70
        else:
            primary_window = "full_truth_replacement_support"
            primary_ratio = full_ratio
            support_ok = full_count > 0
            case_pass = support_ok and full_ratio is not None and full_ratio <= 0.05
        technical &= bool(scored) and support_ok and all(item["config_count"] == 4 and item["prn_count"] >= 4 for item in scored)
        evidence = ssd / "scores" / domain / f"{case_id}.csv.gz"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(evidence, "wt", newline="") as stream:
            fields = (
                "sample",
                "score",
                "alarm_q99",
                "in_full_support",
                "in_active_support",
                "prn_count",
                "config_count",
                "h0_loglike",
                "h1_loglike",
                "penalty",
            )
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for item, alarm, in_full, in_active in zip(scored, alarms, full, active, strict=True):
                writer.writerow(
                    {
                        "sample": item["sample"],
                        "score": item["score"],
                        "alarm_q99": int(alarm),
                        "in_full_support": int(in_full),
                        "in_active_support": int(in_active),
                        "prn_count": item["prn_count"],
                        "config_count": item["config_count"],
                        "h0_loglike": item["h0_loglike"],
                        "h1_loglike": item["h1_loglike"],
                        "penalty": item["penalty"],
                    }
                )
        evidence_binding = file_binding(evidence)
        metrics.append(
            {
                "domain": domain,
                "case_id": case_id,
                "family": row["family"],
                "mode": row["mode"],
                "delay_chips": row["delay_chips"],
                "power_db": row["power_db"],
                "target_prns": row["target_prns"],
                "technical_status": "PASS" if technical else "INCONCLUSIVE_SUPPORT",
                "valid_full_epochs": full_count,
                "full_alarm_count": full_alarm,
                "alarm_ratio_q99_full_support": full_ratio if full_ratio is not None else "",
                "valid_active_epochs": active_count,
                "active_alarm_count": active_alarm,
                "alarm_ratio_q99_active_support": active_ratio if active_ratio is not None else "",
                "primary_window": primary_window,
                "primary_alarm_ratio_q99": primary_ratio if primary_ratio is not None else "",
                "threshold_q99": DOMAIN[domain]["q99"],
                "case_gate_status": "PASS" if technical and case_pass else ("FAIL" if technical else "INCONCLUSIVE_SUPPORT"),
                "score_evidence_path": evidence_binding["path"],
                "score_evidence_size_bytes": evidence_binding["size_bytes"],
                "score_evidence_sha256": evidence_binding["sha256"],
            }
        )
        print(json.dumps({"analysis": index, "case_id": case_id, "status": metrics[-1]["case_gate_status"]}), flush=True)
    technical_ok = len(replay_rows) == 264 and all(row["technical_status"] == "PASS" for row in metrics)
    gate = evaluate_primary_gate(metrics, clean_results, technical_ok)
    if not technical_ok:
        verdict = "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE"
        next_state = "NOT_AUTHORIZED"
        status = "INCONCLUSIVE"
    elif gate["status"] != "PASS":
        verdict = "NO_GO_CRID_CLEAN_PHYSICAL_IDENTIFIABILITY"
        next_state = "NOT_AUTHORIZED"
        status = "NO_GO"
    else:
        verdict = "CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_PASS"
        next_state = "READY_FOR_CRID_PHASE_B"
        status = "PASS"
    write_csv(artifact / "replay_completion.csv", REPLAY_FIELDS, replay_rows)
    write_csv(artifact / "physical_control_metrics.csv", METRIC_FIELDS, metrics)
    write_csv(artifact / "positive_response_surface.csv", METRIC_FIELDS, [row for row in metrics if row["family"] == "positive"])
    threshold_doc = json.loads((artifact / "clean_threshold_binding.json").read_text())
    threshold_doc["status"] = "PASS"
    for domain in DOMAIN:
        threshold_doc["domains"][domain].update(
            {
                "recomputed_q99": clean_results[domain]["q99"],
                "recomputed_holdout_fpr_q99": clean_results[domain]["holdout_fpr_q99"],
                "causal_delays_ms": clean_results[domain]["causal_delays_ms"],
                "exact_match": True,
            }
        )
    dump_json(artifact / "clean_threshold_binding.json", threshold_doc)
    dump_json(
        artifact / "phase_a_gate.json",
        {
            "schema": "gnss-doppler-lab.crid-r4-phase-a-gate.v1",
            "status": gate["status"],
            "definition": primary_gate_definition(),
            "results": gate,
        },
    )
    dump_json(
        artifact / "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.crid-r4-final-verdict.v1",
            "status": status,
            "verdict": verdict,
            "next_state": next_state,
            "freeze_sha": freeze_sha,
            "phase_a_executed": True,
            "phase_a_replays": 264,
            "phase_b_executed": False,
            "actual_spoofing_evaluation_executed": False,
            "attack_bytes_read": 0,
            "post_result_code_threshold_window_score_gate_changes": False,
        },
    )
    audit = json.loads((artifact / "attack_access_audit.json").read_text())
    audit.update({"status": "PASS", "attack_bytes_read": 0, "phase_a_replays": 264, "phase_b_executed": False})
    dump_json(artifact / "attack_access_audit.json", audit)
    (artifact / "README.md").write_text(
        "# CRID Stage-0 R4 Phase A clean physical-identifiability\n\n"
        f"Final verdict: `{verdict}`\n\nNext state: `{next_state}`\n\n"
        "All 66 frozen R3 clean controls were replayed sequentially through C0/C1/C2/C3 "
        "from the pushed pre-result freeze SHA. Positive primary ratios use only truth epochs "
        "with physically active delay; negative primary ratios use the full replacement support. "
        "Full-support and active-support results are both retained.\n\n"
        "No attack payload was statted, hashed, opened, or mapped. Attack bytes read: 0. "
        "Phase B and actual spoofing evaluation were not executed.\n"
    )
    _write_minimal_plot(artifact, metrics)
    dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))
    result = {"status": status, "verdict": verdict, "next_state": next_state, "gate": gate}
    atomic_json(ssd / "analysis_summary.json", result)
    return result


def _write_minimal_plot(artifact: Path, metrics: list[Mapping[str, object]]) -> None:
    import matplotlib.pyplot as plt

    positive = [row for row in metrics if row["family"] == "positive"]
    labels = [str(row["case_id"]) for row in positive]
    values = [float(row["primary_alarm_ratio_q99"]) for row in positive]
    colors = ["#2a9d8f" if row["case_gate_status"] == "PASS" else "#e76f51" for row in positive]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(range(len(values)), values, color=colors)
    axis.axhline(0.70, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1)
    axis.set_ylabel("active-support q99 alarm ratio")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_title("Frozen CRID R4 positive response surface")
    figure.tight_layout()
    path = artifact / "plots/positive_response_surface.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
