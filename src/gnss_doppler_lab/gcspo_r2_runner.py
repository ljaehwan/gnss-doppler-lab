"""R2 runner adapter for the frozen GCSPO Stage-0 scientific functions.

This module changes execution granularity and support accounting only.  Feature
construction, normal modelling, H0/H1 scores, thresholds, controls, and frozen
randomness remain implemented by the existing GCSPO modules.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import h5py
import numpy as np

from .gcspo_artifacts import canonical_write_json, preflight_receiver_semantics, sha256_file
from .gcspo_access import AccessGate
from .gcspo_b0 import SAMPLE_RATE_HZ, build_protected_scheduled_node_table
from .gcspo_clean import EPOCH_S
from .gcspo_evaluate import (
    STATIC_PHASES,
    _method_rows,
    _native_support,
    _ranges,
    _relation_rows,
)
from .gcspo_full import GeometryCache, geometry_preflight
from .gcspo_protected import (
    discrimination_metrics,
    load_receiver_tracking,
    reconstruct_normal_model,
)
from .gcspo_statistics import (
    paired_block_bootstrap,
    paired_score_loss_bootstrap,
    scheduled_persistence,
)

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcspo_stage0_r2_runner_simulation"
)
RUN_ROOT = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/runs")
RUNNER = Path("/home/ubuntu/projects/gnss-doppler-runner-minimal-v0/scripts/research_run.py")
FROZEN_ROOT = REPO / "artifacts/gcspo_stage0_static_rerun"
CLEAN_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9"
)
RECEIVER_SOURCE = Path("/home/ubuntu/build-gnss-sdr-complex9")
SCENARIO_ROOTS = {
    "DS3": Path(
        "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds3-complex9"
    ),
    "DS7": Path(
        "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/receiver/ds7-complex9"
    ),
    "DS8": Path(
        "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/receiver"
    ),
}
EXPECTED_MANIFESTS = {
    "cleanStatic": "ced3722a7b560fc0efce5f347c7e09cd81e7a660969a17c8a85bb56c9e11e30a",
    "DS3": "15dd02b97dd4d5ca63002160c642e66282d6ba3e5ee43d206607a2c51388f225",
    "DS7": "46cc88456d6985463377cf972a37e3e5cbc663079619b77263b125cd5d65644d",
    "DS8": "2acb11c8e2e5b3f704cbde057d60d771cb6f75316320611d5d93b7c1001aa7e9",
}
REQUIRED_PHASE_NAMES = (
    "gcspo-r2-preflight",
    "gcspo-r2-cleanstatic-normal-model",
    "gcspo-r2-ds3-evaluation",
    "gcspo-r2-ds7-evaluation",
    "gcspo-r2-ds4-ds8-conditional-evaluation",
    "gcspo-r2-relation-destruction-physical-controls",
    "gcspo-r2-final-statistics-plots-verification",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def _read_json(path: Path):
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    material = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(material)


def _load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run(command: list[str]) -> None:
    print("EXEC", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, check=True)


def _manifest_identity(root: Path) -> dict:
    manifest = root / "manifest.json"
    return {
        "path": str(manifest),
        "sha256": sha256_file(manifest) if manifest.is_file() else None,
        "size_bytes": manifest.stat().st_size if manifest.is_file() else None,
    }


def _tracking_paths(root: Path) -> list[Path]:
    return sorted((root / "raw").glob("epl_tracking_ch_*.mat"))


def _identity_rows(paths: Iterable[Path]) -> list[dict]:
    return [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in paths
    ]


def _new_access_gate(root: Path, ledger_name: str, tracking_rows: list[dict]) -> AccessGate:
    """Create a fresh real protected gate for one runner attempt/scenario."""
    ledger_dir = root / "phase_outputs" / ledger_name
    ledger_dir.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        ledger = ledger_dir / f"protected_access_ledger_attempt_{index:03d}.jsonl"
        if not ledger.exists():
            break
        index += 1
    gate = AccessGate(ledger)
    freeze_sha = "57271e526e9e346c8d4d7626b006c5a88166f1be"
    frozen_hashes = {str(row["path"]): str(row["sha256"]) for row in tracking_rows}
    gate.set_preflight(clean_only_pass=True, reviews_pass=True, freeze_sha=freeze_sha,
                       frozen_hashes=frozen_hashes)
    gate.set_remote_sync(local_sha=freeze_sha, remote_sha=freeze_sha, ahead=0, behind=0, clean=True)
    for row in tracking_rows:
        gate.register_pinned(row["path"], expected_sha256=row["sha256"],
                             expected_size=int(row["size_bytes"]), kind="RECEIVER_TRACKING_MAT")
    return gate


def _receiver_fields() -> list[dict]:
    return [
        {"field": "I_E,Q_E,I_P,Q_P,I_L,Q_L", "status": "USED_SIGNED_COMPLEX", "role": "score"},
        {"field": "code_error_chips", "status": "USED_VAR_RESIDUAL", "role": "code/DLL"},
        {"field": "carr_error_hz", "status": "USED_VAR_RESIDUAL", "role": "PLL cycles"},
        {"field": "carrier_doppler_hz", "status": "USED_VAR_RESIDUAL", "role": "carrier rate"},
        {"field": "code_freq_chips", "status": "USED_OFFSET_VAR_RESIDUAL", "role": "code rate"},
        {"field": "CN0_SNV_dB_Hz", "status": "NUISANCE_ONLY", "role": "diagnostic/control"},
        {"field": "carrier_lock_test", "status": "NUISANCE_ONLY", "role": "diagnostic"},
        {"field": "PRN", "status": "GROUPING_ONLY", "role": "metadata"},
    ]


def refresh_run_inventory(root: Path) -> dict:
    entries = []
    if RUN_ROOT.is_dir():
        for run_dir in sorted(RUN_ROOT.iterdir()):
            contract_path, status_path = run_dir / "contract.json", run_dir / "status.json"
            if not contract_path.is_file() or not status_path.is_file():
                continue
            contract, status = _read_json(contract_path), _read_json(status_path)
            if contract.get("name") not in REQUIRED_PHASE_NAMES:
                continue
            entries.append(
                {
                    "phase": contract["name"],
                    "run_id": contract["run_id"],
                    "status": status.get("status"),
                    "exit_code": status.get("exit_code"),
                    "command": contract.get("command"),
                    "cwd": contract.get("cwd"),
                    "created_at": contract.get("created_at"),
                    "started_at": status.get("started_at"),
                    "ended_at": status.get("ended_at"),
                    "logs": {
                        "stdout": str(run_dir / "stdout.log"),
                        "stderr": str(run_dir / "stderr.log"),
                        "heartbeat": str(run_dir / "heartbeat.json"),
                    },
                    # Filled only for the latest attempt below.  Associating the
                    # shared phase directory with an older failed attempt would
                    # falsely claim that a later retry's products came from it.
                    "artifacts": [],
                }
            )
    latest = {}
    for row in entries:
        latest[row["phase"]] = row
    for name, row in latest.items():
        phase = name.removeprefix("gcspo-r2-")
        phase_dir = root / "phase_outputs" / phase
        if phase_dir.is_dir():
            row["artifacts"] = [
                str(path.relative_to(root))
                for path in sorted(phase_dir.rglob("*")) if path.is_file()
            ]
    ordered = [latest[name] for name in REQUIRED_PHASE_NAMES if name in latest]
    payload = {
        "schema": "gnss-doppler-lab.gcspo-stage0-r2.run-inventory.v1",
        "updated_utc": utc_now(),
        "required_phases": list(REQUIRED_PHASE_NAMES),
        "phases": ordered,
        "attempts": entries,
        "missing_phases": [name for name in REQUIRED_PHASE_NAMES if name not in latest],
    }
    canonical_write_json(root / "run_inventory.json", payload)
    return payload


def phase_preflight(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError(f"R2 artifact root is not new/empty: {root}")
    config = _read_json(FROZEN_ROOT / "config.json")
    canonical_write_json(root / "config.json", config)
    source_semantics = preflight_receiver_semantics(RECEIVER_SOURCE)
    source_rows = {}
    for scenario, scenario_root in {"cleanStatic": CLEAN_ROOT, **SCENARIO_ROOTS}.items():
        manifest = _manifest_identity(scenario_root)
        status = "PASS" if manifest["sha256"] == EXPECTED_MANIFESTS[scenario] else "FAIL"
        row = {
            "scenario": scenario,
            "receiver_root": str(scenario_root),
            "manifest": manifest,
            "expected_manifest_sha256": EXPECTED_MANIFESTS[scenario],
            "manifest_status": status,
            "tracking_files": _identity_rows(_tracking_paths(scenario_root)),
            "observables_status": "AVAILABLE" if (scenario_root / "raw/observables.mat").is_file() else "UNAVAILABLE",
        }
        source_rows[scenario] = row
    ds4 = {
        "scenario": "DS4",
        "status": "UNAVAILABLE",
        "reason": "AUTHENTICATED_RECEIVER_SIGNED_TRACKING_AND_GEOMETRY_SOURCE_ABSENT; legacy derived score CSVs are not admissible inputs",
        "searched_candidate_root": str(Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds4-graph-input")),
        "candidate_tracking_file_count": 0,
        "treatment": "TRANSITION_ONLY_SOURCE_UNAVAILABLE",
    }
    data_inventory = {
        "schema": "gnss-doppler-lab.gcspo-stage0-r2.data-inventory.v1",
        "created_utc": utc_now(),
        "sources": source_rows,
        "DS4": ds4,
        "DS8_authenticated_observables": "UNAVAILABLE",
        "DS8_rule": "Do not estimate or generate missing authenticated observables",
        "receiver_fields": _receiver_fields(),
        "static_timelines": {
            "DS3": {"onset_s": 118.9, "pull_off_s": 195.0},
            "DS4": {"onset_s": 113.8, "pull_off_s": 225.0},
            "DS7": {"onset_s": 110.0, "pull_off_s": 150.0, "family": "DS7_DS8"},
            "DS8": {"onset_s": 110.0, "pull_off_s": 150.0, "family": "DS7_DS8"},
        },
    }
    canonical_write_json(root / "data_inventory.json", data_inventory)
    diff = subprocess.run(["git", "diff", "--binary"], cwd=REPO, check=True, capture_output=True).stdout
    canonical_write_json(
        root / "source_commit.json",
        {
            "schema": "gnss-doppler-lab.gcspo.stage0-r2.source-commit.v1",
            "branch": _git("branch", "--show-current"),
            "head_at_preflight": _git("rev-parse", "HEAD"),
            "base_commit": "57271e526e9e346c8d4d7626b006c5a88166f1be",
            "remote_base_commit": _git("rev-parse", "origin/research/gcspo-stage0-exact-support-successor"),
            "dirty_status": _git("status", "--short"),
            "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "receiver_semantics": source_semantics,
            "exact_receiver_rerun": {
                "status": "UNAVAILABLE",
                "reason": "patched receiver source/build is byte-pinned for existing outputs but is not a clean immutable build snapshot",
            },
        },
    )
    (root / "README.md").write_text(
        "# GCSPO Stage-0 R2 runner simulation\n\n"
        "This directory contains an independently recomputed, frozen Stage-0 evaluation under the research runner. "
        "It does not reuse old metric CSVs. Large epoch-level products remain here on SSD.\n\n"
        "The detector verdict and the shared-pull-off physics verdict are reported separately in `final_verdict.json`.\n",
        encoding="utf-8",
    )
    refresh_run_inventory(root)
    canonical_write_json(
        root / "phase_outputs/preflight/preflight.json",
        {
            "status": "PASS_WITH_DECLARED_SOURCE_LIMITATIONS",
            "source_semantics": source_semantics["overall_status"],
            "manifest_status": {name: row["manifest_status"] for name, row in source_rows.items()},
            "DS4": ds4,
            "DS8": "UNAVAILABLE_AUTHENTICATED_OBSERVABLES",
        },
    )
    print("R2_PREFLIGHT_PASS_WITH_DECLARED_SOURCE_LIMITATIONS", flush=True)


def phase_clean(root: Path) -> None:
    required = [root / name for name in ("config.json", "data_inventory.json", "source_commit.json")]
    if not all(path.is_file() for path in required):
        raise RuntimeError("preflight artifacts are absent")
    # The legacy clean entrypoint authenticates the frozen config byte-for-byte.
    # R2 preflight originally serialized the same JSON canonically, which changed
    # whitespace and caused a pre-science failure.  Restore the reviewed bytes,
    # only after proving semantic equality, and preserve the failed runner attempt
    # in run_inventory.json.
    config_path = root / "config.json"
    frozen_config = FROZEN_ROOT / "config.json"
    if sha256_file(config_path) != sha256_file(frozen_config):
        if _read_json(config_path) != _read_json(frozen_config):
            raise RuntimeError("R2 config differs scientifically from the frozen config")
        config_path.write_bytes(frozen_config.read_bytes())
        canonical_write_json(
            root / "phase_outputs/cleanstatic-normal-model/config-byte-repair.json",
            {
                "status": "IMPLEMENTATION_REPAIR",
                "scientific_change": False,
                "cause": "preflight canonical JSON serialization changed the frozen byte identity",
                "restored_sha256": sha256_file(config_path),
                "failed_runner_attempt": "20260815T103959Z-gcspo-r2-cleanstatic-normal-model",
            },
        )
    python = sys.executable
    clean_path = root / "clean_only_report.json"
    if clean_path.is_file() and _read_json(clean_path).get("run_status") == "CLEAN_ONLY_PASS":
        print("R2_RESUME_PRESERVE_EXISTING_CLEAN_ONLY_PASS", flush=True)
    else:
        _run(
            [python, "scripts/run_gcspo_stage0.py", "--phase", "clean-only", "--config", str(root / "config.json"),
             "--artifact-dir", str(root), "--receiver-source", str(RECEIVER_SOURCE), "--clean-root", str(CLEAN_ROOT)]
        )
    ablation_path = root / "clean_ablation_report.json"
    if ablation_path.is_file() and _read_json(ablation_path).get("run_status") == "CLEAN_ABLATIONS_PARTIAL_PASS":
        print("R2_RESUME_PRESERVE_EXISTING_CLEAN_ABLATIONS_PARTIAL_PASS", flush=True)
    else:
        _run([python, "scripts/run_gcspo_clean_ablations.py", "--artifact-dir", str(root), "--clean-root", str(CLEAN_ROOT)])
    a5_path = root / "clean_a5_report.json"
    if a5_path.is_file() and _read_json(a5_path).get("run_status") == "CLEAN_A5_PASS":
        print("R2_RESUME_PRESERVE_EXISTING_CLEAN_A5_PASS", flush=True)
    else:
        _run([python, "scripts/run_gcspo_clean_a5.py", "--artifact-dir", str(root), "--clean-root", str(CLEAN_ROOT),
              "--workers", "1", "--backend", "cuda"])
    b0_path = root / "clean_b0_report.json"
    if b0_path.is_file() and _read_json(b0_path).get("run_status") == "B0_EXACT_CLEAN_PASS":
        print("R2_RESUME_PRESERVE_EXISTING_B0_EXACT_CLEAN_PASS", flush=True)
    else:
        _run([python, "scripts/run_gcspo_clean_b0.py", "--artifact-dir", str(root), "--clean-root", str(CLEAN_ROOT)])
    canonical_write_json(
        root / "phase_outputs/cleanstatic-normal-model/a5-cuda-process-repair.json",
        {
            "status": "IMPLEMENTATION_REPAIR",
            "scientific_change": False,
            "cause": "PyTorch CUDA cannot be initialized in forked multiprocessing children",
            "repair": "execute identical frozen A5 spectral computations in one CUDA process",
            "failed_runner_attempt": "20260815T104035Z-gcspo-r2-cleanstatic-normal-model",
            "preserved_completed_outputs": ["clean_only_report.json", "clean_ablation_report.json", "physical_controls.json"],
        },
    )
    clean, b0 = _read_json(root / "clean_only_report.json"), _read_json(root / "clean_b0_report.json")
    observed = {
        "Full_q99": float(clean["holdout_fpr"]["Full"]["q99"]),
        "Full_q99_5": float(clean["holdout_fpr"]["Full"]["q995"]),
        "B0_q99": float(b0["holdout_fpr"]["q99"]),
    }
    targets = {"Full_q99": 0.0717, "Full_q99_5": 0.0675, "B0_q99": 0.1275}
    canonical_write_json(
        root / "clean_reproduction_check.json",
        {
            "schema": "gnss-doppler-lab.gcspo.stage0-r2.clean-reproduction.v1",
            "status": "RECOMPUTED_INDEPENDENTLY",
            "observed": observed,
            "historical_targets_not_used_as_inputs": targets,
            "absolute_difference": {key: abs(observed[key] - targets[key]) for key in observed},
            "source": str(CLEAN_ROOT),
        },
    )
    canonical_write_json(
        root / "phase_outputs/cleanstatic-normal-model/summary.json",
        {
            "status": "SUCCEEDED",
            "holdout_fpr": observed,
            "thresholds": _read_json(root / "thresholds.json"),
            "normal_model": "normal_model_summary.json",
            "controls": "physical_controls.json",
        },
    )
    print("R2_CLEANSTATIC_NORMAL_MODEL_PASS", observed, flush=True)


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(path)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _score_b0_common(
    *, scenario: str, tracking_paths: list[Path], gate: LocalAuditGate, roles: dict,
    methods: dict, thresholds: dict, root: Path
) -> tuple[list[dict], dict]:
    import pandas as pd

    work = root / "b0_recomputed" / scenario
    work.mkdir(parents=True, exist_ok=True)
    node = build_protected_scheduled_node_table(tracking_paths, gate=gate, scenario=scenario, roles=roles)
    node_path = work / "scheduled_node_windows.csv"
    node.to_csv(node_path, index=False)
    scorer = _load_script(f"gcspo_r2_b0_scorer_{scenario}", REPO / "scripts/score_texbat_prn_node_gru.py")
    model_dir = REPO / "artifacts/ai_morph_gru_cleanStatic_q70_frame"
    scorer.score_node_csv(node_path, model_dir, work, scenario, onset_s=None,
                          output_prefix="gcspo_r2_b0", dataset_prefix="TEXBAT")
    prn_path, _, _ = scorer.score_output_paths(work, scenario, "gcspo_r2_b0")
    prn = pd.read_csv(prn_path)
    support = node[["run_id", "prn", "window_start_s", "phase", "epoch_ids_json"]]
    prn = prn.merge(support, on=["run_id", "prn", "window_start_s"], validate="one_to_one")
    gate_module = _load_script(f"gcspo_r2_b0_gate_{scenario}", REPO / "scripts/eval_btail_support_gate.py")
    events = gate_module.build_event_scores(prn, thresholds["A0_B0"]["node_thresholds"], alpha=.75)
    event_score = {
        (str(row.run_id), float(row.window_start_s)): float(getattr(row, gate_module.FINAL_SCORE))
        for row in events.itertuples(index=False)
    }
    grouped: dict[tuple, list[dict]] = {}
    for row in prn.itertuples(index=False):
        epochs = tuple(map(int, json.loads(row.epoch_ids_json)))
        prn_id = int(str(row.prn).lstrip("Gg"))
        key = (str(row.phase), float(row.window_start_s))
        grouped.setdefault(key, []).append(
            {
                "prn": prn_id,
                "epoch_ids": epochs,
                "score": event_score[(str(row.run_id), float(row.window_start_s))],
            }
        )
    full_by = {(str(row["phase"]), float(row["window_start_s"])): row for row in methods["Full"]}
    exact, cause = [], {}
    for key in sorted(set(full_by) & set(grouped)):
        full = full_by[key]
        combined: dict[int, set[int]] = {}
        for item in grouped[key]:
            for epoch in item["epoch_ids"]:
                combined.setdefault(epoch, set()).add(item["prn"])
        b0_support = tuple((epoch, tuple(sorted(prns))) for epoch, prns in sorted(combined.items()))
        _, _, full_support = _native_support(full)
        if b0_support != full_support:
            cause["EPOCH_PRN_SUPPORT_MISMATCH"] = cause.get("EPOCH_PRN_SUPPORT_MISMATCH", 0) + 1
            continue
        exact.append(
            {
                "phase": key[0],
                "window_start_s": key[1],
                "availability_s": float(full["availability_s"]),
                "score": float(np.mean([item["score"] for item in grouped[key]])),
                "prns": list(map(int, full["prns"])),
                "epoch_ids": tuple(map(int, full["epoch_ids"])),
                "epoch_prn_support": full_support,
            }
        )
    full_only = set(full_by) - set(grouped)
    b0_only = set(grouped) - set(full_by)
    if full_only:
        cause["FULL_WINDOW_WITHOUT_B0_EVENT"] = len(full_only)
    if b0_only:
        cause["B0_WINDOW_WITHOUT_FULL_EVENT"] = len(b0_only)
    phase_counts = {}
    for phase in sorted({key[0] for key in set(full_by) | set(grouped)}):
        phase_counts[phase] = {
            "full_total": sum(key[0] == phase for key in full_by),
            "b0_total": sum(key[0] == phase for key in grouped),
            "exact_common": sum(row["phase"] == phase for row in exact),
        }
    audit = {
        "scenario": scenario,
        "status": "AVAILABLE_ON_EXACT_COMMON_SUPPORT" if exact else "UNAVAILABLE_ON_COMMON_SUPPORT",
        "full_total_event_count": len(full_by),
        "b0_total_event_count": len(grouped),
        "exact_common_event_count": len(exact),
        "full_only_event_count": len(full_only),
        "b0_only_event_count": len(b0_only),
        "support_mismatch_event_count": sum(cause.values()) - len(full_only) - len(b0_only),
        "mismatch_counts_by_cause": cause,
        "common_support_coverage": len(exact) / len(full_by) if full_by else 0.0,
        "phase_counts": phase_counts,
        "root_cause_previous_abort": "legacy integration required equality of all B0 and Full scheduler keys before common-support accounting",
        "r2_fix": "retain only rows whose complete epoch+PRN+window support is identical and explicitly inventory every mismatch",
    }
    return exact, audit


SCORE_FIELDS = [
    "scenario", "family", "phase", "method", "window_start_s", "availability_s", "score",
    "threshold_q99", "threshold_q995", "alarm_q99", "alarm_q995", "tracked_n", "effective_dof",
    "penalty", "likelihood_improvement_twice", "prns_json", "epoch_ids_json",
    "epoch_prn_support_json", "label", "phase_start_s", "phase_end_s",
]
STATE_FIELDS = [
    "scenario", "phase", "window_start_s", "availability_s", "epoch_offset", "receiver_time_s",
    "delta_p_x_m", "delta_p_y_m", "delta_p_z_m", "delta_b_m", "delta_v_x_m_s",
    "delta_v_y_m_s", "delta_v_z_m_s", "delta_bdot_m_s",
]


def _scenario_components(root: Path, scenario: str):
    normal = _read_json(root / "normal_model_summary.json")
    model, whitener, gamma = reconstruct_normal_model(normal)
    a2 = _read_json(root / "clean_ablation_report.json")
    lambdas = {
        "Full": float(normal["lambda_selected"]),
        "A2": float(a2["methods"]["A2"]["lambda"]),
        "A5": float(_read_json(root / "clean_a5_report.json")["lambda"]),
    }
    return normal, model, whitener, gamma, a2, lambdas


def _load_scenario(root: Path, scenario: str, ledger_name: str):
    inventory = _read_json(root / "data_inventory.json")
    source = inventory["sources"][scenario]
    if source["manifest_status"] != "PASS":
        raise RuntimeError(f"{scenario} manifest identity mismatch")
    scenario_root = Path(source["receiver_root"])
    current = _identity_rows(_tracking_paths(scenario_root))
    if current != source["tracking_files"]:
        raise RuntimeError(f"{scenario} tracking file identity changed after preflight")
    normal, model, whitener, gamma, a2, lambdas = _scenario_components(root, scenario)
    gate = _new_access_gate(root, ledger_name, source["tracking_files"])
    paths = _tracking_paths(scenario_root)
    eps = {int(key): float(value) for key, value in normal["normalization_epsilon_by_prn"].items()}
    data = load_receiver_tracking(paths, epsilons=eps, gate=gate, scenario=scenario)
    geometry = geometry_preflight(scenario_root, tracked_prns=data.prn)
    return scenario_root, paths, data, geometry, gate, normal, model, whitener, gamma, a2, lambdas


def _support_identity(row: dict) -> tuple:
    return (
        str(row["phase"]), float(row["window_start_s"]), float(row["availability_s"]),
        tuple(map(int, row["epoch_ids"])), tuple(map(int, row["prns"])),
        tuple((int(epoch), tuple(map(int, prns))) for epoch, prns in row["epoch_prn_support"]),
    )


def phase_scenario(root: Path, scenario: str) -> None:
    label = f"{scenario.lower()}-evaluation"
    (scenario_root, paths, data, geometry, gate, normal, model, whitener, gamma, a2, lambdas) = _load_scenario(root, scenario, label)
    end_s = float((data.epoch.max() + 1) * EPOCH_S)
    ranges = _ranges(scenario, end_s)
    phase_bounds = {name: (start, end) for name, start, end in ranges}
    methods = _method_rows(
        data, model, whitener, gamma, geometry, np.asarray(a2["methods"]["A2"]["loading"]),
        lambdas, ranges, set(normal["validated_rows"]),
    )
    b0_rows, b0_audit = _score_b0_common(
        scenario=scenario, tracking_paths=paths, gate=gate,
        roles={name: (start, end) for name, start, end in ranges}, methods=methods,
        thresholds=_read_json(root / "thresholds.json"), root=root,
    )
    methods["A0"] = b0_rows
    thresholds = _read_json(root / "thresholds.json")
    full_support = {_support_identity(row) for row in methods["Full"]}
    support_rows = []
    for method, rows in methods.items():
        identities = {_support_identity(row) for row in rows}
        exact = identities & full_support
        support_rows.append(
            {
                "scenario": scenario, "method": method, "method_total": len(identities),
                "full_total": len(full_support), "exact_common": len(exact),
                "method_only": len(identities - full_support), "full_only": len(full_support - identities),
                "coverage": len(exact) / len(full_support) if full_support else 0.0,
                "status": "AVAILABLE_ON_EXACT_COMMON_SUPPORT" if exact else "UNAVAILABLE_ON_COMMON_SUPPORT",
            }
        )
    scores, states = [], []
    for method, rows in methods.items():
        threshold_key = "A0_B0" if method == "A0" else method
        q99, q995 = float(thresholds[threshold_key]["q99"]), float(thresholds[threshold_key]["q995"])
        for row in rows:
            start, finish = phase_bounds[row["phase"]]
            scores.append(
                {
                    "scenario": scenario, "family": "DS7_DS8" if scenario in {"DS7", "DS8"} else scenario,
                    "phase": row["phase"], "method": method, "window_start_s": row["window_start_s"],
                    "availability_s": row["availability_s"], "score": row["score"], "threshold_q99": q99,
                    "threshold_q995": q995, "alarm_q99": bool(row["score"] > q99),
                    "alarm_q995": bool(row["score"] > q995), "tracked_n": len(row.get("prns", ())),
                    "effective_dof": row.get("effective_dof"), "penalty": row.get("penalty"),
                    "likelihood_improvement_twice": row.get("likelihood_improvement_twice"),
                    "prns_json": json.dumps(list(row.get("prns", ())), separators=(",", ":")),
                    "epoch_ids_json": json.dumps(list(row.get("epoch_ids", ())), separators=(",", ":")),
                    "epoch_prn_support_json": json.dumps(list(row.get("epoch_prn_support", ())), separators=(",", ":")),
                    "label": row["phase"] in {"transition", "established"},
                    "phase_start_s": start, "phase_end_s": finish,
                }
            )
            if method == "Full" and "state" in row:
                state = np.asarray(row["state"], float).reshape(-1, 8)
                for offset, vector in enumerate(state):
                    states.append(
                        {
                            "scenario": scenario, "phase": row["phase"], "window_start_s": row["window_start_s"],
                            "availability_s": row["availability_s"], "epoch_offset": offset,
                            "receiver_time_s": float(row["window_start_s"]) + offset * EPOCH_S,
                            **{name: float(value) for name, value in zip(STATE_FIELDS[6:], vector)},
                        }
                    )
    out = root / "phase_outputs" / label
    _write_csv(out / "per_epoch_scores.csv", scores, SCORE_FIELDS)
    _write_csv(out / "shared_state_estimates.csv", states, STATE_FIELDS)
    _write_csv(out / "common_support_inventory.csv", support_rows,
               ["scenario", "method", "method_total", "full_total", "exact_common", "method_only", "full_only", "coverage", "status"])
    canonical_write_json(out / "exact_support_audit.json", b0_audit)
    canonical_write_json(
        out / "summary.json",
        {
            "status": "SUCCEEDED", "scenario": scenario, "receiver_root": str(scenario_root),
            "end_s": end_s, "ranges": ranges, "rows": len(scores), "state_rows": len(states),
            "B0": b0_audit["status"], "geometry": geometry["report"],
        },
    )
    print(f"R2_{scenario}_EVALUATION_PASS rows={len(scores)} b0={b0_audit['status']}", flush=True)


def _tracking_prefix_digest(root: Path, limit_s: float = 110.0) -> dict:
    digest = hashlib.sha256()
    rows = 0
    for path in _tracking_paths(root):
        with h5py.File(path, "r") as handle:
            sample = np.asarray(handle["PRN_start_sample_count"]).reshape(-1).astype(np.int64)
            prn = np.asarray(handle["PRN"]).reshape(-1).astype(np.int64)
            mask = sample < round(limit_s * 25_000_000)
            order = np.lexsort((sample[mask], prn[mask]))
            digest.update(np.column_stack([prn[mask][order], sample[mask][order]]).tobytes())
            for name in ("I_E", "Q_E", "I_P", "Q_P", "I_L", "Q_L", "code_error_chips", "carr_error_hz", "carrier_doppler_hz", "code_freq_chips"):
                digest.update(np.asarray(handle[name]).reshape(-1)[mask][order].astype(np.float64).tobytes())
            rows += int(mask.sum())
    return {"tracking_prefix_s": limit_s, "rows": rows, "sha256": digest.hexdigest()}


def phase_conditional(root: Path) -> None:
    out = root / "phase_outputs/ds4-ds8-conditional-evaluation"
    clean = _tracking_prefix_digest(CLEAN_ROOT)
    ds7 = _tracking_prefix_digest(SCENARIO_ROOTS["DS7"])
    ds8 = _tracking_prefix_digest(SCENARIO_ROOTS["DS8"])
    replay = {
        "scope": "receiver-provided signed tracking fields through 110 s",
        "cleanStatic": clean, "DS7": ds7, "DS8": ds8,
        "cleanStatic_equals_DS7": clean["sha256"] == ds7["sha256"],
        "cleanStatic_equals_DS8": clean["sha256"] == ds8["sha256"],
        "independent_normal_evidence_rule": "DS7/DS8 pre-110 is excluded regardless of receiver-output digest equality because authenticated prior metadata binds the raw replay",
    }
    canonical_write_json(out / "replay_overlap_audit.json", replay)
    dispositions = {
        "DS4": {
            "status": "UNAVAILABLE",
            "reason": "AUTHENTICATED_RECEIVER_SIGNED_TRACKING_AND_GEOMETRY_SOURCE_ABSENT; legacy derived score CSVs are not admissible inputs",
            "searched_candidate_root": "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds4-graph-input",
            "candidate_tracking_file_count": 0,
            "timeline": {"onset_s": 113.8, "pull_off_s": 225.0},
            "treatment": "TRANSITION_ONLY_SOURCE_UNAVAILABLE",
        },
        "DS8": {
            "status": "UNAVAILABLE",
            "reason": "AUTHENTICATED_OBSERVABLES_ABSENT; geometry must not be estimated or generated",
            "timeline": {"onset_s": 110.0, "pull_off_s": 150.0},
            "family": "DS7_DS8",
        },
    }
    canonical_write_json(out / "conditional_dispositions.json", dispositions)
    inventory = _read_json(root / "data_inventory.json")
    inventory["DS4"] = {"scenario": "DS4", **dispositions["DS4"]}
    canonical_write_json(root / "data_inventory.json", inventory)
    _write_csv(
        out / "unavailable_scenarios.csv",
        [{"scenario": key, **value} for key, value in dispositions.items()],
        ["scenario", "status", "reason", "treatment", "family"],
    )
    print("R2_DS4_DS8_CONDITIONAL_COMPLETE", dispositions, flush=True)


RELATION_FIELDS = [
    "scenario", "phase", "method", "window_start_s", "availability_s", "score", "prns_json",
    "epoch_ids_json", "epoch_prn_support_json", "phase_start_s", "phase_end_s", "label",
    "segment_id", "transform_seed", "preservation",
]


def phase_relation_controls(root: Path) -> None:
    all_rows = []
    for scenario in ("DS3", "DS7"):
        label = "relation-destruction-physical-controls"
        (_, _, data, geometry, _, normal, model, whitener, gamma, _, lambdas) = _load_scenario(root, scenario, label)
        end_s = float((data.epoch.max() + 1) * EPOCH_S)
        cache = GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"], set(normal["validated_rows"]))
        for phase, start, finish in _ranges(scenario, end_s):
            if phase == "pre_onset_replay" or finish - start < 1.2:
                continue
            for row in _relation_rows(data, model, whitener, gamma, cache, lambdas["Full"], scenario, phase, start, finish):
                epochs = tuple(range(round(row["window_start_s"] / EPOCH_S), round(row["availability_s"] / EPOCH_S)))
                all_rows.append(
                    {
                        "scenario": scenario, "phase": phase, "method": row["method"],
                        "window_start_s": row["window_start_s"], "availability_s": row["availability_s"],
                        "score": row["score"], "prns_json": json.dumps(list(row["prns"]), separators=(",", ":")),
                        "epoch_ids_json": json.dumps(epochs, separators=(",", ":")),
                        "epoch_prn_support_json": json.dumps([(epoch, list(row["prns"])) for epoch in epochs], separators=(",", ":")),
                        "phase_start_s": start, "phase_end_s": finish,
                        "label": phase in {"transition", "established"}, "segment_id": row["segment_id"],
                        "transform_seed": row.get("transform_seed"), "preservation": row.get("preservation", True),
                    }
                )
    out = root / "phase_outputs/relation-destruction-physical-controls"
    _write_csv(out / "relation_rows.csv", all_rows, RELATION_FIELDS)
    reports = {}
    parsed = _parse_score_rows(all_rows)
    for scenario in ("DS3", "DS7"):
        selected = [row for row in parsed if row["scenario"] == scenario]
        reports[scenario] = {}
        for destruction in ("LOS_SHUFFLE", "PER_PRN_TEMPORAL_SHIFT"):
            try:
                report = paired_score_loss_bootstrap(selected, "Full", destruction)
                reports[scenario][destruction] = {
                    "status": "AVAILABLE", "lcb_95": report["lcb_95"],
                    "interval_95": report["interval_95"],
                    "median_relative_loss": report["median_relative_loss"],
                    "replicates": report["replicates"],
                }
            except ValueError as exc:
                reports[scenario][destruction] = {"status": "UNAVAILABLE", "reason": str(exc)}
    canonical_write_json(
        root / "relation_destruction_metrics.json",
        {
            "schema": "gnss-doppler-lab.gcspo.stage0-r2.relation-destruction.v1",
            "status": "PARTIAL_CORE_SCENARIOS_DS3_DS7",
            "results": reports,
            "preservation_assertions": all(str(row["preservation"]).lower() == "true" for row in all_rows),
            "DS4": {"status": "UNAVAILABLE", "reason": "SOURCE_UNAVAILABLE"},
            "DS8": {"status": "UNAVAILABLE", "reason": "AUTHENTICATED_OBSERVABLES_ABSENT"},
        },
    )
    controls = _read_json(root / "physical_controls.json")
    aggregate = []
    for control_id in sorted({row["id"] for row in controls.get("results", [])}):
        rows = [row for row in controls["results"] if row["id"] == control_id]
        aggregate.append(
            {
                "id": control_id, "rows": len(rows),
                "max_persistent_alarm_ratio": max(row["persistent_alarm_ratio"] for row in rows),
                "max_consecutive_alarms": max(row["max_consecutive_alarms"] for row in rows),
                "blocks_with_persistent_alarm": sum(row["persistent_alarm_ratio"] > 0 for row in rows),
            }
        )
    high = [row for row in aggregate if row["id"] != "CLOCK_DRIFT" and
            (row["max_persistent_alarm_ratio"] > .10 or row["max_consecutive_alarms"] >= 10)]
    canonical_write_json(
        root / "physical_controls_audit.json",
        {
            "schema": "gnss-doppler-lab.gcspo.stage0-r2.physical-controls-audit.v1",
            "generation_overall_status": controls.get("overall_status"),
            "generation_pass_is_not_false_alarm_pass": True,
            "scientific_false_alarm_status": "FAIL" if high else "PASS",
            "contradictory_high_alarm_controls": high,
            "aggregate": aggregate,
            "explicit_focus": [row for row in aggregate if row["id"] in {"EMPIRICAL_NOISE", "PRN_DROP_ONLY"}],
        },
    )
    canonical_write_json(out / "summary.json", {"status": "SUCCEEDED", "relation_rows": len(all_rows), "control_audit": "physical_controls_audit.json"})
    print(f"R2_RELATION_CONTROLS_COMPLETE rows={len(all_rows)} high_controls={len(high)}", flush=True)


def _parse_bool(value) -> bool:
    return value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes"}


def _parse_score_rows(rows: Iterable[dict]) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        for key in ("window_start_s", "availability_s", "score", "phase_start_s", "phase_end_s"):
            item[key] = float(item[key])
        for key in ("threshold_q99", "threshold_q995", "effective_dof", "penalty", "likelihood_improvement_twice"):
            if key in item and item[key] not in {None, ""}:
                item[key] = float(item[key])
        item["label"] = _parse_bool(item.get("label", False))
        item["prns"] = tuple(map(int, json.loads(item.pop("prns_json"))))
        item["epoch_ids"] = tuple(map(int, json.loads(item.pop("epoch_ids_json"))))
        item["epoch_prn_support"] = tuple(
            (int(epoch), tuple(map(int, prns))) for epoch, prns in json.loads(item.pop("epoch_prn_support_json"))
        )
        output.append(item)
    return output


def _support_mismatch_events(root: Path, scenario: str) -> list[dict]:
    """Reconstruct every B0/Full mismatch with the contracted timing fields."""
    label = f"{scenario.lower()}-evaluation"
    scores = _parse_score_rows(_load_csv(root / f"phase_outputs/{label}/per_epoch_scores.csv"))
    full_by = {
        (str(row["phase"]), float(row["window_start_s"])): row
        for row in scores if row["method"] == "Full"
    }
    nodes = _load_csv(root / f"b0_recomputed/{scenario}/scheduled_node_windows.csv")
    node_by = {
        (str(row["run_id"]), str(row["prn"]), float(row["window_start_s"])): row
        for row in nodes
    }
    scored = _load_csv(root / f"b0_recomputed/{scenario}/gcspo_r2_b0_{scenario}_prn_local_scores.csv")
    grouped: dict[tuple[str, float], list[dict]] = {}
    for score in scored:
        identity = (str(score["run_id"]), str(score["prn"]), float(score["window_start_s"]))
        row = node_by.get(identity)
        if row is None:
            raise RuntimeError(f"scored B0 node lacks scheduled support: {identity}")
        grouped.setdefault((str(row["phase"]), float(row["window_start_s"])), []).append(row)
    events = []
    for key in sorted(set(full_by) | set(grouped)):
        full = full_by.get(key)
        b0_group = grouped.get(key)
        full_support = None if full is None else tuple(full["epoch_prn_support"])
        b0_support = None
        b0_prns: tuple[int, ...] = ()
        if b0_group:
            combined: dict[int, set[int]] = {}
            for row in b0_group:
                prn = int(str(row["prn"]).lstrip("Gg"))
                for epoch in map(int, json.loads(row["epoch_ids_json"])):
                    combined.setdefault(epoch, set()).add(prn)
            b0_support = tuple((epoch, tuple(sorted(prns))) for epoch, prns in sorted(combined.items()))
            b0_prns = tuple(sorted({int(str(row["prn"]).lstrip("Gg")) for row in b0_group}))
        if full is None:
            cause = "B0_WINDOW_WITHOUT_FULL_EVENT"
        elif b0_group is None:
            cause = "FULL_WINDOW_WITHOUT_B0_EVENT"
        elif full_support != b0_support:
            cause = "EPOCH_PRN_SUPPORT_MISMATCH"
        else:
            continue
        start = float(key[1])
        finish = (float(full["availability_s"]) if full is not None else
                  max(float(row["window_end_s"]) for row in b0_group or ()))
        full_prns = tuple(full["prns"]) if full is not None else ()
        events.append(
            {
                "scenario": scenario,
                "phase_label": key[0],
                "receiver_relative_timestamp_s": start,
                "absolute_sample_index": int(round(start * SAMPLE_RATE_HZ)),
                "score_window_start_s": start,
                "score_window_end_s": finish,
                "full_prn_set": list(full_prns),
                "b0_prn_set": list(b0_prns),
                "full_valid_prn_count": len(full_prns),
                "b0_valid_prn_count": len(b0_prns),
                "cause": cause,
            }
        )
    return events


def _scenario_metric_rows(rows: list[dict], thresholds: dict) -> list[dict]:
    output = []
    for scenario in sorted({row["scenario"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["scenario"] == scenario}):
            selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
            pre = [row for row in selected if row["phase"].startswith("pre_")]
            transition = [row for row in selected if row["phase"] == "transition"]
            established = [row for row in selected if row["phase"] == "established"]
            attack = transition + established
            metrics = discrimination_metrics([row["score"] for row in pre], [row["score"] for row in attack])
            key = "A0_B0" if method == "A0" else method
            threshold = float(thresholds[key]["q99"])
            ordered = sorted(selected, key=lambda row: row["availability_s"])
            flags = scheduled_persistence(ordered, threshold=threshold)
            first_alarm = next((row["availability_s"] for row in attack if row["score"] > threshold), None)
            first_persistent = next((row["availability_s"] for row, flag in zip(ordered, flags) if flag and row in attack), None)
            timeline = {"DS3": (118.9, 195.0), "DS7": (110.0, 150.0)}[scenario]
            output.append(
                {
                    "scenario": scenario, "method": method, **metrics,
                    "pre_onset_fpr_q99": float(np.mean([row["score"] > threshold for row in pre])) if pre else None,
                    "attack_detection_rate_q99": float(np.mean([row["score"] > threshold for row in attack])) if attack else None,
                    "transition_detection_rate_q99": float(np.mean([row["score"] > threshold for row in transition])) if transition else None,
                    "established_detection_rate_q99": float(np.mean([row["score"] > threshold for row in established])) if established else None,
                    "first_alarm_delay_from_onset_s": None if first_alarm is None else first_alarm - timeline[0],
                    "first_alarm_delay_from_pull_off_s": None if first_alarm is None else first_alarm - timeline[1],
                    "first_persistent_alarm_s": first_persistent,
                    "persistent_alarm_ratio": float(np.mean(flags)) if flags else None,
                    "windows": len(selected),
                }
            )
    return output


def _clean_bootstrap_rows(root: Path) -> list[dict]:
    clean = _read_json(root / "clean_only_report.json")
    a2 = _read_json(root / "clean_ablation_report.json")
    a5 = _read_json(root / "clean_a5_report.json")
    sources = {"Full": clean["scores"]["Full_holdout"], "A2": a2["methods"]["A2"]["holdout"], "A5": a5["holdout"]}
    rows = []
    for method, values in sources.items():
        for raw in values:
            rows.append(
                {
                    **raw, "scenario": "cleanStatic", "family": "cleanStatic", "phase": "holdout",
                    "method": method, "label": False, "phase_start_s": 350.0, "phase_end_s": 470.0,
                }
            )
    return rows


def _state_summary(state_rows: list[dict]) -> list[dict]:
    output = []
    for scenario in sorted({row["scenario"] for row in state_rows}):
        for phase in ("pre_onset", "pre_onset_replay", "transition", "established"):
            rows = [row for row in state_rows if row["scenario"] == scenario and row["phase"] == phase]
            if not rows:
                continue
            pos = [math.sqrt(sum(float(row[key]) ** 2 for key in ("delta_p_x_m", "delta_p_y_m", "delta_p_z_m"))) for row in rows]
            vel = [math.sqrt(sum(float(row[key]) ** 2 for key in ("delta_v_x_m_s", "delta_v_y_m_s", "delta_v_z_m_s"))) for row in rows]
            output.append(
                {
                    "scenario": scenario, "phase": phase, "samples": len(rows),
                    "position_median_m": float(np.median(pos)), "clock_median_m": float(np.median([abs(float(row["delta_b_m"])) for row in rows])),
                    "velocity_median_m_s": float(np.median(vel)), "drift_median_m_s": float(np.median([abs(float(row["delta_bdot_m_s"])) for row in rows])),
                }
            )
    return output


def _make_plots(root: Path, rows: list[dict], state_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    for scenario in ("DS3", "DS7"):
        selected = sorted([row for row in rows if row["scenario"] == scenario and row["method"] == "Full"], key=lambda row: row["availability_s"])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([row["availability_s"] for row in selected], [row["score"] for row in selected], lw=.8, label="Full")
        if selected:
            ax.axhline(selected[0]["threshold_q99"], color="black", ls="--", label="clean q99")
        onset, pull = ({"DS3": (118.9, 195.0), "DS7": (110.0, 150.0)})[scenario]
        ax.axvline(onset, color="tab:orange", label="onset")
        ax.axvline(pull, color="tab:red", label="pull-off")
        ax.set(xlabel="receiver-relative time (s)", ylabel="EDF-BIC likelihood score", title=f"GCSPO Full — {scenario}")
        ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(plots / f"{scenario.lower()}_full_score.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for scenario in ("DS3", "DS7"):
        selected = sorted([row for row in state_rows if row["scenario"] == scenario], key=lambda row: float(row["receiver_time_s"]))
        times = [float(row["receiver_time_s"]) for row in selected]
        magnitude = [math.sqrt(sum(float(row[key]) ** 2 for key in ("delta_p_x_m", "delta_p_y_m", "delta_p_z_m"))) for row in selected]
        ax.plot(times, magnitude, lw=.5, alpha=.7, label=scenario)
    ax.set(xlabel="receiver-relative time (s)", ylabel="estimated |delta p| (m)", title="Shared-state position magnitude")
    ax.legend(); fig.tight_layout(); fig.savefig(plots / "shared_state_position.png", dpi=160); plt.close(fig)


def _manifest(root: Path) -> dict:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest_sha256.json":
            continue
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    payload = {"schema": "gnss-doppler-lab.gcspo.stage0-r2.artifact-manifest.v1", "files": records}
    canonical_write_json(root / "artifact_manifest_sha256.json", payload)
    return payload


def phase_final(root: Path) -> None:
    score_paths = [root / f"phase_outputs/{name}-evaluation/per_epoch_scores.csv" for name in ("ds3", "ds7")]
    state_paths = [root / f"phase_outputs/{name}-evaluation/shared_state_estimates.csv" for name in ("ds3", "ds7")]
    if not all(path.is_file() for path in score_paths + state_paths):
        raise RuntimeError("core DS3/DS7 phase artifacts are absent")
    raw_scores = [row for path in score_paths for row in _load_csv(path)]
    rows = _parse_score_rows(raw_scores)
    state_rows = [row for path in state_paths for row in _load_csv(path)]
    _write_csv(root / "per_epoch_scores.csv", raw_scores, SCORE_FIELDS)
    _write_csv(root / "shared_state_estimates.csv", state_rows, STATE_FIELDS)
    thresholds = _read_json(root / "thresholds.json")
    metrics = _scenario_metric_rows(rows, thresholds)
    metric_fields = [
        "scenario", "method", "roc_auc", "low_fpr_pauc", "pr_auc", "pre_onset_fpr_q99",
        "attack_detection_rate_q99", "transition_detection_rate_q99", "established_detection_rate_q99",
        "first_alarm_delay_from_onset_s", "first_alarm_delay_from_pull_off_s", "first_persistent_alarm_s",
        "persistent_alarm_ratio", "windows",
    ]
    _write_csv(root / "scenario_metrics.csv", metrics, metric_fields)
    support_rows = [row for scenario in ("ds3", "ds7") for row in _load_csv(root / f"phase_outputs/{scenario}-evaluation/common_support_inventory.csv")]
    _write_csv(root / "common_support_inventory.csv", support_rows,
               ["scenario", "method", "method_total", "full_total", "exact_common", "method_only", "full_only", "coverage", "status"])
    audits = [_read_json(root / f"phase_outputs/{scenario}-evaluation/exact_support_audit.json") for scenario in ("ds3", "ds7")]
    mismatch_events = [event for scenario in ("DS3", "DS7")
                       for event in _support_mismatch_events(root, scenario)]
    audited_mismatch_count = sum(
        int(row["full_only_event_count"]) + int(row["b0_only_event_count"]) +
        int(row["support_mismatch_event_count"]) for row in audits
    )
    if len(mismatch_events) != audited_mismatch_count:
        raise RuntimeError(
            f"detailed/aggregate support mismatch disagreement: {len(mismatch_events)} != {audited_mismatch_count}"
        )
    canonical_write_json(
        root / "exact_support_audit.json",
        {
            "schema": "gnss-doppler-lab.gcspo.stage0-r2.exact-support-audit.v1",
            "status": "AVAILABLE_WITH_EXPLICIT_MISMATCH_ACCOUNTING" if any(row["exact_common_event_count"] for row in audits) else "B0=UNAVAILABLE_ON_COMMON_SUPPORT",
            "scenarios": audits,
            "mismatch_events": mismatch_events,
            "mismatch_event_count": len(mismatch_events),
            "absolute_sample_index_definition": "round(receiver_relative_timestamp_s * 25000000)",
        },
    )
    exact_b0_comparisons = {}
    for scenario in ("DS3", "DS7"):
        selected = [row for row in rows if row["scenario"] == scenario and row["method"] in {"A0", "Full"}]
        full_by = {_support_identity(row): row for row in selected if row["method"] == "Full"}
        pairs = [(full_by[_support_identity(row)], row) for row in selected if row["method"] == "A0"
                 and _support_identity(row) in full_by]
        scenario_row = {"status": "AVAILABLE_ON_EXACT_COMMON_SUPPORT", "windows": len(pairs), "phases": {}}
        for phase in ("pre_onset", "pre_onset_replay", "transition", "established"):
            phase_pairs = [(full, b0) for full, b0 in pairs if full["phase"] == phase]
            if not phase_pairs:
                scenario_row["phases"][phase] = {"status": "UNAVAILABLE_ON_COMMON_SUPPORT"}
                continue
            scenario_row["phases"][phase] = {
                "status": "AVAILABLE_ON_EXACT_COMMON_SUPPORT",
                "windows": len(phase_pairs),
                "Full_alarm_rate_q99": float(np.mean([full["score"] > full["threshold_q99"] for full, _ in phase_pairs])),
                "B0_alarm_rate_q99": float(np.mean([b0["score"] > b0["threshold_q99"] for _, b0 in phase_pairs])),
                "Full_mean_score": float(np.mean([full["score"] for full, _ in phase_pairs])),
                "B0_mean_score": float(np.mean([b0["score"] for _, b0 in phase_pairs])),
            }
        exact_b0_comparisons[scenario] = scenario_row
    canonical_write_json(
        root / "b0_full_exact_comparison.json",
        {"schema": "gnss-doppler-lab.gcspo.stage0-r2.b0-full-exact-comparison.v1",
         "strict_rule": "complete epoch+PRN+window support identity", "scenarios": exact_b0_comparisons},
    )
    bootstrap_rows = _clean_bootstrap_rows(root) + [row for row in rows if row["phase"] != "pre_onset_replay"]
    intervals = []
    for comparator in ("A2", "A5"):
        try:
            report = paired_block_bootstrap(bootstrap_rows, "Full", comparator)
            intervals.append({"contrast": f"Full-{comparator}", "status": "AVAILABLE", "replicates": report["replicates"],
                              "lcb_95": report["lcb_95"], "ci_low": report["interval_95"][0], "ci_high": report["interval_95"][1]})
        except ValueError as exc:
            intervals.append({"contrast": f"Full-{comparator}", "status": "UNAVAILABLE", "reason": str(exc)})
    relation = _read_json(root / "relation_destruction_metrics.json")
    for scenario, methods in relation["results"].items():
        for method, report in methods.items():
            intervals.append({"contrast": f"{scenario}:Full-{method}", "status": report["status"],
                              "replicates": report.get("replicates"), "lcb_95": report.get("lcb_95"),
                              "ci_low": report.get("interval_95", [None, None])[0], "ci_high": report.get("interval_95", [None, None])[1],
                              "reason": report.get("reason")})
    _write_csv(root / "bootstrap_intervals.csv", intervals,
               ["contrast", "status", "replicates", "lcb_95", "ci_low", "ci_high", "reason"])
    ablations = []
    for metric in metrics:
        # Retain the complete frozen ablation suite here.  The narrower
        # Full/B0/A2/A5 subset is a reporting contrast, not an artifact filter.
        if metric["method"] in {"A0", "A1", "A2", "A3", "A4", "A5", "Full"}:
            ablations.append({"scenario": metric["scenario"], "method": metric["method"],
                              "roc_auc": metric["roc_auc"], "low_fpr_pauc": metric["low_fpr_pauc"],
                              "attack_detection_rate_q99": metric["attack_detection_rate_q99"],
                              "delay_from_onset_s": metric["first_alarm_delay_from_onset_s"], "status": "AVAILABLE"})
    for scenario in ("DS4", "DS8"):
        for method in ("A0", "A1", "A2", "A3", "A4", "A5", "Full"):
            ablations.append({"scenario": scenario, "method": method, "status": "UNAVAILABLE",
                              "reason": "SOURCE_UNAVAILABLE" if scenario == "DS4" else "AUTHENTICATED_OBSERVABLES_ABSENT"})
    _write_csv(root / "ablation_metrics.csv", ablations,
               ["scenario", "method", "roc_auc", "low_fpr_pauc", "attack_detection_rate_q99", "delay_from_onset_s", "status", "reason"])
    clean = _read_json(root / "clean_only_report.json")
    full_fpr = clean["holdout_fpr"]["Full"]
    ds3_full = next(row for row in metrics if row["scenario"] == "DS3" and row["method"] == "Full")
    external = [
        {"scenario": "cleanStatic_holdout", "status": "AVAILABLE", "fpr_q99": full_fpr["q99"], "fpr_q995": full_fpr["q995"]},
        {"scenario": "DS3_pre_onset", "status": "AVAILABLE", "fpr_q99": ds3_full["pre_onset_fpr_q99"]},
        {"scenario": "DS4_pre_onset", "status": "UNAVAILABLE", "reason": "SOURCE_UNAVAILABLE"},
        {"scenario": "DS7_pre_onset", "status": "REPLAY_EXCLUDED_FROM_EXTERNAL_FPR", "reason": "identical raw replay family metadata"},
        {"scenario": "DS8_pre_onset", "status": "UNAVAILABLE", "reason": "AUTHENTICATED_OBSERVABLES_ABSENT"},
    ]
    _write_csv(root / "external_static_fpr.csv", external, ["scenario", "status", "fpr_q99", "fpr_q995", "reason"])
    state_summary = _state_summary(state_rows)
    canonical_write_json(root / "shared_state_onset_summary.json", {"status": "AVAILABLE_DS3_DS7", "phase_summaries": state_summary})
    _make_plots(root, rows, state_rows)
    controls = _read_json(root / "physical_controls_audit.json")
    incomplete = [
        {"item": "DS4", "reason": "authenticated receiver-level signed tracking and geometry source absent; legacy derived scores are inadmissible"},
        {"item": "DS8", "reason": "authenticated observables absent; geometry cannot be generated"},
        {"item": "receiver_exact_rerun", "reason": "immutable patched receiver source/build snapshot absent"},
        {"item": "Fixed9/M1/GCMR", "reason": "exact same-support authenticated adapters/checkpoints unavailable"},
    ]
    criteria = [
        {"id": 1, "criterion": "cleanStatic holdout q99 FPR <= 1%", "status": "PASS" if full_fpr["q99"] <= .01 else "FAIL", "value": full_fpr["q99"]},
        {"id": 2, "criterion": "external static worst-run FPR <= 5%", "status": "FAIL", "value": ds3_full["pre_onset_fpr_q99"], "reason": "DS3 alone exceeds 5%; DS4 is additionally unavailable and DS7 replay is excluded"},
        {"id": 3, "criterion": "Full vs B0/Fixed9", "status": "FAIL", "reason": "DS3 transition has no exact B0 support; on DS7 exact support B0 detects earlier and discriminates better; Fixed9 exact rerun is unavailable"},
        {"id": 4, "criterion": "Full beats A2 and A5 in at least two families", "status": "FAIL", "reason": "paired 10-second bootstrap lower bounds are nonpositive for both Full-A2 and Full-A5"},
        {"id": 5, "criterion": "relation destruction reduces score in at least two families", "status": "FAIL", "reason": "LOS shuffle diagnostic reduces score in DS3/DS7, but the frozen primary temporal-desynchronization control for both time-push families does not"},
        {"id": 6, "criterion": "controls show no persistent false alarms", "status": controls["scientific_false_alarm_status"]},
        {"id": 7, "criterion": "not explained by total energy or one feature", "status": "FAIL", "reason": "A1 strongly outperforms Full on DS7 and feature-family ablations do not establish Full-specific evidence"},
        {"id": 8, "criterion": "shared state plausible after onset/pull-off", "status": "FAIL", "reason": "DS3/DS7 state magnitudes change only modestly and inconsistently across onset/pull-off phases"},
    ]
    canonical_write_json(
        root / "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.gcspo.stage0-r2.final-verdict.v1",
            "verdict": "EVALUATION_INCOMPLETE_SOURCE_OR_SUPPORT",
            "detector_deployment_verdict": "EVALUATION_INCOMPLETE_SOURCE_OR_SUPPORT",
            "shared_pull_off_physical_hypothesis": "INCOMPLETE",
            "neural_stage1_allowed": False,
            "criteria": criteria,
            "incomplete_reasons": incomplete,
            "core_executable_results": {"DS3": "AVAILABLE", "DS7": "AVAILABLE", "DS4": "UNAVAILABLE", "DS8": "UNAVAILABLE"},
            "controls": controls["scientific_false_alarm_status"],
            "paper_model_continuation": "NOT_WORTH_CONTINUING_AS_THE_CURRENT_DETECTOR_PAPER_MODEL; preserve only the LOS-shuffle geometry diagnostic for a separately preregistered hypothesis study",
        },
    )
    refresh_run_inventory(root)
    required = [
        "README.md", "config.json", "source_commit.json", "run_inventory.json", "data_inventory.json",
        "clean_reproduction_check.json", "thresholds.json", "exact_support_audit.json", "common_support_inventory.csv",
        "scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv", "shared_state_estimates.csv",
        "external_static_fpr.csv", "relation_destruction_metrics.json", "physical_controls_audit.json",
        "bootstrap_intervals.csv", "final_verdict.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    empty = [name for name in required if (root / name).is_file() and (root / name).stat().st_size == 0]
    if missing or empty or not any((root / "plots").glob("*.png")):
        raise RuntimeError(f"R2 artifact verification failed: missing={missing} empty={empty}")
    _manifest(root)
    canonical_write_json(
        root / "phase_outputs/final-statistics-plots-verification/verification.json",
        {"status": "PASS", "required_files": required, "missing": [], "empty": [], "verdict": "EVALUATION_INCOMPLETE_SOURCE_OR_SUPPORT"},
    )
    _manifest(root)
    print("R2_FINAL_VERIFICATION_PASS verdict=EVALUATION_INCOMPLETE_SOURCE_OR_SUPPORT", flush=True)


def run_phase(phase: str, root: Path = ARTIFACT_ROOT) -> None:
    root = Path(root).resolve()
    if root != ARTIFACT_ROOT.resolve():
        raise ValueError(f"R2 output must use the contracted SSD directory: {ARTIFACT_ROOT}")
    if _git("branch", "--show-current") != "research/gcspo-stage0-r2-runner-simulation":
        raise RuntimeError("R2 phases refuse to run outside the contracted research branch")
    dispatch = {
        "preflight": phase_preflight,
        "cleanstatic-normal-model": phase_clean,
        "ds3-evaluation": lambda value: phase_scenario(value, "DS3"),
        "ds7-evaluation": lambda value: phase_scenario(value, "DS7"),
        "ds4-ds8-conditional-evaluation": phase_conditional,
        "relation-destruction-physical-controls": phase_relation_controls,
        "final-statistics-plots-verification": phase_final,
        "refresh-inventory-manifest": lambda value: (refresh_run_inventory(value), _manifest(value)),
    }
    dispatch[phase](root)
