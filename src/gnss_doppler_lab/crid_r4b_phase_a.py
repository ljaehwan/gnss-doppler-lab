"""R4b execution wrapper for the frozen CRID clean Phase A contract.

R4b deliberately reuses the committed R4 replay and analysis machinery.  The
only methodological repair is the already-committed R4a authorization: the R2
threshold literals remain authoritative and no exact-float threshold
recomputation is used as a gate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from . import crid_r4_phase_a as r4
from .crid import (
    CONFIG_ORDER,
    chronological_split,
    estimate_causal_delays,
    fit_normal_model,
    score_aligned as frozen_score_aligned,
)


BASE_SHA = "970f457daa1cd5677a1942198c92bf715a516050"
BRANCH = "research/crid-stage0-r4b-phase-a-physical-identifiability-execution"
R4_FINAL_SHA = "04ea478e5cb9f4d5da05563c1d883b2f7b20a28b"
R4_ART = "artifacts/crid_stage0_r4_phase_a_physical_identifiability"
R4A_ART = "artifacts/crid_stage0_r4a_threshold_decision_equivalence_repair"
R4B_ART = "artifacts/crid_stage0_r4b_phase_a_physical_identifiability_execution"
R4B_SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "crid-stage0-r4b-phase-a-physical-identifiability-execution"
)
AUTHORITATIVE_THRESHOLDS = {
    "OAK": -21.705587048010322,
    "TEX": -21.942672917134093,
}
EXECUTABLE_FILES = (
    "src/gnss_doppler_lab/crid_r4b_phase_a.py",
    "scripts/run_crid_r4b_phase_a.py",
    "scripts/verify_crid_r4b_phase_a.py",
)
R4A_FILES = (
    f"{R4A_ART}/artifact_manifest_sha256.json",
    f"{R4A_ART}/final_verdict.json",
    f"{R4A_ART}/threshold_numeric_comparison.json",
    f"{R4A_ART}/holdout_alarm_equivalence.json",
    f"{R4A_ART}/source_binding.json",
)
REPLAY_EXTRA_FIELDS = ("common_support_status", "common_valid_epochs")

BindingError = r4.BindingError
sha256_file = r4.sha256_file
require_file_binding = r4.require_file_binding
support_masks = r4.support_masks
evaluate_primary_gate = r4.evaluate_primary_gate
_aggregate_output_sha = r4._aggregate_output_sha
_load_checkpoint = r4._load_checkpoint

_R4_RUN_ONE = r4.run_one_replay
_R4_VALIDATE_COMPLETED = r4.validate_completed_replay
_REPRODUCTION: list[dict[str, object]] = []
_FIT_DOMAIN_INDEX = 0
_AUTHORIZED: dict[str, object] = {}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _configure_r4_runtime() -> None:
    r4.BASE_R3B_SHA = BASE_SHA
    r4.BRANCH = BRANCH
    r4.R4_SSD = R4B_SSD


def assert_branch(repo: Path) -> None:
    if git(repo, "branch", "--show-current") != BRANCH:
        raise BindingError("wrong R4b branch")
    if git(repo, "merge-base", "HEAD", BASE_SHA) != BASE_SHA:
        raise BindingError("R4b branch is not based on the required base commit")


def assert_pushed_freeze(repo: Path, freeze_sha: str) -> None:
    assert_branch(repo)
    head = git(repo, "rev-parse", "HEAD")
    remote = git(repo, "rev-parse", f"origin/{BRANCH}")
    if head != freeze_sha or remote != freeze_sha:
        raise BindingError("HEAD and origin R4b branch must equal the pushed freeze SHA")
    if git(repo, "status", "--porcelain=v1"):
        raise BindingError("R4b execution checkout is not clean")


def compact_manifest(artifact: Path) -> dict[str, object]:
    value = r4.compact_manifest(artifact)
    value["schema"] = "gnss-doppler-lab.crid-r4b-artifact-manifest.v1"
    return value


def seal_manifest(artifact: Path) -> None:
    r4.dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))


def _repo_binding(repo: Path, relative: str) -> dict[str, object]:
    return r4.repo_file_binding(repo, relative)


def validate_authorization_documents(
    final: Mapping[str, object],
    numeric: Mapping[str, object],
    holdout: Mapping[str, object],
) -> dict[str, object]:
    if not (
        final.get("status") == "PASS"
        and final.get("verdict") == "THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS"
        and final.get("next_state") == "READY_TO_REPEAT_CRID_PHASE_A"
        and final.get("phase_a_executed") is False
        and final.get("phase_b_executed") is False
        and final.get("attack_bytes_read") == 0
    ):
        raise BindingError("R4a final verdict does not authorize repeat Phase A")
    if numeric.get("status") != "PASS" or numeric.get("authoritative_threshold_policy") != "COMMITTED_R2_LITERALS_ONLY":
        raise BindingError("R4a numeric authorization is not PASS")
    if holdout.get("status") != "PASS" or holdout.get("comparison") != "score > threshold":
        raise BindingError("R4a holdout decision equivalence is not PASS")
    domains: dict[str, object] = {}
    for domain, threshold in AUTHORITATIVE_THRESHOLDS.items():
        nrow = numeric.get("domains", {}).get(domain, {})
        hrow = holdout.get("domains", {}).get(domain, {})
        if not (
            final.get("authoritative_thresholds", {}).get(domain) == threshold
            and nrow.get("authoritative_threshold_retained") == threshold
            and nrow.get("committed_q99") == threshold
            and nrow.get("status") == "PASS"
            and hrow.get("status") == "PASS"
            and hrow.get("alarm_vectors_byte_identical") is True
            and hrow.get("false_positive_count_and_fpr_equal") is True
            and hrow.get("all_scored_epochs_finite_four_config_min_four_prn") is True
            and float(hrow.get("committed_fpr")) <= 0.02
        ):
            raise BindingError(f"R4a authorization mismatch: {domain}")
        domains[domain] = {
            "authoritative_threshold": threshold,
            "holdout_fpr_q99": float(hrow["committed_fpr"]),
            "holdout_score_count": int(hrow["holdout_score_count"]),
            "alarm_sha256": hrow["committed_alarm_sha256"],
            "causal_delays_ms": hrow["causal_delays_ms"],
        }
    return {
        "status": "PASS",
        "verdict": final["verdict"],
        "next_state": final["next_state"],
        "comparison": "score > authoritative_threshold",
        "threshold_recomputation_executed": False,
        "domains": domains,
    }


def _load_authorization(repo: Path) -> dict[str, object]:
    return validate_authorization_documents(
        json.loads((repo / R4A_ART / "final_verdict.json").read_text()),
        json.loads((repo / R4A_ART / "threshold_numeric_comparison.json").read_text()),
        json.loads((repo / R4A_ART / "holdout_alarm_equivalence.json").read_text()),
    )


def primary_gate_definition() -> dict[str, object]:
    return r4.primary_gate_definition()


def _preregistration() -> dict[str, object]:
    return {
        "schema": "gnss-doppler-lab.crid-r4b-preregistration.v1",
        "status": "FROZEN_BEFORE_PHASE_A_RESULTS",
        "scope": "CLEAN_ONLY_CRID_PHASE_A_PHYSICAL_IDENTIFIABILITY_EXECUTION",
        "base_sha": BASE_SHA,
        "inputs": {
            "controls": 66,
            "domains": {"OAK": 33, "TEX": 33},
            "families": {
                "OAK": {"positive": 18, "negative": 15},
                "TEX": {"positive": 18, "negative": 15},
            },
            "configurations": list(CONFIG_ORDER),
            "total_replays": 264,
            "r4_inventory_byte_identity_required": True,
            "control_regeneration": False,
            "truth_modification": False,
            "overwrite": False,
        },
        "score_contract": {
            "implementation": "unchanged committed CRID feature/alignment/predictor/covariance/H0/H1/median-over-common-PRNs pooling",
            "minimum_configurations_per_epoch": 4,
            "minimum_common_prns_per_epoch": 4,
            "comparison": "score > authoritative_threshold",
            "authoritative_thresholds": AUTHORITATIVE_THRESHOLDS,
            "authorization": "R4a THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS",
            "exact_float_recomputation_gate_reused": False,
            "threshold_reestimated_or_replaced": False,
            "neural_model": False,
            "score_fusion": False,
            "cn0_or_power_score": False,
            "prn_identity_feature": False,
        },
        "window_contract": {
            "coordinate": "raw sample",
            "positive_primary": "active truth delay support only",
            "negative_primary": "full truth replacement support",
            "positive_full_45_second_primary_forbidden": True,
            "full_and_active_support_both_reported": True,
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


def prepare_freeze(repo: Path, artifact: Path) -> dict[str, object]:
    _configure_r4_runtime()
    assert_branch(repo)
    if artifact.exists() and any(artifact.iterdir()):
        raise BindingError("R4b compact artifact already exists; no overwrite")
    result = r4.prepare_freeze(repo, artifact)
    r4_inventory = repo / R4_ART / "control_input_inventory.csv"
    new_inventory = artifact / "control_input_inventory.csv"
    if sha256_file(r4_inventory) != sha256_file(new_inventory):
        raise BindingError("R4b inventory is not byte-identical to frozen R4 inventory")
    authorization = _load_authorization(repo)
    source = json.loads((artifact / "source_binding.json").read_text())
    source.update(
        {
            "schema": "gnss-doppler-lab.crid-r4b-source-binding.v1",
            "status": "PASS",
            "base_sha": BASE_SHA,
            "r4_final_sha": R4_FINAL_SHA,
            "r4_tree_sha": git(repo, "rev-parse", f"{R4_FINAL_SHA}:{R4_ART}"),
            "r4_tree_sha_at_base": git(repo, "rev-parse", f"HEAD:{R4_ART}"),
            "r4_bindings": {
                name: _repo_binding(repo, f"{R4_ART}/{name}")
                for name in (
                    "artifact_manifest_sha256.json",
                    "final_verdict.json",
                    "clean_threshold_binding.json",
                    "control_input_inventory.csv",
                )
            },
            "r4a_bindings": {name: _repo_binding(repo, name) for name in R4A_FILES},
            "r4a_authorization": authorization,
            "r4_replay_engine": _repo_binding(repo, "src/gnss_doppler_lab/crid_r4_phase_a.py"),
            "r4_inventory_byte_identical": True,
            "r4_inventory_sha256": sha256_file(r4_inventory),
            "executable_sha256": {name: sha256_file(repo / name) for name in EXECUTABLE_FILES},
            "attack_bytes_read": 0,
        }
    )
    r4.dump_json(artifact / "source_binding.json", source)
    r4.dump_json(artifact / "preregistration.json", _preregistration())
    old_execution = json.loads((artifact / "execution_freeze.json").read_text())
    r4.dump_json(
        artifact / "execution_freeze.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-execution-freeze.v1",
            "status": "FROZEN_PENDING_REMOTE_COMMIT",
            "base_sha": BASE_SHA,
            "branch": BRANCH,
            "output_root": str(R4B_SSD),
            "worker_count": 1,
            "sequential_order": "OAK then TEX; lexical case_id; C0,C1,C2,C3",
            "checkpoint_contract": old_execution["checkpoint_contract"],
            "freeze_authorization": "HEAD, origin branch, supplied freeze SHA identical; clean checkout",
            "commands": {
                "preflight": "python3 scripts/run_crid_r4b_phase_a.py preflight --freeze-sha <PUSHED_FREEZE_SHA>",
                "authorize_threshold": "python3 scripts/run_crid_r4b_phase_a.py authorize-threshold --freeze-sha <PUSHED_FREEZE_SHA>",
                "replay": "python3 scripts/run_crid_r4b_phase_a.py replay --freeze-sha <PUSHED_FREEZE_SHA>",
                "analyze": "python3 scripts/run_crid_r4b_phase_a.py analyze --freeze-sha <PUSHED_FREEZE_SHA>",
            },
            "environment": old_execution["environment"],
            "executable_sha256": {name: sha256_file(repo / name) for name in EXECUTABLE_FILES},
        },
    )
    r4.dump_json(
        artifact / "freeze_commit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-freeze-commit.v1",
            "status": "PENDING_PUSH",
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "freeze_sha": None,
        },
    )
    r4.dump_json(
        artifact / "clean_threshold_binding.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-clean-threshold-binding.v1",
            "status": "FROZEN_R4A_DECISION_EQUIVALENCE_AUTHORIZED",
            "comparison": "score > authoritative_threshold",
            "authoritative_source": "committed R2 literals authorized by R4a decision equivalence",
            "threshold_recomputation_executed": False,
            "r2_thresholds": _repo_binding(repo, f"{r4.R2_ART}/thresholds.json"),
            "r4a_final_verdict": _repo_binding(repo, f"{R4A_ART}/final_verdict.json"),
            "domains": authorization["domains"],
        },
    )
    r4.dump_json(
        artifact / "phase_a_gate.json",
        {"schema": "gnss-doppler-lab.crid-r4b-phase-a-gate.v1", "status": "FROZEN_NOT_EVALUATED", "definition": primary_gate_definition(), "results": None},
    )
    r4.dump_json(
        artifact / "support_audit.json",
        {"schema": "gnss-doppler-lab.crid-r4b-support-audit.v1", "status": "FROZEN_NOT_EVALUATED", "minimum_configurations": 4, "minimum_common_prns": 4, "cases": []},
    )
    r4.dump_json(
        artifact / "deterministic_reproduction.json",
        {"schema": "gnss-doppler-lab.crid-r4b-deterministic-reproduction.v1", "status": "FROZEN_NOT_EVALUATED", "cases": []},
    )
    r4.dump_json(
        artifact / "attack_access_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-attack-access-audit.v1",
            "status": "PASS",
            "explicit_clean_allowlist_only": True,
            "attack_stats": 0,
            "attack_hashes": 0,
            "attack_opens": 0,
            "attack_mmaps": 0,
            "attack_bytes_read": 0,
            "phase_b_executed": False,
        },
    )
    r4.dump_json(
        artifact / "final_verdict.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-final-verdict.v1",
            "status": "FROZEN_NOT_EXECUTED",
            "verdict": "NOT_EVALUATED_PRE_RESULT_FREEZE",
            "next_state": "RUN_PHASE_A_FROM_PUSHED_FREEZE_SHA",
            "phase_a_executed": False,
            "phase_b_executed": False,
            "attack_bytes_read": 0,
        },
    )
    (artifact / "README.md").write_text(
        "# CRID Stage-0 R4b Phase A physical-identifiability execution\n\n"
        f"Pre-result executable freeze from base `{BASE_SHA}`. R4 and R4a are immutable inputs. "
        "The exact 66 frozen R4 controls will be replayed sequentially through C0-C3 only after the freeze commit is pushed.\n\n"
        "The committed R2 threshold literals are authoritative under the R4a decision-equivalence authorization. "
        "The historical R4 exact-float check is not reused as an execution gate. Phase B and attack data are out of scope.\n"
    )
    seal_manifest(artifact)
    return {"status": "READY_TO_VERIFY_FREEZE", "inventory_rows": 66, "r4_inventory_byte_identical": True, **result}


def preflight_inputs(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    _configure_r4_runtime()
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != R4B_SSD:
        raise BindingError("R4b SSD output root mismatch")
    path = ssd / "input_preflight.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("status") == "PASS" and existing.get("freeze_sha") == freeze_sha and existing.get("control_rows") == 66:
            return existing
        raise BindingError("stale or mismatched R4b input preflight; no overwrite")
    result = r4.preflight_inputs(repo, artifact, ssd, freeze_sha)
    result.update({"schema": "gnss-doppler-lab.crid-r4b-input-preflight.v1", "r4_inventory_byte_identical": True})
    r4.atomic_json(path, result)
    return result


def _write_once_json(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise BindingError(f"existing result mismatch; no overwrite: {path}")
        return
    r4.atomic_json(path, value)


def authorize_threshold(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    _configure_r4_runtime()
    assert_pushed_freeze(repo, freeze_sha)
    if ssd != R4B_SSD:
        raise BindingError("R4b SSD output root mismatch")
    source = json.loads((artifact / "source_binding.json").read_text())
    for binding in source["r4a_bindings"].values():
        path = repo / str(binding["path"])
        require_file_binding(path, int(binding["size_bytes"]), str(binding["sha256"]))
    authorization = _load_authorization(repo)
    result = {
        "schema": "gnss-doppler-lab.crid-r4b-threshold-authorization.v1",
        "status": "PASS",
        "freeze_sha": freeze_sha,
        "method": "R4A_THRESHOLD_DECISION_EQUIVALENCE_REPAIR",
        "comparison": "score > authoritative_threshold",
        "threshold_recomputation_executed": False,
        "historical_r4_exact_float_gate_reused": False,
        "authorization": authorization,
        "attack_bytes_read": 0,
    }
    _write_once_json(ssd / "threshold_authorization.json", result)
    # The reused R4 replay supervisor checks this fixed filename.  Its content
    # is explicitly an authorization record, never a recomputation.
    _write_once_json(ssd / "threshold_recomputation.json", result)
    return result


def _run_one_r4b(row: Mapping[str, str], config: str, out: Path, receiver_sha256: str) -> dict[str, object]:
    manifest = _R4_RUN_ONE(row, config, out, receiver_sha256)
    manifest.update(
        {
            "schema": "gnss-doppler-lab.crid-r4b-replay.v1",
            "execution_stage": "R4b",
            "common_support": {"status": "PENDING_FOUR_CONFIG_ANALYSIS", "minimum_configurations": 4, "minimum_common_prns": 4},
        }
    )
    r4.dump_json(out / "manifest.json", manifest)
    return manifest


def run_all_replays(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    _configure_r4_runtime()
    assert_pushed_freeze(repo, freeze_sha)
    authorization = json.loads((ssd / "threshold_authorization.json").read_text())
    if not (
        authorization.get("status") == "PASS"
        and authorization.get("freeze_sha") == freeze_sha
        and authorization.get("threshold_recomputation_executed") is False
    ):
        raise BindingError("R4a threshold authorization missing or stale")
    previous = r4.run_one_replay
    r4.run_one_replay = _run_one_r4b
    try:
        result = r4.run_all_replays(repo, artifact, ssd, freeze_sha)
    finally:
        r4.run_one_replay = previous
    result.update(
        {
            "schema": "gnss-doppler-lab.crid-r4b-replay-run.v1",
            "threshold_authorization": "R4A_THRESHOLD_DECISION_EQUIVALENCE_REPAIR",
            "threshold_recomputation_executed": False,
        }
    )
    r4.atomic_json(ssd / "replay_run_summary.json", result)
    return result


def validate_completed_replay(manifest_path: Path, row: Mapping[str, str], config: str, receiver_sha256: str) -> dict[str, object]:
    manifest = _R4_VALIDATE_COMPLETED(manifest_path, row, config, receiver_sha256)
    common = manifest.get("common_support", {})
    if manifest.get("schema") != "gnss-doppler-lab.crid-r4b-replay.v1" or common.get("minimum_common_prns") != 4:
        raise BindingError("R4b replay manifest binding mismatch")
    return manifest


def _score_hash(rows: Iterable[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    selected = tuple(rows)
    digest.update(np.asarray([int(row["sample"]) for row in selected], dtype="<i8").tobytes())
    digest.update(np.asarray([int(row["prn_count"]) for row in selected], dtype="<i8").tobytes())
    digest.update(np.asarray([int(row["config_count"]) for row in selected], dtype="<i8").tobytes())
    for name in ("score", "h0_loglike", "h1_loglike", "penalty", "configuration_disagreement"):
        digest.update(np.asarray([float(row[name]) for row in selected], dtype="<f8").tobytes())
    return digest.hexdigest()


def _deterministic_score_aligned(tables, model, delays, minimum_prns: int = 4):
    first = frozen_score_aligned(tables, model, delays, minimum_prns=minimum_prns)
    second = frozen_score_aligned(tables, model, delays, minimum_prns=minimum_prns)
    first_sha = _score_hash(first)
    second_sha = _score_hash(second)
    if first_sha != second_sha:
        raise BindingError("deterministic score reproduction mismatch")
    _REPRODUCTION.append(
        {"status": "PASS", "score_sha256_first": first_sha, "score_sha256_second": second_sha, "epoch_count": len(first)}
    )
    return first


def _authorized_fit_domain(clean_tables):
    global _FIT_DOMAIN_INDEX
    domain = tuple(r4.DOMAIN)[_FIT_DOMAIN_INDEX]
    _FIT_DOMAIN_INDEX += 1
    delays = estimate_causal_delays(clean_tables)
    expected_delays = _AUTHORIZED["domains"][domain]["causal_delays_ms"]
    if delays != expected_delays:
        raise BindingError(f"R4a causal alignment mismatch: {domain}")
    samples = np.concatenate([value.sample for value in clean_tables.values()])
    split = chronological_split(samples)
    model = fit_normal_model(clean_tables, split["train"], split["calibration"])
    threshold = AUTHORITATIVE_THRESHOLDS[domain]
    clean = {"holdout_fpr_q99": _AUTHORIZED["domains"][domain]["holdout_fpr_q99"]}
    return model, delays, split, {"q99": threshold}, [], clean


def authoritative_alarms(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(scores, dtype=float) > float(threshold)


def _write_positive_plot(artifact: Path, metrics: list[Mapping[str, object]]) -> None:
    import matplotlib.pyplot as plt

    positive = [row for row in metrics if row["family"] == "positive"]
    labels = [str(row["case_id"]) for row in positive]
    values = [float(row["primary_alarm_ratio_q99"]) for row in positive]
    colors = ["#2a9d8f" if row["case_gate_status"] == "PASS" else "#e76f51" for row in positive]
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.bar(range(len(values)), values, color=colors)
    axis.axhline(0.70, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0, 1)
    axis.set_ylabel("active-support authoritative-q99 alarm ratio")
    axis.set_xticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_title("CRID R4b positive response surface")
    figure.tight_layout()
    path = artifact / "plots/positive_response_surface.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def analyze_phase_a(repo: Path, artifact: Path, ssd: Path, freeze_sha: str) -> dict[str, object]:
    global _FIT_DOMAIN_INDEX, _AUTHORIZED
    _configure_r4_runtime()
    assert_pushed_freeze(repo, freeze_sha)
    authorization_doc = json.loads((ssd / "threshold_authorization.json").read_text())
    if authorization_doc.get("status") != "PASS" or authorization_doc.get("freeze_sha") != freeze_sha:
        raise BindingError("threshold authorization missing or stale")
    _AUTHORIZED = authorization_doc["authorization"]
    _FIT_DOMAIN_INDEX = 0
    _REPRODUCTION.clear()
    previous_fit = r4.fit_domain
    previous_score = r4.score_aligned
    previous_validate = r4.validate_completed_replay
    r4.fit_domain = _authorized_fit_domain
    r4.score_aligned = _deterministic_score_aligned
    r4.validate_completed_replay = validate_completed_replay
    try:
        result = r4.analyze_phase_a(repo, artifact, ssd, freeze_sha)
    finally:
        r4.fit_domain = previous_fit
        r4.score_aligned = previous_score
        r4.validate_completed_replay = previous_validate
    inventory = r4.read_csv(artifact / "control_input_inventory.csv")
    metrics = r4.read_csv(artifact / "physical_control_metrics.csv")
    replays = r4.read_csv(artifact / "replay_completion.csv")
    if len(_REPRODUCTION) != 66 or len(inventory) != 66 or len(metrics) != 66 or len(replays) != 264:
        raise BindingError("final R4b result counts are incomplete")
    reproduction_cases = []
    common_by_case: dict[str, int] = {}
    for row, reproduction in zip(inventory, _REPRODUCTION, strict=True):
        case = {"domain": row["domain"], "case_id": row["case_id"], **reproduction}
        reproduction_cases.append(case)
        common_by_case[row["case_id"]] = int(reproduction["epoch_count"])
    for replay in replays:
        count = common_by_case[replay["case_id"]]
        replay["common_support_status"] = "PASS" if count > 0 else "INCONCLUSIVE_SUPPORT"
        replay["common_valid_epochs"] = count
    r4.write_csv(artifact / "replay_completion.csv", (*r4.REPLAY_FIELDS, *REPLAY_EXTRA_FIELDS), replays)
    support_cases = []
    for metric in metrics:
        support_cases.append(
            {
                "domain": metric["domain"],
                "case_id": metric["case_id"],
                "family": metric["family"],
                "common_valid_epochs": common_by_case[metric["case_id"]],
                "valid_full_epochs": int(metric["valid_full_epochs"]),
                "valid_active_epochs": int(metric["valid_active_epochs"]),
                "technical_status": metric["technical_status"],
            }
        )
    support_pass = all(
        row["common_valid_epochs"] > 0
        and row["valid_full_epochs"] > 0
        and (row["family"] != "positive" or row["valid_active_epochs"] > 0)
        and row["technical_status"] == "PASS"
        for row in support_cases
    )
    r4.dump_json(
        artifact / "support_audit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-support-audit.v1",
            "status": "PASS" if support_pass else "INCONCLUSIVE_SUPPORT",
            "minimum_configurations": 4,
            "minimum_common_prns": 4,
            "case_count": len(support_cases),
            "zero_support_cases": sum(row["common_valid_epochs"] == 0 for row in support_cases),
            "cases": support_cases,
        },
    )
    reproduction_pass = all(row["score_sha256_first"] == row["score_sha256_second"] for row in reproduction_cases)
    r4.dump_json(
        artifact / "deterministic_reproduction.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-deterministic-reproduction.v1",
            "status": "PASS" if reproduction_pass else "FAIL",
            "case_count": len(reproduction_cases),
            "comparison": "two exact score_aligned evaluations over the same loaded four-config tables",
            "cases": reproduction_cases,
        },
    )
    r4.dump_json(
        artifact / "clean_threshold_binding.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-clean-threshold-binding.v1",
            "status": "PASS",
            "comparison": "score > authoritative_threshold",
            "authoritative_source": "committed R2 literals authorized by R4a decision equivalence",
            "threshold_recomputation_executed": False,
            "historical_r4_exact_float_gate_reused": False,
            "domains": _AUTHORIZED["domains"],
        },
    )
    gate = json.loads((artifact / "phase_a_gate.json").read_text())
    gate.update({"schema": "gnss-doppler-lab.crid-r4b-phase-a-gate.v1", "threshold_authorization": "R4a decision equivalence"})
    r4.dump_json(artifact / "phase_a_gate.json", gate)
    final = {
        "schema": "gnss-doppler-lab.crid-r4b-final-verdict.v1",
        "status": result["status"],
        "verdict": result["verdict"],
        "next_state": result["next_state"],
        "base_sha": BASE_SHA,
        "freeze_sha": freeze_sha,
        "phase_a_executed": True,
        "phase_a_replays": 264,
        "control_metrics": 66,
        "positive_surface_cases": 36,
        "phase_b_executed": False,
        "actual_spoofing_evaluation_executed": False,
        "attack_bytes_read": 0,
        "threshold_recomputation_executed": False,
        "post_result_code_threshold_window_score_gate_changes": False,
        "controls_excluded_after_result": 0,
    }
    if not support_pass or not reproduction_pass:
        final.update(
            {
                "status": "INCONCLUSIVE",
                "verdict": "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE",
                "next_state": "NOT_AUTHORIZED",
            }
        )
    r4.dump_json(artifact / "final_verdict.json", final)
    r4.dump_json(
        artifact / "freeze_commit.json",
        {
            "schema": "gnss-doppler-lab.crid-r4b-freeze-commit.v1",
            "status": "PASS",
            "branch": BRANCH,
            "base_sha": BASE_SHA,
            "freeze_sha": freeze_sha,
            "local_sha_before_execution": freeze_sha,
            "remote_sha_before_execution": freeze_sha,
            "ahead": 0,
            "behind": 0,
            "clean_checkout_before_execution": True,
        },
    )
    execution = json.loads((artifact / "execution_freeze.json").read_text())
    execution.update({"status": "EXECUTED_FROM_PUSHED_FREEZE", "freeze_sha": freeze_sha})
    r4.dump_json(artifact / "execution_freeze.json", execution)
    audit = json.loads((artifact / "attack_access_audit.json").read_text())
    audit.update({"status": "PASS", "phase_a_replays": 264, "phase_a_scores": 66, "phase_b_executed": False, "attack_bytes_read": 0})
    r4.dump_json(artifact / "attack_access_audit.json", audit)
    (artifact / "README.md").write_text(
        "# CRID Stage-0 R4b Phase A physical-identifiability execution\n\n"
        f"Final verdict: `{final['verdict']}`\n\nNext state: `{final['next_state']}`\n\n"
        "All 66 frozen R4 clean controls were replayed sequentially through C0-C3 from the pushed executable freeze. "
        "The committed R2 literals were used under the R4a decision-equivalence authorization; no threshold was re-estimated or replaced.\n\n"
        "Positive primary ratios use active truth-delay support and negative primary ratios use full replacement support. "
        "Full and active metrics are both retained. Phase B was not executed and attack access remained zero.\n"
    )
    _write_positive_plot(artifact, metrics)
    seal_manifest(artifact)
    final_result = {**result, "status": final["status"], "verdict": final["verdict"], "next_state": final["next_state"]}
    r4.atomic_json(ssd / "analysis_summary.json", final_result)
    return final_result
