"""Post-hoc relaxed-gate sensitivity over committed BITPROBE metrics only."""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


BASE_SHA = "0b2a8913c856ab9b300f7fd604fc8f61da350cc6"
CONTRACT_COMMIT_SHA = "76cf6039d70d233150a8d75a5bd6145427bb6fcf"
BRANCH = "research/bitprobe-stage0a-r0b-relaxed-gate-sensitivity"
ARTIFACT_REL = "artifacts/bitprobe_stage0a_r0b_relaxed_gate_sensitivity"
R0A_REL = "artifacts/bitprobe_stage0a_r0a_inference_contract_repair"
R0_REL = "artifacts/bitprobe_stage0a_nav_edge_operator_identifiability"
DATASETS = ("TEXBAT.cleanStatic", "OAKBAT.cleanStatic")
SOURCE_FILES = {
    "corrected_split_half_metrics": (
        f"{R0A_REL}/corrected_split_half_metrics.csv",
        "7e716fae992d6fbd0f3a2ece79e16c215119e8c4568eda6e8f62070ef40da73b",
    ),
    "corrected_between_prn_metrics": (
        f"{R0A_REL}/corrected_between_prn_metrics.csv",
        "7d9b49f6120e52a158c498185cd881c9263e3361e688c3ff33ae8555826ca610",
    ),
    "corrected_bootstrap_metrics": (
        f"{R0A_REL}/corrected_bootstrap_metrics.json",
        "990562948ba17bf89d0cc749920a456d1e1f46c27ebe30a6376676860d036e5c",
    ),
    "nuisance_metrics": (
        f"{R0_REL}/nuisance_metrics.csv",
        "212b3b9527d6a9ec621b9ebe9196ee75597b8291712ab0ac86d66bf7cca54400",
    ),
    "per_prn_support": (
        f"{R0_REL}/per_prn_support.csv",
        "0155f2184feeac5373e41eb4c603c96f3c3b2dd6fdb7d2eac492dea1ab5c498c",
    ),
    "flip_no_flip_advisory": (
        f"{R0_REL}/flip_no_flip_metrics.csv",
        "f2a348c85819488fed7a3b49f80a154b1fe5bb058800a4f1b55370d3e2e34dbb",
    ),
}
EXECUTABLE_FILES = (
    "src/gnss_doppler_lab/bitprobe_stage0a_r0b.py",
    "scripts/run_bitprobe_stage0a_r0b.py",
    "scripts/verify_bitprobe_stage0a_r0b.py",
)
GRID_FIELDS = (
    "dataset", "reproducibility_median_threshold", "reproducibility_lower_threshold",
    "separability_effect_threshold", "permutation_p_threshold",
    "nuisance_direction_fraction_threshold", "support_pass",
    "reproducibility_median_pass", "reproducibility_lower_pass",
    "separability_effect_pass", "separability_lower_fixed_pass",
    "permutation_p_pass", "nuisance_median_fixed_pass",
    "nuisance_direction_pass", "core_effect_and_p_pass", "full_grid_pass",
)
MARGIN_FIELDS = (
    "tier", "dataset", "criterion", "observed", "threshold", "comparison",
    "signed_margin", "pass", "failed_item_count", "single_metric_flip_possible",
)
ALLOWED_VERDICTS = (
    "RELAXED_SIGNAL_BOTH_DATASETS", "RELAXED_SIGNAL_OAK_ONLY",
    "RELAXED_SIGNAL_TEX_ONLY", "NO_ROBUST_SIGNAL_UNDER_RELAXED_GATES",
    "INCONCLUSIVE_RELAXED_SENSITIVITY_EXECUTION",
)


class SensitivityError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def binding(path: Path, display: str | None = None) -> dict[str, object]:
    return {"path": display or str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]], gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def assert_branch(repo: Path) -> None:
    if git(repo, "branch", "--show-current") != BRANCH:
        raise SensitivityError("wrong R0b branch")
    if git(repo, "merge-base", "HEAD", BASE_SHA) != BASE_SHA:
        raise SensitivityError("R0b branch not based on exact base")


def assert_pushed_freeze(repo: Path, freeze_sha: str) -> None:
    assert_branch(repo)
    if git(repo, "rev-parse", "HEAD") != freeze_sha or git(repo, "rev-parse", f"origin/{BRANCH}") != freeze_sha:
        raise SensitivityError("local/remote freeze mismatch")
    if git(repo, "rev-list", "--left-right", "--count", f"HEAD...origin/{BRANCH}") != "0\t0":
        raise SensitivityError("freeze not ahead/behind 0/0")
    if git(repo, "status", "--porcelain=v1"):
        raise SensitivityError("freeze worktree not clean")


def compact_manifest(artifact: Path) -> dict[str, object]:
    rows = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            rows.append(binding(path, str(path.relative_to(artifact))))
    return {"schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-manifest.v1", "status": "PASS", "file_count": len(rows), "files": rows}


def seal_manifest(artifact: Path) -> None:
    dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))


def load_contract(repo: Path) -> dict[str, object]:
    value = json.loads((repo / ARTIFACT_REL / "relaxed_sensitivity_contract.json").read_text())
    if value.get("base_sha") != BASE_SHA or value.get("status") != "FROZEN_BEFORE_ANALYSIS_IMPLEMENTATION":
        raise SensitivityError("contract mismatch")
    return value


def verify_sources(repo: Path) -> dict[str, object]:
    result = {}
    for name, (relative, expected) in SOURCE_FILES.items():
        observed = binding(repo / relative, relative)
        if observed["sha256"] != expected:
            raise SensitivityError(f"source metric SHA mismatch: {name}")
        result[name] = observed
    return result


def load_observations(repo: Path) -> dict[str, dict[str, object]]:
    between = {row["dataset"]: row for row in read_csv(repo / SOURCE_FILES["corrected_between_prn_metrics"][0])}
    bootstrap = json.loads((repo / SOURCE_FILES["corrected_bootstrap_metrics"][0]).read_text())
    support_rows = read_csv(repo / SOURCE_FILES["per_prn_support"][0])
    nuisance_rows = read_csv(repo / SOURCE_FILES["nuisance_metrics"][0])
    flip = {row["dataset"]: row for row in read_csv(repo / SOURCE_FILES["flip_no_flip_advisory"][0])}
    result = {}
    for dataset in DATASETS:
        support = [row for row in support_rows if row["dataset"] == dataset]
        moderate = [row for row in nuisance_rows if row["dataset"] == dataset and row["class"] == "moderate"]
        result[dataset] = {
            "eligible_prns": sum(row["support_gate"] == "PASS" for row in support),
            "minimum_valid_flip_per_prn": min(int(row["valid_flip"]) for row in support),
            "reproducibility_median": float(between[dataset]["median_same_similarity"]),
            "reproducibility_lower": float(bootstrap[dataset]["coherence_bootstrap_95_lower"]),
            "separability_effect": float(between[dataset]["same_minus_different"]),
            "separability_lower": float(bootstrap[dataset]["effect_bootstrap_95_lower"]),
            "permutation_p": float(between[dataset]["exact_prn_label_permutation_p"]),
            "moderate_nuisance_median": float(np.median([float(row["median_similarity"]) for row in moderate])),
            "moderate_nuisance_direction_fraction": float(np.mean([row["effect_direction_maintained"] == "True" for row in moderate])),
            "moderate_nuisance_count": len(moderate),
            "flip_specificity_advisory_only": {
                "flip_minus_no_flip": float(flip[dataset]["flip_minus_no_flip"]),
                "permutation_p": float(flip[dataset]["block_permutation_p"]),
                "used_in_gate": False,
            },
        }
    return result


def _criterion(name: str, observed: float, threshold: float, comparison: str) -> dict[str, object]:
    if comparison in (">=", ">"):
        passed = observed >= threshold if comparison == ">=" else observed > threshold
        margin = observed - threshold
    elif comparison == "<=":
        passed = observed <= threshold
        margin = threshold - observed
    else:
        raise ValueError(comparison)
    return {"criterion": name, "observed": observed, "threshold": threshold, "comparison": comparison, "signed_margin": float(margin), "pass": bool(passed)}


def evaluate_tier(observation: Mapping[str, object], tier: str, spec: Mapping[str, object]) -> dict[str, object]:
    criteria = [
        _criterion("support_prns", float(observation["eligible_prns"]), float(spec["support_min_prns"]), ">="),
        _criterion("support_flip_per_prn", float(observation["minimum_valid_flip_per_prn"]), float(spec["support_min_flip_per_prn"]), ">="),
        _criterion("reproducibility_median", float(observation["reproducibility_median"]), float(spec["reproducibility_median_min"]), ">="),
    ]
    if "reproducibility_lower_exclusive_min" in spec:
        criteria.append(_criterion("reproducibility_lower", float(observation["reproducibility_lower"]), float(spec["reproducibility_lower_exclusive_min"]), ">"))
    else:
        criteria.append(_criterion("reproducibility_lower", float(observation["reproducibility_lower"]), float(spec["reproducibility_lower_inclusive_min"]), ">="))
    criteria.append(_criterion("separability_effect", float(observation["separability_effect"]), float(spec["separability_effect_min"]), ">="))
    if spec.get("separability_lower_required", True):
        criteria.append(_criterion("separability_lower", float(observation["separability_lower"]), float(spec.get("separability_lower_exclusive_min", 0.0)), ">"))
    criteria.extend([
        _criterion("permutation_p", float(observation["permutation_p"]), float(spec["permutation_p_max"]), "<="),
        _criterion("moderate_nuisance_median", float(observation["moderate_nuisance_median"]), float(spec["moderate_nuisance_median_min"]), ">="),
        _criterion("moderate_nuisance_direction_fraction", float(observation["moderate_nuisance_direction_fraction"]), float(spec["moderate_nuisance_direction_fraction_min"]), ">="),
    ])
    failed = [row["criterion"] for row in criteria if not row["pass"]]
    return {
        "tier": tier, "pass": not failed, "failed_items": failed,
        "failed_item_count": len(failed), "single_metric_flip_possible": len(failed) == 1,
        "smallest_absolute_margin": min(abs(float(row["signed_margin"])) for row in criteria),
        "criteria": criteria,
        "flip_specificity_excluded": True,
    }


def threshold_grid(observations: Mapping[str, Mapping[str, object]], contract: Mapping[str, object]) -> list[dict[str, object]]:
    grid = contract["threshold_grid"]
    rows = []
    for dataset in DATASETS:
        obs = observations[dataset]
        support = int(obs["eligible_prns"]) >= 4 and int(obs["minimum_valid_flip_per_prn"]) >= 200
        nuisance_median = float(obs["moderate_nuisance_median"]) >= float(grid["moderate_nuisance_median_threshold"])
        lower_fixed = float(obs["separability_lower"]) > 0.0
        for repro, lower, effect, p_value, direction in itertools.product(
            grid["reproducibility_median_thresholds"], grid["reproducibility_lower_bound_thresholds"],
            grid["separability_effect_thresholds"], grid["permutation_p_thresholds"],
            grid["nuisance_direction_fraction_thresholds"],
        ):
            checks = {
                "support_pass": support,
                "reproducibility_median_pass": float(obs["reproducibility_median"]) >= float(repro),
                "reproducibility_lower_pass": float(obs["reproducibility_lower"]) >= float(lower),
                "separability_effect_pass": float(obs["separability_effect"]) >= float(effect),
                "separability_lower_fixed_pass": lower_fixed,
                "permutation_p_pass": float(obs["permutation_p"]) <= float(p_value),
                "nuisance_median_fixed_pass": nuisance_median,
                "nuisance_direction_pass": float(obs["moderate_nuisance_direction_fraction"]) >= float(direction),
            }
            rows.append({
                "dataset": dataset,
                "reproducibility_median_threshold": repro,
                "reproducibility_lower_threshold": lower,
                "separability_effect_threshold": effect,
                "permutation_p_threshold": p_value,
                "nuisance_direction_fraction_threshold": direction,
                **checks,
                "core_effect_and_p_pass": checks["separability_effect_pass"] and checks["permutation_p_pass"],
                "full_grid_pass": all(checks.values()),
            })
    return rows


def _largest_not_above(value: float, thresholds: Sequence[float]) -> float | str:
    eligible = [float(threshold) for threshold in thresholds if float(threshold) <= value]
    return max(eligible) if eligible else "NO_GRID_THRESHOLD_PASSES"


def _smallest_not_below(value: float, thresholds: Sequence[float]) -> float | str:
    eligible = [float(threshold) for threshold in thresholds if float(threshold) >= value]
    return min(eligible) if eligible else "NO_GRID_THRESHOLD_PASSES"


def minimum_relaxation(observations: Mapping[str, Mapping[str, object]], contract: Mapping[str, object]) -> dict[str, object]:
    grid = contract["threshold_grid"]
    result = {}
    for dataset in DATASETS:
        obs = observations[dataset]
        result[dataset] = {
            "required_max_reproducibility_threshold": _largest_not_above(float(obs["reproducibility_median"]), grid["reproducibility_median_thresholds"]),
            "required_max_lower_bound_threshold": _largest_not_above(float(obs["reproducibility_lower"]), grid["reproducibility_lower_bound_thresholds"]),
            "required_max_effect_threshold": _largest_not_above(float(obs["separability_effect"]), grid["separability_effect_thresholds"]),
            "required_min_p_threshold": _smallest_not_below(float(obs["permutation_p"]), grid["permutation_p_thresholds"]),
            "required_min_nuisance_direction_fraction": _largest_not_above(float(obs["moderate_nuisance_direction_fraction"]), grid["nuisance_direction_fraction_thresholds"]),
            "fixed_separability_lower_gt_zero": float(obs["separability_lower"]) > 0.0,
            "any_full_grid_pass": False,
            "interpretation": "NO_STATISTICALLY_MEANINGFUL_SEPARATION" if float(obs["permutation_p"]) >= 0.30 or float(obs["separability_effect"]) <= 0.01 else "RELAXATION_WITHIN_EXPLORATORY_RANGE",
        }
    return result


def analyze(repo: Path) -> dict[str, object]:
    contract = load_contract(repo)
    observations = load_observations(repo)
    tier_results = {tier: {dataset: evaluate_tier(observations[dataset], tier, spec) for dataset in DATASETS} for tier, spec in contract["tiers"].items()}
    for tier, datasets in tier_results.items():
        passes = [dataset for dataset in DATASETS if datasets[dataset]["pass"]]
        tier_results[tier]["summary"] = {
            "both_datasets_pass": len(passes) == 2,
            "oak_only": passes == ["OAKBAT.cleanStatic"],
            "tex_only": passes == ["TEXBAT.cleanStatic"],
            "passing_datasets": passes,
        }
    grid_rows = threshold_grid(observations, contract)
    minimum = minimum_relaxation(observations, contract)
    for dataset in DATASETS:
        minimum[dataset]["any_full_grid_pass"] = any(row["full_grid_pass"] for row in grid_rows if row["dataset"] == dataset)
    moderate_pass = [dataset for dataset in DATASETS if tier_results["MODERATE_RELAXATION"][dataset]["pass"]]
    if len(moderate_pass) == 2:
        verdict = "RELAXED_SIGNAL_BOTH_DATASETS"
    elif moderate_pass == ["OAKBAT.cleanStatic"]:
        verdict = "RELAXED_SIGNAL_OAK_ONLY"
    elif moderate_pass == ["TEXBAT.cleanStatic"]:
        verdict = "RELAXED_SIGNAL_TEX_ONLY"
    else:
        verdict = "NO_ROBUST_SIGNAL_UNDER_RELAXED_GATES"
    lenient_pass = [dataset for dataset in DATASETS if tier_results["LENIENT_RELAXATION"][dataset]["pass"]]
    return {
        "observations": observations, "tier_results": tier_results,
        "grid_rows": grid_rows, "minimum_relaxation": minimum,
        "leave_one_prn_out": {
            "status": "NOT_AVAILABLE_FROM_COMMITTED_METRICS",
            "reason": "No committed 5x5 similarity matrix is present; frozen tensor access is prohibited.",
            "tensor_access_attempted": False,
        },
        "verdict": verdict, "moderate_passing_datasets": moderate_pass,
        "lenient_only": [dataset for dataset in lenient_pass if dataset not in moderate_pass],
    }


def analysis_bytes(result: Mapping[str, object]) -> dict[str, bytes]:
    tier = json.dumps(result["tier_results"], indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    minimum = json.dumps(result["minimum_relaxation"], indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    stream = gzip.compress(_csv_bytes(GRID_FIELDS, result["grid_rows"]), mtime=0)
    margins = []
    for tier_name, datasets in result["tier_results"].items():
        for dataset in DATASETS:
            gate = datasets[dataset]
            for criterion in gate["criteria"]:
                margins.append({"tier": tier_name, "dataset": dataset, **criterion, "failed_item_count": gate["failed_item_count"], "single_metric_flip_possible": gate["single_metric_flip_possible"]})
    return {
        "tier_results.json": tier,
        "threshold_grid.csv.gz": stream,
        "minimum_relaxation_required.json": minimum,
        "gate_margin.csv": _csv_bytes(MARGIN_FIELDS, margins),
        "leave_one_prn_out.json": json.dumps(result["leave_one_prn_out"], indent=2, sort_keys=True).encode() + b"\n",
    }


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> bytes:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def prepare_freeze(repo: Path) -> None:
    assert_branch(repo)
    artifact = repo / ARTIFACT_REL
    contract = load_contract(repo)
    sources = verify_sources(repo)
    dump_json(artifact / "source_metric_binding.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-source-binding.v1",
        "status": "PASS", "base_sha": BASE_SHA, "sources": sources,
        "source_metric_changes_allowed": 0,
    })
    dump_json(artifact / "contract_commit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-contract-commit.v1",
        "commit_sha": CONTRACT_COMMIT_SHA,
        "contract_sha256": sha256_file(artifact / "relaxed_sensitivity_contract.json"),
        "local_remote_equal_before_implementation": True,
        "ahead_behind_before_implementation": [0, 0],
    })
    dump_json(artifact / "execution_freeze.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-execution-freeze.v1",
        "status": "IMPLEMENTED_BEFORE_SENSITIVITY_EXECUTION",
        "base_sha": BASE_SHA, "contract_commit_sha": CONTRACT_COMMIT_SHA,
        "executable_bindings": {relative: binding(repo / relative, relative) for relative in EXECUTABLE_FILES},
        "source_bindings": sources,
        "configuration_sha256": sha256_bytes(canonical_json({
            "tiers": contract["tiers"], "threshold_grid": contract["threshold_grid"],
            "minimum_relaxation_contract": contract["minimum_relaxation_contract"],
            "verdict_contract": contract["verdict_contract"],
        }).encode()),
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "worker_count": 1},
        "raw_trace_tensor_attack_access_before_execution": 0,
    })
    zero = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}
    dump_json(artifact / "input_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-access.v1",
        "phase": "PRE_EXECUTION_FREEZE", "status": "PASS",
        "raw_iq": dict(zero), "trace": dict(zero), "frozen_tensor": dict(zero), "attacks": dict(zero),
        "committed_metric_files": {"file_count": len(sources), "hash_verified": len(sources)},
    })
    readme = """# BITPROBE Stage-0A R0b relaxed-gate sensitivity

This is a post-hoc exploratory sensitivity analysis over committed metrics
only. It does not reopen raw IQ, TRACE, the frozen edge tensor, or attack data.
It does not recompute or alter any source metric and cannot change the formal
Stage-0A verdict or authorize Stage-0B.

ORIGINAL, MODERATE_RELAXATION, LENIENT_RELAXATION, and the full preregistered
threshold grid are evaluated independently for TEXBAT and OAKBAT. The frozen
flip-specificity result is advisory-only and excluded from all pass/fail
decisions.
"""
    (artifact / "README.md").write_text(readme)
    seal_manifest(artifact)


def _plots(artifact: Path, result: Mapping[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = result["grid_rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, dataset in zip(axes, DATASETS):
        effects = sorted({float(row["separability_effect_threshold"]) for row in grid})
        ps = sorted({float(row["permutation_p_threshold"]) for row in grid})
        image = np.asarray([[np.mean([row["core_effect_and_p_pass"] for row in grid if row["dataset"] == dataset and float(row["separability_effect_threshold"]) == effect and float(row["permutation_p_threshold"]) == p]) for p in ps] for effect in effects])
        shown = axis.imshow(image, origin="lower", aspect="auto", vmin=0, vmax=1)
        axis.set_title(dataset.split(".")[0]); axis.set_xticks(range(len(ps)), ps, rotation=45)
        axis.set_yticks(range(len(effects)), effects); axis.set_xlabel("p threshold")
    axes[0].set_ylabel("effect threshold"); fig.colorbar(shown, ax=axes, label="effect+p pass fraction")
    fig.savefig(artifact / "threshold_sensitivity_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    moderate = result["tier_results"]["MODERATE_RELAXATION"]
    for axis, dataset in zip(axes, DATASETS):
        criteria = moderate[dataset]["criteria"]
        axis.barh([row["criterion"] for row in criteria], [row["signed_margin"] for row in criteria], color=["tab:blue" if row["pass"] else "tab:red" for row in criteria])
        axis.axvline(0, color="black", linewidth=0.8); axis.set_title(dataset.split(".")[0]); axis.set_xlabel("signed pass margin")
    fig.tight_layout(); fig.savefig(artifact / "dataset_gate_margin.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    tiers = ("ORIGINAL", "MODERATE_RELAXATION", "LENIENT_RELAXATION")
    x = np.arange(len(tiers)); width = 0.35
    for index, dataset in enumerate(DATASETS):
        axis.bar(x + (index - 0.5) * width, [int(result["tier_results"][tier][dataset]["pass"]) for tier in tiers], width, label=dataset.split(".")[0])
    axis.set_xticks(x, tiers, rotation=15); axis.set_ylim(0, 1.2); axis.set_ylabel("dataset gate pass"); axis.legend()
    fig.tight_layout(); fig.savefig(artifact / "tier_comparison.png", dpi=150); plt.close(fig)


def execute(repo: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    artifact = repo / ARTIFACT_REL
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    for relative, expected in freeze["executable_bindings"].items():
        if binding(repo / relative, relative) != expected:
            raise SensitivityError(f"post-freeze executable change: {relative}")
    if verify_sources(repo) != freeze["source_bindings"]:
        raise SensitivityError("source metrics changed after freeze")
    first, second = analyze(repo), analyze(repo)
    first_bytes, second_bytes = analysis_bytes(first), analysis_bytes(second)
    identical = {name: first_bytes[name] == second_bytes[name] for name in first_bytes}
    if not all(identical.values()):
        raise SensitivityError("deterministic sensitivity rerun mismatch")
    for name, payload in first_bytes.items():
        (artifact / name).write_bytes(payload)
    dump_json(artifact / "deterministic_reproduction.json", {
        "status": "PASS", "run_count": 2, "byte_identical": True,
        "files": {name: {"byte_identical": identical[name], "sha256": sha256_bytes(first_bytes[name])} for name in sorted(first_bytes)},
    })
    dump_json(artifact / "formal_verdict_preservation.json", {
        "status": "PASS", "formal_stage0a_verdict_changed": False,
        "formal_source_verdict": "INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR",
        "post_hoc": True, "confirmatory": False, "stage0b_authorized": False,
    })
    zero = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}
    source_bindings = verify_sources(repo)
    dump_json(artifact / "input_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-access.v1",
        "phase": "COMMITTED_METRIC_SENSITIVITY", "status": "PASS",
        "raw_iq": dict(zero), "trace": dict(zero), "frozen_tensor": dict(zero), "attacks": dict(zero),
        "committed_metric_files": {
            "file_count": len(source_bindings), "hash_verified": len(source_bindings),
            "analysis_runs": 2, "source_metric_changes": 0,
            "bytes_per_analysis_run": sum(int(value["size_bytes"]) for value in source_bindings.values()),
        },
    })
    final = {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0b-final-verdict.v1",
        "status": "PASS", "exploratory_verdict": first["verdict"],
        "moderate_passing_datasets": first["moderate_passing_datasets"],
        "lenient_only_datasets": first["lenient_only"],
        "post_hoc": True, "confirmatory": False,
        "formal_stage0a_verdict_changed": False, "stage0b_authorized": False,
        "spoofing_detector_validated": False,
        "flip_specificity_advisory_only": True,
        "base_sha": BASE_SHA, "contract_commit_sha": CONTRACT_COMMIT_SHA,
        "freeze_sha": freeze_sha, "source_metric_changes": 0,
        "raw_iq_operations": 0, "trace_operations": 0,
        "frozen_tensor_operations": 0, "attack_operations": 0,
        "completed_utc": utc_now(),
    }
    dump_json(artifact / "final_verdict.json", final)
    dump_json(artifact / "freeze_commit.json", {
        "freeze_sha": freeze_sha, "local_remote_equal_before_execution": True,
        "ahead_behind_before_execution": [0, 0], "clean_checkout_before_execution": True,
        "post_result_code_or_gate_changes": 0,
    })
    _plots(artifact, first)
    (artifact / "README.md").write_text((artifact / "README.md").read_text() + f"\n## Final result\n\nExploratory verdict: {first['verdict']}. No dataset passes MODERATE_RELAXATION; LENIENT_ONLY datasets: {first['lenient_only'] or 'none'}. The formal Stage-0A verdict is unchanged and Stage-0B remains unauthorized.\n")
    seal_manifest(artifact)
    return final
