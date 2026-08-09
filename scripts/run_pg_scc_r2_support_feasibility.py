#!/usr/bin/env python3
"""Protected Phase-2 producer for the preregistered PG-SCC R2 diagnostic.

Do not invoke during Phase 1. The full pushed implementation SHA is verified
before any protected cache, metadata sidecar, score table, or NPZ is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.pg_scc_physics import (  # noqa: E402
    DEFAULT_SEARCH, N_COORDINATES, analytic_same_prn_template, normalize_complex,
)
from gnss_doppler_lab.pg_scc_r2_outcomes import (  # noqa: E402
    calibration_and_pairs as strict_calibration_and_pairs, full_control_results,
)

from gnss_doppler_lab.pg_scc_r2_support import (  # noqa: E402
    aggregate_accounting, universe_support,
)


BRANCH = "research/pg-scc-stage0-r2-support-feasibility"
BASE_SHA = "8cd78ed724e57f97498da26547a9ecbbc2a78fe1"
R1_IMPLEMENTATION_SHA = "5359bfab74d44d7153a32c5cf708ccab240fe219"
R1_PREREGISTRATION_SHA = "5e5339282c6154630fd11f94415cba794d9fa1ec"
OUTPUT = ROOT / "artifacts/pg_scc_stage0_r2_support_feasibility"
CACHE = ROOT / "artifacts/acaf_nf_stage1_r3_static_detection"
FROZEN = ROOT / "artifacts/pg_scc_stage0_static_k9"
CONFIG = ROOT / "configs/pg_scc_stage0_r2_support_feasibility.json"
CONFIG_SHA256 = "013d83dc2245e6cb896607aeec124e7d5cd9fb0a107832affc9aec4d9b12e904"
SUPPORT_SUMMARY_SHA256 = "e55190563d9b4711984a94e79b0d0e6e35c604c3a58e81af39b08d12ad85b513"
SOURCE_SHA256 = "9529f04e66b85861b187f20b124fd003d8ad49de0657ba78fd773d3ae0e9f57a"
R1_FROZEN_HASHES = {
    "config.json": "ade700892fe3e055e0154ae859ba701a36f11a6ec9245db4b39fd087426eb0e6",
    "source_commit.json": "13ab9e544f06903dd6fde04c42e8aefb35a66946bbaf0456e95040fbb777d428",
    "r1_fail_closed_report.json": "041cc432cdc893e9dba867d6d3dc005e3ee7f2c8d25d542d54d4f374ca68e3f5",
}
COMPARISONS = {
    "K9": ("pg_scc_k9", "fixed9", "shuffled_k9"),
    "K5": ("pg_scc_k5", "uniform_k5", "shuffled_k5"),
    "K3": ("pg_scc_k3", "epl3", "shuffled_k3"),
    "DENSE": ("dense_two_source_glrt",),
}
METHOD_BUDGETS = {
    "pg_scc_k9": 9, "fixed9": 9, "shuffled_k9": 9,
    "pg_scc_k5": 5, "uniform_k5": 5, "shuffled_k5": 5,
    "pg_scc_k3": 3, "epl3": 3, "shuffled_k3": 3,
}

GENERATED = {
    "support_accounting.json", "calibration.json", "paired_results.json",
    "control_results.json", "final_diagnostic.json", "fail_closed.json",
    "artifact_manifest_sha256.json",
}
IMPLEMENTATION_FILES = {
    "artifacts/pg_scc_stage0_r2_support_feasibility/README.md",
    "artifacts/pg_scc_stage0_r2_support_feasibility/source_commit.json",
    "artifacts/pg_scc_stage0_r2_support_feasibility/support_inventory_summary.json",
    "configs/pg_scc_stage0_r2_support_feasibility.json",
    "docs/PG_SCC_STAGE0_R2_IMPLEMENTATION_NOTES.md",
    "docs/PG_SCC_STAGE0_R2_SUPPORT_DESIGN_INTENT.md",
    "docs/PG_SCC_STAGE0_R2_SUPPORT_FEASIBILITY_PREREGISTRATION.md",
    "scripts/inventory_pg_scc_r2_support.py",
    "scripts/run_pg_scc_r2_support_feasibility.py",
    "scripts/verify_pg_scc_r2_support_feasibility.py",
    "src/gnss_doppler_lab/pg_scc_r2_outcomes.py",
    "src/gnss_doppler_lab/pg_scc_r2_support.py",
    "tests/test_pg_scc_r2_support_feasibility.py",
    "tests/test_pg_scc_r2_outcomes.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _git(*args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_implementation_freeze(expected_sha: str) -> dict[str, Any]:
    """Verify all freeze conditions before opening any protected input."""
    errors = []
    head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    if len(expected_sha) != 40 or expected_sha != head:
        errors.append("implementation_sha_not_exact_head")
    if branch != BRANCH:
        errors.append("branch_mismatch")
    for ancestor in (BASE_SHA, R1_IMPLEMENTATION_SHA, R1_PREREGISTRATION_SHA):
        if subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, head], cwd=ROOT).returncode:
            errors.append(f"frozen_ancestor_missing:{ancestor}")
    for relative in sorted(IMPLEMENTATION_FILES):
        if subprocess.run(["git", "ls-files", "--error-unmatch", relative], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            errors.append(f"implementation_untracked:{relative}")
    bindings = {
        CONFIG: CONFIG_SHA256,
        OUTPUT / "support_inventory_summary.json": SUPPORT_SUMMARY_SHA256,
        OUTPUT / "source_commit.json": SOURCE_SHA256,
    }
    for path, expected in bindings.items():
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen_binding:{path.name}")
    r1_root = ROOT / "artifacts/pg_scc_stage0_r1_root_cause_audit"
    for name, expected in R1_FROZEN_HASHES.items():
        if not (r1_root / name).is_file() or sha256(r1_root / name) != expected:
            errors.append(f"r1_immutability:{name}")
    frozen_manifest = load_json(FROZEN / "freeze_manifest.json")
    for relative, expected in frozen_manifest.items():
        path = FROZEN / relative
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"frozen_input_drift:{relative}")
    remote = _git("rev-parse", "--verify", f"refs/remotes/origin/{branch}", check=False)
    if remote != head:
        errors.append("local_remote_sha_mismatch")
    if remote and _git("rev-list", "--left-right", "--count", f"{head}...{remote}").split() != ["0", "0"]:
        errors.append("ahead_behind_not_zero_zero")
    dirty = []
    prefix = str(OUTPUT.relative_to(ROOT)) + "/"
    for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        relative = line[3:]
        if relative.startswith(prefix) and relative[len(prefix):] in GENERATED:
            continue
        dirty.append(line)
    if dirty:
        errors.append("dirty_worktree_outside_declared_outputs:" + "|".join(dirty))
    if errors:
        raise RuntimeError("FAIL_CLOSED_IMPLEMENTATION_FREEZE:" + ";".join(errors))
    return {
        "status": "PASS", "implementation_sha": head, "branch": branch,
        "remote_sha": remote, "ahead_behind": [0, 0],
        "config_sha256": CONFIG_SHA256,
        "support_summary_sha256": SUPPORT_SUMMARY_SHA256,
        "source_sha256": SOURCE_SHA256,
        "r1_files_verified": len(R1_FROZEN_HASHES),
        "frozen_input_files_verified": len(frozen_manifest),
    }


def load_support_metadata() -> list[dict[str, Any]]:
    """Load structural JSON sidecars only; no NPZ or scores are opened here."""
    rows = []
    forbidden = {"score", "label", "outcome", "alarm", "threshold", "auroc", "verdict"}
    for role, name in (("clean", "clean_features.json"), ("attack", "attack_features.json")):
        path = CACHE / name
        values = load_json(path)
        for value in values:
            if forbidden & set(value):
                raise RuntimeError("outcome-bearing field in support metadata")
            required = {"scenario", "phase", "second", "prn"}
            if not required.issubset(value):
                raise RuntimeError("support metadata schema mismatch")
            rows.append({"source_role": role, **{key: value[key] for key in required}})
    return rows


def load_protected_records(metadata: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Open protected surface arrays only after freeze and support gates pass."""
    output = []
    offset = 0
    for role, name in (("clean", "clean_features.npz"), ("attack", "attack_features.npz")):
        archive = np.load(CACHE / name, allow_pickle=False)
        count = int(archive["surfaces"].shape[0])
        selected = metadata[offset:offset + count]
        if len(selected) != count or any(row["source_role"] != role for row in selected):
            raise RuntimeError("metadata/surface array binding mismatch")
        for index, row in enumerate(selected):
            output.append({**row, "surface": archive["surfaces"][index]})
        offset += count
    if offset != len(metadata):
        raise RuntimeError("metadata/surface total length mismatch")
    return output


def batch_glrt(surfaces: np.ndarray, auth: np.ndarray, covariance: np.ndarray,
               indices: Sequence[int] | None = None) -> np.ndarray:
    selected = np.arange(N_COORDINATES) if indices is None else np.asarray(indices, int)
    y = np.asarray(surfaces, np.complex128)[:, selected]
    a = np.asarray(auth, np.complex128)[selected]
    cov = np.asarray(covariance, np.complex128)[np.ix_(selected, selected)]
    off = cov - np.diag(np.diag(cov))
    precision = (np.diag(1 / np.maximum(np.real(np.diag(cov)), 1e-12))
                 if np.count_nonzero(off) == 0 else np.linalg.inv(cov))

    def rss(design: np.ndarray) -> np.ndarray:
        gram = design.conj().T @ precision @ design + np.eye(design.shape[1]) * 1e-3
        coefficients = np.linalg.solve(gram, (design.conj().T @ precision @ y.T)).T
        residual = y - coefficients @ design.T
        return np.maximum(np.real(np.sum(np.conj(residual) * (residual @ precision.T), axis=1)), 0)

    null = rss(a[:, None])
    best = null.copy()
    for delay, doppler in DEFAULT_SEARCH:
        alternative = analytic_same_prn_template(delay, doppler)[selected]
        best = np.minimum(best, rss(np.column_stack((a, alternative))))
    return np.maximum(null - best, 0) / len(selected)


def score_methods(records: Sequence[Mapping[str, Any]], masks: Mapping[str, Sequence[int]],
                  auth: np.ndarray, covariance: np.ndarray) -> list[dict[str, Any]]:
    surfaces = np.asarray([normalize_complex(row["surface"], "prompt_phase") for row in records])
    rows = []
    for method in sorted({item for values in COMPARISONS.values() for item in values}):
        selected = None if method == "dense_two_source_glrt" else masks[method]
        if selected is not None and (len(selected) != METHOD_BUDGETS[method] or len(set(selected)) != len(selected)):
            raise RuntimeError(f"frozen mask cardinality mismatch:{method}")
        scores = batch_glrt(surfaces, auth, covariance, selected)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"nonfinite algorithmic result:{method}")
        for metadata, score in zip(records, scores):
            rows.append({**{key: metadata[key] for key in (*("source_role", "scenario", "phase", "second"), "prn")},
                         "method": method, "score": float(score)})
    return rows


def support_fingerprint(values: Mapping[tuple[str, str, str, int], set[int]]) -> str:
    canonical = [[*event, sorted(values[event])] for event in sorted(values)]
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def finalize_manifest() -> dict[str, str]:
    manifest = {str(path.relative_to(OUTPUT)): sha256(path) for path in sorted(OUTPUT.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest_sha256.json"}
    dump_json(OUTPUT / "artifact_manifest_sha256.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-sha", required=True)
    args = parser.parse_args()
    freeze = verify_implementation_freeze(args.implementation_sha)
    gate = 0
    try:
        config = load_json(CONFIG)
        metadata = load_support_metadata()
        inventory = load_json(OUTPUT / "support_inventory_summary.json")
        inferred_rows = [{**row, "method": method} for method in sorted({m for v in COMPARISONS.values() for m in v}) for row in metadata]
        accounting = aggregate_accounting(metadata, inferred_rows, COMPARISONS)
        accounting["universe_fingerprint"] = support_fingerprint(universe_support(metadata))
        accounting["implementation_freeze"] = freeze
        if accounting["universe"]["total_events"] != inventory["total_event_count"]:
            raise RuntimeError("inventory event denominator mismatch")
        if accounting["universe"]["common_unique_prn_histogram"] != inventory["common_unique_prn_histogram"]:
            raise RuntimeError("inventory cardinality histogram mismatch")
        dump_json(OUTPUT / "support_accounting.json", accounting)
        gate = 1

        records = load_protected_records(metadata)
        arrays = np.load(FROZEN / "normalization_covariance.npz", allow_pickle=False)
        masks = load_json(FROZEN / "masks.json")
        scored = score_methods(records, masks, arrays["auth_template"], arrays["covariance"])
        scored_accounting = aggregate_accounting(metadata, scored, COMPARISONS)
        if scored_accounting["comparison_families"] != accounting["comparison_families"]:
            raise RuntimeError("algorithmic method availability changed support accounting")
        gate = 2

        calibration, paired = strict_calibration_and_pairs(scored, config)
        dump_json(OUTPUT / "calibration.json", calibration)
        gate = 3
        controls = full_control_results(records, arrays, masks, calibration)
        dump_json(OUTPUT / "control_results.json", controls)
        dump_json(OUTPUT / "paired_results.json", paired)
        gate = 4
        permutation_cells = [cell for cell in paired["cells"] if cell["control_role"] == "RELATIONSHIP_PERMUTATION"]
        if not permutation_cells:
            raise RuntimeError("relationship permutation evidence missing")
        interpretation = ("DIAGNOSTIC_ONLY" if any(cell["status"] == "AVAILABLE" for cell in permutation_cells)
                          else "NO_CAUSAL_INTERPRETATION")
        final = {
            "schema": "pg_scc_stage0_r2_final_diagnostic.v1",
            "status": "POST_R1_SUPPORT_REPAIRED_DIAGNOSTIC",
            "independent_confirmatory_evidence": False,
            "requires_new_untouched_holdout": True,
            "interpretation": interpretation,
            "total_events": accounting["universe"]["total_events"],
            "relationship_permutation_cells": len(permutation_cells),
            "available_relationship_permutation_cells": sum(cell["status"] == "AVAILABLE" for cell in permutation_cells),
            "all_denominators_explicit": True,
            "k_eff_used": False,
            "implementation_freeze": freeze,
        }
        dump_json(OUTPUT / "final_diagnostic.json", final)
        finalize_manifest()
        print(json.dumps(final, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        if gate >= 1:
            dump_json(OUTPUT / "fail_closed.json", {
                "schema": "pg_scc_stage0_r2_fail_closed.v1", "status": "FAIL_CLOSED",
                "completed_gate": gate, "reason": str(exc), "causal_interpretation": False,
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
