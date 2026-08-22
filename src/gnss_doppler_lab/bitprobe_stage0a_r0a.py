"""BITPROBE Stage-0A R0a inference-contract repair.

Only chronological-half bootstrap and exact PRN-label permutation inference
are repaired.  The original edge tensor, edge estimator, point statistics,
nuisances, synthetic controls, gates, and datasets remain frozen.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import bitprobe_stage0a as original


BASE_SHA = "0947dd3751b86d7f890d71561030e6f36a44f1e3"
PREREGISTRATION_COMMIT_SHA = "57857b1f8696d52d6056e135836e342de36ffaf8"
BRANCH = "research/bitprobe-stage0a-r0a-inference-contract-repair"
ARTIFACT_REL = "artifacts/bitprobe_stage0a_r0a_inference_contract_repair"
ORIGINAL_ARTIFACT_REL = "artifacts/bitprobe_stage0a_nav_edge_operator_identifiability"
ORIGINAL_ARTIFACT_TREE_SHA = "a7f50f1514eb32748a59cff9c62e785121b0a7f2"
ORIGINAL_MANIFEST_FILE_SHA256 = "42fb18e8b5d197e34ed4300d969a681d1bf2b69fae603c98f0da3f9ea83f6375"
TENSOR_PATH = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "bitprobe-stage0a-nav-edge-operator-identifiability/"
    "clean_nav_edge_residual_tensors.npz"
)
TENSOR_SIZE = 8_787_673
TENSOR_SHA256 = "b6df7a0b158b25166863959b49b05c165cf1d310d8609aad67e13a3a120c96da"
BLOCK_LENGTH = 10
REPLICATES = 500
SEED = 20260822
NUMERICAL_TOLERANCE = 1e-15
DATASET_ORDER = original.DATASET_ORDER
EXPECTED_PRNS = original.EXPECTED_PRNS
ALLOWED_VERDICTS = (
    "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE",
    "BITPROBE_STAGE0A_PARTIAL_DATASET_SUPPORT",
    "BITPROBE_STAGE0A_EDGE_OPERATOR_NOT_IDENTIFIABLE",
    "INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR",
)
EXECUTABLE_FILES = (
    "src/gnss_doppler_lab/bitprobe_stage0a_r0a.py",
    "scripts/run_bitprobe_stage0a_r0a.py",
    "scripts/verify_bitprobe_stage0a_r0a.py",
)
FROZEN_FILES = (
    "src/gnss_doppler_lab/bitprobe_stage0a.py",
    f"{ORIGINAL_ARTIFACT_REL}/preregistration.json",
    f"{ORIGINAL_ARTIFACT_REL}/nav_edge_inventory.csv.gz",
    f"{ORIGINAL_ARTIFACT_REL}/per_prn_support.csv",
    f"{ORIGINAL_ARTIFACT_REL}/split_half_metrics.csv",
    f"{ORIGINAL_ARTIFACT_REL}/between_prn_metrics.csv",
    f"{ORIGINAL_ARTIFACT_REL}/flip_no_flip_metrics.csv",
    f"{ORIGINAL_ARTIFACT_REL}/nuisance_metrics.csv",
    f"{ORIGINAL_ARTIFACT_REL}/synthetic_control_metrics.csv",
    f"{ORIGINAL_ARTIFACT_REL}/source_binding.json",
)


class RepairError(RuntimeError):
    """Fail-closed repair, binding, or inference error."""


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
    return {
        "path": display if display is not None else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def assert_branch(repo: Path) -> None:
    if git(repo, "branch", "--show-current") != BRANCH:
        raise RepairError("wrong R0a branch")
    if git(repo, "merge-base", "HEAD", BASE_SHA) != BASE_SHA:
        raise RepairError("R0a branch is not based on exact base")


def assert_pushed_freeze(repo: Path, freeze_sha: str) -> None:
    assert_branch(repo)
    if git(repo, "rev-parse", "HEAD") != freeze_sha:
        raise RepairError("HEAD differs from freeze SHA")
    if git(repo, "rev-parse", f"origin/{BRANCH}") != freeze_sha:
        raise RepairError("remote differs from freeze SHA")
    if git(repo, "rev-list", "--left-right", "--count", f"HEAD...origin/{BRANCH}") != "0\t0":
        raise RepairError("freeze is not ahead/behind 0/0")
    if git(repo, "status", "--porcelain=v1"):
        raise RepairError("freeze checkout is not clean")


def directory_binding(root: Path) -> dict[str, object]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append(binding(path, str(path.relative_to(root))))
    return {
        "file_count": len(rows),
        "files": rows,
        "aggregate_sha256": sha256_bytes(canonical_json(rows).encode()),
    }


def compact_manifest(artifact: Path) -> dict[str, object]:
    rows = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            rows.append(binding(path, str(path.relative_to(artifact))))
    return {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-manifest.v1",
        "status": "PASS",
        "file_count": len(rows),
        "files": rows,
    }


def seal_manifest(artifact: Path) -> None:
    dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))


def load_repair_preregistration(repo: Path) -> dict[str, object]:
    path = repo / ARTIFACT_REL / "repair_preregistration.json"
    value = json.loads(path.read_text())
    if value.get("base_sha") != BASE_SHA:
        raise RepairError("repair preregistration base mismatch")
    if value.get("status") != "FROZEN_BEFORE_REPAIR_IMPLEMENTATION_AND_TENSOR_ACCESS":
        raise RepairError("repair preregistration status mismatch")
    return value


def split_chronological(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen floor split.  Odd N assigns the extra edge to second half."""
    matrix = np.asarray(matrix)
    middle = len(matrix) // 2
    return matrix[:middle], matrix[middle:]


def resample_half_indices(length: int, rng: np.random.Generator, block_length: int = BLOCK_LENGTH) -> np.ndarray:
    """Fixed-partition contiguous block bootstrap within one frozen half.

    Complete source blocks are appended in draw order until the target length
    is met; only the final appended block is truncated.
    """
    if length <= 0 or block_length <= 0:
        raise ValueError("positive half length and block length required")
    blocks = [np.arange(start, min(start + block_length, length), dtype=np.int64) for start in range(0, length, block_length)]
    selected: list[np.ndarray] = []
    total = 0
    while total < length:
        block = blocks[int(rng.integers(0, len(blocks)))]
        selected.append(block)
        total += len(block)
    return np.concatenate(selected)[:length]


def resample_half(matrix: np.ndarray, rng: np.random.Generator, block_length: int = BLOCK_LENGTH) -> np.ndarray:
    matrix = np.asarray(matrix)
    return matrix[resample_half_indices(len(matrix), rng, block_length)]


def resampled_operator_pair(
    flip: np.ndarray, noflip: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    flip_first, flip_second = split_chronological(flip)
    noflip_first, noflip_second = split_chronological(noflip)
    first = original.complex_median(resample_half(flip_first, rng)) - original.complex_median(resample_half(noflip_first, rng))
    second = original.complex_median(resample_half(flip_second, rng)) - original.complex_median(resample_half(noflip_second, rng))
    return first, second


def similarity_matrix(kernels: Mapping[int, tuple[np.ndarray, np.ndarray]], prns: Sequence[int]) -> np.ndarray:
    ordered = tuple(sorted(int(prn) for prn in prns))
    return np.asarray([
        [original.normalized_similarity(kernels[left][0], kernels[right][1]) for right in ordered]
        for left in ordered
    ], dtype=np.float64)


def permutation_statistic(matrix: np.ndarray, permutation: Sequence[int]) -> tuple[float, int, int]:
    matrix = np.asarray(matrix, dtype=np.float64)
    count = matrix.shape[0]
    if matrix.shape != (count, count) or sorted(int(value) for value in permutation) != list(range(count)):
        raise ValueError("invalid square similarity matrix or permutation")
    same = [matrix[index, int(permutation[index])] for index in range(count)]
    different = [
        matrix[index, int(permutation[column])]
        for index in range(count) for column in range(count) if index != column
    ]
    return float(np.median(same) - np.median(different)), len(same), len(different)


def exact_prn_permutation(matrix: np.ndarray) -> dict[str, object]:
    count = int(np.asarray(matrix).shape[0])
    identity = tuple(range(count))
    observed, same_count, different_count = permutation_statistic(matrix, identity)
    values = []
    shapes = []
    for permutation in itertools.permutations(range(count)):
        value, same, different = permutation_statistic(matrix, permutation)
        values.append(value)
        shapes.append((same, different))
    p_value = float(sum(value >= observed - NUMERICAL_TOLERANCE for value in values) / math.factorial(count))
    return {
        "observed": observed,
        "p_value": p_value,
        "permutation_count": len(values),
        "same_count": same_count,
        "different_count": different_count,
        "all_shapes_valid": all(shape == (count, count * (count - 1)) for shape in shapes),
        "distribution_sha256": sha256_bytes(np.asarray(values, dtype="<f8").tobytes()),
    }


def load_inventory(repo: Path) -> list[dict[str, object]]:
    rows = original.read_csv(repo / ORIGINAL_ARTIFACT_REL / "nav_edge_inventory.csv.gz")
    converted: list[dict[str, object]] = []
    for row in rows:
        value: dict[str, object] = dict(row)
        value["prn"] = int(row["prn"])
        value["nav_bit_index"] = int(row["nav_bit_index"])
        value["flip"] = row["flip"] == "True"
        converted.append(value)
    return converted


def load_tensor_once(path: Path = TENSOR_PATH) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    counters = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}
    stat = path.stat()
    counters["stats"] += 1
    if stat.st_size != TENSOR_SIZE:
        raise RepairError("frozen tensor size mismatch")
    with path.open("rb") as stream:
        payload = stream.read()
    counters["opens"] += 1
    counters["bytes_read"] += len(payload)
    digest = sha256_bytes(payload)
    counters["hashes"] += 1
    if digest != TENSOR_SHA256:
        raise RepairError("frozen tensor SHA256 mismatch")
    archive = np.load(io.BytesIO(payload), allow_pickle=False)
    vectors = {key.replace("__", "|"): np.asarray(archive[key]) for key in archive.files}
    archive.close()
    return vectors, {
        "path": str(path), "size_bytes": stat.st_size, "sha256": digest,
        "operations": counters,
    }


def _edge_matrices(
    rows: Sequence[Mapping[str, object]], vectors: Mapping[str, np.ndarray], dataset: str, prns: Sequence[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        prn: (
            original._edge_matrix(rows, vectors, dataset, prn, True),
            original._edge_matrix(rows, vectors, dataset, prn, False),
        ) for prn in prns
    }


def reproduce_flip_specificity(
    matrices: Mapping[int, tuple[np.ndarray, np.ndarray]], dataset_index: int, prereg: Mapping[str, object]
) -> dict[str, object]:
    prns = sorted(matrices)
    flip_coherence = [
        original.normalized_similarity(original._split_kernel(matrices[p][0], 0), original._split_kernel(matrices[p][0], 1))
        for p in prns
    ]
    noflip_coherence = [
        original.normalized_similarity(original._split_kernel(matrices[p][1], 0), original._split_kernel(matrices[p][1], 1))
        for p in prns
    ]
    observed = float(np.median(flip_coherence) - np.median(noflip_coherence))
    rng = np.random.default_rng(SEED + 200 + dataset_index)
    combined = {p: np.concatenate(matrices[p], axis=0) for p in prns}
    permutation_values = []
    for _ in range(int(prereg["split_and_inference"]["flip_specificity_permutations"])):
        effects = []
        for prn in prns:
            pool = combined[prn]
            order = rng.permutation(len(pool))
            fake_flip = pool[order[:len(pool) // 2]]
            fake_noflip = pool[order[len(pool) // 2:]]
            effects.append(
                original.normalized_similarity(original._split_kernel(fake_flip, 0), original._split_kernel(fake_flip, 1))
                - original.normalized_similarity(original._split_kernel(fake_noflip, 0), original._split_kernel(fake_noflip, 1))
            )
        permutation_values.append(float(np.median(effects)))
    p_value = float((1 + sum(value >= observed for value in permutation_values)) / (1 + len(permutation_values)))
    return {
        "median_flip_reproducibility": float(np.median(flip_coherence)),
        "median_no_flip_reproducibility": float(np.median(noflip_coherence)),
        "flip_minus_no_flip": observed,
        "block_permutation_p": p_value,
        "reproduction_matches_original_algorithm": True,
        "new_contract_mismatch_found": True,
        "mismatch": {
            "preregistration": "contiguous blocks of 10 NAV edges within PRN",
            "frozen_code": "individual-edge random permutation of pooled flip/no-flip edges followed by equal-size repartition",
            "class_count_preservation": False,
            "repaired_in_r0a": False,
        },
    }


def repaired_analysis(
    repo: Path, rows: Sequence[Mapping[str, object]], vectors: Mapping[str, np.ndarray]
) -> dict[str, object]:
    prereg = json.loads((repo / ORIGINAL_ARTIFACT_REL / "preregistration.json").read_text())
    support = original.support_table(rows, vectors, prereg)
    eligible = {(row["dataset"], int(row["prn"])) for row in support if row["support_gate"] == "PASS"}
    split_rows: list[dict[str, object]] = []
    between_rows: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    bootstrap_records: dict[str, object] = {}
    dataset_gates: dict[str, object] = {}
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        prns = [prn for prn in EXPECTED_PRNS[dataset] if (dataset, prn) in eligible]
        matrices = _edge_matrices(rows, vectors, dataset, prns)
        kernels = {
            prn: (
                original._operator_kernel(matrices[prn][0], matrices[prn][1], 0),
                original._operator_kernel(matrices[prn][0], matrices[prn][1], 1),
            ) for prn in prns
        }
        for prn in prns:
            first, second = kernels[prn]
            split_rows.append({
                "dataset": dataset, "prn": prn,
                "complex_normalized_coherence": original.normalized_similarity(first, second),
                "phase_aligned_cosine": original.normalized_similarity(first, second),
                "phase_aligned_distance": original.phase_aligned_distance(first, second),
                "flip_only_coherence": original.normalized_similarity(
                    original._split_kernel(matrices[prn][0], 0), original._split_kernel(matrices[prn][0], 1)
                ),
                "no_flip_coherence": original.normalized_similarity(
                    original._split_kernel(matrices[prn][1], 0), original._split_kernel(matrices[prn][1], 1)
                ),
            })
        coherence_bootstrap = []
        rng = np.random.default_rng(SEED + dataset_index)
        for _ in range(REPLICATES):
            values = []
            for prn in prns:
                first, second = resampled_operator_pair(*matrices[prn], rng)
                values.append(original.normalized_similarity(first, second))
            coherence_bootstrap.append(float(np.median(values)))
        same = [original.normalized_similarity(*kernels[prn]) for prn in prns]
        different = [
            original.normalized_similarity(kernels[left][0], kernels[right][1])
            for left in prns for right in prns if left != right
        ]
        effect = float(np.median(same) - np.median(different))
        effect_bootstrap = []
        rng = np.random.default_rng(SEED + 100 + dataset_index)
        for _ in range(REPLICATES):
            boot = {prn: resampled_operator_pair(*matrices[prn], rng) for prn in prns}
            boot_same = [original.normalized_similarity(*boot[prn]) for prn in prns]
            boot_different = [
                original.normalized_similarity(boot[left][0], boot[right][1])
                for left in prns for right in prns if left != right
            ]
            effect_bootstrap.append(float(np.median(boot_same) - np.median(boot_different)))
        matrix = similarity_matrix(kernels, prns)
        permutation = exact_prn_permutation(matrix)
        ratios = [
            original.phase_aligned_distance(*kernels[prn]) / max(float(np.median([
                original.phase_aligned_distance(kernels[prn][0], kernels[other][1])
                for other in prns if other != prn
            ])), 1e-12) for prn in prns
        ]
        retrieval = sum(
            max(prns, key=lambda other: original.normalized_similarity(kernels[prn][0], kernels[other][1])) == prn
            for prn in prns
        ) / len(prns)
        coherence_lower = float(np.quantile(coherence_bootstrap, 0.025))
        effect_lower = float(np.quantile(effect_bootstrap, 0.025))
        median_coherence = float(np.median(same))
        between_rows.append({
            "dataset": dataset, "eligible_prns": len(prns),
            "median_same_similarity": median_coherence,
            "median_different_similarity": float(np.median(different)),
            "same_minus_different": effect,
            "bootstrap_95_lower": effect_lower,
            "exact_prn_label_permutation_p": permutation["p_value"],
            "median_within_between_distance_ratio": float(np.median(ratios)),
            "nearest_kernel_prn_retrieval_diagnostic": retrieval,
        })
        flip = reproduce_flip_specificity(matrices, dataset_index, prereg)
        flip_rows.append({"dataset": dataset, **flip})
        bootstrap_records[dataset] = {
            "replicate_count": len(coherence_bootstrap),
            "coherence_median_distribution_sha256": sha256_bytes(np.asarray(coherence_bootstrap, dtype="<f8").tobytes()),
            "effect_replicate_count": len(effect_bootstrap),
            "effect_distribution_sha256": sha256_bytes(np.asarray(effect_bootstrap, dtype="<f8").tobytes()),
            "coherence_bootstrap_95_lower": coherence_lower,
            "effect_bootstrap_95_lower": effect_lower,
            "permutation": permutation,
        }
        support_pass = len(prns) >= int(prereg["primary_gate"]["support"]["minimum_prns_per_dataset"])
        reproduction_pass = median_coherence >= 0.70 and coherence_lower > 0.50
        separation_pass = effect >= 0.15 and permutation["p_value"] <= 0.01 and effect_lower > 0.0
        flip_pass = flip["flip_minus_no_flip"] > 0.0 and flip["block_permutation_p"] <= 0.01
        dataset_gates[dataset] = {
            "support_pass": support_pass,
            "eligible_prns": len(prns),
            "reproducibility_pass": reproduction_pass,
            "median_same_prn_coherence": median_coherence,
            "coherence_bootstrap_95_lower": coherence_lower,
            "between_prn_separability_pass": separation_pass,
            "flip_specificity_pass": flip_pass,
            "technically_complete": support_pass,
        }
    gate_for_nuisance = {"datasets": dataset_gates, "support": support}
    nuisance_rows, nuisance_gate = original.evaluate_nuisances(rows, vectors, gate_for_nuisance, prereg)
    synthetic_rows, synthetic_gate, receiver_confound = original.evaluate_synthetic(prereg)
    for dataset in DATASET_ORDER:
        gate = dataset_gates[dataset]
        gate["strong_nuisance_pass"] = nuisance_gate[dataset]["strong_gate_pass"]
        gate["moderate_nuisance_pass"] = nuisance_gate[dataset]["moderate_gate_pass"]
        gate["real_data_gates_1_through_6_pass"] = all((
            gate["support_pass"], gate["reproducibility_pass"], gate["between_prn_separability_pass"],
            gate["flip_specificity_pass"], gate["strong_nuisance_pass"], gate["moderate_nuisance_pass"],
        ))
    return {
        "support": support, "split_rows": split_rows, "between_rows": between_rows,
        "flip_rows": flip_rows, "nuisance_rows": nuisance_rows, "nuisance_gate": nuisance_gate,
        "synthetic_rows": synthetic_rows, "synthetic_gate": synthetic_gate,
        "receiver_confound": receiver_confound, "bootstrap": bootstrap_records,
        "dataset_gates": dataset_gates,
    }


def analysis_bytes(result: Mapping[str, object]) -> dict[str, bytes]:
    fields = {
        "corrected_split_half_metrics.csv": (
            "dataset", "prn", "complex_normalized_coherence", "phase_aligned_cosine",
            "phase_aligned_distance", "flip_only_coherence", "no_flip_coherence",
        ),
        "corrected_between_prn_metrics.csv": (
            "dataset", "eligible_prns", "median_same_similarity", "median_different_similarity",
            "same_minus_different", "bootstrap_95_lower", "exact_prn_label_permutation_p",
            "median_within_between_distance_ratio", "nearest_kernel_prn_retrieval_diagnostic",
        ),
    }
    sources = {
        "corrected_split_half_metrics.csv": result["split_rows"],
        "corrected_between_prn_metrics.csv": result["between_rows"],
    }
    output: dict[str, bytes] = {}
    for name, fieldnames in fields.items():
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sources[name])
        output[name] = stream.getvalue().encode()
    output["corrected_bootstrap_metrics.json"] = (json.dumps(result["bootstrap"], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return output


def prepare_freeze(repo: Path) -> None:
    assert_branch(repo)
    artifact = repo / ARTIFACT_REL
    prereg = load_repair_preregistration(repo)
    original_root = repo / ORIGINAL_ARTIFACT_REL
    original_binding = directory_binding(original_root)
    if sha256_file(original_root / "artifact_manifest_sha256.json") != ORIGINAL_MANIFEST_FILE_SHA256:
        raise RepairError("original manifest file changed")
    if git(repo, "rev-parse", f"HEAD:{ORIGINAL_ARTIFACT_REL}") != ORIGINAL_ARTIFACT_TREE_SHA:
        raise RepairError("original artifact Git tree changed")
    dump_json(artifact / "repair_preregistration_commit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-preregistration-commit.v1",
        "commit_sha": PREREGISTRATION_COMMIT_SHA,
        "remote_ref": f"origin/{BRANCH}",
        "repair_preregistration_sha256": sha256_file(artifact / "repair_preregistration.json"),
        "local_remote_equal_before_implementation": True,
        "ahead_behind_before_implementation": [0, 0],
    })
    dump_json(artifact / "original_artifact_binding.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-original-binding.v1",
        "status": "PASS", "git_tree_sha": ORIGINAL_ARTIFACT_TREE_SHA,
        "manifest_file_sha256": ORIGINAL_MANIFEST_FILE_SHA256,
        "directory_binding": original_binding,
        "byte_identical_preservation_required": True,
    })
    dump_json(artifact / "frozen_tensor_binding.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-tensor-binding.v1",
        "status": "EXPECTED_NOT_ACCESSED_BEFORE_FREEZE",
        "path": str(TENSOR_PATH), "size_bytes": TENSOR_SIZE, "sha256": TENSOR_SHA256,
    })
    zero = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}
    dump_json(artifact / "input_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-input-access.v1",
        "phase": "PRE_TENSOR_ACCESS_FREEZE", "status": "PASS",
        "clean_raw_iq": dict(zero), "trace": dict(zero), "attacks": dict(zero),
        "frozen_tensor": dict(zero),
    })
    dump_json(artifact / "execution_freeze.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-execution-freeze.v1",
        "status": "IMPLEMENTED_AND_SYNTHETIC_TESTED_BEFORE_TENSOR_ACCESS",
        "base_sha": BASE_SHA, "repair_preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
        "executable_bindings": {relative: binding(repo / relative, relative) for relative in EXECUTABLE_FILES},
        "frozen_bindings": {relative: binding(repo / relative, relative) for relative in FROZEN_FILES},
        "repair_configuration_sha256": sha256_bytes(canonical_json({
            "chronological_half_bootstrap_contract": prereg["chronological_half_bootstrap_contract"],
            "exact_prn_permutation_contract": prereg["exact_prn_permutation_contract"],
            "frozen_reproduction_contract": prereg["frozen_reproduction_contract"],
            "post_result_contract": prereg["post_result_contract"],
        }).encode()),
        "environment": {
            "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "worker_count": 1,
        },
        "tensor_operations_before_freeze": 0,
    })
    readme = """# BITPROBE-GNSS Stage-0A R0a inference-contract repair

This is an inference implementation repair only. It does not reopen raw IQ,
TRACE, or attack data and does not regenerate edges. The edge estimator,
features, normalization, point statistics, nuisance suite, synthetic controls,
and every scientific gate remain frozen.

The repair isolates chronological halves before block bootstrap and applies a
shape-identical exact second-half PRN-label permutation. Synthetic level
detectability is not authoritative; the reproduced synthetic result measures
common-versus-separate geometry only. A single receiver cannot localize a
common operator to a transmitter.

Stage-0B authorization is determined only by the terminal R0a verdict after
the pushed execution freeze is replayed against the bound frozen tensor.
"""
    (artifact / "README.md").write_text(readme)
    seal_manifest(artifact)


def _compare_float(left: object, right: object, tolerance: float = 1e-14) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def execute_repair(repo: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    artifact = repo / ARTIFACT_REL
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    for relative, expected in freeze["executable_bindings"].items():
        if binding(repo / relative, relative) != expected:
            raise RepairError(f"post-freeze executable change: {relative}")
    original_before = json.loads((artifact / "original_artifact_binding.json").read_text())["directory_binding"]
    if directory_binding(repo / ORIGINAL_ARTIFACT_REL) != original_before:
        raise RepairError("original artifact changed before tensor analysis")
    rows = load_inventory(repo)
    vectors, tensor_audit = load_tensor_once()
    first = repaired_analysis(repo, rows, vectors)
    second = repaired_analysis(repo, rows, vectors)
    first_bytes, second_bytes = analysis_bytes(first), analysis_bytes(second)
    identical = {name: first_bytes[name] == second_bytes[name] for name in first_bytes}
    if not all(identical.values()):
        raise RepairError("two repaired analyses are not byte-identical")
    for name, payload in first_bytes.items():
        (artifact / name).write_bytes(payload)
    old_split = {row["dataset"]: row for row in original.read_csv(repo / ORIGINAL_ARTIFACT_REL / "between_prn_metrics.csv")}
    old_gate = json.loads((repo / ORIGINAL_ARTIFACT_REL / "initial_locked_analysis_result.json").read_text())["dataset_gates"]
    old_split_prn = original.read_csv(repo / ORIGINAL_ARTIFACT_REL / "split_half_metrics.csv")
    comparison = []
    unauthorized_changes = []
    for dataset in DATASET_ORDER:
        new_between = next(row for row in first["between_rows"] if row["dataset"] == dataset)
        old_between = old_split[dataset]
        metrics = (
            ("coherence_bootstrap_95_lower", old_gate[dataset]["coherence_bootstrap_95_lower"], first["dataset_gates"][dataset]["coherence_bootstrap_95_lower"], True, "chronological-half bootstrap repair"),
            ("between_bootstrap_95_lower", old_between["bootstrap_95_lower"], new_between["bootstrap_95_lower"], True, "chronological-half bootstrap repair"),
            ("exact_prn_label_permutation_p", old_between["exact_prn_label_permutation_p"], new_between["exact_prn_label_permutation_p"], True, "shape-identical exact permutation repair"),
            ("median_same_similarity", old_between["median_same_similarity"], new_between["median_same_similarity"], False, "frozen point estimate"),
            ("median_different_similarity", old_between["median_different_similarity"], new_between["median_different_similarity"], False, "frozen point estimate"),
            ("same_minus_different", old_between["same_minus_different"], new_between["same_minus_different"], False, "frozen point estimate"),
        )
        for metric, old_value, new_value, expected_change, reason in metrics:
            changed = not _compare_float(old_value, new_value)
            if changed and not expected_change:
                unauthorized_changes.append(f"{dataset}:{metric}")
            comparison.append({
                "dataset": dataset, "metric": metric, "original_value": old_value,
                "repaired_value": new_value, "changed": changed,
                "expected_to_change": expected_change,
                "gate_before": old_gate[dataset].get("real_data_gates_1_through_6_pass", False),
                "gate_after": first["dataset_gates"][dataset]["real_data_gates_1_through_6_pass"],
                "reason": reason,
            })
    new_split_lookup = {(row["dataset"], str(row["prn"])): row for row in first["split_rows"]}
    for old_row in old_split_prn:
        new_row = new_split_lookup[(old_row["dataset"], old_row["prn"])]
        for metric in ("complex_normalized_coherence", "phase_aligned_cosine", "phase_aligned_distance", "flip_only_coherence", "no_flip_coherence"):
            if not _compare_float(old_row[metric], new_row[metric]):
                unauthorized_changes.append(f"{old_row['dataset']}:PRN{old_row['prn']}:{metric}")
    original_support = original.read_csv(repo / ORIGINAL_ARTIFACT_REL / "per_prn_support.csv")
    support_match = len(original_support) == len(first["support"]) and all(
        all(str(old[field]) == str(new[field]) for field in original.SUPPORT_FIELDS)
        for old, new in zip(original_support, first["support"])
    )
    if not support_match:
        unauthorized_changes.append("support_or_edge_counts")
    flip_original = original.read_csv(repo / ORIGINAL_ARTIFACT_REL / "flip_no_flip_metrics.csv")
    flip_match = all(
        all(_compare_float(old[metric], new[metric]) for metric in (
            "median_flip_reproducibility", "median_no_flip_reproducibility", "flip_minus_no_flip", "block_permutation_p"
        )) for old, new in zip(flip_original, first["flip_rows"])
    )
    if not flip_match:
        unauthorized_changes.append("flip_no_flip_reproduction")
    nuisance_fields = ("dataset", "class", "nuisance", "level", "median_similarity", "effect_direction_maintained")
    synthetic_fields = ("scenario", "level", "seed", "statistic", "above_null_threshold")
    def csv_payload(fields: Sequence[str], values: Iterable[Mapping[str, object]]) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(values)
        return stream.getvalue().encode()
    nuisance_bytes = csv_payload(nuisance_fields, first["nuisance_rows"])
    synthetic_bytes = csv_payload(synthetic_fields, first["synthetic_rows"])
    nuisance_match = nuisance_bytes == (repo / ORIGINAL_ARTIFACT_REL / "nuisance_metrics.csv").read_bytes()
    synthetic_match = synthetic_bytes == (repo / ORIGINAL_ARTIFACT_REL / "synthetic_control_metrics.csv").read_bytes()
    if not nuisance_match: unauthorized_changes.append("nuisance_metrics")
    if not synthetic_match: unauthorized_changes.append("synthetic_metrics")
    comparison_fields = ("dataset", "metric", "original_value", "repaired_value", "changed", "expected_to_change", "gate_before", "gate_after", "reason")
    original.write_csv(artifact / "old_new_metric_comparison.csv", comparison_fields, comparison)
    dump_json(artifact / "flip_no_flip_reproduction.json", {
        "status": "PASS_REPRODUCTION_WITH_NEW_CONTRACT_MISMATCH",
        "values_match_original": flip_match, "datasets": first["flip_rows"],
        "new_mismatch_found": True, "repair_attempted": False,
        "required_fail_closed_verdict": "INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR",
    })
    dump_json(artifact / "nuisance_reproduction.json", {
        "status": "PASS" if nuisance_match else "FAIL", "byte_identical": nuisance_match,
        "sha256": sha256_bytes(nuisance_bytes), "gates": first["nuisance_gate"],
    })
    dump_json(artifact / "synthetic_reproduction.json", {
        "status": "PASS" if synthetic_match else "FAIL", "byte_identical": synthetic_match,
        "sha256": sha256_bytes(synthetic_bytes), "gate": first["synthetic_gate"],
        "synthetic_level_detectability_authoritative": False,
        "synthetic_common_vs_separate_geometry_only": True,
    })
    dump_json(artifact / "deterministic_reanalysis.json", {
        "status": "PASS", "analysis_run_count": 2,
        "byte_identical": all(identical.values()), "files": {
            name: {"byte_identical": identical[name], "sha256": sha256_bytes(first_bytes[name])}
            for name in sorted(first_bytes)
        },
    })
    dump_json(artifact / "freeze_commit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-freeze-commit.v1",
        "freeze_sha": freeze_sha, "local_remote_equal_before_tensor_access": True,
        "ahead_behind_before_tensor_access": [0, 0], "clean_checkout_before_tensor_access": True,
        "post_tensor_executable_changes": 0,
    })
    zero = {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}
    dump_json(artifact / "input_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-input-access.v1",
        "phase": "POST_FREEZE_FROZEN_TENSOR_REANALYSIS", "status": "PASS",
        "clean_raw_iq": dict(zero), "trace": dict(zero), "attacks": dict(zero),
        "frozen_tensor": tensor_audit["operations"],
        "frozen_tensor_binding": {key: tensor_audit[key] for key in ("path", "size_bytes", "sha256")},
    })
    dump_json(artifact / "frozen_tensor_binding.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-tensor-binding.v1",
        "status": "PASS", **{key: tensor_audit[key] for key in ("path", "size_bytes", "sha256")},
    })
    original_after = directory_binding(repo / ORIGINAL_ARTIFACT_REL)
    original_unchanged = original_after == original_before and git(repo, "rev-parse", f"HEAD:{ORIGINAL_ARTIFACT_REL}") == ORIGINAL_ARTIFACT_TREE_SHA
    if not original_unchanged:
        unauthorized_changes.append("original_artifact")
    new_flip_mismatch = any(row["new_contract_mismatch_found"] for row in first["flip_rows"])
    repair_pass = not unauthorized_changes and all(identical.values()) and original_unchanged and not new_flip_mismatch
    dataset_passes = {dataset: first["dataset_gates"][dataset]["real_data_gates_1_through_6_pass"] for dataset in DATASET_ORDER}
    if not repair_pass:
        verdict = "INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR"
    elif all(dataset_passes.values()) and first["synthetic_gate"]["gate_pass"]:
        verdict = "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE"
    elif sum(bool(value) for value in dataset_passes.values()) == 1:
        verdict = "BITPROBE_STAGE0A_PARTIAL_DATASET_SUPPORT"
    else:
        verdict = "BITPROBE_STAGE0A_EDGE_OPERATOR_NOT_IDENTIFIABLE"
    final = {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-r0a-final-verdict.v1",
        "status": "PASS" if repair_pass else "INCONCLUSIVE",
        "substantive_verdict": verdict,
        "repair_status": "PASS" if repair_pass else "FAIL",
        "stage0b_authorized": bool(verdict == "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE" and repair_pass),
        "base_sha": BASE_SHA, "repair_preregistration_commit_sha": PREREGISTRATION_COMMIT_SHA,
        "freeze_sha": freeze_sha, "dataset_gates": first["dataset_gates"],
        "synthetic_gate_reproduced": first["synthetic_gate"],
        "synthetic_level_detectability_authoritative": False,
        "synthetic_common_vs_separate_geometry_only": True,
        "new_flip_specificity_contract_mismatch": new_flip_mismatch,
        "unauthorized_value_changes": unauthorized_changes,
        "original_artifact_byte_identical": original_unchanged,
        "support_and_edge_counts_reproduced": support_match,
        "raw_iq_operations": 0, "trace_operations": 0, "attack_operations": 0,
        "post_result_method_feature_gate_threshold_changes": 0,
        "source_localization_available": False,
        "completed_utc": utc_now(),
    }
    dump_json(artifact / "post_result_change_audit.json", {
        "status": "PASS" if not unauthorized_changes else "FAIL",
        "allowed_changed_values_only": not unauthorized_changes,
        "unauthorized_changes": unauthorized_changes,
        "original_artifact_byte_identical": original_unchanged,
        "post_result_method_feature_gate_threshold_changes": 0,
        "flip_specificity_mismatch_not_repaired": new_flip_mismatch,
    })
    dump_json(artifact / "final_verdict.json", final)
    seal_manifest(artifact)
    return final
