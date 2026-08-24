"""Label-only design helpers for Jammertest 2025 CRPA Stage-0D."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0b import load_label_rows, sha256_file


BASE_COMMIT = "c71b225e3c07f28d685666a977dae94c4cb03214"
SEED = 20_250_823
BLOCK_SIZE = 32
CLASS_BLOCKS = {
    "Spoof": (231, 232, 233, 234, 235),
    "Prn": (95, 96, 97, 98, 99, 100),
}
FOLD_TEST_BLOCKS = (
    {"Spoof": (231,), "Prn": (95, 100)},
    {"Spoof": (232,), "Prn": (96,)},
    {"Spoof": (233,), "Prn": (97,)},
    {"Spoof": (234,), "Prn": (98,)},
    {"Spoof": (235,), "Prn": (99,)},
)
MODEL_NAMES = ("M0", "M1", "M2", "M2R", "M3")
SPATIAL_MODELS = ("M2", "M2R")
CONTROL_NAMES = ("mismatched", "circular_shift", "fourier_phase_randomized")
CALIPERS_DB = (0.10, 0.25, 0.50, 1.00)
PRIMARY_CALIPER_DB = 0.25
VERDICTS = {
    "SPLIT_VALIDITY_FAILURE",
    "SPOOF_EVALUATION_INVALID_NO_RECEIVED_POWER_OVERLAP",
    "NO_TRUE_SPOOF_SPATIAL_SEPARATION",
    "TRUE_SPOOF_SPATIAL_SIGNAL_POWER_CONFOUNDED",
    "TRUE_SPOOF_SPATIAL_SEPARATION_NOT_INCREMENTAL",
    "PHASE_OR_LOCATION_SHORTCUT_ONLY",
    "PROMISING_TRUE_SPOOF_SPATIAL_INCREMENT_PROVENANCE_BLOCKED",
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def eligible_rows(split_root: Path) -> list[dict]:
    return [
        row for row in load_label_rows(split_root)
        if row["area"] == 1
        and row["transmit_power_dbm"] == 40
        and row["class_name"] in CLASS_BLOCKS
    ]


def class_guard_blocks(class_name: str, test_blocks: tuple[int, ...]) -> set[int]:
    """Apply ±1 guard only inside the same class's enumerated block run."""

    run = set(CLASS_BLOCKS[class_name])
    test = set(test_blocks)
    return {
        adjacent
        for block in test
        for adjacent in (block - 1, block + 1)
        if adjacent in run and adjacent not in test
    }


def role_for(class_name: str, block: int, fold: int) -> str:
    test = set(FOLD_TEST_BLOCKS[fold][class_name])
    guard = class_guard_blocks(class_name, FOLD_TEST_BLOCKS[fold][class_name])
    if block in test:
        return "test"
    if block in guard:
        return "guard"
    return "train"


def build_manifests(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    split_rows: list[dict] = []
    guard_rows: list[dict] = []
    for fold in range(len(FOLD_TEST_BLOCKS)):
        for row in rows:
            block = row["sample_index"] // BLOCK_SIZE
            role = role_for(row["class_name"], block, fold)
            record = {
                "fold": fold,
                "sample_index": row["sample_index"],
                "class_name": row["class_name"],
                "binary_label": int(row["class_name"] == "Spoof"),
                "area": row["area"],
                "transmit_power_dbm": int(row["transmit_power_dbm"]),
                "block_size": BLOCK_SIZE,
                "class_block": block,
                "class_block_key": f"{row['class_name']}:{block}",
                "role": role,
            }
            split_rows.append(record)
            if role == "guard":
                guard_rows.append({
                    **record,
                    "guard_reason": "same-class adjacent block to fixed test block",
                })
    return split_rows, guard_rows


def validate_complete_oof(rows: list[dict], split_rows: list[dict]) -> dict:
    errors: list[str] = []
    counts = Counter(row["class_name"] for row in rows)
    expected_counts = {"Spoof": 124, "Prn": 164}
    if dict(counts) != expected_counts:
        errors.append(f"eligible class counts mismatch: {dict(counts)}")

    observed_blocks = {
        class_name: sorted({
            row["sample_index"] // BLOCK_SIZE
            for row in rows if row["class_name"] == class_name
        })
        for class_name in CLASS_BLOCKS
    }
    for class_name, expected in CLASS_BLOCKS.items():
        if observed_blocks[class_name] != list(expected):
            errors.append(f"{class_name} block inventory mismatch")

    test_counts = Counter(
        row["sample_index"] for row in split_rows if row["role"] == "test"
    )
    missing = sorted(row["sample_index"] for row in rows if test_counts[row["sample_index"]] == 0)
    duplicate = sorted(index for index, count in test_counts.items() if count != 1)
    if missing:
        errors.append(f"OOF missing sample count: {len(missing)}")
    if duplicate:
        errors.append(f"OOF duplicate sample count: {len(duplicate)}")

    fold_audits = []
    for fold in range(len(FOLD_TEST_BLOCKS)):
        current = [row for row in split_rows if row["fold"] == fold]
        role_samples = {
            role: {row["sample_index"] for row in current if row["role"] == role}
            for role in ("train", "guard", "test")
        }
        overlap = (
            role_samples["train"] & role_samples["guard"]
            or role_samples["train"] & role_samples["test"]
            or role_samples["guard"] & role_samples["test"]
        )
        train_blocks = {row["class_block_key"] for row in current if row["role"] == "train"}
        test_blocks = {row["class_block_key"] for row in current if row["role"] == "test"}
        class_counts = {
            role: dict(Counter(row["class_name"] for row in current if row["role"] == role))
            for role in ("train", "guard", "test")
        }
        valid = (
            not overlap
            and train_blocks.isdisjoint(test_blocks)
            and all(class_counts[role].get(name, 0) > 0 for role in ("train", "test") for name in CLASS_BLOCKS)
        )
        if not valid:
            errors.append(f"fold {fold} role/block/class validity failure")
        fold_audits.append({
            "fold": fold,
            "valid": valid,
            "class_counts": class_counts,
            "train_class_block_count": len(train_blocks),
            "test_class_block_count": len(test_blocks),
            "sample_role_overlap_count": len(overlap),
            "train_test_class_block_overlap_count": len(train_blocks & test_blocks),
        })

    test_class_counts = Counter(
        row["class_name"] for row in split_rows if row["role"] == "test"
    )
    if dict(test_class_counts) != expected_counts:
        errors.append(f"OOF test class counts mismatch: {dict(test_class_counts)}")
    test_class_blocks = {
        row["class_block_key"] for row in split_rows if row["role"] == "test"
    }
    expected_class_blocks = {
        f"{class_name}:{block}"
        for class_name, blocks in CLASS_BLOCKS.items()
        for block in blocks
    }
    if test_class_blocks != expected_class_blocks:
        errors.append("OOF class-block coverage mismatch")

    return {
        "status": "PASS" if not errors else "SPLIT_VALIDITY_FAILURE",
        "errors": errors,
        "eligible_snapshot_count": len(rows),
        "expected_class_counts": expected_counts,
        "oof_test_class_counts": dict(test_class_counts),
        "oof_missing_count": len(missing),
        "oof_duplicate_count": len(duplicate),
        "enumerated_unique_class_block_count": len(expected_class_blocks),
        "request_terminal_text_says_block_count": 10,
        "block_count_resolution": "The explicit fixed lists enumerate 5 Spoof + 6 Prn = 11 class blocks and are authoritative.",
        "oof_class_blocks": sorted(test_class_blocks),
        "folds": fold_audits,
    }


def label_hashes(split_root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(split_root.glob("*_crpa_*.txt"))
    }


def build_design(split_root: Path, artifact: Path) -> dict:
    rows = eligible_rows(split_root)
    split_rows, guard_rows = build_manifests(rows)
    contract = validate_complete_oof(rows, split_rows)
    if contract["status"] != "PASS":
        raise SystemExit("SPLIT_VALIDITY_FAILURE")
    artifact.mkdir(parents=True, exist_ok=True)
    write_csv(
        artifact / "split_manifest.csv",
        ["fold", "sample_index", "class_name", "binary_label", "area", "transmit_power_dbm", "block_size", "class_block", "class_block_key", "role"],
        split_rows,
    )
    write_csv(
        artifact / "guard_manifest.csv",
        ["fold", "sample_index", "class_name", "binary_label", "area", "transmit_power_dbm", "block_size", "class_block", "class_block_key", "role", "guard_reason"],
        guard_rows,
    )
    write_json(artifact / "complete_oof_contract.json", contract)
    design = {
        "status": "LABEL_ONLY_DESIGN_FREEZE_PRE_IQ",
        "base_commit": BASE_COMMIT,
        "seed": SEED,
        "iq_bytes_read": 0,
        "stage0c_model_results_used": False,
        "label_hashes": label_hashes(split_root),
        "eligibility": {"area": 1, "transmit_power_dbm": 40, "positive_exact_class": "Spoof", "negative_exact_class": "Prn"},
        "class_counts": {"Spoof": 124, "Prn": 164},
        "block_size": BLOCK_SIZE,
        "class_blocks": {name: list(blocks) for name, blocks in CLASS_BLOCKS.items()},
        "fold_test_blocks": [
            {name: list(blocks) for name, blocks in fold.items()}
            for fold in FOLD_TEST_BLOCKS
        ],
        "guard": "test block ±1 only within the same class contiguous enumerated block run",
        "balancing": {"test_sample_removal": False, "train_sample_removal": False, "classifier_class_weight": "balanced"},
        "matching": {
            "feature": "received_power_db = 10*log10(mean over channel,time of abs(x)^2)",
            "algorithm": "fold-local one-to-one without replacement; maximum cardinality then minimum total absolute-power-difference cost",
            "train_test_cross_pairing": False,
            "primary_caliper_db": PRIMARY_CALIPER_DB,
            "sensitivity_calipers_db": [0.10, 0.50, 1.00],
            "inputs": ["class label", "received_power_db"],
        },
        "models": {
            "M0": "received_power_db only",
            "M1": "channel-0 normalized amplitude statistics and 16-bin normalized spectrum; absolute power excluded",
            "M2": "centered/RMS-normalized covariance eigenvalue fractions, lambda1/trace, effective rank, sorted coherence magnitudes and summaries; no channel identity or coherence phase",
            "M2R": "M2 residual after train-only received-power third-degree polynomial Ridge(alpha=1.0)",
            "M3": "M2 plus six complex coherence real/imaginary components; diagnostic only",
        },
        "pipeline": {
            "scaler": "StandardScaler(train fold only)",
            "classifier": {"type": "LogisticRegression", "penalty": "l2", "C": 1.0, "solver": "liblinear", "class_weight": "balanced", "max_iter": 2000, "random_state": SEED},
            "threshold": 0.5,
            "test_label_threshold_use": False,
            "bootstrap_replicates": 2000,
            "bootstrap_unit": "original test class_block_key",
        },
        "controls": {
            "destruction": list(CONTROL_NAMES),
            "fixed_channel_permutation": [2, 0, 3, 1],
            "one_channel_ablation": True,
            "global_gains": [0.1, 1.0, 3.7, 10.0],
            "global_phases_radians": [0.0, 0.37, 1.123, 2.9],
            "invariance_rtol": 1e-10,
            "invariance_atol": 1e-12,
        },
        "verdicts": sorted(VERDICTS),
        "forbidden_claims": ["clean-versus-spoof detection", "general GNSS spoof detector", "recording-independent generalization", "READY_FOR_WCL"],
    }
    write_json(artifact / "design_freeze.json", design)
    return design
