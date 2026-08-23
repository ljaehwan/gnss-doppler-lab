"""Frozen label-only design helpers for Jammertest CRPA Stage-0C."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

from gnss_doppler_lab.jammertest_crpa_stage0b import load_label_rows, sha256_file


SEED = 20_250_823
PRIMARY = {
    "name": "primary",
    "area": 1,
    "powers_dbm": (30.0, 40.0),
    "block_size": 32,
    "folds": 3,
    "guard_blocks_each_side": 1,
}
SENSITIVITY_A = {
    "name": "sensitivity_a",
    "area": 1,
    "powers_dbm": (30.0, 40.0),
    "block_size": 128,
    "folds": 2,
    "guard_blocks_each_side": 1,
}
SENSITIVITY_B = {
    "name": "sensitivity_b",
    "area": 1,
    "powers_dbm": (25.0, 40.0),
    "block_size": 512,
    "folds": 2,
    "guard_blocks_each_side": 1,
}
CONFIGS = (PRIMARY, SENSITIVITY_A, SENSITIVITY_B)
MODEL_NAMES = ("M0", "M1", "M2", "M3")
CONTROL_NAMES = ("actual", "mismatched", "circular_shift", "fourier_phase_randomized")
VERDICTS = {
    "NO_INCREMENTAL_SPATIAL_DISCRIMINATION",
    "WEAK_SPATIAL_SIGNAL_NOT_STABLE",
    "PHASE_OR_LOCATION_SHORTCUT_ONLY",
    "PROMISING_SPATIAL_INCREMENT_PROVENANCE_BLOCKED",
    "SPLIT_OR_NUMERICAL_VALIDITY_FAILURE",
}


def is_positive(row: dict) -> bool:
    return "Spoof" in row["class_name"] or "Meac" in row["class_name"]


def deterministic_digest(*parts: object) -> str:
    payload = ":".join(str(part) for part in (SEED, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


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


def cell_key(row: dict) -> tuple[int, int]:
    return int(row["transmit_power_dbm"]), int(is_positive(row))


def eligible_rows(rows: list[dict], config: dict) -> list[dict]:
    return [
        row
        for row in rows
        if row["area"] == config["area"]
        and row["transmit_power_dbm"] in config["powers_dbm"]
    ]


def cell_block_inventory(rows: list[dict], config: dict) -> dict[tuple[int, int], dict[int, int]]:
    inventory: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in eligible_rows(rows, config):
        inventory[cell_key(row)][row["sample_index"] // config["block_size"]] += 1
    return {cell: dict(blocks) for cell, blocks in inventory.items()}


def _ordered(values: list[int], *context: object) -> list[int]:
    return sorted(values, key=lambda value: deterministic_digest(*context, value))


def solve_test_blocks(rows: list[dict], config: dict) -> tuple[list[dict], dict]:
    """Find deterministic disjoint test blocks with a one-block guard.

    Each fold chooses at most one block per power/class cell.  Candidate order
    is SHA-256 fixed.  A solution must retain both train and test samples in
    every cell after removing adjacent guard blocks.
    """

    inventory = cell_block_inventory(rows, config)
    cells = sorted(inventory, key=lambda cell: (len(inventory[cell]), cell))
    used_test_blocks: set[int] = set()
    folds: list[dict] = []
    feasibility_search = []
    for fold in range(config["folds"]):
        candidates = [
            _ordered(
                [block for block in inventory[cell] if block not in used_test_blocks],
                config["name"], fold, cell,
            )
            for cell in cells
        ]
        found = None
        checked = 0
        for combination in itertools.product(*candidates):
            checked += 1
            test_blocks = set(combination)
            guard_blocks = {
                adjacent
                for block in test_blocks
                for adjacent in (block - 1, block + 1)
            } - test_blocks
            counts = []
            valid = True
            for cell in sorted(inventory):
                blocks = inventory[cell]
                test_count = sum(count for block, count in blocks.items() if block in test_blocks)
                train_count = sum(
                    count
                    for block, count in blocks.items()
                    if block not in test_blocks and block not in guard_blocks
                )
                counts.append(
                    {
                        "power_dbm": cell[0],
                        "binary_class": "positive" if cell[1] else "negative",
                        "train_available": train_count,
                        "test_available": test_count,
                    }
                )
                valid &= train_count > 0 and test_count > 0
            if valid:
                found = {
                    "fold": fold,
                    "test_blocks": sorted(test_blocks),
                    "guard_blocks": sorted(guard_blocks),
                    "cell_available_counts": counts,
                    "combinations_checked": checked,
                }
                break
        feasibility_search.append({"fold": fold, "combinations_checked": checked, "found": found is not None})
        if found is None:
            return [], {
                "feasible": False,
                "reason": "no block-disjoint train/test allocation retains every power/class cell after guard",
                "inventory": inventory_records(inventory),
                "search": feasibility_search,
            }
        folds.append(found)
        used_test_blocks.update(found["test_blocks"])
    return folds, {
        "feasible": True,
        "inventory": inventory_records(inventory),
        "search": feasibility_search,
        "test_blocks_disjoint_across_folds": len(used_test_blocks)
        == sum(len(item["test_blocks"]) for item in folds),
    }


def inventory_records(inventory: dict[tuple[int, int], dict[int, int]]) -> list[dict]:
    return [
        {
            "power_dbm": cell[0],
            "binary_class": "positive" if cell[1] else "negative",
            "sample_count": sum(blocks.values()),
            "block_count": len(blocks),
            "blocks": [
                {"group_key": block, "sample_count": count}
                for block, count in sorted(blocks.items())
            ],
        }
        for cell, blocks in sorted(inventory.items())
    ]


def deterministic_balance(
    rows: list[dict], raw_roles: list[str], config: dict, fold: int
) -> tuple[list[bool], list[int | None], list[str]]:
    selected = [False] * len(rows)
    ranks: list[int | None] = [None] * len(rows)
    digests = [""] * len(rows)
    for role in ("train", "test"):
        for power in config["powers_dbm"]:
            positions_by_class = {}
            for positive in (False, True):
                positions = [
                    position
                    for position, row in enumerate(rows)
                    if raw_roles[position] == role
                    and row["transmit_power_dbm"] == power
                    and is_positive(row) == positive
                ]
                ordered = sorted(
                    positions,
                    key=lambda position: deterministic_digest(
                        config["name"], fold, role, int(power), int(positive), rows[position]["sample_index"]
                    ),
                )
                positions_by_class[positive] = ordered
            count = min(len(positions_by_class[False]), len(positions_by_class[True]))
            if count == 0:
                raise ValueError(f"empty balanced cell: {config['name']} fold={fold} role={role} power={power}")
            for positive in (False, True):
                for rank, position in enumerate(positions_by_class[positive]):
                    digest = deterministic_digest(
                        config["name"], fold, role, int(power), int(positive), rows[position]["sample_index"]
                    )
                    ranks[position] = rank
                    digests[position] = digest
                    selected[position] = rank < count
    return selected, ranks, digests


def build_split_rows(rows: list[dict], config: dict, folds: list[dict]) -> list[dict]:
    eligible = eligible_rows(rows, config)
    output = []
    for fold_info in folds:
        fold = fold_info["fold"]
        test_blocks = set(fold_info["test_blocks"])
        guard_blocks = set(fold_info["guard_blocks"])
        raw_roles = []
        for row in eligible:
            block = row["sample_index"] // config["block_size"]
            role = "test" if block in test_blocks else "guard" if block in guard_blocks else "train"
            raw_roles.append(role)
        selected, ranks, digests = deterministic_balance(eligible, raw_roles, config, fold)
        for position, row in enumerate(eligible):
            raw_role = raw_roles[position]
            final_role = raw_role if raw_role == "guard" or selected[position] else "balance_excluded"
            output.append(
                {
                    "evaluation": config["name"],
                    "fold": fold,
                    "block_size": config["block_size"],
                    "sample_index": row["sample_index"],
                    "group_key": row["sample_index"] // config["block_size"],
                    "area": row["area"],
                    "transmit_power_dbm": int(row["transmit_power_dbm"]),
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "binary_class": "positive" if is_positive(row) else "negative",
                    "raw_role": raw_role,
                    "selected": str(bool(selected[position])).lower(),
                    "final_role": final_role,
                    "selection_rank": "" if ranks[position] is None else ranks[position],
                    "selection_sha256": digests[position],
                }
            )
    return output


def build_exclusion_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        if row["area"] != 1:
            reason = "PRIMARY_EXCLUDED_AREA_NOT_1"
        elif row["transmit_power_dbm"] not in PRIMARY["powers_dbm"]:
            reason = "PRIMARY_EXCLUDED_POWER_NOT_30_OR_40"
        else:
            continue
        output.append(
            {
                "sample_index": row["sample_index"],
                "area": row["area"],
                "transmit_power_dbm": int(row["transmit_power_dbm"]),
                "class_id": row["class_id"],
                "class_name": row["class_name"],
                "binary_class": "positive" if is_positive(row) else "negative",
                "reason": reason,
            }
        )
    return output


def label_file_hashes(split_root: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(split_root.glob("*_crpa_*.txt"))
    }


def build_design(split_root: Path, output: Path) -> dict:
    rows = load_label_rows(split_root)
    output.mkdir(parents=True, exist_ok=True)
    solutions = {}
    all_split_rows = []
    for config in CONFIGS:
        folds, audit = solve_test_blocks(rows, config)
        solutions[config["name"]] = {
            "config": {
                **config,
                "powers_dbm": [int(value) for value in config["powers_dbm"]],
                "group_key": f"floor(sample_index / {config['block_size']})",
            },
            "audit": audit,
            "folds": folds,
        }
        if audit["feasible"]:
            all_split_rows.extend(build_split_rows(rows, config, folds))
    if not solutions["primary"]["audit"]["feasible"]:
        raise SystemExit("LABEL_ONLY_PRIMARY_SPLIT_INFEASIBLE")

    split_fields = [
        "evaluation", "fold", "block_size", "sample_index", "group_key", "area",
        "transmit_power_dbm", "class_id", "class_name", "binary_class", "raw_role",
        "selected", "final_role", "selection_rank", "selection_sha256",
    ]
    exclusion_fields = [
        "sample_index", "area", "transmit_power_dbm", "class_id", "class_name",
        "binary_class", "reason",
    ]
    write_csv(output / "split_manifest.csv", split_fields, all_split_rows)
    exclusions = build_exclusion_rows(rows)
    write_csv(output / "exclusion_manifest.csv", exclusion_fields, exclusions)

    primary_unique = {row["sample_index"] for row in all_split_rows if row["evaluation"] == "primary"}
    excluded_unique = {row["sample_index"] for row in exclusions}
    design = {
        "status": "LABEL_ONLY_DESIGN_FREEZE_PRE_FEATURE",
        "seed": SEED,
        "source_commit": "df98657cf1a50814e169bcd4f91a3b555c9025c0",
        "label_file_sha256": label_file_hashes(split_root),
        "label_row_count": len(rows),
        "label_inputs_used": ["sample_index", "Area", "transmit_power", "class label for binary cell", "block occupancy"],
        "iq_feature_bytes_read": 0,
        "stage0b_class_spatial_results_used": False,
        "primary": solutions["primary"],
        "sensitivity_a": solutions["sensitivity_a"],
        "sensitivity_b": solutions["sensitivity_b"],
        "block_size_2048": {"executable": False, "reason": "support cells insufficient by frozen contract"},
        "coverage": {
            "primary_unique_eligible_indices": len(primary_unique),
            "primary_excluded_unique_indices": len(excluded_unique),
            "all_label_indices_covered_by_primary_or_exclusion": len(primary_unique | excluded_unique) == len(rows),
        },
        "balancing": {
            "method": "minority count within each fold/role/power; majority chosen by ascending SHA-256 rank",
            "random_resampling": False,
        },
        "numerical_invariance": {
            "rtol": 1e-10,
            "atol": 1e-12,
            "gains": [0.1, 1.0, 3.7, 10.0],
            "phases_radians": [0.0, 0.37, 1.123, 2.9],
            "all_combinations_required": True,
        },
        "models": {
            "M0": "mean and per-channel received log-power; no transmit-power metadata",
            "M1": "channel-0 amplitude statistics and 16-bin normalized spectrum; no absolute phase",
            "M2": "calibration-free eigenvalue fractions/effective rank/lambda1/trace/sorted coherence magnitudes",
            "M3": "M2 plus six complex-coherence real/imaginary pairs; diagnostic only",
        },
        "pipeline": {
            "scaler": "StandardScaler fit on train fold only",
            "classifier": {
                "type": "LogisticRegression",
                "penalty": "l2", "C": 1.0, "solver": "liblinear",
                "max_iter": 2000, "random_state": SEED,
            },
            "threshold": 0.5,
            "test_label_threshold_use": False,
            "bootstrap_replicates": 2000,
            "bootstrap_unit": "test group_key block",
        },
        "forbidden_claims": [
            "clean-versus-spoof detector success", "general GNSS spoofing detector success",
            "READY_FOR_WCL", "recording-independent generalization",
        ],
    }
    write_json(output / "design_freeze.json", design)
    return design
