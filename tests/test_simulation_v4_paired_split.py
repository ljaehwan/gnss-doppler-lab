import copy
import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_simulation_v4_paired_split.py"
CONFIG = ROOT / "configs" / "experiments" / "simulation_v4_paired_split_v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("simulation_v4_paired_split", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frozen_config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def labeled_rows(config):
    rows = []
    for pair in config["pairs"]:
        group = pair["paired_group_id"]
        rows.extend((
            {
                "run_id": f"{group}-normal",
                "source_fingerprint": f"source-{group}-normal",
                "label": "normal",
                "paired_group_id": group,
                "scenario_kind": "steady_normal",
                "is_spoofing": "0",
                "feature": "0.1",
            },
            {
                "run_id": f"{group}-spoof",
                "source_fingerprint": f"source-{group}-spoof",
                "label": "normal",
                "paired_group_id": group,
                "scenario_kind": "carryoff_spoof",
                "is_spoofing": "0",
                "feature": "0.2",
            },
            {
                "run_id": f"{group}-spoof",
                "source_fingerprint": f"source-{group}-spoof",
                "label": "spoofing",
                "paired_group_id": group,
                "scenario_kind": "carryoff_spoof",
                "is_spoofing": "1",
                "feature": "0.3",
            },
        ))
    return rows


def write_dataset(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_frozen_config_is_preassigned_by_atomic_pair():
    module = load_module()
    config = frozen_config()
    fingerprints = module.validate_config(config)
    assert len(fingerprints) == 12
    assert len(set(fingerprints.values())) == 12
    assert {
        partition: sum(pair["split"] == partition for pair in config["pairs"])
        for partition in module.PARTITIONS
    } == {"train": 6, "validation": 3, "test": 3}
    assert set(config["calibration_exclusions"]["paired_group_ids"]) == {
        "iv-s-denver-a",
        "iv-s-seoul-b",
        "iv-d-tokyo-straight",
        "iv-d-london-circle",
        "iv-d-sydney-sweep",
    }


def test_config_rejects_duplicate_seed_and_motion_coverage_drift():
    module = load_module()
    config = frozen_config()

    duplicate_seed = copy.deepcopy(config)
    duplicate_seed["pairs"][1]["receiver_seed"] = duplicate_seed["pairs"][0]["receiver_seed"]
    with pytest.raises(ValueError, match="receiver seeds must be unique"):
        module.validate_config(duplicate_seed, verify_prior_source=False)

    no_train_circle = copy.deepcopy(config)
    pair = no_train_circle["pairs"][4]
    pair["motion"] = {"kind": "straight", "speed_mps": 7.0, "heading_deg": 10.0}
    with pytest.raises(ValueError, match="train motion coverage mismatch"):
        module.validate_config(no_train_circle, verify_prior_source=False)


def test_plan_writes_disjoint_group_and_scenario_catalogs(tmp_path):
    module = load_module()
    output = tmp_path / "plan"
    manifest_path = module.create_plan(CONFIG, output)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["leakage_checks"]["all_partition_intersections_empty"] is True
    assert manifest["test_release"]["status"] == "locked"
    assert manifest["partitions"]["train"]["pair_count"] == 6
    assert manifest["partitions"]["validation"]["pair_count"] == 3
    assert manifest["partitions"]["test"]["pair_count"] == 3
    assert Path(manifest["partitions"]["test"]["path"]).name == "test_groups.locked.txt"

    with (output / "scenario_catalog.csv").open(newline="") as stream:
        scenarios = list(csv.DictReader(stream))
    assert len(scenarios) == 24
    by_group = {}
    for row in scenarios:
        by_group.setdefault(row["paired_group_id"], set()).add(row["scenario_kind"])
    assert all(kinds == {"steady_normal", "carryoff_spoof"} for kinds in by_group.values())


def test_completed_dataset_is_pair_atomic_and_partition_disjoint():
    module = load_module()
    config = frozen_config()
    rows = labeled_rows(config)
    result = module.validate_dataset_rows(rows, config)
    assert result["row_counts"] == {"train": 18, "validation": 9, "test": 9}
    assert result["paired_group_count"] == 12
    assert result["pair_atomic"] is True
    assert result["source_fingerprint_partition_overlap"] == []


def test_dataset_rejects_source_and_pair_member_leakage():
    module = load_module()
    config = frozen_config()
    rows = labeled_rows(config)
    rows[18]["source_fingerprint"] = rows[0]["source_fingerprint"]
    with pytest.raises(ValueError, match="source fingerprint leaks"):
        module.validate_dataset_rows(rows, config)

    missing_spoof = [
        row for row in labeled_rows(config)
        if not (
            row["paired_group_id"] == "pv1-pair-012"
            and row["scenario_kind"] == "carryoff_spoof"
        )
    ]
    with pytest.raises(ValueError, match="both planned scenario kinds"):
        module.validate_dataset_rows(missing_spoof, config)


def test_materialization_exports_train_validation_but_locks_test(tmp_path):
    module = load_module()
    config = frozen_config()
    dataset = tmp_path / "labeled.csv"
    write_dataset(dataset, labeled_rows(config))
    plan = tmp_path / "plan"
    module.create_plan(CONFIG, plan)
    manifest_path = module.materialize_dataset(dataset, CONFIG, plan)
    manifest = json.loads(manifest_path.read_text())
    partitions = manifest["partitions"]
    assert Path(partitions["train"]["path"]).is_file()
    assert Path(partitions["validation"]["path"]).is_file()
    assert partitions["test"] == {
        "path": None,
        "row_count": 9,
        "sha256": None,
        "status": "locked",
    }
    assert not (plan / "dataset_partitions" / "test.csv").exists()


def test_test_release_requires_hash_pinned_model_freeze(tmp_path):
    module = load_module()
    config = frozen_config()
    dataset = tmp_path / "labeled.csv"
    write_dataset(dataset, labeled_rows(config))
    plan = tmp_path / "plan"
    split_manifest = module.create_plan(CONFIG, plan)

    with pytest.raises(ValueError, match="requires --freeze-manifest"):
        module.materialize_dataset(dataset, CONFIG, plan, release_test=True)
    assert not (plan / "dataset_partitions").exists()

    for name in ("model", "preprocessing", "thresholds"):
        (tmp_path / f"{name}.bin").write_bytes(name.encode())
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "schema": "gnss-doppler-lab.simulation-v4-model-freeze",
        "split_config_sha256": module._sha256(CONFIG),
        "split_manifest_sha256": module._sha256(split_manifest),
        "artifacts": {
            name: {
                "path": str(tmp_path / f"{name}.bin"),
                "sha256": module._sha256(tmp_path / f"{name}.bin"),
            }
            for name in ("model", "preprocessing", "thresholds")
        },
    }))
    released_manifest = module.materialize_dataset(
        dataset,
        CONFIG,
        plan,
        release_test=True,
        freeze_manifest=freeze,
    )
    released = json.loads(released_manifest.read_text())
    assert released["test_release"]["released"] is True
    assert released["partitions"]["test"]["status"] == "released"
    assert Path(released["partitions"]["test"]["path"]).is_file()
