from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_cmte_a2_epochfix_exploratory.py"
    spec = importlib.util.spec_from_file_location("cmte_a2_epochfix_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verify_checksums_accepts_state_flat_and_packaged_nested_schemas(tmp_path):
    module = _load_runner()
    for nested in (False, True):
        root = tmp_path / ("nested" if nested else "flat")
        root.mkdir()
        (root / "payload.txt").write_text("frozen\n")
        digest = module.sha256(root / "payload.txt")
        document = {"algorithm": "sha256", "files": {"payload.txt": digest}} if nested else {"payload.txt": digest}
        (root / "checksums.json").write_text(json.dumps(document))
        assert module.verify_checksums(root) == 1


def test_tracked_n_histogram_records_clean_and_scenario_distributions():
    module = _load_runner()
    epoch = module.pd.DataFrame({"tracked_prn_count": [3, 3, 4, 5, 5, 5]})
    rows = module.tracked_n_histogram("cleanStatic_test", epoch)
    assert [(row["N"], row["epoch_count"]) for row in rows] == [(3, 2), (4, 1), (5, 3)]
    assert all(row["median_N"] == 4.5 for row in rows)
    assert all(row["min_N"] == 3 and row["max_N"] == 5 for row in rows)
