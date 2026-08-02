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
