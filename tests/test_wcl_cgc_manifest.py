import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "audit_wcl_cgc_manifest",
    ROOT / "scripts" / "audit_wcl_cgc_manifest.py",
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_current_manifest_has_complete_tracked_core_map():
    path = ROOT / "configs" / "paper" / "wcl_cgc_v1_manifest.json"
    result = module.audit_manifest(json.loads(path.read_text(encoding="utf-8")))
    assert result["status"] == "pass"
    assert result["missing_required_paths"] == []
    assert result["required_paths_not_git_tracked"] == []
    assert result["classification_counts"]["paper_core"] >= 4
    assert result["classification_counts"]["negative_or_failed"] >= 1
    assert result["classification_counts"]["uncommitted_review"] == 11


def test_rejects_path_escape():
    try:
        module._relative_path("../outside")
    except ValueError:
        pass
    else:
        raise AssertionError("path escape was accepted")
