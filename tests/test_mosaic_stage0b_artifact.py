import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"


def test_stage0a_source_sha_consistency_and_clean_only_contract():
    binding = json.loads((ART / "raw_source_binding.json").read_text())
    for dataset in ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic"):
        assert binding[dataset]["stage0a_full_sha256"] == binding[dataset]["expected_sha256"]
        assert binding[dataset]["stat_identity_match"] is True
    config = json.loads((ART / "config.json").read_text())
    assert config["attack_data_used"] is False
    assert config["synthetic_injection_performed"] is False
    assert config["constant_plus_one_fallback"] is False


def test_multi_prn_coverage_and_fresh_clone_artifact_verifier():
    coverage = json.loads((ART / "coverage_summary.json").read_text())
    assert coverage["total_validated_prns"] == 10
    assert coverage["total_validated_bits"] == 6000
    assert all(item["validated_prns"] >= 4 for item in coverage["datasets"].values())
    path = ROOT / "scripts/verify_mosaic_stage0b_r0.py"
    spec = importlib.util.spec_from_file_location("verify_mosaic_stage0b_r0", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()
