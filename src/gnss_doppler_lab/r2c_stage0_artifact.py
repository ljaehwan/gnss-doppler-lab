"""Shared, frozen Stage-0-fix artifact schema (structure only, no decisions)."""
from pathlib import Path

TOP_LEVEL_FILES = frozenset({
    "README.md", "config.json", "provenance.json", "input_validity.json",
    "b0_interface_validation.json", "time_geometry_validation.json",
    "training_summary.json", "thresholds.json", "scenario_metrics.csv",
    "ablation_metrics.csv", "per_epoch_scores.csv", "bootstrap_comparisons.json",
    "gain_invariance.json", "phase_invariance.json", "noise_control.json",
    "multipath_control.json", "second_source_injection.json",
    "relation_destruction.json", "decision.json", "verification.json", "hashes.json",
})
TOP_LEVEL_DIRECTORIES = frozenset({"plots"})
HASH_EXCLUDED = frozenset({"hashes.json"})
PRESERVED_TREE = "53f7cdab9ac324c08b94a5d6f38f6a32d3ec16b7"

FIX_SOURCE_FILES = (
    "pyproject.toml",
    "configs/r2c_gnss_stage0_fix.json",
    "configs/detectors/texbat_btail_gate_v1.json",
    "scripts/run_r2c_gnss_stage0_fix.py",
    "scripts/verify_r2c_gnss_stage0_fix.py",
    "scripts/reconstruct_r2c_time_geometry.py",
    "scripts/validate_r2c_stage0_pre_campaign.py",
    "scripts/train_prn_node_gru.py",
    "scripts/score_texbat_prn_node_gru.py",
    "scripts/eval_btail_support_gate.py",
    "src/gnss_doppler_lab/r2c_stage0_fix.py",
    "src/gnss_doppler_lab/r2c_stage0_artifact.py",
    "src/gnss_doppler_lab/tracking_feature_dataset.py",
    "src/gnss_doppler_lab/normal_multi_prn_dataset.py",
    "tests/test_r2c_stage0_fix.py",
    "tests/test_r2c_stage0_correction1.py",
    "tests/test_r2c_stage0_correction2.py",
    "tests/test_r2c_stage0_correction4.py",
    "tests/test_r2c_stage0_correction5.py",
)

def expected_hash_keys(root: Path):
    return frozenset(str(path.relative_to(root)) for path in root.rglob("*")
                     if path.is_file() and path.name not in HASH_EXCLUDED)
