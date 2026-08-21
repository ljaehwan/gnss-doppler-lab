import ast
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.acquisition_surface import gps_l1ca_code
from gnss_doppler_lab.crid_control_joint_reference import (
    independent_ca,
    require_absolute_start,
    require_file_binding,
    solve_joint_system,
)


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"
MODULE = ROOT / "src/gnss_doppler_lab/crid_control_joint_reference.py"


def _system(seed=4):
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(100, 5)) + 1j * rng.normal(size=(100, 5))
    beta = rng.normal(size=5) + 1j * rng.normal(size=5)
    data = design @ beta
    return design.conj().T @ design, design.conj().T @ data, float(np.vdot(data, data).real), beta


def test_independent_ca_matches_committed_reference_codes():
    for prn in (3, 10, 13, 16, 19, 21, 24, 27, 30):
        assert np.array_equal(independent_ca(prn), gps_l1ca_code(prn))


def test_joint_coefficient_is_prn_permutation_invariant():
    gram, rhs, energy, beta = _system()
    baseline = solve_joint_system(gram, rhs, energy).coefficients
    permutation = np.array([3, 0, 4, 1, 2])
    permuted = solve_joint_system(gram[np.ix_(permutation, permutation)], rhs[permutation], energy).coefficients
    restored = np.empty(5, complex)
    restored[permutation] = permuted
    assert np.allclose(restored, baseline)
    assert np.allclose(baseline, beta)


def test_deleted_template_fails_exact_five_column_contract():
    gram, rhs, energy, _ = _system()
    with pytest.raises(ValueError, match="exactly five"):
        solve_joint_system(gram[:4, :4], rhs[:4], energy)


def test_duplicated_template_rank_deficiency_fails_closed():
    gram, rhs, energy, _ = _system()
    gram[:, 4] = gram[:, 3]
    gram[4, :] = gram[3, :]
    rhs[4] = rhs[3]
    with pytest.raises(ValueError, match="rank-deficient"):
        solve_joint_system(gram, rhs, energy)


def test_condition_limit_fails_closed():
    gram = np.diag([1.0, 1.0, 1.0, 1.0, 1e-7]).astype(complex)
    with pytest.raises(ValueError, match="condition number"):
        solve_joint_system(gram, np.ones(5, complex), 10.0)


def test_nav_source_and_output_hash_mutations_are_detected(tmp_path):
    payload = tmp_path / "bound.bin"
    payload.write_bytes(b"authenticated")
    expected = hashlib.sha256(payload.read_bytes()).hexdigest()
    require_file_binding(payload, payload.stat().st_size, expected)
    payload.write_bytes(b"AuthenticateD")
    with pytest.raises(ValueError, match="SHA-256"):
        require_file_binding(payload, payload.stat().st_size, expected)


def test_nav_sign_mutation_changes_bound_mapping_hash(tmp_path):
    mapping = tmp_path / "nav.csv"
    mapping.write_text("prn,bit_value_pm1\n21,-1\n")
    expected = hashlib.sha256(mapping.read_bytes()).hexdigest()
    mapping.write_text("prn,bit_value_pm1\n21,1\n")
    with pytest.raises(ValueError, match="SHA-256"):
        require_file_binding(mapping, mapping.stat().st_size, expected)


def test_raw_start_sample_mutation_is_detected():
    require_absolute_start(150275296, 150275296)
    with pytest.raises(ValueError, match="raw start mutation"):
        require_absolute_start(150275297, 150275296)


def test_generator_import_is_forbidden_and_absent():
    tree = ast.parse(MODULE.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any("crid_control_generator" in name for name in imports)
    assert "frozen_phase" not in MODULE.read_text()
    assert "amplitude_envelope" not in MODULE.read_text()


def test_attack_path_sentinel_absent():
    text = (MODULE.read_text() + (ROOT / "scripts/validate_crid_r3a_joint_reference.py").read_text()).lower()
    for token in ("ds1.bin", "ds2.bin", "ds3.bin", "ds4.bin", "ds7.bin", "os1.bin", "os2.bin", "os3.bin", "os4.bin"):
        assert token not in text


def test_committed_inventory_legacy_and_joint_results():
    legacy_summary = json.loads((ART / "legacy_reproduction_summary.json").read_text())
    joint_summary = json.loads((ART / "joint_reference_summary.json").read_text())
    assert (legacy_summary["passed"], legacy_summary["failed"]) == (171, 9)
    assert legacy_summary["same_nine_oak_prn21_failures"]
    assert legacy_summary["numeric_match_to_committed_r3"]
    assert (joint_summary["passed"], joint_summary["failed"]) == (180, 0)
    with (ART / "joint_reference_validation.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 180
    assert {row["domain"] for row in rows} == {"OAK", "TEX"}
    assert len({row["case_id"] for row in rows}) == 36
    inventory = {(row["domain"], row["case_id"]) for row in rows}
    assert all(sum(r["case_id"] == case and r["domain"] == domain for r in rows) == 5 for domain, case in inventory)
    assert all(row["status"] == "PASS" for row in rows)


def test_no_oak_prn21_exception_in_source_or_result():
    source = (ROOT / "scripts/validate_crid_r3a_joint_reference.py").read_text()
    assert "PRN 21 exception" not in source
    with (ART / "joint_reference_validation.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    oak21 = [row for row in rows if row["domain"] == "OAK" and int(row["prn"]) == 21]
    assert len(oak21) == 18 and all(row["status"] == "PASS" for row in oak21)


def test_artifact_byte_flip_is_detected(tmp_path):
    copied = tmp_path / "artifact"
    shutil.copytree(ART, copied)
    target = copied / "README.md"
    target.write_bytes(target.read_bytes() + b"X")
    result = subprocess.run(
        ["python3", str(ROOT / "scripts/verify_crid_r3a_estimand_repair.py"), "--artifact", str(copied)],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert '\"status\": \"FAIL\"' in result.stdout
