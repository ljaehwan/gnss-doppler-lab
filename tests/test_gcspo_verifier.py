import json
from pathlib import Path

import pytest


def test_final_required_set_matches_frozen_contract_exactly():
    from gnss_doppler_lab.gcspo_verify import FINAL_REQUIRED

    assert FINAL_REQUIRED == {
        "README.md", "config.json", "preregistration.json", "source_commit.json", "data_inventory.json",
        "normal_model_summary.json", "thresholds.json", "scenario_metrics.csv", "ablation_metrics.csv",
        "per_epoch_scores.csv", "shared_state_estimates.csv", "external_static_fpr.csv",
        "relation_destruction_metrics.json", "physical_controls.json", "bootstrap_intervals.csv",
        "final_verdict.json", "access_ledger.jsonl", "artifact_manifest_sha256.json",
        "verifier_report.json", "fresh_clone_verifier_report.json",
        "clean_only_report.json", "clean_ablation_report.json", "clean_a5_report.json",
        "clean_b0_report.json", "clean_reproduction_evidence.json",
        "reproduction_run_1.json", "reproduction_run_2.json", "protected_capabilities.json",
    }


def test_delivered_runners_use_contract_allowed_implementation_manifest_name():
    root = Path(__file__).parents[1]
    for relative in ("scripts/run_gcspo_stage0.py", "scripts/verify_gcspo_stage0.py"):
        text = (root / relative).read_text()
        assert '"implementation_freeze.json"' not in text
        assert '"implementation_manifest.json"' in text



def test_clean_ready_report_does_not_require_final_manifest(tmp_path):
    import importlib.util
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("gcspo_verify_runner", root / "scripts/verify_gcspo_stage0.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._artifact_manifest_sha(tmp_path, "clean-ready") is None
    with pytest.raises(FileNotFoundError):
        module._artifact_manifest_sha(tmp_path, "final")

def test_clean_ready_verifier_requires_all_methods_and_zero_access(tmp_path, monkeypatch):
    import gnss_doppler_lab.gcspo_verify as verify

    monkeypatch.setattr(verify, "verify_reproduction_manifests", lambda _root: {"status": "PASS"})
    verify_clean_ready = verify.verify_clean_ready

    (tmp_path / "clean_only_report.json").write_text(json.dumps({
        "run_status": "CLEAN_ONLY_PASS", "protected_attack_rows_read": False,
        "attack_access_count": 0, "all_methods": ["A0", "A1", "A2", "A3", "A4", "A5", "Full"],
        "deterministic_rerun": "PASS",
    }))
    (tmp_path / "preflight_report.json").write_text(json.dumps({
        "overall_status": "PASS", "attack_access_count": 0,
        "synthetic_physical_recovery": {
            "overall_status": "PASS", "var_transfer_application_count": 1,
            "maximum_scaled_state_error": 1e-9,
            "tolerance": {"maximum_scaled_state_error": 1e-5},
        },
    }))
    canonical = (Path(__file__).parents[1] /
                 "artifacts/gcspo_stage0_successor_launch" /
                 "gcspo-stage0-successor-launch-77e586dfdb50a008ed2f0b31052e33bb700e191641e7ffbb4845860df15cf48e")
    for name in ("reproduction_run_1.json", "reproduction_run_2.json",
                 "clean_a5_report.json", "clean_b0_report.json", "thresholds.json"):
        (tmp_path / name).write_bytes((canonical / name).read_bytes())
    assert verify_clean_ready(tmp_path)["status"] == "PASS"
    doc = json.loads((tmp_path / "clean_only_report.json").read_text())
    doc["all_methods"].remove("A5")
    (tmp_path / "clean_only_report.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="methods"):
        verify_clean_ready(tmp_path)


def test_final_verifier_rejects_manifest_or_access_count_mismatch(tmp_path):
    from gnss_doppler_lab.gcspo_artifacts import build_artifact_manifest, canonical_write_json
    from gnss_doppler_lab.gcspo_verify import verify_final

    (tmp_path / "final_verdict.json").write_text(json.dumps({"verdict": "NO_GO_PHYSICAL_HYPOTHESIS", "protected_run_count": 1}))
    (tmp_path / "access_ledger.jsonl").write_text(json.dumps({"operation": "OPEN", "scenario": "DS3"}) + "\n")
    canonical_write_json(tmp_path / "artifact_manifest_sha256.json", build_artifact_manifest(tmp_path))
    with pytest.raises(ValueError, match="access ledger"):
        verify_final(tmp_path)
    (tmp_path / "access_ledger.jsonl").write_text("")
    with pytest.raises(ValueError, match="access ledger"):
        verify_final(tmp_path)


def test_access_ledger_reconstruction_rejects_unpaired_or_tampered_records():
    import hashlib
    from gnss_doppler_lab.gcspo_verify import validate_access_ledger

    def seal(payload, sequence, previous):
        row = {**payload, "sequence": sequence, "previous_record_sha256": previous}
        row["record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return row
    common = {"actor": "gnss_doppler_lab.gcspo.AccessGate", "canonical_path": "/sealed/ds3.mat",
              "scenario": "DS3", "phase": "transition", "purpose": "tracking",
              "authorization_sha": "a" * 40, "run_identity": "a" * 40, "access_counter": 1,
              "expected_sha256": "b" * 64, "expected_size": 7, "byte_range": "[0,7)",
              "row_range": "ALL_ROWS_IN_BYTE_RANGE", "operation": "READ_HDF5", "kind": "MAT",
              "identity_source": "AUTHENTICATED_MANIFEST:/sealed/manifest.json"}
    pre = seal({**common, "record_type": "PRE", "outcome": "OPEN_PENDING",
                "timestamp_utc": "2026-08-12T01:02:03.000001Z"}, 1, "0" * 64)
    post = seal({**common, "record_type": "POST", "outcome": "SUCCESS",
                 "timestamp_utc": "2026-08-12T01:02:03.000002Z",
                 "observed_sha256": "b" * 64, "observed_size": 7}, 2, pre["record_sha256"] )
    assert validate_access_ledger([pre, post])["successful_files"] == 1
    with pytest.raises(ValueError): validate_access_ledger([pre])
    post["observed_size"] = 8
    with pytest.raises(ValueError): validate_access_ledger([pre, post])


def test_access_ledger_rejects_missing_non_utc_reordered_counter_run_path_and_forged_minimal():
    import hashlib
    from gnss_doppler_lab.gcspo_verify import validate_access_ledger

    def chain(payloads):
        result, previous = [], "0" * 64
        for sequence, payload in enumerate(payloads, 1):
            row = {**payload, "sequence": sequence, "previous_record_sha256": previous}
            row["record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            result.append(row); previous = row["record_sha256"]
        return result

    common = {"actor": "gnss_doppler_lab.gcspo.AccessGate", "canonical_path": "/sealed/ds3.mat",
              "scenario": "DS3", "phase": "transition", "purpose": "tracking",
              "authorization_sha": "a" * 40, "run_identity": "a" * 40, "access_counter": 1,
              "expected_sha256": "b" * 64, "expected_size": 7, "byte_range": "[0,7)",
              "row_range": "ALL_ROWS_IN_BYTE_RANGE", "operation": "READ_HDF5", "kind": "MAT",
              "identity_source": "AUTHENTICATED_MANIFEST:/sealed/manifest.json"}
    pre = {**common, "record_type": "PRE", "outcome": "OPEN_PENDING",
           "timestamp_utc": "2026-08-12T01:02:03.000001Z"}
    post = {**common, "record_type": "POST", "outcome": "SUCCESS",
            "timestamp_utc": "2026-08-12T01:02:03.000002Z",
            "observed_sha256": "b" * 64, "observed_size": 7}
    assert validate_access_ledger(chain([pre, post]))["successful_files"] == 1
    mutations = [
        ({key: value for key, value in pre.items() if key != "timestamp_utc"}, post),
        ({**pre, "timestamp_utc": "2026-08-12T10:02:03.000001+09:00"}, post),
        ({**pre, "timestamp_utc": "2026-08-12T01:02:04.000001Z"}, post),
        (pre, {**post, "access_counter": 2}),
        (pre, {**post, "run_identity": "c" * 40}),
        (pre, {**post, "canonical_path": "/sealed/other.mat"}),
    ]
    for first, second in mutations:
        with pytest.raises(ValueError):
            validate_access_ledger(chain([first, second]))
    minimal = {key: pre[key] for key in ("record_type", "outcome", "canonical_path", "scenario", "phase",
                                          "purpose", "authorization_sha", "expected_sha256", "expected_size",
                                          "byte_range", "row_range", "operation", "kind", "identity_source")}
    minimal_post = {**minimal, "record_type": "POST", "outcome": "SUCCESS",
                    "observed_sha256": "b" * 64, "observed_size": 7}
    with pytest.raises(ValueError):
        validate_access_ledger(chain([minimal, minimal_post]))


def test_verifier_recomputes_gates_and_rejects_placeholder_or_semantic_mutation():
    from gnss_doppler_lab.gcspo_verify import verify_evidence_document

    evidence = {
        "clean_holdout_fpr": .01, "external_pre_fpr": {"DS3": .04},
        "incremental_lcb": {"Full-A1": .1, "Full-A2": .1},
        "destruction": {
            "policy": {}, "required_available_scenarios": ["DS3", "DS7"],
            "scenario_results": {
                scenario: {"status": "AVAILABLE", "mandatory": True, "lcb": .1,
                           "median_relative_loss": .3, "replicates": 2000,
                           "contrast": "PAIRED_SCORE_LOSS_NOT_BINARY_PAUC"}
                for scenario in ("DS3", "DS7")
            } | {"DS4": {"status": "LIMITED_TRANSITION_ONLY", "mandatory": False},
                 "DS8": {"status": "UNAVAILABLE", "mandatory": False}},
        },
        "persistence": {"DS3": {"ratio": .6, "delay_s": 2.}, "DS7_DS8": {"ratio": .6, "delay_s": 2.}},
        "controls": [{"id": "COMMON_GAIN", "persistent_alarm_ratio": 0., "max_consecutive_alarms": 0},
                     {"id": "CLOCK_DRIFT", "specificity_ratio": 0.}],
        "shared": {"full_pauc": .8, "a5_pauc": .8, "full_median_edf": 2., "a5_median_edf": 3.},
    }
    from gnss_doppler_lab.gcspo_statistics import compute_scientific_gates
    gates = compute_scientific_gates(evidence)
    assert verify_evidence_document({"scientific_status": "VALID_SCIENCE", "evidence": evidence, "gates": gates,
                                     "verdict": "GO_FOR_NEURAL_STAGE1", "protected_run_count": 1})["verdict"] == "GO_FOR_NEURAL_STAGE1"
    gates[1]["status"] = "FAIL"
    with pytest.raises(ValueError, match="recomputed"): verify_evidence_document({"scientific_status": "VALID_SCIENCE",
        "evidence": evidence, "gates": gates, "verdict": "NO_GO_PHYSICAL_HYPOTHESIS", "protected_run_count": 1})
    evidence["incremental_lcb"]["Full-A1"] = "NA"
    with pytest.raises((ValueError, TypeError)): verify_evidence_document({"scientific_status": "VALID_SCIENCE",
        "evidence": evidence, "gates": gates, "verdict": "NO_GO_PHYSICAL_HYPOTHESIS", "protected_run_count": 1})


def test_fresh_clone_helper_clones_local_remote_and_checks_exact_commit(tmp_path):
    import subprocess
    from gnss_doppler_lab.gcspo_fresh_clone import clone_exact

    source = tmp_path / "source"; source.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    (source / "tracked.txt").write_text("tracked\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source, check=True, text=True, capture_output=True).stdout.strip()
    (source / "untracked-secret.txt").write_text("must not leak\n")
    checkout = clone_exact(str(source), commit, tmp_path / "clone")
    assert checkout["head"] == commit and (tmp_path / "clone" / "tracked.txt").is_file()
    assert not (tmp_path / "clone" / "untracked-secret.txt").exists()
