from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gnss_doppler_lab import gcspo_evaluate, gcspo_r1_runner as runner, gcspo_verify_artifacts
from gnss_doppler_lab.gcspo_r1_runner import (
    INVOCATION_ID,
    R1ScenarioAccessGate,
    _bind_scenario_manifest_paths,
    _build_scenario_access_plan,
    _verify_authorization_documents,
    _verify_frozen_science_manifest,
    _verify_identity_rows,
    _verify_no_runtime_shadows,
    _verify_preregistration,
    _write_full_manifest,
    claim_once,
    install_and_verify_adapter,
    verify_effective_preregistration,
    verify_r1_final_manifest,
    verify_zero_access_state,
)
from gnss_doppler_lab.gcspo_r1_support import (
    exact_b0_full_contrast_r1,
    integrate_protected_b0_r1,
    validate_protected_method_support_r1,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_r1_adapter_bindings():
    originals = (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    )
    yield
    (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    ) = originals


def test_install_and_verify_adapter_is_scoped_and_restores_every_binding():
    originals = (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    )
    with install_and_verify_adapter():
        assert gcspo_evaluate.integrate_protected_b0 is integrate_protected_b0_r1
        assert (gcspo_evaluate.validate_protected_method_support is
                validate_protected_method_support_r1)
        assert gcspo_evaluate.exact_b0_full_contrast is exact_b0_full_contrast_r1
        assert gcspo_verify_artifacts.exact_b0_full_contrast is exact_b0_full_contrast_r1
    assert (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
        gcspo_verify_artifacts.exact_b0_full_contrast,
    ) == originals


def test_zero_access_state_rejects_marker_ledger_or_verdict(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    marker = artifact / "marker.json"
    verify_zero_access_state(artifact, marker)

    marker.write_text("{}\n")
    with pytest.raises(ValueError, match="marker"):
        verify_zero_access_state(artifact, marker)
    marker.unlink()

    (artifact / "access_ledger.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="ledger"):
        verify_zero_access_state(artifact, marker)
    (artifact / "access_ledger.jsonl").unlink()

    (artifact / "final_verdict.json").write_text("{}\n")
    with pytest.raises(ValueError, match="verdict"):
        verify_zero_access_state(artifact, marker)
    (artifact / "final_verdict.json").unlink()

    marker.symlink_to("missing-target")
    with pytest.raises(ValueError, match="marker"):
        verify_zero_access_state(artifact, marker)
    marker.unlink()
    (artifact / "access_ledger.jsonl").symlink_to("missing-target")
    with pytest.raises(ValueError, match="ledger"):
        verify_zero_access_state(artifact, marker)


def test_claim_is_o_excl_and_bound_to_new_invocation(tmp_path):
    marker = tmp_path / "marker.json"
    info = tmp_path.stat()
    parent_binding = {"dev": info.st_dev, "ino": info.st_ino}
    first = claim_once(marker, wrapper_commit="a" * 40, target_commit="b" * 40,
                       parent_binding=parent_binding)
    assert first["invocation_id"] == INVOCATION_ID
    assert first["protected_run_count"] == 1
    with pytest.raises(FileExistsError):
        claim_once(marker, wrapper_commit="a" * 40, target_commit="b" * 40,
                   parent_binding=parent_binding)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True,
    ).stdout.strip()


def _identity(path: Path, relative: str) -> dict:
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _identity_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    _git(repo, "add", "a.txt", "b.txt")
    _git(repo, "commit", "-qm", "target")
    target = _git(repo, "rev-parse", "HEAD")
    rows = [_identity(repo / name, name) for name in ("a.txt", "b.txt")]
    return repo, target, rows


def test_identity_manifest_requires_exact_canonical_target_tree_set(tmp_path):
    repo, target, rows = _identity_repo(tmp_path)
    expected = ("a.txt", "b.txt")
    assert _verify_identity_rows(
        repo, target, rows, "runtime freeze", expected_paths=expected) == rows

    bad_path_rows = []
    for path in ("/a.txt", "../a.txt", "a/../a.txt", "a//txt", "./a.txt"):
        bad_path_rows.append([{**rows[0], "path": path}, rows[1]])
    mutations = [
        rows[:1],
        rows + [{"path": "c.txt", "sha256": "0" * 64, "size_bytes": 1}],
        [rows[0], rows[0]],
        [{"path": "a.txt", "sha256": rows[0]["sha256"]}, rows[1]],
        [{**rows[0], "sha256": "0" * 64}, rows[1]],
        *bad_path_rows,
    ]
    for mutation in mutations:
        with pytest.raises(ValueError):
            _verify_identity_rows(
                repo, target, mutation, "runtime freeze", expected_paths=expected)


def test_identity_manifest_rejects_git_symlink_blob(tmp_path):
    repo, _target, _rows = _identity_repo(tmp_path)
    (repo / "link.txt").symlink_to("a.txt")
    _git(repo, "add", "link.txt")
    _git(repo, "commit", "-qm", "symlink")
    target = _git(repo, "rev-parse", "HEAD")
    row = {
        "path": "link.txt",
        "sha256": hashlib.sha256(b"a.txt").hexdigest(),
        "size_bytes": 5,
    }
    with pytest.raises(ValueError, match="regular file"):
        _verify_identity_rows(
            repo, target, [row], "runtime freeze", expected_paths=("link.txt",))


def test_frozen_science_manifest_rejects_schema_and_baseline_tamper(monkeypatch):
    science = json.loads(
        (ROOT / runner.SCIENCE_MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    target = "target"
    mapping = {
        (target, runner.SCIENCE_MANIFEST_RELATIVE):
            json.dumps(science, allow_nan=False).encode("utf-8"),
        (target, f"{runner.ARTIFACT_RELATIVE}/original_preregistration.json"):
            (ROOT / runner.ARTIFACT_RELATIVE / "original_preregistration.json").read_bytes(),
        (target, f"{runner.ARTIFACT_RELATIVE}/config.json"):
            (ROOT / runner.ARTIFACT_RELATIVE / "config.json").read_bytes(),
        (target, runner._AUDIT_PATH): (ROOT / runner._AUDIT_PATH).read_bytes(),
        (target, runner._CONTROL_PATH): (ROOT / runner._CONTROL_PATH).read_bytes(),
        (target, runner.PREREG_RELATIVE):
            (ROOT / runner.PREREG_RELATIVE).read_bytes(),
        (target, runner.AMENDMENT_RELATIVE):
            (ROOT / runner.AMENDMENT_RELATIVE).read_bytes(),
        (target, runner.AUDIT_RECEIPT_RELATIVE):
            (ROOT / runner.AUDIT_RECEIPT_RELATIVE).read_bytes(),
    }
    for relative in (*runner.FROZEN_SCIENCE_PATHS, *runner.PIPELINE_BASELINE_PATHS):
        baseline = subprocess.run(
            ["git", "show", f"{runner.BASE_COMMIT}:{relative}"], cwd=ROOT,
            check=True, capture_output=True,
        ).stdout
        mapping[(runner.BASE_COMMIT, relative)] = baseline
        mapping[(target, relative)] = (ROOT / relative).read_bytes()

    def fake_blob(_root, commit, relative, _label):
        return mapping[(commit, relative)]

    monkeypatch.setattr(runner, "_git_regular_blob", fake_blob)
    monkeypatch.setattr(runner, "_verify_audit_git_transition", lambda *_args: None)
    assert _verify_frozen_science_manifest(ROOT, target) == science

    extra = {**science, "unexpected": True}
    mapping[(target, runner.SCIENCE_MANIFEST_RELATIVE)] = json.dumps(extra).encode()
    with pytest.raises(ValueError, match="exact schema"):
        _verify_frozen_science_manifest(ROOT, target)

    mapping[(target, runner.SCIENCE_MANIFEST_RELATIVE)] = json.dumps(science).encode()
    relative = runner.PIPELINE_BASELINE_PATHS[0]
    original = mapping[(runner.BASE_COMMIT, relative)]
    mapping[(runner.BASE_COMMIT, relative)] = b"tampered baseline"
    with pytest.raises(ValueError, match="pipeline baseline"):
        _verify_frozen_science_manifest(ROOT, target)
    mapping[(runner.BASE_COMMIT, relative)] = original

    relative = runner.FROZEN_SCIENCE_PATHS[0]
    mapping[(target, relative)] = b"tampered target"
    with pytest.raises(ValueError, match="frozen science hash changed"):
        _verify_frozen_science_manifest(ROOT, target)


def test_completion_preregistration_binds_all_join_dimensions():
    prereg = json.loads(
        (ROOT / runner.ARTIFACT_RELATIVE / "completion_preregistration.json")
        .read_text(encoding="utf-8"))
    _verify_preregistration(prereg)
    assert prereg["support_contract"]["join_keys"] == list(runner._BASE_JOIN_KEYS)
    effective = verify_effective_preregistration(ROOT / runner.ARTIFACT_RELATIVE)
    assert effective["support_contract"]["join_keys"] == list(runner.JOIN_KEYS)
    assert effective["status"] == "BASE_PREREGISTRATION_PLUS_POST_IMPLEMENTATION_AMENDMENT"
    tampered = json.loads(json.dumps(prereg))
    tampered["support_contract"]["join_keys"].remove("epoch_prn_support")
    with pytest.raises(ValueError, match="nested type/value"):
        _verify_preregistration(tampered)


def test_full_manifest_rejects_symlink_and_nonregular_member(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "regular.json").write_text("{}\n", encoding="utf-8")
    link = artifact / "alias.json"
    link.symlink_to("regular.json")
    with pytest.raises(ValueError, match="symlink"):
        _write_full_manifest(artifact)
    link.unlink()

    fifo = artifact / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="nonregular"):
        _write_full_manifest(artifact)


def test_actual_standalone_verifier_registers_r1_from_production_manifests(tmp_path):
    artifact = tmp_path / "r1"
    artifact.mkdir()
    for name in (
            "completion_preregistration.json", "completion_preregistration_amendment.json",
            "frozen_science_hashes.json", "physical_controls_audit.json",
            "physical_controls_audit_receipt.json"):
        source = ROOT / runner.ARTIFACT_RELATIVE / name
        (artifact / name).write_bytes(source.read_bytes())
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/verify_gcspo_stage0.py"),
            "--artifact-dir", str(artifact), "--r1-adapter-check",
        ],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "R1_ADAPTER_CHECK_PASS"


def test_actual_evaluator_entrypoint_registers_r1_before_loading_inputs(tmp_path):
    artifact = tmp_path / "r1"
    artifact.mkdir()
    for name in ("completion_preregistration.json",
                 "completion_preregistration_amendment.json"):
        source = ROOT / runner.ARTIFACT_RELATIVE / name
        (artifact / name).write_bytes(source.read_bytes())
    originals = (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
    )
    with pytest.raises(FileNotFoundError):
        gcspo_evaluate.run_one_shot(
            artifact_dir=artifact, repo_root=tmp_path, inventory={},
            gate=None, manifest_identities={}, clean_identities={},
            capabilities={},
        )
    assert (
        gcspo_evaluate.integrate_protected_b0,
        gcspo_evaluate.validate_protected_method_support,
        gcspo_evaluate.exact_b0_full_contrast,
    ) == originals


def _scenario_plan_fixture(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    roots, children, available, identities = {}, {}, {}, {}
    for index, scenario in enumerate(("DS3", "DS7"), start=1):
        directory = tmp_path / scenario
        directory.mkdir()
        root = directory / "manifest.json"
        child = directory / f"child-{index}.mat"
        root.write_bytes(f"root-{scenario}".encode())
        child.write_bytes(f"child-{scenario}".encode())
        root_info, child_info = root.stat(), child.stat()
        root_sha = hashlib.sha256(root.read_bytes()).hexdigest()
        child_sha = hashlib.sha256(child.read_bytes()).hexdigest()
        roots[scenario], children[scenario] = root, child
        identities[scenario] = {
            "scenario": scenario, "path": str(root), "sha256": root_sha,
            "size_bytes": root_info.st_size,
            "binding": {"dev": root_info.st_dev, "ino": root_info.st_ino},
        }
        available[scenario] = {
            "sidecar": {
                "scenario": scenario,
                "children": [{
                    "canonical_path": str(child), "sha256": child_sha,
                    "size_bytes": child_info.st_size, "scenario": scenario,
                    "purpose": "test", "_preclaim_dev": child_info.st_dev,
                    "_preclaim_ino": child_info.st_ino,
                }],
            },
        }
    return roots, children, {"available": available}, identities


def test_access_plan_rejects_child_path_or_object_reuse(tmp_path):
    _roots, children, capabilities, identities = _scenario_plan_fixture(tmp_path)
    duplicate = dict(capabilities["available"]["DS7"]["sidecar"]["children"][0])
    duplicate["canonical_path"] = str(children["DS3"])
    capabilities["available"]["DS7"]["sidecar"]["children"] = [duplicate]
    with pytest.raises(ValueError, match="reused"):
        _build_scenario_access_plan(capabilities, identities)

    _roots, children, capabilities, identities = _scenario_plan_fixture(tmp_path / "hardlink")
    replacement = children["DS7"]
    replacement.unlink()
    os.link(children["DS3"], replacement)
    info = replacement.stat()
    row = capabilities["available"]["DS7"]["sidecar"]["children"][0]
    row["canonical_path"] = str(replacement)
    row["size_bytes"] = info.st_size
    row["_preclaim_dev"], row["_preclaim_ino"] = info.st_dev, info.st_ino
    with pytest.raises(ValueError, match="reused|single-link"):
        _build_scenario_access_plan(capabilities, identities)


def test_scenario_bound_gate_rejects_relabel_before_consumer(tmp_path):
    roots, _children, capabilities, identities = _scenario_plan_fixture(tmp_path)
    plan = _build_scenario_access_plan(capabilities, identities)
    gate = R1ScenarioAccessGate(tmp_path / "ledger.jsonl", plan)
    gate.state = "VALID_FOR_PROTECTED_ACCESS"
    identity = identities["DS3"]
    gate.register_pinned(
        roots["DS3"], expected_sha256=identity["sha256"],
        expected_size=identity["size_bytes"], kind="RECEIVER_MANIFEST",
        preclaim_dev=identity["binding"]["dev"],
        preclaim_ino=identity["binding"]["ino"],
    )
    exposed = []
    with pytest.raises(PermissionError, match="different scenario"):
        gate.consume(
            roots["DS3"], scenario="DS7", phase="all", purpose="relabel test",
            consumer=lambda handle: exposed.append(handle.read()),
        )
    assert exposed == []


def test_runtime_shadow_scan_rejects_same_stem_pyc(monkeypatch, tmp_path):
    package = tmp_path / "src/gnss_doppler_lab"
    scripts = tmp_path / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (package / "runtime_module.py").write_text("VALUE = 1\n")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "runtime_module.cpython-311.pyc").write_bytes(b"shadow")
    monkeypatch.setattr(runner, "_git", lambda *_args: "")
    with pytest.raises(ValueError, match="import shadow"):
        _verify_no_runtime_shadows(
            tmp_path, ("src/gnss_doppler_lab/runtime_module.py",))


def _complete_final_tree(tmp_path):
    artifact = tmp_path / "final"
    artifact.mkdir()
    for relative in runner._FINAL_EXPECTED_FILES:
        if relative == runner.FINAL_MANIFEST_NAME:
            continue
        path = artifact / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (artifact / "implementation_manifest.json").write_text(
        json.dumps({"target_commit": "1" * 40}) + "\n")
    _write_full_manifest(artifact)
    return artifact


def test_final_manifest_and_failure_quarantine_support_held_directory_fd(tmp_path):
    artifact = _complete_final_tree(tmp_path)
    (artifact / runner.FINAL_MANIFEST_NAME).unlink()
    descriptor = os.open(artifact, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        bound = Path(f"/proc/self/fd/{descriptor}")
        _write_full_manifest(bound)
        verify_r1_final_manifest(bound)
    finally:
        os.close(descriptor)
    verify_r1_final_manifest(artifact)

    failed = tmp_path / "failed"
    failed.mkdir()
    (failed / "final_verdict.json").write_text("quarantine-me")
    failed_descriptor = os.open(failed, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        failed_bound = Path(f"/proc/self/fd/{failed_descriptor}")
        quarantined = runner._quarantine_failed_final_verdict_bound(failed_bound)
        assert quarantined.read_text() == "quarantine-me"
        assert not (failed_bound / "final_verdict.json").exists()
    finally:
        os.close(failed_descriptor)


def test_final_manifest_rejects_extra_regular_file_and_directory(tmp_path):
    artifact = _complete_final_tree(tmp_path)
    verify_r1_final_manifest(artifact)
    extra = artifact / "unexpected.bin"
    extra.write_bytes(b"x")
    with pytest.raises(ValueError, match="exact file/directory inventory"):
        verify_r1_final_manifest(artifact)
    extra.unlink()
    directory = artifact / "unexpected-directory"
    directory.mkdir()
    with pytest.raises(ValueError, match="exact file/directory inventory"):
        verify_r1_final_manifest(artifact)


def test_final_manifest_rejects_unbound_post_attestation_reports(tmp_path):
    artifact = _complete_final_tree(tmp_path)
    (artifact / "verifier_report.json").write_text("{}\n")
    with pytest.raises(ValueError, match="post-attestation report binding"):
        verify_r1_final_manifest(artifact)
    (artifact / "fresh_clone_verifier_report.json").write_text("{}\n")
    with pytest.raises(ValueError, match="post-attestation report binding"):
        verify_r1_final_manifest(artifact)


def test_claim_rejects_changed_parent_binding_before_marker_creation(tmp_path):
    marker = tmp_path / "marker.json"
    info = tmp_path.stat()
    with pytest.raises(ValueError, match="directory changed"):
        claim_once(
            marker, wrapper_commit="a" * 40, target_commit="b" * 40,
            parent_binding={"dev": info.st_dev, "ino": info.st_ino + 1},
        )
    assert not marker.exists()


def test_stable_read_rejects_same_size_in_place_mutation(monkeypatch, tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"a" * ((1 << 20) * 2 + 17))
    original_read = runner.os.read
    mutated = False

    def changed_read(descriptor, size):
        nonlocal mutated
        block = original_read(descriptor, size)
        if block and not mutated:
            mutated = True
            with path.open("r+b", buffering=0) as handle:
                handle.seek((1 << 20) + 10)
                handle.write(b"b")
                os.fsync(handle.fileno())
        return block

    monkeypatch.setattr(runner.os, "read", changed_read)
    with pytest.raises(ValueError, match="changed while read"):
        runner._read_regular_bytes(path, "same-size mutation fixture")


def test_runtime_shadow_scan_rejects_untracked_dependency_name(monkeypatch, tmp_path):
    package = tmp_path / "src/gnss_doppler_lab"
    scripts = tmp_path / "scripts"
    package.mkdir(parents=True)
    scripts.mkdir()
    (scripts / "numpy.py").write_text("raise RuntimeError('shadow')\n")
    monkeypatch.setattr(runner, "_git", lambda *_args: "?? scripts/numpy.py")
    with pytest.raises(ValueError, match="runtime shadow|import shadow"):
        _verify_no_runtime_shadows(tmp_path, ())


def test_runtime_shadow_scan_rejects_untracked_src_top_level_dependency(monkeypatch, tmp_path):
    (tmp_path / "src/gnss_doppler_lab").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/numpy.py").write_text("raise RuntimeError('shadow')\n")

    def fake_git(_root, *args):
        if args[:3] == ("status", "--porcelain=v1", "--ignored") and "src" in args:
            return "?? src/numpy.py"
        return ""

    monkeypatch.setattr(runner, "_git", fake_git)
    with pytest.raises(ValueError, match="import shadow"):
        _verify_no_runtime_shadows(tmp_path, ())


def test_final_inventory_covers_reconstruction_and_b0_outputs():
    expected = set(runner._FINAL_EXPECTED_FILES)
    assert "implementation_manifest.json" in expected
    for scenario in runner._FINAL_SCENARIOS:
        prefix = f"b0_protected_recomputed/{scenario}"
        assert f"{prefix}/scheduled_node_windows.csv" in expected
        assert f"{prefix}/gcspo_b0_{scenario}_prn_local_scores.csv" in expected
        assert f"{prefix}/gcspo_b0_{scenario}_prn_local_event_scores.csv" in expected
        assert f"{prefix}/gcspo_b0_{scenario}_prn_local_onset_summary.json" in expected
        assert f"{prefix}/gcspo_b0_{scenario}_prn_local_score_vs_time.png" in expected
        assert prefix in runner._FINAL_EXPECTED_DIRECTORIES
    assert "b0_protected_recomputed" in runner._FINAL_EXPECTED_DIRECTORIES


def test_r1_implementation_manifest_uses_portable_artifact_relative_paths(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    identities = []
    for name in ("clean_only_report.json", "clean_ablation_report.json",
                 "clean_a5_report.json", "clean_reproduction_evidence.json"):
        path = artifact / name
        payload = (name + "\n").encode()
        path.write_bytes(payload)
        identities.append({"path": str(path.resolve()),
                           "sha256": hashlib.sha256(payload).hexdigest(),
                           "size_bytes": len(payload)})
    runner._write_r1_implementation_manifest(
        artifact, {"target_commit": "a" * 40, "clean_identities": identities})
    document = json.loads((artifact / "implementation_manifest.json").read_text())
    assert [row["path"] for row in document["clean_scientific_artifacts"]] == [
        "clean_only_report.json", "clean_ablation_report.json",
        "clean_a5_report.json", "clean_reproduction_evidence.json",
    ]


def test_preflight_projects_clean_identity_into_sealed_snapshot(monkeypatch, tmp_path):
    root = tmp_path / "original"
    snapshot = tmp_path / "reviewed-snapshot"
    artifact = root / runner.ARTIFACT_RELATIVE
    snapshot_artifact = snapshot / runner.ARTIFACT_RELATIVE
    artifact.mkdir(parents=True)
    snapshot_artifact.mkdir(parents=True)

    clean_relative = f"{runner.ARTIFACT_RELATIVE}/clean_only_report.json"
    checked = {
        "wrapper_commit": "a" * 40,
        "target_commit": "b" * 40,
        "runtime_rows": [],
        "clean_rows": [{
            "path": clean_relative,
            "sha256": "0" * 64,
            "size_bytes": 1,
        }],
        "r1_identity_map": {},
    }

    monkeypatch.setenv("GCSPO_R1_RUNTIME_SNAPSHOT", str(snapshot))
    monkeypatch.setenv("GCSPO_R1_SEALED_FD_MAP", "{}")
    monkeypatch.setattr(runner, "install_and_verify_adapter", lambda: nullcontext())
    monkeypatch.setattr(runner, "_verify_adapter_bindings", lambda: None)
    monkeypatch.setattr(
        runner, "verify_execution_freeze", lambda *_args, **_kwargs: checked,
    )
    monkeypatch.setattr(runner, "_verify_preaccess_input_tree", lambda *_args: None)
    monkeypatch.setattr(runner, "_verify_no_runtime_shadows", lambda *_args: None)
    monkeypatch.setattr(
        runner, "verify_zero_access_state", lambda *_args: {"dev": 1, "ino": 2},
    )
    monkeypatch.setattr(runner, "verify_effective_preregistration", lambda *_args: {})
    monkeypatch.setattr(runner, "_read_regular_bytes", lambda *_args: b"{}")
    monkeypatch.setattr(
        runner,
        "validate_preaccess_capabilities",
        lambda *_args: {"available": {}, "unavailable": {}},
    )
    monkeypatch.setattr(
        runner,
        "validate_protected_manifest_inventory",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(runner, "_bind_scenario_manifest_paths", lambda *_args: {})
    monkeypatch.setattr(runner, "_build_scenario_access_plan", lambda *_args: ())

    observed = {}

    def sealed_clean_validator(artifact_dir, identities):
        observed["relative"] = (
            Path(identities[0]["path"]).relative_to(snapshot).as_posix()
        )
        observed["artifact"] = Path(artifact_dir)

    monkeypatch.setattr(
        runner, "validate_clean_contrast_preaccess", sealed_clean_validator,
    )

    result = runner.preflight(root, check_remote=False)

    assert observed["artifact"] == snapshot_artifact
    assert observed["relative"] == clean_relative
    assert result["clean_identities"][0]["path"] == str(root / clean_relative)
