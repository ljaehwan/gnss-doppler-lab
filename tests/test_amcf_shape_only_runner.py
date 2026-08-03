from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("shape_runner", "scripts/run_amcf_shape_only.py")
summary = load("shape_summary", "scripts/summarize_amcf_shape_only.py")


def fixture_npz(path: Path, *, zero_prompt: bool = False, include_cn0: bool = True):
    rng = np.random.default_rng(44)
    identities = []
    for segment, base, span in ((0, 0.0, 9.0), (1, 250.0, 9.0),
                                (2, 340.0, 9.0), (3, 420.0, 9.0)):
        for prn in (3, 7, 11):
            for t in np.arange(base + .025, base + span, .05):
                identities.append((segment, prn, t))
    n = len(identities)
    z = rng.normal(size=(n, 9)) + 1j * rng.normal(size=(n, 9))
    z[:, 4] = 5.0 * np.exp(1j * rng.normal(size=n))
    if zero_prompt:
        z[0, 4] = 0
    order = rng.permutation(n)
    kw = {
        "complex_iq": np.stack((z.real, z.imag), axis=-1).astype("f4")[order],
        "time_s": np.asarray([x[2] for x in identities])[order],
        "prn": np.asarray([x[1] for x in identities], dtype="i2")[order],
        "channel": np.asarray([0] * n, dtype="i2")[order],
        "segment_index": np.asarray([x[0] for x in identities], dtype="i4")[order],
        "sample_count": np.arange(n, dtype="u8")[order],
    }
    if include_cn0:
        kw["cn0_db_hz"] = np.full(n, 999.0, dtype="f4")
    np.savez(path, **kw)


def test_red_contract_constants_and_exact_final_path():
    assert set(runner.CANONICAL) == {"cleanStatic", "DS1", "DS2", "DS3", "DS7", "DS8"}
    assert runner.FINAL_ARTIFACT == Path("artifacts/amcf_r1_shape_only")
    assert {"README.md", "config.json", "feature_schema.json", "provenance.json",
            "input_hashes.json", "training_history.csv", "convergence_audit.json",
            "thresholds.json", "scenario_metrics.csv", "seed_metrics.csv",
            "paired_comparisons.csv", "per_epoch", "plots", "models", "hashes.json"} == set(runner.REQUIRED_INVENTORY)


def test_loader_hash_shape_tap_order_stable_sort_and_never_reads_cn0(tmp_path):
    p = tmp_path / "x.npz"
    fixture_npz(p, include_cn0=True)
    digest = runner.sha256(p)
    data, audit = runner.load_canonical_npz(p, digest, "cleanStatic", tap_order=runner.TAP_NAMES)
    assert set(data) == {"complex_iq", "time_s", "prn", "channel", "segment_index", "sample_count"}
    assert "cn0" not in json.dumps(audit).lower()
    keys = list(zip(data["segment_index"], data["channel"], data["prn"],
                    data["time_s"], data["sample_count"]))
    assert keys == sorted(keys)
    assert audit["tap_order"] == list(runner.TAP_NAMES) and audit["shape"][1:] == [9, 2]
    with pytest.raises(ValueError, match="SHA-256"):
        runner.load_canonical_npz(p, "0" * 64, "cleanStatic", tap_order=runner.TAP_NAMES)
    with pytest.raises(ValueError, match="tap order"):
        runner.load_canonical_npz(p, digest, "cleanStatic", tap_order=tuple(reversed(runner.TAP_NAMES)))
    bad = tmp_path / "bad.npz"
    np.savez(bad, complex_iq=np.zeros((2, 8, 2)), time_s=np.arange(2), prn=np.ones(2),
             channel=np.ones(2), segment_index=np.ones(2), sample_count=np.arange(2))
    with pytest.raises(ValueError, match=r"\[N,9,2\]"):
        runner.load_canonical_npz(bad, runner.sha256(bad), "DS1", tap_order=runner.TAP_NAMES)


def test_prompt_zero_floor_roles_literal_features_min_rows_and_provenance(tmp_path):
    p = tmp_path / "x.npz"; fixture_npz(p, zero_prompt=True)
    data, input_audit = runner.load_canonical_npz(p, runner.sha256(p), "cleanStatic", tap_order=runner.TAP_NAMES)
    gate = runner.fit_gate_from_clean_train(data)
    assert gate.minimum > 0 and gate.minimum == max(gate.raw_quantile, gate.positive_floor)
    bundle, qa = runner.build_feature_bundle(data, "cleanStatic", gate, min_valid_rows=5)
    assert qa["zero_prompt_rejected"] >= 1
    assert set(bundle) == {"complex", "magnitude"}
    assert bundle["complex"]["features"].shape[1:] == (8, 4)
    assert bundle["magnitude"]["features"].shape[1:] == (8, 2)
    assert set(np.unique(bundle["complex"]["role"])) <= {"train", "validation", "calibration", "clean_test"}
    assert np.all(bundle["complex"]["valid_count"] >= 5)
    schema = runner.feature_schema_document(5)
    text = json.dumps(schema).lower()
    assert not any(x in text for x in ("cn0", "context", "prompt_magnitude", "valid_fraction"))
    assert schema["representations"]["complex"]["dimensions"] == list(runner.COMPLEX_SCHEMA)
    assert schema["representations"]["magnitude"]["dimensions"] == list(runner.MAGNITUDE_SCHEMA)
    evidence = runner.bind_feature_provenance("cleanStatic", input_audit, gate, schema, bundle, qa)
    runner.verify_feature_provenance(evidence, input_audit, gate, schema, bundle, qa)
    edited = {k: dict(v) for k, v in bundle.items()}
    edited["complex"] = dict(edited["complex"])
    edited["complex"]["features"] = edited["complex"]["features"].copy()
    edited["complex"]["features"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="feature provenance"):
        runner.verify_feature_provenance(evidence, input_audit, gate, schema, edited, qa)
    bad_qa = dict(qa); bad_qa["rejected_rows"] += 1
    with pytest.raises(ValueError, match="feature provenance"):
        runner.verify_feature_provenance(evidence, input_audit, gate, schema, bundle, bad_qa)


def test_causal_history_gap_split_no_padding_common_identity_and_wholly_post(tmp_path):
    p = tmp_path / "x.npz"; fixture_npz(p)
    data, _ = runner.load_canonical_npz(p, runner.sha256(p), "cleanStatic", tap_order=runner.TAP_NAMES)
    gate = runner.fit_gate_from_clean_train(data)
    bundle, _ = runner.build_feature_bundle(data, "cleanStatic", gate, min_valid_rows=5)
    ex = runner.build_examples(bundle["complex"])
    assert ex["history"].shape[1] == 12 and len(ex["current"]) > 0
    assert np.all(ex["history_end"] < ex["source_end"][:, None])
    np.testing.assert_allclose(np.diff(ex["history_end"], axis=1), .5, rtol=0, atol=1e-9)
    assert np.all(ex["role"][:, None] == ex["history_role"])
    labels = runner.phase_labels(np.array([99., 100., 100.5]),
                                 np.array([100., 101., 101.5]), 100.)
    assert labels["post"].tolist() == [False, True, True]


def test_b0_loader_only_timestamp_score_and_duplicate_rejected(tmp_path):
    p = tmp_path / "b0.csv"
    p.write_text("decision_time_s,score_B0_Exact,alarm_primary_q99,phase\n1.0,2.0,true,post\n1.5,3.0,false,post\n")
    rows = runner.load_b0_exact(p, "DS1")
    assert set(rows) == {"decision_time_s", "score_B0_Exact"}
    p.write_text("decision_time_s,score_B0_Exact\n1.0,2.0\n1.0,3.0\n")
    with pytest.raises(ValueError, match="duplicate"):
        runner.load_b0_exact(p, "DS1")


def test_metric_alarm_recompute_timestamp_join_and_prn_permutation():
    rows = []
    for t, score, phase in ((30., 0., "stable_pre"), (30.5, 0., "stable_pre"),
                            (100., 3., "post"), (100.5, 4., "post"), (101., 5., "post")):
        rows.append({"decision_time_s": t, "source_start": t, "source_end": t + 1,
                     "score_ensemble": score, "phase": phase,
                     "alarm_q99": score > 2., "alarm_q995": score > 4.})
    metrics = runner.recompute_scenario_metrics("DS1", rows, 2., 4., onset_s=100.)
    assert metrics["stable_pre_fpr"] == 0 and metrics["post_detection"] == 1
    assert metrics["sustained3_delay_s"] == 0
    bad = [dict(x) for x in rows]; bad[0]["alarm_q99"] = True
    with pytest.raises(ValueError, match="alarm"):
        runner.recompute_scenario_metrics("DS1", bad, 2., 4., onset_s=100.)
    joined = runner.common_timestamp_join(
        [{"decision_time_s": .5, "x": 1}, {"decision_time_s": 1., "x": 2}],
        [{"decision_time_s": 1., "y": 3}, {"decision_time_s": 1.5, "y": 4}], "x", "y")
    assert joined == [{"decision_time_s": 1.0, "complex": 2.0, "comparator": 3.0}]
    assert runner.aggregate_prn_scores({"G03": 4., "G01": 1., "G02": 7.}) == runner.aggregate_prn_scores({"G02": 7., "G03": 4., "G01": 1.})


def test_gpu_probe_requires_real_op():
    class FakeCuda:
        @staticmethod
        def is_available(): return False
    class FakeTorch:
        cuda = FakeCuda()
    with pytest.raises(RuntimeError, match="CUDA"):
        runner.gpu_probe(torch_module=FakeTorch(), required=True)


def test_dirty_tree_refusal_and_baseline_inventory(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "T"], check=True)
    (repo / "a").write_text("a\n")
    subprocess.run(["git", "-C", str(repo), "add", "a"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    runner.verify_primary_source_state(repo, head, allowed_files={"a"})
    (repo / "dirty").write_text("x")
    with pytest.raises(RuntimeError, match="dirty"):
        runner.verify_primary_source_state(repo, "HEAD", allowed_files={"a"})
    actual = runner.diff_inventory(ROOT, "ef36f26")
    assert actual <= runner.SHAPE_ONLY_ALLOWED_FILES
    assert not any(x.startswith("artifacts/amcf_r1_texbat/") or x.startswith("artifacts/cmte_a2_texbat_epochfix/") for x in actual)


def test_atomic_no_overwrite_hash_inventory_and_deterministic_summary_tamper(tmp_path):
    out = tmp_path / "smoke"
    result = runner.run_smoke(out, fixture_seed=19)
    assert result["status"] == "SMOKE-NO-GO" and out.is_dir()
    assert set(runner.REQUIRED_INVENTORY) <= {p.name for p in out.iterdir()}
    assert runner.verify_hashes(out)
    assert not list(out.parent.glob(out.name + ".tmp-*"))
    with pytest.raises(FileExistsError):
        runner.run_smoke(out, fixture_seed=19)
    first = summary.verify_and_summarize(out)
    second = summary.verify_and_summarize(out)
    assert first == second and first["byte_identical"]
    assert first["primary_decision"] == "NO-GO"

    for rel in ("feature_schema.json", "convergence_audit.json", "scenario_metrics.csv",
                "per_epoch/DS1.csv", "feature_cache/cleanStatic_complex.npz"):
        clone = tmp_path / ("tamper_" + rel.replace("/", "_")); shutil.copytree(out, clone)
        q = clone / rel
        b = bytearray(q.read_bytes()); b[max(0, len(b)//2)] ^= 1; q.write_bytes(bytes(b))
        with pytest.raises(ValueError, match="hash"):
            summary.verify_and_summarize(clone)


def test_smoke_is_outside_final_and_readme_claim_limits(tmp_path):
    with pytest.raises(ValueError, match="outside final"):
        runner.run_smoke(ROOT / runner.FINAL_ARTIFACT, fixture_seed=2)
    out = tmp_path / "smoke2"; runner.run_smoke(out, fixture_seed=2)
    text = (out / "README.md").read_text()
    assert "exploratory/developmental" in text
    assert "Not claimable" in text and "q995" in text and "NO-GO" in text
