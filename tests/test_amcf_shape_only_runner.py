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
            "paired_comparisons.csv", "decision.json", "feature_audit.json", "window_qa.json",
            "feature_cache", "per_epoch", "plots", "models", "hashes.json"} == set(runner.REQUIRED_INVENTORY)
    assert set(runner.PRIMARY_REQUIRED_INVENTORY) == set(runner.REQUIRED_INVENTORY) | {"calibration_evidence.json"}


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


# Release-blocker regressions: these are intentionally end-to-end contract tests,
# not self-reported boolean checks.
def test_release_primary_cli_and_frozen_constants():
    assert runner.PRIMARY_BASE_SHA == "ef36f265fca856914b7c79b77fd0a2f85f89b4d1"
    assert runner.PRIMARY_MIN_VALID_ROWS == 5
    assert runner.B0_FROZEN_Q99 == pytest.approx(1.7035537524611113, rel=0, abs=0)
    assert runner.B0_FROZEN_INTERVAL == (300.0, 330.0)
    opts = {a.dest for a in runner.parser()._actions}
    assert "baseline" not in opts and "min_valid_rows" not in opts


def test_release_metric_key_q995_cannot_overwrite_primary():
    rows = [
        {"scenario":"DS1","model":"Complex all9","operating_point":"q99","roc_auc":.9},
        {"scenario":"DS1","model":"Complex all9","operating_point":"q995","roc_auc":0.,"diagnostic_only":True},
    ]
    index = runner.index_metric_rows(rows)
    assert len(index) == 2
    assert runner.select_primary_metric(index, "DS1", "Complex all9")["roc_auc"] == .9
    with pytest.raises(ValueError, match="duplicate"):
        runner.index_metric_rows(rows + [dict(rows[0])])


def _score_rows(offset=0.0):
    return [
        {"decision_time_s":340.0+i*.5,"source_start":339.0+i*.5,
         "source_end":340.0+i*.5,"identity_hash":f"id{i}","score":offset+i}
        for i in range(6)
    ]


def test_release_exact_seed_calibration_alignment_and_recomputation():
    banks = {101:_score_rows(0), 202:_score_rows(.1), 303:_score_rows(.2)}
    result = runner.build_calibration_evidence(
        "complex_all9", banks, source_commit="a"*40,
        score_bank_hash="b"*64, index_hash="c"*64)
    assert result["common_identity"] == [f"id{i}" for i in range(6)]
    assert len(result["per_seed_raw_scores"]) == 3
    checked = runner.recompute_calibration_evidence(result)
    assert checked["q99"] == result["q99"] and checked["q995"] == result["q995"]
    assert result["threshold_digest"] == checked["threshold_digest"]
    for mut in (lambda x: x[202].pop(),
                lambda x: x[303].__setitem__(1, {**x[303][1], "identity_hash":"other"}),
                lambda x: x[101].reverse()):
        bad={k:[dict(r) for r in v] for k,v in banks.items()}; mut(bad)
        with pytest.raises(ValueError, match="exact.*seed|identity|order"):
            runner.build_calibration_evidence("v",bad,source_commit="a"*40,
                                              score_bank_hash="b"*64,index_hash="c"*64)
    edited=json.loads(json.dumps(result)); edited["per_seed_raw_scores"]["101"][0] += .5
    with pytest.raises(ValueError, match="calibration|threshold"):
        runner.recompute_calibration_evidence(edited)


def test_release_b0_frozen_contract_from_protected_artifact(tmp_path):
    d=tmp_path/"b0"; d.mkdir()
    thresholds={"primary":{"B0-Exact":{"q99":runner.B0_FROZEN_Q99,
                                          "q995":runner.B0_FROZEN_Q99}}}
    config={"threshold_fit":"cleanStatic [300,330) only"}
    provenance={"source_commit":"1fc5f710c27bd1c179df565132cdcda4df62c7f5"}
    for name,obj in (("thresholds.json",thresholds),("config.json",config),("provenance.json",provenance)):
        (d/name).write_text(json.dumps(obj,sort_keys=True)+"\n")
    c=runner.load_frozen_b0_contract(d, expected_hashes={n:runner.sha256(d/n) for n in
                                         ("thresholds.json","config.json","provenance.json")})
    assert c["q99"] == runner.B0_FROZEN_Q99 and c["fit_interval"] == [300.0,330.0]
    thresholds["primary"]["B0-Exact"]["q99"] = 1.0
    (d/"thresholds.json").write_text(json.dumps(thresholds)+"\n")
    with pytest.raises(ValueError, match="B0|hash"):
        runner.load_frozen_b0_contract(d, expected_hashes={"thresholds.json":runner.sha256(d/"thresholds.json"),
          "config.json":runner.sha256(d/"config.json"),"provenance.json":runner.sha256(d/"provenance.json")})


def test_release_epoch_scores_preserve_actual_intervals_and_alignment():
    ex={"source_start":np.array([1.1,1.1,2.1]),"source_end":np.array([2.,2.,3.]),
        "prn":np.array([3,7,3]),"segment_index":np.array([0,0,0]),"channel":np.array([1,1,1])}
    rows=runner._epoch_scores(ex,np.array([1.,3.,5.]))
    assert rows[2.0]["source_start"] == 1.1 and rows[2.0]["source_end"] == 2.0
    assert rows[2.0]["score"] == 2.0
    bad={k:np.array(v,copy=True) for k,v in ex.items()}; bad["source_start"][1]=1.0
    with pytest.raises(ValueError,match="source interval"):
        runner._epoch_scores(bad,np.array([1.,3.,5.]))
    a=[{"decision_time_s":2.,"source_start":1.1,"source_end":2.,"phase":"stable_pre","x":1}]
    b=[{"decision_time_s":2.,"source_start":1.,"source_end":2.,"phase":"stable_pre","y":2}]
    with pytest.raises(ValueError,match="interval"):
        runner.common_timestamp_join(a,b,"x","y")


def test_release_bootstrap_roc_honors_phase_mask():
    t=np.arange(6.); y=np.array([0,0,1,1,1,1],bool)
    a=np.array([0.,1.,100.,3.,4.,5.]); b=np.array([1.,0.,-100.,2.,3.,4.])
    use=np.array([1,1,0,1,1,1],bool)
    got=runner._bootstrap_delta(t,y,a,b,"roc_auc",0,0,use,reps=20,seed=3)
    expected=runner._roc_auc(y[use],a[use])-runner._roc_auc(y[use],b[use])
    assert got["estimate"] == expected
    assert got["phase_population_hash"] == runner._digest_value({"t":t[use],"y":y[use]})


def test_release_fit_manifest_chain_and_checkpoint_metadata(tmp_path):
    torch=pytest.importorskip("torch")
    arr=np.arange(24,dtype="f4").reshape(3,8,1)
    ex={"current":arr,"history":np.repeat(arr[:,None],12,axis=1),"source_start":np.arange(3.),
        "source_end":np.arange(3.)+1,"prn":np.arange(3),"segment_index":np.zeros(3),
        "channel":np.zeros(3),"role":np.array(["train"]*3)}
    manifest=runner.make_fit_manifest(ex, canonical_input_hash="a"*64,prompt_gate_hash="b"*64,
        scaler_hash="c"*64,feature_tensor_hash=runner._digest_value(arr),validation_bank_hash="d"*64,
        fit_config=runner.PRIMARY_FIT_CONFIG,source_interval_hash=runner._digest_value({"s":ex["source_start"],"e":ex["source_end"]}))
    runner.verify_fit_manifest(manifest,ex)
    for key in ("current","history","source_start"):
        bad={k:np.array(v,copy=True) for k,v in ex.items()}; bad[key].flat[0]+=1
        with pytest.raises(ValueError,match="fit manifest"):
            runner.verify_fit_manifest(manifest,bad)
    p=tmp_path/"m.pt"; torch.save({"state_dict":{},"upstream_digests":dict(manifest.upstream_digests),
                                   "fit_manifest_digest":manifest.manifest_digest},p)
    runner.verify_checkpoint_metadata(p,manifest)
    obj=torch.load(p,weights_only=False); obj["upstream_digests"]["scaler_hash"]="f"*64; torch.save(obj,p)
    with pytest.raises(ValueError,match="checkpoint metadata"):
        runner.verify_checkpoint_metadata(p,manifest)


def test_release_weighted_validation_and_all_gradient_audit():
    assert runner.weighted_validation_loss([(2.,3),(8.,1)]) == pytest.approx(3.5)
    class P:
        requires_grad=True
        grad=None
    with pytest.raises(RuntimeError,match="every trainable"):
        runner.require_finite_trainable_gradients([P()])


def test_release_ds4_na_and_readme_protocol_limit():
    row=runner.ds4_na_row()
    assert row["scenario"]=="DS4" and row["status"]=="NA" and row["included_in_attack_go"] is False
    text=runner.render_readme("NO-GO","PRIMARY COMPLETE",runner._smoke_criteria())
    assert "[300,330)" in text and "[340,410)" in text
    assert "threshold-dependent" in text and "DS4" in text



def test_release_regenerated_outer_hashes_do_not_rescue_feature_tamper(tmp_path):
    out=tmp_path/"smoke-bound"; runner.run_smoke(out,fixture_seed=31)
    q=out/"feature_cache"/"DS7_complex.npz"
    with np.load(q,allow_pickle=False) as f: part={k:np.array(f[k],copy=True) for k in f.files}
    part["features"][0,0,0]+=10
    np.savez(q,**part)
    runner.write_hashes(out)
    with pytest.raises(ValueError,match="feature|sufficient-stat"):
        summary.verify_and_summarize(out)


def test_release_smoke_cannot_be_relabelled_primary_even_with_new_hashes(tmp_path):
    out=tmp_path/"fake-primary"; runner.run_smoke(out,fixture_seed=32)
    config=json.loads((out/"config.json").read_text());config["mode"]="primary-full"
    (out/"config.json").write_text(json.dumps(config,sort_keys=True,indent=2)+"\n")
    runner.write_hashes(out)
    with pytest.raises(ValueError,match="primary|source|config|digest"):
        summary.verify_and_summarize(out)


def test_release_fit_manifest_is_immutable_and_direct_dict_fit_rejected():
    arr=np.ones((2,8,1),dtype="f4")
    ex={"current":arr,"history":np.repeat(arr[:,None],12,axis=1),"source_start":np.arange(2.),
        "source_end":np.arange(2.)+1,"prn":np.arange(2),"segment_index":np.zeros(2),
        "channel":np.zeros(2),"role":np.array(["train"]*2)}
    m=runner.make_fit_manifest(ex,canonical_input_hash="a"*64,prompt_gate_hash="b"*64,
        scaler_hash="c"*64,feature_tensor_hash=runner._digest_value(arr),validation_bank_hash="d"*64,
        fit_config=runner.PRIMARY_FIT_CONFIG,source_interval_hash=runner._example_component_digests(ex)["examples_source_interval_hash"])
    with pytest.raises(TypeError): m.upstream_digests["scaler_hash"]="e"*64
    with pytest.raises(TypeError,match="arbitrary dict"):
        runner.verify_audited_fit_input({"train":ex},"all9",101)



def _write_real_checkpoint_inventory(out: Path):
    torch=pytest.importorskip("torch")
    import sys
    sys.path.insert(0,str(ROOT/"src"))
    from gnss_doppler_lab import amcf_shape_only as core
    (out/"models").mkdir(parents=True)
    audits={}; derived={}; history=[]
    for rep,dim in (("complex",4),("magnitude",2)):
        for objective in ("all9","EPL"):
            for seed in runner.SEEDS:
                key=f"{rep}_{objective}_seed{seed}"
                model=core.ShapeOnlyModel(dim,hidden=32,df=4.0)
                fit_digest=runner._digest_value({"fit":key})
                upstream={"fixture":runner._digest_value({"upstream":key})}
                audit={"seed":seed,"representation":rep,"objective":objective,"finite":True,
                       "epochs_run":1,"optimizer_updates":1,"gradient_audited_updates":1,
                       "every_trainable_parameter_finite_gradient_each_update":True,
                       "patience_early_stop":True,"converged":True,
                       "fit_manifest_digest":fit_digest,"upstream_digests":upstream}
                path=out/"models"/f"{key}.pt"
                torch.save({"checkpoint_role":"amcf-shape-only-primary-real-torch",
                            "state_dict":model.state_dict(),"optimizer":{"state":{"fixture":1}},
                            "audit":audit,"feature_dim":dim,"fit_manifest_digest":fit_digest,
                            "upstream_digests":upstream,"fit_config":dict(runner.PRIMARY_FIT_CONFIG)},path)
                audit["checkpoint_sha256"]=runner.sha256(path);audits[key]=audit
                derived[key]={"fit_manifest_digest":fit_digest,"upstream_digests":upstream}
                history.append({"representation":rep,"objective":objective,"seed":str(seed),"optimizer_updates":"1"})
    return {"audits":audits,"exact_three_converged_per_variant":True},derived,history


def test_release_convergence_loads_all_twelve_real_checkpoints_without_mapping_shadow(tmp_path):
    convergence,derived,history=_write_real_checkpoint_inventory(tmp_path)
    summary._verify_convergence(tmp_path,convergence,history,"primary-full",derived)
    assert len(convergence["audits"])==12


def _primary_feature_audit_fixture(out: Path):
    schema={"fixture":"shape"}; provenance={"scalers":{"complex":{"hash":"c"*64},"magnitude":{"hash":"m"*64}}}
    scenarios={}
    (out/"feature_cache").mkdir();(out/"per_epoch").mkdir()
    for si,name in enumerate(("cleanStatic","DS7","DS8")):
        checks={}
        for rep,d in (("complex",4),("magnitude",2)):
            x=(np.arange(12*8*d,dtype=float).reshape(12,8,d)+si).astype("f4")
            bundle={"features":x,"role":np.array(["test"]*len(x))}
            rel=f"feature_cache/{name}_{rep}.npz";np.savez(out/rel,**bundle)
            iq=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
            checks[rep]={"canonical_fields":list(runner.ALLOWED_NPZ_FIELDS),"feature_cache":rel,
                "source_feature_digest":runner._digest_value(bundle),"feature_tensor_digest":runner._digest_value(x),
                "schema_hash":runner._digest_value(schema),"scaler_hash":provenance["scalers"][rep]["hash"],
                "shape":list(x.shape),"finite_count":int(np.isfinite(x).sum()),"element_count":int(x.size),
                "iqr":iq.tolist(),"pass":True}
        rows=[]
        for j,alarm in enumerate((False,False)):
            end=(j+1) if name=="cleanStatic" else (41+j)
            rows.append({"decision_time_s":end,"source_start":end-.9,"source_end":end,
                         "phase":"clean_test" if name=="cleanStatic" else "stable_pre",
                         "tracked_prn_count":2+j,"score_complex_all9_ensemble":.2+.1*j,
                         "alarm_complex_all9_q99":alarm})
        runner.write_csv(out/"per_epoch"/f"{name}.csv",rows)
        evidence=[(float(r["decision_time_s"]),float(r["source_start"]),float(r["source_end"]),
                   float(r["score_complex_all9_ensemble"]),1.0,bool(r["alarm_complex_all9_q99"])) for r in rows]
        scenarios[name]={"representation_checks":checks,"tracked_prn_median":2.5,
                         "stable_pre_alarm_rate":0.0,
                         "stable_pre_score_threshold_alarm_digest":runner._digest_value(evidence),"pass":True}
    binding={"canonical_fields":list(runner.ALLOWED_NPZ_FIELDS),"forbidden_fields":["cn0_db_hz","context"],"scenarios":scenarios}
    audit={**binding,"pass":True,"audit_digest":runner._digest_value(binding)}
    (out/"thresholds.json").write_text(json.dumps({"complex_all9":{"q99":1.0}}))
    return audit,schema,provenance


def test_release_ds78_collapse_rederives_tracked_median_and_scenario_pass(tmp_path):
    audit,schema,provenance=_primary_feature_audit_fixture(tmp_path)
    assert summary._verify_feature_audit(tmp_path,audit,schema,provenance)["pass"]
    audit["scenarios"]["DS7"]["tracked_prn_median"]=1.0
    audit["scenarios"]["DS7"]["pass"]=True
    binding={k:audit[k] for k in ("canonical_fields","forbidden_fields","scenarios")}
    audit["audit_digest"]=runner._digest_value(binding)
    runner.write_json(tmp_path/"feature_audit.json",audit);runner.write_hashes(tmp_path)
    with pytest.raises(ValueError,match="tracked|collapse|scenario pass"):
        summary._verify_feature_audit(tmp_path,audit,schema,provenance)


def test_release_paired_rejects_1999_reps_before_recompute(monkeypatch,tmp_path):
    monkeypatch.setattr(summary.runner,"ONSETS",{"DS7":100.0})
    paired=[]
    for comparator in ("Magnitude all9","B0 Exact"):
        for metric in ("roc_auc","post_detection","stable_pre_fpr"):
            paired.append({"scenario":"DS7","comparator":comparator,"metric":metric,
                "operating_point":"q99","reps":"1999","block_s":"10.0",
                "bootstrap_seed":str(101+sum(map(ord,"DS7"+comparator+metric)))})
    config={"bootstrap_reps":2000,"bootstrap_block_seconds":10.0,
            "bootstrap_seed_rule":"101+sum(ord(scenario+comparator+metric))"}
    runner.write_csv(tmp_path/"paired_comparisons.csv",paired);runner.write_json(tmp_path/"config.json",config);runner.write_hashes(tmp_path)
    with pytest.raises(ValueError,match="2000|reps"):
        summary._verify_paired(tmp_path,summary._rows(tmp_path/"paired_comparisons.csv"),{},config)


def _checkpoint_score_fixture(out: Path):
    torch=pytest.importorskip("torch")
    convergence,_,_=_write_real_checkpoint_inventory(out)
    rng=np.random.default_rng(881);examples={}
    for ni,name in enumerate(runner.CANONICAL):
        for rep,d in (("complex",4),("magnitude",2)):
            current=rng.normal(size=(2,8,d)).astype("f4");history=rng.normal(size=(2,12,8,d)).astype("f4")
            examples[name,rep]={"current":current,"history":history,"source_start":np.array([ni*10+.1,ni*10+1.1]),
                "source_end":np.array([ni*10+1.,ni*10+2.]),"prn":np.array([3,7]),
                "segment_index":np.array([ni,ni]),"channel":np.array([0,0])}
    scenario_rows={}
    for name in runner.CANONICAL:
        ex=examples[name,"complex"];scenario_rows[name]=[{"decision_time_s":float(ex["source_end"][1]),
            "source_start":float(ex["source_start"][1]),"source_end":float(ex["source_end"][1]),
            "epoch_identity_hash":runner._epoch_scores(ex,np.array([0.,0.]))[float(ex["source_end"][1])]["identity_hash"]}]
    calibration={};calibration_csv=None
    for rep in ("complex","magnitude"):
        for objective in ("all9","EPL"):
            variant=f"{rep}_{objective}";seed_rows={}
            for seed in runner.SEEDS:
                key=f"{variant}_seed{seed}";obj=torch.load(out/"models"/f"{key}.pt",weights_only=False)
                import sys;sys.path.insert(0,str(ROOT/"src"));from gnss_doppler_lab.amcf_shape_only import ShapeOnlyModel
                model=ShapeOnlyModel(obj["feature_dim"],hidden=32,df=4.0);model.load_state_dict(obj["state_dict"])
                maps={}
                for name in runner.CANONICAL:
                    raw=runner._score_examples(model,examples[name,rep],objective,device="cpu");maps[name]=runner._epoch_scores(examples[name,rep],raw)
                    row=scenario_rows[name][0];row[f"score_{variant}_seed{seed}"]=maps[name][row["decision_time_s"]]["score"]
                first=maps["cleanStatic"][float(examples["cleanStatic",rep]["source_end"][0])]
                seed_rows[seed]=[{**first}]
            calibration[variant]=runner.build_calibration_evidence(variant,seed_rows,source_commit="a"*40,score_bank_hash="b"*64,index_hash="c"*64)
    (out/"per_epoch").mkdir()
    for name,rows in scenario_rows.items():runner.write_csv(out/"per_epoch"/f"{name}.csv",rows)
    common=calibration["complex_all9"]["common_rows"]
    calibration_csv=[{"decision_time_s":common["decision_time_s"][0],"source_start":common["source_start"][0],
        "source_end":common["source_end"][0],"epoch_identity_hash":common["identity"][0]}]
    for variant,cal in calibration.items():
        cal["checkpoint_inference_digests"]={}
        for seed in runner.SEEDS:
            calibration_csv[0][f"score_{variant}_seed{seed}"]=cal["per_seed_raw_scores"][str(seed)][0]
            key=f"{variant}_seed{seed}";digest=runner._digest_value(runner.build_inference_result_binding(key,variant,seed,cal,scenario_rows))
            cal["checkpoint_inference_digests"][str(seed)]=digest;convergence["audits"][key]["inference_result_digest"]=digest
            obj=torch.load(out/"models"/f"{key}.pt",weights_only=False);obj["inference_result_digest"]=digest;obj["audit"]["inference_result_digest"]=digest;torch.save(obj,out/"models"/f"{key}.pt")
            column=f"inference_result_digest_{variant}_seed{seed}";calibration_csv[0][column]=digest
            for rows in scenario_rows.values():rows[0][column]=digest
        binding={k:cal[k] for k in ("variant","source_commit","score_bank_hash","index_hash","common_rows","per_seed_raw_scores","checkpoint_inference_digests")};cal["threshold_digest"]=runner._digest_value(binding)
    for name,rows in scenario_rows.items():runner.write_csv(out/"per_epoch"/f"{name}.csv",rows)
    runner.write_csv(out/"per_epoch/cleanStatic_calibration.csv",calibration_csv)
    return convergence,examples,calibration


def test_release_checkpoint_reinference_rejects_raw_tamper_with_regenerated_chains(tmp_path):
    torch=pytest.importorskip("torch");convergence,examples,calibration=_checkpoint_score_fixture(tmp_path)
    assert summary._verify_checkpoint_scores(tmp_path,examples,convergence,calibration)==12
    rows={name:summary._rows(tmp_path/"per_epoch"/f"{name}.csv") for name in runner.CANONICAL}
    rows["DS7"][0]["score_complex_all9_seed101"]=str(float(rows["DS7"][0]["score_complex_all9_seed101"])+.25)
    cal=calibration["complex_all9"];key="complex_all9_seed101";digest=runner._digest_value(runner.build_inference_result_binding(key,"complex_all9",101,cal,rows))
    cal["checkpoint_inference_digests"]["101"]=digest;convergence["audits"][key]["inference_result_digest"]=digest
    checkpoint=tmp_path/"models"/f"{key}.pt";obj=torch.load(checkpoint,weights_only=False);obj["inference_result_digest"]=digest;obj["audit"]["inference_result_digest"]=digest;torch.save(obj,checkpoint)
    convergence["audits"][key]["checkpoint_sha256"]=runner.sha256(checkpoint)
    digest_binding={k:cal[k] for k in ("variant","source_commit","score_bank_hash","index_hash","common_rows","per_seed_raw_scores","checkpoint_inference_digests")};cal["threshold_digest"]=runner._digest_value(digest_binding)
    column="inference_result_digest_complex_all9_seed101"
    for name,part in rows.items():
        for row in part:row[column]=digest
        runner.write_csv(tmp_path/"per_epoch"/f"{name}.csv",part)
    cal_rows=summary._rows(tmp_path/"per_epoch/cleanStatic_calibration.csv")
    for row in cal_rows:row[column]=digest
    runner.write_csv(tmp_path/"per_epoch/cleanStatic_calibration.csv",cal_rows)
    runner.write_json(tmp_path/"calibration_evidence.json",calibration);runner.write_json(tmp_path/"convergence_audit.json",convergence);runner.write_hashes(tmp_path)
    with pytest.raises(ValueError,match="checkpoint output raw NLL mismatch"):
        summary._verify_checkpoint_scores(tmp_path,examples,convergence,calibration)
