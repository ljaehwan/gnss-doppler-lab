from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab import nc_topi_stage0b as a

ROOT = Path(__file__).resolve().parents[1]


def test_boolean_parser_fails_closed():
    assert a.parse_bool("True") is True
    assert a.parse_bool("false") is False
    for bad in ("", "1", "yes", "TRUE ", None, 1):
        with pytest.raises(ValueError):
            a.parse_bool(bad)


def test_empirical_tails_include_ties_and_add_one():
    ref = np.array([1., 2., 2., 4.])
    got = a.empirical_iq_ood_score(ref, np.array([0., 2., 5.]))
    assert np.allclose(got, -np.log([.4, 1., .4]), rtol=0, atol=1e-15)


def test_higher_quantile_strict_alarm_and_boundaries():
    values = np.arange(100., dtype=float)
    assert a.higher_quantile(values, .99) == 99.
    assert np.array_equal(a.strict_alarms([98., 99., 100.], 99.), [False, False, True])
    scores, inferred = a.reconstruct_original_nc(
        np.array([0., 2e-12, 2.]), np.array([1e-20, 1e-20, 20.]), 10.)
    assert np.array_equal(scores, [0., 2., .2])
    assert np.array_equal(inferred, [1e-12, 1e-12, 10.])
    a.check_effective_scale(np.array([0., 2e-12]), scores[:2], inferred[:2])
    with pytest.raises(ValueError, match="illegal zero division"):
        a.check_effective_scale(np.array([1.]), np.array([0.]), np.array([1.]))


def test_target_conditioner_is_sealed_target_tagged_and_disjoint():
    x = np.column_stack((np.arange(12.), np.ones(12), np.arange(12.) % 3, np.linspace(.1,.4,12)))
    ids = tuple(f"row-{i}" for i in range(12))
    model = a.TargetConditioner.fit("TOPI", x[:8], np.arange(8.) + 1, ids[:8])
    assert model.target == "TOPI" and model.audit["target"] == "TOPI"
    assert model.audit["forbidden_inputs"] == {"attack": False, "label": False, "scenario": False, "onset": False, "prn": False}
    assert model.median.flags.writeable is False and model.coef.flags.writeable is False
    bounds = model.calibration_bounds(x[8:], ids[8:], lower_q=.01, upper_q=.99)
    assert bounds.lower <= bounds.upper
    with pytest.raises(ValueError, match="disjoint"):
        model.calibration_bounds(x[:4], ids[:4], lower_q=.01, upper_q=.99)
    with pytest.raises(ValueError, match="target"):
        a.TargetConditioner.fit("shared", x[:8], np.arange(8.) + 1, ids[:8])


def _write_parent_fixture(root: Path):
    root.mkdir(parents=True)
    fields = ["scenario","physical_recording_id","event_id","target_index","availability_time_s",
              "source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count",
              "row_level","prn","prn_target_index","B0","total","TOPI","NC_TOPI"]
    base=dict(scenario="cleanStatic",physical_recording_id="rec",event_id="rec@1",target_index="0",
              availability_time_s="2",source_start_s="1",source_end_s="2",role="normal_train",
              phase="normal",label="",valid="True",tracked_prn_count="2")
    rows = [{**base,"row_level":"prn","prn":"G01","prn_target_index":"3","B0":"1","total":"2","TOPI":"1","NC_TOPI":".5"},
            {**base,"row_level":"prn","prn":"G02","prn_target_index":"4","B0":"2","total":"3","TOPI":"2","NC_TOPI":"1"},
            {**base,"row_level":"event","prn":"","prn_target_index":"","B0":"","total":"","TOPI":"","NC_TOPI":""}]
    with (root/"per_epoch_scores.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    iq={"scenario":"cleanStatic","physical_recording_id":"rec","block_recording_id":"rec","event_id":"rec@1",
        "window_bin_s":"1","target_source_start_s":"1","history_blocks":"4","cadence_seconds":".5",
        "block_end_s":"0;.1;.2;.3","block_start_s":"-.1;0;.1;.2","sample_offset":"0;1;2;3",
        "sample_count":"1;1;1;1","block_features_json":json.dumps([[1,2,3,4]]*4),
        "context_features_json":json.dumps([1,2,3,4]),"linked_prns":"G01;G02","linked_pair_count":"2",
        "history_reducer":"arithmetic_mean_per_feature"}
    with (root/"iq_context.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(iq));w.writeheader();w.writerow(iq)


def test_exact_event_join_rejects_duplicate_missing_extra_and_link_mismatch(tmp_path):
    root=tmp_path/"parent";_write_parent_fixture(root)
    data=a.load_parent_evidence(root, verify_binding=False)
    assert len(data.prn_rows)==2 and data.features.shape==(2,4)
    assert np.array_equal(data.features[0], data.features[1])
    rows=list(csv.DictReader((root/"iq_context.csv").open()));rows[0]["linked_prns"]="G01"
    with (root/"iq_context.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError, match="linked PRN"):
        a.load_parent_evidence(root, verify_binding=False)


def test_profile_d_support_is_pre_fit_chronological_and_insufficient():
    times=[*range(33), *range(50,83), *range(100,133)]
    events=[{"event_id":str(i),"effective_start_s":float(t),"effective_end_s":float(t)+1} for i,t in enumerate(times)]
    calls=[]
    result=a.check_profile_d_support(events, fit_callback=lambda *_: calls.append(1))
    assert result["status"]=="INSUFFICIENT_NORMAL_SUPPORT"
    assert result["best_counts"]=={"normal_train":33,"normal_calibration":33,"normal_holdout":33}
    assert result["fit_profile_d"] is False and not calls
    assert result["random_split"] is False and result["b0_history_windows"]==12


def test_bootstrap_complete_blocks_and_unavailable_no_iid():
    labels=np.r_[np.zeros(20),np.ones(20)]
    times=np.r_[np.arange(20)*.5,np.arange(20)*.5+20]
    result=a.paired_block_bootstrap(labels,np.arange(40.),np.arange(40.)[::-1],np.array(["r"]*40),times,reps=20)
    assert result["available"] is False
    assert result["iid_fallback"] is False and result["reason"]


def _decision_evidence():
    methods=["B0","TOPI","NC_TOPI_original","IQ_LOW_ONLY","IQ_OOD_ONLY","NC_TOPI_clamped","NC_B0_clamped","NC_total_clamped"]
    pauc={s:{m:.4 for m in methods} for s in ("DS7","DS8")}
    for s in pauc:
        pauc[s].update(NC_TOPI_clamped=.8,NC_B0_clamped=.6,IQ_LOW_ONLY=.5,IQ_OOD_ONLY=.4,B0=.5,NC_TOPI_original=.7)
    ci={s:{"NC_TOPI_clamped_minus_IQ_LOW_ONLY":{"available":True,"lower":.1,"upper":.2},
           "NC_TOPI_clamped_minus_IQ_OOD_ONLY":{"available":True,"lower":.1,"upper":.2},
           "NC_TOPI_clamped_minus_NC_B0_clamped":{"available":True,"lower":.1,"upper":.3}} for s in ("DS7","DS8")}
    return {"pauc":pauc,"paired_ci":ci,"alarm_overlap":{"DS7":.1,"DS8":.1},
            "q99_fpr":{"cleanDynamic":.1,"cleanStatic_holdout":.01,
                       "stable_pre":{"DS1":.01,"DS2":.01,"DS3":.01,"DS7":.01,"DS8":.01}},
            "profile_d":{"status":"INSUFFICIENT_NORMAL_SUPPORT"}}


def test_decision_shortcut_precedence_and_three_statuses():
    evidence=_decision_evidence()
    assert a.evaluate_decision(evidence)["status"]=="TANGENT_SUPPORTED"
    evidence["pauc"]["DS7"]["IQ_LOW_ONLY"]=.9;evidence["pauc"]["DS8"]["IQ_LOW_ONLY"]=.9
    assert a.evaluate_decision(evidence)["status"]=="IQ_SHORTCUT_DOMINATED"
    del evidence["pauc"]["DS8"]["NC_TOPI_clamped"]
    result=a.evaluate_decision(evidence)
    assert result["status"] in {"IQ_SHORTCUT_DOMINATED","INCONCLUSIVE"}
    assert result["validation_errors"]


def test_parent_hash_gate_and_generation_source_binding():
    report=a.verify_parent_binding(ROOT/"artifacts/nc_topi_stage0", repo=ROOT)
    assert report["ok"] and report["parent_artifact_commit"].startswith("6fe5315")
    assert report["parent_generation_source_commit"].startswith("c94af28")
    assert set(report["consumed_file_hashes"])=={"per_epoch_scores.csv","iq_context.csv"}


def test_exact_original_reconstruction_gate():
    report=a.run_original_reconstruction_gate(ROOT/"artifacts/nc_topi_stage0", repo=ROOT)
    assert report["rows"]==55591
    assert report["all_rows_within_rel_abs_1e12"] is True
    assert report["max_relative_error"] <= 1e-12


def test_artifact_stage_atomic_no_overwrite_and_failed_marker(tmp_path):
    final=tmp_path/"out"
    with a.ArtifactStage(final) as stage:
        (stage.path/"x").write_text("ok")
        stage.publish(lambda p:{"ok":True,"errors":[]})
    assert (final/"x").read_text()=="ok"
    with pytest.raises(FileExistsError):
        with a.ArtifactStage(final): pass
    with pytest.raises(RuntimeError):
        with a.ArtifactStage(tmp_path/"bad") as stage:
            raise RuntimeError("boom")
    assert list(tmp_path.glob(".bad.tmp.*"))


def test_independent_verifier_does_not_import_runner_and_rejects_hash_tamper(tmp_path):
    text=(ROOT/"scripts/summarize_nc_topi_stage0b_audit.py").read_text()
    assert "audit_nc_topi_shortcut" not in text
    root=tmp_path/"artifact";root.mkdir();(root/"a.txt").write_text("a")
    a.write_hash_manifest(root)
    spec=importlib.util.spec_from_file_location("v",ROOT/"scripts/summarize_nc_topi_stage0b_audit.py")
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    assert v.verify_hashes(root)["ok"]
    (root/"a.txt").write_text("tampered")
    assert not v.verify_hashes(root)["ok"]


def test_public_cli_has_no_fixture_escape():
    text=(ROOT/"scripts/audit_nc_topi_shortcut.py").read_text()
    assert "test_fixture" not in text
    spec=importlib.util.spec_from_file_location("runner",ROOT/"scripts/audit_nc_topi_shortcut.py")
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    args=runner.parse_args(["--out","/tmp/audit","--verify-after-run"])
    assert args.out==Path("/tmp/audit") and args.verify_after_run is True


def test_parent_evidence_gate_never_opens_configured_raw_iq(monkeypatch):
    original=Path.open;opened=[]
    def guarded(self,*args,**kwargs):
        if self.suffix==".bin":
            opened.append(str(self));raise AssertionError("raw IQ opened")
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guarded)
    report=a.verify_parent_binding(ROOT/"artifacts/nc_topi_stage0",repo=ROOT)
    data=a.load_parent_evidence(ROOT/"artifacts/nc_topi_stage0",verify_binding=False)
    assert report["ok"] and len(data.event_rows)==5283 and opened==[]


def test_time_shuffle_is_target_only_same_permutation_and_deterministic():
    n=20;ids=tuple(f"id-{i}" for i in range(n));x=np.arange(n*4,dtype=float).reshape(n,4);y=np.arange(n,dtype=float)+1
    permutation=np.random.default_rng(0).permutation(n)
    one=a.TargetConditioner.fit("TOPI",x,y[permutation],ids)
    two=a.TargetConditioner.fit("TOPI",x,y[np.random.default_rng(0).permutation(n)],ids)
    assert one.seal==two.seal
    assert one.audit["feature_digest_sha256"]==a._digest_array(x)
    assert one.audit["identity_digest_sha256"]==a._digest_json(list(ids))


def test_profile_d_sufficient_support_fits_once_after_precheck():
    times=[*range(50),*range(70,171),*range(190,240)]
    events=[{"event_id":str(i),"effective_start_s":float(t),"effective_end_s":float(t)+.5} for i,t in enumerate(times)]
    calls=[];result=a.check_profile_d_support(events,fit_callback=lambda rows,witness:calls.append((len(rows),witness)))
    assert result["status"]=="AVAILABLE" and result["fit_profile_d"] is True
    assert len(calls)==1 and calls[0][0]==201


def test_shortcut_ci_including_zero_has_precedence_over_tangent_points():
    evidence=_decision_evidence()
    for scenario in ("DS7","DS8"):
        evidence["paired_ci"][scenario]["NC_TOPI_clamped_minus_NC_B0_clamped"]={"available":True,"lower":0.,"upper":.1}
    result=a.evaluate_decision(evidence)
    assert result["status"]=="IQ_SHORTCUT_DOMINATED"
    assert result["shortcut_triggers"]["b_nc_vs_ncb0_statistically_indistinguishable"] is True
