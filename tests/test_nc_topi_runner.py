"""TDD contract tests for the NC-TOPI Stage-0 runner and independent verifier."""
from __future__ import annotations
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).parents[1]
def load(name):
    spec=importlib.util.spec_from_file_location(name[:-3],ROOT/"scripts"/name)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module);return module
r=load("eval_texbat_nc_topi_stage0.py")
v=load("summarize_nc_topi_stage0.py")


def make_npz(path):
    rows=[]
    for seg,start,prn,ch in [(0,0.,2,1),(0,0.,1,0),(1,10.,1,0)]:
        for i in range(11):
            t=start+i*.1; mag=np.arange(1.,10.)*(1+i/100)
            rows.append((seg,ch,prn,t,int(t*1000),np.stack((mag,np.zeros(9)),axis=1)))
    rows=rows[::-1]
    np.savez(path,complex_iq=np.stack([x[5] for x in rows]).astype(np.float32),
        time_s=[x[3] for x in rows],prn=[x[2] for x in rows],channel=[x[1] for x in rows],
        segment_index=[x[0] for x in rows],sample_count=np.array([x[4] for x in rows],np.uint64))


def test_npz_to_windows_sort_normalization_and_source_support(tmp_path):
    path=tmp_path/"x.npz";make_npz(path)
    nodes,audit=r.build_node_windows(path,"cleanStatic")
    assert audit["stable_sort"]==["segment_index","channel","prn","time_s","sample_count","original_index"]
    assert audit["tap_coordinates_chips"]==[-.5,-.375,-.25,-.125,0.,.125,.25,.375,.5]
    node=next(x for x in nodes if x.segment_index==0 and x.prn=="G01")
    assert (node.source_start_s,node.source_end_s,node.epoch_count)==(0.,1.,11)
    assert node.actual_raw[4]==pytest.approx(1.,abs=1e-6)
    assert node.actual_raw[0]==pytest.approx(.2,rel=1e-5)
    assert (node.source_sample_min,node.source_sample_max)==(0,1000)


def test_sequence_history_is_segment_and_gap_safe():
    nodes=[]
    for seg,start in [(0,0.),(1,10.)]:
        for i in range(14):
            t=start+i*.5
            nodes.append(r.NodeWindow("rec","cleanStatic","G01",0,seg,i,t,t+1,t+.5,t+.5,4,i,i,np.ones(9)))
    examples,audit=r.build_sequence_examples(nodes)
    assert len(examples)==4 and audit["cross_segment_examples"]==0
    assert all(all(x.segment_index==e.target.segment_index for x in e.history) for e in examples)
    broken=list(nodes[:14]); data=broken[6].to_dict();data["window_bin_s"]=99
    broken[6]=r.NodeWindow(**data)
    assert not r.build_sequence_examples(broken)[0]


def test_frozen_checkpoint_schema_no_train_and_deterministic_inference():
    config=json.loads((ROOT/"configs/nc_topi_stage0.json").read_text())
    model=r.FrozenB0(ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt",config["b0"],device="cpu")
    assert model.feature_order==("E4","E3","E2","E","P","L","L2","L3","L4")
    x=np.ones((3,12,9),np.float32);a=model.predict(x);b=model.predict(x)
    assert a.shape==(3,9) and np.array_equal(a,b) and not model.model.training
    with pytest.raises(RuntimeError,match="frozen"):model.train()


def test_prediction_inverse_alignment_pairs_and_common_mask():
    fake=object.__new__(r.FrozenB0);fake.mean=np.arange(9,dtype=np.float32);fake.std=np.arange(1,10,dtype=np.float32)
    targets=np.stack([np.arange(9.),np.arange(9.)+1]);pred=np.stack([np.ones(9),np.ones(9)*2])
    ids=[r.identity("rec","cleanStatic","G01",i,2+i) for i in range(2)]
    pairs=r.make_peak_pairs(targets,pred,fake,ids)
    assert np.allclose(pairs[0].predicted_raw,fake.mean+fake.std)
    assert np.allclose(pairs[0].residual_standardized,(targets[0]-pairs[0].predicted_raw)/fake.std)
    r.assert_same_epoch_mask({"B0":pairs,"TOPI":pairs,"NC_TOPI":pairs})
    with pytest.raises(ValueError,match="same exact"):r.assert_same_epoch_mask({"B0":pairs,"TOPI":pairs[:-1]})


def test_attack_label_loader_requires_all_clean_fit_seals():
    gate=r.StageGate()
    with pytest.raises(RuntimeError,match="sealed"):gate.load_attack_labels({"DS1":100})
    for name in gate.REQUIRED_FITS:gate.seal_fit(name,["cleanStatic:normal_train:x"])
    gate.freeze();assert gate.load_attack_labels({"DS1":100})=={"DS1":100.}
    assert gate.audit["attack_fit"] is False
    with pytest.raises(RuntimeError,match="frozen"):gate.seal_fit("thresholds",["cleanStatic:x"])


def test_iq_features_memmap_and_strict_history4_causality(tmp_path):
    path=tmp_path/"iq.bin";z=np.empty((2010,2),np.int16);z[:,0]=np.arange(len(z))%17;z[:,1]=np.arange(len(z))%11;z.tofile(path)
    blocks=r.extract_iq_blocks(path,"rec",sample_rate_hz=1000,block_duration_s=.01,block_stride_s=.5)
    assert len(blocks)==5 and blocks[0].sample_count==10 and np.isfinite(blocks[0].features).all()
    context,audit=r.causal_iq_context([1.51],["rec"],blocks,history=4,cadence=.5)
    assert context.shape==(1,4) and audit["strict_causal"]
    assert max(audit["selected_block_ends"][0])<=1.51
    with pytest.raises(ValueError,match="history"):r.causal_iq_context([.4],["rec"],blocks,history=4)


def test_variable_prn_aggregation_permutation_and_method_specific_thresholds():
    scores={"G03":3.,"G01":1.,"G02":2.,"G04":100.,"G05":4.}
    assert r.aggregate_event(scores,"median")==3
    assert r.aggregate_event(dict(reversed(list(scores.items()))),"top25_mean")==52
    thresholds=r.calibrate_all_thresholds({"B0":{"median":[1,2,3]},"TOPI":{"median":[10,20,30]}},[.99])
    assert thresholds["B0/median/q99"]["value"]==3
    assert thresholds["TOPI/median/q99"]["value"]==30
    assert thresholds["B0/median/q99"]["score_digest_sha256"]!=thresholds["TOPI/median/q99"]["score_digest_sha256"]


def tiny_artifact(root):
    root.mkdir();(root/"plots").mkdir()
    # Valid 1x1 RGBA PNG.
    (root/"plots"/"scores.png").write_bytes(bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63606060f80f0001040100b51c0c020000000049454e44ae426082"))
    rows=[]
    base={"scenario":"cleanStatic","physical_recording_id":"rec","event_id":"e0","target_index":"1","availability_time_s":"421","source_start_s":"420","source_end_s":"421","role":"normal_calibration","phase":"normal","label":"","valid":"1","tracked_prn_count":"2"}
    rows.append({**base,"row_level":"prn","prn":"G01","B0":"1","TOPI":"2","NC_TOPI":"3"})
    rows.append({**base,"row_level":"prn","prn":"G02","B0":"3","TOPI":"4","NC_TOPI":"5"})
    rows.append({**base,"row_level":"event","prn":"","B0_median":"2","TOPI_median":"3","NC_TOPI_median":"4","B0_top25_mean":"3","TOPI_top25_mean":"4","NC_TOPI_top25_mean":"5"})
    fields=sorted(set().union(*(x.keys() for x in rows)))
    with (root/"per_epoch_scores.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    thresholds={"NC_TOPI/median/q99":{"detector":"NC_TOPI","aggregator":"median","quantile":.99,"method":"higher","comparison":"strict >","value":4,"clean_role":"normal_calibration"}}
    payloads={"config.json":{"schema":"gnss-doppler-lab.nc-topi-stage0.v1"},"data_manifest.json":{},"thresholds.json":thresholds,
      "synthetic_physics_tests.json":{"raw_trials":[],"criteria":{}},"bootstrap_results.json":{"comparisons":[]},
      "decision.json":{"status":"INCONCLUSIVE"},"fit_audit.json":{"attack_fit":False,"fit_scenarios":["cleanStatic"]},"provenance.json":{"attack_fit":False}}
    for name,data in payloads.items():(root/name).write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
    for name in ("scenario_metrics.csv","ablation_metrics.csv"):(root/name).write_text("scenario,method,aggregator,value\n")
    # README depends only on reconstructed evidence; then hash everything except hashes.json.
    provisional=v.compute_summary(root,verify_hashes=False,verify_source=False);provisional["ok"]=True;provisional["errors"]=[]
    (root/"README.md").write_text(v.render_readme(provisional));v.write_hash_inventory(root)
    # hashed_files changes once inventory exists; regenerate README and inventory deterministically.
    summary=v.compute_summary(root,verify_hashes=True,verify_source=False);summary["ok"]=True;summary["errors"]=[]
    (root/"README.md").write_text(v.render_readme(summary));v.write_hash_inventory(root)
    return root


def test_independent_verifier_and_tamper_probes(tmp_path):
    root=tiny_artifact(tmp_path/"artifact")
    result=v.verify_artifact(root,verify_source=False)
    assert result["ok"],result["errors"]
    for filename in ("thresholds.json","scenario_metrics.csv","per_epoch_scores.csv","README.md","plots/scores.png"):
        bad=tmp_path/("bad_"+filename.replace("/","_"));shutil.copytree(root,bad)
        path=bad/filename;path.write_bytes(path.read_bytes()+b"x")
        assert not v.verify_artifact(bad,verify_source=False)["ok"]
    missing=tmp_path/"missing";shutil.copytree(root,missing);(missing/"ablation_metrics.csv").unlink()
    assert not v.verify_artifact(missing,verify_source=False)["ok"]


def test_artifact_stage_no_overwrite_and_failed_marker(tmp_path):
    final=tmp_path/"out"
    with r.ArtifactStage(final) as stage:
        (stage.path/"x").write_text("ok");stage.publish(lambda p:{"ok":True,"errors":[]})
    assert (final/"x").read_text()=="ok"
    with pytest.raises(FileExistsError):
        with r.ArtifactStage(final):pass
    with pytest.raises(RuntimeError):
        with r.ArtifactStage(tmp_path/"fail") as stage:raise RuntimeError("boom")
    markers=list(tmp_path.glob(".fail.tmp.*"));assert markers and (markers[0]/"FAILED.json").exists()


def test_synthetic_physics_raw_trials_are_deterministic_and_threshold_free():
    peak=np.exp(-4*r.core.CANONICAL_TAP_COORDS**2);std=np.linspace(.5,1.3,9);pairs=[]
    for i in range(30):
        ident=r.identity("rec","cleanStatic","G01",i,100+i)
        residual=np.linspace(-.2,.2,9)+i/100
        pairs.append(r.core.PeakPredictionPair(peak+residual,peak,residual/std,std,ident,r.core.RAW_SPACE,r.core.RAW_SPACE,r.core.STANDARDIZED_SPACE,r.core.CANONICAL_TAP_COORDS))
    provenance=r.core.FitProvenance("cleanStatic","normal_train",tuple(x.identity for x in pairs))
    covariance=r.core.fit_shrinkage_covariance(pairs,provenance=provenance)
    a=r.run_synthetic_physics(pairs,covariance);b=r.run_synthetic_physics(pairs,covariance)
    assert len(a["raw_trials"])==100 and len(a["second_peak_grid"])==25
    assert a["attack_thresholds_used"] is False and a["raw_trials"]==b["raw_trials"]
    assert max(x["b0_relative_difference"] for x in a["raw_trials"])<=1e-8
