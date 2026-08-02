"""Regression tests for the CMTE Critical/Important review gaps."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from gnss_doppler_lab.cmte import (
    RESIDUAL_COLUMNS, attach_calibration, fit_distribution, score_residuals,
    sequential_scores, baseline_epoch_scores, validate_clean_provenance,
)
from gnss_doppler_lab.cmte_inputs import extract_role_innovations, convert_complex_npz


def residual_frame(offset=0., run="cleanStatic-role"):
    rows=[]
    for i in range(20):
        x=np.arange(9,dtype=float)+offset+i*.1
        row={"run_id":run,"prn":"G01","window_start_s":i-.5,"window_end_s":i,
             "window_mid_s":i-.25,"window_bin_s":i-.5,"target_window_index":12+i,
             "b0_prn_node_rmse":float(np.sqrt(np.mean(x*x)))}
        row.update(dict(zip(RESIDUAL_COLUMNS,x))); rows.append(row)
    return pd.DataFrame(rows)


def manifest(tmp_path, scenario="cleanStatic", grade="verified_node_artifact"):
    p=tmp_path/"manifest.json"
    p.write_text(json.dumps({"schema":"cmte-input-v1","scenario":scenario,"role":"normal_clean",
      "producer_grade":grade,"source_sha256":"a"*64,"checkpoint_sha256":"b"*64,
      "node_sha256":"c"*64}))
    return p


def test_distribution_fit_and_validation_calibration_are_separate():
    val=residual_frame(20)
    a=attach_calibration(fit_distribution(residual_frame(0)),val)
    b=attach_calibration(fit_distribution(residual_frame(100)),val)
    assert a.metadata["calibration_source"]=="validation_only"
    for method, values in a.calibration.items():
        expected=np.sort(score_residuals(val,fit_distribution(residual_frame(0)),require_calibration=False)[f"q_{method}"])
        assert np.allclose(values,expected)
    assert not np.allclose(a.calibration["full_shrinkage_mahalanobis"],b.calibration["full_shrinkage_mahalanobis"])


def test_positive_allowlist_rejects_attack_tokens_anywhere(tmp_path):
    good=manifest(tmp_path); validate_clean_provenance(good, residual_frame())
    bad=residual_frame(run="texbat-ds1-method-a-9tap-external-validation")
    with pytest.raises(ValueError,match="forbidden"):
        validate_clean_provenance(good,bad)
    doc=json.loads(good.read_text()); doc["scenario"]="DS2"; good.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="cleanStatic"):
        validate_clean_provenance(good,residual_frame())


class Last(torch.nn.Module):
    def forward(self,x): return x[:,-1,:]


def node_frame(n=40, run="continuous"):
    rows=[]
    feats=[f"tap_{x}_rel_prompt_mean" for x in ("E4","E3","E2","E","P","L","L2","L3","L4")]
    for i in range(n):
        row={"run_id":run,"prn":"G01","window_bin_s":i*.5+.5,"window_start_s":i*.5,
             "window_end_s":i*.5+1,"window_mid_s":i*.5+.5}
        row.update({f:float(i+j) for j,f in enumerate(feats)}); rows.append(row)
    return pd.DataFrame(rows),feats


def test_role_local_extraction_resets_history_and_no_crossing():
    nodes,features=node_frame(800)
    roles=extract_role_innovations(nodes,Last(),features,np.zeros(9),np.ones(9),seq_len=12)
    assert set(roles)=={"train","validation","test"}
    for role,out in roles.items():
        assert out.groupby(["run_id","prn"]).head(1).target_window_index.eq(12).all()
    assert roles["train"].window_start_s.min()>=0 and roles["train"].window_end_s.max()<=240
    # A continuous extraction would let validation use train/gap history and starts earlier.
    assert roles["validation"].window_start_s.min()>=256


def test_full_and_diag_q_are_squared_quadratic():
    train=residual_frame(); state=fit_distribution(train,shrinkage=1)
    scored=score_residuals(train,state,require_calibration=False)
    d=train[list(RESIDUAL_COLUMNS)].to_numpy()-state.mean
    expected=np.einsum("ni,ij,nj->n",d,np.linalg.inv(state.covariance),d)
    assert np.allclose(scored.q_full_shrinkage_mahalanobis,expected)
    assert np.allclose(scored.q_diag_mahalanobis,np.sum((d/state.diagonal_scales)**2,axis=1))


def test_s1_fixed_prior_fund_accounting_and_a2_distinct_a4():
    z=sequential_scores(np.log([2.,2.,.5]),["r"]*3,drift=0.,horizon=8)
    assert np.allclose(z.s1_total_fund,1.)
    assert z.iloc[0].s1_capital==pytest.approx(.5+.5*2.)
    assert not np.allclose(z.s1_log_capital,z.s2_e_cusum)
    scored=residual_frame(); scored["q"]=np.arange(len(scored)); scored["p"]=np.linspace(.1,.9,len(scored)); scored["e"]=np.linspace(.2,3,len(scored))
    b=baseline_epoch_scores(scored)
    assert not np.allclose(b.A2,b.A4)


def test_converter_does_not_bridge_segments_and_records_grade(tmp_path):
    iq=np.ones((30,9,2)); prn=np.array([1]*30); segment=np.array([0]*15+[1]*15); t=np.r_[np.arange(15)*.1,np.arange(15)*.1+4]
    src=tmp_path/"x.npz"; np.savez(src,complex_iq=iq,prn=prn,segment_index=segment,time_s=t)
    out=tmp_path/"nodes.csv"; man=tmp_path/"manifest.json"
    frame=convert_complex_npz(src,out,man,scenario="DS1")
    assert not frame.empty and json.loads(man.read_text())["producer_grade"]=="reconstructed_equivalence"
    assert frame.groupby(["prn","segment"]).size().size==2


def test_cli_contracts_use_nodes_manifests_checkpoint_and_required_outputs(tmp_path):
    root=Path(__file__).resolve().parents[1]
    train=(root/"scripts/train_cmte_texbat.py").read_text()
    evaluate=(root/"scripts/eval_cmte_texbat.py").read_text()
    prepare=(root/"scripts/prepare_cmte_texbat_inputs.py").read_text()
    assert "--clean-node-csv" in train and "--clean-manifest" in train
    assert "extract_role_innovations" in train and "--clean-prn-csv" not in train
    assert "--checkpoint" in evaluate and "node.csv=/manifest.json" in evaluate
    for artifact in ("scenario_metrics.csv","ablation_metrics.csv","per_prn_evidence_summary.csv","test_summary.json","checksums.json"):
        assert artifact in evaluate
    assert "required=True" in prepare and "glob(" not in prepare and "rglob(" not in prepare


def test_actual_png_writer_and_metric_schema(tmp_path):
    import importlib.util
    path=Path(__file__).resolve().parents[1]/"scripts/eval_cmte_texbat.py"
    spec=importlib.util.spec_from_file_location("cmte_eval_test",path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    epoch=pd.DataFrame({"availability_time_s":[30.,31.,110.,111.],"N":[2,2,2,2]})
    png=tmp_path/"plot.png"; mod.save_plot(png,epoch,[0.,1.,2.,3.],1.5,"DS1")
    assert png.read_bytes()[:8]==b"\x89PNG\r\n\x1a\n" and png.stat().st_size>1000
    row=mod.metric_row("DS1","Full",[0.,1.,2.,3.],[30.,31.,110.,111.],1.5,.01)
    required={"roc_auc","pr_auc","independent_clean_fpr","stable_pre_fpr","false_alarms_per_min","detection","first_alarm_delay_s","persistent_detection","pre_summary","post_summary"}
    assert required <= set(row)
