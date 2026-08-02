import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnss_doppler_lab.clif_ip_synthetic import (
    DOMAINS, IMPAIRMENT_AXES, TAP_ORDER, PipelinePaths, artifact_checksums,
    build_final_index, cleanup_after_success, domain_gap, exact_iq_bytes,
    extract_m1_features, fit_multirun_ar, history_design, iq_memmap,
    permutation_test, publish_success, rinex_nav_validity, target_spec,
    validate_final_index,evaluation_masks,adapt_m1_residual_state,array_hash,
    tex_npz_magnitude_nodes,
    validate_run_bundle,
)


def test_final_index_exact_split_disjoint_and_deterministic():
    a=build_final_index(duration_s=120); b=build_final_index(duration_s=120)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==60 and set(a.domain)==set(DOMAINS)
    for domain,g in a.groupby("domain"):
        assert g.split.value_counts().to_dict()=={"train":24,"validation":3,"synthetic_test":3}
        assert g.duration_s.eq(120).all() and g.location_id.nunique()>=10
        for key in ("run_id","location_id","utc","impairment_seed"):
            groups=[set(x[key]) for _,x in g.groupby("split")]
            assert all(groups[i].isdisjoint(groups[j]) for i in range(3) for j in range(i))
    assert not a.label.str.contains("spoof|attack",case=False).any()
    validate_final_index(a)


def test_final_index_uses_actual_rinex_validity_and_preserves_completed_prefix():
    idx=build_final_index(duration_s=120)
    root=Path(__file__).resolve().parents[1]

    # Every requested start is accepted by gps-sdr-sim's RINEX epoch range.
    # Unfinished scenarios additionally reserve the complete 120 s before tmax;
    # row 16 is the immutable, already-successful boundary-start exception.
    for nav_rel,g in idx.groupby("rinex_nav"):
        tmin,tmax=rinex_nav_validity(root/nav_rel)
        starts=pd.to_datetime(g.utc,utc=True)
        assert starts.ge(tmin).all() and starts.le(tmax).all()
        safe=g[~g.run_id.eq("syn-oak-r4-016")]
        ends=pd.to_datetime(safe.utc,utc=True)+pd.to_timedelta(safe.duration_s,unit="s")
        assert ends.le(tmax).all()

    # OAK 1--16 are already published and their complete row identity is immutable.
    prefixes=[]
    oak_prefix=idx[idx.domain.eq("SYN-OAK")].iloc[:16]
    prefixes.append(oak_prefix)
    got=hashlib.sha256(oak_prefix.to_csv(index=False,lineterminator="\n").encode()).hexdigest()
    assert got=="46eba730ffdd3848ee9d2e9f146b49840731bfe9c05ebd6d3ab30cec9d360630"
    prefixes.append(idx[idx.domain.eq("SYN-TEX")].iloc[:16])

    # New scenarios are paired across targets, deterministic, and collision-free.
    oak=idx[idx.domain.eq("SYN-OAK")].iloc[16:].reset_index(drop=True)
    tex=idx[idx.domain.eq("SYN-TEX")].iloc[16:].reset_index(drop=True)
    assert oak.utc.tolist()==tex.utc.tolist()
    assert oak.utc.nunique()==14
    assert set(oak.utc).isdisjoint(set(pd.concat(prefixes).utc))


def test_target_specs_and_exact_bytes():
    assert target_spec("SYN-OAK")=={"sample_rate_hz":5_000_000,"sample_format":"s16le_iq","gnss_sdr_item_type":"ishort"}
    assert target_spec("SYN-TEX")["sample_rate_hz"]==25_000_000
    assert exact_iq_bytes("SYN-OAK",2)==40_000_000
    assert exact_iq_bytes("SYN-TEX",2)==200_000_000


def test_impairment_manifest_has_every_axis_and_no_spoofing():
    d=build_final_index().iloc[0]
    imp=json.loads(d.impairments_json)
    assert set(IMPAIRMENT_AXES)<=set(imp)
    assert imp["attack"] is False and imp["spoofing"] is False
    assert all(imp[k] is not None for k in IMPAIRMENT_AXES)


def test_format_reader_interleaved_little_endian_and_duration(tmp_path):
    p=tmp_path/"iq.bin"; np.array([1,-2,300,-400],dtype="<i2").tofile(p)
    z=iq_memmap(p)
    assert z.dtype==np.dtype("<i2") and z.shape==(2,2)
    assert z.tolist()==[[1,-2],[300,-400]]
    with pytest.raises(ValueError): iq_memmap(tmp_path/"missing")


def test_m1_extractor_memmap_ranges_finite_and_exact_rows(tmp_path):
    fs=1000; p=tmp_path/"x.bin"
    t=np.arange(2000); np.c_[1000*np.sin(t/7),1000*np.cos(t/11)].astype("<i2").tofile(p)
    out=extract_m1_features(p,"r",fs,2.0,block_ms=10,stride_s=.5)
    assert len(out)==4 and np.isfinite(out.select_dtypes("number")).all().all()
    assert out.start_sample.tolist()==[0,500,1000,1500]
    assert out.end_sample.tolist()==[10,510,1010,1510]


def test_multirun_ar_fit_counts_and_history_reset():
    frames=[]
    for run in ("a","b"):
        x=np.c_[np.arange(10),np.arange(10)**2].astype(float)
        frames.append(pd.DataFrame({"run_id":run,"split":"train","t":np.arange(10)*.5,"f0":x[:,0],"f1":x[:,1]}))
    state,audit=fit_multirun_ar(pd.concat(frames),["f0","f1"],pca_dim=2,lag=2)
    assert audit=={"fit_runs":2,"fit_rows":20,"ar_target_rows":16,"history_resets":2}
    assert state["lag"]==2
    b=np.arange(2*7*9,dtype=float).reshape(14,9)-20
    d=pd.DataFrame({"run_id":["a"]*7+["b"]*7,"prn":[1]*14,"t":list(np.arange(7)*.5)*2})
    X,y,meta=history_design(d,b,np.ones((14,2)),lag=2,kind="P3")
    assert y.shape[1]==9 and np.any(y<0) and len(meta)==10
    assert meta.iloc[5].run_id=="b" and meta.iloc[5].t==1.0
    assert "prn" not in meta.attrs["predictor_columns"]


def test_actual_producer_schema_maps_exact_canonical_nine_taps():
    # This is the exact b0_nodes producer spelling/layout, including E/L rather
    # than the legacy E1/L1 aliases. Mapping must preserve all nine dimensions.
    from scripts.train_clif_synthetic import tap_columns
    layout="E4,E3,E2,E,P,L,L2,L3,L4"
    cols=[f"tap_{tap}_rel_prompt_mean" for tap in layout.split(",")]
    d=pd.DataFrame({"tap_layout":[layout], **{c:[i+.25] for i,c in enumerate(cols)}})
    assert tap_columns(d)==cols
    assert d[tap_columns(d)].iloc[0].tolist()==[i+.25 for i in range(9)]


def test_tap_mapping_rejects_alias_scalar_or_absolute_compression():
    from scripts.train_clif_synthetic import tap_columns
    layout="E4,E3,E2,E,P,L,L2,L3,L4"
    absolute=[f"tap_{tap}_mean" for tap in layout.split(",")]
    with pytest.raises(ValueError,match="relative-prompt"):
        tap_columns(pd.DataFrame({"tap_layout":[layout], **{c:[1.] for c in absolute}}))
    with pytest.raises(ValueError,match="exactly"):
        tap_columns(pd.DataFrame({"tap_layout":[layout], "tap_rel_prompt_mean":[1.]}))
    wrong="E4,E3,E2,E1,P,L1,L2,L3,L4"
    with pytest.raises(ValueError,match="layout"):
        tap_columns(pd.DataFrame({"tap_layout":[wrong], **{c:[1.] for c in absolute}}))


def test_same_target_support_signed_order_no_prn_identity():
    d=pd.DataFrame({"run_id":["r"]*8,"prn":["G01"]*8,"t":np.arange(8)*.5})
    b=np.arange(72,dtype=float).reshape(8,9)-40;m=np.ones((8,3))
    supports=[]
    for kind in ("P0","P1","P2","P3"):
        _,y,meta=history_design(d,b,m,2,kind)
        supports.append(list(zip(meta.run_id,meta.prn,meta.t)))
        assert y.shape[1]==9 and tuple(meta.attrs["target_order"])==TAP_ORDER
        assert all("prn" not in c.lower() for c in meta.attrs["predictor_columns"])
    assert supports.count(supports[0])==4


def test_cleanup_only_after_valid_success_same_iq_hash(tmp_path):
    paths=PipelinePaths.for_run(tmp_path,"x")
    paths.iq.parent.mkdir(parents=True); paths.iq.write_bytes(b"iq"); paths.raw.mkdir(parents=True); (paths.raw/"dump").write_bytes(b"x")
    with pytest.raises(RuntimeError): cleanup_after_success(paths)
    m={"run_id":"x","iq_sha256":hashlib.sha256(b"iq").hexdigest(),"b0_iq_sha256":hashlib.sha256(b"iq").hexdigest(),"m1_iq_sha256":hashlib.sha256(b"iq").hexdigest(),"b0_rows":1,"m1_rows":1,"finite":True,"zero_placeholder":False}
    publish_success(paths,m)
    validate_run_bundle(paths)
    cleanup_after_success(paths)
    assert not paths.iq.exists() and not paths.raw.exists()


def test_permutation_199_resolution_region_marginals_reproducible():
    rng=np.random.default_rng(4); m=rng.normal(size=(48,3)); b=m[:,0]+rng.normal(scale=.1,size=48)
    a=permutation_test(b,m,repetitions=199,seed=9,block=8,region="attack_established")
    c=permutation_test(b,m,repetitions=199,seed=9,block=8,region="attack_established")
    assert a==c and a["repetitions"]==199 and a["p_value_resolution"]==pytest.approx(.005)
    assert a["marginals_preserved"] and a["region"]=="attack_established" and len(a["raw_metrics"])==199


def test_domain_gap_finite_and_rmse_ratio():
    a=np.arange(60,dtype=float).reshape(20,3); b=a*2+1
    got=domain_gap(a,b)
    assert {"smd_mean","wasserstein_mean","mmd_rbf","rmse_ratio_5p5x"}<=set(got)
    assert all(np.isfinite(v) for v in got.values())


def test_domain_specific_onset_masks_exact_boundaries():
    t=np.array([109.9,110.,120.,129.9,130.]);oak=evaluation_masks(t,"SYN-OAK")
    assert oak["stable_pre"].tolist()==[True,False,False,False,False]
    assert oak["transition"].tolist()==[False,True,True,True,False]
    assert oak["established_post"].tolist()==[False,False,False,False,True]
    assert oak["nominal_onset_s"]==120
    tex=evaluation_masks(np.array([29.9,30.,89.9,90.,109.9,110.]),"SYN-TEX")
    assert tex["stable_pre"].tolist()==[False,True,True,False,False,False]
    assert tex["transition"].tolist()==[False,False,False,True,True,False]
    assert tex["established_post"].tolist()==[False,False,False,False,False,True]
    assert tex["nominal_onset_s"]==100


def test_s1_m1_adaptation_freezes_basis_and_changes_ar():
    rng=np.random.default_rng(2);frames=[]
    for run in ("a","b"):frames.append(pd.DataFrame({"run_id":run,"split":"train","f0":rng.normal(size=30),"f1":rng.normal(size=30)}))
    d=pd.concat(frames,ignore_index=True);base,_=fit_multirun_ar(d,["f0","f1"],pca_dim=2,lag=2)
    before=array_hash(base["components"]);adapt,audit=adapt_m1_residual_state(base,d,["f0","f1"])
    assert before==array_hash(adapt["components"])
    assert audit["basis_hash_before"]==audit["basis_hash_after"]
    assert audit["ar_hash_before"]!=audit["ar_hash_after"]
    assert {"residual_center","residual_scale","residual_cov"}<=set(adapt)


def test_tex_npz_converter_materializes_real_nine_tap_schema(tmp_path):
    from scripts.train_clif_synthetic import tap_columns
    n=20;iq=np.ones((n,9,2),np.float32);iq[:,4]=2
    src=tmp_path/"ds.npz";np.savez(src,complex_iq=iq,time_s=np.arange(n)*.1,prn=np.repeat([1,2],10))
    out=tmp_path/"nodes.csv";d=tex_npz_magnitude_nodes(src,out,"DS1")
    assert out.is_file() and out.with_suffix(".provenance.json").is_file() and len(d)>0
    assert tap_columns(d)==[f"tap_{x}_rel_prompt_mean" for x in TAP_ORDER]
    assert np.isfinite(d.select_dtypes("number")).all().all()


def test_artifact_schema_and_checksums(tmp_path):
    required=("config.json","synthetic_run_manifest.csv","generation_summary.json","impairment_distribution.json","training_summary.json","predictor_comparison.csv","scenario_metrics.csv","domain_gap_metrics.csv","alignment_destruction_metrics.json","test_summary.txt","README.md")
    (tmp_path/"plots").mkdir()
    for name in required: (tmp_path/name).write_text("x\n")
    checks=artifact_checksums(tmp_path,required)
    assert set(checks)==set(required)
    assert all(len(x)==64 for x in checks.values())


def test_actual_r4_training_is_real_adaptation_and_leakage_safe():
    root=Path(__file__).resolve().parents[1]/"artifacts/clif_ip_synthetic_normal_r4"
    summary=json.loads((root/"training_summary.json").read_text())
    assert summary["regimes"]["R0"]["OAKBAT"]["modified"] is False
    import torch
    for domain in DOMAINS:
        s0=summary["regimes"]["S0"][domain];s1=summary["regimes"]["S1"][domain]
        assert s0["real_rows"]==s0["attack_rows"]==0 and s0["train_runs"]==24
        assert s1["weights_changed"] and s1["before_weight_sha256"]!=s1["after_weight_sha256"]
        assert s1["real_clean_b0_rows"]>0 and s1["real_clean_m1_rows"]>0 and s1["attack_rows"]==0
        ck=torch.load(root/"models"/f"S1_{domain}.pt",map_location="cpu",weights_only=False)
        assert ck["fit_roles"]==["synthetic_train","real_clean_train_0_240"]
        assert ck["attack_rows"]==0 and set(ck["allowed_source_hashes"])=={"b0_cleanStatic","m1_cleanStatic"}
        assert ck["before_weight_sha256"]!=ck["after_weight_sha256"]
        audit=ck["fit_audit"]["real_adaptation"]
        assert audit["basis_hash_before"]==audit["basis_hash_after"]
        assert audit["ar_hash_before"]!=audit["ar_hash_after"]


def test_actual_complete_r4_outputs_are_recomputable_and_nonempty():
    root=Path(__file__).resolve().parents[1]/"artifacts/clif_ip_synthetic_normal_r4"
    pred=pd.read_csv(root/"predictor_comparison.csv")
    for regime,domain in (("S0","SYN-OAK"),("S1","SYN-OAK"),("S0","SYN-TEX"),("S1","SYN-TEX"),("R0","SYN-TEX")):
        q=pred[(pred.regime==regime)&(pred.target_domain==domain)&pred.split.isin(["clean_validation","independent_clean_test"])]
        assert set(q.model)==set(("P0","P1","P2","P3")) and len(q)==8
        assert q.groupby("split").support_sha256.nunique().eq(1).all()
        assert pd.to_numeric(q.mse).notna().all() and pd.to_numeric(q.mae).notna().all()
    s=pd.read_csv(root/"scenario_metrics.csv")
    tex=s[(s.target_domain=="SYN-TEX")&s.regime.isin(["R0","S0","S1"])]
    assert set(tex.scenario)=={"DS1","DS2","DS3","DS4"}
    assert not tex[tex.comparison_scope.eq("r4_same_protocol")].roc_auc.isna().any()
    for scenario in ("DS1","DS2","DS3"):
        p=root/"real_inputs"/f"texbat_{scenario}_method_a_magnitude_nodes.csv"
        assert p.is_file() and p.stat().st_size>1000 and p.with_suffix(".provenance.json").is_file()
    # Scenario AUC can be independently reconstructed from retained epoch rows.
    row=s[(s.regime=="S1")&(s.target_domain=="SYN-TEX")&(s.scenario=="DS1")&(s.model=="Full")].iloc[0]
    e=pd.read_csv(root/"per_epoch/S1_SYN-TEX_DS1.csv");m=evaluation_masks(e.available_s,"SYN-TEX")
    from sklearn.metrics import roc_auc_score
    got=roc_auc_score(np.r_[np.zeros(m["stable_pre"].sum()),np.ones(m["established_post"].sum())],np.r_[e.loc[m["stable_pre"],"Full"],e.loc[m["established_post"],"Full"]])
    assert got==pytest.approx(float(row.roc_auc),abs=1e-12)


def test_actual_destruction_is_199_true_9d_rescoring_and_ledger():
    root=Path(__file__).resolve().parents[1]/"artifacts/clif_ip_synthetic_normal_r4"
    raw=pd.read_csv(root/"alignment_destruction_raw_metrics.csv")
    assert set(raw.model)=={"P2","P3","Full"} and raw.predictor_scoring_called.all()
    assert raw.support_preserved.all() and raw.marginals_preserved.all()
    assert raw.groupby(["region_key","model"]).size().eq(199).all()
    assert set(pd.to_numeric(raw[raw.model.isin(["P2","P3"])].target_dimensions))=={9}
    summary=json.loads((root/"alignment_destruction_metrics.json").read_text())
    assert summary["actual_predictor_rescoring"] and summary["p_value_resolution"]==.005
    assert all(x["replicates"]==199 for x in summary["results"].values())
    ledger=pd.read_csv(root/"synthetic_bundle_ledger.csv")
    assert len(ledger)==60 and ledger.run_id.nunique()==60
    assert ledger.b0_rows.gt(0).all() and ledger.m1_rows.gt(0).all()
    assert ledger.leaf_sha256.str.len().eq(64).all()


def test_actual_r4_index_checksums_and_receiver_backed_smoke():
    root=Path(__file__).resolve().parents[1]/"artifacts/clif_ip_synthetic_normal_r4"
    idx=pd.read_csv(root/"synthetic_run_manifest.csv"); validate_final_index(idx)
    checks=json.loads((root/"checksums.json").read_text())["files"]
    for rel,digest in checks.items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest
    smoke=root/"smoke"/"runs"
    for run,fs,size in (("smoke-syn-oak",5_000_000,400_000_000),("smoke-syn-tex",25_000_000,2_000_000_000)):
        p=PipelinePaths.for_run(root/"smoke",run);m=validate_run_bundle(p)
        assert m["sample_rate_hz"]==fs and m["iq_bytes"]==size and m["duration_s"]==20
        assert m["receiver"]["tracking_rows"]>0 and m["receiver"]["tracked_prns"]
        assert m["b0_rows"]>0 and m["m1_rows"]>0 and m["finite"] and not m["zero_placeholder"]
        assert m["iq_sha256"]==m["b0_iq_sha256"]==m["m1_iq_sha256"]
        assert not p.iq.exists() and not p.raw.exists()
