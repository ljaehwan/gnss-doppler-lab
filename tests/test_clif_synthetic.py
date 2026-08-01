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
    permutation_test, publish_success, target_spec, validate_final_index,
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


def test_artifact_schema_and_checksums(tmp_path):
    required=("config.json","synthetic_run_manifest.csv","generation_summary.json","impairment_distribution.json","training_summary.json","predictor_comparison.csv","scenario_metrics.csv","domain_gap_metrics.csv","alignment_destruction_metrics.json","test_summary.txt","README.md")
    (tmp_path/"plots").mkdir()
    for name in required: (tmp_path/name).write_text("x\n")
    checks=artifact_checksums(tmp_path,required)
    assert set(checks)==set(required)
    assert all(len(x)==64 for x in checks.values())


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
