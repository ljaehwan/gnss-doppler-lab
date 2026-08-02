from __future__ import annotations
import hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


def load_script(name):
    path=ROOT/"scripts"/name
    spec=importlib.util.spec_from_file_location(name.replace(".","_"),path)
    assert spec and spec.loader
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def complex_npz(path, *, broken=False):
    if broken:
        np.savez(path,nope=np.ones(2)); return
    n=50; iq=np.ones((n,9,2),np.float32); iq[:,:,1]=np.arange(9)
    np.savez(path,complex_iq=iq,prn=np.ones(n,int),time_s=np.arange(n)*.05,
             segment=np.zeros(n,int),channel=np.zeros(n,int))


def test_committed_ds8_template_exact_placeholders_and_receiver_contract():
    p=ROOT/"configs/cmte_a2_ds8_receiver.conf"; text=p.read_text()
    assert text.count("@CMTE_A2_DS8_INPUT_RAW@") == 1
    assert text.count("@CMTE_A2_DS8_OUTPUT_DIR@") >= 1
    for token in ("SignalSource.item_type=ishort","SignalSource.sampling_frequency=25000000",
                  "Channels_1C.count=11","Tracking_1C.tap_count=9","Tracking_1C.tap_spacing_chips=0.125"):
        assert token in text


def test_named_prepare_is_atomic_and_binds_campaign_fingerprint(tmp_path):
    from gnss_doppler_lab.cmte_a2_inputs import prepare_named_complex_inputs
    a=tmp_path/"a.npz"; b=tmp_path/"b.npz"; complex_npz(a); complex_npz(b)
    docs=prepare_named_complex_inputs([f"cleanStatic={a}",f"DS8={b}"],tmp_path/"ok")
    campaign=json.loads((tmp_path/"ok/manifest.json").read_text())
    assert set(docs)=={"CLEANSTATIC","DS8"}
    fp=campaign["campaign_converter_fingerprint"]
    assert len(fp)==64 and campaign["converter_content_sha256"]
    assert campaign["window_seconds"]==1.0 and campaign["stride_seconds"]==0.5
    assert campaign["prompt_normalization"]=="per_epoch_tap_magnitude_divided_by_prompt_magnitude_then_window_mean"
    bad=tmp_path/"bad.npz"; complex_npz(bad,broken=True)
    with pytest.raises(ValueError):
        prepare_named_complex_inputs([f"DS1={a}",f"DS2={bad}"],tmp_path/"partial")
    assert not (tmp_path/"partial").exists()


def test_ds4_separate_sensitivity_markers(tmp_path):
    from gnss_doppler_lab.cmte_a2_inputs import prepare_ds4_sensitivity
    # Reuse a canonical output produced by the exact wrapper.
    source=tmp_path/"source.npz"; complex_npz(source)
    from gnss_doppler_lab.cmte_a2_inputs import prepare_named_complex_inputs
    prepare_named_complex_inputs([f"DS3={source}"],tmp_path/"up")
    doc=prepare_ds4_sensitivity(tmp_path/"up/DS3_nodes.csv",tmp_path/"ds4")
    assert doc["tier"]=="development_sensitivity"
    assert doc["mixed_producer"] is True and doc["confirmatory_eligible"] is False


def test_ds8_prevalidation_failure_is_atomic_explicit_na(tmp_path,monkeypatch):
    mod=load_script("prepare_cmte_a2_ds8_complex.py")
    monkeypatch.setattr(mod,"RAW_BYTES",4); monkeypatch.setattr(mod,"RAW_SHA","0"*64)
    iq=tmp_path/"iq.bin"; iq.write_bytes(b"bad!")
    exe=tmp_path/"rx"; exe.write_bytes(b"x"); exe.chmod(0o755)
    monkeypatch.setattr(mod,"EXEC_SHA",hashlib.sha256(b"x").hexdigest())
    out=tmp_path/"prepared"
    with pytest.raises(ValueError): mod.main(["--iq",str(iq),"--receiver-executable",str(exe),"--output-root",str(out)])
    failure=out.with_name(out.name+".failed")/"failure.json"
    assert failure.is_file(); doc=json.loads(failure.read_text())
    assert doc["primary_result"]=="NA" and doc["silent_fallback"] is False
    assert not out.exists() and not list(tmp_path.glob("*.tmp-*"))


def test_confirm_input_builder_binds_all_ds7_ds8_layers(tmp_path):
    from gnss_doppler_lab.cmte_a2_campaign import build_confirm_input_manifest, file_sha256, CONVERTER_SHA, RECEIVER_SHA, EXPORTER_SHA
    files={}
    for name in ("ds7.raw","ds7.npz","ds7.node","ds7.input.json","ds8.raw","ds8.npz","ds8.conf","ds8.prep.json","ds8.node","ds8.input.json"):
        p=tmp_path/name; p.write_text(name); files[name]=p
    from gnss_doppler_lab.cmte_a2_inputs import CONVERTER_SEMANTICS, _fingerprint
    wrapper=file_sha256(ROOT/"src/gnss_doppler_lab/cmte_a2_inputs.py"); fingerprint=_fingerprint(CONVERTER_SHA,wrapper)
    for n in ("ds7.input.json","ds8.input.json"):
        scenario="DS7" if n.startswith("ds7") else "DS8"
        files[n].write_text(json.dumps({"scenario":scenario,"campaign_converter_fingerprint":fingerprint,
          "converter_content_sha256":CONVERTER_SHA,"converter_semantics":CONVERTER_SEMANTICS,"wrapper_content_sha256":wrapper,
          "source_sha256":file_sha256(files[n.replace("input.json","npz")]),"node_sha256":file_sha256(files[n.replace("input.json","node")])}))
    prep={"status":"prepared","raw_sha256":file_sha256(files["ds8.raw"]),"npz":{"sha256":file_sha256(files["ds8.npz"])},
          "rendered_config_sha256":file_sha256(files["ds8.conf"])}
    files["ds8.prep.json"].write_text(json.dumps(prep))
    doc=build_confirm_input_manifest(tmp_path/"confirm.json",ds7_raw=files["ds7.raw"],ds7_npz=files["ds7.npz"],
      ds7_node=files["ds7.node"],ds7_input_manifest=files["ds7.input.json"],ds8_raw=files["ds8.raw"],
      ds8_npz=files["ds8.npz"],ds8_rendered_config=files["ds8.conf"],ds8_prep_manifest=files["ds8.prep.json"],
      ds8_node=files["ds8.node"],ds8_input_manifest=files["ds8.input.json"],expected_ds7_raw_sha=file_sha256(files["ds7.raw"]),
      expected_ds8_raw_sha=file_sha256(files["ds8.raw"]),code_hashes={"converter":CONVERTER_SHA,"wrapper":wrapper,"receiver":RECEIVER_SHA,
      "exporter":EXPORTER_SHA,"template":"f"*64})
    assert set(doc["scenarios"])=={"DS7","DS8"}; assert doc["campaign_converter_fingerprint"]==fingerprint
    assert len(doc["files"])==10 and all(len(x["sha256"])==64 for x in doc["files"].values())


def test_confirm_script_has_guard_before_holdout_resolution_and_no_fit_imports():
    text=(ROOT/"scripts/confirm_cmte_a2_texbat.py").read_text()
    lower=text.lower()
    assert lower.index("validate_trust_anchor") < lower.index("resolve_confirmatory_inputs")
    assert lower.index("resolve_confirmatory_inputs") < lower.index("create_ledger")
    assert "fit_distribution" not in lower and "train_b0" not in lower and "threshold_operating_points" not in lower
    assert "o_excl" in lower


def test_finalize_rejects_empty_required_artifacts(tmp_path):
    from gnss_doppler_lab.cmte_a2_campaign import require_nonempty
    p=tmp_path/"empty"; p.write_bytes(b"")
    with pytest.raises(ValueError,match="empty"): require_nonempty(p)


def test_campaign_scripts_compile_and_prereg_immutable():
    for name in ("build_cmte_a2_confirm_input_manifest.py","freeze_cmte_a2_campaign.py",
                 "confirm_cmte_a2_texbat.py","finalize_cmte_a2_campaign.py"):
        subprocess.run([sys.executable,"-m","py_compile",str(ROOT/"scripts"/name)],check=True)
    expected={"docs/CMTE_A2_PREREGISTRATION.md":"5bc92fb711ed85ee20f67e9c8deac7b10bbc9de01d65736eaf4b797d87b6a64f",
              "configs/cmte_a2_preregistration.json":"c2e090aba28acbbd094272aa6bd2c13edab4399d8e406811ec0417d941ebfd8f"}
    for name,digest in expected.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest


def test_runtime_provenance_rejects_forged_source_and_mixed_converter_before_output(tmp_path):
    from gnss_doppler_lab.cmte_a2_campaign import (build_confirm_input_manifest,file_sha256,CONVERTER_SHA,RECEIVER_SHA,EXPORTER_SHA)
    from gnss_doppler_lab.cmte_a2_inputs import CONVERTER_SEMANTICS,_fingerprint
    paths={name:tmp_path/name for name in ("d7raw","d7npz","d7node","d7manifest","d8raw","d8npz","d8conf","d8prep","d8node","d8manifest")}
    for name,p in paths.items(): p.write_text(name)
    wrapper=file_sha256(ROOT/"src/gnss_doppler_lab/cmte_a2_inputs.py"); fp=_fingerprint(CONVERTER_SHA,wrapper)
    def input_doc(scenario,npz,node,fingerprint=fp):
        return {"scenario":scenario,"campaign_converter_fingerprint":fingerprint,"converter_content_sha256":CONVERTER_SHA,
          "converter_semantics":CONVERTER_SEMANTICS,"wrapper_content_sha256":wrapper,"source_sha256":file_sha256(npz),"node_sha256":file_sha256(node)}
    paths["d7manifest"].write_text(json.dumps(input_doc("DS7",paths["d7npz"],paths["d7node"])))
    paths["d8manifest"].write_text(json.dumps(input_doc("DS8",paths["d8npz"],paths["d8node"])))
    paths["d8prep"].write_text(json.dumps({"status":"prepared","raw_sha256":file_sha256(paths["d8raw"]),
      "npz":{"sha256":file_sha256(paths["d8npz"])},"rendered_config_sha256":file_sha256(paths["d8conf"])}))
    kwargs=dict(ds7_raw=paths["d7raw"],ds7_npz=paths["d7npz"],ds7_node=paths["d7node"],ds7_input_manifest=paths["d7manifest"],
      ds8_raw=paths["d8raw"],ds8_npz=paths["d8npz"],ds8_rendered_config=paths["d8conf"],ds8_prep_manifest=paths["d8prep"],
      ds8_node=paths["d8node"],ds8_input_manifest=paths["d8manifest"],expected_ds7_raw_sha=file_sha256(paths["d7raw"]),
      expected_ds8_raw_sha=file_sha256(paths["d8raw"]),code_hashes={"converter":CONVERTER_SHA,"wrapper":wrapper,
      "receiver":RECEIVER_SHA,"exporter":EXPORTER_SHA,"template":"b"*64})
    forged=input_doc("DS7",paths["d7npz"],paths["d7node"]); forged["source_sha256"]="0"*64
    paths["d7manifest"].write_text(json.dumps(forged))
    with pytest.raises(ValueError,match="source NPZ"): build_confirm_input_manifest(tmp_path/"no1.json",**kwargs)
    assert not (tmp_path/"no1.json").exists()
    paths["d7manifest"].write_text(json.dumps(input_doc("DS7",paths["d7npz"],paths["d7node"],"f"*64)))
    with pytest.raises(ValueError,match="mixed|stale"): build_confirm_input_manifest(tmp_path/"no2.json",**kwargs)
    assert not (tmp_path/"no2.json").exists()


def test_trust_anchor_tamper_and_confirm_input_tamper_fail(monkeypatch,tmp_path):
    import gnss_doppler_lab.cmte_a2_campaign as campaign
    monkeypatch.setattr(campaign,"validate_source_tree",lambda *a,**k: None)
    state=tmp_path/"state"; state.write_text("state")
    anchor_doc={"source_commit":"a"*40,"pre_holdout_files":{"state":{"path":str(state),"sha256":campaign.file_sha256(state)}}}
    anchor=tmp_path/"anchor.json"; anchor.write_text(json.dumps(anchor_doc)); digest=campaign.file_sha256(anchor)
    assert campaign.validate_trust_anchor(anchor,digest,repo=tmp_path)["source_commit"]=="a"*40
    state.write_text("tampered")
    with pytest.raises(ValueError,match="frozen pre-holdout"): campaign.validate_trust_anchor(anchor,digest,repo=tmp_path)
    node=tmp_path/"node"; node.write_text("node"); im=tmp_path/"im"; im.write_text("manifest")
    files={}
    for scenario in ("DS7","DS8"):
      files[f"{scenario}/node"]={"path":str(node),"sha256":campaign.file_sha256(node)}
      files[f"{scenario}/input_manifest"]={"path":str(im),"sha256":campaign.file_sha256(im)}
    manifest=tmp_path/"confirm.json"; manifest.write_text(json.dumps({"files":files}))
    holder={"confirm_input_manifest":{"path":str(manifest),"sha256":campaign.file_sha256(manifest)}}
    node.write_text("changed")
    with pytest.raises(ValueError,match="checksum"): campaign.resolve_confirmatory_inputs(holder)


def test_one_shot_ledger_second_run_refused_and_failure_retained(tmp_path):
    from gnss_doppler_lab.cmte_a2_campaign import create_ledger,update_ledger
    ledger=tmp_path/"ledger.json"; create_ledger(ledger,"a"*64); update_ledger(ledger,status="failed",detail="synthetic")
    with pytest.raises(FileExistsError): create_ledger(ledger,"a"*64)
    doc=json.loads(ledger.read_text()); assert doc["status"]=="failed" and doc["detail"]=="synthetic"


def test_finalizer_contract_names_all_six_and_no_opposite_tier_placeholder():
    text=(ROOT/"scripts/finalize_cmte_a2_campaign.py").read_text()
    for name in ("scenario_metrics.csv","confirmatory_metrics.csv","development_metrics.csv","baseline_metrics.csv",
                 "bootstrap_cis.csv","exact_n_diagnostics.csv","matched_fpr.csv","test_summary.json","preregistration.json"):
        assert name in text
    assert '("DS1","DS2","DS3","DS4","DS7","DS8")' in text
    assert "empty placeholder table rejected" in text


def test_confirm_guard_spy_observes_no_holdout_access_before_anchor_failure(tmp_path,monkeypatch):
    mod=load_script("confirm_cmte_a2_texbat.py"); observed={"holdout":False}
    def fail_anchor(*args,**kwargs): raise ValueError("anchor guard failure")
    def holdout_spy(*args,**kwargs): observed["holdout"]=True; raise AssertionError("holdout opened before guard")
    monkeypatch.setattr(mod,"validate_trust_anchor",fail_anchor)
    monkeypatch.setattr(mod,"resolve_confirmatory_inputs",holdout_spy)
    ledger=tmp_path/"ledger.json"
    with pytest.raises(ValueError,match="anchor guard"):
        mod.main(["--trust-anchor",str(tmp_path/"anchor"),"--expected-sha256","a"*64,"--ledger",str(ledger),"--out",str(tmp_path/"out")])
    assert observed["holdout"] is False and not ledger.exists()


def test_checksum_inventory_rejects_checkpoint_and_threshold_mutation(tmp_path):
    from gnss_doppler_lab.cmte_a2_campaign import file_sha256,verify_checksums
    checkpoint=tmp_path/"b0_model.pt"; threshold=tmp_path/"thresholds.json"
    checkpoint.write_bytes(b"checkpoint"); threshold.write_text("{}")
    original={"b0_model.pt":file_sha256(checkpoint),"thresholds.json":file_sha256(threshold)}
    (tmp_path/"checksums.json").write_text(json.dumps(original))
    assert verify_checksums(tmp_path)==original
    checkpoint.write_bytes(b"mutation")
    with pytest.raises(ValueError,match="checksum mismatch"): verify_checksums(tmp_path)
