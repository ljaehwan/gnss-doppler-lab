import csv,hashlib,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest

ROOT=Path(__file__).parents[1]
from gnss_doppler_lab.r2c_stage0_artifact import PRESERVED_TREE,TOP_LEVEL_FILES
from gnss_doppler_lab.r2c_stage0_fix import (B0_FEATURES,ComplexWhitener,SmallNuisanceConditioner,
 TemplateProvider,build_b0_node_windows,joint_profile_glrt,replay_b0_events,run_full_controls,validate_b0_nodes)

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/"scripts"/f"{name}.py");m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def test_original_stage0_bundle_is_exact_base_blob_and_tree_constant():
    base="beb69f4eccfdc9176f742845e965ba0065496742"
    for relative in ("configs/r2c_gnss_stage0.json","scripts/run_r2c_gnss_stage0.py","scripts/verify_r2c_gnss_stage0.py"):
        expected=subprocess.check_output(["git","show",f"{base}:{relative}"],cwd=ROOT)
        assert (ROOT/relative).read_bytes()==expected
    assert subprocess.check_output(["git","rev-parse","HEAD:artifacts/r2c_gnss_stage0"],cwd=ROOT,text=True).strip()==PRESERVED_TREE

def whitener(seed=2):
    rng=np.random.default_rng(seed);z=.03*(rng.normal(size=(100,9))+1j*rng.normal(size=(100,9)))
    return ComplexWhitener(shrinkage=.1).fit(z,["normal_train"]*100)

def geometry():
    u=np.array([[1,0,0],[0,1,0],[0,0,1],[-.6,-.5,-.6245],[.5,-.7,.5099]],float);u/=np.linalg.norm(u,axis=1)[:,None]
    return {i+1:x for i,x in enumerate(u)}

def observations(beta=(20,-15,7,80)):
    taps=np.arange(-.5,.5001,.125);p=TemplateProvider.analytic();los=geometry();out={}
    for prn,u in los.items():
        d=(-u@np.asarray(beta[:3])+beta[3])/299792458*1023000
        out[prn]=np.array([np.exp(.2j)*p.evaluate(taps)+(.3-.1j)*p.evaluate(taps-d),1.1*p.evaluate(taps)+(.2+.2j)*p.evaluate(taps-d)])
    return taps,p,los,out

def test_fixed_covariance_likelihood_bic_and_data_optimized_beta():
    taps,p,los,obs=observations();w=whitener();grid=np.arange(-.5,.5001,.125)
    h0=joint_profile_glrt(obs,los,p,taps,grid,hypothesis="H0",whitener=w)
    shared=joint_profile_glrt(obs,los,p,taps,grid,hypothesis="H1-shared",whitener=w)
    assert shared.valid and shared.beta_m==pytest.approx((20,-15,7,80),abs=3)
    assert shared.score==pytest.approx(2*(shared.log_likelihood-h0.log_likelihood)-(shared.k-h0.k)*np.log(shared.n),abs=1e-10)
    assert h0.k==2*h0.epoch_count+h0.prn_count
    assert shared.k==4*shared.epoch_count+shared.prn_count+4
    # Fixed covariance reference: log-likelihood difference is exactly RSS reduction.
    assert shared.log_likelihood-h0.log_likelihood==pytest.approx(h0.rss-shared.rss,abs=1e-10)

def test_neural_reload_equality_and_separate_energy_fit():
    rng=np.random.default_rng(8);z=rng.normal(size=(30,9))+1j*rng.normal(size=(30,9));x=rng.normal(size=(30,2));xe=np.c_[x,rng.normal(size=30)]
    no=SmallNuisanceConditioner(["cn0","h0_residual_quality"],hidden=3).fit(x,z,["normal_train"]*30,epochs=3)
    restored=SmallNuisanceConditioner.deserialize(no.serialize());assert restored.predict(x)==pytest.approx(no.predict(x))
    energy=SmallNuisanceConditioner(["cn0","h0_residual_quality","explicit_energy"],hidden=3,with_energy=True).fit(xe,z,["normal_train"]*30,epochs=3)
    assert energy.summary["weights_sha256"]!=no.summary["weights_sha256"] and energy.with_energy and not no.with_energy

def test_controls_are_computed_and_threshold_scorer_los_mutations_change_results():
    taps,p,los,obs=observations()
    def scorer(value,pairing):return float(sum(np.mean(np.abs(y)) for y in value.values())+20*sum(p*v[0] for p,v in pairing.items()))
    low=run_full_controls(scorer,obs,los,0,p,taps,seed=7);high=run_full_controls(scorer,obs,los,1e9,p,taps,seed=7)
    assert low["computed_rows"]>=24 and any(r["kind"]=="relation_destruction" and r["effect_size"]!=0 for r in low["rows"])
    assert any(a["post_alarm"]!=b["post_alarm"] for a,b in zip(low["rows"],high["rows"]))
    changed=run_full_controls(lambda value,pairing:2*scorer(value,pairing),obs,los,0,p,taps,seed=7)
    assert changed["rows"][0]["post_score"]!=low["rows"][0]["post_score"]

def raw_rows(count=30):
    rows=[]
    for segment in (0,1):
        for i in range(count):
            row={"time_s":i*.1,"prn":3,"channel":1,"segment_index":segment}
            for tap in ("E4","E3","E2","E","P","L","L2","L3","L4"):row[f"tap_{tap}"]=1+0j
            rows.append(row)
    return pd.DataFrame(rows)

def test_b0_min_epochs_segments_gaps_and_zero_initialized_ewma():
    assert build_b0_node_windows(raw_rows(3),run_id="r").empty
    nodes=build_b0_node_windows(raw_rows(),run_id="r");assert set(nodes.segment_index)=={0,1}
    one=nodes[nodes.segment_index==0].copy();gap=one.drop(one.index[len(one)//2])
    with pytest.raises(ValueError,match="gap"):validate_b0_nodes(gap)
    scores=pd.DataFrame([{"run_id":"r","prn":p,"window_bin_s":.5,"window_start_s":0.,"window_mid_s":.5,"prn_node_rmse":.2} for p in range(1,6)])
    event=replay_b0_events(scores);assert event.btail_max_507080_ewma075.iloc[0]==pytest.approx(.25*event.btail_max_507080.iloc[0])

def test_b0_event_replay_parity_with_tracked_canonical_gate():
    canonical=load_script("eval_btail_support_gate")
    rows=[]
    for bin_s in (.5,1.,1.5):
        for prn,value in enumerate((.05,.11,.14,.18,.22),1):
            rows.append({"run_id":"r","prn":f"G{prn:02d}","window_bin_s":bin_s,"window_start_s":bin_s-.5,
                         "window_mid_s":bin_s,"prn_node_rmse":value+bin_s/100})
    scores=pd.DataFrame(rows);ours=replay_b0_events(scores)
    reference=canonical.build_event_scores(scores,{"q50":.0914354398846626,"q70":.12956311106681812,"q80":.1630456149578094},alpha=.75)
    assert ours["btail_max_507080"].to_numpy()==pytest.approx(reference["btail_max_507080"].to_numpy())
    assert ours["btail_max_507080_ewma075"].to_numpy()==pytest.approx(reference["btail_max_507080_ewma075"].to_numpy())

def test_geometry_expected_assertion_never_changes_derived_time():
    external=Path.home()/"ssd_data/gnss-early-detection/artifacts"
    m=load_script("reconstruct_r2c_time_geometry");directory=external/"r2c-cleanDynamic-geometry-20260804";selected=external/"rc9-real-clean-domain-poc1/exports/cleanDynamic.npz"
    config=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text())["geometry"]
    good=m.reconstruct("cleanDynamic",directory,selected,config,1614,429594);bad=m.reconstruct("cleanDynamic",directory,selected,config,1614,1)
    assert good["derived_time"]["start_tow_s"]==bad["derived_time"]["start_tow_s"]
    assert good["derived_time"]["assertions"]["tow"]["status"]=="PASS" and bad["derived_time"]["assertions"]["tow"]["status"]=="FAIL"
    assert good["offline_geometry_coverage"]["coverage"]<1

def test_runner_guard_and_synthetic_end_to_end_then_verifier_negative(tmp_path):
    runner=load_script("run_r2c_gnss_stage0_fix");verifier=load_script("verify_r2c_gnss_stage0_fix")
    existing=tmp_path/"exists";existing.mkdir()
    with pytest.raises(FileExistsError):runner.validate_destination(existing,test_mode=True)
    output=tmp_path/"campaign";config=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text());head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    runner.run_synthetic(output,config,head);assert {p.name for p in output.iterdir() if p.is_file()}==TOP_LEVEL_FILES
    errors=verifier.verify(output,require_committed=False);assert not errors
    rows=(output/"per_epoch_scores.csv").read_text().splitlines();rows[1]=rows[1].replace(",A1,",",A2,");(output/"per_epoch_scores.csv").write_text("\n".join(rows)+"\n")
    hashes=json.loads((output/"hashes.json").read_text());hashes["files"]["per_epoch_scores.csv"]=hashlib.sha256((output/"per_epoch_scores.csv").read_bytes()).hexdigest();(output/"hashes.json").write_text(json.dumps(hashes))
    assert verifier.verify(output,require_committed=False)

def test_verifier_tamper_negative_suite(tmp_path):
    runner=load_script("run_r2c_gnss_stage0_fix");verifier=load_script("verify_r2c_gnss_stage0_fix")
    base=tmp_path/"base";config=json.loads((ROOT/"configs/r2c_gnss_stage0_fix.json").read_text());head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    runner.run_synthetic(base,config,head)
    def case(name,mutate,update_hashes=True):
        target=tmp_path/name;shutil.copytree(base,target);mutate(target)
        # A data attacker is allowed to update hashes.json too; semantic checks must still fail.
        if update_hashes:
            manifest=json.loads((target/"hashes.json").read_text())
            manifest["files"]={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in target.rglob("*") if p.is_file() and p.name!="hashes.json"}
            (target/"hashes.json").write_text(json.dumps(manifest))
        assert verifier.verify(target,require_committed=False),name
    case("empty_hashes",lambda p:(p/"hashes.json").write_text(json.dumps({"algorithm":"sha256","policy":"all files recursively except hashes.json","files":{}})),False)
    def gates(p):
        d=json.loads((p/"decision.json").read_text());d["verdict"]="PHYSICS_SUPPORTED";(p/"decision.json").write_text(json.dumps(d))
    case("fabricated_gates",gates)
    def alarm(p):
        d=json.loads((p/"gain_invariance.json").read_text());d["rows"][0]["post_alarm"]=not d["rows"][0]["post_alarm"];(p/"gain_invariance.json").write_text(json.dumps(d))
    case("constant_alarm",alarm)
    def alias(p):
        rows=list(csv.DictReader((p/"per_epoch_scores.csv").open()));a1=next(float(r["score"]) for r in rows if r["detector"]=="A1")
        for r in rows:
            if r["detector"]=="A2":r["score"]=a1
        with (p/"per_epoch_scores.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    case("alias",alias)
    def availability(p):
        rows=list(csv.DictReader((p/"per_epoch_scores.csv").open()));rows[0]["availability_time_s"]="nan"
        with (p/"per_epoch_scores.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    case("availability",availability)
    def source(p):
        d=json.loads((p/"provenance.json").read_text());d["source_commit"]="0"*40;(p/"provenance.json").write_text(json.dumps(d))
    case("source",source)
    def perturbation(p):
        d=json.loads((p/"noise_control.json").read_text());d["rows"]=[r for r in d["rows"] if r["kind"]!="quantization"];(p/"noise_control.json").write_text(json.dumps(d))
    case("perturbation",perturbation)
    def bootstrap(p):
        d=json.loads((p/"bootstrap_comparisons.json").read_text());d["comparisons"]=[{"draw_index_sha256":"a"*64}];d["draw_index_sha256"]="b"*64;(p/"bootstrap_comparisons.json").write_text(json.dumps(d))
    case("bootstrap",bootstrap)
    external=tmp_path/"external_input.dat";external.write_text("authentic")
    def external_hash(p):
        d=json.loads((p/"provenance.json").read_text());d["external_inputs"]=[{"path":str(external),"sha256":"0"*64}];(p/"provenance.json").write_text(json.dumps(d))
    case("external",external_hash)
