from __future__ import annotations
import csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import pytest
from gnss_doppler_lab import nc_topi as stage0

ROOT=Path(__file__).resolve().parents[1]
PY=Path("/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python")
RUN=ROOT/"scripts/audit_nc_topi_shortcut.py"
VERIFY=ROOT/"scripts/summarize_nc_topi_stage0b_audit.py"

def _sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def _manifest(root):
 files={str(p.relative_to(root)):_sha(p) for p in sorted(root.rglob("*")) if p.is_file() and p.name!="hashes.json"}
 (root/"hashes.json").write_text(json.dumps({"files":files},sort_keys=True,indent=2)+"\n")
def _refresh_artifact(root):
 sys.path.insert(0,str(ROOT/"scripts"));import summarize_nc_topi_stage0b_audit as v;v.write_hash_manifest(root)
def _write_csv(path,rows,fields):
 with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def make_parent(root):
 root.mkdir();fields=["scenario","physical_recording_id","event_id","target_index","availability_time_s","source_start_s","source_end_s","role","phase","label","valid","tracked_prn_count","row_level","prn","prn_target_index","pair_sequence_index","B0","total","TOPI","NC_TOPI"]
 specs=[]
 for role,start,n in (("normal_train",10,12),("normal_calibration",20,6),("normal_holdout",30,6)):
  specs += [("cleanStatic",role,"normal","",start+i*.5) for i in range(n)]
 specs += [("cleanDynamic","external_cleanDynamic_diagnostic","normal","",40+i*.5) for i in range(6)]
 for scenario,onset in (("DS1",100),("DS2",100),("DS3",100),("DS7",110),("DS8",110)):
  specs += [(scenario,"attack_evaluation","stable_pre","0",30+i*.5) for i in range(40)]
  specs += [(scenario,"attack_evaluation","post","1",onset+i*.5) for i in range(40)]
 prn=[];events=[];iq=[];features=[]
 for i,(scenario,role,phase,label,t) in enumerate(specs):
  rec=scenario;eid=f"{scenario}@{i:04d}";start=t-.5
  feature=[1.+.01*i,2.+.005*(i%17),.2+.001*(i%31),.1+.002*(i%13)]
  attack=1. if label=="1" else 0.;topi=.5+.02*(i%11)+.8*attack;b0=.4+.01*(i%7)+.3*attack;total=topi+b0+.2
  base={"scenario":scenario,"physical_recording_id":rec,"event_id":eid,"target_index":str(i),"availability_time_s":str(t),"source_start_s":str(start),"source_end_s":str(t),"role":role,"phase":phase,"label":label,"valid":"True","tracked_prn_count":"1"}
  prn.append({**base,"row_level":"prn","prn":"G01","prn_target_index":str(i),"pair_sequence_index":str(i+20),"B0":str(b0),"total":str(total),"TOPI":str(topi),"NC_TOPI":"0"});events.append({**base,"row_level":"event","prn":"","prn_target_index":"","pair_sequence_index":"","B0":"","total":"","TOPI":"","NC_TOPI":""});features.append(feature)
  blocks=[feature]*4;iq.append({"scenario":scenario,"physical_recording_id":rec,"block_recording_id":rec,"event_id":eid,"window_bin_s":str(t),"target_source_start_s":str(start),"history_blocks":"4","cadence_seconds":"0.5","block_end_s":";".join(str(start-1.5+j*.5) for j in range(4)),"block_start_s":";".join(str(start-2+j*.5) for j in range(4)),"sample_offset":"0;1;2;3","sample_count":"10;10;10;10","block_features_json":json.dumps(blocks,separators=(",",":")),"context_features_json":json.dumps(feature,separators=(",",":")),"linked_prns":"G01","linked_pair_count":"1","history_reducer":"arithmetic_mean_per_feature"})
 x=np.asarray(features);train=[i for i,r in enumerate(prn) if r["scenario"]=="cleanStatic" and r["role"]=="normal_train"];cal=[i for i,r in enumerate(prn) if r["scenario"]=="cleanStatic" and r["role"]=="normal_calibration"]
 ids=[stage0.EpochIdentity(r["physical_recording_id"],r["scenario"],r["prn"],int(r["prn_target_index"]),float(r["availability_time_s"])) for r in prn];m=stage0.RobustConditioner().fit(x[train],[float(prn[i]["TOPI"]) for i in train],provenance=stage0.FitProvenance("cleanStatic","normal_train",tuple(ids[i] for i in train)));m.calibrate_cap(x[cal],provenance=stage0.FitProvenance("cleanStatic","normal_calibration",tuple(ids[i] for i in cal)));scale=m.predict_scale(x)
 for i,r in enumerate(prn):r["NC_TOPI"]=repr(float(float(r["TOPI"])/scale[i]))
 rows=[]
 for a,b in zip(prn,events):rows.extend((a,b))
 _write_csv(root/"per_epoch_scores.csv",rows,fields);_write_csv(root/"iq_context.csv",iq,list(iq[0]));(root/"decision.json").write_text(json.dumps({"status":"NO-GO"})+"\n");(root/"provenance.json").write_text(json.dumps({"synthetic":True})+"\n");_manifest(root)

def make_config(path):
 cfg=json.loads((ROOT/"configs/nc_topi_stage0b_audit.json").read_text());cfg["schema"]="gnss-doppler-lab.nc-topi-stage0b-audit.synthetic-test.v1";path.write_text(json.dumps(cfg,indent=2)+"\n")

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
 root=tmp_path_factory.mktemp("stage0b-e2e");parent=root/"parent";out=root/"artifact";cfg=root/"config.json";make_parent(parent);make_config(cfg)
 cmd=[str(PY),str(RUN),"--synthetic-fixture-mode","--config",str(cfg),"--parent",str(parent),"--out",str(out)]
 got=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900)
 assert got.returncode==0,(got.stdout,got.stderr)
 return root,parent,out,cfg,cmd

def verify(root,parent):return subprocess.run([str(PY),str(VERIFY),str(root),"--parent",str(parent)],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900)
def clone(out,tmp_path):dst=tmp_path/"artifact";shutil.copytree(out,dst);return dst

def test_tiny_full_subprocess_publication_no_raw_iq_and_overwrite(tiny):
 root,parent,out,cfg,cmd=tiny;report=json.loads(verify(out,parent).stdout);assert report["ok"] and report["status"]=="VERIFIED"
 assert json.loads((out/"provenance.json").read_text())["raw_iq_opened"] is False
 again=subprocess.run(cmd,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900);assert again.returncode!=0 and (out/"verification.json").is_file()

def test_tampered_parent_bound_event_fields_rejected_with_refreshed_hashes(tiny,tmp_path):
 _,parent,out,_,_=tiny
 for field in ("label","phase","source_start_s","availability_time_s"):
  art=clone(out,tmp_path/field);rows=list(csv.DictReader((art/"event_scores.csv").open()));rows[-1][field]="tampered";_write_csv(art/"event_scores.csv",rows,list(rows[0]));_refresh_artifact(art);got=verify(art,parent);assert got.returncode!=0 and not json.loads(got.stdout)["ok"]

def test_semantic_tampers_rejected_even_after_hash_refresh(tiny,tmp_path):
 _,parent,out,_,_=tiny
 cases=[]
 art=clone(out,tmp_path/"metric");rows=list(csv.DictReader((art/"model_metrics.csv").open()));row=next(x for x in rows if x["scenario"]=="cleanDynamic" and x["method"]=="NC_TOPI_clamped");row["normal_fpr"]="0.99";_write_csv(art/"model_metrics.csv",rows,list(rows[0]));cases.append(art)
 art=clone(out,tmp_path/"scale");rows=list(csv.DictReader((art/"per_prn_scores.csv").open()));rows[0]["predicted_TOPI_scale"]="999";_write_csv(art/"per_prn_scores.csv",rows,list(rows[0]));cases.append(art)
 art=clone(out,tmp_path/"diag");rows=list(csv.DictReader((art/"scale_diagnostics.csv").open()));rows[0]["predicted_scale_q1"]="999";_write_csv(art/"scale_diagnostics.csv",rows,list(rows[0]));cases.append(art)
 art=clone(out,tmp_path/"boot");obj=json.loads((art/"paired_comparisons.json").read_text());obj["comparisons"][-1]=obj["comparisons"][0];(art/"paired_comparisons.json").write_text(json.dumps(obj,indent=2)+"\n");cases.append(art)
 art=clone(out,tmp_path/"iq-check");obj=json.loads((art/"diagnostics/iq_scale_checks.json").read_text());obj["event_common_scale_all"]=False;(art/"diagnostics/iq_scale_checks.json").write_text(json.dumps(obj,indent=2)+"\n");cases.append(art)
 art=clone(out,tmp_path/"extra");(art/"surprise.txt").write_text("x");cases.append(art)
 for art in cases:_refresh_artifact(art);got=verify(art,parent);assert got.returncode!=0,(art,got.stdout);assert not json.loads(got.stdout)["ok"]

def test_unrelated_valid_png_rejected_even_with_sidecar_and_hash_refresh(tiny,tmp_path):
 _,parent,out,_,_=tiny;art=clone(out,tmp_path);target=art/"plots/clamp_hit_ratio.png";target.write_bytes((art/"plots/roc_methods.png").read_bytes());side=json.loads((art/"plot_data.json").read_text());side["plots"]["clamp_hit_ratio"]["png_sha256"]=_sha(target);(art/"plot_data.json").write_text(json.dumps(side,indent=2)+"\n");_refresh_artifact(art);got=verify(art,parent);assert got.returncode!=0 and not json.loads(got.stdout)["ok"]
