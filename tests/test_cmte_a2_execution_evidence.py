from __future__ import annotations
import ast, hashlib, importlib.util, inspect, json, subprocess, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]

def load_script(name):
 path=ROOT/"scripts"/name; spec=importlib.util.spec_from_file_location("evidence_"+name.replace(".","_"),path); assert spec and spec.loader
 mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def test_eval_module_is_development_only_by_ast_and_confirm_capability_is_first_gate(tmp_path):
 path=ROOT/"scripts/eval_cmte_a2_texbat.py"; text=path.read_text(); tree=ast.parse(text)
 functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 assert "expected_tier" not in text
 assert "_confirm_main" not in functions and "validate_confirm_capability" not in text
 assert "_run" not in functions
 confirm=load_script("score_cmte_a2_confirmatory.py")
 sig=inspect.signature(confirm.score_confirmatory)
 assert next(iter(sig.parameters))=="capability"
 missing=tmp_path/"must-not-be-opened"
 with pytest.raises(PermissionError,match="capability"):
  confirm.score_confirmatory(object(),["--state-dir",str(missing),"--scenario",f"DS7={missing}={missing}","--out",str(tmp_path/"out")])
 assert not (tmp_path/"out").exists()

def _head():
 return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()

def _attestation(tmp_path, commit=None, exit_code=0, failed=0, passed=35):
 from gnss_doppler_lab.cmte_a2_campaign import canonical_preflight_argv
 commit=commit or _head(); argv=[list(x) for x in canonical_preflight_argv(ROOT)]
 per_command=[passed,0]; log=tmp_path/"preflight.log"
 log.write_text("".join(f"## command {i}\n{per_command[i-1]} passed, {failed if i==1 else 0} failed\n" for i in range(1,len(argv)+1)))
 commands=[]
 for index,command in enumerate(argv):
  commands.append({"argv":command,"cwd":str(ROOT),"started_utc":"2026-08-02T00:00:00Z","completed_utc":"2026-08-02T00:01:00Z",
   "exit_code":exit_code if index==0 else 0,"passed":per_command[index],"failed":failed if index==0 else 0,"skipped":0})
 doc={"schema":"gnss-doppler-lab.cmte-a2-test-attestation.v1","source_commit":commit,"clean_tree_asserted":True,
  "started_utc":"2026-08-02T00:00:00Z","completed_utc":"2026-08-02T00:01:00Z","exit_code":exit_code,"commands":commands,
  "summary":{"passed":passed,"failed":failed,"skipped":0,"tests":passed+failed},
  "log":{"path":str(log.resolve()),"sha256":hashlib.sha256(log.read_bytes()).hexdigest(),"bytes":log.stat().st_size},
  "python":{"executable":argv[0][0],"version":sys.version,"platform":sys.platform},"environment":{"PYTHONHASHSEED":""},
  "fixed_suite":True,"holdout_accessed":False,"subprocess_e2e_attested":False}
 att=tmp_path/"attestation.json"; att.write_text(json.dumps(doc)); return att,doc,log

def test_attestation_validator_rejects_missing_fake_wrong_commit_and_nonzero(tmp_path):
 from gnss_doppler_lab.cmte_a2_campaign import validate_test_attestation
 commit=_head()
 with pytest.raises((FileNotFoundError,ValueError)): validate_test_attestation(tmp_path/"missing.json",commit)
 att,doc,log=_attestation(tmp_path)
 assert validate_test_attestation(att,commit)["summary"]["passed"]==35
 doc["source_commit"]="b"*40; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="commit"): validate_test_attestation(att,commit)
 doc["source_commit"]=commit; doc["exit_code"]=1; doc["commands"][0]["exit_code"]=1; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="exit|failed"): validate_test_attestation(att,commit)
 doc["exit_code"]=0; doc["commands"][0]["exit_code"]=0; doc["summary"]["failed"]=1; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="failed"): validate_test_attestation(att,commit)
 doc["summary"]["failed"]=0; log.write_text("tampered") ; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="log"): validate_test_attestation(att,commit)

def test_attestation_validator_rejects_forged_true_command_and_false_contract_flags(tmp_path):
 from gnss_doppler_lab.cmte_a2_campaign import validate_test_attestation
 att,doc,_=_attestation(tmp_path)
 doc["commands"][0]["argv"]=["/bin/true"]
 doc["fixed_suite"]=False
 doc["holdout_accessed"]=True
 att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="fixed|holdout|command"):
  validate_test_attestation(att,doc["source_commit"])

@pytest.mark.parametrize(("mutation","match"),[
 ("wrong_command","command"),("wrong_order","command"),("wrong_cwd","cwd"),
 ("fixed_false","fixed_suite"),("fixed_one","fixed_suite"),
 ("holdout_true","holdout_accessed"),("holdout_zero","holdout_accessed"),
])
def test_attestation_validator_authenticates_exact_suite_contract(tmp_path,mutation,match):
 from gnss_doppler_lab.cmte_a2_campaign import validate_test_attestation
 att,doc,_=_attestation(tmp_path)
 if mutation=="wrong_command": doc["commands"][0]["argv"]=["/bin/true"]
 elif mutation=="wrong_order": doc["commands"][0]["argv"][-2:]=reversed(doc["commands"][0]["argv"][-2:])
 elif mutation=="wrong_cwd": doc["commands"][0]["cwd"]=str(tmp_path)
 elif mutation.startswith("fixed"): doc["fixed_suite"]=False if mutation.endswith("false") else 1
 else: doc["holdout_accessed"]=True if mutation.endswith("true") else 0
 att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match=match): validate_test_attestation(att,doc["source_commit"])

def test_freeze_and_finalizer_require_real_attestation_and_sealed_provenance_names():
 campaign=(ROOT/"src/gnss_doppler_lab/cmte_a2_campaign.py").read_text()
 freeze=(ROOT/"scripts/freeze_cmte_a2_campaign.py").read_text()
 final=(ROOT/"scripts/finalize_cmte_a2_campaign.py").read_text()
 train=(ROOT/"scripts/train_cmte_a2_texbat.py").read_text()
 assert "--test-attestation" in freeze and "validate_test_attestation" in campaign and "test_summary.txt" not in train
 for rel in ("provenance/trust_anchor.json","provenance/confirm_input_manifest.json","provenance/one_shot_ledger.json",
             "provenance/development.json","provenance/confirmatory.json","provenance/hash_inventory.json","provenance/test_attestation.json"):
  assert rel in final
 assert "producer_hashes" in final and "prepared_inputs" in final and "qcal_sha256" in final and "scaler_sha256" in final
