from __future__ import annotations
import ast, hashlib, importlib.util, inspect, json, sys
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

def _attestation(tmp_path, commit="a"*40, exit_code=0, failed=0, passed=35):
 log=tmp_path/"preflight.log"; log.write_text(f"## command 1\n{passed} passed, {failed} failed\n")
 doc={"schema":"gnss-doppler-lab.cmte-a2-test-attestation.v1","source_commit":commit,"clean_tree_asserted":True,
  "started_utc":"2026-08-02T00:00:00Z","completed_utc":"2026-08-02T00:01:00Z","exit_code":exit_code,
  "commands":[{"argv":[sys.executable,"-m","pytest","tests/test_cmte_a2.py"],"cwd":str(ROOT),"started_utc":"2026-08-02T00:00:00Z","completed_utc":"2026-08-02T00:01:00Z","exit_code":exit_code,"passed":passed,"failed":failed,"skipped":0}],
  "summary":{"passed":passed,"failed":failed,"skipped":0,"tests":passed+failed},
  "log":{"path":str(log.resolve()),"sha256":hashlib.sha256(log.read_bytes()).hexdigest(),"bytes":log.stat().st_size},
  "python":{"executable":sys.executable,"version":sys.version,"platform":sys.platform},"environment":{"PYTHONHASHSEED":""}}
 att=tmp_path/"attestation.json"; att.write_text(json.dumps(doc)); return att,doc,log

def test_attestation_validator_rejects_missing_fake_wrong_commit_and_nonzero(tmp_path):
 from gnss_doppler_lab.cmte_a2_campaign import validate_test_attestation
 with pytest.raises((FileNotFoundError,ValueError)): validate_test_attestation(tmp_path/"missing.json","a"*40)
 att,doc,log=_attestation(tmp_path)
 assert validate_test_attestation(att,"a"*40)["summary"]["passed"]==35
 doc["source_commit"]="b"*40; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="commit"): validate_test_attestation(att,"a"*40)
 doc["source_commit"]="a"*40; doc["exit_code"]=1; doc["commands"][0]["exit_code"]=1; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="exit|failed"): validate_test_attestation(att,"a"*40)
 doc["exit_code"]=0; doc["commands"][0]["exit_code"]=0; doc["summary"]["failed"]=1; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="failed"): validate_test_attestation(att,"a"*40)
 doc["summary"]["failed"]=0; log.write_text("tampered") ; att.write_text(json.dumps(doc))
 with pytest.raises(ValueError,match="log"): validate_test_attestation(att,"a"*40)

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
