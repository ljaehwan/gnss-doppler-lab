"""Security, provenance, freeze, and finalization primitives for CMTE-A2.

This module deliberately contains no detector/model/training implementation.
It validates bytes and provenance before callers enter a scoring import boundary.
"""
from __future__ import annotations
import hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

PREREG_COMMIT="e7cb2e5822923a129d72c475706f87721ddd8104"
PREREG_HASHES={
 "docs/CMTE_A2_PREREGISTRATION.md":"5bc92fb711ed85ee20f67e9c8deac7b10bbc9de01d65736eaf4b797d87b6a64f",
 "configs/cmte_a2_preregistration.json":"c2e090aba28acbbd094272aa6bd2c13edab4399d8e406811ec0417d941ebfd8f",
}
DS7_RAW_SHA="d5fb1430d476f68930f3bb0290b80b649f08012eb6b6981d112493528813400e"
DS7_NPZ_SHA="d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8"
DS8_RAW_SHA="1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78"
RECEIVER_SHA="6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb"
EXPORTER_SHA="30a45f988cec15fdce84552ff30747b472c7d76df07d93f79d6ae236166d4039"
CONVERTER_SHA="5e6db2ef2a07b01a5753d2ac3729df0da47c95821ce61fc4118b634efb671f5a"
_VALIDATED_ANCHORS:dict[int,tuple[dict[str,Any],str]]={}
_ISSUED_CONFIRM_CAPABILITIES:set[Any]=set()


def file_sha256(path:str|Path)->str:
 h=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for chunk in iter(lambda:stream.read(1<<20),b""): h.update(chunk)
 return h.hexdigest()


def canonical_sha(document:Any)->str:
 return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def atomic_json(path:str|Path,document:Mapping[str,Any],*,exclusive:bool=False)->None:
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 payload=(json.dumps(dict(document),indent=2,sort_keys=True)+"\n").encode()
 if exclusive:
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
  try:
   with os.fdopen(fd,"wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
  except Exception:
   try: os.unlink(path)
   except FileNotFoundError: pass
   raise
  return
 tmp=path.with_name(path.name+f".tmp-{os.getpid()}")
 with tmp.open("wb") as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
 os.replace(tmp,path)


def require_nonempty(path:str|Path)->Path:
 p=Path(path).resolve(strict=True)
 if not p.is_file() or p.stat().st_size==0: raise ValueError(f"empty required artifact: {p}")
 if p.suffix.lower()==".csv" and len(p.read_text(errors="replace").splitlines())<2:
  raise ValueError(f"empty CSV artifact: {p}")
 return p


def _git(repo:Path,*args:str,bytes_:bool=False):
 return subprocess.check_output(["git",*args],cwd=repo,stderr=subprocess.STDOUT,text=not bytes_).strip()


def validate_preregistration(repo:str|Path,source_commit:str)->None:
 root=Path(repo).resolve(strict=True)
 if _git(root,"merge-base","--is-ancestor",PREREG_COMMIT,source_commit)!="":
  # merge-base emits no output; non-zero is raised.
  raise ValueError("unexpected merge-base output")
 for rel,digest in PREREG_HASHES.items():
  current=root/rel
  if file_sha256(current)!=digest: raise ValueError(f"preregistration content changed: {rel}")
  blob=_git(root,"show",f"{PREREG_COMMIT}:{rel}",bytes_=True)
  # check_output.strip() is inappropriate for immutable bytes; restore exact file
  blob=subprocess.check_output(["git","show",f"{PREREG_COMMIT}:{rel}"],cwd=root)
  if hashlib.sha256(blob).hexdigest()!=digest: raise ValueError(f"preregistration commit blob mismatch: {rel}")


def validate_source_tree(repo:str|Path,source_commit:str,*,require_clean:bool=True)->None:
 root=Path(repo).resolve(strict=True)
 if _git(root,"rev-parse","HEAD")!=source_commit: raise ValueError("current HEAD differs from generating source commit")
 if require_clean and _git(root,"status","--porcelain"): raise ValueError("campaign operation requires a clean git tree")
 validate_preregistration(root,source_commit)


def verify_checksums(directory:str|Path)->dict[str,str]:
 root=Path(directory).resolve(strict=True); path=require_nonempty(root/"checksums.json")
 doc=json.loads(path.read_text())
 if not isinstance(doc,dict) or not doc: raise ValueError(f"empty checksums inventory: {path}")
 for rel,digest in doc.items():
  candidate=require_nonempty(root/rel)
  if file_sha256(candidate)!=digest: raise ValueError(f"checksum mismatch: {candidate}")
 return doc


def inventory(paths:Mapping[str,str|Path])->dict[str,dict[str,Any]]:
 result={}
 for name,raw in paths.items():
  p=require_nonempty(raw); result[name]={"path":str(p),"sha256":file_sha256(p),"bytes":p.stat().st_size}
 return result


def build_confirm_input_manifest(output:str|Path,*,ds7_raw,ds7_npz,ds7_node,ds7_input_manifest,
 ds8_raw,ds8_npz,ds8_rendered_config,ds8_prep_manifest,ds8_node,ds8_input_manifest,
 expected_ds7_raw_sha=DS7_RAW_SHA,expected_ds8_raw_sha=DS8_RAW_SHA,code_hashes:Mapping[str,str])->dict[str,Any]:
 paths={"DS7/raw":ds7_raw,"DS7/npz":ds7_npz,"DS7/node":ds7_node,"DS7/input_manifest":ds7_input_manifest,
        "DS8/raw":ds8_raw,"DS8/npz":ds8_npz,"DS8/rendered_config":ds8_rendered_config,
        "DS8/prep_manifest":ds8_prep_manifest,"DS8/node":ds8_node,"DS8/input_manifest":ds8_input_manifest}
 files=inventory(paths)
 if files["DS7/raw"]["sha256"]!=expected_ds7_raw_sha: raise ValueError("forged or unexpected DS7 raw source SHA")
 if files["DS8/raw"]["sha256"]!=expected_ds8_raw_sha: raise ValueError("forged or unexpected DS8 raw source SHA")
 d7=json.loads(Path(ds7_input_manifest).read_text()); d8=json.loads(Path(ds8_input_manifest).read_text())
 if str(d7.get("scenario","")).upper()!="DS7" or str(d8.get("scenario","")).upper()!="DS8": raise ValueError("input manifest scenario mismatch")
 fp7=d7.get("campaign_converter_fingerprint"); fp8=d8.get("campaign_converter_fingerprint")
 if not fp7 or fp7!=fp8: raise ValueError("mixed or stale campaign converter fingerprint")
 semantics={"magnitude":"hypot(I,Q)","tap_order":["E4","E3","E2","E","P","L","L2","L3","L4"],
  "window_seconds":1.0,"stride_seconds":0.5,"prompt_relative":True,
  "prompt_normalization":"per_epoch_tap_magnitude_divided_by_prompt_magnitude_then_window_mean",
  "gap_bridging":False,"source_tensor":"complex_iq[N,9,2]","component_order":["I","Q"]}
 required={"converter","wrapper","receiver","exporter","template"}
 if set(code_hashes)!=required or any(len(str(v))!=64 for v in code_hashes.values()): raise ValueError("exact producer code hashes required")
 wrappers={d7.get("wrapper_content_sha256"),d8.get("wrapper_content_sha256")}
 if None in wrappers or len(wrappers)!=1: raise ValueError("input manifest wrapper content hashes disagree")
 wrapper_sha=next(iter(wrappers))
 if wrapper_sha!=code_hashes.get("wrapper"): raise ValueError("input manifest wrapper differs from code_hashes wrapper")
 expected_fp=canonical_sha({"converter_content_sha256":CONVERTER_SHA,"wrapper_content_sha256":wrapper_sha,**semantics})
 if fp7!=expected_fp: raise ValueError("forged/stale campaign converter+wrapper fingerprint")
 for document,scenario in ((d7,"DS7"),(d8,"DS8")):
  if document.get("converter_content_sha256")!=CONVERTER_SHA or document.get("converter_semantics")!=semantics:
   raise ValueError(f"{scenario} converter content/semantics mismatch")
  if document.get("source_sha256")!=files[f"{scenario}/npz"]["sha256"]: raise ValueError(f"{scenario} source NPZ SHA mismatch")
  if document.get("node_sha256")!=files[f"{scenario}/node"]["sha256"]: raise ValueError(f"{scenario} node SHA mismatch")
 prep=json.loads(Path(ds8_prep_manifest).read_text())
 if prep.get("status")!="prepared": raise ValueError("DS8 source preparation is missing or failed")
 checks=(("raw_sha256","DS8/raw"),("rendered_config_sha256","DS8/rendered_config"))
 for field,key in checks:
  if prep.get(field)!=files[key]["sha256"]: raise ValueError(f"DS8 prep {field} mismatch")
 if prep.get("npz",{}).get("sha256")!=files["DS8/npz"]["sha256"]: raise ValueError("DS8 prep NPZ mismatch")
 required={"converter","wrapper","receiver","exporter","template"}
 if set(code_hashes)!=required or any(len(str(v))!=64 for v in code_hashes.values()): raise ValueError("exact producer code hashes required")
 if code_hashes["converter"]!=CONVERTER_SHA: raise ValueError("stale converter content hash")
 if code_hashes["receiver"]!=RECEIVER_SHA or code_hashes["exporter"]!=EXPORTER_SHA: raise ValueError("pinned receiver/exporter hash mismatch")
 doc={"schema":"gnss-doppler-lab.cmte-a2-confirm-inputs.v1","scenarios":["DS7","DS8"],
      "campaign_converter_fingerprint":fp7,"files":files,"producer_hashes":dict(code_hashes),
      "preregistration":{"commit":PREREG_COMMIT,"content_sha256":dict(PREREG_HASHES)},
      "no_model_scoring_performed":True,"confirmatory_eligible":True}
 atomic_json(output,doc,exclusive=True); return doc


def validate_trust_anchor(anchor_path:str|Path,expected_sha:str,*,repo:str|Path)->dict[str,Any]:
 # This function intentionally opens only the caller-supplied trust anchor and repository metadata.
 anchor=Path(anchor_path)
 raw=anchor.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected_sha: raise ValueError("trust anchor SHA mismatch")
 doc=json.loads(raw)
 source=doc.get("source_commit")
 if not source: raise ValueError("trust anchor source commit missing")
 validate_source_tree(repo,source,require_clean=True)
 pre=doc.get("pre_holdout_files",{})
 if not isinstance(pre,dict) or not pre: raise ValueError("trust anchor pre-holdout inventory missing")
 for key,item in pre.items():
  p=require_nonempty(item["path"])
  if file_sha256(p)!=item["sha256"]: raise ValueError(f"frozen pre-holdout file mismatch: {key}")
 _VALIDATED_ANCHORS[id(doc)]=(doc,expected_sha)
 return doc


def resolve_confirmatory_inputs(anchor:Mapping[str,Any])->dict[str,dict[str,Path]]:
 # Called only after validate_trust_anchor and O_EXCL ledger. First holdout boundary.
 entry=anchor.get("confirm_input_manifest") or {}; manifest=require_nonempty(entry.get("path",""))
 if file_sha256(manifest)!=entry.get("sha256"): raise ValueError("confirm input manifest hash mismatch")
 doc=json.loads(manifest.read_text())
 # Validate every source/preparation/config/node/manifest byte before semantic use.
 for key,item in doc.get("files",{}).items():
  p=require_nonempty(item["path"])
  if file_sha256(p)!=item["sha256"]: raise ValueError(f"confirm input checksum mismatch: {key}")
 producer=doc.get("producer_hashes",{}); current_wrapper=file_sha256(Path(__file__).with_name("cmte_a2_inputs.py"))
 current_template=file_sha256(Path(__file__).resolve().parents[2]/"configs/cmte_a2_ds8_receiver.conf")
 expected={"converter":CONVERTER_SHA,"wrapper":current_wrapper,"receiver":RECEIVER_SHA,"exporter":EXPORTER_SHA,"template":current_template}
 if producer!=expected: raise ValueError("confirm input producer converter/wrapper/receiver/exporter/template mismatch")
 expected_fp=canonical_sha({"converter_content_sha256":CONVERTER_SHA,"wrapper_content_sha256":current_wrapper,
   "magnitude":"hypot(I,Q)","tap_order":["E4","E3","E2","E","P","L","L2","L3","L4"],"window_seconds":1.0,
   "stride_seconds":0.5,"prompt_relative":True,"prompt_normalization":"per_epoch_tap_magnitude_divided_by_prompt_magnitude_then_window_mean",
   "gap_bridging":False,"source_tensor":"complex_iq[N,9,2]","component_order":["I","Q"]})
 resolved={}
 for scenario in ("DS7","DS8"):
  resolved[scenario]={}
  for kind in ("node","input_manifest"):
   item=doc.get("files",{}).get(f"{scenario}/{kind}")
   if not item: raise ValueError(f"frozen confirm input missing {scenario}/{kind}")
   resolved[scenario][kind]=Path(item["path"]).resolve(strict=True)
  input_doc=json.loads(resolved[scenario]["input_manifest"].read_text())
  if input_doc.get("converter_content_sha256")!=CONVERTER_SHA or input_doc.get("wrapper_content_sha256")!=current_wrapper:
   raise ValueError(f"confirm input expected converter+wrapper mismatch: {scenario}")
  if input_doc.get("campaign_converter_fingerprint")!=expected_fp or doc.get("campaign_converter_fingerprint")!=expected_fp:
   raise ValueError(f"confirm input converter+wrapper campaign fingerprint mismatch: {scenario}")
 return resolved


_CONFIRM_CAPABILITY_SECRET=object()


class _ConfirmCapability:
 __slots__=("_secret","anchor_sha256","ledger_path")
 def __init__(self,secret,anchor_sha256,ledger_path):
  if secret is not _CONFIRM_CAPABILITY_SECRET: raise PermissionError("confirm capability cannot be constructed")
  self._secret=secret; self.anchor_sha256=str(anchor_sha256); self.ledger_path=str(ledger_path)


def _issue_confirm_capability(anchor:Mapping[str,Any],ledger_path:str|Path,anchor_sha:str)->_ConfirmCapability:
 """Issue only after byte/source guards and an O_EXCL started ledger exist."""
 ledger=json.loads(require_nonempty(ledger_path).read_text())
 if ledger.get("status")!="started" or ledger.get("trust_anchor_sha256")!=anchor_sha:
  raise PermissionError("confirm capability requires the started one-shot ledger")
 registered=_VALIDATED_ANCHORS.get(id(anchor))
 if registered is None or registered[0] is not anchor or registered[1]!=anchor_sha:
  raise PermissionError("confirm capability requires a byte/source-validated trust anchor")
 capability=_ConfirmCapability(_CONFIRM_CAPABILITY_SECRET,anchor_sha,Path(ledger_path).resolve())
 _ISSUED_CONFIRM_CAPABILITIES.add(capability); return capability


def validate_confirm_capability(value:Any)->None:
 if (not isinstance(value,_ConfirmCapability) or value._secret is not _CONFIRM_CAPABILITY_SECRET
     or value not in _ISSUED_CONFIRM_CAPABILITIES):
  raise PermissionError("valid issued internal confirm capability required")
 ledger=json.loads(require_nonempty(value.ledger_path).read_text())
 if ledger.get("status")!="started" or ledger.get("trust_anchor_sha256")!=value.anchor_sha256:
  raise PermissionError("confirm capability ledger is no longer in started state")


def create_ledger(path:str|Path,anchor_sha:str)->dict[str,Any]:
 doc={"schema":"gnss-doppler-lab.cmte-a2-one-shot-ledger.v1","status":"started",
      "trust_anchor_sha256":anchor_sha,"one_shot":True}
 atomic_json(path,doc,exclusive=True); return doc


def update_ledger(path:str|Path,*,status:str,detail:str|None=None,result_sha256:str|None=None)->None:
 p=Path(path); doc=json.loads(p.read_text()); doc.update({"status":status,"detail":detail,"result_sha256":result_sha256})
 # Preserve the O_EXCL-created inode's one-shot evidence via atomic replacement; path can never be reused by create_ledger.
 atomic_json(p,doc)


def collect_freeze(repo:str|Path,state_dir:str|Path,development_dir:str|Path,confirm_input_manifest:str|Path)->dict[str,Any]:
 root=Path(repo).resolve(strict=True); state=Path(state_dir).resolve(strict=True); dev=Path(development_dir).resolve(strict=True)
 state_checks=verify_checksums(state); dev_checks=verify_checksums(dev)
 required_state=("b0_model.pt","a2_state.json","config.json","training.json","calibration.json","thresholds.json",
                 "preregistration.json","provenance.json","freeze_manifest.json","historical_b0_gate_equivalence.json","checksums.json")
 required_dev=("development_metrics.csv","baseline_metrics.csv","bootstrap.csv","exact_n_diagnostics.csv","matched_fpr.csv",
              "audit.json","test_summary.json","success_audit.json","prn_dependence.json","historical_b0_gate_equivalence.json","provenance/provenance.json","checksums.json")
 for name in required_state: require_nonempty(state/name)
 for name in required_dev: require_nonempty(dev/name)
 for s in ("DS1","DS2","DS3","DS4"):
  require_nonempty(dev/"per_epoch"/f"{s}.csv"); require_nonempty(dev/"per_prn"/f"{s}.csv")
 training=json.loads((state/"training.json").read_text()); state_prov=json.loads((state/"provenance.json").read_text())
 dev_prov=json.loads((dev/"provenance/provenance.json").read_text())
 source=state_prov.get("execution_source_commit")
 if not source or training.get("execution",{}).get("source_commit")!=source: raise ValueError("training source commit assertion missing")
 if training.get("execution",{}).get("clean_tree_asserted") is not True: raise ValueError("training clean-tree assertion missing")
 if dev_prov.get("execution_source_commit")!=source or dev_prov.get("clean_tree_asserted") is not True:
  raise ValueError("development execution source/clean-tree assertion missing")
 validate_source_tree(root,source,require_clean=True)
 confirm=require_nonempty(confirm_input_manifest); cdoc=json.loads(confirm.read_text())
 if cdoc.get("scenarios")!=["DS7","DS8"] or not cdoc.get("confirmatory_eligible"): raise ValueError("invalid confirm input manifest")
 code_rel=("src/gnss_doppler_lab/cmte_a2.py","src/gnss_doppler_lab/cmte_a2_inputs.py","src/gnss_doppler_lab/cmte_a2_campaign.py",
 "scripts/train_cmte_a2_texbat.py","scripts/eval_cmte_a2_texbat.py","scripts/check_cmte_a2_historical_b0.py",
 "scripts/smoke_cmte_a2_receiver_config.py","scripts/prepare_cmte_a2_inputs.py",
 "scripts/prepare_cmte_a2_ds8_complex.py","scripts/build_cmte_a2_confirm_input_manifest.py","scripts/freeze_cmte_a2_campaign.py",
 "scripts/confirm_cmte_a2_texbat.py","scripts/finalize_cmte_a2_campaign.py","configs/cmte_a2_ds8_receiver.conf")
 pre={}
 for rel in required_state: pre[f"state/{rel}"]={"path":str((state/rel).resolve()),"sha256":file_sha256(state/rel)}
 for rel in required_dev: pre[f"development/{rel}"]={"path":str((dev/rel).resolve()),"sha256":file_sha256(dev/rel)}
 for rel in code_rel: pre[f"code/{rel}"]={"path":str((root/rel).resolve()),"sha256":file_sha256(root/rel)}
 historical_rel=("artifacts/cmte_texbat_poc/per_prn/DS1.csv","artifacts/cmte_a2_historical_b0/ds1_golden_events.csv",
  "artifacts/cmte_a2_receiver_smoke/exporter_fixture.npz","artifacts/cmte_a2_receiver_smoke/exporter_fixture.json")
 for rel in historical_rel: pre[f"fixture/{rel}"]={"path":str((root/rel).resolve()),"sha256":file_sha256(root/rel)}
 return {"schema":"gnss-doppler-lab.cmte-a2-campaign-freeze.v1","source_commit":source,"prereg_commit":PREREG_COMMIT,
  "preregistration_hashes":dict(PREREG_HASHES),"state_dir":str(state),"development_dir":str(dev),
  "state_checksums_sha256":file_sha256(state/"checksums.json"),"development_checksums_sha256":file_sha256(dev/"checksums.json"),
  "confirm_input_manifest":{"path":str(confirm),"sha256":file_sha256(confirm)},"pre_holdout_files":pre,
  "every_confirm_input_hash_bound":True,"immutable":True}


def copy_tree_files(source:Path,destination:Path,prefix:str="")->None:
 for p in sorted(source.rglob("*")):
  if p.is_file():
   target=destination/prefix/p.relative_to(source); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,target)
