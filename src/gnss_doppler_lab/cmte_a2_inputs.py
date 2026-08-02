"""CMTE-A2 canonical input wrappers and tier/provenance validation."""
from __future__ import annotations
import hashlib, json, os, re, shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
import pandas as pd
from gnss_doppler_lab.cmte_inputs import convert_complex_npz, copy_verified_ds4, sha256, validate_node_table

DEVELOPMENT=frozenset({"DS1","DS2","DS3","DS4"})
CONFIRMATORY=frozenset({"DS7","DS8"})
CANONICAL_COMPLEX_SCENARIOS=("cleanStatic","DS1","DS2","DS3","DS7","DS8")
UNIFORM_COMPLEX=frozenset(CANONICAL_COMPLEX_SCENARIOS)
TAP_ORDER=["E4","E3","E2","E","P","L","L2","L3","L4"]
CONVERTER_SEMANTICS={"magnitude":"hypot(I,Q)","tap_order":TAP_ORDER,"window_seconds":1.0,"stride_seconds":0.5,
 "prompt_relative":True,"prompt_normalization":"per_epoch_tap_magnitude_divided_by_prompt_magnitude_then_window_mean",
 "gap_bridging":False,"source_tensor":"complex_iq[N,9,2]","component_order":["I","Q"]}


def _content_sha(path:Path)->str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(converter_sha:str,wrapper_sha:str|None=None)->str:
 if wrapper_sha is None: wrapper_sha=_content_sha(Path(__file__).resolve(strict=True))
 payload={"converter_content_sha256":converter_sha,"wrapper_content_sha256":wrapper_sha,**CONVERTER_SEMANTICS}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def parse_scenario_mappings(items:Iterable[str],*,tier:str,require_exists:bool=True)->dict[str,Path]:
 if tier not in {"development","confirmatory"}: raise ValueError("tier must be development or confirmatory")
 specs={}
 for item in items:
  if "=" not in item: raise ValueError("scenario mapping must be NAME=/path")
  name,raw=item.split("=",1); name=name.upper(); path=Path(raw)
  if name in specs: raise ValueError(f"duplicate scenario {name}")
  if require_exists:path=path.resolve(strict=True)
  specs[name]=path
 required=DEVELOPMENT if tier=="development" else CONFIRMATORY
 if set(specs)!=required: raise ValueError(f"{tier} requires exactly {sorted(required)}")
 if tier=="development":
  corpus=" ".join(f"{k}={v}" for k,v in specs.items())
  if re.search(r"(?:^|[^A-Z0-9])DS[78](?:[^A-Z0-9]|$)",corpus,re.I): raise ValueError("development paths/tokens must not contain DS7/DS8")
 return specs


def _named_specs(items:Mapping[str,str|Path])->dict[str,Path]:
 if not isinstance(items,Mapping):
  raise TypeError("named complex sources must be a Mapping of canonical scenario name to path")
 if not items: raise ValueError("at least one named complex source required")
 specs={}
 for name,raw in items.items():
  if not isinstance(name,str) or not name: raise ValueError("scenario name must be non-empty")
  if name not in UNIFORM_COMPLEX: raise ValueError(f"unsupported complex scenario {name}")
  if not isinstance(raw,(str,os.PathLike)) or not os.fspath(raw):
   raise ValueError(f"scenario path must be non-empty: {name}")
  specs[name]=Path(raw).resolve(strict=True)
 return specs


def prepare_named_complex_inputs(items:Mapping[str,str|Path],out_dir:str|Path,*,source_metadata:dict|None=None)->dict[str,dict]:
 """Atomically apply one exact converter to cleanStatic/DS1-3/DS7/DS8."""
 specs=_named_specs(items); out=Path(out_dir).resolve()
 if out.exists(): raise FileExistsError("atomic non-overwrite output required")
 out.parent.mkdir(parents=True,exist_ok=True); staging=out.with_name(out.name+f".tmp-{os.getpid()}")
 if staging.exists(): raise FileExistsError(staging)
 staging.mkdir(); manifests={}
 converter_path=Path(convert_complex_npz.__code__.co_filename).resolve(strict=True)
 wrapper_path=Path(__file__).resolve(strict=True); converter_sha=_content_sha(converter_path); wrapper_sha=_content_sha(wrapper_path)
 fingerprint=_fingerprint(converter_sha,wrapper_sha)
 try:
  for name,source in specs.items():
   scenario=name
   csv_path=staging/f"{scenario}_nodes.csv"; manifest_path=staging/f"{scenario}_manifest.json"
   metadata=dict(source_metadata or {}); metadata.update({"source_npz_sha256":sha256(source),"campaign_converter_fingerprint":fingerprint})
   convert_complex_npz(source,csv_path,manifest_path,scenario=scenario,source_metadata=metadata)
   doc=json.loads(manifest_path.read_text()); doc.update({"schema":"gnss-doppler-lab.cmte-a2-input.v2",
    "node_path":str((out/f"{scenario}_nodes.csv").resolve()),"converter_semantics":dict(CONVERTER_SEMANTICS),
    "converter_content_sha256":converter_sha,"wrapper_content_sha256":wrapper_sha,
    "campaign_converter_fingerprint":fingerprint,"source_sha256":sha256(source),"mixed_producer":False,
    "confirmatory_eligible":name in {"DS7","DS8"}})
   manifest_path.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); manifests[name]=doc
  campaign={"schema":"gnss-doppler-lab.cmte-a2-input-campaign.v2","scenarios":list(specs),
   "uniform_converter_semantics":True,"mixed_producer_DS4":False,"converter_content_sha256":converter_sha,
   "wrapper_content_sha256":wrapper_sha,"campaign_converter_fingerprint":fingerprint,
   "window_seconds":1.0,"stride_seconds":0.5,"tap_order":TAP_ORDER,
   "prompt_normalization":CONVERTER_SEMANTICS["prompt_normalization"],
   "source_sha256":{name:sha256(source) for name,source in specs.items()},
   "manifests":{name:f"{name}_manifest.json" for name in manifests}}
  (staging/"manifest.json").write_text(json.dumps(campaign,indent=2,sort_keys=True)+"\n")
  os.replace(staging,out); return manifests
 except Exception:
  shutil.rmtree(staging,ignore_errors=True); raise


def prepare_ds4_sensitivity(source:str|Path,out_dir:str|Path,*,source_manifest:str|Path|None=None,
                            source_metadata:dict|None=None,onset_s:float|None=None)->dict:
 out=Path(out_dir).resolve()
 if out.exists(): raise FileExistsError("atomic non-overwrite output required")
 out.parent.mkdir(parents=True,exist_ok=True); staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir()
 try:
  csv=staging/"DS4_nodes.csv"; manifest=staging/"DS4_manifest.json"
  copy_verified_ds4(source,csv,manifest,source_manifest=source_manifest)
  doc=json.loads(manifest.read_text()); upstream=json.loads(Path(source_manifest).read_text()) if source_manifest else {}
  upstream_meta=upstream.get("source_metadata",{}) or {}
  if not isinstance(upstream_meta,Mapping): raise ValueError("upstream source_metadata must be a mapping")
  supplied_meta=dict(source_metadata or {})
  source_meta=dict(upstream_meta); source_meta.update(supplied_meta)
  if onset_s is not None:
   explicit=float(onset_s)
   supplied=supplied_meta.get("onset_s")
   if supplied is not None and float(supplied)!=explicit: raise ValueError("conflicting authoritative onset metadata")
   source_meta["onset_s"]=explicit
  if source_meta.get("onset_s") is not None:
   onset=float(source_meta["onset_s"])
   if "onset_s" in supplied_meta:
    origin=supplied_meta.get("onset_origin","caller:source_metadata")
    grade=supplied_meta.get("onset_grade","authoritative")
   elif onset_s is not None:
    origin=source_meta.get("onset_origin","explicit:onset_s")
    grade=source_meta.get("onset_grade","authoritative")
   else:
    origin=source_meta.get("onset_origin","upstream_manifest:source_metadata")
    grade=source_meta.get("onset_grade","authoritative")
  elif upstream.get("onset_s") is not None:
   onset=float(upstream["onset_s"])
   origin=upstream.get("onset_origin","upstream_manifest:onset_s")
   grade=upstream.get("onset_grade","authoritative")
  else:
   onset=origin=grade=None
  if onset is not None:
   source_meta.update({"onset_s":onset,"onset_origin":origin,"onset_grade":grade})
  doc.update({"schema":"gnss-doppler-lab.cmte-a2-input.v2","scenario":"DS4",
   "node_path":str((out/"DS4_nodes.csv").resolve()),"tier":"development_sensitivity","mixed_producer":True,
   "confirmatory_eligible":False,"caveat":"historical mixed-producer development sensitivity",
   "onset_s":onset,"onset_origin":origin,"onset_grade":grade,"source_metadata":source_meta})
  manifest.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n")
  campaign={"schema":"gnss-doppler-lab.cmte-a2-input-campaign.v2","scenarios":["DS4"],"tier":"development_sensitivity",
   "mixed_producer":True,"confirmatory_eligible":False,"manifests":{"DS4":"DS4_manifest.json"}}
  (staging/"manifest.json").write_text(json.dumps(campaign,indent=2,sort_keys=True)+"\n")
  os.replace(staging,out); return doc
 except Exception:
  shutil.rmtree(staging,ignore_errors=True); raise


def validate_input_manifest(path:str|Path,node_path:str|Path,scenario:str,*,confirmatory:bool=False,expected_fingerprint:str|None=None)->dict:
 doc=json.loads(Path(path).read_text()); node=Path(node_path).resolve(strict=True)
 if str(doc.get("scenario","")).upper()!=scenario.upper(): raise ValueError("input manifest scenario mismatch")
 if sha256(node)!=doc.get("node_sha256"): raise ValueError("input node checksum mismatch")
 frame=validate_node_table(pd.read_csv(node))
 if confirmatory:
  if doc.get("mixed_producer") or doc.get("confirmatory_eligible") is not True: raise ValueError("mixed/ineligible producer is not confirmatory eligible")
  if not doc.get("campaign_converter_fingerprint"): raise ValueError("confirmatory converter fingerprint missing")
 if expected_fingerprint and doc.get("campaign_converter_fingerprint")!=expected_fingerprint: raise ValueError("campaign converter fingerprint mismatch")
 return {"manifest":doc,"frame":frame}
