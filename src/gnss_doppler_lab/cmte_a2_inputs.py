"""CMTE-A2 canonical input wrappers and tier/provenance validation."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd

from gnss_doppler_lab.cmte_inputs import convert_complex_npz, copy_verified_ds4, sha256, validate_node_table

DEVELOPMENT=frozenset({"DS1","DS2","DS3","DS4"})
CONFIRMATORY=frozenset({"DS7","DS8"})
UNIFORM_COMPLEX=frozenset({"CLEANSTATIC","DS1","DS2","DS3","DS7","DS8"})


def parse_scenario_mappings(items:Iterable[str],*,tier:str,require_exists:bool=True)->dict[str,Path]:
    if tier not in {"development","confirmatory"}: raise ValueError("tier must be development or confirmatory")
    specs={}
    for item in items:
        if "=" not in item: raise ValueError("scenario mapping must be NAME=/path")
        name,raw=item.split("=",1); name=name.upper(); path=Path(raw)
        if name in specs: raise ValueError(f"duplicate scenario {name}")
        if require_exists: path=path.resolve(strict=True)
        specs[name]=path
    required=DEVELOPMENT if tier=="development" else CONFIRMATORY
    if set(specs)!=required: raise ValueError(f"{tier} requires exactly {sorted(required)}")
    if tier=="development":
        corpus=" ".join(f"{k}={v}" for k,v in specs.items())
        if re.search(r"(?:^|[^A-Z0-9])DS[78](?:[^A-Z0-9]|$)",corpus,re.I):
            raise ValueError("development paths/tokens must not contain DS7/DS8")
    return specs


def prepare_named_complex_inputs(items:Iterable[str],out_dir:str|Path,*,source_metadata:dict|None=None)->dict[str,dict]:
    """Apply the same base converter semantics to named complex NPZ inputs."""
    specs={}
    for item in items:
        if "=" not in item: raise ValueError("mapping must be NAME=/complex.npz")
        name,raw=item.split("=",1); name=name.upper()
        if name in specs: raise ValueError(f"duplicate {name}")
        if name not in UNIFORM_COMPLEX: raise ValueError(f"unsupported complex scenario {name}")
        specs[name]=Path(raw).resolve(strict=True)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=False); manifests={}
    producer=None
    for name,source in sorted(specs.items()):
        scenario="cleanStatic" if name=="CLEANSTATIC" else name
        csv_path=out/f"{scenario}_nodes.csv"; manifest_path=out/f"{scenario}_manifest.json"
        convert_complex_npz(source,csv_path,manifest_path,scenario=scenario,source_metadata=source_metadata)
        doc=json.loads(manifest_path.read_text()); doc["schema"]="gnss-doppler-lab.cmte-a2-input.v1"
        doc["converter_semantics"]={"magnitude":"hypot(I,Q)","tap_order":["E4","E3","E2","E","P","L","L2","L3","L4"],
                                    "window_seconds":1.,"stride_seconds":.5,"prompt_relative":True,"gap_bridging":False}
        signature=json.dumps(doc["converter_semantics"],sort_keys=True)
        if producer is None: producer=signature
        if signature!=producer: raise ValueError("complex producer semantics differ across scenarios")
        manifest_path.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); manifests[name]=doc
    campaign={"schema":"gnss-doppler-lab.cmte-a2-input-campaign.v1","scenarios":sorted(specs),
              "uniform_converter_semantics":True,"mixed_producer_DS4":False,
              "manifests":{name:f"{('cleanStatic' if name=='CLEANSTATIC' else name)}_manifest.json" for name in manifests}}
    (out/"manifest.json").write_text(json.dumps(campaign,indent=2,sort_keys=True)+"\n")
    return manifests


def prepare_ds4_sensitivity(source:str|Path,out_dir:str|Path,*,source_manifest:str|Path|None=None)->dict:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=False)
    csv=out/"DS4_nodes.csv"; manifest=out/"DS4_manifest.json"
    copy_verified_ds4(source,csv,manifest,source_manifest=source_manifest)
    doc=json.loads(manifest.read_text()); doc.update({"schema":"gnss-doppler-lab.cmte-a2-input.v1","scenario":"DS4",
        "tier":"development_sensitivity","mixed_producer":True,"confirmatory_eligible":False,
        "caveat":"historical mixed-producer development sensitivity"})
    manifest.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); return doc


def validate_input_manifest(path:str|Path,node_path:str|Path,scenario:str,*,confirmatory:bool=False)->dict:
    doc=json.loads(Path(path).read_text()); node=Path(node_path).resolve(strict=True)
    if str(doc.get("scenario","")).upper()!=scenario.upper(): raise ValueError("input manifest scenario mismatch")
    if sha256(node)!=doc.get("node_sha256"): raise ValueError("input node checksum mismatch")
    frame=validate_node_table(pd.read_csv(node))
    if confirmatory and doc.get("mixed_producer"): raise ValueError("mixed producer is not confirmatory eligible")
    return {"manifest":doc,"frame":frame}
