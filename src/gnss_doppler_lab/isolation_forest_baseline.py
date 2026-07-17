"""Leakage-safe, atomic IsolationForest baseline for three-tap E/P/L features."""
from __future__ import annotations
import csv,hashlib,json,os,tempfile
from dataclasses import asdict,dataclass
from pathlib import Path
from typing import Literal,Sequence
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import IsolationForest
from .tracking_feature_windows import TrackingWindowFeatureRecord
MORPHOLOGY_FEATURE_COLUMNS=["near_sym_mean","near_sym_std","sharp_narrow_mean","sharp_narrow_std","sharp_narrow_slope"]
DYNAMICS_FEATURE_COLUMNS=["doppler_std","doppler_slope","cn0_std","code_err_abs_mean","code_err_std","prompt_mag_cv"]
FEATURE_GROUPS={"morphology-only":MORPHOLOGY_FEATURE_COLUMNS,"dynamics-only":DYNAMICS_FEATURE_COLUMNS,"combined":MORPHOLOGY_FEATURE_COLUMNS+DYNAMICS_FEATURE_COLUMNS}
ALL_COLUMNS=list(TrackingWindowFeatureRecord.__dataclass_fields__); METADATA_COLUMNS=[x for x in ALL_COLUMNS if x not in FEATURE_GROUPS["combined"]]
@dataclass(frozen=True)
class IsolationForestConfig:
 seed:int=42; n_estimators:int=200; contamination:float|Literal["auto"]="auto"
 def validate(self):
  if self.n_estimators<=0: raise ValueError("n_estimators must be positive")
  if self.contamination!="auto" and not 0<float(self.contamination)<=.5: raise ValueError("contamination must be 'auto' or in (0, 0.5]")
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def _ids(v,n):
 x=sorted(v)
 if not x: raise ValueError(f"{n} must contain at least one run_id")
 if len(x)!=len(set(x)): raise ValueError(f"{n} contains duplicate run_id values")
 return x
def _temp(p):
 fd,n=tempfile.mkstemp(prefix=f".{p.name}.",suffix=".tmp",dir=p.parent);os.close(fd);return Path(n)
def run_normal_only_isolation_forest(dataset_path,score_output_path,manifest_output_path,*,train_run_ids,test_run_ids,feature_group="combined",config=None):
 if feature_group not in FEATURE_GROUPS: raise ValueError(f"unknown feature_group {feature_group!r}; choose from {sorted(FEATURE_GROUPS)}")
 dataset,score,manifest=map(Path,(dataset_path,score_output_path,manifest_output_path)); resolved=[p.resolve() for p in (dataset,score,manifest)]
 if len(set(resolved))!=3: raise ValueError("dataset/score/manifest path alias is forbidden")
 train,test=_ids(train_run_ids,"train_run_ids"),_ids(test_run_ids,"test_run_ids")
 if set(train)&set(test): raise ValueError(f"train/test run overlap would leak data: {sorted(set(train)&set(test))}")
 cfg=config or IsolationForestConfig();cfg.validate(); input_sha=_sha(dataset)
 dataset_manifest_path=dataset.with_suffix(".manifest.json")
 if dataset_manifest_path.exists():
  try: dataset_manifest=json.loads(dataset_manifest_path.read_text())
  except (OSError,json.JSONDecodeError) as e: raise ValueError(f"malformed dataset manifest: {e}") from e
  if dataset_manifest.get("output_csv",{}).get("sha256")!=input_sha: raise ValueError("dataset CSV/manifest hash mismatch")
  if dataset_manifest.get("tap_count")!=3 or dataset_manifest.get("tap_layout")!=["E","P","L"]: raise ValueError("dataset manifest is not the current three-tap E/P/L schema")
 with dataset.open(newline="",encoding="utf-8") as h:
  reader=csv.DictReader(h)
  if reader.fieldnames!=ALL_COLUMNS: raise ValueError(f"dataset feature schema mismatch: expected {ALL_COLUMNS}, got {reader.fieldnames}")
  rows=list(reader)
 available={r["run_id"] for r in rows};missing=sorted((set(train)|set(test))-available)
 if missing: raise ValueError(f"requested run_id values are absent from dataset: {missing}")
 selected=sorted((r for r in rows if r["run_id"] in set(train+test)),key=lambda r:(r["run_id"],r["prn"],int(r["channel"]),int(r["segment_index"]),int(r["window_index"])))
 tr=[r for r in selected if r["run_id"] in set(train)];te=[r for r in selected if r["run_id"] in set(test)]
 if not tr or not te: raise ValueError("train and test splits must both contain rows")
 if any(r["label"]!="normal" for r in tr): raise ValueError("normal-only baseline requires every training row label to be 'normal'")
 train_fp={r["source_fingerprint"] for r in tr};test_fp={r["source_fingerprint"] for r in te}
 if "" in train_fp|test_fp: raise ValueError("source_fingerprint must be non-empty")
 if train_fp&test_fp: raise ValueError(f"train/test source content leakage: {sorted(train_fp&test_fp)}")
 features=FEATURE_GROUPS[feature_group]
 def matrix(src):
  try: x=np.asarray([[float(r[c]) for c in features] for r in src],dtype=float)
  except (TypeError,ValueError) as e: raise ValueError(f"feature values must be numeric: {e}") from e
  if x.size==0 or not np.isfinite(x).all(): raise ValueError("feature values must be non-empty and finite")
  return x
 model=IsolationForest(n_estimators=cfg.n_estimators,contamination=cfg.contamination,random_state=cfg.seed,n_jobs=1);model.fit(matrix(tr));scores=-model.decision_function(matrix(selected))
 if not np.isfinite(scores).all(): raise ValueError("anomaly scores must be finite")
 score.parent.mkdir(parents=True,exist_ok=True);manifest.parent.mkdir(parents=True,exist_ok=True);ts,tm=_temp(score),_temp(manifest);cols=METADATA_COLUMNS+["split","anomaly_score"]
 try:
  with ts.open("w",newline="",encoding="utf-8") as h:
   w=csv.DictWriter(h,fieldnames=cols,lineterminator="\n");w.writeheader()
   for r,s in zip(selected,scores,strict=True):
    out={c:r[c] for c in METADATA_COLUMNS};out["split"]="train" if r["run_id"] in set(train) else "test";out["anomaly_score"]=format(float(s),".17g");w.writerow(out)
  if _sha(dataset)!=input_sha: raise ValueError("dataset changed while baseline was running")
  test_labels=sorted({r["label"] for r in te})
  evaluation=("AUC not reported: test labels are normal-only, so no positive attack class exists." if test_labels==["normal"] else "Label-aware evaluation is not implemented in this baseline; test labels include non-normal data and require a separate evaluation.")
  doc={"schema":"gnss-doppler-lab.normal-only-isolation-forest-epl","schema_version":3,"tap_count":3,"tap_layout":["E","P","L"],"estimator":"sklearn.ensemble.IsolationForest","scikit_learn_version":sklearn_version,"dataset":{"path":str(dataset),"sha256":input_sha},"feature_group":feature_group,"feature_columns":features,"metadata_columns":METADATA_COLUMNS,"score_semantics":"higher anomaly_score means more anomalous","score_transform":"anomaly_score = -IsolationForest.decision_function(X)","config":asdict(cfg),"split":{"unit":"receiver run","train_run_ids":train,"test_run_ids":test,"train_row_count":len(tr),"test_row_count":len(te)},"score_csv":{"path":str(score),"row_count":len(selected),"sha256":_sha(ts)},"labels":sorted({r["label"] for r in selected}),"test_labels":test_labels,"evaluation":evaluation}
  tm.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n");manifest.unlink(missing_ok=True);os.replace(ts,score);os.replace(tm,manifest)
 finally: ts.unlink(missing_ok=True);tm.unlink(missing_ok=True)
 return score,manifest
