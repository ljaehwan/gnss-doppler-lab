from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from gnss_doppler_lab.tracking_feature_windows import TrackingWindowFeatureRecord
from gnss_doppler_lab.tracking_feature_dataset import export_tracking_feature_dataset
from gnss_doppler_lab.isolation_forest_baseline import (
    FEATURE_GROUPS,
    IsolationForestConfig,
    run_normal_only_isolation_forest,
)


def _record(run_id: str, window_index: int = 0) -> TrackingWindowFeatureRecord:
    values = {}
    for index, name in enumerate(TrackingWindowFeatureRecord.__dataclass_fields__):
        if name == "run_id": values[name] = run_id
        elif name == "source_fingerprint": values[name] = f"fingerprint-{run_id}"
        elif name == "label": values[name] = "normal"
        elif name == "prn": values[name] = "G05"
        elif name in {"channel", "segment_index", "window_index"}: values[name] = window_index
        elif name in {"sample_rate_hz", "epoch_count"}: values[name] = 10
        else: values[name] = float(index + window_index)
    return TrackingWindowFeatureRecord(**values)


def test_dataset_exporter_reuses_collector_and_is_reproducible(tmp_path: Path, monkeypatch) -> None:
    calls = []
    def fake_collect(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        return [_record(Path(path).name, 1), _record(Path(path).name, 0)]
    def make_source(name):
        run=tmp_path/name; (run/"raw").mkdir(parents=True); (run/"raw"/"epl_tracking_ch_0.mat").write_bytes(name.encode()); (run/"manifest.json").write_text(json.dumps({"tracking":{"raw_directory":"raw"}}))
    make_source("run-a"); make_source("run-b")
    def fake_collect_with_fingerprint(path, **kwargs):
        calls.append((Path(path).name, kwargs))
        from gnss_doppler_lab.tracking_feature_dataset import _source
        fp=_source(Path(path))["source_fingerprint"]
        from dataclasses import replace
        return [replace(_record(Path(path).name, i), source_fingerprint=fp) for i in (1,0)]
    monkeypatch.setattr("gnss_doppler_lab.tracking_feature_dataset.collect_receiver_run_tracking_feature_records", fake_collect_with_fingerprint)
    output = tmp_path / "dataset.csv"
    export_tracking_feature_dataset([tmp_path / "run-b", tmp_path / "run-a"], output_path=output)
    rows = list(csv.DictReader(output.open(newline="", encoding="utf-8")))
    assert [row["run_id"] for row in rows] == ["run-a", "run-a", "run-b", "run-b"]
    assert [(int(row["segment_index"]), int(row["window_index"])) for row in rows] == [(0, 0), (1, 1), (0, 0), (1, 1)]
    assert [name for name, _ in calls] == ["run-a", "run-b"]
    first = output.read_bytes()
    export_tracking_feature_dataset([tmp_path / "run-a", tmp_path / "run-b"], output_path=output)
    assert output.read_bytes() == first


def test_dataset_exporter_rejects_empty_duplicate_and_bad_schema(tmp_path: Path, monkeypatch) -> None:
    with pytest.raises(ValueError, match="at least one"):
        export_tracking_feature_dataset([], output_path=tmp_path / "x.csv")
    with pytest.raises(ValueError, match="duplicate run_id"):
        export_tracking_feature_dataset([tmp_path / "same", tmp_path / "same"], output_path=tmp_path / "x.csv")
    bad=tmp_path/"bad"; (bad/"raw").mkdir(parents=True); (bad/"raw"/"epl_tracking_ch_0.mat").write_bytes(b"bad"); (bad/"manifest.json").write_text(json.dumps({"tracking":{"raw_directory":"raw"}}))
    class BadRecord:
        def to_row(self): return {"run_id": "bad"}
    monkeypatch.setattr("gnss_doppler_lab.tracking_feature_dataset.collect_receiver_run_tracking_feature_records", lambda *a, **k: [BadRecord()])
    with pytest.raises(ValueError, match="feature schema"):
        export_tracking_feature_dataset([tmp_path / "bad"], output_path=tmp_path / "x.csv")


def _dataset(path: Path) -> None:
    fieldnames = list(TrackingWindowFeatureRecord.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_offset, run_id in enumerate(("train-a", "train-b", "holdout")):
            for index in range(8):
                row = _record(run_id, index).to_row()
                for feature in FEATURE_GROUPS["combined"]:
                    row[feature] = float(run_offset * 0.1 + index * 0.01 + fieldnames.index(feature) * 0.001)
                writer.writerow(row)


def test_normal_only_baseline_separates_schema_and_writes_reproducible_scores(tmp_path: Path) -> None:
    dataset = tmp_path / "features.csv"; _dataset(dataset)
    scores = tmp_path / "scores.csv"; manifest = tmp_path / "manifest.json"
    config = IsolationForestConfig(seed=17, n_estimators=25, contamination="auto")
    run_normal_only_isolation_forest(dataset, scores, manifest, train_run_ids=["train-a", "train-b"], test_run_ids=["holdout"], feature_group="dynamics-only", config=config)
    score_rows = list(csv.DictReader(scores.open(newline="", encoding="utf-8")))
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(score_rows) == 24
    assert {row["split"] for row in score_rows} == {"train", "test"}
    assert all(row["label"] == "normal" for row in score_rows)
    assert data["feature_group"] == "dynamics-only"
    assert data["feature_columns"] == FEATURE_GROUPS["dynamics-only"]
    assert not set(data["metadata_columns"]) & set(data["feature_columns"])
    assert "segment_index" in data["metadata_columns"]
    assert data["score_semantics"] == "higher anomaly_score means more anomalous"
    assert data["config"] == {"contamination": "auto", "n_estimators": 25, "seed": 17}
    assert data["split"]["train_run_ids"] == ["train-a", "train-b"]
    first_scores, first_manifest = scores.read_bytes(), manifest.read_bytes()
    run_normal_only_isolation_forest(dataset, scores, manifest, train_run_ids=["train-b", "train-a"], test_run_ids=["holdout"], feature_group="dynamics-only", config=config)
    assert scores.read_bytes() == first_scores
    assert manifest.read_bytes() == first_manifest


def test_baseline_rejects_run_leakage_unknown_group_schema_and_non_normal_training(tmp_path: Path) -> None:
    dataset = tmp_path / "features.csv"; _dataset(dataset)
    args = (dataset, tmp_path / "scores.csv", tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="overlap"):
        run_normal_only_isolation_forest(*args, train_run_ids=["train-a"], test_run_ids=["train-a"], feature_group="combined")
    with pytest.raises(ValueError, match="feature_group"):
        run_normal_only_isolation_forest(*args, train_run_ids=["train-a"], test_run_ids=["holdout"], feature_group="unknown")
    text = dataset.read_text(encoding="utf-8").replace("normal", "spoofed", 1)
    dataset.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="normal-only"):
        run_normal_only_isolation_forest(*args, train_run_ids=["train-a"], test_run_ids=["holdout"], feature_group="morphology-only")


def test_dataset_writes_provenance_manifest_and_rejects_zero_row_run(tmp_path: Path, monkeypatch) -> None:
    run = tmp_path / "run-a"; (run / "raw").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"source":{"sample_rate_hz":10},"tracking":{"raw_directory":"raw"}}))
    (run / "raw" / "epl_tracking_ch_0.mat").write_bytes(b"real-source")
    monkeypatch.setattr("gnss_doppler_lab.tracking_feature_dataset.collect_receiver_run_tracking_feature_records", lambda *a, **k: [])
    with pytest.raises(ValueError, match="zero rows.*run-a"):
        export_tracking_feature_dataset([run], output_path=tmp_path / "features.csv")


def test_baseline_rejects_path_alias_and_source_content_leakage(tmp_path: Path) -> None:
    dataset = tmp_path / "features.csv"; _dataset(dataset)
    with pytest.raises(ValueError, match="path alias"):
        run_normal_only_isolation_forest(dataset, dataset, tmp_path / "m.json", train_run_ids=["train-a"], test_run_ids=["holdout"])


def test_baseline_manifest_hashes_score_and_label_aware_evaluation(tmp_path: Path) -> None:
    dataset = tmp_path / "features.csv"; _dataset(dataset)
    scores = tmp_path / "scores.csv"; manifest = tmp_path / "scores.manifest.json"
    run_normal_only_isolation_forest(dataset, scores, manifest, train_run_ids=["train-a", "train-b"], test_run_ids=["holdout"])
    data = json.loads(manifest.read_text())
    import hashlib
    assert data["score_csv"]["sha256"] == hashlib.sha256(scores.read_bytes()).hexdigest()
    assert "AUC" in data["evaluation"] and "normal" in data["evaluation"]


def test_baseline_rejects_absent_duplicate_nan_malformed_and_fingerprint_leakage(tmp_path: Path) -> None:
    dataset=tmp_path/"features.csv"; _dataset(dataset); args=(dataset,tmp_path/"scores.csv",tmp_path/"manifest.json")
    with pytest.raises(ValueError,match="absent"):
        run_normal_only_isolation_forest(*args,train_run_ids=["missing"],test_run_ids=["holdout"])
    with pytest.raises(ValueError,match="duplicate"):
        run_normal_only_isolation_forest(*args,train_run_ids=["train-a","train-a"],test_run_ids=["holdout"])
    rows=list(csv.DictReader(dataset.open())); fields=list(rows[0]); rows[0][FEATURE_GROUPS["combined"][0]]="nan"
    with dataset.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match="finite"):
        run_normal_only_isolation_forest(*args,train_run_ids=["train-a"],test_run_ids=["holdout"])
    _dataset(dataset); rows=list(csv.DictReader(dataset.open())); fields=list(rows[0]); [row.pop(fields[-1]) for row in rows]
    with dataset.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields[:-1]);w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match="schema mismatch"):
        run_normal_only_isolation_forest(*args,train_run_ids=["train-a"],test_run_ids=["holdout"])
    _dataset(dataset); text=dataset.read_text().replace("fingerprint-holdout","fingerprint-train-a") ; dataset.write_text(text)
    with pytest.raises(ValueError,match="source content leakage"):
        run_normal_only_isolation_forest(*args,train_run_ids=["train-a"],test_run_ids=["holdout"])


def test_baseline_evaluation_text_reflects_attack_test_label(tmp_path: Path) -> None:
    dataset=tmp_path/"features.csv"; _dataset(dataset); dataset.write_text(dataset.read_text().replace("holdout,", "holdout,", 1))
    rows=list(csv.DictReader(dataset.open())); fields=list(rows[0]);
    for row in rows:
        if row["run_id"]=="holdout": row["label"]="attack"
    with dataset.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    manifest=tmp_path/"manifest.json"
    run_normal_only_isolation_forest(dataset,tmp_path/"scores.csv",manifest,train_run_ids=["train-a"],test_run_ids=["holdout"] )
    assert "not implemented" in json.loads(manifest.read_text())["evaluation"]


def test_baseline_score_direction_and_failed_pair_publish_removes_stale_manifest(tmp_path: Path, monkeypatch) -> None:
    dataset=tmp_path/"features.csv"; _dataset(dataset); scores=tmp_path/"scores.csv"; manifest=tmp_path/"manifest.json"
    class FakeForest:
        def __init__(self,**kwargs): pass
        def fit(self,x): return self
        def decision_function(self,x):
            import numpy as np
            return np.arange(len(x),dtype=float)
    monkeypatch.setattr("gnss_doppler_lab.isolation_forest_baseline.IsolationForest",FakeForest)
    run_normal_only_isolation_forest(dataset,scores,manifest,train_run_ids=["train-a"],test_run_ids=["holdout"] )
    vals=[float(r["anomaly_score"]) for r in csv.DictReader(scores.open())]
    assert vals[0]==0.0 and vals[-1]==-(len(vals)-1)
    real_replace=__import__("os").replace
    def fail_manifest(src,dst):
        if Path(dst)==manifest: raise OSError("publish failure")
        return real_replace(src,dst)
    monkeypatch.setattr("gnss_doppler_lab.isolation_forest_baseline.os.replace",fail_manifest)
    with pytest.raises(OSError,match="publish failure"):
        run_normal_only_isolation_forest(dataset,scores,manifest,train_run_ids=["train-a"],test_run_ids=["holdout"] )
    assert not manifest.exists()


def test_baseline_rejects_stale_dataset_manifest_hash(tmp_path: Path) -> None:
    dataset=tmp_path/"features.csv"; _dataset(dataset)
    dataset.with_suffix(".manifest.json").write_text(json.dumps({"tap_count":3,"tap_layout":["E","P","L"],"output_csv":{"sha256":"bad"}}))
    with pytest.raises(ValueError,match="hash mismatch"):
        run_normal_only_isolation_forest(dataset,tmp_path/"scores.csv",tmp_path/"m.json",train_run_ids=["train-a"],test_run_ids=["holdout"] )
