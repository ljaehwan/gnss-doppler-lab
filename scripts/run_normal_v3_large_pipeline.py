#!/usr/bin/env python3
"""Run the normal-v3 300-candidate GPS L1 C/A generation pipeline.

For each candidate row this script:
1. writes a strict RF YAML config;
2. generates 5-minute gps-sdr-sim s8 IQ;
3. runs GNSS-SDR and normalizes tracking MAT outputs;
4. exports per-run E/P/L tracking-window features;
5. deletes bulky transient IQ and raw receiver MAT files after feature extraction.

After all selected runs, it builds combined PRN-node and receiver-graph tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gnss_doppler_lab.gnss_sdr import run_receiver  # noqa: E402
from gnss_doppler_lab.normal_multi_prn_dataset import export_normal_multi_prn_dataset  # noqa: E402
from gnss_doppler_lab.tracking_feature_dataset import export_tracking_feature_dataset  # noqa: E402

DEFAULT_RUN_INDEX = Path("configs/generated/normal_v3_large_300/run_index.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/model_datasets/normal_v3_large_300")
DEFAULT_RF_ROOT = Path("artifacts/rf_runs/normal_v3_large_300")
DEFAULT_RECEIVER_ROOT = Path("artifacts/receiver_runs/normal_v3_large_300")
DEFAULT_SIM = Path(".tools/gps-sdr-sim-src/gps-sdr-sim")


def load_rows(path: Path, *, limit: int | None, only_split: str | None, resume_after: str | None) -> list[dict[str, str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if only_split:
        rows = [r for r in rows if r.get("split") == only_split]
    if resume_after:
        seen = False
        kept = []
        for row in rows:
            if seen:
                kept.append(row)
            elif row["run_id"] == resume_after:
                seen = True
        if not seen:
            raise SystemExit(f"--resume-after run_id not found: {resume_after}")
        rows = kept
    if limit is not None:
        rows = rows[:limit]
    return rows


def actual_run_id(row: dict[str, str]) -> str:
    compact_utc = row["utc"].replace("-", "").replace(":", "")
    return f"{row['run_id']}_{compact_utc}"


def write_rf_config(row: dict[str, str], config_dir: Path, rf_root: Path, simulator: Path) -> Path:
    run_id = row["run_id"]
    utc = row["utc"]
    nav = row["rinex_nav"]
    lat = float(row["latitude_deg"])
    lon = float(row["longitude_deg"])
    alt = float(row["altitude_m"])
    duration = int(row["duration_seconds"])
    # Paths in YAML are resolved relative to config path, so use paths relative to config_dir.
    def rel(p: Path | str) -> str:
        return os.path.relpath((REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p), config_dir.resolve())

    sim_for_yaml = os.path.relpath((REPO_ROOT / simulator).resolve() if not simulator.is_absolute() else simulator.resolve(), REPO_ROOT)
    text = f"""version: 1
scenario:
  name: {run_id}
  constellation: GPS
  signal: L1CA
  utc: {utc}
  duration_seconds: {duration}
  position:
    type: static
    latitude_deg: {lat:.8f}
    longitude_deg: {lon:.8f}
    altitude_m: {alt:.3f}
input:
  rinex_nav: {rel(nav)}
output:
  root: {rel(rf_root)}
  rf_sample_rate_hz: 2600000
  sample_format: s8_iq
simulator:
  executable: {sim_for_yaml}
"""
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{run_id}.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def run_cmd(cmd: list[str], *, timeout: int, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def assert_run_id_from_manifest(manifest_path: Path, expected_run_id: str) -> None:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    if doc.get("run_id", doc.get("receiver_run_id")) != expected_run_id:
        raise RuntimeError(f"manifest run_id mismatch for {expected_run_id}: {manifest_path}")


def delete_transients(rf_manifest: Path, receiver_manifest: Path, *, keep_iq: bool, keep_raw: bool) -> dict[str, object]:
    deleted: list[str] = []
    if not keep_iq:
        rf_doc = json.loads(rf_manifest.read_text(encoding="utf-8"))
        iq_path = rf_manifest.parent / rf_doc.get("iq", {}).get("path", "gps_l1ca_s8_iq.bin")
        if iq_path.exists():
            iq_path.unlink()
            deleted.append(str(iq_path))
    if not keep_raw:
        rec_doc = json.loads(receiver_manifest.read_text(encoding="utf-8"))
        raw_name = rec_doc.get("tracking", {}).get("raw_directory", "raw")
        raw_dir = receiver_manifest.parent / raw_name
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            deleted.append(str(raw_dir))
    return {"deleted": deleted, "keep_iq": keep_iq, "keep_raw": keep_raw}


def concatenate_feature_csvs(paths: Iterable[Path], out_csv: Path, split_by_run: dict[str, str]) -> tuple[int, Path]:
    paths = list(paths)
    if not paths:
        raise RuntimeError("no per-run feature CSVs to concatenate")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = out_csv.with_suffix(".manifest.json")
    total = 0
    fieldnames: list[str] | None = None
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as out:
        writer = None
        for path in paths:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                    writer = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
                    writer.writeheader()
                elif list(reader.fieldnames or []) != fieldnames:
                    raise RuntimeError(f"feature schema mismatch while concatenating: {path}")
                assert writer is not None
                for row in reader:
                    run_id = row.get("run_id")
                    if run_id not in split_by_run:
                        raise RuntimeError(f"unknown run_id in feature row: {run_id}")
                    writer.writerow(row)
                    total += 1
    os.replace(tmp, out_csv)
    manifest.write_text(json.dumps({
        "schema": "gnss-doppler-lab.normal-v3-large-combined-tracking-features",
        "row_count": total,
        "source_csv_count": len(paths),
        "path": str(out_csv),
        "note": "Strict TrackingWindowFeatureRecord schema; split is applied to node/graph tables after conversion.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return total, manifest


def annotate_split(csv_path: Path, split_by_run: dict[str, str]) -> int:
    tmp = csv_path.with_suffix(csv_path.suffix + ".split.tmp")
    count = 0
    with csv_path.open(newline="", encoding="utf-8") as inp, tmp.open("w", newline="", encoding="utf-8") as out:
        reader = csv.DictReader(inp)
        fields = list(reader.fieldnames or [])
        if "split" not in fields:
            fields.append("split")
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            row["split"] = split_by_run.get(row.get("run_id", ""), "unknown")
            writer.writerow(row)
            count += 1
    os.replace(tmp, csv_path)
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-index", type=Path, default=DEFAULT_RUN_INDEX)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--rf-root", type=Path, default=DEFAULT_RF_ROOT)
    ap.add_argument("--receiver-root", type=Path, default=DEFAULT_RECEIVER_ROOT)
    ap.add_argument("--simulator", type=Path, default=DEFAULT_SIM)
    ap.add_argument("--gnss-sdr", default="gnss-sdr")
    ap.add_argument("--channel-count", type=int, default=11)
    ap.add_argument("--receiver-timeout", type=int, default=1200)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-split", choices=["train", "val", "test"])
    ap.add_argument("--resume-after")
    ap.add_argument("--keep-iq", action="store_true")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--force", action="store_true", help="remove existing per-run RF/receiver outputs before rerun")
    args = ap.parse_args(argv)

    run_index = args.run_index.resolve()
    rows = load_rows(run_index, limit=args.limit, only_split=args.only_split, resume_after=args.resume_after)
    if not rows:
        raise SystemExit("no rows selected")
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    config_dir = out / "rf_configs"
    per_run_feature_dir = out / "per_run_tracking_features"
    per_run_feature_dir.mkdir(parents=True, exist_ok=True)
    rf_root = args.rf_root.resolve(); receiver_root = args.receiver_root.resolve()
    rf_root.mkdir(parents=True, exist_ok=True); receiver_root.mkdir(parents=True, exist_ok=True)
    split_by_run = {actual_run_id(r): r.get("split", "unknown") for r in rows}
    candidate_to_actual = {r["run_id"]: actual_run_id(r) for r in rows}
    run_reports=[]; feature_csvs=[]

    started = time.time()
    for i, row in enumerate(rows, 1):
        candidate_id = row["run_id"]
        run_id = candidate_to_actual[candidate_id]
        print(f"[{i}/{len(rows)}] {candidate_id} -> {run_id}", flush=True)
        rf_dir = rf_root / run_id
        rec_dir = receiver_root / run_id
        if args.force:
            shutil.rmtree(rf_dir, ignore_errors=True)
            shutil.rmtree(rec_dir, ignore_errors=True)
        feature_csv = per_run_feature_dir / f"{run_id}.csv"
        if feature_csv.exists() and rec_dir.exists():
            feature_csvs.append(feature_csv)
            run_reports.append({"candidate_id": candidate_id, "run_id": run_id, "status": "skipped_existing_feature"})
            continue

        cfg = write_rf_config(row, config_dir, rf_root, args.simulator)
        gen = run_cmd([sys.executable, "scripts/generate_iq.py", "generate", str(cfg)], timeout=1800)
        if gen.returncode != 0:
            raise RuntimeError(f"IQ generation failed for {candidate_id}\nSTDOUT:\n{gen.stdout}\nSTDERR:\n{gen.stderr}")
        rf_manifest = Path(gen.stdout.strip().splitlines()[-1]).resolve()
        assert_run_id_from_manifest(rf_manifest, run_id)

        receiver_manifest = run_receiver(rf_manifest, receiver_root, executable=args.gnss_sdr, channel_count=args.channel_count, timeout_seconds=args.receiver_timeout)
        assert_run_id_from_manifest(receiver_manifest, run_id)

        export_tracking_feature_dataset([receiver_manifest.parent], output_path=feature_csv, manifest_output_path=feature_csv.with_suffix(".manifest.json"), window_s=1.0, stride_s=0.5, min_epochs=4, label="normal")
        feature_csvs.append(feature_csv)
        cleanup = delete_transients(rf_manifest, receiver_manifest, keep_iq=args.keep_iq, keep_raw=args.keep_raw)
        run_reports.append({"candidate_id": candidate_id, "run_id": run_id, "status": "ok", "rf_manifest": str(rf_manifest), "receiver_manifest": str(receiver_manifest), "feature_csv": str(feature_csv), **cleanup})

    combined_features = out / "normal_tracking_features.csv"
    feature_rows, feature_manifest = concatenate_feature_csvs(feature_csvs, combined_features, split_by_run)
    node_csv, graph_csv, multi_manifest = export_normal_multi_prn_dataset(combined_features, output_dir=out / "normal_multi_prn_morphology_dynamics_v3", stride_s=0.5, min_prns_per_graph=2)
    node_rows = annotate_split(node_csv, split_by_run)
    graph_rows = annotate_split(graph_csv, split_by_run)
    summary = {
        "schema": "gnss-doppler-lab.normal-v3-large-pipeline-run-summary",
        "run_index": str(run_index),
        "selected_run_count": len(rows),
        "feature_rows": feature_rows,
        "node_rows": node_rows,
        "graph_rows": graph_rows,
        "combined_features": str(combined_features),
        "combined_features_manifest": str(feature_manifest),
        "node_csv": str(node_csv),
        "graph_csv": str(graph_csv),
        "multi_prn_manifest": str(multi_manifest),
        "elapsed_seconds": round(time.time() - started, 3),
        "keep_iq": args.keep_iq,
        "keep_raw": args.keep_raw,
        "reports": run_reports,
    }
    summary_path = out / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
