#!/usr/bin/env python3
"""Run MOSAIC-GNSS Stage-0A foundation inventory/alignment checks."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mosaic_alignment import (  # noqa: E402
    navigation_bit_provenance,
    sample_bounds_status,
    stable_epoch_rows,
    validate_causal_alignment,
)
from gnss_doppler_lab.mosaic_iq_injector import design_sha256  # noqa: E402

ART = ROOT / "artifacts/mosaic_stage0ab_foundation"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a")
DATASETS = {
    "TEXBAT.cleanStatic": ("texbat_cleanstatic", 25_000_000.0),
    "OAKBAT.cleanStatic": ("oakbat_cleanstatic", 5_000_000.0),
}


def dump_json(name: str, value: object) -> None:
    (ART / name).parent.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_manifest(slug: str, loop: str = "slow", rep: int = 1) -> dict[str, object] | None:
    path = SSD / slug / loop / f"rep{rep}" / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def stat_binding(path: str | None) -> dict[str, object]:
    if not path:
        return {"exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    return {"path": str(p), "exists": True, "device": st.st_dev, "inode": st.st_ino, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def freeze_design() -> list[dict[str, object]]:
    rows = []
    # Compact stratified preregistration, <=40 per dataset, not a Cartesian product.
    taus = [0.0, -0.05, 0.05, -0.10, 0.10, -0.25, 0.25]
    dfs = [0.0, -5.0, 5.0, -25.0, 25.0, -50.0, 50.0]
    rhos = [-10.0, -6.0, -3.0, 0.0]
    phis = [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469]
    for dataset in DATASETS:
        for i in range(28):
            rows.append({
                "case_id": f"{dataset}.single.{i:02d}",
                "dataset": dataset,
                "mode": "single_prn",
                "delta_tau_chips": taus[i % len(taus)],
                "delta_f_hz": dfs[(i * 2) % len(dfs)],
                "rho_db": rhos[(i * 3) % len(rhos)],
                "delta_phi_rad": phis[(i * 5) % len(phis)],
                "view_before_results": True,
            })
        for i in range(8):
            rows.append({
                "case_id": f"{dataset}.four.{i:02d}",
                "dataset": dataset,
                "mode": "four_prn_diagnostic_after_single_prn_pass_only",
                "delta_tau_chips": taus[(i + 1) % len(taus)],
                "delta_f_hz": dfs[(i + 3) % len(dfs)],
                "rho_db": rhos[i % len(rhos)],
                "delta_phi_rad": phis[i % len(phis)],
                "view_before_results": True,
            })
    return rows


def main() -> int:
    started = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "plots").mkdir(exist_ok=True)
    head = os.popen("git rev-parse HEAD").read().strip()
    dump_json("source_commit.json", {"repo": str(ROOT), "head": head, "branch": os.popen("git branch --show-current").read().strip()})
    dump_json("config.json", {"schema": "gnss-doppler-lab.mosaic-stage0ab-config.v1", "artifact_root": str(ART), "datasets": list(DATASETS), "attack_scenarios_used": False, "neural_training": False})
    dump_json("preregistration.json", {"allowed_verdicts": ["GO_FOR_MOSAIC_NEURAL_STAGE1", "NO_GO_MOSAIC_INJECTOR_PHYSICS", "INCONCLUSIVE_INPUT_OR_ALIGNMENT", "INCONCLUSIVE_INPUT_OR_RECEIVER_UNAVAILABLE", "INCONCLUSIVE_TRACKER_RAW_ALIGNMENT", "INCONCLUSIVE_NAVIGATION_BIT_PROVENANCE"], "thresholds": {"caf_center_delay_chip_abs": 0.125, "caf_center_doppler_hz_abs": 50, "tap_cosine_median_min": 0.90, "tap_spearman_median_min": 0.90}, "attack_scores_computed": False})
    dump_json("environment_inventory.json", {"platform": platform.platform(), "python": sys.version, "ssd_artifact_root": str(SSD), "ssd_artifact_root_exists": SSD.exists()})

    data_inv: dict[str, object] = {}
    raw_bind: dict[str, object] = {}
    field_contract: dict[str, object] = {"native_trace_schema": "TRACE TRC1MS02 v2", "required_fields_present_in_dump_schema": True, "raw_iq_sample_format_assumption_from_existing_manifest": "interleaved signed int8 complex", "unavailable_fields": ["decoded navigation bit sequence", "receiver replay of MOSAIC injected IQ"]}
    split_audit: dict[str, object] = {"chronological_nonoverlap_short_segments": True, "guard_interval_s": 5.0, "attack_data_used": False, "datasets": {}}
    align_metrics: dict[str, object] = {"schema": "gnss-doppler-lab.mosaic-stage0a-alignment.v1", "datasets": {}, "stage0a_pass": False}
    all_rows: list[dict[str, object]] = []
    receiver_unavailable = False

    for dataset, (slug, expected_fs) in DATASETS.items():
        manifest = load_manifest(slug)
        if manifest is None:
            receiver_unavailable = True
            data_inv[dataset] = {"status": "MISSING_MANIFEST"}
            continue
        raw = manifest.get("raw_iq", {})
        recv = manifest.get("receiver", {})
        raw_stat = stat_binding(raw.get("path"))
        recv_stat = stat_binding(recv.get("path"))
        available = bool(raw_stat.get("exists") and recv_stat.get("exists") and manifest.get("raw_stable") and manifest.get("receiver_stable"))
        receiver_unavailable |= not available
        data_inv[dataset] = {"manifest": str(SSD / slug / "slow/rep1/manifest.json"), "raw_iq": raw | {"stat_now": raw_stat}, "receiver": recv | {"stat_now": recv_stat}, "handoff": manifest.get("handoff"), "available": available, "sample_rate_hz_expected": expected_fs}
        raw_bind[dataset] = {"raw_manifest_sha256": raw.get("sha256"), "raw_stat_binding": raw_stat, "receiver_sha256": recv.get("sha256"), "receiver_stat_binding": recv_stat}
        dump_dir = SSD / slug / "slow/rep1"
        try:
            table = stable_epoch_rows(dump_dir, dataset=dataset, recording=slug, limit=120)
            rows = table.rows
            all_rows.extend(rows)
            bounds = sample_bounds_status(rows, int(raw.get("size_bytes", 0)))
            causal = validate_causal_alignment([dump_dir])
            prns = sorted({r["prn"] for r in rows})
            align_metrics["datasets"][dataset] = {"status": "TRACE_SCHEMA_AND_BOUNDS_PASS_RAW_RECORRELATION_NOT_RUN", "sample_rate_hz": table.sample_rate_hz, "stable_epoch_rows_sampled": len(rows), "stable_prns_sampled": prns, "stable_prn_count": len(prns), "raw_sample_bounds": bounds, "causal_alignment": causal, "local_raw_recorrelation": {"status": "NOT_RUN_FAIL_CLOSED", "reason": "Navigation bit/provenance and exact IF/baseband sign contract not sufficient to validate reconstructed complex taps without tuning."}}
            split_audit["datasets"][dataset] = {"sampled_rows": len(rows), "raw_sample_min": min((r["raw_sample_start"] for r in rows), default=None), "raw_sample_max": max((r["raw_sample_end"] for r in rows), default=None), "raw_overlap_in_sample": False}
        except Exception as exc:  # fail-closed inventory evidence
            align_metrics["datasets"][dataset] = {"status": "FAIL", "error": repr(exc)}

    nav = navigation_bit_provenance(all_rows)
    stage0a_structural = (not receiver_unavailable) and all(d.get("raw_sample_bounds", {}).get("status") == "PASS" and d.get("causal_alignment", {}).get("status") == "PASS" for d in align_metrics["datasets"].values())
    align_metrics["stage0a_structural_trace_alignment_pass"] = bool(stage0a_structural)
    align_metrics["stage0a_pass"] = False
    align_metrics["stage0a_terminal_reason"] = "raw-to-reconstructed complex tap agreement not established; Stage-0B receiver-in-loop not authorized"

    with gzip.open(ART / "stage0a_per_epoch_sample.csv.gz", "wt", newline="") as f:
        fields = ["dataset", "recording", "prn", "channel", "loop_sequence", "raw_sample_start", "raw_sample_end", "receiver_timestamp_s", "cn0_db_hz", "lock_tracking_quality"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})

    design = freeze_design()
    dump_json("injection_design.json", design)
    dump_json("injection_design_sha256.json", {"sha256": design_sha256(design), "case_count": len(design), "frozen_before_recovery": True})
    dump_json("data_inventory.json", data_inv)
    dump_json("receiver_field_contract.json", field_contract | {"navigation_bit_provenance": nav})
    dump_json("raw_source_binding.json", raw_bind)
    dump_json("clean_split_audit.json", split_audit)
    dump_json("stage0a_alignment_metrics.json", align_metrics)
    # Placeholder CSVs are explicit NOT_RUN records, not synthetic metrics.
    for name in ["injection_physics_metrics.csv", "parameter_recovery_metrics.csv", "residual_caf_metrics.csv"]:
        (ART / name).write_text("status,reason\nNOT_RUN,Stage-0A did not pass and/or navigation-bit provenance unavailable; no receiver-in-loop injection executed\n")
    dump_json("physical_controls.json", {"all_required_controls_pass": False, "status": "NOT_RUN", "reason": "Stage-0B not authorized", "controls": {"identity_replay": "NOT_RUN", "zero_amplitude": "NOT_RUN", "common_gain_scaling": "NOT_RUN", "global_phase_rotation": "NOT_RUN", "awgn_0p5x_1x_2x": "NOT_RUN", "navigation_bit_sign_reversal": "NOT_RUN"}})

    if receiver_unavailable:
        verdict = "INCONCLUSIVE_INPUT_OR_RECEIVER_UNAVAILABLE"
    elif nav["status"] == "UNAVAILABLE":
        verdict = "INCONCLUSIVE_NAVIGATION_BIT_PROVENANCE"
    else:
        verdict = "INCONCLUSIVE_TRACKER_RAW_ALIGNMENT"
    final = {"schema": "gnss-doppler-lab.mosaic-stage0ab-final-verdict.v1", "verdict": verdict, "go": False, "stage0a_pass": False, "stage0b_run": False, "attack_scores_computed": False, "neural_training": False, "reason": align_metrics["stage0a_terminal_reason"] if verdict != "INCONCLUSIVE_NAVIGATION_BIT_PROVENANCE" else nav["reason"], "recommended_next_action": "Obtain decoded/validated cleanStatic navigation-bit provenance and rerun Stage-0A raw re-correlation without changing gates."}
    dump_json("final_verdict.json", final)
    dump_json("resource_profile.json", {"runtime_s": time.time() - started, "rows_sampled": len(all_rows)})
    (ART / "README.md").write_text("# MOSAIC Stage-0A/0B Foundation Artifacts\n\nFail-closed foundation bundle. No neural training, attack scoring, or synthetic raw-IQ persistence was performed. See `final_verdict.json`.\n")
    # Manifest last, excluding itself.
    manifest_rows = {}
    for p in sorted(ART.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest_sha256.json":
            manifest_rows[str(p.relative_to(ART))] = hashlib.sha256(p.read_bytes()).hexdigest()
    dump_json("artifact_manifest_sha256.json", manifest_rows)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
