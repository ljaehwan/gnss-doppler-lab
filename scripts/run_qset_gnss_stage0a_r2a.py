#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab import qset_stage0a_r2a as Q

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("preregister", "run-variants", "clean-regression", "manifest")); parser.add_argument("--selected", choices=("V1", "V2", "V3")); args = parser.parse_args()
    if args.command == "preregister":
        Q.ARTIFACT.mkdir(parents=True, exist_ok=True); Q.SSD_ROOT.mkdir(parents=True, exist_ok=True)
        baseline = Q.baseline_result(); Q.write_json(Q.ARTIFACT / "preregistration.json", Q.make_preregistration())
        Q.write_json(Q.ARTIFACT / "original_r2_evidence.json", baseline)
        Q.write_json(Q.ARTIFACT / "receiver_variant_matrix.json", {"status": "FROZEN", "variants": Q.VARIANTS})
        Q.write_json(Q.ARTIFACT / "source_binding.json", {"status": "PASS", "base_sha": Q.BASE_SHA, "receiver": Q.file_binding(Q.RECEIVER), "r2_artifact_manifest": Q.file_binding(Q.R2_ARTIFACT / "artifact_manifest_sha256.json"), "normal_model": Q.file_binding(Q.R2_ARTIFACT / "normal_model.json"), "threshold_binding": Q.file_binding(Q.R2_ARTIFACT / "threshold_binding.json")})
        Q.write_json(Q.ARTIFACT / "baseline_receiver_forensics.json", {"status": "PASS", "baseline": baseline, "root_cause_evidence": {"scheduler": "round-robin PRN queue advances after acquisition failure", "runtime_coverage": "tracking of E30/E36 proves reassignment beyond initial E01-E12", "failed_attempt_detail": "not observable because DLOG is compiled out in frozen Release receiver"}})
        attempts = [{"event": "initial_assignment", **row} for row in baseline["events"]["initial_assignments"]] + [{"event": "acquisition_success_tracking_start", **row} for row in baseline["events"]["tracking_starts"]]
        Q.write_csv(Q.ARTIFACT / "acquisition_attempt_timeline.csv", attempts, ["event", "channel", "prn"])
        occupancy = [{"second": second, "panel_size": len(panel), "prns": ";".join(map(str, panel))} for second, panel in baseline["support"]["panels"].items()]
        Q.write_csv(Q.ARTIFACT / "channel_occupancy_timeline.csv", occupancy, ["second", "panel_size", "prns"])
        Q.write_json(Q.ARTIFACT / "access_audit.json", {"status": "PASS_PRE_EXECUTION", "allowed_inputs": ["R2 C-1/C-3/SS-1 decoded IQ, logs, TRACE"], "forbidden_raw_access": {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}, "ss1_raw_redecode": False, "ss1_spoofing_score_operations": 0, "downstream_attack_raw_opened": []})
        Q.write_json(Q.ARTIFACT / "deterministic_reproduction.json", {"status": "FROZEN_PRE_EXECUTION", "commands": ["/usr/bin/python3 scripts/run_qset_gnss_stage0a_r2a.py run-variants", "/usr/bin/python3 scripts/run_qset_gnss_stage0a_r2a.py clean-regression --selected <selected>", "/usr/bin/python3 scripts/verify_qset_gnss_stage0a_r2a.py"], "worker_count": 1, "overwrite": False})
        (Q.ARTIFACT / "README.md").write_text("# Q-SET-GNSS Stage-0A R2a\n\nLocked SS-1 receiver-support root-cause audit. This is not a spoofing detection evaluation. No SS-1 score, ROC, AUC, or morphology is computed. SS-3/SS-5 and all other attack raw payloads remain unopened.\n", encoding="utf-8")
        result = {"status": "PASS_PREREGISTRATION", "baseline_tracked_prns": baseline["support"]["tracked_prns"]}
    elif args.command == "run-variants":
        result = {"V0": Q.baseline_result()}
        for name in ("V1", "V2", "V3"): result[name] = Q.run_variant(name)
        Q.write_json(Q.SSD_ROOT / "variant_results.json", result)
    elif args.command == "clean-regression":
        if not args.selected: parser.error("--selected required")
        result = Q.score_clean_regression(args.selected); Q.write_json(Q.SSD_ROOT / "clean_regression.json", result)
    else:
        result = Q.manifest_for_artifact(); Q.write_json(Q.ARTIFACT / "artifact_manifest_sha256.json", result)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
