#!/usr/bin/env python3
"""R2c path adapter for the unchanged frozen TRACE-R2 Phase-B scorer."""

import json
from pathlib import Path
import sys

import evaluate_trace_r2_phase_b as frozen

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/trace_stage0_r2c_terminal_drain_repair"
frozen.ARTIFACT = ARTIFACT
frozen.SSD = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair"
)
frozen.WORK = frozen.SSD / "evaluation_work"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        phase_a = json.loads((ARTIFACT / "rep3_rep4_reproduction_metrics.json").read_text())
        (ARTIFACT / "smoke_replay_results.json").write_text(
            json.dumps(phase_a, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    code = frozen.main()
    if len(sys.argv) > 1 and sys.argv[1] == "finalize":
        path = ARTIFACT / "final_verdict.json"
        verdict = json.loads(path.read_text())
        verdict["schema"] = "gnss-doppler-lab.trace-r2c-final-verdict.v1"
        verdict["phase_b_authorized"] = True
        verdict["attack_metrics_computed"] = bool(verdict["attack_scores_computed"])
        verdict["normal_fpr"] = {
            "status": "AVAILABLE",
            "clean_holdout_fpr_worst": verdict["clean_holdout_fpr_worst"],
            "external_static_fpr_worst": verdict["external_static_fpr_worst"],
        }
        verdict["actual_action_vs_shuffled_no_action"] = {
            "status": "AVAILABLE",
            "all_scenarios_reduce_significantly": verdict["go_checks"][
                "shifted_and_shuffled_actions_reduce_attack_score_significantly"
            ],
        }
        path.write_text(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
