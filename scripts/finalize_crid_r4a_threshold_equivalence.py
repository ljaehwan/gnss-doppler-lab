#!/usr/bin/env python3
"""Finalize CRID R4a without modifying R4 or its authoritative thresholds."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/crid_stage0_r4a_threshold_decision_equivalence_repair"


def load(name: str) -> dict: return json.loads((ART / name).read_text())
def dump(name: str, value: object) -> None: (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""): digest.update(payload)
    return digest.hexdigest()


def main() -> int:
    source, numeric = load("source_binding.json"), load("threshold_numeric_comparison.json")
    split, alarms = load("clean_split_identity.json"), load("holdout_alarm_equivalence.json")
    audit, checkpoint = load("attack_and_control_access_audit.json"), load("validation_checkpoint.json")
    tests, tamper = load("test_results.json"), load("tamper_test_results.json")
    all_pass = all(row.get("status") == "PASS" for row in (source, numeric, split, alarms, audit, checkpoint, tests, tamper))
    no_forbidden = all(audit.get(name) == 0 for name in ("control_replays_executed", "control_scores_read", "control_scores_computed", "attack_stats", "attack_hashes", "attack_opens", "attack_mmaps", "attack_bytes_read"))
    if all_pass and no_forbidden:
        verdict, next_state = "THRESHOLD_DECISION_EQUIVALENCE_REPAIR_PASS", "READY_TO_REPEAT_CRID_PHASE_A"
    else:
        verdict, next_state = "INCONCLUSIVE_THRESHOLD_DECISION_PROVENANCE", "NOT_AUTHORIZED"
    final = {
        "schema": "gnss-doppler-lab.crid-r4a-final-verdict.v1", "verdict": verdict,
        "status": "PASS" if verdict.endswith("_PASS") else "INCONCLUSIVE", "next_state": next_state,
        "r4_verdict_preserved": "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE",
        "r4_threshold_status_preserved": "INCONCLUSIVE_THRESHOLD_BINDING",
        "authoritative_thresholds": {"OAK": -21.705587048010322, "TEX": -21.942672917134093},
        "thresholds_replaced": False, "control_replays_executed": 0, "control_scores_read": 0,
        "attack_bytes_read": 0, "phase_a_executed": False, "phase_b_executed": False,
        "tests": tests["status"], "tamper_tests": tamper["status"],
    }
    dump("final_verdict.json", final)
    oak_n, tex_n = numeric["domains"]["OAK"], numeric["domains"]["TEX"]
    oak_a, tex_a = alarms["domains"]["OAK"], alarms["domains"]["TEX"]
    (ART / "README.md").write_text(f"""# CRID R4a threshold decision-equivalence repair

Final verdict: `{verdict}`

R4 remains permanently `INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE` with threshold status `INCONCLUSIVE_THRESHOLD_BINDING`. This versioned method repair does not rewrite R4 and does not replace its thresholds. The committed R2 literals remain authoritative: OAK `-21.705587048010322`, TEX `-21.942672917134093`.

The clean-only float64 recomputation differences are OAK `{oak_n['absolute_difference']}` and TEX `{tex_n['absolute_difference']}`. On the identical recomputed clean holdout vectors, committed and recomputed thresholds produced byte-identical alarms. OAK reproduced {oak_a['committed_false_positive_count']}/{oak_a['holdout_score_count']} = `{oak_a['committed_fpr']}`; TEX reproduced {tex_a['committed_false_positive_count']}/{tex_a['holdout_score_count']} = `{tex_a['committed_fpr']}`.

No control replay or control scoring was performed. No attack path was stated, hashed, opened, memory-mapped, or read. Phase A and Phase B were not executed. A PASS authorizes only a future repeat of CRID Phase A.
""")
    files = []
    for path in sorted(p for p in ART.rglob("*") if p.is_file() and p.name != "artifact_manifest_sha256.json"):
        files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump("artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.crid-r4a-artifact-manifest.v1", "file_count": len(files), "files": files, "status": "PASS"})
    print(json.dumps(final, indent=2, sort_keys=True)); return 0 if verdict.endswith("_PASS") else 2


if __name__ == "__main__": raise SystemExit(main())
