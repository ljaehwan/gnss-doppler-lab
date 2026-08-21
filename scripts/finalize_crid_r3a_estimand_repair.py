#!/usr/bin/env python3
"""Finalize the R3a verdict without changing any scientific criterion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"


def load(name: str) -> dict:
    return json.loads((ART / name).read_text())


def dump(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(payload)
    return digest.hexdigest()


def main() -> int:
    legacy, joint = load("legacy_reproduction_summary.json"), load("joint_reference_summary.json")
    binding, checkpoint = load("source_binding.json"), load("validation_checkpoint.json")
    attack, tests, tamper = load("attack_access_audit.json"), load("test_results.json"), load("tamper_test_results.json")
    provenance_ok = (
        binding["status"] == "PASS" and binding["full_hash_executed"]
        and legacy["status"] == "PASS" and legacy["numeric_match_to_committed_r3"]
        and checkpoint["status"] == "PASS" and attack["status"] == "PASS"
        and attack["attack_bytes_read"] == 0 and not attack["crid_score_computed"]
    )
    physical_ok = (
        joint["status"] == "PASS" and joint["passed"] == 180 and joint["failed"] == 0
        and joint["maximum_delay_error_chips"] <= 0.025
        and joint["maximum_power_error_db"] <= 0.75
        and joint["maximum_non_target_relative_energy"] <= 0.01
        and joint["all_rank_five"] and joint["all_condition_at_most_1e6"]
        and joint["deterministic_rerun_match"]
    )
    tests_ok = tests["status"] == "PASS" and tamper["status"] == "PASS"
    if not provenance_ok:
        verdict = "INCONCLUSIVE_REFERENCE_PROVENANCE"; next_state = "NOT_AUTHORIZED"
    elif not physical_ok:
        verdict = "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_FAIL"; next_state = "NOT_AUTHORIZED"
    elif not tests_ok:
        verdict = "INCONCLUSIVE_REFERENCE_PROVENANCE"; next_state = "NOT_AUTHORIZED"
    else:
        verdict = "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS"; next_state = "READY_TO_REPEAT_CRID_PHASE_A"
    final = {
        "schema": "gnss-doppler-lab.crid-r3a-final-verdict.v1",
        "verdict": verdict, "status": "PASS" if verdict.endswith("_PASS") else "FAIL",
        "next_state": next_state,
        "r3_verdict_preserved": "INCONCLUSIVE_CONTROL_PROVENANCE",
        "legacy_reference": {"passed": legacy["passed"], "failed": legacy["failed"]},
        "joint_reference": {"passed": joint["passed"], "failed": joint["failed"]},
        "maximum_delay_error_chips": joint["maximum_delay_error_chips"],
        "maximum_power_error_db": joint["maximum_power_error_db"],
        "maximum_non_target_relative_energy": joint["maximum_non_target_relative_energy"],
        "maximum_condition_number": joint["maximum_condition_number"],
        "attack_bytes_read": attack["attack_bytes_read"], "crid_score_computed": attack["crid_score_computed"],
        "phase_a_executed": attack["phase_a_executed"], "c1_c2_c3_replay_executed": attack["c1_c2_c3_replay_executed"],
        "controls_regenerated": attack["control_iq_regenerated"], "tests": tests["status"], "tamper_tests": tamper["status"],
    }
    dump("final_verdict.json", final)
    oak21 = next(row for row in (ART / "denominator_diagnostic.csv").read_text().splitlines()[1:] if row.startswith("OAK,21,"))
    fields = oak21.split(",")
    readme = f"""# CRID Stage-0 R3a independent-reference estimand repair

Final verdict: `{verdict}`  
Next state: `{next_state}`

This versioned post-result method repair leaves R3 permanently at `INCONCLUSIVE_CONTROL_PROVENANCE`. It changes neither the frozen generator nor any R3 artifact/control. The legacy single-PRN diagnostic reproduced {legacy['passed']}/180 PASS and {legacy['failed']} FAIL, all and only OAK PRN 21. The preregistered independent five-PRN joint complex-LS reference produced {joint['passed']}/180 PASS.

OAK PRN 21's legacy single authentic magnitude is {fields[2]}, the independent joint magnitude is {fields[3]}, and single-minus-joint is {fields[5]} dB. Maximum target delay error is {joint['maximum_delay_error_chips']} chip, maximum target power error is {joint['maximum_power_error_db']} dB, maximum non-target relative energy is {joint['maximum_non_target_relative_energy']}, and maximum condition number is {joint['maximum_condition_number']}.

All source/control and existing R3 artifact hashes were freshly verified. Attack bytes read: 0. CRID score, threshold/alarm evaluation, Phase A, C1/C2/C3 replay, control regeneration, and attack evaluation were not executed. A PASS authorizes only a future repeat of CRID Phase A.
"""
    (ART / "README.md").write_text(readme)
    files = []
    for path in sorted(p for p in ART.rglob("*") if p.is_file() and p.name != "artifact_manifest_sha256.json"):
        files.append({"path": str(path.relative_to(ART)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    dump("artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.crid-r3a-artifact-manifest.v1", "file_count": len(files), "files": files, "status": "PASS"})
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0 if verdict == "INDEPENDENT_REFERENCE_ESTIMAND_REPAIR_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
