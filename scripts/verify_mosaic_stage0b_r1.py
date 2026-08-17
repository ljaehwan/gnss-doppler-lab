#!/usr/bin/env python3
"""Fresh-clone verifier for the frozen R1 preregistration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.mosaic_iq_injector import design_sha256  # noqa: E402
from gnss_doppler_lab.mosaic_receiver_in_loop import assign_case_targets  # noqa: E402
from gnss_doppler_lab.mosaic_stage0b_metrics import caf_grids  # noqa: E402

ART = ROOT / "artifacts/mosaic_stage0b_r1_receiver_in_loop"
R0C = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
EXPECTED_BASE = "e0993bd6b16628681b52c1abd52cf177af67e10a"
EXPECTED_DESIGN = "b1a06556f7cd67738274c132f80b0581b20914d971f72f4e4ab0b5efc9a7facf"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest() -> int:
    value = json.loads((ART / "artifact_manifest_sha256.json").read_text())
    for item in value["files"]:
        path = ART / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {item['path']}")
    return len(value["files"])


def main() -> None:
    checksums = verify_manifest()
    source = json.loads((ART / "source_commit.json").read_text())
    if source["required_base_commit"] != EXPECTED_BASE or not source["base_match"]:
        raise ValueError("R0c base mismatch")
    r0c = json.loads((ART / "r0c_input_binding.json").read_text())
    verdict = json.loads((R0C / "final_verdict.json").read_text())
    if r0c["r0c_verdict"] != verdict["verdict"] or verdict["verdict"] != "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION":
        raise ValueError("R0c verdict mismatch")
    design = json.loads((ART / "frozen_injection_design.json").read_text())
    binding = json.loads((ART / "frozen_injection_design_sha256.json").read_text())
    if len(design) != 72 or design_sha256(design) != EXPECTED_DESIGN or binding["canonical_json_sha256"] != EXPECTED_DESIGN:
        raise ValueError("frozen design mismatch")
    assignment = json.loads((ART / "case_target_assignment.json").read_text())
    common = r0c["common_intervals"]
    expected = assign_case_targets(design, {d: common[d]["included_prns"] for d in common})
    if assignment["assignments"] != expected or assignment["case_count"] != 72:
        raise ValueError("case target assignment mismatch")
    formats = json.loads((ART / "sample_format_validation.json").read_text())
    if formats["status"] != "PASS" or formats["foundation_int8_quantizer_used"]:
        raise ValueError("int16 sample format mismatch")
    if any(v["bytes_per_complex_sample"] != 4 or v["iq_ordering"] != "I_then_Q" for v in formats["datasets"].values()):
        raise ValueError("I/Q ordering mismatch")
    delay, doppler = caf_grids()
    if len(delay) != 29 or len(doppler) != 31 or delay[0] != -.35 or delay[-1] != .35 or doppler[0] != -75 or doppler[-1] != 75:
        raise ValueError("CAF grid mismatch")
    prereg = json.loads((ART / "preregistration.json").read_text())
    if prereg["scientific_verdict_generated"] or prereg["raw_rss_improvement_is_sufficient"] or len(prereg["go_criteria"]) != 14:
        raise ValueError("preregistration criteria mismatch")
    status = json.loads((ART / "execution_status.json").read_text())
    expected_status = {"status": "READY_FOR_R1_EXECUTION", "injection_executed": False,
        "attack_data_accessed": False, "results_viewed": False, "receiver_replay_executed": False,
        "scientific_verdict_generated": False}
    if status != expected_status:
        raise ValueError("execution status mismatch")
    print(f"PASS: verified R1 preregistration, 72 frozen cases, int16 I/Q bindings, deterministic targets, and {checksums} checksums; READY_FOR_R1_EXECUTION; injection not executed")


if __name__ == "__main__":
    main()
