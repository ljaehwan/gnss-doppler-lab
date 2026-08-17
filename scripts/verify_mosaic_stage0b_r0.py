#!/usr/bin/env python3
"""Fresh-clone verifier for the self-contained Stage-0B R0 artifact."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
REQUIRED = {
    "README.md", "config.json", "source_commit.json", "execution_environment.json",
    "raw_source_binding.json", "navigation_source_inventory.json", "receiver_telemetry_inventory.json",
    "bit_boundary_candidates.csv", "decoded_nav_bits.csv.gz", "navbit_sample_mapping.csv.gz",
    "preamble_detections.csv", "parity_validation.csv", "tow_continuity.csv",
    "per_prn_validation.csv", "coverage_summary.json", "rejected_intervals.csv",
    "final_verdict.json", "artifact_manifest_sha256.json",
}
PLOTS = {
    "prompt_phase_sign_and_20ms_boundary.png", "decoded_bits_and_preambles.png",
    "parity_valid_word_timeline.png", "raw_sample_to_bit_mapping.png",
    "per_prn_coverage_confidence.png", "boundary_consistency_separated_intervals.png",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rows(name: str, gz: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if gz else open
    kwargs = {"mode": "rt", "encoding": "utf-8", "newline": ""} if gz else {"mode": "r", "encoding": "utf-8", "newline": ""}
    with opener(ART / name, **kwargs) as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    assert ART.is_dir(), ART
    assert REQUIRED <= {p.name for p in ART.iterdir() if p.is_file()}
    assert PLOTS <= {p.name for p in (ART / "plots").iterdir() if p.is_file()}
    manifest = json.loads((ART / "artifact_manifest_sha256.json").read_text())
    for item in manifest["files"]:
        path = ART / item["path"]
        assert path.stat().st_size == item["size_bytes"]
        assert digest(path) == item["sha256"]

    config = json.loads((ART / "config.json").read_text())
    assert config["attack_data_used"] is False
    assert config["synthetic_injection_performed"] is False
    assert config["model_used"] is False and config["threshold_tuning_performed"] is False
    assert config["constant_plus_one_fallback"] is False
    assert "receiver data_symbol_boundary only" in config["boundary_selection"]
    binding = json.loads((ART / "raw_source_binding.json").read_text())
    assert binding["overall_status"] == "PASS"
    for dataset in ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic"):
        assert binding[dataset]["status"] == "PASS"
        assert binding[dataset]["stage0a_full_sha256"] == binding[dataset]["expected_sha256"]

    validations = rows("per_prn_validation.csv")
    for dataset in ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic"):
        selected = [r for r in validations if r["dataset"] == dataset]
        assert len(selected) >= 4
        assert len({r["prn"] for r in selected}) == len(selected)
        assert all(r["status"] == "PASS" and int(r["valid_subframes"]) >= 2 for r in selected)
        assert all(int(r["valid_words"]) == 20 and int(r["parity_failures"]) == 0 for r in selected)
        assert all(int(r["sample_boundary_error_samples"]) == 0 for r in selected)

    parity = rows("parity_validation.csv")
    assert len(parity) == 20 * len(validations)
    assert all(r["parity_valid"] == "True" for r in parity)
    preambles = rows("preamble_detections.csv")
    assert len(preambles) == 2 * len(validations)
    assert all(r["observed_decoded_preamble"] == "10001011" and r["valid"] == "True" for r in preambles)
    tow = rows("tow_continuity.csv")
    assert len(tow) == len(validations)
    assert all(int(r["delta_s"]) == 6 and r["tow_continuity_valid"] == "True" for r in tow)

    candidates = rows("bit_boundary_candidates.csv")
    for row in validations:
        subset = [r for r in candidates if r["dataset"] == row["dataset"] and r["prn"] == row["prn"]]
        assert len(subset) == 20 and sum(r["selected"] == "True" for r in subset) == 1
        selected = next(r for r in subset if r["selected"] == "True")
        assert selected["receiver_flag_match"] == "True"

    mapping = rows("navbit_sample_mapping.csv.gz", gz=True)
    valid = [r for r in mapping if r["validated_navbit"] == "True"]
    assert len(valid) == 600 * len(validations)
    assert {-1, 1} == {int(r["bit_value_pm1"]) for r in valid}
    for row in valid:
        dataset = row["dataset"]
        fs = 5_000_000 if dataset.startswith("OAK") else 25_000_000
        duration = int(row["raw_end_sample_exclusive"]) - int(row["raw_start_sample"])
        assert abs(duration - 20 * fs // 1000) <= 3
        assert row["parity_valid"] == "True"
    verdict = json.loads((ART / "final_verdict.json").read_text())
    assert verdict["verdict"] == "STAGE0B_NAVBIT_PROVENANCE_PASS" and verdict["pass"] is True
    print(f"PASS: {len(validations)} PRNs, {len(valid)} validated bits, {len(manifest['files'])} checksums")


if __name__ == "__main__":
    main()
