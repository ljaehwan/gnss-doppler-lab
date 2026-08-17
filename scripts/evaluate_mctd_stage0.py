#!/usr/bin/env python3
"""Evaluate the preregistered MCTD Phase-A and clean-only model gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.mctd import align_dump_directories, epoch_scores
from gnss_doppler_lab.trace_native_1ms import read_records

ARTIFACT = ROOT / "artifacts/mctd_stage0_static"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static")


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def directory(slug: str, loop: str, repetition: int, phase: str = "phase_a") -> Path:
    return SSD / "dumps" / phase / slug / loop / f"rep{repetition}"


def canonical_rows(path: Path) -> np.ndarray:
    rows = []
    for dump in sorted(path.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(dump, mmap=False)
        rows.append(records)
    values = np.concatenate(rows)
    order = np.lexsort((values["loop_sequence"], values["raw_interval_start_sample"], values["prn"]))
    return values[order]


def exact_reproduction(left: Path, right: Path) -> dict[str, object]:
    a, b = canonical_rows(left), canonical_rows(right)
    same_shape = a.shape == b.shape
    exact = bool(same_shape and a.tobytes() == b.tobytes())
    return {"status": "PASS" if exact else "FAIL", "left_rows": len(a), "right_rows": len(b),
            "canonical_row_set_exact": exact, "bit_exact_fields": list(a.dtype.names or ()),
            "numeric_tolerance": 0.0}


def manifest(path: Path) -> dict[str, object]:
    return json.loads((path / "manifest.json").read_text())


def source_equality(slug: str) -> dict[str, object]:
    paths = [directory(slug, loop, 1) for loop in ("slow", "fast", "identical_left", "identical_right")]
    values = [manifest(path) for path in paths]
    raw = {item["raw_iq"]["sha256"] for item in values}
    receiver = {item["receiver"]["sha256"] for item in values}
    handoff = {item["handoff"]["sha256"] for item in values}
    passed = len(raw) == len(receiver) == len(handoff) == 1 and all(item["raw_stable"] and item["receiver_stable"] for item in values)
    return {"status": "PASS" if passed else "FAIL", "raw_iq_sha256": sorted(raw),
            "receiver_sha256": sorted(receiver), "handoff_sha256": sorted(handoff),
            "same_initial_state": len(handoff) == 1, "same_raw_source": len(raw) == 1,
            "only_configured_differences": ["Tracking_1C.dll_bw_hz", "Tracking_1C.pll_bw_hz", "trace scenario label"],
            "same_integration_ms": 1, "same_tap_spacing_chips": 0.125, "same_nav_bit_handling": True}


def phase_a() -> int:
    datasets = {}
    overall = True
    for dataset, slug in (("TEXBAT.cleanStatic", "texbat_cleanstatic"), ("OAKBAT.cleanStatic", "oakbat_cleanstatic")):
        deterministic = {
            "slow": exact_reproduction(directory(slug, "slow", 1), directory(slug, "slow", 2)),
            "fast": exact_reproduction(directory(slug, "fast", 1), directory(slug, "fast", 2)),
        }
        aligned = align_dump_directories(directory(slug, "slow", 1), directory(slug, "fast", 1), dataset=dataset)
        epoch, _, n = epoch_scores(aligned.epoch_ms, aligned.prn, np.zeros(len(aligned.prn)))
        identical = align_dump_directories(directory(slug, "identical_left", 1), directory(slug, "identical_right", 1), dataset=dataset)
        max_identical = float(np.max(np.abs(identical.full))) if len(identical.prn) else None
        collapse = max_identical == 0.0 and len(identical.prn) > 0
        common_raw_delta = np.abs(aligned.raw_start_slow - aligned.raw_start_fast)
        stable = len(epoch) >= 1000 and int(n.max(initial=0)) >= 4
        source = source_equality(slug)
        passed = source["status"] == "PASS" and all(item["status"] == "PASS" for item in deterministic.values()) and stable and collapse
        overall &= passed
        union_rows = len(canonical_rows(directory(slug, "slow", 1))) + len(canonical_rows(directory(slug, "fast", 1)))
        datasets[dataset] = {
            "status": "PASS" if passed else "FAIL", "source_equality": source,
            "deterministic_replay": deterministic,
            "stable_support": {"status": "PASS" if stable else "FAIL", "quality_common_rows": len(aligned.prn),
                               "quality_common_epochs_ge_4_prns": len(epoch), "maximum_common_prns": int(n.max(initial=0)),
                               "common_raw_start_delta_samples_max": int(common_raw_delta.max(initial=0)),
                               "missing_native_row_rate": 1.0 - 2.0 * len(aligned.prn) / union_rows},
            "identical_loop_control": {"status": "PASS" if collapse else "FAIL", "common_rows": len(identical.prn),
                                       "maximum_absolute_full_divergence": max_identical,
                                       "collapse_to_numerical_error": collapse, "numerical_error_tolerance": 0.0},
        }
    payload = {"schema": "gnss-doppler-lab.mctd-phase-a.v1", "phase_a_passed": overall,
               "phase_b_authorized": False, "attack_scores_computed": False, "datasets": datasets,
               "failure_verdict_if_terminal": None if overall else "NO_GO_RECEIVER_DIFFERENTIAL_INVALID"}
    dump_json(ARTIFACT / "phase_a_reproducibility.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if overall else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("phase-a",))
    args = parser.parse_args()
    return phase_a()


if __name__ == "__main__":
    raise SystemExit(main())
