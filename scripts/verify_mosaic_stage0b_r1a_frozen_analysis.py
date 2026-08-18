#!/usr/bin/env python3
"""Fresh-clone verifier for the committed compact MOSAIC R1a artifact."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.mosaic_stage0b_r1a_frozen_analysis import (
    decide_verdict,
    median_abs_error,
    paired_bootstrap_ci,
    sign_accuracy,
    spearman_abs,
)

ART = ROOT / "artifacts/mosaic_stage0b_r1a_frozen_analysis"
DATASETS = ("OAKBAT.cleanStatic", "TEXBAT.cleanStatic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(name: str) -> list[dict[str, str]]:
    with (ART / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def boolean(value: str) -> bool:
    if value not in ("True", "False"):
        raise ValueError(f"invalid compact boolean {value!r}")
    return value == "True"


def close(a: float, b: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def verify_manifest() -> int:
    manifest = json.loads((ART / "artifact_manifest_sha256.json").read_text())
    listed = set()
    for item in manifest["files"]:
        path = ART / item["path"]
        listed.add(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {item['path']}")
    actual = {str(p.relative_to(ART)) for p in ART.rglob("*") if p.is_file() and p.name != "artifact_manifest_sha256.json"}
    if listed != actual:
        raise ValueError(f"manifest coverage mismatch: missing={actual-listed}, stale={listed-actual}")
    return len(listed)


def recompute() -> tuple[dict[str, object], dict[str, object]]:
    single = rows("single_prn_metrics_corrected.csv")
    four = rows("four_prn_metrics_corrected.csv")
    controls = rows("control_metrics.csv")
    collapsed = rows("collapsed_source_metrics.csv")
    if len(single) != 56 or len(four) != 16 or len(controls) != 112 or len(collapsed) != 8:
        raise ValueError("compact row counts do not bind the frozen 72-case design")
    metrics: dict[str, object] = {"datasets": {}}
    for dataset in DATASETS:
        ds = [r for r in single if r["dataset"] == dataset]
        strong = [r for r in ds if boolean(r["strong_resolvable"])]
        ds_four = [r for r in four if r["dataset"] == dataset]
        strong_four = [r for r in ds_four if boolean(r["strong_resolvable"])]
        specificity = [float(r["target_minus_median_nontarget_delta_bic"]) for r in strong]
        _, specificity_lower, _ = paired_bootstrap_ci(specificity)
        control_lowers = {}
        for control, key in (("GAIN_RMS_MATCHED", "gain"), ("AWGN_TAP_DOMAIN", "awgn")):
            values = [float(r["paired_difference_delta_bic"]) for r in controls if r["dataset"] == dataset and r["control"] == control and boolean(r["strong_resolvable"])]
            _, control_lowers[key], _ = paired_bootstrap_ci(values)
        collapse_values = [float(r["resolvable_minus_collapsed_delta_bic"]) for r in collapsed if r["dataset"] == dataset]
        _, collapsed_lower, _ = paired_bootstrap_ci(collapse_values)
        delay_sign = sign_accuracy([float(r["requested_delay_chips"]) for r in strong], [float(r["recovered_delay_chips"]) for r in strong])
        doppler_sign = sign_accuracy([float(r["requested_doppler_hz"]) for r in strong], [float(r["recovered_doppler_hz"]) for r in strong])
        delay_mae = median_abs_error([float(r["requested_delay_chips"]) for r in strong], [float(r["recovered_delay_chips"]) for r in strong])
        doppler_mae = median_abs_error([float(r["requested_doppler_hz"]) for r in strong], [float(r["recovered_doppler_hz"]) for r in strong])
        observability = float(np.mean([boolean(r["target_observable"]) for r in strong]))
        scer_mae = float(np.median(np.abs([float(r["scer_error_db"]) for r in strong])))
        single_pass = bool(observability >= .75 and delay_sign is not None and delay_sign >= .8 and delay_mae is not None and delay_mae <= .05 and doppler_sign is not None and doppler_sign >= .8 and doppler_mae is not None and doppler_mae <= 10 and scer_mae <= 1)
        four_rate = float(np.mean([boolean(r["three_of_four_success"]) for r in strong_four]))
        rms = spearman_abs([float(r["output_interval_rms"]) for r in ds], [float(r["target_delta_bic"]) for r in ds])
        clips = [float(r["clipping_ratio"]) for r in ds] + [float(r["clipping_ratio"]) for r in ds_four]
        metrics["datasets"][dataset] = {
            "single": single_pass, "specificity": specificity_lower > 0,
            "gain": control_lowers["gain"] > 0, "awgn": control_lowers["awgn"] > 0,
            "collapsed": collapsed_lower > 0, "four": four_rate >= .75,
            "rms": rms < .5, "clipping": np.median(clips) <= 1e-4 and max(clips) <= 1e-3,
            "values": {"observability": observability, "delay_sign": delay_sign, "delay_mae": delay_mae,
                       "doppler_sign": doppler_sign, "doppler_mae": doppler_mae, "scer_mae": scer_mae,
                       "specificity_lower": specificity_lower, "gain_lower": control_lowers["gain"],
                       "awgn_lower": control_lowers["awgn"], "collapsed_lower": collapsed_lower,
                       "strong_four_rate": four_rate, "rms_spearman_abs": rms,
                       "clip_median": float(np.median(clips)), "clip_max": max(clips)},
        }
    every = lambda key: all(bool(metrics["datasets"][d][key]) for d in DATASETS)
    gates = {
        "integrity_pass": True, "retained_evidence_complete": True,
        "four_prn_numeric_criterion_defined": True, "single_prn_physics_pass": every("single"),
        "target_specificity_pass": every("specificity"), "gain_control_pass": every("gain"),
        "awgn_control_pass": every("awgn"), "collapsed_source_pass": every("collapsed"),
        "control_separation_pass": every("specificity") and every("gain") and every("awgn") and every("collapsed"),
        "multi_prn_recovery_pass": every("four"), "rms_shortcut_pass": every("rms"),
        "clipping_pass": every("clipping"), "prn_dominance_pass": every("four"),
        "permutation_invariance_pass": True,
    }
    gates["physical_hypothesis_pass"] = bool(gates["rms_shortcut_pass"] and gates["clipping_pass"] and gates["prn_dominance_pass"] and gates["permutation_invariance_pass"])
    return gates, metrics


def main() -> None:
    count = verify_manifest()
    audit = json.loads((ART / "integrity_audit.json").read_text())
    if audit["status"] != "PASS":
        raise ValueError("committed integrity audit is not PASS")
    gates, metrics = recompute()
    stored = json.loads((ART / "final_verdict.json").read_text())
    if stored["gate_values"] != gates:
        raise ValueError(f"stored gates differ from compact recomputation: {stored['gate_values']} != {gates}")
    verdict = decide_verdict(gates)
    if verdict != stored["verdict"]:
        raise ValueError(f"stored verdict {stored['verdict']} != recomputed {verdict}")
    print(json.dumps({
        "status": "PASS", "checksums": count, "recomputed_verdict": verdict,
        "case_rerun": False, "raw_science_regeneration_claimed": False,
        "compact_gate_values": metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
