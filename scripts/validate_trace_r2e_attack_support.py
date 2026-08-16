#!/usr/bin/env python3
"""Validate repaired DS7/OS4 support before frozen metric evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.trace_equivariance import robust_epoch_blocks
from gnss_doppler_lab.trace_native_1ms import load_native_trace_pairs, sha256_file
import run_trace_stage0_r2e as r2e

ARTIFACT = r2e.driver.ARTIFACT
SPECS = {
    "TEXBAT.DS7": {"slug": "texbat_ds7", "onset_s": 110.0, "audit": "ds7_attack_support_audit.json"},
    "OAKBAT.OS4": {"slug": "oakbat_os4", "onset_s": 120.0, "audit": "os4_attack_support_audit.json"},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SPECS), required=True)
    args = parser.parse_args()
    spec = SPECS[args.scenario]
    dump = r2e.driver.SSD_ROOT / "dumps/phase_b" / spec["slug"] / "rep1"
    manifest = json.loads((dump / "manifest.json").read_text())
    pairs = load_native_trace_pairs(dump, cn0_min_db_hz=28.0, lock_min=0.85, prompt_epsilon=1e-12)
    common = pairs.valid_support[:, np.arange(1, 8)].all(axis=1)
    finite = (
        np.isfinite(pairs.current[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.current[:, 1:8].imag).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].real).all(axis=1)
        & np.isfinite(pairs.target[:, 1:8].imag).all(axis=1)
    )
    selected = pairs.take(common & finite)
    blocks = robust_epoch_blocks(
        selected,
        np.zeros(len(selected.time_s), dtype=np.float64),
        block_s=0.5,
        minimum_prns=4,
    )
    pre = blocks[blocks["block_start_s"] < spec["onset_s"]]
    post = blocks[blocks["block_start_s"] >= spec["onset_s"]]
    passed = len(pre) > 0 and len(post) > 0
    prior = json.loads((ARTIFACT / spec["audit"]).read_text())
    payload = {
        "schema": "gnss-doppler-lab.trace-r2e-attack-support-audit.v2",
        "scenario": args.scenario,
        "status": "PASS" if passed else "FAIL_CLOSED",
        "failure_label_if_any": None if passed else f"{spec['slug'].upper()}_REPAIRED_SUPPORT_INCOMPLETE",
        "parent_failure_audit": prior,
        "repair": {
            "mapping": "scenario-specific preregistered pre-onset target-aligned handoff",
            "handoff_path": manifest["frozen_handoff_path"],
            "handoff_sha256": manifest["frozen_handoff_sha256"],
            "receiver_config_sha256": manifest["receiver_config_sha256"],
            "receiver_executable_sha256": manifest["receiver_executable"]["sha256"],
            "receiver_manifest_path": str(dump / "manifest.json"),
            "receiver_manifest_sha256": sha256_file(dump / "manifest.json"),
        },
        "frozen_support": {
            "selected_pair_count": int(len(selected.time_s)),
            "unique_prn_count": int(len(np.unique(selected.prn))),
            "unique_prns": sorted(map(int, np.unique(selected.prn))),
            "time_start_s": float(selected.time_s.min()),
            "time_end_s": float(selected.time_s.max()),
            "valid_four_prn_block_count": int(len(blocks)),
            "pre_onset_four_prn_block_count": int(len(pre)),
            "post_onset_four_prn_block_count": int(len(post)),
            "onset_s": spec["onset_s"],
        },
        "frozen_contract": {
            "block_s": 0.5,
            "minimum_prns": 4,
            "cn0_min_db_hz": 28.0,
            "lock_min": 0.85,
            "post_onset_required": True,
        },
        "trace_scores_read_or_computed": False,
        "performance_claimed": False,
    }
    (ARTIFACT / spec["audit"]).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
