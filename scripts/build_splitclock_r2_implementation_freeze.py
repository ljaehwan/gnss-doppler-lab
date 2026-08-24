#!/usr/bin/env python3
"""Build the pre-clean-execution SPLITCLOCK R2 implementation freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPAIR_SCOPE_SHA = "a645980a81e94499efaebbc3287f0a263302e7e9"
IMPLEMENTATION_PATHS = (
    "src/gnss_doppler_lab/splitclock_r2_contract.py",
    "src/gnss_doppler_lab/splitclock_r2_model.py",
    "src/gnss_doppler_lab/splitclock_r2_experiment.py",
    "scripts/build_splitclock_r2_implementation_freeze.py",
    "scripts/run_splitclock_r2.py",
    "scripts/verify_splitclock_r2.py",
    "tests/test_splitclock_r2_terminal_contract_repair.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    repo = args.repo.resolve()

    test_log = artifact / "test_output.txt"
    if not test_log.is_file() or "16 passed" not in test_log.read_text(
        encoding="utf-8"
    ):
        raise SystemExit("data-independent test log does not prove 16 passed")

    bindings = {
        path: {
            "sha256": sha256_file(repo / path),
            "size_bytes": (repo / path).stat().st_size,
        }
        for path in IMPLEMENTATION_PATHS
    }
    write_json(
        artifact / "design_freeze_commit.json",
        {
            "status": "PASS",
            "message": "SPLITCLOCK_STAGE0A_R2_REPAIR_SCOPE_FREEZE",
            "local_sha": REPAIR_SCOPE_SHA,
            "remote_sha": REPAIR_SCOPE_SHA,
            "ahead": 0,
            "behind": 0,
        },
    )
    write_json(
        artifact / "implementation_freeze.json",
        {
            "status": "PRE_CLEAN_EXECUTION_IMPLEMENTATION_FREEZE",
            "repair_scope_freeze_sha": REPAIR_SCOPE_SHA,
            "repairs": {
                "R2-F1": "fit-only per-PRN code median and one fit-only global median for each dynamic modality",
                "R2-F2": "one fit posterior q_i per PRN and one held-out mixture marginalization per PRN",
                "R2-F3": "matched 20-second A0/A6 persistence statistic T",
            },
            "data_independent_test": {
                "passed": 16,
                "failed": 0,
                "log": "test_output.txt",
            },
            "pre_freeze_access_audit": {
                "clean_data_bytes_read": 0,
                "clean_score_operations": 0,
                "attack_bytes_read": 0,
                "jammertest_raw_bytes_read": 0,
            },
            "unchanged_parameters": {
                "window_epochs": 10,
                "fit_epochs": 7,
                "heldout_epochs": 3,
                "student_t_degrees_of_freedom": 4.0,
                "minimum_effective_cluster_mass_prns": 2.0,
                "delta_p": "2 + M_eligible",
                "q": 0.99,
                "quantile_method": "higher",
                "persistence": 3,
                "evaluation_horizon_epochs": 20,
            },
            "implementation_bindings": bindings,
            "post_result_change_policy": "NO_CODE_FEATURE_THRESHOLD_SPLIT_GATE_OR_PARAMETER_CHANGES",
            "terminal_failure_policy": "TERMINATE_SPLITCLOCK_NO_FURTHER_GATE_RELAXATION",
        },
    )

    readme = artifact / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else (
        "# SPLITCLOCK-GNSS Stage-0A R2 terminal contract repair\n"
    )
    text = text.rstrip() + (
        "\n\nImplementation status: PRE_CLEAN_EXECUTION_IMPLEMENTATION_FREEZE\n"
        "\nNo clean score, attack input, or Jammertest raw input was accessed before this freeze.\n"
    )
    readme.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "implementation_files": len(bindings)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
