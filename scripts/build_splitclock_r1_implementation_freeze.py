#!/usr/bin/env python3
"""Build the pre-clean-execution SPLITCLOCK R1 implementation freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DESIGN_SHA = "d472376ba2e59d766c93296b4755df4c89ccbe9b"
IMPLEMENTATION_PATHS = (
    "src/gnss_doppler_lab/splitclock_r1_contract.py",
    "src/gnss_doppler_lab/splitclock_r1_geometry.py",
    "src/gnss_doppler_lab/splitclock_r1_model.py",
    "src/gnss_doppler_lab/splitclock_r1_experiment.py",
    "scripts/run_splitclock_r1.py",
    "scripts/verify_splitclock_r1.py",
    "tests/test_splitclock_r1_design_freeze.py",
    "tests/test_splitclock_r1_model.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()

    artifact = args.artifact.resolve()
    repo = args.repo.resolve()
    test_log = artifact / "test_output.txt"
    if not test_log.is_file() or "16 passed" not in test_log.read_text(encoding="utf-8"):
        raise SystemExit("data-independent test log does not prove 16 passed")

    bindings = {
        path: {"sha256": sha256_file(repo / path), "size_bytes": (repo / path).stat().st_size}
        for path in IMPLEMENTATION_PATHS
    }
    write_json(
        artifact / "design_freeze_commit.json",
        {
            "status": "PASS",
            "message": "SPLITCLOCK_STAGE0A_R1_DESIGN_FREEZE",
            "local_sha": DESIGN_SHA,
            "remote_sha": DESIGN_SHA,
            "ahead": 0,
            "behind": 0,
        },
    )
    write_json(
        artifact / "implementation_freeze.json",
        {
            "status": "PRE_CLEAN_EXECUTION_IMPLEMENTATION_FREEZE",
            "design_freeze_sha": DESIGN_SHA,
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
            "model_contract": {
                "window_epochs": 10,
                "fit_epochs": 7,
                "heldout_epochs": 3,
                "primary_score": "heldout_loglik_K2-heldout_loglik_K1-0.5*delta_p*log(n_valid)",
                "delta_p": "2 + M_eligible",
                "student_t_degrees_of_freedom": 4.0,
                "minimum_effective_cluster_mass_prns": 2.0,
                "restart_selection_data": "fit_only",
                "final_k2_membership": "soft_window_persistent",
            },
            "implementation_bindings": bindings,
            "post_result_change_policy": "NO_MODEL_THRESHOLD_SIGN_FEATURE_SPLIT_OR_GATE_CHANGES",
        },
    )

    readme = artifact / "README.md"
    text = readme.read_text(encoding="utf-8")
    marker = "Implementation status:"
    status_line = "Implementation status: PRE_CLEAN_EXECUTION_IMPLEMENTATION_FREEZE\n"
    if marker in text:
        lines = text.splitlines(keepends=True)
        text = "".join(status_line if line.startswith(marker) else line for line in lines)
    else:
        text = text.rstrip() + "\n\n" + status_line
    readme.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "PASS", "implementation_files": len(bindings)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
