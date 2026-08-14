#!/usr/bin/env python3
"""Launch one committed-nonce clean A5 workload with causal provenance."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.gcspo_provenance import complete_run, prepare_run, utc_now


def _python_command_path(value):
    """Keep a venv launcher path intact; resolving its symlink loses site-packages."""
    return str(Path(value).absolute())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--challenge-file", required=True, type=Path)
    parser.add_argument("--challenge-id", required=True)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--seed-artifact-dir", required=True, type=Path)
    parser.add_argument("--clean-root", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers != 1:
        raise ValueError("causal A5 runs require exactly one worker")

    output = args.scratch_root / "output"
    command = [
        _python_command_path(args.python), str((ROOT / "scripts/run_gcspo_clean_a5.py").resolve()),
        "--artifact-dir", str(output.resolve()), "--clean-root", str(args.clean_root.resolve()),
        "--workers", "1", "--backend", args.backend, "--numeric-trace",
    ]
    prepared = prepare_run(
        repo_root=ROOT, source_commit=args.source_commit,
        challenge_path=args.challenge_file, challenge_id=args.challenge_id,
        backend=args.backend, argv=command, scratch_root=args.scratch_root,
        evidence_root=args.evidence_root,
        input_files={
            "config.json": args.seed_artifact_dir / "config.json",
            "thresholds.json": args.seed_artifact_dir / "thresholds.json",
            "clean_dataset_manifest": args.clean_root / "manifest.json",
        },
        output_names=("clean_a5_report.json", "thresholds.json",
                      "a5_numeric_trace.json", "a5_backend_truth.json"),
    )
    output.mkdir()
    temporary = args.scratch_root / "tmp"
    temporary.mkdir()
    shutil.copy2(args.seed_artifact_dir / "config.json", output / "config.json")
    shutil.copy2(args.seed_artifact_dir / "thresholds.json", output / "thresholds.json")
    stdout_path = args.evidence_root / "stdout.txt"
    stderr_path = args.evidence_root / "stderr.txt"
    environment = dict(os.environ)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src"),
                        "TMPDIR": str(temporary)})
    started = utc_now()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        exit_code = child.wait()
    finished = utc_now()
    if exit_code != 0:
        raise RuntimeError(f"causal A5 child exit is nonzero: {exit_code}")
    truth = json.loads((output / "a5_backend_truth.json").read_text())
    complete_run(
        prepared_path=prepared, pid=child.pid, started_utc=started, finished_utc=finished,
        exit_code=exit_code,
        backend_truth={key: truth[key] for key in
                       ("requested", "resolved", "cuda_available", "device")},
        stdout_path=stdout_path, stderr_path=stderr_path, output_dir=output,
    )
    print(f"CAUSAL_A5_RUN_PASS id={args.challenge_id} backend={args.backend} pid={child.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

