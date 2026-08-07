#!/usr/bin/env python3
"""Build Stage-1 R1 continuous tracker artifacts (checkpoint A).

Checkpoint A produces:
- tracker_cadence_audit.json
- tracker_cadence_by_channel.csv
Checkpoint B produces:
- continuous_tracker_manifest.json
- continuous_tracker_cleanStatic.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnss_doppler_lab.acaf_nf_stage1_continuous_tracker import build_audit, build_continuous_tracker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(output_dir: Path) -> None:
    files = {
        str(path.relative_to(output_dir)): {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in {"checksums.json", "verification_report.json"}
    }
    (output_dir / "checksums.json").write_text(
        json.dumps({"algorithm": "sha256", "files": files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    output_dir: Path,
    command: str,
    source_binding: str | Path,
    scenarios: list[str] | None,
    *,
    checkpoint: str,
    filename: str = "execution_manifest.json",
) -> None:
    manifest = {
        "command": command,
        "checkpoint": checkpoint,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "output_root": str(output_dir),
        "source_binding": str(source_binding),
        "scenarios": scenarios,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / filename).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-binding", default="configs/acaf_nf_stage1_source_binding.json")
    parser.add_argument("--output", default="artifacts/acaf_nf_stage1_r1_continuous_tracker")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--checkpoint", choices=("A", "B"), default="A")
    args = parser.parse_args()

    output = Path(args.output)
    all_scenarios = ("cleanStatic", "ds3", "ds4", "ds7", "ds8")
    if args.checkpoint == "A":
        output = build_audit(
            args.source_binding,
            output,
            scenarios=tuple(args.scenario) if args.scenario else None,
        )
    else:
        requested = tuple(args.scenario) if args.scenario else ("cleanStatic",)
        if len(requested) != 1 or requested[0] != "cleanStatic":
            raise ValueError("checkpoint B only supports --scenario cleanStatic")
        output = build_continuous_tracker(
            args.source_binding,
            output,
            scenario="cleanStatic",
        )

    if args.checkpoint == "B":
        requested = ["cleanStatic"]
    else:
        requested = list(args.scenario) if args.scenario else list(all_scenarios)

    command = shlex.join(
        [
            "PYTHONPATH=src",
            "python3",
            "scripts/build_acaf_nf_continuous_tracker.py",
            "--checkpoint",
            args.checkpoint,
            "--source-binding",
            str(args.source_binding),
            "--output",
            str(args.output),
        ]
    )
    if requested:
        command += " " + " ".join([shlex.join(["--scenario", s]) for s in requested])
    manifest_path = "execution_manifest.json" if args.checkpoint == "A" else "execution_manifest_checkpoint_b.json"
    _write_manifest(
        output,
        command,
        args.source_binding,
        requested,
        checkpoint=args.checkpoint,
        filename=manifest_path,
    )
    (output / "verification_report.json").write_text(
        json.dumps({"status": "PENDING_INDEPENDENT_VERIFICATION", "checkpoint": args.checkpoint}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_checksums(output)
    print(f"built: {output}")


if __name__ == "__main__":
    main()
