#!/usr/bin/env python3
"""Receiver-in-the-loop smoke test for real complex nine-tap GNSS-SDR dumps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnss_doppler_lab.gnss_sdr import run_receiver  # noqa: E402
from gnss_doppler_lab.tracking_peaks import (  # noqa: E402
    available_tracking_prns,
    load_receiver_tracking_peak_series_segments,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/cgc_complex9_receiver_smoke_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: str) -> Path:
    return (REPO_ROOT / value).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = _repo_path(config["output_root"])
    receiver = config["gnss_sdr"]
    executable = _repo_path(receiver["executable"])
    patch_path = _repo_path(receiver["patch"])
    source_manifest = _repo_path(config["source_rf_manifest"])
    if output_root.exists():
        if not args.verify_existing:
            raise FileExistsError(output_root)
        candidates = sorted(output_root.glob("*/manifest.json"))
        if len(candidates) != 1:
            raise ValueError("existing smoke root must contain exactly one receiver manifest")
        receiver_manifest = candidates[0]
        receiver_document = json.loads(
            receiver_manifest.read_text(encoding="utf-8")
        )
        if receiver_document["source"]["rf_manifest_sha256"] != _sha256(source_manifest):
            raise ValueError("existing receiver source provenance mismatch")
        if receiver_document["receiver"]["executable_sha256"] != _sha256(executable):
            raise ValueError("existing receiver executable provenance mismatch")
    else:
        receiver_manifest = run_receiver(
            source_manifest,
            output_root,
            executable=executable,
            channel_count=int(receiver["channel_count"]),
            timeout_seconds=int(receiver["timeout_seconds"]),
            tracking_tap_count=int(receiver["tracking_tap_count"]),
            tracking_tap_spacing_chips=float(receiver["tracking_tap_spacing_chips"]),
        )
    run_dir = receiver_manifest.parent

    prn_records: list[dict[str, object]] = []
    total_epochs = 0
    maximum_magnitude_error = 0.0
    maximum_prompt_tap_error = 0.0
    for prn in available_tracking_prns(run_dir):
        segments = load_receiver_tracking_peak_series_segments(
            run_dir,
            prn,
            tap_count=9,
            require_complex_taps=True,
        )
        epoch_count = sum(len(segment.time_s) for segment in segments)
        total_epochs += epoch_count
        magnitude_error = max(
            float(
                np.max(
                    np.abs(
                        np.hypot(
                            segment.tap_i.astype(np.float32),
                            segment.tap_q.astype(np.float32),
                        ).astype(np.float64) - segment.magnitudes
                    )
                )
            )
            for segment in segments
        )
        prompt_tap_error = max(
            float(
                max(
                    np.max(np.abs(segment.tap_i[:, 4] - segment.prompt_i)),
                    np.max(np.abs(segment.tap_q[:, 4] - segment.prompt_q)),
                )
            )
            for segment in segments
        )
        maximum_magnitude_error = max(maximum_magnitude_error, magnitude_error)
        maximum_prompt_tap_error = max(maximum_prompt_tap_error, prompt_tap_error)
        prn_records.append(
            {
                "prn": prn,
                "segment_count": len(segments),
                "epoch_count": epoch_count,
                "magnitude_reconstruction_max_abs_error": magnitude_error,
                "prompt_tap_max_abs_error": prompt_tap_error,
                "nonzero_quadrature_fraction": float(
                    np.mean(
                        np.concatenate(
                            [
                                np.abs(segment.tap_q).reshape(-1)
                                for segment in segments
                            ]
                        )
                        > 1e-9
                    )
                ),
            }
        )

    gates_config = config["gates"]
    gates = {
        "minimum_complex_prns": {
            "observed": len(prn_records),
            "required": int(gates_config["minimum_complex_prns"]),
            "passed": len(prn_records) >= int(gates_config["minimum_complex_prns"]),
        },
        "minimum_total_epochs": {
            "observed": total_epochs,
            "required": int(gates_config["minimum_total_epochs"]),
            "passed": total_epochs >= int(gates_config["minimum_total_epochs"]),
        },
        "maximum_magnitude_reconstruction_error": {
            "observed": maximum_magnitude_error,
            "required": float(gates_config["maximum_magnitude_reconstruction_error"]),
            "passed": maximum_magnitude_error
            <= float(gates_config["maximum_magnitude_reconstruction_error"]),
        },
    }
    overall_passed = all(record["passed"] for record in gates.values())
    summary = {
        "schema": "gnss-doppler-lab.cgc-complex9-receiver-smoke-result",
        "schema_version": 1,
        "scope": config["scope"],
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_rf_manifest": str(source_manifest),
        "source_rf_manifest_sha256": _sha256(source_manifest),
        "receiver_manifest": str(receiver_manifest),
        "receiver_manifest_sha256": _sha256(receiver_manifest),
        "gnss_sdr": {
            "executable": str(executable),
            "executable_sha256": _sha256(executable),
            "upstream_commit": receiver["upstream_commit"],
            "patch": str(patch_path),
            "patch_sha256": _sha256(patch_path),
        },
        "complex_prn_count": len(prn_records),
        "total_epochs": total_epochs,
        "maximum_magnitude_reconstruction_error": maximum_magnitude_error,
        "maximum_prompt_tap_error": maximum_prompt_tap_error,
        "prns": prn_records,
        "gates": gates,
        "overall_passed": overall_passed,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if overall_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
