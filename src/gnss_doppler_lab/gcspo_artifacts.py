"""Provenance, semantic preflight, and canonical artifact helpers for GCSPO."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

SOURCE_HASHES = {
    "src/algorithms/tracking/libs/tracking_discriminators.cc": "00cb68a02fdd3ab51395e4af2e758060a7e299a76a68424c37fcf6cba38f69ec",
    "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc": "6d2db43fea2728acc35fb29b4cb5027b62be442b8966108bf6b955d5b95f486c",
}
RECEIVER_BINARY = "build-complex/src/main/gnss-sdr"
RECEIVER_BINARY_SHA256 = "6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb"
MANIFEST_EXCLUSIONS = {"artifact_manifest_sha256.json", "verifier_report.json", "fresh_clone_verifier_report.json"}
INVALID_REQUIRED = {
    "README.md", "config.json", "data_inventory.json", "preregistration.json", "source_commit.json",
    "access_ledger.jsonl", "invalid_run.json", "artifact_manifest_sha256.json",
    "verifier_report.json", "fresh_clone_verifier_report.json",
}
VALID_SCIENCE_REQUIRED = {
    "README.md", "config.json", "preregistration.json", "source_commit.json", "data_inventory.json",
    "normal_model_summary.json", "thresholds.json", "scenario_metrics.csv", "ablation_metrics.csv",
    "per_epoch_scores.csv", "shared_state_estimates.csv", "external_static_fpr.csv",
    "relation_destruction_metrics.json", "physical_controls.json", "bootstrap_intervals.csv",
    "final_verdict.json", "access_ledger.jsonl", "artifact_manifest_sha256.json",
    "verifier_report.json", "fresh_clone_verifier_report.json",
}
VALID_ADDITIONS = {
    "implementation_manifest.json", "data_manifest.json", "run_manifest.json", "file_access_trace.jsonl",
}

FROZEN_HASHES = {
    "README.md": "eea2e10885d66bfc762f33b2e25147ab07b1bbceace505078e8770e4cdc18ac2",
    "config.json": "0db816116b95b41db8b7af7379cd7411cc52d43b6428ae00ab02d6ccac19f4ad",
    "data_inventory.json": "4faffaede28119f7655da25b44129b09e76f1bb49ec5169861b6336abaea3631",
    "preregistration.json": "715e11965854f785487e9d2c747718747c1d31cdd8603696ea7af126a45a70da",
    "source_commit.json": "34a0eab36e4cbf6b16cf0a7075bc3c8a61008c34b9962cf584e04d4fa9f80b72",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(type(value).__name__)

def canonical_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, default=_json_default) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _source_snapshot(source_root):
    root = Path(source_root).resolve()
    records = []
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        records.append({"path": str(path), "expected_sha256": expected, "observed_sha256": actual,
                        "status": "PASS" if actual == expected else "FAIL"})
    binary = root / RECEIVER_BINARY
    observed_binary = sha256_file(binary) if binary.is_file() else None
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(root), "status", "--short"], check=True, text=True, capture_output=True).stdout.splitlines()
        diff = subprocess.run(["git", "-C", str(root), "diff", "--binary"], check=True, capture_output=True).stdout
        diff_sha = hashlib.sha256(diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        head, status, diff_sha = None, ["GIT_STATUS_UNAVAILABLE"], None
    records.append({"path": str(binary), "expected_sha256": RECEIVER_BINARY_SHA256, "observed_sha256": observed_binary,
                    "status": "PASS" if observed_binary == RECEIVER_BINARY_SHA256 else "FAIL"})
    return {"source_root": str(root), "git_head": head, "git_status": status, "git_diff_sha256": diff_sha,
            "files": records, "source_tree_clean": not status}


def preflight_receiver_semantics(source_root):
    """Prove frozen field units/signs from pinned source plus analytic epsilon vectors."""
    snapshot = _source_snapshot(source_root)
    source_ok = all(row["status"] == "PASS" for row in snapshot["files"])
    root = Path(source_root)
    discr = (root / "src/algorithms/tracking/libs/tracking_discriminators.cc").read_text()
    loop = (root / "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc").read_text()
    source_equations = {
        "costas_cycles": "pll_cloop_two_quadrant_atan(d_P_accu) / TWO_PI" in loop,
        "dll_chips": "dll_nc_e_minus_l_normalized" in discr and "(P_early - P_late) / E_plus_L" in discr,
        "carrier_nco": "d_carrier_doppler_hz = d_carr_error_filt_hz" in loop,
        "code_nco": "d_code_freq_chips = d_code_chip_rate - d_code_error_filt_chips" in loop,
        "carrier_aiding": "d_code_freq_chips += d_carrier_doppler_hz * d_code_chip_rate / d_signal_carrier_freq" in loop,
        "cadence_1ms_base": "d_correlation_length_ms = 1" in loop,
        "tap_order": "d_local_code_shift_chips[n] = static_cast<float>(n - 4) * spacing" in loop,
    }
    # Source tap order [-... E(-d),P(0),L(+d) ...] and triangular C/A
    # autocorrelation: positive physical delay makes E-L negative.
    spacing, epsilon_chip = .125, .01 / (299_792_458 / 1_023_000)
    triangular = lambda offset: max(1 - abs(offset), 0)
    early, late = triangular(epsilon_chip + spacing), triangular(epsilon_chip - spacing)
    dll_error = (1 - spacing) * (early - late) / (early + late)
    # Positive range rotates the wiped prompt by -2*pi*rho/lambda.
    wavelength = 299_792_458 / 1_575_420_000
    pll_cycles = np.arctan2(np.sin(-2 * np.pi * .01 / wavelength), np.cos(-2 * np.pi * .01 / wavelength)) / (2 * np.pi)
    ratio = 1_023_000 / 1_575_420_000
    synthetic_ok = dll_error < 0 and pll_cycles < 0 and abs(ratio - 1 / 1540) <= 1e-15
    equations_ok = all(source_equations.values())
    validated = ["code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"] if source_ok and equations_ok and synthetic_ok else []
    return {
        "schema": "gnss-doppler-lab.gcspo-stage0.receiver-semantic-preflight.v1",
        "overall_status": "PASS" if len(validated) == 4 else "FAIL",
        "validated_rows": validated,
        "source_snapshot": snapshot,
        "source_equations": source_equations,
        "field_semantics": {"carr_error_hz": "cycles", "code_error_chips": "chips", "carrier_doppler_hz": "Hz NCO state", "code_freq_chips": "chips/s NCO state"},
        "wrap_linear_range_cycles": [-.25, .25],
        "integration": {"base_integration_s": .001, "scientific_epoch_s": .02},
        "synthetic_vectors": {
            "range_impulse": {"rho_m": .01, "code_error_chips": float(dll_error), "pll_phase_error_cycles": float(pll_cycles), "sign": "negative"},
            "rate_ramp": {"rhodot_m_s": .001, "carrier_doppler_hz": -.001 / wavelength,
                          "code_frequency_offset_chips_s": -.001 / (299_792_458 / 1_023_000), "carrier_aiding_ratio": ratio},
        },
        "dirty_source_interpretation": "permitted for byte-pinned existing outputs; exact replay remains unavailable",
    }


def build_artifact_manifest(artifact_dir):
    root = Path(artifact_dir)
    temporary = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and (p.name.endswith(".tmp") or ".tmp." in p.name)]
    if temporary: raise ValueError(f"temporary artifacts are forbidden: {temporary}")
    records = []
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in MANIFEST_EXCLUSIONS: continue
        records.append({"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"schema": "gnss-doppler-lab.gcspo-stage0.artifact-manifest.v1", "files": records}

def prepare_valid_artifact_manifest(artifact_dir):
    """Quarantine pre-access-only files and write the frozen valid manifest."""
    root = Path(artifact_dir)
    required_before_reports = VALID_SCIENCE_REQUIRED - MANIFEST_EXCLUSIONS
    missing = sorted(name for name in required_before_reports if not (root / name).is_file())
    plot_files = [path for path in (root / "plots").rglob("*") if path.is_file()] if (root / "plots").is_dir() else []
    if missing or not plot_files:
        raise ValueError(f"valid science artifacts incomplete before manifest: missing={missing} plots={len(plot_files)}")
    keep = required_before_reports | VALID_ADDITIONS | {"plots"}
    extras = [child for child in root.iterdir() if child.name not in keep]
    if extras:
        quarantine = Path(tempfile.mkdtemp(prefix=f".{root.name}.valid-quarantine-", dir=root.parent))
        for child in extras:
            shutil.move(str(child), quarantine / child.name)
    payload = build_artifact_manifest(root)
    observed = {row["path"] for row in payload["files"]}
    required_members = required_before_reports | {path.relative_to(root).as_posix() for path in plot_files}
    if not required_members.issubset(observed):
        raise ValueError("valid manifest omits required science artifacts")
    canonical_write_json(root / "artifact_manifest_sha256.json", payload)
    return payload


def quarantine_failed_final_verdict(artifact_dir):
    """Prevent a post-start exception from leaving a claimable science verdict."""
    root = Path(artifact_dir)
    verdict = root / "final_verdict.json"
    if not verdict.is_file():
        return None
    quarantine = Path(tempfile.mkdtemp(prefix=f".{root.name}.poststart-quarantine-", dir=root.parent))
    target = quarantine / verdict.name
    shutil.move(str(verdict), target)
    return target

def verify_artifact_manifest(artifact_dir, manifest=None):
    root = Path(artifact_dir)
    payload = manifest if manifest is not None else json.loads((root / "artifact_manifest_sha256.json").read_text())
    expected_paths = [row["path"] for row in payload.get("files", [])]
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        raise ValueError("manifest paths are not sorted and unique")
    actual = build_artifact_manifest(root)
    if payload != actual:
        raise ValueError("artifact manifest checksum/membership mismatch")
    return payload


def write_fail_closed_invalid(artifact_dir, *, reason_codes, failed_checks, target_commit, started_utc=None):
    root = Path(artifact_dir); root.mkdir(parents=True, exist_ok=True)
    ledger = root / "access_ledger.jsonl"
    if ledger.is_file() and ledger.stat().st_size:
        raise RuntimeError("pre-access invalid path cannot erase protected access records")
    extras = [child for child in root.iterdir() if child.name not in FROZEN_HASHES]
    if extras:
        quarantine = Path(tempfile.mkdtemp(prefix=f".{root.name}.invalid-quarantine-", dir=root.parent))
        for child in extras:
            shutil.move(str(child), quarantine / child.name)
    (root / "access_ledger.jsonl").write_bytes(b"")
    invalid = {"schema": "gnss-doppler-lab.gcspo-stage0.invalid-run.v1", "run_status": "INVALID_EXPERIMENT_NO_ATTACK_ACCESS",
               "reason_codes": sorted(set(reason_codes)), "failed_checks": failed_checks, "attack_access_count": 0,
               "target_commit": target_commit, "config_sha256": FROZEN_HASHES["config.json"],
               "started_utc": started_utc or utc_now(), "failed_utc": utc_now()}
    canonical_write_json(root / "invalid_run.json", invalid)
    canonical_write_json(root / "artifact_manifest_sha256.json", build_artifact_manifest(root))
    return invalid


def verify_invalid_artifacts(artifact_dir, *, allow_missing_reports=False):
    root = Path(artifact_dir)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    expected = set(INVALID_REQUIRED)
    if allow_missing_reports: expected -= {"verifier_report.json", "fresh_clone_verifier_report.json"}
    if actual != expected: raise ValueError(f"invalid artifact exact-set mismatch: expected={sorted(expected)} actual={sorted(actual)}")
    if (root / "final_verdict.json").exists(): raise ValueError("invalid run cannot contain final verdict")
    for name, expected_hash in FROZEN_HASHES.items():
        if sha256_file(root / name) != expected_hash: raise ValueError(f"frozen checksum mismatch: {name}")
    payload = json.loads((root / "invalid_run.json").read_text())
    if payload.get("run_status") != "INVALID_EXPERIMENT_NO_ATTACK_ACCESS" or payload.get("attack_access_count") != 0:
        raise ValueError("invalid run status/access count mismatch")
    verify_artifact_manifest(root)
    return payload
