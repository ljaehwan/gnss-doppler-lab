"""Bounded CRPA validation helpers for Jammertest 2025 Stage-0B.

The raw NPY is always opened read-only with ``np.load(..., mmap_mode="r",
allow_pickle=False)``.  Destruction controls operate on in-memory batches and
never modify the source object.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


EXPECTED_BYTES = 1_398_308_992
EXPECTED_SHA256 = "d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f"
EXPECTED_SHAPE = (42_673, 4, 1_024)
SEED = 20_250_823
PAIR_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
BLOCK_SIZES = (32, 128, 512, 2_048)
ALLOWED_VERDICTS = {
    "STOP_SCHEMA_INVALID",
    "STOP_NO_USABLE_SPATIAL_COHERENCE",
    "NO_SPATIAL_INCREMENT_FOR_SPOOF_VS_JAMMER",
    "INCONCLUSIVE_SPATIAL_SIGNAL_PROVENANCE_BLOCKED",
    "PROMISING_EXPLORATORY_SPATIAL_INCREMENT_REQUIRES_PROVENANCE",
}
REQUIRED_ARTIFACTS = {
    "README.md",
    "final_verdict.json",
    "download_integrity.json",
    "npy_schema_observed.json",
    "access_audit.json",
    "label_binding_audit.json",
    "channel_quality.json",
    "spatial_metrics_summary.json",
    "lag_coherence_results.csv",
    "destruction_control_results.json",
    "blocked_split_manifest.json",
    "block_sensitivity.json",
    "confound_analysis.md",
    "artifact_manifest_sha256.json",
    "test_output.txt",
    "verifier_output.txt",
}


CLASS_NAMES = {
    0: "CW",
    1: "Sweep",
    2: "Prn",
    3: "Meac",
    4: "Spoof",
    5: "Chirp",
    6: "ChirpB",
    7: "Triang",
    8: "Meac,Prn",
    9: "Meac,Spoof",
    10: "Mod",
    11: "ChirpM",
    12: "ChirpMS",
    13: "FmS",
    14: "Chirp,Spoof",
    15: "Chirp,Prn",
    16: "Chirp,Prn,Triang",
}


@dataclass
class FeatureSet:
    power: np.ndarray
    single: np.ndarray
    spatial: np.ndarray
    mean_coherence: np.ndarray
    lambda1_fraction: np.ndarray
    effective_rank: np.ndarray
    condition_number: np.ndarray
    eigenvalue_fractions: np.ndarray
    coherences: np.ndarray
    peak_lags: np.ndarray
    peak_lag_coherence: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(artifact: Path) -> dict:
    excluded = {
        "artifact_manifest_sha256.json",
        "test_output.txt",
        "verifier_output.txt",
    }
    files = {
        path.relative_to(artifact).as_posix(): sha256_file(path)
        for path in sorted(artifact.rglob("*"))
        if path.is_file() and path.name not in excluded
    }
    value = {
        "algorithm": "sha256",
        "excluded_self_and_mutable_logs": sorted(excluded),
        "files": files,
    }
    write_json(artifact / "artifact_manifest_sha256.json", value)
    return value


def open_crpa_memmap(path: Path) -> np.memmap:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise ValueError("NPY did not open as a read-only memmap")
    if array.mode != "r":
        raise ValueError(f"unexpected memmap mode: {array.mode}")
    return array


def observe_npy_schema(path: Path) -> tuple[np.memmap, dict]:
    array = open_crpa_memmap(path)
    # Only the small NPY header is parsed separately; all array access remains
    # through the allow_pickle=False read-only memmap above.
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)
    observed = {
        "npy_version": list(version),
        "header": {
            "descr": dtype.str,
            "fortran_order": bool(fortran_order),
            "shape": list(shape),
        },
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "dtype_str": array.dtype.str,
        "byte_order": array.dtype.byteorder,
        "c_contiguous": bool(array.flags.c_contiguous),
        "f_contiguous": bool(array.flags.f_contiguous),
        "data_offset_bytes": int(array.offset),
        "file_size_bytes": path.stat().st_size,
        "read_mode": "np.load(path, mmap_mode='r', allow_pickle=False)",
        "schema_valid": bool(
            array.shape == EXPECTED_SHAPE
            and array.dtype == np.dtype(np.complex64)
            and array.flags.c_contiguous
            and not array.flags.f_contiguous
            and path.stat().st_size == EXPECTED_BYTES
        ),
    }
    return array, observed


def load_label_rows(split_root: Path) -> list[dict]:
    rows: list[dict] = []
    for area in (1, 2):
        for split in ("train", "test"):
            path = split_root / f"{split}_crpa_{area}.txt"
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                fields = line.split("\t")
                if len(fields) != 4:
                    raise ValueError(f"{path}:{line_number}: expected four fields")
                class_id = int(fields[1])
                rows.append(
                    {
                        "sample_index": int(fields[0]),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "transmit_power_dbm": float(fields[2]),
                        "bandwidth_mhz": float(fields[3]),
                        "area": area,
                        "public_split": split,
                    }
                )
    rows.sort(key=lambda row: row["sample_index"])
    indices = [row["sample_index"] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate sample_index across public CRPA split files")
    return rows


def public_reader_view(snapshot: np.ndarray) -> np.ndarray:
    """Reproduce the official utilities_crpa.py reshape without torch."""

    return (
        snapshot.view(np.float32)
        .reshape((-1, 1_024, 2))
        .transpose((1, 0, 2))
        .reshape((1_024, 8))
        .transpose((1, 0))
    )


def expected_reader_view(snapshot: np.ndarray) -> np.ndarray:
    return np.stack(
        [component for channel in snapshot for component in (channel.real, channel.imag)],
        axis=0,
    )


def channel_quality(array: np.memmap, batch_size: int = 512) -> dict:
    channels = array.shape[1]
    total = array.shape[0] * array.shape[2]
    finite = np.zeros(channels, dtype=np.int64)
    nan = np.zeros(channels, dtype=np.int64)
    inf = np.zeros(channels, dtype=np.int64)
    zeros = np.zeros(channels, dtype=np.int64)
    sum_power = np.zeros(channels, dtype=np.float64)
    repeated = np.zeros(channels, dtype=np.int64)
    extrema = [
        {"real_min": np.inf, "real_max": -np.inf, "imag_min": np.inf, "imag_max": -np.inf,
         "real_min_count": 0, "real_max_count": 0, "imag_min_count": 0, "imag_max_count": 0}
        for _ in range(channels)
    ]
    pair_equal = np.zeros(len(PAIR_INDICES), dtype=np.int64)

    def update_extreme(state: dict, key: str, value: float, count: int, is_min: bool) -> None:
        old = state[key]
        if (is_min and value < old) or ((not is_min) and value > old):
            state[key] = value
            state[key + "_count"] = count
        elif value == old:
            state[key + "_count"] += count

    for start in range(0, array.shape[0], batch_size):
        x = np.asarray(array[start : start + batch_size])
        valid = np.isfinite(x)
        finite += valid.sum(axis=(0, 2))
        nan += (np.isnan(x.real) | np.isnan(x.imag)).sum(axis=(0, 2))
        inf += (np.isinf(x.real) | np.isinf(x.imag)).sum(axis=(0, 2))
        zeros += (x == 0).sum(axis=(0, 2))
        safe = np.where(valid, x, 0)
        sum_power += np.sum(np.abs(safe) ** 2, axis=(0, 2), dtype=np.float64)
        repeated += (x[:, :, 1:] == x[:, :, :-1]).sum(axis=(0, 2))
        for channel in range(channels):
            real = x[:, channel, :].real
            imag = x[:, channel, :].imag
            for component, values in (("real", real), ("imag", imag)):
                local_min = float(np.nanmin(values))
                local_max = float(np.nanmax(values))
                update_extreme(extrema[channel], component + "_min", local_min, int((values == local_min).sum()), True)
                update_extreme(extrema[channel], component + "_max", local_max, int((values == local_max).sum()), False)
        for pair_no, (left, right) in enumerate(PAIR_INDICES):
            pair_equal[pair_no] += int((x[:, left, :] == x[:, right, :]).sum())

    per_channel = []
    repeat_denominator = array.shape[0] * (array.shape[2] - 1)
    for channel in range(channels):
        ext = extrema[channel]
        clipped_count = sum(ext[key] for key in (
            "real_min_count", "real_max_count", "imag_min_count", "imag_max_count"
        ))
        per_channel.append(
            {
                "channel": channel,
                "rms": float(np.sqrt(sum_power[channel] / finite[channel])),
                "finite_ratio": float(finite[channel] / total),
                "nan_ratio": float(nan[channel] / total),
                "inf_ratio": float(inf[channel] / total),
                "zero_ratio": float(zeros[channel] / total),
                "adjacent_exact_repeat_ratio": float(repeated[channel] / repeat_denominator),
                "observed_extreme_component_ratio": float(clipped_count / (2 * total)),
                **{key: (int(value) if key.endswith("_count") else float(value)) for key, value in ext.items()},
            }
        )
    duplicates = [
        {
            "pair": f"{left}-{right}",
            "exact_equal_ratio": float(pair_equal[pair_no] / total),
            "byte_identical_channels": bool(pair_equal[pair_no] == total),
        }
        for pair_no, (left, right) in enumerate(PAIR_INDICES)
    ]
    return {
        "complex_values_per_channel": total,
        "channels": per_channel,
        "channel_pair_exact_equality": duplicates,
        "any_byte_identical_channel_pair": any(item["byte_identical_channels"] for item in duplicates),
        "clipping_definition": "fraction of real/imag components equal to each channel's observed extrema",
        "repeat_definition": "fraction of adjacent complex time samples exactly equal within a snapshot",
    }


def normalize_channels(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x, dtype=np.complex128)
    centered = values - np.mean(values, axis=-1, keepdims=True)
    power = np.mean(np.abs(centered) ** 2, axis=-1)
    scale = np.sqrt(np.maximum(power, np.finfo(np.float64).tiny))
    return centered / scale[:, :, None], power


def compute_features(x: np.ndarray) -> FeatureSet:
    y, channel_power = normalize_channels(x)
    n_samples = y.shape[-1]
    covariance = np.einsum("bct,bdt->bcd", y, y.conj(), optimize=True) / n_samples
    covariance = (covariance + covariance.conj().transpose(0, 2, 1)) / 2
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0)
    fractions = eigenvalues[:, ::-1] / np.maximum(
        eigenvalues.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny
    )
    positive_fractions = np.maximum(fractions, np.finfo(np.float64).tiny)
    entropy = -np.sum(np.where(fractions > 0, fractions * np.log(positive_fractions), 0), axis=1)
    effective_rank = np.exp(entropy)
    condition = eigenvalues[:, -1] / np.maximum(eigenvalues[:, 0], np.finfo(np.float64).eps)
    coherences = np.stack([covariance[:, left, right] for left, right in PAIR_INDICES], axis=1)
    mean_coherence = np.mean(np.abs(coherences), axis=1)

    spectrum = np.fft.fft(y, axis=-1)
    single_spectrum = np.abs(spectrum[:, 0, :]) ** 2
    band_energy = single_spectrum.reshape((-1, 16, 64)).sum(axis=2)
    band_fraction = band_energy / np.maximum(band_energy.sum(axis=1, keepdims=True), np.finfo(np.float64).tiny)
    amplitude = np.abs(y[:, 0, :])
    single = np.column_stack(
        [
            10 * np.log10(np.maximum(channel_power[:, 0], np.finfo(np.float64).tiny)),
            amplitude.mean(axis=1),
            amplitude.std(axis=1),
            np.quantile(amplitude, 0.9, axis=1),
            np.quantile(amplitude, 0.99, axis=1),
            np.log10(np.maximum(band_fraction, np.finfo(np.float64).tiny)),
        ]
    )
    power = 10 * np.log10(
        np.maximum(channel_power.mean(axis=1), np.finfo(np.float64).tiny)
    )[:, None]
    spatial = np.column_stack(
        [
            fractions,
            coherences.real,
            coherences.imag,
            np.abs(coherences),
            effective_rank,
            np.log10(np.maximum(condition, 1.0)),
        ]
    )

    lags = np.arange(-16, 17)
    lag_indices = np.mod(lags, n_samples)
    peak_lags = np.empty((x.shape[0], len(PAIR_INDICES)), dtype=np.int16)
    peak_values = np.empty((x.shape[0], len(PAIR_INDICES)), dtype=np.float64)
    for pair_no, (left, right) in enumerate(PAIR_INDICES):
        correlation = np.fft.ifft(spectrum[:, left, :] * spectrum[:, right, :].conj(), axis=-1) / n_samples
        window = np.abs(correlation[:, lag_indices])
        argmax = np.argmax(window, axis=1)
        peak_lags[:, pair_no] = lags[argmax]
        peak_values[:, pair_no] = window[np.arange(x.shape[0]), argmax]
    return FeatureSet(
        power=power,
        single=single,
        spatial=spatial,
        mean_coherence=mean_coherence,
        lambda1_fraction=fractions[:, 0],
        effective_rank=effective_rank,
        condition_number=condition,
        eigenvalue_fractions=fractions,
        coherences=coherences,
        peak_lags=peak_lags,
        peak_lag_coherence=peak_values,
    )


def mismatch_source_map(rows: list[dict]) -> np.ndarray:
    groups: dict[tuple, list[int]] = {}
    for position, row in enumerate(rows):
        key = (row["area"], row["class_id"], row["transmit_power_dbm"])
        groups.setdefault(key, []).append(position)
    mapping = np.empty((len(rows), 4), dtype=np.int64)
    for positions in groups.values():
        positions_array = np.asarray(positions, dtype=np.int64)
        if len(positions_array) < 2:
            raise ValueError("cannot construct mismatched tuple from singleton stratum")
        for channel in range(4):
            offset = 1 + channel
            offset = 1 + ((offset - 1) % (len(positions_array) - 1))
            mapping[positions_array, channel] = np.roll(positions_array, offset)
    return mapping


def mismatch_batch(all_data: np.ndarray, mapping: np.ndarray, positions: np.ndarray) -> np.ndarray:
    result = np.empty((len(positions), 4, all_data.shape[2]), dtype=all_data.dtype)
    for channel in range(4):
        result[:, channel, :] = all_data[mapping[positions, channel], channel, :]
    return result


def circular_shift_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shifts = rng.integers(0, x.shape[-1], size=(x.shape[0], x.shape[1]))
    sample_axis = np.arange(x.shape[-1])[None, None, :]
    gather = (sample_axis - shifts[:, :, None]) % x.shape[-1]
    return np.take_along_axis(x, gather, axis=-1)


def phase_randomize_batch(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.fft(x, axis=-1)
    phase = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
    return np.fft.ifft(spectrum * np.exp(1j * phase), axis=-1).astype(np.complex64)


def block_bootstrap_mean_difference(
    actual: np.ndarray,
    control: np.ndarray,
    sample_indices: np.ndarray,
    block_size: int,
    *,
    seed: int = SEED,
    replicates: int = 2_000,
) -> dict:
    difference = np.asarray(actual) - np.asarray(control)
    block_ids = sample_indices // block_size
    unique, inverse = np.unique(block_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=difference)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed + block_size)
    bootstrap = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        stop = min(start + 100, replicates)
        draws = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        bootstrap[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "block_size": block_size,
        "block_count": int(len(unique)),
        "sample_count": int(len(difference)),
        "mean_difference": float(np.mean(difference)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "actual_significantly_higher": bool(low > 0),
        "bootstrap_replicates": replicates,
        "grouping": "floor(sample_index / block_size), resampled as contiguous-coordinate blocks",
    }


def blocked_roles(sample_indices: np.ndarray, block_size: int) -> np.ndarray:
    phase = (sample_indices // block_size) % 5
    return np.where(np.isin(phase, (0, 1)), "train", np.where(phase == 3, "test", "guard"))


def balanced_classification_positions(
    rows: list[dict], block_size: int, *, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, dict]:
    indices = np.asarray([row["sample_index"] for row in rows], dtype=np.int64)
    roles = blocked_roles(indices, block_size)
    positive = np.asarray([
        ("Spoof" in row["class_name"] or "Meac" in row["class_name"])
        for row in rows
    ])
    eligible = np.asarray([
        row["area"] == 1 and row["transmit_power_dbm"] in (15.0, 25.0, 30.0, 35.0, 40.0)
        for row in rows
    ])
    powers = np.asarray([row["transmit_power_dbm"] for row in rows])
    rng = np.random.default_rng(seed + block_size)
    selected: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    counts: dict[str, dict[str, dict[str, int]]] = {"train": {}, "test": {}}
    for role in ("train", "test"):
        for power in (15.0, 25.0, 30.0, 35.0, 40.0):
            pos = np.flatnonzero(eligible & (roles == role) & (powers == power) & positive)
            neg = np.flatnonzero(eligible & (roles == role) & (powers == power) & ~positive)
            count = min(len(pos), len(neg))
            if count == 0:
                raise ValueError(f"empty balanced cell: block={block_size} role={role} power={power}")
            pos = np.sort(rng.choice(pos, size=count, replace=False))
            neg = np.sort(rng.choice(neg, size=count, replace=False))
            selected[role].extend((pos, neg))
            counts[role][str(int(power))] = {
                "positive": count,
                "negative": count,
                "total": 2 * count,
            }
    train = np.sort(np.concatenate(selected["train"]))
    test = np.sort(np.concatenate(selected["test"]))
    manifest = {
        "block_size": block_size,
        "role_pattern": ["train", "train", "guard", "test", "guard"],
        "group_formula": "floor(sample_index / block_size)",
        "proxy_only_not_recording_safe": True,
        "common_transmit_power_dbm": [15, 25, 30, 35, 40],
        "counts": counts,
        "train_count": int(len(train)),
        "test_count": int(len(test)),
        "guard_eligible_count": int(np.sum(eligible & (roles == "guard"))),
        "train_sample_index_sha256": hashlib.sha256(indices[train].astype("<i8").tobytes()).hexdigest(),
        "test_sample_index_sha256": hashlib.sha256(indices[test].astype("<i8").tobytes()).hexdigest(),
    }
    return train, test, manifest


def verify_artifact(artifact: Path) -> list[str]:
    errors: list[str] = []
    classifier_options = {"classifier_results.json", "classifier_not_run.json"}
    missing = sorted(name for name in REQUIRED_ARTIFACTS if not (artifact / name).is_file())
    if missing:
        errors.append(f"missing required artifacts: {missing}")
    present_classifier = sorted(name for name in classifier_options if (artifact / name).is_file())
    if len(present_classifier) != 1:
        errors.append(f"expected exactly one classifier artifact, found {present_classifier}")
    figures = artifact / "figures"
    if not figures.is_dir() or not any(figures.glob("*.png")):
        errors.append("figures directory has no PNG files")

    try:
        manifest = read_json(artifact / "artifact_manifest_sha256.json")
        for relative, expected in manifest["files"].items():
            path = artifact / relative
            if not path.is_file():
                errors.append(f"manifest target missing: {relative}")
            elif sha256_file(path) != expected:
                errors.append(f"manifest hash mismatch: {relative}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid manifest: {exc}")

    try:
        integrity = read_json(artifact / "download_integrity.json")
        if integrity["actual_size_bytes"] != EXPECTED_BYTES:
            errors.append("download size binding mismatch")
        if integrity["actual_sha256"] != EXPECTED_SHA256:
            errors.append("download hash binding mismatch")
        if integrity["downloaded_lfs_object_count"] != 1:
            errors.append("download object count is not one")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid download integrity: {exc}")

    try:
        schema = read_json(artifact / "npy_schema_observed.json")
        if schema["schema_valid"] != (
            schema["shape"] == list(EXPECTED_SHAPE) and schema["dtype"] == "complex64"
        ):
            errors.append("schema gate contradicts observations")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid schema artifact: {exc}")

    try:
        access = read_json(artifact / "access_audit.json")
        if access["downloaded_payload_bytes"] != EXPECTED_BYTES:
            errors.append("access download byte count mismatch")
        for field in (
            "other_lfs_payload_bytes",
            "innosense_hdf5_bytes",
            "texbat_payload_bytes",
            "oakbat_payload_bytes",
            "tuni_payload_bytes",
        ):
            if access[field] != 0:
                errors.append(f"forbidden access nonzero: {field}")
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        errors.append(f"invalid access audit: {exc}")

    try:
        verdict = read_json(artifact / "final_verdict.json")
        if verdict["verdict"] not in ALLOWED_VERDICTS:
            errors.append(f"invalid verdict: {verdict['verdict']}")
        if verdict["ready_for_wcl_declared"]:
            errors.append("forbidden READY_FOR_WCL claim")
        if verdict["task"] != "spoof_meacon_vs_non_deceptive_terrestrial_jammer":
            errors.append("task contract mismatch")
        if verdict["verdict"] == "STOP_SCHEMA_INVALID" and schema["schema_valid"]:
            errors.append("STOP_SCHEMA_INVALID contradicts observed schema")
        if verdict["classification_run"] != ("classifier_results.json" in present_classifier):
            errors.append("classification_run contradicts classifier artifact")
        if (not verdict["spatial_gate_passed"]) and verdict["classification_run"]:
            errors.append("classifier ran despite spatial gate failure")
    except (OSError, KeyError, json.JSONDecodeError, UnboundLocalError) as exc:
        errors.append(f"invalid final verdict: {exc}")
    return errors
