
"""Per-PRN tracking-correlator peak extraction from GNSS-SDR MAT dumps."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Iterable

import h5py
import numpy as np


# GNSS-SDR 0.0.19 emits real E/P/L only; abs_VE/abs_VL are zero placeholders.
TAP_DATASETS: tuple[tuple[str, str], ...] = (("E", "abs_E"), ("P", "abs_P"), ("L", "abs_L"))


@dataclass(frozen=True)
class TrackingPeakSeries:
    """Tracked correlator-tap magnitudes for one PRN over time."""

    prn: str
    channel: int
    sample_rate_hz: int
    time_s: np.ndarray
    tap_names: tuple[str, ...]
    magnitudes: np.ndarray
    carrier_doppler_hz: np.ndarray
    cn0_db_hz: np.ndarray
    prompt_i: np.ndarray
    prompt_q: np.ndarray
    code_error_chips: np.ndarray
    code_freq_chips: np.ndarray
    source_mat_path: Path
    # Zero-based position of this contiguous PRN run within its channel file.
    segment_index: int = 0

    @property
    def prompt_magnitude(self) -> np.ndarray:
        return np.hypot(self.prompt_i, self.prompt_q)


def _read_vector(handle: h5py.File, name: str) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"Tracking MAT is missing dataset: {name}")
    return np.asarray(handle[name]).reshape(-1)


def _channel_from_name(path: Path) -> int:
    match = re.search(r"_ch_(\d+)\.mat$", path.name)
    if not match:
        raise ValueError(f"Cannot determine channel from {path.name}")
    return int(match.group(1))


def _normalized_prn(prn: str | int) -> str:
    if isinstance(prn, int):
        return f"G{prn:02d}"
    text = str(prn).strip().upper()
    if text.startswith("G") and text[1:].isdigit():
        return f"G{int(text[1:]):02d}"
    if text.isdigit():
        return f"G{int(text):02d}"
    raise ValueError(f"Unsupported PRN identifier: {prn}")


def _receiver_manifest(receiver_run_dir: str | Path) -> dict[str, object]:
    manifest_path = Path(receiver_run_dir) / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _raw_mat_paths(receiver_run_dir: str | Path) -> list[Path]:
    run_dir = Path(receiver_run_dir)
    manifest = _receiver_manifest(run_dir)
    raw_directory = manifest.get("tracking", {}).get("raw_directory", "raw")
    return sorted((run_dir / str(raw_directory)).glob("epl_tracking_ch_*.mat"), key=_channel_from_name)


def _valid_prn_number(value: object) -> int | None:
    """Return a valid GPS L1 C/A PRN number, excluding dump sentinels."""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if 1 <= number <= 32 else None


def available_tracking_prns(receiver_run_dir: str | Path) -> list[str]:
    """Discover every valid epoch PRN in deterministic first-seen order."""
    discovered: list[str] = []
    seen: set[str] = set()
    for path in _raw_mat_paths(receiver_run_dir):
        with h5py.File(path, "r") as handle:
            values = _read_vector(handle, "PRN")
        for value in values:
            number = _valid_prn_number(value)
            if number is None:
                continue
            prn = _normalized_prn(number)
            if prn not in seen:
                seen.add(prn)
                discovered.append(prn)
    return discovered


def _slice_indices(indices: np.ndarray, *, max_epochs: int | None, epoch_step: int) -> np.ndarray:
    if epoch_step < 1:
        raise ValueError("epoch_step must be at least 1")
    selected = indices[::epoch_step]
    if max_epochs is not None:
        if max_epochs < 0:
            raise ValueError("max_epochs must be non-negative or None")
        selected = selected[:max_epochs]
    return selected


def _series_from_indices(mat_path: Path, handle: h5py.File, indices: np.ndarray, *, sample_rate_hz: int, prn: str, segment_index: int) -> TrackingPeakSeries:
    epoch_count = len(_read_vector(handle, "PRN"))
    def take(name: str) -> np.ndarray:
        values = _read_vector(handle, name)
        if len(values) != epoch_count:
            raise ValueError(f"Tracking MAT dataset length mismatch for {name}: {mat_path}")
        return values[indices]
    sample_counts = take("PRN_start_sample_count")
    tap_names: list[str] = []
    tap_columns: list[np.ndarray] = []
    for label, dataset in TAP_DATASETS:
        if dataset not in handle:
            continue
        values = take(dataset).astype(np.float64)
        if np.allclose(values, 0.0):
            raise ValueError(f"Real E/P/L correlator tap {label} is all-zero: {mat_path}")
        tap_names.append(label)
        tap_columns.append(values)
    if indices.size == 0:
        raise ValueError(f"Tracking segment has no epochs: {mat_path}")
    if not tap_columns:
        raise ValueError(f"Tracking MAT does not contain non-zero correlator taps: {mat_path}")
    return TrackingPeakSeries(
        prn=prn, channel=_channel_from_name(mat_path), sample_rate_hz=sample_rate_hz,
        time_s=sample_counts.astype(np.float64) / float(sample_rate_hz), tap_names=tuple(tap_names),
        magnitudes=np.column_stack(tap_columns), carrier_doppler_hz=take("carrier_doppler_hz").astype(np.float64),
        cn0_db_hz=take("CN0_SNV_dB_Hz").astype(np.float64), prompt_i=take("Prompt_I").astype(np.float64),
        prompt_q=take("Prompt_Q").astype(np.float64), code_error_chips=take("code_error_chips").astype(np.float64),
        code_freq_chips=take("code_freq_chips").astype(np.float64), source_mat_path=mat_path,
        segment_index=segment_index,
    )


def load_receiver_tracking_peak_series_segments(receiver_run_dir: str | Path, prn: str | int, *, max_epochs: int | None = None, epoch_step: int = 1) -> list[TrackingPeakSeries]:
    """Load distinct contiguous channel segments for prn."""
    run_dir = Path(receiver_run_dir)
    manifest = _receiver_manifest(run_dir)
    sample_rate_hz = int(manifest["source"]["sample_rate_hz"])
    target = _normalized_prn(prn)
    result: list[TrackingPeakSeries] = []
    for mat_path in _raw_mat_paths(run_dir):
        with h5py.File(mat_path, "r") as handle:
            raw_prns = _read_vector(handle, "PRN")
            normalized = [_normalized_prn(number) if (number := _valid_prn_number(value)) is not None else None for value in raw_prns]
            start = 0
            segment_index = 0
            while start < len(normalized):
                value = normalized[start]
                end = start + 1
                while end < len(normalized) and normalized[end] == value:
                    end += 1
                if value is not None:
                    if value == target:
                        indices = _slice_indices(np.arange(start, end), max_epochs=max_epochs, epoch_step=epoch_step)
                        if len(indices):
                            result.append(_series_from_indices(mat_path, handle, indices, sample_rate_hz=sample_rate_hz, prn=target, segment_index=segment_index))
                    segment_index += 1
                start = end
    if not result:
        raise FileNotFoundError(f"No tracking MAT segment found for {target}")
    return result


def load_receiver_tracking_peak_series(receiver_run_dir: str | Path, prn: str | int, *, max_epochs: int | None = None, epoch_step: int = 1) -> TrackingPeakSeries:
    """Load one PRN, concatenating segments for backwards compatibility."""
    segments = load_receiver_tracking_peak_series_segments(receiver_run_dir, prn, max_epochs=max_epochs, epoch_step=epoch_step)
    if len(segments) == 1:
        return segments[0]
    first = segments[0]
    return TrackingPeakSeries(
        prn=first.prn, channel=first.channel, sample_rate_hz=first.sample_rate_hz,
        time_s=np.concatenate([x.time_s for x in segments]), tap_names=first.tap_names,
        magnitudes=np.concatenate([x.magnitudes for x in segments]),
        carrier_doppler_hz=np.concatenate([x.carrier_doppler_hz for x in segments]),
        cn0_db_hz=np.concatenate([x.cn0_db_hz for x in segments]), prompt_i=np.concatenate([x.prompt_i for x in segments]),
        prompt_q=np.concatenate([x.prompt_q for x in segments]), code_error_chips=np.concatenate([x.code_error_chips for x in segments]),
        code_freq_chips=np.concatenate([x.code_freq_chips for x in segments]), source_mat_path=first.source_mat_path,
        segment_index=first.segment_index,
    )


def render_tracking_peak_dashboard(
    series: TrackingPeakSeries,
    *,
    output_path: str | Path,
    title: str = "GNSS-SDR tracking correlator peak dashboard",
    epoch_index: int | None = None,
) -> Path:
    """Render a paper-friendly dashboard for one PRN's tracked peak slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    if series.magnitudes.ndim != 2 or series.magnitudes.shape[0] == 0:
        raise ValueError("Tracking peak magnitudes must be a non-empty 2D matrix")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    chosen_epoch = series.magnitudes.shape[0] // 2 if epoch_index is None else int(epoch_index)
    if chosen_epoch < 0 or chosen_epoch >= series.magnitudes.shape[0]:
        raise IndexError("epoch_index is out of range")

    tap_indices = np.arange(len(series.tap_names), dtype=np.float64)
    tap_grid, time_grid = np.meshgrid(tap_indices, series.time_s, indexing="xy")

    fig = plt.figure(figsize=(17, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_heat = fig.add_subplot(grid[0, 1])
    ax_profile = fig.add_subplot(grid[1, 1])

    ax3d.plot_surface(
        tap_grid,
        time_grid,
        series.magnitudes,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=min(200, series.magnitudes.shape[0]),
        ccount=len(series.tap_names),
    )
    ax3d.set_title(f"PRN {series.prn} tracked peak ridge")
    ax3d.set_xlabel("Correlator tap")
    ax3d.set_ylabel("Time (s)")
    ax3d.set_zlabel("Magnitude")
    ax3d.set_xticks(tap_indices)
    ax3d.set_xticklabels(series.tap_names)
    ax3d.view_init(elev=28, azim=-130)

    heat = ax_heat.imshow(
        series.magnitudes,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[
            -0.5,
            len(series.tap_names) - 0.5,
            float(series.time_s[0]),
            float(series.time_s[-1]),
        ],
    )
    fig.colorbar(heat, ax=ax_heat, shrink=0.8, pad=0.02, label="Magnitude")
    ax_heat.set_title("Time-varying correlator peak slice")
    ax_heat.set_xlabel("Correlator tap")
    ax_heat.set_ylabel("Time (s)")
    ax_heat.set_xticks(tap_indices)
    ax_heat.set_xticklabels(series.tap_names)
    ax_heat.axhline(series.time_s[chosen_epoch], color="cyan", linestyle="--", linewidth=1.0)

    ax_profile.plot(tap_indices, series.magnitudes[chosen_epoch], marker="o", linewidth=1.5, label="tap magnitude")
    ax_profile.set_xticks(tap_indices)
    ax_profile.set_xticklabels(series.tap_names)
    ax_profile.set_xlabel("Correlator tap")
    ax_profile.set_ylabel("Magnitude")
    ax_profile.set_title(
        f"Selected epoch @ {series.time_s[chosen_epoch]:.3f} s | Doppler {series.carrier_doppler_hz[chosen_epoch]:.1f} Hz"
    )
    ax_profile.grid(alpha=0.25)

    twin = ax_profile.twinx()
    twin.plot(series.time_s, series.prompt_magnitude, color="tab:green", alpha=0.55, linewidth=1.0, label="|Prompt|")
    twin.plot(series.time_s, series.cn0_db_hz, color="tab:orange", alpha=0.55, linewidth=1.0, label="C/N0")
    twin.set_ylabel("Prompt/CN0 trend")

    lines = ax_profile.get_lines() + twin.get_lines()
    ax_profile.legend(lines, [line.get_label() for line in lines], loc="best")

    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.06, top=0.92, wspace=0.28, hspace=0.25)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output
