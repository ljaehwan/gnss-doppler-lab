"""Low-cost physical statistics for overlapping GNSS correlation peaks."""
from __future__ import annotations

import math
import re
from typing import Any, Iterable

import numpy as np


MAD_TO_SIGMA = 1.4826


def smoothstep_progress(time_s: float, start_s: float, transition_s: float) -> float:
    """C1-smooth carry-off progress in [0, 1]."""
    values = (float(time_s), float(start_s), float(transition_s))
    if not all(math.isfinite(value) for value in values) or transition_s <= 0:
        raise ValueError("time, start, and positive transition must be finite")
    fraction = min(1.0, max(0.0, (time_s - start_s) / transition_s))
    return fraction * fraction * (3.0 - 2.0 * fraction)


def linear_amplitude_ratio(
    time_s: float,
    *,
    start_s: float,
    ramp_s: float,
    initial_advantage_db: float,
    final_advantage_db: float,
) -> float:
    """Counterfeit/authentic voltage ratio used by simulation-v4."""
    values = (time_s, start_s, ramp_s, initial_advantage_db, final_advantage_db)
    if not all(math.isfinite(float(value)) for value in values) or ramp_s < 0:
        raise ValueError("amplitude-envelope inputs must be finite and ramp nonnegative")
    if time_s < start_s:
        return 0.0
    initial = 10.0 ** (initial_advantage_db / 20.0)
    final = 10.0 ** (final_advantage_db / 20.0)
    if ramp_s == 0:
        return final
    fraction = min(1.0, max(0.0, (time_s - start_s) / ramp_s))
    return initial + fraction * (final - initial)


def mixture_variance_excess(relative_amplitude: float, delay_chips: float) -> float:
    """Variance added by mixing two equal-shape peaks separated by delay.

    Treating a normalized correlation profile as a discrete distribution, a
    mixture with weights 1 and ``relative_amplitude`` obeys the exact
    within/between-mixture variance identity

        delta_V = rho / (1 + rho)^2 * delta_tau^2.
    """
    rho = float(relative_amplitude)
    delay = float(delay_chips)
    if not math.isfinite(rho) or rho < 0 or not math.isfinite(delay):
        raise ValueError("relative amplitude must be nonnegative and inputs finite")
    return rho / (1.0 + rho) ** 2 * delay**2


def displacement_envelope_proxy(
    event: dict[str, Any], time_s: float, *, chip_length_m: float
) -> float:
    """Geometry-agnostic upper-envelope proxy for carry-off peak broadening."""
    chip = float(chip_length_m)
    if not math.isfinite(chip) or chip <= 0:
        raise ValueError("chip_length_m must be positive and finite")
    target = np.asarray(event["target_offset_enu_m"], dtype=np.float64)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_offset_enu_m must contain three finite values")
    progress = smoothstep_progress(
        time_s, float(event["start_seconds"]), float(event["transition_seconds"])
    )
    delay_chips = float(np.linalg.norm(target)) * progress / chip
    ratio = linear_amplitude_ratio(
        time_s,
        start_s=float(event["start_seconds"]),
        ramp_s=float(event["power_ramp_seconds"]),
        initial_advantage_db=float(event["initial_advantage_db"]),
        final_advantage_db=float(event["final_advantage_db"]),
    )
    return mixture_variance_excess(ratio, delay_chips)


_GPS_SDR_SIM_LOS_ROW = re.compile(
    r"^(?P<prn>\d{2})\s+"
    r"(?P<azimuth>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"(?P<elevation>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"(?P<range>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    r"(?P<iono>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$",
    re.MULTILINE,
)


def enu_line_of_sight(
    azimuth_deg: float, elevation_deg: float
) -> tuple[float, float, float]:
    """Convert clockwise-from-north azimuth/elevation to an ENU unit vector."""
    azimuth = float(azimuth_deg)
    elevation = float(elevation_deg)
    if (
        not math.isfinite(azimuth)
        or not 0.0 <= azimuth < 360.0
        or not math.isfinite(elevation)
        or not -90.0 <= elevation <= 90.0
    ):
        raise ValueError("azimuth/elevation outside the finite physical range")
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    horizontal = math.cos(elevation_rad)
    vector = (
        horizontal * math.sin(azimuth_rad),
        horizontal * math.cos(azimuth_rad),
        math.sin(elevation_rad),
    )
    norm = math.sqrt(sum(value * value for value in vector))
    return tuple(value / norm for value in vector)


def parse_gps_sdr_sim_los_table(text: str) -> dict[str, tuple[float, float, float]]:
    """Parse the startup PRN azimuth/elevation table preserved by gps-sdr-sim."""
    result: dict[str, tuple[float, float, float]] = {}
    for match in _GPS_SDR_SIM_LOS_ROW.finditer(str(text)):
        prn = f"G{int(match.group('prn')):02d}"
        if prn in result:
            raise ValueError(f"duplicate simulator LOS row: {prn}")
        range_m = float(match.group("range"))
        ionosphere_m = float(match.group("iono"))
        if not math.isfinite(range_m) or range_m <= 0 or not math.isfinite(ionosphere_m):
            raise ValueError(f"invalid simulator LOS row: {prn}")
        result[prn] = enu_line_of_sight(
            float(match.group("azimuth")), float(match.group("elevation"))
        )
    if not result:
        raise ValueError("gps-sdr-sim log contains no LOS table")
    return result


def los_displacement_proxy(
    event: dict[str, Any],
    time_s: float,
    los_enu: Iterable[float],
    *,
    chip_length_m: float,
) -> float:
    """Peak-mixture variance using carry-off projected onto one satellite LOS."""
    chip = float(chip_length_m)
    target = np.asarray(event["target_offset_enu_m"], dtype=np.float64)
    los = np.asarray(tuple(los_enu), dtype=np.float64)
    if chip <= 0 or not math.isfinite(chip):
        raise ValueError("chip_length_m must be positive and finite")
    if (
        target.shape != (3,)
        or los.shape != (3,)
        or not np.isfinite(target).all()
        or not np.isfinite(los).all()
    ):
        raise ValueError("target displacement and LOS must contain three finite values")
    if not math.isclose(
        float(np.linalg.norm(los)), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("LOS vector must have unit norm")
    progress = smoothstep_progress(
        time_s, float(event["start_seconds"]), float(event["transition_seconds"])
    )
    delay_chips = float(np.dot(target, los)) * progress / chip
    ratio = linear_amplitude_ratio(
        time_s,
        start_s=float(event["start_seconds"]),
        ramp_s=float(event["power_ramp_seconds"]),
        initial_advantage_db=float(event["initial_advantage_db"]),
        final_advantage_db=float(event["final_advantage_db"]),
    )
    return mixture_variance_excess(ratio, delay_chips)


def robust_center_scale(values: Iterable[float], *, scale_floor: float = 1e-6) -> tuple[float, float]:
    """Median and Gaussian-consistent MAD scale with a declared floor."""
    array = np.asarray(tuple(values), dtype=np.float64)
    floor = float(scale_floor)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite one-dimensional sequence")
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("scale_floor must be positive and finite")
    center = float(np.median(array))
    scale = max(floor, MAD_TO_SIGMA * float(np.median(np.abs(array - center))))
    return center, scale


def first_persistent_crossing(
    times_s: Iterable[float],
    scores: Iterable[float],
    *,
    threshold: float,
    onset_s: float,
    persistence: int,
    expected_step_s: float,
) -> float | None:
    """First causal crossing with consecutive, contiguous score availability."""
    times = np.asarray(tuple(times_s), dtype=np.float64)
    values = np.asarray(tuple(scores), dtype=np.float64)
    if times.shape != values.shape or times.ndim != 1:
        raise ValueError("times and scores must be equal one-dimensional arrays")
    if persistence < 1 or expected_step_s <= 0:
        raise ValueError("persistence and expected step must be positive")
    if not math.isfinite(threshold) or not math.isfinite(onset_s):
        raise ValueError("threshold and onset must be finite")
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    for start in range(max(0, len(times) - persistence + 1)):
        stop = start + persistence
        block_t, block_v = times[start:stop], values[start:stop]
        if block_t[0] < onset_s or not np.isfinite(block_v).all():
            continue
        if len(block_t) > 1 and np.any(np.diff(block_t) > expected_step_s * 1.5):
            continue
        if np.all(block_v > threshold):
            return float(block_t[0])
    return None
