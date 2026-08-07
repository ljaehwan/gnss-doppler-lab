"""Reusable, outcome-blind ACAF-NF Stage-1 mathematics.

Production attack data must only reach these scoring routines after an
independent support gate.  The campaign runner deliberately never does so when
the frozen R1.4 tracker/source contract is unavailable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import (
    Candidate, carrier_wipeoff, code_replica,
)
from gnss_doppler_lab.acquisition_surface import gps_l1ca_code

FS_HZ = 25_000_000.0
SUPPORT_SAMPLES = 25_000
WINDOW_LENGTH = 20
DELAYS = tuple(np.round(np.arange(-1.0, 1.0001, .125), 3))
DOPPLERS = tuple(float(x) for x in range(-250, 251, 50))
CENTER = (DOPPLERS.index(0.0), DELAYS.index(0.0))
H1_COORDINATES = tuple((i, j) for i in range(len(DOPPLERS))
                       for j in range(len(DELAYS)) if (i, j) != CENTER)
FROZEN_CANDIDATE = Candidate("previous", "previous", -1, -1, 0)


@dataclass(frozen=True)
class FrozenStage1Config:
    signal: str = "canonical_gps_l1_ca"
    fs_hz: float = FS_HZ
    raw_format: str = "signed_int16_interleaved_complex_iq"
    global_raw_offset_samples: int = 0
    nco_row: str = "previous"
    aux_row: str = "previous"
    remnant_sign: int = -1
    carrier_sign: int = -1
    replica_direction: str = "forward"
    prompt_row: str = "current"
    support_samples: int = SUPPORT_SAMPLES
    window_length: int = WINDOW_LENGTH
    delay_start_chip: float = -1.0
    delay_stop_chip: float = 1.0
    delay_step_chip: float = .125
    doppler_start_hz: float = -250.0
    doppler_stop_hz: float = 250.0
    doppler_step_hz: float = 50.0
    h1_center_excluded: bool = True

    def document(self) -> dict:
        return asdict(self)


FROZEN_CONFIG = FrozenStage1Config()


def dense_complex_caf(iq: np.ndarray, prn: int, code_freq_chips: float,
                      aux1_samples: float, tracker_doppler_hz: float,
                      *, fs_hz: float = FS_HZ) -> np.ndarray:
    """Complex CAF using the audited R1.3 physical replica/wipe primitives."""
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 1 or x.size != SUPPORT_SAMPLES or not np.isfinite(x).all():
        raise ValueError("CAF requires exactly 25,000 finite complex samples")
    replicas = [code_replica(prn, x.size, fs_hz, code_freq_chips, aux1_samples,
                             -1, d, replica_direction=1)[0] for d in DELAYS]
    wipes = [carrier_wipeoff(x.size, fs_hz, tracker_doppler_hz, f, -1)[0]
             for f in DOPPLERS]
    return (np.asarray(wipes) * x[None, :]) @ np.asarray(replicas).T


def normalize_caf(caf: np.ndarray, floor: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Remove center phase and gain, returning the unhidden raw diagnostics."""
    c = np.asarray(caf, dtype=np.complex128)
    if c.shape[-2:] != (len(DOPPLERS), len(DELAYS)) or floor <= 0:
        raise ValueError("invalid CAF shape or normalization floor")
    center = c[..., CENTER[0], CENTER[1]]
    denominator = np.maximum(np.abs(center), float(floor))
    y = c * np.exp(-1j * np.angle(center))[..., None, None] / denominator[..., None, None]
    return y, {"center_complex_real": np.real(center), "center_complex_imag": np.imag(center),
               "center_magnitude": np.abs(center), "normalization_floor": float(floor),
               "floor_applied": np.abs(center) < floor}


def chronological_split(rows: Sequence[Mapping], fractions=(.6, .2, .2)) -> dict[str, list[Mapping]]:
    """Split only cleanStatic rows in chronological order, with no overlap."""
    if not rows or any(r.get("scenario") != "cleanStatic" for r in rows):
        raise ValueError("normal fitting accepts cleanStatic only")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("invalid split fractions")
    ordered = sorted(rows, key=lambda r: (int(r["support_start_sample"]), str(r.get("channel", ""))))
    n = len(ordered); a = int(n * fractions[0]); b = int(n * (fractions[0] + fractions[1]))
    result = {"train": ordered[:a], "calibration": ordered[a:b], "holdout": ordered[b:]}
    assert_no_raw_overlap(result)
    return result


def assert_no_raw_overlap(roles: Mapping[str, Sequence[Mapping]]) -> None:
    bounds = {}
    for role, rows in roles.items():
        intervals = [(int(r["support_start_sample"]), int(r["support_end_sample"])) for r in rows]
        if any(a >= b for a, b in intervals):
            raise ValueError("invalid raw interval")
        bounds[role] = intervals
    names = list(bounds)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if any(a < d and c < b for a, b in bounds[left] for c, d in bounds[right]):
                raise ValueError("raw overlap across roles")


def audit_provenance_roles(rows: Sequence[Mapping]) -> dict:
    fit_roles = {"train", "calibration"}
    bad = [r for r in rows if r.get("role") in fit_roles and r.get("scenario") != "cleanStatic"]
    return {"status": "PASS" if not bad else "FAIL", "fit_scenarios": sorted({str(r.get("scenario")) for r in rows if r.get("role") in fit_roles}),
            "attack_rows_in_fit_or_calibration": len(bad)}


def consecutive_windows(rows: Sequence[Mapping], length: int = WINDOW_LENGTH) -> list[list[Mapping]]:
    """Exact causal same-PRN 1 ms windows; 20 ms decimation is unavailable."""
    if length != WINDOW_LENGTH:
        raise ValueError("L=20 is frozen")
    grouped: dict[tuple, list[Mapping]] = {}
    for row in rows:
        grouped.setdefault((row.get("channel"), int(row["prn"])), []).append(row)
    out = []
    for group in grouped.values():
        group = sorted(group, key=lambda r: int(r["support_start_sample"]))
        for end in range(length - 1, len(group)):
            w = group[end-length+1:end+1]
            starts = [int(r["support_start_sample"]) for r in w]
            if (all(int(r.get("support_samples", SUPPORT_SAMPLES)) == SUPPORT_SAMPLES for r in w)
                    and all(float(r.get("cn0_db_hz", 0)) >= 28 for r in w)
                    and all(float(r.get("carrier_lock", 0)) >= .85 for r in w)
                    and all(24_999 <= b-a <= 25_001 for a, b in zip(starts, starts[1:]))):
                out.append(w)
    return out


def learn_diagonal_variance(train: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    x = np.asarray(train, dtype=np.complex128)
    if x.ndim < 2 or len(x) < 2:
        raise ValueError("normal train surfaces required")
    variance = np.var(x.real, axis=0, ddof=1) + np.var(x.imag, axis=0, ddof=1)
    return np.maximum(variance, floor)


def _weighted_fit(y: np.ndarray, columns: Sequence[np.ndarray], variance: np.ndarray):
    target = np.asarray(y).reshape(-1)
    design = np.column_stack([np.asarray(c).reshape(-1) for c in columns])
    weights = 1 / np.sqrt(np.broadcast_to(variance, np.asarray(y).shape).reshape(-1))
    coef, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
    residual = (target - design @ coef) * weights
    return float(np.vdot(residual, residual).real), coef


def two_source_wls(y: np.ndarray, template0: np.ndarray,
                   shifted_templates: Mapping[tuple[int, int], np.ndarray],
                   variance: np.ndarray) -> dict:
    """Fit H0=alpha*T0 and H1=alpha*T0+beta*Tdelta by complex WLS."""
    rss0, coef0 = _weighted_fit(y, [template0], variance)
    candidates = []
    for coordinate, template in shifted_templates.items():
        if tuple(coordinate) == CENTER:
            continue
        rss, coef = _weighted_fit(y, [template0, template], variance)
        candidates.append((rss, tuple(coordinate), coef))
    if not candidates:
        raise ValueError("H1 requires at least one non-center delta")
    rss1, delta, coef1 = min(candidates, key=lambda x: x[0])
    rss1 = min(rss0, rss1)
    return {"h0_rss": rss0, "h1_rss": rss1, "raw_s2src": rss0-rss1,
            "selected_delta": delta, "h0_alpha": coef0[0],
            "h1_alpha": coef1[0], "h1_beta": coef1[1]}


def calibrate_scores(raw_scores: Sequence[float], penalties: Sequence[float] | None = None) -> dict:
    x = np.asarray(raw_scores, dtype=float)
    p = np.zeros_like(x) if penalties is None else np.asarray(penalties, dtype=float)
    if x.ndim != 1 or len(x) < 3 or p.shape != x.shape or not np.isfinite(x-p).all():
        raise ValueError("finite clean calibration scores required")
    adjusted = x-p; med = float(np.median(adjusted)); scale = float(np.quantile(adjusted, .75)-np.quantile(adjusted, .25))
    scale = max(scale, np.finfo(float).eps)
    return {"center": med, "scale": scale, "complexity_penalty": float(np.median(p)),
            "threshold": float(np.quantile((adjusted-med)/scale, .99, method="higher")),
            "source": "cleanStatic_calibration_only"}


def standardized_score(raw_s2src: float, calibration: Mapping) -> float:
    return (float(raw_s2src)-float(calibration["complexity_penalty"])-float(calibration["center"])) / float(calibration["scale"])


BASELINE_SELECTORS = {
    "power_only": "raw_iq_mean_power", "prompt_magnitude": "center_abs",
    "epl_3point_complex": (-.5, 0., .5),
    "fixed_9_delay_tap_complex": tuple(np.arange(-.5, .5001, .125)),
    "dense_one_source_residual": "h0_rss", "dense_two_source_score": "calibration_tail_score",
    "B0": "PROVISIONAL_UNAVAILABLE",
}


def pool_prns(values: Mapping[int, float], method: str) -> tuple[float, dict]:
    x = np.asarray(list(values.values()), dtype=float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError("finite PRN scores required")
    if method == "median": pooled = np.median(x)
    elif method == "top50_mean": pooled = np.mean(np.sort(x)[len(x)//2:])
    elif method == "trimmed_mean":
        k = int(np.floor(.1*len(x))); pooled = np.mean(np.sort(x)[k:len(x)-k] if k else x)
    else: raise ValueError("unknown fixed pooling method")
    return float(pooled), {"prn_count": len(x), "dominant_fraction": float(np.max(np.abs(x))/max(np.sum(np.abs(x)), np.finfo(float).eps))}


def choose_pooling(clean_calibration: Sequence[Mapping]) -> str:
    """Normal-only choice: minimum block-to-block median absolute deviation."""
    objectives = {}
    for method in ("median", "top50_mean", "trimmed_mean"):
        pooled = [pool_prns(row["scores"], method)[0] for row in clean_calibration]
        objectives[method] = float(np.median(np.abs(pooled-np.median(pooled))))
    return min(objectives, key=lambda k: (objectives[k], k))


def apply_gain_phase(iq, gain: float, phase_rad: float):
    return np.asarray(iq, complex) * float(gain) * np.exp(1j*float(phase_rad))


def add_awgn(iq, sigma: float, seed: int = 0):
    rng=np.random.default_rng(seed); x=np.asarray(iq, complex)
    return x + sigma/np.sqrt(2)*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))


def noise_floor(iq) -> float:
    return float(np.mean(np.abs(np.asarray(iq))**2))


def amplitude_control(iq, target_rms: float):
    x=np.asarray(iq, complex); rms=np.sqrt(np.mean(np.abs(x)**2))
    if rms == 0: raise ValueError("zero input RMS")
    return x * float(target_rms)/rms


def synthesize_same_prn_second_source(prn: int, n: int, delay_chips: float,
                                      residual_doppler_hz: float, phase_rad: float,
                                      amplitude: float, *, fs_hz: float = FS_HZ,
                                      code_rate: float = 1.023e6) -> np.ndarray:
    """Physical analytic fractional code-phase replica; never zero padded."""
    phase = np.arange(n)*code_rate/fs_hz - float(delay_chips)
    code = gps_l1ca_code(prn)[np.floor(phase).astype(np.int64) % 1023]
    carrier = np.exp(1j*(2*np.pi*float(residual_doppler_hz)*np.arange(n)/fs_hz+float(phase_rad)))
    return float(amplitude)*code*carrier


def binary_metrics(labels, scores, max_fpr=.01) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    y=np.asarray(labels, int); s=np.asarray(scores, float); fpr,tpr,_=roc_curve(y,s)
    mask=fpr<=max_fpr
    pauc=float(np.trapz(tpr[mask], fpr[mask])/max_fpr) if np.sum(mask)>=2 else 0.
    return {"roc_auc":float(roc_auc_score(y,s)), "average_precision":float(average_precision_score(y,s)), "partial_auc":pauc, "max_fpr":max_fpr}


def alarm_metrics(times, scores, threshold, onset) -> dict:
    t=np.asarray(times,float); alarm=np.asarray(scores,float)>=threshold; pre=t<onset; post=t>=onset
    first=t[post & alarm]
    return {"pre_onset_fpr":float(np.mean(alarm[pre])) if np.any(pre) else None,
            "detection_fraction":float(np.mean(alarm[post])) if np.any(post) else None,
            "alarm_delay_s":float(first[0]-onset) if len(first) else None}


def block_bootstrap_effect(normal, other, block_ids, replicates=1000, seed=1) -> dict:
    a=np.asarray(normal,float); b=np.asarray(other,float); ids=np.asarray(block_ids)
    unique=np.unique(ids); rng=np.random.default_rng(seed); estimates=[]
    for _ in range(replicates):
        chosen=rng.choice(unique,len(unique),replace=True); mask=np.concatenate([np.flatnonzero(ids==x) for x in chosen])
        estimates.append(float(np.mean(b[mask])-np.mean(a[mask])))
    return {"effect":float(np.mean(b-a)), "ci95":[float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))], "block_seconds":10, "seed":seed, "replicates":replicates}
