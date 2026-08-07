"""Reusable, outcome-blind ACAF-NF Stage-1 mathematics.

Production attack data must only reach these scoring routines after an
independent support gate.  The campaign runner deliberately never does so when
the frozen R1.4 tracker/source contract is unavailable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence
import hashlib
import json

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
H1_GRID_SHA256 = hashlib.sha256(json.dumps(
    {"delays": DELAYS, "dopplers": DOPPLERS}, separators=(",", ":"),
    sort_keys=True).encode()).hexdigest()
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


@dataclass(frozen=True)
class H1Template:
    coordinate: tuple[int, int]
    surface: np.ndarray
    source_role: str
    construction_method: str
    digest: str
    lineage_json: str


def _canonical_h1_lineage(lineage, construction_method: str) -> str:
    required = {"recording_sha256", "scenario", "role", "raw_intervals",
                "construction_method", "algorithm", "version", "grid_sha256"}
    if not isinstance(lineage, Mapping) or set(lineage) != required:
        raise ValueError("exact H1 provenance keys are required")
    is_hash = lambda value: (isinstance(value, str) and len(value) == 64
                             and all(c in "0123456789abcdef" for c in value))
    if (not is_hash(lineage["recording_sha256"])
            or lineage["scenario"] != "cleanStatic"
            or lineage["role"] != "normal_train"
            or lineage["construction_method"] != construction_method
            or lineage["algorithm"] != "dense_complex_caf_periodic_l1ca"
            or lineage["version"] != "1"
            or lineage["grid_sha256"] != H1_GRID_SHA256
            or not isinstance(lineage["raw_intervals"], list)
            or not lineage["raw_intervals"]):
        raise ValueError("invalid H1 provenance values")
    for interval in lineage["raw_intervals"]:
        if (not isinstance(interval, Mapping)
                or set(interval) != {"start", "end", "sha256", "recording_sha256"}
                or type(interval["start"]) is not int or type(interval["end"]) is not int
                or interval["start"] < 0 or interval["end"] <= interval["start"]
                or not is_hash(interval["sha256"])
                or interval["recording_sha256"] != lineage["recording_sha256"]):
            raise ValueError("invalid H1 raw interval provenance")
    return json.dumps(lineage, separators=(",", ":"), sort_keys=True,
                      ensure_ascii=True, allow_nan=False)


def _h1_digest(surface, coordinate, source_role, construction_method,
               lineage_json) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(surface, dtype=np.complex128).view(np.uint8))
    h.update(json.dumps(list(coordinate), separators=(",", ":")).encode())
    h.update(source_role.encode())
    h.update(construction_method.encode())
    h.update(lineage_json.encode())
    return h.hexdigest()


def _build_h1_template_from_surface_for_test(coordinate, surface, *, source_role,
                                              construction_method, lineage) -> H1Template:
    coordinate = tuple(coordinate)
    value = np.asarray(surface, dtype=np.complex128)
    if coordinate not in H1_COORDINATES:
        raise ValueError("coordinate is outside the frozen non-center H1 grid")
    if value.shape != (len(DOPPLERS), len(DELAYS)) or not np.isfinite(value).all():
        raise ValueError("H1 template must be a finite frozen-grid surface")
    if source_role != "normal_train":
        raise ValueError("H1 templates must be constructed from normal_train")
    if construction_method != "raw_iq_periodic_recorrelation":
        raise ValueError("untrusted H1 construction method")
    lineage_json = _canonical_h1_lineage(lineage, construction_method)
    frozen = np.array(value, dtype=np.complex128, copy=True, order="C")
    frozen.flags.writeable = False
    digest = _h1_digest(frozen, coordinate, source_role, construction_method,
                        lineage_json)
    return H1Template(coordinate, frozen, source_role, construction_method,
                      digest, lineage_json)


def build_h1_template_from_raw_recorrelation(
        coordinate, raw_iq_intervals, *, lineage, prn: int,
        code_freq_chips: float, aux1_samples: float,
        tracker_doppler_hz: float, fs_hz: float = FS_HZ) -> H1Template:
    """Construct H1 only by recomputing authenticated cleanStatic raw intervals."""
    coordinate = tuple(coordinate)
    lineage_json = _canonical_h1_lineage(
        lineage, "raw_iq_periodic_recorrelation")
    intervals = list(raw_iq_intervals)
    if len(intervals) != len(lineage["raw_intervals"]):
        raise ValueError("raw interval count does not match lineage")
    if (type(prn) is not int or not 1 <= prn <= 32
            or not all(np.isfinite(v) for v in (code_freq_chips, aux1_samples,
                                                tracker_doppler_hz, fs_hz))
            or code_freq_chips <= 0 or fs_hz <= 0):
        raise ValueError("invalid raw recorrelation controls")
    surfaces = []
    di, dj = coordinate
    for raw, provenance in zip(intervals, lineage["raw_intervals"]):
        raw_array = np.ascontiguousarray(raw)
        if (not np.iscomplexobj(raw_array) or raw_array.ndim != 1
                or raw_array.size != SUPPORT_SAMPLES
                or not np.isfinite(raw_array).all()
                or provenance["end"] - provenance["start"] != raw_array.size):
            raise ValueError("raw interval must be exactly one finite support")
        actual = hashlib.sha256(raw_array.view(np.uint8)).hexdigest()
        if actual != provenance["sha256"]:
            raise ValueError("raw interval SHA-256 mismatch")
        x = np.asarray(raw_array, dtype=np.complex128)
        surfaces.append(dense_complex_caf(
            x, prn, code_freq_chips, aux1_samples + DELAYS[dj],
            tracker_doppler_hz + DOPPLERS[di], fs_hz=fs_hz))
    surface = np.mean(np.stack(surfaces), axis=0)
    return _build_h1_template_from_surface_for_test(
        coordinate, surface, source_role="normal_train",
        construction_method="raw_iq_periodic_recorrelation",
        lineage=json.loads(lineage_json))


def dense_complex_caf(iq: np.ndarray, prn: int, code_freq_chips: float,
                      aux1_samples: float, tracker_doppler_hz: float,
                      *, fs_hz: float = FS_HZ) -> np.ndarray:
    """Complex CAF using the audited R1.3 physical replica/wipe primitives."""
    raw = np.asarray(iq)
    x = np.asarray(raw, dtype=np.complex128)
    if (not np.iscomplexobj(raw) or x.ndim != 1 or x.size != SUPPORT_SAMPLES
            or not np.isfinite(x).all()):
        raise ValueError("CAF requires exactly 25,000 finite complex samples")
    replicas = [code_replica(prn, x.size, fs_hz, code_freq_chips, aux1_samples,
                             -1, d, replica_direction=1)[0] for d in DELAYS]
    wipes = [carrier_wipeoff(x.size, fs_hz, tracker_doppler_hz, f, -1)[0]
             for f in DOPPLERS]
    return (np.asarray(wipes) * x[None, :]) @ np.asarray(replicas).T


def normalize_caf(caf: np.ndarray, floor: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Remove center phase and gain, returning the unhidden raw diagnostics."""
    c = np.asarray(caf, dtype=np.complex128)
    if (c.shape[-2:] != (len(DOPPLERS), len(DELAYS)) or not np.isfinite(c).all()
            or not np.isfinite(floor) or floor <= 0):
        raise ValueError("invalid CAF shape or normalization floor")
    center = c[..., CENTER[0], CENTER[1]]
    if np.any(np.abs(center) < floor):
        raise ValueError("normalization floor applied; CAF is ineligible")
    denominator = np.maximum(np.abs(center), float(floor))
    y = c * np.exp(-1j * np.angle(center))[..., None, None] / denominator[..., None, None]
    y[..., CENTER[0], CENTER[1]] = 1.0 + 0.0j
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
        intervals = [(str(r.get("recording_sha256", r.get("recording_id", ""))),
                      int(r["support_start_sample"]), int(r["support_end_sample"])) for r in rows]
        if any(not identity or b-a != SUPPORT_SAMPLES or a < 0
               for identity, a, b in intervals):
            raise ValueError("invalid raw interval")
        bounds[role] = intervals
    names = list(bounds)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if any(identity == other and a < d and c < b
                   for identity, a, b in bounds[left]
                   for other, c, d in bounds[right]):
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
            identities=[(str(r.get("channel")),int(r["prn"]),int(r.get("tracker_row", r.get("row", starts[k]//SUPPORT_SAMPLES)))) for k,r in enumerate(w)]
            if (len(set(identities)) == length
                    and all(int(r["support_end_sample"])-int(r["support_start_sample"]) == SUPPORT_SAMPLES for r in w)
                    and len({r.get("phase") for r in w if "phase" in r}) <= 1
                    and all(float(r.get("cn0_db_hz", 0)) >= 28 for r in w)
                    and all(float(r.get("carrier_lock", 0)) >= .85 for r in w)
                    and all(24_999 <= b-a <= 25_001 for a, b in zip(starts, starts[1:]))):
                out.append(w)
    return out


def learn_diagonal_variance(train: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    x = np.asarray(train, dtype=np.complex128)
    if x.ndim < 2 or len(x) < 2:
        raise ValueError("normal train surfaces required")
    if not np.isfinite(x).all() or not np.isfinite(floor) or floor <= 0:
        raise ValueError("finite training surfaces and positive floor required")
    variance = np.var(x.real, axis=0, ddof=1) + np.var(x.imag, axis=0, ddof=1)
    variance[CENTER] = np.nan
    return np.maximum(variance, floor)


def _weighted_fit(y: np.ndarray, columns: Sequence[np.ndarray], variance: np.ndarray):
    target = np.asarray(y).reshape(-1)
    design = np.column_stack([np.asarray(c).reshape(-1) for c in columns])
    var = np.broadcast_to(variance, np.asarray(y).shape).reshape(-1)
    mask = np.ones(np.asarray(y).shape, dtype=bool); mask[CENTER] = False; mask=mask.reshape(-1)
    if (not np.isfinite(target[mask]).all() or not np.isfinite(design[mask]).all()
            or not np.isfinite(var[mask]).all() or np.any(var[mask] <= 0)):
        raise ValueError("diagonal quasi-WLS requires finite values and positive variance")
    target=target[mask];design=design[mask];weights = 1 / np.sqrt(var[mask])
    if np.linalg.matrix_rank(design * weights[:,None]) != design.shape[1]:
        raise ValueError("degenerate WLS design")
    coef, *_ = np.linalg.lstsq(design * weights[:, None], target * weights, rcond=None)
    residual = (target - design @ coef) * weights
    return float(np.vdot(residual, residual).real), coef


def two_source_wls(y: np.ndarray, template0: np.ndarray,
                   shifted_templates: Mapping[tuple[int, int], np.ndarray],
                   variance: np.ndarray, *, _test_allow_incomplete_grid: bool = False) -> dict:
    """Fit H0=alpha*T0 and H1=alpha*T0+beta*Tdelta by complex WLS."""
    rss0, coef0 = _weighted_fit(y, [template0], variance)
    if not _test_allow_incomplete_grid and set(shifted_templates) != set(H1_COORDINATES):
        raise ValueError("H1 requires the exact complete frozen coordinate grid")
    candidates = []
    for coordinate, template in shifted_templates.items():
        if not isinstance(template,H1Template) or tuple(coordinate)!=template.coordinate:
            raise ValueError("validated H1Template required")
        try:
            lineage_json = _canonical_h1_lineage(
                json.loads(template.lineage_json), template.construction_method)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("H1 template provenance failure") from None
        actual = _h1_digest(template.surface, template.coordinate,
                            template.source_role, template.construction_method,
                            lineage_json)
        if (template.surface.flags.writeable or actual != template.digest
                or template.source_role != "normal_train"
                or template.construction_method != "raw_iq_periodic_recorrelation"
                or template.lineage_json != lineage_json):
            raise ValueError("H1 template immutability/digest failure")
        rss, coef = _weighted_fit(y, [template0, template.surface], variance)
        candidates.append((rss, tuple(coordinate), coef))
    if not candidates:
        raise ValueError("H1 requires at least one non-center delta")
    rss1, delta, coef1 = min(candidates, key=lambda x: x[0])
    tolerance=1e-10*max(1.,rss0)
    if rss1 > rss0+tolerance:
        raise ArithmeticError("nested H1 RSS exceeds H0")
    return {"h0_rss": rss0, "h1_rss": rss1, "raw_s2src": rss0-rss1,
            "selected_delta": delta, "h0_alpha": coef0[0],
            "h1_alpha": coef1[0], "h1_beta": coef1[1]}


def calibrate_scores(raw_scores: Sequence[float], penalties: Sequence[float] | None = None) -> dict:
    x = np.asarray(raw_scores, dtype=float)
    p = np.zeros_like(x) if penalties is None else np.asarray(penalties, dtype=float)
    if x.ndim != 1 or len(x) < 3 or p.shape != x.shape or not np.isfinite(x-p).all():
        raise ValueError("finite clean calibration scores required")
    if np.any(p != p[0]):raise ValueError("complexity penalty must be one frozen scalar")
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


def choose_pooling(clean_selection: Sequence[Mapping], *, cleanstatic_sha256: str | None = None) -> str:
    """Normal-only choice: minimum block-to-block median absolute deviation."""
    if (cleanstatic_sha256 is None or not clean_selection
            or any(row.get("role") not in {"clean_train", "selection"}
                   or row.get("scenario") != "cleanStatic"
                   or row.get("recording_sha256") != cleanstatic_sha256
                   for row in clean_selection)):
        raise ValueError("pooling selection accepts clean_train or selection only")
    objectives = {}
    for method in ("median", "top50_mean", "trimmed_mean"):
        pooled = [pool_prns(row["scores"], method)[0] for row in clean_selection]
        objectives[method] = float(np.median(np.abs(pooled-np.median(pooled))))
    return min(objectives, key=lambda k: (objectives[k], k))


def apply_gain_phase(iq, gain: float, phase_rad: float):
    raw=np.asarray(iq);x=np.asarray(raw,complex)
    if not np.iscomplexobj(raw) or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(gain) or gain<=0 or not np.isfinite(phase_rad):raise ValueError("invalid gain/phase control")
    return x * float(gain) * np.exp(1j*float(phase_rad))


def add_awgn(iq, sigma: float, seed: int = 0):
    rng=np.random.default_rng(seed);raw=np.asarray(iq);x=np.asarray(raw, complex)
    if not np.iscomplexobj(raw) or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(sigma) or sigma<0:raise ValueError("invalid AWGN control")
    return x + sigma/np.sqrt(2)*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))


def noise_floor(iq) -> float:
    raw=np.asarray(iq);x=np.asarray(raw,complex)
    if not np.iscomplexobj(raw) or x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError("finite nonempty complex IQ required")
    value=float(np.mean(np.abs(x)**2))
    if not np.isfinite(value):raise ValueError("nonfinite noise floor")
    return value


def amplitude_control(iq, target_rms: float):
    raw=np.asarray(iq);x=np.asarray(raw, complex); rms=np.sqrt(np.mean(np.abs(x)**2))
    if not np.iscomplexobj(raw) or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not np.isfinite(target_rms) or target_rms<=0 or not np.isfinite(rms) or rms<=0: raise ValueError("invalid RMS control")
    return x * float(target_rms)/rms


def synthesize_same_prn_second_source(prn: int, n: int, delay_chips: float,
                                      residual_doppler_hz: float, phase_rad: float,
                                      amplitude: float, *, fs_hz: float = FS_HZ,
                                      code_rate: float = 1.023e6) -> np.ndarray:
    """Physical analytic fractional code-phase replica; never zero padded."""
    if (type(prn) is not int or not 1<=prn<=32 or type(n) is not int or n<=0
            or not all(np.isfinite(v) for v in (delay_chips,residual_doppler_hz,
                                                phase_rad,amplitude,fs_hz,code_rate))
            or amplitude<0 or fs_hz<=0 or code_rate<=0):
        raise ValueError("invalid physical synthesis control")
    phase = np.arange(n)*code_rate/fs_hz - float(delay_chips)
    code = gps_l1ca_code(prn)[np.floor(phase).astype(np.int64) % 1023]
    carrier = np.exp(1j*(2*np.pi*float(residual_doppler_hz)*np.arange(n)/fs_hz+float(phase_rad)))
    return float(amplitude)*code*carrier


def _trapezoid_integral(y, x) -> float:
    """NumPy-version-independent composite trapezoid integral."""
    yy = np.asarray(y, dtype=float)
    xx = np.asarray(x, dtype=float)
    if yy.ndim != 1 or xx.ndim != 1 or yy.shape != xx.shape or yy.size < 2:
        raise ValueError("aligned one-dimensional integration coordinates required")
    if not np.isfinite(yy).all() or not np.isfinite(xx).all():
        raise ValueError("finite integration coordinates required")
    return float(np.sum(np.diff(xx) * (yy[:-1] + yy[1:]) * 0.5))


def binary_metrics(labels, scores, max_fpr=.01) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    if not np.isfinite(max_fpr) or not 0 < max_fpr <= 1: raise ValueError("max_fpr must be in (0,1]")
    y=np.asarray(labels, int); s=np.asarray(scores, float)
    if y.ndim!=1 or s.shape!=y.shape or not np.isfinite(s).all():raise ValueError("finite aligned labels/scores required")
    fpr,tpr,_=roc_curve(y,s); stop=np.searchsorted(fpr,max_fpr,"right")
    xf=list(fpr[:stop]);yt=list(tpr[:stop])
    if not xf or xf[-1] < max_fpr:
        xf.append(max_fpr);yt.append(float(np.interp(max_fpr,fpr,tpr)))
    # Standardized to [0,1] over the requested false-positive range.
    raw=_trapezoid_integral(yt,xf)
    minimum=.5*max_fpr**2
    maximum=max_fpr
    pauc=.5*(1+(raw-minimum)/(maximum-minimum)) if max_fpr < 1 else raw
    return {"roc_auc":float(roc_auc_score(y,s)), "average_precision":float(average_precision_score(y,s)), "partial_auc":pauc, "max_fpr":max_fpr}


def alarm_metrics(times, scores, threshold, onset) -> dict:
    t=np.asarray(times,float);s=np.asarray(scores,float)
    if (t.ndim!=1 or not len(t) or s.shape!=t.shape or not np.isfinite(t).all()
            or not np.isfinite(s).all() or not np.isfinite(threshold)
            or not np.isfinite(onset) or np.any(np.diff(t)<=0)):raise ValueError("sorted unique finite aligned times/scores required")
    alarm=s>=threshold; pre=t<onset; post=t>=onset
    first=t[post & alarm]
    return {"pre_onset_fpr":float(np.mean(alarm[pre])) if np.any(pre) else None,
            "detection_fraction":float(np.mean(alarm[post])) if np.any(post) else None,
            "alarm_delay_s":float(first[0]-onset) if len(first) else None}


def block_bootstrap_effect(normal, other, replicates=1000, seed=1,
                           *, times=None, block_seconds=10.0, block_origin_s=None,
                           block_ids=None) -> dict:
    a=np.asarray(normal,float); b=np.asarray(other,float); t=np.asarray(times,float)
    if (a.ndim!=1 or not len(a) or b.shape!=a.shape or t.shape!=a.shape
            or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(t).all()
            or np.any(np.diff(t)<0) or not isinstance(replicates, (int, np.integer))
            or replicates<=0 or block_seconds != 10.0
            or block_origin_s is None or not np.isfinite(block_origin_s)):
        raise ValueError("finite paired arrays, chronological times, and positive replicates required")
    derived=np.floor((t-float(block_origin_s))/10.0).astype(np.int64)
    if block_ids is not None:
        supplied=np.asarray(block_ids)
        if supplied.shape!=a.shape or not np.array_equal(supplied,derived):
            raise ValueError("supplied block IDs do not equal exact derived IDs")
    ids=derived
    unique=np.unique(ids); rng=np.random.default_rng(seed); estimates=[]
    for _ in range(replicates):
        chosen=rng.choice(unique,len(unique),replace=True); mask=np.concatenate([np.flatnonzero(ids==x) for x in chosen])
        estimates.append(float(np.mean(b[mask])-np.mean(a[mask])))
    ci=[float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))]
    if not np.isfinite(ci).all():raise ArithmeticError("nonfinite bootstrap CI")
    return {"effect":float(np.mean(b-a)), "ci95":ci, "block_seconds":10.0,
            "block_origin_s":float(block_origin_s), "times_verified":True,
            "seed":seed, "replicates":int(replicates)}
