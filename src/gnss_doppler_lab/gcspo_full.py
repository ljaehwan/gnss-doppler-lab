"""Geometry construction and Full shared-state clean-only GCSPO scorer."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .gcspo_clean import EPOCH_S, residual_table, window_endpoints
from .gcspo_core import build_physical_loading, build_state_prior_precision, common_epoch_covariance, geometry_observability
from .gcspo_ephemeris import parse_ephemeris_handle
from .gcmr_geometry import (ephemeris_health_selection, parse_gnss_sdr_gps_ephemeris_xml,
                            satellite_observation, validate_ephemeris_time_alignment)
from .trajectory import llh_to_ecef

GPS_UTC_LEAP_OFFSET_S = 18
TOW0_S = 477900.0


def _valid_sentence(line):
    line = line.strip()
    if not line.startswith("$") or "*" not in line: return None
    body, raw = line[1:].rsplit("*", 1); check = 0
    for character in body: check ^= ord(character)
    try: valid = check == int(raw[:2], 16)
    except ValueError: valid = False
    return body.split(",") if valid else None


def _hms(value):
    return int(value[:2]), int(value[2:4]), float(value[4:])


def _degree(value, hemisphere):
    width = 2 if hemisphere in ("N", "S") else 3
    result = float(value[:width]) + float(value[width:]) / 60
    return -result if hemisphere in ("S", "W") else result


def _gps_week_tow(date, hms):
    hour, minute, second = hms
    instant = datetime(date.year, date.month, date.day, hour, minute, int(second), tzinfo=timezone.utc)
    seconds = (instant - datetime(1980, 1, 6, tzinfo=timezone.utc)).total_seconds() + GPS_UTC_LEAP_OFFSET_S + second - int(second)
    return int(seconds // 604800), seconds % 604800


def geometry_preflight(receiver_root, *, tracked_prns):
    root = Path(receiver_root)
    with h5py.File(root / "raw/observables.mat", "r") as handle:
        if "RX_time" not in handle: raise ValueError("observables MAT is missing RX_time")
        rx = np.asarray(handle["RX_time"]).reshape(-1).astype(float)
    usable = rx[np.isfinite(rx) & (rx > 0)]
    if not len(usable) or abs(float(usable.min()) - TOW0_S) > .05 + 1e-12: raise ValueError("observables RX_time anchor mismatch")
    current_date, weeks, points = None, [], []
    for line in (root / "nmea_pvt.nmea").read_text(errors="replace").splitlines():
        fields = _valid_sentence(line)
        if not fields: continue
        kind = fields[0][-3:]
        try:
            if kind == "RMC" and len(fields) > 9 and fields[2] == "A":
                current_date = datetime.strptime(fields[9], "%d%m%y").date()
                week, tow = _gps_week_tow(current_date, _hms(fields[1])); relative = (tow - TOW0_S + 302400) % 604800 - 302400
                if 0 <= relative < 120: weeks.append(week)
            elif kind == "GGA" and current_date is not None and len(fields) > 11 and int(fields[6]) > 0 and fields[9] and fields[10] == "M":
                _, tow = _gps_week_tow(current_date, _hms(fields[1])); relative = (tow - TOW0_S + 302400) % 604800 - 302400
                if 20 <= relative <= 90: points.append((relative, _degree(fields[2], fields[3]), _degree(fields[4], fields[5]), float(fields[9])))
        except (ValueError, IndexError): continue
    if not weeks or len(set(weeks)) != 1: raise ValueError("NMEA full GPS week preflight failed")
    if not points: raise ValueError("no clean preflight static NMEA points")
    eph = parse_gnss_sdr_gps_ephemeris_xml(root / "gps_ephemeris.xml")
    alignment = validate_ephemeris_time_alignment(eph, full_gps_week=weeks[0], recording_start_tow_s=TOW0_S, max_toe_age_s=7200)
    healthy, health = ephemeris_health_selection(eph, tracked_prns=sorted(set(map(int, tracked_prns))), min_prns=4)
    array = np.asarray(points, float); llh = tuple(float(np.median(array[:, index])) for index in (1, 2, 3))
    return {"ephemerides": healthy, "receiver_ecef": np.asarray(llh_to_ecef(*llh)),
            "report": {"overall_status": "PASS", "recording_start_rx_time_s": float(usable.min()), "configured_tow0_s": TOW0_S,
                       "full_gps_week": weeks[0], "receiver_llh": llh, "nmea_sample_count": len(points),
                       "ephemeris_alignment": alignment, "ephemeris_health": health}}


def protected_geometry_preflight(*, gate, observables_path, nmea_path, ephemeris_path, scenario, tracked_prns):
    arrays = gate.read_h5(observables_path, datasets=["RX_time"], scenario=scenario, phase="all_frozen_phases",
                          purpose="receiver time anchor")
    rx = np.asarray(arrays["RX_time"]).reshape(-1).astype(float)
    usable = rx[np.isfinite(rx) & (rx > 0)]
    if not len(usable) or abs(float(usable.min()) - TOW0_S) > .05 + 1e-12: raise ValueError("observables RX_time anchor mismatch")
    nmea = gate.read_text(nmea_path, scenario=scenario, phase="all_frozen_phases", purpose="receiver time and position")
    current_date, weeks, points = None, [], []
    for line in nmea.splitlines():
        fields = _valid_sentence(line)
        if not fields: continue
        kind = fields[0][-3:]
        try:
            if kind == "RMC" and len(fields) > 9 and fields[2] == "A":
                current_date = datetime.strptime(fields[9], "%d%m%y").date()
                week, tow = _gps_week_tow(current_date, _hms(fields[1])); relative = (tow - TOW0_S + 302400) % 604800 - 302400
                if 0 <= relative < 120: weeks.append(week)
            elif kind == "GGA" and current_date is not None and len(fields) > 11 and int(fields[6]) > 0 and fields[9] and fields[10] == "M":
                _, tow = _gps_week_tow(current_date, _hms(fields[1])); relative = (tow - TOW0_S + 302400) % 604800 - 302400
                if 20 <= relative <= 90: points.append((relative, _degree(fields[2], fields[3]), _degree(fields[4], fields[5]), float(fields[9])))
        except (ValueError, IndexError): continue
    if not weeks or len(set(weeks)) != 1: raise ValueError("NMEA full GPS week preflight failed")
    if not points: raise ValueError("no receiver static NMEA points")
    eph = gate.consume(ephemeris_path, scenario=scenario, phase="all_frozen_phases", purpose="broadcast ephemeris",
                       consumer=parse_ephemeris_handle, operation="READ_XML")
    alignment = validate_ephemeris_time_alignment(eph, full_gps_week=weeks[0], recording_start_tow_s=TOW0_S, max_toe_age_s=7200)
    healthy, health = ephemeris_health_selection(eph, tracked_prns=sorted(set(map(int, tracked_prns))), min_prns=4)
    array = np.asarray(points, float); llh = tuple(float(np.median(array[:, index])) for index in (1, 2, 3))
    return {"ephemerides": healthy, "receiver_ecef": np.asarray(llh_to_ecef(*llh)),
            "report": {"overall_status": "PASS", "recording_start_rx_time_s": float(usable.min()),
                       "configured_tow0_s": TOW0_S, "full_gps_week": weeks[0], "receiver_llh": llh,
                       "nmea_sample_count": len(points), "ephemeris_alignment": alignment, "ephemeris_health": health}}


class GeometryCache:
    def __init__(self, ephemerides, receiver_ecef, validated_rows):
        self.ephemerides, self.receiver, self.validated_rows, self.cache = ephemerides, receiver_ecef, set(validated_rows), {}

    def loading(self, epoch, prn):
        key = int(epoch), int(prn)
        if key not in self.cache:
            if key[1] not in self.ephemerides: return None
            observation = satellite_observation(self.receiver, self.ephemerides[key[1]], TOW0_S + (key[0] + 1) * EPOCH_S)
            los = np.asarray(observation.los_ecef, float)
            self.cache[key] = (los, build_physical_loading(los, validated_rows=self.validated_rows))
        return self.cache[key]


def _geometry_epoch_prn_support(epoch_ids, prns_by_epoch, geometry):
    return tuple((int(epoch), tuple(prn for prn in sorted(prns_by_epoch[int(epoch)])
                                    if geometry.loading(epoch, prn) is not None))
                 for epoch in epoch_ids)


def _window_normal_terms(epoch_ids, prns_by_epoch, z_lookup, geometry, model, whitener, gamma):
    epoch_count, state_width = len(epoch_ids), 8
    total_state = epoch_count * state_width
    normal = np.zeros((total_state, total_state)); vector = np.zeros(total_state); yty = 0.; nobs = 0
    epoch_position = {int(epoch): index for index, epoch in enumerate(epoch_ids)}
    for epoch in epoch_ids:
        index = epoch_position[int(epoch)]; prns = sorted(prns_by_epoch[int(epoch)])
        los_rows, observations, designs = [], [], []
        for prn in prns:
            physical = geometry.loading(epoch, prn)
            if physical is None: continue
            los, current_b = physical; los_rows.append(los); observations.append(z_lookup[(int(epoch), int(prn))])
            design = np.zeros((10, total_state)); design[:, index * 8:(index + 1) * 8] = current_b
            for lag, coefficient in enumerate(model.coefficients, start=1):
                prior_epoch = int(epoch) - lag
                if prior_epoch in epoch_position and (prior_epoch, int(prn)) in z_lookup:
                    prior = geometry.loading(prior_epoch, prn)
                    if prior is not None:
                        prior_index = epoch_position[prior_epoch]
                        design[:, prior_index * 8:(prior_index + 1) * 8] -= coefficient @ prior[1]
            designs.append(whitener.inverse_sqrt @ design)
        if len(observations) < 4: return None
        observable = geometry_observability(np.vstack(los_rows))
        if not observable["available"]: return None
        y = np.concatenate(observations); g = np.vstack(designs)
        covariance = common_epoch_covariance(gamma, prn_count=len(observations))
        factor = np.linalg.cholesky(covariance)
        white_y, white_g = np.linalg.solve(factor, y), np.linalg.solve(factor, g)
        normal += white_g.T @ white_g; vector += white_g.T @ white_y; yty += float(white_y @ white_y); nobs += len(white_y)
    return normal, vector, yty, nobs


def _score_terms(terms, prior):
    h, vector, yty, nobs = terms
    matrix = h + prior
    try:
        factor = cho_factor(matrix, lower=True, check_finite=False)
        state = cho_solve(factor, vector, check_finite=False)
        influence_normal = cho_solve(factor, h, check_finite=False)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(matrix, rcond=1e-10); state = inverse @ vector; influence_normal = inverse @ h
    rss = yty - 2 * float(state @ vector) + float(state @ h @ state)
    improvement = yty - rss; edf = float(np.trace(influence_normal)); rank = int(np.linalg.matrix_rank(h, tol=1e-10))
    if edf < -1e-7 or edf > rank + 1e-7 or rank > nobs: raise ValueError("Full influence trace bounds failed")
    edf = min(max(edf, 0), float(rank)); penalty = edf * math.log(nobs)
    return {"score": improvement - penalty, "state": state, "rss": rss, "likelihood_improvement_twice": improvement,
            "effective_dof": edf, "penalty": penalty, "n_obs": nobs, "rank": rank,
            "gcv": nobs * rss / max(nobs - edf, 1e-12) ** 2}


def role_full_terms(data, model, whitener, gamma, geometry, start_s, end_s):
    epochs, prns, _, z = residual_table(data, model, whitener, start_s, end_s)
    z_lookup = {(int(e), int(p)): value for e, p, value in zip(epochs, prns, z)}
    prns_by_epoch = {int(e): [] for e in np.unique(epochs)}
    for e, p in zip(epochs, prns): prns_by_epoch[int(e)].append(int(p))
    rows = []
    for endpoint in window_endpoints(start_s, end_s):
        epoch_ids = np.arange(round((endpoint - 1) / EPOCH_S), round(endpoint / EPOCH_S), dtype=np.int64)
        if not all(int(epoch) in prns_by_epoch and len(prns_by_epoch[int(epoch)]) >= 4 for epoch in epoch_ids): continue
        support = _geometry_epoch_prn_support(epoch_ids, prns_by_epoch, geometry)
        if any(len(epoch_prns) < 4 for _, epoch_prns in support): continue
        actual_by_epoch = {epoch: list(epoch_prns) for epoch, epoch_prns in support}
        terms = _window_normal_terms(epoch_ids, actual_by_epoch, z_lookup, geometry, model, whitener, gamma)
        if terms is not None:
            rows.append({"window_start_s": endpoint - 1, "availability_s": endpoint, "terms": terms,
                         "epoch_ids": tuple(map(int, epoch_ids)), "epoch_prn_support": support,
                         "prns": sorted(set().union(*(set(ps) for _, ps in support)))})
    return rows


def role_full_terms_from_z(epochs, prns, z, model, gamma, geometry, start_s, end_s):
    """Score already-whitened innovations for frozen relation destructions."""
    epochs, prns, z = np.asarray(epochs, np.int64), np.asarray(prns, np.int64), np.asarray(z, float)
    if epochs.ndim != 1 or prns.shape != epochs.shape or z.shape != (len(epochs), 10):
        raise ValueError("pre-whitened relation table shape mismatch")
    keys = [(int(epoch), int(prn)) for epoch, prn in zip(epochs, prns)]
    if len(set(keys)) != len(keys): raise ValueError("duplicate relation epoch/PRN identity")
    lookup = {key: value for key, value in zip(keys, z)}
    by_epoch = {int(epoch): [] for epoch in np.unique(epochs)}
    for epoch, prn in keys: by_epoch[epoch].append(prn)
    identity_whitener = type("IdentityWhitener", (), {"inverse_sqrt": np.eye(10)})()
    rows = []
    for endpoint in window_endpoints(start_s, end_s):
        ids = np.arange(round((endpoint - 1) / EPOCH_S), round(endpoint / EPOCH_S), dtype=np.int64)
        if not all(int(epoch) in by_epoch and len(by_epoch[int(epoch)]) >= 4 for epoch in ids): continue
        support = _geometry_epoch_prn_support(ids, by_epoch, geometry)
        if any(len(epoch_prns) < 4 for _, epoch_prns in support): continue
        actual_by_epoch = {epoch: list(epoch_prns) for epoch, epoch_prns in support}
        terms = _window_normal_terms(ids, actual_by_epoch, lookup, geometry, model, identity_whitener, gamma)
        if terms is not None:
            rows.append({"window_start_s": endpoint - 1, "availability_s": endpoint, "terms": terms,
                         "epoch_ids": tuple(map(int, ids)), "epoch_prn_support": support,
                         "prns": sorted(set().union(*(set(ps) for _, ps in support)))})
    return rows


def select_full_lambda(validation_terms, lambda_grid):
    if len(validation_terms) < 100: raise ValueError("Full has fewer than 100 common lambda-validation windows")
    objectives = []
    priors = {float(value): build_state_prior_precision(epoch_count=50, smoothness=float(value)) for value in lambda_grid}
    for value in map(float, lambda_grid):
        scores = [_score_terms(row["terms"], priors[value]) for row in validation_terms]
        objectives.append({"lambda": value, "mean_gcv": float(np.mean([score["gcv"] for score in scores]))})
    best = objectives[0]
    for candidate in objectives[1:]:
        scale = max(abs(candidate["mean_gcv"]), abs(best["mean_gcv"]), 1)
        if candidate["mean_gcv"] < best["mean_gcv"] - 1e-12 * scale or (abs(candidate["mean_gcv"] - best["mean_gcv"]) <= 1e-12 * scale and candidate["lambda"] > best["lambda"]): best = candidate
    return best["lambda"], objectives


def score_full_terms(rows, smoothness):
    prior = build_state_prior_precision(epoch_count=50, smoothness=float(smoothness))
    return [{**{key: row[key] for key in ("window_start_s", "availability_s", "prns", "epoch_ids", "epoch_prn_support")},
             **_score_terms(row["terms"], prior)} for row in rows]
