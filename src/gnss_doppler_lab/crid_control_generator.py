"""Frozen clean-only CRID R3 raw-IQ physical-control generator.

This module contains no CRID feature, score, predictor, threshold, or attack
reader.  It operates only on the two clean source windows frozen in
``control_spec.json`` and the authenticated MOSAIC R0c TRACE/NAV lineage.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .acquisition_surface import gps_l1ca_code
from .trace_native_1ms import read_records

BYTES_PER_COMPLEX = 4
CHUNK_SAMPLES = 250_000
INT16_MIN, INT16_MAX = -32768, 32767


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(chunk), b""):
            digest.update(payload)
    return digest.hexdigest()


def decode(payload: bytes) -> np.ndarray:
    if len(payload) % 4:
        raise ValueError("non-integral complex int16 payload")
    raw = np.frombuffer(payload, dtype="<i2")
    return raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64)


def encode(values: np.ndarray) -> tuple[bytes, int, int]:
    z = np.asarray(values, np.complex128).reshape(-1)
    i, q = np.rint(z.real), np.rint(z.imag)
    ci, cq = (i < INT16_MIN) | (i > INT16_MAX), (q < INT16_MIN) | (q > INT16_MAX)
    out = np.empty(2 * len(z), dtype="<i2")
    out[0::2] = np.clip(i, INT16_MIN, INT16_MAX).astype("<i2")
    out[1::2] = np.clip(q, INT16_MIN, INT16_MAX).astype("<i2")
    return out.tobytes(), int(np.count_nonzero(ci | cq)), int(np.count_nonzero(ci) + np.count_nonzero(cq))


def _hash_unit(seed: str) -> float:
    word = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big")
    return word / 2**64


def frozen_phase(seed: str, case_id: str, prn: int) -> float:
    return -math.pi + 2 * math.pi * _hash_unit(f"{seed}|{case_id}|PRN={prn}")


def amplitude_envelope(t: np.ndarray, duration: float) -> np.ndarray:
    x = np.asarray(t, float); out = np.ones_like(x)
    lead = x < .5; out[lead] = .5 * (1 - np.cos(np.pi * np.maximum(x[lead], 0) / .5))
    tail = x > duration - .5
    out[tail] = .5 * (1 + np.cos(np.pi * np.minimum(np.maximum(x[tail] - (duration - .5), 0), .5) / .5))
    out[(x < 0) | (x > duration)] = 0
    return out


def pull_delay(t: np.ndarray, terminal: float) -> np.ndarray:
    x = np.asarray(t, float); out = np.zeros_like(x)
    ramp = (x >= .5) & (x < 4.5)
    out[ramp] = terminal * .5 * (1 - np.cos(np.pi * (x[ramp] - .5) / 4))
    out[x >= 4.5] = terminal
    return out


def pull_rate(t: np.ndarray, terminal: float) -> np.ndarray:
    x = np.asarray(t, float); out = np.zeros_like(x)
    ramp = (x >= .5) & (x < 4.5)
    out[ramp] = terminal * .5 * (1 - np.cos(np.pi * (x[ramp] - .5) / 4))
    out[x >= 4.5] = terminal
    return out


def pull_integral(t: np.ndarray, terminal: float) -> np.ndarray:
    x = np.asarray(t, float); out = np.zeros_like(x)
    ramp = (x >= .5) & (x < 4.5); u = x[ramp] - .5
    out[ramp] = terminal * (.5 * u - 2 / np.pi * np.sin(np.pi * u / 4))
    hold = x >= 4.5; out[hold] = terminal * (2 + x[hold] - 4.5)
    return out


@dataclass(frozen=True)
class ControlCase:
    domain: str
    case_id: str
    family: str
    kind: str
    targets: tuple[int, ...]
    delay_chips: float = 0.0
    power_db: float = 0.0
    mode: str = "negative"
    ordinal: int = 0


class Replica:
    def __init__(self, prn: int, trace_path: Path, nav_rows: list[dict[str, str]], start: int, fs: int):
        self.prn, self.start, self.fs = int(prn), int(start), int(fs)
        _, records = read_records(trace_path, mmap=True); self.records = records
        self.starts = records["raw_interval_start_sample"].astype(np.int64)
        self.ends = records["raw_interval_end_sample"].astype(np.int64)
        self.code = gps_l1ca_code(prn).astype(np.float64)
        rows = sorted((r for r in nav_rows if int(r["prn"]) == prn), key=lambda r: int(r["corrected_raw_start_sample"]))
        self.nav_starts = np.array([int(r["corrected_raw_start_sample"]) for r in rows], np.int64)
        self.nav_ends = np.array([int(r["corrected_raw_end_sample_exclusive"]) for r in rows], np.int64)
        self.nav_signs = np.array([int(r["bit_value_pm1"]) for r in rows], np.float64)

    def render(self, absolute: int, count: int, delay: float | np.ndarray = 0.0,
               extra_phase: float | np.ndarray = 0.0, nav_override: np.ndarray | None = None) -> np.ndarray:
        pos = absolute + np.arange(count, dtype=np.int64)
        row = np.searchsorted(self.starts, pos, side="right") - 1
        # R0c authenticated the receiver code-NCO endpoint jitter and permits
        # exactly one uncovered raw sample at a 1 ms join. Extrapolate that
        # sample from the preceding row; larger gaps remain fail-closed.
        if row.min(initial=0) < 0 or np.any(pos < self.starts[row]) or np.any(pos > self.ends[row]):
            raise ValueError(f"PRN {self.prn} outside exact TRACE support")
        nav = np.searchsorted(self.nav_starts, pos, side="right") - 1
        if nav.min(initial=0) < 0 or np.any(pos > self.nav_ends[nav]):
            raise ValueError(f"PRN {self.prn} outside exact NAV support")
        local = pos - self.starts[row]
        code_phase = (local * self.records["action_used_code_phase_step_chips_per_sample"][row]
                      - self.records["action_used_residual_code_phase_chips"][row] + delay)
        phase = (self.records["action_used_residual_carrier_phase_rad"][row]
                 + local * self.records["action_used_carrier_phase_step_rad_per_sample"][row] + extra_phase)
        signs = self.nav_signs[nav] if nav_override is None else nav_override
        return self.code[np.floor(code_phase).astype(np.int64) % 1023] * signs * np.exp(1j * phase)

    def nav(self, absolute: int, count: int) -> np.ndarray:
        pos = absolute + np.arange(count, dtype=np.int64)
        nav = np.searchsorted(self.nav_starts, pos, side="right") - 1
        if nav.min(initial=0) < 0 or np.any(pos > self.nav_ends[nav]):
            raise ValueError("NAV bounds")
        return self.nav_signs[nav]


class FrozenContext:
    def __init__(self, root: Path, domain: str):
        self.root = Path(root); self.spec = json.loads((root / "artifacts/crid_stage0_r3_control_generator_foundation/control_spec.json").read_text())
        self.domain = domain; self.ds = self.spec["datasets"][domain]
        self.fs = int(self.ds["sample_rate_hz"]); self.start = int(self.ds["absolute_start_sample"])
        self.end = int(self.ds["absolute_end_sample_exclusive"]); self.count = self.end - self.start
        self.source = Path(self.ds["source_path"])
        if self.source.stat().st_size != int(self.ds["source_size_bytes"]):
            raise ValueError("source size mismatch")
        mapping = root / self.spec["lineage"]["nav_mapping"]
        with gzip.open(mapping, "rt", newline="") as stream: self.nav_rows = list(csv.DictReader(stream))
        continuity = root / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation/tracking_continuity.csv"
        with continuity.open(newline="") as stream: rows = list(csv.DictReader(stream))
        dataset = self.ds["dataset"]
        paths = {int(r["prn"]): Path(r["trace_path"]) for r in rows if r["dataset"] == dataset and r["status"] == "PASS"}
        self.prns = tuple(int(v) for v in self.ds["validated_prns_sorted"])
        if set(paths) != set(self.prns): raise ValueError("TRACE lineage PRN mismatch")
        self.replicas = {p: Replica(p, paths[p], self.nav_rows, self.start, self.fs) for p in self.prns}

    def source_payload(self, absolute: int, count: int, extra: int = 0) -> bytes:
        with self.source.open("rb") as stream:
            stream.seek(absolute * 4); payload = stream.read((count + extra) * 4)
        if len(payload) != (count + extra) * 4: raise EOFError("clean window bounds")
        return payload


def enumerate_cases(spec: dict, domain: str) -> list[ControlCase]:
    ds = spec["datasets"][domain]; prns = tuple(ds["validated_prns_sorted"]); out = []
    ordinal = 0
    for delay in spec["positive_controls"]["cartesian_grid"]["terminal_delay_chips"]:
        for power in spec["positive_controls"]["cartesian_grid"]["relative_power_db"]:
            for mode in spec["positive_controls"]["cartesian_grid"]["target_mode"]:
                if mode == "single": targets = (ds["single_assignment_cycle"][ordinal % 5],)
                else:
                    excluded = ds["four_assignment_exclusion_cycle"][ordinal % 5]
                    targets = tuple(p for p in prns if p != excluded)
                cid = f"{domain}.positive.d{int(round(delay*1000)):03d}.p{power:+03d}.{mode}"
                out.append(ControlCase(domain, cid, "positive", "replica", targets, delay, power, mode, ordinal)); ordinal += 1
    primary, secondary = prns[:2]
    for row in spec["negative_controls"]["controls"]:
        target_def = row["targets"]
        if target_def == "negative_primary": targets = (primary,)
        elif target_def == "negative_secondary": targets = (secondary,)
        elif target_def == "first four sorted validated PRNs": targets = prns[:4]
        elif isinstance(target_def, list): targets = (primary, secondary)
        else: targets = ()
        out.append(ControlCase(domain, f"{domain}.negative.{row['id']}", "negative", row["id"], targets))
    return out


def estimate_joint_amplitudes(ctx: FrozenContext) -> dict[int, complex]:
    gram = np.zeros((5, 5), np.complex128); rhs = np.zeros(5, np.complex128)
    absolute, end = ctx.start, ctx.start + ctx.fs
    while absolute < end:
        n = min(CHUNK_SAMPLES, end - absolute); x = decode(ctx.source_payload(absolute, n))
        design = np.column_stack([ctx.replicas[p].render(absolute, n) for p in ctx.prns])
        gram += design.conj().T @ design; rhs += design.conj().T @ x; absolute += n
    alpha = np.linalg.solve(gram, rhs)
    return {p: complex(alpha[i]) for i, p in enumerate(ctx.prns)}


def empirical_noise_sigma(ctx: FrozenContext) -> float:
    x = decode(ctx.source_payload(ctx.start, 1_000_000)); scale = .6744897501960817 * math.sqrt(2)
    si = np.median(np.abs(np.diff(x.real) - np.median(np.diff(x.real)))) / scale
    sq = np.median(np.abs(np.diff(x.imag) - np.median(np.diff(x.imag)))) / scale
    return float((si + sq) / 2)


def _rng(spec: dict, domain: str, kind: str) -> np.random.Generator:
    seed = spec["negative_controls"]["seed"]
    word = int.from_bytes(hashlib.sha256(f"{seed}|{domain}|{kind}".encode()).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(word))


def _prn_delta(ctx: FrozenContext, case: ControlCase, alpha: dict[int, complex], absolute: int, count: int) -> np.ndarray:
    t = (absolute + np.arange(count) - ctx.start) / ctx.fs; duration = ctx.count / ctx.fs
    out = np.zeros(count, np.complex128); primary = ctx.prns[0]
    if case.family == "positive":
        env, delay = amplitude_envelope(t, duration), pull_delay(t, case.delay_chips)
        seed = ctx.spec["positive_controls"]["phase_seed"]
        for p in case.targets:
            phase = frozen_phase(seed, case.case_id, p)
            out += alpha[p] * 10 ** (case.power_db / 20) * env * np.exp(1j * phase) * ctx.replicas[p].render(absolute, count, delay)
    elif case.kind == "bitwise_nav_sign":
        nav = ctx.replicas[primary].nav(absolute, count); mask = nav < 0
        out += -2 * alpha[primary] * ctx.replicas[primary].render(absolute, count) * mask
    elif case.kind == "single_source_code_ramp":
        base = ctx.replicas[primary].render(absolute, count); shifted = ctx.replicas[primary].render(absolute, count, pull_delay(t, .05))
        out += alpha[primary] * (shifted - base)
    elif case.kind == "single_source_doppler_ramp":
        base = ctx.replicas[primary].render(absolute, count); phase = 2 * np.pi * pull_integral(t, 5.0)
        out += alpha[primary] * (ctx.replicas[primary].render(absolute, count, extra_phase=phase) - base)
    elif case.kind == "prn_drop_add":
        second = ctx.prns[1]
        out += -alpha[primary] * ctx.replicas[primary].render(absolute, count)
        out += alpha[second] * 10 ** (3 / 20) * ctx.replicas[second].render(absolute, count)
    elif case.kind == "single_prn_disturbance":
        phase = np.zeros(count); inside = (t >= 4) & (t <= 8)
        u = np.clip((t[inside] - 4) / .5, 0, 1); v = np.clip((8 - t[inside]) / .5, 0, 1)
        phase[inside] = (np.pi / 4) * np.minimum(.5 * (1 - np.cos(np.pi*u)), .5 * (1 - np.cos(np.pi*v)))
        base = ctx.replicas[primary].render(absolute, count)
        out += alpha[primary] * (np.exp(1j * phase) - 1) * base
    elif case.kind == "independent_multipath":
        delays, phases = [.07,.11,.17,.23], [.2,-.7,1.1,-1.9]
        for p,d,ph in zip(ctx.prns[:4], delays, phases, strict=True):
            out += alpha[p] * 10 ** (-9 / 20) * np.exp(1j*ph) * ctx.replicas[p].render(absolute, count, d)
    elif case.kind == "zero_delay_collapsed_duplicate":
        out += alpha[primary] * 10 ** (-3 / 20) * ctx.replicas[primary].render(absolute, count)
    return out


def generate_case(ctx: FrozenContext, case: ControlCase, output: Path, alpha: dict[int, complex], noise_sigma: float,
                  *, overwrite: bool = False) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite: raise FileExistsError(output)
    mode = "wb" if overwrite else "xb"; digest = hashlib.sha256(); source_digest = hashlib.sha256()
    clipped_samples = clipped_components = changed = 0; rng = _rng(ctx.spec, ctx.domain, case.kind)
    absolute = ctx.start
    with output.open(mode) as sink:
        while absolute < ctx.end:
            count = min(CHUNK_SAMPLES, ctx.end - absolute); payload = ctx.source_payload(absolute, count)
            source_digest.update(payload); x = decode(payload); y = x.copy(); t = (absolute + np.arange(count) - ctx.start) / ctx.fs
            if case.kind == "gain": y *= .8
            elif case.kind == "global_phase": y *= np.exp(1j*np.pi/6)
            elif case.kind.startswith("empirical_awgn_") or case.kind == "cn0_reduction":
                mult = {"empirical_awgn_0p5sigma":.5,"empirical_awgn_1sigma":1.,"empirical_awgn_2sigma":2.,"cn0_reduction":math.sqrt(10**.3-1)}[case.kind]
                y += mult * noise_sigma * (rng.standard_normal(count) + 1j*rng.standard_normal(count))
            elif case.kind == "common_receiver_clock_drift":
                extra = 1 if absolute + count < ctx.end else 0; xx = decode(ctx.source_payload(absolute, count, extra))
                shift = pull_delay(t, .25); lo = np.arange(count); hi = np.minimum(lo+1, len(xx)-1)
                y = xx[lo]*(1-shift) + xx[hi]*shift
            elif case.kind != "byte_identical": y += _prn_delta(ctx, case, alpha, absolute, count)
            if case.kind == "byte_identical": packed, cs, cc = payload, 0, 0
            else: packed, cs, cc = encode(y)
            sink.write(packed); digest.update(packed); clipped_samples += cs; clipped_components += cc
            changed += int(np.count_nonzero(np.frombuffer(payload, dtype=np.uint8) != np.frombuffer(packed, dtype=np.uint8))); absolute += count
    scalars = 2 * ctx.count; total_fraction = clipped_components / scalars
    if total_fraction > ctx.spec["sample_contract"]["clipping_fail_closed"]["maximum_total_clip_fraction"]:
        raise RuntimeError(f"clipping fail closed: {total_fraction}")
    return {"case_id":case.case_id,"domain":ctx.domain,"family":case.family,"kind":case.kind,"targets":list(case.targets),
            "delay_chips":case.delay_chips,"power_db":case.power_db,"absolute_start_sample":ctx.start,"absolute_end_sample_exclusive":ctx.end,
            "complex_samples":ctx.count,"size_bytes":output.stat().st_size,"source_window_sha256":source_digest.hexdigest(),
            "output_sha256":digest.hexdigest(),"changed_bytes":changed,"clipped_samples":clipped_samples,
            "clipped_components":clipped_components,"clipping_fraction":total_fraction,"output_path":str(output),
            "authentic_amplitudes":{str(p):[alpha[p].real,alpha[p].imag] for p in ctx.prns},"noise_sigma":noise_sigma}
