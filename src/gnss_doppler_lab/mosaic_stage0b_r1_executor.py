"""Frozen full-prefix injector, receiver replay, and TRACE scorer for R1."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import numpy as np

from .acquisition_surface import gps_l1ca_code
from .mosaic_iq_injector_int16 import decode_interleaved_int16, inject_payload
from .mosaic_stage0b_r1_execution_metrics import bic, raised_cosine_envelope, raised_cosine_integral
from .trace_native_1ms import TAPS, complex_taps, read_records, sha256_file

BYTES_PER_SAMPLE = 4
CHUNK_SAMPLES = 250_000
COPY_BYTES = 8 * 1024 * 1024
DELAY_GRID = np.round(np.arange(-.35, .350001, .025), 12)
DOPPLER_GRID = np.arange(-75., 75.0001, 5.)


def load_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def trace_for_prn(directory: Path, prn: int) -> Path:
    matches = []
    for path in sorted(directory.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        if len(records) and int(records["prn"][-1]) == prn:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"TRACE lookup failed for PRN {prn}: {matches}")
    return matches[0]


class FrozenReplica:
    """Absolute-sample replica using authenticated per-epoch receiver NCO state."""
    def __init__(self, prn: int, trace_path: Path, mapping_rows: list[dict[str, str]], interval_start: int, fs: float):
        self.prn = prn; self.fs = float(fs); self.interval_start = int(interval_start)
        _, self.records = read_records(trace_path)
        self.starts = self.records["raw_interval_start_sample"].astype(np.int64)
        self.ends = self.records["raw_interval_end_sample"].astype(np.int64)
        self.code = gps_l1ca_code(prn).astype(np.float64)
        rows = sorted((r for r in mapping_rows if int(r["prn"]) == prn), key=lambda r: int(r["corrected_raw_start_sample"]))
        self.nav_starts = np.asarray([int(r["corrected_raw_start_sample"]) for r in rows], dtype=np.int64)
        self.nav_signs = np.asarray([int(r["bit_value_pm1"]) for r in rows], dtype=np.float64)
        self.nav_end = int(rows[-1]["corrected_raw_end_sample_exclusive"])

    def render(self, absolute_start: int, count: int, *, delay_chips: float, delta_f_hz: float,
               phase0_rad: float, duration_seconds: float, scheduled: bool) -> np.ndarray:
        positions = absolute_start + np.arange(count, dtype=np.int64)
        row = np.searchsorted(self.starts, positions, side="right") - 1
        if count and (row.min() < 0 or np.any(positions > self.ends[np.minimum(row, len(self.ends)-1)] + 1)):
            raise ValueError(f"PRN {self.prn}: sample outside authenticated TRACE NCO support")
        nav = np.searchsorted(self.nav_starts, positions, side="right") - 1
        if count and (nav.min() < 0 or positions[-1] >= self.nav_end):
            raise ValueError(f"PRN {self.prn}: sample outside R0c NAV mapping")
        local = positions - self.starts[row]
        t = (positions - self.interval_start) / self.fs
        envelope = raised_cosine_envelope(t, duration_seconds) if scheduled else np.ones(count)
        delay = envelope * float(delay_chips)
        code_phase = (local * self.records["action_used_code_phase_step_chips_per_sample"][row]
                      - self.records["action_used_residual_code_phase_chips"][row] + delay)
        chips = self.code[np.floor(code_phase).astype(np.int64) % 1023]
        base_phase = (self.records["action_used_residual_carrier_phase_rad"][row]
                      + local * self.records["action_used_carrier_phase_step_rad_per_sample"][row])
        extra = float(phase0_rad) + 2 * np.pi * float(delta_f_hz) * raised_cosine_integral(t, duration_seconds) if scheduled else 0.0
        return (chips * self.nav_signs[nav] * np.exp(1j * (base_phase + extra))).astype(np.complex128)


def _copy_exact(source, target, byte_count: int, digest) -> None:
    remaining = byte_count
    while remaining:
        payload = source.read(min(remaining, COPY_BYTES))
        if not payload:
            raise EOFError("source raw IQ truncated")
        target.write(payload); digest.update(payload); remaining -= len(payload)


def estimate_authentic_amplitudes(raw_path: Path, replicas: dict[int, FrozenReplica], interval_start: int, fs: int) -> dict[int, complex]:
    numerators = {prn: 0j for prn in replicas}; denominators = {prn: 0.0 for prn in replicas}
    end = interval_start + 2 * fs
    with raw_path.open("rb") as stream:
        stream.seek(interval_start * BYTES_PER_SAMPLE)
        absolute = interval_start
        while absolute < end:
            count = min(CHUNK_SAMPLES, end - absolute)
            clean = decode_interleaved_int16(stream.read(count * BYTES_PER_SAMPLE))
            for prn, replica in replicas.items():
                a = replica.render(absolute, count, delay_chips=0, delta_f_hz=0, phase0_rad=0,
                                   duration_seconds=(end - interval_start) / fs + 10, scheduled=False)
                numerators[prn] += np.vdot(a, clean); denominators[prn] += float(np.vdot(a, a).real)
            absolute += count
    return {prn: numerators[prn] / denominators[prn] for prn in replicas}


def generate_injected_prefix(raw_path: Path, output_path: Path, *, total_samples: int, interval: tuple[int, int], fs: int,
                             targets: list[int], rho_db: float, delay_chips: float, delta_f_hz: float,
                             phase0_rad: float, trace_dir: Path, mapping_rows: list[dict[str, str]]) -> dict[str, object]:
    start_time = time.monotonic(); interval_start, interval_end = interval
    replicas = {prn: FrozenReplica(prn, trace_for_prn(trace_dir, prn), mapping_rows, interval_start, fs) for prn in targets}
    alpha_auth = estimate_authentic_amplitudes(raw_path, replicas, interval_start, fs)
    requested = {prn: abs(alpha_auth[prn]) * 10 ** (rho_db / 20) for prn in targets}
    output_path.parent.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(); clipped = 0; interval_samples = 0
    gram = np.zeros((len(targets), len(targets)), dtype=np.complex128)
    rhs = np.zeros(len(targets), dtype=np.complex128)
    clean_energy = 0.0; output_energy = 0.0
    with raw_path.open("rb") as source, output_path.open("xb") as output:
        _copy_exact(source, output, interval_start * BYTES_PER_SAMPLE, digest)
        absolute = interval_start
        duration = (interval_end - interval_start) / fs
        while absolute < interval_end:
            count = min(CHUNK_SAMPLES, interval_end - absolute)
            payload = source.read(count * BYTES_PER_SAMPLE)
            clean = decode_interleaved_int16(payload)
            columns = []
            for prn in targets:
                replica = replicas[prn].render(absolute, count, delay_chips=delay_chips,
                    delta_f_hz=delta_f_hz, phase0_rad=phase0_rad, duration_seconds=duration, scheduled=True)
                envelope = raised_cosine_envelope((absolute + np.arange(count) - interval_start) / fs, duration)
                columns.append(replica * envelope)
            design = np.column_stack(columns)
            counterfeit = design @ np.asarray([requested[p] for p in targets], dtype=float)
            encoded, q = inject_payload(payload, counterfeit)
            output.write(encoded); digest.update(encoded)
            quantized = decode_interleaved_int16(encoded); difference = quantized - clean
            hold_t = (absolute + np.arange(count) - interval_start) / fs
            hold = (hold_t >= 4) & (hold_t < 10)
            if hold.any():
                dh = design[hold]; gram += dh.conj().T @ dh; rhs += dh.conj().T @ difference[hold]
            clipped += int(q["clipped_sample_count"]); interval_samples += count
            clean_energy += float(np.vdot(clean, clean).real); output_energy += float(np.vdot(quantized, quantized).real)
            absolute += count
        suffix_samples = total_samples - interval_end
        _copy_exact(source, output, suffix_samples * BYTES_PER_SAMPLE, digest)
    realized = np.linalg.solve(gram + 1e-12 * np.eye(len(targets)), rhs)
    return {"path": str(output_path), "sha256": digest.hexdigest(), "size_bytes": output_path.stat().st_size,
        "sample_count": total_samples, "interval_samples": interval_samples, "clipped_sample_count": clipped,
        "clipping_ratio": clipped / interval_samples, "alpha_auth": {str(p): [alpha_auth[p].real, alpha_auth[p].imag] for p in targets},
        "requested_spoof_amplitude": {str(p): requested[p] for p in targets},
        "realized_spoof_amplitude": {str(p): [realized[i].real, realized[i].imag] for i, p in enumerate(targets)},
        "realized_scer_db": {str(p): float(20*np.log10(abs(realized[i])/abs(alpha_auth[p]))) for i,p in enumerate(targets)},
        "clean_interval_rms": float(np.sqrt(clean_energy/interval_samples)),
        "output_interval_rms": float(np.sqrt(output_energy/interval_samples)),
        "runtime_seconds": time.monotonic()-start_time, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def render_receiver_config(base_path: Path, raw_path: Path) -> tuple[str, list[str]]:
    lines = base_path.read_text().splitlines()
    changed = []
    out = []
    for line in lines:
        if line.startswith("SignalSource.filename="):
            out.append(f"SignalSource.filename={raw_path}"); changed.append("SignalSource.filename")
        else:
            out.append(line)
    if changed != ["SignalSource.filename"]:
        raise ValueError("receiver config allowlist violation")
    return "\n".join(out) + "\n", changed


def run_receiver(executable: Path, base_config: Path, raw_path: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    config, changed = render_receiver_config(base_config, raw_path)
    config_path = output_dir / "receiver.conf"; config_path.write_text(config)
    started = time.monotonic()
    with (output_dir / "receiver.log").open("wb") as log:
        result = subprocess.run([str(executable), f"--config_file={config_path}", "--keyboard=false"],
                                cwd=output_dir, stdout=log, stderr=subprocess.STDOUT, check=False)
    traces = sorted(output_dir.glob("trace_native_1ms_ch_*.bin"))
    return {"exit_code": result.returncode, "runtime_seconds": time.monotonic()-started,
        "config_sha256": sha256_file(config_path), "config_changed_keys": changed,
        "receiver_binary_sha256": sha256_file(executable), "trace_count": len(traces),
        "trace_files": [{"path": str(p), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in traces],
        "log_path": str(output_dir / "receiver.log"), "status": "PASS" if result.returncode == 0 and len(traces) >= 4 else "FAIL"}


def compare_identity(reference_dir: Path, observed_dir: Path, prns: list[int]) -> dict[str, object]:
    rows = []; passed = 0
    float_fields = [f"{tap}_{axis}" for tap in TAPS for axis in ("i","q")] + [
        "action_used_code_nco_rate_chips_s", "action_used_carrier_doppler_hz",
        "action_used_residual_code_phase_chips", "action_used_residual_carrier_phase_rad"]
    for prn in prns:
        _, a = read_records(trace_for_prn(reference_dir, prn)); _, b = read_records(trace_for_prn(observed_dir, prn))
        n = min(len(a), len(b)); a=a[:n]; b=b[:n]
        integers = all(np.array_equal(a[field], b[field]) for field in ("prn","loop_sequence","raw_interval_start_sample","raw_interval_end_sample"))
        max_abs = max(float(np.max(np.abs(a[field].astype(float)-b[field].astype(float)))) for field in float_fields)
        floats = all(np.allclose(a[field], b[field], rtol=1e-6, atol=1e-6) for field in float_fields)
        ok = n > 0 and integers and floats
        passed += int(ok); rows.append({"prn":prn,"rows":n,"integer_alignment":integers,"float_alignment":floats,"max_abs_error":max_abs,"status":"PASS" if ok else "FAIL"})
    return {"per_prn":rows,"passed_prns":passed,"minimum_four_prns":passed>=4,"status":"PASS" if passed==len(prns) and len(prns)>=4 else "FAIL"}


def _aligned_taps(reference_dir: Path, case_dir: Path, prn: int, lo: int, hi: int):
    _, a = read_records(trace_for_prn(reference_dir,prn)); header,b = read_records(trace_for_prn(case_dir,prn))
    amap={int(v):i for i,v in enumerate(a["raw_interval_start_sample"])}
    take_b=np.flatnonzero((b["raw_interval_start_sample"]>=lo)&(b["raw_interval_start_sample"]<hi))
    pairs=[(amap.get(int(b["raw_interval_start_sample"][j])),j) for j in take_b]
    pairs=[p for p in pairs if p[0] is not None]
    ia=np.asarray([p[0] for p in pairs]); ib=np.asarray([p[1] for p in pairs])
    return header, complex_taps(a[ia]), complex_taps(b[ib]), b["raw_interval_start_sample"][ib].astype(np.int64)


def score_trace_prn(reference_dir: Path, case_dir: Path, prn: int, interval_start: int, fs: int) -> dict[str, float | int]:
    lo=interval_start+4*fs; hi=interval_start+10*fs
    header,auth,observed,starts=_aligned_taps(reference_dir,case_dir,prn,lo,hi)
    y=observed.reshape(-1); a=auth.reshape(-1,1)
    c0,*_=np.linalg.lstsq(a,y,rcond=None); residual=(y-a@c0).reshape(len(starts),len(TAPS))
    rss0=float(np.vdot(residual,residual).real); nobs=2*y.size; b0=bic(max(rss0,1e-300),nobs,2)
    t=(starts-interval_start)/fs
    carrier=np.exp(1j*2*np.pi*np.outer(t,DOPPLER_GRID))
    best=None
    for delay in DELAY_GRID:
        spatial=np.maximum(1-np.abs(np.asarray(header.tap_offsets_chips)-delay),0.0)
        templates=carrier[:,:,None]*spatial[None,None,:]
        flat=templates.transpose(0,2,1).reshape(y.size,len(DOPPLER_GRID))
        projection=a.conj().T@flat; orth=flat-a@(projection/(a.conj().T@a))
        denom=np.sum(np.abs(orth)**2,axis=0); numer=np.abs(orth.conj().T@residual.reshape(-1))**2
        rss1=np.maximum(rss0-numer/np.maximum(denom,1e-300),1e-300)
        b1=nobs*np.log(rss1/nobs)+4*np.log(nobs); db=b0-b1
        k=int(np.argmax(db)); candidate=(float(db[k]),float(delay),float(DOPPLER_GRID[k]),float(rss1[k]),float(b1[k]))
        if best is None or candidate[0]>best[0]: best=candidate
    assert best is not None
    return {"prn":prn,"epochs":len(starts),"rss_h0":rss0,"rss_h1":best[3],"bic_h0":b0,"bic_h1":best[4],
        "delta_bic":best[0],"recovered_delay_chips":best[1],"recovered_doppler_hz":best[2],
        "tap_rms":float(np.sqrt(np.mean(np.abs(observed)**2)))}
