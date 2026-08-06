"""ACAF-NF Stage-0-R1: validated GPS L1 C/A complex CAF primitives.

This module intentionally consumes tracker lineage only for centers and rereads raw
interleaved int16 IQ for every CAF.  It never uses historical CAF/tap scores as
ACAF evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np

GPS_CA_CHIP_RATE_HZ = 1.023e6
CA_DELAY_CHIPS = np.arange(-1.0, 1.0001, 0.125, dtype=np.float64)
CA_DOPPLER_HZ = np.arange(-250.0, 250.1, 50.0, dtype=np.float64)

# IS-GPS-200 table 3-Ia. Values are 1-indexed G2 stages.
_G2_TAPS = ((2,6),(3,7),(4,8),(5,9),(1,9),(2,10),(1,8),(2,9),
            (3,10),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),(9,10),
            (1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(1,3),(4,6),
            (5,7),(6,8),(7,9),(8,10),(1,6),(2,7),(3,8),(4,9))

@dataclass(frozen=True)
class TrackerCenter:
    prn: int
    code_phase_chips: float
    carrier_doppler_hz: float
    code_rate_hz: float
    sample_count: int


def ca_code(prn: int) -> np.ndarray:
    """One canonical GPS L1 C/A code period in +/-1 convention."""
    if not 1 <= int(prn) <= 32:
        raise ValueError("GPS L1 C/A PRN must be 1..32")
    tap_a, tap_b = _G2_TAPS[int(prn)-1]
    g1 = np.ones(10, dtype=np.uint8)
    g2 = np.ones(10, dtype=np.uint8)
    chips = np.empty(1023, dtype=np.int8)
    for i in range(1023):
        # G2 stage indices are one-based in the ICD; do not use negative indexing.
        bit = g1[-1] ^ g2[tap_a-1] ^ g2[tap_b-1]
        chips[i] = 1 if bit == 0 else -1
        g1_fb = g1[2] ^ g1[9]
        g2_fb = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1[1:] = g1[:-1]; g1[0] = g1_fb
        g2[1:] = g2[:-1]; g2[0] = g2_fb
    return chips


def sampled_replica(prn: int, fs_hz: float, samples: int, *, code_phase_chips: float,
                    code_rate_hz: float) -> np.ndarray:
    if fs_hz <= 0 or samples <= 0 or not np.isfinite(code_rate_hz) or code_rate_hz <= 0:
        raise ValueError("invalid sampling/code-rate parameters")
    phase = (float(code_phase_chips) + np.arange(samples, dtype=np.float64) * float(code_rate_hz) / float(fs_hz)) % 1023.0
    return ca_code(prn)[np.floor(phase).astype(np.int64)].astype(np.complex64)


def tracker_center_from_row(row: Mapping[str, object], fs_hz: float) -> TrackerCenter:
    required = ("PRN", "PRN_start_sample_count", "carrier_doppler_hz", "code_freq_chips", "aux1")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(f"tracker row lacks mandatory center lineage: {missing}")
    values = {name: float(row[name]) for name in required}
    if not all(np.isfinite(v) for v in values.values()) or values["code_freq_chips"] <= 0:
        raise ValueError("non-finite tracker center lineage")
    prn = int(values["PRN"])
    if not 1 <= prn <= 32:
        raise ValueError("tracker PRN outside GPS L1 C/A range")
    # GNSS-SDR's aux1 is the tracker code phase at PRN_start_sample_count, in chips.
    return TrackerCenter(prn, values["aux1"] % 1023.0, values["carrier_doppler_hz"],
                         values["code_freq_chips"], int(values["PRN_start_sample_count"]))


def raw_iq_epoch(path: str, sample_count: int, fs_hz: int, coherent_ms: int = 1) -> np.ndarray:
    n = int(round(fs_hz * coherent_ms / 1000.0))
    if sample_count < 0:
        raise ValueError("negative raw sample count")
    mm = np.memmap(path, dtype="<i2", mode="r", offset=int(sample_count)*4, shape=(n,2))
    return mm[:,0].astype(np.float32) + 1j*mm[:,1].astype(np.float32)


def caf_complex_grid(iq: np.ndarray, center: TrackerCenter, fs_hz: float,
                     delay_chips: np.ndarray = CA_DELAY_CHIPS,
                     doppler_hz: np.ndarray = CA_DOPPLER_HZ) -> np.ndarray:
    """Compute true complex local-replica CAF over delay/Doppler offsets."""
    x = np.asarray(iq, dtype=np.complex64).reshape(-1)
    n = len(x)
    t = np.arange(n, dtype=np.float64) / float(fs_hz)
    replicas = np.stack([sampled_replica(center.prn, fs_hz, n,
                                          code_phase_chips=center.code_phase_chips+d,
                                          code_rate_hz=center.code_rate_hz)
                         for d in np.asarray(delay_chips, float)], axis=0)
    wiped = x[None,:] * np.exp(-2j*np.pi*(center.carrier_doppler_hz + np.asarray(doppler_hz)[:,None])*t[None,:])
    # C(f,tau) = sum x e^-j2pift conj(local code) / N.
    return (wiped @ np.conj(replicas).T) / float(n)


def normalize_complex_caf(field: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, dict]:
    z = np.asarray(field, dtype=np.complex128)
    flat = z.ravel(); center = len(flat)//2
    anchor = center if abs(flat[center]) > eps else int(np.argmax(np.abs(flat)))
    norm = float(np.linalg.norm(flat))
    phase = float(np.angle(flat[anchor])) if norm > eps else 0.0
    return z*np.exp(-1j*phase)/(norm+eps), {"anchor_index":anchor, "fallback":anchor != center, "l2_norm":norm}


def complex_vector(field: np.ndarray) -> np.ndarray:
    z=np.asarray(field).ravel()
    return np.concatenate((z.real,z.imag))


def complex_coordinate_indices(field_shape: tuple[int,int]) -> np.ndarray:
    return np.arange(int(field_shape[0])*int(field_shape[1]), dtype=np.int64)


def select_complex_coordinates(clean_vectors: np.ndarray, field_shape: tuple[int,int], k: int) -> list[int]:
    """Clean-only deterministic coordinate selection; one index means one complex CAF cell."""
    cells = field_shape[0]*field_shape[1]
    if not 1 <= k <= cells:
        raise ValueError("invalid complex-coordinate budget")
    x=np.asarray(clean_vectors)
    if x.ndim != 2 or x.shape[1] not in (cells, 2*cells):
        raise ValueError("expected clean vectors of complex cells or real/imag expansion")
    if x.shape[1] == 2*cells:
        variance=np.var(x[:,:cells],axis=0)+np.var(x[:,cells:],axis=0)
    else:
        variance=np.var(x,axis=0)
    return sorted(np.argsort(-variance, kind="stable")[:k].astype(int).tolist())


def robust_h0_fit(clean_vectors: np.ndarray, ridge: float = 0.10) -> dict:
    x=np.asarray(clean_vectors,float)
    if x.ndim != 2 or x.shape[0] < 20:
        raise ValueError("H0 requires at least 20 chronological clean observations")
    median=np.median(x,axis=0); d=x-median
    mad=np.median(np.abs(d),axis=0)+1e-8
    clipped=np.clip(d, -5*mad, 5*mad)
    cov=(clipped.T@clipped)/max(1,len(x)-1)
    scale=max(float(np.trace(cov)/cov.shape[0]),1e-10)
    cov=(1-ridge)*cov+ridge*scale*np.eye(cov.shape[0])+1e-9*np.eye(cov.shape[0])
    return {"mean":median,"precision":np.linalg.pinv(cov,rcond=1e-9),"ridge":ridge,"n":int(len(x))}


def h0_score(vector: np.ndarray, h0: dict) -> float:
    d=np.asarray(vector,float)-h0["mean"]
    return float(d@h0["precision"]@d)


def two_source_same_prn_fit(field: np.ndarray, delay_chips: np.ndarray, doppler_hz: np.ndarray) -> dict:
    """Diagnostic two-component CAF dictionary from the same PRN's observed template shifts."""
    y=np.asarray(field,dtype=np.complex128).ravel(); shape=field.shape
    base=np.asarray(field,dtype=np.complex128)
    atoms=[]; coords=[]
    for fi,f in enumerate(doppler_hz):
        for di,d in enumerate(delay_chips):
            atoms.append(np.roll(np.roll(base,fi-shape[0]//2,axis=0),di-shape[1]//2,axis=1).ravel()); coords.append((float(d),float(f)))
    A=np.stack(atoms,axis=1); residual=[]
    for j in range(A.shape[1]):
        q=np.linalg.lstsq(A[:,[j]],y,rcond=None)[0]; residual.append(float(np.vdot(y-A[:,[j]]@q,y-A[:,[j]]@q).real))
    j1=int(np.argmin(residual)); best=None
    for j2 in range(A.shape[1]):
        if j2==j1: continue
        q=np.linalg.lstsq(A[:,[j1,j2]],y,rcond=None)[0]; r=float(np.vdot(y-A[:,[j1,j2]]@q,y-A[:,[j1,j2]]@q).real)
        if best is None or r<best[0]: best=(r,j2,q)
    r1=residual[j1]; r2,j2,q=best; n=len(y)
    bic1=n*np.log(r1/n+1e-15)+2*np.log(n); bic2=n*np.log(r2/n+1e-15)+4*np.log(n)
    return {"single_residual":r1,"two_residual":r2,"bic_improvement":float(bic1-bic2),
            "second_delay_chip":coords[j2][0],"second_doppler_hz":coords[j2][1],
            "second_amplitude_ratio":float(abs(q[1])/(abs(q[0])+1e-12)),
            "grid_boundary":bool(abs(coords[j2][0])>=max(abs(delay_chips)) or abs(coords[j2][1])>=max(abs(doppler_hz)))}
