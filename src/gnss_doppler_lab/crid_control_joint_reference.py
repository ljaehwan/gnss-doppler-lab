"""Independent five-PRN joint reference correlator for CRID R3a.

The module deliberately does not import the CRID control generator.  It
reconstructs GPS L1 C/A, TRACE/NAV sample binding, templates, and complex-LS
systems independently from the frozen inputs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping, Sequence

import numpy as np


_TAPS = {
    1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),
    9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),
    16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),
    23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),
    30:(2,7),31:(3,8),32:(4,9),
}
EXPECTED_COLUMNS = 5
CONDITION_LIMIT = 1_000_000.0


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for payload in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(payload)
    return digest.hexdigest()


def require_file_binding(path: Path, expected_size: int, expected_sha256: str) -> None:
    actual_path = Path(path)
    if actual_path.stat().st_size != int(expected_size):
        raise ValueError(f"file size binding failure: {actual_path}")
    if sha256_file(actual_path) != str(expected_sha256):
        raise ValueError(f"file SHA-256 binding failure: {actual_path}")


def require_absolute_start(actual: int, expected: int) -> None:
    if int(actual) != int(expected):
        raise ValueError(f"absolute raw start mutation: {actual} != {expected}")


def independent_ca(prn: int) -> np.ndarray:
    """Generate one GPS L1 C/A period with local shift registers."""
    if prn not in _TAPS:
        raise ValueError(f"unsupported GPS L1 C/A PRN: {prn}")
    g1, g2 = [1] * 10, [1] * 10
    tap_a, tap_b = _TAPS[prn]
    out = np.empty(1023, np.float64)
    for index in range(1023):
        out[index] = 1.0 if (g1[9] ^ (g2[tap_a - 1] ^ g2[tap_b - 1])) == 0 else -1.0
        feedback_1 = g1[2] ^ g1[9]
        feedback_2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = [feedback_1] + g1[:-1]
        g2 = [feedback_2] + g2[:-1]
    return out


def decode_iq(payload: bytes) -> np.ndarray:
    if len(payload) % 4:
        raise ValueError("non-integral interleaved int16 complex payload")
    raw = np.frombuffer(payload, dtype="<i2").reshape(-1, 2)
    return raw[:, 0].astype(np.float64) + 1j * raw[:, 1].astype(np.float64)


def read_exact_iq(stream: BinaryIO, sample_offset: int, count: int) -> np.ndarray:
    stream.seek(int(sample_offset) * 4)
    payload = stream.read(int(count) * 4)
    if len(payload) != int(count) * 4:
        raise EOFError(f"IQ read short: {len(payload)} != {int(count) * 4}")
    return decode_iq(payload)


class JointReferenceReplica:
    """Independently reconstructed per-PRN replica on absolute samples."""

    def __init__(self, prn: int, records: np.ndarray, nav_rows: Sequence[Mapping[str, str]]):
        self.prn = int(prn)
        self.records = records
        self.starts = records["raw_interval_start_sample"].astype(np.int64)
        self.ends = records["raw_interval_end_sample"].astype(np.int64)
        if len(self.starts) == 0 or np.any(np.diff(self.starts) <= 0):
            raise ValueError(f"PRN {prn} invalid TRACE starts")
        rows = sorted(
            (row for row in nav_rows if int(row["prn"]) == self.prn),
            key=lambda row: int(row["corrected_raw_start_sample"]),
        )
        if not rows:
            raise ValueError(f"PRN {prn} has no NAV rows")
        self.nav_starts = np.array([int(row["corrected_raw_start_sample"]) for row in rows], np.int64)
        self.nav_ends = np.array([int(row["corrected_raw_end_sample_exclusive"]) for row in rows], np.int64)
        self.nav_signs = np.array([int(row["bit_value_pm1"]) for row in rows], np.float64)
        if not set(np.unique(self.nav_signs)).issubset({-1.0, 1.0}):
            raise ValueError(f"PRN {prn} invalid NAV signs")
        self.code = independent_ca(self.prn)

    def _coordinates(self, absolute: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = int(absolute) + np.arange(int(count), dtype=np.int64)
        trace_index = np.searchsorted(self.starts, positions, side="right") - 1
        if (
            trace_index.min(initial=0) < 0
            or np.any(positions < self.starts[trace_index])
            or np.any(positions > self.ends[trace_index])
        ):
            raise ValueError(f"PRN {self.prn} outside exact TRACE support")
        nav_index = np.searchsorted(self.nav_starts, positions, side="right") - 1
        if nav_index.min(initial=0) < 0 or np.any(positions > self.nav_ends[nav_index]):
            raise ValueError(f"PRN {self.prn} outside exact NAV support")
        local = positions - self.starts[trace_index]
        code_phase = (
            local * self.records["action_used_code_phase_step_chips_per_sample"][trace_index]
            - self.records["action_used_residual_code_phase_chips"][trace_index]
        )
        carrier_phase = (
            self.records["action_used_residual_carrier_phase_rad"][trace_index]
            + local * self.records["action_used_carrier_phase_step_rad_per_sample"][trace_index]
        )
        return code_phase, carrier_phase, self.nav_signs[nav_index]

    def render(self, absolute: int, count: int, delay_chips: float = 0.0) -> np.ndarray:
        code_phase, carrier_phase, nav = self._coordinates(absolute, count)
        chips = self.code[np.floor(code_phase + float(delay_chips)).astype(np.int64) % 1023]
        return chips * nav * np.exp(1j * carrier_phase)

    def render_grid(self, absolute: int, count: int, delay_grid: np.ndarray) -> np.ndarray:
        code_phase, carrier_phase, nav = self._coordinates(absolute, count)
        indices = np.floor(code_phase[None, :] + np.asarray(delay_grid)[:, None]).astype(np.int64) % 1023
        return self.code[indices] * (nav * np.exp(1j * carrier_phase))[None, :]


@dataclass(frozen=True)
class JointSolution:
    coefficients: np.ndarray
    rank: int
    condition_number: float
    residual_norm: float


def solve_joint_system(
    gram: np.ndarray,
    rhs: np.ndarray,
    data_energy: float,
    *,
    expected_columns: int = EXPECTED_COLUMNS,
    condition_limit: float = CONDITION_LIMIT,
) -> JointSolution:
    matrix = np.asarray(gram, np.complex128)
    vector = np.asarray(rhs, np.complex128)
    if matrix.shape != (expected_columns, expected_columns) or vector.shape != (expected_columns,):
        raise ValueError("joint system must contain exactly five template columns")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)) or not np.isfinite(data_energy):
        raise ValueError("non-finite joint system")
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix))
    if rank != expected_columns:
        raise ValueError(f"rank-deficient joint system: {rank} != {expected_columns}")
    if not np.isfinite(condition) or condition > float(condition_limit):
        raise ValueError(f"joint system condition number {condition} exceeds {condition_limit}")
    coefficients = np.linalg.solve(matrix, vector)
    fitted_energy = 2.0 * np.real(np.vdot(coefficients, vector)) - np.real(np.vdot(coefficients, matrix @ coefficients))
    residual_sq = max(0.0, float(data_energy) - float(fitted_energy))
    return JointSolution(coefficients, rank, condition, float(np.sqrt(residual_sq)))


def accumulate_authentic_system(
    source: Path,
    absolute_start: int,
    sample_rate_hz: int,
    replicas: Mapping[int, JointReferenceReplica],
    prn_order: Sequence[int],
    *,
    complete_epochs: int = 1000,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(prn_order) != EXPECTED_COLUMNS or len(set(prn_order)) != EXPECTED_COLUMNS:
        raise ValueError("authentic joint fit requires five distinct PRNs")
    samples_per_epoch = int(sample_rate_hz) // 1000
    gram = np.zeros((EXPECTED_COLUMNS, EXPECTED_COLUMNS), np.complex128)
    rhs = np.zeros(EXPECTED_COLUMNS, np.complex128)
    energy = 0.0
    with Path(source).open("rb") as stream:
        for epoch in range(int(complete_epochs)):
            absolute = int(absolute_start) + epoch * samples_per_epoch
            x = read_exact_iq(stream, absolute, samples_per_epoch)
            design = np.column_stack([replicas[prn].render(absolute, samples_per_epoch) for prn in prn_order])
            gram += design.conj().T @ design
            rhs += design.conj().T @ x
            energy += float(np.vdot(x, x).real)
    return gram, rhs, energy


@dataclass
class TerminalBatch:
    gram_by_delay: np.ndarray
    rhs_by_case_delay_prn: np.ndarray
    residual_energy_by_case: np.ndarray
    single_authentic_numerator: np.ndarray
    single_authentic_denominator: np.ndarray


def accumulate_terminal_batch(
    source: Path,
    outputs: Sequence[Path],
    absolute_start: int,
    sample_rate_hz: int,
    replicas: Mapping[int, JointReferenceReplica],
    prn_order: Sequence[int],
    delay_grid: np.ndarray,
) -> TerminalBatch:
    """Accumulate all positive cases together on the frozen terminal support."""
    if len(prn_order) != EXPECTED_COLUMNS or len(set(prn_order)) != EXPECTED_COLUMNS:
        raise ValueError("terminal joint fit requires five distinct PRNs")
    grid = np.asarray(delay_grid, np.float64)
    if not np.array_equal(grid, np.round(np.arange(-0.4, 0.4001, 0.01), 2)):
        raise ValueError("delay grid differs from frozen [-0.4,0.4]/0.01 grid")
    samples_per_epoch = int(sample_rate_hz) // 1000
    case_count = len(outputs)
    gram = np.zeros((len(grid), EXPECTED_COLUMNS, EXPECTED_COLUMNS), np.complex128)
    rhs = np.zeros((case_count, len(grid), EXPECTED_COLUMNS), np.complex128)
    energy = np.zeros(case_count, np.float64)
    auth_num = np.zeros(EXPECTED_COLUMNS, np.complex128)
    auth_den = np.zeros(EXPECTED_COLUMNS, np.float64)
    source_stream = Path(source).open("rb")
    output_streams = [Path(path).open("rb") for path in outputs]
    try:
        zero_index = int(np.flatnonzero(grid == 0.0)[0])
        for epoch in range(10):
            absolute = int(absolute_start) + int((5.0 + 0.1 * epoch) * int(sample_rate_hz))
            x = read_exact_iq(source_stream, absolute, samples_per_epoch)
            residuals = np.column_stack(
                [read_exact_iq(stream, absolute - int(absolute_start), samples_per_epoch) - x for stream in output_streams]
            )
            energy += np.sum(np.abs(residuals) ** 2, axis=0)
            templates = [replicas[prn].render_grid(absolute, samples_per_epoch, grid) for prn in prn_order]
            for column, template in enumerate(templates):
                rhs[:, :, column] += residuals.T @ template.conj().T
                zero = template[zero_index]
                auth_num[column] += np.vdot(zero, x)
                auth_den[column] += float(np.vdot(zero, zero).real)
            for left, template_left in enumerate(templates):
                for right in range(left, EXPECTED_COLUMNS):
                    values = np.einsum("gn,gn->g", template_left.conj(), templates[right], optimize=True)
                    gram[:, left, right] += values
                    if right != left:
                        gram[:, right, left] += values.conj()
    finally:
        source_stream.close()
        for stream in output_streams:
            stream.close()
    return TerminalBatch(gram, rhs, energy, auth_num, auth_den)


def coefficient_json(values: Iterable[complex], prns: Sequence[int]) -> str:
    import json
    return json.dumps({str(prn): [float(value.real), float(value.imag)] for prn, value in zip(prns, values, strict=True)}, sort_keys=True)
