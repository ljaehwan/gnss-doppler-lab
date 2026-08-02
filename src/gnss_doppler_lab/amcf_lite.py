"""AMCF-Lite: causal prompt-invariant masked correlation-field feasibility tools.

The module deliberately has only NumPy/SciPy dependencies.  It implements a
small coordinate-conditioned masked-set neural-process (a deterministic random
shared token encoder and Student-t fitted shared decoder).  PRN identifiers are
never model features; they are used only for causal grouping and diagnostics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
# 0.125-chip receiver spacing, prompt at zero.
TAP_COORDS = np.arange(-4, 5, dtype=float) * 0.125
SEED_TAPS = (3, 4, 5)
FIXED_ORDERS = {
    3: (3, 4, 5),
    5: (3, 4, 5, 2, 6),
    7: (3, 4, 5, 1, 2, 6, 7),
    9: (3, 4, 5, 0, 1, 2, 6, 7, 8),
}


@dataclass(frozen=True)
class PromptGate:
    min_prompt_magnitude: float
    quantile: float = 0.005
    fit_rows: int = 0
    fit_interval: list[float] | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_prompt_magnitude) or self.min_prompt_magnitude < 0:
            raise ValueError("prompt threshold must be finite and nonnegative")
        if not 0 <= self.quantile <= 1:
            raise ValueError("prompt quantile must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {"min_prompt_magnitude": float(self.min_prompt_magnitude), "quantile": float(self.quantile),
                "fit_rows": int(self.fit_rows), "fit_interval": self.fit_interval,
                "fit_source": "cleanStatic train only", "attack_fit": False}


@dataclass(frozen=True)
class Prediction:
    location: np.ndarray
    scale: np.ndarray


@dataclass
class DecisionRows:
    complex_iq: np.ndarray
    time_s: np.ndarray
    decision_time_s: np.ndarray
    prn: np.ndarray
    channel: np.ndarray
    segment_index: np.ndarray
    sample_count: np.ndarray
    recording_id: np.ndarray
    source_index: np.ndarray

    def __len__(self) -> int:
        return len(self.time_s)

    def take(self, index: np.ndarray | Sequence[int]) -> "DecisionRows":
        idx = np.asarray(index)
        return DecisionRows(**{name: np.asarray(getattr(self, name))[idx] for name in self.__dataclass_fields__})


def _complex(iq: np.ndarray) -> np.ndarray:
    arr = np.asarray(iq)
    if arr.ndim != 3 or arr.shape[1:] != (9, 2):
        raise ValueError("complex_iq must have shape [N,9,2] with I/Q last")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise ValueError("complex_iq must be finite numeric I/Q")
    return arr[..., 0].astype(np.float64) + 1j * arr[..., 1].astype(np.float64)


def fit_prompt_gate(iq: np.ndarray, time_s: np.ndarray, *, train_end_s: float = 240.0,
                    quantile: float = 0.005) -> PromptGate:
    """Fit the sole quality parameter on clean train rows, never later data."""
    z = _complex(iq)
    t = np.asarray(time_s, dtype=float)
    if t.shape != (len(z),) or not np.isfinite(t).all():
        raise ValueError("finite time_s[N] required")
    keep = (t >= 0.0) & (t < float(train_end_s))
    prompt = np.abs(z[keep, 4])
    prompt = prompt[np.isfinite(prompt)]
    if not len(prompt):
        raise ValueError("no clean train prompt rows for quality gate")
    threshold = float(np.quantile(prompt, quantile, method="higher"))
    return PromptGate(threshold, quantile, int(len(prompt)), [0.0, float(train_end_s)])


def normalize_prompt(iq: np.ndarray, gate: PromptGate) -> tuple[np.ndarray, np.ndarray]:
    """Return C*conj(P)/|P|^2 after rejecting low prompts.

    The exact valid-row denominator gives global carrier-phase and common nav-bit
    sign invariance. Rejected rows are NaN rather than large finite artifacts.
    """
    z = _complex(iq)
    p = z[:, 4]
    mag = np.abs(p)
    valid = np.isfinite(mag) & (mag >= gate.min_prompt_magnitude) & (mag > 0)
    out = np.full(z.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    denominator = np.maximum(np.square(mag[valid]), np.finfo(np.float64).tiny)
    out[valid] = z[valid] * np.conjugate(p[valid, None]) / denominator[:, None]
    return out, valid


def tapwise_complex_qa(normalized: np.ndarray) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per-tap magnitude, phase and wrapped second-phase-difference QA."""
    z = np.asarray(normalized, np.complex128)
    if z.ndim != 2 or z.shape[1] != 9 or not np.isfinite(z).all():
        raise ValueError("finite normalized complex field [N,9] required")

    def summary(values: np.ndarray) -> dict[str, float | int]:
        x = np.asarray(values, float)
        x = x[np.isfinite(x)]
        if not len(x):
            return {"count": 0, "q01": float("nan"), "median": float("nan"), "q99": float("nan")}
        q = np.quantile(x, [.01, .5, .99])
        return {"count": int(len(x)), "q01": float(q[0]), "median": float(q[1]), "q99": float(q[2])}

    phase = np.angle(z)
    curvature = np.full(phase.shape, np.nan, dtype=float)
    curvature[:, 1:-1] = np.angle(np.exp(1j * (phase[:, :-2] - 2 * phase[:, 1:-1] + phase[:, 2:])))
    return {
        name: {
            "magnitude": summary(np.abs(z[:, index])),
            "phase_rad": summary(phase[:, index]),
            "phase_curvature_rad": summary(curvature[:, index]),
        }
        for index, name in enumerate(TAP_NAMES)
    }


def represent_values(normalized: np.ndarray, kind: str) -> np.ndarray:
    z = np.asarray(normalized, dtype=np.complex128)
    if z.ndim != 2 or z.shape[1] != 9:
        raise ValueError("normalized field must have shape [N,9]")
    if kind == "complex":
        return np.stack((z.real, z.imag), axis=-1)
    if kind == "magnitude":
        return np.abs(z)[..., None]
    if kind == "phase":
        mag = np.abs(z)
        unit = np.ones(z.shape, dtype=np.complex128)
        nz = mag > np.finfo(float).tiny
        unit[nz] = z[nz] / mag[nz]
        return np.stack((unit.real, unit.imag), axis=-1)
    raise ValueError("representation must be complex, magnitude, or phase")


def causal_decision_rows(complex_iq: np.ndarray, *, time_s: np.ndarray, prn: np.ndarray,
                         channel: np.ndarray, segment_index: np.ndarray, sample_count: np.ndarray,
                         recording_id: str, stride_s: float = 0.5,
                         tolerance_s: float = 1e-9) -> DecisionRows:
    """Map each row to its first causal grid and retain latest row per epoch/PRN.

    The stable tie is time, sample count, segment, channel, then source index.
    Recording identity is explicit and no row is carried into a later epoch.
    """
    z = np.asarray(complex_iq)
    _complex(z)
    n = len(z)
    arrays = [np.asarray(x) for x in (time_s, prn, channel, segment_index, sample_count)]
    if any(x.shape != (n,) for x in arrays):
        raise ValueError("all row metadata must have shape [N]")
    t, p, ch, seg, count = arrays
    if not np.isfinite(t.astype(float)).all() or stride_s <= 0 or tolerance_s < 0:
        raise ValueError("finite timestamps, positive stride, nonnegative tolerance required")
    grid_index = np.ceil((t.astype(float) - tolerance_s) / stride_s).astype(np.int64)
    decision = grid_index.astype(float) * stride_s
    source = np.arange(n, dtype=np.int64)
    # Lexicographic maximum is latest time then sample/segment/channel.  Select
    # group ends vectorially: canonical files have ~500k raw rows, so a Python
    # loop over every 1 ms row would dominate the feasibility runtime.
    order = np.lexsort((ch, seg, count, t, p, decision))
    od, op = decision[order], p[order]
    same_next = (od[:-1] == od[1:]) & (op[:-1] == op[1:])
    end_positions = np.r_[np.flatnonzero(~same_next), len(order)-1]
    idx = order[end_positions].astype(np.int64, copy=True)
    # An exact metadata tie is resolved by the lexicographically maximum stable
    # SHA-256 of I/Q plus row metadata, never by source/input order.
    def digest(j: int) -> str:
        h = hashlib.sha256(np.ascontiguousarray(z[j]).tobytes())
        h.update(json.dumps([float(t[j]), str(p[j]), str(ch[j]), str(seg[j]), str(count[j])],
                            separators=(",", ":")).encode())
        return h.hexdigest()
    starts = np.r_[0, end_positions[:-1] + 1]
    for out_i, (start, end) in enumerate(zip(starts, end_positions)):
        if end <= start: continue
        candidate = order[end]
        rank = (t[candidate], count[candidate], seg[candidate], ch[candidate])
        tied = []
        pos = end
        while pos >= start:
            j = order[pos]
            if (t[j], count[j], seg[j], ch[j]) != rank: break
            tied.append(int(j)); pos -= 1
        if len(tied) > 1: idx[out_i] = max(tied, key=digest)
    idx = idx[np.lexsort((count[idx], p[idx], decision[idx]))]
    if np.any(t[idx].astype(float) > decision[idx] + tolerance_s):
        raise AssertionError("future row survived causal decision mapping")
    rec = np.full(len(idx), str(recording_id), dtype=object)
    return DecisionRows(z[idx].copy(), t[idx].astype(float), decision[idx], p[idx].copy(), ch[idx].copy(),
                        seg[idx].copy(), count[idx].copy(), rec, idx)


def student_t_nll(y: np.ndarray, location: np.ndarray, scale: np.ndarray, df: float = 4.0) -> np.ndarray:
    y, mu, s = np.broadcast_arrays(np.asarray(y, float), np.asarray(location, float), np.asarray(scale, float))
    if df <= 0 or np.any(~np.isfinite(y)) or np.any(~np.isfinite(mu)) or np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("finite values, positive scale and df required")
    r = (y - mu) / s
    return (-gammaln((df + 1) / 2) + gammaln(df / 2) + .5 * math.log(df * math.pi)
            + np.log(s) + (df + 1) / 2 * np.log1p(np.square(r) / df))


class MaskedSetModel:
    """Small shared coordinate-conditioned masked-set Student-t predictor.

    The shared token encoder is deterministic and frozen (an ELM-style neural
    feature map); both location and heteroscedastic scale decoders are fitted by
    exact Student-t(df=4) NLL. This keeps feasibility runs reproducible and CPU
    practical while preserving masked-set/neural-process semantics.
    """
    def __init__(self, representation_dim: int, *, hidden: int = 32, seed: int = 20260802,
                 epochs: int = 25, df: float = 4.0, ridge: float = 1e-4):
        if representation_dim not in (1, 2): raise ValueError("representation_dim must be 1 or 2")
        if not 1 <= hidden <= 64: raise ValueError("hidden must be in [1,64]")
        if not 1 <= epochs <= 50: raise ValueError("epochs must be in [1,50]")
        self.representation_dim, self.hidden, self.seed = representation_dim, hidden, int(seed)
        self.epochs, self.df, self.ridge = int(epochs), float(df), float(ridge)
        rng = np.random.default_rng(self.seed)
        self.encoder_weight = rng.normal(scale=1 / math.sqrt(representation_dim + 2),
                                         size=(representation_dim + 2, hidden))
        self.encoder_bias = rng.uniform(-math.pi, math.pi, size=hidden)
        self.location_weight: np.ndarray | None = None
        self.log_scale_weight: np.ndarray | None = None
        self.fit_audit: dict[str, Any] = {}

    @property
    def feature_dim(self) -> int: return 3 + 2 * self.hidden

    def _context(self, observed: Mapping[int, np.ndarray]) -> np.ndarray:
        tokens = []
        for tap in sorted(observed):
            if int(tap) not in range(9): raise ValueError("tap index outside [0,8]")
            value = np.asarray(observed[tap], float)
            if value.shape != (self.representation_dim,) or not np.isfinite(value).all():
                raise ValueError("observed value shape/finite mismatch")
            token = np.r_[TAP_COORDS[int(tap)], value, 1.0]
            tokens.append(np.tanh(token @ self.encoder_weight + self.encoder_bias))
        return np.mean(tokens, axis=0) if tokens else np.zeros(self.hidden)

    def _feature(self, observed: Mapping[int, np.ndarray], target: int) -> np.ndarray:
        c = float(TAP_COORDS[int(target)]); context = self._context(observed)
        return np.r_[1.0, c, c*c, context, c*context]

    def _examples(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 991)
        features, targets = [], []
        for row in x:
            # Three random clean-train masks per field; target is always hidden.
            for _ in range(3):
                target = int(rng.integers(0, 9))
                available = [i for i in range(9) if i != target]
                count = int(rng.integers(2, 8))
                observed_idx = sorted(rng.choice(available, size=count, replace=False).tolist())
                observed = {i: row[i] for i in observed_idx}
                features.append(self._feature(observed, target)); targets.append(row[target])
        return np.asarray(features), np.asarray(targets)

    def fit(self, clean_train: np.ndarray, validation: np.ndarray | None = None) -> "MaskedSetModel":
        x = np.asarray(clean_train, float)
        if x.ndim != 3 or x.shape[1:] != (9, self.representation_dim) or not np.isfinite(x).all():
            raise ValueError("finite clean_train[N,9,D] required")
        if len(x) < 2: raise ValueError("at least two clean train fields required")
        f, y = self._examples(x)
        d, p = self.representation_dim, self.feature_dim
        initial = np.zeros((2, d, p), dtype=float)
        initial[1, :, 0] = np.log(np.maximum(np.std(y, axis=0), 1e-2))

        def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
            weights = flat.reshape(2, d, p); b, g = weights
            mu = f @ b.T
            log_s = np.clip(f @ g.T, -8.0, 5.0); s = np.exp(log_s)
            residual = y - mu; r2 = np.square(residual / s)
            nll = student_t_nll(y, mu, s, self.df)
            d_mu = -(self.df + 1) * residual / (self.df * np.square(s) + np.square(residual))
            d_log_s = 1.0 - (self.df + 1) * r2 / (self.df + r2)
            # Stop gradients at numerical clipping boundaries.
            raw_log_s = f @ g.T
            d_log_s[(raw_log_s <= -8.0) | (raw_log_s >= 5.0)] = 0
            norm = float(y.size)
            grad_b = d_mu.T @ f / norm + 2 * self.ridge * b
            grad_g = d_log_s.T @ f / norm + 2 * self.ridge * g
            loss = float(np.mean(nll) + self.ridge * (np.square(b).sum() + np.square(g).sum()))
            return loss, np.stack((grad_b, grad_g)).ravel()

        result = minimize(objective, initial.ravel(), method="L-BFGS-B", jac=True,
                          options={"maxiter": self.epochs, "ftol": 1e-10, "gtol": 1e-7, "maxls": 20})
        weights = result.x.reshape(2, d, p)
        self.location_weight, self.log_scale_weight = weights[0], weights[1]
        val_loss = None
        if validation is not None and len(validation):
            vf, vy = self._examples(np.asarray(validation, float))
            mu = vf @ self.location_weight.T
            scale = np.exp(np.clip(vf @ self.log_scale_weight.T, -8, 5))
            val_loss = float(np.mean(student_t_nll(vy, mu, scale, self.df)))
        state = np.concatenate([self.encoder_weight.ravel(), self.encoder_bias,
                                self.location_weight.ravel(), self.log_scale_weight.ravel()])
        self.fit_audit = {"train_rows": int(len(x)), "masked_examples": int(len(f)), "iterations": int(result.nit),
                          "epochs_cap": self.epochs, "optimizer_success": bool(result.success),
                          "optimizer_message": str(result.message), "train_objective": float(result.fun),
                          "validation_nll": val_loss, "df": self.df, "hidden": self.hidden,
                          "seed": self.seed, "prn_feature": False,
                          "state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}
        return self

    def predict(self, observed: Mapping[int, np.ndarray], target: int) -> Prediction:
        if self.location_weight is None or self.log_scale_weight is None: raise RuntimeError("model is not fitted")
        feature = self._feature(observed, int(target))
        location = self.location_weight @ feature
        scale = np.exp(np.clip(self.log_scale_weight @ feature, -8.0, 5.0))
        return Prediction(location, scale)


def select_next_uncertain(model: MaskedSetModel, observed: Mapping[int, np.ndarray],
                          candidates: Sequence[int]) -> int:
    """Select using predictive uncertainty only; no hidden-value argument exists."""
    if not candidates: raise ValueError("candidate taps must be nonempty")
    values = [(float(np.mean(model.predict(observed, int(tap)).scale)), -int(tap), int(tap)) for tap in candidates]
    return max(values)[2]


def random_query_order(k: int, *, seed: int, values: Any = None) -> list[int]:
    """Seeded value-blind random extras; ``values`` is accepted but never touched."""
    if k not in (5, 7): raise ValueError("random query count must be 5 or 7")
    rng = np.random.default_rng(int(seed))
    remaining = np.array([0, 1, 2, 6, 7, 8], dtype=int)
    extras = rng.choice(remaining, size=k-3, replace=False).tolist()
    return [*SEED_TAPS, *map(int, extras)]


def adaptive_query_order(model: MaskedSetModel, values: np.ndarray, k: int) -> list[int]:
    if k not in (5, 7): raise ValueError("adaptive query count must be 5 or 7")
    row = np.asarray(values, float)
    observed = {i: row[i] for i in SEED_TAPS}; order = list(SEED_TAPS)
    while len(order) < k:
        tap = select_next_uncertain(model, observed, [i for i in range(9) if i not in observed])
        order.append(tap); observed[tap] = row[tap]  # reveal only after selection
    return order


def score_query_path(model: MaskedSetModel, values: np.ndarray, query_order: Sequence[int]) -> dict[str, Any]:
    row = np.asarray(values, float)
    if row.shape != (9, model.representation_dim) or not np.isfinite(row).all():
        raise ValueError("finite values[9,D] required")
    order = [int(x) for x in query_order]
    if order[:3] != list(SEED_TAPS) or len(set(order)) != len(order):
        raise ValueError("query path must begin E/P/L and contain unique taps")
    nll: list[float] = []
    # Each seed is scored leave-one-out against the other two seeds.
    for tap in SEED_TAPS:
        observed = {j: row[j] for j in SEED_TAPS if j != tap}
        pred = model.predict(observed, tap)
        nll.append(float(np.mean(student_t_nll(row[tap], pred.location, pred.scale, model.df))))
    observed = {j: row[j] for j in SEED_TAPS}
    for tap in order[3:]:
        pred = model.predict(observed, tap)
        nll.append(float(np.mean(student_t_nll(row[tap], pred.location, pred.scale, model.df))))
        observed[tap] = row[tap]
    return {"query_order": order, "queried_nll": nll, "score": robust_top2(nll)}


def robust_top2(values: Sequence[float]) -> float:
    x = np.sort(np.asarray(values, float))
    if not len(x) or not np.isfinite(x).all(): raise ValueError("finite nonempty values required")
    return float(np.mean(x[-min(2, len(x)):]))


def aggregate_epoch_scores(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["recording_id"]), float(row["decision_time_s"]))
        groups.setdefault(key, []).append(row)
    result = []
    for (recording, decision), group in sorted(groups.items()):
        prns = [str(x["prn"]) for x in group]
        if len(set(prns)) != len(prns): raise ValueError("duplicate PRN in decision epoch")
        score = np.asarray([x["score"] for x in group], float)
        if not np.isfinite(score).all(): raise ValueError("epoch scores must be finite")
        histogram: dict[str, int] = {}
        for item in group:
            for tap in item.get("query_order", []):
                key = str(int(tap)); histogram[key] = histogram.get(key, 0) + 1
        result.append({"recording_id": recording, "decision_time_s": decision,
                       "tracked_prn_count": len(group), "score": float(np.median(score)),
                       "selected_tap_histogram_json": json.dumps(histogram, sort_keys=True, separators=(",", ":"))})
    return result


def assign_clean_role(decision_time_s: float) -> str | None:
    t = float(decision_time_s)
    if 0 <= t < 240: return "train"
    if 250 <= t < 330: return "validation"
    if 340 <= t < 410: return "calibration"
    if t >= 420: return "clean_test"
    return None


def higher_quantile(values: Sequence[float], probability: float) -> float:
    x = np.sort(np.asarray(values, float))
    if not len(x) or not np.isfinite(x).all() or not 0 <= probability <= 1:
        raise ValueError("finite nonempty values and probability in [0,1] required")
    return float(x[int(math.ceil(probability * (len(x) - 1)))])


def calibrate_normal_thresholds(scores: Sequence[float], roles: Sequence[str],
                                scenarios: Sequence[str]) -> dict[str, Any]:
    x = np.asarray(scores, float); r = np.asarray(roles, object); s = np.asarray(scenarios, object)
    if not (x.shape == r.shape == s.shape) or not len(x) or not np.isfinite(x).all():
        raise ValueError("aligned finite calibration arrays required")
    if np.any(r != "calibration"): raise ValueError("threshold fitting accepts calibration role only")
    if np.any(s != "cleanStatic"): raise ValueError("normal-only calibration forbids attack scenarios")
    return {"q99": higher_quantile(x, .99), "q995": higher_quantile(x, .995),
            "fit_count": int(len(x)), "fit_role": "calibration", "fit_scenario": "cleanStatic",
            "attack_fit": False, "comparison": "strict_greater"}


def binary_auc(negative_scores: Sequence[float], positive_scores: Sequence[float]) -> tuple[float, float]:
    """Return ROC-AUC and average precision; 0.5 ROC is chance, 1 is perfect."""
    negative = np.asarray(negative_scores, float)
    positive = np.asarray(positive_scores, float)
    if not len(negative) or not len(positive) or not np.isfinite(negative).all() or not np.isfinite(positive).all():
        raise ValueError("finite nonempty negative and positive scores required")
    comparison = positive[:, None] - negative[None, :]
    roc = float(np.mean(comparison > 0) + .5 * np.mean(comparison == 0))
    scores = np.r_[negative, positive]
    labels = np.r_[np.zeros(len(negative), dtype=int), np.ones(len(positive), dtype=int)]
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_true = np.cumsum(sorted_labels)
    precision = cumulative_true / np.arange(1, len(sorted_labels) + 1)
    average_precision = float(np.sum(precision * sorted_labels) / len(positive))
    return roc, average_precision


def first_sustained_alarm_delay(decision_times: Sequence[float], alarms: Sequence[bool], *,
                                onset_s: float, run_length: int = 3, cadence_s: float = .5) -> float | None:
    times = np.asarray(decision_times, float)
    flags = np.asarray(alarms, bool)
    if times.shape != flags.shape or not np.isfinite(times).all() or run_length < 1 or cadence_s <= 0:
        raise ValueError("aligned finite times/alarms and positive run/cadence required")
    order = np.argsort(times, kind="mergesort"); times = times[order]; flags = flags[order]
    eligible = np.flatnonzero(times >= float(onset_s))
    for start in range(max(0, len(eligible) - run_length + 1)):
        index = eligible[start:start + run_length]
        if flags[index].all() and np.allclose(np.diff(times[index]), cadence_s, atol=1e-9, rtol=0):
            return float(times[index[0]] - float(onset_s))
    return None


def phase_masks(decision_times: Sequence[float], onset_s: float) -> dict[str, np.ndarray]:
    end = np.asarray(decision_times, float); start = end - .5; onset = float(onset_s)
    contained = lambda a, b: (start >= a) & (end <= b)
    return {"stable_pre": contained(30., onset-20.), "transition": contained(onset-20., onset),
            "post": start >= onset, "ramp": contained(onset, onset+20),
            "takeover": contained(onset+20, onset+40), "persistent": start >= onset+40}


def state_json(models: Mapping[str, MaskedSetModel], gate: PromptGate) -> str:
    doc = {"schema": "gnss-doppler-lab.amcf-lite-state.v1", "prompt_gate": gate.to_dict(),
           "models": {name: model.fit_audit for name, model in sorted(models.items())}}
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))
