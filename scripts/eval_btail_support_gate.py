#!/usr/bin/env python3
"""Evaluate the frozen clean-calibrated binomial-tail PRN support gate."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FINAL_SCORE = "btail_max_507080_ewma075"
NODE_QUANTILES = {"q50": 0.50, "q70": 0.70, "q80": 0.80}
DEFAULT_ONSETS = {"ds1": 100.0, "ds2": 100.0, "ds3": 100.0, "ds4": 100.0, "ds7": 110.0, "ds8": 110.0}
DS7_DS8_PHASE_WINDOWS = (
    ("110_130", 110.0, 130.0),
    ("130_150", 130.0, 150.0),
    ("150_end", 150.0, None),
)


def _validate_alpha(alpha: float) -> None:
    if float(alpha) != 0.75:
        raise ValueError("alpha must be 0.75 because the frozen detector is named ewma075")


@dataclass
class GateCalibration:
    node_thresholds: dict[str, float]
    event_q99_threshold: float
    clean_static_events: pd.DataFrame
    clean_dynamic_events: pd.DataFrame


def binomial_tail_surprise(k: int, n: int, exceedance_probability: float) -> float:
    """Return -ln(P[X >= k]) for X~Binomial(n,p), matching the frozen probe."""
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        return -math.log(1e-300)
    p = float(exceedance_probability)
    tail = sum(math.comb(n, i) * p**i * (1.0 - p) ** (n - i) for i in range(k, n + 1))
    return -math.log(max(tail, 1e-300))


def _validated_prn_scores(prn_scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "run_id", "prn", "window_bin_s", "window_start_s",
        "window_mid_s", "prn_node_rmse",
    }
    missing = sorted(required - set(prn_scores.columns))
    if missing:
        raise ValueError(f"PRN score CSV missing columns: {missing}")
    df = prn_scores.copy()
    for column in ("run_id", "prn"):
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"PRN score CSV contains null or empty {column}")
    for column in ("window_bin_s", "window_start_s", "window_mid_s", "prn_node_rmse"):
        try:
            values = pd.to_numeric(df[column], errors="raise").to_numpy(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"PRN score CSV contains non-numeric {column}") from exc
        if not np.isfinite(values).all():
            raise ValueError(f"PRN score CSV contains non-finite {column}")
        df[column] = values
    return df


def build_event_scores(prn_scores: pd.DataFrame, node_thresholds: dict[str, float], alpha: float = 0.75) -> pd.DataFrame:
    """Aggregate PRN scores by the receiver's frozen half-second event bins.

    ``alpha`` is the previous-state retention weight, not pandas' current-sample
    EWM alpha: state[t] = alpha*state[t-1] + (1-alpha)*raw[t], state[-1] = 0.
    """
    _validate_alpha(alpha)
    if set(node_thresholds) != set(NODE_QUANTILES):
        raise ValueError(f"node_thresholds must contain exactly {sorted(NODE_QUANTILES)}")
    if not all(np.isfinite(float(node_thresholds[name])) for name in NODE_QUANTILES):
        raise ValueError("node_thresholds must be finite")
    df = _validated_prn_scores(prn_scores)
    duplicate_key = ["run_id", "window_bin_s", "prn"]
    if df.duplicated(duplicate_key, keep=False).any():
        raise ValueError("duplicate (run_id, window_bin_s, prn) rows in PRN scores")

    rows: list[dict[str, float | int | str]] = []
    for (run_id, window_bin_s), group in df.groupby(["run_id", "window_bin_s"], sort=True):
        scores = group["prn_node_rmse"].to_numpy(float)
        n = int(len(scores))
        row: dict[str, float | int | str] = {
            "run_id": run_id,
            "window_bin_s": float(window_bin_s),
            "window_start_s": float(group["window_start_s"].min()),
            "window_mid_s": float(group["window_mid_s"].min()),
            "tracked_prn_count": n,
        }
        surprises = []
        for name, q in NODE_QUANTILES.items():
            k = int(np.sum(scores > float(node_thresholds[name])))
            surprise = binomial_tail_surprise(k, n, 1.0 - q)
            row[f"k_{name}"] = k
            row[f"btail_{name}"] = surprise
            surprises.append(surprise)
        row["btail_max_507080"] = float(max(surprises))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["run_id", "window_bin_s"]).reset_index(drop=True)

    smoothed = np.empty(len(out), dtype=float)
    for _, index in out.groupby("run_id", sort=False).groups.items():
        previous = 0.0
        for position in index:
            current = float(out.at[position, "btail_max_507080"])
            previous = alpha * previous + (1.0 - alpha) * current
            smoothed[position] = previous
    out[FINAL_SCORE] = smoothed
    return out


def calibrate_clean_gate(clean_static: pd.DataFrame, clean_dynamic: pd.DataFrame, alpha: float = 0.75) -> GateCalibration:
    clean_static = _validated_prn_scores(clean_static)
    clean_dynamic = _validated_prn_scores(clean_dynamic)
    combined_prn = pd.concat([clean_static, clean_dynamic], ignore_index=True)
    thresholds = {name: float(combined_prn["prn_node_rmse"].quantile(q)) for name, q in NODE_QUANTILES.items()}
    static_events = build_event_scores(clean_static, thresholds, alpha=alpha)
    dynamic_events = build_event_scores(clean_dynamic, thresholds, alpha=alpha)
    clean_events = pd.concat([static_events, dynamic_events], ignore_index=True)
    event_q99 = float(clean_events[FINAL_SCORE].quantile(0.99))
    return GateCalibration(thresholds, event_q99, static_events, dynamic_events)


def evaluate_scenario(
    events: pd.DataFrame,
    threshold: float,
    onset_s: float,
    phase_windows: tuple[tuple[str, float, float | None], ...] | None = None,
    onset_buffer_s: float = 10.0,
) -> dict[str, object]:
    flags = events[FINAL_SCORE] > threshold
    event_time = events["window_start_s"]
    pre = event_time < onset_s - onset_buffer_s
    post = event_time >= onset_s + onset_buffer_s
    if not pre.any() or not post.any():
        raise ValueError("evaluation requires both buffered pre-onset and post-onset windows")
    first = event_time[flags & post].min() if (flags & post).any() else np.nan
    result: dict[str, object] = {
        "onset_s": float(onset_s),
        "onset_buffer_s": float(onset_buffer_s),
        "q99_threshold": float(threshold),
        "pre_windows": int(pre.sum()),
        "pre_false_flags": int((flags & pre).sum()),
        "pre_false_positive_rate": float((flags & pre).sum() / int(pre.sum())),
        "post_windows": int(post.sum()),
        "post_detection_flags": int((flags & post).sum()),
        "post_detection_rate": float((flags & post).sum() / int(post.sum())),
        "first_detection_s": None if np.isnan(first) else float(first),
        "first_delay_s": None if np.isnan(first) else float(first - onset_s),
    }
    if phase_windows is not None:
        phases: dict[str, dict[str, object]] = {}
        for name, start_s, end_s in phase_windows:
            in_phase = event_time >= start_s
            if end_s is not None:
                in_phase &= event_time < end_s
            phase_flags = flags & in_phase
            count = int(in_phase.sum())
            first_phase = event_time[phase_flags].min() if phase_flags.any() else np.nan
            phases[name] = {
                "start_s": float(start_s),
                "end_s": None if end_s is None else float(end_s),
                "windows": count,
                "detection_flags": int(phase_flags.sum()),
                "detection_rate": float(phase_flags.sum() / count) if count else 0.0,
                "first_detection_s": None if np.isnan(first_phase) else float(first_phase),
                "first_delay_s": None if np.isnan(first_phase) else float(first_phase - start_s),
            }
        result["phases"] = phases
    return result


def read_prn_scores(score_root: Path, scenario: str) -> pd.DataFrame:
    path = score_root / scenario / f"texbat_{scenario}_prn_local_scores.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenarios", default="ds1,ds2,ds3,ds4")
    parser.add_argument("--alpha", type=float, choices=[0.75], default=0.75,
                        help="Frozen previous-state EWMA retention weight.")
    parser.add_argument("--onset-buffer-s", type=float, default=10.0,
                        help="Exclude this many seconds on both sides of onset for overall metrics.")
    parser.add_argument("--onsets-json", default="", help="Optional JSON object overriding scenario onset seconds.")
    args = parser.parse_args()

    score_root = Path(args.score_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onsets = dict(DEFAULT_ONSETS)
    if args.onsets_json:
        onsets.update({k: float(v) for k, v in json.loads(args.onsets_json).items()})

    clean_static = read_prn_scores(score_root, "cleanStatic")
    clean_dynamic = read_prn_scores(score_root, "cleanDynamic")
    calibration = calibrate_clean_gate(clean_static, clean_dynamic, alpha=args.alpha)
    calibration.clean_static_events.to_csv(out_dir / "cleanStatic_event_scores.csv", index=False)
    calibration.clean_dynamic_events.to_csv(out_dir / "cleanDynamic_event_scores.csv", index=False)

    summary: dict[str, object] = {
        "schema": "gnss-doppler-lab.clean-calibrated-btail-gate.v1",
        "detector": FINAL_SCORE,
        "calibration": "cleanStatic+cleanDynamic only",
        "node_thresholds": calibration.node_thresholds,
        "event_q99_threshold": calibration.event_q99_threshold,
        "surprise_log_base": "e",
        "ewma_previous_state_weight": args.alpha,
        "ewma_current_score_weight": 1.0 - args.alpha,
        "evaluation_onset_buffer_s": args.onset_buffer_s,
        "scenarios": {},
    }
    for scenario in [s.strip() for s in args.scenarios.split(",") if s.strip()]:
        prn = read_prn_scores(score_root, scenario)
        events = build_event_scores(prn, calibration.node_thresholds, alpha=args.alpha)
        events.to_csv(out_dir / f"{scenario}_event_scores.csv", index=False)
        phase_windows = DS7_DS8_PHASE_WINDOWS if scenario in {"ds7", "ds8"} else None
        summary["scenarios"][scenario] = evaluate_scenario(
            events,
            calibration.event_q99_threshold,
            onsets[scenario],
            phase_windows=phase_windows,
            onset_buffer_s=args.onset_buffer_s,
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
