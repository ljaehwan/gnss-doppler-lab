#!/usr/bin/env python3
"""Physically complex-domain synthetic sanity evaluation for frozen TRACE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.trace_action_warp import TAP_COORDS_CHIPS, prompt_normalize, warp_complex_taps


def ca_correlation(delay_chips: float) -> np.ndarray:
    """Ideal GPS L1 C/A triangular autocorrelation over the TRACE aperture."""
    return np.maximum(0.0, 1.0 - np.abs(TAP_COORDS_CHIPS - delay_chips)).astype(np.complex128)


def normalized(values: np.ndarray) -> np.ndarray | None:
    out, valid = prompt_normalize(values)
    if not bool(valid):
        return None
    return out


def residual_score(current: np.ndarray, target: np.ndarray, action: float) -> float | None:
    current_normalized = normalized(current)
    actual = normalized(target)
    if current_normalized is None or actual is None:
        return None
    predicted, valid = warp_complex_taps(current_normalized, action, 0.0)
    common = valid & (np.arange(9) > 0) & (np.arange(9) < 8)
    return float(np.mean(np.abs(actual[common] - predicted[common]) ** 2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/trace_stage0_static")
    parser.add_argument("--seed", type=int, default=23017)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    action = 0.02
    dt = 0.001
    controls = []
    two_source = []
    plot_rows = []
    prompt_masked = 0
    powers = (-3.0, 0.0, 3.0)
    phases = np.linspace(0.0, 2 * np.pi, 16, endpoint=False)
    delays = np.linspace(0.0, 0.5, 11)
    dopplers = np.linspace(0.0, 50.0, 6)
    for prn in range(4):
        for reference in range(8):
            common_phase = rng.uniform(-np.pi, np.pi)
            auth = ca_correlation(0.0) * np.exp(1j * common_phase)
            auth_next = ca_correlation(-action) * np.exp(1j * common_phase)
            base = residual_score(auth, auth_next, action)
            nuisance = (
                residual_score(auth, 2.0 * auth_next, action),
                residual_score(auth, auth_next * np.exp(1j * 1.1), action),
                residual_score(auth, -auth_next, action),
                residual_score(auth, auth_next + (rng.normal(0, 0.003, 9) + 1j * rng.normal(0, 0.003, 9)), action),
            )
            controls.extend((base, *nuisance))
            for power_db in powers:
                amplitude = np.sqrt(10 ** (power_db / 10))
                for phase in phases:
                    for delay in delays:
                        for doppler in dopplers:
                            spoof = amplitude * np.exp(1j * (common_phase + phase)) * ca_correlation(delay)
                            drift = doppler * dt * 0.001  # small resolvable code/carrier mismatch coupling
                            spoof_next = amplitude * np.exp(1j * (common_phase + phase + 2 * np.pi * doppler * dt)) * ca_correlation(delay - action + drift)
                            score = residual_score(auth + spoof, auth_next + spoof_next, action)
                            if score is None:
                                prompt_masked += 1
                                continue
                            two_source.append(score)
                            plot_rows.append((power_db, phase, delay, doppler, score))
    controls_array = np.asarray(controls)
    two_array = np.asarray(two_source)
    resolvable = np.asarray([row[2] >= 0.125 for row in plot_rows])
    threshold = float(np.quantile(controls_array, 0.99))
    paired_control = rng.choice(controls_array, size=len(two_array), replace=True)
    differences = two_array - paired_control
    bootstrap_source = differences[: min(10_000, len(differences))]
    bootstrap_means = np.asarray([
        np.mean(rng.choice(bootstrap_source, size=len(bootstrap_source), replace=True)) for _ in range(999)
    ])
    ci = np.quantile(bootstrap_means, [0.025, 0.975])
    statistic = wilcoxon(differences[: min(50_000, len(differences))], alternative="greater")
    metrics = {
        "schema": "gnss-doppler-lab.trace-synthetic-physics.v1",
        "seed": args.seed,
        "complex_superposition": True,
        "direct_magnitude_addition": False,
        "sweep": {"relative_power_db": list(powers), "phase_count": len(phases), "delay_chips": delays.tolist(), "residual_doppler_hz": dopplers.tolist(), "prns": 4, "reference_epochs": 8},
        "single_source_control_count": int(len(controls_array)),
        "two_source_count": int(len(two_array)),
        "low_prompt_masked_count": prompt_masked,
        "control_median_score": float(np.median(controls_array)),
        "two_source_median_score": float(np.median(two_array)),
        "paired_mean_effect": float(np.mean(differences)),
        "paired_mean_effect_bootstrap_95_ci": [float(ci[0]), float(ci[1])],
        "wilcoxon_p_greater": float(statistic.pvalue),
        "control_q99_threshold": threshold,
        "resolvable_detection_probability": float(np.mean(two_array[resolvable] > threshold)),
        "physics_pass": bool(ci[0] > 0 and statistic.pvalue < 0.01 and np.mean(two_array[resolvable] > threshold) > 0.5),
    }
    (args.out_dir / "synthetic_physics_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    array = np.asarray(plot_rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for power in powers:
        mask = array[:, 0] == power
        medians = [np.median(array[mask & np.isclose(array[:, 2], delay), 4]) for delay in delays]
        axes[0].plot(delays, medians, label=f"{power:+.0f} dB")
    axes[0].axhline(threshold, color="black", linestyle="--", label="control q99")
    axes[0].set(xlabel="delay separation (chip)", ylabel="TRACE residual", title="Phase-marginal delay response")
    axes[0].legend()
    phase_medians = [np.median(array[np.isclose(array[:, 1], phase), 4]) for phase in phases]
    axes[1].plot(phases, phase_medians)
    axes[1].set(xlabel="relative carrier phase (rad)", ylabel="TRACE residual", title="Phase response")
    fig.tight_layout()
    fig.savefig(args.out_dir / "plots/synthetic_delay_power_phase_response.png", dpi=150)
    plt.close(fig)
    return 0 if metrics["physics_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
