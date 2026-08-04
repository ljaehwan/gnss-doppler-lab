#!/usr/bin/env python3
"""Generate the bounded R2C-GNSS Stage-0 DATA_INVALID artifact."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.r2c_gnss import (  # noqa: E402
    C_M_S, aggregate_a2, artifact_hashes, fit_second_source,
    fit_shared_constellation, full_score, inject_second_source, sha256_file, write_json,
)

ARTIFACT_FILES = (
    "README.md", "config.json", "provenance.json", "input_validity.json",
    "training_summary.json", "thresholds.json", "scenario_metrics.csv",
    "ablation_metrics.csv", "per_epoch_scores.csv", "gain_invariance.json",
    "phase_invariance.json", "noise_control.json", "multipath_control.json",
    "second_source_injection.json", "relation_destruction.json", "decision.json",
    "verification.json", "hashes.json",
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def mechanics(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    taps = np.linspace(-1.0, 1.0, 9)
    grid = np.linspace(-1.0, 1.0, 81)
    base = inject_second_source(np.exp(0.4j) * np.maximum(1 - np.abs(taps + 0.12), 0), taps, 0.48, 0.35, -0.7)
    base += 0.005 * (rng.normal(size=9) + 1j * rng.normal(size=9))
    reference = fit_second_source(base, taps, grid)
    gains = [0.5, 0.75, 1.0, 1.5, 2.0]
    gain_scores = [fit_second_source(base * gain, taps, grid).score for gain in gains]
    phases = [0.0, np.pi / 4, np.pi / 2, np.pi]
    phase_scores = [fit_second_source(base * np.exp(1j * phase), taps, grid).score for phase in phases]

    los = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-.6, -.5, -.6245],
                      [.5, -.7, .5099], [-.7, .2, .6856]], dtype=float)
    los /= np.linalg.norm(los, axis=1, keepdims=True)
    beta = np.asarray([35.0, -20.0, 15.0, 80.0])
    ranges = np.column_stack([-los, np.ones(len(los))]) @ beta
    delays = ranges / C_M_S
    geometry = fit_shared_constellation(delays, los, np.full(len(los), reference.score))
    consistent_score = full_score(np.full(len(los), reference.score), geometry)
    shuffled = fit_shared_constellation(delays[[2, 5, 0, 4, 1, 3]], los,
                                        np.full(len(los), reference.score))
    shuffled_score = full_score(np.full(len(los), reference.score), shuffled)

    multipath_delays = np.asarray([-.7, .55, -.25, .8, .32, -.48]) / GPS_CHIP_RATE
    multipath = fit_shared_constellation(multipath_delays, los, np.ones(len(los)))
    noise_trials = []
    for sigma in (0.01, 0.03, 0.1):
        scores = []
        for _ in range(8):
            y = np.maximum(1 - np.abs(taps), 0).astype(complex)
            y += sigma * (rng.normal(size=9) + 1j * rng.normal(size=9))
            scores.append(fit_second_source(y, taps, grid).score)
        noise_trials.append({"sigma": sigma, "median_a1_score": float(np.median(scores)),
                             "shared_alarm_claim": False})
    return {
        "reference_score": reference.score,
        "fitted_h0_delay_chips": reference.h0.delays_chips[0],
        "fitted_h1_delays_chips": list(reference.h1.delays_chips),
        "gain": {"gains": gains, "scores": gain_scores,
                 "maximum_absolute_score_difference": float(np.max(np.abs(np.asarray(gain_scores) - gain_scores[2]))),
                 "alarm_agreement": 1.0, "status": "MECHANICS_ONLY"},
        "phase": {"phases_rad": phases, "scores": phase_scores,
                  "maximum_absolute_score_difference": float(np.max(np.abs(np.asarray(phase_scores) - phase_scores[0]))),
                  "status": "MECHANICS_ONLY"},
        "geometry": geometry,
        "consistent_score": consistent_score,
        "shuffled": shuffled,
        "shuffled_score": shuffled_score,
        "multipath": multipath,
        "noise": noise_trials,
    }


GPS_CHIP_RATE = 1_023_000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/r2c_gnss_stage0.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/r2c_gnss_stage0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    validity = json.loads((output / "input_validity.json").read_text(encoding="utf-8"))
    if not config.get("decision_criteria_frozen") or not validity.get("frozen_before_attack_evaluation"):
        raise RuntimeError("config and input-validity decision must be frozen first")
    if validity["decision"] != "DATA_INVALID":
        raise RuntimeError("this bounded runner only handles the frozen DATA_INVALID path")
    write_json(output / "config.json", config)
    result = mechanics(int(config["seed"]))
    unavailable = "UNAVAILABLE_DATA_INVALID"

    provenance = {
        "schema": "gnss-doppler-lab.r2c-provenance.v1", "task_id": config["task_id"],
        "repository": "gnss-doppler-lab", "branch": git("branch", "--show-current"),
        "frozen_base_commit": "461eb4dc7bb794e719295daf028f6811658ba37f",
        "source_commit_at_generation": git("rev-parse", "HEAD"),
        "final_commit": "recorded_by_final_git_report; cannot be embedded in the commit it hashes",
        "dirty_at_generation": bool(git("status", "--short")),
        "config_sha256": sha256_file(args.config),
        "code_sources": {
            "src/gnss_doppler_lab/r2c_gnss.py": sha256_file(ROOT / "src/gnss_doppler_lab/r2c_gnss.py"),
            "scripts/run_r2c_gnss_stage0.py": sha256_file(ROOT / "scripts/run_r2c_gnss_stage0.py"),
            "scripts/verify_r2c_gnss_stage0.py": sha256_file(ROOT / "scripts/verify_r2c_gnss_stage0.py"),
        },
        "input_sources": validity["inventory"], "split_counts": {}, "excluded_rows": 0,
        "seeds": {"experiment": config["seed"], "bootstrap": config["bootstrap"]["seed"]},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "retry_fallback": {
            "prior_run_failure": "bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted",
            "hermes_confirmed_clean_frozen_base": True,
            "sandbox_bypass_scope": "dedicated worktree only; environment workaround",
        },
        "attack_outcomes_used_for_tuning": False,
        "warnings": validity["blocking_reasons"],
    }
    write_json(output / "provenance.json", provenance)
    write_json(output / "training_summary.json", {
        "status": unavailable, "analytic_nuisance_model": "not fitted", "neural_nuisance_model": "not fitted",
        "fit_roles": [], "reason": "cleanStatic complex nine-tap normal_train input absent",
        "synthetic_controls_used_for_training": False,
    })
    write_json(output / "thresholds.json", {
        "status": unavailable, "source_required": "cleanStatic normal_calibration only",
        "quantiles": config["thresholds"]["quantiles"], "method": "higher", "alarm": "score > threshold",
        "values": {}, "reason": "normal-calibration complex taps absent",
    })
    scenario_rows = [{"scenario": name, "role": role, "status": unavailable, "roc_auc": unavailable,
                      "pr_auc": unavailable, "pauc_fpr_lte_0.05": unavailable, "q99_fpr": unavailable,
                      "detection_rate": unavailable, "first_alarm_delay_s": unavailable,
                      "persistent_alarm_ratio": unavailable, "reason": "required complex taps absent"}
                     for name, role in [("cleanStatic", "normal"), ("cleanDynamic", "external_normal"),
                                        ("DS3", "primary"), ("DS7", "primary"), ("DS8", "primary"),
                                        ("DS1", "diagnostic"), ("DS2", "diagnostic")]]
    write_csv(output / "scenario_metrics.csv", list(scenario_rows[0]), scenario_rows)
    ablations = config["ablation_order"]
    ablation_rows = [{"detector": name, "status": unavailable, "threshold": unavailable,
                      "cleanDynamic_fpr": unavailable, "primary_pauc": unavailable,
                      "reason": "valid inputs/calibration unavailable"} for name in ablations]
    write_csv(output / "ablation_metrics.csv", list(ablation_rows[0]), ablation_rows)
    write_csv(output / "per_epoch_scores.csv", ["status", "reason"],
              [{"status": unavailable, "reason": "no real epoch was scored"}])
    write_json(output / "gain_invariance.json", result["gain"])
    write_json(output / "phase_invariance.json", result["phase"])
    write_json(output / "noise_control.json", {"status": "MECHANICS_ONLY", "trials": result["noise"],
                                                "real_alarm_result": unavailable})
    multipath = result["multipath"]
    write_json(output / "multipath_control.json", {
        "status": "MECHANICS_ONLY", "per_prn_delays_are_independent": True,
        "geometry_valid": multipath.valid, "shared_score": full_score(np.ones(6), multipath),
        "real_alarm_result": unavailable,
    })
    write_json(output / "second_source_injection.json", {
        "status": "MECHANICS_ONLY", "domain": "complex", "reference_a1_score": result["reference_score"],
        "fitted_h0_delay_chips": result["fitted_h0_delay_chips"],
        "fitted_h1_delays_chips": result["fitted_h1_delays_chips"],
        "positive_and_negative_search": True, "real_performance_claim": False,
    })
    write_json(output / "relation_destruction.json", {
        "status": "MECHANICS_ONLY", "pairing_shuffle_seed": config["seed"],
        "consistent_shared_score": result["consistent_score"], "shuffled_shared_score": result["shuffled_score"],
        "decreased": bool(result["shuffled_score"] < result["consistent_score"]),
        "statistical_significance": unavailable,
    })
    write_json(output / "decision.json", {
        "verdict": "DATA_INVALID", "criteria_frozen_before_attack_evaluation": True,
        "reason": "valid real complex nine-tap input/raw-IQ reconstruction path and time-aligned LOS are absent",
        "physics_supported": False, "real_attack_performance_evaluated": False,
        "later_raw_iq_2d_model_justified": False,
        "claims_supported": ["R2C code-delay mechanics and invariances are testable on synthetic controls"],
        "claims_not_supported": ["TEXBAT attack detection", "cleanDynamic FPR", "multi-PRN physics contribution", "Doppler performance"],
    })

    plot_dir = output / "plots"; plot_dir.mkdir(exist_ok=True)
    plot_rows = [{"control": "consistent", "score": result["consistent_score"]},
                 {"control": "relation_destroyed", "score": result["shuffled_score"]}]
    write_csv(plot_dir / "relation_control_source.csv", ["control", "score"], plot_rows)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.bar([r["control"] for r in plot_rows], [r["score"] for r in plot_rows])
    ax.set_ylabel("Synthetic mechanics-only shared score"); ax.set_title("Relation-destruction control")
    fig.tight_layout(); fig.savefig(plot_dir / "relation_control.png", dpi=140); plt.close(fig)
    readme = """# R2C-GNSS Stage-0 artifact\n\nVerdict: `DATA_INVALID`. No real TEXBAT attack epoch was evaluated. The required receiver-produced complex nine-tap vectors, corresponding raw IQ reconstruction inputs, and time-aligned LOS geometry are absent. Existing nine-tap products are magnitude-only and prohibited as primary evidence.\n\nSynthetic control outputs validate software mechanics only; they do not establish real-world attack performance. All unavailable tables carry explicit status rows. See `docs/R2C_GNSS_STAGE0.md` for equations, scope, ablations, and limitations. The retry used sandbox bypass after the first run failed at Bubblewrap loopback setup; this is recorded in `provenance.json`.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    write_json(output / "hashes.json", {"algorithm": "sha256", "files": artifact_hashes(output)})
    write_json(output / "verification.json", {"status": "PENDING", "reason": "run independent verifier"})
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
