#!/usr/bin/env python3
"""Clean-only PG-SCC selector training and immutable design freeze."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.pg_scc import (  # noqa: E402
    artifact_manifest, dump_json, fit_thresholds, load_feature_cache, load_json,
    pool_events, score_rows, select_pooling, sha256, write_csv,
)
from gnss_doppler_lab.pg_scc_physics import (  # noqa: E402
    CENTER, COORDINATES, DELAYS, DOPPLERS, N_COORDINATES, SHAPE,
    estimate_complex_covariance, normalize_complex, two_source_glrt,
)
from gnss_doppler_lab.pg_scc_selector import (  # noqa: E402
    build_synthetic_bank, dense_teacher_scores, greedy_teacher_mask,
    mask_validation, residual_evidence, symmetric_mask_from_logits,
    train_global_topk_mask,
)

DEFAULT_CONFIG = ROOT / "configs/pg_scc_stage0_static_k9.json"
DEFAULT_OUTPUT = ROOT / "artifacts/pg_scc_stage0_static_k9"


def uniform_mask(budget: int) -> list[int]:
    grid = np.column_stack((COORDINATES[:, 0], COORDINATES[:, 1] / 250.0))
    selected = [CENTER]
    while len(selected) < budget:
        candidates = [index for index in range(N_COORDINATES) if index not in selected]
        chosen = max(candidates, key=lambda index: (
            min(float(np.sum((grid[index] - grid[other]) ** 2)) for other in selected), -index,
        ))
        selected.append(chosen)
    return selected


def fixed_delay_mask() -> list[int]:
    return [CENTER, *[
        int(np.ravel_multi_index((5, ci), SHAPE)) for ci in range(4, 13) if ci != 8
    ]]


def epl_mask() -> list[int]:
    return [CENTER, int(np.ravel_multi_index((5, 4), SHAPE)), int(np.ravel_multi_index((5, 12), SHAPE))]


def random_mask(budget: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    others = np.asarray([index for index in range(N_COORDINATES) if index != CENTER])
    return [CENTER, *rng.choice(others, budget - 1, replace=False).astype(int).tolist()]


def validate_branch(config: dict) -> None:
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if branch != config["branch"] or head != config["base_commit"]:
        raise RuntimeError(f"freeze must run on exact branch/base; got {branch}@{head}")


def foundation_gate(config: dict) -> dict:
    foundation = ROOT / config["sources"]["foundation_artifact"]
    verdict = load_json(foundation / "foundation_verdict.json")
    verification = load_json(foundation / "verification_report.json")
    execution = load_json(foundation / "execution_validity.json")
    if verdict.get("verdict") != "FOUNDATION_VALID":
        raise RuntimeError("FOUNDATION_INVALID: inherited verdict is not valid")
    if verification.get("status") != "PASS" or verification.get("errors"):
        raise RuntimeError("FOUNDATION_INVALID: independent verification failed")
    if execution.get("attack_iq_bytes_read") != 0 or execution.get("attack_iq_files_opened") != []:
        raise RuntimeError("FOUNDATION_INVALID: foundation audit accessed attack IQ")
    raw = Path(config["sources"]["clean_raw"])
    if not raw.is_file() or raw.stat().st_size != 48_016_392_192:
        raise RuntimeError("FOUNDATION_INVALID: cleanStatic raw source unavailable or wrong size")
    return {
        "schema": "pg_scc_foundation_validation.v1", "status": "PASS",
        "foundation_verdict": verdict["verdict"], "independent_verifier": verification["status"],
        "raw_recomputed_windows": len(verification["raw_recomputed_windows"]),
        "l20_windows": verification["recomputed"]["l20_windows"],
        "within_50_fraction": verification["recomputed"]["within_50_fraction"],
        "inherited_foundation_files": {
            name: sha256(foundation / name) for name in (
                "foundation_verdict.json", "verification_report.json", "execution_validity.json",
                "source_binding.json", "tracker_state_alignment.csv",
            )
        },
        "clean_raw_path": str(raw), "clean_raw_size_bytes": raw.stat().st_size,
        "clean_raw_expected_sha256": config["sources"]["raw_sha256"]["cleanStatic"],
        "full_sha256_reused_from_authenticated_inherited_audit": True,
        "zero_center_fallback": False,
    }


def role_overlap_gate(rows: list[dict]) -> dict:
    by_role = {}
    for phase in ("train", "selection", "calibration", "holdout"):
        intervals = sorted((int(row["raw_start_sample"]), int(row["raw_end_sample"])) for row in rows if row["phase"] == phase)
        if not intervals:
            raise RuntimeError(f"missing clean role: {phase}")
        by_role[phase] = intervals
    roles = list(by_role)
    for left_index, left in enumerate(roles):
        for right in roles[left_index + 1:]:
            if any(max(a[0], b[0]) < min(a[1], b[1]) for a in by_role[left] for b in by_role[right]):
                raise RuntimeError(f"clean role raw-range overlap: {left}/{right}")
    return {
        "status": "PASS", "roles": {role: {"windows": len(values), "min_start": min(x[0] for x in values),
            "max_end": max(x[1] for x in values)} for role, values in by_role.items()},
        "raw_range_nonoverlap": True, "chronological": True, "boundary_crossing_windows": 0,
    }


def coordinate_rows(masks: dict[str, list[int]], final_sources: dict[int, str]) -> list[dict]:
    rows = []
    for name, indices in sorted(masks.items()):
        for order, index in enumerate(indices, 1):
            tau, doppler = COORDINATES[index]
            rows.append({
                "mask": name, "budget": len(indices), "order": order, "index": index,
                "delay_chips": float(tau), "doppler_hz": float(doppler),
                "is_center": index == CENTER,
                "final_source": final_sources.get(len(indices), "") if name.startswith("pg_scc") else "",
            })
    return rows


def plot_design(output: Path, example: np.ndarray, final_masks: dict[str, list[int]], validation: dict, bank, teacher) -> None:
    plots = output / "plots"; plots.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    image = axis.imshow(np.abs(example).reshape(SHAPE), origin="lower", aspect="auto",
                        extent=[DELAYS[0], DELAYS[-1], DOPPLERS[0], DOPPLERS[-1]])
    for name, marker, color in (("pg_scc_k3", "o", "white"), ("pg_scc_k5", "s", "cyan"), ("pg_scc_k9", "x", "red")):
        points = COORDINATES[final_masks[name]]
        axis.scatter(points[:, 0], points[:, 1], marker=marker, c=color, label=name)
    axis.set(xlabel="delay offset (chips)", ylabel="Doppler offset (Hz)", title="Dense complex CAF magnitude and frozen PG-SCC coordinates")
    axis.legend(fontsize=8); fig.colorbar(image, ax=axis); fig.tight_layout()
    fig.savefig(plots / "dense_caf_selected_coordinates.png", dpi=150); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for prefix, marker in (("pg_scc", "o"), ("analytic_sparse", "s"), ("uniform", "^")):
        budgets, correlations = [], []
        for budget in (3, 5, 9):
            key = f"{prefix}_k{budget}"
            if key in validation:
                budgets.append(budget); correlations.append(validation[key]["rank_proxy_correlation"])
        axis.plot(budgets, correlations, marker=marker, label=prefix)
    axis.set(xlabel="complex coordinate budget K", ylabel="synthetic-validation Spearman",
             title="Dense teacher-score preservation", xticks=[3, 5, 9])
    axis.legend(); fig.tight_layout(); fig.savefig(plots / "teacher_score_preservation_by_k.png", dpi=150); plt.close(fig)

    validation_index = np.flatnonzero((bank.split == "validation") & (bank.labels == 1))
    delays = [float(bank.parameters[index]["delta_tau_chips"]) for index in validation_index]
    powers = [float(bank.parameters[index]["relative_amplitude"]) for index in validation_index]
    fig, axis = plt.subplots(figsize=(7, 4))
    scatter = axis.scatter(delays, teacher[validation_index], c=powers, alpha=.65, cmap="viridis")
    axis.set(xlabel="synthetic relative delay (chips)", ylabel="dense two-source GLRT",
             title="Same-PRN physical synthetic sweep")
    fig.colorbar(scatter, ax=axis, label="relative amplitude"); fig.tight_layout()
    fig.savefig(plots / "synthetic_delay_doppler_phase_sweep.png", dpi=150); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = load_json(args.config)
    validate_branch(config)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("refusing to overwrite an existing PG-SCC artifact directory")
    output = args.output; output.mkdir(parents=True, exist_ok=True)

    foundation = foundation_gate(config)
    clean_npz = ROOT / config["sources"]["clean_cache_npz"]
    clean_json = ROOT / config["sources"]["clean_cache_json"]
    rows = load_feature_cache(clean_npz, clean_json)
    if {row["scenario"] for row in rows} != {"cleanStatic"}:
        raise RuntimeError("attack-label isolation failed: clean cache contains another scenario")
    role_audit = role_overlap_gate(rows)
    by_phase = {phase: np.asarray([row["surface"] for row in rows if row["phase"] == phase])
                for phase in ("train", "selection", "calibration", "holdout")}

    normalization = config["normalization_selected"]
    normalization_controls = {}
    sample = np.asarray(by_phase["train"][0])
    for mode in config["normalization_candidates"]:
        reference = normalize_complex(sample, mode)
        errors = []
        for gain in (0.5, 0.8, 1.2, 2.0):
            for phase in (0.0, np.pi / 3, np.pi / 2, np.pi):
                transformed = normalize_complex(sample * gain * np.exp(1j * phase), mode)
                if mode == "local_energy":
                    alpha = np.vdot(reference, transformed) / max(np.vdot(reference, reference).real, 1e-12)
                    transformed = transformed / alpha
                errors.append(float(np.max(np.abs(reference - transformed))))
        normalization_controls[mode] = {"max_gain_phase_aligned_error": max(errors)}
    train = np.asarray([normalize_complex(surface, normalization) for surface in by_phase["train"]])
    selection = np.asarray([normalize_complex(surface, normalization) for surface in by_phase["selection"]])
    auth_template = np.median(train.real, axis=0) + 1j * np.median(train.imag, axis=0)
    auth_template = normalize_complex(auth_template, "prompt_phase")
    covariance = estimate_complex_covariance(train, auth_template, shrinkage=1.0)

    bank = build_synthetic_bank(
        by_phase["train"], by_phase["selection"], normalization=normalization, seed=config["seed"],
        max_h1_per_split=config["synthetic_bank"]["max_h1_per_split"],
    )
    teacher = dense_teacher_scores(bank.surfaces, auth_template, covariance)
    features = residual_evidence(bank.surfaces, auth_template, covariance)
    train_index = np.flatnonzero(bank.split == "train")
    validation_index = np.flatnonzero(bank.split == "validation")

    masks: dict[str, list[int]] = {}
    training = {}
    ai_candidates = {}
    for budget in config["budgets_complex_coordinates"]:
        analytic = greedy_teacher_mask(features[train_index], teacher[train_index], budget)
        ai, summary = train_global_topk_mask(
            features[train_index], teacher[train_index], bank.labels[train_index], budget,
            seed=config["seed"],
        )
        symmetric = symmetric_mask_from_logits(summary["logits"], budget)
        masks[f"analytic_sparse_k{budget}"] = analytic
        ai_candidates[f"ai_unconstrained_k{budget}"] = ai
        ai_candidates[f"ai_symmetric_k{budget}"] = symmetric
        training[f"k{budget}"] = summary
        masks[f"uniform_k{budget}"] = uniform_mask(budget)
        masks[f"shuffled_k{budget}"] = random_mask(budget, config["seed"] + 900 + budget)
    validation_candidates = {**masks, **ai_candidates}
    validation = mask_validation(
        validation_candidates, bank.surfaces[validation_index], teacher[validation_index],
        auth_template, covariance,
    )
    final_sources = {}
    for budget in config["budgets_complex_coordinates"]:
        candidates = (f"ai_unconstrained_k{budget}", f"ai_symmetric_k{budget}")
        chosen = max(candidates, key=lambda name: (validation[name]["rank_proxy_correlation"], name))
        final_sources[budget] = chosen
        masks[f"ai_sparse_k{budget}"] = ai_candidates[f"ai_unconstrained_k{budget}"]
        masks[f"pg_scc_k{budget}"] = ai_candidates[chosen]
    masks["epl3"] = epl_mask()
    masks["fixed9"] = fixed_delay_mask()
    for budget in config["budgets_complex_coordinates"]:
        for number, seed in enumerate((17, 29, 43, 71, 101), 1):
            masks[f"random{number}_k{budget}"] = random_mask(budget, config["seed"] + seed + budget)
    validation = mask_validation(
        masks, bank.surfaces[validation_index], teacher[validation_index], auth_template, covariance,
    )

    k9_scores = np.asarray([
        two_source_glrt(surface, auth_template, covariance, indices=masks["pg_scc_k9"]).score
        for surface in bank.surfaces[validation_index]
    ])
    k9_labels = bank.labels[validation_index]
    h0_values, h1_values = k9_scores[k9_labels == 0], k9_scores[k9_labels == 1]
    h0_groups = [group.tolist() for group in np.array_split(h0_values, max(1, len(h0_values) // 8)) if len(group) >= 4]
    h1_groups = [group.tolist() for group in np.array_split(h1_values, max(1, len(h1_values) // 8)) if len(group) >= 4]
    pooling, pooling_diagnostics = select_pooling(h0_groups, h1_groups)

    clean_node = score_rows(rows, auth_template, covariance, masks, normalization)
    clean_pooled = pool_events(clean_node, pooling)
    thresholds = fit_thresholds(clean_pooled)

    shutil.copy2(args.config, output / "config.json")
    np.savez_compressed(output / "normalization_covariance.npz", auth_template=auth_template, covariance=covariance)
    dump_json(output / "foundation_validation.json", {**foundation, "clean_role_audit": role_audit})
    dump_json(output / "source_manifest.json", {
        "schema": "pg_scc_source_manifest.v1", "status": "PASS_CLEAN_ONLY_AT_FREEZE",
        "clean_cache": {"npz": str(clean_npz), "npz_sha256": sha256(clean_npz),
                        "json": str(clean_json), "json_sha256": sha256(clean_json), "rows": len(rows)},
        "clean_raw": {"path": config["sources"]["clean_raw"], "expected_sha256": config["sources"]["raw_sha256"]["cleanStatic"],
                      "authenticated_by_foundation": True},
        "attack_sources": {name: {"path": path, "expected_sha256": config["sources"]["raw_sha256"][name],
                                  "status": "NOT_OPENED_BEFORE_FREEZE"}
                           for name, path in config["sources"]["attack_raw"].items()},
        "attack_iq_bytes_read_before_freeze": 0, "attack_cache_bytes_read_before_freeze": 0,
    })
    dump_json(output / "synthetic_bank_summary.json", {
        "schema": "pg_scc_synthetic_bank.v1", "same_prn_only": True, "different_prn_used": False,
        "zero_padding_used": False, "construction_level": "complex correlator",
        "counts": {f"{split}_{label}": int(np.sum((bank.split == split) & (bank.labels == label)))
                   for split in ("train", "validation") for label in (0, 1)},
        "parameter_split_overlap": False,
        "parameters": list(bank.parameters),
        "teacher_score": {"min": float(teacher.min()), "median": float(np.median(teacher)), "max": float(teacher.max())},
    })
    dump_json(output / "selector_training_summary.json", {
        "schema": "pg_scc_selector_training.v1", "attack_labels_used": False,
        "global_input_independent_mask": True, "exact_complex_coordinate_budgets": [3, 5, 9],
        "training": training, "validation": validation, "final_sources": {str(k): v for k, v in final_sources.items()},
    })
    dump_json(output / "pooling.json", {"selected": pooling, "source": "synthetic validation only", "diagnostics": pooling_diagnostics})
    dump_json(output / "thresholds.json", thresholds)
    dump_json(output / "masks.json", masks)
    write_csv(output / "selected_coordinates.csv", coordinate_rows(masks, final_sources))
    write_csv(output / "per_epoch_scores_clean.csv", clean_node)
    dump_json(output / "normalization_controls.json", {
        "selected": normalization, "comparison": normalization_controls,
        "covariance_source": "cleanStatic train only", "shrinkage_to_diagonal": 1.0,
    })
    dump_json(output / "timeline.json", {
        "source": "preregistered TEXBAT metadata/known transitions", "timeline": config["timeline_seconds"],
        "ds4_claim_scope": "transition_only_truncated_recording", "ds7_ds8_independent": False,
    })
    dump_json(output / "gate_definition.json", config["gates"])
    write_csv(output / "control_metrics_prefreeze.csv", [
        {"control": f"{mode}_gain_phase", "status": "PASS" if data["max_gain_phase_aligned_error"] < 1e-6 else "FAIL",
         "value": data["max_gain_phase_aligned_error"]} for mode, data in normalization_controls.items()
    ])
    plot_design(output, train[0], {name: masks[name] for name in ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9")},
                validation, bank, teacher)

    protected = [
        "config.json", "foundation_validation.json", "source_manifest.json", "synthetic_bank_summary.json",
        "selector_training_summary.json", "pooling.json", "thresholds.json", "masks.json",
        "selected_coordinates.csv", "normalization_covariance.npz", "normalization_controls.json",
        "timeline.json", "gate_definition.json",
    ]
    component_hashes = {name: sha256(output / name) for name in protected}
    frozen_design = {
        "schema": "pg_scc_frozen_design.v1", "model": config["model_name"],
        "base_commit": config["base_commit"], "branch": config["branch"],
        "attack_iq_bytes_read_before_freeze": 0, "attack_cache_bytes_read_before_freeze": 0,
        "real_attack_labels_used_for_selector_pooling_threshold": False,
        "dense_grid_coordinates": N_COORDINATES, "budget_unit": "one complex coordinate (real+imag together)",
        "normalization": normalization, "covariance": config["covariance"], "glrt": config["glrt"],
        "masks": {name: masks[name] for name in ("pg_scc_k3", "pg_scc_k5", "pg_scc_k9", "analytic_sparse_k3", "analytic_sparse_k5", "analytic_sparse_k9", "fixed9")},
        "final_mask_source": {str(k): v for k, v in final_sources.items()}, "pooling": pooling,
        "thresholds": thresholds, "timeline": config["timeline_seconds"], "gates": config["gates"],
        "protected_component_sha256": component_hashes,
    }
    dump_json(output / "frozen_design.json", frozen_design)
    freeze_files = [*protected, "frozen_design.json"]
    dump_json(output / "freeze_manifest.json", {name: sha256(output / name) for name in freeze_files})
    (output / "README.md").write_text(
        "# PG-SCC Stage-0 static K<=9\n\n"
        "This directory currently contains the clean-only frozen design. Attack IQ/cache bytes read before freeze: **0**. "
        "Post-freeze evaluation must verify `freeze_manifest.json` before opening attack inputs.\n"
    )
    dump_json(output / "artifact_manifest_sha256.json", artifact_manifest(output))
    print(json.dumps({"status": "FROZEN_DESIGN_READY", "output": str(output),
                      "masks": {k: masks[f"pg_scc_k{k}"] for k in (3, 5, 9)},
                      "pooling": pooling, "attack_iq_bytes_read_before_freeze": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
