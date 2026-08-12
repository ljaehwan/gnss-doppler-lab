"""Execution and evidence packaging for the frozen clean control grid."""
from __future__ import annotations

import numpy as np

from .gcspo_statistics import scheduled_persistence

CONTROL_LEVELS = {
    "COMMON_GAIN": (.5, .8, 1.2, 2.),
    "PROMPT_AMPLITUDE": (.5, .8, 1.2, 2.),
    "CN0_METADATA_EXCLUSION_INVARIANCE": (-3., -6., -10.),
    "PRN_DROP_ONLY": (1, 2, 4),
    "EMPIRICAL_NOISE": (.5, 1., 2.),
    "ONE_PRN_DISTURBANCE": (1., 2., 4.),
    "INDEPENDENT_MULTIPATH_LIKE": (.5, 1., 2.),
    "CLOCK_DRIFT": (.1, 1., 5.),
}


def _max_alarm_run(alarms):
    longest = current = 0
    for alarm in alarms:
        current = current + 1 if alarm else 0
        longest = max(longest, current)
    return longest


def generate_control_grid(contexts, *, scenario, phase, var_coefficients, threshold, scorer):
    """Execute, semantically verify, and score every frozen block/control/level."""
    from .gcspo_controls import apply_control

    rows, failures = [], []
    for block_id, block_start_s, context in contexts:
        for control_id, levels in CONTROL_LEVELS.items():
            for level in levels:
                try:
                    selected_context = context(control_id, level) if callable(context) else context
                    result = apply_control(selected_context, control_id=control_id, level=level,
                                           scenario=scenario, phase=phase, block_id=block_id,
                                           var_coefficients=var_coefficients)
                    values = [float(value) for value in scorer(result)]
                    if not all(np.isfinite(values)):
                        raise ValueError("control scorer returned nonfinite values")
                    score_rows = [{"availability_s": block_start_s + 1. + .5 * index,
                                   "score": value} for index, value in enumerate(values)]
                    alarms = [value > float(threshold) for value in values]
                    persistent = scheduled_persistence(score_rows, threshold=threshold) if score_rows else []
                    rows.append({
                        "id": control_id, "level": level, "block_id": block_id,
                        "block_start_s": block_start_s, "block_end_s": block_start_s + 10.,
                        "stage": result.stage, "seed_material": result.seed_material, "seed": result.seed,
                        "numpy_version": result.numpy_version, "history_reset": result.history_reset,
                        "first_eligible_epoch": result.first_eligible_epoch,
                        "source_block_index": result.source_block_index,
                        "source_mapping": result.source_mapping, "prn_phases_rad": result.prn_phases_rad,
                        "var_transfer_application_count": result.var_transfer_application_count,
                        "state_energy": result.state_energy,
                        "specificity_ratio": result.state_energy.get("specificity_ratio", 0.),
                        "scores": values, "score_count": len(values), "alarm_count": sum(alarms),
                        "persistent_alarm_ratio": float(np.mean(persistent)) if persistent else 0.,
                        "max_consecutive_alarms": _max_alarm_run(alarms), "status": "PASS",
                    })
                except Exception as exc:
                    failures.append({"id": control_id, "level": level, "block_id": block_id,
                                     "error_type": type(exc).__name__, "detail": str(exc)})
    return {
        "schema": "gnss-doppler-lab.gcspo-stage0.clean-control-generation.v2",
        "overall_status": "PASS" if not failures else "FAIL",
        "results": rows, "failures": failures, "block_count": len(contexts),
        "numpy_version": np.__version__, "protected_attack_rows_read": False,
        "attack_access_count": 0,
    }
