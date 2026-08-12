"""Frozen protected-evaluation scheduling, pooling, bootstrap, and gates."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .gcspo_core import (block_index, content_seed, los_derangement,
                         nearest_rank_percentile, temporal_desynchronization,
                         weighted_low_fpr_pauc)


def phase_contained(row, phase_start, phase_end):
    """The full window and right endpoint must be inside a half-open phase."""
    start = float(row["window_start_s"]); end = float(row["availability_s"])
    return start >= float(phase_start) and end < float(phase_end)


def scheduled_persistence(rows, *, threshold, slot_s=.5):
    """Causal 3-of-5 on exact scheduled slots; a missing slot resets history."""
    ordered = sorted(rows, key=lambda row: float(row["availability_s"]))
    output, run = [], []
    previous = None
    for row in ordered:
        current = float(row["availability_s"])
        if previous is None or not np.isclose(current - previous, slot_s, rtol=0, atol=1e-12):
            run = []
        run.append(float(row["score"]) > float(threshold))
        if len(run) > 5: run.pop(0)
        output.append(len(run) == 5 and sum(run) >= 3)
        previous = current
    return output


def _authenticated_actual_support(row):
    try:
        epochs = tuple(map(int, row["epoch_ids"]))
        support = tuple((int(epoch), tuple(map(int, prns)))
                        for epoch, prns in row["epoch_prn_support"])
        prns = tuple(map(int, row["prns"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing authenticated actual support") from None
    valid = bool(epochs) and tuple(epoch for epoch, _ in support) == epochs
    valid = valid and epochs == tuple(sorted(set(epochs)))
    valid = valid and all(len(epoch_prns) >= 4 and epoch_prns == tuple(sorted(set(epoch_prns)))
                          for _, epoch_prns in support)
    union = tuple(sorted(set().union(*(set(epoch_prns) for _, epoch_prns in support)))) if support else ()
    if not valid or prns != union:
        raise ValueError("malformed authenticated actual support")
    return epochs, prns, support


def _event_key(row):
    epochs, prns, support = _authenticated_actual_support(row)
    return (str(row["scenario"]), str(row["phase"]), float(row["window_start_s"]),
            float(row["availability_s"]), epochs, prns, support)


def exact_contrast_support(rows: Iterable[dict], methods):
    """Return only events with identical event/epoch/PRN support for all methods."""
    methods = tuple(methods)
    grouped: dict[tuple, dict[str, dict]] = {}
    seen = set()
    for row in rows:
        method = str(row["method"])
        if method not in methods or not np.isfinite(float(row["score"])):
            continue
        duplicate_key = (str(row["scenario"]), str(row["phase"]), float(row["window_start_s"]),
                         float(row["availability_s"]), method)
        if duplicate_key in seen:
            raise ValueError("duplicate method row on exact scientific support")
        seen.add(duplicate_key)
        bucket = grouped.setdefault(_event_key(row), {})
        bucket[method] = row
    result = []
    for key in sorted(grouped):
        bucket = grouped[key]
        if set(bucket) != set(methods): continue
        reference = bucket[methods[0]]
        result.append({key: reference[key] for key in ("scenario", "phase", "window_start_s", "availability_s")}
                      | {"epoch_ids": tuple(reference["epoch_ids"]), "prns": tuple(reference["prns"]),
                         "epoch_prn_support": tuple(reference["epoch_prn_support"]),
                         "label": bool(reference.get("label", reference["phase"] not in {"pre_onset", "holdout"})),
                         "phase_start_s": float(reference.get("phase_start_s", 0.)),
                         "phase_end_s": float(reference.get("phase_end_s", math.inf)),
                         "scores": {method: float(bucket[method]["score"]) for method in methods}})
    return result


def _logical_cell(row):
    scenario, phase, label = str(row["scenario"]), str(row["phase"]), bool(row["label"])
    prefix = "positive" if label else "negative"
    family = "DS7_DS8" if scenario in {"DS7", "DS8"} else scenario
    return f"{prefix}:{family}:{phase}"


def primary_pauc_rows(rows, *, method):
    selected = []
    for original in rows:
        if original.get("method") != method or not np.isfinite(float(original["score"])): continue
        scenario, phase = str(original["scenario"]), str(original["phase"])
        if scenario in {"DS1", "DS2", "DS5", "DS6", "cleanDynamic"}: continue
        if scenario in {"DS7", "DS8"} and phase == "pre_onset_replay": continue
        if phase not in {"holdout", "pre_onset", "transition", "established"}: continue
        row = dict(original); row["logical_cell"] = _logical_cell(row); selected.append(row)
    return selected


def hierarchical_scenario_phase_weights(rows):
    """Equalize classes/cells, then available physical members, then member rows."""
    rows = list(rows)
    if not rows: raise ValueError("hierarchical weights require rows")
    if any(not isinstance(row.get("scenario"), str) or not row["scenario"] for row in rows):
        raise ValueError("hierarchical weight member is missing")
    labels = np.asarray([bool(row["label"]) for row in rows])
    cells = np.asarray([row.get("logical_cell", _logical_cell(row)) for row in rows], dtype=object).astype(str)
    members = np.asarray([row["scenario"] for row in rows], dtype=object).astype(str)
    weights = np.zeros(len(rows), dtype=np.float64)
    for positive in (True, False):
        names = sorted(set(cells[labels == positive]))
        if not names: raise ValueError("hierarchical weights require both classes")
        for name in names:
            cell_indices = np.flatnonzero((labels == positive) & (cells == name))
            available = sorted(set(members[cell_indices]))
            for member in available:
                indices = cell_indices[members[cell_indices] == member]
                weights[indices] = 1 / (len(names) * len(available) * len(indices))
    return weights


def scenario_phase_balanced_pauc(rows, *, score_field="score"):
    rows = list(rows)
    if not rows: raise ValueError("pAUC support is empty")
    scores = np.asarray([float(row[score_field]) for row in rows], dtype=np.float64)
    labels = np.asarray([bool(row["label"]) for row in rows])
    cells = np.asarray([row.get("logical_cell", _logical_cell(row)) for row in rows])
    weights = hierarchical_scenario_phase_weights(rows)
    return weighted_low_fpr_pauc(scores, labels, cells, alpha=.05, row_weights=weights)


def _bootstrap_eligible(row):
    phase_start, phase_end = float(row["phase_start_s"]), float(row["phase_end_s"])
    index = block_index(row["availability_s"], phase_start=phase_start)
    lower, upper = phase_start + index * 10., min(phase_start + (index + 1) * 10., phase_end)
    if upper - lower < 5.: return None
    if float(row["window_start_s"]) < lower or float(row["availability_s"]) > upper + 1e-12: return None
    return index


def paired_block_bootstrap(rows, first_method, second_method, *, replicates=2000, seed=23):
    if replicates != 2000 or seed != 23:
        raise ValueError("frozen bootstrap requires 2000 replicates and seed 23")
    paired = exact_contrast_support(rows, (first_method, second_method))
    cells: dict[str, dict[int, list[dict]]] = {}
    for row in paired:
        block = _bootstrap_eligible(row)
        if block is None: continue
        cell = _logical_cell(row)
        cells.setdefault(cell, {}).setdefault(block, []).append(row)
    if not cells or not any(cell.startswith("positive:") for cell in cells) or not any(cell.startswith("negative:") for cell in cells):
        raise ValueError("bootstrap requires eligible positive and negative cells")
    rng = np.random.Generator(np.random.PCG64(seed)); values = np.empty(replicates)
    first_family_draws = []
    for replicate in range(replicates):
        sampled = []
        for cell in sorted(cells):
            blocks = sorted(cells[cell]); draws = rng.integers(0, len(blocks), size=len(blocks))
            for draw in draws:
                selected = blocks[int(draw)]; sampled.extend(cells[cell][selected])
                if replicate == 0 and "DS7_DS8" in cell:
                    first_family_draws.append({"DS7_DS8:transition:DS7": selected,
                                               "DS7_DS8:transition:DS8": selected})
        def pauc(method):
            material = [{**row, "score": row["scores"][method], "logical_cell": _logical_cell(row)} for row in sampled]
            return scenario_phase_balanced_pauc(material)
        values[replicate] = pauc(first_method) - pauc(second_method)
    return {"replicates": replicates, "seed": seed, "values": values,
            "lcb_95": nearest_rank_percentile(values, .05),
            "interval_95": [nearest_rank_percentile(values, .025), nearest_rank_percentile(values, .975)],
            "family_coupled": True, "first_replicate_family_draws": first_family_draws}


def execute_relation_destructions(residual, los, prns, *, scenario, phase, segment_id):
    residual, los, prns = np.asarray(residual, float), np.asarray(los, float), np.asarray(prns, int)
    order = np.argsort(prns); residual, los, prns = residual[order], los[order], prns[order]
    if residual.ndim != 3 or los.ndim not in (2, 3) or los.shape[0] != len(prns) or los.shape[-1] != 3 or residual.shape[0] != len(prns):
        raise ValueError("relation destruction shape mismatch")
    los_seed = content_seed("LOS_SHUFFLE", scenario, phase, 0, "NA", segment_id)
    _, permutation, los_record = los_derangement(los[:, 0] if los.ndim == 3 else los, seed=los_seed)
    shuffled = los[permutation]
    temporal_seed = content_seed("PER_PRN_TEMPORAL_SHIFT", scenario, phase, 0, "NA", segment_id)
    shifted, shifts = temporal_desynchronization(residual, seed=temporal_seed)
    los_preserved = (sorted(row.tobytes() for row in shuffled) == sorted(row.tobytes() for row in los)
                     and not np.any(permutation == np.arange(len(prns))))
    residual_preserved = all(sorted(map(tuple, before)) == sorted(map(tuple, after))
                             for before, after in zip(residual, shifted))
    residual_preserved = bool(residual_preserved and np.isclose(np.sum(residual * residual), np.sum(shifted * shifted)))
    if not los_preserved or not residual_preserved: raise RuntimeError("relation preservation assertion failed")
    return {"sorted_prns": prns, "shuffled_los": shuffled, "los_permutation": permutation,
            "shifted_residual": shifted, "temporal_shifts": shifts,
            "seeds": {"LOS_SHUFFLE": los_seed, "PER_PRN_TEMPORAL_SHIFT": temporal_seed},
            "draw": los_record["draw"],
            "preservation": {"LOS_SHUFFLE": los_preserved, "PER_PRN_TEMPORAL_SHIFT": residual_preserved},
            "numpy_version": np.__version__}


def compute_scientific_gates(evidence):
    external = evidence["external_pre_fpr"]
    g1 = evidence["clean_holdout_fpr"] <= .02 and bool(external) and max(external.values()) <= .05
    incremental = evidence["incremental_lcb"]
    g2 = set(incremental) == {"Full-A1", "Full-A2"} and all(value > 0 for value in incremental.values())
    destruction = evidence["destruction"]
    required_destructions = {"LOS_SHUFFLE", "PER_PRN_TEMPORAL_SHIFT"}
    g3 = set(destruction) == required_destructions and all(
        destruction[name]["lcb"] > 0 and destruction[name]["median_relative_loss"] >= .25
        for name in required_destructions)
    persistence = evidence["persistence"]
    g4 = set(persistence) >= {"DS3", "DS7_DS8"} and all(
        persistence[name]["ratio"] >= .5 and persistence[name]["delay_s"] <= 10
        for name in ("DS3", "DS7_DS8"))
    controls = evidence["controls"]
    clock = [row for row in controls if row["id"] == "CLOCK_DRIFT"]
    nonclock = [row for row in controls if row["id"] != "CLOCK_DRIFT"]
    g5 = bool(clock) and bool(nonclock) and all(row["specificity_ratio"] <= .25 for row in clock)
    g5 = g5 and all(row["persistent_alarm_ratio"] <= .10 and row["max_consecutive_alarms"] < 10 for row in nonclock)
    shared = evidence["shared"]
    g6 = shared["full_pauc"] >= shared["a5_pauc"] - .01 and shared["full_median_edf"] < shared["a5_median_edf"]
    values = (g1, g2, g3, g4, g5, g6)
    ids = ("G1_FALSE_ALARM", "G2_INCREMENTAL", "G3_GEOMETRY", "G4_PERSISTENCE", "G5_CONTROLS", "G6_SHARED")
    return [{"id": name, "status": "PASS" if passed else "FAIL", "computed": True}
            for name, passed in zip(ids, values)]
