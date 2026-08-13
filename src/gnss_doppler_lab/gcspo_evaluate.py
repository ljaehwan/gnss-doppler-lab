"""Delivered one-shot evaluator for protected static TEXBAT receiver outputs."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys

import numpy as np

from .gcspo_a5 import role_a5_terms, score_a5_terms
from .gcspo_b0 import build_protected_scheduled_node_table
from .gcspo_ablations import role_a2_terms, score_a2_terms
from .gcspo_artifacts import canonical_write_json, sha256_file
from .gcspo_clean import EPOCH_S, a1_role_scores, residual_table
from .gcspo_full import (GeometryCache, geometry_preflight, protected_geometry_preflight, role_full_terms, role_full_terms_from_z,
                         score_full_terms)
from .gcspo_statistics import (compute_scientific_gates, exact_b0_full_contrast,
                               exact_contrast_support, execute_relation_destructions, paired_block_bootstrap,
                               paired_score_loss_bootstrap, primary_pauc_rows,
                               RELATION_POLICY, scenario_phase_balanced_pauc, scheduled_persistence,
                               validate_mandatory_relation_evidence)
from .gcspo_protected import (discrimination_metrics, load_receiver_tracking, phase_rows,
                              reconstruct_normal_model, scientific_verdict, score_metrics, write_csv)

STATIC_PHASES = {
    "DS3": (("pre_onset", 0., 118.9), ("transition", 118.9, 195.), ("established", 195., None)),
    "DS4": (("pre_onset", 0., 113.8), ("transition", 113.8, 225.), ("established", 225., None)),
    "DS7": (("pre_onset_replay", 0., 110.), ("transition", 110., 150.), ("established", 150., None)),
    "DS8": (("pre_onset_replay", 0., 110.), ("transition", 110., 150.), ("established", 150., None)),
}


def _ranges(scenario, end_s):
    if scenario not in STATIC_PHASES:
        return (("diagnostic", 0., end_s),)
    return tuple((name, start, end_s if final is None else min(final, end_s)) for name, start, final in STATIC_PHASES[scenario] if start < end_s)


_CLEAN_CONTRAST_FILES = ("clean_only_report.json", "clean_ablation_report.json",
                         "clean_a5_report.json", "clean_reproduction_evidence.json")


def load_clean_contrast_rows(root, identities):
    """Load the frozen cleanStatic negative cell from byte-pinned clean evidence."""
    root = Path(root).resolve()
    by_path = {str(Path(row.get("path", "")).resolve()): row for row in identities if isinstance(row, dict)}
    documents, source = {}, {}
    for name in _CLEAN_CONTRAST_FILES:
        path = (root / name).resolve(); identity = by_path.get(str(path))
        if identity is None or identity.get("sha256") != sha256_file(path) or identity.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"clean contrast identity mismatch: {name}")
        documents[name] = json.loads(path.read_text())
        source[name] = {"sha256": identity["sha256"], "size_bytes": identity["size_bytes"]}
    clean, ablation, a5, reproduction = (documents[name] for name in _CLEAN_CONTRAST_FILES)
    statuses = ((clean, "CLEAN_ONLY_PASS"), (ablation, "CLEAN_ABLATIONS_PASS"), (a5, "CLEAN_A5_PASS"))
    if any(doc.get("run_status") != status or doc.get("attack_access_count") != 0 or
           doc.get("protected_attack_rows_read") is not False for doc, status in statuses):
        raise ValueError("clean contrast source status/identity mismatch")
    try:
        method_rows = {"Full": clean["scores"]["Full_holdout"], "A1": clean["scores"]["A1_holdout"],
                       "A2": ablation["methods"]["A2"]["holdout"], "A5": a5["holdout"]}
    except (KeyError, TypeError):
        raise ValueError("clean contrast cell is missing") from None
    expected_count = reproduction.get("counts", {}).get("clean_contrast_holdout_windows")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("clean contrast expected count binding is absent")
    if any(len(rows) != expected_count for rows in method_rows.values()):
        raise ValueError("clean contrast cell count mismatch")
    source["clean_reproduction_evidence.json"]["expected_holdout_windows"] = expected_count
    result = []
    for method, rows in method_rows.items():
        for raw in rows:
            row = dict(raw)
            try:
                row["prns"] = tuple(map(int, row["prns"]))
                row["epoch_ids"] = tuple(map(int, row["epoch_ids"]))
                row["epoch_prn_support"] = tuple((int(epoch), tuple(map(int, prns)))
                                                 for epoch, prns in row["epoch_prn_support"])
                row["score"] = float(row["score"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("clean contrast cell support is malformed") from None
            result.append({**row, "scenario": "cleanStatic", "family": "cleanStatic", "phase": "holdout",
                           "method": method, "label": False, "phase_start_s": 350., "phase_end_s": 470.})
    return result, source


def validate_clean_contrast_preaccess(root, identities):
    """Authenticate the complete clean contrast before a protected attempt exists."""
    rows, source = load_clean_contrast_rows(root, identities)
    methods = {row["method"] for row in rows}
    if methods != {"Full", "A1", "A2", "A5"}:
        raise ValueError("clean contrast method set mismatch")
    return {"status": "PASS", "rows": len(rows), "source": source}


def _native_support(row):
    try:
        epochs = tuple(map(int, row["epoch_ids"]))
        prns = tuple(map(int, row["prns"]))
        support = tuple((int(epoch), tuple(map(int, values)))
                        for epoch, values in row["epoch_prn_support"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("protected method native support is malformed") from None
    if (not epochs or tuple(epoch for epoch, _ in support) != epochs or
            epochs != tuple(sorted(set(epochs))) or
            any(not values or values != tuple(sorted(set(values))) for _, values in support)):
        raise ValueError("protected method native support is malformed")
    union = tuple(sorted(set().union(*(set(values) for _, values in support))))
    if prns != union:
        raise ValueError("protected method native support is malformed")
    return epochs, prns, support


def integrate_protected_b0(methods, b0_rows, *, score_column):
    """Attach A0/B0 only when every Full event has exact native B0 support."""
    full_rows = list(methods.get("Full", ()))
    b0_rows = list(b0_rows)
    if not full_rows or not b0_rows:
        raise ValueError("protected B0/Full exact support is empty")
    full_by = {}
    for row in full_rows:
        key = (row.get("phase"), float(row["window_start_s"]))
        if key in full_by:
            raise ValueError("duplicate Full window on protected exact support")
        _native_support(row)
        full_by[key] = row
    grouped = {}
    seen_prn = set()
    for row in b0_rows:
        if score_column not in row:
            raise ValueError("protected B0 score column is absent")
        try:
            score = float(row[score_column])
            key = (row.get("phase"), float(row["window_start_s"]))
        except (TypeError, ValueError):
            raise ValueError("protected B0 join/score is malformed") from None
        if not np.isfinite(score):
            raise ValueError("protected B0 score is nonfinite")
        if "prn" in row:
            try:
                prn = int(str(row["prn"]).lstrip("Gg"))
                epochs = tuple(map(int, row["epoch_ids"]))
                support = tuple((int(epoch), tuple(map(int, values)))
                                for epoch, values in row["epoch_prn_support"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("protected B0 native support is malformed") from None
            identity = (*key, prn)
            if identity in seen_prn:
                raise ValueError("duplicate B0 window/PRN scientific row")
            seen_prn.add(identity)
            if (not epochs or tuple(epoch for epoch, _ in support) != epochs or
                    any(values != (prn,) for _, values in support)):
                raise ValueError("protected B0 native support is malformed")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(full_by):
        raise ValueError("protected B0/Full unsupported join or partial support")
    a0 = []
    for key in sorted(full_by, key=lambda value: (str(value[0]), value[1])):
        full, group = full_by[key], grouped[key]
        if any(("prn" in row) != ("prn" in group[0]) for row in group):
            raise ValueError("protected B0 support representation is mixed")
        if "prn" in group[0]:
            combined = {}
            for row in group:
                for epoch, values in row["epoch_prn_support"]:
                    combined.setdefault(int(epoch), set()).update(map(int, values))
            support = tuple((epoch, tuple(sorted(values))) for epoch, values in sorted(combined.items()))
            score = float(np.mean([float(row[score_column]) for row in group]))
        else:
            if len(group) != 1:
                raise ValueError("duplicate aggregated B0 scientific row")
            support = tuple((int(epoch), tuple(map(int, values)))
                            for epoch, values in group[0]["epoch_prn_support"])
            score = float(group[0][score_column])
        _, _, full_support = _native_support(full)
        if support != full_support:
            raise ValueError("protected B0/Full exact native support mismatch")
        a0.append({"window_start_s": float(full["window_start_s"]),
                   "availability_s": float(full["availability_s"]), "score": score,
                   "prns": list(map(int, full["prns"])), "epoch_ids": tuple(map(int, full["epoch_ids"])),
                   "epoch_prn_support": full_support,
                   **({"phase": full["phase"]} if "phase" in full else {})})
    if not a0:
        raise ValueError("protected A0 has zero usable windows")
    return {**methods, "A0": a0}


def validate_protected_method_support(methods, *, required_phases):
    """Require every mandatory method/phase on Full's exact native support."""
    expected = ("A0", "A1", "A2", "A3", "A4", "A5", "Full")
    if set(methods) != set(expected):
        raise ValueError("protected mandatory method set is incomplete")
    phases = tuple(required_phases)
    if not phases or len(phases) != len(set(phases)):
        raise ValueError("protected mandatory phase set is invalid")
    supports = {}; counts = {}
    for method in expected:
        rows = list(methods[method]); by_phase = {phase: set() for phase in phases}
        for row in rows:
            phase = row.get("phase")
            if phase not in by_phase:
                raise ValueError(f"{method} emitted an unsupported protected phase")
            epochs, prns, support = _native_support(row)
            identity = (float(row["window_start_s"]), float(row["availability_s"]),
                        epochs, prns, support)
            if identity in by_phase[phase]:
                raise ValueError(f"duplicate {method} protected support row")
            by_phase[phase].add(identity)
        if any(not by_phase[phase] for phase in phases):
            raise ValueError(f"{method} silently empty on a mandatory protected phase")
        supports[method] = by_phase
        counts[method] = {phase: len(by_phase[phase]) for phase in phases}
    for method in expected[:-1]:
        if supports[method] != supports["Full"]:
            raise ValueError(f"{method}/Full exact native support mismatch")
    return {"methods": list(expected), "phase_counts": counts}


def _load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    if spec.loader is None: raise ImportError(path)
    spec.loader.exec_module(module)
    return module


def score_protected_b0(*, tracking_paths, gate, scenario, roles, methods, thresholds,
                       artifact, repo):
    """Execute frozen B0 and bind its event scores to exact Full support."""
    pd = __import__("pandas")
    work = artifact / "b0_protected_recomputed" / scenario
    work.mkdir(parents=True, exist_ok=True)
    node = build_protected_scheduled_node_table(tracking_paths, gate=gate, scenario=scenario, roles=roles)
    node_path = work / "scheduled_node_windows.csv"; node.to_csv(node_path, index=False)
    scorer = _load_script(f"gcspo_b0_scorer_{scenario}", repo / "scripts/score_texbat_prn_node_gru.py")
    model_dir = repo / "artifacts/ai_morph_gru_cleanStatic_q70_frame"
    scorer.score_node_csv(node_path, model_dir, work, scenario, onset_s=None,
                          output_prefix="gcspo_b0", dataset_prefix="TEXBAT")
    prn_path, _, _ = scorer.score_output_paths(work, scenario, "gcspo_b0")
    prn = pd.read_csv(prn_path)
    support = node[["run_id", "prn", "window_start_s", "phase", "epoch_ids_json"]]
    prn = prn.merge(support, on=["run_id", "prn", "window_start_s"], validate="one_to_one")
    gate_module = _load_script(f"gcspo_b0_gate_{scenario}", repo / "scripts/eval_btail_support_gate.py")
    events = gate_module.build_event_scores(prn, thresholds["A0_B0"]["node_thresholds"], alpha=.75)
    event_score = {(str(row.run_id), float(row.window_start_s)): float(getattr(row, gate_module.FINAL_SCORE))
                   for row in events.itertuples(index=False)}
    rows = []
    for row in prn.itertuples(index=False):
        epochs = tuple(map(int, json.loads(row.epoch_ids_json)))
        numeric_prn = int(str(row.prn).lstrip("Gg"))
        rows.append({"phase": row.phase, "window_start_s": float(row.window_start_s),
                     "availability_s": float(row.window_start_s) + 1., "prn": numeric_prn,
                     "event_score": event_score[(str(row.run_id), float(row.window_start_s))],
                     "epoch_ids": epochs,
                     "epoch_prn_support": tuple((epoch, (numeric_prn,)) for epoch in epochs)})
    return integrate_protected_b0(methods, rows, score_column="event_score")


def _method_rows(data, model, whitener, gamma, geometry, a2_loading, lambdas, ranges,
                 validated_rows):
    validated_rows = set(validated_rows)
    result = {name: [] for name in ("A1", "A2", "A3", "A4", "A5", "Full")}
    for phase, start, end in ranges:
        if end - start < 1.2:
            continue
        method_sets = {
            "A1": a1_role_scores(data, model, whitener, start, end),
            "A2": score_a2_terms(role_a2_terms(data, model, whitener, gamma, a2_loading, start, end), lambdas["A2"]),
            "A3": score_full_terms(role_full_terms(data, model, whitener, gamma,
                    GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"],
                                  validated_rows & {"code_error_chips", "pll_phase_error_cycles"}), start, end), lambdas["Full"]),
            "A4": score_full_terms(role_full_terms(data, model, whitener, gamma,
                    GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"],
                                  validated_rows & {"carrier_doppler_hz", "code_frequency_offset_chips_s"}), start, end), lambdas["Full"]),
            "Full": score_full_terms(role_full_terms(data, model, whitener, gamma,
                    GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"], validated_rows), start, end), lambdas["Full"]),
            "A5": score_a5_terms(role_a5_terms(data, model, whitener, gamma, validated_rows, start, end), lambdas["A5"]),
        }
        for method, rows in method_sets.items():
            for row in rows:
                row.pop("state", None) if method != "Full" else None
                row["phase"] = phase
                required = {"prns", "epoch_ids", "epoch_prn_support"}
                if not required <= set(row):
                    raise RuntimeError(f"{method} scorer omitted authenticated actual support")
            result[method].extend(rows)
    return result


class _RelationGeometry:
    def __init__(self, lookup): self.lookup = lookup
    def loading(self, epoch, prn): return self.lookup.get((int(epoch), int(prn)))


def _constant_mask_segments(epochs, prns):
    by_epoch = {int(epoch): tuple(sorted(map(int, prns[epochs == epoch]))) for epoch in np.unique(epochs)}
    ordered = sorted(by_epoch); segments = []; start = 0
    for index in range(1, len(ordered) + 1):
        if index == len(ordered) or ordered[index] != ordered[index - 1] + 1 or by_epoch[ordered[index]] != by_epoch[ordered[index - 1]]:
            segments.append((ordered[start:index], by_epoch[ordered[index - 1]])); start = index
    return segments


def _relation_rows(data, model, whitener, gamma, geometry, smoothness, scenario, phase, start_s, end_s):
    epochs, prns, residual, _ = residual_table(data, model, whitener, start_s, end_s)
    result = []
    for segment_number, (segment_epochs, mask) in enumerate(_constant_mask_segments(epochs, prns)):
        if len(segment_epochs) < 50 or len(mask) < 4: continue
        residual_lookup = {(int(epoch), int(prn)): value for epoch, prn, value in zip(epochs, prns, residual)}
        cube = np.stack([[residual_lookup[(epoch, prn)] for epoch in segment_epochs] for prn in mask])
        los_cube = np.empty((len(mask), len(segment_epochs), 3)); loading_lookup = {}
        available = True
        for pi, prn in enumerate(mask):
            for ti, epoch in enumerate(segment_epochs):
                physical = geometry.loading(epoch, prn)
                if physical is None: available = False; break
                los_cube[pi, ti] = physical[0]; loading_lookup[(epoch, prn)] = physical
            if not available: break
        if not available: continue
        transformed = execute_relation_destructions(cube, los_cube, np.asarray(mask), scenario=scenario, phase=phase,
                                                     segment_id=f"segment-{segment_number}")
        flat_epochs = np.repeat(np.asarray(segment_epochs, np.int64), len(mask))
        flat_prns = np.tile(np.asarray(mask, np.int64), len(segment_epochs))
        original_z = whitener.transform(np.transpose(cube, (1, 0, 2)).reshape(-1, 10))
        shifted_z = whitener.transform(np.transpose(transformed["shifted_residual"], (1, 0, 2)).reshape(-1, 10))
        permutation = np.asarray(transformed["los_permutation"], int)
        shuffled_lookup = {}
        for target_index, target_prn in enumerate(mask):
            source_prn = mask[int(permutation[target_index])]
            for epoch in segment_epochs:
                shuffled_lookup[(epoch, target_prn)] = loading_lookup[(epoch, source_prn)]
        segment_start = max(start_s, segment_epochs[0] * EPOCH_S)
        segment_end = min(end_s, (segment_epochs[-1] + 1) * EPOCH_S)
        variants = {
            "Full": (original_z, _RelationGeometry(loading_lookup)),
            "LOS_SHUFFLE": (original_z, _RelationGeometry(shuffled_lookup)),
            "PER_PRN_TEMPORAL_SHIFT": (shifted_z, _RelationGeometry(loading_lookup)),
        }
        for method, (z_values, relation_geometry) in variants.items():
            terms = role_full_terms_from_z(flat_epochs, flat_prns, z_values, model, gamma, relation_geometry, segment_start, segment_end)
            for row in score_full_terms(terms, smoothness):
                result.append({**row, "method": method, "phase": phase, "prns": list(mask),
                               "segment_id": f"segment-{segment_number}",
                               "transform_seed": transformed["seeds"].get(method),
                               "preservation": transformed["preservation"].get(method, True)})
    return result


def run_one_shot(*, artifact_dir, repo_root, inventory, gate, manifest_identities, clean_identities,
                 capabilities):
    artifact = Path(artifact_dir); repo = Path(repo_root)
    normal = json.loads((artifact / "normal_model_summary.json").read_text())
    thresholds = json.loads((artifact / "thresholds.json").read_text())
    a2_doc = json.loads((artifact / "clean_ablation_report.json").read_text())
    model, whitener, gamma = reconstruct_normal_model(normal)
    validated_rows = set(normal.get("validated_rows", ()))
    if not validated_rows:
        raise RuntimeError("source-verified field inventory is absent from the frozen normal model")
    eps = {int(key): float(value) for key, value in normal["normalization_epsilon_by_prn"].items()}
    lambdas = {"Full": float(normal["lambda_selected"]), "A2": float(a2_doc["methods"]["A2"]["lambda"]),
               "A5": float(json.loads((artifact / "clean_a5_report.json").read_text())["lambda"])}
    scenario_inventory = {row["id"]: row for row in inventory["scenario_inventory"]}
    score_rows, state_rows, scenario_rows, relation_rows = [], [], [], []
    unavailable = [{"scenario": scenario, **row}
                   for scenario, row in sorted(capabilities["unavailable"].items())]
    available_scenarios = tuple(sorted(capabilities["available"]))
    for scenario in available_scenarios:
        identity = manifest_identities[scenario]; manifest_path = Path(identity["path"])
        binding = capabilities["available"][scenario]["manifest_binding"]
        gate.register_pinned(manifest_path, expected_sha256=identity["sha256"], expected_size=identity["size_bytes"],
                             kind="RECEIVER_MANIFEST", preclaim_dev=binding["dev"],
                             preclaim_ino=binding["ino"])
        gate.read_json(manifest_path, scenario=scenario, phase="all_frozen_phases",
                       purpose="receiver root manifest identity")
        sidecar = capabilities["available"][scenario]["sidecar"]
        child_paths = list(gate.register_sidecar_children(manifest_path, sidecar["children"]))
        tracking_paths = sorted(path for path in child_paths if path.name.startswith("epl_tracking_ch_") and path.suffix == ".mat")
        observables = [path for path in child_paths if path.name == "observables.mat"]
        nmea = [path for path in child_paths if path.suffix.lower() == ".nmea"]
        ephemeris = [path for path in child_paths if "ephemeris" in path.name.lower() and path.suffix.lower() == ".xml"]
        if not tracking_paths or len(observables) != 1 or len(nmea) != 1 or len(ephemeris) != 1:
            raise RuntimeError(f"authenticated receiver manifest is incomplete: {scenario}")
        data = load_receiver_tracking(tracking_paths, epsilons=eps, gate=gate, scenario=scenario)
        end_s = float((data.epoch.max() + 1) * .02)
        ranges = _ranges(scenario, end_s)
        phase_bounds = {name: (start, end) for name, start, end in ranges}
        try:
            geometry = protected_geometry_preflight(gate=gate, observables_path=observables[0], nmea_path=nmea[0],
                                                    ephemeris_path=ephemeris[0], scenario=scenario, tracked_prns=data.prn)
        except Exception as exc:
            unavailable.append({"scenario": scenario, "reason": "UNAVAILABLE_GEOMETRY", "detail": str(exc)}); continue
        methods = _method_rows(data, model, whitener, gamma, geometry, np.asarray(a2_doc["methods"]["A2"]["loading"]), lambdas, ranges,
                               validated_rows)
        methods = score_protected_b0(tracking_paths=tracking_paths, gate=gate, scenario=scenario,
                                     roles={name: (start, end) for name, start, end in ranges},
                                     methods=methods, thresholds=thresholds, artifact=artifact, repo=repo)
        validate_protected_method_support(
            methods, required_phases=tuple(name for name, _start, _end in ranges))
        relation_geometry = GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"], validated_rows)
        for relation_phase, relation_start, relation_end in ranges:
            if relation_phase != "pre_onset_replay" and relation_end - relation_start >= 1.2:
                relation_rows.extend({**row, "scenario": scenario, "label": relation_phase in {"transition", "established"},
                                      "phase_start_s": relation_start, "phase_end_s": relation_end,
                                      "epoch_ids": tuple(range(round(row["window_start_s"] / EPOCH_S), round(row["availability_s"] / EPOCH_S))),
                                      "epoch_prn_support": tuple((epoch, tuple(row["prns"])) for epoch in
                                                                 range(round(row["window_start_s"] / EPOCH_S), round(row["availability_s"] / EPOCH_S)))} for row in
                                     _relation_rows(data, model, whitener, gamma, relation_geometry, lambdas["Full"],
                                                    scenario, relation_phase, relation_start, relation_end))
        for method, rows in methods.items():
            threshold_key = "A0_B0" if method == "A0" else method
            q99 = float(thresholds[threshold_key]["q99"])
            for row in rows:
                record = {"scenario": scenario, "family": "DS7_DS8" if scenario in {"DS7", "DS8"} else scenario,
                          "phase": row["phase"], "method": method, "window_start_s": row["window_start_s"],
                          "availability_s": row["availability_s"], "score": row["score"], "threshold_q99": q99,
                          "alarm_q99": bool(row["score"] > q99), "tracked_n": len(row.get("prns", [])) or None,
                          "effective_dof": row.get("effective_dof"), "penalty": row.get("penalty"),
                          "likelihood_improvement_twice": row.get("likelihood_improvement_twice"),
                          "prns": tuple(row.get("prns", ())), "epoch_ids": tuple(row.get("epoch_ids", ())),
                          "epoch_prn_support": tuple(row.get("epoch_prn_support", ())),
                          "label": row["phase"] in {"transition", "established"},
                          "phase_start_s": phase_bounds[row["phase"]][0], "phase_end_s": phase_bounds[row["phase"]][1],
                          "prns_json": json.dumps(list(row.get("prns", ())), separators=(",", ":")),
                          "epoch_ids_json": json.dumps(list(row.get("epoch_ids", ())), separators=(",", ":")),
                          "epoch_prn_support_json": json.dumps(list(row.get("epoch_prn_support", ())), separators=(",", ":"))}
                score_rows.append(record)
                if method == "Full" and "state" in row:
                    state = np.asarray(row["state"], float).reshape(-1, 8)
                    for index, vector in enumerate(state):
                        state_rows.append({"scenario": scenario, "phase": row["phase"], "window_start_s": row["window_start_s"],
                                           "epoch_offset": index, **{f"state_{j}": value for j, value in enumerate(vector)}})
        for method, rows in methods.items():
            pre = [row["score"] for row in rows if row["phase"].startswith("pre_")]
            attack = [row["score"] for row in rows if row["phase"] in {"transition", "established"}]
            metric = discrimination_metrics(pre, attack)
            alarm = score_metrics(rows, threshold=float(thresholds["A0_B0" if method == "A0" else method]["q99"]))
            scenario_rows.append({"scenario": scenario, "method": method, **metric, **alarm,
                                  "pre_onset_fpr": float(np.mean(np.asarray(pre) > float(thresholds["A0_B0" if method == "A0" else method]["q99"]))) if pre else None})
    optional = normal.get("optional_method_capabilities", {})
    for method in ("M1", "Fixed9", "GCMR"):
        capability = optional.get(method, {})
        if capability.get("status") != "AVAILABLE":
            unavailable.append({"method": method, "status": capability.get("status", "UNAVAILABLE"),
                                "reason": capability.get("reason", "AUTHENTICATED_INPUT_OR_CHECKPOINT_ABSENT")})
    score_fields = ["scenario", "family", "phase", "method", "window_start_s", "availability_s", "score", "threshold_q99",
                    "alarm_q99", "tracked_n", "effective_dof", "penalty", "likelihood_improvement_twice", "prns_json", "epoch_ids_json", "epoch_prn_support_json",
                    "label", "phase_start_s", "phase_end_s"]
    write_csv(artifact / "per_epoch_scores.csv", score_rows, score_fields)
    write_csv(artifact / "shared_state_estimates.csv", state_rows,
              ["scenario", "phase", "window_start_s", "epoch_offset", *[f"state_{j}" for j in range(8)]])
    write_csv(artifact / "scenario_metrics.csv", scenario_rows,
              ["scenario", "method", "roc_auc", "low_fpr_pauc", "pr_auc", "windows", "alarm_ratio", "persistent_alarm_ratio", "first_persistent_alarm_s", "pre_onset_fpr"])
    full_by = {row["scenario"]: row for row in scenario_rows if row["method"] == "Full"}
    clean_fpr = float(json.loads((artifact / "clean_only_report.json").read_text())["holdout_fpr"]["Full"]["q99"])
    external_fpr = {scenario: full_by[scenario]["pre_onset_fpr"] for scenario in ("DS3", "DS4")
                    if scenario in full_by and full_by[scenario]["pre_onset_fpr"] is not None}
    clean_contrast_rows, clean_contrast_source = load_clean_contrast_rows(artifact, clean_identities)
    bootstrap_source = clean_contrast_rows + [row for row in score_rows if row["scenario"] in {"DS3", "DS4", "DS7", "DS8"}
                                               and row["phase"] != "pre_onset_replay"]
    bootstrap_reports = {}
    for comparator in ("A1", "A2"):
        bootstrap_reports[f"Full-{comparator}"] = paired_block_bootstrap(bootstrap_source, "Full", comparator)
    required_relation = sorted(set(available_scenarios) & {"DS3", "DS7", "DS8"})
    relation_reports = {"policy": RELATION_POLICY, "required_available_scenarios": required_relation,
                        "scenario_results": {}}
    for scenario, policy in RELATION_POLICY.items():
        if scenario not in available_scenarios:
            disposition = capabilities["unavailable"].get(scenario, {})
            relation_reports["scenario_results"][scenario] = {
                "status": disposition.get("status", "UNAVAILABLE"), "primary": policy["primary"],
                "mandatory": False, "reason": disposition.get("reason", "CAPABILITY_UNAVAILABLE")}
            continue
        scenario_relation = [row for row in relation_rows if row["scenario"] == scenario]
        established = any(row["phase"] == "established" for row in scenario_relation)
        if policy["requires_established"] and not established:
            relation_reports["scenario_results"][scenario] = {
                "status": "LIMITED_TRANSITION_ONLY", "primary": policy["primary"],
                "mandatory": False, "reason": "AUTHENTICATED_ESTABLISHED_PULL_OFF_COVERAGE_ABSENT"}
            continue
        primary = policy["primary"]
        try:
            report = paired_score_loss_bootstrap(scenario_relation, "Full", primary)
        except ValueError as exc:
            relation_reports["scenario_results"][scenario] = {
                "status": "UNAVAILABLE", "primary": primary, "mandatory": scenario in {"DS3", "DS7", "DS8"},
                "reason": str(exc)}
            continue
        relation_reports["scenario_results"][scenario] = {
            "status": "AVAILABLE", "primary": primary, "mandatory": scenario in {"DS3", "DS7", "DS8"},
            "lcb": report["lcb_95"], "interval_95": report["interval_95"],
            "median_relative_loss": report["median_relative_loss"], "replicates": report["replicates"],
            "contrast": report["contrast"]}
    validate_mandatory_relation_evidence(relation_reports, required_scenarios=required_relation)
    persistence = {}
    per_scenario_persistence = {}
    for scenario in tuple(name for name in ("DS3", "DS7", "DS8") if name in available_scenarios):
        rows = sorted([row for row in score_rows if row["scenario"] == scenario and row["method"] == "Full"
                       and row["phase"] == "established"], key=lambda row: row["availability_s"])
        if not rows: raise RuntimeError(f"mandatory established persistence unavailable: {scenario}")
        flags = scheduled_persistence(rows, threshold=float(thresholds["Full"]["q99"]))
        first = next((row["availability_s"] for row, flag in zip(rows, flags) if flag), None)
        if first is None: delay = float("inf")
        else: delay = float(first) - float(rows[0]["phase_start_s"])
        per_scenario_persistence[scenario] = {"ratio": float(np.mean(flags)), "delay_s": delay}
    if "DS3" not in per_scenario_persistence:
        raise RuntimeError("mandatory DS3 persistence unavailable")
    family_members = [name for name in ("DS7", "DS8") if name in per_scenario_persistence]
    if not family_members:
        raise RuntimeError("mandatory DS7/DS8 family persistence unavailable")
    persistence["DS3"] = per_scenario_persistence["DS3"]
    persistence["DS7_DS8"] = {"ratio": min(per_scenario_persistence[name]["ratio"] for name in family_members),
                               "delay_s": max(per_scenario_persistence[name]["delay_s"] for name in family_members),
                               "available_members": family_members}
    controls = json.loads((artifact / "physical_controls.json").read_text())
    control_evidence = controls.get("results")
    if not isinstance(control_evidence, list) or not control_evidence:
        raise RuntimeError("mandatory generated control score evidence is absent")
    shared_support = exact_contrast_support(bootstrap_source, ("Full", "A5"))
    if not shared_support: raise RuntimeError("Full/A5 exact support is absent")
    def paired_pauc(method):
        rows = [{**row, "score": row["scores"][method],
                 "logical_cell": f"{'positive' if row['label'] else 'negative'}:{'DS7_DS8' if row['scenario'] in {'DS7', 'DS8'} else row['scenario']}:{row['phase']}"}
                for row in shared_support]
        return scenario_phase_balanced_pauc(rows)
    support_keys = {(row["scenario"], row["phase"], row["availability_s"], tuple(row["prns"])) for row in shared_support}
    full_edf = [row["effective_dof"] for row in bootstrap_source if row["method"] == "Full" and
                (row["scenario"], row["phase"], row["availability_s"], tuple(row["prns"])) in support_keys]
    a5_edf = [row["effective_dof"] for row in bootstrap_source if row["method"] == "A5" and
              (row["scenario"], row["phase"], row["availability_s"], tuple(row["prns"])) in support_keys]
    shared = {"full_pauc": paired_pauc("Full"), "a5_pauc": paired_pauc("A5"),
              "full_median_edf": float(np.median(full_edf)), "a5_median_edf": float(np.median(a5_edf))}
    b0_exact = exact_b0_full_contrast(score_rows, required_scenarios=required_relation)

    evidence = {"clean_holdout_fpr": clean_fpr, "clean_contrast_source": clean_contrast_source,
                "external_pre_fpr": external_fpr,
                "incremental_lcb": {name: row["lcb_95"] for name, row in bootstrap_reports.items()},
                "b0_exact_support": b0_exact, "destruction": relation_reports, "persistence": persistence, "controls": control_evidence, "shared": shared}
    gates = compute_scientific_gates(evidence); verdict = scientific_verdict(gates)
    write_csv(artifact / "ablation_metrics.csv", [{"contrast": row["id"], "status": row["status"]} for row in gates], ["contrast", "status"])
    write_csv(artifact / "external_static_fpr.csv", [{"scenario": "cleanStatic_holdout", "fpr_q99": clean_fpr},
              *[{"scenario": scenario, "fpr_q99": value} for scenario, value in sorted(external_fpr.items())]], ["scenario", "fpr_q99"])
    interval_rows = [{"contrast": name, "replicates": report["replicates"], "lcb_95": report["lcb_95"],
                      "ci_low": report["interval_95"][0], "ci_high": report["interval_95"][1]}
                     for name, report in sorted(bootstrap_reports.items())]
    interval_rows += [{"contrast": f"Full-{name}", "replicates": report["replicates"], "lcb_95": report["lcb"],
                       "ci_low": report["interval_95"][0], "ci_high": report["interval_95"][1]}
                      for name, report in sorted(relation_reports["scenario_results"].items())
                      if report.get("status") == "AVAILABLE"]
    write_csv(artifact / "bootstrap_intervals.csv", interval_rows, ["contrast", "replicates", "lcb_95", "ci_low", "ci_high"])
    canonical_write_json(artifact / "relation_destruction_metrics.json",
                         {"schema": "gnss-doppler-lab.gcspo-stage0.relation-destruction.v2",
                          "results": relation_reports, "transforms_executed": True,
                          "preservation_assertions": all(row.get("preservation") is True for row in relation_rows),
                          "seeds": sorted({row["transform_seed"] for row in relation_rows if row.get("transform_seed") is not None}),
                          "numpy_version": np.__version__,
                          "rows": [{key: value for key, value in row.items() if key not in {"state", "terms"}}
                                   for row in relation_rows]})
    controls["alarm_behavior_evaluated"] = True; controls["gate_status"] = next(row["status"] for row in gates if row["id"] == "G5_CONTROLS")
    canonical_write_json(artifact / "physical_controls.json", controls)
    plots = artifact / "plots"; plots.mkdir(exist_ok=True)
    write_csv(plots / "full_score_numeric_sidecar.csv", [row for row in score_rows if row["method"] == "Full"], score_fields)
    canonical_write_json(artifact / "final_verdict.json", {"schema": "gnss-doppler-lab.gcspo-stage0.final-verdict.v2",
                         "verdict": verdict, "scientific_status": "VALID_SCIENCE", "protected_run_count": 1,
                         "gates": gates, "evidence": evidence, "unavailable": unavailable})
    return verdict
