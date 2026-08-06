"""R12 cleanStatic raw-IQ/tracker alignment contracts and small pure helpers."""
from __future__ import annotations
from collections import Counter

FS = 25_000_000
MIN_VALIDATION_EPOCHS_PER_PRN = 50
# Retained cleanStatic fixture has 19 usable PRNs; 19 * 50 guarantees the per-PRN validation floor.
DEFAULT_VALIDATION_EPOCHS = 950


def aux_samples_to_chips(aux, code_freq, fs=FS):
    return float(aux) * float(code_freq) / float(fs)


def apply_origin(sample, offset):
    return int(sample) + int(offset)


def interval_bounds(row, interval):
    """Return actual tracker-row consecutive sample bounds; never assume 25,000."""
    if interval == "prev_to_cur":
        return int(row["previous_sample_count"]), int(row["sample_count"])
    if interval == "cur_to_next":
        return int(row["sample_count"]), int(row["next_sample_count"])
    raise ValueError("only consecutive source-semantic intervals are permitted")


def alignment_candidates():
    """Pre-registered source-semantic candidate family (24 hypotheses)."""
    return [
        {
            "name": f"{interval}_aux_{aux}_rem_{rem:+d}_carrier_{carrier:+d}_replica_{replica:+d}",
            "interval": interval,
            "aux_row": aux,
            "remnant_sign": rem,
            "carrier_sign": carrier,
            "replica_sign": replica,
        }
        for interval in ("prev_to_cur", "cur_to_next")
        for aux in ("previous", "current", "next")
        for rem in (-1, 1)
        for carrier in (-1, 1)
        for replica in (1,)
    ]


def wide_grid():
    return {
        "delay_chips": [round(-1 + i * 0.125, 3) for i in range(17)],
        "doppler_hz": list(range(-250, 251, 50)),
    }


def dominant_fraction(prns):
    return max(Counter(prns).values()) / len(prns) if prns else 1.0


def roles_nonoverlap(intervals):
    """Only time-role intervals must not overlap; cross-PRN epoch overlap is allowed."""
    by_role = {}
    for row in intervals:
        by_role.setdefault(row["role"], []).append((row["start"], row["end"]))
    roles = sorted(by_role)
    for i, left in enumerate(roles):
        for right in roles[i + 1 :]:
            if any(a[0] < b[1] and b[0] < a[1] for a in by_role[left] for b in by_role[right]):
                return False
    return True


def select_role_stratified(rows, n):
    """Round-robin PRNs after time-first role assignment; never reuse a tracker row."""
    queues = {prn: [] for prn in sorted({row["prn"] for row in rows})}
    seen = set()
    for row in sorted(rows, key=lambda x: (x["sample_count"], x["prn"], x["tracker_row"])):
        key = (row.get("channel"), row.get("tracker_row"))
        if key not in seen:
            queues[row["prn"]].append(row)
            seen.add(key)
    selected = []
    while len(selected) < n and any(queues.values()):
        for prn in sorted(queues):
            if queues[prn] and len(selected) < n:
                selected.append(queues[prn].pop(0))
    return sorted(selected, key=lambda x: (x["sample_count"], x["prn"], x["tracker_row"]))


def center_stats(rows):
    n = len(rows)
    return {
        "exact_center_fraction": sum(r["peak_delay_offset_chips"] == 0 and r["peak_doppler_offset_hz"] == 0 for r in rows) / n if n else 0.0,
        "within_tolerance_fraction": sum(abs(r["peak_delay_offset_chips"]) <= 0.125 and abs(r["peak_doppler_offset_hz"]) <= 50 for r in rows) / n if n else 0.0,
        "boundary_fraction": sum(r.get("grid_boundary", False) for r in rows) / n if n else 1.0,
    }


def gate_alignment(s):
    """Fail closed: A2/A3 failures force selected_alignment to JSON null."""
    checks = {
        "A1_source_binding": s.get("binding") in ("exact_same_raw", "same_record_header_offset"),
        "A2_interval_alignment": s.get("n", 0) >= 800 and s.get("prn_count", 0) >= 8 and s.get("min_prn_epochs", 0) >= MIN_VALIDATION_EPOCHS_PER_PRN and s.get("dominant_fraction", 1) <= 0.2 and bool(s.get("consistent_time")),
        "A3_recovery": s.get("within_tolerance_fraction", 0) >= 0.95 and s.get("pooled_spearman", -1) >= 0.90 and s.get("median_prn_spearman", -1) >= 0.80 and s.get("boundary_fraction", 1) <= 0.05,
    }
    selected = s.get("candidate") if all(checks.values()) else None
    return {
        **{key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "selected_alignment": selected,
        "diagnostic_best_candidate": s.get("candidate"),
        "status": "PASS" if selected else "FAIL_CLOSED",
    }


def clean_only_guard(scenarios):
    if set(scenarios) != {"cleanStatic"}:
        raise ValueError("R12 accepts cleanStatic only")
    return True
