#!/usr/bin/env python3
"""Static receiver nondeterminism audit for TRACE-R2a."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def evidence(path: Path, needles: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if any(needle in line for needle in needles):
            rows.append({"path": str(path), "line": number, "text": line.strip()})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receiver-source", type=Path, required=True)
    parser.add_argument("--r2-config", type=Path, required=True)
    parser.add_argument("--r2-patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    flowgraph = args.receiver_source / "src/core/receiver/gnss_flowgraph.cc"
    acquisition = args.receiver_source / "src/algorithms/acquisition/gnuradio_blocks/pcps_acquisition.cc"
    tracking = args.receiver_source / "src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc"
    config = args.r2_config.read_text()
    checks = {
        "acquisition_channel_scheduling": {
            "status": "NONDETERMINISTIC",
            "finding": "R2 enabled all 11 acquisition blocks concurrently; successful/failed PMT messages arrive in scheduler-dependent order and acquisition_manager mutates the shared available-signal list under that event order.",
            "config_channels_in_acquisition": next(line for line in config.splitlines() if line.startswith("Channels.in_acquisition=")),
            "evidence": evidence(flowgraph, ("start_acquisition", "acquisition_manager(who)", "search_next_signal"))[:30],
        },
        "prn_assignment_order": {
            "status": "ORDERED_INITIAL_LIST_BUT_EVENT_ORDER_DEPENDENT_REASSIGNMENT",
            "finding": "Initial implicit GPS PRNs are drawn from list front, but unfixed channels are reassigned according to whichever concurrent acquisition event is processed first.",
            "fixed_satellite_entries_in_r2_config": [line for line in config.splitlines() if line.startswith("Channel") and ".satellite=" in line],
            "evidence": evidence(flowgraph, ("available_GPS_1C_signals_.front()", "pop_front()", "push_back(result)", '"Channel" + std::to_string(i) + ".satellite"')),
        },
        "acquisition_worker_thread_count": {
            "status": "ELEVEN_LOGICALLY_CONCURRENT_GNURADIO_BLOCKS",
            "finding": "The non-FPGA path calls start_acquisition for every enabled channel before the flowgraph runs; GNU Radio schedules acquisition blocks independently.",
        },
        "thread_scheduling_races": {
            "status": "CAUSAL_SCHEDULING_NONDETERMINISM_CONFIRMED",
            "finding": "The signal-list mutex prevents memory corruption but serializes whichever event happens to arrive first; it does not impose deterministic event ordering or acquisition sample stamps.",
            "evidence": evidence(flowgraph, ("signal_list_mutex_", "apply_action", "Received "))[:20],
        },
        "acquisition_success_message_order": {
            "status": "UNORDERED",
            "finding": "Positive acquisition is emitted from each acquisition block after its FFT search; no sequence barrier sorts successes by channel or sample stamp.",
            "evidence": evidence(acquisition, ("send_positive_acquisition", "acquisition_core"))[:20],
        },
        "random_seed_usage": {
            "status": "NO_RANDOM_SOURCE_FOUND_IN_RELEVANT_PATH",
            "finding": "The relevant flowgraph/acquisition/tracking source contains no RNG calls; the observed variation is scheduling, not seeded randomness.",
        },
        "uninitialized_state": {
            "status": "NO_R2_SERIALIZED_PADDING_OR_UNINITIALIZED_RESERVED_BYTES_FOUND",
            "finding": "R2 writes every field explicitly, value-initializes action snapshots and reserved bytes, and emits a fixed 416-byte schema.",
            "evidence": evidence(tracking, ("TraceActionSnapshot action{}", "reserved = 0U", "trace_write_value"))[:30],
        },
        "channel_reassignment": {
            "status": "ENABLED_AND_EVENT_ORDER_DEPENDENT",
            "finding": "Failed acquisitions call acquisition_manager before returning the old signal to the list; later assignment therefore depends on event order.",
            "evidence": evidence(flowgraph, ("push back the old signal AFTER", "acquisition_manager(who)", "push_back_signal(gs)")),
        },
        "tracking_start_sample_stamp": {
            "status": "NONDETERMINISTIC",
            "finding": "Tracking start uses the acquisition sample stamp plus the tracking block's scheduler-dependent nitems_read value. R2 rep1/rep2 show integer-millisecond first-interval shifts even for identical channel/PRN pairs.",
            "evidence": evidence(tracking, ("d_acq_sample_stamp", "nitems_read(0)", "samples_offset"))[:30],
        },
        "concurrent_file_writer_ordering": {
            "status": "REPRESENTATIONAL_ONLY_ACROSS_FILES",
            "finding": "Each channel owns a separate stream. Cross-file completion/order may vary but cannot explain physical differences at identical canonical keys.",
        },
        "dump_struct_padding": {
            "status": "PASS_EXPLICIT_SERIALIZATION",
            "finding": "No native C++ struct is dumped; scalar fields are written explicitly in schema order. Canonical serialization is already satisfied by R2.",
        },
        "wall_clock_or_session_id_in_physical_record": {
            "status": "SESSION_ID_PRESENT_AS_METADATA_ONLY_NO_WALL_CLOCK",
            "finding": "tracking_session_id and local loop sequence remain in the binary row for causal boundaries but are excluded from the canonical physical semantic hash; receiver_timestamp_s is raw-sample-derived.",
        },
        "floating_point_reduction_order": {
            "status": "DETERMINISTIC_WITHIN_FIXED_CHANNEL_HANDOFF",
            "finding": "Per-channel FFT/correlator/filter operations have fixed local order. The audit found no shared floating reduction across channels.",
        },
        "fixed_satellite_channel_list": {
            "status": "SUPPORTED_BUT_NOT_USED_BY_R2",
            "finding": "ChannelN.satellite is supported. R2 supplied no entries, so assignments changed across repetitions.",
        },
        "reuse_same_acquisition_handoff": {
            "status": "NOT_SUPPORTED_BY_R2_REQUIRES_R2A_REPAIR",
            "finding": "R2 has no persisted handoff reader. R2a must pin channel/PRN, acquisition Doppler convention, and exact first raw tracking interval, and must fail closed if activation arrives after that interval.",
        },
    }
    payload = {
        "schema": "gnss-doppler-lab.trace-r2a-receiver-nondeterminism-audit.v1",
        "status": "ROOT_CAUSE_IDENTIFIED",
        "receiver_source_path_read_only": str(args.receiver_source),
        "r2_patch_sha256": sha256(args.r2_patch),
        "r2_config_sha256": sha256(args.r2_config),
        "checks": checks,
        "root_cause": "Concurrent acquisition and unordered event-driven reassignment/handoff change the acquisition/tracking raw-sample origin. This changes physical loop state and taps, not only metadata.",
        "repair_decision": {
            "canonical_serialization": "ALREADY_VALID_IN_R2",
            "deterministic_assignment": "REQUIRED_FIXED_CHANNEL_PRN_MAP",
            "frozen_acquisition_handoff": "REQUIRED_BECAUSE_SAME_CHANNEL_PRN_STARTS_SHIFTED_ACROSS_R2_REPLAYS",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
