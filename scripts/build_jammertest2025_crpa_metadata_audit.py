#!/usr/bin/env python3
"""Build the Jammertest 2025 CRPA metadata-only feasibility artifact."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from gnss_doppler_lab.jammertest_metadata_audit import (
    counter_records,
    infer_crpa_npy_layout,
    parse_crpa_rows,
    parse_lfs_pointer,
    sha256_file,
    write_json,
    write_manifest,
)


CLASS_NAMES = {
    0: "CW",
    1: "Sweep",
    2: "Prn",
    3: "Meac",
    4: "Spoof",
    5: "Chirp",
    6: "ChirpB",
    7: "Triang",
    8: "Meac,Prn",
    9: "Meac,Spoof",
    10: "Mod",
    11: "ChirpM",
    12: "ChirpMS",
    13: "FmS",
    14: "Chirp,Spoof",
    15: "Chirp,Prn",
    16: "Chirp,Prn,Triang",
}

DATASET_SHA = "3b778a12147ded5c86c3edfc586b5de6ae6a67d7"
PLAN_SHA = "1ab4ae055a291e69ea9476c88f3903b5cbd9bd64"
CRPA_OID = "d869fa20d552288002e4d2a5b6c5d1300083a6348c01a956cd6a34ff232e0a3f"
CRPA_SIZE = 1_398_308_992
TOTAL_SIZE = 360_208_806_569
VERDICT = "INCONCLUSIVE_SCHEMA_REQUIRES_ONE_BOUNDED_H5_SAMPLE"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_sha_map(root: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_file(root / name) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--zenodo-json", type=Path, required=True)
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    plan = args.plan_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if git(dataset, "rev-parse", "HEAD") != DATASET_SHA:
        raise SystemExit("dataset repository SHA mismatch")
    if git(plan, "rev-parse", "HEAD") != PLAN_SHA:
        raise SystemExit("transmission-plan repository SHA mismatch")

    zenodo = json.loads(args.zenodo_json.read_text())
    if zenodo["id"] != 21_332_689 or zenodo["doi"] != "10.5281/zenodo.21332689":
        raise SystemExit("Zenodo identity mismatch")

    pointer_paths = [dataset / "all_crpa_files.npy", *sorted(dataset.glob("dataset/*.h5.part-*"))]
    pointer_rows = []
    grouped_h5: dict[str, int] = defaultdict(int)
    for path in pointer_paths:
        pointer = parse_lfs_pointer(path.read_text())
        relative = path.relative_to(dataset).as_posix()
        is_crpa = relative == "all_crpa_files.npy"
        area_match = re.search(r"Sample_(Area[12])_", relative)
        area = "ALL_RELEASED_CRPA" if is_crpa else area_match.group(1)
        reconstructed = re.sub(r"\.part-[a-z]+$", "", relative)
        if not is_crpa:
            grouped_h5[reconstructed] += pointer.size
        pointer_rows.append(
            {
                "path": relative,
                "receiver": "CRPA" if is_crpa else "Innosense",
                "area": area,
                "band": "UNKNOWN_MIXED_WITHIN_OBJECT",
                "reconstructed_object": reconstructed,
                "oid_sha256": pointer.oid_sha256,
                "logical_size_bytes": pointer.size,
                "payload_present": "false",
            }
        )
    if len(pointer_rows) != 231 or sum(row["logical_size_bytes"] for row in pointer_rows) != TOTAL_SIZE:
        raise SystemExit("LFS inventory arithmetic mismatch")
    write_csv(
        output / "lfs_pointer_inventory.csv",
        [
            "path",
            "receiver",
            "area",
            "band",
            "reconstructed_object",
            "oid_sha256",
            "logical_size_bytes",
            "payload_present",
        ],
        pointer_rows,
    )

    crpa_rows = parse_crpa_rows(
        (
            (dataset / f"splits/{split}_crpa_{area}.txt", area, split)
            for area in (1, 2)
            for split in ("train", "test")
        )
    )
    if len(crpa_rows) != 36_186 or max(row["sample_index"] for row in crpa_rows) != 42_672:
        raise SystemExit("CRPA split inventory mismatch")
    if len({row["sample_index"] for row in crpa_rows}) != len(crpa_rows):
        raise SystemExit("duplicate CRPA split sample indices")

    innosense = {}
    for area in (1, 2):
        rows = []
        for split in ("train", "test"):
            for line in (dataset / f"splits/{split}_{area}.txt").read_text().splitlines():
                fields = line.split("\t")
                if len(fields) != 5:
                    raise SystemExit("unexpected Innosense split schema")
                rows.append(fields)
        innosense[str(area)] = {
            "rows": len(rows),
            "band_counts": dict(sorted(Counter(row[3] for row in rows).items())),
            "antenna_power_flag_counts": dict(sorted(Counter(row[4] for row in rows).items())),
        }

    area_bytes = {
        area: sum(row["logical_size_bytes"] for row in pointer_rows if row["receiver"] == "Innosense" and row["area"] == area)
        for area in ("Area1", "Area2")
    }
    disk = shutil.disk_usage(output)

    plan_names = [
        "README.md",
        "LICENSE",
        "equipment.json",
        "plan-monday-2025-09-15.json",
        "plan-tuesday-2025-09-16.json",
        "plan-wednesday-2025-09-17.json",
        "plan-thursday-2025-09-18.json",
        "plan-friday-2025-09-19.json",
        "testcatalog2025.json",
        "Transmissionplan.pdf",
        "Testcatalog.pdf",
    ]
    plan_events = []
    for name in plan_names[3:8]:
        obj = json.loads((plan / name).read_text())
        for location in obj["locations"]:
            for event in location["tests"]:
                plan_events.append({"source": name, "location": location, "event": event})

    source_binding = {
        "audit_date": "2026-08-23",
        "base_repository": {
            "url": "https://github.com/ljaehwan/gnss-doppler-lab.git",
            "branch": "origin/research/texbat-first-spoofing-model-design-audit",
            "commit": "88ef328eeead5053a3ba9c4bf7cc888a6de549fa",
        },
        "dataset": {
            "record_id": 21_332_689,
            "doi": "10.5281/zenodo.21332689",
            "title": zenodo["metadata"]["title"],
            "publication_date": zenodo["metadata"]["publication_date"],
            "version": zenodo["metadata"]["version"],
            "record_created": zenodo["created"],
            "record_updated": zenodo["updated"],
            "authors": zenodo["metadata"]["creators"],
            "institution": "Fraunhofer Institute for Integrated Circuits IIS, Nürnberg, Germany",
            "github_url": "https://github.com/FelixOtt94/FraunhoferIIS_Jammertest2025",
            "github_ref": "refs/tags/dataset",
            "github_commit": DATASET_SHA,
            "readme_sha256": sha256_file(dataset / "README.md"),
            "reader_sha256": sha256_file(dataset / "utilities_crpa.py"),
            "gitattributes_sha256": sha256_file(dataset / ".gitattributes"),
            "data_paper": {
                "title": "Analyzing and Characterizing Multi-Source Interference Effects at Jammertest Norway 2025",
                "arxiv": "2608.15819",
                "url": "https://arxiv.org/abs/2608.15819",
                "pdf_sha256": sha256_file(args.paper_pdf),
                "pdf_bytes": args.paper_pdf.stat().st_size,
            },
            "zenodo_archive": {
                "name": zenodo["files"][0]["key"],
                "bytes": zenodo["files"][0]["size"],
                "checksum": zenodo["files"][0]["checksum"],
                "contains": "ordinary Git snapshot and LFS pointer text, not LFS payloads",
            },
            "license_conflict": {
                "zenodo_structured_metadata": zenodo["metadata"]["license"]["id"],
                "repository_readme": "CC-BY-NC-SA-4.0",
                "status": "CONFLICT_REQUIRES_PUBLISHER_CLARIFICATION",
            },
        },
        "transmission_plan": {
            "url": "https://github.com/NPRA/jammertest-plan",
            "commit": PLAN_SHA,
            "commit_date": git(plan, "show", "-s", "--format=%aI", "HEAD"),
            "license": "MIT",
            "file_sha256": json_sha_map(plan, plan_names),
        },
    }
    write_json(output / "official_source_binding.json", source_binding)

    write_json(
        output / "repository_inventory.json",
        {
            "dataset_commit": DATASET_SHA,
            "dataset_tag": "dataset",
            "metadata_only_worktree_bytes": 41_000_000,
            "lfs_pointer_count": len(pointer_rows),
            "lfs_payload_file_count": 0,
            "split_files": {
                name: sum(1 for _ in (dataset / "splits" / name).open())
                for name in sorted(path.name for path in (dataset / "splits").glob("*.txt"))
            },
            "ordinary_code": ["README.md", "utilities_crpa.py", "utilities_innosense.py"],
            "crpa_storage": "single Git-LFS NPY object plus four ordinary-text split files",
            "innosense_storage": "86 HDF5 objects split across 230 Git-LFS part objects",
        },
    )

    write_json(
        output / "logical_size_summary.json",
        {
            "all_lfs_pointers": {"count": 231, "bytes": TOTAL_SIZE, "decimal_gb": TOTAL_SIZE / 1e9},
            "crpa": {
                "object_count": 1,
                "bytes": CRPA_SIZE,
                "gib": CRPA_SIZE / 2**30,
                "area_size_status": "ONE_SHARED_OBJECT; cannot download by area",
                "selected_split_snapshot_payload_estimate": {
                    "Area1": 22_756 * 32_768,
                    "Area2": 13_430 * 32_768,
                    "unselected_indices": 6_487 * 32_768,
                    "npy_header": 128,
                },
                "band_size": "UNKNOWN: CRPA split metadata has no band field",
            },
            "innosense": {
                "hdf5_object_count": len(grouped_h5),
                "part_pointer_count": 230,
                "bytes": sum(area_bytes.values()),
                "area_bytes": area_bytes,
                "band_size": "UNKNOWN: band is stored per raw HDF5 label and objects mix labels",
                "smallest_reconstructed_hdf5": min(grouped_h5.items(), key=lambda item: item[1]),
                "largest_reconstructed_hdf5": max(grouped_h5.items(), key=lambda item: item[1]),
            },
            "arithmetic_verified": True,
        },
    )

    layout = infer_crpa_npy_layout(CRPA_SIZE, 42_672)
    write_json(
        output / "crpa_schema_audit.json",
        {
            "array": "2x2 patch array",
            "stored_channel_count": 4,
            "channel_representation": "four complex channels exposed as 8 float I/Q channels by reader",
            "reader_transform": "complex array -> view(float32) -> (-1,1024,2) -> (1024,8)",
            "snapshot_shape_exposed_to_model": [8, 1024],
            "one_snapshot_candidate_physical_shape": [4, 1024],
            "npy_layout_inference": layout,
            "crpa_hdf5_dataset_name": "NOT_APPLICABLE: released CRPA payload is all_crpa_files.npy",
            "innosense_hdf5_dataset_names": ["/data", "/label"],
            "innosense_data_shape": ["records", 263120],
            "innosense_label_shape": ["records", 5],
            "innosense_data_dtype": "8-bit quantized interleaved I/Q; signedness not documented",
            "crpa_released_storage_dtype": "complex64 candidate inferred from reader and exact LFS size; header not opened",
            "crpa_original_adc_quantization": "UNKNOWN",
            "sample_rate": "not explicitly stated for CRPA; 1024 samples/10 us implies 102.4 MS/s only if duration is exact",
            "bandwidth_hz": 100_000_000,
            "snapshot_duration_us": 10,
            "band_labels": "UNKNOWN for CRPA split; repository prose says E1/E5a campaign coverage",
            "antenna_element_order": "UNKNOWN",
            "element_spacing": "UNKNOWN",
            "array_geometry_beyond_2x2": "UNKNOWN",
            "phase_calibration": "UNKNOWN",
            "cable_receiver_channel_phase_offsets": "UNKNOWN",
            "receiver_orientation": "UNKNOWN",
            "automatic_gain_control": False,
            "vga_metadata_for_crpa": "ABSENT from released CRPA split rows",
            "raw_or_beamformed": "reader exposes four per-channel complex streams; no beamforming step is documented, but raw-front-end status is not explicitly attested",
            "synchronous_sampling": "INTENDED/IMPLIED by array snapshot reader, not directly attested with clock/trigger specification",
            "relative_phase_preservation": "NOT DIRECTLY DOCUMENTED OR VALIDATED",
            "schema_gate": "FAIL_CLOSED",
        },
    )

    (output / "array_phase_coherence_evidence.md").write_text(
        "# Array phase-coherence evidence\n\n"
        "## Directly supported\n\n"
        "The official README/data paper calls the receiver a 2×2 patch CRPA, says it is quadrature sampled over 10 µs, and states an array/direction-finding use. The released reader takes one array item, exposes four complex streams as eight I/Q columns, and never beamforms or magnitude-reduces those columns. The LFS size is exactly consistent with 42,673 × 4 × 1024 complex64 values plus a 128-byte NPY header.\n\n"
        "## Not supported\n\n"
        "No released text defines element order, element coordinates/spacing, common clock/trigger, cable or RF-chain phase offsets, phase calibration, receiver orientation, or a phase-coherence acceptance measurement. The payload header was not opened. Therefore the size-consistent shape is an inference, and four-channel relative phase preservation is not direct evidence.\n\n"
        "## Gate\n\n"
        "The mandatory phase gate is closed. `READY_FOR_CRPA_MINIMAL_SUBSET_DOWNLOAD` is forbidden. The next bounded schema step would require the single CRPA LFS object (NPY, despite the contractual verdict name mentioning H5) and a publisher statement or calibration record; a sample alone cannot establish absolute array calibration.\n",
        encoding="utf-8",
    )

    by_area = {}
    for area in (1, 2):
        rows = [row for row in crpa_rows if row["area"] == area]
        by_area[str(area)] = {
            "count": len(rows),
            "class_counts": [
                {"class_id": key, "class": CLASS_NAMES[key], "count": value}
                for key, value in sorted(Counter(row["class_id"] for row in rows).items())
            ],
            "power_counts": counter_records(Counter(row["transmit_power_dbm"] for row in rows), "power_dbm"),
            "bandwidth_counts": counter_records(Counter(row["bandwidth_mhz"] for row in rows), "bandwidth_mhz"),
            "sample_index_min": min(row["sample_index"] for row in rows),
            "sample_index_max": max(row["sample_index"] for row in rows),
        }
    spoofish = {3, 4, 9, 14}
    overlaps = {}
    for area in (1, 2):
        rows = [row for row in crpa_rows if row["area"] == area]
        positive = {row["transmit_power_dbm"] for row in rows if row["class_id"] in spoofish}
        negative = {row["transmit_power_dbm"] for row in rows if row["class_id"] not in spoofish}
        overlaps[str(area)] = sorted(positive & negative)
    write_json(
        output / "label_distribution_audit.json",
        {
            "receiver_counts": {"CRPA": len(crpa_rows), "Innosense": sum(v["rows"] for v in innosense.values())},
            "crpa": {
                "rows": len(crpa_rows),
                "unique_sample_indices": len({row["sample_index"] for row in crpa_rows}),
                "source_object_candidate_snapshots": 42_673,
                "indices_absent_from_public_splits": 6_487,
                "area": by_area,
                "clean_no_transmission_count": 0,
                "binary_task_counts": {"spoof_or_meacon": 18023, "nondeceptive_jammer": 18163},
                "spoof_meacon_vs_nonspoof_power_overlap_dbm": overlaps,
                "released_label_dictionary": {str(key): value for key, value in CLASS_NAMES.items()},
                "direct_fields": ["sample_index", "type", "transmit power", "bandwidth"],
                "inherited_field": ["area from split filename"],
                "unknown_fields": ["date", "time", "E1/E5a band", "VGA", "jammer/spoofer ID", "recording ID", "snapshot cadence", "start/end time", "transmitter position", "receiver position/orientation"],
                "multi_emitter_classes_present_in_split": [],
            },
            "innosense_text_split_only": innosense,
            "unknown_is_not_zero": True,
        },
    )

    write_json(
        output / "campaign_alignment_audit.json",
        {
            "transmission_plan_commit": PLAN_SHA,
            "scheduled_events": len(plan_events),
            "scheduled_events_area_1_2": sum(item["location"]["location_id"] in (1, 2) for item in plan_events),
            "unique_scheduled_test_ids": len({item["event"]["test_id"] for item in plan_events}),
            "schedule_time_bounds": [
                min(item["event"]["start_time"] for item in plan_events),
                max(item["event"]["end_time"] for item in plan_events),
            ],
            "schedule_has_zero_transmission_or_grace_families": True,
            "released_crpa_split_has_clean_label": False,
            "crpa_sample_index_to_timestamp_mapping": "ABSENT",
            "crpa_sample_index_to_test_id_mapping": "ABSENT",
            "crpa_recording_chunk_mapping": "ABSENT",
            "transmitter_receiver_coordinate_binding": "NOT RELEASED WITH CRPA ROWS",
            "azimuth_elevation_computable": False,
            "schedule_to_label_alignment_verifiable": False,
            "continuous_snapshot_independence": "UNKNOWN; sample_index alone does not establish independent recordings",
            "train_test_leakage_risk": "HIGH: published train/test split separates snapshot indices, not documented recording groups",
            "status": "INCONCLUSIVE_ALIGNMENT",
        },
    )

    write_csv(
        output / "shortcut_confound_matrix.csv",
        ["shortcut", "available_for_crpa", "evidence", "gate_status"],
        [
            {"shortcut": "transmit_power", "available_for_crpa": "true", "evidence": "Area1 spoofish/non-spoof overlap: 15,25,30,35,40 dBm; Area2 overlap: none", "gate_status": "PARTIAL_CONFOUND"},
            {"shortcut": "VGA", "available_for_crpa": "false", "evidence": "not present in CRPA split", "gate_status": "UNKNOWN_NOT_ZERO"},
            {"shortcut": "band", "available_for_crpa": "false", "evidence": "not present in CRPA split", "gate_status": "UNKNOWN_NOT_ZERO"},
            {"shortcut": "bandwidth", "available_for_crpa": "true", "evidence": "spoof/meacon commonly encoded as -1; many jammer classes have class-specific widths", "gate_status": "SEVERE_CONFOUND"},
            {"shortcut": "area", "available_for_crpa": "true", "evidence": "Meacon only Area1; Sweep only Area1; Chirp/ChirpB/Triang only Area2", "gate_status": "SEVERE_CONFOUND"},
            {"shortcut": "day", "available_for_crpa": "false", "evidence": "not bound to CRPA indices", "gate_status": "UNKNOWN_NOT_ZERO"},
            {"shortcut": "transmitter_id", "available_for_crpa": "false", "evidence": "not bound to CRPA indices", "gate_status": "UNKNOWN_NOT_ZERO"},
            {"shortcut": "clean_class", "available_for_crpa": "false", "evidence": "0 rows", "gate_status": "MISSING_CONTROL"},
        ],
    )

    (output / "spatial_identifiability.md").write_text(
        "# Spatial identifiability review\n\n"
        "A 4×4 sample covariance is algebraically possible if the inferred four-channel complex layout is confirmed. A 1024-sample snapshot gives at most 1024 temporal observations, enough to form a full-rank 4×4 covariance, but not enough by itself to make its eigenvectors stable under nonstationary wideband waveforms, clipping, multipath, or channel mismatch. Stability must be measured, not assumed.\n\n"
        "The working physical hypothesis is narrower than satellite-AoA resolution: a strong terrestrial emitter can create cross-channel coherence or a dominant spatial mode. Authentic GNSS is normally below the pre-correlation noise floor in a 10 µs wideband snapshot, so the released representation does not justify a claim that separate satellite directions will be visible in raw covariance. Thus low rank may indicate any dominant terrestrial emitter, including a non-deceptive jammer, rather than spoofing.\n\n"
        "MUSIC/MVDR requires element coordinates/order and steering calibration; these are absent. Calibration-free pairwise coherence/eigenvalue ratios could be computed after phase continuity is proven, but fixed RF-chain offsets, element gains, and coupling remain nuisance factors. Multipath can raise apparent rank; multiple emitters can raise it further; a distributed/multi-antenna spoofer can defeat the common-direction premise. Power matching and phase-destruction controls are mandatory to separate spatial evidence from received-power shortcuts.\n\n"
        "Conclusion: covariance is potentially measurable, but spoof-vs-jammer identifiability and calibrated direction inference are not established from metadata.\n",
        encoding="utf-8",
    )

    literature = [
        {"year": 2014, "title": "GNSS Spoofing Detection Using Two-Antenna Differential Carrier Phase", "doi_or_official_url": "https://www.ion.org/publications/abstract.cfm?articleID=12530", "antennas": "2", "input": "per-satellite differential carrier phase", "hypothesis": "authentic satellites have diverse AoA; one spoofer has common AoA", "learning": "physics/unsupervised", "dataset": "live-signal yacht spoofing", "evaluation": "real-time attack", "difference_from_sparc": "post-correlation phase, not 10 us raw covariance"},
        {"year": 2019, "title": "Blind Spoofing GNSS Constellation Detection Using a Multi-Antenna Snapshot Receiver", "doi_or_official_url": "https://doi.org/10.3390/s19245439", "antennas": "6", "input": "snapshot acquisition steering vectors, eigenvalue ratio, clustering", "hypothesis": "spoofed constellation has similar steering vectors", "learning": "physics/unsupervised", "dataset": "open-sky six-element array plus splitter emulation", "evaluation": "single authentic and single spoof scenario", "difference_from_sparc": "closest prior art; already establishes blind snapshot eigen/spatial metrics"},
        {"year": 2020, "title": "Blind Spoofing Detection for Multi-Antenna Snapshot Receivers using Machine-Learning Techniques", "doi_or_official_url": "https://doi.org/10.33012/2020.17564", "antennas": "multi-antenna snapshot array (paper builds on six-element platform)", "input": "blind spatial snapshot metrics", "hypothesis": "ML combines array spoof metrics without trusted calibration", "learning": "supervised LR/KNN/NB/DT/SVM", "dataset": "simulated training and real recorded validation", "evaluation": "cross simulated-to-real; reported F1 >99% for several models", "difference_from_sparc": "directly overlaps the proposed small spatial ML detector"},
        {"year": 2019, "title": "A Two-Stage Interference Suppression Scheme Based on Antenna Array for GNSS Jamming and Spoofing", "doi_or_official_url": "https://doi.org/10.3390/s19183870", "antennas": "12-element 3x4 example", "input": "spatial/cyclic correlation matrices", "hypothesis": "subspaces separate jamming and common-DOA spoof signals", "learning": "physics/unsupervised", "dataset": "simulation", "evaluation": "MUSIC/beam pattern and detection statistic", "difference_from_sparc": "calibrated cyclic MUSIC/mitigation rather than field covariance classification"},
        {"year": 2019, "title": "A Multi-Antenna Scheme for Early Detection and Mitigation of Intermediate GNSS Spoofing", "doi_or_official_url": "https://doi.org/10.3390/s19102411", "antennas": "M-element simulated array", "input": "estimated steering vectors and multipath components", "hypothesis": "counterfeit components share an AoA", "learning": "physics/unsupervised", "dataset": "simulation", "evaluation": "MVDR/LCMV detection and mitigation", "difference_from_sparc": "tracking/acquisition-aware simulated attack"},
        {"year": 2021, "title": "A Spatial-Temporal Approach Based on Antenna Array for GNSS Anti-Spoofing", "doi_or_official_url": "https://doi.org/10.3390/s21030929", "antennas": "simulation array/subarrays", "input": "array eigenspace and cross-correlation", "hypothesis": "spatial-temporal structure separates correlated spoof sources", "learning": "physics/unsupervised", "dataset": "simulation", "evaluation": "detection and suppression", "difference_from_sparc": "calibrated signal model rather than metadata-bound field snapshots"},
        {"year": 2023, "title": "Interference Detection, Localization, and Mitigation Capabilities of CRPA for Aviation", "doi_or_official_url": "https://doi.org/10.3390/ENC2023-15452", "antennas": "CRPA element count not disclosed in abstract", "input": "array covariance/eigendecomposition and MUSIC", "hypothesis": "dominant interference subspace enables detection/localization", "learning": "physics/unsupervised", "dataset": "simulation", "evaluation": "static/dynamic simulated interference", "difference_from_sparc": "general interference mitigation, not spoof-vs-jammer discrimination"},
        {"year": 2024, "title": "GLRT-Based Spacetime Detection Algorithms Via Joint DoA and Doppler Shift", "doi_or_official_url": "https://doi.org/10.1109/JIOT.2024.3413954", "antennas": "array count not exposed by publisher abstract", "input": "relative DoA and Doppler difference", "hypothesis": "spoof signals combine angular inconsistency with similar Doppler", "learning": "physics/GLRT", "dataset": "controlled experiment", "evaluation": "detection probability versus DoA baselines", "difference_from_sparc": "requires GNSS signal/Doppler extraction"},
        {"year": 2025, "title": "Attention-Based Fusion of IQ and FFT Spectrograms with AoA Features for GNSS Jammer Localization", "doi_or_official_url": "https://doi.org/10.1109/RadarConf2559087.2025.11204959", "antennas": "array count not stated in official abstract", "input": "IQ, FFT spectrograms, 22 AoA features", "hypothesis": "learned fusion improves jammer range/azimuth/elevation in multipath", "learning": "supervised deep learning", "dataset": "indoor moving-jammer array dataset", "evaluation": "classification and 3D localization", "difference_from_sparc": "jammer localization; large vision/time-series benchmark"},
        {"year": 2026, "title": "Real-World Jammer and Spoofer Localization Using a Low-Cost Array-Based SDR", "doi_or_official_url": "https://doi.org/10.33012/navi.735", "antennas": "5 coherent KrakenSDR channels", "input": "array DoA over campaign motion", "hypothesis": "coherent array bearings localize terrestrial emitters", "learning": "physics/DoA", "dataset": "Jammertest 2023", "evaluation": "real jammer and driving spoofer localization", "difference_from_sparc": "already supplies real Jammertest spatial jammer/spoofer localization evidence"},
        {"year": 2026, "title": "Analyzing and Characterizing Multi-Source Interference Effects at Jammertest Norway 2025", "doi_or_official_url": "https://arxiv.org/abs/2608.15819", "antennas": "2x2 CRPA", "input": "CRPA/Innosense time series and spectrograms", "hypothesis": "waveform structure supports interference characterization and domain-shift evaluation", "learning": "supervised 17-model benchmark and object detection", "dataset": "Jammertest 2025 release", "evaluation": "within/cross-area type, power, bandwidth; YOLO/RF-DETR", "difference_from_sparc": "same data, but no reported phase-destruction-controlled 4x4 covariance spoof-vs-jammer study"},
        {"year": 2026, "title": "Experimental Evaluation of Spatial-Temporal Interference Mitigation in CRPA GNSS Receivers Under Jamming and Spoofing", "doi_or_official_url": "https://doi.org/10.3390/electronics15122544", "antennas": "two commercial CRPA configurations; element counts undisclosed", "input": "receiver/PVT/RSSI logs", "hypothesis": "STAP gives resilience under jamming/spoofing", "learning": "system comparison", "dataset": "15-minute vehicle trials", "evaluation": "HDOP, satellites, availability", "difference_from_sparc": "system mitigation, not open raw covariance detector"},
        {"year": 2026, "title": "GNSS-Spoofing Detection via ML with In-Domain and Cross-Domain Testing", "doi_or_official_url": "https://doi.org/10.1109/ACCESS.2026.3693081", "antennas": "1", "input": "post-correlation TEXBAT/OAKBAT features", "hypothesis": "RF fingerprints can transfer across attack datasets", "learning": "supervised", "dataset": "TEXBAT and OAKBAT", "evaluation": "in-domain and cross-dataset", "difference_from_sparc": "single-channel reference; cross-domain result motivates group/domain-safe evaluation"},
    ]
    write_json(output / "literature_sources.json", {"as_of": "2026-08-23", "sources": literature})
    (output / "literature_review.md").write_text(
        "# Literature and novelty audit (through 2026-08-23)\n\n"
        "Multi-antenna GNSS spoofing detection is established prior art. Two-antenna differential carrier phase, array steering-vector similarity, eigenspectrum tests, MUSIC/MVDR, cyclic covariance, and blind snapshot ML have all been published. The closest work is the 2019/2020 Fraunhofer six-element snapshot program: it already uses blind spatial/eigen features and then classical ML for spoof detection. Recent work also covers field CRPA mitigation, Jammertest array localization, joint DoA/Doppler GLRT, and AoA-feature deep fusion.\n\n"
        "The Jammertest 2025 data paper benchmarks 17 time-series models and spectrogram detectors and reports severe cross-area degradation, but does not report a phase/coherence-destruction-controlled 4×4 covariance comparison for spoof/meacon versus non-deceptive jammers. That narrow evaluation could differ experimentally, especially if power-matched and recording-safe. It is not currently a defensible WCL claim because released CRPA metadata cannot establish phase provenance, clean controls, recording groups, bands, VGA, or campaign alignment.\n\n"
        "Novelty verdict: `CONDITIONALLY_DIFFERENT_EXPERIMENT, NOT A CURRENTLY DEFENSIBLE SINGLE CLAIM`. “First AI on this dataset” is explicitly rejected. The broad SPARC idea overlaps prior blind snapshot ML; only the narrow cross-area, matched-power, destruction-controlled field validation could be new after provenance closure.\n",
        encoding="utf-8",
    )

    (output / "sparc_candidate_spec.md").write_text(
        "# SPARC-GNSS candidate status\n\n"
        "`WITHHELD_BY_SCHEMA_LABEL_AND_NOVELTY_GATES`\n\n"
        "No working model design is authorized at Stage-0. If provenance is later closed, the only defensible primary task is **spoof/meacon versus non-deceptive terrestrial jammer discrimination**, because no clean CRPA class is released and direction truth is not bound to snapshots. Candidate primary inputs would be antenna-normalized 4×4 Hermitian covariance, eigenvalue ratios, and pairwise complex coherence. Power, VGA, bandwidth, band, area, and day remain audit/nuisance fields, never primary score features. Baselines would be physics-only eigen/coherence, single-channel spectrogram, power/VGA/bandwidth shortcut, and a small Hermitian encoder. A raw-IQ large CNN is out of scope.\n\n"
        "This conditional description is not implementation authorization.\n",
        encoding="utf-8",
    )

    write_json(
        output / "leakage_safe_split_plan.json",
        {
            "status": "NOT_EXECUTABLE_WITH_RELEASED_CRPA_METADATA",
            "required_group_key": "original recording/event, with all consecutive snapshots and derivative chunks kept together",
            "prohibited": "snapshot-level random split as primary evidence",
            "required_evaluations": [
                "transmitter/spoofer ID holdout",
                "day holdout",
                "Area1-to-Area2 and reverse external validation",
                "E1-to-E5a and reverse transfer",
                "matched transmit/received power and VGA",
                "multi-emitter stress set",
            ],
            "metrics": ["macro AUROC", "macro AUPRC", "per-class recall", "FPR at fixed TPR", "matched-power performance", "cross-area performance", "cross-band performance", "recording-level bootstrap CI"],
            "threshold_rule": "set without test labels",
            "missing_bindings": ["recording ID", "timestamp", "day", "transmitter ID", "band", "VGA", "multi-emitter indicator", "received power"],
        },
    )

    write_json(
        output / "destruction_controls.json",
        {
            "negative_shortcut_controls": ["total power only", "VGA only", "bandwidth only", "area/day only", "single antenna channel", "channel magnitude with phase removed"],
            "destruction_tests": ["antenna channel permutation", "independent per-antenna phase randomization", "channel synchronization destruction", "remove off-diagonal covariance", "remove eigenvectors and retain eigenvalues", "power matching", "recording-group label shuffle"],
            "physics_gate": "performance that survives phase/coherence destruction is shortcut evidence, not spatial-physics evidence",
            "status": "PREREGISTERED_FOR_A_FUTURE_AUTHORIZED_EXPERIMENT_ONLY",
        },
    )

    write_json(
        output / "minimal_download_plan.json",
        {
            "current_authorization": "NOT_AUTHORIZED",
            "reason": "direct relative-phase evidence and leakage-safe class/domain bindings are absent",
            "forty_gb_cap_bytes": 40_000_000_000,
            "crpa_is_single_lfs_object": True,
            "single_bounded_object_required_for_next_schema_step": {
                "path": "all_crpa_files.npy",
                "oid_sha256": CRPA_OID,
                "logical_size_bytes": CRPA_SIZE,
                "gib": CRPA_SIZE / 2**30,
                "official_lfs_object_granularity": "entire object; no arbitrary partial reconstruction",
                "receiver": "CRPA",
                "class": "all released CRPA classes; no clean class",
                "area": "Area1 and Area2 shared object",
                "band": "UNKNOWN",
                "recording_group": "UNKNOWN",
            },
            "balanced_clean_spoof_nonspoof_plan": "IMPOSSIBLE: CRPA clean count is zero",
            "area_band_transmitter_constraints": "IMPOSSIBLE TO VERIFY: absent CRPA bindings",
            "would_fit_size_cap": True,
            "download_command_intentionally_omitted": True,
            "clarification": "The contractual verdict name says H5, but the official CRPA payload is NPY. Downloading any HDF5 part would sample Innosense, not the array, and cannot close the CRPA phase gate.",
        },
    )

    write_json(
        output / "ssd_capacity_audit.json",
        {
            "path": str(output),
            "filesystem_total_bytes": disk.total,
            "filesystem_used_bytes": disk.used,
            "filesystem_free_bytes": disk.free,
            "read_only_check": True,
            "files_deleted": 0,
            "crpa_object_fits_current_free_space": disk.free > CRPA_SIZE,
            "full_dataset_fits_current_free_space": disk.free > TOTAL_SIZE,
        },
    )

    write_json(
        output / "access_audit.json",
        {
            "mode": "METADATA_ONLY",
            "git_lfs_skip_smudge": True,
            "git_lfs_pull_or_fetch_executed": False,
            "git_lfs_payload_bytes_downloaded": 0,
            "raw_hdf5_bytes_downloaded": 0,
            "raw_iq_bytes_opened": 0,
            "tuni_raw_payload_bytes_accessed": 0,
            "texbat_raw_payload_bytes_accessed": 0,
            "oakbat_raw_payload_bytes_accessed": 0,
            "models_implemented": False,
            "models_trained": False,
            "scores_computed": False,
            "allowed_downloads": {"Zenodo_JSON_bytes": args.zenodo_json.stat().st_size, "data_paper_PDF_bytes": args.paper_pdf.stat().st_size},
            "ordinary_git_objects_read": ["README", "reader scripts", "split text", "LFS pointers", "transmission-plan JSON/PDF"],
        },
    )

    gates = {
        "four_channel_synchronous_complex_iq_schema": False,
        "direct_relative_phase_preservation_evidence": False,
        "usable_clean_and_spoof_meacon_labels": False,
        "matched_power_vga_comparison": False,
        "recording_safe_split": False,
        "approximately_40gb_minimal_subset_plan": False,
        "defensible_single_wcl_claim": False,
    }
    write_json(
        output / "final_verdict.json",
        {
            "verdict": VERDICT,
            "gates": gates,
            "primary_blocker": "No direct phase-coherence/calibration evidence for the four released CRPA channels",
            "additional_blockers": ["no CRPA clean class", "no recording/day/band/VGA/transmitter binding", "no Area2 power overlap for spoof vs jammer", "broad blind snapshot spatial ML novelty already exists", "Zenodo/repository license conflict"],
            "raw_download_authorized": False,
            "model_implementation_authorized": False,
            "next_state": "AWAIT_PUBLISHER_PHASE_AND_METADATA_PROVENANCE_OR_SEPARATELY_AUTHORIZE_ONE_BOUNDED_CRPA_OBJECT",
            "contract_note": "Verdict token is retained exactly; the one relevant bounded object is NPY, because released HDF5 files are Innosense-only.",
        },
    )

    (output / "README.md").write_text(
        "# Jammertest 2025 CRPA Stage-0 metadata-only feasibility audit\n\n"
        f"Final verdict: `{VERDICT}`. No raw IQ/HDF5/LFS payload was downloaded or opened, and no model, training, or score was run.\n\n"
        "The release contains a 2×2 CRPA and a reader that exposes four complex channels, but it does **not** directly bind channel order, synchronization, relative phase preservation, calibration, geometry, or orientation. The CRPA is a single 1,398,308,992-byte NPY LFS object, not the split HDF5 series described generically in the README. Public CRPA split rows contain type/power/bandwidth and area-by-filename, but no clean label, band, VGA, time/day, transmitter, recording group, or position.\n\n"
        "Consequently, neither a leakage-safe balanced subset nor a power/VGA-matched spatial spoofing experiment can be authorized. The exact next payload object is documented only as a bounded follow-up target; this audit does not authorize downloading it. The literature audit also finds direct prior blind multi-antenna snapshot/eigen/ML work, so novelty would need the narrower destruction-controlled cross-domain field claim after provenance closure.\n",
        encoding="utf-8",
    )

    write_manifest(output)
    print(json.dumps({"status": "BUILT", "verdict": VERDICT, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
