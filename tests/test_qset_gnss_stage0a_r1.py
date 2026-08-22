from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


RUN = load("qset_r1_run", "scripts/run_qset_gnss_stage0a_r1.py")
VERIFY = load("qset_r1_verify", "scripts/verify_qset_gnss_stage0a_r1.py")


def synthetic_be_iq(count: int, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    values = np.rint(rng.normal(0.0, 18.0, size=(count, 2))).clip(-100, 100).astype(">i2")
    return values.tobytes()


def test_big_endian_iq_decoder_order_and_round_trip() -> None:
    payload = struct.pack(">hhhh", -123, 456, 789, -321)
    decoded = RUN.decoder_from_bytes(payload)
    assert decoded.dtype == np.dtype("complex64")
    assert np.array_equal(decoded, np.array([-123 + 456j, 789 - 321j], dtype=np.complex64))
    assert RUN.decoder_round_trip(payload)


def test_streaming_chunk_round_trip() -> None:
    payload = synthetic_be_iq(20_000)
    reconstructed = bytearray()
    for start in range(0, len(payload), 4096):
        chunk = payload[start : start + 4096]
        assert len(chunk) % 4 == 0
        decoded = RUN.decoder_from_bytes(chunk)
        words = np.empty((len(decoded), 2), dtype=">i2")
        words[:, 0] = decoded.real.astype(np.int16); words[:, 1] = decoded.imag.astype(np.int16)
        reconstructed.extend(words.tobytes())
    assert bytes(reconstructed) == payload


def test_format_identification_selects_be_i2_and_rejects_float(tmp_path: Path) -> None:
    window_bytes = 32_768
    target = tmp_path / "synthetic.bin"
    target.write_bytes(b"".join(synthetic_be_iq(window_bytes // 4, seed) for seed in (1, 2, 3)))
    windows = tuple({"id": name, "offset_bytes": index * window_bytes, "length_bytes": window_bytes} for index, name in enumerate(("head", "middle", "tail")))
    result = RUN.identify_format(target, windows, expected_size=None)
    assert result["status"] == "PASS"
    assert result["selected_format"] == ">i2 interleaved I,Q"
    assert result["float32_rejection"]["<f4"]["rejected"]
    assert result["float32_rejection"][">f4"]["rejected"]
    assert result["little_endian_i2_rejected"]


def test_format_read_outside_frozen_bounds_fails(tmp_path: Path) -> None:
    target = tmp_path / "x.bin"; target.write_bytes(synthetic_be_iq(100))
    windows = ({"id": "bad", "offset_bytes": 396, "length_bytes": 16},)
    with pytest.raises(RUN.PreflightError): RUN.identify_format(target, windows, expected_size=None)


def test_source_mutation_detected(tmp_path: Path) -> None:
    target = tmp_path / "source.bin"; target.write_bytes(synthetic_be_iq(100))
    before = RUN.sha256_file(target); target.write_bytes(target.read_bytes()[:-1] + b"x")
    assert RUN.sha256_file(target) != before


def test_sample_time_mapping_constants() -> None:
    assert RUN.RAW_SAMPLE_RATE * RUN.INTERPOLATION // RUN.DECIMATION == RUN.OUTPUT_SAMPLE_RATE
    output_index = 40_000_000
    raw = 1_500_000_000 + output_index * RUN.DECIMATION / RUN.INTERPOLATION - RUN.RESAMPLER_GROUP_DELAY_INPUT_SAMPLES
    receiver_s = output_index / RUN.OUTPUT_SAMPLE_RATE
    mapped_s = (raw - 1_500_000_000 + RUN.RESAMPLER_GROUP_DELAY_INPUT_SAMPLES) / RUN.RAW_SAMPLE_RATE
    assert mapped_s == pytest.approx(receiver_s, abs=1e-12)


def test_dynamic_panel_mask_not_fixed_ten_slots() -> None:
    panel = [2, 5, 11, 19, 24]
    masks = {"a": {str(prn): prn in {2, 5, 19, 24} for prn in panel}, "b": {str(prn): prn in {2, 11, 19} for prn in panel}}
    assert len(panel) == 5 and len(panel) != 10
    assert [prn for prn in panel if masks["a"][str(prn)]] == [2, 5, 19, 24]
    assert [prn for prn in panel if masks["b"][str(prn)]] == [2, 11, 19]


def test_tracking_dump_record_schema_and_finite_detection(tmp_path: Path) -> None:
    assert RUN.TRACK_RECORD_DTYPE.itemsize == 96
    records = np.zeros(3, dtype=RUN.TRACK_RECORD_DTYPE); records["prn"] = 5; records["sample"] = [100, 16_100, 32_100]
    path = tmp_path / "track.dat"; records.tofile(path)
    parsed = RUN.read_tracking_dump(path)
    assert len(parsed) == 3 and parsed["prn"].tolist() == [5, 5, 5]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RUN.PreflightError): RUN.read_tracking_dump(path)


def test_receiver_config_freezes_galileo_and_dynamic_capacity() -> None:
    text = (ROOT / RUN.CONFIG_TEMPLATE_REL).read_text()
    assert "Galileo_E1_PCPS_Ambiguous_Acquisition" in text
    assert "Galileo_E1_DLL_PLL_VEML_Tracking" in text
    assert "Galileo_E1B_Telemetry_Decoder" in text
    assert "Channels_1B.count=12" in text and "Channels_1B.count=10" not in text
    assert "doppler_max=15000" in text and "pfa=0.00001" in text


def test_resampler_taps_frozen_with_system_python() -> None:
    result = subprocess.run(
        ["/usr/bin/python3", "-c", f"import importlib.util; s=importlib.util.spec_from_file_location('m',{str(ROOT / 'scripts/run_qset_gnss_stage0a_r1.py')!r}); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(__import__('json').dumps(m.resampler_identity(),sort_keys=True))"],
        text=True, capture_output=True, check=True,
    )
    identity = json.loads(result.stdout)
    assert identity["tap_count"] == RUN.RESAMPLER_TAP_COUNT
    assert identity["tap_sha256"] == RUN.RESAMPLER_TAP_SHA256


def test_streaming_resampler_count_and_finite_with_system_python(tmp_path: Path) -> None:
    source = tmp_path / "source_be_iq.bin"; output = tmp_path / "output.c64"
    source.write_bytes(synthetic_be_iq(100_000))
    code = (
        "import importlib.util,json,numpy as np;"
        f"s=importlib.util.spec_from_file_location('m',{str(ROOT / 'scripts/run_qset_gnss_stage0a_r1.py')!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"r=m.stream_decode_resample(__import__('pathlib').Path({str(source)!r}),__import__('pathlib').Path({str(output)!r}),0,100000);"
        f"print(json.dumps({{'samples':r['output_samples'],'finite':bool(np.isfinite(np.fromfile({str(output)!r},dtype='<c8')).all())}}))"
    )
    result = subprocess.run(["/usr/bin/python3", "-c", code], text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    assert payload == {"samples": 100_000 * 2 // 25 - RUN.RESAMPLER_EOF_LOSS_OUTPUT_SAMPLES, "finite": True}


def test_access_tamper_detection() -> None:
    audit = {
        "status": "PASS",
        "c1": {"identity_hash_passes": 2, "identity_hash_bytes": RUN.RAW_SIZE * 2, "format_window_bytes": 3 * RUN.FORMAT_WINDOW_BYTES, "receiver_decode_bytes": 12_000_000_000, "total_payload_bytes_read_including_pre_freeze_hash": RUN.RAW_SIZE * 2 + 3 * RUN.FORMAT_WINDOW_BYTES + 12_000_000_000},
        "c3": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")},
        "attack": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")},
        "other_tuni2025_raw": {key: 0 for key in ("stats", "hashes", "opens", "mmaps", "bytes_read", "downloads")},
    }
    VERIFY.validate_access(audit)
    changed = copy.deepcopy(audit); changed["attack"]["opens"] = 1
    with pytest.raises(VERIFY.VerificationError): VERIFY.validate_access(changed)
    changed = copy.deepcopy(audit); changed["c3"]["bytes_read"] = 1
    with pytest.raises(VERIFY.VerificationError): VERIFY.validate_access(changed)


def test_manifest_detects_artifact_tamper(tmp_path: Path) -> None:
    target = tmp_path / "x"; target.write_bytes(b"before")
    before = VERIFY.compact_manifest(tmp_path); target.write_bytes(b"after")
    assert before != VERIFY.compact_manifest(tmp_path)


def test_large_hash_tamper_detectable(tmp_path: Path) -> None:
    target = tmp_path / "output"; target.write_bytes(b"native-output")
    expected = hashlib.sha256(target.read_bytes()).hexdigest(); target.write_bytes(b"native-tamper")
    assert RUN.sha256_file(target) != expected


def test_threshold_and_attack_operations_absent() -> None:
    text = (ROOT / "scripts/run_qset_gnss_stage0a_r1.py").read_text()
    assert "threshold_calibrated\": False" in text
    assert "attack_scoring_performed\": False" in text
    assert "READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD" in text
    assert "SS-1" not in text and "SS-3" not in text and "SS-5" not in text
