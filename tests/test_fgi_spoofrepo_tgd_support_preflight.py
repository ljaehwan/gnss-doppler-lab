from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_fgi_spoofrepo_tgd_support.py"
SPEC = importlib.util.spec_from_file_location("fgi_tgd_support_preflight", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config() -> dict:
    return json.loads(MODULE.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_frozen_config_is_valid() -> None:
    MODULE.validate_config(config())


def test_real_if_adapter_is_byte_xlating_complex9(tmp_path: Path) -> None:
    text = MODULE.render_config(iq_path=tmp_path / "input.dat", output_dir=tmp_path, config=config())
    assert "GNSS-SDR.internal_fs_sps=6500000" in text
    assert "SignalSource.item_type=byte" in text
    assert "SignalSource.samples=6240000000" in text
    assert "DataTypeAdapter.implementation=Byte_To_Short" in text
    assert "InputFilter.implementation=Freq_Xlating_Fir_Filter" in text
    assert "InputFilter.IF=6390000" in text
    assert "InputFilter.decimation_factor=4" in text
    assert "Tracking_1C.tap_count=9" in text
    assert "Tracking_1C.tap_spacing_chips=0.125" in text


def test_byte_source_count_has_one_item_per_real_sample() -> None:
    assert MODULE.byte_source_item_count(10.0) == 260_000_000

