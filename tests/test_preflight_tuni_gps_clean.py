from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/preflight_tuni_gps_clean.py"


def _load():
    spec = importlib.util.spec_from_file_location("preflight_tuni_gps_clean", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PREFLIGHT = _load()


def test_ishort_item_count_has_i_and_q() -> None:
    assert PREFLIGHT.ishort_source_item_count(10.0) == 1_000_000_000


def test_render_config_is_gps_baseband_complex_nine_tap(tmp_path: Path) -> None:
    text = PREFLIGHT.render_config(
        iq_path=tmp_path / "c5.bin",
        output_dir=tmp_path / "receiver",
        duration_s=10.0,
    )

    assert "SignalSource.item_type=ishort" in text
    assert "DataTypeAdapter.swap_endian=true" in text
    assert "SignalSource.sampling_frequency=50000000" in text
    assert "SignalSource.samples=1000000000" in text
    assert "Resampler.sample_freq_out=5000000" in text
    assert "Channels_1C.count=31" in text
    assert "Channels_1B.count" not in text
    assert "Tracking_1C.tap_count=9" in text
    assert "Tracking_1C.tap_spacing_chips=0.125" in text
