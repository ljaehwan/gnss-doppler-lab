from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_tuni_galileo_authentic_windows",
    ROOT / "scripts" / "audit_tuni_galileo_authentic_windows.py",
)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_window_roster_is_nonoverlapping_and_inside_recording() -> None:
    assert MOD.OFFSETS_SECONDS == (30, 90, 150, 210, 270)
    assert all(left + MOD.DURATION_SECONDS <= right for left, right in zip(
        MOD.OFFSETS_SECONDS, MOD.OFFSETS_SECONDS[1:]
    ))
    assert MOD.OFFSETS_SECONDS[-1] + MOD.DURATION_SECONDS < 299.9


def test_config_skips_to_frozen_offset_and_excludes_spoof_prns(tmp_path: Path) -> None:
    spoofed = MOD.SCENARIOS["SS-13"][1]
    authentic = tuple(prn for prn in range(1, 37) if prn not in spoofed)
    text = MOD.render_window_config(
        iq_path=tmp_path / "input.bin", output_dir=tmp_path / "out",
        offset_s=150, authentic_prns=authentic,
    )
    assert "SignalSource.seconds_to_skip=150" in text
    assert "Channels_1B.count=32" in text
    assigned = {
        int(line.split("=", 1)[1]) for line in text.splitlines()
        if line.startswith("Channel") and ".satellite=" in line
    }
    assert assigned == set(range(1, 37)) - spoofed
    assert "Acquisition_1B.pfa=0.000001" in text
    assert "Tracking_1B.tap_count=9" in text
