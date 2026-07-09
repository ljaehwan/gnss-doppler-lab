from pathlib import Path

from gnss_doppler_lab.pipeline import run_visibility_pipeline


def test_run_visibility_pipeline_writes_expected_outputs(tmp_path: Path) -> None:
    result = run_visibility_pipeline(
        config_path=Path("configs/seoul_poc.yaml"),
        output_root=tmp_path,
    )

    summary_path = result.output_dir / "visibility_summary.csv"
    timeseries_path = result.output_dir / "doppler_timeseries.csv"

    assert summary_path.exists()
    assert timeseries_path.exists()
    assert result.records_written > 0
    assert result.visible_satellite_count > 0

    summary_text = summary_path.read_text()
    timeseries_text = timeseries_path.read_text()

    assert "prn" in summary_text
    assert "doppler_hz" in timeseries_text
    assert "G" in summary_text or "G" in timeseries_text
