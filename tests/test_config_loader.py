from pathlib import Path

from gnss_doppler_lab.config_loader import load_scenario_config


def test_load_scenario_config_reads_extended_fields() -> None:
    config = load_scenario_config(Path("configs/seoul_poc.yaml"))

    assert config.scenario_name == "seoul_poc"
    assert config.latitude_deg == 37.5665
    assert config.mask_angle_deg == 10.0
    assert config.receiver_speed_mps == 22.0
    assert config.num_satellites == 24
    assert config.epoch_count == 11
    assert config.rinex_nav_path is not None
    assert Path(config.rinex_nav_path).name == "BRDC00WRD_S_20240010000_01D_MN.rnx"
    assert Path(config.rinex_nav_path).is_absolute()
