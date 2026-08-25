import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.rf_impairments import (
    CompositeChannelProcessor,
    ImpairmentConfig,
)
from gnss_doppler_lab.simulation_v4 import (
    OutageEvent,
    SimulationScenario,
    SimulationV4Error,
    SpoofEvent,
    build_carryoff_rows,
    compare_prefix,
    compose_paired_iq,
    generate_simulation_campaign,
    load_simulation_campaign,
    outage_envelope,
    spoof_power_envelope,
)
from gnss_doppler_lab.trajectory import llh_to_enu


def _nav(path: Path) -> None:
    path.write_text(
        "     2.10           N: GPS NAV DATA                         RINEX VERSION / TYPE\n"
        "    18                                                      LEAP SECONDS\n"
        "                                                            END OF HEADER\n",
        encoding="ascii",
    )


def _campaign_config(tmp_path: Path, *, keep_components: bool = False) -> Path:
    _nav(tmp_path / "input.nav")
    simulator = tmp_path / "gps-sdr-sim"
    simulator.write_text("fake")
    config = tmp_path / "campaign.yaml"
    config.write_text(f"""version: 1
campaign:
  name: paired-pilot
  utc: 2022-01-01T00:00:00Z
  duration_seconds: 5
  position:
    type: static
    latitude_deg: 30.2851494
    longitude_deg: -97.7339352
    altitude_m: 180
input:
  rinex_nav: input.nav
output:
  root: runs
  rf_sample_rate_hz: 1000000
  sample_format: s8_iq
  keep_component_iq: {str(keep_components).lower()}
receiver:
  seed: 77
  sample_snr_db: 10
  normal_target_rms: 20
  carrier_offset_hz: 0
  phase_noise_std_rad_per_sqrt_sample: 0
  chunk_samples: 7
scenarios:
  - name: steady
    kind: steady_normal
  - name: recovery
    kind: recovery_normal
    outage:
      start_seconds: 1
      end_seconds: 2
      attenuation_db: -80
      recovery_ramp_seconds: 0.5
  - name: spoof
    kind: carryoff_spoof
    spoofing:
      start_seconds: 1
      transition_seconds: 1
      target_offset_enu_m: [100, 0, 10]
      initial_advantage_db: -20
      final_advantage_db: 3
      power_ramp_seconds: 1
simulator:
  executable: gps-sdr-sim
""", encoding="utf-8")
    return config


def _receiver(chunk_samples: int = 7) -> ImpairmentConfig:
    return ImpairmentConfig(
        enabled=True,
        profile="explicit",
        seed=123,
        sample_snr_db=8.0,
        carrier_offset_hz=0.2,
        frequency_drift_hz_per_s=0.01,
        phase_noise_std_rad_per_sqrt_sample=0.0001,
        frontend_cutoff_hz=None,
        iq_gain_imbalance_db=0.1,
        iq_phase_imbalance_deg=0.2,
        dc_i=0.1,
        dc_q=-0.1,
        gain=1.0,
        agc_target_rms=None,
        clip_level=127.0,
        chunk_samples=chunk_samples,
    )


def _scenarios() -> tuple[SimulationScenario, ...]:
    return (
        SimulationScenario("steady", "steady_normal"),
        SimulationScenario("recovery", "recovery_normal", outage=OutageEvent(2.0, 4.0, -80.0, 1.0)),
        SimulationScenario(
            "spoof",
            "carryoff_spoof",
            spoofing=SpoofEvent(2.0, 2.0, (20.0, 0.0, 0.0), -20.0, 3.0, 2.0),
        ),
    )


def _write_iq(path: Path, i: np.ndarray, q: np.ndarray) -> None:
    values = np.empty(i.size * 2, dtype=np.int8)
    values[0::2] = i
    values[1::2] = q
    values.tofile(path)


def test_load_strict_campaign_resolves_paths_and_requires_all_three_families(tmp_path):
    path = _campaign_config(tmp_path)
    campaign = load_simulation_campaign(path)

    assert campaign.name == "paired-pilot"
    assert campaign.output_root == (tmp_path / "runs").resolve()
    assert campaign.base_rf_config.input.rinex_nav == (tmp_path / "input.nav").resolve()
    assert [scenario.kind for scenario in campaign.scenarios] == [
        "steady_normal", "recovery_normal", "carryoff_spoof",
    ]
    assert campaign.receiver.agc_target_rms is None

    path.write_text(path.read_text().replace("  - name: steady\n    kind: steady_normal\n", ""))
    with pytest.raises(SimulationV4Error, match="requires steady_normal"):
        load_simulation_campaign(path)


def test_campaign_rejects_unknown_receiver_key_and_invalid_event_timing(tmp_path):
    path = _campaign_config(tmp_path)
    path.write_text(path.read_text().replace("  seed: 77", "  seed: 77\n  agc_target_rms: 24"))
    with pytest.raises(SimulationV4Error, match="unknown receiver"):
        load_simulation_campaign(path)

    path = _campaign_config(tmp_path)
    path.write_text(path.read_text().replace("      end_seconds: 2", "      end_seconds: 5"))
    with pytest.raises(SimulationV4Error, match="outage"):
        load_simulation_campaign(path)


def test_outage_and_spoof_envelopes_have_exact_causal_boundaries():
    outage = OutageEvent(1.0, 2.0, -40.0, 1.0)
    envelope = outage_envelope(8, 2, outage)
    assert envelope[:2].tolist() == [1.0, 1.0]
    assert envelope[2] == pytest.approx(0.01)
    assert envelope[3] == pytest.approx(0.01)
    assert envelope[4] == pytest.approx(0.01)
    assert envelope[5] == pytest.approx(0.505, abs=1e-6)
    assert envelope[6:].tolist() == [1.0, 1.0]

    spoof = SpoofEvent(1.0, 1.0, (10.0, 0.0, 0.0), -20.0, 0.0, 2.0)
    power = spoof_power_envelope(8, 2, spoof)
    assert power[:2].tolist() == [0.0, 0.0]
    assert power[2] == pytest.approx(0.1)
    assert power[6] == pytest.approx(1.0)


def test_carryoff_rows_are_identical_through_onset_and_reach_target_offset():
    authentic = tuple((index / 10, 30.2851494, -97.7339352, 180.0) for index in range(40))
    event = SpoofEvent(1.0, 1.0, (100.0, 0.0, 10.0), -20.0, 3.0, 1.0)
    rows = build_carryoff_rows(authentic, event)

    assert rows[:11] == authentic[:11]
    east, north, up = llh_to_enu(*rows[-1][1:], *authentic[-1][1:])
    assert east == pytest.approx(100.0, abs=0.02)
    assert north == pytest.approx(0.0, abs=0.02)
    assert up == pytest.approx(10.0, abs=0.02)


def test_composite_channel_complex_entry_matches_iq8_entry():
    raw = np.array([10, -3, 2, 4, -8, 9, 1, -2], dtype=np.int8).tobytes()
    config = _receiver()
    from_bytes = CompositeChannelProcessor(10.0, config).process(raw)
    values = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
    from_complex = CompositeChannelProcessor(10.0, config).process_complex(values[0::2] + 1j * values[1::2])
    assert np.array_equal(from_bytes, from_complex)
    with pytest.raises(ValueError, match="one-dimensional"):
        CompositeChannelProcessor(10.0, config).process_complex(np.zeros((2, 2), complex))


def test_paired_composer_freezes_gain_and_makes_pre_event_prefix_byte_identical(tmp_path):
    samples = 80
    indices = np.arange(samples)
    authentic = tmp_path / "auth.bin"
    counterfeit = tmp_path / "fake.bin"
    _write_iq(authentic, (15 + indices % 9).astype(np.int8), (-10 + indices % 7).astype(np.int8))
    _write_iq(counterfeit, (-12 + indices % 5).astype(np.int8), (8 - indices % 3).astype(np.int8))
    outputs = {name: tmp_path / name / "iq.bin" for name in ("steady", "recovery", "spoof")}

    report = compose_paired_iq(
        authentic,
        counterfeit,
        outputs,
        _scenarios(),
        sample_rate_hz=10,
        receiver=_receiver(),
        normal_target_rms=20.0,
    )

    assert report["reference"]["fixed_receiver_gain"] > 0
    assert report["processing"]["agc"].startswith("disabled")
    assert compare_prefix(outputs["steady"], outputs["recovery"], 20)["byte_identical"]
    assert compare_prefix(outputs["steady"], outputs["spoof"], 20)["byte_identical"]
    steady = np.fromfile(outputs["steady"], dtype=np.int8)
    recovery = np.fromfile(outputs["recovery"], dtype=np.int8)
    spoof = np.fromfile(outputs["spoof"], dtype=np.int8)
    assert not np.array_equal(steady[40:80], recovery[40:80])
    assert not np.array_equal(steady[40:], spoof[40:])
    assert report["scenarios"]["steady"]["clipping_fraction"] < 0.05


def test_generate_campaign_publishes_receiver_compatible_manifests_and_truth(tmp_path):
    campaign = load_simulation_campaign(_campaign_config(tmp_path))
    rate = 10
    campaign = replace(
        campaign,
        base_rf_config=replace(
            campaign.base_rf_config,
            output=replace(campaign.base_rf_config.output, rf_sample_rate_hz=rate),
        ),
        receiver=replace(campaign.receiver, chunk_samples=7),
    )

    class FakeRunner:
        identity = "fake-gps-sdr-sim"
        executable = "/fake/gps-sdr-sim"
        provenance = "unit-test"
        cli_contract = "fake-contract"

        def expected_output_bytes(self, config):
            return config.scenario.duration_seconds * config.output.rf_sample_rate_hz * 2

        def run(self, config, output, log):
            count = self.expected_output_bytes(config) // 2
            index = np.arange(count)
            if config.scenario.position.__class__.__name__ == "TrajectoryPosition":
                i = (-15 + index % 11).astype(np.int8)
                q = (8 - index % 5).astype(np.int8)
            else:
                i = (12 + index % 7).astype(np.int8)
                q = (-9 + index % 5).astype(np.int8)
            _write_iq(output, i, q)
            log.write_text("fake\n")
            return {"command": [self.executable, "-o", output.name], "actual_bytes": output.stat().st_size}

    manifest_path = generate_simulation_campaign(campaign, FakeRunner())
    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema"] == "gnss-doppler-lab.simulation-v4"
    assert set(manifest["rf_manifests"]) == {"steady", "recovery", "spoof"}
    assert manifest["paired_prefix_checks"]["recovery"]["byte_identical"] is True
    assert manifest["paired_prefix_checks"]["spoof"]["byte_identical"] is True
    assert manifest["truth"]["realized_final_offset_enu_m"][0] == pytest.approx(100.0, abs=0.02)
    assert not (campaign.output_root / "components" / "authentic_gps_l1ca_s8_iq.bin").exists()
    for path in manifest["rf_manifests"].values():
        run_manifest = json.loads(Path(path).read_text())
        iq = Path(path).parent / run_manifest["iq"]["path"]
        assert iq.is_file()
        assert run_manifest["iq"]["sha256"]
        assert run_manifest["iq"]["rf_sample_rate_hz"] == rate
        assert run_manifest["simulation_v4"]["scope"].startswith("offline baseband")
