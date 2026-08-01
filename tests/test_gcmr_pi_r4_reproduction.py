import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.gcmr_pi_r4_reproduction import (
    component_agreement,
    reconstruct_and_rescore,
)


def test_component_agreement_records_error_statistics_alarm_disagreements_and_correlations():
    ref = np.array([0.0, 4.0, 4.0, 9.0])
    actual = np.array([0.0, 2.0, 4.0, 8.0])
    result = component_agreement(ref, actual, threshold=3.0, times=np.array([1.0, 2.0, 3.0, 4.0]))
    assert result["max_abs_error"] == 2.0
    assert result["max_abs_error_time"] == 2.0
    assert result["alarm_agreement_rate"] == 0.75
    assert result["alarm_disagreement_events"] == [2.0]
    assert result["pearson"] is not None and result["spearman"] is not None


def test_component_agreement_uses_null_alarm_fields_without_a_frozen_threshold():
    result = component_agreement([1.0, 2.0], [1.1, 2.1], threshold=None, times=[1.0, 2.0])
    assert result["threshold"] is None
    assert result["alarm_agreement_rate"] is None
    assert result["alarm_disagreement_events"] is None


def test_reconstruct_and_rescore_passes_z_then_residual():
    residual = np.array([11.0, 12.0])
    z = np.array([21.0, 22.0])
    captured = {}

    def reconstruct(pipe, event):
        assert pipe == "pipe" and event == "event"
        return residual, z

    def rescore(pipe, event, supplied_z, supplied_residual):
        captured["args"] = (pipe, event, supplied_z, supplied_residual)
        return "diagnostics", {"Full": 1.0}

    diagnostics, scores = reconstruct_and_rescore("pipe", "event", reconstruct=reconstruct, rescore=rescore)
    assert (diagnostics, scores) == ("diagnostics", {"Full": 1.0})
    assert captured["args"][2] is z
    assert captured["args"][3] is residual


def test_reconstruct_and_rescore_detects_a_reversed_argument_order():
    residual = np.array([11.0, 12.0])
    z = np.array([21.0, 22.0])

    def reconstruct(_pipe, _event):
        return residual, z

    def rescore(_pipe, _event, supplied_z, supplied_residual):
        if not np.array_equal(supplied_z, z) or not np.array_equal(supplied_residual, residual):
            raise AssertionError("rescore requires z, residual")
        return "diagnostics", {"Full": 1.0}

    reconstruct_and_rescore("pipe", "event", reconstruct=reconstruct, rescore=rescore)
    with pytest.raises(AssertionError, match="rescore requires z, residual"):
        rescore("pipe", "event", residual, z)


def test_runner_rescore_event_passes_z_then_residual(monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("r4_reproduction_runner_order", repo / "scripts/run_gcmr_pi_r4_reproduction.py")
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    residual = np.array([11.0])
    z = np.array([21.0])
    monkeypatch.setattr(runner, "reconstruct_event_innovation", lambda _pipe, _event: (residual, z))
    captured = {}

    def rescore(_pipe, _event, supplied_z, supplied_residual):
        captured["args"] = (supplied_z, supplied_residual)
        return "diagnostics", {"Full": 1.0}

    monkeypatch.setattr(runner, "rescore_from_innovations", rescore)
    assert runner.rescore_event("pipe", "event") == ("diagnostics", {"Full": 1.0})
    assert captured["args"][0] is z
    assert captured["args"][1] is residual


def test_frozen_event_helper_matches_direct_rescore():
    repo = Path(__file__).resolve().parents[1]
    frozen = repo / "artifacts/frozen/gcmr-pi-oakbat-3t-r3"
    if not frozen.exists():
        pytest.skip("immutable r3 frozen artifact is not available")
    script_path = repo / "scripts/run_gcmr_pi_r4_reproduction.py"
    spec = importlib.util.spec_from_file_location("r4_reproduction_runner", script_path)
    assert spec and spec.loader
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    import torch
    from run_gcmr_oakbat_poc import SCENARIOS as source_scenarios, load_scenario
    from train_gcmr_peak_innovation import peak_indexes, records
    from gnss_doppler_lab.gcmr_pi_r4_corrected import reconstruct_event_innovation, rescore_from_innovations

    seed = 7
    pipe = runner.load_pipe(frozen / "model.pt", torch.device("cpu"), seed)
    events, _, _ = load_scenario("os1", frozen / "cache", False)
    built, _ = records(events, source_scenarios["os1"], pipe.window, peak_indexes(events, source_scenarios["os1"], pipe.feature_dim))
    event = built[0]
    residual, z = reconstruct_event_innovation(pipe, event)
    direct_diagnostics, direct_scores = rescore_from_innovations(pipe, event, z, residual)
    helper_diagnostics, helper_scores = reconstruct_and_rescore(pipe, event)
    assert helper_diagnostics.s_common == pytest.approx(direct_diagnostics.s_common, abs=1e-12)
    assert helper_diagnostics.n_eff == pytest.approx(direct_diagnostics.n_eff, abs=1e-12)
    assert helper_diagnostics.s_pair == pytest.approx(direct_diagnostics.s_pair, abs=1e-12)
    assert helper_diagnostics.energy == pytest.approx(direct_diagnostics.energy, abs=1e-12)
    assert helper_scores["Full"] == pytest.approx(direct_scores["Full"], abs=1e-12)
