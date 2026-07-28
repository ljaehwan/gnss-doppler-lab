import importlib.util
import inspect
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "score_texbat_prn_node_gru.py"
    spec = importlib.util.spec_from_file_location("score_node_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_scorer_supports_generic_prefix_and_onset_without_changing_defaults():
    mod = _load_module()
    sig = inspect.signature(mod.score_node_csv)
    assert sig.parameters["onset_s"].default == 100.0
    assert sig.parameters["output_prefix"].default == "texbat"
    assert mod.score_output_paths(Path("out"), "os1", "oakbat")[0].name == "oakbat_os1_prn_local_scores.csv"


def test_checkpoint_provenance_records_hash_and_feature_contract(tmp_path):
    mod = _load_module()
    ckpt = tmp_path / "model.pt"; ckpt.write_bytes(b"frozen")
    provenance = mod.checkpoint_provenance(ckpt, ["tap_E4", "tap_P"])
    assert provenance["checkpoint_sha256"] == "ffb304816a1090313e833215c08dae3d209cfad1ffd1f674f0909a2ae99e1394"
    assert provenance["node_feature_columns"] == ["tap_E4", "tap_P"]



def test_oakbat_threshold_summary_uses_dynamic_onset_neutral_keys():
    mod = _load_module()
    import pandas as pd
    event = pd.DataFrame({"window_start_s": [109.0, 119.0, 120.0, 130.0]})
    flags = pd.Series([True, False, True, True])
    metrics = mod.threshold_flag_metrics(event, flags, onset_s=120.0, legacy_aliases=False)
    assert metrics["pre_attack_windows"] == 2
    assert metrics["post_attack_windows"] == 2
    assert metrics["buffered_post_windows"] == 1
    assert metrics["detection_delay_s"] == 0.0
    assert "100" not in __import__("json").dumps(metrics)
    assert "90" not in __import__("json").dumps(metrics)
    assert "110" not in __import__("json").dumps(metrics)


def test_clean_threshold_summary_has_only_false_positive_metrics():
    mod = _load_module()
    import pandas as pd
    event = pd.DataFrame({"window_start_s": [0.0, 0.5, 1.0]})
    flags = pd.Series([False, True, False])
    metrics = mod.threshold_flag_metrics(event, flags, onset_s=None, legacy_aliases=False)
    assert metrics["false_positive_flags"] == 1
    assert metrics["false_positive_exceedance_rate"] == 1 / 3
    assert "attack" not in __import__("json").dumps(metrics)
    assert "onset" not in __import__("json").dumps(metrics)


def test_default_plot_restores_exact_texbat_time_label(tmp_path):
    mod = _load_module()
    import pandas as pd
    event = pd.DataFrame({"window_start_s": [0.0], "prn_node_rmse_max": [0.1],
                          "prn_node_rmse_top3_mean": [0.1], "tracked_prn_count": [1]})
    captured = {}
    old_close = mod.plt.close
    mod.plt.close = lambda fig: captured.setdefault("fig", fig)
    try:
        mod.make_plot(event, {}, tmp_path / "p.png", "ds4")
    finally:
        mod.plt.close = old_close
    assert captured["fig"].axes[2].get_xlabel() == "TEXBAT time (s)"
    old_close(captured["fig"])


def test_scorer_roots_are_resolved_from_script_not_cwd():
    mod = _load_module()
    expected = Path(mod.__file__).resolve().parents[1]
    assert mod.ROOT == expected
    assert mod.SRC_ROOT == expected / 'src'


def test_checkpoint_loader_uses_safe_load_and_fails_closed_on_structure(tmp_path, monkeypatch):
    mod = _load_module()
    calls = {}
    monkeypatch.setattr(mod.torch, 'load', lambda *a, **k: calls.update(k) or {})
    with __import__('pytest').raises(ValueError, match='checkpoint'):
        mod.load_checkpoint(tmp_path / 'bad.pt')
    assert calls['weights_only'] is True


def test_numeric_feature_validation_rejects_nan_and_infinity():
    mod = _load_module(); import pandas as pd; import pytest
    frame = pd.DataFrame({'run_id':['r','r'], 'prn':['G01','G02'], 'window_bin_s':[0.0,0.0],
                          'window_start_s':[0.0,0.0], 'window_end_s':[1.0,1.0], 'window_mid_s':[0.5,0.5],
                          'f':[float('nan'), float('inf')]})
    with pytest.raises(ValueError, match='finite'):
        mod.validate_node_inputs(frame, ['f'])


def test_event_aggregation_keeps_runs_separate_and_one_run_values():
    mod = _load_module(); import pandas as pd
    scores = pd.DataFrame([
      {'run_id':'r1','prn':'G01','window_bin_s':0.5,'window_start_s':0.5,'window_end_s':1.5,'prn_node_rmse':1.0},
      {'run_id':'r1','prn':'G02','window_bin_s':0.5,'window_start_s':0.5,'window_end_s':1.5,'prn_node_rmse':3.0},
      {'run_id':'r2','prn':'G01','window_bin_s':0.5,'window_start_s':0.5,'window_end_s':1.5,'prn_node_rmse':10.0},
    ])
    event = mod.aggregate_event_scores(scores)
    assert list(zip(event.run_id, event.window_bin_s)) == [('r1',0.5),('r2',0.5)]
    assert event.prn_node_rmse_max.tolist() == [3.0,10.0]
    assert event.prn_node_rmse_top3_mean.tolist() == [2.0,10.0]


def test_score_timing_contract_records_frozen_start_and_availability_offset():
    mod = _load_module()
    assert mod.TIMING_CONTRACT == {
      'score_time_field':'window_start_s',
      'window_duration_s':1.0,
      'window_availability_offset_s':1.0,
      'interpretation':'scores timestamped at frozen window start become available one full window later',
    }


import pytest

@pytest.mark.parametrize("contents", [
    "prn_node_mae\n1.0\n",
    "prn_node_rmse\n",
    "prn_node_rmse\n\n",
    "prn_node_rmse\nnot-a-number\n",
    "prn_node_rmse\nnan\n",
    "prn_node_rmse\ninf\n",
    "prn_node_rmse\n-inf\n",
])
def test_validation_prn_scores_fail_closed_before_quantiles(tmp_path, contents):
    mod = _load_module()
    path = tmp_path / "validation_prn_node_scores.csv"
    path.write_text(contents)
    with pytest.raises(ValueError, match="validation.*prn_node_rmse|prn_node_rmse.*finite|empty"):
        mod.load_validation_prn_scores(path)


def test_validation_prn_scores_accept_numeric_nonempty_finite_values(tmp_path):
    mod = _load_module()
    path = tmp_path / "validation_prn_node_scores.csv"
    path.write_text("prn_node_rmse,prn_node_mae\n1.0,0.5\n2.5,1.0\n")
    frame = mod.load_validation_prn_scores(path)
    assert frame["prn_node_rmse"].tolist() == [1.0, 2.5]
