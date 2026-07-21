import importlib.util
from pathlib import Path

import pandas as pd


def _load_eval_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "eval_ai_morph_gru_q70_frame.py"
    spec = importlib.util.spec_from_file_location("eval_ai_morph_gru_q70_frame", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_add_score_persistence_is_causal_rolling_max_and_keeps_base_cols():
    mod = _load_eval_module()
    ev = pd.DataFrame({"score": [1.0, 0.5, 2.0, 1.5]})

    cols = mod.add_score_persistence(ev, ["score"], 3)

    assert cols == ["score", "score_persist3"]
    assert ev["score"].tolist() == [1.0, 0.5, 2.0, 1.5]
    assert ev["score_persist3"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_add_score_persistence_disabled_returns_original_cols_without_mutation():
    mod = _load_eval_module()
    ev = pd.DataFrame({"score": [1.0, 2.0]})

    cols = mod.add_score_persistence(ev, ["score"], 1)

    assert cols == ["score"]
    assert list(ev.columns) == ["score"]

def test_q70_eval_defaults_match_validated_morphology_quorum_frame():
    mod = _load_eval_module()

    assert mod.LOW_NODE_Q == 0.70
    assert mod.AGG_Q == 0.65
    assert mod.ROLL_WINDOW == 4
    assert mod.SCORE_PERSISTENCE_WINDOW == 10
    assert mod.QUORUM_TAU == 0.50


def test_event_scores_uses_current_tracked_set_quantile_and_quorum_gate():
    mod = _load_eval_module()
    prn_scores = pd.DataFrame({
        "window_mid_s": [0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
        "prn_node_rmse": [1.0, 2.0, 10.0, 1.0, 9.0, 11.0],
    })

    ev = mod.event_scores(prn_scores, low_thr=8.0, aggregation_quantile=0.65, roll_window=1)

    assert ev["tracked_prn_count"].tolist() == [3, 3]
    assert ev["low_high_fraction"].tolist() == [1 / 3, 2 / 3]
    assert ev.loc[0, "ai_rmse_q"] == pd.Series([1.0, 2.0, 10.0]).quantile(0.65)
    assert ev.loc[0, "ai_rmse_q_tau50_gate"] == ev.loc[0, "ai_rmse_q"]
    expected_gated = ev.loc[1, "ai_rmse_q"] * (1.0 + mod.SOFT_ALPHA * (2 / 3)) + mod.OFFSET_BETA * (2 / 3)
    assert abs(ev.loc[1, "ai_rmse_q_tau50_gate"] - expected_gated) < 1e-12

