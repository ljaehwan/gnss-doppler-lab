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
