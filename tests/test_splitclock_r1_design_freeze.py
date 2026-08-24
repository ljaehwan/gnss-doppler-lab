from gnss_doppler_lab.splitclock_r1_contract import ALLOWED_VERDICTS, FORBIDDEN_MODEL_INPUTS, frozen_design


def test_design_is_pre_score_and_base_bound():
    design = frozen_design()
    assert design["status"] == "PRE_CLEAN_SCORE_DESIGN_FREEZE"
    assert design["base_sha"] == "04ba01f2f01d08882e069917bfe062c93abe1585"
    assert design["scope"]["attack_stats_hashes_opens_mmaps_bytes"] == 0


def test_r0_contract_repairs_are_semantically_separate():
    observable = frozen_design()["observable_contract"]
    assert observable["carrier"].startswith("carrier_increment_m = +lambda_E1")
    assert observable["cadence"]["acquisition_coherent_integration_ms"] == 8
    assert observable["cadence"]["native_trace_cadence_must_not_be_hardcoded"]


def test_soft_state_space_is_primary_not_hard_clustering():
    model = frozen_design()["model"]
    assert "soft EM" in model["estimator"]
    assert model["hard_clustering_final_forbidden"]
    assert model["effective_mass"] == "sum(pi)>=2 and sum(1-pi)>=2"


def test_dynamic_panel_and_heldout_are_causal():
    design = frozen_design()
    assert design["dynamic_panel"]["same_observation_mask_assertion"]
    assert "heldout inaccessible" in design["model"]["restart_selection"]
    assert design["score"]["fit_epochs"] == 7
    assert design["score"]["heldout_epochs"] == 3


def test_forbidden_inputs_and_verdict_inventory_are_frozen():
    design = frozen_design()
    assert set(design["observable_contract"]["forbidden_model_inputs"]) == set(FORBIDDEN_MODEL_INPUTS)
    assert tuple(design["verdicts"]) == ALLOWED_VERDICTS
    assert len(ALLOWED_VERDICTS) == 9
