from gnss_doppler_lab.acaf_nf_stage1_r1 import _evenly, independent_block_effect, select_clean_roles


def test_even_selection_is_deterministic_and_bounded():
    assert _evenly(list(range(10)),3)==[0,4,9]
    assert _evenly([1,2],3)==[1,2]


def test_clean_roles_are_chronological_and_disjoint():
    bins=[{"second":i,"pairs":{}} for i in range(40)]
    roles=select_clean_roles(bins)
    assert set(roles)=={"train","selection","calibration","holdout"}
    assert max(x["second"] for x in roles["train"])<min(x["second"] for x in roles["selection"])
    assert max(x["second"] for x in roles["calibration"])<min(x["second"] for x in roles["holdout"])


def test_block_effect_uses_exact_ten_second_blocks():
    pre=[{"time_s":float(x),"score":0.0} for x in (1,2,11,12)]
    post=[{"time_s":float(x),"score":2.0} for x in (21,22,31,32)]
    result=independent_block_effect(pre,post,7)
    assert result["status"]=="PASS" and result["effect"]==2.0
    assert result["ci95"]==[2.0,2.0] and result["block_seconds"]==10
