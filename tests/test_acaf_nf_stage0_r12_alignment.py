import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1] / 'src/gnss_doppler_lab/acaf_nf_stage0_r12_alignment.py'
spec=importlib.util.spec_from_file_location('r12', P); r12=importlib.util.module_from_spec(spec); spec.loader.exec_module(r12)

def test_candidate_set_is_preregistered_and_consecutive_not_fixed_length():
    names={x['name'] for x in r12.alignment_candidates()}
    assert len(names)==24
    assert all(x['interval'] in {'prev_to_cur','cur_to_next'} for x in r12.alignment_candidates())
    assert r12.interval_bounds({'sample_count':100,'next_sample_count':137}, 'cur_to_next') == (100,137)

def test_same_epoch_cross_prn_allowed_role_overlap_forbidden():
    rows=[{'prn':p,'sample_count':0,'next_sample_count':25,'role':'train','tracker_row':p} for p in range(1,15)]
    selected=r12.select_role_stratified(rows, 14)
    assert len(selected)==14 and len({x['prn'] for x in selected})==14
    assert r12.roles_nonoverlap([{'role':'train','start':0,'end':10},{'role':'holdout','start':9,'end':12}]) is False

def test_14_to_at_least_8_and_dominance():
    rows=[{'prn':p,'sample_count':p,'next_sample_count':p+1,'role':'train','tracker_row':p} for p in range(1,15)]
    chosen=r12.select_role_stratified(rows, 800)
    assert len({x['prn'] for x in chosen}) >= 8
    assert r12.dominant_fraction([1]*2+[2]*8) == .8

def test_selection_gives_each_feasible_prn_at_least_50_validation_epochs():
    rows=[
        {'prn':prn,'sample_count':epoch * 100 + prn,'next_sample_count':epoch * 100 + prn + 1,
         'role':'holdout','tracker_row':epoch,'channel':f'ch_{prn}'}
        for prn in range(1, 20) for epoch in range(50)
    ]
    chosen=r12.select_role_stratified(rows, r12.DEFAULT_VALIDATION_EPOCHS)
    counts={prn: sum(row['prn'] == prn for row in chosen) for prn in range(1, 20)}
    assert len(chosen) == 950
    assert min(counts.values()) >= 50

def test_aux_mapping_global_offsets_wide_grid_and_exact_separate():
    assert r12.aux_samples_to_chips(25_000,1_023_000,25_000_000)==1023
    assert r12.apply_origin(12, -7)==5
    grid=r12.wide_grid(); assert len(grid['delay_chips'])==17 and len(grid['doppler_hz'])==11
    s=r12.center_stats([{'peak_delay_offset_chips':0,'peak_doppler_offset_hz':0},{'peak_delay_offset_chips':.125,'peak_doppler_offset_hz':50}])
    assert s['exact_center_fraction']==.5 and s['within_tolerance_fraction']==1

def test_exact_raw_binding_passes_a1_and_recovery_failure_remains_null():
    gates=r12.gate_alignment({'binding':'exact_same_raw','candidate':'registered','n':800,'prn_count':8,'min_prn_epochs':50,'dominant_fraction':.2,'within_tolerance_fraction':.94,'pooled_spearman':1,'median_prn_spearman':1,'boundary_fraction':0,'consistent_time':True})
    assert gates['A1_source_binding']=='PASS' and gates['A2_interval_alignment']=='PASS'
    assert gates['A3_recovery']=='FAIL' and gates['selected_alignment'] is None
    assert gates['diagnostic_best_candidate']=='registered'
    assert r12.clean_only_guard(['cleanStatic']) is True

def test_different_raw_fails_a1():
    gates=r12.gate_alignment({'binding':'different','candidate':'registered','n':800,'prn_count':8,'min_prn_epochs':50,'dominant_fraction':.2,'within_tolerance_fraction':1,'pooled_spearman':1,'median_prn_spearman':1,'boundary_fraction':0,'consistent_time':True})
    assert gates['selected_alignment'] is None and gates['A1_source_binding']=='FAIL'
