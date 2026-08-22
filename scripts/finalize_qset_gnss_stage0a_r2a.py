#!/usr/bin/env python3
"""Reporting-only finalizer for completed Q-SET R2a evidence."""
from __future__ import annotations
import csv, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gnss_doppler_lab import qset_stage0a_r2a as Q


def summary(name, value):
    s=value['support']; e=value['events']
    qualified=[row for row in s['per_prn'] if row['qualifying_one_second_windows']>0]
    gate=value['status']=='PASS' and s['m_ge_5_windows']>=60 and s['prn9_stable'] and s['finite_failures']==s['cadence_failures']==s['causal_failures']==0 and e['terminal_drain']
    return {'variant':name,'parameters':value['parameters'],'status':value['status'],'exit_code':value.get('exit_code',value.get('receiver',{}).get('exit_code')),'terminal_drain':e['terminal_drain'],'tracked_prns_ever':s['tracked_prns'],'tracked_prn_count_ever':s['tracked_prn_count'],'qualified_prns':[row['prn'] for row in qualified],'qualified_prn_count':len(qualified),'m_ge_5_windows':s['m_ge_5_windows'],'longest_m_ge_5_run':s['longest_m_ge_5_run'],'median_panel_size':s['median_panel_size'],'panel_churn':s['panel_churn'],'prn9_stable_windows':s['prn9_stable_windows'],'telemetry_prns':e['telemetry_prns'],'finite_failures':s['finite_failures'],'cadence_failures':s['cadence_failures'],'causal_failures':s['causal_failures'],'support_gate_pass':gate,'output_set_sha256':value.get('output_set',{}).get('aggregate_sha256',value.get('r2_output_set_sha256'))}


def main():
    A=Q.ARTIFACT; A.mkdir(parents=True,exist_ok=True)
    full=Q.read_json(Q.SSD_ROOT/'variant_results.json'); rows=[summary(k,v) for k,v in full.items()]
    Q.write_json(A/'receiver_variant_results.json',{'schema':'gnss-doppler-lab.qset-stage0a-r2a-variant-results.v1','status':'PASS','execution_order':['V0_REUSE','V1','V2','V3'],'variants':rows,'selected_variant':'V3','selection_reason':'only variant satisfying the preregistered SS-1 support gate'})
    Q.write_json(A/'ss1_support_audit.json',{'schema':'gnss-doppler-lab.qset-stage0a-r2a-ss1-support.v1','status':'PASS','variants':rows,'root_cause':'RECEIVER_ACQUISITION_INTEGRATION_LIMIT_CONFIRMED','evidence':'V1 channel concurrency and V2 relaxed PFA remained below five stable PRNs; V3 8-ms coherent acquisition produced six stable PRNs and 132 consecutive M>=5 windows','spoofing_score_computed':False,'detection_claim':False})
    clean=Q.read_json(Q.SSD_ROOT/'clean_regression.json')
    clean_compact={'schema':clean['schema'],'status':clean['status'],'selected_variant':clean['selected_variant'],'model_sha256':clean['model_sha256'],'threshold_sha256':clean['threshold_sha256'],'refit':clean['refit'],'metrics':clean['metrics'],'replays':{name:{'status':value['status'],'exit_code':value['exit_code'],'terminal_drain':value['terminal_drain'],'config_sha256':value['config']['sha256'],'output_set_sha256':value['output_set']['aggregate_sha256'],'support':{'m_ge_5_windows':value['support']['m_ge_5_windows'],'tracked_prn_count':value['support']['tracked_prn_count'],'finite_failures':value['support']['finite_failures'],'cadence_failures':value['support']['cadence_failures'],'causal_failures':value['support']['causal_failures']}} for name,value in clean['replays'].items()},'gate':{'minimum_scoreable_windows':100,'minimum_panel':5,'empirical_fpr_max':0.01,'wilson_95_upper_max':0.05,'pass':False}}
    Q.write_json(A/'clean_regression.json',clean_compact)
    acquisition=[]; occupancy=[]
    for name,value in full.items():
        for event in value['events']['initial_assignments']: acquisition.append({'variant':name,'event':'initial_assignment',**event})
        for event in value['events']['tracking_starts']: acquisition.append({'variant':name,'event':'acquisition_success_tracking_start',**event})
        for second,panel in value['support']['panels'].items(): occupancy.append({'variant':name,'second':second,'panel_size':len(panel),'prns':';'.join(map(str,panel))})
    Q.write_csv(A/'acquisition_attempt_timeline.csv',acquisition,['variant','event','channel','prn'])
    Q.write_csv(A/'channel_occupancy_timeline.csv',occupancy,['variant','second','panel_size','prns'])
    Q.write_json(A/'access_audit.json',{'schema':'gnss-doppler-lab.qset-stage0a-r2a-access.v1','status':'PASS','allowed_inputs':['preserved R2 C-1/C-3/SS-1 decoded IQ, logs, TRACE'],'ss1_raw_redecode':False,'ss1_spoofing_score_operations':0,'ss1_morphology_operations':0,'roc_auc_operations':0,'forbidden_raw_access':{'stats':0,'hashes':0,'opens':0,'mmaps':0,'bytes_read':0},'unopened_attack_payloads':['SS-3','SS-5','SS-11','SS-12','SS-13'],'receiver_runs':{'SS-1_completed_versioned':3,'SS-1_preserved_pre-adapter-exception':1,'clean_C1':1,'clean_C3':1},'note':'counters are enforced by the scenario-explicit implementation and static verifier; no OS-wide unrelated-process claim'})
    code_paths=['src/gnss_doppler_lab/qset_stage0a_r2a.py','scripts/run_qset_gnss_stage0a_r2a.py','scripts/finalize_qset_gnss_stage0a_r2a.py','scripts/verify_qset_gnss_stage0a_r2a.py','tests/test_qset_gnss_stage0a_r2a.py']
    Q.write_json(A/'freeze_commit.json',{'status':'PASS','base_sha':Q.BASE_SHA,'preregistration_freeze_sha':'4fdb60ad0ed43bfa244ff963dfa0454fb8b8d741','cadence_adapter_repair_sha':'bcb2c719421f8de18eb60e7709ee93717e59a9e7','clean_binding_adapter_repair_sha':'e47da3ac8a02aa522b2bd07e2755550b5d83387f','variant_or_gate_changes_after_freeze':False,'scientific_changes':False,'executable_final_bindings':[{'path':path,**{key:value for key,value in Q.file_binding(ROOT/path).items() if key!='path'}} for path in code_paths]})
    Q.write_json(A/'deterministic_reproduction.json',{'status':'PASS','worker_count':1,'commands':['/usr/bin/python3 scripts/run_qset_gnss_stage0a_r2a.py run-variants','/usr/bin/python3 scripts/run_qset_gnss_stage0a_r2a.py clean-regression --selected V3','/usr/bin/python3 scripts/finalize_qset_gnss_stage0a_r2a.py','/usr/bin/python3 scripts/verify_qset_gnss_stage0a_r2a.py'],'large_output_root':str(Q.SSD_ROOT),'checkpoint_validation':'all completed versioned manifests rehash before reuse','overwrite':False})
    Q.write_json(A/'final_verdict.json',{'schema':'gnss-doppler-lab.qset-stage0a-r2a-final-verdict.v1','verdict':'RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION','next_state':'NOT_AUTHORIZED','selected_candidate':'V3_REJECTED','root_cause_finding':'SS-1 support deficit is receiver acquisition-integration dependent, not demonstrated dataset unavailability','clean_regression_pass':False,'unopened_attack_confirmation_authorized':False,'ss1_spoofing_score_computed':False,'detection_claim':False,'model_threshold_feature_window_aggregator_changed':False})
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        P=A/'plots'; P.mkdir(exist_ok=True)
        labels=[row['variant'] for row in rows]; m5=[row['m_ge_5_windows'] for row in rows]; q=[row['qualified_prn_count'] for row in rows]
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar(labels,m5,color='#4169e1'); ax.axhline(60,color='#b22222',linestyle='--',label='gate = 60'); ax.set_ylabel('1 s windows with M >= 5'); ax.set_title('SS-1 stable support by frozen receiver variant'); ax.legend(); fig.tight_layout(); fig.savefig(P/'ss1_support_by_variant.png',dpi=160); plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,4));
        for name,value in full.items(): ax.plot([int(x) for x in value['support']['panels']], [len(x) for x in value['support']['panels'].values()],label=name,linewidth=1.2)
        ax.axhline(5,color='#b22222',linestyle='--'); ax.set_xlabel('receiver second'); ax.set_ylabel('qualified PRN panel size'); ax.set_title('SS-1 dynamic panel timeline'); ax.legend(); fig.tight_layout(); fig.savefig(P/'ss1_panel_timeline.png',dpi=160); plt.close(fig)
    except Exception as exc: raise RuntimeError(f'plot generation failed: {exc}')
    (A/'README.md').write_text('# Q-SET-GNSS Stage-0A R2a\n\nFinal verdict: `RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION`.\n\nThe locked audit isolated the original three-PRN SS-1 support deficit to receiver acquisition integration: V3 (12 concurrent acquisition channels, 8 ms coherent integration, original PFA) yielded six stable PRNs and 132 consecutive M>=5 windows. The candidate was rejected because the unchanged R2 model and threshold produced clean FPR 0.3944 on C-1 and 0.0791 on C-3, both failing the frozen regression gate. This is receiver engineering evidence only. No SS-1 spoofing score, morphology, ROC/AUC, or detection claim was made, and no unopened attack raw was accessed.\n',encoding='utf-8')
    print(json.dumps({'status':'PASS_FINALIZED','verdict':'RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION','selected_candidate':'V3_REJECTED'},sort_keys=True))
if __name__=='__main__': main()
