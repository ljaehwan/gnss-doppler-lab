from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

ROOT = Path('/home/ubuntu/projects/gnss-doppler-lab')
OUT = ROOT / 'artifacts' / 'sci_ds4_baseline_sweep'
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / 'artifacts/texbat_9tap_external_validation_cleanStatic_normalized_dmcpd'
PATHS = {
    'cleanStatic': ROOT / 'artifacts/texbat_clean_normal_9tap_features/cleanStatic_normalized_dmcpd/tap9_tracking_features_w1.0_s0.5_dmcpd.csv',
    'ds1': BASE / 'ds1/tap9_tracking_features_w1.0_s0.5_dmcpd.csv',
    'ds2': BASE / 'ds2/tap9_tracking_features_w1.0_s0.5_dmcpd.csv',
    'ds4': BASE / 'ds4/tap9_tracking_features_w1.0_s0.5_dmcpd.csv',
}
cd = ROOT / 'artifacts/texbat_clean_normal_9tap_features/cleanDynamic/tap9_tracking_features_w1.0_s0.5.csv'
# skip cleanDynamic here because existing file is not normalized_dmcpd node-compatible
# if cd.exists(): PATHS['cleanDynamic'] = cd
META_COLS = {'run_id','source_fingerprint','label','window_bin_s','window_start_s','window_end_s','window_mid_s','prn','channel','segment_index','window_index','epoch_count','tap_count','tap_layout','sample_rate_hz'}

def numeric_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]

def feature_groups(cols):
    groups={}
    groups['doppler_only']=[c for c in cols if c.startswith('doppler')]
    groups['code_only']=[c for c in cols if c.startswith('code_err')]
    groups['dmcpd_only']=[c for c in cols if c.startswith('dmcpd_')]
    groups['tap_rel_cv_only']=[c for c in cols if c.startswith('tap_') and ('_rel_' in c or c.endswith('_cv'))]
    groups['tap_cv_only']=[c for c in cols if c.startswith('tap_') and c.endswith('_cv')]
    groups['tap_rel_prompt_only']=[c for c in cols if c.startswith('tap_') and '_rel_prompt_' in c]
    groups['tap_rel_sum_only']=[c for c in cols if c.startswith('tap_') and '_rel_sum_' in c]
    groups['peak_shape_only']=[c for c in cols if c.startswith('peak_') or c.startswith('left_right_imbalance')]
    groups['morph_core_no_doppler_no_cn0_no_power']=[c for c in cols if (c.startswith('peak_') or c.startswith('left_right_imbalance') or c.startswith('code_err') or c.endswith('_cv') or '_rel_' in c or c.startswith('dmcpd_')) and not c.startswith('doppler') and not c.startswith('cn0')]
    groups['full_no_raw_power']=[c for c in cols if not c.endswith('_mean') or c.startswith('dmcpd_') or c.startswith('peak_') or c.startswith('left_right_imbalance') or c.startswith('code_err') or c.startswith('doppler') or c.startswith('cn0')]
    groups['full_all_features'] = cols[:]
    return {k:v for k,v in groups.items() if v}

def robust_fit(df, cols):
    x=df[cols].replace([np.inf,-np.inf],np.nan).astype(float)
    med=x.median(); mad=(x-med).abs().median(); scale=1.4826*mad
    std=x.std().replace(0,np.nan); scale=scale.where(scale>1e-9,std).fillna(1.0)
    return med,scale

def node_scores(df, cols, med, scale):
    use=[c for c in cols if c in df.columns]
    x=df[use].replace([np.inf,-np.inf],np.nan).astype(float).fillna(med[use])
    z=((x-med[use])/scale[use]).clip(-20,20)
    out=df[['window_mid_s','window_start_s','window_end_s','prn']].copy()
    out['node_score']=np.sqrt((z.to_numpy()**2).mean(axis=1))
    return out

def event_scores(ns):
    ns = ns.copy()
    ns['event_bin_s'] = (ns['window_mid_s'] * 2).round() / 2.0
    def topk(a,k):
        a=np.sort(np.asarray(a,float)); return float(a[-min(k,len(a)):].mean())
    g=ns.groupby('event_bin_s')['node_score']
    return pd.DataFrame({'window_mid_s':g.mean().index,'score_mean':g.mean().values,'score_max':g.max().values,'score_top3':g.apply(lambda a:topk(a,3)).values,'score_top5':g.apply(lambda a:topk(a,5)).values,'prn_count':g.size().values}).sort_values('window_mid_s').reset_index(drop=True)

def auc_rank(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); n1=(y==1).sum(); n0=(y==0).sum()
    if n1==0 or n0==0: return None
    ranks=pd.Series(s).rank(method='average').to_numpy(); r1=ranks[y==1].sum()
    return float((r1-n1*(n1+1)/2)/(n1*n0))

def metrics(ev, th, onset=100.0, buffer=10.0):
    pre=ev.window_mid_s<(onset-buffer); post=ev.window_mid_s>=(onset+buffer); mask=pre|post
    out={'rows':int(len(ev)),'pre_windows_t_lt_90':int(pre.sum()),'post_windows_t_ge_110':int(post.sum()),'auc_pre_vs_post_buffered':auc_rank(post[mask].astype(int),ev.loc[mask,'event_score']),'pre_score_median':float(ev.loc[pre,'event_score'].median()) if pre.any() else None,'post_score_median':float(ev.loc[post,'event_score'].median()) if post.any() else None,'pre_score_q95':float(ev.loc[pre,'event_score'].quantile(.95)) if pre.any() else None,'post_score_q95':float(ev.loc[post,'event_score'].quantile(.95)) if post.any() else None}
    for n,t in th.items():
        flags=ev.event_score>t; pf=int((flags&pre).sum()); df=int((flags&post).sum()); first=ev.loc[flags&post,'window_mid_s'].min() if (flags&post).any() else np.nan
        out[f'{n}_pre_fp_rate']=float(pf/max(1,int(pre.sum()))); out[f'{n}_post_det_rate']=float(df/max(1,int(post.sum()))); out[f'{n}_first_delay_s']=None if np.isnan(first) else float(first-onset)
    return out

print('Loading data...')
dfs={k:pd.read_csv(p) for k,p in PATHS.items() if p.exists()}
print({k:v.shape for k,v in dfs.items()})
clean=dfs['cleanStatic']; cols=numeric_feature_cols(clean); groups=feature_groups(cols)
print('groups:',{k:len(v) for k,v in groups.items()})
rows=[]
for gname,gcols in groups.items():
    med,scale=robust_fit(clean,gcols); ns_clean=node_scores(clean,gcols,med,scale); ev_clean=event_scores(ns_clean)
    for agg in ['score_mean','score_top3','score_top5','score_max']:
        th={'pfa10_q90':float(ev_clean[agg].quantile(.90)),'pfa5_q95':float(ev_clean[agg].quantile(.95)),'pfa1_q99':float(ev_clean[agg].quantile(.99)),'pfa0_5_q995':float(ev_clean[agg].quantile(.995)),'pfa0_1_q999':float(ev_clean[agg].quantile(.999))}
        for scen,df in dfs.items():
            ev=event_scores(node_scores(df,gcols,med,scale)); ev['event_score']=ev[agg]
            if scen.startswith('ds'): m=metrics(ev,th)
            else:
                m={'rows':int(len(ev))}
                for n,t in th.items(): m[f'{n}_normal_flag_rate']=float((ev.event_score>t).mean())
            rows.append({'group':gname,'agg':agg,'scenario':scen,'feature_count':len(gcols),**m})
            if scen.startswith('ds'):
                ev.to_csv(OUT/f'events_{gname}_{agg}_{scen}.csv',index=False)
res=pd.DataFrame(rows); res.to_csv(OUT/'baseline_sweep_metrics.csv',index=False)
ds4=res[res.scenario=='ds4'].copy()
for c in ['pfa1_q99_post_det_rate','pfa1_q99_first_delay_s','pfa5_q95_post_det_rate','auc_pre_vs_post_buffered','pfa1_q99_pre_fp_rate']:
    if c not in ds4: ds4[c]=np.nan
rank=ds4.sort_values(['pfa1_q99_post_det_rate','auc_pre_vs_post_buffered','pfa5_q95_post_det_rate'],ascending=[False,False,False])
rank.to_csv(OUT/'ds4_method_ranking.csv',index=False)
# feature shift
sf=[]; ds4df=dfs['ds4']; pre=ds4df.window_mid_s<90; post=ds4df.window_mid_s>=110
for c in cols:
    if c not in ds4df.columns: continue
    med,scale=robust_fit(clean,[c]); sc=float(scale[c]) if float(scale[c])!=0 else 1.0
    zpre=((ds4df.loc[pre,c].astype(float)-float(med[c]))/sc).abs().replace([np.inf,-np.inf],np.nan).dropna(); zpost=((ds4df.loc[post,c].astype(float)-float(med[c]))/sc).abs().replace([np.inf,-np.inf],np.nan).dropna()
    if len(zpre) and len(zpost): sf.append({'feature':c,'pre_absz_median':float(zpre.median()),'post_absz_median':float(zpost.median()),'delta_post_minus_pre':float(zpost.median()-zpre.median()),'post_absz_q95':float(zpost.quantile(.95)),'pre_absz_q95':float(zpre.quantile(.95))})
sf=pd.DataFrame(sf).sort_values('delta_post_minus_pre',ascending=False); sf.to_csv(OUT/'ds4_single_feature_shift_ranking.csv',index=False)
md=['# SCI ds4 baseline sweep\n\n','Normal-only robust-z baseline. Thresholds are calibrated on cleanStatic only; ds records are only evaluation.\n\n','## Best ds4 methods by q99 post detection\n']
for _,r in rank.head(25).iterrows(): md.append(f"- {r.group}/{r.agg}: q99 det={r.get('pfa1_q99_post_det_rate',np.nan):.3f}, q99 FP={r.get('pfa1_q99_pre_fp_rate',np.nan):.3f}, delay={r.get('pfa1_q99_first_delay_s')}, AUC={r.get('auc_pre_vs_post_buffered',np.nan):.3f}, q95 det={r.get('pfa5_q95_post_det_rate',np.nan):.3f}\n")
md.append('\n## Top ds4 shifted features\n')
for _,r in sf.head(30).iterrows(): md.append(f"- {r.feature}: median |z| pre={r.pre_absz_median:.3f}, post={r.post_absz_median:.3f}, delta={r.delta_post_minus_pre:.3f}, post_q95={r.post_absz_q95:.3f}\n")
(OUT/'README.md').write_text(''.join(md))
summary={'out_dir':str(OUT.relative_to(ROOT)),'metrics_csv':str((OUT/'baseline_sweep_metrics.csv').relative_to(ROOT)),'ds4_ranking_csv':str((OUT/'ds4_method_ranking.csv').relative_to(ROOT)),'single_feature_csv':str((OUT/'ds4_single_feature_shift_ranking.csv').relative_to(ROOT)),'top_ds4':rank.head(10).replace({np.nan:None}).to_dict(orient='records'),'top_features':sf.head(20).replace({np.nan:None}).to_dict(orient='records')}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2)[:6000])
