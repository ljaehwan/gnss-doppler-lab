from pathlib import Path
import csv,json,numpy as np
import matplotlib.pyplot as p
from matplotlib.ticker import PercentFormatter
A=Path(__file__).parent; D=['DS1','DS2','DS3','DS4']; T=.09195067847110018; Z=8.157904927719994
def R(n,k='events'):
 with open(A/f'{n}-{k}.csv') as f:return list(csv.DictReader(f))
def M(n):return json.load(open(A/f'{n}-metrics.json'))
f=p.figure(figsize=(18,12),layout='constrained');g=f.add_gridspec(2,2,height_ratios=[1.3,1]);a=f.add_subplot(g[0,0])
for n,c in zip(D,['#4c78a8','#f58518','#54a24b','#e45756']):
 r=R(n);a.plot([float(x['availability_s']) for x in r],[float(x['multi_score']) for x in r],lw=1.2,label=n,color=c)
a.axvspan(0,90,color='#d9ead3',alpha=.5,label='pre ≤90 s');a.axvspan(90,110,color='#ffe599',alpha=.55,label='transition 90–110 s');a.axvspan(110,500,color='#f4cccc',alpha=.35,label='post ≥110 s');a.axhline(T,color='k',ls='--',label=f'R threshold {T:.8f}');a.set(xlim=(25,500),xlabel='Availability = window end (s)',ylabel='Multi-PRN spoof score R',title='A. Frozen v2 multi-score timelines');a.legend(ncol=2,fontsize=9);a.grid(alpha=.2)
a=f.add_subplot(g[0,1]);x=np.arange(4);w=.22;ms=[M(n) for n in D];sets=[([m['post']['single_alarm_rate'] for m in ms],'post single-PRN tracking fault','#f28e2b'),([m['post']['multi_alarm_rate'] for m in ms],'post FINAL multi-PRN spoof alarm','#d62728'),([m['pre']['single_alarm_rate'] for m in ms],'pre single-PRN tracking fault','#76b7b2'),([m['pre']['multi_alarm_rate'] for m in ms],'pre FINAL multi-PRN spoof alarm','#4e79a7')]
for i,(v,l,c) in enumerate(sets):
 b=a.bar(x+(i-1.5)*w,v,w,label=l,color=c);a.bar_label(b,labels=[f'{q:.2%}' for q in v],rotation=90,fontsize=8,padding=2)
a.set_xticks(x,D);a.yaxis.set_major_formatter(PercentFormatter(1));a.set_ylim(0,max(q for v,_,_ in sets for q in v)*1.35);a.set(title='B. Single-PRN tracking fault ≠ final multi-PRN spoof alarm',ylabel='Alarm rate');a.legend(fontsize=9);a.grid(axis='y',alpha=.2)
a=f.add_subplot(g[1,0]);e=R('clean-calibration');n=R('clean-calibration','nodes');z={int(q['event_index']):float(q['z']) for q in n if q['prn']=='8'};t=np.array([float(q['availability_s']) for q in e]);r=np.array([float(q['multi_score']) for q in e]);q=np.array([z.get(int(v['event_index']),np.nan) for v in e]);h=np.array([v['single_alarm'].lower()=='true' and v['candidate_prn']=='8' for v in e]);a.plot(t,q,color='#9467bd',label='PRN8 standardized score z');a.axhline(Z,color='#9467bd',ls='--',label=f'single τ {Z:.8f}');a.scatter(t[h],q[h],s=60,facecolors='none',edgecolors='red',label=f'PRN8 single spike ({h.sum()})');a.set(title='C. Calibration: PRN8 single spike retained, aggregate R bounded',xlabel='Availability (s)',ylabel='PRN8 z');a.grid(alpha=.2);b=a.twinx();b.plot(t,r,color='#2ca02c',label='aggregate R');b.axhline(T,color='k',ls=':',label=f'R threshold {T:.8f}');b.set_ylabel('Multi score R');u,v=a.get_legend_handles_labels();u2,v2=b.get_legend_handles_labels();a.legend(u+u2,v+v2,fontsize=8)
a=f.add_subplot(g[1,1]);a.axis('off');s='''D. AUDITED VERDICT

Frozen thresholds: τ = 8.157904927719994
R threshold = 0.09195067847110018
Sealed clean FINAL multi: 0 / 119

POST — final multi-PRN alarm / single-PRN tracking fault
DS1  0/702 (0.00%) / 49/702
DS2  138/693 (19.91%), first 111.0 s / 149/693
DS3  37/694 (5.33%), first 120.5 s / 122/694
DS4  0/35 (0.00%) / 11/35

PRE — final multi-PRN alarms all 0
single-PRN tracking fault: DS1 0; DS2 2/119; DS3 1/119; DS4 3/119

Verdict: frozen multi-PRN spoof anomaly occurs on DS2 and DS3.
DS1 and DS4 have no final multi-PRN spoof alarms. Single-PRN tracking
fault flags are diagnostic; they are not final spoof alarms.''';a.text(0,1,s,va='top',fontsize=11.3,linespacing=1.38,bbox=dict(boxstyle='round,pad=.7',facecolor='#f7f7f7',edgecolor='#888'))
f.suptitle('GCMR v2 — frozen cleanStatic→TEXBAT DS1–DS4 dashboard (seed 23)',fontsize=18,fontweight='bold');f.savefig(A/'gcmr_v2_dashboard.png',dpi=180,facecolor='white');print(A/'gcmr_v2_dashboard.png')
