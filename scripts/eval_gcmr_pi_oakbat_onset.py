#!/usr/bin/env python3
"""Onset-aware, attack-score-only evaluation for frozen GCMR-PI score CSVs."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def rows(path: Path):
    with path.open(newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def metrics(items, score_name, threshold, onset=120., guard=10.):
    t = np.asarray([x['time'] for x in items])
    s = np.asarray([x[score_name] for x in items])
    pre = t < onset - guard
    post = t >= onset + guard
    keep = pre | post
    y = post[keep].astype(int)
    x = s[keep]
    alarm = s > threshold
    post_alarm = alarm & post
    first = t[post_alarm][0] if post_alarm.any() else None
    return {
        'threshold': float(threshold), 'onset_s': onset, 'guard_s': guard,
        'pre_events': int(pre.sum()), 'pre_fpr': float(alarm[pre].mean()) if pre.any() else None,
        'post_events': int(post.sum()), 'post_detection_rate': float(alarm[post].mean()) if post.any() else None,
        'first_alarm_delay_s': None if first is None else float(first-onset),
        'persistence': float(alarm[post].mean()) if post.any() else None,
        'roc_auc': float(roc_auc_score(y, x)) if len(np.unique(y)) == 2 else None,
        'pr_auc': float(average_precision_score(y, x)) if y.any() else None,
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('out',type=Path); p.add_argument('--onset',type=float,default=120.); p.add_argument('--guard',type=float,default=10.)
    a=p.parse_args(); thresholds=json.loads((a.out/'thresholds.json').read_text())
    result={'contract':{'attack_fit':False,'onset_s':a.onset,'guard_s':a.guard,'pre':'time < onset-guard','post':'time >= onset+guard'},'scenarios':{}}
    for name in ('os1','os2','os3','os4'):
        data=rows(a.out/f'{name}_scores.csv'); record={}
        for ab in ('A0','A1','A2','A3','A4','Full'):
            record[ab]={q:metrics(data,ab, thresholds[ab][q],a.onset,a.guard) for q in ('q99','q995','FPR1')}
        sp=np.array([r['S_pair'] for r in data]); destroyed=np.array([r['relation_destruction'] for r in data])
        record['relation_destruction']={'events':len(data),'s_pair_mean':float(sp.mean()),'destroyed_pair_mean':float(destroyed.mean()),'mean_delta':float((sp-destroyed).mean()),'fraction_decreased':float((destroyed<sp).mean())}
        result['scenarios'][name]=record
    (a.out/'onset_metrics.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
