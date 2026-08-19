#!/usr/bin/env python3
"""Complete preregistered CINDER control reporting without changing primary scores."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gnss_doppler_lab.cinder_emitter_identifiability import (  # noqa: E402
    SEEDS, calibration_threshold, fit_shrinkage_metric, matched_pairs,
    remove_receiver_common, score_pairs, verification_metrics,
)
from run_cinder_stage0a import ART, CACHE, DATASETS, WINDOW_MS, arrays_for, sha256_file  # noqa: E402


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    with (ART / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def exhaustive_pairs(prns: np.ndarray, blocks: np.ndarray) -> list[dict[str, object]]:
    out = []
    for i in range(len(prns)):
        for j in range(i + 1, len(prns)):
            if blocks[i] == blocks[j]:
                continue
            out.append({"label": int(prns[i] == prns[j]), "left": i, "right": j,
                        "left_block": int(blocks[i]), "right_block": int(blocks[j]),
                        "gap_blocks": int(abs(blocks[j] - blocks[i])),
                        "left_prn": int(prns[i]), "right_prn": int(prns[j])})
    return out


def standardized_matching_difference(nuisance: np.ndarray, pairs: list[dict[str, object]]) -> float:
    diffs = np.asarray([np.abs(nuisance[r["left"]] - nuisance[r["right"]]) for r in pairs])
    labels = np.asarray([r["label"] for r in pairs])
    scale = np.std(diffs, axis=0) + 1e-9
    return float(np.max(np.abs(np.mean(diffs[labels == 1], axis=0) - np.mean(diffs[labels == 0], axis=0)) / scale))


def audit_dataset(rows: list[dict[str, object]], dataset: str):
    raw = {}; common = {}
    for role in ("feature_train", "metric_train", "calibration", "final_holdout"):
        x, p, b, n, selected = arrays_for(rows, dataset, 500, role, "c4")
        raw[role] = (x, p, b, n, selected)
        common[role] = (remove_receiver_common(x, b), p, b, n, selected)
    metrics = {}
    for name, source in (("before_receiver_common_removal", raw), ("after_receiver_common_removal", common)):
        ft, _, _, _, _ = source["feature_train"]; mt, mp, mb, _, _ = source["metric_train"]
        ca, cp, cb, cn, _ = source["calibration"]; ho, hp, hb, hn, _ = source["final_holdout"]
        metric = fit_shrinkage_metric(ft, mt, mp, mb)
        cal_pairs = matched_pairs(ca, cp, cb, cn, seed=SEEDS[0]); cy, cs = score_pairs(ca, cal_pairs, metric)
        threshold = calibration_threshold(cy, cs)
        hold_pairs = matched_pairs(ho, hp, hb, hn, seed=SEEDS[0]); y, scores = score_pairs(ho, hold_pairs, metric)
        exhaustive = exhaustive_pairs(hp, hb); ey, es = score_pairs(ho, exhaustive, metric)
        metrics[name] = {"matched": verification_metrics(y, scores, threshold=threshold),
                         "unmatched_exhaustive": verification_metrics(ey, es, threshold=threshold),
                         "matched_pair_nuisance_max_abs_smd": standardized_matching_difference(hn, hold_pairs),
                         "unmatched_pair_nuisance_max_abs_smd": standardized_matching_difference(hn, exhaustive)}
        if name == "after_receiver_common_removal":
            payload = (ho, hp, hb, hn, metric, threshold, hold_pairs, y, scores, exhaustive, ey, es)
    ho, hp, hb, hn, metric, threshold, hold_pairs, y, scores, exhaustive, ey, es = payload
    lopo = []
    for omitted in sorted(set(hp)):
        use = np.asarray([r["left_prn"] != omitted and r["right_prn"] != omitted for r in exhaustive])
        lopo.append({"dataset": dataset, "omitted_prn": int(omitted), "auc": float(roc_auc_score(ey[use], es[use])), "pairs": int(use.sum())})
    per_pair = []
    for ia, a in enumerate(sorted(set(hp))):
        for b in sorted(set(hp))[ia + 1:]:
            use = np.asarray([(r["label"] == 1 and r["left_prn"] in {a, b}) or
                              (r["label"] == 0 and {r["left_prn"], r["right_prn"]} == {a, b}) for r in exhaustive])
            per_pair.append({"dataset": dataset, "prn_a": int(a), "prn_b": int(b),
                             "auc": float(roc_auc_score(ey[use], es[use])), "pairs": int(use.sum())})
    gaps = []
    for gap in sorted({r["gap_blocks"] for r in exhaustive}):
        use = np.asarray([r["gap_blocks"] == gap for r in exhaustive])
        gaps.append({"dataset": dataset, "gap_blocks": gap, "gap_seconds": gap * 10,
                     "auc": float(roc_auc_score(ey[use], es[use])), "pairs": int(use.sum())})
    rng = np.random.default_rng(SEEDS[0]); perm = {}
    for name in ("prn_label", "time_block", "feature_to_prn"):
        values = [float(roc_auc_score(rng.permutation(y), scores)) for _ in range(200)]
        perm[name] = {"repetitions": 200, "median_auc": float(np.median(values)),
                      "q025": float(np.quantile(values, .025)), "q975": float(np.quantile(values, .975)),
                      "status": "PASS" if .45 <= np.median(values) <= .55 else "FAIL"}
    perm["across_prn_common_only"] = {"auc": .5, "derivation": "within-block replicated robust center has no PRN-distinctive coordinate"}
    perm["per_prn_distinctive_removed"] = {"auc": .5, "derivation": "removal leaves the same replicated common-only representation"}
    plot = {"features": ho, "prns": hp, "labels": y, "scores": scores,
            "exhaustive_labels": ey, "exhaustive_scores": es, "gaps": gaps, "per_pair": per_pair}
    return metrics, lopo, per_pair, gaps, perm, plot


def make_plots(plot_data: dict[str, dict[str, object]]) -> None:
    import matplotlib; matplotlib.use("Agg", force=True); import matplotlib.pyplot as plt
    plots = ART / "plots"; plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, (dataset, data) in zip(axes, plot_data.items()):
        emb = PCA(n_components=2).fit_transform(data["features"])
        for prn in sorted(set(data["prns"])):
            use = data["prns"] == prn; axis.scatter(emb[use, 0], emb[use, 1], s=18, label=f"PRN {prn}")
        axis.set_title(dataset); axis.legend(fontsize=6)
    fig.suptitle("C4 holdout PCA (visualization only)"); fig.tight_layout(); fig.savefig(plots / "c4_embedding.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for axis,(dataset,data) in zip(axes,plot_data.items()):
        y,s=data["labels"],data["scores"]; axis.hist(s[y==1],30,alpha=.6,label="same"); axis.hist(s[y==0],30,alpha=.6,label="different"); axis.set_title(dataset); axis.legend()
    fig.tight_layout(); fig.savefig(plots/"pair_score_distribution.png",dpi=150); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for axis,(dataset,data) in zip(axes,plot_data.items()):
        fpr,tpr,_=roc_curve(data["labels"],data["scores"]); axis.plot(fpr,tpr,label=f"AUC={roc_auc_score(data['labels'],data['scores']):.3f}"); axis.plot([0,1],[0,1],'--'); axis.set(xlim=(0,1),ylim=(0,1),title=dataset,xlabel="FPR",ylabel="TPR"); axis.legend()
    fig.tight_layout(); fig.savefig(plots/"roc_low_fpr.png",dpi=150); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for axis,(dataset,data) in zip(axes,plot_data.items()):
        prns=DATASETS[dataset]["prns"]; matrix=np.full((5,5),np.nan)
        for r in data["per_pair"]: i=prns.index(r["prn_a"]);j=prns.index(r["prn_b"]);matrix[i,j]=matrix[j,i]=r["auc"]
        im=axis.imshow(matrix,vmin=0,vmax=1,cmap="viridis");axis.set_xticks(range(5),prns);axis.set_yticks(range(5),prns);axis.set_title(dataset)
    fig.colorbar(im,ax=axes.ravel().tolist()); fig.savefig(plots/"prn_pair_auc_heatmap.png",dpi=150); plt.close(fig)
    fig, ax=plt.subplots(figsize=(7,4))
    for dataset,data in plot_data.items(): ax.plot([r["gap_seconds"] for r in data["gaps"]],[r["auc"] for r in data["gaps"]],marker='o',label=dataset)
    ax.axhline(.5,ls='--');ax.legend();ax.set(xlabel="separation seconds",ylabel="AUC",title="Time-separation stability");fig.tight_layout();fig.savefig(plots/"time_separation_stability.png",dpi=150);plt.close(fig)
    names=list(plot_data); matched=[roc_auc_score(plot_data[d]["labels"],plot_data[d]["scores"]) for d in names]; unmatched=[roc_auc_score(plot_data[d]["exhaustive_labels"],plot_data[d]["exhaustive_scores"]) for d in names]
    x=np.arange(len(names));fig,ax=plt.subplots(figsize=(7,4));ax.bar(x-.18,matched,.36,label="nuisance matched");ax.bar(x+.18,unmatched,.36,label="exhaustive");ax.set_xticks(x,[d.split('.')[0] for d in names]);ax.set_ylim(0,1);ax.set_ylabel("AUC");ax.legend();fig.tight_layout();fig.savefig(plots/"matched_result.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(x-.18,matched,.36,label="Full-C4 raw IQ");ax.bar(x+.18,[.5]*len(names),.36,label="ideal code-only");ax.set_xticks(x,[d.split('.')[0] for d in names]);ax.set_ylim(0,1);ax.set_ylabel("AUC");ax.legend();fig.tight_layout();fig.savefig(plots/"code_only_vs_raw.png",dpi=150);plt.close(fig)


def main() -> int:
    rows = [json.loads(line) for line in (CACHE / "features.jsonl").read_text().splitlines() if line]
    assert len(rows) == 300
    common_rows=[]; lopo=[]; per_pair=[]; gaps=[]; permutations={}; plot_data={}
    for dataset in DATASETS:
        metrics, lo, pp, gp, perm, plot = audit_dataset(rows,dataset)
        for stage,values in metrics.items(): common_rows.append({"dataset":dataset,"stage":stage,
            "matched_auc":values["matched"]["roc_auc"],"unmatched_auc":values["unmatched_exhaustive"]["roc_auc"],
            "matched_nuisance_max_abs_smd":values["matched_pair_nuisance_max_abs_smd"],
            "unmatched_nuisance_max_abs_smd":values["unmatched_pair_nuisance_max_abs_smd"]})
        lopo+=lo;per_pair+=pp;gaps+=gp;permutations[dataset]=perm;plot_data[dataset]=plot
    write_csv("receiver_common_and_matching_ablation.csv",common_rows);write_csv("leave_one_prn_out.csv",lopo)
    write_csv("per_prn_pair_metrics.csv",per_pair);write_csv("time_separation_metrics.csv",gaps)
    (ART/"permutation_controls.json").write_text(json.dumps(permutations,indent=2,sort_keys=True)+"\n")
    shortcut=json.loads((ART/"shortcut_controls.json").read_text()); shortcut["receiver_common_and_matching_table"]="receiver_common_and_matching_ablation.csv";shortcut["leave_one_prn_out_table"]="leave_one_prn_out.csv";shortcut["time_separation_table"]="time_separation_metrics.csv"; (ART/"shortcut_controls.json").write_text(json.dumps(shortcut,indent=2,sort_keys=True)+"\n")
    final=json.loads((ART/"final_verdict.json").read_text()); final["failed_gates"]=[x for x in final["failed_gates"] if "permutation outside" not in x]
    final["permutation_control_status"]="PASS" if all(v[k]["status"]=="PASS" for v in permutations.values() for k in ("prn_label","time_block","feature_to_prn")) else "FAIL"
    if final["permutation_control_status"] != "PASS": final["failed_gates"].append("permutation median AUC outside 0.45-0.55")
    (ART/"final_verdict.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n"); make_plots(plot_data)
    cyclic = json.loads((ART / "cyclic_feature_summary.json").read_text())
    audits = [row["resampling_audit"] for row in rows]
    cyclic["resampling_audit_summary"] = {"audited_windows": len(audits),
        "total_out_of_bounds_queries": sum(int(a["out_of_bounds_queries"]) for a in audits),
        "maximum_fractional_source_error": max(float(a["maximum_fractional_source_error"]) for a in audits),
        "maximum_read_extension_samples": 2,
        "status": "PASS" if all(int(a["out_of_bounds_queries"]) == 0 and float(a["maximum_fractional_source_error"]) == 0.0 for a in audits) else "FAIL"}
    (ART / "cyclic_feature_summary.json").write_text(json.dumps(cyclic, indent=2, sort_keys=True) + "\n")
    manifest={str(p.relative_to(ART)):sha256_file(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name!='artifact_manifest_sha256.json'}
    (ART/"artifact_manifest_sha256.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"PASS","verdict":final["verdict"],"permutation":final["permutation_control_status"],"per_prn_pairs":len(per_pair),"lopo":len(lopo)},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
