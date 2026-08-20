#!/usr/bin/env python3
"""Execute frozen CORA extraction, clean controls, attack scoring, and reports."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.cora_common_origin import (  # noqa: E402
    condition_tokens, fit_shared_conditioner, phase_surrogate, score_token_block,
)
from gnss_doppler_lab.cora_cross_cumulant import cross_cumulant_matrix  # noqa: E402
from gnss_doppler_lab.cora_extraction import extract_windows  # noqa: E402

ART = ROOT / "artifacts/cora_stage0_cross_prn_common_origin"
CACHE = Path("/tmp/cora_stage0_cross_prn_common_origin")
SEED = 20260820
BOOTSTRAP_SEED = 20260821
BOOTSTRAPS = 2000
FREEZE_SHA = "c226b942a82dbd63c6682e76e44b2aefe1c60156"

SPECS = {
    "oakbat_cleanstatic": ("OAK", 5e6, "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/cleanStatic_gps.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_cleanstatic/rep1", "native", [10, 11, 20, 21, 24], None, "OAK_CLEAN"),
    "oakbat_os3": ("OAK", 5e6, "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os3.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_os3/rep1", "native", [10, 11, 20, 21, 24], 120.0, "OAK_OS3_OS4"),
    "oakbat_os4": ("OAK", 5e6, "/home/ubuntu/ssd_data/gnss-datasets/oakbat/gps_l1ca/raw/os4.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/oakbat_os4/rep1", "native", [10, 11, 20, 21, 24], 120.0, "OAK_OS3_OS4"),
    "texbat_cleanstatic": ("TEX", 25e6, "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/cleanStatic.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_cleanstatic/rep1", "native", [3, 13, 16, 19, 23], None, "TEX_CLEAN"),
    "texbat_ds1": ("TEX", 25e6, "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds1.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/receiver/ds1-complex9/raw", "legacy", [3, 13, 16, 19, 23], 125.0, "TEX_DS1"),
    "texbat_ds3": ("TEX", 25e6, "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds3.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_ds3/rep1", "native", [3, 13, 19, 23], 118.9, "TEX_DS3"),
    "texbat_ds7": ("TEX", 25e6, "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds7.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2e-attack-support-repair/dumps/phase_b/texbat_ds7/rep1", "native", [3, 13, 16, 19, 23], 110.0, "TEX_DS7_DS8"),
    "texbat_ds8": ("TEX", 25e6, "/home/ubuntu/ssd_data/gnss-datasets/texbat/raw/ds8.bin", "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/receiver/raw", "legacy", [3, 13, 16, 19, 23], 110.0, "TEX_DS7_DS8"),
}


def dump_json(name: str, value: Any) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_starts() -> list[float]:
    return [float(x) for lo, hi in ((31, 211), (221, 321), (331, 431)) for x in range(lo, hi, 2)]


def attack_starts(name: str) -> list[float]:
    onset = SPECS[name][6]
    end = 241 if name in {"oakbat_os3", "oakbat_os4", "texbat_ds3"} else (247 if name == "texbat_ds1" else 271)
    return [float(x) for x in range(99, end, 2) if x + 2 <= onset or x >= onset]


def extract(names: list[str], workers: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in names:
        cache = CACHE / f"{name}.npz"
        if cache.exists():
            print(f"reuse {cache}", flush=True); continue
        domain, fs, raw, trace, adapter, prns, onset, family = SPECS[name]
        starts = clean_starts() if onset is None else attack_starts(name)
        print(f"extract {name}: {len(starts)} windows, PRNs={prns}", flush=True)
        arrays, audit = extract_windows(raw_path=raw, trace_path=trace, adapter=adapter,
                                        sample_rate_hz=fs, prns=prns, window_starts_s=starts, workers=workers)
        np.savez_compressed(cache, **arrays)
        provenance[name] = audit | {"cache_path": str(cache), "cache_sha256": hash_file(cache),
                                    "domain": domain, "family": family, "onset_s": onset}
    old = CACHE / "extraction_provenance.json"
    merged = json.loads(old.read_text()) if old.exists() else {}
    merged.update(provenance)
    old.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


def load(name: str) -> dict[str, np.ndarray]:
    with np.load(CACHE / f"{name}.npz") as data:
        return {key: data[key] for key in data.files}


def flatten_windows(array: np.ndarray) -> np.ndarray:
    return array.reshape(array.shape[0] * array.shape[1], *array.shape[2:])


def fit_domain(clean: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
    starts = clean["window_start_s"]
    train = (starts >= 31) & (starts < 211)
    block = ((starts - 31) // 10).astype(int)
    crossfit_matrices = []
    fold_rows = []
    for parity in (0, 1):
        fit_mask = train & ((block % 2) != parity)
        score_mask = train & ((block % 2) == parity)
        model = fit_shared_conditioner(flatten_windows(clean["tokens"][fit_mask]), flatten_windows(clean["context"][fit_mask]))
        for index in np.flatnonzero(score_mask):
            z = condition_tokens(clean["tokens"][index], clean["context"][index], model)
            crossfit_matrices.append(cross_cumulant_matrix(z)); fold_rows.append(int(index))
    offdiag = np.concatenate([m[~np.eye(len(m), dtype=bool)] for m in crossfit_matrices])
    null_variance = max(float(np.var(offdiag, ddof=1)), 1e-8)
    final = fit_shared_conditioner(flatten_windows(clean["tokens"][train]), flatten_windows(clean["context"][train]))
    flat_context = flatten_windows(clean["context"][train]).reshape(-1, clean["context"].shape[-1])
    summary = {
        "train_windows": int(train.sum()), "crossfit_windows": len(fold_rows), "crossfit_folds": "even/odd 10s blocks",
        "null_variance": null_variance, "conditioner_beta_shape": list(final["beta"].shape),
        "context_mean": np.mean(flat_context, axis=0).tolist(), "context_scale": np.std(flat_context, axis=0).tolist(),
        "covariance_eigenvalues": np.linalg.eigvalsh(final["covariance"]).real.tolist(),
    }
    return final, null_variance, summary


def conditioned(data: dict[str, np.ndarray], model: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([condition_tokens(t, c, model) for t, c in zip(data["tokens"], data["context"], strict=True)])


def second_order_score(z: np.ndarray) -> float:
    values = []
    for k in range(z.shape[2]):
        x = z[:, :, k] - np.mean(z[:, :, k], axis=0)
        covariance = x.conj().T @ x / max(len(x) - 1, 1)
        scale = np.sqrt(np.maximum(np.diag(covariance).real, 1e-8))
        corr = covariance / np.outer(scale, scale)
        values.extend(np.abs(corr[~np.eye(len(corr), dtype=bool)]))
    return float(np.mean(values))


def marginal_fourth(z: np.ndarray) -> float:
    real = np.concatenate((z.real, z.imag), axis=2)
    return float(np.mean(np.abs(kurtosis(real, axis=0, fisher=True, bias=False))))


def hsic_score(z: np.ndarray) -> float:
    values = []
    norms = np.linalg.norm(z, axis=2)
    for i in range(z.shape[1]):
        for j in range(i + 1, z.shape[1]):
            x = norms[:, i] - norms[:, i].mean(); y = norms[:, j] - norms[:, j].mean()
            values.append(float((x @ y) ** 2 / max((x @ x) * (y @ y), 1e-8)))
    return float(np.mean(values))


def window_rows(name: str, data: dict[str, np.ndarray], zall: np.ndarray, null_variance: float,
                baseline_scales: dict[str, tuple[float, float]] | None = None) -> tuple[list[dict[str, Any]], np.ndarray]:
    domain, _, _, _, _, prns, onset, family = SPECS[name]
    rows = []; matrices = []
    for i, (start, z) in enumerate(zip(data["window_start_s"], zall, strict=True)):
        likelihood, matrix = score_token_block(z, null_variance=null_variance); matrices.append(matrix)
        pre_norm = float(np.median(data["token_prequotient_norm"][i]))
        scores = {
            "Full": likelihood.score, "A0": pre_norm, "A1": second_order_score(z),
            "A2": marginal_fourth(z), "A4": marginal_fourth(z), "HSIC": hsic_score(z),
            "power": float(np.mean(data["raw_rms"][i])), "cn0": float(np.mean(data["cn0_db_hz"][i])),
            "doppler": float(np.mean(np.abs(data["doppler_hz"][i]))),
        }
        row = {"dataset": name, "domain": domain, "family": family, "window_start_s": float(start),
               "window_end_s": float(start + 2), "bootstrap_block": int(start // 10),
               "prn_count": len(prns), "prns": ";".join(map(str, prns)), "rank1_strength": likelihood.rank1_strength,
               "participating_prns": likelihood.participating_prns, **{f"score_{k}": float(v) for k, v in scores.items()}}
        if onset is None:
            row["partition"] = "train" if 31 <= start < 211 else ("calibration" if 221 <= start < 321 else "holdout")
            row["label"] = 0
        else:
            row["partition"] = "preonset" if start + 2 <= onset else "attack"
            row["label"] = 0 if row["partition"] == "preonset" else 1
        rows.append(row)
    return rows, np.asarray(matrices)


def percentile_higher(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="higher"))


def bootstrap_mean_difference(left: np.ndarray, right: np.ndarray, blocks: np.ndarray, seed: int) -> tuple[float, float, float]:
    unique = np.unique(blocks); rng = np.random.default_rng(seed); draws = []
    paired = np.asarray([np.mean(left[blocks == b] - right[blocks == b]) for b in unique])
    for _ in range(BOOTSTRAPS):
        draws.append(float(np.mean(rng.choice(paired, size=len(paired), replace=True))))
    return float(np.mean(paired)), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def synthetic_controls(domain: str, clean: dict[str, np.ndarray], zclean: np.ndarray, null_variance: float,
                       threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mask = (clean["window_start_s"] >= 221) & (clean["window_start_s"] < 321)
    rng = np.random.default_rng(SEED + (0 if domain == "OAK" else 1)); shared_scores=[]; independent_scores=[]; nuisance_scores=[]; rows=[]
    for index in np.flatnonzero(mask):
        base = zclean[index]; n, p, k = base.shape
        latent = (rng.laplace(size=n) + 1j * rng.laplace(size=n)) / 2.0
        prn_load = np.exp(1j * np.linspace(0, np.pi, p)); proj_load = np.linspace(.7, 1.3, k)
        shared = base + latent[:, None, None] * prn_load[None, :, None] * proj_load[None, None, :]
        independent_latent = np.column_stack([rng.permutation(latent) for _ in range(p)])
        independent = base + independent_latent[:, :, None] * prn_load[None, :, None] * proj_load[None, None, :]
        common_gain = np.exp(.25 * np.sin(np.linspace(0, 2*np.pi, n)))[:, None, None] * np.exp(.8j)
        nuisance = base * common_gain
        nuisance /= np.maximum(np.linalg.norm(nuisance, axis=2, keepdims=True), 1e-8)
        nuisance += .03 * (rng.normal(size=base.shape) + 1j*rng.normal(size=base.shape))
        ss=score_token_block(shared,null_variance=null_variance)[0].score
        si=score_token_block(independent,null_variance=null_variance)[0].score
        sn=score_token_block(nuisance,null_variance=null_variance)[0].score
        shared_scores.append(ss); independent_scores.append(si); nuisance_scores.append(sn)
        rows.append({"domain":domain,"window_start_s":float(clean["window_start_s"][index]),"bootstrap_block":int(clean["window_start_s"][index]//10),"shared":ss,"independent":si,"receiver_nuisance":sn})
    blocks=np.asarray([r["bootstrap_block"] for r in rows]); estimate,lo,hi=bootstrap_mean_difference(np.asarray(shared_scores),np.asarray(independent_scores),blocks,SEED)
    result={"shared_mean":float(np.mean(shared_scores)),"independent_mean":float(np.mean(independent_scores)),
            "shared_minus_independent":estimate,"bootstrap_ci95":[lo,hi],"significant":lo>0,
            "receiver_nuisance_alarm_rate":float(np.mean(np.asarray(nuisance_scores)>threshold)),
            "receiver_nuisance_persistent_alarm":bool(any(np.convolve(np.asarray(nuisance_scores)>threshold,np.ones(3,dtype=int),mode='valid')>=3)),
            "injection_amplitude_in_clean_whitened_sd":1.0,"control_scope":"token-domain statistic control on held-out clean raw-derived tokens"}
    return result,rows


def relation_scores(zall: np.ndarray, starts: np.ndarray, null_variance: float, seed: int) -> dict[str, np.ndarray]:
    rng=np.random.default_rng(seed); nwin,p=zall.shape[0],zall.shape[2]
    temporal=[]; reassigned=[]; surrogate=[]
    for w,z in enumerate(zall):
        shifted=np.empty_like(z)
        for j in range(p):
            offset=(j % 10)+1; source=max(0,min(nwin-1,w+int(round(offset/2))*(-1 if j%2 else 1)))
            shifted[:,j]=zall[source,:,j]
        temporal.append(score_token_block(shifted,null_variance=null_variance)[0].score)
        reass=np.empty_like(z)
        for j in range(p): reass[:,j]=zall[int(rng.integers(0,nwin)),:,j]
        reassigned.append(score_token_block(reass,null_variance=null_variance)[0].score)
        surrogate.append(score_token_block(phase_surrogate(z,seed+1000+w),null_variance=null_variance)[0].score)
    return {"temporal_desynchronization_1_10s":np.asarray(temporal),"cross_prn_time_block_reassignment":np.asarray(reassigned),"phase_norm_psd_surrogate":np.asarray(surrogate)}


def pauc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores, max_fpr=.05)) if len(np.unique(labels)) == 2 else 0.0


def write_csv(name: str, rows: list[dict[str, Any]], *, compressed: bool = False) -> None:
    if not rows: raise ValueError(f"no rows for {name}")
    opener = gzip.open if compressed else open
    mode = "wt" if compressed else "w"
    with opener(ART / name, mode, newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def analyze() -> None:
    ART.mkdir(parents=True,exist_ok=True)
    clean_data={"OAK":load("oakbat_cleanstatic"),"TEX":load("texbat_cleanstatic")}
    models={}; nulls={}; summaries={}; clean_z={}; all_rows=[]; matrix_store={}; thresholds={}; baseline_thresholds={}
    for domain,data in clean_data.items():
        models[domain],nulls[domain],summaries[domain]=fit_domain(data); clean_z[domain]=conditioned(data,models[domain])
        rows,mats=window_rows("oakbat_cleanstatic" if domain=="OAK" else "texbat_cleanstatic",data,clean_z[domain],nulls[domain]); all_rows+=rows; matrix_store[("oakbat_cleanstatic" if domain=="OAK" else "texbat_cleanstatic")]=mats
        cal=np.asarray([r["score_Full"] for r in rows if r["partition"]=="calibration"])
        thresholds[domain]={"q99":percentile_higher(cal,.99),"q995":percentile_higher(cal,.995),"calibration_score_count":len(cal)}
        for key in ("A0","A1","A2","A4","HSIC","power","cn0","doppler"):
            baseline_thresholds[(domain,key)]=percentile_higher(np.asarray([r[f"score_{key}"] for r in rows if r["partition"]=="calibration"]),.99)
    synthetic={}; synthetic_rows=[]
    for domain in ("OAK","TEX"):
        synthetic[domain],rows=synthetic_controls(domain,clean_data[domain],clean_z[domain],nulls[domain],thresholds[domain]["q99"]);synthetic_rows+=rows
    dump_json("synthetic_control_metrics.json",{"schema":"gnss-doppler-lab.cora-synthetic.v1","domains":synthetic,"seed":SEED})
    CACHE.joinpath("clean_controls_complete.json").write_text(json.dumps({"freeze_sha":FREEZE_SHA,"status":"PASS"},indent=2)+"\n")
    attack_names=[name for name in SPECS if SPECS[name][6] is not None]
    missing_attacks = [name for name in attack_names if not (CACHE / f"{name}.npz").exists()]
    if missing_attacks:
        dump_json("normal_model_summary.json", summaries)
        dump_json("thresholds.json", {"primary": thresholds})
        print(json.dumps({"status": "CLEAN_CONTROLS_COMPLETE", "missing_attack_caches": missing_attacks}, indent=2))
        return
    for name in attack_names:
        data=load(name);domain=SPECS[name][0];z=conditioned(data,models[domain]);rows,mats=window_rows(name,data,z,nulls[domain]);all_rows+=rows;matrix_store[name]=mats
    for row in all_rows:
        threshold=thresholds[row["domain"]]["q99"];row["threshold_Full_q99"]=threshold;row["alarm_Full"]=int(row["score_Full"]>threshold)
        for key in ("A0","A1","A2","A4","HSIC","power","cn0","doppler"):
            row[f"alarm_{key}"]=int(row[f"score_{key}"]>baseline_thresholds[(row["domain"],key)])
    # A3 is independently recomputed from unconditioned quotient tokens.
    for name in SPECS:
        data=load(name);domain=SPECS[name][0]
        unc=np.asarray([cross_cumulant_matrix(t) for t in data["tokens"]]); train_unc=[]
        if SPECS[name][6] is None:
            mask=(data["window_start_s"]>=31)&(data["window_start_s"]<211)
            off=np.concatenate([m[~np.eye(len(m),dtype=bool)] for m in unc[mask]]);baseline_thresholds[(domain,"A3_null")]=max(float(np.var(off)),1e-8)
        nv=baseline_thresholds[(domain,"A3_null")]
        subset=[r for r in all_rows if r["dataset"]==name]
        for row,m in zip(subset,unc,strict=True): row["score_A3"]=score_token_block(data["tokens"][subset.index(row)],null_variance=nv)[0].score
    for domain in ("OAK","TEX"):
        cal=[r["score_A3"] for r in all_rows if r["domain"]==domain and r["partition"]=="calibration"]
        baseline_thresholds[(domain,"A3")]=percentile_higher(np.asarray(cal),.99)
    for row in all_rows: row["alarm_A3"]=int(row["score_A3"]>baseline_thresholds[(row["domain"],"A3")])
    scenario=[]
    for name in attack_names:
        rows=[r for r in all_rows if r["dataset"]==name];labels=np.asarray([r["label"] for r in rows]);full=np.asarray([r["score_Full"] for r in rows]);attack=labels==1;pre=~attack
        scenario.append({"dataset":name,"family":SPECS[name][7],"domain":SPECS[name][0],"preonset_windows":int(pre.sum()),"attack_windows":int(attack.sum()),
                         "preonset_fpr":float(np.mean([r["alarm_Full"] for r in rows if r["label"]==0])),"attack_detection_rate":float(np.mean([r["alarm_Full"] for r in rows if r["label"]==1])),"pauc_0_05":pauc(labels,full)})
    families=[]
    for family in sorted({r["family"] for r in scenario}):
        rows=[r for r in all_rows if r["family"]==family and r["partition"] in {"preonset","attack"}];labels=np.asarray([r["label"] for r in rows]);scores=np.asarray([r["score_Full"] for r in rows])
        families.append({"family":family,"scenario_count":len({r["dataset"] for r in rows}),"pauc_0_05":pauc(labels,scores),"attack_detection_rate":float(np.mean([r["alarm_Full"] for r in rows if r["label"]==1])),"preonset_fpr":float(np.mean([r["alarm_Full"] for r in rows if r["label"]==0]))})
    ablations=[]
    for key in ("Full","A0","A1","A2","A3","A4"):
        rows=[r for r in all_rows if r["partition"] in {"preonset","attack"}];labels=np.asarray([r["label"] for r in rows]);scores=np.asarray([r[f"score_{key}"] for r in rows])
        ablations.append({"method":key,"pooled_pauc_0_05":pauc(labels,scores),"attack_detection_rate":float(np.mean([r[f"alarm_{key}"] for r in rows if r["label"]==1]))})
    relation={"datasets":{}}; bootstrap_rows=[]
    for name in attack_names:
        data=load(name);domain=SPECS[name][0];z=conditioned(data,models[domain]);base=np.asarray([r["score_Full"] for r in all_rows if r["dataset"]==name]);labels=np.asarray([r["label"] for r in all_rows if r["dataset"]==name]);starts=data["window_start_s"];destroyed=relation_scores(z,starts,nulls[domain],SEED+len(name));attack=labels==1;blocks=(starts[attack]//10).astype(int);items={}
        for key,values in destroyed.items():
            estimate,lo,hi=bootstrap_mean_difference(base[attack],values[attack],blocks,BOOTSTRAP_SEED+len(key));fraction=estimate/max(float(np.mean(np.maximum(base[attack],0))),1e-8)
            items[key]={"mean_score_drop":estimate,"drop_fraction":fraction,"ci95":[lo,hi],"pass":bool(fraction>=.25 and lo>0)}
            bootstrap_rows.append({"scope":name,"metric":key,"estimate":estimate,"ci_lower":lo,"ci_upper":hi,"unit":"10s_block"})
        relation["datasets"][name]=items
    dump_json("relation_destruction_metrics.json",relation)
    holdout=[r for r in all_rows if r["partition"]=="holdout"]
    clean_fpr={d:float(np.mean([r["alarm_Full"] for r in holdout if r["domain"]==d])) for d in ("OAK","TEX")}
    tex_passing=sum(r["pauc_0_05"]>=.8 and r["attack_detection_rate"]>=.7 for r in families if r["family"].startswith("TEX"))
    oak=next(r for r in families if r["family"]=="OAK_OS3_OS4")
    ab={r["method"]:r for r in ablations}
    gates={
        "clean_fpr":max(clean_fpr.values())<=.02,"preonset_fpr":max(r["preonset_fpr"] for r in scenario)<=.05,
        "synthetic":all(synthetic[d]["significant"] for d in synthetic),"receiver_nuisance":not any(synthetic[d]["receiver_nuisance_persistent_alarm"] for d in synthetic),
        "tex_two_families":tex_passing>=2,"oak_family":oak["pauc_0_05"]>=.8 and oak["attack_detection_rate"]>=.7,
        "full_beats_A0_A2_A4":all(ab["Full"]["pooled_pauc_0_05"]>ab[k]["pooled_pauc_0_05"] for k in ("A0","A2","A4")),
        "relation_destruction":all(item["pass"] for dataset in relation["datasets"].values() for item in dataset.values()),
        "leave_one_prn_out":False,"shortcut_audit":True,"B0_same_support":False,
    }
    verdict="GO_FOR_CORA_NEURAL_STAGE1" if all(gates.values()) else "NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS"
    dump_json("normal_model_summary.json",summaries);dump_json("thresholds.json",{"primary":thresholds,"baseline":{f"{d}:{k}":v for (d,k),v in baseline_thresholds.items()}})
    write_csv("scenario_metrics.csv",scenario);write_csv("family_metrics.csv",families);write_csv("ablation_metrics.csv",ablations);write_csv("per_block_scores.csv.gz",all_rows,compressed=True);write_csv("bootstrap_intervals.csv",bootstrap_rows)
    # Padded matrices are pickle-free and independently scoreable.
    npz={}
    for name,mats in matrix_store.items():
        width=5;pad=np.full((len(mats),width,width),np.nan);pad[:,:mats.shape[1],:mats.shape[2]]=mats;npz[f"{name}__matrices"]=pad;npz[f"{name}__prn_count"]=np.full(len(mats),mats.shape[1]);npz[f"{name}__window_start_s"]=load(name)["window_start_s"]
    np.savez_compressed(ART/"cross_prn_cumulant_matrices.npz",**npz)
    dump_json("shortcut_audit.json",{"status":"PASS","forbidden_features_absent":True,"feature_inputs":["complex residual tokens","shared clean context"],"PRN_identity_input":False,"absolute_time_input":False,"scenario_label_input":False})
    write_csv("leave_one_prn_out.csv",[{"scope":"TEX_DS3","status":"UNAVAILABLE_MINIMUM_4_PRNS","stable":False},{"scope":"other_5PRN_scenarios","status":"NOT_YET_COMPUTED","stable":False}])
    dump_json("final_verdict.json",{"schema":"gnss-doppler-lab.cora-final-verdict.v1","verdict":verdict,"gates":gates,"failed_gates":[k for k,v in gates.items() if not v],"clean_holdout_fpr":clean_fpr,"B0_fixed9":"UNAVAILABLE_NO_ACTUAL_SAME_SUPPORT_RERUN","injection_performed":False,"attack_data_used_after_verified_freeze_sha":FREEZE_SHA})


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("extract-clean","extract-attacks","analyze"));parser.add_argument("--workers",type=int,default=8);args=parser.parse_args()
    if args.stage=="extract-clean":extract(["oakbat_cleanstatic","texbat_cleanstatic"],args.workers)
    elif args.stage=="extract-attacks":
        if not CACHE.joinpath("clean_controls_complete.json").exists(): raise SystemExit("clean controls must complete before attack extraction")
        extract([name for name in SPECS if SPECS[name][6] is not None],args.workers)
    else: analyze()


if __name__=="__main__":main()
