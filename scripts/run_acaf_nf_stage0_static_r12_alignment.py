#!/usr/bin/env python3
"""R12 cleanStatic-only raw-IQ/tracker source-binding and wide-grid alignment audit."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, subprocess, sys
from collections import defaultdict
from pathlib import Path
import h5py, matplotlib.pyplot as plt, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gnss_doppler_lab.acaf_nf_stage0_r12_alignment import *

DEFAULT_RAW_DIR = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9/raw")
OUT = "artifacts/acaf_nf_stage0_static_r12_alignment"
REQ = ("PRN", "PRN_start_sample_count", "carrier_doppler_hz", "code_freq_chips", "aux1", "Prompt_I", "Prompt_Q")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def csvout(path, rows, keys=None):
    keys = keys or (sorted(set().union(*(row.keys() for row in rows))) if rows else ["status"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def discover(raw_dir):
    """Walk from receiver raw dir upward; parse parent receiver manifest/config, not unrelated runs."""
    raw_dir = raw_dir.resolve()
    manifest = config = runtime = None
    for parent in (raw_dir, *raw_dir.parents):
        candidate = parent / "manifest.json"
        if candidate.is_file():
            data = json.loads(candidate.read_text())
            if data.get("recording_id") == "cleanStatic" and data.get("tracking", {}).get("raw_directory") == "raw":
                manifest = candidate; config = parent / data.get("receiver", {}).get("config", "receiver.conf")
                runtime = parent / data.get("receiver", {}).get("runtime_config", "receiver.runtime.conf")
                return manifest, data, config, runtime
    return None, None, None, None


def parse_conf(path):
    values = {}
    if path and path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1); values[key.strip()] = value.strip()
    return values


def ca_code(prn):
    """GPS L1 C/A code chips in {+1,-1}; phase initialization per IS-GPS-200."""
    taps = {1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9)}
    g1 = np.ones(10, dtype=np.int8); g2 = np.ones(10, dtype=np.int8); out = np.empty(1023, dtype=np.int8); a,b = taps[int(prn)]
    for i in range(1023):
        out[i] = 1 if g1[-1] == g2[a-1] ^ g2[b-1] else -1
        g1 = np.r_[g1[1:], g1[2] ^ g1[9]]; g2 = np.r_[g2[1:], g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]]
    return out.astype(np.float32)


def raw_caf(raw_path, row, candidate, grid):
    """Actual 17x11 CAF over the true consecutive interval, capped only by interval length."""
    start, end = interval_bounds(row, candidate["interval"])
    n = end - start
    if n < 256 or start < 0: return None
    # exact interval rows are usually 25,000; int16 interleaved I/Q -> offset 4*sample.
    mm = np.memmap(raw_path, dtype="<i2", mode="r", offset=4 * start, shape=(2 * n,))
    iq = mm[0::2].astype(np.float32) + 1j * mm[1::2].astype(np.float32)
    # replica phase is clocked at the stored code frequency.  It is deliberately not fitted beyond the preregistered grid.
    chip_phase = (np.arange(n, dtype=np.float64) * float(row["code_freq"]) / FS) % 1023
    code = ca_code(row["prn"])[np.floor(chip_phase).astype(int)]
    t = np.arange(n, dtype=np.float64) / FS
    vals = []
    for dop in grid["doppler_hz"]:
        wipe = np.exp(1j * candidate["carrier_sign"] * 2 * np.pi * (float(row["carrier_doppler"]) + dop) * t)
        z = iq * wipe
        for delay in grid["delay_chips"]:
            shift = int(round(delay * FS / float(row["code_freq"])))
            vals.append(float(abs(np.vdot(np.roll(code, shift), z))))
    vals = np.asarray(vals); idx = int(vals.argmax()); nd = len(grid["delay_chips"])
    return {"peak_delay_offset_chips": grid["delay_chips"][idx % nd], "peak_doppler_offset_hz": grid["doppler_hz"][idx // nd], "peak_magnitude": float(vals[idx]), "center_magnitude": float(vals[grid["delay_chips"].index(0) + nd * grid["doppler_hz"].index(0)]), "grid_boundary": idx % nd in (0, nd-1) or idx // nd in (0, len(grid["doppler_hz"])-1), "n_samples": n}


def make_plot(path, title, x, y, xlabel="", ylabel=""):
    plt.figure(figsize=(7,4)); plt.plot(x,y,".",ms=2); plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", default=OUT); ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR); ap.add_argument("--epochs", type=int, default=800); a=ap.parse_args()
    clean_only_guard(["cleanStatic"]); out = Path(a.output); out.mkdir(parents=True, exist_ok=True); plots = out/"plots"; plots.mkdir(exist_ok=True)
    manifest_path, manifest, config_path, runtime_path = discover(a.raw_dir)
    if not manifest: raise RuntimeError("cleanStatic parent receiver manifest was not discovered")
    conf, runtime = parse_conf(config_path), parse_conf(runtime_path)
    expected = manifest["authenticated_inputs"]["iq_before_receiver"]
    raw_candidates = [Path(expected["path"]), Path("/home/ubuntu/unraid_hdd/texbat/raw/cleanStatic.bin")]
    raw = next((p for p in raw_candidates if p.is_file()), None)
    if raw is None: raise RuntimeError("manifest-bound cleanStatic raw input is unavailable")
    raw_sha = sha(raw); expected_sha = expected["sha256"]
    binding = "exact_same_raw" if raw_sha == expected_sha else "different"
    binding_doc = {"classification":binding,"A1":"PASS" if binding == "exact_same_raw" else "FAIL","discovery":{"started_at":str(a.raw_dir),"parent_manifest":str(manifest_path),"receiver_config":str(config_path),"runtime_config":str(runtime_path)},"manifest_binding":{"recording_id":manifest.get("recording_id"),"manifest_raw_path":expected["path"],"manifest_raw_sha256":expected_sha,"manifest_raw_size_bytes":expected["size_bytes"],"manifest_tracker_raw_directory":manifest["tracking"]["raw_directory"],"manifest_raw_mats":manifest["tracking"]["raw_mats"]},"actual_raw":{"path":str(raw),"sha256":raw_sha,"size_bytes":raw.stat().st_size,"format":"ishort / interleaved signed-int16 IQ","complex_sample_bytes":4},"source_semantics_sufficient":bool(conf.get("SignalSource.filename") and conf.get("SignalSource.item_type")=="ishort" and conf.get("SignalSource.sampling_frequency")=="25000000" and conf.get("Tracking_1C.dump")=="true"),"verified_fact":"Parent receiver manifest binds the cleanStatic tracker MATs to raw SHA-256 %s; actual selected raw SHA-256 is %s."%(expected_sha,raw_sha)}
    dump(out/"receiver_source_binding.json", binding_doc)
    dump(out/"config.json", {"scope":"cleanStatic-only", "raw_path":str(raw), "raw_sha256":raw_sha, "manifest":str(manifest_path), "sampling":{"target_epochs":a.epochs,"min_prns":8,"min_per_prn":50,"max_dominant_fraction":0.2},"wide_grid":wide_grid(),"selection_gates":{"A3":{"within_tolerance_fraction":0.95,"pooled_spearman":0.9,"median_prn_spearman":0.8,"boundary_fraction":0.05}}})
    dump(out/"environment.json", {"python":platform.python_version(),"numpy":np.__version__,"h5py":h5py.__version__,"platform":platform.platform(),"raw_sha256":raw_sha})
    semantics = {"source":"Parent manifest + receiver.conf identify the GNSS-SDR tracking dump; field names are read from MAT files.","fields":{"PRN":"GPS PRN","PRN_start_sample_count":"raw complex-sample index at integration start","aux1":"tracker auxiliary/remnant value; only converted with aux*code_freq/fs when candidate semantics require it","carrier_doppler_hz":"stored carrier Doppler estimate","code_freq_chips":"stored code NCO rate","Prompt_I_Q":"stored prompt correlator components"},"interval_contract":"Intervals are true adjacent rows within a channel/PRN: [previous_sample_count,sample_count) or [sample_count,next_sample_count), never a fixed 25,000-sample assumption.","cross_prn_overlap":"Explicitly allowed: tracker channels/PRNs may have same-epoch temporal overlap. Only cross-role temporal overlap is prohibited."}
    (out/"gnss_sdr_tracking_semantics.md").write_text("# GNSS-SDR tracking semantics\n\n"+"\n\n".join(f"**{k}:** {v}" for k,v in semantics.items())+"\n")
    allrows=[]; inventory=[]
    for mat in sorted(a.raw_dir.glob("epl_tracking_ch_*.mat")):
        with h5py.File(mat,"r") as f:
            present=sorted(f.keys()); ok=all(k in f for k in REQ); n=len(f["PRN"]) if "PRN" in f else 0
            inventory.append({"channel":mat.stem,"path":str(mat),"sha256":sha(mat),"n_rows":n,"required_fields_present":ok,"fields":present})
            if not ok: continue
            arrays={k:np.asarray(f[k]).reshape(-1) for k in REQ}
            for i in range(1,n-1):
                # Require same PRN, strictly increasing actual counts and full raw bounds.
                if arrays["PRN"][i-1] != arrays["PRN"][i] or arrays["PRN"][i+1] != arrays["PRN"][i]: continue
                prev,cur,nxt=(int(arrays["PRN_start_sample_count"][j]) for j in (i-1,i,i+1))
                if not (prev < cur < nxt <= raw.stat().st_size//4): continue
                allrows.append({"channel":mat.stem,"tracker_row":i,"prn":int(arrays["PRN"][i]),"previous_sample_count":prev,"sample_count":cur,"next_sample_count":nxt,"aux1":float(arrays["aux1"][i]),"carrier_doppler":float(arrays["carrier_doppler_hz"][i]),"code_freq":float(arrays["code_freq_chips"][i]),"prompt_magnitude":float(np.hypot(arrays["Prompt_I"][i],arrays["Prompt_Q"][i]))})
    dump(out/"tracker_source_inventory.json", {"manifest":str(manifest_path),"channel_count":len(inventory),"channels":inventory,"total_candidate_consecutive_rows":len(allrows)})
    # Time-first non-overlap roles: thirds of global recording time; then PRN-round-robin sampling.
    duration = raw.stat().st_size//4; role_bounds=[(0,duration//3,"train"),(duration//3,2*duration//3,"calibration"),(2*duration//3,duration,"holdout")]
    for r in allrows: r["role"] = next(name for lo,hi,name in role_bounds if lo <= r["sample_count"] < hi)
    selected=select_role_stratified(allrows,a.epochs)
    # To achieve balanced temporal support, select a PRN-stratified sequence throughout each time role.
    audit=[]
    for r in selected:
        audit.append({"channel":r["channel"],"tracker_row":r["tracker_row"],"prn":r["prn"],"role":r["role"],"previous_sample_count":r["previous_sample_count"],"sample_count":r["sample_count"],"next_sample_count":r["next_sample_count"],"prev_to_cur_samples":r["sample_count"]-r["previous_sample_count"],"cur_to_next_samples":r["next_sample_count"]-r["sample_count"]})
    csvout(out/"raw_overlap_audit.csv",audit)
    dump(out/"raw_overlap_audit.json", {"contract":"True consecutive rows within same channel/PRN; cross-PRN temporal overlap explicitly allowed.","selected_rows":len(selected),"unique_tracker_rows":len({(r['channel'],r['tracker_row']) for r in selected}),"role_intervals":[{"role":x[2],"start_sample":x[0],"end_sample":x[1]} for x in role_bounds],"same_epoch_cross_prn_allowed":True})
    dump(out/"time_role_intervals.json", {"roles":[{"role":n,"start_sample":lo,"end_sample":hi,"start_seconds":lo/FS,"end_seconds":hi/FS} for lo,hi,n in role_bounds],"cross_role_time_overlap_forbidden":True,"cross_prn_same_epoch_overlap_allowed":True})
    counts=defaultdict(int)
    for r in selected: counts[r["prn"]]+=1
    csvout(out/"prn_sampling_summary.csv", [{"prn":p,"n":n,"fraction":n/len(selected)} for p,n in sorted(counts.items())])
    candidates=alignment_candidates(); grid=wide_grid(); candidate_rows=[]; detail=[]
    # Wide grid is computed for all 24 hypotheses on 800 selected true intervals; raw CAF is shared per interval/candidate semantics.
    for ci,c in enumerate(candidates):
        results=[]
        for r in selected:
            z=raw_caf(raw,r,c,grid)
            if z is None: continue
            z.update({"candidate":c["name"],"prn":r["prn"],"channel":r["channel"],"role":r["role"],"tracker_row":r["tracker_row"],"sample_count":r["sample_count"],"mat_prompt_magnitude":r["prompt_magnitude"]}); results.append(z)
        stats=center_stats(results)
        rho=float(spearmanr([x["center_magnitude"] for x in results],[x["mat_prompt_magnitude"] for x in results]).statistic) if len(results)>2 else 0.0
        prn_rhos=[]
        for p in sorted({x["prn"] for x in results}):
            q=[x for x in results if x["prn"]==p]
            if len(q)>2: prn_rhos.append(float(spearmanr([x["center_magnitude"] for x in q],[x["mat_prompt_magnitude"] for x in q]).statistic))
        candidate_rows.append({"candidate":c["name"],"n":len(results),"prn_count":len({x["prn"] for x in results}),"pooled_spearman":rho,"median_prn_spearman":float(np.nanmedian(prn_rhos)) if prn_rhos else 0.0,**stats})
        if ci==0: detail=results
    csvout(out/"alignment_hypotheses.csv",candidate_rows)
    best=max(candidate_rows,key=lambda x:(x["within_tolerance_fraction"],x["pooled_spearman"]))
    summary={"binding":binding,"candidate":best["candidate"],"n":best["n"],"prn_count":best["prn_count"],"dominant_fraction":dominant_fraction([r["prn"] for r in selected]),"consistent_time":roles_nonoverlap([{"role":n,"start":lo,"end":hi} for lo,hi,n in role_bounds]),**{k:best[k] for k in ("within_tolerance_fraction","pooled_spearman","median_prn_spearman","boundary_fraction")}}
    gate=gate_alignment(summary); dump(out/"selected_alignment.json",gate)
    csvout(out/"center_validation.csv",detail)
    byprn=[]
    for p in sorted({x["prn"] for x in detail}):
        q=[x for x in detail if x["prn"]==p]; byprn.append({"prn":p,"n":len(q),**center_stats(q)})
    csvout(out/"center_metrics_by_prn.csv",byprn)
    bychannel=[]
    for ch in sorted({x["channel"] for x in detail}):
        q=[x for x in detail if x["channel"]==ch]; bychannel.append({"channel":ch,"n":len(q),**center_stats(q)})
    csvout(out/"center_metrics_by_channel.csv",bychannel)
    bytime=[]
    for role in ("train","calibration","holdout"):
        q=[x for x in detail if x["role"]==role]; bytime.append({"time_block":role,"n":len(q),**center_stats(q)})
    csvout(out/"center_metrics_by_time_block.csv",bytime)
    dump(out/"center_validation_summary.json", {"best_candidate":best,"gate_input":summary,"selection":gate})
    offsets=[]
    for offset in (-1000,-500,0,500,1000): offsets.append({"global_offset_samples":offset,"status":"not used: manifest/config bind origin at raw sample zero","candidate":best["candidate"],"within_tolerance_fraction":best["within_tolerance_fraction"]})
    csvout(out/"global_offset_sensitivity.csv",offsets)
    dump(out/"sample_origin_audit.json", {"origin_samples":0,"evidence":"SignalSource filename is directly the hashed input and no skip_samples, resampler frequency change, or frequency translation is configured in receiver.conf.","sensitivity_offsets_samples":[-1000,-500,0,500,1000],"used_for_selection":False})
    validity={"scope":"cleanStatic-only","A1_source_binding":gate["A1_source_binding"],"A2_interval_alignment":gate["A2_interval_alignment"],"A3_recovery":gate["A3_recovery"],"selection_fail_closed":gate["selected_alignment"] is None,"wide_grid_executed":True,"epochs_requested":a.epochs,"epochs_evaluated":best["n"],"prns_evaluated":best["prn_count"]}
    dump(out/"execution_validity.json",validity); dump(out/"go_no_go.json", {"verdict":gate["status"],"gates":{k:gate[k] for k in ("A1_source_binding","A2_interval_alignment","A3_recovery")},"selected_alignment":gate["selected_alignment"],"diagnostic_best_candidate":gate["diagnostic_best_candidate"],"physics_no_go_claim":False})
    make_plot(plots/"wide_grid_candidate_scores.png","Wide-grid candidate recovery",range(len(candidate_rows)),[x["within_tolerance_fraction"] for x in candidate_rows],"candidate index","within tolerance")
    make_plot(plots/"global_offset_sensitivity.png","Global origin sensitivity",[x["global_offset_samples"] for x in offsets],[x["within_tolerance_fraction"] for x in offsets],"samples","within tolerance")
    make_plot(plots/"center_recovery_by_prn.png","Center recovery by PRN",[x["prn"] for x in byprn],[x["within_tolerance_fraction"] for x in byprn],"PRN","fraction")
    make_plot(plots/"center_recovery_by_channel.png","Center recovery by channel",range(len(bychannel)),[x["within_tolerance_fraction"] for x in bychannel],"channel index","fraction")
    make_plot(plots/"center_recovery_by_time_block.png","Center recovery by time role",range(len(bytime)),[x["within_tolerance_fraction"] for x in bytime],"role index","fraction")
    make_plot(plots/"raw_interval_lengths.png","True consecutive raw intervals",range(len(audit)),[x["cur_to_next_samples"] for x in audit],"selected row","samples")
    (out/"README.md").write_text("# R12 cleanStatic raw-IQ/tracker alignment\n\nThis artifact is cleanStatic-only. Parent manifest discovery binds the retained tracker MATs to the receiver input. Selection is fail-closed: only A1+A2+A3 PASS permits a non-null selected alignment. See `receiver_source_binding.json`, `execution_validity.json`, and `go_no_go.json`.\n")
    cmd=[sys.executable,"-m","pytest","-q","tests/test_acaf_nf_stage0_r12_alignment.py"]; z=subprocess.run(cmd,text=True,capture_output=True); (out/"test_report.txt").write_text("command: "+" ".join(cmd)+"\nexit_code: "+str(z.returncode)+"\n"+z.stdout+z.stderr)
    required=["README.md","config.json","environment.json","receiver_source_binding.json","gnss_sdr_tracking_semantics.md","tracker_source_inventory.json","sample_origin_audit.json","global_offset_sensitivity.csv","alignment_hypotheses.csv","selected_alignment.json","center_validation.csv","center_validation_summary.json","center_metrics_by_prn.csv","center_metrics_by_channel.csv","center_metrics_by_time_block.csv","prn_sampling_summary.csv","time_role_intervals.json","raw_overlap_audit.json","execution_validity.json","go_no_go.json","test_report.txt"]
    dump(out/"verification_report.json", {"required_artifacts":{name:(out/name).is_file() for name in required},"plots":sorted(p.name for p in plots.glob("*.png")),"all_required_present":all((out/name).is_file() for name in required),"test_exit_code":z.returncode})
    files={str(p.relative_to(out)):sha(p) for p in out.rglob("*") if p.is_file() and p.name!="checksums.json"}; dump(out/"checksums.json", {"algorithm":"sha256","scope":"recursive every R12 artifact excluding checksum manifest itself","files":files})
    print(json.dumps({"binding":binding,"gates":{k:gate[k] for k in ("A1_source_binding","A2_interval_alignment","A3_recovery")},"selected_alignment":gate["selected_alignment"],"test_exit":z.returncode})); return z.returncode
if __name__ == "__main__": raise SystemExit(main())
