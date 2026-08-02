#!/usr/bin/env python3
"""Source-only DS8 GNSS-SDR receiver and complex 9-tap export.

No model, residual, conformal, detector, threshold, or metric code is imported
or executed here.  Failure is explicit and leaves primary DS8 unavailable.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.gnss_sdr import export_tracking_csv, parse_acquired_prns, parse_receiver_reported_prns

RAW_SHA="1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78"
EXEC_SHA="6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb"
EXPORTER_SHA="30a45f988cec15fdce84552ff30747b472c7d76df07d93f79d6ae236166d4039"
RAW_BYTES=47_000_141_168
DEFAULT_IQ=Path("/home/ubuntu/unraid/gnss-datasets/texbat/raw/ds8.bin")
DEFAULT_EXEC=Path("/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr")
DEFAULT_TEMPLATE=Path("/home/ubuntu/projects/gnss-doppler-cmte-a2-texbat-ds78/data/external/texbat/receiver_templates/complex9/receiver.conf")
EXPORTER_GIT_DIR=Path("/home/ubuntu/projects/gnss-doppler-lab/.git")
EXPORTER_OBJECT="f8a97987a8f48bd8f2f15dc249fc181a65d97842:src/gnss_doppler_lab/multiview/complex_taps.py"


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def atomic_json(path,doc):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(doc,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)


def validate_source_contract(iq,executable,template):
    if not iq.is_file() or iq.stat().st_size!=RAW_BYTES or sha(iq)!=RAW_SHA: raise ValueError("DS8 raw IQ identity mismatch")
    if not executable.is_file() or sha(executable)!=EXEC_SHA: raise ValueError("patched GNSS-SDR binary identity mismatch")
    text=template.read_text()
    required=("SignalSource.item_type=ishort","Channels_1C.count=11","Tracking_1C.tap_count=9","Tracking_1C.tap_spacing_chips=0.125")
    if not all(token in text for token in required): raise ValueError("receiver template violates 11-channel/9-tap/.125-chip/ishort contract")
    return text


def load_pinned_exporter(destination):
    content=subprocess.check_output(["git",f"--git-dir={EXPORTER_GIT_DIR}","show",EXPORTER_OBJECT])
    if hashlib.sha256(content).hexdigest()!=EXPORTER_SHA: raise ValueError("pinned complex exporter object hash mismatch")
    path=Path(destination)/"pinned_complex_taps.py"; path.write_bytes(content)
    spec=importlib.util.spec_from_file_location("cmte_a2_pinned_complex_taps",path)
    if spec is None or spec.loader is None: raise RuntimeError("cannot load pinned source-only complex exporter")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.export_complex_9tap_npz


def main(argv=None):
    forbidden=("--model","--checkpoint","--residual","--score","--threshold","--calibr","--fit","--train")
    raw_argv=sys.argv[1:] if argv is None else argv
    if any(any(str(x).startswith(token) for token in forbidden) for x in raw_argv): raise SystemExit("model/residual/scoring arguments are forbidden")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iq",type=Path,default=DEFAULT_IQ); parser.add_argument("--receiver-executable",type=Path,default=DEFAULT_EXEC)
    parser.add_argument("--receiver-template",type=Path,default=DEFAULT_TEMPLATE); parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--receiver-timeout-seconds",type=int,default=7200)
    args=parser.parse_args(argv)
    iq=args.iq.resolve(); executable=args.receiver_executable.resolve(); template=args.receiver_template.resolve()
    text=validate_source_contract(iq,executable,template)
    out=args.output_root.resolve()
    if out.exists(): raise FileExistsError("atomic non-overwrite output required")
    staging=out.with_name(out.name+f".tmp-{os.getpid()}"); staging.mkdir(parents=True)
    try:
        run=staging/"receiver"; raw=run/"raw"; raw.mkdir(parents=True)
        config=run/"receiver.conf"
        rendered=text.replace("SignalSource.filename=DS8_IQ_PATH",f"SignalSource.filename={iq}")
        rendered=rendered.replace("OUTPUT_DIRECTORY",str(run))
        config.write_text(rendered)
        command=[str(executable),f"--config_file={config}","--keyboard=false"]
        started=time.time()
        with (run/"receiver.log").open("w") as log:
            result=subprocess.run(command,cwd=run,stdout=log,stderr=subprocess.STDOUT,timeout=args.receiver_timeout_seconds,check=False)
        if result.returncode: raise RuntimeError(f"GNSS-SDR exited {result.returncode}")
        mats=sorted(raw.glob("epl_tracking_ch_*.mat"))
        if not mats: raise RuntimeError("receiver produced no tracking MAT source")
        report=export_tracking_csv(mats,run/"tracking.csv",run/"tracking_summary.csv",sample_rate_hz=25_000_000)
        log_text=(run/"receiver.log").read_text(errors="replace")
        receiver_doc={"schema":"gnss-doppler-lab.cmte-a2-ds8-receiver-source.v1","scenario":"DS8","source_only":True,
          "aggregate_scoring_performed":False,"residual_inference_performed":False,"model_loaded":False,
          "source":{"path":str(iq),"sha256":RAW_SHA,"bytes":RAW_BYTES,"sample_rate_hz":25_000_000,"format":"ishort interleaved IQ"},
          "receiver":{"path":str(executable),"sha256":EXEC_SHA,"channels":11,"taps":9,"tap_spacing_chips":.125,
                      "config":"receiver.conf","config_sha256":sha(config),"command":command,"elapsed_s":time.time()-started},
          "acquisition":{"tracked_prns":parse_acquired_prns(log_text),"receiver_reported_prns":parse_receiver_reported_prns(log_text)},
          "tracking":{**report,"raw_directory":"raw","tap_count":9,"tap_spacing_chips":.125}}
        atomic_json(run/"manifest.json",receiver_doc)
        export=staging/"exports"/"ds8.npz"; export.parent.mkdir()
        manifest_path=load_pinned_exporter(staging)(run,export)
        with np.load(export,allow_pickle=False) as arrays:
            if set(("complex_iq","prn","time_s"))-set(arrays.files): raise ValueError("complex export schema incomplete")
            if arrays["complex_iq"].ndim!=3 or arrays["complex_iq"].shape[1:]!=(9,2): raise ValueError("complex export shape must be [N,9,2]")
            rows=len(arrays["complex_iq"])
        export_doc=json.loads(manifest_path.read_text())
        final={"schema":"gnss-doppler-lab.cmte-a2-ds8-source-preparation.v1","scenario":"DS8","status":"prepared",
          "source_only":True,"model_or_scoring_imported":False,"receiver_manifest":"receiver/manifest.json",
          "raw_sha256":RAW_SHA,"binary_sha256":EXEC_SHA,"exporter_content_sha256":EXPORTER_SHA,"rendered_config_sha256":sha(config),
          "wrapper_sha256":sha(Path(__file__)),"npz":{"path":"exports/ds8.npz","sha256":sha(export),"rows":rows,"shape_tail":[9,2]},
          "upstream_export_manifest":export_doc,"producer":{"channels":11,"taps":9,"spacing_chips":.125,"sample_rate_hz":25_000_000,"format":"ishort"}}
        atomic_json(staging/"manifest.json",final); os.replace(staging,out)
        print(json.dumps({"out":str(out),"status":"prepared","rows":rows,"aggregate_scoring_performed":False},sort_keys=True))
    except Exception as exc:
        failure={"schema":"gnss-doppler-lab.cmte-a2-ds8-source-preparation-failure.v1","scenario":"DS8","status":"NA",
                 "primary_result":"NA","silent_fallback":False,"reason":f"{type(exc).__name__}: {exc}"}
        atomic_json(staging/"failure.json",failure)
        failed=out.with_name(out.name+".failed")
        if failed.exists(): raise FileExistsError(f"failure output already exists: {failed}") from exc
        os.replace(staging,failed); raise
if __name__=="__main__": main()
