#!/usr/bin/env python3
"""Parse/run the committed config with pinned GNSS-SDR and synthetic ishort."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PINNED=Path("/home/ubuntu/build-gnss-sdr-complex9/build-complex/src/main/gnss-sdr")
PINNED_SHA="6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb"
TEMPLATE=ROOT/"configs/cmte_a2_ds8_receiver.conf"
FIXTURE=ROOT/"artifacts/cmte_a2_receiver_smoke/exporter_fixture.npz"
FIXTURE_DOC=ROOT/"artifacts/cmte_a2_receiver_smoke/exporter_fixture.json"
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--receiver",type=Path,default=PINNED); p.add_argument("--output-json"); a=p.parse_args(argv)
 receiver=a.receiver.resolve(strict=True)
 if sha(receiver)!=PINNED_SHA: raise ValueError("receiver is not the pinned complex9 binary")
 template=TEMPLATE.read_text();
 if template.count("[GNSS-SDR]")!=1 or template.count("[")!=1: raise ValueError("config must use one GNSS-SDR property section")
 fixture=json.loads(FIXTURE_DOC.read_text())
 if sha(FIXTURE)!=fixture["npz_sha256"]: raise ValueError("exporter fixture checksum mismatch")
 with np.load(FIXTURE,allow_pickle=False) as z:
  if z["complex_iq"].shape!=(fixture["rows"],9,2) or set(("prn","time_s"))-set(z.files): raise ValueError("exporter fixture schema mismatch")
 with tempfile.TemporaryDirectory(prefix="cmte-a2-config-smoke-") as raw:
  run=Path(raw); (run/"raw").mkdir(); iq=run/"synthetic.ishort"; iq.write_bytes(bytes(400000)) # 100000 complex ishort samples
  config=run/"receiver.conf"; config.write_text(template.replace("@CMTE_A2_DS8_INPUT_RAW@",str(iq)).replace("@CMTE_A2_DS8_OUTPUT_DIR@",str(run)))
  result=subprocess.run([str(receiver),f"--config_file={config}","--keyboard=false"],cwd=run,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30,check=False)
  log=result.stdout
  if result.returncode!=0 or "too short" not in log or "Stopping GNSS-SDR" not in log: raise RuntimeError("unexpected synthetic no-signal receiver exit")
  if "configuration file is not well formatted" in log.lower() or "must define at least one signalsource" in log.lower(): raise RuntimeError("GNSS-SDR rejected config syntax")
  evidence={"schema":"gnss-doppler-lab.cmte-a2-config-smoke.v1","passed":True,"actual_binary_executed":True,
   "receiver_sha256":sha(receiver),"template_sha256":sha(TEMPLATE),"synthetic_ishort":True,"holdout_opened":False,
   "sample_count":100000,"returncode":result.returncode,"controlled_exit":"too_short_no_signal",
   "log_markers":["too short","Stopping GNSS-SDR"],"exporter_fixture_validated":True,
   "exporter_fixture_sha256":fixture["npz_sha256"],"exporter_content_sha256":fixture["exporter_content_sha256"]}
 if a.output_json: Path(a.output_json).write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
 print(json.dumps(evidence,sort_keys=True)); return evidence
if __name__=="__main__": main()
