#!/usr/bin/env python3
"""Pre-holdout historical B0 gate equivalence check (DS1 only; no holdout)."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from gnss_doppler_lab.cmte_a2 import historical_gate_equivalence_files

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--node-score-csv",default=str(ROOT/"artifacts/cmte_texbat_poc/per_prn/DS1.csv"))
 p.add_argument("--golden-event-csv",default=str(ROOT/"artifacts/cmte_a2_historical_b0/ds1_golden_events.csv"))
 p.add_argument("--historical-evaluator",default=str(ROOT/"scripts/eval_btail_support_gate.py"))
 p.add_argument("--calibration",default=str(ROOT/"configs/detectors/texbat_btail_gate_v1.json")); p.add_argument("--output",required=True)
 a=p.parse_args(argv); evidence=historical_gate_equivalence_files(a.node_score_csv,a.golden_event_csv,evaluator_path=a.historical_evaluator,
  calibration_path=a.calibration,evidence_path=a.output)
 if evidence.get("passed") is not True: raise SystemExit("historical B0 gate equivalence FAILED")
 print(json.dumps({"output":str(Path(a.output).resolve()),"passed":True,"max_absolute_error":evidence["max_absolute_error"]},sort_keys=True))
if __name__=="__main__": main()
