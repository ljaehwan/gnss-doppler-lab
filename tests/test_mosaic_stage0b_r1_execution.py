import json
from pathlib import Path

import numpy as np
import pytest

from gnss_doppler_lab.mosaic_iq_injector_int16 import encode_interleaved_int16, inject_payload
from gnss_doppler_lab.mosaic_receiver_in_loop import ReplicaState, StatefulReplica
from gnss_doppler_lab.mosaic_stage0b_r1_execution_metrics import (
    bic, fit_complex_models, paired_bootstrap_ci, raised_cosine_envelope,
    raised_cosine_integral, strong_resolvable,
)
from gnss_doppler_lab.mosaic_stage0b_r1_executor import canonical_sha, render_receiver_config

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts/mosaic_stage0b_r1_execution"
PREREG=ROOT/"artifacts/mosaic_stage0b_r1_receiver_in_loop"


def test_zero_amplitude_byte_identity_and_sample_count():
    payload,_=encode_interleaved_int16(np.array([1+2j,-3+4j]))
    output,m=inject_payload(payload,np.zeros(2,complex))
    assert output==payload and len(output)==len(payload) and m["clipped_sample_count"]==0


def test_int16_iq_ordering_and_clipping_accounting():
    payload,m=encode_interleaved_int16(np.array([40000-2j]))
    assert np.frombuffer(payload,dtype="<i2").tolist()==[32767,-2]
    assert m["clipped_component_count"]==1 and m["clipped_sample_count"]==1


def test_delay_ramp_and_carrier_integral_are_sample_continuous():
    t=np.arange(0,12,1/10000)
    e=raised_cosine_envelope(t,12);integ=raised_cosine_integral(t,12)
    assert np.max(np.abs(np.diff(e)))<.001
    assert np.max(np.abs(np.diff(integ)-e[:-1]/10000))<1e-7
    phase=2*np.pi*75*integ
    assert np.max(np.abs(np.diff(phase)))<.05


def test_chunk_boundary_invariance_with_nav_change():
    signs=np.r_[np.ones(51),-np.ones(49)]
    a=StatefulReplica(3,1_023_000,ReplicaState(0,.2,.3));whole=a.render(100,code_rate_chips_s=1_023_001,carrier_doppler_hz=50,nav_signs=signs)
    b=StatefulReplica(3,1_023_000,ReplicaState(0,.2,.3));parts=np.r_[b.render(51,code_rate_chips_s=1_023_001,carrier_doppler_hz=50,nav_signs=signs[:51]),b.render(49,code_rate_chips_s=1_023_001,carrier_doppler_hz=50,nav_signs=signs[51:])]
    assert np.allclose(whole,parts) and b.state.absolute_sample_index==100


def test_receiver_config_diff_allowlist(tmp_path):
    base=tmp_path/"base.conf";base.write_text("A=1\nSignalSource.filename=/old\nB=2\n")
    text,changed=render_receiver_config(base,tmp_path/"new.iq")
    assert changed==["SignalSource.filename"] and "A=1" in text and "B=2" in text


def test_bic_complexity_and_two_source_fit():
    n=200;a=np.exp(1j*np.arange(n)/10);s=np.exp(1j*np.arange(n)/7);y=2*a+.3*s
    result=fit_complex_models(y,a,s)
    assert result["rss_h1"]<result["rss_h0"] and result["delta_bic"]>0
    assert bic(100,1000,4)>bic(100,1000,2)


def test_deterministic_paired_bootstrap():
    a=paired_bootstrap_ci(np.arange(10.));b=paired_bootstrap_ci(np.arange(10.))
    assert a==b and a[1]<a[0]<a[2]


def test_target_nontarget_accounting_and_dataset_gate_are_not_pooled():
    assignments=json.loads((PREREG/"case_target_assignment.json").read_text())["assignments"]
    assert len(assignments)==72
    assert all(len(x["target_prns"]) in (1,4) for x in assignments)
    assert {x["dataset"] for x in assignments}=={"OAKBAT.cleanStatic","TEXBAT.cleanStatic"}


def test_strong_resolvable_definition_is_frozen():
    assert strong_resolvable(-6,.1,0) and strong_resolvable(0,0,25)
    assert not strong_resolvable(-10,.25,50)


def test_design_hash_and_case_order_are_frozen():
    design=json.loads((PREREG/"frozen_injection_design.json").read_text())
    assert len(design)==72
    assert canonical_sha(design)=="b1a06556f7cd67738274c132f80b0581b20914d971f72f4e4ab0b5efc9a7facf"


def test_freeze_precedes_results_and_records_code_hashes():
    freeze=json.loads((ART/"executor_freeze.json").read_text())
    assert freeze["status"]=="PRE_EXECUTION_FREEZE" and not freeze["results_viewed"] and freeze["cases_executed"]==0
    assert len(freeze["executor_code_sha256"])==5


def test_permutation_invariance_of_complex_fit():
    n=128;a=np.exp(1j*np.arange(n)/13);s1=np.exp(1j*np.arange(n)/9);s2=np.exp(1j*np.arange(n)/7);y=a+.2*s1-.3j*s2
    x=fit_complex_models(y,a,np.column_stack([s1,s2]));z=fit_complex_models(y,a,np.column_stack([s2,s1]))
    assert x["delta_bic"]==pytest.approx(z["delta_bic"],abs=1e-9)


def test_artifact_freeze_manifest():
    import importlib.util
    path=ROOT/"scripts/verify_mosaic_stage0b_r1_results.py";spec=importlib.util.spec_from_file_location("r1_result_verify",path)
    module=importlib.util.module_from_spec(spec);assert spec.loader is not None;spec.loader.exec_module(module)
    module.verify_freeze();assert module.verify_manifest()>=5
