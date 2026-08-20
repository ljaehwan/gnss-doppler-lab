"""Independent sparse raw-IQ reference correlator for CRID R3.

No generator rendering, scheduling, injection, or amplitude routine is imported.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

_TAPS={1:(2,6),2:(3,7),3:(4,8),4:(5,9),5:(1,9),6:(2,10),7:(1,8),8:(2,9),9:(3,10),10:(2,3),11:(3,4),12:(5,6),13:(6,7),14:(7,8),15:(8,9),16:(9,10),17:(1,4),18:(2,5),19:(3,6),20:(4,7),21:(5,8),22:(6,9),23:(1,3),24:(4,6),25:(5,7),26:(6,8),27:(7,9),28:(8,10),29:(1,6),30:(2,7),31:(3,8),32:(4,9)}

def independent_ca(prn:int)->np.ndarray:
    g1=[1]*10;g2=[1]*10;out=np.empty(1023,float);a,b=_TAPS[prn]
    for k in range(1023):
        out[k]=1. if (g1[9]^(g2[a-1]^g2[b-1]))==0 else -1.
        f1=g1[2]^g1[9];f2=g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9]
        g1=[f1]+g1[:-1];g2=[f2]+g2[:-1]
    return out

class ReferenceReplica:
    def __init__(self,prn:int,records:np.ndarray,nav_rows:list[dict[str,str]]):
        self.prn=prn;self.r=records;self.starts=records['raw_interval_start_sample'].astype(np.int64);self.ends=records['raw_interval_end_sample'].astype(np.int64);self.code=independent_ca(prn)
        rows=sorted((x for x in nav_rows if int(x['prn'])==prn),key=lambda x:int(x['corrected_raw_start_sample']))
        self.ns=np.array([int(x['corrected_raw_start_sample']) for x in rows]);self.ne=np.array([int(x['corrected_raw_end_sample_exclusive']) for x in rows]);self.nv=np.array([int(x['bit_value_pm1']) for x in rows])
    def render(self,absolute:int,count:int,delay:float)->np.ndarray:
        pos=absolute+np.arange(count);row=np.searchsorted(self.starts,pos,side='right')-1
        if row.min()<0 or np.any(pos>self.ends[row]):raise ValueError('reference TRACE bounds')
        nav=np.searchsorted(self.ns,pos,side='right')-1
        if nav.min()<0 or np.any(pos>self.ne[nav]):raise ValueError('reference NAV bounds')
        local=pos-self.starts[row];cp=local*self.r['action_used_code_phase_step_chips_per_sample'][row]-self.r['action_used_residual_code_phase_chips'][row]+delay
        phase=self.r['action_used_residual_carrier_phase_rad'][row]+local*self.r['action_used_carrier_phase_step_rad_per_sample'][row]
        return self.code[np.floor(cp).astype(np.int64)%1023]*self.nv[nav]*np.exp(1j*phase)

def _iq(payload:bytes)->np.ndarray:
    z=np.frombuffer(payload,dtype='<i2').reshape(-1,2);return z[:,0].astype(float)+1j*z[:,1].astype(float)

def recover_delay_power(source:Path,output:Path,start:int,fs:int,replicas:dict[int,ReferenceReplica],targets:list[int],requested_delay:float,requested_power_db:float)->dict:
    epochs=[start+int((5.0+.1*k)*fs) for k in range(10)];n=fs//1000;grids=np.round(np.arange(-.4,.4001,.01),2);per={}
    with source.open('rb') as a,output.open('rb') as b:
        for prn,rep in replicas.items():
            numer=np.zeros(len(grids),complex);denom=np.zeros(len(grids));auth_num=0j;auth_den=0.
            for absolute in epochs:
                a.seek(absolute*4);x=_iq(a.read(n*4));b.seek((absolute-start)*4);y=_iq(b.read(n*4));res=y-x
                s0=rep.render(absolute,n,0.);auth_num+=np.vdot(s0,x);auth_den+=float(np.vdot(s0,s0).real)
                for j,d in enumerate(grids):
                    s=rep.render(absolute,n,float(d));numer[j]+=np.vdot(s,res);denom[j]+=float(np.vdot(s,s).real)
            coef=numer/denom;j=int(np.argmax(np.abs(coef)));alpha=auth_num/auth_den
            per[str(prn)]={'recovered_delay_chips':float(grids[j]),'requested_delay_chips':requested_delay if prn in targets else None,'coefficient_magnitude':float(abs(coef[j])),'authentic_magnitude':float(abs(alpha)),'realized_power_db':float(20*np.log10(max(abs(coef[j]),1e-15)/max(abs(alpha),1e-15))),'is_target':prn in targets}
    target_ok=all(abs(per[str(p)]['recovered_delay_chips']-requested_delay)<=.025 and abs(per[str(p)]['realized_power_db']-requested_power_db)<=.75 for p in targets)
    nontarget_ok=all(10**(per[str(p)]['realized_power_db']/10)<=.01 for p in replicas if p not in targets)
    return {'per_prn':per,'target_count':len(targets),'target_delay_power_pass':target_ok,'non_target_relative_energy_pass':nontarget_ok,'status':'PASS' if target_ok and nontarget_ok else 'FAIL'}
