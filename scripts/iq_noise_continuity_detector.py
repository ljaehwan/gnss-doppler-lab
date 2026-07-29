#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS = 25_000_000
DEFAULT_ONSETS = {'ds1':100.0,'ds2':100.0,'ds3':100.0,'ds4':110.0,'ds5':100.0,'ds7':110.0,'ds8':110.0}
ART = Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts')
EPS = 1e-9

def robust_scale(x):
    med=np.nanmedian(x,axis=0)
    mad=np.nanmedian(np.abs(x-med),axis=0)*1.4826
    std=np.nanstd(x,axis=0)
    scale=np.where(np.isfinite(mad)&(mad>EPS),mad,std)
    scale=np.where(np.isfinite(scale)&(scale>EPS),scale,1.0)
    return med,scale

def z_l2(x,med,scale):
    z=(x-med)/scale
    z=np.nan_to_num(z,nan=0,posinf=0,neginf=0)
    return np.sqrt(np.mean(z*z,axis=-1))

def block_features(z: np.ndarray, fs: int) -> dict[str,float]:
    # z: complex64/complex128 raw IQ block, intentionally before GNSS tracking/GRU residuals.
    z=z.astype(np.complex64, copy=False)
    i=z.real.astype(np.float32); q=z.imag.astype(np.float32)
    amp=np.abs(z).astype(np.float32)
    power=(amp*amp).astype(np.float32)
    # Remove block DC only; this is not despreading/tracking residual.
    zc=z - np.mean(z)
    ac=[]
    denom=float(np.mean(np.abs(zc)**2)+EPS)
    for lag in [1,2,4,8,16,32,64]:
        c=np.mean(zc[lag:]*np.conj(zc[:-lag]))/denom
        ac += [float(np.real(c)), float(np.imag(c)), float(np.abs(c))]
    # phase increment/coherence fingerprints
    dz=z[1:]*np.conj(z[:-1])
    phase=np.angle(dz).astype(np.float32)
    # FFT on capped subset for speed; bandpowers are receiver/IQ-noise texture features.
    nfft=min(65536, len(zc))
    x=zc[:nfft]
    win=np.hanning(nfft).astype(np.float32)
    spec=np.fft.fftshift(np.fft.fft(x*win))
    psd=(np.abs(spec)**2).astype(np.float64)+EPS
    psd=psd/psd.sum()
    bands=np.array_split(psd,16)
    bandpowers=[float(b.sum()) for b in bands]
    entropy=float(-(psd*np.log(psd)).sum()/math.log(len(psd)))
    flatness=float(np.exp(np.mean(np.log(psd)))/(np.mean(psd)+EPS))
    # high-frequency amplitude micro texture
    da=np.diff(amp)
    def moments(x, prefix):
        mu=float(np.mean(x)); sd=float(np.std(x)+EPS)
        xc=(x-mu)/sd
        return {f'{prefix}_mean':mu, f'{prefix}_std':sd, f'{prefix}_skew':float(np.mean(xc**3)), f'{prefix}_kurt':float(np.mean(xc**4))}
    out={
        'i_mean':float(i.mean()), 'q_mean':float(q.mean()), 'i_std':float(i.std()+EPS), 'q_std':float(q.std()+EPS),
        'iq_corr':float(np.corrcoef(i,q)[0,1]) if len(i)>2 else 0.0,
        'power_mean':float(power.mean()), 'power_std':float(power.std()+EPS),
        'phase_inc_mean':float(np.mean(phase)), 'phase_inc_std':float(np.std(phase)+EPS),
        'phase_coh':float(np.abs(np.mean(np.exp(1j*phase)))),
        'psd_entropy':entropy, 'psd_flatness':flatness,
    }
    out.update(moments(amp,'amp'))
    out.update(moments(da,'damp'))
    for idx,val in enumerate(ac): out[f'ac_{idx:02d}']=val
    for idx,val in enumerate(bandpowers): out[f'psd_band_{idx:02d}']=val
    return out

def extract_feature_frame(raw: Path, scenario: str, *, block_ms: float, stride_s: float, max_s: float|None) -> pd.DataFrame:
    mm=np.memmap(raw, dtype=np.int16, mode='r')
    total_complex=mm.size//2
    duration=total_complex/FS
    end_s=min(duration, max_s) if max_s else duration
    block_n=int(round(FS*block_ms/1000.0))
    stride_n=int(round(FS*stride_s))
    rows=[]
    starts=np.arange(0, max(0,total_complex-block_n+1), stride_n, dtype=np.int64)
    for idx,start in enumerate(starts):
        t=start/FS
        if t>end_s: break
        inter=mm[start*2:(start+block_n)*2].astype(np.float32)
        z=inter[0::2] + 1j*inter[1::2]
        feats=block_features(z, FS)
        feats.update({'scenario':scenario,'window_index':idx,'window_start_s':float(t),'window_mid_s':float(t+block_ms/2000.0),'window_end_s':float(t+block_ms/1000.0),'block_ms':float(block_ms),'stride_s':float(stride_s)})
        rows.append(feats)
    return pd.DataFrame(rows)

def pca_fit_transform(Xfit, Xall, ncomp):
    mu=Xfit.mean(axis=0); sd=Xfit.std(axis=0)+EPS
    Zfit=(Xfit-mu)/sd; Zall=(Xall-mu)/sd
    U,S,Vt=np.linalg.svd(Zfit, full_matrices=False)
    k=min(ncomp, Vt.shape[0], Vt.shape[1])
    V=Vt[:k].T
    return Zfit@V, Zall@V, {'mean':mu.tolist(),'std':sd.tolist(),'components':V.tolist()}

def make_ar(P, times, lag):
    X=[]; Y=[]; meta=[]
    for i in range(lag, len(P)):
        X.append(P[i-lag:i].reshape(-1)); Y.append(P[i]); meta.append(i)
    return np.asarray(X), np.asarray(Y), np.asarray(meta)

def first_persistent(times, flags, start, n=3):
    for i in range(len(flags)-n+1):
        if times[i]>=start and np.all(flags[i:i+n]): return float(times[i])
    return None

def run(args):
    sc=args.scenario
    onset=DEFAULT_ONSETS.get(sc,100.0) if args.onset_s is None else args.onset_s
    raw=args.raw or Path(f'data/external/texbat/raw/{sc}.bin')
    out=args.out or ART/f'{sc}-raw-iq-noise-continuity-20260729-v0'
    out.mkdir(parents=True, exist_ok=True)
    feat_csv=out/f'{sc}_raw_iq_noise_features.csv'
    if feat_csv.exists() and not args.force_extract:
        df=pd.read_csv(feat_csv)
    else:
        df=extract_feature_frame(raw, sc, block_ms=args.block_ms, stride_s=args.stride_s, max_s=args.max_s)
        df.to_csv(feat_csv,index=False)
    meta_cols={'scenario','window_index','window_start_s','window_mid_s','window_end_s','block_ms','stride_s'}
    feat_cols=[c for c in df.columns if c not in meta_cols]
    X=df[feat_cols].apply(pd.to_numeric, errors='raise').to_numpy(float)
    times=df.window_start_s.to_numpy(float)
    fit_mask=times<=args.fit_end_s
    Xfit=X[fit_mask]
    Pfit, Pall, pca=pca_fit_transform(Xfit, X, args.pca_dim)
    Xar,Yar,idx=make_ar(Pall,times,args.lag)
    fit_ar=times[idx]<=args.fit_end_s
    # least-squares linear continuity model in IQ-noise PCA space.
    W=np.linalg.lstsq(Xar[fit_ar], Yar[fit_ar], rcond=None)[0]
    pred=Xar@W
    err=np.sqrt(np.mean((Yar-pred)**2,axis=1))
    fit_err=err[fit_ar]
    med_e,scale_e=robust_scale(fit_err[:,None])
    ar_score=((err-med_e[0])/scale_e[0]).clip(min=0)
    # feature-level drift score as companion, still raw-IQ based.
    med_x,scale_x=robust_scale(Xfit)
    level=z_l2(X,med_x,scale_x)
    ev=df.iloc[idx][['scenario','window_index','window_start_s','window_mid_s','window_end_s']].copy().reset_index(drop=True)
    ev['iq_noise_ar_rmse']=err
    ev['iq_noise_ar_score']=ar_score
    ev['iq_noise_level_l2z']=level[idx]
    ev['iq_noise_ar_ewma075']=ev.iq_noise_ar_score.ewm(alpha=.25, adjust=False).mean()
    ev['iq_noise_level_ewma075']=ev.iq_noise_level_l2z.ewm(alpha=.25, adjust=False).mean()
    fit_ev=ev[ev.window_start_s<=args.fit_end_s]
    th_ar=float(fit_ev.iq_noise_ar_ewma075.quantile(args.q))
    th_lv=float(fit_ev.iq_noise_level_ewma075.quantile(args.q))
    ev['ar_ratio']=ev.iq_noise_ar_ewma075/max(th_ar,EPS)
    ev['level_ratio']=ev.iq_noise_level_ewma075/max(th_lv,EPS)
    ev['combined_ratio']=ev[['ar_ratio','level_ratio']].max(axis=1)
    th_comb=float(ev.loc[ev.window_start_s<=args.fit_end_s,'combined_ratio'].quantile(args.q))
    ev['alarm_combined']=ev.combined_ratio>th_comb
    event_csv=out/f'{sc}_raw_iq_noise_continuity_event_scores.csv'
    ev.to_csv(event_csv,index=False)
    t=ev.window_start_s.to_numpy(float); flags=ev.alarm_combined.to_numpy(bool)
    pre=t<onset; post=t>=onset; gap=(t>args.fit_end_s)&(t<onset)
    summary={
      'schema':'gnss-doppler-lab.raw-iq-noise-continuity.v0',
      'scenario':sc,'raw':str(raw),'raw_seconds':Path(raw).stat().st_size/4/FS,
      'intent':'learn continuity of raw IQ noise/fingerprint features from 0-90s; no GRU residuals, no tracking residuals, no spoof labels',
      'feature_extraction':{'block_ms':args.block_ms,'stride_s':args.stride_s,'feature_count':len(feat_cols),'feature_cols':feat_cols,'feature_csv':str(feat_csv)},
      'model':{'kind':'PCA + linear autoregressive predictor over raw-IQ noise fingerprint features','pca_dim':args.pca_dim,'lag_windows':args.lag,'fit_rule':f'window_start_s <= {args.fit_end_s}','threshold_quantile':args.q},
      'onset_s_assumed':onset,
      'thresholds':{'ar_ewma_q':th_ar,'level_ewma_q':th_lv,'combined_ratio_q':th_comb},
      'detections':{
        'pre_onset_alarm_rate':float(flags[pre].mean()),'pre_onset_alarm_count':int(flags[pre].sum()),
        'gap_fit_to_onset_alarm_rate':float(flags[gap].mean()) if gap.any() else None,'gap_fit_to_onset_alarm_count':int(flags[gap].sum()),
        'post_onset_alarm_rate':float(flags[post].mean()),'post_onset_alarm_count':int(flags[post].sum()),
        'first_post_alarm_s':None if not np.any(flags&post) else float(t[flags&post][0]),
        'first_post_alarm_delay_s':None if not np.any(flags&post) else float(t[flags&post][0]-onset),
        'first_post_3consecutive_alarm_s':first_persistent(t,flags,onset,3),
      },
      'outputs':{'event_scores_csv':str(event_csv)}
    }
    fp=summary['detections']['first_post_3consecutive_alarm_s']
    summary['detections']['first_post_3consecutive_delay_s']=None if fp is None else float(fp-onset)
    fig,ax=plt.subplots(figsize=(13,5.5))
    ax.plot(ev.window_start_s, ev.ar_ratio, label='IQ-noise AR continuity ratio')
    ax.plot(ev.window_start_s, ev.level_ratio, label='IQ-noise feature level ratio', alpha=.75)
    ax.plot(ev.window_start_s, ev.combined_ratio, label='combined', color='black', lw=1.4)
    ax.axvspan(0,args.fit_end_s,color='green',alpha=.08,label=f'fit 0-{args.fit_end_s:g}s')
    ax.axvline(onset,color='red',ls='--',label=f'onset {onset:g}s')
    ax.axhline(th_comb,color='black',ls=':',label='fit q threshold')
    ax.set_title(f'{sc.upper()} raw-IQ noise continuity detector (not GRU residual)')
    ax.set_xlabel('time / s'); ax.set_ylabel('normalized score ratio'); ax.grid(True,alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); plot=out/f'{sc}_raw_iq_noise_continuity_timeline.png'; fig.savefig(plot,dpi=160)
    summary['outputs']['timeline_png']=str(plot)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2,sort_keys=True))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario',default='ds3')
    ap.add_argument('--raw',type=Path,default=None)
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--fit-end-s',type=float,default=90.0)
    ap.add_argument('--onset-s',type=float,default=None)
    ap.add_argument('--block-ms',type=float,default=10.0)
    ap.add_argument('--stride-s',type=float,default=0.5)
    ap.add_argument('--max-s',type=float,default=None)
    ap.add_argument('--pca-dim',type=int,default=8)
    ap.add_argument('--lag',type=int,default=6)
    ap.add_argument('--q',type=float,default=0.99)
    ap.add_argument('--force-extract',action='store_true')
    run(ap.parse_args())
if __name__=='__main__': main()
