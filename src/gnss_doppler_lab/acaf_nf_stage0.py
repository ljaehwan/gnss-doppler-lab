"""Direct raw-IQ ACAF Stage-0 primitives; no historical CAF/tap reuse."""
import hashlib,json
import numpy as np
CAF_CODE=np.arange(-1.,1.0001,.125); CAF_DOPPLER=np.arange(-250.,250.1,50.)
def ca_code(prn):
 if not 1<=prn<=32: raise ValueError("PRN 1..32")
 g1=np.ones(10,dtype=np.int8);g2=np.ones(10,dtype=np.int8);o=[];t=[(2,6),(3,7),(4,8),(5,9),(1,9),(2,10),(1,8),(2,9),(3,10),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),(9,10),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(1,3),(4,6),(5,7),(6,8),(7,9),(8,10),(1,6),(2,7),(3,8),(4,9)];a,b=t[prn-1]
 for _ in range(1023):
  o.append(g1[-1]^g2[-a]^g2[-b]);f1=g1[2]^g1[9];f2=g2[1]^g2[2]^g2[5]^g2[7]^g2[8]^g2[9];g1[1:]=g1[:-1];g1[0]=f1;g2[1:]=g2[:-1];g2[0]=f2
 return 1-2*np.array(o,dtype=np.int8)
def replica(prn,fs,n,offset=0): return ca_code(prn)[np.floor((np.arange(n)*1023/fs+offset)%1023).astype(int)].astype(np.complex64)
def caf_surface(iq,prn,fs,center_code,center_doppler,code_grid=CAF_CODE,doppler_grid=CAF_DOPPLER):
 n=fs//1000;x=np.asarray(iq[:n],np.complex64);tt=np.arange(n,dtype=np.float32)/fs;z=np.empty((len(doppler_grid),len(code_grid)))
 for i,d in enumerate(doppler_grid):
  y=x*np.exp(-2j*np.pi*(center_doppler+d)*tt)
  for j,c in enumerate(code_grid): z[i,j]=abs(np.vdot(replica(prn,fs,n,center_code+c),y))**2/n**2
 return z
def augment(x,gain,phase,awgn_sigma,seed):
 r=np.random.default_rng(seed);return np.asarray(x)*gain*np.exp(1j*phase)+awgn_sigma*(r.normal(size=len(x))+1j*r.normal(size=len(x)))/np.sqrt(2)
def two_source_control(a,b,fs,seed):
 n=fs//1000;return augment(replica(a,fs,n)+.7*np.exp(.37j)*replica(b,fs,n,.375),1,0,.03,seed)
def select_k(x):
 z=sorted(set(x));
 if not set(z)<={3,5,9,16}:raise ValueError("selected K")
 return z
def chronological_clean_split(x):
 n=len(x);return {"fit":list(x[:n//2]),"threshold":list(x[n//2:3*n//4]),"held_clean":list(x[3*n//4:])}
def attack_free_fit(split,roles): return roles.get("fit")=="cleanStatic" and "attack" not in str(split["fit"]).lower()
def ds78_overlap_status(a,b): return {"status":"INCONCLUSIVE","reason":"exact-time nonoverlap provenance absent; fail closed"} if a is None or b is None else {"status":"VERIFIED_NONOVERLAP" if a[1]<=b[0] or b[1]<=a[0] else "INCONCLUSIVE"}
def onset_alignment(a,b): return {"candidate_s":a,"observed_s":b,"aligned":b is not None and abs(a-b)<=1}
def same_epochs(a,b): return list(a)==list(b)
def strict_manifest(x): return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)
def sha256_file(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def raw_epoch(p,seconds,fs=25000000):
 r=np.memmap(p,dtype="<i2",mode="r",offset=int(seconds*fs)*4,shape=(fs//1000,2));return r[:,0].astype(np.float32)+1j*r[:,1].astype(np.float32)
def feature(s):
 p=float(s.max());m=float(np.median(s));return {"peak_to_median":p/(m+1e-12),"rawpower":p,"flatness":float(s.mean()/(p+1e-12))}
def bootstrap_mean(x,seed=0):
 x=np.array(x,float)
 if len(x)<2:return [None,None]
 r=np.random.default_rng(seed);z=[r.choice(x,len(x)).mean() for _ in range(500)];return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
