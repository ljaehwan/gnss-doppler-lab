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


# Stage-0 complex-field model helpers.  These deliberately have no attack inputs.
def caf_complex(iq, prn, fs, center_code, center_doppler, code_grid=CAF_CODE, doppler_grid=CAF_DOPPLER):
    n = int(fs // 1000); x=np.asarray(iq[:n],np.complex64); tt=np.arange(n,dtype=np.float64)/fs
    out=np.empty((len(doppler_grid),len(code_grid)),np.complex128)
    for i,d in enumerate(doppler_grid):
        y=x*np.exp(-2j*np.pi*(center_doppler+d)*tt)
        for j,c in enumerate(code_grid): out[i,j]=np.vdot(replica(prn,fs,n,center_code+c),y)/n
    return out

def normalize_caf(c, eps=1e-12, center=None):
    c=np.asarray(c,np.complex128); flat=c.ravel();
    if center is None: anchor=int(np.argmax(np.abs(flat)))
    else: anchor=int(center)
    if abs(flat[anchor]) <= eps: anchor=int(np.argmax(np.abs(flat)))
    phase=np.angle(flat[anchor]) if abs(flat[anchor])>eps else 0.0
    norm=np.linalg.norm(flat)
    return (c*np.exp(-1j*phase)/(norm+eps)), {'anchor_index':anchor,'fallback':center is not None and anchor!=center,'norm':float(norm)}

def complex_vector(c):
    z=np.asarray(c).ravel(); return np.concatenate([z.real,z.imag])

def fit_h0(clean_vectors, ridge=1e-3):
    x=np.asarray(clean_vectors,float)
    if x.ndim!=2 or len(x)<2: raise ValueError('at least two clean train vectors required')
    mu=np.median(x,axis=0)
    xc=x-mu; emp=(xc.T@xc)/max(1,len(x)-1); scale=float(np.trace(emp)/emp.shape[0])
    cov=(1-ridge)*emp+ridge*scale*np.eye(emp.shape[0])+1e-8*np.eye(emp.shape[0])
    return {'mu':mu,'precision':np.linalg.pinv(cov,rcond=1e-8),'ridge':ridge}

def h0_score(v,h0):
    d=np.asarray(v)-h0['mu']; return float(d@h0['precision']@d)

def clean_query_indices(clean_vectors,k):
    x=np.asarray(clean_vectors,float)
    if k<1 or k>x.shape[1]: raise ValueError('invalid K')
    # deterministic pivoted greedy variance selection, clean-only
    remain=list(range(x.shape[1])); chosen=[]; residual=x-x.mean(0)
    for _ in range(k):
        j=max(remain,key=lambda q: (float(np.var(residual[:,q])),-q)); chosen.append(j); remain.remove(j)
    return chosen

def two_source_fit(c, atoms, boundary_flags=None):
    y=np.asarray(c).ravel(); A=np.asarray(atoms).reshape(len(atoms),-1).T
    one=[]
    for j in range(A.shape[1]):
        q=np.linalg.lstsq(A[:,[j]],y,rcond=None)[0]; one.append((float(np.vdot(y-A[:,[j]]@q,y-A[:,[j]]@q).real),j,q))
    r1,j1,a1=min(one,key=lambda z:z[0]); candidates=[]
    for j2 in range(A.shape[1]):
        if j2==j1: continue
        q=np.linalg.lstsq(A[:,[j1,j2]],y,rcond=None)[0]; rr=float(np.vdot(y-A[:,[j1,j2]]@q,y-A[:,[j1,j2]]@q).real); candidates.append((rr,j2,q))
    r2,j2,a=min(candidates,key=lambda z:z[0])
    n=len(y); bic1=n*np.log(r1/n+1e-15)+2*np.log(n); bic2=n*np.log(r2/n+1e-15)+4*np.log(n)
    return {'single_residual':r1,'two_residual':r2,'bic_improvement':float(bic1-bic2),'first_index':j1,'second_index':j2,'amplitude_ratio':float(abs(a[1])/(abs(a[0])+1e-12)),'boundary':bool(boundary_flags[j2]) if boundary_flags is not None else False}
