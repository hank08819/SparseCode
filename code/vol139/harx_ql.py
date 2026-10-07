"""Post-hoc linear baselines fitted by QLIKE (HAR_QL, HAR_X_QL) for every pair; CPU only. Writes vol139/harx/<asset>_<period>.json."""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import optimize
sys.path.insert(0, os.path.expanduser("~/MISA_PR")); from msca_sparsity import build_features
DATA=os.path.expanduser("~/MISA_PR/data"); DATE={"P1":"20221101","P2":"20231001","P3":"20240301"}; W,H=30,5; RV_FLOOR=1e-10; LAGS=(5,30,288); OUT=os.path.expanduser("~/MISA_PR/vol139/harx"); os.makedirs(OUT,exist_ok=True)
def csv_path(a,p): return os.path.join(DATA,f"{a}_5m_{p}_{DATE[p]}.csv")
def lags(close):
    r=np.diff(np.log(np.maximum(close,1e-10))); cs=np.concatenate([[0.0],np.cumsum(r**2)]); t=np.arange(len(close)); cols=[]
    for L in LAGS: lo=np.maximum(0,t-L); cnt=np.maximum(1,t-lo); cols.append((cs[t]-cs[lo])/cnt*H)
    out=np.stack(cols,1); out[0]=out[1]; return out
def qlike_vec(s2,rv): s2=np.maximum(s2,RV_FLOOR); r=rv/s2; return r-np.log(r)-1
def fit_ql(A,rv,b0):
    f=lambda b: qlike_vec(np.maximum(A@b,RV_FLOOR),rv).mean()
    res=optimize.minimize(f,b0,method="Nelder-Mead",options=dict(maxiter=20000,xatol=1e-12,fatol=1e-12)); return res.x
PAIRS=sorted(json.load(open(os.path.expanduser("~/MISA_PR/results/expanded_results.json"))).keys())
for key in PAIRS:
    a,p=key.split("_"); ca="ETH" if a=="BTC" else "BTC"
    try:
        close,ft=build_features(csv_path(a,p),7); cclose,fca=build_features(csv_path(ca,p),7)
        lr=np.diff(np.log(np.maximum(close,1e-10))); n=min(len(ft),len(fca)); idx=np.arange(n-W-H+1); org=idx+W-1
        rv=np.array([(lr[i+W-1:i+W+H-1]**2).sum() for i in idx]); N=len(rv); nt=int(N*0.70); nv=int(N*0.10)
        floor=max(RV_FLOOR,float(np.percentile(rv[:nt][rv[:nt]>0],1))); rv=np.maximum(rv,floor); tr=slice(0,nt); te=slice(nt+nv,N)
        Ht=lags(close)[org]; Hc=lags(cclose)[org]
        A=np.c_[np.ones(N),Ht]; Ax=np.c_[A,Hc]
        b=np.linalg.lstsq(A[tr],rv[tr],rcond=None)[0]; bx=np.linalg.lstsq(Ax[tr],rv[tr],rcond=None)[0]
        bq=fit_ql(A[tr],rv[tr],b); bxq=fit_ql(Ax[tr],rv[tr],np.r_[bq,np.zeros(3)])
        out=dict(asset=a,period=p,HAR_QL=np.maximum(A[te]@bq,floor).tolist(),HAR_X_QL=np.maximum(Ax[te]@bxq,floor).tolist(),rv_test=rv[te].tolist(),
                 qlike=dict(HAR_QL=float(qlike_vec(np.maximum(A[te]@bq,floor),rv[te]).mean()),HAR_X_QL=float(qlike_vec(np.maximum(Ax[te]@bxq,floor),rv[te]).mean())))
        json.dump(out,open(os.path.join(OUT,f"{key}.json"),"w")); print(key,{k:round(v,3) for k,v in out["qlike"].items()},flush=True)
    except Exception as ex: print("FAILED",key,ex,flush=True)
print("DONE")
