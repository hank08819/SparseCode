"""Aggregate vol139: everything recomputed from saved test predictions (incl. QLIKE-fit HAR-X from vol139/harx)."""
import json, glob, os, numpy as np
from scipy import stats as sstats
H=5; RV_FLOOR=1e-10
def ql(sig2, rv): sig2=np.maximum(sig2,RV_FLOOR); r=rv/sig2; return r-np.log(r)-1
def dm(d,h=H):
    n=len(d); db=d.mean(); lag=max(1,int(np.ceil(n**(1/3)))); nw=np.var(d,ddof=1)
    for j in range(1,lag+1): nw+=2*(1-j/(lag+1))*np.mean((d[j:]-db)*(d[:-j]-db))
    nw=max(nw,1e-18); cf=np.sqrt((n+1-2*h+h*(h-1)/n)/n); return float(db/(cf*np.sqrt(nw/n)))
R={}
for f in glob.glob(os.path.expanduser("~/MISA_PR/vol139/results/*.json")):
    d=json.load(open(f)); R.setdefault((d["asset"],d["period"]),{})[d["seed"]]=d
NEU=("MISA","LSTM","GRU"); BASE=["naive_last5","EWMA","HAR","HAR_QL","HAR_log","HAR_X","HAR_X_QL","GARCH11"]; METHODS=BASE+list(NEU)
rows=[]
for (a,p),S in sorted(R.items()):
    if len(S)<3: continue
    rv=np.array(S[42]["rv_test"]); hx=json.load(open(os.path.expanduser(f"~/MISA_PR/vol139/harx/{a}_{p}.json")))
    assert np.allclose(hx["rv_test"], rv), (a,p)
    base={m:np.array(S[42]["pred_test"][m]) for m in BASE if m!="HAR_X_QL"}; base["HAR_X_QL"]=np.array(hx["HAR_X_QL"])
    per_seed={}
    for s in (42,43,44):
        pred=dict(base); pred.update({m:np.array(S[s]["pred_test"][m]) for m in NEU}); L={m:ql(pred[m],rv) for m in METHODS}
        per_seed[s]=dict(q={m:float(L[m].mean()) for m in METHODS}, dm={(m,ref):dm(L[ref]-L[m]) for m in NEU for ref in METHODS if ref!=m})
    pred=dict(base); pred.update({m:np.exp(np.mean([np.log(np.array(S[s]["pred_test"][m])) for s in (42,43,44)],0)) for m in NEU}); L={m:ql(pred[m],rv) for m in METHODS}
    rows.append(dict(asset=a,period=p,rv_med=float(np.median(rv)),q={m:float(L[m].mean()) for m in METHODS},dm={(m,ref):dm(L[ref]-L[m]) for m in NEU for ref in METHODS if ref!=m},per_seed=per_seed))
print(f"complete pairs: {len(rows)}")
REFS=["HAR_QL","HAR_X_QL","HAR","HAR_log","GARCH11","EWMA","naive_last5","MISA","LSTM","GRU"]
def WL(rows,m,ref,seed=None):
    st=np.array([(r["per_seed"][seed]["dm"] if seed else r["dm"])[(m,ref)] for r in rows]); return int((st>1.96).sum()),int((st<-1.96).sum())
def table(rows,label):
    print(f"\n### {label}: {len(rows)} pairs")
    print("  median QLIKE (seed-avg): "+"  ".join(f"{m} {np.median([r['q'][m] for r in rows]):.3f}" for m in METHODS))
    for m in NEU:
        print(f"  {m:5s} seed-avg forecasts: "+"; ".join(f"vs {ref} {WL(rows,m,ref)[0]}W/{WL(rows,m,ref)[1]}L" for ref in REFS if ref!=m))
    for s in (42,43,44):
        print(f"  MISA seed {s}:          "+"; ".join(f"vs {ref} {WL(rows,'MISA',ref,s)[0]}W/{WL(rows,'MISA',ref,s)[1]}L" for ref in ("HAR_QL","HAR_X_QL","GARCH11","LSTM","GRU")))
    best=[min(METHODS,key=lambda m:r["q"][m]) for r in rows]; print("  lowest QLIKE per pair (seed-avg):",{m:best.count(m) for m in METHODS if best.count(m)})
    lowq=lambda m,ref: sum(r["q"][m]<r["q"][ref] for r in rows); print(f"  MISA lower QLIKE than: HAR_QL {lowq('MISA','HAR_QL')}, HAR_X_QL {lowq('MISA','HAR_X_QL')}, GARCH {lowq('MISA','GARCH11')}, GRU {lowq('MISA','GRU')}, LSTM {lowq('MISA','LSTM')} of {len(rows)}")
table(rows,"ALL 139")
for p in ("P1","P2","P3"): table([r for r in rows if r["period"]==p], f"period {p}")
med=np.median([r["rv_med"] for r in rows]); table([r for r in rows if r["rv_med"]<=med],"calm half (test median RV <= cross-pair median)"); table([r for r in rows if r["rv_med"]>med],"volatile half")
json.dump([dict(asset=r["asset"],period=r["period"],rv_med=r["rv_med"],q=r["q"],dm={f"{m}|{ref}":v for (m,ref),v in r["dm"].items()},per_seed={s:dict(q=v["q"],dm={f"{m}|{ref}":x for (m,ref),x in v["dm"].items()}) for s,v in r["per_seed"].items()}) for r in rows], open(os.path.expanduser("~/MISA_PR/vol139/summary.json"),"w"), indent=1)
print("\nwrote vol139/summary.json")
