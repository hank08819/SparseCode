"""Ablation aggregate on the volatility target: FULL vs ABL_A (softmax), ABL_B (single-scale), ABL_C (no cross-asset); seed-averaged and per-seed DM on QLIKE."""
import json, glob, os, numpy as np
H=5; RV_FLOOR=1e-10
def ql(s2, rv): s2=np.maximum(s2,RV_FLOOR); r=rv/s2; return r-np.log(r)-1
def dm(d,h=H):
    n=len(d); db=d.mean(); lag=max(1,int(np.ceil(n**(1/3)))); nw=np.var(d,ddof=1)
    for j in range(1,lag+1): nw+=2*(1-j/(lag+1))*np.mean((d[j:]-db)*(d[:-j]-db))
    return float(db/(np.sqrt((n+1-2*h+h*(h-1)/n)/n)*np.sqrt(max(nw,1e-18)/n)))
full={}
for f in glob.glob(os.path.expanduser("~/MISA_PR/vol139/results/*.json")):
    d=json.load(open(f)); full.setdefault((d["asset"],d["period"]),{})[d["seed"]]=d
abl={}
for f in glob.glob(os.path.expanduser("~/MISA_PR/vol139/results_abl/*.json")):
    d=json.load(open(f)); abl.setdefault((d["asset"],d["period"]),{}).setdefault(d["variant"],{})[d["seed"]]=d
rows=[]
for key,S in sorted(full.items()):
    if key not in abl or any(len(abl[key].get(v,{}))<3 for v in ("ABL_A","ABL_B","ABL_C")): continue
    rv=np.array(S[42]["rv_test"]); geo=lambda preds: np.exp(np.mean([np.log(np.array(p)) for p in preds],0))
    pf=geo([S[s]["pred_test"]["MISA"] for s in (42,43,44)]); Lf=ql(pf,rv)
    r=dict(asset=key[0],period=key[1],q_full=float(Lf.mean()))
    for v in ("ABL_A","ABL_B","ABL_C"):
        pv=geo([abl[key][v][s]["pred_test"] for s in (42,43,44)]); Lv=ql(pv,rv); r[v]=dict(q=float(Lv.mean()),dm=dm(Lv-Lf),
            per_seed=[dm(ql(np.array(abl[key][v][s]["pred_test"]),rv)-ql(np.array(S[s]["pred_test"]["MISA"]),rv)) for s in (42,43,44)])
    rows.append(r)
print("pairs with all variants:",len(rows))
def rep(rows,label):
    print(f"\n### {label}: {len(rows)} pairs   (positive DM = FULL better than the variant)")
    if not rows: return
    for v,name in (("ABL_A","softmax instead of sparsemax"),("ABL_B","single scale"),("ABL_C","no cross-asset gate")):
        st=np.array([r[v]["dm"] for r in rows]); dq=np.array([r[v]["q"]-r["q_full"] for r in rows])
        ps=np.array([r[v]["per_seed"] for r in rows])
        print(f"  {v} {name:28s}: FULL sig better {int((st>1.96).sum()):3d}, variant sig better {int((st<-1.96).sum()):3d}; FULL lower QLIKE on {int((dq>0).sum())}/{len(rows)}; median QLIKE gain {np.median(dq):+.4f} ({np.median(dq/np.array([r['q_full'] for r in rows]))*100:+.1f}%); per-seed W/L: "+" ".join(f"{int((ps[:,i]>1.96).sum())}/{int((ps[:,i]<-1.96).sum())}" for i in range(3)))
rep(rows,"ALL")
for p in ("P1","P2","P3"): rep([r for r in rows if r["period"]==p],f"period {p}")
json.dump(rows,open(os.path.expanduser("~/MISA_PR/vol139/summary_ablation.json"),"w"),indent=1)
