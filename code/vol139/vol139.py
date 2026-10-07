"""Decisive volatility study for PR2027: realized variance of the next 5 bars, all 47 assets x 3 periods, 3 seeds.

Target  : RV_t = sum of squared 5-min log-returns over the next H=5 bars (level), forecast origin = last observed bar.
Neural  : MISA / LSTM / GRU on the paper's 7 features + log RV lags (5, 30, 288 bars) for target and leader asset,
          trained with the QLIKE loss  RV*exp(-s) + s  on s = log sigma^2 (early stopping on validation QLIKE).
Baselines (fit on the training split only, parameters then fixed):
          naive_last5, EWMA (RiskMetrics lambda=0.94 on squared returns, x H), HAR level (OLS), HAR_QL (HAR fit by QLIKE),
          HAR_log (OLS on log RV, smearing-corrected), HAR_X level (target + leader lags), GARCH(1,1)-AR(1) 5-step variance.
Metrics : QLIKE, MSE on RV level (x1e8), MSE on log RV; Harvey-corrected DM on QLIKE loss differentials (positive = model better).
Usage   : python vol139.py ASSET --seed 42 --gpu 0 [--epochs 150]
"""
import os, sys, json, time, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy import stats as sstats, optimize
sys.path.insert(0, os.path.expanduser("~/MISA_PR"))
from msca_sparsity import MSCABiGRU, build_features
from arch import arch_model

ap = argparse.ArgumentParser(); ap.add_argument("asset"); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--out", default=os.path.expanduser("~/MISA_PR/vol139/results")); ap.add_argument("--periods", default="P1,P2,P3"); ap.add_argument("--variant", default="FULL", choices=["FULL","ABL_A","ABL_B","ABL_C"])
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
DEV = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
DATA = os.path.expanduser("~/MISA_PR/data"); DATE = {"P1": "20221101", "P2": "20231001", "P3": "20240301"}
CFG = dict(window=30, horizon=5, hidden=64, n_heads=4, dropout=0.25, lr=5e-4, epochs=args.epochs, patience=20, warmup=20, batch_size=64, train_ratio=0.70, val_frac=0.10, n_features=7)
W, H, SEED = CFG["window"], CFG["horizon"], args.seed; RV_FLOOR = 1e-10; LAGS = (5, 30, 288)

class LSTMBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__(); self.lstm = nn.LSTM(feat_dim, hidden, num_layers=2, batch_first=True, dropout=dropout); self.fc = nn.Linear(hidden, 1)
    def forward(self, x_tgt, x_ca=None):
        out, _ = self.lstm(x_tgt); return self.fc(out[:, -1, :]).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)
class GRUBaseline(nn.Module):
    def __init__(self, feat_dim=7, hidden=64, dropout=0.20):
        super().__init__(); self.gru = nn.GRU(feat_dim, hidden, num_layers=2, batch_first=True, dropout=dropout); self.fc = nn.Linear(hidden, 1)
    def forward(self, x_tgt, x_ca=None):
        out, _ = self.gru(x_tgt); return self.fc(out[:, -1, :]).squeeze(-1), None, torch.tensor(0., device=x_tgt.device)

def csv_path(asset, period): return os.path.join(DATA, f"{asset}_5m_{period}_{DATE[period]}.csv")
def rv_lags_per_bar(close):
    """log of (mean squared return over the last L bars) x H, known at bar t (uses returns ending at bar t)."""
    r = np.diff(np.log(np.maximum(close, 1e-10))); cs = np.concatenate([[0.0], np.cumsum(r**2)]); t = np.arange(len(close)); cols = []
    for L in LAGS:
        lo = np.maximum(0, t - L); cnt = np.maximum(1, t - lo); cols.append(np.log((cs[t] - cs[lo]) / cnt * H + RV_FLOOR))
    out = np.stack(cols, 1); out[0] = out[1]; return out

def prepare(asset, period):
    ca = "ETH" if asset == "BTC" else "BTC"
    close, ft = build_features(csv_path(asset, period), CFG["n_features"]); cclose, fca = build_features(csv_path(ca, period), CFG["n_features"])
    ft = np.c_[ft, rv_lags_per_bar(close)[:len(ft)]]; fca = np.c_[fca, rv_lags_per_bar(cclose)[:len(fca)]]
    lr = np.diff(np.log(np.maximum(close, 1e-10))); n = min(len(ft), len(fca)); idx = np.arange(n - W - H + 1)
    X = np.stack([ft[i:i+W] for i in idx]).astype(np.float32); Xca = np.stack([fca[i:i+W] for i in idx]).astype(np.float32)
    rv = np.array([(lr[i+W-1:i+W+H-1]**2).sum() for i in idx])
    org = idx + W - 1                                             # origin bar index
    N = len(rv); nt = int(N*CFG["train_ratio"]); nv = int(N*CFG["val_frac"]); F = X.shape[-1]
    floor = max(RV_FLOOR, float(np.percentile(rv[:nt][rv[:nt] > 0], 1)) if (rv[:nt] > 0).any() else RV_FLOOR); rv = np.maximum(rv, floor)
    for Z in (X, Xca):                                            # clip the three log-RV columns at training quantiles before min-max scaling
        for c in range(F-3, F):
            lo_q, hi_q = np.percentile(Z[:nt, :, c], [0.5, 99.5]); Z[:, :, c] = np.clip(Z[:, :, c], lo_q, hi_q)
    har_t = np.exp(ft[org, -3:]); har_c = np.exp(fca[org, -3:])  # level lags at the origin (target, leader)
    sc, sca = MinMaxScaler(), MinMaxScaler(); sc.fit(X[:nt].reshape(-1, F)); sca.fit(Xca[:nt].reshape(-1, F))
    X = sc.transform(X.reshape(-1, F)).reshape(X.shape).astype(np.float32); Xca = sca.transform(Xca.reshape(-1, F)).reshape(Xca.shape).astype(np.float32)
    sl = dict(tr=slice(0, nt), va=slice(nt, nt+nv), te=slice(nt+nv, N))
    return dict(X=X, Xca=Xca, rv=rv, rv_floor=floor, har_t=har_t, har_c=har_c, lr=lr, idx=idx, org=org, sl=sl, N=N, nt=nt, nv=nv)

# ---------------------------------------------------------------- losses / tests
def qlike_vec(sig2, rv): sig2 = np.maximum(sig2, RV_FLOOR); r = rv / sig2; return r - np.log(r) - 1
def dm_loss(d, h=H):
    """Harvey-corrected DM on a loss differential series d = L_ref - L_model (positive mean = model better)."""
    n = len(d); db = d.mean(); lag = max(1, int(np.ceil(n ** (1/3)))); nw = np.var(d, ddof=1)
    for j in range(1, lag+1): nw += 2*(1 - j/(lag+1))*np.mean((d[j:]-db)*(d[:-j]-db))
    nw = max(nw, 1e-18); cf = np.sqrt((n + 1 - 2*h + h*(h-1)/n)/n); st = db/(cf*np.sqrt(nw/n)); return float(st), float(2*(1 - sstats.t.cdf(abs(st), df=n-1)))
def metrics(sig2, rv): return dict(qlike=float(qlike_vec(sig2, rv).mean()), mse_level_e8=float(np.mean((sig2 - rv)**2))*1e8, mse_log=float(np.mean((np.log(np.maximum(sig2, RV_FLOOR)) - np.log(rv))**2)))

# ---------------------------------------------------------------- neural training with QLIKE
def train_qlike(model_cls, D, kw):
    torch.manual_seed(SEED); np.random.seed(SEED)
    s = D["sl"]; model = model_cls(feat_dim=D["X"].shape[-1], **kw).to(DEV)
    rv = D["rv"].astype(np.float32); scale = float(np.median(rv[s["tr"]]))          # work in units of the training median so s ~ 0
    y = (rv / scale).astype(np.float32)
    ds = TensorDataset(torch.tensor(D["X"][s["tr"]]), torch.tensor(D["Xca"][s["tr"]]), torch.tensor(y[s["tr"]])); loader = DataLoader(ds, batch_size=CFG["batch_size"], shuffle=True)
    Xv, Xcv, yv = (torch.tensor(D["X"][s["va"]], device=DEV), torch.tensor(D["Xca"][s["va"]], device=DEV), torch.tensor(y[s["va"]], device=DEV))
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"]); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG["epochs"])
    def ql(sv, yb): return (yb*torch.exp(-sv) + sv).mean()      # QLIKE up to a constant, sv = log sigma^2 (in scaled units)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(CFG["epochs"]):
        model.train()
        for Xb, Xcb, yb in loader:
            Xb, Xcb, yb = Xb.to(DEV), Xcb.to(DEV), yb.to(DEV); opt.zero_grad(); p, _, gl = model(Xb, Xcb); (ql(p, yb) + gl).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step(); model.eval()
        with torch.no_grad(): vl = ql(model(Xv, Xcv)[0], yv).item()
        if vl < best: best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        elif ep >= CFG["warmup"]:
            bad += 1
            if bad >= CFG["patience"]: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): st = model(torch.tensor(D["X"][s["te"]], device=DEV), torch.tensor(D["Xca"][s["te"]], device=DEV))[0].cpu().numpy()
    return np.exp(st.astype(np.float64)) * scale, ep + 1

# ---------------------------------------------------------------- baselines
def baselines(D):
    s, rv = D["sl"], D["rv"]; tr, te = s["tr"], s["te"]; out, pred = {}, {}
    Ht, Hc = D["har_t"], D["har_c"]
    pred["naive_last5"] = Ht[te, 0]
    # EWMA on squared returns (lambda 0.94), x H, at the origin
    r2 = D["lr"]**2; ew = np.empty_like(r2); ew[0] = r2[0]; lam = 0.94
    for t in range(1, len(r2)): ew[t] = lam*ew[t-1] + (1-lam)*r2[t]
    pred["EWMA"] = ew[D["org"] - 1][te] * H
    A = np.c_[np.ones(len(rv)), Ht]; b = np.linalg.lstsq(A[tr], rv[tr], rcond=None)[0]; pred["HAR"] = np.maximum(A[te] @ b, RV_FLOOR)
    Ax = np.c_[A, Hc]; bx = np.linalg.lstsq(Ax[tr], rv[tr], rcond=None)[0]; pred["HAR_X"] = np.maximum(Ax[te] @ bx, RV_FLOOR)
    Al = np.c_[np.ones(len(rv)), np.log(Ht)]; bl, *_ = np.linalg.lstsq(Al[tr], np.log(rv[tr]), rcond=None); resid = np.log(rv[tr]) - Al[tr] @ bl
    pred["HAR_log"] = np.exp(Al[te] @ bl) * float(np.mean(np.exp(resid)))      # Duan smearing
    def qobj(beta): sig = np.maximum(A[tr] @ beta, RV_FLOOR); return qlike_vec(sig, rv[tr]).mean()
    res = optimize.minimize(qobj, b, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-12, fatol=1e-12)); pred["HAR_QL"] = np.maximum(A[te] @ res.x, RV_FLOOR)
    # GARCH(1,1)-AR(1) fitted on training returns, fixed parameters, 5-step variance at each origin
    lr4 = D["lr"]*1e4; last_train_ret = D["idx"][tr][-1] + W - 2
    am = arch_model(lr4, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal", rescale=False); fit = am.fit(last_obs=last_train_ret + 1, disp="off")
    fc = fit.forecast(horizon=H, start=0, reindex=True, align="origin"); var5 = fc.variance.values.sum(axis=1)/1e8
    pred["GARCH11"] = np.maximum(var5[D["idx"] + W - 2][te], RV_FLOOR)
    return {k: np.maximum(np.asarray(v, dtype=np.float64), D["rv_floor"]) for k, v in pred.items()}

VARIANTS = {"FULL": dict(use_sparsemax=True, use_multiscale=True, use_cross_asset=True), "ABL_A": dict(use_sparsemax=False, use_multiscale=True, use_cross_asset=True),
            "ABL_B": dict(use_sparsemax=True, use_multiscale=False, use_cross_asset=True), "ABL_C": dict(use_sparsemax=True, use_multiscale=True, use_cross_asset=False)}
def run_ablation(asset, period):
    """Train one MISA ablation variant with the QLIKE objective and save its test forecasts (baselines come from the FULL run)."""
    t0 = time.time(); D = prepare(asset, period); s = D["sl"]; rvte = D["rv"][s["te"]]
    kw = dict(hidden=CFG["hidden"], n_heads=CFG["n_heads"], dropout=CFG["dropout"], **VARIANTS[args.variant])
    pr, ep = train_qlike(MSCABiGRU, D, kw); pr = np.maximum(pr, D["rv_floor"]); q = float(qlike_vec(pr, rvte).mean())
    json.dump(dict(asset=asset, period=period, seed=SEED, variant=args.variant, epochs=ep, qlike=q, pred_test=pr.tolist(), rv_test=rvte.tolist(), minutes=round((time.time()-t0)/60, 2)),
              open(os.path.join(args.out, f"{asset}_{period}_s{SEED}_{args.variant}.json"), "w"))
    print(f"{asset} {period} s{SEED} {args.variant} | QLIKE {q:.3f} | ep {ep} | {round((time.time()-t0)/60,2)} min", flush=True)

def run_pair(asset, period):
    if args.variant != "FULL": return run_ablation(asset, period)
    t0 = time.time(); D = prepare(asset, period); s = D["sl"]; rvte = D["rv"][s["te"]]
    pred = baselines(D); epochs = {}
    for name, cls, kw in [("MISA", MSCABiGRU, dict(hidden=CFG["hidden"], n_heads=CFG["n_heads"], dropout=CFG["dropout"])),
                          ("LSTM", LSTMBaseline, dict(hidden=CFG["hidden"], dropout=0.20)), ("GRU", GRUBaseline, dict(hidden=CFG["hidden"], dropout=0.20))]:
        pred[name], epochs[name] = train_qlike(cls, D, kw); pred[name] = np.maximum(pred[name], D["rv_floor"])
    L = {k: qlike_vec(v, rvte) for k, v in pred.items()}
    out = dict(asset=asset, period=period, seed=SEED, n_test=int(len(rvte)), rv_floor=D["rv_floor"], epochs=epochs, metrics={k: metrics(v, rvte) for k, v in pred.items()},
               dm={m: {ref: dm_loss(L[ref] - L[m]) for ref in pred if ref != m} for m in ("MISA", "LSTM", "GRU", "HAR_X", "HAR_QL")},
               pred_test={k: v.tolist() for k, v in pred.items()}, rv_test=rvte.tolist(), minutes=round((time.time()-t0)/60, 2))
    json.dump(out, open(os.path.join(args.out, f"{asset}_{period}_s{SEED}.json"), "w"))
    m = out["metrics"]; d = out["dm"]["MISA"]
    print(f"{asset} {period} s{SEED} | QLIKE naive {m['naive_last5']['qlike']:.3f} EWMA {m['EWMA']['qlike']:.3f} HAR {m['HAR']['qlike']:.3f} HAR_QL {m['HAR_QL']['qlike']:.3f} HAR_X {m['HAR_X']['qlike']:.3f} GARCH {m['GARCH11']['qlike']:.3f} | MISA {m['MISA']['qlike']:.3f} (DM vs HAR_QL {d['HAR_QL'][0]:+.2f}, vs GARCH {d['GARCH11'][0]:+.2f}, vs GRU {d['GRU'][0]:+.2f}) LSTM {m['LSTM']['qlike']:.3f} GRU {m['GRU']['qlike']:.3f} | ep {epochs} | {out['minutes']} min", flush=True)

PAIRS = set(json.load(open(os.path.expanduser("~/MISA_PR/results/expanded_results.json"))).keys())   # the paper's 139 asset-period pairs
for period in args.periods.split(","):
    if f"{args.asset}_{period}" not in PAIRS or not os.path.exists(csv_path(args.asset, period)): print(f"SKIP {args.asset} {period}", flush=True); continue
    try: run_pair(args.asset, period)
    except Exception as ex:
        import traceback; traceback.print_exc(); print(f"FAILED {args.asset} {period}: {ex}", flush=True)
print("DONE", args.asset, SEED, flush=True)
