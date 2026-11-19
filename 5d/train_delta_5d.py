#!/usr/bin/env python3
"""train_delta_5d.py -- NPLM nuisance networks delta_1, delta_2 for the 5-D benchmark.

Method (d'Agnolo et al., EPJC 82:275 (2022), Sects. 2.3 and 4.2):
    log r(x; nu) = nu d1(x) + nu^2 d2(x) / 2          (Eq. 31; nu_n handled analytically)
    c(x; nu)     = 1 / (1 + r(x; nu))
    L            = sum_nu [ sum_{S0(nu)} w c^2 + sum_{S1} w (1 - c)^2 ]      (Eq. 17)
  d1, d2: separate fully connected networks, 5 hidden layers x 10 ReLU units, linear output.
  Optimizer as in LMNP (EPJC 81:89): Adam, beta1 = 0.9, beta2 = 0.99, eps = 1e-7, lr = 1e-3,
  full-batch (one step per epoch), Keras update form and Glorot-uniform initialization.
  Preprocessing as in LMNP: eta's standardized; positive variables (pT, delta_phi) divided by
  their mean (statistics from the S1 training events).
  NPLM train about 2000 epochs with early stopping on reconstruction quality; here the
  exported model is the one with the lowest validation loss (a held-out fraction of source
  events), and the paired Fig.-14 check is reported for it.

Inputs (from utils_5d.R: make_delta_samples_5d + export_delta_samples_5d):
    <stem>.csv.gz        pt1,pt2,eta1,eta2,delta_phi,nu,target,w,id
    <stem>_check.csv.gz  pt1,pt2,eta1,eta2,delta_phi,nu,w,id   (S0 source events on a nu grid incl. 0)
Outputs:
    <out>.json           preprocessing, weights/biases of both nets, metadata, R-verification set
    <out>_history.csv    epoch, train loss, validation loss
    <out>_fig14.csv      binned observed vs predicted log(n_nu / n_0) on validation events

Usage:
    python3 train_delta_5d.py --csv delta/delta_raw5_e.csv.gz \
        --out delta/delta_raw5_e --epochs 2000 --threads 8
"""
import argparse
import os
import sys


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="training table from export_delta_samples_5d")
    ap.add_argument("--check", default=None, help="paired check table (default: <stem>_check.csv.gz)")
    ap.add_argument("--out", required=True, help="output stem (no extension)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=5, help="hidden layers")
    ap.add_argument("--width", type=int, default=10, help="units per hidden layer")
    ap.add_argument("--val-frac", type=float, default=0.2, help="fraction of source events held out")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--select", choices=["best", "final"], default="best")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0, help="BLAS threads (0 = library default)")
    ap.add_argument("--min-count", type=int, default=30, help="min events per bin in the Fig.-14 check")
    ap.add_argument("--check-only", action="store_true",
                    help="skip training; load <out>.json and rerun the Fig.-14 check (same seed/split)")
    return ap.parse_args(argv)


ARGS = parse_args(sys.argv[1:])
if ARGS.threads > 0:                               # must be set before numpy is imported
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(ARGS.threads)

import json   # noqa: E402
import time   # noqa: E402
import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

FEATS = ["pt1", "pt2", "eta1", "eta2", "delta_phi"]


# ----------------------------------------------------------------------------- model
class MLP:
    """Fully connected ReLU network with scalar linear output."""

    def __init__(self, sizes, rng):
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            lim = np.sqrt(6.0 / (a + b))                 # Glorot uniform (Keras default)
            self.W.append(rng.uniform(-lim, lim, size=(a, b)))
            self.b.append(np.zeros(b))

    def params(self):
        return self.W + self.b

    def forward(self, X, keep=True):
        h, hs = X, [X]
        for l in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[l] + self.b[l], 0.0)
            hs.append(h)
        if keep:
            self._hs = hs
        return (h @ self.W[-1] + self.b[-1])[:, 0]

    def backward(self, g):
        """g = dL/d(output), shape (n,); returns grads aligned with params()."""
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        d = g[:, None]
        for l in range(L - 1, -1, -1):
            h = self._hs[l]
            gW[l] = h.T @ d
            gb[l] = d.sum(axis=0)
            if l > 0:
                d = (d @ self.W[l].T) * (h > 0)
        return gW + gb

    def to_json(self):
        return {"W": [w.tolist() for w in self.W], "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_json(cls, J):
        m = cls.__new__(cls)
        m.W = [np.array(w, dtype=float) for w in J["W"]]
        m.b = [np.array(b, dtype=float) for b in J["b"]]
        return m

    def copy_params(self):
        return [p.copy() for p in self.params()]

    def set_params(self, ps):
        L = len(self.W)
        self.W = [p.copy() for p in ps[:L]]
        self.b = [p.copy() for p in ps[L:]]


class Adam:
    """Keras-form Adam: lr_t = lr sqrt(1 - b2^t) / (1 - b1^t); p -= lr_t m / (sqrt(v) + eps)."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.99, eps=1e-7):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= lr_t * m / (np.sqrt(v) + self.eps)


def expit_neg(g):
    """c = 1 / (1 + e^g), numerically stable."""
    out = np.empty_like(g)
    pos = g >= 0
    e = np.exp(-g[pos]); out[pos] = e / (1 + e)
    e = np.exp(g[~pos]); out[~pos] = 1 / (1 + e)
    return out


# ----------------------------------------------------------------------------- data
def preproc_stats(X):
    shift, scale = np.zeros(X.shape[1]), np.ones(X.shape[1])
    for j, nm in enumerate(FEATS):
        if nm.startswith("eta"):
            shift[j], scale[j] = X[:, j].mean(), X[:, j].std()
        else:
            scale[j] = X[:, j].mean()
    return shift, scale


def build_split(df, ids, shift, scale):
    """Unique events of one split: S0 rows are unique (event x nu); S1 rows are shared across nu."""
    d = df[df["id"].isin(ids)]
    s0 = d[d["target"] == 0]
    s1 = d[d["target"] == 1]
    s1u = s1.drop_duplicates("id")
    Xu = np.vstack([s0[FEATS].to_numpy(), s1u[FEATS].to_numpy()])
    pos1 = pd.Series(np.arange(len(s1u)) + len(s0), index=s1u["id"].to_numpy())
    ui = np.concatenate([np.arange(len(s0)), pos1.loc[s1["id"].to_numpy()].to_numpy()])
    rows = pd.concat([s0, s1])
    return {"X": (Xu - shift) / scale, "ui": ui,
            "nu": rows["nu"].to_numpy(float), "t": rows["target"].to_numpy(float),
            "w": rows["w"].to_numpy(float), "n_u": len(Xu), "n_rows": len(rows)}


def loss_and_grads(nets, S, need_grad=True):
    d1 = nets[0].forward(S["X"], keep=need_grad)
    d2 = nets[1].forward(S["X"], keep=need_grad)
    nu, ui = S["nu"], S["ui"]
    g = nu * d1[ui] + 0.5 * nu * nu * d2[ui]
    c = expit_neg(g)
    r = S["t"] - c
    loss = float(np.sum(S["w"] * r * r))
    if not need_grad:
        return loss, None
    dLdg = 2.0 * S["w"] * r * c * (1.0 - c)            # d/dg of w (t - c)^2, with dc/dg = -c(1-c)
    g1 = np.bincount(ui, weights=dLdg * nu, minlength=S["n_u"])
    g2 = np.bincount(ui, weights=dLdg * 0.5 * nu * nu, minlength=S["n_u"])
    return loss, nets[0].backward(g1) + nets[1].backward(g2)


def predict(nets, Xraw, shift, scale):
    Z = (Xraw - shift) / scale
    return nets[0].forward(Z, keep=False), nets[1].forward(Z, keep=False)


def mll_of(X):
    return np.sqrt(2 * X[:, 0] * X[:, 1] * (np.cosh(X[:, 2] - X[:, 3]) - np.cos(X[:, 4])))


def fig14_check(nets, chk, ids, shift, scale, min_count):
    """Binned observed vs predicted log(n_nu / n_0) on the paired check sample (events in ids).

    Standard error of the observed paired log-ratio from the events that migrate: with
    n_in (n_out) events entering (leaving) a bin between nu = 0 and nu, and constant weights,
    se ~ sqrt(n_in + n_out) / n_0. z = (pred - obs) / se.
    """
    c = chk[chk["id"].isin(ids)]
    c0 = c[c["nu"] == 0]
    X0 = c0[FEATS].to_numpy()
    w0 = c0["w"].to_numpy()
    id0 = c0["id"].to_numpy()
    d1, d2 = predict(nets, X0, shift, scale)
    bins = {"pt1": np.array([20, 40, 60, 80, 100, 150, 200, 300, 500, 1000.]),
            "pt2": np.array([20, 30, 40, 50, 60, 80, 100, 150, 300, 1000.]),
            "mll": np.array([100, 110, 120, 140, 170, 200, 300, 500, 2000.])}
    val = lambda X, v: mll_of(X) if v == "mll" else X[:, FEATS.index(v)]
    rows = []
    for nu in sorted(set(c["nu"]) - {0.0}):
        cn = c[c["nu"] == nu]
        Xn, wn, idn = cn[FEATS].to_numpy(), cn["w"].to_numpy(), cn["id"].to_numpy()
        rw = w0 * np.exp(nu * d1 + 0.5 * nu * nu * d2)
        n_in, n_out = np.setdiff1d(idn, id0).size, np.setdiff1d(id0, idn).size
        rows.append(dict(nu=nu, var="total", lo=np.nan, hi=np.nan, n0=len(w0),
                         obs=np.log(wn.sum() / w0.sum()), pred=np.log(rw.sum() / w0.sum()),
                         se=np.sqrt(n_in + n_out) / len(w0)))
        for v, br in bins.items():
            k0 = np.digitize(val(X0, v), br) - 1
            kn = np.digitize(val(Xn, v), br) - 1
            b0 = pd.Series(k0, index=id0)
            bn = pd.Series(kn, index=idn)
            for b in range(len(br) - 1):
                n0 = int(np.sum(k0 == b))
                if n0 < min_count:
                    continue
                in0, inn = set(b0.index[b0.to_numpy() == b]), set(bn.index[bn.to_numpy() == b])
                s0 = w0[k0 == b].sum()
                rows.append(dict(nu=nu, var=v, lo=br[b], hi=br[b + 1], n0=n0,
                                 obs=np.log(wn[kn == b].sum() / s0),
                                 pred=np.log(rw[k0 == b].sum() / s0),
                                 se=np.sqrt(len(inn - in0) + len(in0 - inn)) / n0))
    out = pd.DataFrame(rows)
    out["diff"] = out["pred"] - out["obs"]
    out["z"] = np.where(out["se"] > 0, out["diff"] / out["se"].where(out["se"] > 0), np.nan)
    return out


def report_check(tab, out, min_count):
    tab.to_csv(out + "_fig14.csv", index=False)
    print(f"\nFig.-14 check on held-out events ({out}_fig14.csv): log(n_nu/n_0), observed vs predicted")
    tot = tab[tab["var"] == "total"][["nu", "obs", "pred", "diff", "se", "z"]]
    print(tot.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    b = tab[tab["var"] != "total"]
    g = b.groupby(["nu", "var"])
    summ = pd.DataFrame({"max_abs_z": g["z"].apply(lambda z: np.nanmax(np.abs(z))),
                         "chi2_per_bin": g["z"].apply(lambda z: np.nanmean(z ** 2)),
                         "nbins": g["z"].size()}).unstack("var")
    print(f"\nbinned residuals (bins with >= {min_count} events): max|z| and chi2/bin "
          f"(noise alone: chi2/bin ~ 1, max|z| ~ 2-2.5 over ~8 bins)")
    print(summ.to_string(float_format=lambda x: f"{x:.2f}"))


# ----------------------------------------------------------------------------- main
def main(a):
    t_start = time.time()
    rng = np.random.default_rng(a.seed)
    df = pd.read_csv(os.path.expanduser(a.csv))
    miss = [c for c in FEATS + ["nu", "target", "w", "id"] if c not in df.columns]
    if miss:
        sys.exit(f"missing columns in {a.csv}: {miss} (re-export with the current utils_5d.R)")
    chk_path = a.check or os.path.expanduser(a.csv).replace(".csv.gz", "_check.csv.gz")
    chk = pd.read_csv(chk_path) if os.path.exists(chk_path) else None

    # split by source event over the union of ids in both files: events that pass the cuts
    # only at check-grid nu values (never at a training nu) must still be assignable to the
    # validation set, or the check at large |nu| would silently drop the entering events
    ids = np.unique(np.concatenate([df["id"].to_numpy()] + ([chk["id"].to_numpy()] if chk is not None else [])))
    val_ids = rng.choice(ids, size=int(round(a.val_frac * len(ids))), replace=False)
    tr_ids = np.setdiff1d(ids, val_ids)
    if a.check_only:
        out = os.path.expanduser(a.out)
        with open(out + ".json") as f:
            J = json.load(f)
        nets = [MLP.from_json(J["nets"]["delta1"]), MLP.from_json(J["nets"]["delta2"])]
        shift, scale = np.array(J["preproc"]["shift"]), np.array(J["preproc"]["scale"])
        if J["meta"].get("seed") != a.seed or J["meta"].get("val_frac") != a.val_frac:
            print("warning: --seed/--val-frac differ from training; the check is not on the held-out split")
        if chk is None:
            sys.exit(f"no check file at {chk_path}")
        report_check(fig14_check(nets, chk, val_ids, shift, scale, a.min_count), out, a.min_count)
        return
    s1_tr = df[(df["target"] == 1) & df["id"].isin(tr_ids)].drop_duplicates("id")
    shift, scale = preproc_stats(s1_tr[FEATS].to_numpy())
    TR = build_split(df, tr_ids, shift, scale)
    VA = build_split(df, val_ids, shift, scale)
    nus = sorted(set(df["nu"]))
    print(f"training rows {TR['n_rows']} ({TR['n_u']} unique events), "
          f"validation rows {VA['n_rows']} ({VA['n_u']}); nu points {nus}")

    sizes = [len(FEATS)] + [a.width] * a.layers + [1]
    nets = [MLP(sizes, rng), MLP(sizes, rng)]
    params = nets[0].params() + nets[1].params()
    opt = Adam(params, lr=a.lr)
    n0p = len(nets[0].params())

    hist, best = [], (np.inf, 0, None)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        loss, grads = loss_and_grads(nets, TR)
        opt.step(params, grads)
        if ep % a.eval_every == 0 or ep == a.epochs:
            vloss, _ = loss_and_grads(nets, VA, need_grad=False)
            hist.append((ep, loss, vloss))
            if vloss < best[0]:
                best = (vloss, ep, [n.copy_params() for n in nets])
            if ep % (a.eval_every * 20) == 0 or ep == a.epochs:
                el = time.time() - t0
                print(f"  epoch {ep:6d}  train {loss:.6f}  val {vloss:.6f}  "
                      f"(best {best[0]:.6f} @ {best[1]})  {el:.0f}s, ~{el / ep * (a.epochs - ep):.0f}s left",
                      flush=True)

    final_ep = a.epochs
    if a.select == "best":
        for n, ps in zip(nets, best[2]):
            n.set_params(ps)
        sel_ep = best[1]
    else:
        sel_ep = final_ep
    print(f"selected epoch {sel_ep} ({a.select}); best validation loss {best[0]:.6f} at epoch {best[1]}")

    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame(hist, columns=["epoch", "train_loss", "val_loss"]).to_csv(out + "_history.csv", index=False)

    # R-verification set: raw features and predictions for 2000 training events
    Xv = df[FEATS].to_numpy()[:2000]
    v1, v2 = predict(nets, Xv, shift, scale)
    model = {
        "features": FEATS,
        "preproc": {"shift": shift.tolist(), "scale": scale.tolist()},
        "activation": "relu",
        "nets": {"delta1": nets[0].to_json(), "delta2": nets[1].to_json()},
        "meta": {"method": "NPLM EPJC 82:275 Eq.17; 5x10 ReLU; Adam(0.9,0.99,1e-7), lr 1e-3, full batch",
                 "sizes": sizes, "epochs": a.epochs, "selected_epoch": int(sel_ep), "select": a.select,
                 "best_val_loss": best[0], "nu_points": nus, "val_frac": a.val_frac, "seed": a.seed,
                 "n_train_rows": TR["n_rows"], "n_val_rows": VA["n_rows"], "csv": a.csv,
                 "seconds": round(time.time() - t_start, 1)},
        "check": {"x": Xv.tolist(), "delta1": v1.tolist(), "delta2": v2.tolist()},
    }
    with open(out + ".json", "w") as f:
        json.dump(model, f)
    print(f"wrote {out}.json, {out}_history.csv")

    if chk is not None:
        report_check(fig14_check(nets, chk, val_ids, shift, scale, a.min_count), out, a.min_count)
    else:
        print(f"(no check file at {chk_path}; Fig.-14 check skipped)")
    print(f"total time {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main(ARGS)