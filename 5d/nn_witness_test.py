#!/usr/bin/env python3
"""nn_witness_test.py -- per-fold neural-network witnesses for the cross-fitted 5-D test.

For each toy in --toys (files toy_XXXX.in.bin written by nn_test_5d.R::export_nn_batch_5d):
  * trains one network per fold (hidden layers --hidden, sigmoid, NPLM loss
        L_k = - sum_{training data} f + sum_{training base} omega_tr (exp(f) - 1),
    Adam 0.9/0.99/1e-7, --lr, weight clipping --clip after every step, --epochs, all
    networks of all toys in one batched model on the GPU);
  * writes toy_XXXX.out.bin with, per fold, the network parameters and its values at the
    held-out data (XE), the test base at nu-hat (XT) and at nu-hat +- eps (XTp, XTm); and on
    the common reference measure Xref (wp = 1/nR) the Gauss-Newton blocks
        H_k  = sum_ref wp expo_tr,k e^{f_k} J_k J_k'       (P x P)
        M_kj = sum_ref wp J_k J_j'                         (P x P, all fold pairs)
        mean_k = sum_ref wp J_k                            (P)
    with J_k = d f_k / d theta at the trained parameters (all P parameters; R selects the
    free ones, |theta| < clip).
Preprocessing as in LMNP, statistics from the toy's reference sample (no data dependence).
--mock: no training; one random network shared by all toys and folds, centred on the reference
        (a fixed witness); numpy only --
        for testing the plumbing and the fixed-witness variance (var(S) vs V_D + V_R + v_aux).
--selfcheck: compare the analytic Jacobian with torch autograd and exit.

Binary layouts (little-endian; matrices row-major, 5 columns pt1 pt2 eta1 eta2 delta_phi):
  in : int32 K, nR; f32 Xref[nR*5];  per fold: int32 nD,nB,nE,nT,nTp,nTm; f64 om_tr, om_te;
       f32 XD, XB, XE, XT, XTp, XTm
  out: int32 K, P;  per fold: f32 theta[P]; f64 phiE[nE], phiT[nT], phiTp[nTp], phiTm[nTm];
       f64 H[K*P*P] (k,p,q), M[K*K*P*P] (k,j,p,q), mean[K*P]
"""
import argparse
import os
import struct
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--toys", required=True, help="range a-b or comma list")
ap.add_argument("--hidden", default="5,5,5")
ap.add_argument("--clip", type=float, default=0.72)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=2800)
ap.add_argument("--checkpoints", default="",
                help="comma list of epochs at which to write outputs (toy_XXXX.e<ep>.out.bin); "
                     "default: only --epochs, written as toy_XXXX.out.bin")
ap.add_argument("--duty", type=float, default=1.0)
ap.add_argument("--duty-block", type=int, default=1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--chunk", type=int, default=4000, help="reference rows per Jacobian chunk")
ap.add_argument("--mock", action="store_true")
ap.add_argument("--selfcheck", action="store_true")
ap.add_argument("--log-every", type=int, default=500)
a = ap.parse_args()
D = os.path.expanduser(a.dir)
PROG = os.path.join(D, "python_progress.log")
sizes = [5] + [int(h) for h in a.hidden.split(",")] + [1]
LAYERS = list(zip(sizes[:-1], sizes[1:]))
P = sum(i * o + o for i, o in LAYERS)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


def toy_ids(s):
    if "-" in s:
        lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


# ------------------------------------------------------------------ I/O
def read_in(path):
    with open(path, "rb") as f:
        K, nR = struct.unpack("<2i", f.read(8))
        mat = lambda n: np.frombuffer(f.read(20 * n), "<f4").reshape(n, 5).astype(np.float64)
        Xref = mat(nR); folds = []
        for _ in range(K):
            nD, nB, nE, nT, nTp, nTm = struct.unpack("<6i", f.read(24))
            om_tr, om_te = struct.unpack("<2d", f.read(16))
            folds.append(dict(XD=mat(nD), XB=mat(nB), XE=mat(nE), XT=mat(nT), XTp=mat(nTp), XTm=mat(nTm),
                              om_tr=om_tr, om_te=om_te))
    return Xref, folds


def prep_stats(Xref):
    shift = np.zeros(5); scale = np.ones(5)
    for j in (2, 3):
        shift[j], scale[j] = Xref[:, j].mean(), Xref[:, j].std()
    for j in (0, 1, 4):
        scale[j] = Xref[:, j].mean()
    return shift, scale


# ------------------------------------------------------------------ numpy network math (float64)
def unflatten(theta):
    """theta (P,) -> [(W (i,o), b (o,)), ...] in the order W1, b1, W2, b2, ..."""
    out, p = [], 0
    for i, o in LAYERS:
        W = theta[p:p + i * o].reshape(i, o); p += i * o
        b = theta[p:p + o]; p += o
        out.append((W, b))
    return out


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def fwd(theta, Z):
    h = Z
    L = unflatten(theta)
    for l, (W, b) in enumerate(L):
        h = h @ W + b
        if l < len(L) - 1:
            h = sig(h)
    return h[:, 0]


def fwd_jac(theta, Z):
    """values f (n,) and Jacobian J (n, P) w.r.t. theta, analytic backprop"""
    L = unflatten(theta); hs = [Z]; h = Z
    for l, (W, b) in enumerate(L):
        h = h @ W + b
        if l < len(L) - 1:
            h = sig(h)
        hs.append(h)
    n = Z.shape[0]; blocks = [None] * (2 * len(L))
    g = np.ones((n, 1))                                   # d f / d z_L
    for l in range(len(L) - 1, -1, -1):
        W, b = L[l]; hin = hs[l]
        blocks[2 * l] = (hin[:, :, None] * g[:, None, :]).reshape(n, -1)   # d f / d W_l (row-major i,o)
        blocks[2 * l + 1] = g                                               # d f / d b_l
        if l > 0:
            g = (g @ W.T) * hin * (1.0 - hin)             # through sigmoid of layer l-1
    return hs[-1][:, 0], np.concatenate(blocks, axis=1)


def gn_blocks(thetas, Zref, expo_tr, chunk):
    """H_k, M_kj, mean_k on the common measure (wp = 1/nR) for the K networks of one toy"""
    K = len(thetas); nR = Zref.shape[0]
    H = np.zeros((K, P, P)); G = np.zeros((K * P, K * P)); mean = np.zeros((K, P))
    for st in range(0, nR, chunk):
        Zc = Zref[st:st + chunk]
        FJ = [fwd_jac(th, Zc) for th in thetas]
        Jall = np.concatenate([J for _, J in FJ], axis=1)          # (n_c, K*P)
        G += Jall.T @ Jall
        for k, (f, J) in enumerate(FJ):
            mu = expo_tr[k] * np.exp(f)
            H[k] += (J * mu[:, None]).T @ J
            mean[k] += J.sum(0)
    H /= nR; G /= nR; mean /= nR
    M = G.reshape(K, P, K, P).transpose(0, 2, 1, 3)               # (k, j, p, q)
    return H, M, mean


def write_out(path, thetas, phis, H, M, mean):
    K = len(thetas)
    with open(path, "wb") as f:
        f.write(struct.pack("<2i", K, P))
        for th, ph in zip(thetas, phis):
            f.write(np.asarray(th, "<f4").tobytes())
            for key in ("XE", "XT", "XTp", "XTm"):
                f.write(np.asarray(ph[key], "<f8").tobytes())
        f.write(np.ascontiguousarray(H, "<f8").tobytes())
        f.write(np.ascontiguousarray(M, "<f8").tobytes())
        f.write(np.ascontiguousarray(mean, "<f8").tobytes())


# ------------------------------------------------------------------ training (torch)
def train(toys, ids):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = [(t, k) for t in range(len(toys)) for k in range(len(toys[t]["folds"]))]
    F = len(nets)
    nmax = max(len(fo["XD"]) + len(fo["XB"]) for tt in toys for fo in tt["folds"])
    X = np.zeros((F, nmax, 5), np.float32); W = np.zeros((F, nmax), np.float32); I = np.zeros((F, nmax), np.float32)
    for i, (t, k) in enumerate(nets):
        fo = toys[t]["folds"][k]; sh, sc = toys[t]["prep"]
        nD, nB = len(fo["XD"]), len(fo["XB"])
        X[i, :nD] = (fo["XD"] - sh) / sc; X[i, nD:nD + nB] = (fo["XB"] - sh) / sc
        I[i, :nD] = 1.0; W[i, nD:nD + nB] = fo["om_tr"]
    X = torch.from_numpy(X).to(dev); W = torch.from_numpy(W).to(dev); I = torch.from_numpy(I).to(dev)
    params = []
    for i_, o_ in LAYERS:                              # Glorot uniform per network, reproducible per toy
        Wl = torch.empty(F, i_, o_); lim = (6.0 / (i_ + o_)) ** 0.5
        for t, tid in enumerate(ids):
            g = torch.Generator().manual_seed(a.seed * 1000003 + tid * 101 + i_ * 7 + o_)
            idx = [j for j, (tt, _) in enumerate(nets) if tt == t]
            Wl[idx] = (torch.rand(len(idx), i_, o_, generator=g) * 2 - 1) * lim
        params += [Wl.to(dev).requires_grad_(), torch.zeros(F, 1, o_, device=dev).requires_grad_()]
    nL = len(params) // 2

    def forward(Xb):
        h = Xb
        for l in range(nL):
            h = torch.baddbmm(params[2 * l + 1], h, params[2 * l])
            if l < nL - 1:
                h = torch.sigmoid(h)
        return h.squeeze(-1)

    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-7)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    t0 = time.perf_counter(); blk = time.perf_counter(); slept = 0.0
    snaps = {}

    def snapshot():
        th = []
        for l in range(nL):
            th.append(params[2 * l].detach().cpu().double().numpy().reshape(F, -1))
            th.append(params[2 * l + 1].detach().cpu().double().numpy().reshape(F, -1))
        theta = np.concatenate(th, axis=1)              # (F, P) in the order W1, b1, W2, b2, ...
        out = {}
        for i, (t, k) in enumerate(nets):
            out.setdefault(t, {})[k] = theta[i]
        return out

    for ep in range(1, EP_MAX + 1):
        opt.zero_grad(set_to_none=True)
        f = forward(X)
        loss = (-(I * f) + W * torch.expm1(f)).sum()
        loss.backward(); opt.step()
        if a.clip > 0:
            with torch.no_grad():
                for p in params:
                    p.clamp_(-a.clip, a.clip)
        if a.duty < 1.0 and ep % a.duty_block == 0:
            sync(); busy = time.perf_counter() - blk; pause = busy * (1 / a.duty - 1); time.sleep(pause)
            slept += pause; blk = time.perf_counter()
        if ep % a.log_every == 0:
            el = time.perf_counter() - t0
            log(f"toys {ids[0]}-{ids[-1]}: epoch {ep}/{EP_MAX}  loss/net {loss.item() / F:.4f}  "
                f"{el / 60:.1f} min (idle {100 * slept / max(el, 1e-9):.0f}%)")
        if ep in CKS:
            snaps[ep] = snapshot()
    return snaps


def selfcheck():
    import torch
    rng = np.random.default_rng(0); th = rng.uniform(-a.clip, a.clip, P); Z = rng.normal(size=(7, 5))
    f, J = fwd_jac(th, Z)
    tt = torch.tensor(th, requires_grad=False, dtype=torch.float64)

    def tf(theta):
        h = torch.tensor(Z, dtype=torch.float64); p = 0
        for l, (i, o) in enumerate(LAYERS):
            Wl = theta[p:p + i * o].reshape(i, o); p += i * o
            bl = theta[p:p + o]; p += o
            h = h @ Wl + bl
            if l < len(LAYERS) - 1:
                h = torch.sigmoid(h)
        return h[:, 0]

    Jt = torch.autograd.functional.jacobian(tf, tt).numpy()
    ft = tf(tt).numpy()
    log(f"selfcheck: max |f - f_torch| {np.abs(f - ft).max():.2e}, max |J - J_torch| {np.abs(J - Jt).max():.2e}")


# ------------------------------------------------------------------ main
if a.selfcheck:
    selfcheck(); raise SystemExit(0)
CKS = sorted({int(c) for c in a.checkpoints.split(",") if c.strip()}) or [a.epochs]
MULTI = bool(a.checkpoints.strip())
EP_MAX = max(CKS)
ids = toy_ids(a.toys)
toys = []
for tid in ids:
    Xref, folds = read_in(os.path.join(D, f"toy_{tid:04d}.in.bin"))
    toys.append(dict(Xref=Xref, folds=folds, prep=prep_stats(Xref)))
log(f"toys {ids[0]}-{ids[-1]}: {len(ids)} toys x {len(toys[0]['folds'])} folds, network {sizes} ({P} parameters), "
    f"clip {a.clip}, epochs {CKS}{' [MOCK]' if a.mock else ''}")
if a.mock:
    # one random network per toy, shared by all its folds: a FIXED witness (the analogue of the
    # dictionary's fixed-witness check; the plug-in's (K-1) shortcut assumes similar fold witnesses)
    # one random network for ALL toys and folds (drawn from --seed only), output bias set so that
    # f has mean 0 over the toy's reference sample (reference-only information: still a fixed witness)
    th0 = np.random.default_rng(a.seed * 7919).uniform(-a.clip, a.clip, P)
    thetas = {}
    for t in range(len(toys)):
        sh, sc = toys[t]["prep"]; th = th0.copy()
        th[P - 1] -= fwd(th, (toys[t]["Xref"] - sh) / sc).mean()
        thetas[t] = {k: th for k in range(len(toys[t]["folds"]))}
    snaps = {ck: thetas for ck in CKS}
else:
    snaps = train(toys, ids)
for ck in CKS:
    thetas = snaps[ck]; sfx = f".e{ck}" if MULTI else ""
    for t, tid in enumerate(ids):
        sh, sc = toys[t]["prep"]; folds = toys[t]["folds"]; K = len(folds)
        th = [thetas[t][k] for k in range(K)]
        phis = [{key: fwd(th[k], (folds[k][key] - sh) / sc) for key in ("XE", "XT", "XTp", "XTm")} for k in range(K)]
        expo_tr = [folds[k]["om_tr"] * len(folds[k]["XB"]) for k in range(K)]
        H, M, mean = gn_blocks(th, (toys[t]["Xref"] - sh) / sc, expo_tr, a.chunk)
        write_out(os.path.join(D, f"toy_{tid:04d}{sfx}.out.bin"), th, phis, H, M, mean)
    log(f"toys {ids[0]}-{ids[-1]}: outputs written for epoch {ck}")
