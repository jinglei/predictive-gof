#!/usr/bin/env python3
"""nn1d_test.py -- per-fold neural-network witnesses for the 1-D cross-fitted test (nn1d_test.R).

For every toy in --toys and every fold k, trains NPLM's (1, h, 1) sigmoid network f_k by
    L_k = - sum_{training events} f(x) + sum_{grid} w_g (exp(f(x_g)) - 1)
against the fold's fitted null on its quadrature grid (w_g = expo_tr * wp), all networks batched
(Adam 0.9/0.99/1e-7, weight clipping after each step; --no-bias-clip clips weight matrices only).
Inputs x / --scale (LMNP: positive feature over the nominal null mean; 1 for Exp(1)).

Writes per toy (toy_XXXX.out.bin), for nn1d_test.R to assemble the statistic:
  - f_k at the held-out events and on the fold's grid (the witness phi_k; Poisson mode, unnormalized);
  - the Gauss-Newton blocks on the toy's common reference measure (fold 1's grid, weights wp_ref):
        H_k  = J_k' diag(expo_tr,k * wp_ref * exp(f_k)) J_k,     M_kj = J_k' diag(wp_ref) J_j,
    J_k the Jacobian of f_k in its parameters [W1 (h), b1 (h), W2 (h), b2];
  - the free-parameter mask (not at the clip bound) and the parameters.
Input layout  (toy_XXXX.in.bin, little-endian):
  int32 K, n_ref; float32 x_ref[n_ref]; float64 wp_ref[n_ref];
  per fold: int32 nD, nE, nG; float64 expo_tr; float32 xD[nD], xE[nE], xG[nG]; float64 wG[nG]
Output layout (toy_XXXX.out.bin):
  int32 K, P; per fold: int32 nE, nG; float64 phiE[nE], phiG[nG]; int32 free[P]; float64 theta[P], H[P*P];
  then float64 M[K*K*P*P] (C order: k, j, p, q)
Usage:
  python3 nn1d_test.py --dir <work_dir> --toys 1-100 --clip 4 --no-bias-clip --epochs 30000
  python3 nn1d_test.py --dir /tmp --selfcheck          (analytic Jacobian vs finite differences)
  --mock: no training; one fixed random network for all folds and toys (fixed-witness plumbing check).
"""
import argparse
import os
import struct
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--toys", default="")
ap.add_argument("--hidden", type=int, default=4)
ap.add_argument("--clip", type=float, default=4.0, help="0 = no clipping")
ap.add_argument("--no-bias-clip", action="store_true")
ap.add_argument("--epochs", type=int, default=30000)
ap.add_argument("--checkpoints", default="",
                help="comma list of epochs; writes toy_XXXX.e<epoch>.out.bin at each (default: --epochs only, toy_XXXX.out.bin)")
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--duty", type=float, default=1.0)
ap.add_argument("--duty-block", type=int, default=1)
ap.add_argument("--log-every", type=int, default=5000)
ap.add_argument("--mock", action="store_true")
ap.add_argument("--selfcheck", action="store_true")
a = ap.parse_args()
D = os.path.expanduser(a.dir)
h = a.hidden
P = 3 * h + 1
PROG = os.path.join(D, "python_progress.log")


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(PROG, "a") as f:
        f.write(line + "\n")


# ---- network in numpy: theta = [W1 (h), b1 (h), W2 (h), b2] -------------------------------
def unpack(th):
    return th[:h], th[h:2 * h], th[2 * h:3 * h], th[3 * h]


def fwd(th, x):
    W1, b1, W2, b2 = unpack(th)
    s = 1.0 / (1.0 + np.exp(-(x[:, None] * W1[None, :] + b1[None, :])))
    return s @ W2 + b2


def jac(th, x):
    W1, b1, W2, b2 = unpack(th)
    s = 1.0 / (1.0 + np.exp(-(x[:, None] * W1[None, :] + b1[None, :])))
    ds = s * (1.0 - s) * W2[None, :]
    return np.concatenate([ds * x[:, None], ds, s, np.ones((len(x), 1))], axis=1)


if a.selfcheck:
    rng = np.random.default_rng(3); x = rng.exponential(1.0, 50); err = 0.0
    for _ in range(5):
        th = rng.normal(0, 2, P); J = jac(th, x); eps = 1e-6
        Jn = np.stack([(fwd(th + eps * e, x) - fwd(th - eps * e, x)) / (2 * eps) for e in np.eye(P)], 1)
        err = max(err, np.abs(J - Jn).max())
    print(f"selfcheck: max |J - J_finite-diff| = {err:.2e}  ({'OK' if err < 1e-6 else 'FAIL'})")
    raise SystemExit(0)


def read_toy(path):
    with open(path, "rb") as f:
        K, nr = struct.unpack("<2i", f.read(8))
        xr = np.frombuffer(f.read(4 * nr), "<f4").astype(np.float64)
        wr = np.frombuffer(f.read(8 * nr), "<f8").copy()
        folds = []
        for _ in range(K):
            nD, nE, nG = struct.unpack("<3i", f.read(12)); et = struct.unpack("<d", f.read(8))[0]
            xD = np.frombuffer(f.read(4 * nD), "<f4").astype(np.float64)
            xE = np.frombuffer(f.read(4 * nE), "<f4").astype(np.float64)
            xG = np.frombuffer(f.read(4 * nG), "<f4").astype(np.float64)
            wG = np.frombuffer(f.read(8 * nG), "<f8").copy()
            folds.append(dict(expo_tr=et, xD=xD, xE=xE, xG=xG, wG=wG))
    return dict(K=K, xr=xr, wr=wr, folds=folds)


lo, hi = (int(v) for v in a.toys.split("-"))
ids = [b for b in range(lo, hi + 1) if os.path.exists(os.path.join(D, f"toy_{b:04d}.in.bin"))]
toys = [read_toy(os.path.join(D, f"toy_{b:04d}.in.bin")) for b in ids]
nets = [(t, k) for t in range(len(toys)) for k in range(toys[t]["K"])]
F = len(nets); sc = a.scale
log(f"toys {ids[0]}-{ids[-1]} ({len(ids)}): {F} networks (1,{h},1); clip {a.clip}"
    f"{' (weights only)' if a.no_bias_clip else ''}, epochs {a.checkpoints or a.epochs}, lr {a.lr}{' [MOCK]' if a.mock else ''}")

CKS = sorted({int(float(c)) for c in a.checkpoints.split(",") if c.strip()}) or [a.epochs]
MULTI = bool(a.checkpoints.strip()); EPMAX = max(CKS)


def free_mask(theta):
    if a.clip <= 0:
        return np.ones(theta.shape, bool)
    isw = np.array([j < h or 2 * h <= j < 3 * h for j in range(P)])          # W1, W2 entries
    at = np.abs(theta) >= a.clip * (1 - 1e-6)
    return ~(at & (isw[None, :] | (not a.no_bias_clip)))


snaps = {}                                                                     # epoch -> theta (F, P)
if a.mock:
    rng = np.random.default_rng(a.seed)
    th0 = np.concatenate([rng.uniform(-2, 2, h), rng.uniform(-2, 2, h), rng.uniform(-0.5, 0.5, h), [0.0]])
    for ck in CKS:
        snaps[ck] = np.tile(th0, (F, 1))
else:
    import torch
    nmax = max(len(fd["xD"]) + len(fd["xG"]) for tt in toys for fd in tt["folds"])
    X = np.zeros((F, nmax, 1), np.float32); W = np.zeros((F, nmax), np.float32); I = np.zeros((F, nmax), np.float32)
    for i, (t, k) in enumerate(nets):
        fd = toys[t]["folds"][k]; nD, nG = len(fd["xD"]), len(fd["xG"])
        X[i, :nD, 0] = fd["xD"] / sc; X[i, nD:nD + nG, 0] = fd["xG"] / sc
        I[i, :nD] = 1.0; W[i, nD:nD + nG] = fd["wG"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(a.seed)
    Xt = torch.from_numpy(X).to(dev); Wt = torch.from_numpy(W).to(dev); It = torch.from_numpy(I).to(dev)
    sizes = [1, h, 1]; params = []
    for i_, o_ in zip(sizes[:-1], sizes[1:]):
        lim = (6.0 / (i_ + o_)) ** 0.5
        params.append((torch.rand(F, i_, o_, device=dev) * 2 - 1).mul_(lim).requires_grad_())
        params.append(torch.zeros(F, 1, o_, device=dev).requires_grad_())

    def forward(Xb):
        z = torch.sigmoid(torch.baddbmm(params[1], Xb, params[0]))
        return torch.baddbmm(params[3], z, params[2]).squeeze(-1)

    def extract():
        with torch.no_grad():
            return torch.cat([params[0].reshape(F, -1), params[1].reshape(F, -1),
                              params[2].reshape(F, -1), params[3].reshape(F, -1)], 1).double().cpu().numpy()

    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.99), eps=1e-7)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    t0 = time.perf_counter(); blk = t0; slept = 0.0
    for ep in range(1, EPMAX + 1):
        opt.zero_grad(set_to_none=True)
        f = forward(Xt); loss = (-(It * f) + Wt * torch.expm1(f)).sum(); loss.backward(); opt.step()
        if a.clip > 0:
            with torch.no_grad():
                for j, p in enumerate(params):
                    if j % 2 == 0 or not a.no_bias_clip:
                        p.clamp_(-a.clip, a.clip)
        if a.duty < 1.0 and ep % a.duty_block == 0:
            sync(); busy = time.perf_counter() - blk; pause = busy * (1 / a.duty - 1); time.sleep(pause)
            slept += pause; blk = time.perf_counter()
        if ep in CKS:
            snaps[ep] = extract()
        if ep % a.log_every == 0 or ep == EPMAX:
            el = time.perf_counter() - t0
            log(f"toys {ids[0]}-{ids[-1]}: epoch {ep}/{EPMAX}  loss/net {loss.item() / F:.4f}  {el / 60:.1f} min "
                f"(idle {100 * slept / max(el, 1e-9):.0f}%), ~{el / ep * (EPMAX - ep) / 60:.1f} min to go")
    with torch.no_grad():
        theta = snaps[EPMAX]
        ## consistency of the numpy forward with the trained torch model (first network, held-out events)
        fd = toys[0]["folds"][0]; xe = torch.from_numpy((fd["xE"] / sc).astype(np.float32)).to(dev).view(1, -1, 1)
        ft = forward(xe.expand(F, -1, -1).contiguous())[0].double().cpu().numpy()
        dmax = np.abs(ft - fwd(theta[0], fd["xE"].astype(np.float32).astype(np.float64) / sc)).max()
        log(f"numpy vs torch forward: max |diff| {dmax:.2e}")
        if dmax > 1e-3:
            raise SystemExit("parameter layout mismatch between torch and numpy")

# ---- outputs, one file per checkpoint (toy_XXXX.e<epoch>.out.bin when --checkpoints is given) ----
for ck in CKS:
    theta = snaps[ck]; free = free_mask(theta); sfx = f".e{ck}" if MULTI else ""
    log(f"epoch {ck}: free parameters per network: mean {free.sum(1).mean():.1f} of {P}")
    for t, b in enumerate(ids):
        tt = toys[t]; K = tt["K"]; idx = [i for i, (tq, _) in enumerate(nets) if tq == t]
        xr = tt["xr"] / sc; J = np.empty((K, len(xr), P))
        with open(os.path.join(D, f"toy_{b:04d}{sfx}.out.bin"), "wb") as f:
            f.write(struct.pack("<2i", K, P))
            for k in range(K):
                th = theta[idx[k]]; fd = tt["folds"][k]
                J[k] = jac(th, xr)
                mu = fd["expo_tr"] * tt["wr"] * np.exp(np.minimum(fwd(th, xr), 50.0))
                H = J[k].T @ (mu[:, None] * J[k])
                phiE = fwd(th, fd["xE"] / sc); phiG = fwd(th, fd["xG"] / sc)
                f.write(struct.pack("<2i", len(phiE), len(phiG)))
                f.write(phiE.astype("<f8").tobytes()); f.write(phiG.astype("<f8").tobytes())
                f.write(free[idx[k]].astype("<i4").tobytes())
                f.write(th.astype("<f8").tobytes()); f.write(H.astype("<f8").tobytes())
            M = np.einsum("kgp,g,jgq->kjpq", J, tt["wr"], J)
            f.write(M.astype("<f8").tobytes())
log(f"toys {ids[0]}-{ids[-1]}: outputs written")
