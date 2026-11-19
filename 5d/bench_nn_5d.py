#!/usr/bin/env python3
"""bench_nn_5d.py -- time batched training of NPLM-style witness networks on GPU/CPU.

All fold networks (and optionally several toys) are trained AT ONCE as one batched model:
weights have a leading "network" dimension and the forward pass uses batched matmuls.
Loss per network (NPLM extended likelihood, fixed base at the null fit):
    L = - sum_{data} f(x) + sum_{base} omega (exp(f(x)) - 1)
Data and base sets of different sizes are padded with zero weight.
Optimizer: Adam (beta 0.9, 0.99, eps 1e-7, lr 1e-3, as in LMNP); weight clipping after each step.

Usage:
    python3 bench_nn_5d.py                         # 10 folds x 1 toy, (5,5,5), float32, cuda
    python3 bench_nn_5d.py --toys 8                # 80 networks at once
    python3 bench_nn_5d.py --device cpu --threads 8
"""
import argparse
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--folds", type=int, default=10)
ap.add_argument("--toys", type=int, default=1)
ap.add_argument("--n-data", type=int, default=7830, help="training data events per fold")
ap.add_argument("--n-base", type=int, default=39000, help="training base points per fold")
ap.add_argument("--hidden", default="5,5,5")
ap.add_argument("--clip", type=float, default=2.16)
ap.add_argument("--epochs", type=int, default=2000)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--graph", action="store_true",
                help="capture the whole training step in a CUDA graph and replay it (removes launch overhead)")
a = ap.parse_args()
if a.threads > 0:
    torch.set_num_threads(a.threads)
dev, dt = torch.device(a.device), getattr(torch, a.dtype)
F = a.folds * a.toys                                   # networks trained together
sizes = [5] + [int(h) for h in a.hidden.split(",")] + [1]
n_par = sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))
torch.manual_seed(0)

# synthetic stand-in data with realistic shapes (standardized features)
nmax = a.n_data + a.n_base
X = torch.randn(F, nmax, 5, device=dev, dtype=dt)
w = torch.zeros(F, nmax, device=dev, dtype=dt)          # base weights (omega); 0 on data rows
w[:, a.n_data:] = a.n_data / a.n_base
is_data = torch.zeros(F, nmax, device=dev, dtype=dt); is_data[:, :a.n_data] = 1.0

# batched MLP parameters: W_l (F, in, out), b_l (F, 1, out); Glorot-uniform init
params = []
for i, o in zip(sizes[:-1], sizes[1:]):
    lim = (6.0 / (i + o)) ** 0.5
    params.append((torch.rand(F, i, o, device=dev, dtype=dt) * 2 - 1).mul_(lim).requires_grad_())
    params.append(torch.zeros(F, 1, o, device=dev, dtype=dt).requires_grad_())


def forward(X):
    h = X
    L = len(params) // 2
    for l in range(L):
        h = torch.baddbmm(params[2 * l + 1], h, params[2 * l])
        if l < L - 1:
            h = torch.sigmoid(h)
    return h.squeeze(-1)                                 # (F, nmax)


use_graph = a.graph and dev.type == "cuda"
opt = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.99), eps=1e-7, capturable=use_graph)


def body():
    f = forward(X)
    loss = (-(is_data * f) + w * torch.expm1(f)).sum()   # sum over all networks: independent gradients
    loss.backward()
    opt.step()
    with torch.no_grad():
        for p in params:
            p.clamp_(-a.clip, a.clip)
    return loss


if use_graph:
    # standard whole-network capture: warm up on a side stream, then record one step
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            opt.zero_grad(set_to_none=True); body()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_loss = body()

    def step():
        graph.replay()
        return static_loss
else:
    def step():
        opt.zero_grad(set_to_none=True)
        return body()


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


for _ in range(100):                                      # warm-up
    step()
sync(); t0 = time.perf_counter()
for _ in range(a.epochs):
    step()
sync(); dt_ep = (time.perf_counter() - t0) / a.epochs

name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else f"CPU ({torch.get_num_threads()} threads)"
print(f"device {name}, {a.dtype}{', CUDA graph' if use_graph else ''}; {F} networks ({a.folds} folds x {a.toys} toys) of {sizes} = {n_par} parameters each")
print(f"rows per network {nmax} ({a.n_data} data + {a.n_base} base)")
print(f"{dt_ep * 1e3:.3f} ms per epoch for all {F} networks together")
for ep in (1e5, 3e5, 1e6):
    print(f"  {ep:9.0f} epochs: {dt_ep * ep / 60:7.1f} min for these {a.toys} toy(s)  "
          f"-> {dt_ep * ep / a.toys / 60:6.2f} min per toy")
if dev.type == "cuda":
    print(f"peak GPU memory {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MB")
