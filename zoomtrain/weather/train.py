"""Compute-matched training of a 6 h ERA5 forecaster: Zoom x {AdamW, Muon}.

--zoom 1                        baseline: full-resolution tokens throughout
--zoom 4:0.6,1:0.4              Zoom Training: 2x2 token merge for 60% of FLOPs
--zoom 4:0.6,1:0.4 --coarse_mode lowres
                                progressive resizing (prior art): 60% of FLOPs on
                                2x-coarsened fields, then full resolution
Train years 2010-2017, test year 2018 (one-step loss and 6/24/72 h rollout RMSE).
"""
import torch

torch.set_flush_denormal(True)  # before any parallel op (see zoomtrain.device)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from zoomtrain.data.gridpack import GridPack  # noqa: E402
from zoomtrain.device import (DutyCycleLimiter, describe, maybe_compile,  # noqa: E402
                              setup_runtime)
from zoomtrain.models import ZoomViT  # noqa: E402
from zoomtrain.optim import build_optimizer  # noqa: E402
from zoomtrain.zoom import ZoomSchedule, lr_at  # noqa: E402

REPORT = ["z500", "t850", "t2m"]


class ERA5:
    """Normalized fields in memory plus forcings; reads fields.npy or a GridPack."""

    def __init__(self, path, device):
        meta = json.load(open(os.path.join(path, "meta.json")))
        gp = os.path.join(path, "fields.gridpack")
        if os.path.exists(gp):
            raw = torch.from_numpy(GridPack(gp).read_all())
            self.source = "gridpack"
        else:
            raw = torch.from_numpy(np.load(os.path.join(path, "fields.npy")))
            self.source = "npy"
        self.mean = torch.tensor(meta["mean"]).view(1, -1, 1, 1)
        self.std = torch.tensor(meta["std"]).view(1, -1, 1, 1)
        self.dstd = torch.tensor(meta["diff_std"]).view(1, -1, 1, 1)
        self.x = ((raw - self.mean) / self.std).to(device)             # (T, C, H, W)
        self.channels = meta["channels"]
        self.n_train, self.n_total = meta["n_train"], meta["n_total"]
        lat = torch.tensor(meta["lat"], dtype=torch.float32)
        lon = torch.tensor(meta["lon"], dtype=torch.float32)
        w = torch.cos(torch.deg2rad(lat))
        self.latw = (w / w.mean()).view(1, 1, -1, 1).to(device)
        times = np.array(meta["times"], dtype="datetime64[h]")
        hours = (times - times.astype("datetime64[D]")).astype(int)
        doy = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(int)
        self.hour = torch.tensor(hours, dtype=torch.float32, device=device)
        self.doy = torch.tensor(doy, dtype=torch.float32, device=device)
        self.lon = lon.to(device)
        self.static = torch.from_numpy(np.load(os.path.join(path, "static.npy"))).to(device)
        self.H, self.W = self.x.shape[-2:]
        self.res_scale = (self.dstd / self.std).to(device)   # residual target scaling
        self.std = self.std.to(device)
        self.device = device

    def forcings(self, t):
        """(B, 6, H, W): static fields + local solar time + day of year."""
        B = len(t)
        local = (self.hour[t].view(B, 1) + self.lon.view(1, -1) / 15.0) * (2 * math.pi / 24)
        lh = torch.stack([local.sin(), local.cos()], 1).view(B, 2, 1, self.W)
        lh = lh.expand(-1, -1, self.H, -1)
        ang = self.doy[t] * (2 * math.pi / 365.25)
        dy = torch.stack([ang.sin(), ang.cos()], 1).view(B, 2, 1, 1).expand(-1, -1, self.H, self.W)
        st = self.static.unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([st, lh, dy], 1)

    def inputs(self, xn, t):
        return torch.cat([xn, self.forcings(t)], 1)

    def batch(self, t):
        xn = self.x[t]
        y = (self.x[t + 1] - xn) / self.res_scale
        return self.inputs(xn, t), y


def lat_mse(pred, y, w):
    return ((pred.float() - y) ** 2 * w).mean()


@torch.no_grad()
def quick_eval(model, ds, idx, rt):
    model.eval()
    tot = 0.0
    for i in range(0, len(idx), 64):
        t = idx[i:i + 64]
        x, y = ds.batch(t)
        with rt.autocast():
            tot += lat_mse(model(x, 1), y, ds.latw).item() * len(t)
    model.train()
    return tot / len(idx)


@torch.no_grad()
def full_eval(model, ds, one_step_idx, ic_idx, rt, steps=(1, 4, 12)):
    """Latitude-weighted RMSE in physical units at +6 h, +24 h, +72 h vs persistence."""
    model.eval()
    ci = [ds.channels.index(c) for c in REPORT]
    phys = ds.std[0, ci].view(1, -1, 1, 1)
    out = {"val_loss": quick_eval(model, ds, one_step_idx, rt)}
    max_s = max(steps)
    se = {s: torch.zeros(len(ci), device=ds.device) for s in steps}
    se_p = {s: torch.zeros(len(ci), device=ds.device) for s in steps}
    for i in range(0, len(ic_idx), 32):
        t0 = ic_idx[i:i + 32]
        xn = ds.x[t0].clone()
        for s in range(1, max_s + 1):
            t = t0 + s - 1
            with rt.autocast():
                delta = model(ds.inputs(xn, t), 1).float()
            xn = xn + delta * ds.res_scale
            if s in steps:
                truth = ds.x[t0 + s]
                err = ((xn[:, ci] - truth[:, ci]) * phys) ** 2 * ds.latw
                se[s] += err.mean(dim=(2, 3)).sum(0)
                perr = ((ds.x[t0][:, ci] - truth[:, ci]) * phys) ** 2 * ds.latw
                se_p[s] += perr.mean(dim=(2, 3)).sum(0)
    n = len(ic_idx)
    for s in steps:
        h = 6 * s
        for j, c in enumerate(REPORT):
            out[f"rmse_{c}_{h}h"] = float((se[s][j] / n).sqrt())
            out[f"persist_{c}_{h}h"] = float((se_p[s][j] / n).sqrt())
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--zoom", default="1")
    ap.add_argument("--coarse_mode", default="merge", choices=["merge", "lowres"])
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--state_bits", type=int, default=32, choices=[32, 8])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--lr_restart", type=int, default=0,
                    help="restart warmup+cosine at every zoom stage switch")
    ap.add_argument("--mix_fine", type=float, default=0.0,
                    help="in coarse stages, run a k=1 step with this probability")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--gated", type=int, default=1)
    ap.add_argument("--merge_mode", default="", help="gate|mean|linear (overrides --gated)")
    ap.add_argument("--evals", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "weather"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rt = setup_runtime(args.device, args.precision, args.compile, args.threads, args.seed)
    if rt.device.type == "cpu":
        rt.amp_dtype = None
    ds = ERA5(args.data, rt.device)
    sched = ZoomSchedule.parse(args.zoom)
    c_in = len(ds.channels) + 6
    model = ZoomViT(c_in, len(ds.channels), (ds.H, ds.W), 2, args.d, args.layers, args.heads,
                    zoom_ks=sched.ks if args.coarse_mode == "merge" else (),
                    gated=bool(args.gated), merge_mode=args.merge_mode or None).to(rt.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optimizer(model, args.optim, args.lr, args.wd, state_bits=args.state_bits)
    fwd = maybe_compile(model, rt)
    limiter = DutyCycleLimiter(args.duty, rt.device)

    g = torch.Generator().manual_seed(args.seed)
    train_idx = torch.randperm(ds.n_train - 1, generator=g).to(rt.device)
    test_all = torch.arange(ds.n_train, ds.n_total - 13, device=rt.device)
    quick_idx = test_all[::11][:128]
    one_step_idx = test_all[::3]
    ic_idx = test_all[::12][:96]

    log = {"args": vars(args), "schedule": repr(sched), "params": n_params, "evals": [],
           "train": [], "runtime": describe(rt), "data_source": ds.source}
    print(f"[{args.name}] params={n_params/1e6:.2f}M zoom={sched} mode={args.coarse_mode} "
          f"optim={args.optim} budget={args.budget:.3e} data={ds.source} {describe(rt)}",
          flush=True)
    flops, step, samples, train_time, pos = 0.0, 0, 0, 0.0, 0
    next_eval = 0.0
    mix_rng = np.random.default_rng(args.seed + 1)
    while True:
        frac = flops / args.budget
        if frac >= next_eval - 1e-12:
            vl = quick_eval(model, ds, quick_idx, rt)
            log["evals"].append({"frac": frac, "flops": flops, "samples": samples,
                                 "step": step, "train_time": train_time, "val_loss": vl})
            print(f"  frac={frac:.3f} step={step} samples={samples} t={train_time:.0f}s "
                  f"val={vl:.4f}", flush=True)
            next_eval += 1.0 / args.evals
        if frac >= 1.0:
            break
        k, s0, s1 = sched.stage_at(frac)
        if k > 1 and args.mix_fine > 0 and mix_rng.random() < args.mix_fine:
            k = 1
        for gr in opt.param_groups:
            gr["lr"] = lr_at(frac, args.lr, warmup=args.warmup,
                             stage=(s0, s1) if args.lr_restart else None)
        t0 = time.perf_counter()
        if pos + args.batch > len(train_idx):
            pos = 0
        t = train_idx[pos:pos + args.batch]
        pos += args.batch
        x, y = ds.batch(t)
        with rt.autocast():
            if k > 1 and args.coarse_mode == "lowres":
                f = int(round(k ** 0.5))
                pred = model.forward_lowres(F.avg_pool2d(x, f), f)
                loss = lat_mse(pred, F.avg_pool2d(y, f),
                               F.avg_pool2d(ds.latw.expand(1, 1, -1, f), f))
            else:
                loss = lat_mse(fwd(x, k), y, ds.latw)
        opt.zero_grad(set_to_none=True)
        rt.backward(loss, opt, model.parameters())
        if rt.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        train_time += dt
        limiter(dt)
        flops += model.flops_per_step(args.batch, k)
        samples += args.batch
        step += 1
        if step % 25 == 0:
            log["train"].append({"step": step, "frac": frac, "k": k, "loss": loss.item(),
                                 "train_time": train_time})

    log["final"] = full_eval(model, ds, one_step_idx, ic_idx, rt)
    log["train_time"], log["samples"], log["steps"] = train_time, samples, step
    log["epochs"] = samples / len(train_idx)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
        json.dump(log, f, indent=1)
    fin = log["final"]
    print(f"[{args.name}] val={fin['val_loss']:.4f} "
          + " ".join(f"{c}@{h}h={fin[f'rmse_{c}_{h}h']:.2f}" for h in (6, 24, 72) for c in REPORT)
          + f" train_time={train_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
