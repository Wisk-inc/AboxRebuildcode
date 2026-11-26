"""Compute-matched training of a 6 h ERA5 forecaster: Zoom x {AdamW, Muon}.

--zoom 1                        baseline: full-resolution tokens throughout
--zoom 4:0.6,1:0.4              Zoom Training: 2x2 token merge for 60% of FLOPs
--zoom 4:0.6,1:0.4 --coarse_mode lowres
                                progressive resizing (prior art): 60% of FLOPs on
                                2x-coarsened fields, then full resolution
--recipe best                   the best settings measured so far (small scale, CPU)
                                for every option the command line does not give

Splits come from meta.json: by default train on the years before the test year
(meta n_train; 2018 for the default prepare-era5 range) and evaluate on the test
year. With --val_year Y, train on the years before Y, run the periodic evals on
the validation years [Y, test year) and touch the test year only in the final
evaluation; data prepared with prepare-era5 --val_year (meta n_val / splits) uses
its validation split that way by default. Metrics: one-step loss and latitude-weighted 6/24/72 h rollout RMSE
and ACC, next to persistence and a train-years climatology.

--save PATH keeps the trained model; roll it out with
    bantam weather --forecast --load PATH --data DIR --init 2018-03-01T00 --steps 12
"""
import torch

torch.set_flush_denormal(True)  # before any parallel op (see bantamtrain.device)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ..data.gridpack import GridPack  # noqa: E402
from ..device import (DutyCycleLimiter, describe, maybe_compile,  # noqa: E402
                              setup_runtime)
from ..lowmem import load_checkpoint  # noqa: E402
from ..models import ZoomViT  # noqa: E402
from ..optim import build_optimizer  # noqa: E402
from ..zoom import ZoomSchedule, lr_at  # noqa: E402
from .recipes import NAMES as RECIPE_NAMES, apply_recipe  # noqa: E402

REPORT = ["z500", "t850", "t2m"]
# options a --resume checkpoint must share with the command line: the model, and (as
# pretrain's RESUME_FIELDS) the data order and the LR / zoom / FLOP schedule
MODEL_FIELDS = ("zoom", "coarse_mode", "d", "layers", "heads", "gated", "merge_mode", "optim",
                "state_bits", "val_year")
RESUME_FIELDS = MODEL_FIELDS + ("budget", "flops_mode", "batch", "batch_scale_k", "lr", "warmup",
                                "sched", "decay_frac", "ns", "lr_restart", "mix_fine", "wd", "seed")


def time_features(times):
    """(hour of day, 0-based day of year) int arrays for datetime64 times."""
    times = np.asarray(times, dtype="datetime64[h]")
    hours = (times - times.astype("datetime64[D]")).astype(int)
    doy = (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(int)
    return hours, doy


def forcings(static, hour, doy, lon):
    """(B, 6, H, W): static fields + local solar time + day of year."""
    B, (H, W) = len(hour), static.shape[-2:]
    local = (hour.view(B, 1) + lon.view(1, -1) / 15.0) * (2 * math.pi / 24)
    lh = torch.stack([local.sin(), local.cos()], 1).view(B, 2, 1, W)
    lh = lh.expand(-1, -1, H, -1)
    ang = doy * (2 * math.pi / 365.25)
    dy = torch.stack([ang.sin(), ang.cos()], 1).view(B, 2, 1, 1).expand(-1, -1, H, W)
    st = static.unsqueeze(0).expand(B, -1, -1, -1)
    return torch.cat([st, lh, dy], 1)


def pack_holds_npy(gp, npy, meta):
    """True if fields.gridpack was made from this fields.npy: it records meta.json's prep_id
    (pack-era5 >= 0.3; a different prep_id means stale), or else (older packs) its first, middle and last time steps agree
    with fields.npy to within one quantization step. Modification times are not used: a
    copy of the data directory (cp, rsync without -t) reorders them."""
    pack = GridPack(gp, verify=False)
    pid = (pack.h.get("source") or {}).get("prep_id")
    if pid is not None and meta.get("prep_id") is not None:
        return pid == meta["prep_id"]
    x = np.load(npy, mmap_mode="r")
    if list(x.shape) != list(pack.h["shape"]):
        return False
    tol = pack.scales.reshape(-1).astype(np.float64) * 1.01
    for t in sorted({0, len(x) // 2, len(x) - 1}):
        err = np.abs(pack.read(t, 1)[0].astype(np.float64) - x[t]).max(axis=(1, 2))
        if (err > tol).any():
            return False
    return True


def field_source(path, meta):
    """("gridpack" | "npy", file) holding the fields, checked against meta.json.

    A GridPack that does not hold the data of fields.npy (pack_holds_npy: another
    prepare-era5 run, other shape) is stale: it is skipped for fields.npy with a warning,
    and without fields.npy it is an error. Data whose shape disagrees with meta.json is an
    error."""
    gp, npy = os.path.join(path, "fields.gridpack"), os.path.join(path, "fields.npy")
    kind, f = ("gridpack", gp) if os.path.exists(gp) else ("npy", npy)
    if kind == "gridpack" and os.path.exists(npy) and not pack_holds_npy(gp, npy, meta):
        print(f"warning: {gp} does not hold the data of fields.npy (stale pack from an earlier "
              "prepare-era5 run?); reading fields.npy. Rerun pack-era5 or delete it", flush=True)
        kind, f = "npy", npy
    shape = GridPack(f).h["shape"] if kind == "gridpack" else np.load(f, mmap_mode="r").shape
    want = (len(meta["times"]), len(meta["channels"]), len(meta["lat"]), len(meta["lon"]))
    if tuple(shape) != want:
        raise ValueError(f"{f} has shape {tuple(shape)} but meta.json describes {want}: stale or "
                         "mismatched data; rerun prepare-era5"
                         + (" and pack-era5" if kind == "gridpack" else ""))
    return kind, f


def read_fields(kind, f, start=0, length=None):
    """float32 rows [start, start + length) of the fields (all rows by default)."""
    if kind == "gridpack":
        gp = GridPack(f)
        if start or length is not None:
            return gp.read(start, length if length is not None else gp.T - start)
        out = np.empty((gp.T, gp.C, gp.H, gp.W), np.float32)   # chunk by chunk: one copy in RAM
        for i in range(0, gp.T, gp.chunk):
            out[i:i + gp.chunk] = gp.read_chunk(i // gp.chunk)
        return out
    if start == 0 and length is None:
        return np.load(f)
    return np.array(np.load(f, mmap_mode="r")[start:None if length is None else start + length])


def train_stats(raw, n, chunk=256):
    """Per-channel mean, std and 6 h-change std over rows [0, n), as prepare-era5 computes them."""
    _, C, H, W = raw.shape
    s = np.zeros(C)
    for i in range(0, n, chunk):
        s += raw[i:min(i + chunk, n)].sum(axis=(0, 2, 3), dtype=np.float64)
    mean = s / (n * H * W)
    ss, ds, dss = np.zeros(C), np.zeros(C), np.zeros(C)
    for i in range(0, n, chunk):
        b = raw[i:min(i + chunk, n)].astype(np.float64) - mean[:, None, None]
        ss += (b * b).sum(axis=(0, 2, 3))
        j = min(i + chunk, n - 1)                      # pairs (t, t+1) with t+1 < n
        if j > i:
            d = raw[i + 1:j + 1].astype(np.float64) - raw[i:j]
            ds += d.sum(axis=(0, 2, 3))
            dss += (d * d).sum(axis=(0, 2, 3))
    m = (n - 1) * H * W
    return {"mean": mean.tolist(), "std": np.sqrt(ss / (n * H * W)).tolist(),
            "diff_std": np.sqrt(np.maximum(dss / m - (ds / m) ** 2, 0.0)).tolist()}


def _circular_box(a, w):
    """Sums of a over a circular window of 2w+1 entries along dim 0."""
    w = min(w, (len(a) - 1) // 2)
    if w <= 0:
        return a
    p = torch.cat([a[-w:], a, a[:w]])
    c = torch.cat([torch.zeros_like(p[:1]), p.cumsum(0)])
    return c[2 * w + 1:] - c[:-(2 * w + 1)]


class ERA5:
    """Normalized fields in memory plus forcings; reads fields.npy or a GridPack.

    stats_end: normalize with statistics of rows [0, stats_end) (the train years of a
    --val_year split) instead of the ones in meta.json.
    """

    def __init__(self, path, device, stats_end=None):
        meta = json.load(open(os.path.join(path, "meta.json")))
        self.source, f = field_source(path, meta)
        raw = read_fields(self.source, f)
        own = stats_end is not None and stats_end != meta["n_train"]
        stats = train_stats(raw, stats_end) if own else meta
        self.stats_rows = stats_end if own else meta["n_train"]
        self.mean = torch.tensor(stats["mean"]).view(1, -1, 1, 1)
        self.std = torch.tensor(stats["std"]).view(1, -1, 1, 1)
        self.dstd = torch.tensor(stats["diff_std"]).view(1, -1, 1, 1)
        # in place: one copy of the data in RAM, same values as (raw - mean) / std
        self.x = torch.from_numpy(raw).sub_(self.mean).div_(self.std).to(device)  # (T, C, H, W)
        del raw
        self.channels = meta["channels"]
        self.n_train, self.n_total = meta["n_train"], meta["n_total"]
        self.lat_deg, self.lon_deg = meta["lat"], meta["lon"]
        lat = torch.tensor(meta["lat"], dtype=torch.float32)
        lon = torch.tensor(meta["lon"], dtype=torch.float32)
        w = torch.cos(torch.deg2rad(lat))
        self.latw = (w / w.mean()).view(1, 1, -1, 1).to(device)
        self.times = np.array(meta["times"], dtype="datetime64[h]")
        hours, doy = time_features(self.times)
        self.step_hours = int((self.times[1] - self.times[0]).astype(int)) if len(hours) > 1 else 6
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
        return forcings(self.static, self.hour[t], self.doy[t], self.lon)

    def inputs(self, xn, t):
        return torch.cat([xn, self.forcings(t)], 1)

    def batch(self, t):
        xn = self.x[t]
        y = (self.x[t + 1] - xn) / self.res_scale
        return self.inputs(xn, t), y

    @torch.no_grad()
    def climatology(self, end, channels, window=30):
        """clim(t) -> (B, len(channels), H, W): day-of-year x hour-of-day mean of rows [0, end)
        (the train years), smoothed over a circular (2*window+1)-day box. Days the box never
        reaches (short train spans) fall back to that hour's train mean."""
        dev = self.x.device
        hours = self.hour.long()
        uh = torch.unique(hours)
        hi = torch.searchsorted(uh, hours)
        nh, C = len(uh), len(channels)
        slot = self.doy.long() * nh + hi
        ch = torch.tensor(channels, device=dev)
        S = torch.zeros(366 * nh, C, self.H, self.W, dtype=torch.float64, device=dev)
        N = torch.zeros(366 * nh, dtype=torch.float64, device=dev)
        for i in range(0, end, 256):
            j = min(i + 256, end)
            S.index_add_(0, slot[i:j], self.x[i:j, ch].double())
            N.index_add_(0, slot[i:j], torch.ones(j - i, dtype=torch.float64, device=dev))
        S, N = S.view(366, nh, C, self.H, self.W), N.view(366, nh)
        table = torch.empty(366, nh, C, self.H, self.W, device=dev)
        for h in range(nh):
            s, n = _circular_box(S[:, h], window), _circular_box(N[:, h], window)
            fallback = S[:, h].sum(0) / N[:, h].sum().clamp_min(1)
            mean = s / n.clamp_min(1).view(-1, 1, 1, 1)
            table[:, h] = torch.where((n > 0).view(-1, 1, 1, 1), mean, fallback).float()
        doy = self.doy.long()
        return lambda t: table[doy[t], hi[t]]


def lat_mse(pred, y, w):
    return ((pred.float() - y) ** 2 * w).mean()


def lat_acc(f, o, w):
    """Latitude-weighted anomaly correlation per (sample, channel); f, o: (B, C, H, W)."""
    num = (f * o * w).sum((2, 3))
    den = ((f * f * w).sum((2, 3)) * (o * o * w).sum((2, 3))).sqrt()
    return num / den.clamp_min(1e-30)


@torch.no_grad()
def quick_eval(model, ds, idx, rt):
    if len(idx) == 0:
        raise ValueError("quick_eval: no evaluation times (empty split)")
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
def full_eval(model, ds, one_step_idx, ic_idx, rt, steps=(1, 4, 12), clim=None):
    """Latitude-weighted RMSE in physical units at +6 h, +24 h, +72 h vs persistence.

    With a climatology clim(t) (ERA5.climatology) also the climatology RMSE (clim_*) and
    the latitude-weighted ACC of the model (acc_*) and of persistence (acc_persist_*)."""
    if len(ic_idx) == 0:
        raise ValueError("full_eval: no rollout initial conditions (empty split)")
    model.eval()
    ci = [ds.channels.index(c) for c in REPORT]
    phys = ds.std[0, ci].view(1, -1, 1, 1)
    out = {"val_loss": quick_eval(model, ds, one_step_idx, rt)}
    max_s = max(steps)
    zero = lambda: {s: torch.zeros(len(ci), device=ds.device) for s in steps}  # noqa: E731
    se, se_p, se_c, acc, acc_p = zero(), zero(), zero(), zero(), zero()
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
                if clim is not None:
                    c = clim(t0 + s)
                    o = truth[:, ci] - c
                    se_c[s] += ((o * phys) ** 2 * ds.latw).mean(dim=(2, 3)).sum(0)
                    acc[s] += lat_acc(xn[:, ci] - c, o, ds.latw).sum(0)
                    acc_p[s] += lat_acc(ds.x[t0][:, ci] - c, o, ds.latw).sum(0)
    n = len(ic_idx)
    for s in steps:
        h = 6 * s
        for j, c in enumerate(REPORT):
            out[f"rmse_{c}_{h}h"] = float((se[s][j] / n).sqrt())
            out[f"persist_{c}_{h}h"] = float((se_p[s][j] / n).sqrt())
    if clim is not None:
        for s in steps:
            for j, c in enumerate(REPORT):
                out[f"clim_{c}_{6 * s}h"] = float((se_c[s][j] / n).sqrt())
                out[f"acc_{c}_{6 * s}h"] = float(acc[s][j] / n)
                out[f"acc_persist_{c}_{6 * s}h"] = float(acc_p[s][j] / n)
    model.train()
    return out


class Sampler:
    """Training start times in a seeded random order, reshuffled every epoch."""

    def __init__(self, n, seed, device):
        self.n, self.device = n, device
        self.g = torch.Generator().manual_seed(seed)
        self.idx = torch.randperm(n, generator=self.g).to(device)
        self.pos, self.epochs = 0, 0

    def next(self, batch):
        if self.pos + batch > self.n:
            self.idx = torch.randperm(self.n, generator=self.g).to(self.device)
            self.pos, self.epochs = 0, self.epochs + 1
        t = self.idx[self.pos:self.pos + batch]
        self.pos += batch
        return t

    def state_dict(self):
        return {"idx": self.idx.cpu(), "pos": self.pos, "epochs": self.epochs,
                "gen": self.g.get_state()}

    def load_state_dict(self, s):
        self.idx, self.pos, self.epochs = s["idx"].to(self.device), s["pos"], s["epochs"]
        self.g.set_state(s["gen"])


def optimizer_step_flops(opt):
    """FLOPs of one optimizer step: bantamtrain.optim.optimizer_flops when it exists, else
    Muon's Newton-Schulz matmuls plus ~12 FLOPs per element-wise (AdamW) parameter."""
    try:
        from ..optim import optimizer_flops
    except ImportError:
        optimizer_flops = None
    if optimizer_flops is not None:
        return float(optimizer_flops(opt))
    tot = 0.0
    for o in getattr(opt, "optimizers", [opt]):
        for gr in o.param_groups:
            for p in gr["params"]:
                if not p.requires_grad:
                    continue
                if "ns_steps" in gr and p.dim() == 2:
                    s = getattr(p, "muon_split", 1)
                    m, n = sorted((p.shape[0] // s, p.shape[1]))
                    tot += gr["ns_steps"] * s * (4 * m * m * n + 2 * m ** 3)
                else:
                    tot += 12 * p.numel()
    return float(tot)


def first_test_row(meta):
    """First test row: meta splits/n_val (prepare-era5 --val_year) put a validation split
    between train and test; older meta.json files end the train years at the test year."""
    sp = meta.get("splits") or {}
    return int(sp["test"][0]) if "test" in sp else int(meta["n_train"] + meta.get("n_val", 0))


def year_start(times, year):
    """Index of the first of the (sorted) times in `year` or later."""
    years = times.astype("datetime64[Y]").astype(int) + 1970
    return int(np.searchsorted(years, int(year)))


def span(times, idx):
    """JSON summary of the times at these indices."""
    idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx, dtype=np.int64)
    if len(idx) == 0:
        return {"n": 0}
    t = times[idx]
    return {"n": int(len(t)), "first": str(t[0]), "last": str(t[-1]),
            "hours": sorted({int(h) for h in time_features(t)[0]})}


def save_ckpt(path, state):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_ckpt(path):
    """torch.load(weights_only=True); pickled 8-bit QState optimizer state (bantamtrain <= 0.2)
    is allow-listed, anything else that is not plain data is refused."""
    return load_checkpoint(path)[0]


def save_model(path, model, ds, args=None):
    """Weights, architecture and normalization stats: all forecast() needs besides the data.
    Plain tensors and Python values only, so torch.load(path, weights_only=True) reads it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    args = dict(vars(args)) if args is not None else {}
    if args.get("data"):
        args["data"] = os.path.abspath(args["data"])
    torch.save({"format": "bantam-weather-1", "config": model.config(),
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "norm": {"mean": ds.mean.flatten().tolist(), "std": ds.std.flatten().tolist(),
                         "diff_std": ds.dstd.flatten().tolist()},
                "channels": list(ds.channels), "lat": list(ds.lat_deg), "lon": list(ds.lon_deg),
                "step_hours": ds.step_hours, "args": args}, path)


def load_model(path, device="cpu"):
    """(ZoomViT in eval mode, checkpoint dict) from a --save file."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = ZoomViT(**ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def forecast(load, data, init, steps=12, out=None, device="cpu"):
    """Roll a --save'd model forward `steps` 6 h steps from the analysis at `init`.

    data: the ERA5 dir holding `init` (None: the one the model was trained on).
    Returns (steps + 1, C, H, W) float32 in physical units, index 0 = the initial state.
    With `out`, writes it as .npy plus a .json sidecar: valid times, channels, grid and,
    where `data` covers the valid times, the latitude-weighted RMSE against it."""
    dev = torch.device(device)
    model, ck = load_model(load, dev)
    data = data or ck.get("args", {}).get("data")
    if not data:
        raise ValueError("forecast: pass the ERA5 data dir (--data)")
    meta = json.load(open(os.path.join(data, "meta.json")))
    for key in ("channels", "lat", "lon"):
        if list(meta[key]) != list(ck[key]):
            raise ValueError(f"{data}: {key} differ from the ones the model was trained on")
    times = np.array(meta["times"], dtype="datetime64[h]")
    t0 = np.datetime64(init, "h")
    i0 = int(np.searchsorted(times, t0))
    if i0 >= len(times) or times[i0] != t0:
        raise ValueError(f"init time {init} is not in {data}/meta.json "
                         f"({times[0]} .. {times[-1]})")
    kind, f = field_source(data, meta)
    valid = t0 + np.arange(steps + 1) * np.timedelta64(int(ck.get("step_hours", 6)), "h")
    n_truth = min(steps + 1, len(times) - i0)
    hit = times[i0:i0 + n_truth] == valid[:n_truth]   # truth only up to a gap in the data
    n_truth = n_truth if hit.all() else int(np.argmin(hit))
    truth = read_fields(kind, f, i0, n_truth)
    mean, std, dstd = (torch.tensor(ck["norm"][k], device=dev).view(1, -1, 1, 1)
                       for k in ("mean", "std", "diff_std"))
    static = torch.from_numpy(np.load(os.path.join(data, "static.npy"))).to(dev)
    lon = torch.tensor(meta["lon"], dtype=torch.float32, device=dev)
    hours, doy = (torch.tensor(a, dtype=torch.float32, device=dev) for a in time_features(valid))
    xn = (torch.from_numpy(truth[:1]).to(dev) - mean) / std
    states = [xn]
    for s in range(steps):
        inp = torch.cat([xn, forcings(static, hours[s:s + 1], doy[s:s + 1], lon)], 1)
        xn = xn + model(inp, 1).float() * (dstd / std)
        states.append(xn)
    fc = (torch.cat(states) * std + mean).cpu().numpy().astype(np.float32)
    fc[0] = truth[0]
    lead = [int(h) for h in (valid - t0).astype(int)]
    side = {"init": str(t0), "valid_times": [str(v) for v in valid], "lead_hours": lead,
            "channels": list(ck["channels"]), "lat": list(ck["lat"]), "lon": list(ck["lon"]),
            "model": os.path.abspath(load)}
    if n_truth > 1:
        w = np.cos(np.deg2rad(np.asarray(meta["lat"])))
        err = (fc[:n_truth] - truth) ** 2 * (w / w.mean())[None, None, :, None]
        side["rmse"] = {c: np.sqrt(err[:, i].mean(axis=(1, 2))).tolist()
                        for i, c in enumerate(side["channels"])}
        side["rmse_lead_hours"] = lead[:n_truth]
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        np.save(out, fc)
        with open(os.path.splitext(out)[0] + ".json", "w") as fh:
            json.dump(side, fh, indent=1)
    return fc


RECIPE_HELP = (
    "best: the best settings measured so far, for each of these options that the command line "
    "does not give: --optim muon --lr 1.6e-2 --coarse_mode lowres --zoom 4:0.6,1:0.4 "
    "(low-resolution warm-up for 60%% of the FLOPs; the LR only with --optim muon). Measured at "
    "small scale on CPU only (ERA5 5.625 deg 64x32, 1.26M-param ZoomViT, 5.5e13 FLOPs, 2018 "
    "test year): one-step MSE 0.112 (1 seed) vs 0.124 for the Muon baseline (2 seeds); not "
    "tested on larger models or GPUs. none (default): no change. The log records args.recipe, "
    "the resolved values and args.recipe_applied (the options it set)")


def build_parser():
    ap = argparse.ArgumentParser(prog="bantam weather", description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="dir written by prepare-era5 (--forecast: "
                                                 "default = the model's training data)")
    ap.add_argument("--recipe", default="none", choices=RECIPE_NAMES, help=RECIPE_HELP)
    ap.add_argument("--zoom", default="1")
    ap.add_argument("--coarse_mode", default="merge", choices=["merge", "lowres"])
    ap.add_argument("--budget", type=float, default=None, help="training FLOPs (required to train)")
    ap.add_argument("--flops_mode", default="model", choices=["model", "total"],
                    help="what the budget counts: model fwd/bwd FLOPs, or also optimizer FLOPs "
                         "(Muon Newton-Schulz); both are logged either way")
    ap.add_argument("--optim", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--state_bits", type=int, default=32, choices=[32, 8])
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch_scale_k", type=int, default=0,
                    help="use batch*k samples at zoom k (constant tokens per step; both modes)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "wsd"])
    ap.add_argument("--decay_frac", type=float, default=0.45)
    ap.add_argument("--ns", default="quintic", choices=["quintic", "polar"])
    ap.add_argument("--lr_restart", type=int, default=0,
                    help="restart warmup+cosine at every zoom stage switch")
    ap.add_argument("--mix_fine", type=float, default=0.0,
                    help="in coarse stages, run a k=1 step with this probability")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--gated", type=int, default=1)
    ap.add_argument("--merge_mode", default="", choices=["", "gate", "mean", "linear"],
                    help="gate|mean|linear (overrides --gated)")
    ap.add_argument("--val_year", type=int, default=None,
                    help="train on years before this one, run the periodic evals on it (up to "
                         "the test year) and evaluate the test year only at the end")
    ap.add_argument("--evals", type=int, default=10, help="periodic evals (0: first + final)")
    ap.add_argument("--ic_stride", type=int, default=12,
                    help="rollout initial conditions: every Nth 6 h time of the eval split "
                         "(12 = 00 UTC every 3 days; 14 alternates 00/12 UTC)")
    ap.add_argument("--max_ics", type=int, default=96, help="cap on rollout ICs (0 = no cap)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--precision", default="auto")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--duty", type=float, default=1.0)
    ap.add_argument("--name", default=None, help="run name (required to train)")
    ap.add_argument("--out", default=os.path.join("results", "weather"))
    ap.add_argument("--save", default="", help="write the trained model (weights, config, "
                                               "normalization stats) to this file")
    ap.add_argument("--ckpt_every", type=int, default=0,
                    help="write a resumable checkpoint (model+optimizer+data order) every N steps")
    ap.add_argument("--ckpt", default="", help="checkpoint path (default <out>/<name>.ckpt)")
    ap.add_argument("--resume", default="", help="resume from this resumable checkpoint")
    ap.add_argument("--stop_step", type=int, default=0, help="stop after this many steps (tests)")
    ap.add_argument("--forecast", action="store_true",
                    help="roll out a --save'd model (--load, --init, --steps) instead of training")
    ap.add_argument("--load", default="", help="--forecast: model file written by --save")
    ap.add_argument("--init", default="", help="--forecast: initial time, e.g. 2018-03-01T00")
    ap.add_argument("--steps", type=int, default=12, help="--forecast: number of 6 h steps")
    ap.add_argument("--forecast_out", default="",
                    help="--forecast: output .npy (default <out>/forecasts/forecast_<init>.npy)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.forecast:
        if not (args.load and args.init) or args.steps < 0:
            ap.error("--forecast needs --load PATH, --init TIME and --steps >= 0")
        # own subdir: run logs are <out>/*.json, and summarize.py reads them all as runs
        out = args.forecast_out or os.path.join(args.out, "forecasts", f"forecast_{args.init}.npy")
        dev = args.device if args.device != "auto" else \
            ("cuda" if torch.cuda.is_available() else "cpu")
        fc = forecast(args.load, args.data, args.init, args.steps, out, dev)
        print(f"forecast {tuple(fc.shape)} from {args.init} -> {out}", flush=True)
        return
    if args.data is None or args.budget is None or not args.name:
        ap.error("the following arguments are required to train: --data, --budget, --name")
    recipe_note = apply_recipe(args, build_parser, argv, "weather")
    if args.evals < 0 or args.ic_stride < 1 or args.max_ics < 0:
        ap.error("--evals and --max_ics must be >= 0, --ic_stride >= 1")
    if not args.duty > 0:
        ap.error(f"--duty must be in (0, 1], got {args.duty}")
    args.ckpt = args.ckpt or os.path.join(args.out, f"{args.name}.ckpt")

    os.makedirs(args.out, exist_ok=True)
    rt = setup_runtime(args.device, args.precision, args.compile, args.threads, args.seed)
    if rt.device.type == "cpu":
        rt.amp_dtype = None
    meta = json.load(open(os.path.join(args.data, "meta.json")))
    times = np.array(meta["times"], dtype="datetime64[h]")
    test_start, n_total = first_test_row(meta), meta["n_total"]
    if n_total - 13 <= test_start:
        ap.error(f"{args.data}: fewer than 14 time steps after the train years, nothing to test on")
    # validation years: --val_year, else the val split prepare-era5 --val_year wrote (n_val)
    val = args.val_year is not None or test_start > meta["n_train"]
    train_end = year_start(times, args.val_year) if args.val_year is not None else meta["n_train"]
    if val and not 1 < train_end < test_start - 13:
        ap.error(f"--val_year {args.val_year}: needs train data before it and at least 14 steps "
                 f"of it before the test period ({times[test_start]} onwards); "
                 f"the data covers {times[0]} .. {times[-1]}")
    resume = load_ckpt(args.resume) if args.resume else None
    if resume is not None:
        bad = [k for k in RESUME_FIELDS if resume["args"].get(k) != getattr(args, k)]
        if bad:
            ap.error(f"--resume: options differ from the checkpoint: {bad}")
    ds = ERA5(args.data, rt.device, stats_end=train_end)
    try:
        sched = ZoomSchedule.parse(args.zoom)
    except ValueError as e:
        ap.error(f"--zoom {args.zoom!r}: {e}")
    c_in = len(ds.channels) + 6
    torch.manual_seed(args.seed)          # model init depends on --seed only
    try:
        model = ZoomViT(c_in, len(ds.channels), (ds.H, ds.W), 2, args.d, args.layers, args.heads,
                        zoom_ks=sched.ks if args.coarse_mode == "merge" else (),
                        gated=bool(args.gated), merge_mode=args.merge_mode or None)
        fs = {k: model.zoom_factor(k) for k, _ in sched.stages}     # every stage, k=1 too
    except ValueError as e:
        ap.error(str(e))
    model = model.to(rt.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optimizer(model, args.optim, args.lr, args.wd, state_bits=args.state_bits,
                          ns=args.ns)
    opt_flops = optimizer_step_flops(opt)
    fwd = maybe_compile(model, rt)
    low = maybe_compile(model.forward_lowres, rt) if args.coarse_mode == "lowres" else None
    limiter = DutyCycleLimiter(args.duty, rt.device)

    sampler = Sampler(train_end - 1, args.seed, rt.device)   # t and t+1 both in the train years
    test_all = torch.arange(test_start, n_total - 13, device=rt.device)
    eval_all = torch.arange(train_end, test_start - 13, device=rt.device) if val else test_all
    quick_idx = eval_all[::11][:128]
    ics = lambda a: a[::args.ic_stride][:args.max_ics or None]  # noqa: E731
    one_step_idx, ic_idx = test_all[::3], ics(test_all)
    split = {"eval": "val" if val else "test", "val_from": ("--val_year" if args.val_year is not None
                                                             else "meta.json" if val else None),
             "train": span(times, np.arange(train_end)),
             "test": span(times, np.arange(test_start, n_total)), "test_ics": span(times, ic_idx),
             "norm_stats_rows": ds.stats_rows}
    if val:
        split["val"] = span(times, np.arange(train_end, test_start))
        split["val_ics"] = span(times, ics(eval_all))

    log = {"args": vars(args), "schedule": repr(sched), "params": n_params, "evals": [],
           "train": [], "runtime": describe(rt), "data_source": ds.source, "split": split,
           "flops_mode": args.flops_mode, "opt_flops_per_step": opt_flops}
    print(f"[{args.name}] params={n_params/1e6:.2f}M zoom={sched} mode={args.coarse_mode} "
          f"optim={args.optim} budget={args.budget:.3e} ({args.flops_mode} FLOPs) "
          f"data={ds.source} {describe(rt)}", flush=True)
    if recipe_note:
        print(f"  {recipe_note}", flush=True)
    tr, te = split["train"], split["test"]
    print(f"  split: train {tr['first']}..{tr['last']}, periodic evals on "
          + (f"val {split['val']['first']}..{split['val']['last']}" if val else "the test period")
          + f", final eval on test {te['first']}..{te['last']} ({len(ic_idx)} rollout ICs)",
          flush=True)
    flops, model_flops, opt_total = 0.0, 0.0, 0.0
    step, samples, train_time, compile_time = 0, 0, 0.0, 0.0
    next_eval = 0.0
    mix_rng = np.random.default_rng(args.seed + 1)
    compiled = set()
    if resume is not None:
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["opt"])
        sampler.load_state_dict(resume["sampler"])
        pr = resume["progress"]
        flops, model_flops, opt_total = pr["flops"], pr["model_flops"], pr["opt_flops"]
        step, samples, train_time, compile_time, next_eval = (
            pr["step"], pr["samples"], pr["train_time"], pr["compile_time"], pr["next_eval"])
        log = resume["log"]       # this invocation's args, as pretrain records them
        log.setdefault("resume_history", []).append({"step": step, "args": log["args"]})
        log.update(args=vars(args), resumed_from=args.resume)
        torch.set_rng_state(resume["rng"])
        mix_rng.bit_generator.state = resume["mix_rng"]
        if rt.scaler is not None and "scaler" in resume:
            rt.scaler.load_state_dict(resume["scaler"])
        print(f"  resumed at step {step} (frac={flops / args.budget:.3f})", flush=True)

    def checkpoint():
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "sampler": sampler.state_dict(), "log": log, "args": vars(args),
                 "progress": {"flops": flops, "model_flops": model_flops, "opt_flops": opt_total,
                              "step": step, "samples": samples, "train_time": train_time,
                              "compile_time": compile_time, "next_eval": next_eval},
                 "rng": torch.get_rng_state(), "mix_rng": mix_rng.bit_generator.state}
        if rt.scaler is not None:
            state["scaler"] = rt.scaler.state_dict()
        save_ckpt(args.ckpt, state)

    while True:
        frac = flops / args.budget
        if frac >= next_eval - 1e-12:
            vl = quick_eval(model, ds, quick_idx, rt)
            log["evals"].append({"frac": frac, "flops": flops, "samples": samples,
                                 "step": step, "train_time": train_time, "val_loss": vl})
            print(f"  frac={frac:.3f} step={step} samples={samples} t={train_time:.0f}s "
                  f"val={vl:.4f}", flush=True)
            next_eval += 1.0 / args.evals if args.evals else math.inf
        if frac >= 1.0:
            break
        k, s0, s1 = sched.stage_at(frac)
        if k > 1 and args.mix_fine > 0 and mix_rng.random() < args.mix_fine:
            k = 1
        lr = lr_at(frac, args.lr, warmup=args.warmup, stage=(s0, s1) if args.lr_restart else None,
                   sched=args.sched, decay_frac=args.decay_frac)
        for gr in opt.param_groups:
            gr["lr"] = lr * gr.get("lr_scale", 1.0)
        t0 = time.perf_counter()
        batch = args.batch * (k if args.batch_scale_k else 1)
        lowres = k > 1 and args.coarse_mode == "lowres"
        t = sampler.next(batch)
        x, y = ds.batch(t)
        with rt.autocast():
            if lowres:
                f = fs[k]
                pred = low(F.avg_pool2d(x, f), f)
                loss = lat_mse(pred, F.avg_pool2d(y, f),
                               F.avg_pool2d(ds.latw.expand(1, 1, -1, f), f))
            else:
                loss = lat_mse(fwd(x, k), y, ds.latw)
        opt.zero_grad(set_to_none=True)
        rt.backward(loss, opt, model.parameters())
        if rt.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if rt.compile and (k, batch, lowres) not in compiled:
            compiled.add((k, batch, lowres))     # first step at a new shape includes compilation
            compile_time += dt
        else:
            train_time += dt
        limiter(dt)
        mf = model.flops_per_step(batch, k, lowres=lowres)
        model_flops += mf
        opt_total += opt_flops
        flops += mf + (opt_flops if args.flops_mode == "total" else 0.0)
        samples += batch
        step += 1
        if step % 25 == 0:
            log["train"].append({"step": step, "frac": frac, "k": k, "loss": loss.item(),
                                 "train_time": train_time})
        if args.ckpt_every and step % args.ckpt_every == 0:
            checkpoint()
        if args.stop_step and step >= args.stop_step:
            print(f"  stopped at step {step} (--stop_step)", flush=True)
            if args.ckpt_every:
                checkpoint()
            return

    clim = ds.climatology(train_end, [ds.channels.index(c) for c in REPORT])
    log["final"] = full_eval(model, ds, one_step_idx, ic_idx, rt, clim=clim)
    if val:
        log["final_val"] = full_eval(model, ds, eval_all[::3], ics(eval_all), rt, clim=clim)
    log["train_time"], log["samples"], log["steps"] = train_time, samples, step
    log["epochs"] = samples / sampler.n
    log["compile_time"] = compile_time
    log["model_flops"], log["opt_flops"] = model_flops, opt_total
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"{args.name}.json"), "w") as f:
        json.dump(log, f, indent=1)
    if args.save:
        save_model(args.save, model, ds, args)
    fin = log["final"]
    print(f"[{args.name}] val={fin['val_loss']:.4f} "
          + " ".join(f"{c}@{h}h={fin[f'rmse_{c}_{h}h']:.2f}" for h in (6, 24, 72) for c in REPORT)
          + f" acc z500@72h={fin['acc_z500_72h']:.3f} train_time={train_time:.0f}s"
          + f" opt_flops={opt_total / max(model_flops + opt_total, 1):.1%}", flush=True)


if __name__ == "__main__":
    main()
