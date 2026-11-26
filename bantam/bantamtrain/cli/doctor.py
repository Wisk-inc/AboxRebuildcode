"""Environment report for bug reports: `bantam doctor` (or `bantam doctor --json`).

Versions (Python, bantamtrain, torch, numpy, transformers, peft, ...), CUDA / MPS
devices with names and memory, bf16 / fp16 / TF32 support, the flush-denormal
state, thread counts, free RAM and disk, and which optional extras are importable.
Read-only: nothing is changed in the running process.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile

# pip extra -> the modules it provides (pyproject.toml [project.optional-dependencies])
EXTRAS = {
    "data": ["tokenizers", "pyarrow", "huggingface_hub"],
    "finetune": ["transformers", "safetensors", "datasets", "sentencepiece"],
    "weather": ["xarray", "zarr", "fsspec", "aiohttp"],
    "test": ["pytest", "peft", "matplotlib"],
}
PACKAGES = ["bantamtrain", "torch", "numpy", "zstandard", "transformers", "peft", "safetensors",
            "tokenizers", "datasets", "pyarrow", "huggingface_hub", "accelerate", "xarray"]


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _try(fn, default=None):
    try:
        return fn()
    except Exception:  # noqa: BLE001 - a report must never crash on an odd platform
        return default


def _free_ram():
    """(available, total) bytes."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.available, vm.total
    except ImportError:
        pass
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
        return info.get("MemAvailable"), info.get("MemTotal")
    except OSError:
        pg = os.sysconf("SC_PAGE_SIZE")
        return None, _try(lambda: os.sysconf("SC_PHYS_PAGES") * pg)


def _disk(path):
    u = _try(lambda: shutil.disk_usage(path))
    return {"path": path, "free_bytes": u.free if u else None, "total_bytes": u.total if u else None}


def collect():
    """The report as a JSON-serializable dict."""
    import torch
    from .. import __version__
    rep = {
        "bantamtrain": __version__,
        "bantamtrain_metadata_version": _version("bantamtrain"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {p: _version(p) for p in PACKAGES},
        "torch": {"version": torch.__version__, "cuda": torch.version.cuda,
                  "hip": getattr(torch.version, "hip", None),
                  "cudnn": _try(torch.backends.cudnn.version),
                  "nccl": _try(lambda: ".".join(map(str, torch.cuda.nccl.version()))),
                  "debug": torch.version.debug,
                  "cpu_capability": _try(torch.backends.cpu.get_cpu_capability)},
    }
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            free, total = _try(lambda: torch.cuda.mem_get_info(i), (None, p.total_memory))
            gpus.append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                         "total_bytes": total, "free_bytes": free,
                         "multiprocessors": p.multi_processor_count})
    mps = bool(_try(torch.backends.mps.is_available, False))
    rep["cuda"] = {"available": torch.cuda.is_available(), "devices": gpus,
                   "bf16": bool(_try(torch.cuda.is_bf16_supported, False)) if gpus else False,
                   "fp16": bool(gpus),
                   "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                   "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                   "tf32_capable": any(float(g["capability"]) >= 8.0 for g in gpus)}
    rep["mps"] = {"available": mps, "built": bool(_try(torch.backends.mps.is_built, False)),
                  "bf16": bool(_try(lambda: torch.ones(1, dtype=torch.bfloat16, device="mps")
                                    is not None, False)) if mps else False}
    rep["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    # flush-to-zero: a denormal survives x * 2 only when FTZ/DAZ is off on this thread
    tiny = torch.tensor([1e-40], dtype=torch.float32)
    flushed = float(tiny * 2) == 0.0
    rep["flush_denormal"] = {"enabled": flushed,
                             "supported": bool(torch.set_flush_denormal(flushed))}  # unchanged
    rep["threads"] = {"torch_intraop": torch.get_num_threads(),
                      "torch_interop": torch.get_num_interop_threads(),
                      "cpu_count": os.cpu_count(),
                      "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else None,
                      **{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    avail, total = _free_ram()
    rep["ram"] = {"available_bytes": avail, "total_bytes": total}
    rep["disk"] = [_disk(os.getcwd()), _disk(tempfile.gettempdir())]
    rep["extras"] = {extra: {m: importlib.util.find_spec(m) is not None for m in mods}
                     for extra, mods in EXTRAS.items()}
    return rep


def _gb(b):
    return "?" if b is None else f"{b / 2**30:.1f} GiB"


def format_report(r):
    yes = {True: "yes", False: "no"}
    pk = ", ".join(f"{k} {v}" for k, v in r["packages"].items() if v and k != "bantamtrain")
    lines = [f"bantamtrain {r['bantamtrain']}"
             + ("" if r["bantamtrain_metadata_version"] in (None, r["bantamtrain"])
                else f" (installed metadata says {r['bantamtrain_metadata_version']})"),
             f"python      {r['python']} on {r['platform']}",
             f"packages    {pk}",
             f"torch       {r['torch']['version']} (CUDA {r['torch']['cuda']}, cuDNN "
             f"{r['torch']['cudnn']}, NCCL {r['torch']['nccl']}, CPU {r['torch']['cpu_capability']})"]
    c = r["cuda"]
    if c["devices"]:
        for g in c["devices"]:
            lines.append(f"cuda:{g['index']}      {g['name']} (sm {g['capability']}), "
                         f"{_gb(g['free_bytes'])} free of {_gb(g['total_bytes'])}")
        lines.append(f"precision   bf16 {yes[c['bf16']]}, fp16 {yes[c['fp16']]}, TF32 capable "
                     f"{yes[c['tf32_capable']]} (matmul {c['tf32_matmul']}, cudnn {c['tf32_cudnn']})")
    else:
        lines.append("cuda        not available")
    lines.append(f"mps         {'available' if r['mps']['available'] else 'not available'}")
    lines.append(f"fp32 matmul precision {r['float32_matmul_precision']}; flush-denormal "
                 f"{'on' if r['flush_denormal']['enabled'] else 'off'} (supported: "
                 f"{yes[r['flush_denormal']['supported']]})")
    t = r["threads"]
    lines.append(f"threads     torch {t['torch_intraop']} intra-op / {t['torch_interop']} inter-op, "
                 f"{t['cpu_count']} CPUs ({t['affinity']} usable), OMP_NUM_THREADS="
                 f"{t['OMP_NUM_THREADS']}, MKL_NUM_THREADS={t['MKL_NUM_THREADS']}")
    lines.append(f"ram         {_gb(r['ram']['available_bytes'])} available of "
                 f"{_gb(r['ram']['total_bytes'])}")
    for d in r["disk"]:
        lines.append(f"disk        {_gb(d['free_bytes'])} free of {_gb(d['total_bytes'])} at {d['path']}")
    for extra, mods in r["extras"].items():
        missing = [m for m, ok in mods.items() if not ok]
        lines.append(f"extra {extra:8s} " + ("ok" if not missing else
                                              f"missing {', '.join(missing)}: "
                                              f"pip install 'bantamtrain[{extra}]'"))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print an environment report for bug reports.")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)
    rep = collect()
    print(json.dumps(rep, indent=1) if a.json else format_report(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
