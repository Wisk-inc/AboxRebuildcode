"""Runtime setup: one code path for CPU and GPU.

CPU
  * flush-to-zero / denormals-are-zero. Must be enabled before the first
    parallel op so every OpenMP worker inherits it; call `enable_ftz()` right
    after `import torch`. Measured here: denormal softmax gradients made a
    (8192x4096)@(4096x256) matmul 15x slower (897 ms -> 57 ms with FTZ).
GPU
  * bf16 autocast on compute capability >= 8.0, fp16 + GradScaler before
    (precision="auto"), TF32 matmuls, fused AdamW, torch.compile,
    SDPA/FlashAttention (used by the models).
  * "Maximize compute without stressing": a memory-headroom batch finder so
    runs never sit at the OOM edge (`bantam pretrain --auto_batch`), and a
    duty-cycle limiter that caps the average load when a hard power cap
    (nvidia-smi -pl) is not available.
"""
import contextlib
import os
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue

import torch


def enable_ftz():
    torch.set_flush_denormal(True)


@dataclass
class Runtime:
    device: torch.device
    amp_dtype: torch.dtype | None
    scaler: "torch.amp.GradScaler | None"
    compile: bool

    def autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def backward(self, loss, optimizer, params, clip=1.0, step=True):
        """loss.backward() (+ unscale + clip + step unless step=False), AMP-aware."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if step:
            self.step(optimizer, params, clip)

    def step(self, optimizer, params, clip=1.0):
        """Clip the global grad norm and step; returns that norm.

        With a GradScaler (fp16) every sub-optimizer of a CombinedOptimizer is
        unscaled first and one finite check covers all of them: an overflow
        anywhere skips the whole step (no sub-optimizer steps) and lowers the
        scale. Sub-optimizers without grads are left out (GradScaler asserts on
        an optimizer that recorded no inf check).
        """
        params = list(params)
        if self.scaler is None:
            norm = torch.nn.utils.clip_grad_norm_(params, clip)
            optimizer.step()
            return norm
        opts = [o for o in getattr(optimizer, "optimizers", [optimizer])
                if any(p.grad is not None for g in o.param_groups for p in g["params"])]
        for o in opts:
            self.scaler.unscale_(o)
        norm = torch.nn.utils.clip_grad_norm_(params, clip)   # inf/nan iff any grad overflowed
        if opts:
            if torch.isfinite(norm):
                for o in opts:
                    self.scaler.step(o)
            self.scaler.update()
        return norm

    def to(self, x):
        if self.device.type == "cuda":
            return x.pin_memory().to(self.device, non_blocking=True)
        return x


def auto_precision(device):
    """precision="auto": bf16 on CUDA compute capability >= 8.0 (Ampere+), fp16 (+ GradScaler)
    on older GPUs, fp32 elsewhere. torch.cuda.is_bf16_supported() also accepts sm70/sm75,
    where bf16 is only emulated (no bf16 tensor cores, no flash attention)."""
    if device.type != "cuda":
        return "fp32"
    return "bf16" if tuple(torch.cuda.get_device_capability(device)) >= (8, 0) else "fp16"


def setup_runtime(device="auto", precision="auto", compile=False, threads=None, seed=0):
    """precision: 'auto' | 'fp32' | 'bf16' | 'fp16'."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    torch.manual_seed(seed)
    if dev.type == "cpu":
        enable_ftz()
        if threads:
            torch.set_num_threads(threads)
    amp_dtype, scaler = None, None
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    if precision == "auto":
        precision = auto_precision(dev)
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda")
    return Runtime(dev, amp_dtype, scaler, compile)


def maybe_compile(model, rt):
    if rt.compile and hasattr(torch, "compile"):
        return torch.compile(model, dynamic=False)
    return model


def find_batch_size(try_step, start=8, limit=4096, headroom=0.85, device=None):
    """Largest power-of-two batch in [start, limit] whose peak memory stays under
    `headroom` x total, or None when even `start` does not fit.

    try_step(batch) must run one full fwd/bwd/step on a throwaway model and
    optimizer (it is called several times; probing the real ones would train
    them on FLOPs nobody counts). CPU: returns `start` without probing.
    Staying below the OOM edge avoids allocator thrash and fragmentation
    retries, which cost more throughput than the last few % of batch size.
    """
    if device is None or device.type != "cuda":
        return start
    total = torch.cuda.get_device_properties(device).total_memory
    best, b = None, start
    while b <= limit:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            try_step(b)
            torch.cuda.synchronize(device)
            if torch.cuda.max_memory_allocated(device) > headroom * total:
                break
            best, b = b, b * 2
        except torch.OutOfMemoryError:
            break
    torch.cuda.empty_cache()
    return best


class DutyCycleLimiter:
    """Cap average accelerator load at `target` (0 < target <= 1).

    After a step that took t seconds, idles t * (1/target - 1). Throughput
    scales ~linearly with target. A hardware power cap is more efficient
    (GPUs typically keep most of their throughput at ~70% power because
    clocks/voltage drop super-linearly), so prefer `nvidia-smi -pl <watts>`
    when you have permission; this is the portable fallback.
    """

    def __init__(self, target=1.0, device=None):
        if not target > 0:                       # also rejects NaN
            raise ValueError(f"duty cycle target must be in (0, 1], got {target}")
        self.target = target
        self.device = device

    def __call__(self, step_seconds):
        if self.target >= 1.0:
            return
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        time.sleep(step_seconds * (1.0 / self.target - 1.0))

    @staticmethod
    def power_cap_hint(fraction=0.7):
        return (f"sudo nvidia-smi -pl <{int(fraction * 100)}% of default TDP>  "
                "# e.g. -pl 250 on a 350 W card")


class Prefetcher:
    """Background thread that builds the next batches while the model trains.

    An exception in make_batch is re-raised by next() (after the batches made
    before it); close() stops the worker even while it waits on a full queue.
    """

    _END = object()

    def __init__(self, make_batch, depth=2):
        self.q = Queue(maxsize=depth)
        self.make_batch = make_batch
        self._stop = threading.Event()
        self._err = None
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.05)
                return True
            except Full:
                pass
        return False

    def _run(self):
        try:
            while not self._stop.is_set():
                if not self._put(self.make_batch()):
                    return
        except BaseException as e:               # handed to the consumer by next()
            self._err = e
            self._put(self._END)

    def next(self):
        while True:
            if (self._err is not None or self._stop.is_set()) and self.q.empty():
                break
            try:
                item = self.q.get(timeout=0.05)
            except Empty:
                continue
            if item is not self._END:
                return item
            break
        if self._err is not None:
            raise self._err
        raise RuntimeError("Prefetcher is closed")

    def _drain(self):
        try:
            while True:
                self.q.get_nowait()
        except Empty:
            pass

    def close(self):
        self._stop.set()
        self._drain()                  # frees a worker blocked on the full queue
        self.t.join(timeout=1.0)
        self._drain()


def describe(rt):
    s = f"device={rt.device} amp={rt.amp_dtype} compile={rt.compile}"
    if rt.device.type == "cpu":
        s += f" threads={torch.get_num_threads()}"
    elif rt.device.type == "cuda":
        s += f" gpu={torch.cuda.get_device_name(rt.device)}"
    return s + f" pid={os.getpid()}"
