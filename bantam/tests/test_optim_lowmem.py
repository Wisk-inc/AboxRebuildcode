"""8-bit state numerics (log map, unbiased v), chunked steps and their memory, and
counter-based stochastic rounding."""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain import lowmem  # noqa: E402
from bantamtrain.optim import AdamWLowMem, Muon  # noqa: E402


@pytest.mark.parametrize("n", [7, 32, 255, 300, 1000])
def test_log_quant_partial_blocks_and_zeros(n):
    torch.manual_seed(n)
    v = 10 ** (torch.rand(n) - 6)                       # one decade, 1e-6 .. 1e-5
    v[torch.randperm(n)[:max(1, n // 50)]] = 0.0        # exact zeros inside blocks
    q, lo, hi = lowmem.quantize_log(v, 256)
    y = lowmem.dequantize_log(q, lo, hi, v.shape)
    nz = v > 0
    assert torch.all(y[~nz] == 0)
    assert ((y[nz] - v[nz]).abs() / v[nz]).max() < 0.01  # was ~12 % with zero padding


def test_adamw8_small_param_v_tracks_fp32():
    torch.manual_seed(0)
    w8, w32 = torch.nn.Parameter(torch.zeros(32)), torch.nn.Parameter(torch.zeros(32))
    o8 = AdamWLowMem([w8], lr=0.0, state_bits=8)
    o32 = AdamWLowMem([w32], lr=0.0, state_bits=32)
    scale = 10 ** (torch.rand(32) * 0.5 - 3)
    for _ in range(50):
        g = torch.randn(32) * scale
        w8.grad, w32.grad = g, g.clone()
        o8.step()
        o32.step()
    r = o8.state[w8]["v"].get() / o32.state[w32]["v"]
    assert 0.97 < r.min() and r.max() < 1.03             # was [0.71, 1.31]


def _v_ratio(b2, v_rounding, steps=1500, n=2048):
    torch.manual_seed(0)
    scale = 10 ** (torch.rand(n) * 2 - 3)               # 2 decades of gradient scale per block
    w8, w32 = torch.nn.Parameter(torch.zeros(n)), torch.nn.Parameter(torch.zeros(n))
    o8 = AdamWLowMem([w8], lr=0.0, betas=(0.9, b2), state_bits=8, v_rounding=v_rounding)
    o32 = AdamWLowMem([w32], lr=0.0, betas=(0.9, b2), state_bits=32)
    gen = torch.Generator().manual_seed(1)
    for _ in range(steps):
        g = torch.randn(n, generator=gen) * scale
        w8.grad, w32.grad = g, g.clone()
        o8.step()
        o32.step()
    return (o8.state[w8]["v"].get() / o32.state[w32]["v"]).median().item()


def test_adamw8_v_unbiased_high_beta2():
    assert _v_ratio(0.999, "nearest") > 1.1              # the round-to-nearest ratchet
    assert 0.97 < _v_ratio(0.999, "auto") < 1.03         # stochastic codes: unbiased
    assert 0.97 < _v_ratio(0.95, "auto", steps=300) < 1.03


def test_qstate_chunked_get_set_matches_whole():
    torch.manual_seed(0)
    x = torch.randn(1000, 3) * torch.rand(1000, 3)
    for kind, data in (("signed", x), ("log", x * x)):
        whole = lowmem.QState(x, kind)
        whole.set(data)
        parts = lowmem.QState(x, kind)
        flat = data.reshape(-1)
        for s in range(0, flat.numel(), 512):
            parts.set(flat[s:s + 512], s)
        assert torch.equal(whole.q, parts.q)
        assert torch.equal(whole.get(), parts.get())
        assert torch.equal(whole.get(512, 1024), whole.get().reshape(-1)[512:1024])
    st = lowmem.QState(x, "log")
    st.set(x * x)
    back = lowmem.QState.from_tensors(x.shape, st.tensors())
    assert torch.equal(back.get(), st.get())


def test_stochastic_rounding_generator_is_reproducible():
    x = torch.full((10_000,), 1.0 + 2 ** -10)
    a = lowmem.to_bf16_stochastic(x, torch.Generator().manual_seed(5))
    torch.manual_seed(123)
    b = lowmem.to_bf16_stochastic(x, torch.Generator().manual_seed(5))
    assert torch.equal(a, b)
    assert abs(a.float().mean().item() - x[0].item()) < 2e-4


@pytest.mark.parametrize("cls", [AdamWLowMem, Muon])
@pytest.mark.parametrize("dtype,bits", [(torch.float32, 8), (torch.float32, 32),
                                        (torch.bfloat16, 8)])
def test_chunked_step_matches_single_chunk(cls, dtype, bits):
    """Chunking only changes memory, not results (bf16 noise is per chunk, so compare
    deterministic rounding there)."""
    torch.manual_seed(0)
    w0 = torch.randn(96, 40).to(dtype)
    ws = [torch.nn.Parameter(w0.clone()) for _ in range(2)]
    kw = dict(lr=1e-2, weight_decay=0.1, state_bits=bits, stochastic=dtype == torch.float32)
    opts = [cls([ws[0]], chunk=1 << 20, **kw), cls([ws[1]], chunk=256, **kw)]
    for _ in range(3):
        g = torch.randn(96, 40).to(dtype)
        for w, o in zip(ws, opts):
            w.grad = g.clone()
            o.step()
    assert torch.equal(ws[0], ws[1])


def test_non_contiguous_params():
    torch.manual_seed(0)
    base = torch.randn(40, 24)
    for cls in (AdamWLowMem, Muon):
        a = torch.nn.Parameter(base.T.contiguous())
        b = torch.nn.Parameter(base.clone().T)                 # same values, transposed storage
        assert not b.is_contiguous()
        oa, ob = cls([a], lr=1e-2, state_bits=8), cls([b], lr=1e-2, state_bits=8)
        for _ in range(2):
            g = torch.randn(24, 40)
            a.grad, b.grad = g.clone(), g.clone()
            oa.step()
            ob.step()
        assert torch.allclose(a, b, atol=1e-6)


_MEASURE = textwrap.dedent("""
    import gc, sys, torch
    sys.path.insert(0, {root!r})
    from bantamtrain.optim import AdamWLowMem
    def hwm():
        return next(int(l.split()[1]) * 1024 for l in open("/proc/self/status")
                    if l.startswith("VmHWM"))
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(2048, 2048, dtype=torch.bfloat16))
    p.grad = torch.randn(2048, 2048, dtype=torch.bfloat16) * 1e-3
    opt = AdamWLowMem([p], lr=1e-3, state_bits=8, chunk=1 << 18)
    opt.step()
    gc.collect()
    open("/proc/self/clear_refs", "w").write("5")
    base = hwm()
    opt.step()
    print((hwm() - base) / p.numel())
""")


def test_8bit_step_transient_memory_is_per_chunk():
    if not os.access("/proc/self/clear_refs", os.W_OK):
        pytest.skip("needs /proc/self/clear_refs to reset the peak RSS")
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MALLOC_MMAP_THRESHOLD_": "131072"}
    out = subprocess.run([sys.executable, "-c", _MEASURE.format(root=ROOT)], env=env,
                         capture_output=True, text=True, check=True).stdout
    assert float(out.strip()) < 6.0                    # bytes/param; 36 before chunking
