"""Optimizer state: exact resume, weights_only checkpoints, device moves, old formats,
and CombinedOptimizer as a torch Optimizer."""
import copy
import io
import os
import pickle
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain import lowmem  # noqa: E402
from bantamtrain.models import ZoomGPT  # noqa: E402
from bantamtrain.optim import AdamWLowMem, CombinedOptimizer, Muon, build_optimizer  # noqa: E402

BATCHES = [torch.randint(0, 64, (2, 18), generator=torch.Generator().manual_seed(i))
           for i in range(6)]


def _make(optim, dtype, bits):
    torch.manual_seed(0)
    m = ZoomGPT(64, 32, 1, 2, zoom_ks=[2]).to(dtype)
    return m, build_optimizer(m, optim, 1e-2, 0.1, state_bits=bits, fused=False)


def _train(m, opt, batches, start=0):
    for i, x in enumerate(batches, start):
        k = 2 if i % 2 == 0 else 1                 # merge params idle (grad None) at k=1
        loss = m(x[:, :16], x[:, k:16 + k], k=k)
        opt.zero_grad(set_to_none=True)
        loss.float().backward()
        opt.step()


def _roundtrip(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, weights_only=True)


@pytest.mark.parametrize("optim,dtype,bits", [("adamw", torch.bfloat16, 32),
                                              ("muon", torch.bfloat16, 32),
                                              ("muon", torch.bfloat16, 8),
                                              ("adamw", torch.bfloat16, 8),
                                              ("muon", torch.float32, 8)])
def test_resume_is_exact_with_weights_only_checkpoint(optim, dtype, bits):
    m1, o1 = _make(optim, dtype, bits)
    _train(m1, o1, BATCHES)                                     # uninterrupted
    m2, o2 = _make(optim, dtype, bits)
    _train(m2, o2, BATCHES[:3])
    ck = _roundtrip({"model": m2.state_dict(), "opt": o2.state_dict()})
    m3, o3 = _make(optim, dtype, bits)
    m3.load_state_dict(ck["model"])
    o3.load_state_dict(ck["opt"])
    for o in o3.optimizers:                                     # fp32 state stays fp32
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v) and v.is_floating_point():
                    assert v.dtype == torch.float32
    torch.manual_seed(1234)                                     # global RNG must not matter
    _train(m3, o3, BATCHES[3:], 3)
    for a, b in zip(m1.parameters(), m3.parameters()):
        assert torch.equal(a, b)


def test_bf16_params_fp32_state_survive_load_unchanged():
    m, opt = _make("muon", torch.bfloat16, 32)
    _train(m, opt, BATCHES[:2])
    before = {k: v.clone() for o in opt.optimizers for st in o.state.values()
              for k, v in st.items() if torch.is_tensor(v)}
    sd = _roundtrip(opt.state_dict())
    m2, opt2 = _make("muon", torch.bfloat16, 32)
    m2.load_state_dict(m.state_dict())
    opt2.load_state_dict(sd)
    after = {k: v for o in opt2.optimizers for st in o.state.values()
             for k, v in st.items() if torch.is_tensor(v)}
    assert before.keys() == after.keys()
    assert all(after[k].dtype == torch.float32 and torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("cls,key", [(AdamWLowMem, "m"), (Muon, "buf")])
@pytest.mark.parametrize("bits", [8, 32])
@pytest.mark.parametrize("device", ["meta", "cuda"])
def test_state_loads_onto_the_param_device(cls, key, bits, device):
    """A CPU checkpoint (pretrain loads with map_location='cpu') lands on the param device."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    w = torch.nn.Parameter(torch.randn(40, 24))
    opt = cls([w], lr=1e-2, state_bits=bits)
    w.grad = torch.randn_like(w)
    opt.step()
    sd = _roundtrip(opt.state_dict())                           # weights_only=True works
    assert not any(isinstance(v, lowmem.QState) for st in sd["state"].values() for v in st.values())
    wm = torch.nn.Parameter(torch.zeros(40, 24, device=device))
    om = cls([wm], lr=1e-2, state_bits=bits)
    om.load_state_dict(sd)
    st = om.state[wm][key]
    tensors = st.tensors().values() if isinstance(st, lowmem.QState) else [st]
    assert all(t.device.type == device for t in tensors)
    wm.grad = torch.ones_like(wm)
    om.step()                                                   # no device mismatch


def test_changed_state_bits_raise_clear_errors():
    m, opt = _make("muon", torch.float32, 32)
    _train(m, opt, BATCHES[:1])
    sd = opt.state_dict()
    for optim, dtype, bits in (("muon", torch.float32, 8), ("adamw", torch.bfloat16, 32)):
        _, other = _make(optim, dtype, bits)
        with pytest.raises(ValueError, match="state_bits|optimizers"):
            other.load_state_dict(sd)
    w = torch.nn.Parameter(torch.randn(8, 8))
    w.grad = torch.randn_like(w)
    m32 = Muon([w], state_bits=32)
    m32.step()
    with pytest.raises(ValueError, match="state_bits"):
        Muon([w], state_bits=8).load_state_dict(m32.state_dict())
    with pytest.raises(ValueError, match="optimizer states"):
        opt.load_state_dict([o.state_dict() for o in opt.optimizers][:1])


def _old_quantize_log(v, block=256, eps=1e-30):
    b, _ = lowmem._blocks(v.float(), block)
    lv = (b + eps).log()
    lo, hi = lv.amin(dim=1), lv.amax(dim=1)
    span = (hi - lo).clamp_min(1e-12)
    return torch.round((lv - lo[:, None]) / span[:, None] * 255).to(torch.uint8), lo, hi


def test_loads_checkpoints_of_bantamtrain_0_2():
    """Old format: a list with one state dict per optimizer, 8-bit buffers as pickled
    QState objects (old log map), fp32 moments of bf16 params."""
    m, opt = _make("adamw", torch.float32, 8)
    _train(m, opt, BATCHES[:2])
    adam = opt.optimizers[0]
    old = adam.state_dict()
    vs = {}
    for i, st in old["state"].items():
        p = [p for g in adam.param_groups for p in g["params"]][i]
        q = lowmem.QState.__new__(lowmem.QState)            # as unpickled from 0.2: no _fmt
        q.shape, q.kind, q.block, q.vmap = tuple(p.shape), "log", 256, "sqrt"
        q.q, q.lo, q.hi = _old_quantize_log(adam.state[p]["v"].get())
        vs[i] = lowmem._dequantize_log_v1(q.q, q.lo, q.hi, q.shape)   # what 0.2 decoded
        st = {k: x for k, x in st.items() if not k.startswith("v.")}
        st["v"] = q
        st["m"] = lowmem.QState.from_tensors(p.shape, {"q": st.pop("m.q"), "s": st.pop("m.s")})
        old["state"][i] = st
    for g in old["param_groups"]:
        g.pop("v_rounding"), g.pop("sr_seed"), g.pop("lr_scale")
    buf = io.BytesIO()
    torch.save([old], buf)
    buf.seek(0)
    ck = torch.load(buf, weights_only=False)
    m2, opt2 = _make("adamw", torch.float32, 8)
    opt2.load_state_dict(ck)
    a2 = opt2.optimizers[0]
    for i, p in enumerate(p for g in a2.param_groups for p in g["params"]):
        if i in vs:
            got = a2.state[p]["v"].get()
            assert torch.allclose(got, vs[i], rtol=0.03, atol=0)
    assert all(g["lr_scale"] == 1.0 and g["sr_seed"] == 0 for g in opt2.param_groups)
    _train(m2, opt2, BATCHES[2:3], 2)

    mb, ob = _make("muon", torch.bfloat16, 32)                  # old list of fp32 states
    _train(mb, ob, BATCHES[:1])
    mb2, ob2 = _make("muon", torch.bfloat16, 32)
    ob2.load_state_dict([o.state_dict() for o in ob.optimizers])
    assert all(v.dtype == torch.float32 for o in ob2.optimizers for st in o.state.values()
               for v in st.values() if torch.is_tensor(v))


def test_combined_optimizer_is_a_torch_optimizer():
    m, opt = _make("muon", torch.float32, 32)
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(opt.param_groups) == sum(len(o.param_groups) for o in opt.optimizers)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    assert all(g["lr"] == 5e-3 for g in opt.param_groups)
    hits = []
    opt.register_step_post_hook(lambda o, a, k: hits.append(1))
    _train(m, opt, BATCHES[:1])
    sched.step()
    assert hits == [1]
    opt.zero_grad()
    assert all(p.grad is None for p in m.parameters())
    # state_dict round trip: a copy continues exactly like the original
    m2, opt2 = _make("muon", torch.float32, 32)
    m2.load_state_dict(m.state_dict())
    opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
    _train(m, opt, BATCHES[1:3], 1)
    _train(m2, opt2, BATCHES[1:3], 1)
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))
    assert isinstance(pickle.loads(pickle.dumps(opt)), CombinedOptimizer)
    assert [n for n, _ in opt.summary()] == ["AdamW", "Muon"]
    extra = torch.nn.Parameter(torch.randn(4, 4))
    opt.add_param_group({"params": [extra], "use_muon": True})
    assert opt.optimizers[1].param_groups[-1]["lr_scale"] == 1.0


def test_combined_optimizer_with_grad_scaler():
    try:
        scaler = torch.amp.GradScaler("cpu", init_scale=1024.0)
    except Exception as e:                                      # pragma: no cover
        pytest.skip(f"no CPU GradScaler: {e}")
    m, opt = _make("muon", torch.float32, 8)
    x = BATCHES[0]
    scaler.scale(m(x[:, :16], x[:, 1:17], k=1)).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    before = [p.detach().clone() for p in m.parameters()]
    scaler.step(opt)
    scaler.update()
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    # an inf in any parameter's grad skips every sub-optimizer's step
    opt.zero_grad()
    scaler.scale(m(x[:, :16], x[:, 1:17], k=1)).backward()
    m.emb.weight.grad[0, 0] = float("inf")
    before = [p.detach().clone() for p in m.parameters()]
    scaler.step(opt)
    scaler.update()
    assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert scaler.get_scale() == 512.0
