"""2-process gloo DDP: bf16 stochastic rounding must keep the replicas identical even
when every rank seeds the global RNG differently (finetune seeds with seed + rank)."""
import os
import sys

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _worker(rank, world, init_file, out_dir):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from bantamtrain.models import ZoomGPT
    from bantamtrain.optim import build_optimizer
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank,
                            world_size=world)
    result = {}
    for optim in ("adamw", "muon"):
        torch.manual_seed(0)                       # same init on every rank
        m = ZoomGPT(64, 32, 1, 2).to(torch.bfloat16)
        ddp = DDP(m)
        opt = build_optimizer(m, optim, 1e-2, 0.1, state_bits=8)
        torch.manual_seed(1000 + rank)             # rank-dependent global RNG
        gen = torch.Generator().manual_seed(rank)  # each rank its own data
        for _ in range(5):
            x = torch.randint(0, 64, (2, 17), generator=gen)
            loss = ddp(x[:, :16], x[:, 1:], k=1)
            opt.zero_grad(set_to_none=True)
            loss.float().backward()
            opt.step()
        flat = torch.cat([p.detach().float().reshape(-1) for p in m.parameters()])
        got = [torch.empty_like(flat) for _ in range(world)]
        dist.all_gather(got, flat)
        result[optim] = int((got[0] != got[1]).sum())
    if rank == 0:
        torch.save(result, os.path.join(out_dir, "diff.pt"))
    dist.destroy_process_group()


def test_bf16_stochastic_rounding_identical_across_ddp_ranks(tmp_path):
    mp.spawn(_worker, args=(2, str(tmp_path / "init"), str(tmp_path)), nprocs=2, join=True)
    diff = torch.load(tmp_path / "diff.pt")
    assert diff == {"adamw": 0, "muon": 0}
