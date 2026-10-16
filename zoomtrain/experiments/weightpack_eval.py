"""Measure checkpoint compression vs quality on a trained model.

    python experiments/weightpack_eval.py --run results/llm_lowmem/lm_muon32.json \
        --weights results/llm_lowmem/lm_muon32.pt

Reports bytes for torch.save (fp32), WeightPack fp32/bf16/int8, and the
validation loss of the model after a save -> load round trip of each.
"""
import argparse
import json
import os
import sys

import torch

torch.set_flush_denormal(True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "llm"))
from zoomtrain import weightpack  # noqa: E402
from zoomtrain.device import setup_runtime  # noqa: E402
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.zoom import ZoomSchedule  # noqa: E402
from train import evaluate, open_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--eval_seqs", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    log = json.load(open(args.run))
    a = argparse.Namespace(**log["args"])
    rt = setup_runtime("cpu", "fp32", False, args.threads, 0)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    sched = ZoomSchedule.parse(a.zoom)
    model = ZoomGPT(meta["vocab_size"], a.d, a.layers, a.heads, zoom_ks=sched.ks,
                    gated=bool(a.gated), out_mode=a.out_mode, rope_pos=a.rope_pos,
                    qk_norm=bool(a.qk_norm))
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    val = open_tokens(a.data, "val")
    ev = lambda m: evaluate(m, val, a, rt, args.eval_seqs)  # noqa: E731
    out = {"torch_save_fp32_bytes": weightpack.torch_save_bytes(model.state_dict()),
           "params": sum(p.numel() for p in model.parameters())}
    tmp = args.weights + ".zwpk"
    for bits in (32, 16, 8):
        before, after, nbytes = weightpack.roundtrip_delta(model, ev, tmp, bits)
        out[f"wp{bits}"] = {"bytes": nbytes, "val_loss": after, "delta": after - before,
                            "ratio_vs_torch_fp32": out["torch_save_fp32_bytes"] / nbytes,
                            "bytes_per_param": nbytes / out["params"]}
        out["val_loss_fp32"] = before
    os.remove(tmp)
    print(json.dumps(out, indent=1))
    with open(args.run.replace(".json", "_weightpack.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
