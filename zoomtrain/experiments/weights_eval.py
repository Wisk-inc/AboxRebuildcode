"""Weight storage vs quality on a trained LM: RAM formats and checkpoint files.

    python experiments/weights_eval.py --run results/llm_lowmem/save_muon_base.json \
        --weights results/llm_lowmem/save_muon_base.pt

For each format reports bytes and the validation loss (same 2048 held-out
sequences as the main runs):
  RAM:  fp32, bf16, int8 weight-only (block 128), int4 weight-only (block 64)
  disk: torch.save fp32, WeightPack 32 / 16 / 8 / 4 bits (save -> load -> eval)
Writes <run>_weights.json next to the run log.
"""
import argparse
import copy
import json
import os
import sys

import torch

torch.set_flush_denormal(True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "llm"))
from zoomtrain import qweights, weightpack  # noqa: E402
from zoomtrain.device import setup_runtime  # noqa: E402
from zoomtrain.models import ZoomGPT  # noqa: E402
from zoomtrain.zoom import ZoomSchedule  # noqa: E402
from train import evaluate, open_tokens  # noqa: E402


def build(a, vocab):
    sched = ZoomSchedule.parse(a.zoom)
    return ZoomGPT(vocab, a.d, a.layers, a.heads, zoom_ks=sched.ks, gated=bool(a.gated),
                   out_mode=a.out_mode, rope_pos=a.rope_pos, qk_norm=bool(a.qk_norm))


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
    vocab = json.load(open(os.path.join(a.data, "meta.json")))["vocab_size"]
    base = build(a, vocab)
    base.load_state_dict(torch.load(args.weights, map_location="cpu"))
    base.eval()
    val = open_tokens(a.data, "val")

    def ev(m):
        return evaluate(m, val, a, rt, args.eval_seqs)

    out = {"params": sum(p.numel() for p in base.parameters()), "ram": {}, "disk": {}}
    ref = ev(base)
    out["ram"]["fp32"] = {"bytes": qweights.model_bytes(base), "val_loss": ref, "delta": 0.0}

    m = copy.deepcopy(base).to(torch.bfloat16)
    bf16_bytes = qweights.model_bytes(m)                     # measure before .float() (in place)
    lb = evaluate(m.float(), val, a, rt, args.eval_seqs)    # bf16-rounded weights, fp32 compute
    out["ram"]["bf16"] = {"bytes": bf16_bytes, "val_loss": lb, "delta": lb - ref}

    for bits in (8, 4):
        m = copy.deepcopy(base)
        rep = qweights.quantize_model(m, bits)
        lq = ev(m)
        out["ram"][f"int{bits}"] = {"bytes": rep["bytes_after"], "val_loss": lq,
                                    "delta": lq - ref, "ratio_vs_fp32": rep["ratio"]}

    out["disk"]["torch_save_fp32"] = {"bytes": weightpack.torch_save_bytes(base.state_dict())}
    tmp = args.weights + ".zwpk"
    for bits in (32, 16, 8, 4):
        before, after, nbytes = weightpack.roundtrip_delta(base, ev, tmp, bits)
        out["disk"][f"weightpack{bits}"] = {"bytes": nbytes, "val_loss": after,
                                            "delta": after - before}
    os.remove(tmp)
    for k, v in list(out["ram"].items()) + list(out["disk"].items()):
        v["bytes_per_param"] = v["bytes"] / out["params"]
    print(json.dumps(out, indent=1))
    with open(args.run.replace(".json", "_weights.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
