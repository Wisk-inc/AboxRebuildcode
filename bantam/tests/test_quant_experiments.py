"""experiments/weightpack_eval.py and weights_eval.py build the run's real architecture."""
import argparse
import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_weightpack_eval_builds_the_logged_architecture(tmp_path):
    """EXP-1: mlp / softcap / merge_mode come from the run's args (was GELU, no softcap)."""
    from bantamtrain.cli.pretrain import build_model
    a = argparse.Namespace(d=64, layers=2, heads=2, zoom="1", gated=1, out_mode="slot",
                           rope_pos="token", qk_norm=0, merge_mode="", mlp="relu2",
                           softcap=15.0, zero_init_proj=0)
    for name in ("weightpack_eval", "weights_eval"):
        mod = _script(name)
        m = mod.build(a, 300)
        ref = build_model(a, 300)
        assert m.softcap == 15.0 and m.blocks[0].mlp == "relu2"
        assert m.state_dict().keys() == ref.state_dict().keys()
        sd = {k: v.clone() for k, v in ref.state_dict().items()}
        torch.save(sd, tmp_path / "w.pt")
        from safetensors.torch import save_file
        save_file(sd, tmp_path / "w.safetensors")
        for f in ("w.pt", "w.safetensors"):
            back = mod.load_weights(str(tmp_path / f))
            assert all(torch.equal(back[k], v) for k, v in sd.items())
