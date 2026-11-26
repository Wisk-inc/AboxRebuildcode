"""Self-describing ZoomGPT checkpoints: config / save_pretrained / from_pretrained."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomGPT  # noqa: E402

KW = dict(zoom_ks=[2, 4], merge_mode="linear", qk_norm=True, mlp="relu2", softcap=15.0,
          rope_pos="patch", block_size=64)


def test_config_rebuilds_the_architecture():
    m = ZoomGPT(300, 32, 2, 2, **KW)
    cfg = m.config()
    assert cfg["block_size"] == 64 and cfg["zoom_ks"] == [2, 4] and cfg["vocab"] == 300
    json.dumps(cfg)                                          # plain JSON types only
    m2 = ZoomGPT(**cfg)
    assert {n: t.shape for n, t in m.state_dict().items()} == \
        {n: t.shape for n, t in m2.state_dict().items()}
    cfg["zoom_ks"].append(8)                                 # a copy, not the model's own
    assert m.config()["zoom_ks"] == [2, 4]


def test_save_pretrained_round_trip(tmp_path):
    torch.manual_seed(0)
    m = ZoomGPT(300, 32, 2, 2, **KW).eval()
    x = torch.randint(0, 300, (2, 20))
    d = m.save_pretrained(str(tmp_path / "plain"))
    assert sorted(os.listdir(d)) == ["config.json", "model.safetensors"]
    r = ZoomGPT.from_pretrained(d)
    assert not r.training and r.config()["zoom_ks"] == [] and r.block_size == 64
    assert not any(n.startswith(("merge.", "unmerge.")) for n in r.state_dict())
    assert torch.equal(r(x[:, :16]), m(x[:, :16]))
    assert torch.equal(r.generate(x[:, :5], 6, temperature=0),
                       m.generate(x[:, :5], 6, temperature=0))
    assert r.muon_params()[0].muon_split == 3                # re-tagged after assign-loading
    z = ZoomGPT.from_pretrained(m.save_pretrained(str(tmp_path / "zoom"), strip_zoom=False))
    assert z.config() == m.config()
    for k in (2, 4):
        assert torch.equal(z(x[:, :16], x[:, k:16 + k], k=k), m(x[:, :16], x[:, k:16 + k], k=k))


def test_save_pretrained_bf16_tokenizer_and_eos(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "tokenizer.json").write_text("{}")
    (data / "meta.json").write_text(json.dumps({"vocab_size": 64, "eos_id": 7}))
    m = ZoomGPT(64, 16, 1, 2).to(torch.bfloat16)
    out = m.save_pretrained(str(tmp_path / "m"), tokenizer_path=str(data))
    cfg = json.loads((tmp_path / "m" / "config.json").read_text())
    assert cfg["eos_id"] == 7 and cfg["torch_dtype"] == "bfloat16"
    assert cfg["model_type"] == "zoomgpt"
    assert (tmp_path / "m" / "tokenizer.json").read_text() == "{}"
    r = ZoomGPT.from_pretrained(out)
    assert r.emb.weight.dtype == torch.bfloat16
    assert all(torch.equal(a, b) for a, b in zip(r.state_dict().values(), m.state_dict().values()))


def test_save_pretrained_without_safetensors(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)     # import -> ImportError
    m = ZoomGPT(64, 16, 1, 2, zoom_ks=[2])
    d = m.save_pretrained(str(tmp_path / "pt"))
    assert sorted(os.listdir(d)) == ["config.json", "model.pt"]
    torch.load(os.path.join(d, "model.pt"), weights_only=True)
    r = ZoomGPT.from_pretrained(d)
    x = torch.randint(0, 64, (1, 8))
    assert torch.equal(r(x), m(x))

