"""compare_published: end-of-story NLL reported apart, local copies of hub models, format
overrides as part of the scoring signature."""
import json
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import compare_published as cp  # noqa: E402

V = 11


def _context_free(x, mask):
    """Logits that depend only on the current input token: how a story is chunked or what
    follows it cannot change its NLL."""
    return 3.0 * F.one_hot(x % V, V).float() + 0.1 * torch.arange(V).float()


def test_end_tokens_scored_apart_even_when_split_across_chunks():
    ids, prefix, end = [5, 6, 7, 8, 9, 5, 6], [1, 2], [3, 4]
    for ctx in (2, 3, 4, 8, 100):     # ctx=3: the 2-token end straddles the last two chunks
        trip = cp.story_windows(ids, prefix, ctx, end)
        assert [(c, s) for c, s, _ in trip] == cp.boundary_windows(ids, prefix, ctx, end)
        assert sum(ne for _, _, ne in trip) == len(end)
        nll, n, nll_end, n_end = cp.score_split(_context_free, trip)
        body, n_body = cp.score(_context_free, cp.boundary_windows(ids, prefix, ctx, []))
        assert (n, n_end, n_body) == (len(ids) + len(end), len(end), len(ids))
        assert nll_end == pytest.approx(nll - body, rel=1e-6)
        assert cp.score(_context_free, trip) == pytest.approx((nll, n))      # old API unchanged
    split = cp.story_windows(ids, prefix, 3, end)
    assert [ne for _, _, ne in split][-2:] == [1, 1]
    assert cp.story_windows(ids, prefix, 3)[-1][2] == 0                  # no end: nothing marked


def test_hf_local_dir_spec_keeps_the_repo_name():
    assert cp.split_local("roneneldan/TinyStories-1M#eos=/m/ts1m") == ("roneneldan/TinyStories-1M#eos", "/m/ts1m")
    assert cp.split_local("org/m") == ("org/m", None)
    assert cp.parse_hf_spec("roneneldan/TinyStories-1M=/m/ts1m") == ("roneneldan/TinyStories-1M", None, "eos_nl")
    assert cp.parse_hf_spec("org/m@org/tok#none=/m/x") == ("org/m", "org/tok", "none")


def _parquet(path, texts):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"text": texts}), str(path))
    return str(path)


TEXTS = [f"Story {i}.\n\nThe end {i}." for i in range(6)]


def test_format_overrides_are_a_different_convention(tmp_path, monkeypatch):
    pq = _parquet(tmp_path / "val.parquet", TEXTS)
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps({"clean_indices": [1, 2, 4], "source": {
        "n_texts": len(TEXTS), "text_sha256": cp.text_sha256([t.strip() for t in TEXTS])}}))
    out = str(tmp_path / "out.json")
    calls = []

    def fake_hf(repo, stories, ctx_cap, tok_repo=None, end_token=True, fmt=None, local_dir=None):
        calls.append((repo, fmt, local_dir))
        return {"name": repo, "family": "published", "params": 1, "params_non_embedding": 1,
                "vocab": 9, "context": 8, "nll_nats": 20.0, "tokens": 10, "nll_end_nats": 4.0,
                "end_tokens": 2, "boundary_format": fmt}

    monkeypatch.setattr(cp, "eval_hf", fake_hf)
    base = ["--val_parquet", pq, "--stories", "2", "--clean", str(clean), "--out", out]
    assert cp.main(base + ["--hf", "roneneldan/A=" + str(tmp_path)]) == 0
    res = json.load(open(out))
    assert calls[-1] == ("roneneldan/A", "eos_nl", str(tmp_path))
    assert res["_eval"]["signature"]["boundary_overrides"] == []
    n_bytes = res["_eval"]["bytes"]
    assert res["roneneldan/A"]["bpb_no_end"] == pytest.approx(16.0 / math.log(2) / n_bytes)
    # rows in another boundary format never join a native file silently
    with pytest.raises(SystemExit):
        cp.main(base + ["--hf", "roneneldan/B#eos"])
    assert set(json.load(open(out))) == {"_eval", "roneneldan/A"}
    assert cp.main(base + ["--hf", "roneneldan/B#eos", "org/C#eos", "--overwrite"]) == 0
    res = json.load(open(out))
    assert res["_eval"]["signature"]["boundary_overrides"] == ["roneneldan/B#eos"]   # org/C is native
    assert res["roneneldan/B"]["boundary_format"] == "eos"


def test_eval_hf_local_copy_end_to_end(tmp_path):
    """A tiny random GPT-2 saved locally, scored as a roneneldan model: the V1 boundary format,
    the weight hashes, and an end-of-story NLL equal to (with end) - (without end)."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except Exception:  # noqa: BLE001 - no cached gpt2 tokenizer offline
        pytest.skip("gpt2 tokenizer not cached")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=16, n_layer=1, n_head=2)
    d = tmp_path / "tiny"
    transformers.GPT2LMHeadModel(cfg).save_pretrained(str(d))
    tok.save_pretrained(str(d))
    stories = ["Once upon a time there was a cat.\n\nThe cat sat.", "Tom ran home. He was happy."]
    with_end = cp.eval_hf("roneneldan/tiny", stories, 6, local_dir=str(d))
    no_end = cp.eval_hf("roneneldan/tiny", stories, 6, end_token=False, local_dir=str(d))
    assert with_end["name"] == "roneneldan/tiny" and with_end["boundary_format"] == "eos_nl"
    assert with_end["end_tokens"] == 2 * len(stories) and no_end["end_tokens"] == 0
    assert with_end["tokens"] - no_end["tokens"] == 2 * len(stories)
    assert with_end["nll_end_nats"] == pytest.approx(with_end["nll_nats"] - no_end["nll_nats"], rel=1e-4)
    assert set(with_end["weights_sha256"]) == {"model.safetensors"}
    assert len(with_end["weights_sha256"]["model.safetensors"]) == 64
