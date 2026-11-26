"""experiments/compare_published.py and plot_published.py: boundaries, story set, signatures."""
import ast
import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import compare_published as cp  # noqa: E402


def _uniform(V):
    return lambda x, mask: torch.zeros(*x.shape, V)


def test_prefix_is_context_and_every_story_token_scored_once():
    V, ids, prefix, end = 10, [5, 6, 7, 8, 9, 5, 6], [1, 2], [3, 4]
    for ctx in (2, 3, 4, 100):
        chunks = cp.boundary_windows(ids, prefix, ctx, end)
        scored = [t for c, skip in chunks for t in c[1 + skip:]]
        assert scored == ids + end                       # no prefix token, nothing twice
        assert all(len(c) <= ctx + 1 for c, _ in chunks)
        nll, n = cp.score(_uniform(V), chunks)
        assert n == len(ids) + len(end)
        assert nll == pytest.approx((len(ids) + len(end)) * math.log(V), rel=1e-5)
    # the old one-token API is unchanged
    assert cp.windows([5, 6, 7], 0, 2, 9) == [[0, 5, 6], [6, 7, 9]]
    assert cp.windows([5, 6, 7], 0, 2) == [[0, 5, 6], [6, 7]]


class _Tok:
    eos_token_id, bos_token_id = 50256, None

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [198] if s == "\n" else [1, 2]}


def test_boundary_format_per_family():
    assert cp.parse_hf_spec("roneneldan/TinyStories-1M") == ("roneneldan/TinyStories-1M", None, "eos_nl")
    assert cp.parse_hf_spec("org/m@org/tok") == ("org/m", "org/tok", "eos")
    assert cp.parse_hf_spec("roneneldan/TinyStories-1M#eos")[2] == "eos"
    with pytest.raises(SystemExit):
        cp.parse_hf_spec("org/m#bogus")
    tok = _Tok()
    assert cp.hf_boundary("eos_nl", tok) == ([50256, 198], [198, 50256])   # V1 .txt layout
    assert cp.hf_boundary("eos", tok) == ([50256], [50256])
    assert cp.hf_boundary("none", tok) == ([50256], [])
    assert cp.hf_boundary("eos_nl", tok, end_token=False) == ([50256, 198], [])


def _parquet(path, texts):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    pq.write_table(pa.table({"text": texts}), str(path))
    return str(path)


def _clean(path, texts, idx, with_hash=True):
    meta = {"clean_indices": idx, "rule": "test"}
    if with_hash:
        meta["source"] = {"n_texts": len(texts), "text_sha256": cp.text_sha256([t.strip() for t in texts])}
    path.write_text(json.dumps(meta))
    return str(path)


TEXTS = [f"Story {i}.\n\nThe end {i}." for i in range(8)]


def test_load_stories_checks_the_split(tmp_path):
    pq = _parquet(tmp_path / "val.parquet", TEXTS)
    good = _clean(tmp_path / "clean.json", TEXTS, [5, 1, 7])
    assert cp.load_stories(pq, 2, good) == [TEXTS[5], TEXTS[1]]
    bad = _clean(tmp_path / "bad.json", TEXTS[::-1], [5, 1, 7])
    with pytest.raises(ValueError, match="positional indices"):
        cp.load_stories(pq, 2, bad)
    legacy = _clean(tmp_path / "legacy.json", TEXTS, [5, 1], with_hash=False)
    assert cp.load_stories(pq, 5, legacy) == [TEXTS[5], TEXTS[1]]
    assert cp.load_stories(pq, 2) == TEXTS[:2]                       # --all_val
    # the committed clean set is pinned to a revision and a text hash
    src = json.load(open(cp.CLEAN))["source"]
    assert src["n_texts"] == 21990 and len(src["text_sha256"]) == 64 and src["revision"]


def test_clean_set_by_default_and_no_mixed_conventions(tmp_path, monkeypatch):
    pq = _parquet(tmp_path / "val.parquet", TEXTS)
    clean = _clean(tmp_path / "clean.json", TEXTS, [3, 4, 6])
    out = str(tmp_path / "out.json")
    calls = []

    def fake_l2c(pt, spm_path, stories, end_token=True):
        calls.append((pt, list(stories), end_token))
        return {"name": os.path.basename(pt)[:-3], "family": "published-llama2c", "params": 1,
                "params_non_embedding": 1, "vocab": 32, "context": 8, "nll_nats": 10.0 + end_token,
                "tokens": 5, "boundary_format": "bos"}

    monkeypatch.setattr(cp, "eval_llama2c", fake_l2c)
    # default --clean is the committed clean set; its hash check rejects this tiny parquet
    with pytest.raises(ValueError):
        cp.main(["--val_parquet", pq, "--stories", "2", "--out", out])
    base = ["--val_parquet", pq, "--stories", "2", "--clean", clean, "--out", out]
    assert cp.main(base + ["--l2c", "a.pt:t.model"]) == 0
    res = json.load(open(out))
    assert calls[-1][1] == [TEXTS[3], TEXTS[4]]
    assert res["_eval"]["clean_subset"] and res["_eval"]["signature"]["end_of_story_scored"]
    assert res["a"]["signature_id"] == res["_eval"]["signature_id"]
    assert cp.main(base + ["--l2c", "b.pt:t.model"]) == 0              # same convention: merge
    assert set(json.load(open(out))) == {"_eval", "a", "b"}
    with pytest.raises(SystemExit):                                   # another convention: refuse
        cp.main(base + ["--l2c", "c.pt:t.model", "--no_end"])
    assert cp.main(base + ["--l2c", "c.pt:t.model", "--no_end", "--overwrite"]) == 0
    res = json.load(open(out))
    assert set(res) == {"_eval", "c"} and not res["_eval"]["end_of_story_scored"]
    # a legacy file (no signature) is never silently extended
    legacy = {"_eval": {"stories": 2, "bytes": 10, "end_of_story_scored": True}, "old": {"nll_nats": 1}}
    with open(out, "w") as f:
        json.dump(legacy, f)
    with pytest.raises(SystemExit):
        cp.main(base + ["--l2c", "d.pt:t.model"])
    assert cp.main(["--val_parquet", pq, "--stories", "2", "--all_val", "--overwrite", "--out", out,
                    "--l2c", "e.pt:t.model"]) == 0
    assert not json.load(open(out))["_eval"]["clean_subset"] and calls[-1][1] == TEXTS[:2]


def test_checkpoint_compute_matched_from_any_cwd(tmp_path, monkeypatch):
    run_dir = tmp_path / "results" / "published"
    run_dir.mkdir(parents=True)
    ck = run_dir / "run_f0.10.pt"
    ck.write_bytes(b"x")
    final = run_dir / "run.pt"
    final.write_bytes(b"y")
    log = {"args": {"budget": 1e9}, "tokens": None, "in_progress": True,
           "evals": [{"checkpoint": "results/published/run_f0.10.pt", "tokens": 111, "flops": 1e8,
                      "train_time": 5.0}]}
    run_json = str(run_dir / "run.json")
    got = []
    for cwd in (str(tmp_path), "/", ROOT):
        monkeypatch.chdir(cwd)
        got.append(cp.train_cost(log, str(ck), run_json))
    assert got == [(111, 1e8, 5.0)] * 3
    with pytest.raises(ValueError, match="in-progress"):
        cp.train_cost(log, str(final), run_json)
    done = {**log, "in_progress": False, "tokens": 999, "train_time": 9.0}
    assert cp.train_cost(done, str(final), run_json) == (999, 1e9, 9.0)


def test_torch_load_is_weights_only():
    src = open(cp.__file__).read()
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "load" and getattr(n.func.value, "id", None) == "torch"]
    assert calls
    for c in calls:
        kw = {k.arg: k.value for k in c.keywords}
        assert isinstance(kw.get("weights_only"), ast.Constant) and kw["weights_only"].value is True


def test_l2c_rewrite_merges_newlines_counted():
    assert cp.l2c_text("a\nb") == cp.l2c_text("a\n\nb")
    assert cp.l2c_lossy(["a\n\nb", "a\nb", "a\n\n\nb", "ab"]) == 2


def test_first_reach_and_frontier():
    import plot_published as pp
    assert pp.first_reach([(1e13, 0.9), (2e13, 1.1), (3e13, 0.95)], 1.0) == 1e13
    c, kind = pp.first_reach([(1e13, 2.0), (1e15, 1.0)], 1.5, with_kind=True)
    assert kind == "crossing" and c == pytest.approx(1e14)
    assert pp.first_reach([(1e13, 0.9)], 1.0, with_kind=True) == (1e13, "upper bound")
    assert pp.first_reach([(1e13, 2.0)], 1.0) is None
    assert pp.frontier([(3e13, 0.95), (1e13, 1.2), (2e13, 1.3), (4e13, 0.9)]) == \
        [(1e13, 1.2), (3e13, 0.95), (4e13, 0.9)]


def _row(name, family, bpb, **kw):
    return {"name": name, "family": family, "params": 1e6, "bpb": bpb, "nll_nats": 1.0, "tokens": 1, **kw}


def test_plot_published_unknown_model_and_signature(tmp_path):
    pytest.importorskip("matplotlib")
    import plot_published as pp
    res = {"_eval": {"stories": 10, "bytes": 100, "end_of_story_scored": True},
           "ours_run": _row("ours_run", "ours", 1.0, train_tokens=1e6),
           "stories260K": _row("stories260K", "published-llama2c", 1.1),
           "EleutherAI/gpt-neo-125M": _row("EleutherAI/gpt-neo-125M", "published", 0.9)}
    path = tmp_path / "cmp.json"
    path.write_text(json.dumps(res))
    with pytest.warns(UserWarning, match="compute unknown"):
        assert pp.main([str(path), "--png", str(tmp_path / "p.png")]) == 0
    table = json.loads((tmp_path / "cmp_table.json").read_text())
    rows = {r["name"]: r for r in table["rows"]}
    assert rows["EleutherAI/gpt-neo-125M"]["compute"] is None
    assert rows["stories260K"]["match_kind"] == "upper bound"
    assert (tmp_path / "p.png").exists()
    res["_eval"]["signature_id"] = "abc"
    for k in ("ours_run", "stories260K", "EleutherAI/gpt-neo-125M"):
        res[k]["signature_id"] = "abc"
    res["stories260K"]["signature_id"] = "other"
    path.write_text(json.dumps(res))
    with pytest.raises(SystemExit, match="another convention"):
        pp.main([str(path), "--png", str(tmp_path / "p.png")])
