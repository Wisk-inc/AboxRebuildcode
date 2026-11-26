"""prepare-text / pack-text: val leakage, literal EOS, uint32 ids, HF shard picking, stale
packs, streaming memory, reusing a Hugging Face tokenizer, document index, open_tokens."""
import contextlib
import io
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("tokenizers")
from bantamtrain.cli import pack_text, prepare_text  # noqa: E402
from bantamtrain.data import open_tokens, tokpack  # noqa: E402

WORDS = "the cat sat on a mat and then ran far away from home to see big red ball Lily Tom".split()


def _docs(n, seed=0, lo=5, hi=40):
    rng = np.random.default_rng(seed)
    return [" ".join(rng.choice(WORDS, rng.integers(lo, hi))) + "." for _ in range(n)]


def _parquet(path, texts, rows=500):
    pq.write_table(pa.table({"text": texts}), path, row_group_size=rows)
    return str(path)


def _run(mod, argv):
    with contextlib.redirect_stdout(io.StringIO()):
        mod.main(argv)


def _bin_docs(path, meta):
    a = np.fromfile(path, dtype=meta.get("dtype", "uint16"))
    ends = np.flatnonzero(a == meta["eos_id"]) + 1
    return [a[s:e].tobytes() for s, e in zip(np.r_[0, ends[:-1]], ends)]


def test_prepare_text_drops_train_docs_identical_to_val(tmp_path):
    val = _docs(40, seed=1)
    train = _docs(600) + val[:5] + ["  " + val[7] + "\n"]          # 6 leaked copies (stripped)
    out = tmp_path / "out"
    _run(prepare_text, ["--train", _parquet(tmp_path / "t.parquet", train),
                        "--val", _parquet(tmp_path / "v.parquet", val), "--out", str(out),
                        "--vocab", "300"])
    meta = json.load(open(out / "meta.json"))
    assert meta["train_docs_dropped_exact_val"] == 6 and meta["train_docs"] == 600
    vdocs = set(_bin_docs(out / "val.bin", meta))
    assert len(vdocs) == 40 and not vdocs & set(_bin_docs(out / "train.bin", meta))
    # without a validation split the held-out tail is protected the same way
    rows = _docs(990, seed=2)
    rows = rows[:400] + rows[985:] + rows[400:]          # docs of the last 1 % also early on
    _run(prepare_text, ["--train", _parquet(tmp_path / "h.parquet", rows), "--out",
                        str(tmp_path / "h"), "--vocab", "300", "--val_frac", "0.01"])
    m = json.load(open(tmp_path / "h" / "meta.json"))
    assert m["val_docs"] == 9 and m["train_docs_dropped_exact_val"] == 5
    assert not set(_bin_docs(tmp_path / "h" / "val.bin", m)) & set(_bin_docs(tmp_path / "h" / "train.bin", m))


def test_literal_eos_text_not_split(tmp_path):
    tok = prepare_text.train_tokenizer(_docs(300), 300)
    eos = tok.token_to_id(prepare_text.EOS)
    docs = ["Tom wrote <|eos|> on the board.", "x"]
    n = prepare_text.encode_to_memmap(tok, docs, tmp_path / "v.bin")
    a = np.fromfile(tmp_path / "v.bin", dtype=np.uint16)
    assert n == len(a) and int((a == eos).sum()) == 2
    assert tok.decode(a[:np.flatnonzero(a == eos)[0]].tolist()) == docs[0]
    assert tok.encode_special_tokens is False            # restored after encoding


def _wordlevel_tokenizer(path, vocab):
    from tokenizers import Tokenizer, models, pre_tokenizers
    v = {f"w{i}": i for i in range(vocab - 2)}
    v.update({"<unk>": vocab - 2, "</s>": vocab - 1})
    tok = Tokenizer(models.WordLevel(v, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["</s>"])
    tok.save(str(path))


def test_encode_large_vocab_uses_uint32(tmp_path):
    class Enc:
        ids = [70000, 5]

    class Tok:
        def token_to_id(self, s):
            return 0

        def get_vocab_size(self):
            return 100_000

        def encode_batch(self, texts, **kw):
            return [Enc() for _ in texts]

    assert prepare_text.encode_to_memmap(Tok(), ["a", "b"], tmp_path / "x.bin") == 6
    assert np.fromfile(tmp_path / "x.bin", np.uint32).tolist() == [70000, 5, 0, 70000, 5, 0]
    with pytest.raises(ValueError, match="uint16"):
        prepare_text.encode_to_memmap(Tok(), ["a"], tmp_path / "y.bin", dtype=np.uint16)
    # end to end: a 70k-word tokenizer -> uint32 .bin -> pack-text -> TokPack
    _wordlevel_tokenizer(tmp_path / "tokenizer.json", 70_000)
    rng = np.random.default_rng(0)
    texts = [" ".join(f"w{i}" for i in rng.integers(60_000, 69_998, 30)) for _ in range(300)]
    out = tmp_path / "out"
    _run(prepare_text, ["--train", _parquet(tmp_path / "t.parquet", texts), "--out", str(out),
                        "--tokenizer", str(tmp_path / "tokenizer.json"), "--eos", "</s>"])
    meta = json.load(open(out / "meta.json"))
    assert meta["dtype"] == "uint32" and meta["vocab_size"] == 70_000 and meta["eos_id"] == 69_999
    a = np.fromfile(out / "train.bin", np.uint32)
    assert a.max() == 69_999 and len(a) == meta["train_tokens"] and a[a != 69_999].min() >= 60_000
    _run(pack_text, ["--data", str(out), "--level", "1"])
    tp = tokpack.TokPack(out / "train.tokpack")
    assert tp.out_dtype == np.uint32 and np.array_equal(tp[:], a)
    toks, src = open_tokens(str(out), "val", prefer_pack=False)
    assert src == "bin" and toks.dtype == np.uint32


def test_hf_shards_selection(monkeypatch):
    listings = {
        "wikitext": ["wikitext-103-raw-v1/test-00000-of-00001.parquet",
                     "wikitext-103-raw-v1/train-00000-of-00002.parquet",
                     "wikitext-103-raw-v1/train-00001-of-00002.parquet",
                     "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
                     "wikitext-103-v1/test-00000-of-00001.parquet",
                     "wikitext-103-v1/train-00000-of-00002.parquet",
                     "wikitext-103-v1/validation-00000-of-00001.parquet"],
        "tinystories": ["data/train-00000-of-00004-2d5a.parquet", "data/train-00001-of-00004-5852.parquet",
                        "data/validation-00000-of-00001-869c.parquet", "TinyStories-train.txt"],
        "fineweb": ["data/CC-MAIN-2024-10/000_00000.parquet", "sample/10BT/000_00000.parquet",
                    "sample/10BT/001_00000.parquet"],
        "testonly": ["data/test-0.parquet", "data/train-0.parquet"],
        "jsonl": ["data/train.jsonl.zst"],
        "jsonl@refs/convert/parquet": ["default/partial-train/0000.parquet",
                                       "default/partial-train/0001.parquet",
                                       "default/validation/0000.parquet"],
        "nothing": ["README.md"],
    }
    hub = types.ModuleType("huggingface_hub")

    class HfApi:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            key = f"{repo}@{revision}" if revision else repo
            if key not in listings:
                raise FileNotFoundError(key)
            return listings[key]

    hub.HfApi = HfApi
    hub.hf_hub_download = lambda repo, f, repo_type=None, cache_dir=None, revision=None: \
        f"{revision or 'main'}:{f}"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    tr, va = prepare_text.hf_shards("wikitext", 3, None)
    assert tr == ["main:wikitext-103-raw-v1/train-00000-of-00002.parquet",
                  "main:wikitext-103-raw-v1/train-00001-of-00002.parquet"]       # one config
    assert va == ["main:wikitext-103-raw-v1/validation-00000-of-00001.parquet"]  # not test
    tr, va = prepare_text.hf_shards("wikitext", 1, None, config="wikitext-103-v1")
    assert tr == ["main:wikitext-103-v1/train-00000-of-00002.parquet"] and "v1/valid" in va[0]
    assert prepare_text.hf_shards("tinystories", 1, None) == (
        ["main:data/train-00000-of-00004-2d5a.parquet"], ["main:data/validation-00000-of-00001-869c.parquet"])
    assert prepare_text.hf_shards("fineweb", 5, None) == (["main:data/CC-MAIN-2024-10/000_00000.parquet"], [])
    assert len(prepare_text.hf_shards("fineweb", 5, None, config="10BT")[0]) == 2
    assert prepare_text.hf_shards("testonly", 1, None)[1] == ["main:data/test-0.parquet"]
    tr, va = prepare_text.hf_shards("jsonl", 1, None)
    assert tr == ["refs/convert/parquet:default/partial-train/0000.parquet"] and len(va) == 1
    with pytest.raises(SystemExit, match="no parquet"):
        prepare_text.hf_shards("nothing", 1, None)
    with pytest.raises(SystemExit, match="config"):
        prepare_text.hf_shards("wikitext", 1, None, config="nope")


def test_reprepare_invalidates_stale_tokpack(tmp_path):
    t = _parquet(tmp_path / "t.parquet", _docs(800))
    v = _parquet(tmp_path / "v.parquet", _docs(40, seed=5))
    out = tmp_path / "out"
    _run(prepare_text, ["--train", t, "--val", v, "--out", str(out), "--vocab", "300"])
    _run(pack_text, ["--data", str(out), "--level", "1"])
    assert open_tokens(str(out), "train")[1] == "tokpack"
    old_pack = (out / "train.tokpack").read_bytes()
    _run(prepare_text, ["--train", t, "--val", v, "--out", str(out), "--vocab", "260"])
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "tokenizer.json", "train.bin",
                                                     "val.bin"]
    toks, src = open_tokens(str(out), "train")
    meta = json.load(open(out / "meta.json"))
    assert src == "bin" and toks.max() < meta["vocab_size"] == 260
    (out / "train.tokpack").write_bytes(old_pack)       # e.g. copied back by hand
    with pytest.warns(UserWarning, match="prep_id"):
        toks, src = open_tokens(str(out), "train")
    assert src == "bin"
    with pytest.raises(tokpack.StalePackError):
        tokpack.TokPack(out / "train.tokpack")
    with open(out / "train.bin", "ab") as f:            # .bin no longer matches the manifest
        f.write(b"\0\0")
    with pytest.raises(SystemExit, match="different prepare-text runs"):
        _run(pack_text, ["--data", str(out), "--level", "1"])


def test_pack_text_document_index_and_val_overlap(tmp_path):
    val = _docs(30, seed=9)
    train = _docs(500) + _docs(500)[:60] + val[:4]          # 60 exact train dups, 4 leaked val docs
    out = tmp_path / "out"
    _run(prepare_text, ["--train", _parquet(tmp_path / "t.parquet", train), "--val",
                        _parquet(tmp_path / "v.parquet", val), "--out", str(out), "--vocab", "300",
                        "--drop_val_overlap", "0"])
    meta = json.load(open(out / "meta.json"))
    assert meta["train_docs_dropped_exact_val"] is None and meta["train_docs"] == 564
    _run(pack_text, ["--data", str(out), "--level", "1", "--dedup", "0"])
    st = json.load(open(out / "pack_stats.json"))
    assert st["val_overlap"]["train_docs_dropped"] == 4 and st["prep_id"] == meta["prep_id"]
    tp = tokpack.TokPack(out / "train.tokpack")
    assert tp.n_docs == st["train"]["docs"] == 560
    assert not {d.tobytes() for d in tp.docs()} & set(_bin_docs(out / "val.bin", meta))
    _run(pack_text, ["--data", str(out), "--level", "1"])
    st = json.load(open(out / "pack_stats.json"))
    assert st["dedup"]["train_docs_dropped"] == 64 and st["dedup"]["docs_per_source"] == [30, 564]
    tp = tokpack.TokPack(out / "train.tokpack")
    assert tp.n_docs == 500 and tp.h["source"]["prep_id"] == meta["prep_id"]
    vp = tokpack.TokPack(out / "val.tokpack")
    assert [d.tobytes() for d in vp.docs()] == _bin_docs(out / "val.bin", meta)


def test_prepare_text_reuses_hf_tokenizer(tmp_path):
    transformers = pytest.importorskip("transformers")
    tok = prepare_text.train_tokenizer(_docs(300), 280)
    tok.add_special_tokens(["</s>", "<s>"])
    hf = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="</s>", bos_token="<s>")
    hf.save_pretrained(tmp_path / "tk")
    texts = _docs(200, seed=4)
    out = tmp_path / "out"
    _run(prepare_text, ["--train", _parquet(tmp_path / "t.parquet", texts), "--out", str(out),
                        "--tokenizer", str(tmp_path / "tk")])
    meta = json.load(open(out / "meta.json"))
    eos = hf.convert_tokens_to_ids("</s>")
    assert meta["eos_token"] == "</s>" and meta["eos_id"] == eos and meta["tokenizer"] == str(tmp_path / "tk")
    a = np.fromfile(out / "train.bin", np.uint16)
    want = np.concatenate([hf(t, add_special_tokens=False)["input_ids"] + [eos] for t in texts[:-2]])
    assert np.array_equal(a, want)                     # no BOS, EOS after each document
    reloaded, eos_tok = prepare_text.load_tokenizer(str(out / "tokenizer.json"))
    assert eos_tok == "</s>" and reloaded.get_vocab_size() == meta["vocab_size"]


def test_prepare_text_streams_with_bounded_memory(tmp_path):
    """Peak RSS does not grow with the corpus (the old prepare held every text and array)."""
    rng = np.random.default_rng(0)
    big = [" ".join(rng.choice(WORDS, 120)) for _ in range(12_000)]      # ~7 MB of text
    _parquet(tmp_path / "small.parquet", big[:2000], rows=1000)
    _parquet(tmp_path / "big.parquet", big, rows=1000)
    code = f"""
import resource, sys
sys.path.insert(0, {ROOT!r})
import pyarrow.parquet, tokenizers, numpy
from bantamtrain.cli import prepare_text
def peak():
    try:
        return int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])
    except (OSError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
try:
    open("/proc/self/clear_refs", "w").write("5")
except OSError:
    pass
base = peak()
sys.argv = ["x", "--train", sys.argv[1], "--out", sys.argv[2], "--vocab", "300", "--tok_docs", "1000"]
prepare_text.main()
print(peak() - base)
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "RAYON_NUM_THREADS": "2"}
    grow = []
    for name in ("small", "big"):
        r = subprocess.run([sys.executable, "-c", code, str(tmp_path / f"{name}.parquet"),
                            str(tmp_path / name)], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        grow.append(int(r.stdout.strip().splitlines()[-1]))
    assert json.load(open(tmp_path / "big" / "meta.json"))["train_docs"] > 11_000
    # 10000 more documents (~6 MB of text, 2.4M tokens): the old in-RAM prepare grew by
    # ~150 MB here (texts, per-document arrays and one 10000-document Encoding batch)
    assert grow[1] - grow[0] < 16 * 1024, grow
