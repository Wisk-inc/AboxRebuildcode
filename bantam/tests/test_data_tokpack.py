"""TokPack: thread safety, pickling, read-only views, slicing, stream coverage, streaming
writer, document index and stale-pack detection."""
import json
import os
import pickle
import subprocess
import sys
import threading

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain.data import tokpack  # noqa: E402


def _zipf(n, vocab, seed=0, eos=None, doc=200):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1)
    t = rng.choice(vocab, size=n, p=p / p.sum()).astype(np.uint16)
    if eos is not None:
        t[t == eos] = eos + 1
        t[doc - 1::doc] = eos
    return t


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    toks = _zipf(50_000, 1000)
    path = tmp_path_factory.mktemp("tp") / "x.tokpack"
    tokpack.write(path, toks, 1000, chunk=4096, level=3)
    return path, toks


def test_tokpack_threadsafe_and_picklable(packed):
    path, toks = packed
    tp = tokpack.TokPack(path, cache_chunks=2)
    bad, errors = [0], []

    def job(seed):
        r = np.random.default_rng(seed)
        for _ in range(200):
            s = int(r.integers(0, len(toks) - 5000))
            try:
                bad[0] += not np.array_equal(tp[s:s + 5000], toks[s:s + 5000])
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

    ts = [threading.Thread(target=job, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert bad[0] == 0 and not errors
    tp2 = pickle.loads(pickle.dumps(tp))
    assert np.array_equal(tp2[1000:9000], toks[1000:9000]) and tp2.cache_chunks == 2


def test_tokpack_returns_readonly_views(packed):
    path, toks = packed
    tp = tokpack.TokPack(path)
    with pytest.raises(ValueError):
        tp[100:110][0] = 1
    with pytest.raises(ValueError):
        tp[4090:4100][0] = 1                              # spans two chunks
    with pytest.raises(ValueError):
        next(tp.stream(256))[0] = 1
    assert np.array_equal(tp[100:110], toks[100:110])


@pytest.mark.parametrize("seq", [1000, 4096, 5000, 20000])
def test_tokpack_stream_covers_every_full_window(packed, seq):
    path, toks = packed
    tp = tokpack.TokPack(path, cache_chunks=4)
    wins = list(tp.stream(seq, seed=3))
    assert len(wins) == len(toks) // seq and all(len(w) == seq for w in wins)
    grid = [toks[g * seq:(g + 1) * seq].tobytes() for g in range(len(toks) // seq)]
    assert sorted(w.tobytes() for w in wins) == sorted(grid)      # every grid window exactly once
    with pytest.raises(ValueError):
        next(tp.stream(0))


def test_tokpack_slicing_matches_numpy(tmp_path, packed):
    path, toks = packed
    tp = tokpack.TokPack(path)
    rng = np.random.default_rng(0)
    n = len(toks)
    for _ in range(1500):
        a, b = (int(v) for v in rng.integers(-n - 10, n + 10, 2))
        step = int(rng.choice([1, 1, 1, 2, 3, -1, -2]))
        assert np.array_equal(tp[a:b:step], toks[a:b:step]), (a, b, step)
    for i in (0, 5, n - 1, -1, -n):
        assert tp[i] == toks[i]
    for i in (n, -n - 1):
        with pytest.raises(IndexError):
            tp[i]
    with pytest.raises(TypeError):
        tp[[1, 2]]
    assert len(tp.read(10, 0)) == 0 and len(tp.read(10, -3)) == 0
    with pytest.raises(IndexError):
        tp.read(n - 2, 5)
    tokpack.write(tmp_path / "e.tokpack", [], 100)       # empty pack from a python list
    e = tokpack.TokPack(tmp_path / "e.tokpack")
    assert len(e) == 0 and len(e[:]) == 0 and list(e.stream(4)) == []


def test_tokpack_bad_input_raises_valueerror(tmp_path):
    with pytest.raises(ValueError, match=r"\[0, 10\)"):
        tokpack.write(tmp_path / "a.tokpack", np.array([1, 2, 10]), 10)
    with pytest.raises(ValueError):
        tokpack.write(tmp_path / "b.tokpack", np.array([1, -1, 2]), 10)
    with pytest.raises(TypeError):
        tokpack.write(tmp_path / "c.tokpack", np.array([1.0, 2.0]), 10)
    with pytest.raises(ValueError):
        tokpack.leb128_encode([1 << 21])
    with pytest.raises(ValueError):
        tokpack.leb128_decode(tokpack.leb128_encode([1, 2, 3]).tobytes(), 4)
    (tmp_path / "junk").write_bytes(b"NOTAPACK" * 4)
    with pytest.raises(ValueError, match="not a TokPack"):
        tokpack.TokPack(tmp_path / "junk")
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())   # side files cleaned up


def test_tokpack_asserts_not_relied_on_under_python_O(tmp_path):
    code = (f"import numpy as np, sys; sys.path.insert(0, {ROOT!r})\n"
            "from bantamtrain.data import tokpack\n"
            f"p = {str(tmp_path / 'o.tokpack')!r}\n"
            "tokpack.write(p, np.arange(100) % 7, 10)\n"
            "for f in (lambda: tokpack.TokPack(p).read(98, 5),\n"
            "          lambda: tokpack.leb128_decode(tokpack.leb128_encode([1, 2, 3]).tobytes(), 4),\n"
            "          lambda: tokpack.write(p, np.array([1, 99]), 10)):\n"
            "    try:\n        f(); print('silent')\n"
            "    except (ValueError, IndexError):\n        print('raised')\n")
    out = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True)
    assert out.stdout.split() == ["raised"] * 3, out.stdout + out.stderr


def test_tokpack_write_stream_matches_write_and_memmap(tmp_path):
    toks = _zipf(400_000, 4096, eos=0)             # > dict sample: dictionary path, held ranks
    a, b = tmp_path / "a.tokpack", tmp_path / "b.tokpack"
    kw = dict(chunk=4096, level=1, eos_id=0, dict_sample_tokens=100_000)
    na = tokpack.write(a, toks, 4096, **kw)
    toks.tofile(tmp_path / "t.bin")
    for src in (np.memmap(tmp_path / "t.bin", dtype=np.uint16, mode="r"),
                tokpack.BinFile(tmp_path / "t.bin", np.uint16)):
        assert tokpack.write(b, src, 4096, **kw) == na
        assert a.read_bytes() == b.read_bytes()
    counts = tokpack.count_tokens(toks, 4096)
    with tokpack.TokPackWriter(b, 4096, counts, **kw) as w:
        for s in range(0, len(toks), 77_777):                # odd block sizes
            w.append(toks[s:s + 77_777])
    assert a.read_bytes() == b.read_bytes()
    tp = tokpack.TokPack(b)
    assert tp.h["dict_bytes"] > 0 and np.array_equal(tp[:], toks)


def test_tokpack_write_memory_is_bounded(tmp_path):
    """Packing a token file holds O(block + dictionary sample), not O(tokens)."""
    rng = np.random.default_rng(0)
    with open(tmp_path / "t.bin", "wb") as f:
        for _ in range(5):   # skewed ids, like text
            (rng.integers(0, 4096, 1 << 20) >> rng.integers(0, 12, 1 << 20)).astype(np.uint16).tofile(f)
    code = f"""
import resource, sys
sys.path.insert(0, {ROOT!r})
from bantamtrain.data import tokpack
tokpack._BLOCK = 1 << 18
src = tokpack.BinFile({str(tmp_path / "t.bin")!r})
src.n = int(sys.argv[1])
def _peak_kb():
    try:
        return int([l for l in open("/proc/self/status") if l.startswith("VmHWM")][0].split()[1])
    except (OSError, IndexError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
def _reset_peak():
    try:
        open("/proc/self/clear_refs", "w").write("5")
    except OSError:
        pass
    return _peak_kb()
base = _reset_peak()
tokpack.write({str(tmp_path / "o.tp")!r}, src, 4096, level=1, dict_size=8192,
              dict_sample_tokens=100_000)
print(_peak_kb() - base)
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    grow = [int(subprocess.run([sys.executable, "-c", code, str(n)], capture_output=True, text=True,
                               env=env, check=True).stdout) for n in (1 << 20, 5 << 20)]
    # 4M more tokens: the old in-RAM writer grew by ~8 B/token (~32 MB) here
    assert grow[1] - grow[0] < (4 << 20) / 1024, grow


def test_tokpack_document_index(tmp_path):
    rng = np.random.default_rng(0)
    docs = [np.append(rng.integers(1, 50, int(rng.integers(1, 3000))), 0) for _ in range(300)]
    toks = np.concatenate(docs + [np.array([7, 8, 9])]).astype(np.uint16)   # trailing doc, no EOS
    tokpack.write(tmp_path / "d.tokpack", toks, 50, chunk=1024, level=1, dict_size=0, eos_id=0)
    tp = tokpack.TokPack(tmp_path / "d.tokpack")
    ends = np.cumsum([len(d) for d in docs])
    assert tp.has_doc_index and tp.h["doc_index"]["n_docs"] == 300
    assert np.array_equal(tp.eos_positions, ends - 1)
    assert np.array_equal(tp.doc_starts, np.concatenate([[0], ends]))
    got = list(tp.docs())
    assert len(got) == tp.n_docs == 301
    assert all(np.array_equal(g, d) for g, d in zip(got, docs)) and got[-1].tolist() == [7, 8, 9]
    assert np.array_equal(tp.doc(5), docs[5]) and np.array_equal(tp.doc(-1), [7, 8, 9])
    assert [len(d) for d in tp.docs(10, 13)] == [len(d) for d in docs[10:13]]
    tokpack.write(tmp_path / "n.tokpack", toks, 50, level=1, dict_size=0)   # no eos_id: no index
    with pytest.raises(ValueError, match="no document index"):
        tokpack.TokPack(tmp_path / "n.tokpack").doc_starts
    tokpack.write(tmp_path / "e.tokpack", np.arange(5) + 1, 50, eos_id=0)   # no EOS at all
    assert tokpack.TokPack(tmp_path / "e.tokpack").doc_starts.tolist() == [0]


def test_tokpack_stale_pack_detected(tmp_path):
    toks = _zipf(5000, 300)
    tokpack.write(tmp_path / "train.tokpack", toks, 300, extra={"source": {"prep_id": "old"}})
    assert tokpack.TokPack(tmp_path / "train.tokpack").h["source"]["prep_id"] == "old"
    json.dump({"vocab_size": 300, "prep_id": "old"}, open(tmp_path / "meta.json", "w"))
    assert tokpack.stale_reason(tmp_path / "train.tokpack") is None
    json.dump({"vocab_size": 300, "prep_id": "new"}, open(tmp_path / "meta.json", "w"))
    assert "prep_id" in tokpack.stale_reason(tmp_path / "train.tokpack")
    with pytest.raises(tokpack.StalePackError):
        tokpack.TokPack(tmp_path / "train.tokpack")
    assert len(tokpack.TokPack(tmp_path / "train.tokpack", verify=False)) == len(toks)
    json.dump({"vocab_size": 260}, open(tmp_path / "meta.json", "w"))   # pre-0.3 meta, smaller vocab
    with pytest.raises(tokpack.StalePackError, match="vocab"):
        tokpack.TokPack(tmp_path / "train.tokpack")
    with pytest.raises(ValueError, match="reserved"):
        tokpack.write(tmp_path / "x.tokpack", toks, 300, extra={"vocab": 1})


def test_binfile_reads_like_memmap(tmp_path):
    a = (np.arange(100_000) * 7 % 70_000).astype(np.uint32)
    a.tofile(tmp_path / "t.bin")
    b = tokpack.BinFile(tmp_path / "t.bin", np.uint32)
    assert len(b) == len(a) and b.nbytes == a.nbytes
    for sl in (slice(0, 10), slice(99_990, 100_100), slice(500, 400), slice(None, None, 7),
               slice(None, None, -1), slice(500, 10, -3), slice(10, 500, -1), slice(-5, None)):
        assert np.array_equal(b[sl], a[sl]), sl
    assert b[-1] == a[-1] and b[3] == a[3]
    assert tokpack.token_dtype(65536) == np.uint16 and tokpack.token_dtype(65537) == np.uint32
