"""dedup: degenerate corpora and the out-of-core dedup_stream (same answer, bounded RAM)."""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain.data import dedup, tokpack  # noqa: E402


@pytest.mark.parametrize("toks", [
    np.concatenate([np.r_[np.arange(1, 1 + k % 7 + 1), 0] for k in range(100)]),   # all < n
    np.array([3, 4, 5, 6, 0]),                                                     # one short doc
    np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),                                    # no EOS
    np.zeros(0),                                                                    # empty
    np.array([0, 0, 0]),                                                            # empty docs
], ids=["all-short", "one-short", "no-eos", "empty", "only-eos"])
def test_dedup_degenerate_corpora(toks, tmp_path):
    toks = toks.astype(np.uint16)
    n_docs = int((toks == 0).sum())
    for keep, st in (dedup.dedup(toks, 0, n=8), dedup.dedup_stream(toks, 0, n=8, tmpdir=tmp_path)):
        assert len(keep) == st["docs"] == n_docs
        assert st["kept_docs"] == int(keep.sum()) and 0.0 <= st["kept_frac"] <= 1.0
        assert len(dedup.filter_tokens(toks, 0, keep)) == st["kept_tokens"]
    with pytest.raises(ValueError):
        dedup.dedup(toks, 0, num_perm=60, bands=8)


def _corpus(seed, nd=1500):
    rng = np.random.default_rng(seed)
    docs = [rng.integers(1, 300, size=rng.integers(2, 80)).astype(np.uint16) for _ in range(nd)]
    for _ in range(nd // 10):
        docs.append(docs[rng.integers(len(docs))].copy())                 # exact
    for _ in range(nd // 10):
        d = docs[rng.integers(len(docs))].copy()
        d[rng.integers(len(d))] = rng.integers(1, 300)                    # near
        docs.append(d)
    order = rng.permutation(len(docs))
    return np.concatenate([np.append(docs[i], 0) for i in order]).astype(np.uint16)


@pytest.mark.parametrize("seed", [0, 1])
def test_dedup_stream_matches_in_memory(seed, tmp_path, monkeypatch):
    toks = _corpus(seed)
    cut = int(np.flatnonzero(toks == 0)[100]) + 1
    for kw in (dict(n=5, num_perm=64, bands=16, threshold=0.5), dict()):
        ref_keep, ref = dedup.dedup(toks, 0, **kw)
        toks[:cut].tofile(tmp_path / "a.bin")
        toks[cut:].tofile(tmp_path / "b.bin")
        srcs = [tokpack.BinFile(tmp_path / "a.bin"), np.memmap(tmp_path / "b.bin", np.uint16, mode="r")]
        keep, st = dedup.dedup_stream(srcs, 0, tmpdir=tmp_path, block=997, **kw)
        assert st.pop("docs_per_source") == [101, ref["docs"] - 101]
        assert np.array_equal(keep, ref_keep) and st == ref
        assert ref["exact_dups"] > 0 and ref["near_dups"] > 0
    # every band key colliding exercises the exact re-check of sorted runs
    monkeypatch.setattr(dedup, "_band_hash", lambda rows: np.zeros(len(rows), np.uint64))
    keep, st = dedup.dedup_stream(toks, 0, tmpdir=tmp_path)
    assert np.array_equal(keep, dedup.dedup(toks, 0)[0])
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("bantam-dedup-")]


def test_dedup_stream_memory_per_document(tmp_path):
    """RAM grows by ~14 B/doc, not by the tens of B/token of the in-memory dedup."""
    rng = np.random.default_rng(0)
    with open(tmp_path / "t.bin", "wb") as f:
        for _ in range(6):
            t = rng.integers(1, 4096, 1 << 20).astype(np.uint16)
            t[::256] = 0
            t.tofile(f)
    code = f"""
import resource, sys
sys.path.insert(0, {ROOT!r})
from bantamtrain.data import dedup, tokpack
dedup._BLOCK = 1 << 18
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
dedup.dedup_stream(src, 0, tmpdir={str(tmp_path)!r})
print(_peak_kb() - base)
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    grow = [int(subprocess.run([sys.executable, "-c", code, str(n)], capture_output=True, text=True,
                               env=env, check=True).stdout) for n in (1 << 20, 6 << 20)]
    # 5M more tokens (~20k docs): dedup() needs ~40 B/token (~200 MB) more here
    assert grow[1] - grow[0] < 8 * 1024, grow
