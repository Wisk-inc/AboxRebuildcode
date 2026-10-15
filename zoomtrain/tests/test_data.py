import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from zoomtrain.data import dedup, gridpack, tokpack  # noqa: E402


def _zipf_tokens(n, vocab, seed=0):
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1)
    p /= p.sum()
    return rng.choice(vocab, size=n, p=p).astype(np.uint16)


def test_tokpack_roundtrip_exact_and_random_access(tmp_path):
    toks = _zipf_tokens(300_000, 4096)
    path = tmp_path / "t.tokpack"
    nbytes = tokpack.write(path, toks, 4096, chunk=65536)
    tp = tokpack.TokPack(path)
    assert len(tp) == len(toks)
    assert np.array_equal(tp[:], toks)
    assert np.array_equal(tp[65530:65550], toks[65530:65550])      # crosses a chunk edge
    assert np.array_equal(tp.read(299_000, 1000), toks[299_000:])
    assert nbytes < toks.nbytes * 0.8                                 # Zipf data compresses
    windows = list(tp.stream(seq=1000, seed=1))
    assert sum(len(w) for w in windows) >= 0.9 * len(toks)
    assert all(len(w) == 1000 for w in windows)


def test_tokpack_no_remap(tmp_path):
    toks = _zipf_tokens(10_000, 300)
    path = tmp_path / "t.tokpack"
    tokpack.write(path, toks, 300, chunk=4096, remap=False)
    assert np.array_equal(tokpack.TokPack(path)[:], toks)


def _corpus_with_dups(eos=0):
    rng = np.random.default_rng(0)
    docs = [rng.integers(1, 500, size=rng.integers(20, 60)).astype(np.uint16) for _ in range(40)]
    docs.append(docs[3].copy())                        # exact dup of 3
    near = docs[5].copy()
    near[2] = 499                                      # near dup of 5 (one token changed)
    docs.append(near)
    docs.append(docs[7][::-1].copy())                  # same multiset, different order: not a dup
    toks = np.concatenate([np.concatenate([d, [eos]]) for d in docs]).astype(np.uint16)
    return toks, len(docs)


def test_dedup_finds_exact_and_near_not_reordered():
    toks, nd = _corpus_with_dups()
    keep, st = dedup.dedup(toks, eos=0, n=5, num_perm=64, bands=16, threshold=0.5)
    assert st["docs"] == nd
    assert not keep[40]                                 # exact dup dropped
    assert not keep[41]                                 # near dup dropped
    assert keep[42]                                     # reversed doc kept
    assert keep[3] and keep[5] and keep[7]
    assert st["exact_dups"] == 1 and st["near_dups"] >= 1
    out = dedup.filter_tokens(toks, 0, keep)
    assert (out == 0).sum() == keep.sum()


def test_gridpack_error_bound_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    T, C, H, W = 50, 3, 8, 16
    base = np.array([5.4e4, 280.0, 1.0e5], dtype=np.float32).reshape(1, C, 1, 1)
    spread = np.array([3000.0, 15.0, 1300.0], dtype=np.float32).reshape(1, C, 1, 1)
    x = (base + spread * rng.standard_normal((T, C, H, W))).astype(np.float32)
    dstd = np.array([230.0, 1.2, 250.0], dtype=np.float32)
    path = tmp_path / "f.gridpack"
    nbytes, max_err = gridpack.write(path, x, dstd, err_frac=0.05, chunk=16)
    gp = gridpack.GridPack(path)
    y = gp.read_all()
    assert y.shape == x.shape
    bound = 0.05 * dstd.reshape(1, C, 1, 1)
    assert (np.abs(y - x) <= bound + 1e-3).all()
    assert (max_err <= 0.05 * dstd + 1e-3).all()
    assert np.array_equal(gp.read(15, 3), y[15:18])
    assert nbytes < x.nbytes / 2
    # float16 would NOT meet this bound for geopotential / pressure
    f16_err = np.abs(x.astype(np.float16).astype(np.float32) - x)
    assert f16_err[:, 0].max() > bound[0, 0] or np.isinf(x.astype(np.float16)).any()
