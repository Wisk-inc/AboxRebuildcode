"""Exact + near-duplicate detection on a pre-tokenized corpus.

Works directly on the uint16 token stream (documents separated by EOS), fully
vectorized with numpy: no per-document Python loops in the hot path.

* exact:  64-bit hash of each document's token sequence.
* near:   MinHash over token n-gram shingles + LSH banding (b bands x r rows).
          Candidate pairs from a shared bucket are confirmed when their
          estimated Jaccard similarity >= threshold; duplicates are clustered
          with union-find and one document per cluster is kept.

Duplicates are redundant compute when data is plentiful (one copy carries the
same information) and are known to hurt generalization/memorization
(Lee et al., 2021, "Deduplicating Training Data Makes Language Models Better").
"""
from collections import defaultdict

import numpy as np

M61 = np.uint64((1 << 61) - 1)


def doc_bounds(tokens, eos):
    ends = np.flatnonzero(tokens == eos) + 1          # exclusive end, includes EOS
    starts = np.concatenate([[0], ends[:-1]])
    return starts, ends


def _ngram_hash(tokens, n):
    """Polynomial hash of every length-n window (uint64 arithmetic wraps)."""
    t = tokens.astype(np.uint64) + np.uint64(1)
    L = len(t) - n + 1
    h = np.zeros(L, dtype=np.uint64)
    base = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * base + t[j:j + L]
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(32)
    return h


def exact_hashes(tokens, starts, ends):
    """One 64-bit hash per document (order-sensitive, length-aware)."""
    t = tokens[:ends[-1]].astype(np.uint64) + np.uint64(1)
    pos = np.arange(len(t), dtype=np.uint64)
    # mix each token with its position *within its document*, then sum per doc
    rel = pos - np.repeat(starts.astype(np.uint64), ends - starts)
    with np.errstate(over="ignore"):
        rel_mixed = t * np.uint64(0x9E3779B97F4A7C15) ^ (rel * np.uint64(0xC2B2AE3D27D4EB4F))
        rel_mixed ^= rel_mixed >> np.uint64(31)
        rel_mixed *= np.uint64(0x94D049BB133111EB)
    sums = np.add.reduceat(rel_mixed, starts)
    with np.errstate(over="ignore"):
        return sums ^ (ends - starts).astype(np.uint64) * np.uint64(0xD6E8FEB86659FD93)


def minhash(tokens, starts, ends, eos, n=8, num_perm=64, seed=0):
    """(n_docs, num_perm) uint32 MinHash signatures; docs shorter than n get all-max."""
    h = _ngram_hash(tokens, n)
    L = len(h)
    # a window is valid if it lies inside one document (no EOS among its first n-1 tokens)
    is_eos = (tokens == eos).astype(np.int32)
    c = np.concatenate([[0], np.cumsum(is_eos)])
    valid = (c[n - 1:n - 1 + L] - c[:L]) == 0
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 1 << 61, size=num_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 1 << 61, size=num_perm, dtype=np.uint64)
    seg_starts = np.minimum(starts, L - 1)
    sig = np.empty((len(starts), num_perm), dtype=np.uint32)
    maxv = np.uint64(0xFFFFFFFFFFFFFFFF)
    for k in range(num_perm):
        with np.errstate(over="ignore"):
            v = (h * a[k] + b[k]) >> np.uint64(32)
        v = np.where(valid, v, maxv)
        sig[:, k] = np.minimum(np.minimum.reduceat(v, seg_starts), 0xFFFFFFFF).astype(np.uint32)
    return sig


class _UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedup(tokens, eos, n=8, num_perm=64, bands=8, threshold=0.8, seed=0):
    """Return (keep_mask over documents, stats dict)."""
    assert num_perm % bands == 0
    starts, ends = doc_bounds(tokens, eos)
    nd = len(starts)
    keep = np.ones(nd, dtype=bool)

    # exact duplicates: keep first occurrence
    eh = exact_hashes(tokens, starts, ends)
    _, first = np.unique(eh, return_index=True)
    exact_keep = np.zeros(nd, dtype=bool)
    exact_keep[first] = True
    n_exact = int(nd - exact_keep.sum())
    keep &= exact_keep

    # near duplicates among the exact-unique docs
    sig = minhash(tokens, starts, ends, eos, n, num_perm, seed)
    has_sh = (ends - starts) > n
    cand = np.flatnonzero(keep & has_sh)
    r = num_perm // bands
    uf = _UF(nd)
    n_pairs = 0
    for bi in range(bands):
        band = np.ascontiguousarray(sig[cand, bi * r:(bi + 1) * r])
        key = band.view(np.dtype((np.void, band.dtype.itemsize * r))).ravel()
        order = np.argsort(key, kind="stable")
        ks = key[order]
        same = np.flatnonzero(ks[1:] == ks[:-1])
        for i in same:  # consecutive equal keys -> candidate pair
            x, y = cand[order[i]], cand[order[i + 1]]
            if np.mean(sig[x] == sig[y]) >= threshold:
                uf.union(x, y)
                n_pairs += 1
    roots = np.array([uf.find(i) for i in cand])
    near_keep = np.zeros(nd, dtype=bool)
    near_keep[np.unique(roots)] = True               # root = smallest index in cluster
    near_drop = np.zeros(nd, dtype=bool)
    near_drop[cand] = ~near_keep[cand]
    keep &= ~near_drop

    stats = {"docs": nd, "exact_dups": n_exact, "near_dups": int(near_drop.sum()),
             "near_pairs_confirmed": n_pairs, "kept_docs": int(keep.sum()),
             "kept_frac": float(keep.mean()),
             "tokens": int(len(tokens)), "kept_tokens": int((ends - starts)[keep].sum()),
             "params": {"ngram": n, "num_perm": num_perm, "bands": bands,
                        "threshold": threshold}}
    return keep, stats


def filter_tokens(tokens, eos, keep):
    starts, ends = doc_bounds(tokens, eos)
    mask = np.repeat(keep, ends - starts)
    return tokens[:len(mask)][mask]
