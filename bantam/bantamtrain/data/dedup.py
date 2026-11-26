"""Exact + near-duplicate detection on a pre-tokenized corpus.

Works directly on the uint16 token stream (documents separated by EOS), fully
vectorized with numpy: no per-document Python loops in the hot path.

* exact:  64-bit hash of each document's token sequence.
* near:   MinHash over token n-gram shingles + LSH banding (b bands x r rows).
          Candidate pairs from a shared bucket are confirmed when their
          estimated Jaccard similarity >= threshold; duplicates are clustered
          with union-find and one document per cluster is kept.

`dedup` holds the whole corpus and its per-token hashes in RAM; `dedup_stream`
returns the same keep mask for memmapped corpora of any size, with
per-document state only (about 14 B/doc of RAM plus ~340 B/doc of temporary disk).

Duplicates are redundant compute when data is plentiful (one copy carries the
same information) and are known to hurt generalization/memorization
(Lee et al., 2021, "Deduplicating Training Data Makes Language Models Better").
"""
import os
import tempfile

import numpy as np

M61 = np.uint64((1 << 61) - 1)
_BLOCK = 1 << 21             # tokens per block in dedup_stream (~40 B/token of temporaries)
_ROWS = 1 << 18              # documents per block when sorting / hashing per-doc arrays


def doc_bounds(tokens, eos):
    """(starts, ends) of the EOS-terminated documents; tokens after the last EOS are ignored."""
    ends = np.flatnonzero(tokens == eos) + 1          # exclusive end, includes EOS
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)[:len(ends)]
    return starts, ends


def _ngram_hash(tokens, n):
    """Polynomial hash of every length-n window (uint64 arithmetic wraps)."""
    t = tokens.astype(np.uint64) + np.uint64(1)
    L = max(len(t) - n + 1, 0)
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
    if len(ends) == 0:
        return np.zeros(0, np.uint64)
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
    if len(tokens) < n or len(starts) == 0:
        return np.full((len(starts), num_perm), 0xFFFFFFFF, np.uint32)
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


def _check_params(num_perm, bands):
    if num_perm < 1 or bands < 1 or num_perm % bands:
        raise ValueError(f"num_perm ({num_perm}) must be a positive multiple of bands ({bands})")


def _stats(nd, n_exact, n_near, n_pairs, n_kept, tokens, kept_tokens, n, num_perm, bands,
           threshold):
    return {"docs": nd, "exact_dups": n_exact, "near_dups": n_near,
            "near_pairs_confirmed": n_pairs, "kept_docs": n_kept,
            "kept_frac": float(n_kept / nd) if nd else 1.0,
            "tokens": int(tokens), "kept_tokens": int(kept_tokens),
            "params": {"ngram": n, "num_perm": num_perm, "bands": bands,
                       "threshold": threshold}}


def dedup(tokens, eos, n=8, num_perm=64, bands=8, threshold=0.8, seed=0):
    """Return (keep_mask over documents, stats dict)."""
    _check_params(num_perm, bands)
    tokens = np.asarray(tokens)
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
    has_sh = (ends - starts) > n
    cand = np.flatnonzero(keep & has_sh)
    near_drop = np.zeros(nd, dtype=bool)
    n_pairs = 0
    if cand.size:
        sig = minhash(tokens, starts, ends, eos, n, num_perm, seed)
        r = num_perm // bands
        uf = _UF(nd)
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
        roots = np.array([uf.find(i) for i in cand], dtype=np.int64)
        near_keep = np.zeros(nd, dtype=bool)
        near_keep[np.unique(roots)] = True           # root = smallest index in cluster
        near_drop[cand] = ~near_keep[cand]
    keep &= ~near_drop

    stats = _stats(nd, n_exact, int(near_drop.sum()), n_pairs, int(keep.sum()), len(tokens),
                   (ends - starts)[keep].sum(), n, num_perm, bands, threshold)
    return keep, stats


def filter_tokens(tokens, eos, keep):
    starts, ends = doc_bounds(tokens, eos)
    mask = np.repeat(keep, ends - starts)
    return tokens[:len(mask)][mask]


# ---------------------------------------------------------------- out of core
def doc_blocks(tokens, eos, block=None):
    """Yield (offset, block) with each block a run of whole EOS-terminated documents.

    Reads `tokens` (array or np.memmap) in slices of about `block` tokens (more
    when a single document is longer); tokens after the last EOS are not yielded.
    """
    pos, n = 0, len(tokens)
    block = block or _BLOCK
    while pos < n:
        size = block
        while True:
            seg = np.asarray(tokens[pos:pos + size])
            e = np.flatnonzero(seg == eos)
            if e.size or pos + size >= n:
                break
            size *= 2
        if not e.size:
            return
        yield pos, seg[:e[-1] + 1]
        pos += int(e[-1]) + 1


def _mix64(x):
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint64(33))
        x = x * np.uint64(0xFF51AFD7ED558CCD)
        x = x ^ (x >> np.uint64(33))
        x = x * np.uint64(0xC4CEB9FE1A85EC53)
        return x ^ (x >> np.uint64(33))


def _band_hash(rows):
    """64-bit hash of each row of a (k, r) uint32 band."""
    h = np.zeros(len(rows), np.uint64)
    with np.errstate(over="ignore"):
        for j in range(rows.shape[1]):
            h = _mix64(h ^ (rows[:, j].astype(np.uint64) + np.uint64(j + 1)))
    return h


def _equal_runs(keys, full_key):
    """Groups of positions whose `full_key` is equal, given `keys` packed as
    (key high bits | position): an in-place sort of one uint64 per item finds
    the candidates; a run of equal high bits is re-checked with full_key(positions).
    Yields arrays of positions in ascending order, one per group of size >= 2."""
    m = len(keys)
    if m < 2:
        return
    bits = max(1, (m - 1).bit_length())
    shift = np.uint64(bits)
    keys.sort()
    run_starts = []
    for a in range(0, m - 1, _ROWS):
        b = min(a + _ROWS, m - 1)
        eq = (keys[a + 1:b + 1] >> shift) == (keys[a:b] >> shift)
        run_starts.append(np.flatnonzero(eq) + a)
    pairs = np.concatenate(run_starts)
    if not pairs.size:
        return
    # pairs[i] and pairs[i]+1 share high bits; split into maximal runs
    brk = np.flatnonzero(np.diff(pairs) != 1) + 1
    low = np.uint64((1 << bits) - 1)
    for grp in np.split(pairs, brk):
        pos = (keys[grp[0]:grp[-1] + 2] & low).astype(np.int64)   # ascending (sorted low bits)
        fk = full_key(pos)
        order = np.argsort(fk, kind="stable")
        fs = fk[order]
        cut = np.flatnonzero(fs[1:] != fs[:-1]) + 1
        for sub in np.split(order, cut):
            if len(sub) > 1:
                yield pos[sub]


def _pack(dst, keys, bits, offset):
    """dst[i] = (keys[i] with the low `bits` cleared) | (offset + i)."""
    hi = np.uint64(~((1 << bits) - 1) & 0xFFFFFFFFFFFFFFFF)
    dst[:] = (keys & hi) | np.arange(offset, offset + len(keys), dtype=np.uint64)


def _rows(path, a, b, dtype="<u8"):
    """Rows [a, b) of a flat binary file, read without mapping it."""
    return np.fromfile(path, dtype=dtype, count=b - a, offset=a * np.dtype(dtype).itemsize)


def dedup_stream(sources, eos, n=8, num_perm=64, bands=8, threshold=0.8, seed=0,
                 tmpdir=None, block=None):
    """dedup() for corpora that do not fit in RAM; returns the same (keep, stats).

    `sources` is one token array or a list of them (np.memmap, tokpack.BinFile),
    whose EOS-terminated documents are numbered consecutively (tokens after a
    source's last EOS are not a document). One pass computes per-document exact
    hashes, MinHash signatures and LSH band hashes block by block into temporary
    files in `tmpdir` (num_perm*4 + (bands+2)*8 B/doc). Duplicates are then found
    by sorting one packed uint64 per document in place, so RAM is about 14 B/doc
    whatever the token count, plus O(block) for the pass. stats["docs_per_source"]
    gives the document count of each source.
    """
    _check_params(num_perm, bands)
    if not isinstance(sources, (list, tuple)):
        sources = [sources]
    r = num_perm // bands
    with tempfile.TemporaryDirectory(dir=tmpdir, prefix="bantam-dedup-") as td:
        sig_path, eh_path, len_path = (os.path.join(td, f) for f in ("sig", "exact", "len"))
        band_paths = [os.path.join(td, f"band{bi}") for bi in range(bands)]
        nd, n_tokens, docs_per_source = 0, 0, []
        outs = [open(f, "wb") for f in [sig_path, eh_path, len_path] + band_paths]
        try:
            for src in sources:
                n_tokens += len(src)
                d0 = nd
                for _, blk in doc_blocks(src, eos, block):
                    starts, ends = doc_bounds(blk, eos)
                    sig = minhash(blk, starts, ends, eos, n, num_perm, seed)
                    outs[0].write(sig.tobytes())
                    outs[1].write(exact_hashes(blk, starts, ends).astype("<u8").tobytes())
                    outs[2].write((ends - starts).astype("<u8").tobytes())
                    for bi in range(bands):
                        outs[3 + bi].write(_band_hash(sig[:, bi * r:(bi + 1) * r]).astype("<u8")
                                           .tobytes())
                    nd += len(starts)
                docs_per_source.append(nd - d0)
        finally:
            for f in outs:
                f.close()
        if nd == 0:
            st = _stats(0, 0, 0, 0, 0, n_tokens, 0, n, num_perm, bands, threshold)
            st["docs_per_source"] = docs_per_source
            return np.ones(0, bool), st
        # random access (few rows: duplicate candidates only) goes through memmaps
        eh = np.memmap(eh_path, dtype="<u8", mode="r")
        sig = np.memmap(sig_path, dtype=np.uint32, mode="r", shape=(nd, num_perm))
        bits = max(1, (nd - 1).bit_length())
        packed = np.empty(nd, np.uint64)

        # exact duplicates: keep the first occurrence of each hash
        keep = np.ones(nd, bool)
        for a in range(0, nd, _ROWS):
            b = min(a + _ROWS, nd)
            _pack(packed[a:b], _rows(eh_path, a, b), bits, a)
        for grp in _equal_runs(packed, lambda pos: np.asarray(eh[pos])):
            keep[grp[1:]] = False
        n_exact = int(nd - keep.sum())

        # near duplicates among the exact-unique docs longer than n
        ctype = np.uint32 if nd < (1 << 32) else np.int64
        cand = np.concatenate([
            (np.flatnonzero(keep[a:a + _ROWS] & (_rows(len_path, a, min(a + _ROWS, nd)) > n)) + a)
            .astype(ctype) for a in range(0, nd, _ROWS)])
        m = len(cand)
        parent = {}

        def find(x):
            root = x
            while parent.get(root, root) != root:
                root = parent[root]
            while x != root:
                parent[x], x = root, parent[x]
            return root

        n_pairs = 0
        cbits = max(1, (m - 1).bit_length())
        ckeys = packed[:m]
        for bi in range(bands):
            cols = slice(bi * r, (bi + 1) * r)
            for a in range(0, nd, _ROWS):
                b = min(a + _ROWS, nd)
                ca, cb = np.searchsorted(cand, [a, b])
                hb = _rows(band_paths[bi], a, b)
                _pack(ckeys[ca:cb], hb[cand[ca:cb].astype(np.int64) - a], cbits, int(ca))

            def full(pos, cols=cols):
                band = np.ascontiguousarray(sig[cand[pos].astype(np.int64), cols])
                return band.view(np.dtype((np.void, 4 * r))).ravel()

            for grp in _equal_runs(ckeys, full):
                docs = cand[grp].astype(np.int64)
                x, y = docs[:-1], docs[1:]           # consecutive members, as dedup() pairs them
                sim = (np.asarray(sig[x]) == np.asarray(sig[y])).mean(axis=1)
                for xi, yi in zip(x[sim >= threshold].tolist(), y[sim >= threshold].tolist()):
                    ra, rb = find(xi), find(yi)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                    n_pairs += 1
        near_drop = [x for x in list(parent) if find(x) != x]   # root = smallest index in cluster
        keep[near_drop] = False
        kept_tokens = sum(int(_rows(len_path, a, min(a + _ROWS, nd))[keep[a:a + _ROWS]].sum())
                          for a in range(0, nd, _ROWS))
        st = _stats(nd, n_exact, len(near_drop), n_pairs, int(keep.sum()), n_tokens, kept_tokens,
                    n, num_perm, bands, threshold)
        st["docs_per_source"] = docs_per_source
        del eh, sig
        return keep, st
