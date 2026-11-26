"""TokPack: lossless, random-access, compressed storage for token streams.

Format v2 (default, b"ZTPK2\\0"):
    1. frequency-rank remap: token id -> rank of its corpus frequency, so the
       common tokens become small integers;
    2. LEB128 varint: ranks < 128 take one byte, < 16384 two, < 2^21 three;
    3. zstd-19 per chunk of `chunk` tokens, all chunks sharing one zstd
       dictionary trained on a sample of the corpus and stored in the file.
    The dictionary is what lets small chunks (fast random access) compress
    like large ones: measured on 18M held-out TinyStories tokens (4096 vocab),
    16K-token chunks reach 5.96 bits/token (0.75 B/token) with a 1 MB
    dictionary vs 7.08 bits/token without one, and decompress in 0.05 ms.

Format v1 (b"ZTPK1\\0", read-only): rank remap -> byte-plane shuffle -> zstd.
    Kept so old files still load. The byte shuffle turned out to *hurt*
    (8.55 bits/token vs 7.86 for plain zstd on uint16), which is why v2
    replaced it.

Layout (little endian)
    magic | u32 header_len | header JSON
    | u64[n_chunks + 1] byte offsets (relative to data start)
    | u16/u32[vocab] rank -> token id table
    | (v2) zstd dictionary bytes
    | chunk payloads
    | (optional) document index: zstd-compressed u64 document lengths (EOS
      included), header["doc_index"] = {eos_id, n_docs, bytes}
All steps are invertible, so decode(encode(x)) == x exactly. Readers that
predate the document index ignore it (it follows the last payload), and the
optional header["source"] records which prepared data a pack was made from
(see `stale_reason`).

The writer streams: `TokPackWriter` / `write` encode chunk by chunk with
memory O(chunk + dict_sample_tokens), so a memmap of any size can be packed.
"""
import json
import os
import shutil
import struct
import threading
from array import array
from collections import OrderedDict

import numpy as np
import zstandard as zstd

MAGIC_V1 = b"ZTPK1\0"
MAGIC_V2 = b"ZTPK2\0"
MAX_VOCAB = 1 << 21          # LEB128 ranks use at most three bytes
_HEADER_KEYS = ("version", "n_tokens", "vocab", "chunk", "n_chunks", "codec", "dict_bytes",
                "table_dtype", "pipeline", "doc_index")
_BLOCK = 1 << 22             # tokens per block when streaming a source


class StalePackError(ValueError):
    """The pack was made from other data than the prepared directory now holds."""


def token_dtype(vocab):
    """Smallest unsigned dtype holding ids < vocab (uint16 up to 65536, else uint32)."""
    return np.dtype(np.uint16 if vocab <= 65536 else np.uint32)


# ---------------------------------------------------------------- varint codec
def leb128_encode(a):
    """Non-negative ints < 2^21 -> LEB128 bytes (vectorized)."""
    a = np.asarray(a, dtype=np.uint32)
    if a.size and int(a.max()) >= MAX_VOCAB:
        raise ValueError(f"LEB128 codec holds values < 2^21, got {int(a.max())}")
    lens = 1 + (a >= 128).astype(np.int64) + (a >= 16384).astype(np.int64)
    pos = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
    out = np.empty(int(lens.sum()), dtype=np.uint8)
    out[pos] = (a & 0x7F) | np.where(lens > 1, 0x80, 0).astype(np.uint32)
    m2 = lens >= 2
    out[pos[m2] + 1] = ((a[m2] >> 7) & 0x7F) | np.where(lens[m2] > 2, 0x80, 0).astype(np.uint32)
    m3 = lens == 3
    out[pos[m3] + 2] = a[m3] >> 14
    return out


def leb128_decode(b, n=None):
    """LEB128 bytes -> uint32 values (vectorized, values < 2^21)."""
    b = np.frombuffer(b, dtype=np.uint8) if isinstance(b, (bytes, bytearray)) else b
    if len(b) == 0:
        v = np.zeros(0, np.uint32)
    else:
        cont = b >= 128
        starts = np.flatnonzero(np.concatenate([[True], ~cont[:-1]]))
        v = (b[starts] & 0x7F).astype(np.uint32)
        has2 = cont[starts]
        i2 = starts[has2] + 1
        v[has2] |= (b[i2] & 0x7F).astype(np.uint32) << 7
        has3 = np.zeros_like(has2)
        has3[has2] = cont[i2]
        v[has3] |= b[starts[has3] + 2].astype(np.uint32) << 14
    if n is not None and len(v) != n:
        raise ValueError(f"corrupt TokPack chunk: decoded {len(v)} values, expected {n}")
    return v


# ---------------------------------------------------------------- v1 helpers
def _unshuffle_bytes(raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(2, n)
    return np.ascontiguousarray(b.T).view(np.uint16).reshape(n)


# ---------------------------------------------------------------- writer
def _as_tokens(a):
    a = np.asarray(a)
    if a.size == 0:
        return a.reshape(0).astype(np.int64)
    if a.dtype.kind not in "iu":
        raise TypeError(f"token ids must be integers, got dtype {a.dtype}")
    return a.reshape(-1)


def _blocks(tokens, block=None):
    """1-D integer blocks of an array, memmap, TokPack or list (read in slices)."""
    if not hasattr(tokens, "__len__") or isinstance(tokens, (list, tuple)):
        tokens = _as_tokens(tokens)
    block = block or _BLOCK
    for s in range(0, len(tokens), block):
        yield _as_tokens(tokens[s:s + block])


def count_tokens(tokens, vocab, block=None):
    """np.bincount(tokens, minlength=vocab) computed block by block."""
    counts = np.zeros(vocab, np.int64)
    for blk in _blocks(tokens, block):
        _check_range(blk, vocab)
        counts += np.bincount(blk.astype(np.int64, copy=False), minlength=vocab)
    return counts


def _check_range(blk, vocab):
    if blk.size and (int(blk.max()) >= vocab or int(blk.min()) < 0):
        raise ValueError(f"token ids must lie in [0, {vocab}): got min {int(blk.min())}, "
                         f"max {int(blk.max())}")


class BinFile:
    """A flat token file (prepare-text's .bin) read by slices with pread.

    Unlike np.memmap, pages read once do not stay mapped in the process, so a
    pass over a corpus larger than RAM keeps RSS at one slice. Slices are copies.
    """

    def __init__(self, path, dtype=np.uint16):
        self.path, self.dtype = os.fspath(path), np.dtype(dtype)
        self.n = os.path.getsize(self.path) // self.dtype.itemsize

    def __len__(self):
        return self.n

    @property
    def nbytes(self):
        return self.n * self.dtype.itemsize

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            i = int(key) + (self.n if key < 0 else 0)
            if not 0 <= i < self.n:
                raise IndexError(f"index {int(key)} out of range for {self.n} tokens")
            return self[i:i + 1][0]
        idx = range(*key.indices(self.n))
        if len(idx) == 0:
            return np.zeros(0, self.dtype)
        lo, hi = min(idx[0], idx[-1]), max(idx[0], idx[-1]) + 1     # covering range, any step
        a = np.fromfile(self.path, dtype=self.dtype, count=hi - lo,
                        offset=lo * self.dtype.itemsize)
        return a if idx.step == 1 else a[idx[0] - lo::idx.step]


class TokPackWriter:
    """Streaming v2 writer: append token blocks, close() writes `path` atomically.

    `counts` is the frequency table of the tokens that will be appended
    (`count_tokens`); it fixes the rank remap before the first chunk is encoded.
    Payloads and the document index go to side files and are spliced in by
    close(), so memory is O(chunk + dict_sample_tokens) whatever the corpus size.
    With `eos_id` set the pack also stores a document index (see TokPack.docs);
    `extra` adds JSON fields to the header (e.g. {"source": {...}}).

        with TokPackWriter(path, vocab, counts, eos_id=eos) as w:
            for block in blocks:
                w.append(block)
    """

    def __init__(self, path, vocab, counts, chunk=16384, level=19, dict_size=1 << 20,
                 dict_sample_tokens=2_000_000, threads=0, eos_id=None, extra=None):
        vocab = int(vocab)
        if not 1 <= vocab <= MAX_VOCAB:
            raise ValueError(f"vocab must be in [1, {MAX_VOCAB}], got {vocab}")
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        if len(counts) > vocab:
            raise ValueError(f"counts has {len(counts)} entries for vocab {vocab}")
        counts = np.pad(counts, (0, vocab - len(counts)))
        bad = set(extra or {}) & set(_HEADER_KEYS)
        if bad:
            raise ValueError(f"extra header fields {sorted(bad)} are reserved")
        self.path, self.vocab, self.chunk, self.level = os.fspath(path), vocab, chunk, level
        self.threads, self.eos_id = threads, eos_id
        self.extra = dict(extra or {})
        self.rank_to_tok = np.argsort(-counts, kind="stable").astype(np.uint32)
        self.tok_to_rank = np.empty(vocab, dtype=np.uint32)
        self.tok_to_rank[self.rank_to_tok] = np.arange(vocab, dtype=np.uint32)
        n = int(counts.sum())
        # the dictionary must pay for itself: cap it at ~5 % of the varint stream
        approx_varint = int(n * (1 + (counts[self.rank_to_tok[128:]].sum() / max(n, 1))))
        self.dict_size = min(dict_size, approx_varint // 20)
        self.want_dict = self.dict_size >= 4096 and n >= 8 * chunk
        self.dict_sample_tokens = dict_sample_tokens
        self.dict_bytes = b""
        self.cctx = None if self.want_dict else zstd.ZstdCompressor(level=level, threads=threads)
        self.held, self.n_held = [], 0           # ranks kept until the dictionary is trained
        self.buf = np.empty(chunk, np.uint32)    # the chunk being filled
        self.fill = 0
        self.n = 0
        self.offsets = array("Q", [0])
        self._payload = open(self.path + ".payload.tmp", "w+b")
        self._docs = open(self.path + ".docs.tmp", "w+b") if eos_id is not None else None
        self._doc_z = zstd.ZstdCompressor(level=10).compressobj() if eos_id is not None else None
        self.n_docs, self.last_eos = 0, -1
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def append(self, tokens):
        """Append token ids (any integer array-like); returns the running token count."""
        if self.closed:
            raise ValueError("TokPackWriter is closed")
        for blk in _blocks(tokens):
            _check_range(blk, self.vocab)
            if self._docs is not None:
                e = np.flatnonzero(blk == self.eos_id).astype(np.int64) + self.n
                if e.size:
                    lens = np.diff(np.concatenate([[self.last_eos], e])).astype("<u8")
                    self._docs.write(self._doc_z.compress(lens.tobytes()))
                    self.n_docs += int(e.size)
                    self.last_eos = int(e[-1])
            self.n += len(blk)
            ranks = self.tok_to_rank[blk]
            if self.cctx is None:                  # still sampling for the dictionary
                self.held.append(ranks)
                self.n_held += len(ranks)
                if self.n_held >= self.dict_sample_tokens:
                    self._train_dict()
            else:
                self._feed(ranks)
        return self.n

    def _train_dict(self):
        sample = np.concatenate(self.held) if self.held else np.zeros(0, np.uint32)
        sample = sample[:self.dict_sample_tokens]
        samples = [leb128_encode(sample[s:s + 4096]).tobytes()
                   for s in range(0, len(sample), 4096)]
        try:
            d = zstd.train_dictionary(self.dict_size, samples, level=self.level,
                                      threads=self.threads or 1)
            self.dict_bytes = d.as_bytes()
        except zstd.ZstdError:
            self.dict_bytes = b""
        zd = zstd.ZstdCompressionDict(self.dict_bytes) if self.dict_bytes else None
        self.cctx = zstd.ZstdCompressor(level=self.level, dict_data=zd, threads=self.threads)
        held, self.held, self.n_held = self.held, [], 0
        for r in held:
            self._feed(r)

    def _feed(self, ranks):
        i = 0
        while i < len(ranks):
            k = min(self.chunk - self.fill, len(ranks) - i)
            self.buf[self.fill:self.fill + k] = ranks[i:i + k]
            self.fill += k
            i += k
            if self.fill == self.chunk:
                self._emit()

    def _emit(self):
        p = self.cctx.compress(leb128_encode(self.buf[:self.fill]).tobytes())
        self._payload.write(p)
        self.offsets.append(self.offsets[-1] + len(p))
        self.fill = 0

    def close(self):
        """Finish the file; returns the payload bytes (chunks + dictionary), like write()."""
        if self.closed:
            raise ValueError("TokPackWriter is closed")
        tmp = self.path + ".tmp"
        try:
            if self.cctx is None:
                self._train_dict()
            if self.fill:
                self._emit()
            header = {"version": 2, "n_tokens": self.n, "vocab": self.vocab, "chunk": self.chunk,
                      "n_chunks": len(self.offsets) - 1, "codec": f"zstd-{self.level}",
                      "dict_bytes": len(self.dict_bytes),
                      "table_dtype": "u16" if self.vocab <= 65536 else "u32",
                      "pipeline": "rank-remap|leb128|zstd+dict"}
            if self._docs is not None:
                self._docs.write(self._doc_z.flush())
                header["doc_index"] = {"eos_id": int(self.eos_id), "n_docs": self.n_docs,
                                       "bytes": self._docs.tell(), "codec": "u64-doclen|zstd"}
            header.update(self.extra)
            hb = json.dumps(header).encode()
            with open(tmp, "wb") as f:
                f.write(MAGIC_V2)
                f.write(struct.pack("<I", len(hb)))
                f.write(hb)
                f.write(np.asarray(self.offsets, dtype=np.uint64).tobytes())
                f.write(self.rank_to_tok.astype(np.uint16 if self.vocab <= 65536 else np.uint32)
                        .tobytes())
                f.write(self.dict_bytes)
                for side in (self._payload, self._docs):
                    if side is not None:
                        side.seek(0)
                        shutil.copyfileobj(side, f, 1 << 24)
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        finally:
            self._cleanup()
        self.result = self.offsets[-1] + len(self.dict_bytes)
        return self.result

    def abort(self):
        """Drop the side files without writing `path`."""
        self._cleanup()

    def _cleanup(self):
        self.closed = True
        for side in (self._payload, self._docs):
            if side is not None and not side.closed:
                side.close()
                os.remove(side.name)


def write(path, tokens, vocab, chunk=16384, level=19, dict_size=1 << 20,
          dict_sample_tokens=2_000_000, threads=0, eos_id=None, extra=None):
    """Write a v2 TokPack. Returns the total payload bytes (chunks + dictionary).

    `tokens` may be an array, np.memmap or anything with len() and slicing: it is
    read in blocks twice (frequencies, then encoding), so memory stays
    O(chunk + dict_sample_tokens). eos_id / extra: see TokPackWriter.
    """
    if not hasattr(tokens, "__len__") or isinstance(tokens, (list, tuple)):
        tokens = _as_tokens(tokens)
    counts = count_tokens(tokens, vocab)
    with TokPackWriter(path, vocab, counts, chunk, level, dict_size, dict_sample_tokens,
                       threads, eos_id, extra) as w:
        for blk in _blocks(tokens):
            w.append(blk)
    return w.result


# ---------------------------------------------------------------- reader
def _meta_for(path):
    meta = os.path.join(os.path.dirname(os.path.abspath(path)), "meta.json")
    if not os.path.exists(meta):
        return None
    try:
        with open(meta) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _source_mismatch(path, header, meta):
    """Why a pack does not match its directory's meta.json (prepare-text), or None."""
    if meta is None:
        return None
    want = meta.get("prep_id")
    have = (header.get("source") or {}).get("prep_id")
    if want is not None and have != want:
        return (f"{path} was packed from another prepare run than its meta.json describes "
                f"(prep_id {have} != {want}); re-run `bantam pack-text` or delete it")
    if "vocab_size" in meta and header.get("vocab", 0) > meta["vocab_size"]:
        return (f"{path} has vocab {header['vocab']} but meta.json says {meta['vocab_size']}: it "
                f"is left over from an earlier prepare run; re-run `bantam pack-text` or delete it")
    return None


def stale_reason(path):
    """None if the TokPack at `path` matches DIR/meta.json, else a message saying why not."""
    tp = TokPack(path, verify=False)
    return _source_mismatch(tp.path, tp.h, _meta_for(tp.path))


class TokPack:
    """Random-access reader with an LRU of decoded chunks.

    Thread-safe (one zstd decompressor per thread, locked cache) and picklable
    (the copy reopens the file, e.g. in a DataLoader worker). Returned arrays are
    read-only views, like np.memmap(mode="r"): copy before modifying. With
    `verify` (default) a pack that does not match its directory's meta.json
    raises StalePackError.
    """

    def __init__(self, path, cache_chunks=64, verify=True):
        self.path = os.fspath(path)
        self._abspath = os.path.abspath(self.path)    # for pickling, should the cwd change
        self.cache_chunks = cache_chunks
        self.verify = verify
        self._open()
        if verify:
            why = _source_mismatch(self.path, self.h, _meta_for(self.path))
            if why:
                raise StalePackError(why)

    def _open(self):
        self.mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        magic = bytes(self.mm[:6])
        if magic not in (MAGIC_V1, MAGIC_V2):
            raise ValueError(f"{self.path} is not a TokPack file")
        self.version = 2 if magic == MAGIC_V2 else 1
        hlen = struct.unpack("<I", bytes(self.mm[6:10]))[0]
        self.h = json.loads(bytes(self.mm[10:10 + hlen]))
        pos = 10 + hlen
        nc = self.h["n_chunks"]
        self.offsets = np.frombuffer(self.mm[pos:pos + 8 * (nc + 1)].tobytes(), dtype=np.uint64)
        pos += 8 * (nc + 1)
        v = self.h["vocab"]
        self._dict = None
        if self.version == 1:
            self.rank_to_tok = None
            if self.h["remap"]:
                self.rank_to_tok = np.frombuffer(self.mm[pos:pos + 2 * v].tobytes(), dtype=np.uint16)
                pos += 2 * v
            self.dict_bytes = 0
        else:
            tdt = np.uint16 if self.h["table_dtype"] == "u16" else np.uint32
            w = np.dtype(tdt).itemsize
            self.rank_to_tok = np.frombuffer(self.mm[pos:pos + w * v].tobytes(), dtype=tdt)
            pos += w * v
            nd = self.h["dict_bytes"]
            self.dict_bytes = nd
            self._dict = self.mm[pos:pos + nd].tobytes() if nd else None
            pos += nd
        self.data_start = pos
        if self.data_start + int(self.offsets[-1]) > len(self.mm):
            raise ValueError(f"{self.path} is truncated")
        self.n = self.h["n_tokens"]
        self.chunk = self.h["chunk"]
        self.out_dtype = np.uint16 if v <= 65536 else np.uint32
        self.cache = OrderedDict()
        self._lock = threading.Lock()
        self._tl = threading.local()
        self._doc_lens = self._starts = None
        self._empty = np.zeros(0, self.out_dtype)
        self._empty.flags.writeable = False

    def __getstate__(self):
        return {"path": self._abspath, "cache_chunks": self.cache_chunks,
                "verify": self.verify}

    def __setstate__(self, state):
        self.__init__(**state)

    @property
    def dctx(self):
        """This thread's zstd decompressor (python-zstandard objects are not thread-safe)."""
        d = getattr(self._tl, "dctx", None)
        if d is None:
            zd = zstd.ZstdCompressionDict(self._dict) if self._dict else None
            d = self._tl.dctx = zstd.ZstdDecompressor(dict_data=zd)
        return d

    def __len__(self):
        return self.n

    @property
    def compressed_bytes(self):
        return int(self.offsets[-1]) + self.dict_bytes

    def _chunk(self, i):
        with self._lock:
            arr = self.cache.get(i)
            if arr is not None:
                self.cache.move_to_end(i)
                return arr
        a = self.data_start + int(self.offsets[i])
        b = self.data_start + int(self.offsets[i + 1])
        n = min(self.chunk, self.n - i * self.chunk)
        raw = self.dctx.decompress(self.mm[a:b].tobytes())
        if self.version == 1:
            arr = _unshuffle_bytes(raw, n)
            if self.rank_to_tok is not None:
                arr = self.rank_to_tok[arr]
        else:
            arr = self.rank_to_tok[leb128_decode(raw, n)].astype(self.out_dtype, copy=False)
        arr.flags.writeable = False
        with self._lock:
            self.cache[i] = arr
            while len(self.cache) > self.cache_chunks:
                self.cache.popitem(last=False)
        return arr

    def read(self, start, length):
        """Tokens [start, start + length) as a read-only array (empty if length <= 0)."""
        start, length = int(start), int(length)
        if length <= 0:
            return self._empty
        end = start + length
        if start < 0 or end > self.n:
            raise IndexError(f"read({start}, {length}) outside [0, {self.n})")
        c0, c1 = start // self.chunk, (end - 1) // self.chunk
        parts = [self._chunk(i) for i in range(c0, c1 + 1)]
        if len(parts) == 1:
            joined = parts[0]
        else:
            joined = np.concatenate(parts)
            joined.flags.writeable = False
        off = start - c0 * self.chunk
        return joined[off:off + length]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            i = int(key)
            if i < 0:
                i += self.n
            if not 0 <= i < self.n:
                raise IndexError(f"index {int(key)} out of range for TokPack of length {self.n}")
            return self.read(i, 1)[0]
        if not isinstance(key, slice):
            raise TypeError(f"TokPack indices must be int or slice, not {type(key).__name__}")
        start, stop, step = key.indices(self.n)
        if step == 1:
            return self.read(start, stop - start)
        idx = range(start, stop, step)
        if len(idx) == 0:
            return self._empty
        lo, hi = min(idx[0], idx[-1]), max(idx[0], idx[-1]) + 1
        return self.read(lo, hi - lo)[idx[0] - lo::step]

    def stream(self, seq, seed=0, buffer_chunks=32):
        """Yield every full length-`seq` window of the grid [g*seq, (g+1)*seq) once, shuffled.

        n // seq windows per epoch: only the final n % seq tokens are never
        yielded. Windows are visited chunk by chunk in a random chunk order (a
        window belongs to the chunk holding its start and may run into the
        following chunks, so seq > chunk works too), then mixed in a buffer of
        about `buffer_chunks` chunks.
        """
        if seq < 1:
            raise ValueError(f"seq must be >= 1, got {seq}")
        rng = np.random.default_rng(seed)
        n_win = self.n // seq
        order = rng.permutation(self.h["n_chunks"])
        pool = []
        per_chunk = max(1, self.chunk // seq)
        for i in order:
            lo = -(-int(i) * self.chunk // seq)                # first window starting in chunk i
            hi = min(n_win, -(-(int(i) + 1) * self.chunk // seq))
            if lo >= hi:
                continue
            c = self._chunk(int(i))
            base = int(i) * self.chunk
            for g in range(lo, hi):
                s = g * seq
                pool.append(c[s - base:s - base + seq] if s + seq <= base + len(c)
                            else self.read(s, seq))
            if len(pool) >= buffer_chunks * per_chunk:
                rng.shuffle(pool)
                half = len(pool) // 2
                yield from pool[half:]
                pool = pool[:half]
        rng.shuffle(pool)
        yield from pool

    # ------------------------------------------------------------ documents
    @property
    def has_doc_index(self):
        return "doc_index" in self.h

    def _lens(self):
        if self._doc_lens is None:
            di = self.h.get("doc_index")
            if di is None:
                raise ValueError(f"{self.path} has no document index (packed before 0.3 or "
                                 f"without eos_id); re-run `bantam pack-text`")
            a = self.data_start + int(self.offsets[-1])
            raw = zstd.ZstdDecompressor().decompress(self.mm[a:a + di["bytes"]].tobytes(),
                                                     max_output_size=8 * di["n_docs"] + 1)
            lens = np.frombuffer(raw, dtype="<u8").astype(np.int64)
            if len(lens) != di["n_docs"]:
                raise ValueError(f"corrupt document index in {self.path}")
            self._doc_lens = lens
        return self._doc_lens

    @property
    def eos_positions(self):
        """Positions of the EOS tokens that end each document (int64)."""
        return np.cumsum(self._lens()) - 1

    @property
    def doc_starts(self):
        """Start offset of every document (int64, read-only). Document i is
        [doc_starts[i], doc_starts[i+1]) (the last one ends at len(self)) and ends
        with its EOS, except a final document without EOS, if any."""
        if self._starts is None:
            starts = np.concatenate([[0], np.cumsum(self._lens())]).astype(np.int64)
            starts = starts if starts[-1] < self.n else starts[:-1]
            starts.flags.writeable = False
            self._starts = starts
        return self._starts

    @property
    def n_docs(self):
        return len(self.doc_starts)

    def doc(self, i):
        starts = self.doc_starts
        i = int(i) + (len(starts) if i < 0 else 0)
        if not 0 <= i < len(starts):
            raise IndexError(f"document {i} out of range ({len(starts)} documents)")
        end = starts[i + 1] if i + 1 < len(starts) else self.n
        return self.read(starts[i], end - starts[i])

    def docs(self, start=0, stop=None):
        """Iterate documents start..stop-1 in order (read-only arrays, EOS included)."""
        starts = self.doc_starts
        ends = np.append(starts[1:], self.n)
        for i in range(*slice(start, stop).indices(len(starts))):
            yield self.read(starts[i], ends[i] - starts[i])
