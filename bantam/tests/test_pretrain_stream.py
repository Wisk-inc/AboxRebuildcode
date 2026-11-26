"""ChunkStream: disjoint DDP shards, per-epoch reshuffle, full batches, zoom padding."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.cli.pretrain import ChunkStream  # noqa: E402

SEQ = 8


def _tokens(n_windows, pad=4):
    return np.arange(n_windows * SEQ + SEQ + pad, dtype=np.uint16)


def _starts(chunk):
    return [int(r[0]) for r in chunk.numpy()]


def test_world1_first_epoch_keeps_the_old_order():
    s = ChunkStream(_tokens(50), SEQ, seed=3)
    want = np.random.default_rng(3).permutation(50) * SEQ          # the pre-0.3 order
    got = [x for _ in range(10) for x in _starts(s.next(5))]
    assert got == list(want)
    assert s.next(4).shape == (4, SEQ + 4)                         # windows hold seq + KMAX tokens


def test_ddp_ranks_read_disjoint_slices_of_one_order():
    world, batch, n = 2, 4, 64
    ranks = [ChunkStream(_tokens(n), SEQ, seed=0, rank=r, world=world) for r in range(world)]
    seen = [[x for _ in range(n // (batch * world)) for x in _starts(s.next(batch))] for s in ranks]
    assert not set(seen[0]) & set(seen[1])                         # no window read twice
    assert sorted(seen[0] + seen[1]) == list(range(0, n * SEQ, SEQ))   # one global epoch
    single = ChunkStream(_tokens(n), SEQ, seed=0)                  # same data as world=1, batch*2
    assert set(_starts(single.next(batch * world))) == set(seen[0][:batch]) | set(seen[1][:batch])
    assert ranks[0].data_epochs == ranks[1].data_epochs == 1.0     # global, not per rank
    # seeds no longer alias across ranks: rank 1 of seed s != rank 0 of seed s+1
    a = ChunkStream(_tokens(n), SEQ, seed=0, rank=1, world=2)
    b = ChunkStream(_tokens(n), SEQ, seed=1, rank=0, world=2)
    assert _starts(a.next(batch)) != _starts(b.next(batch))


def test_wrap_reshuffles_and_never_returns_a_short_batch():
    s = ChunkStream(_tokens(52), SEQ, seed=0)
    epochs = {}
    for _ in range(12):
        c = s.next(16)
        assert c.shape[0] == 16
        epochs.setdefault(s.epochs, []).extend(_starts(c))
    assert len(epochs[0]) == len(epochs[1]) == 48                  # 3 full batches per epoch
    assert epochs[0] != epochs[1]                                  # new order every epoch
    assert set(epochs[0]) | set(epochs[1]) | set(epochs[2]) == set(range(0, 52 * SEQ, SEQ))
    with pytest.raises(ValueError):
        s.next(53)                                                 # more than the corpus


def test_set_position_replays_the_same_batches():
    a = ChunkStream(_tokens(40), SEQ, seed=5)
    for _ in range(7):
        a.next(8)                                                  # into the second epoch
    pos, ep = a.pos, a.epochs
    want = [_starts(a.next(8)) for _ in range(6)]
    b = ChunkStream(_tokens(40), SEQ, seed=5)
    b.set_position(pos, ep)
    assert [_starts(b.next(8)) for _ in range(6)] == want


def test_zoom_levels_above_4_get_long_enough_windows():
    s = ChunkStream(_tokens(30, pad=8), SEQ, seed=0, kmax=8)
    c = s.next(4)
    assert c.shape == (4, SEQ + 8)
    k = 8
    assert c[:, k:SEQ + k].shape == (4, SEQ)                       # full targets at k=8
