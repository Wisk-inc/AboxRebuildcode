"""Wave-2 review (rdw): prepare-text / pack-text edge cases found end to end."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_data_prepare import _docs, _parquet, _run, pack_text, prepare_text  # noqa: E402


def test_failed_prepare_text_does_not_create_out(tmp_path):
    """--out was created before the inputs were checked, so a run that failed on a bad
    --column (or, core-only, on the missing pyarrow) left an empty DIR behind."""
    t = _parquet(tmp_path / "t.parquet", _docs(50))
    out = tmp_path / "new" / "out"
    with pytest.raises(SystemExit):
        _run(prepare_text, ["--train", t, "--out", str(out), "--column", "txt"])
    assert not out.exists() and not out.parent.exists()


def test_pack_stats_report_the_raw_dtype(tmp_path):
    """pack_stats only compared against uint16 bytes, half the real size of a uint32 corpus."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = sorted({w for d in _docs(400) for w in d.replace(".", " .").split()})
    vocab = {"<|eos|>": 0, "<unk>": 1, **{w: 70000 + i for i, w in enumerate(words)}}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(tmp_path / "big.json"))
    out = tmp_path / "out"
    _run(prepare_text, ["--train", _parquet(tmp_path / "t.parquet", _docs(300)),
                        "--val", _parquet(tmp_path / "v.parquet", _docs(30, seed=4)),
                        "--out", str(out), "--tokenizer", str(tmp_path / "big.json"),
                        "--eos", "<|eos|>"])
    _run(pack_text, ["--data", str(out), "--level", "1"])
    st = json.load(open(out / "pack_stats.json"))
    assert st["dtype"] == "uint32"
    for s in ("train", "val"):
        assert st[s]["raw_bytes"] == 4 * st[s]["tokens"] == os.path.getsize(out / f"{s}.bin")
        assert st[s]["ratio_vs_raw"] == pytest.approx(2 * st[s]["ratio_vs_uint16"])


def test_readers_pickle_after_chdir(tmp_path, monkeypatch):
    """__getstate__ resolved the path against the cwd at pickling time: a reader opened by a
    relative path and pickled (DataLoader workers) after a chdir pointed at another file."""
    import pickle

    import numpy as np

    from bantamtrain.data import gridpack, tokpack
    toks = np.arange(5000, dtype=np.uint16) % 300
    tokpack.write(tmp_path / "t.tokpack", toks, 300, chunk=512)
    x = np.random.default_rng(0).standard_normal((20, 2, 4, 8)).astype(np.float32)
    gridpack.write(tmp_path / "f.gridpack", x, np.ones(2), chunk=8)
    monkeypatch.chdir(tmp_path)
    tp, gp = tokpack.TokPack("t.tokpack"), gridpack.GridPack("f.gridpack")
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    assert np.array_equal(pickle.loads(pickle.dumps(tp))[:], toks)
    assert pickle.loads(pickle.dumps(gp)).T == 20
