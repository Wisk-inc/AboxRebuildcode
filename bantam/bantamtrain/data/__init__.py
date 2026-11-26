"""Data layer: compact storage and deduplication."""
import json
import os
import warnings


def open_tokens(data_dir, split, prefer_pack=True):
    """(tokens, source) for a prepare-text directory: DIR/{split}.tokpack when it exists
    and matches DIR/meta.json (source "tokpack"), else DIR/{split}.bin as a read-only
    memmap of meta["dtype"] (uint16 when absent; source "bin").

    A pack left over from an earlier prepare run is skipped with a warning, and so
    is reading a train.bin that was prepared before validation documents were
    removed from it (run `bantam pack-text`, or prepare again).
    """
    import numpy as np

    from .tokpack import TokPack, stale_reason
    meta_path = os.path.join(data_dir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    pack = os.path.join(data_dir, f"{split}.tokpack")
    if prefer_pack and os.path.exists(pack):
        why = stale_reason(pack)
        if why is None:
            return TokPack(pack, verify=False), "tokpack"
        warnings.warn(f"{why}; reading {split}.bin instead", stacklevel=2)
    if split == "train" and "train_docs_dropped_exact_val" not in meta:
        warnings.warn(f"{data_dir}/train.bin may contain validation documents verbatim (prepared "
                      f"before 0.3); run `bantam pack-text --data {data_dir}`", stacklevel=2)
    dtype = np.dtype(meta.get("dtype", "uint16"))
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=dtype, mode="r"), "bin"
