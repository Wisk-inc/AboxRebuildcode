"""Wave-2 core review: ZoomGPT.generate pads a finished row with eos."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bantamtrain.models import ZoomGPT  # noqa: E402


def test_finished_row_is_padded_with_eos_even_when_the_model_continues():
    """A model that always predicts token + 1: row 0 reaches eos (0) and would go on with
    1, 2, ...; the tensor result must hold eos after it (the old test's fixed logits
    emitted eos again anyway, so it never exercised the padding)."""
    V = 8
    m = ZoomGPT(V, 16, 1, 2).eval()
    m._decode = lambda tok, valid, caches, start: F.one_hot(tok, V).float()
    m.logits = lambda h: 10.0 * h.roll(1, dims=-1)              # argmax = last token + 1
    p = torch.tensor([[6], [2]])
    out = m.generate(p, 5, temperature=0, eos_id=0)
    assert out.tolist() == [[6, 7, 0, 0, 0, 0], [2, 3, 4, 5, 6, 7]]
    assert m.generate([[6], [2]], 5, temperature=0, eos_id=0) == [[6, 7, 0], [2, 3, 4, 5, 6, 7]]
    assert m.generate(p, 5, temperature=0).tolist()[0] == [6, 7, 0, 1, 2, 3]   # no eos_id
