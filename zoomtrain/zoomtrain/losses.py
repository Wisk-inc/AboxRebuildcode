"""Chunked cross-entropy: never materializes the full (tokens x vocab) logits.

    loss = chunked_cross_entropy(hidden, head_weight, targets, chunk=2048)

The logits of each chunk of `chunk` positions are computed, reduced to a loss
and discarded; torch.utils.checkpoint recomputes them in the backward pass.
Peak memory for the loss goes from tokens*vocab*4 bytes to chunk*vocab*4, which
for a 128k vocabulary at 16k tokens per step is 8 GB -> 1 GB. Matches
F.cross_entropy exactly up to floating-point summation order.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(h, weight, bias, targets, softcap, ignore_index):
    logits = F.linear(h, weight, bias).float()
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="sum")


def chunked_cross_entropy(h, weight, targets, chunk=2048, bias=None, softcap=0.0,
                          ignore_index=-100):
    """h: (..., d) hidden states; weight: (vocab, d); targets: (...) long. Mean over
    non-ignored targets."""
    h = h.reshape(-1, h.shape[-1])
    targets = targets.reshape(-1)
    n_valid = (targets != ignore_index).sum().clamp_min(1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        total = total + checkpoint(_chunk_loss, h[i:i + chunk], weight, bias,
                                   targets[i:i + chunk], softcap, ignore_index,
                                   use_reentrant=False)
    return total / n_valid
