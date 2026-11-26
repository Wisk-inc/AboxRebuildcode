"""Chunked cross-entropy: never materializes the full (tokens x vocab) logits.

    loss = chunked_cross_entropy(hidden, head_weight, targets, chunk=2048)

The logits of each chunk of `chunk` positions are computed, reduced to a loss
and discarded; torch.utils.checkpoint recomputes them in the backward pass.
Peak memory for the loss goes from tokens*vocab*4 bytes to chunk*vocab*4, which
for a 128k vocabulary at 16k tokens per step is 8 GB -> 1 GB. Matches
F.cross_entropy exactly up to floating-point summation order.

logits = softcap(scale * (h @ W^T + b)) covers Gemma-2's final softcap and
Granite/Cohere-style logit scaling; token_weights gives a weighted mean (e.g. a
prompt-loss weight). `weight` may also be a quantized head (anything with
.matmul(h) = h @ W^T, e.g. qweights.QLinear / QEmbedding / QTiedHead): each chunk
then dequantizes the head itself instead of holding a float copy of the table.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_loss(h, weight, bias, targets, tw, softcap, ignore_index, scale):
    if isinstance(weight, torch.Tensor):
        logits = F.linear(h, weight, bias).float()
    else:
        logits = weight.matmul(h)
        logits = (logits if bias is None else logits + bias.to(logits.dtype)).float()
    if scale != 1.0:
        logits = logits * scale
    if softcap > 0:
        logits = softcap * torch.tanh(logits / softcap)
    if tw is None:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="sum")
    per = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction="none")
    return (per * tw).sum()


def chunked_cross_entropy(h, weight, targets, chunk=2048, bias=None, softcap=0.0,
                          ignore_index=-100, scale=1.0, token_weights=None):
    """h: (..., d) hidden states; weight: (vocab, d) tensor or a quantized head with
    .matmul; targets: (...) long; token_weights: like targets (float), optional.
    Mean over non-ignored targets (weighted mean with token_weights); 0, never NaN, when
    nothing is supervised."""
    h = h.reshape(-1, h.shape[-1])
    targets = targets.reshape(-1)
    valid = targets != ignore_index
    if token_weights is None:
        tw, den = None, valid.sum()
    else:
        tw = torch.where(valid, token_weights.reshape(-1).float(), 0.0)
        den = tw.sum()
    den = torch.where(den > 0, den, torch.ones_like(den)).float()
    if isinstance(weight, torch.Tensor) and weight.dtype != h.dtype \
            and not torch.is_autocast_enabled(h.device.type):
        weight = weight.to(h.dtype)
    grad = torch.is_grad_enabled()
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        args = (h[i:i + chunk], weight, bias, targets[i:i + chunk],
                None if tw is None else tw[i:i + chunk], softcap, ignore_index, scale)
        total = total + (checkpoint(_chunk_loss, *args, use_reentrant=False) if grad
                         else _chunk_loss(*args))
    return total / den
