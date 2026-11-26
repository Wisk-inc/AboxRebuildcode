# Bantam

Bantam (`pip install bantamtrain`, command `bantam`) is a small, pure-PyTorch toolkit
for training and fine-tuning small language models and small weather models on
limited compute. With one command you can pretrain a small GPT on any Hugging Face
text dataset, fine-tune any Hugging Face causal language model with LoRA, QLoRA or
DoRA (and export a standard PEFT adapter), or train a 6-hour ERA5 weather forecaster.
Every result so far was measured on a 4-core CPU; nothing has been run on a GPU yet.
Made by Nathan from Jamaica. MIT license.

**Start here: [bantam/README.md](bantam/README.md)**: install, quickstarts, every
result with its conditions, and the limitations.

```bash
pip install bantamtrain
```

## Three key results

All measured in a CPU-only sandbox (4 cores, 16 GB RAM).

1. **Pretraining.** A 5.8M-parameter GPT on TinyStories, every run at the same budget
   of 6.8e13 FLOPs. QK-norm (a known trick) plus Zoom (merging pairs of tokens early in
   training, a variant of patch-level training) reached the plain Muon baseline's loss
   with **1.41x less compute**: validation loss 2.589 vs 2.737 (2 and 3 seeds). Zoom
   alone gives 1.12x (3 seeds).
2. **Weather.** A 1.26M-parameter ERA5 forecaster at 5.625°. Muon beat AdamW (0.124
   vs 0.229 MSE), but Zoom **hurt** (0.159). A known low-resolution warm-up helped
   (0.112, 1 seed).
3. **Against published models.** Bantam's one CPU run (1.029 bits per byte) is
   **behind every TinyStories model**, starting with TinyStories-1M (0.822, trained
   with about 3,350x more compute). It beats llama2.c stories260K with at least 157x
   less compute, partly because Bantam's model is 22x larger.

The code, experiments, tests, result logs and documentation are all in
[`bantam/`](bantam/).
