# Fine-tuning race

HuggingFaceTB/SmolLM2-135M, databricks-dolly-15k: 800 train / 200 judge examples (split seed 1234), seq 512, micro-batch 2 x accum 2, 1.0 epoch, LR 0.0002, LoRA r=16 alpha=32 dropout=0.05, 4 threads, seed 1234.

Versions: torch 2.15.0+cpu, transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, datasets 5.2.0, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.3, psutil 7.2.2, bantamtrain 0.3.0.dev0, python 3.13.18, bantam_git 33cb0cda9faaf47bf4d8f2cccf29bba08f70a327-dirty.

Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 CPUs, 15.7 GB, load average 3.9 at the end.

| arm | steps | train min | real tok/s | train CPU min | peak RSS GB | final train loss | judge loss (95% CI) | vs base |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| hf_peft | 200 | 13.6 | 147 | 50.2 | 4.18 | 2.604 | 2.260 (2.096-2.415) | - |

Arms:

- **hf_peft**: transformers Trainer + peft, dynamic padding, torch AdamW.

Fairness checks: token parity 0 mismatched of 800; same LoRA modules and trainable parameters in every arm: True; supervised tokens trained: hf_peft 61972; bantam finetune leaves out the windows that do not fill a last optimizer step; bantam's own final eval minus the judge on the same examples: -.

Commands:

```
# hf_peft
/usr/local/bin/python3 /home/user/AboxRebuildcode/bantam/experiments/race_finetune.py _hf --out /home/user/AboxRebuildcode/bantam/results/race_hf_ftz --arm hf_peft
```

Greedy answers:

*What is the capital of France?*
- hf_peft: 'The capital of France is Paris.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO'

*Give me three tips for staying healthy.*
- hf_peft: '1. Eat a balanced diet. 2. Get enough sleep. 3. Exercise. 4. Get enough sleep. 5. Eat a balanced diet. 6. Get enough sleep. 7. Exercise. 8. Eat a balanced diet. 9. Get enough sleep. 10'

*Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.*
- hf_peft: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
