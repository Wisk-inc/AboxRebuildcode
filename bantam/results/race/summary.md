# Fine-tuning race

HuggingFaceTB/SmolLM2-135M, databricks-dolly-15k: 800 train / 200 judge examples (split seed 1234), seq 512, micro-batch 2 x accum 2, 1.0 epoch, LR 0.0002, LoRA r=16 alpha=32 dropout=0.05, 4 threads, seed 1234.

Versions: torch 2.15.0+cpu, transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, datasets 5.2.0, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.3, psutil 7.2.2, bantamtrain 0.3.0.dev0, python 3.13.18, bantam_git e4b629938ebf44e6361d27440f3fe1af92a9ea91.

Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 CPUs, 15.7 GB, load average 4.1 at the end.

| arm | steps | train min | real tok/s | train CPU min | peak RSS GB | final train loss | judge loss (95% CI) | vs base |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| base | - | - | - | - | - | - | 2.645 (2.496-2.796) | +0.000 |
| hf_peft | 200 | 13.2 | 151 | 47.8 | 4.17 | 2.604 | 2.260 (2.096-2.415) | -0.385 |
| bantam_adamw | 58 | 9.9 | 199 | 35.2 | 5.16 | 2.556 | 2.316 (2.156-2.467) | -0.329 |
| bantam | 58 | 9.3 | 213 | 33.6 | 5.28 | 2.696 | 2.535 (2.384-2.684) | -0.110 |
| bantam_nf4 | 58 | 9.2 | 216 | 34.0 | 4.61 | 2.850 | 2.694 (2.544-2.845) | +0.049 |

Arms:

- **base**: base model, no training (reference). Judge loss minus hf_peft: +0.385 (paired 95% CI +0.326 to +0.466).
- **hf_peft**: transformers Trainer + peft, dynamic padding, torch AdamW.
- **bantam_adamw**: bantam finetune, torch AdamW (fp32 state), packing. Effective: optimizers [['AdamW', 4884480]], state_bits 32, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.056 (paired 95% CI +0.046 to +0.068).
- **bantam**: bantam finetune defaults: Muon LoRA, 8-bit state, packing, chunked CE. Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.275 (paired 95% CI +0.231 to +0.338).
- **bantam_nf4**: bantam defaults on an NF4 base (QLoRA via load_quantized). Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant nf4 (embeddings 8). Judge loss minus hf_peft: +0.434 (paired 95% CI +0.390 to +0.492). Its adapters on the float base: judge loss 2.563.

Fairness checks: token parity 0 mismatched of 800; same LoRA modules and trainable parameters in every arm: True; supervised tokens trained: hf_peft 61972, bantam_adamw 61257 (in 232 of 235 packed windows), bantam 61257 (in 232 of 235 packed windows), bantam_nf4 61257 (in 232 of 235 packed windows); bantam finetune leaves out the windows that do not fill a last optimizer step; bantam's own final eval minus the judge on the same examples: bantam_adamw +7.2e-08, bantam -1.0e-07, bantam_nf4 -1.5e-07.

Commands:

```
# hf_peft
/usr/local/bin/python3 /home/user/AboxRebuildcode/bantam/experiments/race_finetune.py _hf --out /home/user/AboxRebuildcode/bantam/results/race --arm hf_peft
# bantam_adamw
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 2 --accum 2 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim adamw --state_bits 32 --pack bfd --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam_adamw
# bantam
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 2 --accum 2 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim muon --pack bfd --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam
# bantam_nf4
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 2 --accum 2 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim muon --pack bfd --quant nf4 --low_ram 1 --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam_nf4
```

Greedy answers:

*What is the capital of France?*
- base: 'What is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?'
- hf_peft: 'The capital of France is Paris.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO'
- bantam_adamw: 'The capital of France is Paris.\n\n|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•'
- bantam: 'What is the capital of France?\n\n## What is the capital of France?\n\nThe capital of France is Paris.\n\n## What is the capital of France?\n\nThe capital of France is Paris.\n\n## What is the capital of France?\n\nThe capital of France is Paris.'
- bantam_nf4: 'What is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat'

*Give me three tips for staying healthy.*
- base: 'Give me three tips for staying healthy.\n\nThe first tip is to eat a balanced diet. This means eating a variety of foods, including fruits, vegetables, whole grains, and lean proteins.\n\nThe second tip is to get enough sleep. Getting enough sleep can help you feel better and help you stay healthy'
- hf_peft: '1. Eat a balanced diet. 2. Get enough sleep. 3. Exercise. 4. Get enough sleep. 5. Eat a balanced diet. 6. Get enough sleep. 7. Exercise. 8. Eat a balanced diet. 9. Get enough sleep. 10'
- bantam_adamw: '1. Eat a balanced diet.\n2. Get enough sleep.\n3. Exercise.\n4. Avoid smoking.\n5. Drink plenty of water.\n6. Avoid alcohol.\n7. Avoid stress.\n8. Avoid junk food.\n9. Avoid caffeine.\n10. Avoid poor'
- bantam: 'Give me three tips for staying healthy.\n\n### 1. Eat a balanced diet.\n\nA balanced diet is one that includes a variety of foods. It should include a variety of fruits, vegetables, grains, and proteins. It should also include a variety of dairy products, such as milk and cheese.'
- bantam_nf4: 'The first tip is to eat a balanced diet. Eat a variety of foods, including fruits, vegetables, whole grains, and lean proteins. Eat a variety of foods, including fruits, vegetables, whole grains, and lean proteins. Eat a variety of foods, including fruits, vegetables, whole grains, and lean proteins'

*Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.*
- base: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical '
- hf_peft: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
- bantam_adamw: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\nLEGATO\nThe Great Wall of China is a series of fortifications built across the historical northern borders of a'
- bantam: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\n## 1.1.1.1.1.1.1.1.1.1.1'
- bantam_nf4: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
