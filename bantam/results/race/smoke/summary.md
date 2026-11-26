# Fine-tuning race (smoke)

HuggingFaceTB/SmolLM2-135M, databricks-dolly-15k: 32 train / 16 judge examples (split seed 1234), seq 512, micro-batch 2 x accum 2, 1.0 epoch, LR 0.0002, LoRA r=16 alpha=32 dropout=0.05, 1 threads, seed 1234.

Versions: torch 2.15.0+cpu, transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, datasets 5.2.0, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.3, psutil 7.2.2, bantamtrain 0.3.0.dev0, python 3.13.18, bantam_git bbc473cc15419b777f6f374cedce3e9290bb37ea.

Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 CPUs, 15.7 GB, load average 2.8 at the end.

| arm | steps | train min | real tok/s | train CPU min | peak RSS GB | final train loss | judge loss (95% CI) | vs base | est. full train min |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| base | - | - | - | - | - | - | 2.561 (2.159-2.939) | +0.000 | - |
| hf_peft | 8 | 1.0 | 76 | 1.0 | 3.99 | 2.628 | 2.527 (2.131-2.892) | -0.033 | 26 |
| bantam_adamw | 2 | 0.8 | 82 | 0.8 | 5.26 | 3.156 | 2.554 (2.154-2.928) | -0.007 | 24 |
| bantam | 2 | 0.7 | 89 | 0.7 | 5.23 | 3.161 | 2.559 (2.158-2.937) | -0.001 | 22 |
| bantam_nf4 | 2 | 0.7 | 83 | 0.7 | 4.61 | 3.348 | 2.768 (2.368-3.145) | +0.207 | 24 |

Arms:

- **base**: base model, no training (reference). Judge loss minus hf_peft: +0.033 (paired 95% CI +0.025 to +0.055).
- **hf_peft**: transformers Trainer + peft, dynamic padding, torch AdamW.
- **bantam_adamw**: bantam finetune, torch AdamW (fp32 state), packing. Effective: optimizers [['AdamW', 4884480]], state_bits 32, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.026 (paired 95% CI +0.020 to +0.044).
- **bantam**: bantam finetune defaults: Muon LoRA, 8-bit state, packing, chunked CE. Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.032 (paired 95% CI +0.024 to +0.053).
- **bantam_nf4**: bantam defaults on an NF4 base (QLoRA via load_quantized). Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant nf4 (embeddings 8). Judge loss minus hf_peft: +0.241 (paired 95% CI +0.207 to +0.281). Its adapters on the float base: judge loss 2.560.

Fairness checks: token parity 0 mismatched of 32; same LoRA modules and trainable parameters in every arm: True; supervised tokens trained: hf_peft 2303, bantam_adamw 1734 (in 8 of 10 packed windows), bantam 1734 (in 8 of 10 packed windows), bantam_nf4 1734 (in 8 of 10 packed windows); bantam finetune leaves out the windows that do not fill a last optimizer step; bantam's own final eval minus the judge on the same examples: bantam_adamw -1.8e-07, bantam +1.6e-07, bantam_nf4 +8.3e-08.

Commands:

```
# hf_peft
/usr/local/bin/python3 /home/user/wt2/race/bantam/experiments/race_finetune.py _hf --out /home/user/wt2/race/bantam/results/race/smoke --arm hf_peft
# bantam_adamw
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/wt2/race/bantam/results/race/smoke/data/train.jsonl --eval_jsonl /home/user/wt2/race/bantam/results/race/smoke/data/bantam_eval.jsonl --seq 512 --batch 2 --accum 2 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 1 --eval_every 0 --save_peft 1 --optim adamw --state_bits 32 --pack bfd --out /home/user/wt2/race/bantam/results/race/smoke/runs/bantam_adamw
# bantam
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/wt2/race/bantam/results/race/smoke/data/train.jsonl --eval_jsonl /home/user/wt2/race/bantam/results/race/smoke/data/bantam_eval.jsonl --seq 512 --batch 2 --accum 2 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 1 --eval_every 0 --save_peft 1 --optim muon --pack bfd --out /home/user/wt2/race/bantam/results/race/smoke/runs/bantam
# bantam_nf4
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/wt2/race/bantam/results/race/smoke/data/train.jsonl --eval_jsonl /home/user/wt2/race/bantam/results/race/smoke/data/bantam_eval.jsonl --seq 512 --batch 2 --accum 2 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 1 --eval_every 0 --save_peft 1 --optim muon --pack bfd --quant nf4 --low_ram 1 --out /home/user/wt2/race/bantam/results/race/smoke/runs/bantam_nf4
```

Greedy answers:

*What is the capital of France?*
- base: 'What is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?'
- hf_peft: 'What is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat'
- bantam_adamw: 'What is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?'
- bantam: 'What is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?'
- bantam_nf4: 'What is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat'

*Give me three tips for staying healthy.*
- base: 'Give me three tips for staying healthy.\n\nThe first tip is to eat a balanced diet. This means eating a variety of foods, including fruits, vegetables, whole grains, and lean proteins.\n\nThe second tip is to get enough sleep. Getting enough sleep can help you feel better and help you stay healthy'
- hf_peft: 'Give me three tips for staying healthy.\n\nThe first tip is to eat a balanced diet. This means eating a variety of foods, including fruits, vegetables, whole grains, and lean proteins.\n\nThe second tip is to get enough sleep. Getting enough sleep can help you feel better and help you stay healthy'
- bantam_adamw: 'Give me three tips for staying healthy.\n\nThe first tip is to eat a balanced diet. This means eating a variety of foods, including fruits, vegetables, whole grains, and lean proteins.\n\nThe second tip is to get enough sleep. Getting enough sleep can help you feel better and help you stay healthy'
- bantam: 'Give me three tips for staying healthy.\n\nThe first tip is to eat a balanced diet. This means eating a variety of foods, including fruits, vegetables, whole grains, and lean proteins.\n\nThe second tip is to get enough sleep. Getting enough sleep can help you feel better and help you stay healthy'
- bantam_nf4: 'The following are some of the most important tips for staying healthy.\n\nGive me three tips for staying healthy.\n\nThe following are some of the most important tips for staying healthy.\n\nGive me three tips for staying healthy.\n\nThe following are some of the most important tips for staying healthy.'

*Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.*
- base: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical '
- hf_peft: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical '
- bantam_adamw: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical '
- bantam: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical '
- bantam_nf4: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
