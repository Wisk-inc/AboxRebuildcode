# Fine-tuning race

HuggingFaceTB/SmolLM2-135M, databricks-dolly-15k: 800 train / 200 judge examples (split seed 1234), seq 512, micro-batch 2 x accum 2, 1.0 epoch, LR 0.0002, LoRA r=16 alpha=32 dropout=0.05, 4 threads, seed 1234.

Versions: torch 2.15.0+cpu, transformers 5.24.0, peft 0.22.0, accelerate 1.16.0, datasets 5.2.0, tokenizers 0.23.3, safetensors 0.8.0, numpy 2.5.3, psutil 7.2.2, bantamtrain 0.3.0.dev0, python 3.13.18, bantam_git 33cb0cda9faaf47bf4d8f2cccf29bba08f70a327-dirty.

Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 4 CPUs, 15.7 GB, load average 4.0 at the end.

| arm | steps | train min | real tok/s | train CPU min | peak RSS GB | final train loss | judge loss (95% CI) | vs base |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| base | - | - | - | - | - | - | 2.645 (2.496-2.796) | +0.000 |
| hf_peft | 200 | 13.2 | 151 | 47.8 | 4.17 | 2.604 | 2.260 (2.096-2.415) | -0.385 |
| bantam_adamw | 58 | 9.9 | 199 | 35.2 | 5.16 | 2.556 | 2.316 (2.156-2.467) | -0.329 |
| bantam | 58 | 9.3 | 213 | 33.6 | 5.28 | 2.696 | 2.535 (2.384-2.684) | -0.110 |
| bantam_nf4 | 58 | 9.2 | 216 | 34.0 | 4.61 | 2.850 | 2.694 (2.544-2.845) | +0.049 |
| bantam_adamw_steps | 235 | 9.7 | 206 | 36.0 | 2.57 | 2.420 | 2.259 (2.096-2.413) | -0.386 |
| bantam_muon_steps | 235 | 9.3 | 216 | 35.4 | 2.56 | 2.464 | 2.303 (2.144-2.455) | -0.341 |
| bantam_muon_steps_lr1e-3 | 235 | 9.0 | 223 | 34.5 | 2.57 | 2.431 | 2.241 (2.076-2.400) | -0.404 |
| bantam_nf4_adamw_steps | 235 | 9.4 | 212 | 36.5 | 2.12 | 2.538 | 2.368 (2.207-2.527) | -0.276 |

Arms:

- **base**: base model, no training (reference). Judge loss minus hf_peft: +0.385 (paired 95% CI +0.326 to +0.466).
- **hf_peft**: transformers Trainer + peft, dynamic padding, torch AdamW.
- **bantam_adamw**: bantam finetune, torch AdamW (fp32 state), packing. Effective: optimizers [['AdamW', 4884480]], state_bits 32, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.056 (paired 95% CI +0.046 to +0.068).
- **bantam**: bantam finetune defaults: Muon LoRA, 8-bit state, packing, chunked CE. Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.275 (paired 95% CI +0.231 to +0.338).
- **bantam_nf4**: bantam defaults on an NF4 base (QLoRA via load_quantized). Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant nf4 (embeddings 8). Judge loss minus hf_peft: +0.434 (paired 95% CI +0.390 to +0.492). Its adapters on the float base: judge loss 2.563.
- **bantam_adamw_steps**: bantam_adamw with 1 window per optimizer step (235 steps, close to hf_peft's 200). Effective: optimizers [['AdamW', 4884480]], state_bits 32, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: -0.001 (paired 95% CI -0.005 to +0.003).
- **bantam_muon_steps**: Muon LoRA (8-bit state) with 1 window per step. Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: +0.044 (paired 95% CI +0.035 to +0.054).
- **bantam_muon_steps_lr1e-3**: Muon LoRA with 1 window per step, LR 1e-3. Effective: optimizers [['AdamWLowMem', 0], ['Muon', 4884480]], state_bits 8, pack bfd, chunked CE True, quant 0. Judge loss minus hf_peft: -0.019 (paired 95% CI -0.027 to -0.012).
- **bantam_nf4_adamw_steps**: QLoRA NF4 base, torch AdamW, 1 window per step. Effective: optimizers [['AdamW', 4884480]], state_bits 32, pack bfd, chunked CE True, quant nf4 (embeddings 8). Judge loss minus hf_peft: +0.109 (paired 95% CI +0.100 to +0.119). Its adapters on the float base: judge loss 2.302.

Fairness checks: token parity 0 mismatched of 800; same LoRA modules and trainable parameters in every arm: True; supervised tokens trained: hf_peft 61972, bantam_adamw 61257 (in 232 of 235 packed windows), bantam 61257 (in 232 of 235 packed windows), bantam_nf4 61257 (in 232 of 235 packed windows), bantam_adamw_steps 61972 (in 235 of 235 packed windows), bantam_muon_steps 61972 (in 235 of 235 packed windows), bantam_muon_steps_lr1e-3 61972 (in 235 of 235 packed windows), bantam_nf4_adamw_steps 61972 (in 235 of 235 packed windows); bantam finetune leaves out the windows that do not fill a last optimizer step; bantam's own final eval minus the judge on the same examples: bantam_adamw +7.2e-08, bantam -1.0e-07, bantam_nf4 -1.5e-07, bantam_adamw_steps +2.0e-08, bantam_muon_steps +1.1e-07, bantam_muon_steps_lr1e-3 +1.9e-08, bantam_nf4_adamw_steps -1.3e-08.

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
# bantam_adamw_steps
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 1 --accum 1 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim adamw --state_bits 32 --pack bfd --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam_adamw_steps
# bantam_muon_steps
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 1 --accum 1 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim muon --pack bfd --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam_muon_steps
# bantam_muon_steps_lr1e-3
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 1 --accum 1 --epochs 1.0 --lr 0.001 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim muon --pack bfd --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam_muon_steps_lr1e-3
# bantam_nf4_adamw_steps
/usr/local/bin/python3 -m bantamtrain finetune --model HuggingFaceTB/SmolLM2-135M --jsonl /home/user/AboxRebuildcode/bantam/results/race/data/train.jsonl --eval_jsonl /home/user/AboxRebuildcode/bantam/results/race/data/bantam_eval.jsonl --seq 512 --batch 1 --accum 1 --epochs 1.0 --lr 0.0002 --sched cosine --warmup 0.03 --wd 0 --lora 16 --lora_alpha 32 --lora_dropout 0.05 --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj --seed 1234 --threads 4 --eval_every 0 --save_peft 1 --optim adamw --state_bits 32 --pack bfd --quant nf4 --low_ram 1 --out /home/user/AboxRebuildcode/bantam/results/race/runs/bantam_nf4_adamw_steps
```

Greedy answers:

*What is the capital of France?*
- base: 'What is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?\nWhat is the capital of France?'
- hf_peft: 'The capital of France is Paris.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO\n\nFrance is a country in Western Europe.LEGATO'
- bantam_adamw: 'The capital of France is Paris.\n\n|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•|•'
- bantam: 'What is the capital of France?\n\n## What is the capital of France?\n\nThe capital of France is Paris.\n\n## What is the capital of France?\n\nThe capital of France is Paris.\n\n## What is the capital of France?\n\nThe capital of France is Paris.'
- bantam_nf4: 'What is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat is the capital of France?\n\nWhat'
- bantam_adamw_steps: 'The capital of France is Paris. It is located in the south of France. It is the largest city in France. It is the capital of the department of department of department of department of department of department of department of department of department of department of department of department of dep'
- bantam_muon_steps: 'The capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe capital of France is Paris.\n\nThe'
- bantam_muon_steps_lr1e-3: 'The capital of France is Paris. It is the largest city in France and the capital of the country. It is located in the south of France. It is the largest city in France and the capital of the country. It is the largest city in France and the capital of the country. It is the largest city in'
- bantam_nf4_adamw_steps: "The capital of France is Paris. It is located in the north of France. It is the largest city in France and the capital of the country. It is the largest city in the European Union. It is also the largest city in the European Union's European Economic Area. It is also the capital of the European Unio"

*Give me three tips for staying healthy.*
- base: 'Give me three tips for staying healthy.\n\nThe first tip is to eat a balanced diet. This means eating a variety of foods, including fruits, vegetables, whole grains, and lean proteins.\n\nThe second tip is to get enough sleep. Getting enough sleep can help you feel better and help you stay healthy'
- hf_peft: '1. Eat a balanced diet. 2. Get enough sleep. 3. Exercise. 4. Get enough sleep. 5. Eat a balanced diet. 6. Get enough sleep. 7. Exercise. 8. Eat a balanced diet. 9. Get enough sleep. 10'
- bantam_adamw: '1. Eat a balanced diet.\n2. Get enough sleep.\n3. Exercise.\n4. Avoid smoking.\n5. Drink plenty of water.\n6. Avoid alcohol.\n7. Avoid stress.\n8. Avoid junk food.\n9. Avoid caffeine.\n10. Avoid poor'
- bantam: 'Give me three tips for staying healthy.\n\n### 1. Eat a balanced diet.\n\nA balanced diet is one that includes a variety of foods. It should include a variety of fruits, vegetables, grains, and proteins. It should also include a variety of dairy products, such as milk and cheese.'
- bantam_nf4: 'The first tip is to eat a balanced diet. Eat a variety of foods, including fruits, vegetables, whole grains, and lean proteins. Eat a variety of foods, including fruits, vegetables, whole grains, and lean proteins. Eat a variety of foods, including fruits, vegetables, whole grains, and lean proteins'
- bantam_adamw_steps: '1. Eat a balanced diet. 2. Get enough sleep. 3. Exercise. 4. Get enough sleep. 5. Eat a balanced diet. 6. Get enough sleep. 7. Exercise. 8. Eat a balanced diet. 9. Get enough sleep. 10'
- bantam_muon_steps: '1. Eat a balanced diet.\n2. Get enough sleep.\n3. Exercise.\n4. Avoid smoking.\n5. Drink alcohol in moderation.\n6. Avoid stress.\n7. Avoid junk food.\n8. Avoid caffeine.\n9. Avoid junk food.\n10. Avoid'
- bantam_muon_steps_lr1e-3: '1. Eat a balanced diet.  Eat a variety of foods to ensure you get all the nutrients you need.  Eat a variety of foods to ensure you get all the nutrients you need.  Eat a variety of foods to ensure you get all the nutrients you need.  Eat a variety of foods to ensure you'
- bantam_nf4_adamw_steps: '1. Eat a balanced diet that includes plenty of fruits and vegetables. 2. Exercise regularly. 3. Get enough sleep. 4. Drink plenty of water. 5. Avoid smoking and drinking. 6. Avoid excess sugar and caffeine. 7. Avoid foods that are high in saturated fat and'

*Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.*
- base: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical '
- hf_peft: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
- bantam_adamw: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\nLEGATO\nThe Great Wall of China is a series of fortifications built across the historical northern borders of a'
- bantam: 'Summarize in one sentence: The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\n## 1.1.1.1.1.1.1.1.1.1.1'
- bantam_nf4: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
- bantam_adamw_steps: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
- bantam_muon_steps: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.LEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO\nLEGATO'
- bantam_muon_steps_lr1e-3: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe. It is also known as the Great Wall of China, the Great Wall of China, the Great Wall of China, the Great Wall '
- bantam_nf4_adamw_steps: 'The Great Wall of China is a series of fortifications built across the historical northern borders of ancient Chinese states to protect them against nomadic groups from the Eurasian Steppe.\n\nThe Great Wall of China is a series of fortifications built across the historical northern borders of ancient'
