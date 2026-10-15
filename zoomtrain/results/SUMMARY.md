# Results summary (auto-generated)

## LLM learning-rate sweep (2.3e13 FLOPs)

| run | optim | lr | steps | val_loss |
|---|---|---|---|---|
| sweep_adamw_lr1e-3 | adamw | 0.001 | 152 | 3.9607 |
| sweep_adamw_lr2e-3 | adamw | 0.002 | 152 | 4.0903 |
| sweep_adamw_lr5e-4 | adamw | 0.0005 | 152 | 4.2252 |
| sweep_muon_lr1.6e-2 | muon | 0.016 | 152 | 3.1759 |
| sweep_muon_lr1e-3 | muon | 0.001 | 152 | 3.9369 |
| sweep_muon_lr2e-3 | muon | 0.002 | 152 | 3.6919 |
| sweep_muon_lr3.2e-2 | muon | 0.032 | 152 | 3.2078 |
| sweep_muon_lr4e-3 | muon | 0.004 | 152 | 3.4402 |
| sweep_muon_lr5e-4 | muon | 0.0005 | 152 | 4.2265 |
| sweep_muon_lr6.4e-2 | muon | 0.064 | 152 | 3.5223 |
| sweep_muon_lr8e-3 | muon | 0.008 | 152 | 3.2302 |


## Weather learning-rate sweep (1.8e13 FLOPs)

| run | optim | lr | steps | val_mse | z500@6h | z500@24h | z500@72h |
|---|---|---|---|---|---|---|---|
| wsweep_adamw_lr1e-3 | adamw | 0.001 | 186 | 0.4143 | 147.0 | 471.7 | 843.6 |
| wsweep_adamw_lr2e-3 | adamw | 0.002 | 186 | 0.3928 | 144.7 | 465.1 | 826.5 |
| wsweep_adamw_lr4e-3 | adamw | 0.004 | 186 | 0.3911 | 145.0 | 467.7 | 839.5 |
| wsweep_adamw_lr5e-4 | adamw | 0.0005 | 186 | 0.4543 | 151.3 | 479.0 | 878.8 |
| wsweep_muon_lr1e-3 | muon | 0.001 | 186 | 0.3686 | 141.6 | 460.6 | 839.6 |
| wsweep_muon_lr2e-3 | muon | 0.002 | 186 | 0.3201 | 134.9 | 450.8 | 831.2 |
| wsweep_muon_lr4e-3 | muon | 0.004 | 186 | 0.2929 | 130.7 | 443.6 | 813.6 |
| wsweep_muon_lr5e-4 | muon | 0.0005 | 186 | 0.4496 | 152.1 | 476.0 | 862.8 |
| wsweep_muon_lr8e-3 | muon | 0.008 | 186 | 0.2672 | 122.4 | 412.4 | 798.0 |
