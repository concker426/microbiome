# Monitor Report — 2026-07-20 08:50:01

## Processes
- **A (merged_all)**: PID=1571156 | CPU=0.0% | MEM=2 MB | Runtime=01:36:57
- **B (two-stage)**: PID=1573686 | CPU=0.0% | MEM=2 MB | Runtime=04:49

## GPU
```
0, NVIDIA L40, 21559 MiB, 46068 MiB, 97 %
1, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
2, NVIDIA L40, 19311 MiB, 46068 MiB, 97 %
```

## Output Health

### A (merged_all)
- Size: 1969 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
  Epoch 4/4 loss=0.0116 time=512s
  Enc+NL=0.8350 NL-only=0.8200 Gap=0.0150 Time=2534s

Seed=456
Loading weights:   0%|          | 0/339 [00:00<?, ?it/s]Loading weights: 100%|██████████| 339/339 [00:00<00:00, 10751.70it/s]
```

### B (two-stage)
- Size: 2893 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
                   ~~~~~~~^^^^^^^^^^^^^^^^^^
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
  Stage 1: Adapter only (Qwen frozen)
    S1 Ep1/3 loss=3.0411
    S1 Ep2/3 loss=1.3289
```

## Results Extracted
```
```

*Auto: 08:50:01*
