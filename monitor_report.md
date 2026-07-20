# Monitor Report — 2026-07-20 08:20:01

## Processes
- **A (merged_all)**: PID=1571156 | CPU=0.0% | MEM=2 MB | Runtime=01:06:57
- **B (two-stage)**: **NOT RUNNING** :red_circle:

## GPU
```
0, NVIDIA L40, 21561 MiB, 46068 MiB, 97 %
1, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
2, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
```

## Output Health

### A (merged_all)
- Size: 1690 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```

Seed=123
Loading weights:   0%|          | 0/339 [00:00<?, ?it/s]Loading weights: 100%|██████████| 339/339 [00:00<00:00, 10847.83it/s]
  Epoch 1/4 loss=0.1457 time=511s
  Epoch 2/4 loss=0.0119 time=519s
```

### B (two-stage)
- Size: 4712 bytes
- **ERROR DETECTED** :red_circle:
```
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
Traceback (most recent call last):
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 2 has a total capacity of 44.39 GiB of which 49.31 MiB is free. Including non-PyTorch memory, this process has 44.34 GiB memory in use. Of the allocated memory 43.42 GiB is allocated by PyTorch, and 419.37 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```
- Latest lines:
```
        p, memory_format=torch.preserve_format
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 2 has a total capacity of 44.39 GiB of which 49.31 MiB is free. Including non-PyTorch memory, this process has 44.34 GiB memory in use. Of the allocated memory 43.42 GiB is allocated by PyTorch, and 419.37 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## Results Extracted
```
```

*Auto: 08:20:02*
