# Monitor Report — 2026-07-20 09:00:01

## Processes
- **A (merged_all)**: PID=1571156 | CPU=0.0% | MEM=2 MB | Runtime=01:46:57
- **B (two-stage)**: **NOT RUNNING** :red_circle:

## GPU
```
0, NVIDIA L40, 21579 MiB, 46068 MiB, 97 %
1, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
2, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
```

## Output Health

### A (merged_all)
- Size: 2003 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
  Enc+NL=0.8350 NL-only=0.8200 Gap=0.0150 Time=2534s

Seed=456
Loading weights:   0%|          | 0/339 [00:00<?, ?it/s]Loading weights: 100%|██████████| 339/339 [00:00<00:00, 10751.70it/s]
  Epoch 1/4 loss=0.1498 time=519s
```

### B (two-stage)
- Size: 4894 bytes
- **ERROR DETECTED** :red_circle:
```
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
Traceback (most recent call last):
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 2 has a total capacity of 44.39 GiB of which 109.31 MiB is free. Including non-PyTorch memory, this process has 44.28 GiB memory in use. Of the allocated memory 43.70 GiB is allocated by PyTorch, and 69.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```
- Latest lines:
```
        p, memory_format=torch.preserve_format
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB. GPU 2 has a total capacity of 44.39 GiB of which 109.31 MiB is free. Including non-PyTorch memory, this process has 44.28 GiB memory in use. Of the allocated memory 43.70 GiB is allocated by PyTorch, and 69.68 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## Results Extracted
```
```

*Auto: 09:00:01*
