# Monitor Report — 2026-08-03 04:40:01

## Processes
- **A (merged_all)**: **NOT RUNNING** :red_circle:
- **B (two-stage)**: **NOT RUNNING** :red_circle:

## GPU
```
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 580.173
```

## Output Health

### A (merged_all)
- Size: 2187 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
  Epoch 4/4 loss=0.0102 time=518s
  Enc+NL=0.8200 NL-only=0.8400 Gap=-0.0200 Time=2549s

MEAN: 0.8341 ±0.0112
DONE
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

*Auto: 04:40:01*
