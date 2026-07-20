# Monitor Report — 2026-07-20 07:40:01

## Processes
- **A (merged_all)**: PID=1571156 | CPU=0.0% | MEM=2 MB | Runtime=26:57
- **B (two-stage)**: PID=1571669 | CPU=105% | MEM=2111 MB | Runtime=09:17

## GPU
```
0, NVIDIA L40, 40828 MiB, 46068 MiB, 100 %
1, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
2, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
```

## Output Health

### A (merged_all)
- Size: 1343 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
    self._handle = _dlopen(self._name, mode)
                   ~~~~~~~^^^^^^^^^^^^^^^^^^
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
  Epoch 1/4 loss=0.1449 time=517s
  Epoch 2/4 loss=0.0107 time=522s
```

### B (two-stage)
- Size: 3332 bytes
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

*Auto: 07:40:01*
