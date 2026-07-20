# Monitor Report — 2026-07-20 07:50:01

## Processes
- **A (merged_all)**: PID=1571156 | CPU=0.0% | MEM=2 MB | Runtime=36:57
- **B (two-stage)**: **NOT RUNNING** :red_circle:

## GPU
```
0, NVIDIA L40, 21541 MiB, 46068 MiB, 98 %
1, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
2, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
```

## Output Health

### A (merged_all)
- Size: 1377 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
                   ~~~~~~~^^^^^^^^^^^^^^^^^^
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
  Epoch 1/4 loss=0.1449 time=517s
  Epoch 2/4 loss=0.0107 time=522s
  Epoch 3/4 loss=0.0103 time=898s
```

### B (two-stage)
- Size: 3469 bytes
- **ERROR DETECTED** :red_circle:
```
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
Traceback (most recent call last):
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB. GPU 0 has a total capacity of 44.39 GiB of which 30.00 MiB is free. Process 1571159 has 21.03 GiB memory in use. Including non-PyTorch memory, this process has 23.32 GiB memory in use. Of the allocated memory 22.60 GiB is allocated by PyTorch, and 215.14 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```
- Latest lines:
```
        t_outputs, *args, **kwargs
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    )  # Calls into the C++ engine to run the backward pass
    ^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 MiB. GPU 0 has a total capacity of 44.39 GiB of which 30.00 MiB is free. Process 1571159 has 21.03 GiB memory in use. Including non-PyTorch memory, this process has 23.32 GiB memory in use. Of the allocated memory 22.60 GiB is allocated by PyTorch, and 215.14 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## Results Extracted
```
```

*Auto: 07:50:01*
