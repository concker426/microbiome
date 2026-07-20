# Monitor Report — 2026-07-20 07:11:44

## Processes
- **A (merged_all)**: PID=1569086 | CPU=0.0% | MEM=2 MB | Runtime=01:13:54
- **B (two-stage)**: PID=1569086 | CPU=0.0% | MEM=2 MB | Runtime=01:13:54

## GPU
```
0, NVIDIA L40, 21555 MiB, 46068 MiB, 97 %
1, NVIDIA L40, 19291 MiB, 46068 MiB, 98 %
2, NVIDIA L40, 3 MiB, 46068 MiB, 0 %
```

## Output Health

### A (merged_all)
- Size: 1275 bytes
- **ERROR DETECTED** :red_circle:
```
Traceback (most recent call last):
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```
- Latest lines:
```
           ~~~~~~~~~~~~~^^^^^^
  File "/home/star/miniconda3/lib/python3.13/ctypes/__init__.py", line 390, in __init__
    self._handle = _dlopen(self._name, mode)
                   ~~~~~~~^^^^^^^^^^^^^^^^^^
OSError: libnvJitLink.so.13: cannot open shared object file: No such file or directory
```

### B (two-stage)
- Size: 1404 bytes
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
  Stage 1: Adapter only (Qwen frozen)
    S1 Ep1/3 loss=1.1918
```

## Results Extracted
```
```

*Auto: 07:11:45*
