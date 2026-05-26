"""Site-specific configuration: disable flash_attn to prevent import errors."""
import os
os.environ.pop("TORCH_FLASH_ATTN_ENABLED", None)  # Remove env that might trigger flash_attn

import sys
# Remove flash_attn from importable modules by blocking its import
if "flash_attn" in sys.modules:
    del sys.modules["flash_attn"]

import builtins
_original_import = builtins.__import__

def _blocked_import(name, *args, **kwargs):
    if "flash_attn" in name:
        raise ImportError(f"flash_attn is disabled: {name}")
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _blocked_import
