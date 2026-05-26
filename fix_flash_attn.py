"""Fix import errors: flash_attn + deepspeed."""
import sys
import types

# ── Patch 1: flash_attn ──
import transformers.utils.import_utils as _iu
_iu.is_flash_attn_2_available = lambda: False

import transformers.utils as _utils
_utils.is_flash_attn_2_available = lambda: False
_iu.is_flash_attn_greater_or_equal_2_10 = lambda: False
_utils.is_flash_attn_greater_or_equal_2_10 = lambda: False

print("[fix_imports] Patched flash_attn=False", flush=True)
