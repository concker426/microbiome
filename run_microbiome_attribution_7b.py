#!/usr/bin/env python3
"""
Genus attribution training: train model to explain WHY a sample is
classified as Healthy/Disease by listing top-K deviating genera.

Architecture: same as run_microbiome_nl_7b.py — MGMEncoder + Qwen2.5-7B LoRA.
Init: transferred from NL checkpoint.
Differs only by data paths and output dirs.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TORCH_FLASH_ATTN_ENABLED"] = "0"

import fix_flash_attn  # noqa: F401
import accelerate.utils.imports as _acc_imports
_acc_imports.is_deepspeed_available = lambda: False
import accelerate.utils.other as _acc_other
_acc_other.is_deepspeed_available = lambda: False

# Patch run_microbiome_nl_7b's module-level paths before importing
import run_microbiome_nl_7b as nl

BASE = "/hd/liujx/microbiome_llm_project"
DATA_DIR = os.path.join(BASE, "data/agp_ftp_processed_attribution")
nl.TRAIN_DATA = os.path.join(DATA_DIR, "train_attribution.jsonl")
nl.TEST_DATA = os.path.join(DATA_DIR, "test_attribution.jsonl")
nl.TRAIN_SEQUENCES = os.path.join(DATA_DIR, "train_genus_sequences.npy")
nl.TRAIN_MASKS = os.path.join(DATA_DIR, "train_genus_masks.npy")
nl.TEST_SEQUENCES = os.path.join(DATA_DIR, "test_genus_sequences.npy")
nl.TEST_MASKS = os.path.join(DATA_DIR, "test_genus_masks.npy")
nl.OUTPUT_DIR = os.path.join(BASE, "saved_models/procyon_attribution_7b")
nl.EVAL_DIR = os.path.join(BASE, "eval_results_procyon_attribution_7b")
os.makedirs(nl.OUTPUT_DIR, exist_ok=True)
os.makedirs(nl.EVAL_DIR, exist_ok=True)

# Keep MGM encoder pretrain (loaded by nl.main() automatically)
# Don't override PRETRAINED_ENCODER — it already points to mgm_encoder_pretrained/mgm_encoder.pt

# Tighter epochs since this is a more focused task
nl.EPOCHS = 3
nl.LR = 1e-4

if __name__ == "__main__":
    print("=" * 60)
    print("  ProCyon Genus Attribution Training")
    print("  (Explanation: why is this sample Healthy/Disease?)")
    print("=" * 60)
    nl.main()
