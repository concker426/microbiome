#!/usr/bin/env python3
"""
Prepare a super-blind held-out test set from existing test samples.

Strategy:
  Hold out 200 samples from the existing test set as a completely blind
  final validation set. The model has never seen these during training.
  These are evaluated ONLY once at the very end.

Usage: python3 prepare_new_test_set.py
Output: data/agp_ftp_processed_nl_new_test/
"""
import json
import os
import random
from collections import Counter

import numpy as np

BASE = "/hd/liujx/microbiome_llm_project"
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Input: existing test data
TEST_NL = os.path.join(BASE, "data/agp_ftp_processed_nl/test_nl.jsonl")
TEST_SEQUENCES = os.path.join(BASE, "data/agp_ftp_processed/test_genus_sequences.npy")
TEST_MASKS = os.path.join(BASE, "data/agp_ftp_processed/test_genus_masks.npy")

# Output
OUTPUT_DIR = os.path.join(BASE, "data/agp_ftp_processed_nl_new_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_HOLDOUT = 200  # 200 samples → 600 NL entries


def main():
    print("=" * 60)
    print("  Super-blind Test Set Preparation")
    print(f"  Hold out {N_HOLDOUT} samples from existing test set")
    print("=" * 60)

    # ── 1. Load test data ──
    print("\n[1/3] Loading existing test NL data...")
    with open(TEST_NL) as f:
        all_data = [json.loads(line) for line in f]
    print(f"  Total NL entries: {len(all_data)}")

    # Group by sample_id
    samples = {}
    for item in all_data:
        sid = item["sample_id"]
        if sid not in samples:
            samples[sid] = []
        samples[sid].append(item)

    print(f"  Unique samples: {len(samples)}")
    label_counts = Counter(
        items[0]["label"] for items in samples.values()
    )
    print(f"  Labels: {dict(label_counts)}")

    # ── 2. Stratified holdout ──
    print(f"\n[2/3] Selecting {N_HOLDOUT} samples for super-blind set...")
    healthy_samples = [
        sid for sid, items in samples.items()
        if items[0]["label"] == "Healthy"
    ]
    disease_samples = [
        sid for sid, items in samples.items()
        if items[0]["label"] == "Disease"
    ]
    random.shuffle(healthy_samples)
    random.shuffle(disease_samples)

    # Proportional holdout: keep same Healthy:Disease ratio
    total_healthy = len(healthy_samples)
    total_disease = len(disease_samples)
    holdout_disease = min(
        total_disease,
        max(1, int(N_HOLDOUT * total_disease / (total_healthy + total_disease)))
    )
    holdout_healthy = min(
        total_healthy,
        N_HOLDOUT - holdout_disease
    )

    held_out_sids = set(
        healthy_samples[:holdout_healthy]
        + disease_samples[:holdout_disease]
    )
    print(f"  Held out: {len(held_out_sids)} samples "
          f"({holdout_healthy} Healthy + {holdout_disease} Disease)")

    # Remaining for standard evaluation
    remaining_sids = set(samples.keys()) - held_out_sids
    print(f"  Remaining for standard eval: {len(remaining_sids)} samples")

    # ── 3. Extract held-out data + genus sequences ──
    print("\n[3/3] Saving held-out data...")
    test_sequences = np.load(TEST_SEQUENCES).astype(np.int64)
    test_masks = np.load(TEST_MASKS)

    # Build sample_id → index mapping (test sequences were deduplicated per task type)
    # The test set has 663 unique samples, and sequences have same length
    # Line up NL entries with genus sequences
    held_out_nl = []
    held_out_seq_indices = set()

    for item in all_data:
        if item["sample_id"] in held_out_sids:
            held_out_nl.append(item)

    # Get unique sample IDs from the held-out set and map to sequence indices
    unique_sids = list(dict.fromkeys(item["sample_id"] for item in held_out_nl))
    sid_to_seq_idx = {sid: i for i, sid in enumerate(unique_sids)}

    # Create genus sequences for held-out samples
    test_sid_order = list(dict.fromkeys(item["sample_id"] for item in all_data))
    sid_to_orig_idx = {sid: i for i, sid in enumerate(test_sid_order)}

    held_out_seqs = []
    held_out_masks = []
    for sid in unique_sids:
        orig_idx = sid_to_orig_idx[sid]
        held_out_seqs.append(test_sequences[orig_idx])
        held_out_masks.append(test_masks[orig_idx])

    held_out_seqs = np.stack(held_out_seqs, axis=0)
    held_out_masks = np.stack(held_out_masks, axis=0)

    # Save
    nl_path = os.path.join(OUTPUT_DIR, "new_test_nl.jsonl")
    with open(nl_path, "w") as f:
        for item in held_out_nl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  NL data saved: {len(held_out_nl)} entries to {nl_path}")

    seq_path = os.path.join(OUTPUT_DIR, "new_test_genus_sequences.npy")
    mask_path = os.path.join(OUTPUT_DIR, "new_test_genus_masks.npy")
    np.save(seq_path, held_out_seqs)
    np.save(mask_path, held_out_masks)
    print(f"  Sequences saved: {held_out_seqs.shape}")
    print(f"  Masks saved: {held_out_masks.shape}")

    # Metadata
    meta = {
        "description": "Super-blind held-out test set",
        "n_samples": len(unique_sids),
        "n_nl_entries": len(held_out_nl),
        "label_distribution": dict(sorted(Counter(
            item["label"] for item in held_out_nl
        ).items())),
        "n_healthy": holdout_healthy,
        "n_disease": holdout_disease,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved")

    print(f"\n{'='*60}")
    print(f"  Super-blind test set ready!")
    print(f"  {meta['n_samples']} samples, {meta['n_nl_entries']} NL entries")
    print(f"  Evaluate with: run_eval_super_blind.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
