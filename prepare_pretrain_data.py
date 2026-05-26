#!/usr/bin/env python3
"""
MGM-style pre-training data preparation.
Converts raw BIOM abundance matrix → genus token sequences for next-genus prediction.

Method (Rank Value Encoding):
  1. Aggregate OTUs to genus level (preserves 1222 genus names)
  2. For each sample: sort genera by abundance descending
  3. Generate token ID sequence: [genus_1_id, genus_2_id, ..., genus_k_id, EOS]
  4. Save as numpy arrays for efficient pre-training

Output:
  pretrain_sequences.npy   — (N, max_seq_len) int32 token IDs, padded with PAD
  pretrain_masks.npy       — (N, max_seq_len) bool, True = valid token
  genus_vocab.json         — genus_id → genus_name mapping
"""
import json, os, sys
from collections import Counter

import biom
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split

# ── Config ──────────────────────────────────────────────────────────
BIOM_PATH = "/hd/liujx/microbiome_llm_project/data/agp_ftp/08-collapsed/100nt/10k/ag-fecal.biom"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
RANDOM_SEED = 42
TEST_SIZE = 0.1            # 10% validation split
MAX_SEQ_LEN = 512          # max genera per sample (≥99th percentile)
MIN_GENERA = 5             # skip samples with fewer than this many genera

# Special token IDs
PAD_ID = 0
EOS_ID = 1
MASK_ID = 2
SPECIAL_TOKENS = 3


def extract_genus(taxonomy_list):
    """Parse taxonomy list to extract genus name."""
    for taxon in taxonomy_list:
        t = taxon.strip()
        if t.startswith("g__") and len(t) > 3:
            return t[3:]
    return "Incertae Sedis"


def main():
    print("=" * 60)
    print("  MGM Pre-training Data Preparation")
    print("  Rank Value Encoding: BIOM → Genus Token Sequences")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load BIOM ─────────────────────────────────────────────
    print("\n[1/4] Loading BIOM table...")
    bt = biom.load_table(BIOM_PATH)
    n_otus, n_samples = bt.shape
    print(f"  BIOM: {n_otus} OTUs × {n_samples} samples")

    # ── 2. Aggregate OTUs → genus level ──────────────────────────
    print("\n[2/4] Aggregating OTUs to genus level...")
    obs_meta = bt.metadata(axis="observation")
    obs_genus = []
    for m in obs_meta:
        obs_genus.append(extract_genus(m["taxonomy"]) if m and "taxonomy" in m else "Incertae Sedis")

    unique_genera = np.array(sorted(set(obs_genus)))
    n_genus = len(unique_genera)
    print(f"  Unique genera: {n_genus}")

    genus_to_idx = {g: i + SPECIAL_TOKENS for i, g in enumerate(unique_genera)}
    idx_to_genus = {i + SPECIAL_TOKENS: g for i, g in enumerate(unique_genera)}
    idx_to_genus[PAD_ID] = "[PAD]"
    idx_to_genus[EOS_ID] = "[EOS]"
    idx_to_genus[MASK_ID] = "[MASK]"

    obs_to_genus = np.array([genus_to_idx[g] for g in obs_genus])
    n_genus_total = n_genus + SPECIAL_TOKENS  # 1225

    # Sparse genus aggregation matrix
    map_data = np.ones(len(obs_to_genus), dtype=np.float64)
    map_rows = obs_to_genus - SPECIAL_TOKENS  # map to 0..n_genus-1 for matrix
    map_cols = np.arange(len(obs_to_genus))
    genus_map = coo_matrix(
        (map_data, (map_rows, map_cols)),
        shape=(n_genus, n_otus),
    ).tocsr()

    genus_counts = (genus_map @ bt.matrix_data).toarray()
    print(f"  Genus abundance matrix: {genus_counts.shape}")

    # Convert to relative abundances (0-100)
    col_sums = genus_counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    genus_abundances = genus_counts / col_sums * 100.0
    del genus_counts, bt  # free memory

    # ── 3. Build token sequences (Rank Value Encoding) ───────────
    print("\n[3/4] Building genus token sequences...")
    all_sequences = []
    genus_token_counts = Counter()
    seq_lens = []

    for sample_idx in range(n_samples):
        abund = genus_abundances[:, sample_idx]

        # Sort non-zero genera by abundance descending
        nonzero = abund > 0
        nz_count = nonzero.sum()
        if nz_count < MIN_GENERA:
            continue

        sorted_idx = np.argsort(-abund[nonzero])
        orig_idx = np.where(nonzero)[0][sorted_idx]

        # Map to token IDs (add SPECIAL_TOKENS offset)
        token_ids = [genus_to_idx[unique_genera[i]] for i in orig_idx]
        token_ids.append(EOS_ID)  # append EOS token

        for tid in token_ids:
            genus_token_counts[tid] += 1

        all_sequences.append((sample_idx, token_ids))
        seq_lens.append(len(token_ids))

    seq_lens = np.array(seq_lens)
    print(f"  Total sequences: {len(all_sequences)}")
    print(f"  Sequence length: mean={seq_lens.mean():.1f}, median={np.median(seq_lens):.1f}, "
          f"min={seq_lens.min()}, max={seq_lens.max()}")
    print(f"  P99 length: {np.percentile(seq_lens, 99):.0f}")

    actual_max_len = min(int(np.percentile(seq_lens, 99)), MAX_SEQ_LEN)
    print(f"  Using max_seq_len: {actual_max_len}")

    # Pad sequences
    n_total = len(all_sequences)
    sequences = np.full((n_total, actual_max_len), PAD_ID, dtype=np.int32)
    masks = np.zeros((n_total, actual_max_len), dtype=bool)

    for i, (_, token_ids) in enumerate(all_sequences):
        truncated = token_ids[:actual_max_len]
        sequences[i, :len(truncated)] = truncated
        masks[i, :len(truncated)] = True

    # ── 4. Train/validation split + save ──────────────────────────
    print("\n[4/4] Splitting and saving...")

    train_seq, val_seq, train_mask, val_mask = train_test_split(
        sequences, masks,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    print(f"  Train: {len(train_seq)} | Validation: {len(val_seq)}")

    # Save
    np.save(os.path.join(OUTPUT_DIR, "pretrain_sequences.npy"), sequences)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_masks.npy"), masks)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_train_sequences.npy"), train_seq)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_train_masks.npy"), train_mask)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_val_sequences.npy"), val_seq)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_val_masks.npy"), val_mask)

    # Vocabulary metadata
    vocab = {
        "n_genera": n_genus,
        "vocab_size": n_genus_total,
        "pad_id": PAD_ID,
        "eos_id": EOS_ID,
        "mask_id": MASK_ID,
        "max_seq_len": actual_max_len,
        "n_train": len(train_seq),
        "n_val": len(val_seq),
        "genus_names": unique_genera.tolist(),
        "special_tokens": {
            "0": "[PAD]",
            "1": "[EOS]",
            "2": "[MASK]",
        },
    }
    with open(os.path.join(OUTPUT_DIR, "genus_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    # Also save genus_names.npy for backward compatibility
    np.save(os.path.join(OUTPUT_DIR, "genus_names.npy"), unique_genera)

    # Quick stats
    print(f"\n  Saved files:")
    for fname in ["pretrain_sequences.npy", "pretrain_masks.npy",
                  "pretrain_train_sequences.npy", "pretrain_train_masks.npy",
                  "pretrain_val_sequences.npy", "pretrain_val_masks.npy",
                  "genus_vocab.json", "genus_names.npy"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            print(f"    {fname}: {size_mb:.2f} MB")

    # Top genera stats
    print(f"\n  Top 20 most common genera in pre-training data:")
    top_genus_ids = [tid for tid, _ in genus_token_counts.most_common(20)]
    for rank, tid in enumerate(top_genus_ids, 1):
        name = idx_to_genus.get(tid, f"UNK_{tid}")
        count = genus_token_counts[tid]
        print(f"    {rank:2d}. {name} ({count:,} occurrences)")

    print(f"\n{'='*60}")
    print(f"  Done! Ready for next-genus pre-training.")
    print(f"  Next: python3 pretrain_mgm_encoder.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
