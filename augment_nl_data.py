#!/usr/bin/env python3
"""
Augment microbiome NL training data, particularly for Disease class.

Strategy (per google.txt):
  1. Genus sequence perturbation on existing Disease samples (4× augmentation)
  2. Healthy→Disease conversion via targeted IBD-pattern perturbation
  3. Re-run NL enrichment pipeline for all synthetic samples
  4. Balanced sampling config for training

Output: augmented train_nl.jsonl with approx 5:2 Healthy:Disease ratio
"""
import json
import os
import random
from collections import Counter

import numpy as np

DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
NL_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed_nl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed_nl_aug"

MODIFIED_VOCAB = os.path.join(DATA_DIR, "genus_vocab.json")
TRAIN_VECTORS = os.path.join(DATA_DIR, "train_set_vectors.npy")
TRAIN_DATA = os.path.join(DATA_DIR, "train_set.jsonl")
TRAIN_SEQUENCES = os.path.join(DATA_DIR, "train_genus_sequences.npy")
TRAIN_MASKS = os.path.join(DATA_DIR, "train_genus_masks.npy")
TEST_DATA = os.path.join(DATA_DIR, "test_set.jsonl")
TEST_VECTORS = os.path.join(DATA_DIR, "test_set_vectors.npy")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── IBD-associated genera (from generate_nl_microbiome_data.py) ──
IBD_DECREASED = {
    "Faecalibacterium", "Roseburia", "Lachnospira", "Blautia",
    "Coprococcus", "Anaerostipes", "Ruminococcus", "Bifidobacterium",
    "Akkermansia", "Prevotella", "Eubacterium", "Butyricicoccus",
    "Dorea", "Collinsella", "Oscillospira",
}
IBD_INCREASED = {
    "Escherichia", "Fusobacterium", "Veillonella", "Streptococcus",
    "Clostridium", "Enterococcus", "Lactobacillus", "Bacteroides",
    "Eggerthella", "Dialister", "Haemophilus", "Campylobacter",
    "Peptostreptococcus", "Pseudomonas",
}

# ONLY Dropout perturbation — scientifically defensible
# (simulates varying sequencing depth, no prior knowledge injected)
NUM_AUGMENT_PER_DISEASE = 2   # create 2 dropout variants per Disease sample
NUM_AUGMENT_PER_HEALTHY = 2   # create 2 dropout variants per Healthy sample (balance)
NUM_HEALTHY_TO_DISEASE = 0    # DISABLED: injects label knowledge


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def genus_id_to_name(genus_names):
    """Return lookup dict."""
    return {i: name for i, name in enumerate(genus_names)}


def name_to_genus_id(genus_names, name):
    """Find genus ID by name, case-insensitive."""
    name_lower = name.lower()
    for i, n in enumerate(genus_names):
        if n.lower() == name_lower:
            return i
    return None


def perturb_genus_sequence(seq, mask, method="noise"):
    """
    Create a perturbed variant of a genus token sequence.

    Methods:
      - "noise":        Swap adjacent positions with low probability
      - "dropout":      Remove some low-rank (high index) genera
      - "ibd_shift":    Boost IBD_increased, reduce IBD_decreased
      - "shuffle_tail": Shuffle the tail of the sequence
    """
    valid_indices = np.where(mask)[0]
    valid_tokens = seq[valid_indices].copy()
    L = len(valid_tokens)

    if L < 5:
        # Too short to perturb, return original
        return seq.copy(), mask.copy()

    new_tokens = valid_tokens.tolist()

    if method == "noise":
        # Swap adjacent positions with 0.15 probability
        for i in range(L - 1):
            if random.random() < 0.15:
                new_tokens[i], new_tokens[i+1] = new_tokens[i+1], new_tokens[i]

    elif method == "dropout":
        # Remove 1-3 genera from the tail (low-abundance end)
        n_drop = random.randint(1, min(3, L - 3))
        keep_len = L - n_drop
        new_tokens = new_tokens[:keep_len]
        # Truncate seq and mask
        new_seq = np.zeros_like(seq)
        new_mask = np.zeros_like(mask)
        new_seq[:keep_len] = new_tokens
        new_mask[:keep_len] = 1
        return new_seq, new_mask

    elif method == "ibd_shift":
        # Boost IBD_increased genera by duplicating them
        # Reduce IBD_decreased by occasionally replacing with another genus
        for i, gid in enumerate(new_tokens):
            if gid >= len(genus_names):
                continue
            gname = genus_names[gid]
            if gname in IBD_INCREASED and random.random() < 0.3:
                # Boost: insert again nearby
                insert_pos = min(i + 1, len(new_tokens) - 1)
                new_tokens.insert(insert_pos, gid)
                if len(new_tokens) >= len(valid_tokens):
                    break
            elif gname in IBD_DECREASED and random.random() < 0.25:
                # Reduce: replace with a random non-IBD genus
                candidates = [j for j in range(len(genus_names))
                              if j not in IBD_DECREASED_GENUS_IDS
                              and j not in IBD_INCREASED_GENUS_IDS][:50]
                if candidates:
                    new_tokens[i] = random.choice(candidates)

    elif method == "shuffle_tail":
        # Shuffle the last 30% of positions
        shuffle_start = max(3, int(L * 0.7))
        tail = new_tokens[shuffle_start:]
        random.shuffle(tail)
        new_tokens = new_tokens[:shuffle_start] + tail

    elif method == "mix":
        # Combine: noise + shuffle_tail
        for i in range(L - 1):
            if random.random() < 0.1:
                new_tokens[i], new_tokens[i+1] = new_tokens[i+1], new_tokens[i]
        shuffle_start = max(3, int(L * 0.8))
        tail = new_tokens[shuffle_start:]
        random.shuffle(tail)
        new_tokens = new_tokens[:shuffle_start] + tail

    # Build output (pad to original length)
    final_len = min(len(new_tokens), len(valid_indices))
    new_seq = np.zeros_like(seq)
    new_mask = np.zeros_like(mask)
    new_seq[:final_len] = new_tokens[:final_len]
    new_mask[:final_len] = 1
    return new_seq, new_mask


def convert_healthy_to_disease(seq, mask, genus_names):
    """
    Convert a Healthy sample to synthetic Disease by perturbing
    IBD-associated genera in the sequence.
    """
    valid_indices = np.where(mask)[0]
    valid_tokens = seq[valid_indices].copy()
    new_tokens = valid_tokens.tolist()

    # Remove some IBD_DECREASED genera
    new_tokens = [t for t in new_tokens
                  if not (t < len(genus_names)
                          and genus_names[t] in IBD_DECREASED
                          and random.random() < 0.4)]

    # Add some IBD_INCREASED genera if not present
    present_ids = set(new_tokens)
    for gid, gname in enumerate(genus_names):
        if gname in IBD_INCREASED and gid not in present_ids and random.random() < 0.5:
            # Insert at a random position near the end (lower abundance)
            insert_pos = random.randint(max(1, len(new_tokens) - 10), len(new_tokens))
            new_tokens.insert(insert_pos, gid)

    # Truncate/pad
    final_len = min(len(new_tokens), len(valid_indices))
    new_seq = np.zeros_like(seq)
    new_mask = np.zeros_like(mask)
    new_seq[:final_len] = new_tokens[:final_len]
    new_mask[:final_len] = 1
    return new_seq, new_mask


def compute_vectors_from_sequence(seq, mask, genus_names, vocab_size=1222):
    """
    Convert a genus token sequence back to a 1222-dim abundance vector.
    Assigns abundance proportional to rank (first = highest).
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    valid_len = int(mask.sum())
    if valid_len == 0:
        return vec
    tokens = seq[:valid_len]
    # Abundance decays exponentially by rank
    # First genus ~30%, second ~15%, etc.
    abundances = np.exp(-np.arange(valid_len) * 0.15) * 0.3
    abundances = abundances / abundances.sum() * 100  # normalize to sum to 100
    for i, gid in enumerate(tokens):
        if gid < vocab_size:
            vec[gid] = abundances[i]
    return vec


def main():
    print("=" * 60)
    print("  NL data augmentation (Disease upsampling)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──
    print("\n[1/5] Loading data...")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    train_vectors = np.load(TRAIN_VECTORS).astype(np.float32)
    train_sequences = np.load(TRAIN_SEQUENCES)
    train_masks = np.load(TRAIN_MASKS)

    with open(MODIFIED_VOCAB) as f:
        vocab = json.load(f)
    genus_names = vocab["genus_names"][:train_vectors.shape[1]]

    original_train_labels = Counter(d["label"] for d in train_data)
    print(f"  Original train: {dict(original_train_labels)}")
    print(f"  Test: {len(test_data)} samples")

    # Build genus name→id lookups for IBD lists
    global IBD_DECREASED_GENUS_IDS, IBD_INCREASED_GENUS_IDS, genus_names_global
    genus_names_global = genus_names
    IBD_DECREASED_GENUS_IDS = set()
    for name in IBD_DECREASED:
        gid = name_to_genus_id(genus_names, name)
        if gid is not None:
            IBD_DECREASED_GENUS_IDS.add(gid)
    IBD_INCREASED_GENUS_IDS = set()
    for name in IBD_INCREASED:
        gid = name_to_genus_id(genus_names, name)
        if gid is not None:
            IBD_INCREASED_GENUS_IDS.add(gid)
    print(f"  IBD_DECREASED found in vocab: {len(IBD_DECREASED_GENUS_IDS)}/{len(IBD_DECREASED)}")
    print(f"  IBD_INCREASED found in vocab: {len(IBD_INCREASED_GENUS_IDS)}/{len(IBD_INCREASED)}")

    # ── Identify indices ──
    disease_indices = [i for i, d in enumerate(train_data) if d["label"] == "Disease"]
    healthy_indices = [i for i, d in enumerate(train_data) if d["label"] == "Healthy"]
    print(f"\n  Disease samples: {len(disease_indices)}")
    print(f"  Healthy samples: {len(healthy_indices)}")

    # ── Step 1: Perturb Disease samples ──
    print("\n[2/5] Augmenting Disease samples (Dropout only)...")
    methods = ["dropout"]  # Only scientifically valid method
    augmented_disease_entries = []

    for idx in disease_indices:
        orig = train_data[idx]
        seq = train_sequences[idx]
        mask = train_masks[idx]

        for aug_i in range(NUM_AUGMENT_PER_DISEASE):
            new_seq, new_mask = perturb_genus_sequence(seq, mask, method="dropout")
            count_nonzero = int(new_mask.sum())
            if count_nonzero < 3:
                continue  # skip degenerate sequences
            new_vec = compute_vectors_from_sequence(new_seq, new_mask, genus_names)

            entry = {
                "original_idx": idx,
                "label": "Disease",
                "dataset_type": orig["dataset_type"],
                "sample_id": f"{orig['sample_id']}_aug_dropout_{aug_i}",
                "genus_sequence": new_seq.tolist(),
                "genus_mask": new_mask.tolist(),
                "genus_vector": new_vec.tolist(),
                "aug_method": "dropout",
            }
            augmented_disease_entries.append(entry)

    print(f"  Created {len(augmented_disease_entries)} augmented Disease variants")

    # ── Step 1b: Perturb Healthy samples (for class balance) ──
    print("\n[2b/5] Augmenting Healthy samples (Dropout only)...")
    augmented_healthy_entries = []

    for idx in healthy_indices:
        orig = train_data[idx]
        seq = train_sequences[idx]
        mask = train_masks[idx]

        for aug_i in range(NUM_AUGMENT_PER_HEALTHY):
            new_seq, new_mask = perturb_genus_sequence(seq, mask, method="dropout")
            count_nonzero = int(new_mask.sum())
            if count_nonzero < 3:
                continue
            new_vec = compute_vectors_from_sequence(new_seq, new_mask, genus_names)

            entry = {
                "original_idx": idx,
                "label": "Healthy",
                "dataset_type": orig["dataset_type"],
                "sample_id": f"{orig['sample_id']}_aug_dropout_{aug_i}",
                "genus_sequence": new_seq.tolist(),
                "genus_mask": new_mask.tolist(),
                "genus_vector": new_vec.tolist(),
                "aug_method": "dropout",
            }
            augmented_healthy_entries.append(entry)

    print(f"  Created {len(augmented_healthy_entries)} augmented Healthy variants")

    # ── Step 2: Healthy→Disease conversion ──
    print("\n[3/5] Converting Healthy samples to synthetic Disease...")
    # Only use a subset of Healthy samples
    n_to_convert = min(NUM_HEALTHY_TO_DISEASE, len(healthy_indices))
    convert_indices = random.sample(healthy_indices, n_to_convert)
    converted_disease_entries = []

    for idx in convert_indices:
        orig = train_data[idx]
        seq = train_sequences[idx]
        mask = train_masks[idx]

        new_seq, new_mask = convert_healthy_to_disease(seq, mask, genus_names)
        count_nonzero = int(new_mask.sum())
        if count_nonzero < 3:
            continue
        new_vec = compute_vectors_from_sequence(new_seq, new_mask, genus_names)

        entry = {
            "original_idx": idx,
            "label": "Disease",
            "dataset_type": orig["dataset_type"],
            "sample_id": f"{orig['sample_id']}_converted_disease",
            "genus_sequence": new_seq.tolist(),
            "genus_mask": new_mask.tolist(),
            "genus_vector": new_vec.tolist(),
            "aug_method": "healthy_to_disease",
        }
        converted_disease_entries.append(entry)

    print(f"  Created {len(converted_disease_entries)} Healthy→Disease conversions")

    # ── Step 3: Generate NL text for synthetic samples ──
    print("\n[4/5] Generating enriched NL for synthetic Disease samples...")

    # Build healthy baseline from original train vectors
    healthy_vec_indices = [i for i, d in enumerate(train_data) if d["label"] == "Healthy"]
    baseline = train_vectors[healthy_vec_indices].mean(axis=0)
    print(f"  Baseline computed from {len(healthy_vec_indices)} Healthy samples")

    # Import NL generation functions
    import sys
    sys.path.insert(0, "/hd/liujx/microbiome_llm_project")
    from generate_nl_microbiome_data import (
        find_top_deviations, make_analysis_text,
        make_marker_text, make_comparison_text,
    )

    synthetic_nl = []

    all_synthetic = augmented_disease_entries + converted_disease_entries + augmented_healthy_entries
    for idx, entry in enumerate(all_synthetic):
        label = entry["label"]  # use actual label (Disease or Healthy)
        vec = np.array(entry["genus_vector"], dtype=np.float32)

        top_devs = find_top_deviations(vec, baseline, genus_names, top_n=8)
        analysis = make_analysis_text(label, top_devs)
        marker = make_marker_text(label, top_devs)
        comparison = make_comparison_text(label, top_devs)

        diag_statement = f"诊断结果：{label}。"

        # Build 3 task variants
        diag_user = (
            "你是一位专业的肠道微生物分析师，擅长从菌群数据中识别疾病标志物并提供循证分析。\n\n"
            "请分析以下肠道微生物样本，完成两项任务：\n"
            "（1）判断该样本的健康状态（Healthy 或 Disease）\n"
            "（2）基于菌群丰度数据，详细说明你的判断依据，指出哪些菌属偏离了健康基线\n\n"
        )
        # Build genus string from vector
        top_genera = sorted(
            [(genus_names[i], vec[i]) for i in range(len(genus_names)) if vec[i] > 0.5],
            key=lambda x: -x[1],
        )[:8]
        genus_str = "，".join(f"{name} ({val:.1f}%)" for name, val in top_genera)
        diag_user += f"【主要菌属构成】: {genus_str}"
        diag_asst = f"{diag_statement}\n\n{analysis}"

        marker_user = (
            "作为微生物组研究专家，请分析以下肠道微生物样本中的关键菌属标志物。\n"
            "识别哪些菌属的相对丰度显著偏离正常水平，并说明其变化方向（增加/减少）和幅度。\n\n"
            f"【菌群数据】: {genus_str}"
        )
        marker_asst = marker

        comp_user = (
            "以下是一个肠道微生物样本的菌群数据。请将其与健康人群的平均菌群组成进行对比，\n"
            "列出每个主要菌属的丰度差异（包括具体数值和变化倍数），并简要说明这些差异的生物学意义。\n\n"
            f"【菌群数据】: {genus_str}"
        )
        comp_asst = comparison

        for task_type, user_msg, asst_msg in [
            ("diagnosis", diag_user, diag_asst),
            ("marker_analysis", marker_user, marker_asst),
            ("comparison", comp_user, comp_asst),
        ]:
            synthetic_nl.append({
                "task_type": task_type,
                "messages": [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ],
                "dataset_type": entry["dataset_type"],
                "sample_id": entry["sample_id"],
                "label": label,  # use actual label from entry
                "is_synthetic": True,
                "aug_method": entry["aug_method"],
            })

        if (idx + 1) % 200 == 0:
            print(f"  Generated NL: {idx + 1}/{len(all_synthetic)}", flush=True)

    print(f"  Total synthetic NL samples: {len(synthetic_nl)}")

    # ── Step 4: Merge and save ──
    print("\n[5/5] Merging with original data and saving...")

    original_nl = load_jsonl(os.path.join(NL_DIR, "train_nl.jsonl"))
    print(f"  Original NL: {len(original_nl)}")

    # Count original Disease NL samples
    orig_disease_nl = [d for d in original_nl if d["label"] == "Disease"]
    print(f"  Original Disease NL: {len(orig_disease_nl)}")

    # Merge: keep all original + synthetic
    merged = original_nl + synthetic_nl
    merged_labels = Counter(d["label"] for d in merged)
    merged_tasks = Counter(d["task_type"] for d in merged)
    print(f"  Merged train NL: {len(merged)}")
    print(f"    By label: {dict(merged_labels)}")
    print(f"    By task: {dict(merged_tasks)}")

    # Save merged
    train_path = os.path.join(OUTPUT_DIR, "train_nl_aug.jsonl")
    with open(train_path, "w") as f:
        for s in merged:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved: {train_path}")

    # Copy test data unchanged
    import shutil
    test_path = os.path.join(OUTPUT_DIR, "test_nl_aug.jsonl")
    shutil.copy2(os.path.join(NL_DIR, "test_nl.jsonl"), test_path)
    print(f"  Test (copied): {test_path}")

    # ── Also save augmented genus sequences for training ──
    # Build expanded arrays: original + synthetic
    orig_seq = train_sequences  # (2649, 148)
    orig_mask = train_masks     # (2649, 148)

    aug_seqs = []
    aug_masks = []
    for entry in all_synthetic:
        aug_seqs.append(entry["genus_sequence"])
        aug_masks.append(entry["genus_mask"])

    # Find max sequence length
    max_len_in_aug = max(len(s) for s in aug_seqs) if aug_seqs else 0
    max_len_final = max(orig_seq.shape[1], max_len_in_aug)

    # Pad augmented sequences to match dimension
    aug_seq_arr = np.zeros((len(aug_seqs), max_len_final), dtype=np.int32)
    aug_mask_arr = np.zeros((len(aug_masks), max_len_final), dtype=np.int32)
    for i, (s, m) in enumerate(zip(aug_seqs, aug_masks)):
        s_len = len(s)
        aug_seq_arr[i, :s_len] = s
        aug_mask_arr[i, :s_len] = m

    # Pad original if needed
    if max_len_final > orig_seq.shape[1]:
        new_orig_seq = np.zeros((orig_seq.shape[0], max_len_final), dtype=np.int32)
        new_orig_mask = np.zeros((orig_mask.shape[0], max_len_final), dtype=np.int32)
        new_orig_seq[:, :orig_seq.shape[1]] = orig_seq
        new_orig_mask[:, :orig_mask.shape[1]] = orig_mask
        orig_seq = new_orig_seq
        orig_mask = new_orig_mask

    final_seq = np.concatenate([orig_seq, aug_seq_arr], axis=0)
    final_mask = np.concatenate([orig_mask, aug_mask_arr], axis=0)
    print(f"\n  Augmented genus sequences: {final_seq.shape}")

    # Need to also expand the original NL data by 3× (for 3 task types)
    # to match the sequence array
    seq_path = os.path.join(OUTPUT_DIR, "train_genus_sequences_aug.npy")
    mask_path = os.path.join(OUTPUT_DIR, "train_genus_masks_aug.npy")
    np.save(seq_path, final_seq)
    np.save(mask_path, final_mask)
    print(f"  Sequences: {seq_path}")
    print(f"  Masks: {mask_path}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Augmentation complete!")
    print(f"  Original Disease NL: {len(orig_disease_nl)} → {merged_labels.get('Disease', 0)}")
    print(f"  Train NL total: {len(original_nl)} → {len(merged)}")
    print(f"  Healthy:Disease ratio: "
          f"{merged_labels.get('Healthy', 0)}:{merged_labels.get('Disease', 0)} "
          f"({merged_labels.get('Healthy', 0)/max(merged_labels.get('Disease', 1), 1):.1f}:1)")
    print(f"{'='*60}")
    print(f"\nNext step: Run training with:")
    print(f"  DATA_DIR = '{OUTPUT_DIR}'")
    print(f"  TRAIN_DATA = '{train_path}'")
    print(f"  TRAIN_SEQUENCES = '{seq_path}'")
    print(f"  TRAIN_MASKS = '{mask_path}'")


if __name__ == "__main__":
    main()
