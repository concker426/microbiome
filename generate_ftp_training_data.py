"""
Merge existing AGP training data with new FTP-sourced genus data.
Output: expanded train/test sets with stratification.

Key design:
- Deduplicates existing JSONL by sample_id (removes oversampling copies)
- Generates Q&A pairs from FTP genus table + metadata in same format
- Merges, splits, re-oversamples for balanced training
"""
import pandas as pd
import numpy as np
import json
import os
import random
from collections import Counter
from sklearn.model_selection import train_test_split

BASE = "/hd/liujx/microbiome_llm_project"
EXISTING_TRAIN = os.path.join(BASE, "data/agp/train_set.jsonl")
EXISTING_TEST = os.path.join(BASE, "data/agp/test_set.jsonl")
FTP_GENUS = os.path.join(BASE, "data/agp_ftp/ftp_genus_table.csv")
FTP_META = os.path.join(BASE, "data/agp_ftp/04-meta/ag-gg-cleaned.txt")
OUTPUT_DIR = os.path.join(BASE, "data/agp_ftp_merged")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TOP_N_GENERA = 15

IBD_DIAG_MAP = {
    "Ulcerative colitis": "UC",
    "Colonic Crohn's Disease": "CD",
    "Ileal Crohn's Disease": "CD",
    "Ileal and Colonic Crohn's Disease": "CD",
    "Microcolitis": "UC",
}


def load_existing_unique():
    """Load existing data, deduplicate by sample_id."""
    samples = []
    seen_ids = set()
    for fn in [EXISTING_TRAIN, EXISTING_TEST]:
        with open(fn) as f:
            for line in f:
                s = json.loads(line)
                sid = s["sample_id"]
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    samples.append(s)
    print(f"Existing unique samples: {len(samples)}")
    labels = Counter(s["label"] for s in samples)
    print(f"  Label distribution: {dict(labels)}")
    return samples


def extract_top_genera(row, genus_columns, top_n=TOP_N_GENERA):
    """Extract top N genera and their relative abundances from a row."""
    genus_data = {}
    for col_name in genus_columns:
        val = row.get(col_name, 0)
        val_f = float(val) if pd.notna(val) and val != "" else 0.0
        if val_f > 0:
            genus_data[col_name] = val_f

    if not genus_data:
        return []

    total = sum(genus_data.values())
    if total == 0:
        return []

    sorted_genera = sorted(genus_data.items(), key=lambda x: -x[1])
    result = []
    for genus_name, rel_ab in sorted_genera[:top_n]:
        rel_ab_norm = rel_ab / total
        if rel_ab_norm >= 0.001:
            result.append((genus_name, rel_ab_norm))
    return result


def make_taxon_description(genera_list):
    """Convert genus list to natural language string."""
    if not genera_list:
        return "（菌群数据不足）"
    parts = [f"{name} ({rel_ab:.2%})" for name, rel_ab in genera_list]
    return "，".join(parts)


def build_qa_samples(df, genus_columns):
    """Generate Q&A pairs for each sample in the dataframe."""
    samples = []
    for idx, (_, row) in enumerate(df.iterrows()):
        genera = extract_top_genera(row, genus_columns)
        if not genera:
            continue

        species_desc = make_taxon_description(genera)
        label = row["label"]
        diagnosis = row.get("diagnosis_detail", "")
        confidence = row.get("label_confidence", "high")

        binary_label = "Disease" if label != "Healthy" else "Healthy"

        user_content = (
            f"你是一位专业的肠道微生物分析师。请分析样本的菌群数据。\n\n"
            f"【主要菌属构成】: {species_desc}\n\n"
            f"请判断该样本的健康状态（Healthy 或 Disease），并简要说明理由。"
        )

        if label == "Healthy":
            assistant_content = (
                f"诊断结果：Healthy。\n\n"
                f"分析理由：该样本菌群构成在正常范围内，各菌属比例均衡，"
                f"未发现明显的菌群失调特征。"
            )
        else:
            assistant_content = (
                f"诊断结果：Disease。\n\n"
                f"分析理由：该样本表现出菌群失调特征，"
                f"其中 {genera[0][0]} 等菌属的丰度变化与肠道炎症相关。"
            )

        samples.append({
            "dataset_type": "agp_ftp",
            "sample_id": row["sample_id"],
            "label": binary_label,
            "label_detail": label,
            "diagnosis": diagnosis,
            "label_confidence": confidence,
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
        })

        if (idx + 1) % 1000 == 0:
            print(f"  Processed: {idx + 1}/{len(df)}")

    return samples


def filter_ftp_samples(meta):
    """Filter metadata to get Healthy and Disease samples with labels."""
    healthy = meta[meta["IBD"] == "I do not have this condition"].copy()
    healthy["label"] = "Healthy"
    healthy["diagnosis_detail"] = ""
    healthy["label_confidence"] = "high"

    # Diagnosed by medical professional
    dx_medical = meta[
        meta["IBD"] == "Diagnosed by a medical professional (doctor, physician assistant)"
    ].copy()
    dx_medical["label"] = dx_medical["IBD_DIAGNOSIS_REFINED"].map(IBD_DIAG_MAP)
    dx_medical["diagnosis_detail"] = dx_medical["IBD_DIAGNOSIS_REFINED"]
    dx_medical["label_confidence"] = "high"
    dx_medical = dx_medical[dx_medical["label"].notna()]

    # Self-diagnosed
    self_dx = meta[meta["IBD"] == "Self-diagnosed"].copy()
    self_dx["label"] = self_dx["IBD_DIAGNOSIS_REFINED"].map(IBD_DIAG_MAP)
    self_dx["diagnosis_detail"] = self_dx["IBD_DIAGNOSIS_REFINED"]
    self_dx["label_confidence"] = "low"
    self_dx = self_dx[self_dx["label"].notna()]

    # Alternative medicine
    alt_dx = meta[
        meta["IBD"] == "Diagnosed by an alternative medicine practitioner"
    ].copy()
    alt_dx["label"] = alt_dx["IBD_DIAGNOSIS_REFINED"].map(IBD_DIAG_MAP)
    alt_dx["diagnosis_detail"] = alt_dx["IBD_DIAGNOSIS_REFINED"]
    alt_dx["label_confidence"] = "low"
    alt_dx = alt_dx[alt_dx["label"].notna()]

    # Unspecified with specific diagnosis
    unspec = meta[
        (meta["IBD"] == "Unspecified") &
        meta["IBD_DIAGNOSIS_REFINED"].isin(IBD_DIAG_MAP.keys())
    ].copy()
    unspec["label"] = unspec["IBD_DIAGNOSIS_REFINED"].map(IBD_DIAG_MAP)
    unspec["diagnosis_detail"] = unspec["IBD_DIAGNOSIS_REFINED"]
    unspec["label_confidence"] = "low"

    result = pd.concat([healthy, dx_medical, self_dx, alt_dx, unspec], ignore_index=True)
    print(f"FTP filtered: {len(result)} total "
          f"(Healthy: {len(healthy)}, Disease: {len(dx_medical) + len(self_dx) + len(alt_dx) + len(unspec)})")
    for grp_name, grp in [("medical", dx_medical), ("self", self_dx),
                           ("alt", alt_dx), ("unspec", unspec)]:
        print(f"  {grp_name}: {len(grp)}")
    return result


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # 1. Load existing unique samples
    print("=" * 60)
    print("Step 1: Load existing unique samples")
    print("=" * 60)
    existing = load_existing_unique()

    # 2. Load FTP data
    print("\n" + "=" * 60)
    print("Step 2: Load and filter FTP data")
    print("=" * 60)
    print("Loading genus table...")
    genus_df = pd.read_csv(FTP_GENUS, dtype={"sample_id": str}, low_memory=False)
    genus_columns = [c for c in genus_df.columns if c != "sample_id"]
    print(f"  Genus table: {genus_df.shape[0]} samples x {len(genus_columns)} genera")

    print("Loading metadata...")
    meta = pd.read_csv(FTP_META, sep="\t", low_memory=False, dtype={"#SampleID": str})

    # Merge genus with metadata
    print("Merging genus + metadata...")
    merged = genus_df.merge(meta, left_on="sample_id", right_on="#SampleID", how="inner")
    print(f"  Merged: {len(merged)} samples")

    # Filter + label
    labeled = filter_ftp_samples(merged)

    # 3. Generate Q&A from FTP
    print("\n" + "=" * 60)
    print("Step 3: Generate FTP Q&A samples")
    print("=" * 60)
    ftp_samples = build_qa_samples(labeled, genus_columns)
    print(f"  Generated: {len(ftp_samples)} FTP samples")
    ftp_labels = Counter(s["label"] for s in ftp_samples)
    print(f"  Label distribution: {dict(ftp_labels)}")

    # 4. Merge existing + FTP
    print("\n" + "=" * 60)
    print("Step 4: Merge datasets")
    print("=" * 60)
    # Check for sample_id overlap
    existing_ids = set(s["sample_id"] for s in existing)
    ftp_ids = set(s["sample_id"] for s in ftp_samples)
    overlap = existing_ids & ftp_ids
    print(f"  Existing unique IDs: {len(existing_ids)}")
    print(f"  FTP unique IDs: {len(ftp_ids)}")
    print(f"  Overlap: {len(overlap)}")

    # Filter out overlapping FTP samples (prefer existing source)
    ftp_new = [s for s in ftp_samples if s["sample_id"] not in existing_ids]
    print(f"  FTP samples after overlap removal: {len(ftp_new)}")

    all_unique = existing + ftp_new
    print(f"  All unique samples: {len(all_unique)}")
    all_labels = Counter(s["label"] for s in all_unique)
    print(f"  Label distribution: {dict(all_labels)}")

    # 5. Split train/test stratified
    print("\n" + "=" * 60)
    print("Step 5: Train/test split (80/20 stratified)")
    print("=" * 60)
    train_data, test_data = train_test_split(
        all_unique, test_size=0.2, random_state=RANDOM_STATE,
        stratify=[s["label"] for s in all_unique]
    )
    print(f"  Train: {len(train_data)}")
    print(f"  Test: {len(test_data)}")
    train_labels = Counter(s["label"] for s in train_data)
    test_labels = Counter(s["label"] for s in test_data)
    print(f"  Train labels: {dict(train_labels)}")
    print(f"  Test labels: {dict(test_labels)}")

    # 6. Oversample training set for balance
    print("\n" + "=" * 60)
    print("Step 6: Oversample training set")
    print("=" * 60)
    max_count = max(train_labels.values())
    grouped = {}
    for s in train_data:
        grouped.setdefault(s["label"], []).append(s)

    balanced_train = []
    for label, samples_list in grouped.items():
        if len(samples_list) == max_count:
            balanced_train.extend(samples_list)
        else:
            times = max_count // len(samples_list)
            remainder = max_count % len(samples_list)
            oversampled = samples_list * times + random.sample(samples_list, remainder)
            balanced_train.extend(oversampled)

    random.shuffle(balanced_train)
    balanced_labels = Counter(s["label"] for s in balanced_train)
    print(f"  Balanced train: {len(balanced_train)}")
    print(f"  Labels: {dict(balanced_labels)}")

    # 7. Save
    print("\n" + "=" * 60)
    print("Step 7: Save datasets")
    print("=" * 60)
    train_path = os.path.join(OUTPUT_DIR, "train_set.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test_set.jsonl")

    with open(train_path, "w") as f:
        for s in balanced_train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(test_path, "w") as f:
        for s in test_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    # Detailed label breakdown
    print(f"\nTrain set -- label_detail breakdown:")
    for k, v in sorted(Counter(s["label_detail"] for s in train_data).items()):
        print(f"  {k}: {v}")
    print(f"\nTest set -- label_detail breakdown:")
    for k, v in sorted(Counter(s["label_detail"] for s in test_data).items()):
        print(f"  {k}: {v}")
    print(f"\nTest set -- label_confidence breakdown:")
    for k, v in sorted(Counter(s.get("label_confidence", "high") for s in test_data).items()):
        print(f"  {k}: {v}")

    print(f"\n✅ Done! Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
