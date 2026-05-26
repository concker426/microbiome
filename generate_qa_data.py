#!/usr/bin/env python3
"""
Generate free-form QA pairs from microbiome samples.

Strategy:
  For each labeled sample, generate 5-10 diverse QA pairs covering:
    - Status inquiry (健康吗？有什么问题？)
    - Genus-specific (为什么XX升高了？XX降低意味着什么？)
    - Comparative (和健康人比有什么不同？)
    - Risk assessment (有什么风险？)
    - Multi-class (是CD还是UC？)

Output: train_qa.jsonl, test_qa.jsonl (same split as existing NL data)
"""
import json
import os
import random
from collections import Counter

import numpy as np

BASE = "/hd/liujx/microbiome_llm_project"
DATA_DIR = os.path.join(BASE, "data/agp_ftp_processed")
OUTPUT_DIR = os.path.join(BASE, "data/agp_ftp_processed_qa")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Paths ──
TRAIN_DATA = os.path.join(DATA_DIR, "train_set.jsonl")
TEST_DATA = os.path.join(DATA_DIR, "test_set.jsonl")
TRAIN_VECTORS = os.path.join(DATA_DIR, "train_set_vectors.npy")
TEST_VECTORS = os.path.join(DATA_DIR, "test_set_vectors.npy")
TRAIN_SEQUENCES = os.path.join(DATA_DIR, "train_genus_sequences.npy")
TRAIN_MASKS = os.path.join(DATA_DIR, "train_genus_masks.npy")
TEST_SEQUENCES = os.path.join(DATA_DIR, "test_genus_sequences.npy")
TEST_MASKS = os.path.join(DATA_DIR, "test_genus_masks.npy")
GENUS_NAMES_PATH = os.path.join(DATA_DIR, "genus_names.npy")

QA_PER_SAMPLE = 8  # target QA pairs per sample

# ── IBD-associated genera ──
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
IBD_GENERA = IBD_DECREASED | IBD_INCREASED


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def find_top_deviations(sample_vec, baseline, genus_names, top_n=10):
    """Find genera with largest deviation from healthy baseline."""
    n_genus = min(len(genus_names), len(sample_vec))
    deviations = []
    for i in range(n_genus):
        sample_val = float(sample_vec[i])
        baseline_val = float(baseline[i]) if i < len(baseline) else 0.0
        if sample_val < 0.001 and baseline_val < 0.001:
            continue
        delta = sample_val - baseline_val
        if baseline_val > 0.1:
            fold_change = delta / baseline_val
            weighted_delta = abs(delta) * 0.5 + abs(fold_change) * 0.5 * baseline_val * 10
        elif baseline_val > 0.01:
            fold_change = delta / max(baseline_val, 0.001)
            weighted_delta = abs(delta) + abs(fold_change) * 0.01
        else:
            fold_change = delta / max(baseline_val, 0.001)
            weighted_delta = abs(delta) * 2
        status = "elevated" if delta > 0 else "reduced"
        deviations.append((genus_names[i], sample_val, baseline_val, delta, status, weighted_delta))
    deviations.sort(key=lambda x: -abs(x[5]))
    return deviations


def make_genus_str(sample_vec, genus_names, max_n=10):
    """Format top genera as readable string."""
    sorted_idx = np.argsort(-sample_vec)
    parts = []
    for idx in sorted_idx:
        if idx < len(genus_names) and sample_vec[idx] > 0.01:
            parts.append(f"{genus_names[idx]} ({sample_vec[idx]:.2f}%)")
    return " → ".join(parts[:max_n])


def status_qa(sample_vec, baseline, genus_names, deviations, label, label_detail):
    """Type 1: Status inquiry questions."""
    is_disease = label == "Disease"
    qas = []

    # Question variants
    q_templates = [
        "这个样本健康吗？",
        "请分析这个样本的肠道菌群状况",
        "这个人的肠道健康如何？",
        "这个样本有问题吗？",
        "请评估该样本的菌群健康状况",
    ]
    random.shuffle(q_templates)

    for q in q_templates[:2]:  # 2 per sample
        if is_disease:
            # Different answer for different question styles
            elevated = [d for d in deviations if d[4] == "elevated"][:3]
            reduced = [d for d in deviations if d[4] == "reduced"][:3]
            parts = [f"诊断结果：{label}"]
            if label_detail:
                parts[0] = f"诊断结果：{label}（{label_detail}）"
            evidence = []
            if elevated:
                evidence.append("促炎/升高菌属：" + ", ".join(f"{d[0]}（{d[1]:.2f}%）" for d in elevated))
            if reduced:
                evidence.append("抗炎/降低菌属：" + ", ".join(f"{d[0]}（{d[1]:.2f}%）" for d in reduced))
            if evidence:
                parts.append("主要异常：" + "；".join(evidence))
            ibd = [d for d in deviations if d[0] in IBD_GENERA]
            if ibd:
                parts.append("部分异常菌属与IBD相关，提示可能存在肠道炎症。")
            else:
                parts.append("该样本菌群结构存在显著异常，需进一步临床评估。")
            a = "\n".join(parts)
        else:
            if "问题" in q or "评估" in q:
                a = f"诊断结果：Healthy。该样本菌群结构在正常范围内，主要菌属构成与健康基线一致，未发现显著异常。"
            else:
                a = f"该样本整体健康，菌群结构正常。"
        qas.append((q, a))

    return qas


def genus_specific_qa(sample_vec, baseline, genus_names, deviations, label):
    """Type 2: Questions about specific genera."""
    qas = []
    # Pick top 3 most deviated genera
    top_genus = deviations[:3]
    for d in top_genus:
        name, sample_val, baseline_val, delta, status, _ = d
        change_word = "升高" if status == "elevated" else "降低"
        direction = "升高" if status == "elevated" else "降低"
        is_ibd = name in IBD_GENERA

        # Multiple question styles per genus
        q_styles = [
            f"为什么{name}{direction}了？",
            f"{name}{direction}有什么影响？",
            f"{name}的水平正常吗？",
            f"请解释{name}的变化",
            f"{name}{direction}意味着什么？",
        ]
        random.shuffle(q_styles)
        q = q_styles[0]

        # Build answer
        ibd_note = ""
        if is_ibd:
            if name in IBD_INCREASED:
                ibd_note = f"。{name}属于促炎菌属，其升高与肠道炎症正相关"
            else:
                ibd_note = f"。{name}属于抗炎菌属，其降低提示抗炎能力减弱"

        a = (
            f"{name}相对丰度为{sample_val:.2f}%，"
            f"健康基线为{baseline_val:.2f}%，"
            f"{direction}了{abs(delta):.2f}%"
            f"{ibd_note}。"
        )
        qas.append((q, a))

    return qas


def comparative_qa(sample_vec, baseline, genus_names, deviations, label):
    """Type 3: Comparative questions."""
    qas = []
    is_disease = label == "Disease"

    q_templates = [
        "这个样本和健康人比有什么不同？",
        "哪些菌属最异常？",
        "最大的问题是什么？",
        "与健康基线相比，这个样本有什么特点？",
    ]
    random.shuffle(q_templates)

    for q in q_templates[:2]:
        if is_disease:
            elevated = [d for d in deviations if d[4] == "elevated"]
            reduced = [d for d in deviations if d[4] == "reduced"]
            parts = ["与健康基线相比，该样本存在以下差异："]
            if elevated:
                parts.append(f"• {len(elevated)}个菌属升高：{', '.join(d[0] for d in elevated[:5])}")
            if reduced:
                parts.append(f"• {len(reduced)}个菌属降低：{', '.join(d[0] for d in reduced[:5])}")
            a = "\n".join(parts)
        else:
            a = "该样本与健康基线无明显差异，主要菌属构成均在正常范围内。"
        qas.append((q, a))

    return qas


def risk_qa(sample_vec, baseline, genus_names, deviations, label, label_detail):
    """Type 4: Risk assessment (Disease only)."""
    qas = []
    if label != "Disease":
        return qas

    q_templates = [
        "有什么健康风险？",
        "需要关注什么？",
        "这个结果意味着什么？",
        "是否需要进一步检查？",
    ]
    random.shuffle(q_templates)

    for q in q_templates[:2]:
        ibd = [d for d in deviations if d[0] in IBD_GENERA]
        ibd_elevated = [d for d in ibd if d[4] == "elevated"]
        ibd_reduced = [d for d in ibd if d[4] == "reduced"]

        parts = ["该样本菌群异常提示以下风险："]
        if ibd_elevated:
            parts.append(f"• 促炎菌属升高（{', '.join(d[0] for d in ibd_elevated[:3])}），提示炎症风险增加")
        if ibd_reduced:
            parts.append(f"• 抗炎菌属降低（{', '.join(d[0] for d in ibd_reduced[:3])}），抗炎能力减弱")
        if label_detail:
            parts.append(f"• 该样本诊断为{label_detail}，需专科随访")
        parts.append("建议结合临床症状综合判断。")
        a = "\n".join(parts)
        qas.append((q, a))

    return qas


def multiclass_qa(sample_vec, baseline, genus_names, deviations, label, label_detail):
    """Type 5: Multi-class questions (for CD/UC samples)."""
    qas = []
    if label != "Disease" or not label_detail:
        return qas

    q_templates = [
        "是什么类型的肠道疾病？",
        "具体是CD还是UC？",
        f"诊断为{label_detail}的依据是什么？",
    ]
    random.shuffle(q_templates)

    for q in q_templates[:2]:
        ibd_elevated = [d for d in deviations if d[0] in IBD_INCREASED and d[4] == "elevated"]
        ibd_reduced = [d for d in deviations if d[0] in IBD_DECREASED and d[4] == "reduced"]

        parts = [f"诊断为{label_detail}。"]
        evidence = []
        if ibd_elevated:
            evidence.append(f"促炎菌属升高：{', '.join(d[0] for d in ibd_elevated[:3])}")
        if ibd_reduced:
            evidence.append(f"抗炎菌属降低：{', '.join(d[0] for d in ibd_reduced[:3])}")
        if evidence:
            parts.append("菌群特征：" + "；".join(evidence))
        a = "\n".join(parts)
        qas.append((q, a))

    return qas


def prescription_qa(sample_vec, baseline, genus_names, deviations, label):
    """Type 6: What interventions might help (Disease only)."""
    qas = []
    if label != "Disease":
        return qas

    q_templates = [
        "如何改善这个菌群？",
        "有什么饮食建议？",
    ]
    random.shuffle(q_templates)

    for q in q_templates[:1]:
        reduced = [d for d in deviations if d[4] == "reduced" and d[0] in IBD_DECREASED]
        if reduced:
            a = (f"该样本抗炎菌属（{'/'.join(d[0] for d in reduced[:3])}）降低，"
                 "建议增加膳食纤维摄入，促进有益菌生长。具体方案需结合临床。")
        else:
            a = "改善方案需结合临床症状综合制定，建议咨询专科医生。"
        qas.append((q, a))

    return qas


def generate_qa_for_sample(sample_vec, baseline, genus_names,
                            label, label_detail, sample_id):
    """Generate all QA pairs for one sample."""
    deviations = find_top_deviations(sample_vec, baseline, genus_names)

    all_qa = []

    # Type 1: Status (2)
    all_qa.extend(status_qa(sample_vec, baseline, genus_names, deviations, label, label_detail))

    # Type 2: Genus-specific (3)
    all_qa.extend(genus_specific_qa(sample_vec, baseline, genus_names, deviations, label))

    # Type 3: Comparative (1-2)
    all_qa.extend(comparative_qa(sample_vec, baseline, genus_names, deviations, label))

    # Type 4: Risk (1-2, Disease only)
    all_qa.extend(risk_qa(sample_vec, baseline, genus_names, deviations, label, label_detail))

    # Type 5: Multi-class (1-2, CD/UC only)
    all_qa.extend(multiclass_qa(sample_vec, baseline, genus_names, deviations, label, label_detail))

    # Type 6: Prescription (0-1, Disease only)
    all_qa.extend(prescription_qa(sample_vec, baseline, genus_names, deviations, label))

    # Shuffle and limit
    random.shuffle(all_qa)
    all_qa = all_qa[:QA_PER_SAMPLE]

    # Format as messages
    results = []
    for q, a in all_qa:
        results.append({
            "task_type": "free_qa",
            "messages": [
                {"role": "user", "content": f"问：{q}"},
                {"role": "assistant", "content": a},
            ],
            "sample_id": sample_id,
            "label": label,
            "label_detail": label_detail or "",
        })
    return results


def main():
    print("=" * 60)
    print("  Free-form QA Data Generation")
    print("=" * 60)

    # ── 1. Load data ──
    print("\n[1/5] Loading data...")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    train_vectors = np.load(TRAIN_VECTORS)
    test_vectors = np.load(TEST_VECTORS)
    genus_names = np.load(GENUS_NAMES_PATH)

    print(f"  Train: {len(train_data)} samples, vectors: {train_vectors.shape}")
    print(f"  Test: {len(test_data)} samples, vectors: {test_vectors.shape}")
    print(f"  Genus names: {len(genus_names)}")
    label_dist = Counter(d["label"] for d in train_data)
    print(f"  Labels: {dict(label_dist)}")

    # ── 2. Compute healthy baseline ──
    print("\n[2/5] Computing healthy baseline...")
    healthy_indices = [i for i, d in enumerate(train_data) if d["label"] == "Healthy"]
    baseline = train_vectors[healthy_indices].mean(axis=0)
    print(f"  Baseline from {len(healthy_indices)} Healthy samples")

    # ── 3. Generate QA for train ──
    print("\n[3/5] Generating train QA pairs...")
    train_qa = []
    for i, item in enumerate(train_data):
        label_detail = item.get("label_detail", "")
        qa = generate_qa_for_sample(
            train_vectors[i], baseline, genus_names,
            item["label"], label_detail, item["sample_id"],
        )
        train_qa.extend(qa)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(train_data)}...", flush=True)
    print(f"  Generated {len(train_qa)} train QA pairs")

    # ── 4. Generate QA for test ──
    print("\n[4/5] Generating test QA pairs...")
    test_qa = []
    for i, item in enumerate(test_data):
        label_detail = item.get("label_detail", "")
        qa = generate_qa_for_sample(
            test_vectors[i], baseline, genus_names,
            item["label"], label_detail, item["sample_id"],
        )
        test_qa.extend(qa)
    print(f"  Generated {len(test_qa)} test QA pairs")

    # ── 5. Save ──
    print("\n[5/5] Saving...")

    # Save QA data
    train_path = os.path.join(OUTPUT_DIR, "train_qa.jsonl")
    with open(train_path, "w") as f:
        for item in train_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    test_path = os.path.join(OUTPUT_DIR, "test_qa.jsonl")
    with open(test_path, "w") as f:
        for item in test_qa:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Copy genus sequences (same as original, just repeated per QA pair)
    import shutil
    for name in ["train_genus_sequences.npy", "train_genus_masks.npy",
                  "test_genus_sequences.npy", "test_genus_masks.npy"]:
        src = os.path.join(DATA_DIR, name)
        dst = os.path.join(OUTPUT_DIR, name)
        shutil.copy2(src, dst)

    print(f"  Train QA: {train_path} ({len(train_qa)} entries)")
    print(f"  Test QA: {test_path} ({len(test_qa)} entries)")

    # Stats
    train_labels = Counter(item["label"] for item in train_qa)
    print(f"  Train label dist: {dict(train_labels)}")

    # Example
    print(f"\n  Example:")
    print(f"  User: {train_qa[0]['messages'][0]['content']}")
    print(f"  Assistant: {train_qa[0]['messages'][1]['content'][:200]}...")

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
