#!/usr/bin/env python3
"""
Generate genus attribution training data.

For each sample, build Q/A pairs that explain WHY the sample was diagnosed
the way it was, by highlighting top-K genera that deviate most from the
healthy baseline.

Output:
  data/agp_ftp_processed_attribution/{train,test}_attribution.jsonl
"""
import os, json, re
import numpy as np

BASE = "/hd/liujx/microbiome_llm_project"
SRC = os.path.join(BASE, "data/agp_ftp_processed")
OUT = os.path.join(BASE, "data/agp_ftp_processed_attribution")
os.makedirs(OUT, exist_ok=True)

TRAIN_JSONL = os.path.join(SRC, "train_set.jsonl")
TEST_JSONL = os.path.join(SRC, "test_set.jsonl")
TRAIN_VEC = os.path.join(SRC, "train_set_vectors.npy")
TEST_VEC = os.path.join(SRC, "test_set_vectors.npy")
GENUS = os.path.join(SRC, "genus_names.npy")

TOP_K = 5
ATTRIB_QUESTIONS = [
    "请解释这个样本被诊断为{label}的关键依据。",
    "为什么判断这个样本属于{label}？请从菌属变化角度说明。",
    "请列出这个样本与健康基线相比变化最大的菌属。",
]


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f]


def fmt_pct(v):
    return f"{v:.2f}%"


def deviation_summary(sample_vec, baseline, genus_names, top_k=5):
    """Top-k genera by |sample - baseline| (in pct points)."""
    diff = sample_vec - baseline
    order = np.argsort(-np.abs(diff))[:top_k]
    parts = []
    for i in order:
        s, b = float(sample_vec[i]), float(baseline[i])
        d = s - b
        sign = "升高" if d > 0 else "降低"
        parts.append(
            f"{genus_names[i]} {fmt_pct(s)}（基线 {fmt_pct(b)}，{sign} {abs(d):.2f}个百分点）"
        )
    return parts


def build_answer(label, deviations, label_detail=""):
    head = (
        "该样本菌群结构整体正常，主要菌属丰度与健康基线接近。"
        if label == "Healthy"
        else "该样本表现出菌群失调特征，与健康基线存在显著偏离。"
    )
    if label == "Disease" and label_detail in ("CD", "UC"):
        head = head[:-1] + f"，并具有{('克罗恩病(CD)' if label_detail=='CD' else '溃疡性结肠炎(UC)')}相关的菌群特征。"
    bullets = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(deviations))
    diag = f"诊断结果：{label}。\n\n关键菌属变化（与健康基线对比）：\n{bullets}\n\n{head}"
    return diag


def gen_attribution_pairs(items, vectors, baseline, genus_names, top_k=5):
    out = []
    for i, item in enumerate(items):
        label = item["label"]
        ld = item.get("label_detail", "")
        dev = deviation_summary(vectors[i], baseline, genus_names, top_k=top_k)
        ans = build_answer(label, dev, ld)
        for q_template in ATTRIB_QUESTIONS:
            q = q_template.format(label=label)
            out.append({
                "task_type": "genus_attribution",
                "sample_id": item["sample_id"],
                "label": label,
                "label_detail": ld,
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": ans},
                ],
            })
    return out


def main():
    print("[1/4] Loading source data...")
    train_items = load_jsonl(TRAIN_JSONL)
    test_items = load_jsonl(TEST_JSONL)
    train_vec = np.load(TRAIN_VEC)
    test_vec = np.load(TEST_VEC)
    genus_names = np.load(GENUS, allow_pickle=True)
    # genus_names may be 1223 while vectors are 1222 — align
    if len(genus_names) > train_vec.shape[1]:
        genus_names = genus_names[:train_vec.shape[1]]
    print(f"  train: {train_vec.shape}, test: {test_vec.shape}, genera: {len(genus_names)}")

    print("[2/4] Computing healthy baseline (median)...")
    h_idx = [i for i, d in enumerate(train_items) if d["label"] == "Healthy"]
    baseline = np.median(train_vec[h_idx], axis=0)
    print(f"  Baseline from {len(h_idx)} healthy samples (using median to avoid skew)")

    print("[3/4] Generating attribution Q/A pairs...")
    train_pairs = gen_attribution_pairs(train_items, train_vec, baseline, genus_names, top_k=TOP_K)
    test_pairs = gen_attribution_pairs(test_items, test_vec, baseline, genus_names, top_k=TOP_K)
    print(f"  Train pairs: {len(train_pairs)} (from {len(train_items)} samples)")
    print(f"  Test pairs:  {len(test_pairs)} (from {len(test_items)} samples)")

    print("[4/4] Saving + copying genus sequences...")
    with open(os.path.join(OUT, "train_attribution.jsonl"), "w") as f:
        for r in train_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "test_attribution.jsonl"), "w") as f:
        for r in test_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import shutil
    for n in ["train_genus_sequences.npy", "train_genus_masks.npy",
              "test_genus_sequences.npy", "test_genus_masks.npy"]:
        shutil.copy2(os.path.join(SRC, n), os.path.join(OUT, n))
    # Sample print
    print("\n--- Sample (Disease) ---")
    print(json.dumps(train_pairs[1], indent=2, ensure_ascii=False)[:1200])
    print(f"\n✅ Done. Files in {OUT}/")


if __name__ == "__main__":
    main()
