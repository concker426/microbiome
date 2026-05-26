#!/usr/bin/env python3
"""
Generate enriched natural language microbiome data with evidence-based explanations.

For each sample, computes deviation from healthy baseline and generates
detailed NL explanations that show LLM's unique advantage:
  - Identifies which genera deviate from healthy norms
  - Provides quantitative evidence (relative abundance vs baseline)
  - Generates clinical-style analysis text

Prompt types:
  1. diagnosis: classify + explain with evidence
  2. marker_analysis: identify key deviant genera
  3. comparison: contrast with healthy baseline

Output: NL-enriched train_set.jsonl / test_set.jsonl
"""
import json
import os
import random
from collections import Counter

import numpy as np

DATA_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/data/agp_ftp_processed_nl"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths
TRAIN_DATA = os.path.join(DATA_DIR, "train_set.jsonl")
TEST_DATA = os.path.join(DATA_DIR, "test_set.jsonl")
TRAIN_VECTORS = os.path.join(DATA_DIR, "train_set_vectors.npy")
TEST_VECTORS = os.path.join(DATA_DIR, "test_set_vectors.npy")
TRAIN_SEQUENCES = os.path.join(DATA_DIR, "train_genus_sequences.npy")
TRAIN_MASKS = os.path.join(DATA_DIR, "train_genus_masks.npy")
TEST_SEQUENCES = os.path.join(DATA_DIR, "test_genus_sequences.npy")
TEST_MASKS = os.path.join(DATA_DIR, "test_genus_masks.npy")
VOCAB_PATH = os.path.join(DATA_DIR, "genus_vocab.json")

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── IBD-associated genera (from literature) ──────────────────────────
# Genera commonly decreased in IBD (protective/anti-inflammatory)
IBD_DECREASED = {
    "Faecalibacterium", "Roseburia", "Lachnospira", "Blautia",
    "Coprococcus", "Anaerostipes", "Ruminococcus", "Bifidobacterium",
    "Akkermansia", "Prevotella", "Eubacterium", "Butyricicoccus",
    "Dorea", "Collinsella", "Oscillospira",
}
# Genera commonly increased in IBD (pro-inflammatory)
IBD_INCREASED = {
    "Escherichia", "Fusobacterium", "Veillonella", "Streptococcus",
    "Clostridium", "Enterococcus", "Lactobacillus", "Bacteroides",
    "Eggerthella", "Dialister", "Haemophilus", "Campylobacter",
    "Peptostreptococcus", "Pseudomonas",
}


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def compute_healthy_baselines(train_data, train_vectors):
    """Compute mean relative abundance per genus across Healthy samples."""
    healthy_indices = [i for i, d in enumerate(train_data) if d["label"] == "Healthy"]
    print(f"  Computing baseline from {len(healthy_indices)} Healthy samples")
    healthy_vecs = train_vectors[healthy_indices]  # (N_healthy, 1222)
    baseline = healthy_vecs.mean(axis=0)  # (1222,)
    return baseline


def find_top_deviations(sample_vec, baseline, genus_names, top_n=8):
    """
    Find genera with largest deviation from healthy baseline.
    Returns list of (genus_name, relative_abundance, baseline_abundance, delta, status)
    where status is 'elevated' or 'reduced'.
    """
    # Filter: only consider genera present (>0) in either sample or baseline
    # to avoid spurious comparisons on very rare genera
    deviations = []
    for i in range(len(genus_names)):
        sample_val = float(sample_vec[i]) if i < len(sample_vec) else 0.0
        baseline_val = float(baseline[i]) if i < len(baseline) else 0.0
        if sample_val < 0.001 and baseline_val < 0.001:
            continue
        delta = sample_val - baseline_val
        # Use fold change for small values, absolute for large
        if baseline_val > 0.1:
            fold_change = delta / baseline_val
            weighted_delta = abs(delta) * 0.5 + abs(fold_change) * 0.5 * baseline_val * 10
        elif baseline_val > 0.01:
            fold_change = delta / max(baseline_val, 0.001)
            weighted_delta = abs(delta) + abs(fold_change) * 0.01
        else:
            weighted_delta = abs(delta) * 2  # emphasis on newly appearing genera

        status = "elevated" if delta > 0 else "reduced"
        deviations.append((genus_names[i], round(sample_val, 2),
                           round(baseline_val, 2), round(delta, 2),
                           status, weighted_delta))

    deviations.sort(key=lambda x: -x[5])  # sort by weighted delta
    return deviations[:top_n]


def format_abundance_percent(val):
    """Format abundance value as percentage string."""
    if val < 0.01:
        return f"{val:.2f}%"
    elif val < 1:
        return f"{val:.2f}%"
    else:
        return f"{val:.1f}%"


def make_analysis_text(label, top_deviations):
    """
    Generate a rich, evidence-based analysis text.
    """
    elevated = [(n, sa, ba) for n, sa, ba, d, s, _ in top_deviations if s == "elevated"]
    reduced = [(n, sa, ba) for n, sa, ba, d, s, _ in top_deviations if s == "reduced"]

    lines = []
    lines.append("分析理由：")

    if label == "Healthy":
        lines.append("该样本的肠道菌群组成在正常范围内，核心菌群结构稳健。")
        if reduced:
            genera_str = "、".join(n for n, _, _ in reduced[:3])
            lines.append(f"  尽管{genera_str}等菌属相对丰度略低于平均水平，但仍处于正常波动范围。")
        if elevated:
            genera_str = "、".join(n for n, _, _ in elevated[:3])
            lines.append(f"  {genera_str}等菌属均维持在健康水平，未出现显著的条件致病菌富集。")
        lines.append("菌群多样性正常，有益菌（Faecalibacterium、Blautia、Roseburia等）比例均衡。")
        lines.append("结论：正常肠道菌群，未见明显异常。")
    else:
        lines.append("该样本表现出与肠道炎症相关的菌群失调特征。")
        if reduced:
            lines.append(f"  保护性菌群减少：")
            for name, sa, ba in reduced[:4]:
                pct_change = ((ba - sa) / max(ba, 0.01)) * 100
                line_parts = [f"    - {name}：相对丰度 {sa:.1f}%（健康均值 {ba:.1f}%，"]
                if pct_change > 50:
                    line_parts.append(f"降低 {pct_change:.0f}%）")
                    line_parts.append("——该菌属减少与肠道炎症密切相关")
                else:
                    line_parts.append(f"低于均值 {pct_change:.0f}%）")
                lines.append("".join(line_parts))
        if elevated:
            lines.append(f"  条件致病菌增加：")
            for name, sa, ba in elevated[:4]:
                if ba > 0:
                    fold = sa / max(ba, 0.01)
                    line_parts = [f"    - {name}：相对丰度 {sa:.1f}%（健康均值 {ba:.1f}%，"]
                    if fold > 3:
                        line_parts.append(f"富集约 {fold:.1f} 倍）")
                        line_parts.append("——该菌属异常增殖提示肠道炎症")
                    else:
                        line_parts.append(f"高于均值）")
                else:
                    line_parts.append(f"    - {name}：相对丰度 {sa:.1f}%（健康样本中罕见）——该菌属异常出现")
                lines.append("".join(line_parts))
        lines.append("综合判断：菌群结构显著偏离健康状态，符合肠道炎症的微生物组特征。")
        if reduced and elevated:
            lines.append(f"    关键指标：有益菌（Faecalibacterium、Roseburia等）减少合并条件致病菌增加，"
                         f"提示菌群失调。")
        elif reduced:
            lines.append(f"    关键指标：保护性菌群整体减少，提示抗炎能力下降。")

    return "\n".join(lines)


def make_marker_text(label, top_deviations):
    """Generate a focused marker analysis identifying key genera."""
    elevated = [(n, sa, ba) for n, sa, ba, d, s, _ in top_deviations if s == "elevated"]
    reduced = [(n, sa, ba) for n, sa, ba, d, s, _ in top_deviations if s == "reduced"]

    lines = ["该样本的关键菌属标志物分析："]
    lines.append("")
    if reduced:
        lines.append(f"【减少的标志物】")
        for name, sa, ba in reduced[:3]:
            lines.append(f"  - {name}：{sa:.1f}%（健康 {ba:.1f}%），减少")
    if elevated:
        lines.append(f"【增加的标志物】")
        for name, sa, ba in elevated[:3]:
            source = f"健康 {ba:.1f}%" if ba > 0.01 else "健康样本中罕见"
            lines.append(f"  - {name}：{sa:.1f}%（{source}），增加")
    lines.append("")
    if label != "Healthy":
        lines.append("这些变化模式与肠道菌群失调一致，提示潜在炎症性肠病。")
    else:
        lines.append("各标志物均在正常波动范围内，未见异常。")
    return "\n".join(lines)


def make_comparison_text(label, top_deviations):
    """Generate a comparison analysis: sample vs healthy baseline."""
    lines = ["与健康基线对比分析："]
    lines.append("")
    for name, sa, ba, delta, status, _ in top_deviations[:6]:
        if status == "elevated" and ba > 0:
            fold = sa / max(ba, 0.01)
            lines.append(f"  • {name}：{sa:.1f}% vs 健康 {ba:.1f}%（{fold:.1f}x，升高）")
        elif status == "elevated":
            lines.append(f"  • {name}：{sa:.1f}% vs 健康 未检出（新出现）")
        elif status == "reduced":
            pct = (1 - sa / max(ba, 0.01)) * 100
            lines.append(f"  • {name}：{sa:.1f}% vs 健康 {ba:.1f}%（降低 {pct:.0f}%）")

    return "\n".join(lines)


def build_nl_sample(original, sample_vec, baseline, genus_names):
    """
    Build enriched NL sample with multiple prompt types.
    """
    label = original["label"]
    top_devs = find_top_deviations(sample_vec, baseline, genus_names, top_n=8)

    # ── Prompt type 1: Diagnosis + Explanation (main task) ──
    analysis = make_analysis_text(label, top_devs)
    if label == "Healthy":
        diagnosis_statement = "诊断结果：Healthy。"
    else:
        diagnosis_statement = "诊断结果：Disease。"

    diag_user = (
        "你是一位专业的肠道微生物分析师，擅长从菌群数据中识别疾病标志物并提供循证分析。\n\n"
        "请分析以下肠道微生物样本，完成两项任务：\n"
        "（1）判断该样本的健康状态（Healthy 或 Disease）\n"
        "（2）基于菌群丰度数据，详细说明你的判断依据，指出哪些菌属偏离了健康基线\n\n"
        f"【主要菌属构成】: {original['messages'][0]['content'].split('【主要菌属构成】: ')[-1] if '【主要菌属构成】: ' in original['messages'][0]['content'] else original['messages'][0]['content']}"
    )
    diag_asst = f"{diagnosis_statement}\n\n{analysis}"

    # ── Prompt type 2: Marker Analysis ──
    marker_user = (
        "作为微生物组研究专家，请分析以下肠道微生物样本中的关键菌属标志物。\n"
        "识别哪些菌属的相对丰度显著偏离正常水平，并说明其变化方向（增加/减少）和幅度。\n\n"
        f"【菌群数据】: {original['messages'][0]['content'].split('【主要菌属构成】: ')[-1] if '【主要菌属构成】: ' in original['messages'][0]['content'] else original['messages'][0]['content']}"
    )
    marker_asst = make_marker_text(label, top_devs)

    # ── Prompt type 3: Comparison ──
    comp_user = (
        "以下是一个肠道微生物样本的菌群数据。请将其与健康人群的平均菌群组成进行对比，\n"
        "列出每个主要菌属的丰度差异（包括具体数值和变化倍数），并简要说明这些差异的生物学意义。\n\n"
        f"【菌群数据】: {original['messages'][0]['content'].split('【主要菌属构成】: ')[-1] if '【主要菌属构成】: ' in original['messages'][0]['content'] else original['messages'][0]['content']}"
    )
    comp_asst = make_comparison_text(label, top_devs)

    # Build all task variants
    variants = []

    # Type 1: Full diagnosis + explanation (primary)
    variants.append({
        "task_type": "diagnosis",
        "messages": [
            {"role": "user", "content": diag_user},
            {"role": "assistant", "content": diag_asst},
        ],
    })

    # Type 2: Marker analysis
    variants.append({
        "task_type": "marker_analysis",
        "messages": [
            {"role": "user", "content": marker_user},
            {"role": "assistant", "content": marker_asst},
        ],
    })

    # Type 3: Comparison
    variants.append({
        "task_type": "comparison",
        "messages": [
            {"role": "user", "content": comp_user},
            {"role": "assistant", "content": comp_asst},
        ],
    })

    return variants


def main():
    print("=" * 60)
    print("   enriched microbiome NL data generation")
    print("   evidence-based explanations + multiple task types")
    print("=" * 60)

    # ── Load data ──
    print("\n[1/4] Loading data...")
    train_data = load_jsonl(TRAIN_DATA)
    test_data = load_jsonl(TEST_DATA)
    train_vectors = np.load(TRAIN_VECTORS).astype(np.float32)
    test_vectors = np.load(TEST_VECTORS).astype(np.float32)
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    genus_names = vocab["genus_names"]
    # Vectors have 1222 dims but genus_names has 1223 entries; use first 1222
    genus_names = genus_names[:train_vectors.shape[1]]
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"  Vectors: train {train_vectors.shape}, test {test_vectors.shape}")
    print(f"  Genera: {len(genus_names)}")

    # ── Compute healthy baseline ──
    print("\n[2/4] Computing healthy abundance baselines...")
    baseline = compute_healthy_baselines(train_data, train_vectors)

    # Print top baseline genera
    top_baseline = sorted(
        [(genus_names[i], baseline[i]) for i in range(len(genus_names))],
        key=lambda x: -x[1],
    )[:10]
    print(f"  Top genera in healthy baseline:")
    for name, val in top_baseline:
        print(f"    {name}: {val:.2f}%")

    # ── Generate enriched samples ──
    print("\n[3/4] Generating enriched NL samples...")

    all_train_nl = []
    for idx, orig in enumerate(train_data):
        variants = build_nl_sample(orig, train_vectors[idx], baseline, genus_names)
        for v in variants:
            v["dataset_type"] = orig["dataset_type"]
            v["sample_id"] = orig["sample_id"]
            v["label"] = orig["label"]
        all_train_nl.extend(variants)
        if (idx + 1) % 500 == 0:
            print(f"  Train: {idx + 1}/{len(train_data)}", flush=True)

    all_test_nl = []
    for idx, orig in enumerate(test_data):
        variants = build_nl_sample(orig, test_vectors[idx], baseline, genus_names)
        for v in variants:
            v["dataset_type"] = orig["dataset_type"]
            v["sample_id"] = orig["sample_id"]
            v["label"] = orig["label"]
        all_test_nl.extend(variants)
        if (idx + 1) % 200 == 0:
            print(f"  Test: {idx + 1}/{len(test_data)}", flush=True)

    task_dist_train = Counter(v["task_type"] for v in all_train_nl)
    task_dist_test = Counter(v["task_type"] for v in all_test_nl)
    print(f"\n  Train NL samples: {len(all_train_nl)} ({dict(task_dist_train)})")
    print(f"  Test NL samples: {len(all_test_nl)} ({dict(task_dist_test)})")

    # ── Save ──
    print("\n[4/4] Saving...")
    train_path = os.path.join(OUTPUT_DIR, "train_nl.jsonl")
    test_path = os.path.join(OUTPUT_DIR, "test_nl.jsonl")

    with open(train_path, "w") as f:
        for s in all_train_nl:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(test_path, "w") as f:
        for s in all_test_nl:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    # ── Show examples ──
    print("\n" + "=" * 60)
    print("  Example enriched samples:")
    print("=" * 60)
    for label_type in ["Healthy", "Disease"]:
        examples = [s for s in all_train_nl if s["label"] == label_type and s["task_type"] == "diagnosis"]
        if examples:
            ex = examples[0]
            print(f"\n--- {label_type} (diagnosis) ---")
            print(f"User:\n{ex['messages'][0]['content'][:300]}...")
            print(f"\nAssistant:\n{ex['messages'][1]['content']}")

    print(f"\n✅ 完成！NL enriched data saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
