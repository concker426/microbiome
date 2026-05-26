#!/usr/bin/env python3
"""
Expand QA training data with more diverse question types.

Covers question categories beyond the current "explain single genus" template:
  - Diet/lifestyle inference from microbiome
  - Drug interaction prediction
  - Cross-sample comparison
  - Multi-marker analysis
  - Risk assessment
  - Probiotic recommendation

Run: python3 expand_qa_data.py
Output: data/agp_ftp_processed_qa/train_qa_expanded.jsonl
"""
import json, os, random, re
import numpy as np

BASE = "/hd/liujx/microbiome_llm_project"
QA_DIR = os.path.join(BASE, "data/agp_ftp_processed_qa")
VECTORS = os.path.join(BASE, "data/agp_ftp_processed/train_set_vectors.npy")
GENUS_NAMES = os.path.join(BASE, "data/agp_ftp_processed/genus_names.npy")
TRAIN_JSONL = os.path.join(BASE, "data/agp_ftp_processed/train_set.jsonl")
OUT_PATH = os.path.join(QA_DIR, "train_qa_expanded.jsonl")

random.seed(42)

# ── Question templates ────────────────────────────────────────────────
# Each template gets (genus_str, deviations_summary, label)
DIET_TEMPLATES = [
    "基于该样本的菌群构成【{genus_str}】，推测此人的饮食习惯可能是什么？素食、高纤维、高脂还是混合型？",
    "此人肠道菌群特征为{genus_str}。请分析其可能的膳食结构并给出建议。",
    "根据以下菌群数据，判断该个体的饮食类型（素食/地中海/西式/高蛋白）：{genus_str}",
]

DRUG_TEMPLATES = [
    "该样本菌群为【{genus_str}】。如果此人需要服用广谱抗生素，基于菌群构成分析可能的耐药风险和益生菌补充建议。",
    "某患者菌群数据：{genus_str}。若需使用免疫抑制剂，从菌群角度评估风险。",
    "肠道菌群：{genus_str}。如需服用二甲双胍，请分析菌群可能的响应。",
]

RISK_TEMPLATES = [
    "根据菌群【{genus_str}】，评估此人未来一年患代谢综合征的风险等级（低/中/高），并说明关键菌属依据。",
    "菌群构成：{genus_str}。评估炎症性肠病发病风险，指出高风险和低风险菌属。",
    "基于{genus_str}的菌群特征，评估结直肠癌风险并列出预警菌属。",
]

COMPARE_TEMPLATES = [
    "样本A：{genus_str}。请将其与健康人群平均菌群对比，列出主要偏离并给出健康评分（1-10）。",
    "将该样本的菌群（{genus_str}）与典型IBD患者菌群对比，判断相似度。",
    "对比此样本（{genus_str}）与30-40岁健康女性平均菌群的差异。",
]

PROBIOTIC_TEMPLATES = [
    "基于菌群数据【{genus_str}】，推荐适合此人的益生菌种类，说明补充理由。",
    "菌群分析：{genus_str}。哪些有益菌属需要补充？推荐什么益生菌或食物？",
    "根据{genus_str}，给出个性化的益生菌和益生元补充方案。",
]

MECHANISM_TEMPLATES = [
    "菌群：{genus_str}。{focus_genus} 的异常可能通过什么代谢通路影响宿主健康？",
    "分析{genus_str}中关键菌属异常与宿主免疫的互作机制。",
    "该样本中{deviating}偏离基线。请从菌群-代谢物-宿主通路角度解释可能的致病机制。",
]

ALL_TEMPLATES = {
    "diet_inference": DIET_TEMPLATES,
    "drug_interaction": DRUG_TEMPLATES,
    "risk_assessment": RISK_TEMPLATES,
    "comparison": COMPARE_TEMPLATES,
    "probiotic": PROBIOTIC_TEMPLATES,
    "mechanism": MECHANISM_TEMPLATES,
}


def load_data():
    items = []
    with open(TRAIN_JSONL) as f:
        for line in f:
            items.append(json.loads(line))
    vectors = np.load(VECTORS)
    genus_names = np.load(GENUS_NAMES, allow_pickle=True)
    return items, vectors, genus_names


def genus_str_from_item(item):
    msg = item["messages"][0]["content"]
    m = re.search(r'【主要菌属构成】[:：]\s*(.+?)(?:\n|【|$)', msg, re.DOTALL)
    if m:
        return m.group(1).strip()
    return msg[:300]


def top_deviations(sample_vec, baseline, genus_names, top_k=5):
    """Return list of 'genus_name +direction' strings."""
    diff = sample_vec - baseline
    order = np.argsort(-np.abs(diff))[:top_k]
    result = []
    for i in order:
        direction = "升高" if diff[i] > 0 else "降低"
        result.append(f"{genus_names[i]}{direction}")
    return result


def build_answer(template_type, genus_str, deviations, label, baseline, genus_names):
    """Generate an answer appropriate for the template type."""
    top5 = top_deviations(deviations[0] if deviations.ndim == 2 else deviations,
                           baseline, genus_names, top_k=5)

    answers = {
        "diet_inference": "基于菌群构成分析：该样本富含{}，推测为{}饮食类型。Bacteroides/Firmicutes 比例提示{}。".format(
            genus_str[:60], "高纤维/植物性" if "Prevotella" in genus_str or label == "Healthy" else "高脂/动物蛋白型",
            "膳食结构均衡" if label == "Healthy" else "可能存在膳食失衡"),
        "drug_interaction": "菌群分析显示{}。抗生素使用后需重点关注{}等菌属的恢复，建议{}。".format(
            "菌群结构稳定，耐药风险较低" if label == "Healthy" else "部分有益菌属偏低，耐药风险中等",
            "、".join([d.split("升高")[0].split("降低")[0] for d in top5[:3]]),
            "补充双歧杆菌和乳酸杆菌类益生菌" if label == "Disease" else "维持现有菌群平衡"),
        "risk_assessment": "风险评估：{}。关键依据：{}。".format(
            "低风险" if label == "Healthy" else "中高风险",
            "；".join(top5[:4])),
        "comparison": "与健康基线对比，该样本综合健康评分为{}分（满分10分）。主要偏离：{}。".format(
            "8-9" if label == "Healthy" else "4-6",
            "；".join(top5[:4])),
        "probiotic": "建议补充{}，食用富含{}的食物。".format(
            "乳杆菌和双歧杆菌制剂" if label == "Disease" else "益生元（菊粉、低聚果糖）维持现有菌群",
            "膳食纤维的全谷物和蔬菜" if label == "Disease" else "多样化蔬果"),
        "mechanism": "{}通过短链脂肪酸/胆汁酸代谢通路影响宿主。{}。".format(
            top5[0].split("升高")[0].split("降低")[0] if top5 else "关键菌属",
            "可能通过调节Treg/Th17平衡影响肠道免疫" if "Bacteroides" in genus_str or "Faecali" in genus_str else "可能影响肠道屏障功能和系统性炎症水平"),
    }
    return answers.get(template_type, "分析完成。")


def main():
    items, vectors, genus_names = load_data()
    baseline = np.mean(vectors, axis=0)

    expanded = []
    for i, item in enumerate(items):
        if i % 1000 == 0:
            print(f"  {i}/{len(items)}", flush=True)

        gs = genus_str_from_item(item)
        label = item.get("label", "Healthy")
        sample_vec = vectors[i]
        devs = sample_vec - baseline

        # 2 random template types per sample
        types = random.sample(list(ALL_TEMPLATES.keys()), min(2, len(ALL_TEMPLATES)))
        for ttype in types:
            tmpl = random.choice(ALL_TEMPLATES[ttype])

            # Fill template
            focus_genus = genus_names[np.argmax(np.abs(devs))]
            top_dev_strs = top_deviations(devs, baseline, genus_names, top_k=3)
            deviating = "、".join(top_dev_strs[:3])

            question = tmpl.format(genus_str=gs, focus_genus=focus_genus, deviating=deviating)
            answer = build_answer(ttype, gs, devs, label, baseline, genus_names)

            expanded.append({
                "task_type": f"expanded_{ttype}",
                "sample_id": item.get("sample_id", f"sample_{i}"),
                "label": label,
                "label_detail": item.get("label_detail", "None"),
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ],
            })

    with open(OUT_PATH, "w") as f:
        for rec in expanded:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nExpanded: {len(expanded)} QA pairs ({len(items)} samples × ~2 types)")
    print(f"Saved to {OUT_PATH}")

    # Stats
    from collections import Counter
    type_counts = Counter(r["task_type"] for r in expanded)
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
