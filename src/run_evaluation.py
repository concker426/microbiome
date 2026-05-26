import argparse
import os
import sys
import torch
import json
import pandas as pd
from tqdm import tqdm

# 导入我们之前写的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from microbiome_inference_utils import MicrobiomeDataLoader, MicrobiomeInputConstructor, IBD_COUNTS_FILE
from microbiome_metrics import MicrobiomeEvaluator
from chat_procyon_style import load_model_and_adapters, run_multimodal_inference, NUM_SPECIES

# ================= 配置区域 =================
# 数据集选择：study (6374特征) 或 ibd (300特征)
DATASET_TYPE = "study"  # 默认值，可由命令行覆盖
LABELS_FILE = "/hd/liujx/microbiome_llm_project/data/sample_labels.json"
METADATA_FILE = "/hd/liujx/microbiome_llm_project/data/ibd_metadata.txt"
EVAL_OUTPUT = "/hd/liujx/microbiome_llm_project/results/evaluation_report.json"
USE_FINE_GRAINED_LABELS = True  # 是否使用细粒度标签（CD/UC/IBD/Healthy）



def load_ground_truth(dataset_type, labels_file, metadata_file, use_fine_grained_labels):
    """加载真实标签，支持二分类和四分类"""
    ground_truth = {}

    # 优先使用细粒度标签（如果启用且数据集匹配）
    if use_fine_grained_labels and dataset_type == "ibd" and os.path.exists(metadata_file):
        try:
            df = pd.read_csv(metadata_file, sep='\t', index_col=0)
            for sample_id, row in df.iterrows():
                disease = str(row['Disease']).strip()
                if disease in ['IBD', 'CD', 'UC', 'Healthy']:
                    ground_truth[sample_id] = disease
            print(f"✅ [Labels] 加载细粒度标签: {len(ground_truth)} 个样本")
            return ground_truth
        except Exception as e:
            print(f"⚠️ [Labels] 细粒度标签加载失败: {e}")

    # 回退到二分类标签
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            loaded_labels = json.load(f)

        # 检查标签ID格式是否与当前数据集匹配
        sample_ids_in_labels = list(loaded_labels.keys())
        if dataset_type == "study":
            # Study数据集的样本ID是DNA序列，不是sample_xxx格式
            if any("sample_" in sid for sid in sample_ids_in_labels[:5]):
                print(f"⚠️ [Labels] 二分类标签文件包含IBD数据集格式的ID，不适用于Study数据集")
                return {}
        elif dataset_type == "ibd":
            # IBD数据集的样本ID是sample_xxx格式
            if not any("sample_" in sid for sid in sample_ids_in_labels[:5]):
                print(f"⚠️ [Labels] 二分类标签文件不包含IBD数据集格式的ID")
                return {}

        ground_truth = loaded_labels
        print(f"✅ [Labels] 加载二分类标签: {len(ground_truth)} 个样本")
        return ground_truth

    print("⚠️ 警告: 未找到标签文件，将仅演示推理流程，不计算分类指标。")
    return {}

def main(dataset_type, labels_file, metadata_file, eval_output, use_fine_grained_labels, max_samples=None):
    print("🚀 [Evaluation] 启动微生物组模型标准化评估流程...")
    
    # 1. 初始化组件
    llm, micro_encoder, projection, tokenizer = load_model_and_adapters(dataset_type)
    
    # 根据数据集类型选择数据文件
    if dataset_type == "ibd":
        data_loader = MicrobiomeDataLoader(IBD_COUNTS_FILE, skiprows=1)
        print("📊 [Info] 使用IBD数据集评估 (300特征)")
    else:  # study
        data_loader = MicrobiomeDataLoader()
        print("📊 [Info] 使用Study数据集评估 (6374特征)")
    
    input_constructor = MicrobiomeInputConstructor(tokenizer)
    evaluator = MicrobiomeEvaluator()
    ground_truth = load_ground_truth(dataset_type, labels_file, metadata_file, use_fine_grained_labels)
    
    # 2. 获取所有待测样本 ID
    sample_ids = data_loader.otu_df.index.tolist()
    
    results = []
    print(f"📊 [Info] 开始对 {len(sample_ids)} 个样本进行批量推理...")

    matched_labels = 0
    max_samples = max_samples if max_samples is not None else (50 if len(ground_truth) == 0 else 10)
    max_samples = min(max_samples, len(sample_ids))
    for sid in tqdm(sample_ids[:max_samples], total=max_samples):
        vector = data_loader.get_sample_vector(sid)
        if vector is None:
            continue
        
        # 构造输入
        input_dict = input_constructor.create_diagnosis_input(sid, vector)
        
        # 执行推理
        try:
            response = run_multimodal_inference(llm, micro_encoder, projection, tokenizer, input_dict)
            
            # 解析预测结果（支持多分类）
            pred_label = parse_prediction_label(response)
            
            # 获取真实标签（如果存在）
            true_label = ground_truth.get(sid)
            if true_label is not None:
                matched_labels += 1
                evaluator.update_classification(true_label, response)
            
            results.append({
                "sample_id": sid,
                "prediction": response,
                "predicted_label": pred_label,
                "true_label": true_label or "Unknown"
            })
        except Exception as e:
            print(f"❌ [Error] 样本 {sid} 推理失败: {e}")

    # 3. 计算并保存指标
    metrics = evaluator.compute_metrics() if matched_labels > 0 else {}
    if matched_labels == 0:
        metrics["warning"] = "未找到可匹配的真实标签，已保存推理结果但不计算分类指标。"
    metrics["Labels_Matched"] = matched_labels
    metrics["Perplexity"] = evaluator.compute_perplexity()
    
    print("\n" + "="*30)
    print("📈 [Results] 评估报告:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    print("="*30)
    
    # 保存结果
    os.makedirs(os.path.dirname(eval_output), exist_ok=True)

    # 转换numpy类型为Python基本类型以支持JSON序列化
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy类型
            return obj.item()
        else:
            return obj

    serializable_metrics = convert_numpy_types(metrics)

    with open(eval_output, 'w') as f:
        json.dump({"metrics": serializable_metrics, "details": results}, f, indent=4)
    print(f"💾 [Save] 详细报告已保存至: {eval_output}")


def parse_prediction_label(response: str) -> str:
    """智能解析模型输出中的预测标签"""
    response_upper = response.upper()
    
    # 优先匹配更具体的标签
    if "CROHN" in response_upper or " CD" in response_upper or "CD" in response_upper:
        return "CD"
    elif "ULCERATIVE" in response_upper or " UC" in response_upper or "UC" in response_upper:
        return "UC"
    elif "IBD" in response_upper:
        return "IBD"
    elif "HEALTHY" in response_upper or "NORMAL" in response_upper:
        return "Healthy"
    else:
        return "IBD"  # 默认

def parse_args():
    parser = argparse.ArgumentParser(description="微生物组评估脚本")
    parser.add_argument("--dataset_type", choices=["study", "ibd"], default=DATASET_TYPE,
                        help="选择评估数据集：study 或 ibd")
    parser.add_argument("--labels_file", default=LABELS_FILE,
                        help="二分类标签文件路径")
    parser.add_argument("--metadata_file", default=METADATA_FILE,
                        help="IBD数据集元数据文件路径，用于细粒度标签")
    parser.add_argument("--output", default=EVAL_OUTPUT,
                        help="评估报告输出路径")
    parser.add_argument("--no_fine_grained", action="store_true",
                        help="不使用细粒度标签，强制仅使用二分类标签")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="限制推理样本数量，默认根据标签可用性自动选择")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.dataset_type, args.labels_file, args.metadata_file, args.output,
         not args.no_fine_grained, args.max_samples)
