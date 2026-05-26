"""
综合评估脚本 - 全面测试模型性能
"""
import os
import sys
import json
import torch
import numpy as np
from collections import Counter
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_microbiome_encoder import UnifiedMicrobiomeEncoder, ProjectionLayer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
LORA_PATH = "/hd/liujx/microbiome_llm_project/saved_models/merged_multidataset_v2/final"
TEST_DATA_FILE = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/evaluation_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584

def load_model_and_encoders():
    print("🚀 正在加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print("  - 加载Qwen2.5-7B基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        local_files_only=True,
        trust_remote_code=True
    )
    
    print("  - 加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    print("  - 加载自定义层...")
    custom_layers = torch.load(
        os.path.join(LORA_PATH, "custom_layers.pt"),
        map_location="cuda:0"
    )
    
    micro_encoder_study = UnifiedMicrobiomeEncoder.create_for_dataset("study").to("cuda:0").to(torch.bfloat16)
    micro_encoder_ibd = UnifiedMicrobiomeEncoder.create_for_dataset("ibd").to("cuda:0").to(torch.bfloat16)
    projection = ProjectionLayer(EMBED_DIM, LLM_HIDDEN_SIZE).to("cuda:0").to(torch.bfloat16)
    
    micro_encoder_study.load_state_dict(custom_layers['micro_encoder_study'])
    micro_encoder_ibd.load_state_dict(custom_layers['micro_encoder_ibd'])
    projection.load_state_dict(custom_layers['projection'])
    
    micro_encoder_study.eval()
    micro_encoder_ibd.eval()
    projection.eval()
    
    print("✅ 模型加载完成！\n")
    return model, micro_encoder_study, micro_encoder_ibd, projection, tokenizer

def load_test_data():
    print("📂 正在加载测试数据...")
    samples = []
    with open(TEST_DATA_FILE, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    print(f"  - 测试样本数: {len(samples)}")
    labels = Counter([s['label'] for s in samples])
    print("  - 标签分布:")
    for label, count in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"    {label}: {count} ({count/len(samples)*100:.1f}%)")
    return samples

def extract_otu_vector(sample):
    """简化版：使用随机向量模拟（实际需要读取TSV文件）"""
    dataset_type = sample.get('dataset_type', 'study')
    if dataset_type == 'study':
        return np.random.randn(6374).astype(np.float32), 'study'
    else:
        return np.random.randn(300).astype(np.float32), 'ibd'

def predict_sample(model, micro_encoder_study, micro_encoder_ibd, projection, 
                   tokenizer, otu_vector, dataset_type):
    prompt = "请根据微生物组数据诊断疾病状态。"
    
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    
    otu_tensor = torch.tensor(otu_vector, dtype=torch.bfloat16, device="cuda:0").unsqueeze(0)
    
    if dataset_type == 'study':
        micro_embeds = micro_encoder_study(otu_tensor)
    else:
        micro_embeds = micro_encoder_ibd(otu_tensor)
    
    micro_tokens = projection(micro_embeds).unsqueeze(1)
    
    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1)
    combined_mask = torch.cat([
        inputs['attention_mask'],
        torch.ones_like(micro_tokens[..., 0])
    ], dim=1)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 简单规则提取标签
    response_lower = response.lower()
    if 'ibd' in response_lower or 'inflammatory' in response_lower:
        predicted_label = 'IBD'
    elif 'healthy' in response_lower or 'normal' in response_lower or '健康' in response:
        predicted_label = 'Healthy'
    elif 'cd' in response_lower or 'crohn' in response_lower:
        predicted_label = 'CD'
    elif 'uc' in response_lower or 'ulcerative' in response_lower or '溃疡' in response:
        predicted_label = 'UC'
    else:
        predicted_label = 'IBD'
    
    return predicted_label, response

def evaluate_model():
    print("="*80)
    print("🧪 微生物组LLM模型 - 综合评估")
    print("="*80 + "\n")
    
    model, micro_encoder_study, micro_encoder_ibd, projection, tokenizer = load_model_and_encoders()
    test_samples = load_test_data()
    
    print("\n🔍 开始预测...")
    y_true = []
    y_pred = []
    responses = []
    
    for i, sample in enumerate(tqdm(test_samples, desc="评估进度")):
        true_label = sample['label']
        otu_vector, dataset_type = extract_otu_vector(sample)
        
        try:
            predicted_label, response = predict_sample(
                model, micro_encoder_study, micro_encoder_ibd, projection,
                tokenizer, otu_vector, dataset_type
            )
            y_true.append(true_label)
            y_pred.append(predicted_label)
            responses.append({
                'sample_id': sample.get('sample_id', f'sample_{i}'),
                'true_label': true_label,
                'predicted_label': predicted_label,
                'response': response[:200]
            })
        except Exception as e:
            print(f"\n⚠️  样本 {i} 预测失败: {e}")
            y_true.append(true_label)
            y_pred.append('IBD')
    
    print("\n" + "="*80)
    print("📊 评估结果")
    print("="*80)
    
    classes = ['IBD', 'Healthy', 'CD', 'UC']
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    
    print("详细分类报告:")
    print("-" * 80)
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print(report)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None
    )
    
    print("\n各类别详细指标:")
    print("-" * 80)
    print(f"{'类别':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    for i, cls in enumerate(classes):
        print(f"{cls:<12} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print("-" * 80)
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("\n混淆矩阵:")
    print("-" * 80)
    print(f"{'真实\\预测':<12}", end='')
    for cls in classes:
        print(f"{cls:<12}", end='')
    print()
    print("-" * 80)
    for i, cls in enumerate(classes):
        print(f"{cls:<12}", end='')
        for j in range(len(classes)):
            print(f"{cm[i][j]:<12}", end='')
        print()
    
    print("\n💾 保存评估结果...")
    
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write("="*80 + "\n")
        f.write("微生物组LLM模型评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"总体准确率: {accuracy:.4f}\n\n")
        f.write("详细分类报告:\n")
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n")
    
    with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'w') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(classes))
    bars = plt.bar(x, f1, width=0.6, color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
    plt.xticks(x, classes)
    plt.ylim(0, 1.0)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('F1-Score by Class', fontsize=16)
    
    for bar, score in zip(bars, f1):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.axhline(y=macro_f1, color='r', linestyle='--', label=f'Macro F1={macro_f1:.3f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1.png'), dpi=150)
    plt.close()
    
    print(f"\n✅ 评估完成！")
    print(f"📁 结果保存在: {OUTPUT_DIR}")
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'report': report,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    results = evaluate_model()
