"""
使用真实OTU数据进行模型评估
"""
import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_microbiome_encoder import UnifiedMicrobiomeEncoder, ProjectionLayer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
LORA_PATH = "/hd/liujx/microbiome_llm_project/saved_models/merged_multidataset_v2/final"
TEST_DATA_FILE = "/hd/liujx/microbiome_llm_project/data/test_set.jsonl"
STUDY_COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"
IBD_COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/evaluation_results_real"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584

def load_otu_data():
    """加载真实OTU数据"""
    print("📂 加载OTU计数数据...")
    
    # 加载Study数据
    study_df = pd.read_csv(STUDY_COUNTS_FILE, sep='\t', index_col=0, skiprows=2, low_memory=False)
    if study_df.shape[0] > study_df.shape[1]:
        study_df = study_df.T
    print(f"  - Study数据集: {study_df.shape}")
    
    # 加载IBD数据
    ibd_df = pd.read_csv(IBD_COUNTS_FILE, sep='\t', index_col=0, skiprows=2, low_memory=False)
    if ibd_df.shape[0] > ibd_df.shape[1]:
        ibd_df = ibd_df.T
    print(f"  - IBD数据集: {ibd_df.shape}")
    
    return study_df, ibd_df

def get_otu_vector(sample_id, dataset_type, study_df, ibd_df):
    """获取真实的OTU向量"""
    try:
        if dataset_type == 'study':
            if str(sample_id) in study_df.index:
                vector = study_df.loc[str(sample_id)].values.astype(np.float32)
                # log转换
                vector = np.log1p(vector)
                return vector, 'study'
        else:  # ibd
            if str(sample_id) in ibd_df.index:
                vector = ibd_df.loc[str(sample_id)].values.astype(np.float32)
                vector = np.log1p(vector)
                return vector, 'ibd'
        
        print(f"  ⚠️  样本 {sample_id} 未找到，使用随机向量")
        if dataset_type == 'study':
            return np.random.randn(6374).astype(np.float32), 'study'
        else:
            return np.random.randn(300).astype(np.float32), 'ibd'
    except Exception as e:
        print(f"  ⚠️  获取OTU向量失败: {e}")
        if dataset_type == 'study':
            return np.random.randn(6374).astype(np.float32), 'study'
        else:
            return np.random.randn(300).astype(np.float32), 'ibd'

def load_model_and_encoders():
    print("🚀 正在加载模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("  - 加载Qwen2.5-7B基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        local_files_only=True, trust_remote_code=True
    )
    
    print("  - 加载LoRA适配器...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    
    print("  - 加载自定义层...")
    custom_layers = torch.load(os.path.join(LORA_PATH, "custom_layers.pt"), map_location="cuda:0")
    
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

def predict_sample(model, micro_encoder_study, micro_encoder_ibd, projection, tokenizer, otu_vector, dataset_type):
    prompt = "请根据微生物组数据诊断疾病状态，输出格式：诊断结果: [IBD/Healthy/CD/UC]"
    
    messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]
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
    combined_mask = torch.cat([inputs['attention_mask'], torch.ones_like(micro_tokens[..., 0])], dim=1)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds, attention_mask=combined_mask,
            max_new_tokens=100, temperature=0.7, top_p=0.9,
            repetition_penalty=1.2, no_repeat_ngram_size=3,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 改进的标签提取
    import re
    match = re.search(r'诊断结果[：:]\s*(IBD|Healthy|CD|UC|健康|炎症)', response)
    if match:
        pred = match.group(1)
        if '健康' in pred:
            predicted_label = 'Healthy'
        elif '炎症' in pred or 'IBD' in pred:
            predicted_label = 'IBD'
        elif pred in ['IBD', 'Healthy', 'CD', 'UC']:
            predicted_label = pred
        else:
            predicted_label = 'IBD'
    else:
        # 备用规则
        response_lower = response.lower()
        if 'healthy' in response_lower or '健康' in response or 'normal' in response_lower:
            predicted_label = 'Healthy'
        elif 'cd' in response_lower or 'crohn' in response_lower:
            predicted_label = 'CD'
        elif 'uc' in response_lower or 'ulcerative' in response_lower:
            predicted_label = 'UC'
        elif 'ibd' in response_lower or 'inflammatory' in response_lower:
            predicted_label = 'IBD'
        else:
            predicted_label = 'IBD'
    
    return predicted_label, response

def evaluate_model():
    print("="*80)
    print("🧪 微生物组LLM模型评估（使用真实OTU数据）")
    print("="*80 + "\n")
    
    study_df, ibd_df = load_otu_data()
    model, micro_encoder_study, micro_encoder_ibd, projection, tokenizer = load_model_and_encoders()
    
    print("📂 加载测试数据...")
    test_samples = []
    with open(TEST_DATA_FILE, 'r') as f:
        for line in f:
            test_samples.append(json.loads(line))
    print(f"  - 测试样本数: {len(test_samples)}\n")
    
    print("🔍 开始预测...\n")
    y_true = []
    y_pred = []
    responses = []
    
    for i, sample in enumerate(tqdm(test_samples, desc="评估进度")):
        true_label = sample['label']
        sample_id = sample.get('sample_id', '')
        dataset_type = sample.get('dataset_type', 'study')
        
        otu_vector, actual_type = get_otu_vector(sample_id, dataset_type, study_df, ibd_df)
        
        try:
            predicted_label, response = predict_sample(
                model, micro_encoder_study, micro_encoder_ibd, projection,
                tokenizer, otu_vector, actual_type
            )
            y_true.append(true_label)
            y_pred.append(predicted_label)
            responses.append({
                'sample_id': sample_id,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'response': response[:300]
            })
        except Exception as e:
            print(f"\n⚠️ 样本 {i} 失败: {e}")
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
    
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
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
    
    # 保存结果
    print("\n💾 保存结果...")
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write(f"总体准确率: {accuracy:.4f}\n\n")
        f.write(report)
        f.write(f"\nMacro F1: {macro_f1:.4f}\nWeighted F1: {weighted_f1:.4f}\n")
    
    with open(os.path.join(OUTPUT_DIR, 'predictions.json'), 'w') as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估完成！结果保存在: {OUTPUT_DIR}")
    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1}

if __name__ == "__main__":
    results = evaluate_model()
