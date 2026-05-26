"""
改进的训练脚本 - 使用Transformer编码器，支持多数据集训练
"""
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
from tqdm import tqdm
import json
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from unified_microbiome_encoder import UnifiedMicrobiomeEncoder, ProjectionLayer

# ================= 配置区域 =================
MODEL_PATH = "/hd/gcr/hf_models/Qwen2.5-7B-Instruct"
DATA_FILE = "/hd/liujx/microbiome_llm_project/data/merged_training_data.jsonl"
STUDY_COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/study_16496_counts.tsv"
IBD_COUNTS_FILE = "/hd/liujx/microbiome_llm_project/data/ibd_counts.tsv"
OUTPUT_DIR = "/hd/liujx/microbiome_llm_project/saved_models/merged_multidataset_v1"

EMBED_DIM = 768
LLM_HIDDEN_SIZE = 3584

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

EPOCHS = 5
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

os.makedirs(OUTPUT_DIR, exist_ok=True)


class MultiDatasetChatDataset(Dataset):
    """支持多个数据集的聊天数据集"""
    def __init__(self, jsonl_path, tokenizer, study_counts_path, ibd_counts_path):
        self.tokenizer = tokenizer
        self.samples = []
        
        print("📂 正在加载数据集...")
        
        # 加载Study数据集 (421维)
        print("  - 加载Study计数表 (421维)...")
        self.study_otu_df = pd.read_csv(study_counts_path, sep='\t', index_col=0, skiprows=2, low_memory=False)
        if self.study_otu_df.shape[0] > self.study_otu_df.shape[1]:
            self.study_otu_df = self.study_otu_df.T
        self.study_otu_df = self.study_otu_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.study_otu_df = np.log1p(self.study_otu_df)
        print(f"    ✅ Study: {self.study_otu_df.shape}")
        
        # 加载IBD数据集 (300维)
        print("  - 加载IBD计数表 (300维)...")
        self.ibd_otu_df = pd.read_csv(ibd_counts_path, sep='\t', index_col=0)
        self.ibd_otu_df = self.ibd_otu_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.ibd_otu_df = np.log1p(self.ibd_otu_df)
        print(f"    ✅ IBD: {self.ibd_otu_df.shape}")
        
        # 加载JSONL样本
        print("  - 加载训练样本...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line))
        print(f"✅ 总计加载 {len(self.samples)} 个训练样本")
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item['messages']
        dataset_type = item.get('dataset_type', 'study')
        sample_id = item.get('sample_id', '')
        
        # 应用chat template
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # 根据数据集类型获取对应的OTU向量
        if dataset_type == "study":
            otu_df = self.study_otu_df
            num_species = 421
        else:  # ibd
            otu_df = self.ibd_otu_df
            num_species = 300
        
        # 查找样本向量
        if str(sample_id) in otu_df.index.astype(str):
            otu_vector = otu_df.loc[str(sample_id)].values.astype(np.float32)
        else:
            # 如果找不到，返回零向量
            otu_vector = np.zeros(num_species, dtype=np.float32)
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            max_length=512, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "otu_vector": torch.tensor(otu_vector),
            "labels": inputs.input_ids.squeeze(),
            "dataset_type": dataset_type,
            "sample_id": str(sample_id)
        }


def main():
    print("="*60)
    print("🚀 开始多数据集联合训练 (Transformer Encoder)")
    print("="*60)
    
    # 1. 初始化Tokenizer
    print("\n📝 初始化Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 初始化LLM (使用LoRA)
    print("\n🤖 初始化LLM模型 (Qwen2.5-7B + LoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        local_files_only=True
    )
    
    llm = prepare_model_for_kbit_training(llm)
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    llm = get_peft_model(llm, lora_config)
    print(f"✅ LLM参数量: {sum(p.numel() for p in llm.parameters() if p.requires_grad):,}")
    
    # 3. 初始化两个Microbiome Encoder (分别对应两个数据集)
    print("\n🧬 初始化微生物组编码器 (Transformer架构)...")
    micro_encoder_study = UnifiedMicrobiomeEncoder.create_for_dataset(
        dataset_type="study",
        hidden_size=EMBED_DIM,
        num_layers=2,
        dropout=0.1
    ).to("cuda:0")
    
    micro_encoder_ibd = UnifiedMicrobiomeEncoder.create_for_dataset(
        dataset_type="ibd",
        hidden_size=EMBED_DIM,
        num_layers=2,
        dropout=0.1
    ).to("cuda:0")
    
    print(f"  - Study Encoder: 421 → {EMBED_DIM}")
    print(f"  - IBD Encoder: 300 → {EMBED_DIM}")
    
    # 4. 投影层 (共享)
    projection = ProjectionLayer(input_dim=EMBED_DIM, output_dim=LLM_HIDDEN_SIZE).to("cuda:0")
    print(f"  - Projection: {EMBED_DIM} → {LLM_HIDDEN_SIZE}")
    
    # 5. 加载数据集
    print("\n📊 加载训练数据...")
    dataset = MultiDatasetChatDataset(DATA_FILE, tokenizer, STUDY_COUNTS_FILE, IBD_COUNTS_FILE)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 6. 优化器
    optimizer = torch.optim.AdamW(
        list(llm.parameters()) + 
        list(micro_encoder_study.parameters()) + 
        list(micro_encoder_ibd.parameters()) + 
        list(projection.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 7. 训练循环
    print("\n" + "="*60)
    print("🔥 开始训练...")
    print("="*60)
    
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        llm.train()
        micro_encoder_study.train()
        micro_encoder_ibd.train()
        projection.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        total_loss = 0
        step_count = 0
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to("cuda:0")
            attention_mask = batch['attention_mask'].to("cuda:0")
            otu_vectors = batch['otu_vector'].to("cuda:0").float()
            labels = batch['labels'].to("cuda:0")
            dataset_types = batch['dataset_type']
            
            # 根据数据集类型选择对应的encoder
            micro_embeds_list = []
            for i, ds_type in enumerate(dataset_types):
                if ds_type == "study":
                    embed = micro_encoder_study(otu_vectors[i:i+1])
                else:
                    embed = micro_encoder_ibd(otu_vectors[i:i+1])
                micro_embeds_list.append(embed)
            
            micro_embeds = torch.cat(micro_embeds_list, dim=0)
            micro_tokens = projection(micro_embeds).unsqueeze(1)
            
            # 获取文本embeddings
            inputs_embeds = llm.base_model.model.model.embed_tokens(input_ids)
            
            # 融合
            combined_embeds = torch.cat([inputs_embeds, micro_tokens], dim=1)
            combined_mask = torch.cat([attention_mask, torch.ones_like(micro_tokens[..., 0])], dim=1)
            
            # 构造labels (忽略micro token位置)
            ignore_index = -100
            padded_labels = torch.cat([labels, torch.full((labels.shape[0], 1), ignore_index, device=labels.device)], dim=1)
            
            # 前向传播
            outputs = llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=padded_labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # 反向传播
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 保存checkpoint
                if global_step % 100 == 0:
                    save_dir = os.path.join(OUTPUT_DIR, f"step_{global_step}")
                    llm.save_pretrained(save_dir)
                    torch.save({
                        'micro_encoder_study': micro_encoder_study.state_dict(),
                        'micro_encoder_ibd': micro_encoder_ibd.state_dict(),
                        'projection': projection.state_dict()
                    }, os.path.join(save_dir, "custom_layers.pt"))
                    print(f"\n💾 Checkpoint saved at step {global_step}")
            
            step_count += 1
            pbar.set_postfix({"loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})
        
        avg_loss = total_loss / step_count
        print(f"\n✅ Epoch {epoch+1} 完成 | 平均 Loss: {avg_loss:.4f}")
        
        # 保存每个epoch的模型
        epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch+1}")
        llm.save_pretrained(epoch_dir)
        torch.save({
            'micro_encoder_study': micro_encoder_study.state_dict(),
            'micro_encoder_ibd': micro_encoder_ibd.state_dict(),
            'projection': projection.state_dict()
        }, os.path.join(epoch_dir, "custom_layers.pt"))
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = os.path.join(OUTPUT_DIR, "best")
            llm.save_pretrained(best_dir)
            torch.save({
                'micro_encoder_study': micro_encoder_study.state_dict(),
                'micro_encoder_ibd': micro_encoder_ibd.state_dict(),
                'projection': projection.state_dict()
            }, os.path.join(best_dir, "custom_layers.pt"))
            print(f"🏆 新的最佳模型! Loss: {best_loss:.4f}")
    
    # 保存最终模型
    final_dir = os.path.join(OUTPUT_DIR, "final")
    llm.save_pretrained(final_dir)
    torch.save({
        'micro_encoder_study': micro_encoder_study.state_dict(),
        'micro_encoder_ibd': micro_encoder_ibd.state_dict(),
        'projection': projection.state_dict()
    }, os.path.join(final_dir, "custom_layers.pt"))
    
    print("\n" + "="*60)
    print("🎉 训练全部完成！")
    print(f"   最佳Loss: {best_loss:.4f}")
    print(f"   模型保存在: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
